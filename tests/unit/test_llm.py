"""ggml_compiler.causal_lm on tiny random Hugging Face models: prefill / decode entry points over an
in-place KV cache match transformers, token for token."""

import numpy as np
import pytest
import torch
from helpers import assert_close

import ggml_compiler
from ggml_compiler.runtime import Session

transformers = pytest.importorskip("transformers")
TOL = {"cpu": 1e-5, "cuda": 3e-3}


def _qwen3():
    cfg = transformers.Qwen3Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=64, eos_token_id=7,
                                   tie_word_embeddings=True)
    return transformers.Qwen3ForCausalLM(cfg)


def _llama():
    cfg = transformers.LlamaConfig(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, eos_token_id=7)
    return transformers.LlamaForCausalLM(cfg)


def _qwen2():  # DeepSeek-R1-Distill-Qwen: biases on Q, K and V
    cfg = transformers.Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, eos_token_id=7)
    return transformers.Qwen2ForCausalLM(cfg)


def _gemma4(**kw):
    """Gemma 4 E2B's features at toy size: sliding-window and full layers with their own head sizes
    and RoPE, the last two layers reusing earlier layers' K/V, per-layer embeddings, double-wide
    MLPs in the sharing layers, logit soft-capping."""
    types = ["sliding_attention"] * 2 + ["full_attention"]
    cfg = transformers.Gemma4TextConfig(**{
        "vocab_size": 500, "vocab_size_per_layer_input": 500, "hidden_size": 128, "hidden_size_per_layer_input": 16,
        "intermediate_size": 256, "num_hidden_layers": 6, "layer_types": types * 2, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 32, "global_head_dim": 64, "sliding_window": 40,
        "num_kv_shared_layers": 2, "use_double_wide_mlp": True, "final_logit_softcapping": 30.0, "eos_token_id": 7,
        "pad_token_id": 0, "bos_token_id": 2, **kw})
    model = transformers.Gemma4ForCausalLM(cfg)
    # Gemma 4 scales attention by 1 after normalizing q and k: with the init's unit q_norm weights
    # the scores reach +-head_dim and every rounding error is amplified (transformers' own f32
    # logits are then 1.6e-5 from float64). Give the scores an ordinary temperature.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_norm.weight.fill_(layer.self_attn.head_dim ** -0.5)
    return model


def _cached_layers(model):
    layers = model.model.layers
    return sum(not getattr(layer.self_attn, "is_kv_shared_layer", False) for layer in layers)


@pytest.fixture(scope="module", params=["qwen3", "llama", "qwen2", "gemma4"])
def lm(request, tmp_path_factory):
    torch.manual_seed(0)
    model = {"qwen3": _qwen3, "llama": _llama, "qwen2": _qwen2, "gemma4": _gemma4}[request.param]().float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=24),
                                out_dir=tmp_path_factory.mktemp(f"lm-{request.param}"))
    return model, art


def test_program_shape(lm):
    model, art = lm
    with Session(art.gguf, backend="cpu") as s:
        assert set(s.entries) == {"prefill", "decode", "generate"}
        assert s.symbols == {"max_new": (1, 768), "s": (2, 24), "kv_blocks": (2, 3)}
        assert s.metadata["eos"] == [7] and s.metadata["max_context"] == 768
    ops = art.report.data["stages"]["decode"]["ops"]
    assert ops["set_rows"] == 2 * _cached_layers(model)  # K and V of each layer with a cache, in place


def test_logits_match_transformers(lm, backend):
    model, art = lm
    ids = torch.randint(0, 500, (1, 300), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        got = {}
        got[249] = g.feed(ids[0, :250].tolist())  # 11 chunks of <= 24: crosses n_kv 512
        for i in range(250, 300):  # one token at a time, past 256 and into the next kv block
            got[i] = g.feed([int(ids[0, i])])
    for i, row in got.items():
        assert_close({"logits": row}, {"logits": ref[i]}, TOL[backend])


def test_greedy_matches_generate(lm, backend):
    model, art = lm
    ids = torch.randint(0, 500, (1, 30), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        # No EOS: min_new_tokens would ban it instead, and the argmax differ when it is EOS.
        ref = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False,
                             pad_token_id=0, eos_token_id=None)[0, 30:].tolist()
    assert len(ref) == 40
    with Session(art.gguf, backend=backend) as s:
        assert ggml_compiler.Generator(s).generate(ids[0].tolist(), max_new=40, stop=[]) == ref


def test_state_persists_until_reset(lm):
    _, art = lm
    with Session(art.gguf, backend="cpu") as s:
        g = ggml_compiler.Generator(s)
        a = g.generate([1, 2, 3, 4], max_new=5, stop=[])
        g.feed([9, 9, 9])  # more context: the cache now holds it
        b = g.generate([1, 2, 3, 4], max_new=5, stop=[])  # generate resets first
        assert a == b


@pytest.fixture(scope="module")
def fast_lm(tmp_path_factory):
    """O3: f16 KV cache, flash attention, ggml RoPE, no fully-masked-row guard."""
    torch.manual_seed(0)
    model = _qwen3().float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=24), opt="O3",
                                out_dir=tmp_path_factory.mktemp("lm-o3"))
    return model, art


def test_fast_build(fast_lm, backend):
    model, art = fast_lm
    for st in ("prefill", "decode"):
        ops = art.report.data["stages"][st]["ops"]
        assert ops["rope"] == 4 and ops["flash_attn"] == 2 and ops["swiglu"] == 2 and "cont" not in ops
    ids = torch.randint(0, 500, (1, 280), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        got = {259: g.feed(ids[0, :260].tolist())}
        for i in range(260, 280):
            got[i] = g.feed([int(ids[0, i])])
    for i, row in got.items():
        assert_close({"logits": row}, {"logits": ref[i]}, 1e-2)


def test_all_logits(tmp_path):
    torch.manual_seed(0)
    model = _qwen3().float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=16, logits="all"),
                                out_dir=tmp_path)
    ids = torch.randint(0, 500, (1, 40), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend="cpu") as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        got = g.feed(ids[0].tolist())
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-5 * max(1.0, np.abs(ref).max()))


def test_generated_cpp_runs_every_entry(tmp_path, backend):
    from helpers import build_generated, check_generated

    torch.manual_seed(0)
    model = _llama().float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=24), opt="O3",
                                out_dir=tmp_path,
                                emit=("gguf", "cpp"), name="lm")
    assert set(art.report.data["entries"]) == {"prefill", "decode", "generate"}
    check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)


def test_device_generate_matches_host(lm, backend):
    """The generate entry runs the decoding loop in the runtime: argmax on the device, positions,
    masks and the KV length set per step, stop at EOS. Same tokens as the host loop."""
    _, art = lm
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        prompt = torch.randint(0, 500, (30,), generator=torch.Generator().manual_seed(5)).tolist()
        host = g.generate(prompt, max_new=300, stop=[7])  # crosses into the third KV block
        dev = g.generate(prompt, max_new=300, device=True)
        assert dev == host
        assert g.n_past == len(prompt) + len(dev) - 1


def test_device_generate_from_third_kv_block(lm, backend):
    """A generation loop that starts at position 512: the prompt's last call read 512 cache rows,
    the loop reads 768. Values sized by the KV length (Gemma 4's sliding-window distances) are
    rebuilt on every step, not hoisted out of the loop at the prompt's length."""
    _, art = lm
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        prompt = torch.randint(0, 500, (512,), generator=torch.Generator().manual_seed(6)).tolist()
        host = g.generate(prompt, max_new=40, stop=[])
        dev = g.generate(prompt, max_new=40, device=True)
        assert dev == host[:len(dev)] and (len(dev) == 40 or dev[-1] == 7)


def test_gemma4_fast_build(tmp_path, backend):
    """O3 on Gemma 4: every RoPE becomes one rope op. Its full-attention layers' proportional RoPE
    rotates a quarter of the head (the zero frequencies become infinite freq_factors), and it
    rotates q and k as [B, S, H, D], before the heads move out."""
    torch.manual_seed(0)
    model = _gemma4().float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=24), opt="O3",
                                out_dir=tmp_path)
    for st in ("prefill", "decode"):
        ops = art.report.data["stages"][st]["ops"]
        assert ops["rope"] == 10 and "sin" not in ops  # q of 6 layers, k of the 4 with their own K/V
    ids = torch.randint(0, 500, (1, 280), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        got = {259: g.feed(ids[0, :260].tolist())}
        for i in range(260, 280):
            got[i] = g.feed([int(ids[0, i])])
    for i, row in got.items():
        assert_close({"logits": row}, {"logits": ref[i]}, 1e-2)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_gemma4_wide_heads(tmp_path, backend, kv_heads, capfd, monkeypatch):
    """Gemma 4's 512-wide full-attention heads fuse into flash attention at O3. CUDA runs that only
    with grouped-query heads (on Turing or newer): with as many K/V heads as query heads the runtime
    builds the node as matmul, softmax, matmul at load instead. Either way the logits match
    transformers, and the generated C++ makes the same choice as the runtime."""
    from helpers import build_generated, check_generated

    torch.manual_seed(0)
    model = _gemma4(global_head_dim=512, num_key_value_heads=kv_heads).float().eval()
    model.config._attn_implementation = "eager"
    art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=768, max_prompt=24), opt="O3",
                                out_dir=tmp_path, emit=("gguf", "cpp"), name="g4")
    assert art.report.data["stages"]["prefill"]["ops"]["flash_attn"] == 2  # the two full-attention layers
    ids = torch.randint(0, 500, (1, 60), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    monkeypatch.setenv("GGML_COMPILER_LOG", "debug")
    capfd.readouterr()
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        got = {39: g.feed(ids[0, :40].tolist())}
        for i in range(40, 60):
            got[i] = g.feed([int(ids[0, i])])
    fell_back = "cannot run flash attention with head size 512" in capfd.readouterr().err
    if backend == "cpu":
        assert not fell_back  # the CPU's flash attention takes any head size
    elif kv_heads == 4:
        assert fell_back  # without grouped-query heads, CUDA has no kernel for 512 on any GPU
    for i, row in got.items():
        assert_close({"logits": row}, {"logits": ref[i]}, 1e-2)
    if kv_heads == 4:
        check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)
