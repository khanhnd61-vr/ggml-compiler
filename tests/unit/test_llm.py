"""ggmlc.causal_lm on tiny random Hugging Face models: prefill / decode entry points over an
in-place KV cache match transformers, token for token."""

import numpy as np
import pytest
import torch
from helpers import assert_close

import ggmlc
from ggmlc.runtime import Session

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


@pytest.fixture(scope="module", params=["qwen3", "llama"])
def lm(request, tmp_path_factory):
    torch.manual_seed(0)
    model = {"qwen3": _qwen3, "llama": _llama}[request.param]().float().eval()
    model.config._attn_implementation = "eager"
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=24),
                        out_dir=tmp_path_factory.mktemp(f"lm-{request.param}"))
    return model, art


def test_program_shape(lm):
    _, art = lm
    with Session(art.gguf, backend="cpu") as s:
        assert set(s.entries) == {"prefill", "decode", "generate"}
        assert s.symbols == {"max_new": (1, 768), "s": (2, 24), "kv_blocks": (2, 3)}
        assert s.metadata["eos"] == [7] and s.metadata["max_context"] == 768
    ops = art.report.data["stages"]["decode"]["ops"]
    assert ops["set_rows"] == 4  # K and V of 2 layers, written in place


def test_logits_match_transformers(lm, backend):
    model, art = lm
    ids = torch.randint(0, 500, (1, 300), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
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
        ref = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, min_new_tokens=40,
                             do_sample=False, pad_token_id=0)[0, 30:].tolist()
    with Session(art.gguf, backend=backend) as s:
        assert ggmlc.Generator(s).generate(ids[0].tolist(), max_new=40, stop=[]) == ref


def test_state_persists_until_reset(lm):
    _, art = lm
    with Session(art.gguf, backend="cpu") as s:
        g = ggmlc.Generator(s)
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
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=24), opt="O3",
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
        g = ggmlc.Generator(s)
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
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=16, logits="all"), out_dir=tmp_path)
    ids = torch.randint(0, 500, (1, 40), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = model(ids).logits[0].numpy()
    with Session(art.gguf, backend="cpu") as s:
        g = ggmlc.Generator(s)
        g.reset()
        got = g.feed(ids[0].tolist())
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-5 * max(1.0, np.abs(ref).max()))


def test_generated_cpp_runs_every_entry(tmp_path, backend):
    from helpers import build_generated, check_generated

    torch.manual_seed(0)
    model = _llama().float().eval()
    model.config._attn_implementation = "eager"
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=24), opt="O3", out_dir=tmp_path,
                        emit=("gguf", "cpp"), name="lm")
    assert set(art.report.data["entries"]) == {"prefill", "decode", "generate"}
    check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)


def test_device_generate_matches_host(lm, backend):
    """The generate entry runs the decoding loop in the runtime: argmax on the device, positions,
    masks and the KV length set per step, stop at EOS. Same tokens as the host loop."""
    _, art = lm
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        prompt = torch.randint(0, 500, (30,), generator=torch.Generator().manual_seed(5)).tolist()
        host = g.generate(prompt, max_new=300, stop=[7])  # crosses into the third KV block
        dev = g.generate(prompt, max_new=300, device=True)
        assert dev == host
        assert g.n_past == len(prompt) + len(dev) - 1
