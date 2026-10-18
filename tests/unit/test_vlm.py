"""ggmlc.causal_lm on a tiny random SmolVLM: the encode entry (a symbolic tile count) writes image
rows into state, prefill_mm places them at the image tokens' positions, and the logits and greedy
tokens match transformers."""

import numpy as np
import pytest
import torch
from helpers import assert_close

import ggmlc
from ggmlc.runtime import Session

transformers = pytest.importorskip("transformers")
TOL = {"cpu": 1e-5, "cuda": 3e-3}
IMAGE = 200  # image token id
ROWS = 4  # rows per 64x64 tile: (64 / 16)^2 patches, pixel shuffle 2


def _smolvlm():
    cfg = transformers.SmolVLMConfig(
        # head size 64 in both towers, one that flash attention takes
        vision_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                           image_size=64, patch_size=16),
        text_config=dict(model_type="llama", hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=256,
                         max_position_embeddings=4096, tie_word_embeddings=False),
        scale_factor=2, image_token_id=IMAGE, vocab_size=256)
    model = transformers.SmolVLMForConditionalGeneration(cfg).eval()
    model.set_attn_implementation("eager")
    return model


def _prompt(tiles: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randn(1, tiles, 3, 64, 64, generator=g)
    text = torch.randint(0, IMAGE, (9,), generator=g).tolist()
    return pixels, text[:3] + [IMAGE] * (ROWS * tiles) + text[3:]


@pytest.fixture(scope="module")
def vlm(tmp_path_factory):
    torch.manual_seed(0)
    model = _smolvlm()
    # max_prompt 8: image tokens span several prefill_mm calls
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=8, logits="all", max_image_tokens=4 * ROWS),
                        out_dir=tmp_path_factory.mktemp("vlm"))
    return model, art


def test_program_shape(vlm):
    _, art = vlm
    with Session(art.gguf, backend="cpu") as s:
        assert set(s.entries) == {"prefill", "decode", "generate", "encode", "prefill_mm"}
        assert s.symbols["tiles"] == (2, 4)
        assert s.metadata["image"] == {"kind": "tiles", "token": IMAGE, "rows": 16, "rows_per_tile": ROWS, "tile": [3, 64, 64],
                                       "max_tiles": 4}
        assert s.entries["encode"][1] == []  # writes state only
    assert art.report.data["stages"]["encode"]["ops"]["set_rows"] == 1


@pytest.mark.parametrize("tiles", [3, 1])
def test_image_logits_match_transformers(vlm, backend, tiles):
    """One tile is encoded twice (a dynamic dim's min is 2) and the copy ignored."""
    model, art = vlm
    pixels, ids = _prompt(tiles, seed=tiles)
    with torch.no_grad():
        ref = model(input_ids=torch.tensor([ids]), pixel_values=pixels).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        g.reset()
        assert g.encode(pixels) == ROWS * tiles
        got = g.feed(ids)
    assert got.shape == ref.shape
    assert_close({"logits": got}, {"logits": ref}, TOL[backend])


def test_greedy_with_image_matches_generate(vlm, backend):
    model, art = vlm
    pixels, ids = _prompt(2, seed=7)
    x = torch.tensor([ids])
    with torch.no_grad():
        ref = model.generate(input_ids=x, attention_mask=torch.ones_like(x), pixel_values=pixels, max_new_tokens=30,
                             min_new_tokens=30, do_sample=False, pad_token_id=0)[0, len(ids):].tolist()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        assert g.generate(ids, max_new=30, stop=[], pixels=pixels) == ref
        eos = s.metadata["eos"]
        host = g.generate(ids, max_new=30, pixels=pixels)
        assert g.generate(ids, max_new=30, pixels=pixels, device=True) == host
        assert host == ref[:next((i + 1 for i, t in enumerate(ref) if t in eos), len(ref))]


def test_image_rows_are_checked(vlm):
    _, art = vlm
    pixels, ids = _prompt(2, seed=8)
    with Session(art.gguf, backend="cpu") as s:
        g = ggmlc.Generator(s)
        g.reset()
        with pytest.raises(ValueError, match="encode"):
            g.feed(ids)  # nothing encoded yet
        with pytest.raises(ValueError, match="pixels"):
            g.encode(torch.randn(5, 3, 64, 64))  # above max_tiles
        g.encode(pixels)
        with pytest.raises(ValueError, match="image tokens"):
            g.feed(ids + [IMAGE] * 2)  # more image tokens than rows


def test_fast_build_and_cpp(tmp_path, backend):
    """O3 (flash attention in the vision tower and the LM, fast GELU, f16 KV cache), and the
    generated C++ running every entry, encode and prefill_mm included."""
    from helpers import build_generated, check_generated

    torch.manual_seed(0)
    model = _smolvlm()
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=24, max_image_tokens=4 * ROWS), opt="O3",
                        out_dir=tmp_path, emit=("gguf", "cpp"), name="vlm")
    for st in ("encode", "prefill_mm"):
        assert art.report.data["stages"][st]["ops"]["flash_attn"] == 2
    pixels, ids = _prompt(3, seed=9)
    with torch.no_grad():
        ref = model(input_ids=torch.tensor([ids]), pixel_values=pixels).logits[0, -1].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        g.reset()
        g.encode(pixels)
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, 1e-2)
    check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)


# --- Qwen3-VL: M-RoPE positions apart from cache rows, deepstack, grid tables from the caller ---

QIMAGE = 250


def _qwen3vl():
    cfg = transformers.Qwen3VLConfig(
        vision_config=dict(depth=3, hidden_size=128, intermediate_size=256, num_heads=2, out_hidden_size=128,
                           patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64,
                           deepstack_visual_indexes=[0, 1, 2]),
        text_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=3, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=64, vocab_size=300, max_position_embeddings=4096,
                         rope_theta=5e6, tie_word_embeddings=True,
                         rope_scaling={"rope_type": "default", "mrope_section": [12, 10, 10], "mrope_interleaved": True}),
        image_token_id=QIMAGE, vision_start_token_id=251, vision_end_token_id=252, tie_word_embeddings=True)
    model = transformers.Qwen3VLForConditionalGeneration(cfg).eval()
    model.set_attn_implementation("eager")
    return model


def _qprompt(h: int, w: int, seed: int):
    """pixels [h*w, 1536], grid [1, 3], ids: text, <vision_start>, h*w/4 image tokens, <vision_end>, text."""
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randn(h * w, 3 * 2 * 16 * 16, generator=g)
    text = torch.randint(0, 250, (6,), generator=g).tolist()
    return pixels, torch.tensor([[1, h, w]]), text[:3] + [251] + [QIMAGE] * (h * w // 4) + [252] + text[3:]


def _hf_qwen(model, ids, pixels, grid):
    x = torch.tensor([ids])
    return dict(input_ids=x, attention_mask=torch.ones_like(x), pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=(x == QIMAGE).int())


@pytest.fixture(scope="module")
def qwen3vl(tmp_path_factory):
    torch.manual_seed(0)
    model = _qwen3vl()
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=8, logits="all", max_image_tokens=32),
                        out_dir=tmp_path_factory.mktemp("qwen3vl"))
    return model, art


def test_qwen3vl_program_shape(qwen3vl):
    _, art = qwen3vl
    with Session(art.gguf, backend="cpu") as s:
        assert [i.name for i in s.entries["prefill_mm"][0]] == ["tokens", "gather", "positions", "mrope", "mask"]
        assert [i.name for i in s.entries["encode"][0]] == ["pixels", "pos_idx", "pos_w", "vpos"]
        assert [i.name for i in s.entries["generate"][0]] == ["first", "start", "rstart"]
        assert s.symbols["vision_tokens"] == (2, 32)
        assert s.metadata["rope"] == "separate" and s.metadata["image"]["kind"] == "qwen3_vl"
    assert art.report.data["stages"]["encode"]["ops"]["set_rows"] == 4  # merged rows + 3 deepstack


@pytest.mark.parametrize("hw", [(6, 8), (4, 4)])
def test_qwen3vl_image_logits_match_transformers(qwen3vl, backend, hw):
    model, art = qwen3vl
    pixels, grid, ids = _qprompt(*hw, seed=hw[0])
    with torch.no_grad():
        ref = model(**_hf_qwen(model, ids, pixels, grid)).logits[0].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        g.reset()
        assert g.encode(pixels, grid[0]) == hw[0] * hw[1] // 4
        got = g.feed(ids)
    assert_close({"logits": got}, {"logits": ref}, TOL[backend])


def test_qwen3vl_greedy_with_image_matches_generate(qwen3vl, backend):
    model, art = qwen3vl
    pixels, grid, ids = _qprompt(6, 8, seed=11)
    with torch.no_grad():
        ref = model.generate(**_hf_qwen(model, ids, pixels, grid), max_new_tokens=30, min_new_tokens=30,
                             do_sample=False, pad_token_id=0)[0, len(ids):].tolist()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        assert g.generate(ids, max_new=30, stop=[], pixels=pixels, grid_thw=grid[0]) == ref
        host = g.generate(ids, max_new=30, pixels=pixels, grid_thw=grid[0])
        assert g.generate(ids, max_new=30, pixels=pixels, grid_thw=grid[0], device=True) == host


def test_qwen3vl_fast_build_and_cpp(tmp_path, backend):
    from helpers import build_generated, check_generated

    torch.manual_seed(0)
    model = _qwen3vl()
    art = ggmlc.compile(ggmlc.causal_lm(model, max_context=768, max_prompt=24, max_image_tokens=32), opt="O3",
                        out_dir=tmp_path, emit=("gguf", "cpp"), name="qwen3vl")
    ops = art.report.data["stages"]
    assert ops["encode"]["ops"]["flash_attn"] == 3 and ops["prefill_mm"]["ops"]["flash_attn"] == 3
    assert ops["decode"]["ops"]["rope"] == 6  # text RoPE fused (q and k of 3 layers)
    pixels, grid, ids = _qprompt(6, 8, seed=12)
    with torch.no_grad():
        ref = model(**_hf_qwen(model, ids, pixels, grid)).logits[0, -1].numpy()
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        g.reset()
        g.encode(pixels, grid[0])
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, 1e-2)
    check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)
