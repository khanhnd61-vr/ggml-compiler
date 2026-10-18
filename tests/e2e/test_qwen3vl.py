"""L5 exit check for Qwen3-VL-2B-Instruct: an image prompt (a 640x480 image, 300 image tokens, M-RoPE
positions, deepstack) through ggml_compiler.causal_lm's encode and prefill_mm matches transformers in f32
(last-token logits at the acceptance tolerances, 64 greedy tokens exactly), on CPU and CUDA, from
the host and the device generation loop. ~13 GB of RAM while compiling: GGML_COMPILER_BIG=1."""

import os
from pathlib import Path

import pytest
import torch
from helpers import assert_close, scratch_dir

import ggml_compiler
from ggml_compiler.runtime import Session

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.environ.get("GGML_COMPILER_BIG"),
                                 reason="Qwen3-VL-2B f32: ~5 min, ~13 GB of RAM (GGML_COMPILER_BIG=1)")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "Qwen/Qwen3-VL-2B-Instruct"
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}
IMAGE = Path(__file__).resolve().parents[2] / ".deps" / "llama.cpp-b11223" / "tools" / "mtmd" / "test-1.jpeg"

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def qwen3vl(tmp_path_factory):
    from PIL import Image

    if not IMAGE.exists():
        pytest.skip(f"{IMAGE} missing (the runtime build fetches the llama.cpp tree)")
    try:
        proc = transformers.AutoProcessor.from_pretrained(REPO)
        model = transformers.AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.float32,
                                                                         attn_implementation="eager").eval()
    except Exception as e:  # not in the local cache
        pytest.skip(f"cannot load {REPO}: {e}")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in detail."}]}]
    inp = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True),
               images=[Image.open(IMAGE).convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        ref = model(**inp).logits[0, -1].numpy()
        greedy = model.generate(**inp, max_new_tokens=64, do_sample=False)[0, inp["input_ids"].shape[1]:].tolist()
    with scratch_dir(tmp_path_factory, "qwen3vl") as out:
        art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=2048), out_dir=out)
        del model
        yield art, inp["input_ids"][0].tolist(), inp["pixel_values"], inp["image_grid_thw"][0], ref, greedy


def test_last_logits(qwen3vl, backend):
    art, ids, pixels, grid, ref, _ = qwen3vl
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        assert g.encode(pixels, grid) == ids.count(s.metadata["image"]["token"]) == 300
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, ACCEPT[backend])


@pytest.mark.parametrize("device", [False, True])
def test_greedy_64(qwen3vl, backend, device):
    art, ids, pixels, grid, _, greedy = qwen3vl
    with Session(art.gguf, backend=backend) as s:
        got = ggml_compiler.Generator(s).generate(ids, max_new=64, pixels=pixels, grid_thw=grid, device=device)
    assert got == greedy
