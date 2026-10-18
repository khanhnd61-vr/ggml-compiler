"""L5 exit check for SmolVLM2-500M-Instruct: an image prompt (17 tiles, 1088 image tokens) through
ggmlc.causal_lm's encode and prefill_mm matches transformers in f32 (last-token logits at the
acceptance tolerances, 64 greedy tokens exactly), on CPU and CUDA, from the host and the device
generation loop."""

import os
from pathlib import Path

import pytest
import torch
from helpers import assert_close, scratch_dir

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "HuggingFaceTB/SmolVLM2-500M-Instruct"
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}
IMAGE = Path(__file__).resolve().parents[2] / ".deps" / "llama.cpp-b11223" / "tools" / "mtmd" / "test-1.jpeg"

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def smolvlm(tmp_path_factory):
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
    with scratch_dir(tmp_path_factory, "smolvlm") as out:
        art = ggmlc.compile(ggmlc.causal_lm(model, max_context=2048), out_dir=out)
        del model
        yield art, inp["input_ids"][0].tolist(), inp["pixel_values"], ref, greedy


def test_last_logits(smolvlm, backend):
    art, ids, pixels, ref, _ = smolvlm
    with Session(art.gguf, backend=backend) as s:
        assert pixels.shape[1] == 17 and ids.count(s.metadata["image"]["token"]) == 17 * 64
        g = ggmlc.Generator(s)
        g.reset()
        g.encode(pixels)
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, ACCEPT[backend])


@pytest.mark.parametrize("device", [False, True])
def test_greedy_64(smolvlm, backend, device):
    art, ids, pixels, _, greedy = smolvlm
    with Session(art.gguf, backend=backend) as s:
        got = ggmlc.Generator(s).generate(ids, max_new=64, pixels=pixels, device=device)
    assert got == greedy
