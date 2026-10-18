"""L3 exit check: Qwen3-0.6B through ggmlc.causal_lm matches transformers in f32 (logits at the
acceptance tolerances, 64 greedy tokens exactly), on CPU and CUDA, from the host and the device
generation loop."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import assert_close, scratch_dir

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "Qwen/Qwen3-0.6B"
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}
TEXT = Path(__file__).resolve().parents[2] / "bench" / "llama" / "text.txt"

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def qwen(tmp_path_factory):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(REPO)
        model = transformers.AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32,
                                                                  attn_implementation="eager").eval()
    except Exception as e:  # not in the local cache
        pytest.skip(f"cannot load {REPO}: {e}")
    ids = tok(TEXT.read_text())["input_ids"][:512]
    with torch.no_grad():
        x = torch.tensor([ids])
        ref = model(x).logits[0, -1].numpy()
        greedy = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=64, min_new_tokens=64,
                                do_sample=False)[0, len(ids):].tolist()
    with scratch_dir(tmp_path_factory, "qwen3") as out:
        art = ggmlc.compile(ggmlc.causal_lm(model, max_context=1024, max_prompt=512), out_dir=out)
        del model
        yield art, ids, ref, greedy


def test_last_logits(qwen, backend):
    art, ids, ref, _ = qwen
    with Session(art.gguf, backend=backend) as s:
        g = ggmlc.Generator(s)
        g.reset()
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, ACCEPT[backend])


@pytest.mark.parametrize("device", [False, True])
def test_greedy_64(qwen, backend, device):
    art, ids, _, greedy = qwen
    with Session(art.gguf, backend=backend) as s:
        got = ggmlc.Generator(s).generate(ids, max_new=64, stop=[] if not device else None, device=device)
        eos = s.metadata["eos"]
    # The device loop stops at the model's EOS; the reference ran on without stopping.
    assert got == greedy[:len(got)] and (len(got) == 64 or got[-1] in eos)


def test_graph(qwen):
    ops = qwen[0].report.data["stages"]["decode"]["ops"]
    assert ops["set_rows"] == 56 and ops["rms_norm"] == 113 and ops["swiglu"] == 28  # 28 layers, K and V
    # O1 copies per layer: exact attention's V transpose, and rotate_half's four slices (fuse_rope,
    # O2, makes RoPE one op). The K cache is read in place.
    assert ops.get("cont", 0) == 5 * 28
