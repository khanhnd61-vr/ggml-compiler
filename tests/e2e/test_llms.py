"""More LLM architectures through ggml_compiler.causal_lm, with tests/e2e/test_qwen3.py's checks: over the
512 tokens of bench/llama/text.txt, last-token logits in f32 at the acceptance tolerances and 64
greedy tokens exactly, on CPU and CUDA, from the host and the device generation loop.

- DeepSeek-R1-Distill-Qwen-1.5B: Qwen2 (biases on Q, K and V). ~7 GB in f32.
- Gemma 4 E2B (its language model): sliding-window layers over the last 512 positions, so the
  greedy tokens after the 512-token prompt depend on the window; K/V shared by the last 20 layers,
  per-layer embeddings, two head sizes. ~20 GB of RAM in f32.

GGML_COMPILER_BIG=1."""

import os
from pathlib import Path

import pytest
import torch
from helpers import assert_close, scratch_dir

import ggml_compiler
from ggml_compiler.runtime import Session

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.environ.get("GGML_COMPILER_BIG"),
                                 reason="7-20 GB of RAM in f32 (GGML_COMPILER_BIG=1)")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPOS = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "google/gemma-4-E2B-it"]
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}
TEXT = Path(__file__).resolve().parents[2] / "bench" / "llama" / "text.txt"

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module", params=REPOS, ids=lambda r: r.split("/")[1])
def llm(request, tmp_path_factory):
    repo = request.param
    try:
        tok = transformers.AutoTokenizer.from_pretrained(repo)
        model = transformers.AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                                                  attn_implementation="eager").eval()
    except Exception as e:  # not in the local cache
        pytest.skip(f"cannot load {repo}: {e}")
    ids = tok(TEXT.read_text())["input_ids"][:512]
    with torch.no_grad():
        x = torch.tensor([ids])
        ref = model(x).logits[0, -1].numpy()
        # No EOS: min_new_tokens would ban it instead, and the argmax differ when it is EOS.
        greedy = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=64, do_sample=False,
                                eos_token_id=None)[0, len(ids):].tolist()
    with scratch_dir(tmp_path_factory, repo.split("/")[1]) as out:
        art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=1024, max_prompt=512), out_dir=out)
        del model
        yield art, ids, ref, greedy


def test_last_logits(llm, backend):
    art, ids, ref, _ = llm
    with Session(art.gguf, backend=backend) as s:
        g = ggml_compiler.Generator(s)
        g.reset()
        assert_close({"logits": g.feed(ids)}, {"logits": ref}, ACCEPT[backend])


@pytest.mark.parametrize("device", [False, True])
def test_greedy_64(llm, backend, device):
    art, ids, _, greedy = llm
    assert len(greedy) == 64
    with Session(art.gguf, backend=backend) as s:
        got = ggml_compiler.Generator(s).generate(ids, max_new=64, stop=[] if not device else None, device=device)
        eos = s.metadata["eos"]
    # The device loop stops at the model's EOS; the reference ran on without stopping.
    assert got == greedy[:len(got)] and (len(got) == 64 or got[-1] in eos)
