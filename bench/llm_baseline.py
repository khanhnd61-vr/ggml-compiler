"""llama.cpp baseline for an LLM: accuracy against transformers f32 and llama-bench speed.

    uv run python bench/llm_baseline.py Qwen/Qwen3-0.6B qwen3-0.6b

Reads the llama.cpp GGUFs `<dir>/<name>-{bf16,f16,q8_0,q4_k_m}.gguf` (made by llama.cpp's
convert_hf_to_gguf.py and llama-quantize) and the builds in `--llama` (build-cpu, build-cuda,
logits-cpu, logits-cuda). Writes `<dir>/<name>/`: tokens.bin (the first 512 tokens of
bench/llama/text.txt), ref.npy (transformers f32 logits), greedy.npy (transformers' 64-token greedy
continuation) and baseline.json; prints a table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ggmlc.verify.parity import logit_stats

TEXT = Path(__file__).parent / "llama" / "text.txt"
TYPES = ["bf16", "f16", "q8_0", "q4_k_m"]


def reference(repo: str, out: Path, n_tokens: int, n_gen: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (out / "ref.npy").exists():
        return
    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok(TEXT.read_text())["input_ids"][:n_tokens]
    if len(ids) < n_tokens:
        raise SystemExit(f"{TEXT} has only {len(ids)} tokens")
    np.asarray(ids, dtype=np.int32).tofile(out / "tokens.bin")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        x = torch.tensor([ids])
        np.save(out / "ref.npy", model(x).logits[0].numpy())
        g = model.generate(x, max_new_tokens=n_gen, do_sample=False, min_new_tokens=n_gen)
    np.save(out / "greedy.npy", g[0, len(ids):].numpy().astype(np.int32))


def llama_logits(llama: Path, backend: str, gguf: Path, out: Path, n_gen: int) -> tuple[np.ndarray, np.ndarray]:
    dst = out / "llama-logits.bin"
    cmd = [str(llama / f"logits-{backend}" / "llama-logits"), str(gguf), str(out / "tokens.bin"), str(dst),
           "--gen", str(n_gen), "--ngl", "99" if backend == "cuda" else "0"]
    subprocess.run(cmd, check=True, capture_output=True)
    n = np.fromfile(out / "tokens.bin", dtype=np.int32).size
    logits = np.fromfile(dst, dtype=np.float32).reshape(n, -1)
    gen = np.fromfile(str(dst) + ".gen", dtype=np.int32)
    dst.unlink()
    return logits, gen


def llama_bench(llama: Path, backend: str, gguf: Path) -> dict[str, float]:
    cmd = [str(llama / f"build-{backend}" / "bin" / "llama-bench"), "-m", str(gguf), "-p", "512", "-n", "128",
           "-t", "6", "-o", "json"] + (["-ngl", "99"] if backend == "cuda" else [])
    rows = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    out = {}
    for r in rows:
        key = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
        out[key] = r["avg_ts"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("name")
    p.add_argument("--dir", default="/mnt/khanh/ggmlc-bench/llm")
    p.add_argument("--llama", default="/mnt/khanh/ggmlc-bench/llama.cpp")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--backends", default="cpu,cuda")
    p.add_argument("--speed-only", action="store_true", help="re-measure llama-bench into an existing baseline.json")
    a = p.parse_args()
    d, llama = Path(a.dir), Path(a.llama)
    out = d / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.speed_only:
        results = json.loads((out / "baseline.json").read_text())
        for t in TYPES:
            for b in a.backends.split(","):
                speed = llama_bench(llama, b, d / f"{a.name}-{t}.gguf")
                results[f"{t}/{b}"].update(speed)
                print(f"{t:>7} {b:<4}  " + "  ".join(f"{k} {v:8.1f} t/s" for k, v in speed.items()), flush=True)
        (out / "baseline.json").write_text(json.dumps(results, indent=1))
        return 0
    reference(a.repo, out, a.tokens, a.gen)
    ref, greedy = np.load(out / "ref.npy"), np.load(out / "greedy.npy")
    results = {}
    for t in TYPES:
        gguf = d / f"{a.name}-{t}.gguf"
        for b in a.backends.split(","):
            logits, gen = llama_logits(llama, b, gguf, out, a.gen)
            s = logit_stats(logits, ref)
            match = int(np.argmax(gen != greedy)) if (gen != greedy).any() else len(greedy)
            speed = llama_bench(llama, b, gguf)
            results[f"{t}/{b}"] = {**s.__dict__, "greedy_match": match, **speed}
            print(f"{t:>7} {b:<4}  {s.row()}  greedy {match:>2}/{len(greedy)}  "
                  + "  ".join(f"{k} {v:8.1f} t/s" for k, v in speed.items()), flush=True)
    (out / "baseline.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
