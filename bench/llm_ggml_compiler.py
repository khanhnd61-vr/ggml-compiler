"""ggml-compiler against the llama.cpp baseline of bench/llm_baseline.py, at llama.cpp's own weight types.

    uv run python bench/llm_ggml_compiler.py Qwen/Qwen3-0.6B qwen3-0.6b --opt O3

For each type, compiles the model with `ggml_compiler.llm.llama_cpp_policy` (the same weight bytes as
llama.cpp's GGUF) twice: logits="last" for speed and greedy decoding, logits="all" for KL / top-1
over the baseline's 512 tokens. All compiles run first, then the measurements, so nothing competes
with the timings. Writes <dir>/<name>/ggml-compiler-<opt>.json and prints each row next to llama.cpp's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import ggml_compiler
from ggml_compiler.llm import Generator, bench, llama_cpp_policy
from ggml_compiler.verify.parity import logit_stats

TYPES = ["bf16", "f16", "q8_0", "q4_k_m"]


def build(repo: str, name: str, types: list[str], opt: str, out: Path, max_context: int,
          ggufs: Path) -> dict[str, dict[str, Path]]:
    from transformers import AutoModelForCausalLM

    model = None
    paths: dict[str, dict[str, Path]] = {}
    for t in types:
        paths[t] = {}
        for logits in ("last", "all"):
            d = out / f"{name}-{t}-{opt}-{logits}"
            paths[t][logits] = d / "model.gguf"
            if (d / "report.json").exists():  # written last: a killed compile leaves none
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager")
            policy = llama_cpp_policy(str(ggufs / f"{name}-{t}.gguf"), model) if t != "f32" else None
            ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=max_context, max_prompt=512,
                                                          logits=logits),
                                  out_dir=d, opt=opt, dtype_policy=policy)
            print(f"compiled {d}", flush=True)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("name")
    p.add_argument("--opt", default="O3")
    p.add_argument("--types", default=",".join(TYPES))
    p.add_argument("--backends", default="cpu,cuda")
    p.add_argument("--dir", default="/mnt/khanh/ggmlc-bench/llm")
    p.add_argument("--out", default="/mnt/khanh/ggmlc-out")
    p.add_argument("--max-context", type=int, default=1024)
    a = p.parse_args()
    d = Path(a.dir) / a.name
    ref, greedy = np.load(d / "ref.npy"), np.load(d / "greedy.npy")
    ids = np.fromfile(d / "tokens.bin", dtype=np.int32).tolist()
    base = json.loads((d / "baseline.json").read_text())
    types = a.types.split(",")
    paths = build(a.repo, a.name, types, a.opt, Path(a.out), a.max_context, Path(a.dir))

    results = {}
    for t in types:
        for b in a.backends.split(","):
            with ggml_compiler.Session(paths[t]["all"], backend=b, n_threads=6) as s:
                g = Generator(s)
                g.reset()
                st = logit_stats(g.feed(ids), ref)
            with ggml_compiler.Session(paths[t]["last"], backend=b, n_threads=6) as s:
                gen = Generator(s).generate(ids, max_new=len(greedy), stop=[])
                match = next((i for i, (x, y) in enumerate(zip(gen, greedy)) if x != y), len(greedy))
                speed = bench(s)
            results[f"{t}/{b}"] = {**st.__dict__, "greedy_match": match, **speed}
            lb = base[f"{t}/{b}"]
            print(f"{t:>7} {b:<4} KL {st.kl_mean:.2e} (llama.cpp {lb['kl_mean']:.2e}, x{st.kl_mean / lb['kl_mean']:.2f})  "
                  f"top-1 {100 * st.top1:.1f}% ({100 * lb['top1']:.1f}%)  greedy {match}/{len(greedy)} "
                  f"({lb['greedy_match']})  pp512 {speed['pp512']:.0f} ({lb['pp512']:.0f}, x{lb['pp512'] / speed['pp512']:.2f})  "
                  f"tg128 {speed['tg128']:.1f} ({lb['tg128']:.1f}, x{lb['tg128'] / speed['tg128']:.2f})", flush=True)
    path = d / f"ggml-compiler-{a.opt}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
