"""ggml-compiler against the llama.cpp baseline of bench/vlm_baseline.py, at llama.cpp's own weight types.

    uv run python bench/vlm_ggml_compiler.py SNAPSHOT_DIR smolvlm2-500m --opt O3

For each type, compiles the model with `ggml_compiler.llm.llama_cpp_policy` (the language model's types from
llama.cpp's GGUF, the vision tower's from its mmproj) twice: logits="last" for speed and greedy
decoding, logits="all" for KL / top-1 over the baseline's answers (pooled over its cases, as the
baseline does; the prompt and the answer are fed in separate calls, as mtmd-logits feeds them). All compiles run first, then the measurements. Timings match mtmd-logits', on the
first case: the image encode, the whole prompt (text and image rows), and --gen greedy tokens after
it, each a mean over --reps runs after a warmup. Writes <dir>/<name>/ggml-compiler-<opt>.json and prints
each row next to llama.cpp's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ggml_compiler
from ggml_compiler.llm import Generator, llama_cpp_policy
from ggml_compiler.verify.parity import stats_of, token_stats

TYPES = ["bf16", "f16", "q8_0", "q4_k_m"]


def build(snapshot: str, name: str, types: list[str], opt: str, out: Path, max_context: int,
          ggufs: Path, only: tuple[str, str] | None = None) -> dict[str, dict[str, Path]]:
    from transformers import AutoModelForImageTextToText

    model = None
    paths: dict[str, dict[str, Path]] = {}
    for t in types:
        paths[t] = {}
        for logits in ("last", "all"):
            d = out / f"{name}-{t}-{opt}-{logits}"
            paths[t][logits] = d / "model.gguf"
            if (d / "report.json").exists() or (only and only != (t, logits)):  # report.json is written last
                continue
            if model is None:
                model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32,
                                                                    attn_implementation="eager")
            policy = llama_cpp_policy(str(ggufs / f"{name}-{t}.gguf"), model, str(ggufs / f"{name}-mmproj-f16.gguf"))
            ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=max_context, logits=logits), out_dir=d,
                                  opt=opt,
                                  dtype_policy=policy)
            print(f"compiled {d}", flush=True)
    return paths


def timings(g: Generator, image: dict, prompt: list[int], gen: int, reps: int) -> dict[str, float]:
    enc, pp, tg = [], [], []
    for r in range(reps + 1):
        g.reset()
        t0 = time.perf_counter()
        g.encode(**image)
        t1 = time.perf_counter()
        logits = g.feed(prompt)
        t2 = time.perf_counter()
        for _ in range(gen):
            logits = g.feed([int(np.argmax(logits))])
        t3 = time.perf_counter()
        if r:
            enc.append(t1 - t0)
            pp.append(t2 - t1)
            tg.append(gen / (t3 - t2))
    return {"encode_ms": 1e3 * float(np.mean(enc)), "prompt_ms": 1e3 * float(np.mean(pp)), "tg": float(np.mean(tg))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot")
    p.add_argument("name")
    p.add_argument("--opt", default="O3")
    p.add_argument("--types", default=",".join(TYPES))
    p.add_argument("--backends", default="cpu,cuda")
    p.add_argument("--dir", default="/mnt/khanh/ggmlc-bench/vlm")
    p.add_argument("--out", default="/mnt/khanh/ggmlc-out")
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--build-one", metavar="TYPE:LOGITS",
                   help="compile one artifact (e.g. q8_0:all) and exit: a 2B model in f32 needs most of 15 GB")
    a = p.parse_args()
    if a.build_one:
        t, logits = a.build_one.split(":")
        build(a.snapshot, a.name, [t], a.opt, Path(a.out), a.max_context, Path(a.dir), only=(t, logits))
        return 0
    d = Path(a.dir) / a.name
    cases = []
    for c in json.loads((d / "cases.json").read_text()):
        cd = d / c
        meta = {"kind": "tiles", **json.loads((cd / "meta.json").read_text())}
        if meta["kind"] == "tiles":
            px = np.fromfile(cd / "pixels.bin", dtype=np.float32)
            image = {"pixels": px.reshape(-1, 3, meta["tile"], meta["tile"])}
        else:
            image = {"pixels": np.load(cd / "pixels.npy"), "grid_thw": meta["grid"]}
        cases.append({"name": c, "dir": cd, "meta": meta, "image": image,
                      "prompt": np.fromfile(cd / "prompt.bin", dtype=np.int32).tolist(),
                      "tokens": np.fromfile(cd / "tokens.bin", dtype=np.int32).tolist(),
                      "greedy": np.load(cd / "answer.npy")[: a.gen].tolist()})
    base = json.loads((d / "baseline.json").read_text())
    types = a.types.split(",")
    paths = build(a.snapshot, a.name, types, a.opt, Path(a.out), a.max_context, Path(a.dir))

    results = {}
    for t in types:
        for b in a.backends.split(","):
            kls, agrees, per_case, matched = [], [], {}, 0
            with ggml_compiler.Session(paths[t]["all"], backend=b, n_threads=6) as s:
                g = Generator(s)
                for c in cases:
                    g.reset()
                    g.encode(**c["image"])
                    # The prompt, then the answer in its own calls, as mtmd-logits feeds them: on CUDA,
                    # how tokens are batched changes cuBLAS's f16-accumulation error (Qwen3-VL F16: KL
                    # 8.3e-4 with the answer inside 512-token chunks, 5.1e-4 fed after the prompt).
                    n = c["meta"]["prompt"]
                    out = np.concatenate([g.feed(c["tokens"][:n])[-1:], g.feed(c["tokens"][n:])])
                    kl, agree = token_stats(out, np.load(c["dir"] / "ref.npy"))
                    kls.append(kl)
                    agrees.append(agree)
                    per_case[c["name"]] = float(kl.mean())
            st = stats_of(np.concatenate(kls), np.concatenate(agrees))
            with ggml_compiler.Session(paths[t]["last"], backend=b, n_threads=6) as s:
                g = Generator(s)
                for c in cases:
                    gen = g.generate(c["prompt"], max_new=a.gen, stop=[], **c["image"])
                    greedy = c["greedy"]
                    matched += next((i for i, (x, y) in enumerate(zip(gen, greedy)) if x != y), min(len(gen), len(greedy)))
                speed = timings(g, cases[0]["image"], cases[0]["prompt"], a.gen, a.reps)
            total = a.gen * len(cases)
            results[f"{t}/{b}"] = {**st.__dict__, "greedy_match": matched, "greedy_total": total, "cases": per_case,
                                   **speed}
            lb = base[f"{t}/{b}"]
            print(f"{t:>7} {b:<4} KL {st.kl_mean:.2e} ({lb['kl_mean']:.2e}, x{st.kl_mean / lb['kl_mean']:.2f})  "
                  f"top-1 {100 * st.top1:.1f}% ({100 * lb['top1']:.1f}%)  greedy {matched}/{total} ({lb['greedy_match']})  "
                  f"encode {speed['encode_ms']:.0f} ms ({lb['encode_ms']:.0f}, x{speed['encode_ms'] / lb['encode_ms']:.2f})  "
                  f"prompt {speed['prompt_ms']:.0f} ms ({lb['prompt_ms']:.0f}, x{speed['prompt_ms'] / lb['prompt_ms']:.2f})  "
                  f"tg {speed['tg']:.1f} ({lb['tg']:.1f}, x{lb['tg'] / speed['tg']:.2f})", flush=True)
    path = d / f"ggml-compiler-{a.opt}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
