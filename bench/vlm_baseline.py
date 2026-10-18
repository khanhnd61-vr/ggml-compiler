"""llama.cpp baseline for a VLM: accuracy against transformers f32, and speed, on a set of image prompts.

    uv run --with scikit-image python bench/vlm_baseline.py --fetch-images   # once: writes <dir>/images
    uv run python bench/vlm_baseline.py SNAPSHOT_DIR smolvlm2-500m

Each case is an image and "Describe this image in detail.", made by the model's own processor:
llama.cpp's tools/mtmd/test-1.jpeg and six of scikit-image's sample photos (public domain / CC0).
transformers f32 answers each greedily (up to --answer tokens); accuracy is KL / top-1 over the
answers, teacher-forced, pooled over every case's positions, so one ambiguous token cannot decide
it. llama.cpp runs through bench/llama/mtmd_logits.cpp (its clip encoder from the mmproj GGUF, fed
the processor's pixels). Greedy decoding: the first --gen tokens of each case, matched tokens summed.
Timings: the first case (test-1.jpeg) only, as in earlier measurements.

Reads `<dir>/<name>-{bf16,f16,q8_0,q4_k_m}.gguf` and `<dir>/<name>-mmproj-f16.gguf`
(vlm/convert.sh). Writes `<dir>/<name>/<case>/`: pixels.npy (the processor's pixel_values) and
pixels.bin (the same pixels as mtmd-logits reads them: SmolVLM's tiles [tiles, 3, S, S], Qwen3-VL's
patches put back into one image [3, H, W]), prompt.bin, tokens.bin (prompt + answer), ref.npy
(transformers' logits over the answer), answer.npy, meta.json; and `<dir>/<name>/cases.json`,
baseline.json; prints a table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ggmlc.verify.parity import stats_of, token_stats

TEST1 = Path(__file__).parents[1] / ".deps" / "llama.cpp-b11223" / "tools" / "mtmd" / "test-1.jpeg"
SAMPLES = ["astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry"]
CASES = ["test-1"] + SAMPLES
QUESTION = "Describe this image in detail."
TYPES = ["bf16", "f16", "q8_0", "q4_k_m"]


def fetch_images(d: Path) -> None:
    from PIL import Image
    from skimage import data

    (d / "images").mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        Image.fromarray(getattr(data, name)()).convert("RGB").save(d / "images" / f"{name}.png")


def image_path(d: Path, case: str) -> Path:
    return TEST1 if case == "test-1" else d / "images" / f"{case}.png"


def reference(snapshot: str, d: Path, out: Path, n_answer: int) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if (out / "ref.npy").exists() and not (out / "test-1").exists():  # the single-case layout: test-1
        (out / "test-1").mkdir()
        for f in out.iterdir():
            if f.is_file() and f.name != "cases.json" and not f.name.startswith(("ggmlc-", "baseline")):
                shutil.move(str(f), out / "test-1" / f.name)
    todo = [c for c in CASES if not (out / c / "ref.npy").exists()]
    if todo:
        proc = AutoProcessor.from_pretrained(snapshot)
        model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32,
                                                            attn_implementation="eager").eval()
        for case in todo:
            one(model, proc, image_path(d, case), out / case, n_answer)
            print(f"reference {case} done", flush=True)
    (out / "cases.json").write_text(json.dumps(CASES))


def one(model, proc, image: Path, out: Path, n_answer: int) -> None:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    inp = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True),
               images=[Image.open(image).convert("RGB")], return_tensors="pt")
    prompt = inp["input_ids"][0]
    tok = model.config.image_token_id
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=n_answer, do_sample=False)[0, len(prompt):]
        seq = torch.cat([prompt, g[:-1]])[None]
        full = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        if "mm_token_type_ids" in inp:
            full["mm_token_type_ids"] = (seq == tok).int()
        logits = model(input_ids=seq, attention_mask=torch.ones_like(seq), **full).logits[0, len(prompt) - 1:]
    px = inp["pixel_values"].numpy().astype(np.float32)
    meta = {"image": image.name, "image_token": tok, "prompt": len(prompt), "answer": len(g), "text": proc.decode(g)}
    if "image_grid_thw" in inp:  # Qwen-VL: [n, C*T*P*P] patches in 2 x 2 merge blocks -> [C, H, W]
        t, gh, gw = inp["image_grid_thw"][0].tolist()
        vc = model.config.vision_config
        m, P, C, T = vc.spatial_merge_size, vc.patch_size, vc.in_channels, vc.temporal_patch_size
        img = px.reshape(gh // m, gw // m, m, m, C, T, P, P)[:, :, :, :, :, 0]
        img = img.transpose(4, 0, 2, 5, 1, 3, 6).reshape(C, gh * P, gw * P)
        np.ascontiguousarray(img).tofile(out / "pixels.bin")
        meta.update(kind="qwen3_vl", grid=[t, gh, gw], width=gw * P, height=gh * P)
    else:  # SmolVLM: [1, tiles, 3, S, S]
        px = px[0]
        px.tofile(out / "pixels.bin")
        meta.update(kind="tiles", tile=int(px.shape[-1]), tiles=int(px.shape[0]))
    np.save(out / "pixels.npy", px)
    np.asarray(prompt, np.int32).tofile(out / "prompt.bin")
    np.asarray(seq[0], np.int32).tofile(out / "tokens.bin")
    np.save(out / "ref.npy", logits.numpy())
    np.save(out / "answer.npy", g.numpy().astype(np.int32))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_case(case_dir: Path) -> dict:
    meta = {"kind": "tiles", **json.loads((case_dir / "meta.json").read_text())}
    if meta["kind"] == "tiles" and "tiles" not in meta:
        meta["tiles"] = np.fromfile(case_dir / "pixels.bin", np.float32).size // (3 * meta["tile"] ** 2)
    return meta


def mtmd_logits(llama: Path, backend: str, gguf: Path, mmproj: Path, out: Path, meta: dict, gen: int = 0,
                reps: int = 0) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Logits over the answer (gen=0), or greedy tokens and timings after the prompt (gen > 0)."""
    dst = out / "mtmd-logits.bin"
    tokens = out / ("prompt.bin" if gen else "tokens.bin")
    cmd = [str(llama / f"logits-{backend}" / "mtmd-logits"), str(gguf), str(mmproj), str(out / "pixels.bin"),
           str(tokens), str(dst), "--image-token", str(meta["image_token"]),
           *(["--image", f"{meta['width']}x{meta['height']}"] if meta["kind"] == "qwen3_vl" else ["--size", str(meta["tile"])]),
           "--from", str(meta["prompt"] - 1), "--ngl", "99" if backend == "cuda" else "0",
           "--gen", str(gen), "--reps", str(reps)]
    subprocess.run(cmd, check=True, capture_output=True)
    timing = json.loads(Path(str(dst) + ".json").read_text())
    logits = gen_ids = None
    if gen:
        gen_ids = np.fromfile(str(dst) + ".gen", dtype=np.int32)
        Path(str(dst) + ".gen").unlink()
    else:
        logits = np.fromfile(dst, dtype=np.float32).reshape(meta["answer"], -1)
    dst.unlink()
    Path(str(dst) + ".json").unlink()
    return logits, gen_ids, timing


def greedy_match(gen, greedy) -> int:
    """Tokens of `gen` equal to the reference's greedy answer before the first difference."""
    gen, greedy = list(gen), list(greedy)
    return next((i for i, (x, y) in enumerate(zip(gen, greedy)) if x != y), min(len(gen), len(greedy)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", nargs="?")
    p.add_argument("name", nargs="?")
    p.add_argument("--dir", default="/mnt/khanh/ggmlc-bench/vlm")
    p.add_argument("--llama", default="/mnt/khanh/ggmlc-bench/llama.cpp")
    p.add_argument("--answer", type=int, default=256)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--backends", default="cpu,cuda")
    p.add_argument("--types", default=",".join(TYPES))
    p.add_argument("--fetch-images", action="store_true", help="write the sample images to <dir>/images and exit")
    p.add_argument("--reference-only", action="store_true", help="compute the transformers references and exit")
    a = p.parse_args()
    d, llama = Path(a.dir), Path(a.llama)
    if a.fetch_images:
        fetch_images(d)
        return 0
    out = d / a.name
    out.mkdir(parents=True, exist_ok=True)
    reference(a.snapshot, d, out, a.answer)
    if a.reference_only:
        return 0
    cases = [(c, load_case(out / c)) for c in CASES]
    mmproj = d / f"{a.name}-mmproj-f16.gguf"
    path = out / "baseline.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    n_pos = sum(m["answer"] for _, m in cases)
    print(f"{len(cases)} cases, {n_pos} answer positions; timings on {cases[0][0]} ({cases[0][1]['prompt']}-token prompt)",
          flush=True)
    for t in a.types.split(","):
        gguf = d / f"{a.name}-{t}.gguf"
        for b in a.backends.split(","):
            kls, agrees, per_case, matched, speed = [], [], {}, 0, {}
            for k, (case, meta) in enumerate(cases):
                cd = out / case
                logits, _, _ = mtmd_logits(llama, b, gguf, mmproj, cd, meta)
                kl, agree = token_stats(logits, np.load(cd / "ref.npy"))
                kls.append(kl)
                agrees.append(agree)
                per_case[case] = float(kl.mean())
                _, gen, timing = mtmd_logits(llama, b, gguf, mmproj, cd, meta, gen=a.gen, reps=a.reps if k == 0 else 0)
                matched += greedy_match(gen, np.load(cd / "answer.npy")[: a.gen])
                if k == 0:
                    speed = {"encode_ms": timing["encode_ms"], "prompt_ms": timing["prompt_ms"],
                             "tg": timing["gen"] / timing["gen_ms"] * 1e3}
            s = stats_of(np.concatenate(kls), np.concatenate(agrees))
            results[f"{t}/{b}"] = {**s.__dict__, "greedy_match": matched, "greedy_total": a.gen * len(cases),
                                   "cases": per_case, **speed}
            print(f"{t:>7} {b:<4}  {s.row()}  greedy {matched}/{a.gen * len(cases)}  "
                  f"encode {speed['encode_ms']:7.1f} ms  prompt {speed['prompt_ms']:7.1f} ms  tg {speed['tg']:6.1f} t/s",
                  flush=True)
    path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
