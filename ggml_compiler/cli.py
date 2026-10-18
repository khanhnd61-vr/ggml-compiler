"""ggml-compiler command line."""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TOL = {"cpu": 1e-4, "cuda": 1e-2}


def _parse_input(spec: str) -> tuple[str, list[int]]:
    """'image:1x3x480x640:f32' -> ('image', [1, 3, 480, 640]); the dtype comes from the model."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"ggml-compiler: --input wants NAME:SHAPE[:DTYPE], got {spec!r}")
    return parts[0], [int(d) for d in parts[1].split("x")]


def _load_program(source: str) -> Any:
    from . import api

    if source.endswith(".onnx"):
        return source
    if ":" not in source:
        raise SystemExit("ggml-compiler: SOURCE must be model.onnx or path/to/script.py:name")
    path, name = source.rsplit(":", 1)
    ns = runpy.run_path(path, run_name="__ggml_compiler__")
    if name not in ns:
        raise SystemExit(f"ggml-compiler: {path} defines no {name!r}")
    obj = ns[name]
    if callable(obj) and not isinstance(obj, api.Program):
        obj = obj()
    if not isinstance(obj, api.Program):
        raise SystemExit(f"ggml-compiler: {source} is a {type(obj).__name__}, expected a ggml_compiler.Program")
    return obj


def _read_gguf(path: str) -> tuple[dict[str, Any], dict[str, str], list[Any]]:
    import gguf

    r = gguf.GGUFReader(path)

    def text(key: str) -> str:
        f = r.fields.get(key)
        if f is None:
            raise SystemExit(f"ggml-compiler: {path} has no {key}; it was not written by ggml-compiler compile")
        return bytes(f.parts[f.data[0]]).decode()

    prog = json.loads(text("ggml_compiler.program"))
    if "ggml_compiler.options" in r.fields:  # written since the optimization levels
        prog["options"] = json.loads(text("ggml_compiler.options"))
    return prog, json.loads(text("ggml_compiler.tensor_names")), list(r.tensors)


def _inputs_for(model: Path, inputs: str | None, sess: Any) -> dict[str, np.ndarray]:
    path = Path(inputs) if inputs else model.parent / "example_inputs.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if inputs:
        raise SystemExit(f"ggml-compiler: {path} not found")
    return {t.name: np.zeros(t.shape, dtype=np.float32) for t in sess.inputs}


def _parse_flags(specs: list[str] | None) -> dict[str, bool] | None:
    """-f NAME turns a flag on, -f no-NAME turns it off."""
    if not specs:
        return None
    return {s[3:] if s.startswith("no-") else s: not s.startswith("no-") for s in specs}


def cmd_compile(a: argparse.Namespace) -> int:
    from .compiler import compile

    program = _load_program(a.source)
    shapes = dict(_parse_input(x) for x in a.input) if a.input else None
    art = compile(program, out_dir=a.out, emit=a.emit.split(","), dtype_policy=a.policy, input_shapes=shapes,
                  prequant=a.prequant, opt=a.opt, flags=_parse_flags(a.flag), precision=a.precision,
                  attention=a.attention, name=a.name,
                  backends=a.backends.split(",") if a.backends else None)
    print(art.report.summary())
    print(f"wrote {art.gguf}")
    return 0


def cmd_inspect(a: argparse.Namespace) -> int:
    prog, names, tensors = _read_gguf(a.model)
    print(f"{a.model}: format {prog['format_version']}")
    if "options" in prog:
        o = prog["options"]
        print(f"options: {o['level']}  " + " ".join(("" if v else "no-") + f for f, v in o["flags"].items()))
        for k, v in o["specialized"].items():
            print(f"  specialized {k}: {v}")
    from .ir import sym

    def shape(dims: list) -> str:
        return "[" + ", ".join(str(sym.from_json(d)) for d in dims) + "]"

    for sy in prog.get("symbols", []):
        print(f"symbol {sy['name']} in [{sy['min']}, {sy['max']}]")
    print("inputs:")
    values = {v["name"]: v for v in prog["values"]}
    for n in prog["inputs"]:
        print(f"  {n:<32} {values[n]['type']} {shape(values[n]['shape'])}")
    print("outputs:")
    for n in prog["outputs"]:
        print(f"  {n:<32} {values[n]['type']} {shape(values[n]['shape'])}")
    print("driver:")

    def show(ops: list[dict], indent: str) -> None:
        for d in ops:
            if d["op"] == "call":
                print(f"{indent}call {d['stage']}({', '.join(d['args'])}) -> {', '.join(d['out'])}")
                continue
            extra = [f"carry {dst} <- {src}" for dst, src in d["carry"]]
            extra += [f"scalar {sc['value']}" for sc in d["scalars"]]
            extra += [f"slice {dst} <- {src}[i]" for dst, src in d.get("slices", [])]
            print(f"{indent}loop x{d['n']}: {', '.join(extra)}")
            show(d["body"], indent + "  ")

    show(prog["driver"], "  ")
    for s in prog["stages"]:
        ops = Counter(n["op"] for n in s["nodes"])
        print(f"stage {s['name']}: {len(s['nodes'])} nodes")
        print("  " + "  ".join(f"{op}x{k}" for op, k in ops.most_common()))
    total = sum(int(t.n_bytes) for t in tensors)
    print(f"tensors: {len(tensors)}, {total / 1e6:.2f} MB")
    if a.tensors:
        for t in tensors:
            print(f"  {t.name:<8} {names.get(t.name, '(constant)'):<56} {t.tensor_type.name:<5} "
                  f"{list(reversed([int(d) for d in t.shape]))}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .runtime import Session

    with Session(a.model, backend=a.backend, n_threads=a.threads) as s:
        out = s.run(**_inputs_for(Path(a.model), a.inputs, s))
        np.savez(a.out, **out)
        print(f"{s.backend_name}: {s.stats['ms_total']:.2f} ms, wrote {a.out}")
    return 0


def _profile(a: argparse.Namespace) -> int:
    """One run, node by node with a sync after each; prints where the time goes."""
    import os
    import tempfile

    prog, _, _ = _read_gguf(a.model)
    meta = {(st["name"], n["id"]): (n["op"], n["loc"]) for st in prog["stages"] for n in st["nodes"]}
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
        path = f.name
    os.environ["GGML_COMPILER_PROFILE"] = path  # read by the runtime when it first runs a graph
    from .runtime import Session

    with Session(a.model, backend=a.backend, n_threads=a.threads) as s:
        inputs = _inputs_for(Path(a.model), a.inputs, s)
        s.run(**inputs)  # warm-up, discarded
        Path(path).write_text("")
        s.run(**inputs)
    by_op: Counter = Counter()
    by_loc: Counter = Counter()
    total = 0.0
    for line in Path(path).read_text().splitlines():
        stage, node, gop, ms = line.split("\t")
        op, loc = meta.get((stage, node), (f"[{gop}]", ""))
        ms = float(ms)
        total += ms
        by_op[(stage, op)] += ms
        by_loc[(stage, op, loc)] += ms
    os.unlink(path)
    print(f"{s.backend_name}: {total:.1f} ms summed over nodes (sync after each node inflates this)")
    print("by op:")
    for (stage, op), ms in by_op.most_common(a.top):
        print(f"  {ms:9.2f} ms {100 * ms / total:5.1f}%  {stage:<8} {op}")
    print("by source line:")
    for (stage, op, loc), ms in by_loc.most_common(a.top):
        print(f"  {ms:9.2f} ms {100 * ms / total:5.1f}%  {stage:<8} {op:<10} {loc}")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    if a.profile:
        return _profile(a)
    from .runtime import Session

    with Session(a.model, backend=a.backend, n_threads=a.threads) as s:
        inputs = _inputs_for(Path(a.model), a.inputs, s)
        for _ in range(a.warmup):
            s.run(**inputs)
        wall, stages = [], {}
        for _ in range(a.iters):
            t0 = time.perf_counter()
            s.run(**inputs)
            wall.append((time.perf_counter() - t0) * 1e3)
            for name, st in s.stats["stages"].items():
                stages.setdefault(name, []).append(st["ms"])
        q = statistics.quantiles(wall, n=10) if len(wall) > 1 else wall * 9
        print(f"{s.backend_name}, {a.iters} runs after {a.warmup} warmup")
        print(f"  end to end  median {statistics.median(wall):8.3f} ms   min {min(wall):8.3f}   p90 {q[-1]:8.3f}")
        for name, ms in stages.items():
            print(f"  {name:<11} median {statistics.median(ms):8.3f} ms")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from .runtime import Session
    from .verify.parity import compare

    model = Path(a.model)
    ref_path = Path(a.ref) if a.ref else model.parent / "reference_outputs.npz"
    if not ref_path.exists():
        raise SystemExit(f"ggml-compiler: no reference outputs at {ref_path}")
    with np.load(ref_path) as z:
        refs = {k: z[k] for k in z.files}
    with Session(model, backend=a.backend, n_threads=a.threads) as s:
        out = s.run(**_inputs_for(model, a.inputs, s))
        backend = s.backend_name.lower().rstrip("0123456789")
    tol = a.tol if a.tol is not None else DEFAULT_TOL.get(backend, 1e-2)
    diffs = compare(out, refs)
    for d in diffs:
        print(f"  {d.name:<32} max|d| {d.max_abs:.3e}  max|ref| {d.ref_range:.3e}  scaled {d.scaled:.3e}"
              f"  {'ok' if d.ok(tol) else 'FAIL'}")
    passed = all(d.ok(tol) for d in diffs)  # not max(): max(0.0, nan) is 0.0
    worst = max((d.scaled for d in diffs), key=lambda x: (x != x, x), default=0.0)
    print(f"{'PASS' if passed else 'FAIL'}: worst scaled error {worst:.3e}, tolerance {tol:.0e} ({s.backend_name})")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ggml-compiler", description="Compile PyTorch models into ggml programs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile a ggml_compiler.Program into an output directory")
    c.add_argument("source",
                   help="model.onnx, or path/to/script.py:name where name is a ggml_compiler.Program or returns one")
    c.add_argument("--out", required=True, help="output directory")
    c.add_argument("--emit", default="gguf", help="gguf, cpp, or gguf,cpp")
    c.add_argument("--policy", help="weight dtype policy (YAML)")
    c.add_argument("--prequant", choices=["modelopt"], help="import a ModelOpt-quantized model")
    c.add_argument("-O", "--opt", choices=["O0", "O1", "O2", "O3"], type=lambda s: s if s.startswith("O") else f"O{s}",
                   help="optimization level (default O1; O2 adds flash attention and table GELU, ~1e-3)")
    c.add_argument("-f", "--flag", action="append", metavar="[no-]NAME",
                   help="turn one flag on or off on top of the level, e.g. -f no-hoist_invariants")
    c.add_argument("--precision", choices=["exact", "fast"], help="older spelling: exact is -O1, fast is -O2")
    c.add_argument("--attention", choices=["exact", "flash"], help="set the flash_attention flag")
    c.add_argument("--backends", help="comma-separated backends to load the result on, e.g. cpu,cuda")
    c.add_argument("--input", action="append", help="ONNX: pin an input's shape, NAME:SHAPE[:DTYPE], e.g. x:1x3x224x224:f32")
    c.add_argument("--name", help="name for the generated C++ (default: the output directory's)")
    c.set_defaults(fn=cmd_compile)

    i = sub.add_parser("inspect", help="show a compiled model's I/O, driver, stages and tensors")
    i.add_argument("model")
    i.add_argument("--tensors", action="store_true", help="list every tensor")
    i.set_defaults(fn=cmd_inspect)

    def runtime_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("model")
        sp.add_argument("--inputs", help=".npz of inputs (default: example_inputs.npz next to the model)")
        sp.add_argument("--backend", default="auto", choices=["cpu", "cuda", "auto"])
        sp.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")

    r = sub.add_parser("run", help="run a compiled model on inputs from an .npz")
    runtime_args(r)
    r.add_argument("--out", required=True, help="where to write outputs (.npz)")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench", help="time a compiled model")
    runtime_args(b)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--iters", type=int, default=50)
    b.add_argument("--profile", action="store_true", help="time every node once and print the hot spots")
    b.add_argument("--top", type=int, default=15, help="rows to print with --profile")
    b.set_defaults(fn=cmd_bench)

    v = sub.add_parser("verify", help="compare a compiled model with the PyTorch outputs saved at compile time")
    runtime_args(v)
    v.add_argument("--ref", help=".npz of reference outputs (default: reference_outputs.npz next to the model)")
    v.add_argument("--tol", type=float, help="tolerance on max|d| / max(1, max|ref|) (default 1e-4 CPU, 1e-2 CUDA)")
    v.set_defaults(fn=cmd_verify)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
