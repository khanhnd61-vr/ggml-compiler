"""ggml_compiler.compile: PyTorch program in, output directory with model.gguf and friends out."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import api
from .ir import Const, Node, Weight
from .ir.program import CompileError, ProgramIR
from .options import resolve


@dataclass
class Report:
    data: dict[str, Any]

    def summary(self) -> str:
        d = self.data
        w = d["weights"]
        lines = [f"ggml-compiler: {len(d['stages'])} stage(s), {w['count']} weights ({w['bytes'] / 1e6:.1f} MB), "
                 f"{d['consts']['count']} constants ({d['consts']['bytes'] / 1e6:.1f} MB)"]
        for name, s in d["stages"].items():
            ops = "  ".join(f"{op}x{n}" for op, n in sorted(s["ops"].items(), key=lambda kv: -kv[1]))
            lines.append(f"  stage {name}: {s['nodes']} nodes  {ops}")
        lines.append("  inputs:  " + ", ".join(f"{k}{v['shape']}" for k, v in d["inputs"].items()))
        lines.append("  outputs: " + ", ".join(f"{k}{v['shape']}" for k, v in d["outputs"].items()))
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.data, indent=1))


@dataclass
class Artifact:
    out_dir: Path
    gguf: Path
    report: Report
    program: ProgramIR


def make_report(prog: ProgramIR) -> Report:
    consts: dict[tuple, Const] = {}  # by content, as the emitter stores them
    stages = {}
    for name, s in prog.stages.items():
        for v in s.leaves():
            if isinstance(v, Const):
                consts[(v.type, v.data.numpy().tobytes())] = v
        stages[name] = {"nodes": len(s.nodes), "ops": dict(Counter(n.op for n in s.nodes))}
    unique = {id(v): v for s in prog.stages.values() for v in s.leaves() if isinstance(v, Weight)}
    return Report({
        "stages": stages,
        "weights": {
            "count": len(unique),
            "bytes": sum(w.data.numel() * w.data.element_size() for w in unique.values()),
            "tensors": {w.fqn: {"shape": list(w.shape), "dtype": w.type.dtype} for w in unique.values()},
        },
        "consts": {"count": len(consts), "bytes": sum(c.data.numel() * c.data.element_size() for c in consts.values())},
        "inputs": {n: {"shape": _shape(prog.values[n].shape), "dtype": prog.values[n].dtype} for n in prog.inputs},
        "outputs": {n: {"shape": _shape(prog.values[n].shape), "dtype": prog.values[n].dtype} for n in prog.outputs},
        "symbols": {k: {"min": lo, "max": hi} for k, (lo, hi) in prog.symbols.items()},
        "entries": {e: {"inputs": {n: _shape(prog.values[n].shape) for n in en.inputs},
                        "outputs": {n: _shape(prog.values[n].shape) for n in en.outputs}}
                    for e, en in prog.entries.items()},
        "state": {n: {"shape": _shape(prog.values[n].shape), "dtype": prog.values[n].dtype} for n in prog.state},
    })


def _shape(shape: tuple) -> list:
    """A shape for JSON: symbolic dims as their expression, e.g. "2*seq"."""
    return [d if isinstance(d, int) else str(d) for d in shape]


def _stored_bytes(prog: ProgramIR) -> int:
    sizes = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32,
             "q5_1": 24 / 32, "iq4_nl": 18 / 32, "q4_k": 144 / 256, "q5_k": 176 / 256, "q6_k": 210 / 256}
    ws = {id(w): w for s in prog.stages.values() for w in s.leaves() if isinstance(w, Weight)}
    return int(sum(w.data.numel() * sizes[w.storage] for w in ws.values()))


def _prequant_report(found: Any, decisions: dict, gguf_path: Path, ref: Any) -> dict[str, Any]:
    """Per imported weight: source quantizer, stored type, error vs the fake-quant weight. Plus the
    end-to-end error of the compiled model (CPU) against the fake-quant PyTorch model."""
    from .runtime import Session
    from .verify.parity import compare

    weights = {}
    for fqn, w in found.weights.items():
        d = decisions.get(fqn, {})
        weights[fqn] = {**w, "stored": d.get("dtype"), "rel_err_vs_fake_quant": d.get("rel_err")}
    with Session(gguf_path, backend="cpu") as s:
        out = s.run(**{k: v.numpy() for k, v in ref.inputs.items()})
    diffs = compare(out, {k: v.numpy() for k, v in ref.outputs.items()})
    return {"source": "modelopt", "weights": weights, "dropped_activation_quantizers": found.dropped,
            "output_error_vs_fake_quant": {d.name: {"max_abs": d.max_abs, "scaled": d.scaled} for d in diffs}}


def _emit_cpp(prog: ProgramIR, ref: Any, out: Path, name: str | None, extra: dict[str, int] | None = None) -> None:
    import re

    from .emit.cpp import write_cpp
    from .emit.gguf_runtime import build_program_json

    program, tensors = build_program_json(prog)  # same naming as the GGUF just written
    fqns = {k: v.fqn for k, v in tensors.items() if isinstance(v, Weight)}
    name = re.sub(r"[^0-9A-Za-z_]", "_", name or out.resolve().name) or "model"
    if name[0].isdigit():
        name = "m_" + name
    from .ir import sym

    binding = sym.solve((prog.values[n].shape, tuple(ref.inputs[n].shape)) for n in prog.all_inputs())
    binding.update(extra or {})  # symbols no input carries (a loop's trip count): their examples
    write_cpp(program, fqns, out / "cpp", name, binding)
    # The check program's inputs: raw bytes, back to back, every entry's in order.
    with open(out / "example_inputs.bin", "wb") as f:
        for n in prog.all_inputs():
            f.write(np.ascontiguousarray(ref.inputs[n].numpy()).tobytes())


def _matmul_acc(prog: Any, program: api.Program) -> None:
    """Mark the matmuls of stages declared Stage(matmul_acc="f32") (and of the invariant stages
    hoisted out of them) to accumulate in f32."""
    for name, st in program.stages.items():
        if st.matmul_acc is None:
            continue
        if st.matmul_acc != "f32":
            raise ValueError(f"stage {name!r}: matmul_acc must be None or 'f32'")
        for sname in (name, f"{name}.invariant"):
            for n in prog.stages[sname].nodes if sname in prog.stages else []:
                if n.op == "mul_mat":
                    n.attrs["acc"] = "f32"


def _attention_acc(prog: Any) -> None:
    """The two matmuls of unfused attention, Q·Kᵀ (read by a softmax) and P·V (reading it),
    accumulate in f32, as flash attention does and llama.cpp asks of Q·Kᵀ. Over an f16 KV cache,
    CUDA would otherwise run them through cuBLAS with f16 accumulation (Gemma 4's 512-wide heads,
    too wide to fuse on every GPU). Only 16-bit operands are affected: an f32 matmul runs the same
    either way."""
    through = {"add", "scale", "cast", "reshape", "permute", "cont"}
    views = {"cast", "reshape", "permute", "cont"}

    def scores(v: Any) -> Node | None:
        while isinstance(v, Node) and v.op in through:
            if v.op == "add":  # the mask is added to the scores, in either order
                return next(filter(None, (scores(x) for x in v.inputs)), None)
            v = v.inputs[0]
        return v if isinstance(v, Node) and v.op == "mul_mat" else None

    for st in prog.stages.values():
        users = st.users()

        def readers(v: Any) -> list[Node]:
            out = []
            for u in users.get(id(v), []):
                out += readers(u) if u.op in views else [u] if u.op == "mul_mat" else []
            return out

        for n in st.nodes:
            if n.op == "soft_max":
                for m in filter(None, [scores(n.inputs[0]), *readers(n)]):
                    m.attrs["acc"] = "f32"


def compile(
    model: torch.nn.Module | api.Program | str | Path,
    example_inputs: Sequence[Any] | dict[str, Any] | None = None,
    out_dir: str | Path = "out",
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    emit: Sequence[str] = ("gguf",),
    opt: str | None = None,
    flags: dict[str, bool] | None = None,
    attention: str | None = None,
    precision: str | None = None,
    dtype_policy: Any = None,
    prequant: Any = None,
    backends: Sequence[str] | None = None,
    name: str | None = None,
    input_shapes: dict[str, Sequence[int]] | None = None,
) -> Artifact:
    """Compile a module (one stage), a Program, or an .onnx file into `out_dir`.

    For ONNX, `input_shapes` pins dynamic input dims and `example_inputs` (optional; seeded random
    data otherwise) feeds onnxruntime for the reference outputs.

    Writes model.gguf, report.json, example_inputs.npz and reference_outputs.npz. With
    `backends`, also loads the result on each backend so unsupported nodes fail here.

    `opt` is the optimization level, "O0" to "O3" (default "O1": numerics-preserving rewrites,
    f32 arithmetic). "O2" adds ~1e-3 approximations for speed: flash attention (K/V in f16) and
    ggml's table GELU on CPU. `flags` toggles single flags on top of the level (ggml_compiler.FLAGS).
    precision="exact" | "fast" is the older spelling of O1 | O2; attention="flash" | "exact" sets
    the flash_attention flag.
    """
    from . import __version__
    from .emit.gguf_runtime import write_gguf
    from .frontend.torch_export import export_program
    from .passes import run_pipeline

    options = resolve(opt, flags, precision=precision, attention=attention)
    if set(emit) - {"gguf", "cpp"}:
        raise ValueError("emit takes 'gguf' and 'cpp'")

    from .frontend.modelopt import ModelOpt, ModelOptImport
    from .quant import DTypePolicy, apply_policy

    if prequant == "modelopt":
        prequant = ModelOpt()
    if prequant is not None and not isinstance(prequant, ModelOpt):
        raise TypeError("prequant must be ggml_compiler.ModelOpt(...) or 'modelopt'")
    found = ModelOptImport()

    if isinstance(model, (str, Path)):
        import onnx

        from .frontend.onnx import export_onnx

        if Path(model).suffix != ".onnx":
            raise ValueError(f"{model}: only .onnx files compile from a path")
        if prequant is not None:
            raise ValueError("prequant applies to PyTorch models (ONNX QDQ import is not supported)")
        shapes = {k: list(v) for k, v in (input_shapes or {}).items()}
        prog, ref = export_onnx(model, example_inputs, shapes)
        source = {"frontend": "onnx", "onnx": onnx.__version__, "model": Path(model).name}
    else:
        if isinstance(model, api.Program):
            if example_inputs is not None or input_names or output_names:
                raise TypeError("a Program carries its own example inputs and names")
            program = model
        else:
            if example_inputs is None:
                raise TypeError("example_inputs is required when compiling a module")
            program = api.Program.single(model, example_inputs, input_names, output_names)
        context = prequant.prepare(program.root, found) if prequant is not None else None
        prog, ref = export_program(program, export_context=context)
        source = {"frontend": "torch.export", "torch": torch.__version__,
                  "root": type(program.root).__name__ if program.root is not None else None}
    source["ggml-compiler"] = __version__
    run_pipeline(prog, options)
    _attention_acc(prog)
    if not isinstance(model, (str, Path)):
        _matmul_acc(prog, program)
    specialized = dict(program.specialized) if not isinstance(model, (str, Path)) else {}
    meta = {**options.to_json(), "specialized": specialized}

    if isinstance(dtype_policy, (str, Path)):
        dtype_policy = DTypePolicy.from_yaml(dtype_policy)
    elif isinstance(dtype_policy, dict):
        dtype_policy = DTypePolicy.from_dict(dtype_policy)
    dtype_policy = dtype_policy or DTypePolicy()
    if found.weights:  # imported weights take their mapped type ahead of the user's rules
        pinned = [(p, w["dtype"]) for fqn, w in found.weights.items() for p in (fqn, f"{fqn}:*")]
        dtype_policy = DTypePolicy(rules=pinned + dtype_policy.rules, float=dtype_policy.float,
                                   fallback=dtype_policy.fallback)
    decisions = apply_policy(prog, dtype_policy)
    for fqn, scale in found.scales.items():
        if fqn in prog.weights:
            prog.weights[fqn].grid_scale = scale  # lets the emitter encode exactly

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gguf_path = out / "model.gguf"

    user_meta = dict(program.metadata) if not isinstance(model, (str, Path)) else {}
    quant_errors = write_gguf(prog, gguf_path, source, meta, user_meta)
    for fqn, err in quant_errors.items():
        decisions[fqn]["rel_err"] = err
    np.savez(out / "example_inputs.npz", **{k: v.numpy() for k, v in ref.inputs.items()})
    if "cpp" in emit:
        extra = {k: v[2] for k, v in program.symbols.items()} if not isinstance(model, (str, Path)) else {}
        _emit_cpp(prog, ref, out, name, extra)
    np.savez(out / "reference_outputs.npz", **{k: v.numpy() for k, v in ref.outputs.items()})
    report = make_report(prog)
    report.data["weights"]["tensors"] = decisions
    report.data["weights"]["stored_bytes"] = _stored_bytes(prog)
    report.data["options"] = meta
    if prequant is not None:
        report.data["prequant"] = _prequant_report(found, decisions, gguf_path, ref)
    report.save(out / "report.json")

    for backend in backends or ():
        from .runtime import GgmlCompilerRuntimeError, Session

        try:
            Session(gguf_path, backend=backend).close()
        except GgmlCompilerRuntimeError as e:
            raise CompileError(f"backend {backend}: {e}") from None

    return Artifact(out, gguf_path, report, prog)
