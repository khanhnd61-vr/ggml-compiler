"""Write a ProgramIR as a GGUF the runtime executes: weights, constants, and the program JSON.

This is where PyTorch order becomes ggml order. Shapes turn into innermost-first `ne`, and
dim-valued attributes into ggml axes, so the C++ runtime only dispatches and checks. A symbolic
dim is written as an expression tree (`ir.sym.to_json`) that the runtime evaluates per binding.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import gguf
import numpy as np

from ..ir import Const, Node, TensorType, Value, Weight, sym
from ..ir.ops import OPS
from ..ir.program import CallOp, CompileError, LoopOp, ProgramIR
from ..quant.types import ggml_name

FORMAT_VERSION = 3  # 2: loop slices; 3: symbolic dims, entry points, state
_FLT_MAX = 3.4028234663852886e38


def ne_of(t: TensorType) -> list[sym.Dim]:
    """ggml ne[] for a PyTorch-order type. Above 4-D the size-1 dims are dropped (the squeeze pass
    gives such stage parameters the same layout); the PyTorch shape stays what the user sees."""
    shape = t.shape if t.rank <= 4 else tuple(d for d in t.shape if d != 1)
    if len(shape) > 4:
        raise CompileError(f"{t} has more than 4 dims that are not 1")
    return list(reversed(shape)) + [1] * (4 - len(shape))


def _ggml_dim(d: int, rank: int) -> int:
    return rank - 1 - d


def _permute_axes(perm: list[int]) -> list[int]:
    """PyTorch permute (out dim i reads in dim perm[i]) to ggml_permute axes (in ggml dim j moves
    to ggml dim axes[j]), after padding both to 4 dims."""
    r = len(perm)
    pad = 4 - r
    full = list(range(pad)) + [p + pad for p in perm]
    axes = [0] * 4
    for i, src in enumerate(full):
        axes[3 - src] = 3 - i
    return axes


def _finite(x: float) -> float:
    if math.isinf(x):
        return _FLT_MAX if x > 0 else -_FLT_MAX
    return float(x)


def _params(n: Node) -> dict[str, Any]:
    a = n.attrs
    r = n.inputs[0].rank if n.inputs else 0
    if n.op == "permute":
        return {"axes": _permute_axes(list(a["perm"]))}
    if n.op == "concat":
        return {"dim": _ggml_dim(a["dim"], r)}
    if n.op == "view":
        return {"dim": _ggml_dim(a["dim"], r), "start": sym.to_json(a["start"])}
    if n.op == "soft_max":
        return {"scale": a.get("scale", 1.0), "max_bias": a.get("max_bias", 0.0)}
    if n.op == "flash_attn":
        return {"scale": a["scale"]}
    if n.op == "mul_mat":
        return {"acc": 1} if a.get("acc") == "f32" else {}
    if n.op == "arange":
        return {"start": sym.to_json(a["start"]), "stop": sym.to_json(a["stop"]), "step": a["step"]}
    if n.op == "rope":  # GGML_ROPE_TYPE_NEOX is 2
        return {"n_dims": a["n_dims"], "mode": 2 if a["mode"] == "neox" else 0, "freq_base": a["freq_base"],
                "freq_scale": a.get("freq_scale", 1.0), "attn_factor": a.get("attn_factor", 1.0)}
    if n.op in ("norm", "rms_norm"):
        return {"eps": a["eps"]}
    if n.op == "scale":
        return {"s": a["s"], "b": a.get("b", 0.0)}
    if n.op == "clamp":
        return {"min": _finite(a["min"]), "max": _finite(a["max"])}
    if n.op in ("im2col", "conv_2d"):  # ggml's dim 0 is width
        (sh, sw), (ph, pw), (dh, dw) = a["stride"], a["padding"], a["dilation"]
        return {"s0": sw, "s1": sh, "p0": pw, "p1": ph, "d0": dw, "d1": dh}
    if n.op == "pool_2d":
        (kh, kw), (sh, sw), (ph, pw) = a["kernel"], a["stride"], a["padding"]
        return {"op": a["op"], "k0": kw, "k1": kh, "s0": sw, "s1": sh, "p0": pw, "p1": ph}
    return {}


def _ggml_type(dtype: str) -> str:
    return {"f32": "f32", "f16": "f16", "bf16": "bf16", "i32": "i32"}[dtype]


def _dims(dims: Any) -> list[Any]:
    return [sym.to_json(d) for d in dims]


def _tensor_json(name: str, t: TensorType) -> dict[str, Any]:
    return {"name": name, "type": _ggml_type(t.dtype), "ne": _dims(ne_of(t)), "shape": _dims(t.shape)}


def _driver_json(op: CallOp | LoopOp) -> dict[str, Any]:
    if isinstance(op, CallOp):
        return {"op": "call", "stage": op.stage, "args": op.args, "out": op.out}
    return {"op": "loop", "n": op.n, "carry": [list(c) for c in op.carry],
            "scalars": [{"value": v, "values": t} for v, t in op.scalars],
            "slices": [list(sl) for sl in op.slices], "counters": [list(c) for c in op.counters],
            "bind": [[k, e] for k, e in op.bind], "masks": [list(m) for m in op.masks],
            "until": [[v, ids] for v, ids in op.until], "collect": [list(c) for c in op.collect],
            "body": [_driver_json(c) for c in op.body]}


def build_program_json(prog: ProgramIR) -> tuple[dict[str, Any], dict[str, Value]]:
    """Returns the program JSON and the GGUF tensors it refers to, by GGUF name."""
    tensors: dict[str, Value] = {}
    names: dict[int, str] = {}
    by_content: dict[tuple, str] = {}  # constants are stored once per distinct content
    n_w = n_c = 0

    def leaf_name(v: Value) -> str:
        nonlocal n_w, n_c
        if id(v) not in names:
            if isinstance(v, Weight):
                names[id(v)] = f"w.{n_w}"
                n_w += 1
            else:
                key = (v.type, hashlib.sha1(v.data.contiguous().numpy().tobytes()).hexdigest())
                if key not in by_content:
                    by_content[key] = f"c.{n_c}"
                    n_c += 1
                names[id(v)] = by_content[key]
            tensors[names[id(v)]] = v
        return names[id(v)]

    stages = []
    for stage in prog.stages.values():
        local: dict[int, str] = {}
        for i, p in enumerate(stage.params):
            local[id(p)] = f"a.{i}"
        nodes = []

        def ref(v: Value) -> str:
            if isinstance(v, (Weight, Const)):
                return leaf_name(v)
            return local[id(v)]

        stored: dict[int, str] = {}  # reshapes of a weight stored as f16/quantized: ggml keeps that type

        def stored_type(v: Value) -> str | None:
            if isinstance(v, Weight):
                return None if v.storage == "f32" else ggml_name(v.storage)
            return stored.get(id(v))

        for i, n in enumerate(stage.nodes):
            d = OPS[n.op]
            if d.tier != 1:
                raise CompileError(f"stage {stage.name!r}: tier-2 op {n.op} reached the emitter ({n.loc})")
            local[id(n)] = f"%{i}"
            if n.op == "reshape" and stored_type(n.inputs[0]) is not None:
                stored[id(n)] = stored_type(n.inputs[0])
            nodes.append({
                "id": f"%{i}",
                "op": n.op,
                "src": [ref(v) for v in n.inputs],
                "type": stored.get(id(n)) or _ggml_type(n.type.dtype),
                "ne": _dims(ne_of(n.type)),
                "p": _params(n),
                "loc": n.loc,
                "scope": n.scope,
            })
        outputs = []
        for v in stage.outputs:
            if isinstance(v, (Weight, Const)):
                # The runtime returns graph values only: copy a leaf through a node.
                local[id(v)] = f"%{len(nodes)}"
                nodes.append({"id": f"%{len(nodes)}", "op": "cont", "src": [leaf_name(v)],
                              "type": _ggml_type(v.type.dtype), "ne": _dims(ne_of(v.type)), "p": {}, "loc": ""})
            outputs.append(local[id(v)])
        stages.append({
            "name": stage.name,
            "params": [_tensor_json(f"a.{i}", p.type) for i, p in enumerate(stage.params)],
            "outputs": outputs,
            "nodes": nodes,
        })

    program = {
        "format_version": FORMAT_VERSION,
        "symbols": [{"name": k, "min": lo, "max": hi} for k, (lo, hi) in prog.symbols.items()],
        "values": [_tensor_json(name, t) for name, t in prog.values.items()],
        "inputs": prog.inputs,
        "outputs": prog.outputs,
        "stages": stages,
        "driver": [_driver_json(op) for op in prog.driver],
        "entries": [{"name": name, "inputs": e.inputs, "outputs": e.outputs,
                     "driver": [_driver_json(op) for op in e.driver]} for name, e in prog.entries.items()],
        "state": prog.state,
    }
    return program, tensors


def write_gguf(prog: ProgramIR, path: str | Path, source: dict[str, Any],
               options: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Write the GGUF. Returns, per weight stored in a lossy type, max|w - stored| / max|w|.

    Tensors are encoded and written one at a time, so a multi-GB model never has a second full
    copy of its weights in memory."""
    program, tensors = build_program_json(prog)
    errors: dict[str, float] = {}
    w = gguf.GGUFWriter(str(path), arch="ggmlc")
    w.add_uint32("ggmlc.format_version", FORMAT_VERSION)
    w.add_string("ggmlc.program", json.dumps(program, separators=(",", ":")))
    w.add_string(
        "ggmlc.tensor_names",
        json.dumps({k: v.fqn for k, v in tensors.items() if isinstance(v, Weight)}, separators=(",", ":")),
    )
    w.add_string("ggmlc.source", json.dumps(source, separators=(",", ":")))
    if options is not None:
        w.add_string("ggmlc.options", json.dumps(options, separators=(",", ":")))
    if metadata:
        w.add_string("ggmlc.meta", json.dumps(metadata, separators=(",", ":")))

    def array(v: Value) -> np.ndarray:
        return v.data.numpy() if v.data.dim() > 0 else v.data.reshape(1).numpy()

    def stored_as(v: Value, data: np.ndarray) -> str:
        storage = getattr(v, "storage", "f32")
        return "raw" if storage == "f32" or data.dtype != np.float32 else storage

    for name, v in tensors.items():
        data = array(v)
        kind = stored_as(v, data)
        if kind == "raw":
            w.add_tensor_info(name, data.shape, data.dtype, data.nbytes)
        elif kind == "f16":
            w.add_tensor_info(name, data.shape, np.dtype(np.float16), data.size * 2)
        else:
            qtype = gguf.GGMLQuantizationType[ggml_name(kind).upper()]
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            w.add_tensor_info(name, data.shape, data.dtype, data.size // block * size, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    for name, v in tensors.items():
        data = array(v)
        kind = stored_as(v, data)
        if kind == "raw":
            w.write_tensor_data(np.ascontiguousarray(data))
            continue
        if kind == "f16":
            stored = data.astype(np.float16)
            w.write_tensor_data(stored)
            back = stored.astype(np.float32)
        else:
            from ..quant.encode import encode_on_grid
            from ..runtime import quantize

            qtype = gguf.GGMLQuantizationType[ggml_name(kind).upper()]
            grid = getattr(v, "grid_scale", None)
            raw = encode_on_grid(data, grid.numpy(), kind) if grid is not None else None
            if raw is None:
                raw = quantize(data, ggml_name(kind))  # ggml's own encoder
            w.write_tensor_data(raw)
            back = gguf.quants.dequantize(raw, qtype).reshape(data.shape)
        scale = float(np.abs(data).max()) or 1.0
        errors[v.fqn] = float(np.abs(back - data).max()) / scale
        del back
    w.close()
    return errors
