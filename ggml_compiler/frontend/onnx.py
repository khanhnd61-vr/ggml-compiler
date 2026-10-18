"""ONNX frontend: a static-shape ONNX model to a one-stage program.

- Shapes come from ONNX shape inference; `input_shapes` pins dynamic input dims.
- Float initializers become weights, named by the initializer (the torch exporter keeps parameter
  paths, so dtype policies written for the PyTorch path still match).
- A node whose inputs are all constants is evaluated here with onnx's reference evaluator (Shape,
  Gather on shapes, Range, ...). One whose only runtime inputs are weights is evaluated into a
  derived weight, except cheap views. Everything else goes through lower/onnx.py.
- Missing lowerings are collected over the whole graph and reported together.
- The reference outputs come from onnxruntime on the example inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..ir import Builder, Stage, TensorType, Value, Weight
from ..ir.program import CallOp, CompileError, ProgramIR
from ..lower.onnx import ONNX_LOWERINGS
from .torch_export import Reference, WeightTable, runtime_tensor

# Ops on weights that ggml does as free views: keep them in the graph rather than store a copy.
_VIEW_OPS = {"Reshape", "Squeeze", "Unsqueeze", "Flatten", "Slice", "Split", "Identity", "Gather"}


def _np_dtype(elem_type: int) -> np.dtype:
    import onnx

    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))


def _evaluate(node: Any, inputs: list[np.ndarray | None], opsets: dict[str, int] | None = None) -> list[np.ndarray]:
    """Run one ONNX node on constant inputs with onnx's reference evaluator."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    names = [f"i{k}" if x is not None else "" for k, x in enumerate(inputs)]
    vi = [onnx.helper.make_tensor_value_info(n, onnx.helper.np_dtype_to_tensor_dtype(x.dtype), x.shape)
          for n, x in zip(names, inputs) if x is not None]
    outs = [f"o{k}" for k in range(len(node.output))]
    g = onnx.helper.make_graph([onnx.helper.make_node(node.op_type, names, outs, domain=node.domain,
                                                      **{a.name: onnx.helper.get_attribute_value(a)
                                                         for a in node.attribute})],
                               "fold", vi, [onnx.helper.make_empty_tensor_value_info(o) for o in outs])
    model = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid(d, v)
                                                     for d, v in (opsets or {"": 20}).items()])
    return list(ReferenceEvaluator(model).run(None, {n: x for n, x in zip(names, inputs) if x is not None}))


def _loc(node: Any) -> str:
    return f"{node.op_type} {node.name}" if node.name else node.op_type


def export_onnx(path: str | Path, example_inputs: Any = None,
                input_shapes: dict[str, list[int]] | None = None) -> tuple[ProgramIR, Reference]:
    import onnx

    model = onnx.load(str(path))
    graph = model.graph
    init_names = {i.name for i in graph.initializer}
    real_inputs = [i for i in graph.input if i.name not in init_names]
    for i in real_inputs:  # pin dynamic dims
        dims = (input_shapes or {}).get(i.name)
        shape = i.type.tensor_type.shape
        if dims is not None:
            if len(dims) != len(shape.dim):
                raise CompileError(f"input {i.name}: {len(dims)} dims given, the model has {len(shape.dim)}")
            for d, v in zip(shape.dim, dims):
                d.ClearField("dim_param")
                d.dim_value = int(v)
        if any(not d.HasField("dim_value") for d in shape.dim):
            raise CompileError(f"input {i.name} has a dynamic shape; fix it with input_shapes / --input")
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    graph = model.graph
    opsets = {o.domain: o.version for o in model.opset_import}

    types: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        t = vi.type.tensor_type
        if t.HasField("shape") and all(d.HasField("dim_value") for d in t.shape.dim):
            types[vi.name] = (tuple(d.dim_value for d in t.shape.dim), _np_dtype(t.elem_type))

    stage = Stage("forward")
    b = Builder(stage)
    weights = WeightTable()
    env: dict[str, Any] = {"": None}
    for init in graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        t = torch.from_numpy(np.array(arr))
        # Scalars (fill values such as -1e9 or -inf, divisors) are constants, not weights, so they
        # take the scalar paths in the lowerings (where() must never multiply an infinity by 0).
        env[init.name] = weights.get(init.name, t) if t.is_floating_point() and t.numel() > 1 else t

    # Inputs and example values.
    if isinstance(example_inputs, dict):
        examples = dict(example_inputs)
    elif example_inputs is not None:
        examples = dict(zip([i.name for i in real_inputs], example_inputs))
    else:
        examples = {}
    g = np.random.default_rng(0)
    for i in real_inputs:
        shape, dtype = types[i.name]
        if i.name not in examples:
            examples[i.name] = (g.standard_normal(shape) if dtype.kind == "f" else np.zeros(shape)).astype(dtype)
        ex = examples[i.name]
        ex = ex.detach().cpu().numpy() if hasattr(ex, "detach") else np.asarray(ex)
        examples[i.name] = ex.astype(dtype)
        integer = dtype.kind in "iu"
        env[i.name] = stage.add_param(TensorType(shape, "i32" if integer else "f32"))

    missing: dict[str, list[str]] = {}
    for node in graph.node:
        ins = [env.get(n, _MISSING) for n in node.input]
        if any(x is _MISSING for x in ins):
            for o in node.output:
                env[o] = _MISSING
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        attrs = {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()}
        values = [x for x in ins if isinstance(x, Value)]

        if node.op_type == "Shape":
            shape = ins[0].shape if isinstance(ins[0], Value) else tuple(ins[0].shape)
            res = [np.array(shape, dtype=np.int64)[attrs.get("start", 0): attrs.get("end", None)]]
        elif node.op_type == "Constant":
            res = _evaluate(node, [], opsets)
        elif not values or (all(isinstance(v, Weight) for v in values) and node.op_type not in _VIEW_OPS):
            const_in = [None if x is None else (x.data if isinstance(x, Weight) else x).numpy() for x in ins]
            res = _evaluate(node, const_in, opsets)
            if values:  # derived from weights: keep it a weight, so dtype policies see it
                res = [weights.derived(f"{values[0].fqn}:{node.op_type}", torch.from_numpy(np.array(res[0])))]
        else:
            lower = ONNX_LOWERINGS.get(node.op_type)
            if lower is None:
                missing.setdefault(node.op_type, []).append(_loc(node))
                for o in node.output:
                    env[o] = _MISSING
                continue
            b.loc = _loc(node)
            outs = [types.get(o) for o in node.output]
            b.out_types = [TensorType(t[0], "f32") if t else None for t in outs]
            b.out_type = b.out_types[0]
            res = lower(b, ins, attrs)
            for o, r in zip(node.output, res):
                t = types.get(o)
                if isinstance(r, Value) and t and r.shape != t[0]:
                    raise CompileError(f"{_loc(node)}: lowering produced {r.shape}, ONNX inferred {t[0]}")
        for o, r in zip(node.output, res):
            env[o] = torch.from_numpy(np.array(r)) if isinstance(r, np.ndarray) else r

    if missing:
        lines = [f"{Path(path).name}: {len(missing)} ONNX op type(s) have no ggml lowering"]
        for op, where in missing.items():
            lines.append(f"  {op:<24} {where[0]}" + (f" (+{len(where) - 1} more)" if len(where) > 1 else ""))
        raise CompileError("\n".join(lines))

    b.loc = ""
    stage.outputs = [env[o.name] if isinstance(env[o.name], Value) else b.const(env[o.name]) for o in graph.output]

    prog = ProgramIR(stages={"forward": stage})
    prog.weights = weights.by_fqn
    prog.inputs = [i.name for i in real_inputs]
    prog.outputs = [o.name for o in graph.output]
    prog.values.update(zip(prog.inputs, (p.type for p in stage.params)))
    prog.values.update(zip(prog.outputs, (v.type for v in stage.outputs)))
    prog.driver = [CallOp("forward", list(prog.inputs), list(prog.outputs))]
    prog.check()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {n: examples[n] for n in prog.inputs})
    ref = Reference(inputs={n: runtime_tensor(torch.from_numpy(examples[n])) for n in prog.inputs},
                    outputs={n: torch.from_numpy(np.asarray(o)).float() for n, o in zip(prog.outputs, outs)})
    return prog, ref


class _Missing:
    pass


_MISSING = _Missing()
