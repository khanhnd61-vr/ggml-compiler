"""Lowerings from ONNX ops to the IR, mostly by delegating to the ATen lowerings.

A lowering receives the Builder, the op's inputs (IR values for runtime tensors, torch.Tensor for
constants, None for omitted optional inputs) and its attributes; it returns one value per output.
Delegating keeps one set of semantics and clean-ups (broadcasting, the stack-of-rows rewrite in
concat, ...) for both frontends.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ..ir import Builder
from ..ir.program import CompileError
from . import aten as A
from .registry import LOWERINGS

ONNX_LOWERINGS: dict[str, Callable[..., list[Any]]] = {}

aten = torch.ops.aten


def onnx_op(*names: str):
    def deco(fn):
        for n in names:
            ONNX_LOWERINGS[n] = fn
        return fn

    return deco


def _aten(name: str) -> Callable[..., Any]:
    return LOWERINGS[name]


def _ints(x: Any) -> list[int]:
    return [int(v) for v in (x.tolist() if isinstance(x, torch.Tensor) else x)]


def _unsupported(b: Builder, what: str) -> CompileError:
    return CompileError(f"{what}" + (f" at {b.loc}" if b.loc else ""))


# --- elementwise ---------------------------------------------------------------------------

for _onnx, _aten_name in [("Add", "aten.add.Tensor"), ("Sub", "aten.sub.Tensor"), ("Mul", "aten.mul.Tensor"),
                          ("Div", "aten.div.Tensor"), ("Equal", "aten.eq.Tensor"), ("Less", "aten.lt.Tensor"),
                          ("Greater", "aten.gt.Tensor"), ("LessOrEqual", "aten.le.Tensor"),
                          ("GreaterOrEqual", "aten.ge.Tensor"), ("And", "aten.logical_and.default"),
                          ("Or", "aten.logical_or.default")]:
    onnx_op(_onnx)((lambda name: lambda b, ins, at: [_aten(name)(b, ins[0], ins[1])])(_aten_name))

for _onnx, _ir in [("Relu", "relu"), ("Sigmoid", "sigmoid"), ("Tanh", "tanh"), ("Exp", "exp"), ("Log", "log"),
                   ("Neg", "neg"), ("Abs", "abs"), ("Sqrt", "sqrt"), ("Sin", "sin"), ("Cos", "cos")]:
    onnx_op(_onnx)((lambda op: lambda b, ins, at: [b.emit(op, [A.V(b, ins[0])])])(_ir))


@onnx_op("Not")
def _not(b, ins, at):
    return [_aten("aten.logical_not.default")(b, ins[0])]


@onnx_op("Reciprocal")
def _reciprocal(b, ins, at):
    return [_aten("aten.reciprocal.default")(b, ins[0])]


@onnx_op("Gelu")
def _gelu(b, ins, at):
    return [_aten("aten.gelu.default")(b, ins[0], approximate=at.get("approximate", "none"))]


@onnx_op("Pow")
def _pow(b, ins, at):
    e = ins[1]
    if not isinstance(e, torch.Tensor) or e.numel() != 1:
        raise _unsupported(b, "Pow with a runtime exponent")
    return [_aten("aten.pow.Tensor_Scalar")(b, ins[0], float(e))]


@onnx_op("Clip")
def _clip(b, ins, at):
    lo = float(ins[1]) if len(ins) > 1 and ins[1] is not None else None
    hi = float(ins[2]) if len(ins) > 2 and ins[2] is not None else None
    return [_aten("aten.clamp.default")(b, ins[0], lo, hi)]


@onnx_op("Where")
def _where(b, ins, at):
    return [A._where(b, ins[0], ins[1], ins[2])]


@onnx_op("Identity", "Dropout")
def _identity(b, ins, at):
    return [ins[0]]


@onnx_op("Cast")
def _cast(b, ins, at):
    to_bool = at["to"] == 9  # onnx.TensorProto.BOOL
    return [_aten("aten._to_copy.default")(b, ins[0], dtype=torch.bool if to_bool else torch.float32)]


# --- matmul, normalisation, attention pieces -----------------------------------------------


@onnx_op("MatMul")
def _matmul(b, ins, at):
    return [_aten("aten.matmul.default")(b, ins[0], ins[1])]


@onnx_op("Gemm")
def _gemm(b, ins, at):
    a, w = A.V(b, ins[0]), A.V(b, ins[1])
    if at.get("transA", 0):
        a = b.transpose(a, 0, 1)
    if not at.get("transB", 0):
        w = b.transpose(w, 0, 1)  # mul_mat wants the weight as [N, K]
    y = b.mul_mat(w, a)
    if at.get("alpha", 1.0) != 1.0:
        y = b.mul_scalar(y, at["alpha"])
    c = ins[2] if len(ins) > 2 else None
    if c is not None:
        c = A.V(b, c)
        if at.get("beta", 1.0) != 1.0:
            c = b.mul_scalar(c, at["beta"])
        y = A._binary(b, "add", y, c)
    return [y]


@onnx_op("LayerNormalization")
def _layer_norm(b, ins, at):
    x = A.V(b, ins[0])
    if at.get("axis", -1) not in (-1, x.rank - 1):
        raise _unsupported(b, "LayerNormalization over more than the last dim")
    bias = ins[2] if len(ins) > 2 else None
    return [_aten("aten.layer_norm.default")(b, x, [x.shape[-1]], ins[1], bias, at.get("epsilon", 1e-5))]


@onnx_op("Softmax")
def _softmax(b, ins, at):
    return [_aten("aten._softmax.default")(b, ins[0], at.get("axis", -1), False)]


def _reduce_for(op_type: str):
    name = {"ReduceMean": "aten.mean.dim", "ReduceSum": "aten.sum.dim_IntList"}[op_type]

    def lower(b, ins, at):
        axes = at.get("axes") if len(ins) < 2 or ins[1] is None else _ints(ins[1])
        return [_aten(name)(b, ins[0], list(axes), bool(at.get("keepdims", 1)))]

    return lower


for _op in ("ReduceMean", "ReduceSum"):
    ONNX_LOWERINGS[_op] = _reduce_for(_op)


# --- convolution and pooling ---------------------------------------------------------------


def _pads(at: dict[str, Any], b: Builder) -> list[int]:
    pads = list(at.get("pads", [0, 0, 0, 0]))
    if at.get("auto_pad", "NOTSET") not in ("NOTSET", b"NOTSET", "VALID", b"VALID"):
        raise _unsupported(b, f"auto_pad={at['auto_pad']}")
    if pads[:2] != pads[2:]:
        raise _unsupported(b, f"asymmetric padding {pads}")
    return pads[:2]


@onnx_op("Conv")
def _conv(b, ins, at):
    if at.get("group", 1) != 1:
        raise _unsupported(b, f"grouped Conv (group={at['group']})")
    bias = ins[2] if len(ins) > 2 else None
    return [_aten("aten.convolution.default")(b, ins[0], ins[1], bias, list(at.get("strides", [1, 1])), _pads(at, b),
                                              list(at.get("dilations", [1, 1])), False, [0, 0], 1)]


@onnx_op("MaxPool")
def _max_pool(b, ins, at):
    if at.get("ceil_mode", 0) or any(d != 1 for d in at.get("dilations", [1, 1])):
        raise _unsupported(b, "MaxPool with ceil_mode or dilation")
    k = list(at["kernel_shape"])
    y, _ = _aten("aten.max_pool2d_with_indices.default")(b, ins[0], k, list(at.get("strides", k)), _pads(at, b))
    return [y]


@onnx_op("AveragePool")
def _avg_pool(b, ins, at):
    k = list(at["kernel_shape"])
    return [_aten("aten.avg_pool2d.default")(b, ins[0], k, list(at.get("strides", k)), _pads(at, b),
                                             bool(at.get("ceil_mode", 0)), bool(at.get("count_include_pad", 0)))]


# --- layout --------------------------------------------------------------------------------


@onnx_op("Reshape", "Squeeze", "Unsqueeze", "Flatten")
def _reshape(b, ins, at):
    return [b.reshape_to(A.V(b, ins[0]), b.out_type.shape)]


@onnx_op("Transpose")
def _transpose(b, ins, at):
    x = A.V(b, ins[0])
    perm = list(at.get("perm", reversed(range(x.rank))))
    return [_aten("aten.permute.default")(b, x, perm)]


@onnx_op("Expand")
def _expand(b, ins, at):
    x = A.V(b, ins[0])
    out = b.out_type.shape
    x = b.reshape_to(x, (1,) * (len(out) - x.rank) + x.shape)
    return [x if x.shape == out else b.repeat(x, shape=list(out))]


@onnx_op("Concat")
def _concat(b, ins, at):
    return [_aten("aten.cat.default")(b, list(ins), at["axis"])]


@onnx_op("Split")
def _split(b, ins, at):
    x = A.V(b, ins[0])
    axis = at.get("axis", 0)
    d = A._dim(axis, x.rank)
    if len(ins) > 1 and ins[1] is not None:
        sizes = _ints(ins[1])
    else:
        n = at.get("num_outputs") or len(b.out_types)  # out_types: set by the ONNX frontend
        sizes = [x.shape[d] // n] * n
    return _aten("aten.split_with_sizes.default")(b, x, sizes, d)


@onnx_op("Slice")
def _slice(b, ins, at):
    x = A.V(b, ins[0])
    starts, ends = _ints(ins[1]), _ints(ins[2])
    axes = _ints(ins[3]) if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
    steps = _ints(ins[4]) if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
    for s, e, a, st in zip(starts, ends, axes, steps):
        n = x.shape[A._dim(a, x.rank)]
        x = _aten("aten.slice.Tensor")(b, x, a, max(-n, min(s, n)), max(-n - 1, min(e, n)), st)
    return [x]


@onnx_op("Gather")
def _gather(b, ins, at):
    data, idx = ins[0], ins[1]
    axis = at.get("axis", 0)
    if isinstance(idx, torch.Tensor) and idx.dim() == 0:
        return [_aten("aten.select.int")(b, data, axis, int(idx))]
    if A._dim(axis, A.V(b, data).rank) == 0:
        return [_aten("aten.embedding.default")(b, data, idx)]
    raise _unsupported(b, "Gather along a dim other than 0 with tensor indices")

