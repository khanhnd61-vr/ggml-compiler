"""Every tier-1 op, run by the runtime, agrees with its PyTorch reference.

These tests build IR by hand, so they pin down the PyTorch-to-ggml translation in the emitter
(ne order, permute axes, concat/view dims) independently of the torch.export frontend.
"""

import pytest
import torch
from helpers import check_stage

from ggml_compiler.ir import Builder, Stage, TensorType, Weight


def _stage(shapes, build, name="op"):
    s = Stage(name)
    b = Builder(s)
    params = [s.add_param(TensorType(shape)) for shape in shapes]
    out = build(b, *params)
    s.outputs = list(out) if isinstance(out, (list, tuple)) else [out]
    return s


def _inputs(shapes, positive=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(shape, generator=g) for shape in shapes]
    return [x.abs() + 0.1 for x in xs] if positive else xs


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("bshape", [(2, 3, 4), (4,), (3, 1), (2, 1, 4), (1, 1, 1)])
def test_binary_broadcast(op, bshape, backend, tmp_path):
    shapes = [(2, 3, 4), bshape]
    xs = _inputs(shapes)
    if op == "div":
        xs[1] = xs[1].abs() + 0.5
    check_stage(_stage(shapes, lambda b, x, y: b.emit(op, [x, y])), xs, backend, tmp_path)


UNARY = ["relu", "gelu_erf", "silu", "tanh", "sigmoid", "exp", "neg", "abs", "sqr", "sin", "cos"]


@pytest.mark.parametrize("op", UNARY)
def test_unary(op, backend, tmp_path):
    shapes = [(3, 5, 7)]
    check_stage(_stage(shapes, lambda b, x: b.emit(op, [x])), _inputs(shapes), backend, tmp_path)


@pytest.mark.parametrize("op", ["log", "sqrt"])
def test_unary_positive(op, backend, tmp_path):
    shapes = [(3, 5, 7)]
    check_stage(_stage(shapes, lambda b, x: b.emit(op, [x])), _inputs(shapes, positive=True), backend, tmp_path)


def test_gelu_tanh_is_approximate_on_cpu(backend, tmp_path):
    # ggml_gelu reads an fp16 table on CPU. This pins down how far off it is, which is why the
    # frontend decomposes tanh-GELU for exact parity.
    shapes = [(4, 64)]
    check_stage(_stage(shapes, lambda b, x: b.gelu(x)), _inputs(shapes), backend, tmp_path, tol=5e-3)


def test_gelu_tanh_is_exact(backend, tmp_path):
    shapes = [(4, 64)]
    check_stage(_stage(shapes, lambda b, x: b.gelu_tanh(b.mul_scalar(x, 3.0))), _inputs(shapes), backend, tmp_path)


def test_scale_clamp(backend, tmp_path):
    shapes = [(4, 9)]
    build = lambda b, x: [b.scale(x, s=-1.5, b=0.25), b.clamp(x, min=-0.5, max=0.7), b.mul_scalar(x, 3.0)]
    check_stage(_stage(shapes, build), _inputs(shapes), backend, tmp_path)


def test_reductions_and_norms(backend, tmp_path):
    shapes = [(2, 3, 16)]
    build = lambda b, x: [b.mean(x), b.sum_rows(x), b.norm(x, eps=1e-5), b.rms_norm(x, eps=1e-6)]
    check_stage(_stage(shapes, build), _inputs(shapes), backend, tmp_path)


def test_soft_max(backend, tmp_path):
    shapes = [(2, 4, 6, 8), (6, 8)]
    build = lambda b, x, m: [b.soft_max(x), b.soft_max(x, scale=0.3), b.soft_max(x, m, scale=0.5)]
    xs = _inputs(shapes)
    xs[1] = torch.full((6, 8), float("-inf")).triu(1)  # causal mask
    check_stage(_stage(shapes, build), xs, backend, tmp_path)


@pytest.mark.parametrize(
    "ashape,bshape",
    [
        ((8, 16), (5, 16)),  # linear
        ((8, 16), (3, 5, 16)),  # weight shared over a batch
        ((3, 8, 16), (3, 5, 16)),  # batched
        ((2, 8, 16), (4, 5, 16)),  # grouped-query: a batch repeat-interleaved
        ((2, 3, 8, 16), (2, 3, 5, 16)),  # 4-D
    ],
)
def test_mul_mat(ashape, bshape, backend, tmp_path):
    check_stage(_stage([ashape, bshape], lambda b, a, x: b.mul_mat(a, x)), _inputs([ashape, bshape]), backend, tmp_path)


def test_mul_mat_with_weight(backend, tmp_path):
    w = Weight(torch.randn(12, 16, generator=torch.Generator().manual_seed(1)), "fc.weight")
    check_stage(_stage([(3, 16)], lambda b, x: b.mul_mat(w, x)), _inputs([(3, 16)]), backend, tmp_path)


@pytest.mark.parametrize(
    "shape,perm",
    [((3, 4), [1, 0]), ((2, 3, 4), [0, 2, 1]), ((2, 3, 4), [2, 0, 1]), ((2, 3, 4, 5), [0, 2, 1, 3]),
     ((2, 3, 4, 5), [3, 1, 0, 2])],
)
def test_permute(shape, perm, backend, tmp_path):
    build = lambda b, x: [b.cont(b.permute(x, perm=perm)), b.permute(x, perm=perm)]
    check_stage(_stage([shape], build), _inputs([shape]), backend, tmp_path)


def test_reshape(backend, tmp_path):
    build = lambda b, x: [b.reshape(x, shape=[6, 4]), b.reshape(x, shape=[24]), b.reshape(x, shape=[2, 3, 2, 2])]
    check_stage(_stage([(2, 3, 4)], build), _inputs([(2, 3, 4)]), backend, tmp_path)


def test_repeat(backend, tmp_path):
    build = lambda b, x: [b.repeat(x, shape=[2, 3, 4]), b.repeat(x, shape=[4, 6, 4])]
    check_stage(_stage([(1, 3, 4)], build), _inputs([(1, 3, 4)]), backend, tmp_path)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_concat(dim, backend, tmp_path):
    a = [2, 3, 4]
    c = list(a)
    c[dim] = 5
    check_stage(_stage([tuple(a), tuple(c)], lambda b, x, y: b.concat(x, y, dim=dim)), _inputs([tuple(a), tuple(c)]),
                backend, tmp_path)


@pytest.mark.parametrize("dim,start,length", [(0, 1, 2), (1, 0, 2), (2, 3, 4), (1, 2, 1)])
def test_view(dim, start, length, backend, tmp_path):
    build = lambda b, x: [b.cont(b.view(x, dim=dim, start=start, length=length))]
    check_stage(_stage([(4, 3, 8)], build), _inputs([(4, 3, 8)]), backend, tmp_path)


def test_cast_roundtrip(backend, tmp_path):
    build = lambda b, x: b.cast(b.cast(x, dtype="f16"), dtype="f32")
    check_stage(_stage([(4, 32)], build), _inputs([(4, 32)]), backend, tmp_path, tol=1e-3)


def test_stage_returning_param_and_weight(backend, tmp_path):
    w = Weight(torch.arange(6.0).reshape(2, 3), "w")
    check_stage(_stage([(2, 3)], lambda b, x: [x, w, b.add(x, w)]), _inputs([(2, 3)]), backend, tmp_path)


@pytest.mark.parametrize("stride,padding,dilation", [((1, 1), (0, 0), (1, 1)), ((2, 2), (3, 3), (1, 1)),
                                                     ((2, 1), (1, 2), (1, 1)), ((1, 1), (2, 2), (2, 2))])
def test_im2col(stride, padding, dilation, backend, tmp_path):
    w = Weight(torch.zeros(4, 3, 3, 5), "kernel")  # only its shape matters
    build = lambda b, x: b.im2col(w, x, stride=list(stride), padding=list(padding), dilation=list(dilation))
    check_stage(_stage([(2, 3, 11, 13)], build), _inputs([(2, 3, 11, 13)]), backend, tmp_path)


@pytest.mark.parametrize("op", ["max", "avg"])
@pytest.mark.parametrize("kernel,stride,padding", [((3, 3), (2, 2), (1, 1)), ((2, 2), (2, 2), (0, 0)),
                                                   ((3, 2), (1, 2), (1, 0))])
def test_pool_2d(op, kernel, stride, padding, backend, tmp_path):
    build = lambda b, x: b.pool_2d(x, op=op, kernel=list(kernel), stride=list(stride), padding=list(padding))
    check_stage(_stage([(1, 4, 9, 12)], build), _inputs([(1, 4, 9, 12)]), backend, tmp_path)
