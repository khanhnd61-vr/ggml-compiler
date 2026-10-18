"""Lowerings from (Core) ATen ops to the IR.

The frontend keeps a few composite ops whole (linear, layer_norm,
scaled_dot_product_attention) so they arrive here as tier-2 IR ops that fusion passes can
still recognise. Everything else maps onto tier-1 ops directly.
"""

from __future__ import annotations

import math
import operator
from typing import Any

import torch

from ..ir import Builder, Node, Value, sym
from ..ir.program import CompileError
from .registry import lowering

aten = torch.ops.aten


def V(b: Builder, x: Any) -> Value:
    """IR value for a tensor argument that may be a compile-time constant."""
    if isinstance(x, Value):
        return x
    if isinstance(x, (torch.Tensor, int, float)):
        return b.const(x if isinstance(x, torch.Tensor) else torch.tensor(float(x)))
    raise CompileError(f"expected a tensor, got {type(x).__name__}")


def _dim(d: int, rank: int) -> int:
    return d + rank if d < 0 else d


def _unsupported(b: Builder, what: str) -> CompileError:
    where = f" at {b.loc}" if b.loc else ""
    return CompileError(f"{what}{where}")


# --- identity ------------------------------------------------------------------------------


@lowering(aten.clone.default, aten.alias.default, aten.detach.default, aten.lift_fresh_copy.default)
def _identity(b, x, *args, **kwargs):
    return x


@lowering(aten.dropout.default)
def _dropout(b, x, p=0.5, train=False):
    if train:
        raise _unsupported(b, "dropout in training mode")
    return x


# --- elementwise ---------------------------------------------------------------------------


def _scalar(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return float(x.reshape(()))
    return None


def _binary(b: Builder, op: str, x: Any, y: Any) -> Value:
    sx, sy = _scalar(x), _scalar(y)
    if sy is not None and isinstance(x, Value):
        if op == "add":
            return b.add_scalar(x, sy)
        if op == "sub":
            return b.add_scalar(x, -sy)
        if op == "mul":
            return b.mul_scalar(x, sy)
        return b.mul_scalar(x, 1.0 / sy)
    if sx is not None and isinstance(y, Value):
        if op == "add":
            return b.add_scalar(y, sx)
        if op == "mul":
            return b.mul_scalar(y, sx)
        if op == "sub":
            return b.scale(y, s=-1.0, b=sx)
        x = b.repeat(b.const(torch.tensor([sx])), shape=list(y.shape))
    X, Y = V(b, x), V(b, y)
    out = sym.broadcast(X.shape, Y.shape)
    if X.shape == out:
        return b.emit(op, [X, Y])
    if Y.shape == out and op in ("add", "mul"):
        return b.emit(op, [Y, X])
    return b.emit(op, [b.repeat(X, shape=list(out)), Y])


@lowering(aten.add.Tensor, aten.add.Scalar)
def _add(b, x, y, alpha=1):
    if alpha != 1:
        y = _binary(b, "mul", y, alpha)
    return _binary(b, "add", x, y)


@lowering(aten.sub.Tensor, aten.sub.Scalar)
def _sub(b, x, y, alpha=1):
    if alpha != 1:
        y = _binary(b, "mul", y, alpha)
    return _binary(b, "sub", x, y)


@lowering(aten.mul.Tensor, aten.mul.Scalar)
def _mul(b, x, y):
    return _binary(b, "mul", x, y)


@lowering(aten.div.Tensor, aten.div.Scalar)
def _div(b, x, y):
    return _binary(b, "div", x, y)


def _unary(op: str):
    return lambda b, x: b.emit(op, [V(b, x)])


for _aten_op, _ir_op in [
    (aten.relu.default, "relu"),
    (aten.silu.default, "silu"),
    (aten.tanh.default, "tanh"),
    (aten.sigmoid.default, "sigmoid"),
    (aten.exp.default, "exp"),
    (aten.log.default, "log"),
    (aten.neg.default, "neg"),
    (aten.abs.default, "abs"),
    (aten.sqrt.default, "sqrt"),
    (aten.sin.default, "sin"),
    (aten.cos.default, "cos"),
]:
    lowering(_aten_op)(_unary(_ir_op))


def _ones_like(b: Builder, x: Value) -> Value:
    return b.repeat(b.const(torch.ones(1)), shape=list(x.shape))


@lowering(aten.reciprocal.default)
def _reciprocal(b, x):
    x = V(b, x)
    return b.div(_ones_like(b, x), x)


@lowering(aten.rsqrt.default)
def _rsqrt(b, x):
    x = V(b, x)
    return b.div(_ones_like(b, x), b.sqrt(x))


@lowering(aten.softplus.default)
def _softplus(b, x, beta=1, threshold=20):
    if beta != 1 or threshold != 20:
        raise _unsupported(b, f"softplus with beta={beta}, threshold={threshold} (ggml has 1 and 20)")
    return b.softplus(V(b, x))


@lowering(aten.mish.default)
def _mish(b, x):
    x = V(b, x)
    return b.mul(x, b.tanh(b.softplus(x)))


@lowering(aten.gelu.default)
def _gelu(b, x, approximate="none"):
    x = V(b, x)
    return b.gelu_erf(x) if approximate == "none" else b.gelu_tanh(x)


@lowering(aten.pow.Tensor_Scalar)
def _pow(b, x, e):
    if e == 2:
        return b.sqr(x)
    if e == 0.5:
        return b.sqrt(x)
    if e == 1:
        return x
    if e in (-1, -0.5):  # as reciprocal / rsqrt lower them, so fuse_rms_norm sees the same graph
        x = V(b, x)
        return b.div(_ones_like(b, x), x if e == -1 else b.sqrt(x))
    raise _unsupported(b, f"pow with exponent {e}")


@lowering(aten.clamp.default)
def _clamp(b, x, min=None, max=None):
    return b.clamp(x, min=float("-inf") if min is None else float(min), max=float("inf") if max is None else float(max))


# --- matmul and normalisation --------------------------------------------------------------


@lowering(aten.linear.default)
def _linear(b, x, w, bias=None):
    ins = [V(b, x), V(b, w)] + ([V(b, bias)] if bias is not None else [])
    return b.linear(*ins)


@lowering(aten.mm.default, aten.bmm.default, aten.matmul.default)
def _matmul(b, x, y):
    X, Y = V(b, x), V(b, y)
    if X.rank < 2 or Y.rank < 2:
        raise _unsupported(b, "matmul with a 1-D operand")
    return b.mul_mat(b.transpose(Y, -2 + Y.rank, -1 + Y.rank), X)


@lowering(aten.layer_norm.default)
def _layer_norm(b, x, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
    if len(normalized_shape) != 1:
        raise _unsupported(b, f"layer_norm over {len(normalized_shape)} dims")
    ins = [V(b, x)] + ([V(b, weight)] if weight is not None else []) + ([V(b, bias)] if bias is not None else [])
    return b.layer_norm(*ins, eps=float(eps), has_weight=weight is not None, has_bias=bias is not None)


@lowering(aten._softmax.default, aten.softmax.int)
def _softmax(b, x, dim, *rest):
    x = V(b, x)
    d = _dim(dim, x.rank)
    if d == x.rank - 1:
        return b.soft_max(x)
    t = b.transpose(x, d, x.rank - 1)
    return b.transpose(b.soft_max(t), d, x.rank - 1)


@lowering(aten.scaled_dot_product_attention.default)
def _sdpa(b, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if dropout_p:
        raise _unsupported(b, "attention dropout")
    Q, K, Vv = V(b, q), V(b, k), V(b, v)
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    mask = None
    sq, sk = Q.shape[-2], K.shape[-2]
    if is_causal:
        if not sym.static((sq, sk)):
            raise _unsupported(b, "is_causal attention over a symbolic length (pass a mask)")
        mask = torch.full((sq, sk), float("-inf")).triu(1)
    if isinstance(attn_mask, Value):
        # A runtime mask. bool (f32 0/1 here): 0 where allowed, float32 min where not; a finite
        # fill keeps a fully masked row a uniform softmax instead of NaN. Float: additive as is.
        m = b.scale(attn_mask, s=-torch.finfo(torch.float32).min, b=torch.finfo(torch.float32).min) \
            if id(attn_mask) in b.bools else attn_mask
        if mask is not None:
            m = b.add(m, b.const(mask))
        return b.sdpa(Q, K, Vv, m, scale=float(scale))
    if attn_mask is not None:
        m = attn_mask
        if m.dtype == torch.bool:
            m = torch.zeros(m.shape).masked_fill(~m, float("-inf"))
        mask = m if mask is None else mask + m
    ins = [Q, K, Vv] + ([b.const(mask.to(torch.float32))] if mask is not None else [])
    return b.sdpa(*ins, scale=float(scale))


# --- reductions ----------------------------------------------------------------------------


def _row_reduce(op: str):
    def lower(b, x, dims, keepdim=False, dtype=None):
        x = V(b, x)
        if dims is None or len(dims) != 1 or _dim(dims[0], x.rank) != x.rank - 1:
            raise _unsupported(b, f"{op} over dims {dims} (only the last dim is supported)")
        y = b.emit(op, [x])
        return y if keepdim else b.reshape_to(y, x.shape[:-1])

    return lower


lowering(aten.mean.dim)(_row_reduce("mean"))
lowering(aten.sum.dim_IntList)(_row_reduce("sum_rows"))


# --- layout --------------------------------------------------------------------------------


@lowering(aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.unsqueeze.default,
          aten.squeeze.dim, aten.squeeze.dims, aten.squeeze.default, aten.flatten.using_ints)
def _reshape(b, x, *args):
    return b.reshape_to(V(b, x), b.out_type.shape)


@lowering(aten.permute.default)
def _permute(b, x, dims):
    x = V(b, x)
    perm = [_dim(d, x.rank) for d in dims]
    if perm == list(range(x.rank)):
        return x
    return b.permute(x, perm=perm)


@lowering(aten.transpose.int)
def _transpose(b, x, d0, d1):
    x = V(b, x)
    return b.transpose(x, _dim(d0, x.rank), _dim(d1, x.rank))


@lowering(aten.t.default)
def _t(b, x):
    x = V(b, x)
    return x if x.rank < 2 else b.transpose(x, 0, 1)


@lowering(aten.expand.default)
def _expand(b, x, size, implicit=False):
    x = V(b, x)
    out = b.out_type.shape
    return x if x.shape == out else b.repeat(x, shape=list(out))


def _unit_slice(v: Value) -> tuple[Value, int, int] | None:
    """(x, dim, i) if v is x's i-th slice along dim, maybe reshaped, where every dim of x before
    dim is 1, so the slices sit back to back in memory."""
    while isinstance(v, Node) and v.op == "reshape":
        v = v.inputs[0]
    if isinstance(v, Node) and v.op == "view" and v.attrs["length"] == 1:
        x, d = v.inputs[0], v.attrs["dim"]
        if all(s == 1 for s in x.shape[:d]):
            return x, d, v.attrs["start"]
    return None


@lowering(aten.cat.default)
def _cat(b, tensors, dim=0):
    """Concatenation, after two clean-ups for code like `torch.stack(list(t) + ...)`: adjacent
    constants are joined at compile time, and a run of consecutive unit slices of one tensor
    becomes a single slice of it."""
    numel = lambda t: t.type.numel() if isinstance(t, Value) else t.numel()  # noqa: E731
    tensors = [t for t in tensors if numel(t) > 0]  # torch.cat skips empty inputs (lazy caches)
    rank = next(t.rank if isinstance(t, Value) else t.dim() for t in tensors)
    d = _dim(dim, rank)
    items: list[Any] = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and items and isinstance(items[-1], torch.Tensor):
            items[-1] = torch.cat([items[-1], t], d)
        else:
            items.append(t)
    vals = [V(b, t) for t in items]

    runs: list[Value] = []
    i = 0
    while i < len(vals):
        v, j = vals[i], i + 1
        origin = _unit_slice(v) if all(s == 1 for s in v.shape[:d]) else None
        if origin is not None:
            x, sd, start = origin
            while (j < len(vals) and vals[j].shape == v.shape and _unit_slice(vals[j]) is not None
                   and _unit_slice(vals[j])[:2] == (x, sd) and _unit_slice(vals[j])[2] == start + (j - i)):
                j += 1
            if j - i > 1:
                k = j - i
                src = x if k == x.shape[sd] else b.view(x, dim=sd, start=start, length=k)
                shape = list(v.shape)
                shape[d] *= k
                v = b.reshape_to(src, tuple(shape))
        runs.append(v)
        i = j

    out = runs[0]
    for v in runs[1:]:
        out = b.concat(out, v, dim=d)
    return out


@lowering(aten.slice.Tensor)
def _slice(b, x, dim=0, start=None, end=None, step=1):
    x = V(b, x)
    d = _dim(dim, x.rank)
    n = x.shape[d]
    start = 0 if start is None else sym.smax(0, sym.smin(n, start + n if sym.definitely(start < 0) else start))
    end = n if end is None else sym.smax(0, sym.smin(n, end + n if sym.definitely(end < 0) else end))
    if step == 1:
        return x if (start, end) == (0, n) else b.view(x, dim=d, start=start, length=end - start)
    if step < 1:
        raise _unsupported(b, f"slice with step {step}")
    # x[start:end:step] along d: take a block of m * step elements beginning at o (at most one
    # step before start, so it fits), split d into (m, step), keep inner index start - o. (A
    # ggml view cannot stride the innermost dim.)
    m = -(-(end - start) // step)
    o = max(0, min(start, n - m * step))
    if m * step > n or start - o >= step:
        raise _unsupported(b, f"slice {start}:{end}:{step} of a dim of {n}")
    y = x if (o, m * step) == (0, n) else b.view(x, dim=d, start=o, length=m * step)
    y = b.reshape_to(y, x.shape[:d] + (m, step) + x.shape[d + 1:])
    y = b.view(y, dim=d + 1, start=start - o, length=1)
    return b.reshape_to(y, x.shape[:d] + (m,) + x.shape[d + 1:])


@lowering(aten.select.int)
def _select(b, x, dim, index):
    x = V(b, x)
    d = _dim(dim, x.rank)
    index = index + x.shape[d] if index < 0 else index
    return b.reshape_to(b.view(x, dim=d, start=index, length=1), b.out_type.shape)


@lowering(aten.split_with_sizes.default, aten.split.Tensor)
def _split(b, x, sizes, dim=0):
    x = V(b, x)
    d = _dim(dim, x.rank)
    if isinstance(sizes, int):  # split.Tensor: chunks of `sizes`
        n = x.shape[d]
        sizes = [min(sizes, n - i) for i in range(0, n, sizes)]
    out, start = [], 0
    for s in sizes:
        out.append(x if s == x.shape[d] else b.view(x, dim=d, start=start, length=s))
        start += s
    return out


@lowering(aten.unbind.int)
def _unbind(b, x, dim=0):
    x = V(b, x)
    d = _dim(dim, x.rank)
    shape = x.shape[:d] + x.shape[d + 1:]
    return [b.reshape_to(b.view(x, dim=d, start=i, length=1), shape) for i in range(x.shape[d])]


class _Unavailable:
    """A result a lowering cannot produce (max-pool indices). Using it is an error."""

    def __init__(self, what: str) -> None:
        self.what = what


@lowering(operator.getitem)
def _getitem(b, seq, i):
    v = seq[i]
    if isinstance(v, _Unavailable):
        raise _unsupported(b, v.what)
    return v


# --- convolution and pooling ---------------------------------------------------------------


def _pair(v: Any) -> list[int]:
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return v * 2 if len(v) == 1 else v


@lowering(aten.convolution.default)
def _convolution(b, x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    x, w = V(b, x), V(b, w)
    if groups != 1:
        raise _unsupported(b, f"grouped convolution (groups={groups})")
    if transposed:
        if w.rank != 3:
            raise _unsupported(b, f"{w.rank - 2}-D transposed convolution (only 1-D)")
        return _conv_transpose_1d(b, x, w, bias, stride[0], padding[0], dilation[0], output_padding[0])
    if w.rank == 3:  # 1-D: a 2-D convolution over [N, C, 1, L]
        n, c, length = x.shape
        y = _convolution(b, b.reshape_to(x, (n, c, 1, length)), b.reshape_to(w, (*w.shape[:2], 1, w.shape[2])), bias,
                         [1, stride[0]], [0, padding[0]], [1, dilation[0]], False, [0, 0], 1)
        return b.reshape_to(y, (n, y.shape[1], y.shape[3]))
    if w.rank > 4 and tuple(x.shape[2:]) == tuple(w.shape[2:]) and not any(padding) \
            and all(d == 1 for d in dilation):
        # A kernel covering the whole input (Qwen-VL's Conv3d patch embedding over [N, C, T, P, P]):
        # one output per sample, i.e. a matmul against the flattened kernel.
        n, k = x.shape[0], w.type.numel() // w.shape[0]
        y = b.mul_mat(b.reshape_to(w, (w.shape[0], k)), b.reshape_to(x, (n, k)))
        if bias is not None:
            y = b.add(y, V(b, bias))
        return b.reshape_to(y, (n, w.shape[0]) + (1,) * (w.rank - 2))
    if w.rank != 4:
        raise _unsupported(b, f"{w.rank - 2}-D convolution (only 1-D, 2-D, and kernels covering the input)")
    ins = [w, x] + ([V(b, bias)] if bias is not None else [])
    return b.conv_2d(*ins, stride=_pair(stride), padding=_pair(padding), dilation=_pair(dilation))


def _conv_transpose_1d(b: Builder, x: Value, w: Value, bias: Any, stride: int, padding: int, dilation: int,
                       output_padding: int) -> Value:
    """Transposed 1-D convolution as one matmul against the kernel plus an overlap-add.

    P[i, oc, k] = sum_ic W[ic, oc, k] x[ic, i] lands at output position i * s + k. With the kernel
    zero-padded to m * s taps, k = k1 * s + k0 and the output, viewed as [L + m - 1, s], is the sum
    over k1 of P[:, :, k1, :] shifted down by k1 rows. ggml's own conv_transpose_1d is a direct loop
    that took a quarter of a Diffusion Policy U-Net step on CUDA."""
    if dilation != 1:
        raise _unsupported(b, f"transposed convolution with dilation {dilation}")
    if output_padding > padding:
        raise _unsupported(b, f"transposed convolution with output_padding {output_padding} > padding {padding}")
    n, c, length = x.shape
    ic, oc, k = w.shape
    s, m = stride, -(-k // stride)
    out_len = (length - 1) * s - 2 * padding + k + output_padding
    if m * s != k:
        w = b.concat(w, b.const(torch.zeros(ic, oc, m * s - k)), dim=2)
    wt = b.reshape_to(b.permute(w, perm=[1, 2, 0]), (oc * m * s, ic))  # folded into a stored weight
    rows = []
    for i in range(n):
        xi = b.reshape_to(b.view(x, dim=0, start=i, length=1) if n > 1 else x, (c, length))
        p = b.reshape_to(b.mul_mat(wt, b.transpose(xi, 0, 1)), (length, oc, m, s))
        acc = None
        for k1 in range(m):
            q = b.reshape_to(b.view(p, dim=2, start=k1, length=1) if m > 1 else p, (length, oc, s))
            if k1:
                q = b.concat(b.const(torch.zeros(k1, oc, s)), q, dim=0)
            if m - 1 - k1:
                q = b.concat(q, b.const(torch.zeros(m - 1 - k1, oc, s)), dim=0)
            acc = q if acc is None else b.add(acc, q)
        y = b.reshape_to(b.permute(acc, perm=[1, 0, 2]), (oc, (length + m - 1) * s))
        if padding or y.shape[1] != out_len:
            y = b.view(y, dim=1, start=padding, length=out_len)
        rows.append(b.reshape_to(y, (1, oc, out_len)))
    y = rows[0]
    for r in rows[1:]:
        y = b.concat(y, r, dim=0)
    if bias is not None:
        y = b.add(y, b.reshape_to(V(b, bias), (oc, 1)))
    return y


@lowering(aten.max_pool2d_with_indices.default)
def _max_pool2d(b, x, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
    if ceil_mode or _pair(dilation) != [1, 1]:
        raise _unsupported(b, "max_pool2d with ceil_mode or dilation")
    k = _pair(kernel_size)
    y = b.pool_2d(V(b, x), op="max", kernel=k, stride=_pair(stride) if stride else k, padding=_pair(padding))
    return [y, _Unavailable("max_pool2d indices")]


@lowering(aten.avg_pool2d.default)
def _avg_pool2d(b, x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True,
                divisor_override=None):
    p = _pair(padding)
    if ceil_mode or divisor_override is not None or (not count_include_pad and p != [0, 0]):
        raise _unsupported(b, "avg_pool2d with ceil_mode, divisor_override or count_include_pad=False")
    k = _pair(kernel_size)
    return b.pool_2d(V(b, x), op="avg", kernel=k, stride=_pair(stride) if stride else k, padding=p)


@lowering(aten.native_group_norm.default)
def _group_norm(b, x, weight, bias, n, c, hw, groups, eps):
    """Normalize each group's C/G * HxW values as one row, then the per-channel affine."""
    x = V(b, x)
    shape = x.shape
    y = b.norm(b.reshape_to(x, (n, groups, (c // groups) * hw)), eps=float(eps))
    y = b.reshape_to(y, shape)
    tail = (1,) * (len(shape) - 2)
    if weight is not None:
        y = b.mul(y, b.reshape_to(V(b, weight), (c,) + tail))
    if bias is not None:
        y = b.add(y, b.reshape_to(V(b, bias), (c,) + tail))
    return [y, _Unavailable("group_norm saved mean"), _Unavailable("group_norm saved rstd")]


@lowering(aten._native_batch_norm_legit_no_training.default)
def _batch_norm_eval(b, x, weight, bias, running_mean, running_var, momentum, eps):
    """Inference batch norm: x * inv + shift per channel (dim 1)."""
    x = V(b, x)
    c = x.shape[1]
    tail = (1,) * (x.rank - 2)
    inv = b.const((running_var.float() + eps).rsqrt().reshape((c,) + tail))
    shift = b.const((-running_mean.float()).reshape((c,) + tail))
    y = b.mul(b.add(x, shift), inv)
    if weight is not None:
        y = b.mul(y, b.reshape_to(V(b, weight), (c,) + tail))
    if bias is not None:
        y = b.add(y, b.reshape_to(V(b, bias), (c,) + tail))
    return [y, _Unavailable("batch_norm saved mean"), _Unavailable("batch_norm saved invstd")]


# --- comparisons, logic, selection: booleans are f32 0/1 ------------------------------------


def _diff(b: Builder, x: Any, y: Any) -> Value:
    """x - y with broadcasting, either side possibly a scalar."""
    return _binary(b, "sub", x, y)


@lowering(aten.gt.Tensor, aten.gt.Scalar)
def _gt(b, x, y):
    return b.step(_diff(b, x, y))


@lowering(aten.lt.Tensor, aten.lt.Scalar)
def _lt(b, x, y):
    return b.step(_diff(b, y, x) if isinstance(y, Value) else b.scale(V(b, x), s=-1.0, b=float(y)))


@lowering(aten.ge.Tensor, aten.ge.Scalar)
def _ge(b, x, y):
    return b.scale(_lt(b, x, y), s=-1.0, b=1.0)


@lowering(aten.le.Tensor, aten.le.Scalar)
def _le(b, x, y):
    return b.scale(_gt(b, x, y), s=-1.0, b=1.0)


@lowering(aten.eq.Tensor, aten.eq.Scalar)
def _eq(b, x, y):
    return b.mul(_le(b, x, y), _ge(b, x, y))


@lowering(aten.ne.Tensor, aten.ne.Scalar)
def _ne(b, x, y):
    return b.scale(_eq(b, x, y), s=-1.0, b=1.0)


@lowering(aten.logical_and.default, aten.bitwise_and.Tensor)
def _and(b, x, y):
    return _binary(b, "mul", x, y)


@lowering(aten.logical_or.default, aten.bitwise_or.Tensor)
def _or(b, x, y):
    return b.step(_binary(b, "add", x, y))


@lowering(aten.logical_not.default, aten.bitwise_not.default)
def _not(b, x):
    return b.scale(V(b, x), s=-1.0, b=1.0)


def _where(b: Builder, cond: Any, x: Any, y: Any) -> Value:
    """cond * x + (1 - cond) * y, written so a huge fill value (finfo.min) never meets a 0 * inf
    or cancels against x."""
    cond = V(b, cond)
    sx, sy = _scalar(x), _scalar(y)
    if sx is not None and sy is not None:  # cond * x + (1 - cond) * y: each term 0 or one fill value
        sx, sy = (max(min(v, 3.4e38), -3.4e38) for v in (sx, sy))
        return _binary(b, "add", b.scale(cond, s=sx, b=0.0), b.scale(cond, s=-sy, b=sy))
    if sy is not None:
        sy = max(min(sy, 3.4e38), -3.4e38)
        return _binary(b, "add", _binary(b, "mul", x, cond), b.scale(cond, s=-sy, b=sy))
    if sx is not None:
        sx = max(min(sx, 3.4e38), -3.4e38)
        notc = b.scale(cond, s=-1.0, b=1.0)
        return _binary(b, "add", _binary(b, "mul", y, notc), b.scale(notc, s=-sx, b=sx))
    notc = b.scale(cond, s=-1.0, b=1.0)
    return _binary(b, "add", _binary(b, "mul", x, cond), _binary(b, "mul", y, notc))


@lowering(aten.where.self, aten.where.ScalarSelf, aten.where.ScalarOther, aten.where.Scalar)
def _where_op(b, cond, x, y):
    return _where(b, cond, x, y)


@lowering(aten.masked_fill.Scalar, aten.masked_fill.Tensor)
def _masked_fill(b, x, mask, value):
    return _where(b, mask, value, x)


@lowering(aten._to_copy.default, aten.to.dtype, aten.type_as.default)
def _to_copy(b, x, *args, dtype=None, **kwargs):
    """Every runtime tensor is f32 except i32 indices, so most conversions are no-ops. Casting
    to bool is x != 0."""
    x = V(b, x)
    return _ne(b, x, 0.0) if dtype == torch.bool else x


@lowering(aten.cumsum.default)
def _cumsum(b, x, dim, dtype=None):
    x = V(b, x)
    d = _dim(dim, x.rank)
    if d == x.rank - 1:
        return b.cumsum(x)
    t = b.transpose(x, d, x.rank - 1)
    return b.transpose(b.cumsum(t), d, x.rank - 1)


@lowering(aten.sum.default)
def _sum_all(b, x, dtype=None):
    x = V(b, x)
    return b.reshape_to(b.sum_rows(b.reshape_to(x, (1, x.type.numel()))), ())


@lowering(aten.embedding.default)
def _embedding(b, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    weight, indices = V(b, weight), V(b, indices)
    if weight.rank != 2:
        raise _unsupported(b, f"embedding table of rank {weight.rank}")
    flat = b.reshape_to(indices, (indices.type.numel(),))
    rows = b.get_rows(weight, flat)
    return b.reshape_to(rows, tuple(indices.shape) + (weight.shape[1],))


@lowering(aten.index_select.default)
def _index_select(b, x, dim, index):
    x, index = V(b, x), V(b, index)
    d = _dim(dim, x.rank)
    if x.rank != 2 or d != 0:
        raise _unsupported(b, "index_select other than rows of a 2-D tensor")
    return b.get_rows(x, index)


@lowering(aten._assert_tensor_metadata.default, aten._assert_scalar.default, aten._assert_async.msg,
          aten.sym_constrain_range_for_size.default)
def _assertion(b, *args, **kwargs):
    """Trace-time checks with no result: nothing to run."""
    return None


# --- functionalized in-place writes ---------------------------------------------------------


@lowering(aten.copy.default)
def _copy(b, dst, src, non_blocking=False):
    """copy(dst, src): src broadcast to dst's shape; dst's old values are gone."""
    src = V(b, src)
    shape = dst.shape if isinstance(dst, (Value, torch.Tensor)) else b.out_type.shape
    shape = tuple(shape)
    return src if src.shape == shape else b.repeat(b.reshape_to(src, (1,) * (len(shape) - src.rank) + src.shape),
                                                   shape=list(shape))


@lowering(aten.slice_scatter.default)
def _slice_scatter(b, x, src, dim=0, start=None, end=None, step=1):
    """x with x[start:end] along dim replaced by src."""
    x, src = V(b, x), V(b, src)
    d = _dim(dim, x.rank)
    n = x.shape[d]
    start = 0 if start is None else (start + n if sym.definitely(start < 0) else start)
    end = n if end is None else sym.smin(n, end + n if sym.definitely(end < 0) else end)
    if step != 1:
        raise _unsupported(b, f"slice_scatter with step {step}")
    parts = []
    if start != 0:
        parts.append(b.view(x, dim=d, start=0, length=start))
    parts.append(src)
    if end != n:
        parts.append(b.view(x, dim=d, start=end, length=n - end))
    out = parts[0]
    for p in parts[1:]:
        out = b.concat(out, p, dim=d)
    return out


@lowering(aten.index_put.default, aten.index_put_.default)
def _index_put(b, x, indices, values, accumulate=False):
    """x[..., idx, :] = values for one index tensor on the second-to-last dim: ggml_set_rows. A KV
    cache written at runtime positions lowers to this, and ggml writes it in place."""
    x, values = V(b, x), V(b, values)
    used = [i for i, t in enumerate(indices) if t is not None]
    if accumulate or len(used) != 1 or used[0] != x.rank - 2:
        raise _unsupported(b, "index_put other than one index tensor on the second-to-last dim (rows)")
    idx = V(b, indices[used[0]])
    if idx.rank != 1:
        raise _unsupported(b, f"index_put with a {idx.rank}-D index")
    src = b.reshape_to(values, x.shape[:-2] + (idx.shape[0], x.shape[-1]))
    return b.set_rows(x, src, idx)


@lowering(aten.index_copy.default)
def _index_copy(b, x, dim, index, source):
    x = V(b, x)
    d = _dim(dim, x.rank)
    return _index_put(b, x, [None] * d + [index], source)


@lowering(aten.arange.default, aten.arange.start, aten.arange.start_step)
def _arange(b, *args, dtype=None, layout=None, device=None, pin_memory=None):
    """Only reached with a symbolic bound (constant ranges are folded): ggml_arange, in f32."""
    start, end, step = (0, args[0], 1) if len(args) == 1 else (args[0], args[1], args[2] if len(args) > 2 else 1)
    if not all(isinstance(v, int) or isinstance(v, sym.sympy.Basic) for v in (start, end, step)) or step == 0:
        raise _unsupported(b, f"arange({start}, {end}, {step})")
    return b.arange(start=start, stop=end, step=step)


# --- min / max reductions: argmax over rows, then gather -----------------------------------


def _to_rows(b: Builder, x: Value, d: int) -> tuple[Value, tuple[int, ...]]:
    """x with dim d moved last and flattened to [rows, n]; also the shape of the rows."""
    if d != x.rank - 1:
        x = b.transpose(x, d, x.rank - 1)
    lead = x.shape[:-1]
    rows = 1
    for s in lead:
        rows *= s
    return b.reshape_to(x, (rows, x.shape[-1])), lead


def _arg_extreme(b: Builder, x: Value, d: int, largest: bool) -> tuple[Value, Value, tuple[int, ...]]:
    """(values [rows, 1], indices [rows] i32, lead shape) of the max (or min) along dim d."""
    m, lead = _to_rows(b, x, d)
    rows, n = m.shape
    idx = b.argmax(m if largest else b.neg(m))
    flat = b.add(b.cast(idx, dtype="f32"), b.const(torch.arange(rows, dtype=torch.float32) * n))
    vals = b.get_rows(b.reshape_to(m, (rows * n, 1)), flat)
    return vals, idx, lead


def _reduce_shape(x: Value, d: int, keepdim: bool) -> tuple[int, ...]:
    return tuple(1 if i == d else s for i, s in enumerate(x.shape)) if keepdim else x.shape[:d] + x.shape[d + 1:]


def _minmax_dim(largest: bool):
    def lower(b, x, dim, keepdim=False):
        x = V(b, x)
        d = _dim(dim, x.rank)
        vals, idx, _ = _arg_extreme(b, x, d, largest)
        shape = _reduce_shape(x, d, keepdim)
        return [b.reshape_to(vals, shape), b.reshape_to(idx, shape)]

    return lower


def _minmax_values(largest: bool):
    def lower(b, x, dim=(), keepdim=False):
        x = V(b, x)
        dims = [dim] if isinstance(dim, int) else list(dim)
        if len(dims) != 1:
            raise _unsupported(b, "amax/amin over several dims")
        d = _dim(dims[0], x.rank)
        vals, _, _ = _arg_extreme(b, x, d, largest)
        return b.reshape_to(vals, _reduce_shape(x, d, keepdim))

    return lower


def _argext(largest: bool):
    def lower(b, x, dim=None, keepdim=False):
        x = V(b, x)
        if dim is None:
            x, dim = b.reshape_to(x, (x.type.numel(),)), 0
        d = _dim(dim, x.rank)
        _, idx, _ = _arg_extreme(b, x, d, largest)
        return b.reshape_to(idx, _reduce_shape(x, d, keepdim))

    return lower


lowering(aten.max.dim)(_minmax_dim(True))
lowering(aten.min.dim)(_minmax_dim(False))
lowering(aten.amax.default)(_minmax_values(True))
lowering(aten.amin.default)(_minmax_values(False))
lowering(aten.argmax.default)(_argext(True))
lowering(aten.argmin.default)(_argext(False))


@lowering(aten.repeat.default)
def _tile(b, x, repeats):
    """Tensor.repeat tiles: out[i] = x[i % n] along every dim, which is ggml_repeat."""
    x = V(b, x)
    out = b.out_type.shape
    x = b.reshape_to(x, (1,) * (len(out) - x.rank) + x.shape)
    return x if x.shape == out else b.repeat(x, shape=list(out))
