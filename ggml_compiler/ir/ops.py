"""Op registry: shape/dtype inference and a PyTorch reference for every IR op.

Tier 1 ops map one-to-one onto a ggml call and are the only ops the emitters accept.
Tier 2 ops are source-level conveniences (linear, layer_norm, sdpa) that
`passes.lower_tier2` rewrites into tier 1. Keeping them until then lets fusion passes see
them whole.

Semantics are PyTorch semantics on PyTorch-order shapes. Where ggml differs from PyTorch,
the op follows ggml and says so (mul_mat batch broadcasting, binary-op broadcasting).

Dims may be symbolic (`ir.sym`). Inference rejects a shape only when it is wrong for every
binding in range; attributes may hold symbolic dims too, and references see them evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import sym
from .graph import TensorType

Infer = Callable[[list[TensorType], dict[str, Any]], TensorType]
Ref = Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor]


@dataclass(frozen=True)
class OpDef:
    name: str
    tier: int
    arity: tuple[int, int]  # min, max inputs
    infer: Infer
    ref: Ref


OPS: dict[str, OpDef] = {}


class ShapeError(ValueError):
    pass


def _register(name: str, tier: int, arity: tuple[int, int], infer: Infer, ref: Ref) -> None:
    OPS[name] = OpDef(name, tier, arity, infer, ref)


def get(name: str) -> OpDef:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"unknown IR op {name!r}") from None


# --- helpers -------------------------------------------------------------------------------


def _same(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    return ts[0]


def _check_rank(t: TensorType, lo: int = 0, hi: int = 4) -> None:
    if not lo <= t.rank <= hi:
        raise ShapeError(f"rank {t.rank} outside [{lo}, {hi}]")


def _broadcast_into(a: TensorType, b: TensorType, what: str) -> None:
    """b must broadcast into a without changing a's shape (ggml_can_repeat, restricted to the
    PyTorch-compatible case where each b dim equals a's or is 1)."""
    if b.rank > a.rank:
        raise ShapeError(f"{what}: operand {b} has higher rank than {a}")
    for da, db in zip(reversed(a.shape), reversed(b.shape)):
        if db != da and db != 1:
            raise ShapeError(f"{what}: cannot broadcast {b} into {a}")


# --- tier 1: elementwise -------------------------------------------------------------------


def _binary(name: str, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    def infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
        _broadcast_into(ts[0], ts[1], name)
        return ts[0]

    _register(name, 1, (2, 2), infer, lambda xs, a: fn(xs[0], xs[1]))


_binary("add", torch.add)
_binary("sub", torch.sub)
_binary("mul", torch.mul)
_binary("div", torch.div)

_UNARY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": torch.relu,
    # ggml_gelu is the tanh approximation; on CPU it goes through an fp16 lookup table.
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_erf": lambda x: F.gelu(x, approximate="none"),
    "silu": F.silu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
    "log": torch.log,
    "neg": torch.neg,
    "abs": torch.abs,
    "sqr": torch.square,
    "sqrt": torch.sqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "softplus": F.softplus,
    "cont": lambda x: x.contiguous(),
}
for _name, _fn in _UNARY.items():
    _register(_name, 1, (1, 1), _same, (lambda f: lambda xs, a: f(xs[0]))(_fn))

def _glu_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    if ts[0].shape != ts[1].shape:
        raise ShapeError(f"swiglu: gate {ts[0]} and up {ts[1]} differ")
    return ts[0]


# swiglu(gate, up) = silu(gate) * up, one ggml_swiglu_split (a SwiGLU MLP).
_register("swiglu", 1, (2, 2), _glu_infer, lambda xs, a: F.silu(xs[0]) * xs[1])
_register("scale", 1, (1, 1), _same, lambda xs, a: xs[0] * a["s"] + a.get("b", 0.0))
_register("step", 1, (1, 1), _same, lambda xs, a: (xs[0] > 0).to(xs[0].dtype))
_register("clamp", 1, (1, 1), _same, lambda xs, a: torch.clamp(xs[0], a["min"], a["max"]))
_register(
    "cast",
    1,
    (1, 1),
    lambda ts, a: TensorType(ts[0].shape, a["dtype"]),
    lambda xs, a: xs[0].to({"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "i32": torch.int32}[a["dtype"]]),
)


# --- tier 1: reductions over the last dim --------------------------------------------------


def _row_reduce(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    t = ts[0]
    if t.rank == 0:
        raise ShapeError("cannot reduce a scalar")
    return TensorType(t.shape[:-1] + (1,), t.dtype)


_register("mean", 1, (1, 1), _row_reduce, lambda xs, a: xs[0].mean(-1, keepdim=True))
_register("cumsum", 1, (1, 1), _same, lambda xs, a: xs[0].cumsum(-1))
_register("sum_rows", 1, (1, 1), _row_reduce, lambda xs, a: xs[0].sum(-1, keepdim=True))


def _norm_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x = xs[0]
    return F.layer_norm(x, (x.shape[-1],), eps=a["eps"])


def _rms_norm_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x = xs[0]
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + a["eps"])


_register("norm", 1, (1, 1), _same, _norm_ref)
_register("rms_norm", 1, (1, 1), _same, _rms_norm_ref)


def _soft_max_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    if len(ts) > 1:
        _broadcast_into(ts[0], ts[1], "soft_max mask")
    return ts[0]


def _soft_max_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x = xs[0] * a.get("scale", 1.0)
    if len(xs) > 1:
        x = x + xs[1]
    return torch.softmax(x, dim=-1)


_register("soft_max", 1, (1, 2), _soft_max_infer, _soft_max_ref)


# --- tier 1: matmul ------------------------------------------------------------------------


def _mul_mat_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    wa, xb = ts
    _check_rank(wa, 2)
    _check_rank(xb, 2)
    if wa.shape[-1] != xb.shape[-1]:
        raise ShapeError(f"mul_mat: inner dims differ, {wa} vs {xb}")
    rank = max(wa.rank, xb.rank)
    pa = (1,) * (rank - wa.rank) + wa.shape
    pb = (1,) * (rank - xb.rank) + xb.shape
    for da, db in zip(pa[:-2], pb[:-2]):
        if not sym.divides(da, db):
            raise ShapeError(f"mul_mat: batch dims of {wa} do not divide those of {xb}")
    return TensorType(pb[:-2] + (pb[-2], pa[-2]), xb.dtype)


def _mul_mat_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    """b @ a^T. A batch dim of `a` smaller than b's is repeat-interleaved, as ggml does: b batch
    i reads a batch i // (nb / na). That is grouped-query attention's head sharing."""
    wa, xb = xs
    rank = max(wa.dim(), xb.dim())
    wa = wa.reshape((1,) * (rank - wa.dim()) + tuple(wa.shape))
    xb = xb.reshape((1,) * (rank - xb.dim()) + tuple(xb.shape))
    for d in range(rank - 2):
        if wa.shape[d] != xb.shape[d]:
            wa = wa.repeat_interleave(xb.shape[d] // wa.shape[d], dim=d)
    return xb.float() @ wa.float().transpose(-1, -2)


_register("mul_mat", 1, (2, 2), _mul_mat_infer, _mul_mat_ref)


# --- tier 1: indexing ----------------------------------------------------------------------


def _get_rows_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    table, idx = ts
    if table.rank != 2 or idx.rank != 1 or idx.dtype != "i32":
        raise ShapeError(f"get_rows: needs a [rows, cols] table and 1-D i32 indices, got {table}, {idx}")
    return TensorType((idx.shape[0], table.shape[1]), "f32")


_register("get_rows", 1, (2, 2), _get_rows_infer, lambda xs, a: xs[0].float()[xs[1].long()])


def _argmax_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    if ts[0].rank != 2:
        raise ShapeError(f"argmax: needs a [rows, n] matrix, got {ts[0]}")
    return TensorType((ts[0].shape[0],), "i32")


_register("argmax", 1, (1, 1), _argmax_infer, lambda xs, a: xs[0].argmax(-1).to(torch.int32))


def _set_rows_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    """set_rows(dst, src, idx): dst with rows idx (along dim -2) replaced by src's rows. ggml writes
    them into dst's own memory: the result is dst, updated (a KV cache written in place)."""
    dst, src, idx = ts
    if dst.rank < 2 or src.rank != dst.rank or idx.rank != 1 or idx.dtype != "i32":
        raise ShapeError(f"set_rows: needs dst and src of one rank >= 2 and 1-D i32 indices, got {dst}, {src}, {idx}")
    if src.shape[-1] != dst.shape[-1] or src.shape[:-2] != dst.shape[:-2] or src.shape[-2] != idx.shape[0]:
        raise ShapeError(f"set_rows: rows {src} with indices {idx} do not fit {dst}")
    return dst


def _set_rows_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    dst, src, idx = xs
    out = dst.clone()
    out[..., idx.long(), :] = src.to(out.dtype)
    return out


_register("set_rows", 1, (3, 3), _set_rows_infer, _set_rows_ref)


def _rope_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    """rope(x [..., S, H, D], positions [S] i32 [, freq_factors [n_dims/2]]): rotary embedding of
    the first n_dims of D, as ggml_rope_ext (mode "neox": the halves rotate together, as in HF's
    rotate_half)."""
    x, pos = ts[0], ts[1]
    if x.rank < 3 or pos.rank != 1 or pos.dtype != "i32" or pos.shape[0] != x.shape[-3]:
        raise ShapeError(f"rope: needs x [..., S, H, D] and i32 positions [S], got {x}, {pos}")
    if a["n_dims"] % 2 or a["n_dims"] > x.shape[-1]:
        raise ShapeError(f"rope: n_dims {a['n_dims']} does not fit head size {x.shape[-1]}")
    if len(ts) > 2 and ts[2].shape != (a["n_dims"] // 2,):
        raise ShapeError(f"rope: freq_factors {ts[2]} for {a['n_dims']} rotated dims")
    return TensorType(x.shape, "f32")


def _rope_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x, pos = xs[0].float(), xs[1].double()
    n = a["n_dims"]
    inv = a["freq_base"] ** (-torch.arange(0, n, 2, dtype=torch.float64) / n) * a.get("freq_scale", 1.0)
    if len(xs) > 2:
        inv = inv / xs[2].double()
    theta = pos[:, None] * inv[None, :]  # [S, n/2]
    shape = (1,) * (x.dim() - 3) + (x.shape[-3], 1, n // 2)
    cos = (torch.cos(theta) * a.get("attn_factor", 1.0)).float().reshape(shape)
    sin = (torch.sin(theta) * a.get("attn_factor", 1.0)).float().reshape(shape)
    x1, x2, rest = x[..., : n // 2], x[..., n // 2: n], x[..., n:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)


_register("rope", 1, (2, 3), _rope_infer, _rope_ref)


def _arange_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    """arange(start, stop, step): [ceil((stop - start) / step)] f32, as ggml_arange; stop may be
    a symbolic dim (a mask over the KV rows in use)."""
    n = sym.norm(-((a["start"] - a["stop"]) // a["step"]))
    if sym.definitely(n < 1):
        raise ShapeError(f"arange: empty range {a['start']}..{a['stop']} step {a['step']}")
    return TensorType((n,), "f32")


_register("arange", 1, (0, 0), _arange_infer,
          lambda xs, a: torch.arange(a["start"], a["stop"], a["step"], dtype=torch.float32))


# --- tier 1: convolution and pooling -------------------------------------------------------
# Attributes are PyTorch-style pairs, (height, width).


def _conv_out(size: int, k: int, s: int, p: int, d: int) -> int:
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def _im2col_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    w, x = ts
    if w.rank != 4 or x.rank != 4 or w.shape[1] != x.shape[1]:
        raise ShapeError(f"im2col: kernel {w} does not fit input {x}")
    (sh, sw), (ph, pw), (dh, dw) = a["stride"], a["padding"], a["dilation"]
    oh = _conv_out(x.shape[2], w.shape[2], sh, ph, dh)
    ow = _conv_out(x.shape[3], w.shape[3], sw, pw, dw)
    if sym.definitely(oh < 1) or sym.definitely(ow < 1):
        raise ShapeError(f"im2col: kernel {w} larger than padded input {x}")
    return TensorType((x.shape[0], oh, ow, w.shape[1] * w.shape[2] * w.shape[3]), x.dtype)


def _im2col_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    """[N, OH, OW, C*KH*KW], patch entries ordered (c, kh, kw) like ggml_im2col."""
    w, x = xs
    n = x.shape[0]
    cols = F.unfold(x.float(), tuple(w.shape[2:]), dilation=tuple(a["dilation"]), padding=tuple(a["padding"]),
                    stride=tuple(a["stride"]))
    out = _im2col_infer([TensorType(tuple(w.shape)), TensorType(tuple(x.shape))], a)
    return cols.reshape(n, out.shape[3], out.shape[1], out.shape[2]).permute(0, 2, 3, 1).contiguous()


_register("im2col", 1, (2, 2), _im2col_infer, _im2col_ref)


def _conv_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    w, x = ts[0], ts[1]
    n, oh, ow, _ = _im2col_infer([w, x], a).shape
    if len(ts) > 2 and ts[2].shape != (w.shape[0],):
        raise ShapeError(f"conv_2d: bias {ts[2]} does not match {w.shape[0]} output channels")
    return TensorType((n, w.shape[0], oh, ow), x.dtype)


def _conv_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    w, x = xs[0].float(), xs[1].float()
    bias = xs[2].float() if len(xs) > 2 else None
    return F.conv2d(x, w, bias, tuple(a["stride"]), tuple(a["padding"]), tuple(a["dilation"]))


# conv_2d(kernel, x [, bias]): the runtime picks the kernel per backend (direct implicit GEMM for
# f16 kernels on a GPU, f32 im2col + matmul otherwise).
_register("conv_2d", 1, (2, 3), _conv_infer, _conv_ref)


def _pool_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    x = ts[0]
    if x.rank != 4:
        raise ShapeError(f"pool_2d: input {x} is not [N, C, H, W]")
    (kh, kw), (sh, sw), (ph, pw) = a["kernel"], a["stride"], a["padding"]
    return TensorType((x.shape[0], x.shape[1], _conv_out(x.shape[2], kh, sh, ph, 1),
                       _conv_out(x.shape[3], kw, sw, pw, 1)), x.dtype)


def _pool_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    k, s, p = tuple(a["kernel"]), tuple(a["stride"]), tuple(a["padding"])
    if a["op"] == "max":
        return F.max_pool2d(xs[0], k, s, p)
    return F.avg_pool2d(xs[0], k, s, p, count_include_pad=True)


_register("pool_2d", 1, (1, 1), _pool_infer, _pool_ref)


# --- tier 1: layout ------------------------------------------------------------------------


def _reshape_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    out = TensorType(tuple(a["shape"]), ts[0].dtype)
    if out.numel() != ts[0].numel():
        raise ShapeError(f"reshape: {ts[0]} has {ts[0].numel()} elements, target {out.shape} has {out.numel()}")
    return out


_register("reshape", 1, (1, 1), _reshape_infer, lambda xs, a: xs[0].reshape(a["shape"]))


def _permute_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    perm = list(a["perm"])
    t = ts[0]
    if sorted(perm) != list(range(t.rank)):
        raise ShapeError(f"permute: {perm} is not a permutation of rank {t.rank}")
    return TensorType(tuple(t.shape[p] for p in perm), t.dtype)


_register("permute", 1, (1, 1), _permute_infer, lambda xs, a: xs[0].permute(a["perm"]))


def _repeat_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    out = TensorType(tuple(a["shape"]), ts[0].dtype)
    t = ts[0]
    if t.rank > out.rank or not all(sym.divides(s, o) for o, s in zip(reversed(out.shape), reversed(t.shape))):
        raise ShapeError(f"repeat: cannot tile {t} to {out.shape}")
    return out


def _repeat_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x = xs[0]
    shape = tuple(a["shape"])
    x = x.reshape((1,) * (len(shape) - x.dim()) + tuple(x.shape))
    return x.repeat(tuple(o // s for o, s in zip(shape, x.shape)))


_register("repeat", 1, (1, 1), _repeat_infer, _repeat_ref)


def _concat_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    x, y = ts
    d = a["dim"]
    if x.rank != y.rank or not 0 <= d < x.rank:
        raise ShapeError(f"concat: bad operands {x}, {y} for dim {d}")
    for i, (p, q) in enumerate(zip(x.shape, y.shape)):
        if i != d and p != q:
            raise ShapeError(f"concat: {x} and {y} differ outside dim {d}")
    shape = list(x.shape)
    shape[d] += y.shape[d]
    return TensorType(tuple(shape), x.dtype)


_register("concat", 1, (2, 2), _concat_infer, lambda xs, a: torch.cat(xs, dim=a["dim"]))


def _view_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    t = ts[0]
    d, start, length = a["dim"], a["start"], a["length"]
    if not 0 <= d < t.rank or sym.definitely(start < 0) or sym.definitely(length < 1) or \
            sym.definitely(start + length > t.shape[d]):
        raise ShapeError(f"view: slice [{start}, {start + length}) out of range for dim {d} of {t}")
    shape = list(t.shape)
    shape[d] = length
    return TensorType(tuple(shape), t.dtype)


_register("view", 1, (1, 1), _view_infer, lambda xs, a: xs[0].narrow(a["dim"], a["start"], a["length"]))


# --- tier 1: fused attention ---------------------------------------------------------------


def _flash_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    q, k, v = ts[:3]
    if q.rank != 4 or k.rank != 4 or v.rank != 4:
        raise ShapeError(f"flash_attn: q, k, v must be [B, H, S, D], got {q}, {k}, {v}")
    if k.shape[-1] != q.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[0] != k.shape[0] or q.shape[1] % k.shape[1]:
        raise ShapeError(f"flash_attn: q {q}, k {k}, v {v} do not line up")
    if len(ts) > 3:
        m = ts[3]
        if m.shape[-1] != k.shape[-2] or sym.definitely(m.shape[-2] < q.shape[-2]):
            raise ShapeError(f"flash_attn: mask {m} does not cover [{q.shape[-2]}, {k.shape[-2]}]")
    return TensorType((q.shape[0], q.shape[2], q.shape[1], v.shape[-1]), "f32")  # [B, Sq, H, Dv]


def _flash_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    q, k, v = (x.float() for x in xs[:3])
    mask = xs[3].float()[..., : q.shape[-2], :] if len(xs) > 3 else None
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a["scale"])
    return out.permute(0, 2, 1, 3).contiguous()


_register("flash_attn", 1, (3, 4), _flash_infer, _flash_ref)


# --- tier 2 --------------------------------------------------------------------------------


def _linear_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    x, w = ts[0], ts[1]
    if w.rank != 2 or x.shape[-1] != w.shape[1]:
        raise ShapeError(f"linear: {x} @ {w}^T")
    out = TensorType(x.shape[:-1] + (w.shape[0],), x.dtype)
    if len(ts) > 2:
        _broadcast_into(out, ts[2], "linear bias")
    return out


_register(
    "linear",
    2,
    (2, 3),
    _linear_infer,
    lambda xs, a: F.linear(xs[0], xs[1], xs[2] if len(xs) > 2 else None),
)


def _layer_norm_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    x = xs[0]
    w = xs[1] if a.get("has_weight") else None
    b = xs[-1] if a.get("has_bias") else None
    return F.layer_norm(x, (x.shape[-1],), w, b, a["eps"])


_register("layer_norm", 2, (1, 3), _same, _layer_norm_ref)
# tanh-approximated GELU in f32. The runtime emits ggml_gelu on GPUs (an f32 formula there) and the
# formula as separate ggml ops on CPU, whose ggml_gelu reads an fp16 table (~1e-3).
# precision="fast" rewrites it to `gelu` (ggml_gelu everywhere).
_register("gelu_tanh", 1, (1, 1), _same, lambda xs, a: F.gelu(xs[0], approximate="tanh"))


def _sdpa_infer(ts: list[TensorType], a: dict[str, Any]) -> TensorType:
    q, k, v = ts[:3]
    if k.shape[-1] != q.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ShapeError(f"sdpa: q {q}, k {k}, v {v} do not line up")
    return TensorType(q.shape[:-1] + (v.shape[-1],), q.dtype)


def _sdpa_ref(xs: list[torch.Tensor], a: dict[str, Any]) -> torch.Tensor:
    q, k, v = xs[:3]
    mask = xs[3] if len(xs) > 3 else None
    if k.shape[-3] != q.shape[-3]:  # grouped-query attention
        r = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(r, dim=-3)
        v = v.repeat_interleave(r, dim=-3)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=a["scale"])


_register("sdpa", 2, (3, 4), _sdpa_infer, _sdpa_ref)


def default_sdpa_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
