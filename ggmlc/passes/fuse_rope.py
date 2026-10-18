"""Fuse rotary position embeddings into ggml_rope_ext (flag fuse_rope, O2).

Hugging Face writes RoPE as

    freqs = inv_freq (const) outer positions        # [.., S, D/2]
    emb = cat(freqs, freqs); cos = emb.cos() * a; sin = emb.sin() * a
    x * cos + cat(-x[..., D/2:], x[..., :D/2]) * sin    # rotate_half

over x [B, H, S, D]. That is ggml_rope_ext in NEOX mode on x with the heads and tokens swapped
(ggml wants the token dim third from the inside), with the base recovered from inv_freq; an
inv_freq that is not a plain base^(-2i/D) (Llama 3's scaled RoPE) becomes ggml's freq_factors.
ggml computes the angles by repeated multiplication in f32, so cos/sin move by ~1e-4 at position
1000: hence O2.
"""

from __future__ import annotations

import math

import torch

from ..ir import Builder, Const, Node, Stage, Value
from . import rewrite

_THROUGH = {"reshape", "permute", "cast", "cont"}


def _strip(v: Value) -> Value:
    while isinstance(v, Node) and v.op in _THROUGH:
        v = v.inputs[0]
    return v


def _scaled(v: Value, op: str) -> tuple[Node, float] | None:
    """v = [reshape*] [scale(s)] op(E): (that op node, s)."""
    v = _strip(v)
    s = 1.0
    if isinstance(v, Node) and v.op == "scale" and v.attrs.get("b", 0.0) == 0.0:
        s, v = float(v.attrs["s"]), _strip(v.inputs[0])
    return (v, s) if isinstance(v, Node) and v.op == op else None


def _table(e: Value) -> tuple[Value, torch.Tensor] | None:
    """e = concat(F, F) with F = positions outer inv_freq: (the i32 positions, inv_freq)."""
    e = _strip(e)
    if not (isinstance(e, Node) and e.op == "concat" and _strip(e.inputs[0]) is _strip(e.inputs[1])):
        return None
    f = _strip(e.inputs[0])
    if not (isinstance(f, Node) and f.op == "mul_mat" and f.inputs[0].shape[-1] == 1):
        return None
    for a, b in ((f.inputs[0], f.inputs[1]), (f.inputs[1], f.inputs[0])):
        c, p = _strip(a), _strip(b)
        if isinstance(c, Const) and isinstance(p, Value) and p.type.dtype == "i32":
            return p, c.data.reshape(-1).double()
    return None


def _rotate_half(r: Value, x: Value) -> bool:
    """r = concat(neg(x[..., h:]), x[..., :h]) along the last dim, h = D / 2."""
    if not (isinstance(r, Node) and r.op == "concat" and r.attrs["dim"] == r.rank - 1):
        return False
    a, b = r.inputs
    d = x.shape[-1]
    if not (isinstance(a, Node) and a.op == "neg" and isinstance(b, Node) and b.op == "view"):
        return False
    a = a.inputs[0]
    ok = lambda v, start: (isinstance(v, Node) and v.op == "view" and v.inputs[0] is x  # noqa: E731
                           and v.attrs["dim"] == x.rank - 1 and v.attrs["start"] == start
                           and v.attrs["length"] == d // 2)
    return ok(a, d // 2) and ok(b, 0)


def _base_and_factors(inv: torch.Tensor, d: int) -> tuple[float, torch.Tensor | None]:
    i = torch.arange(1, len(inv), dtype=torch.float64)
    base = math.exp(float((-torch.log(inv[1:]) * d / (2 * i)).median())) if len(inv) > 1 else 10000.0
    plain = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    factors = plain / inv  # ggml divides each angle by its factor
    if float((factors - 1).abs().max()) < 1e-6:
        return base, None
    return base, factors.float()


def fuse_rope(stage: Stage) -> int:
    count = 0

    def fix(b: Builder, n: Node) -> Value | None:
        nonlocal count
        if n.op != "add" or n.rank < 3:
            return None
        for m1, m2 in ((n.inputs[0], n.inputs[1]), (n.inputs[1], n.inputs[0])):
            if not (isinstance(m1, Node) and m1.op == "mul" and isinstance(m2, Node) and m2.op == "mul"):
                continue
            for x, c in ((m1.inputs[0], m1.inputs[1]), (m1.inputs[1], m1.inputs[0])):
                cos = _scaled(c, "cos")
                if cos is None or x.shape != n.shape:
                    continue
                for r, sv in ((m2.inputs[0], m2.inputs[1]), (m2.inputs[1], m2.inputs[0])):
                    sin = _scaled(sv, "sin")
                    if sin is None or sin[1] != cos[1] or not _rotate_half(r, x):
                        continue
                    if _strip(sin[0].inputs[0]) is not _strip(cos[0].inputs[0]):
                        continue
                    t = _table(cos[0].inputs[0])
                    if t is None:
                        continue
                    pos, inv = t
                    d = x.shape[-1]
                    if len(inv) != d // 2 or pos.type.numel() != x.shape[-2]:
                        continue
                    base, factors = _base_and_factors(inv, d)
                    perm = list(range(x.rank))
                    perm[-2], perm[-3] = perm[-3], perm[-2]  # [.., H, S, D] -> [.., S, H, D]
                    ins = [b.permute(x, perm=perm), b.reshape_to(pos, (pos.type.numel(),))]
                    if factors is not None:
                        ins.append(b.const(factors))
                    y = b.rope(*ins, n_dims=d, mode="neox", freq_base=base, attn_factor=cos[1])
                    count += 1
                    return b.permute(y, perm=perm)
        return None

    rewrite(stage, fix)
    return count
