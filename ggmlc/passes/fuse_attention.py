"""Fuse eager attention into ggml_flash_attn_ext (compile option attention="flash").

Matches  o = mul_mat(V^T, soft_max(scale(mul_mat(K, Q)) [+ mask]))  where the score and
probability nodes have no other readers, and
rewrites it as  permute(flash_attn(Q, f16(K), f16(V) [, f16(mask)]), [0, 2, 1, 3]).
K, V and the mask go through f16 (flash attention reads them in f16), so this changes numerics
at the 1e-3 level: it is a performance option, off for parity builds.
"""

from __future__ import annotations

from ..ir import Builder, Node, Stage, Value
from . import rewrite


# Head sizes every backend's flash attention handles with K and V the same size (CUDA's list at
# llama.cpp b11223; the CPU takes any). Fusing anything else would make a GGUF that loads on the
# CPU but not on a GPU.
PORTABLE_HEAD_DIMS = {40, 64, 72, 80, 96, 112, 128, 256}


def _as4(b: Builder, v: Value) -> Value:
    return b.reshape_to(v, (1,) * (4 - v.rank) + v.shape)


def fuse_attention(stage: Stage, row_guard: bool = True) -> int:
    """row_guard=False (flag assume_attended_rows) trusts that every query row may attend to some
    key, as a causal mask guarantees, and drops the fully-masked-row handling."""
    users = stage.users()
    count = 0

    def only_reader(v: Value, reader: Node) -> bool:
        return all(u is reader for u in users.get(id(v), [])) and not any(o is v for o in stage.outputs)

    def fix(b: Builder, n: Node) -> Value | None:
        nonlocal count
        if n.op != "mul_mat":
            return None
        vt, p = n.inputs
        if not isinstance(p, Node) or p.op != "soft_max" or not only_reader(p, n):
            return None
        s, scale, reader = p.inputs[0], p.attrs.get("scale", 1.0), p
        mask = p.inputs[1] if len(p.inputs) > 1 else None
        keep = None  # 0/1 per (query, key), when the mask came from where()
        if mask is None and scale == 1.0 and isinstance(s, Node) and s.op == "add" and only_reader(s, p):
            x, m = s.inputs
            if (isinstance(x, Node) and x.op == "mul" and isinstance(m, Node) and m.op == "scale"
                    and m.inputs[0] is x.inputs[1] and m.attrs["s"] == -m.attrs.get("b", 0.0) and only_reader(x, s)):
                # where(c, scores, big_neg) lowered as scores * c + (c - 1) * big: with c in {0, 1}
                # that is scores + additive mask (c - 1) * big.
                s, mask, reader, keep = x.inputs[0], m, x, x.inputs[1]
            else:
                # scores + mask, the usual HF form. The softmax has no scale of its own here, so
                # the mask stays unscaled under flash attention too.
                s, mask, reader = x, m, s
        if isinstance(s, Node) and s.op == "scale" and s.attrs.get("b", 0.0) == 0.0 and only_reader(s, reader):
            scale *= s.attrs["s"]
            s, reader = s.inputs[0], s
        if not isinstance(s, Node) or s.op != "mul_mat" or not only_reader(s, reader):
            return None
        k, q = s.inputs
        if q.shape[-1] not in PORTABLE_HEAD_DIMS or vt.shape[-2] != q.shape[-1]:
            return None
        v = b.transpose(vt, vt.rank - 2, vt.rank - 1)  # simplify folds this into vt's own permute
        if not (2 <= q.rank <= 4 and q.rank == k.rank == v.rank) or q.shape[:-3] != k.shape[:-3]:
            return None
        q4, k4, v4 = _as4(b, q), _as4(b, k), _as4(b, v)
        if q4.shape[0] != k4.shape[0] or q4.shape[1] % k4.shape[1]:
            return None
        ins = [q4, b.cast(k4, dtype="f16"), b.cast(v4, dtype="f16")]
        if mask is not None:
            if mask.rank > 4 or mask.shape[-1] != k.shape[-2]:
                return None
            m4 = _as4(b, mask)
            if m4.shape[-2] != q.shape[-2]:  # one row for all queries: flash attention reads one per query
                if m4.shape[-2] != 1:
                    return None
                m4 = b.repeat(m4, shape=list(m4.shape[:2]) + [q.shape[-2], m4.shape[-1]])
            if m4.shape[1] not in (1, q4.shape[1]) or m4.shape[0] not in (1, q4.shape[0]):
                return None  # the mask must broadcast over batch and heads as is
            # f16 saturates at 65504: a finfo.min mask would become -inf, and a fully masked row
            # would turn into NaN.
            ins.append(b.cast(b.clamp(m4, min=-65504.0, max=65504.0), dtype="f16"))
            if not row_guard:
                return _finish(b, ins, scale, n)
            # A fully masked row (padding tokens as queries): PyTorch replaces every score with the
            # fill value, so its softmax is uniform. An additive mask only shifts the row, which
            # softmax ignores. Zeroing that row's query makes all its scores 0: uniform again.
            k4 = _as4(b, keep) if keep is not None else b.step(b.add_scalar(m4, 1e4))  # key visible?
            if k4.shape[-2] == 1 and q.shape[-2] != 1:
                k4 = b.repeat(k4, shape=list(k4.shape[:2]) + [q.shape[-2], k4.shape[-1]])
            any_key = b.step(b.sum_rows(k4))  # [., ., Sq, 1]
            ins[0] = b.mul(q4, any_key)
        return _finish(b, ins, scale, n)

    def _finish(b: Builder, ins: list[Value], scale: float, n: Node) -> Value:
        nonlocal count
        fa = b.flash_attn(*ins, scale=float(scale))  # [B, Sq, H, Dv]
        out = b.permute(fa, perm=[0, 2, 1, 3])  # [B, H, Sq, Dv]
        count += 1
        return b.reshape_to(out, n.shape)

    rewrite(stage, fix)
    return count
