"""Bring tensors above 4-D down to ggml's limit by dropping size-1 dims.

PyTorch code often goes through 5-D shapes on the way between two 4-D ones (the packed q/k/v
projection in nn.MultiheadAttention does unflatten, unsqueeze, transpose, squeeze). With a
static batch of 1 those shapes have size-1 dims to spare. Each value above 4-D gets a
physical stand-in holding its non-unit dims in order; ops that touch it are rewritten against
the stand-in. Ops this pass cannot rewrite are left for the layout pass to report. Parameters above 4-D take
their squeezed type; `emit.gguf_runtime.ne_of` gives their driver values the same layout.
"""

from __future__ import annotations

from ..ir import Builder, Node, Stage, TensorType, Value
from . import rewrite

# Ops that act on each element alone, so dropping size-1 dims cannot change their meaning.
_ELEMENTWISE_UNARY = {"relu", "gelu", "gelu_erf", "silu", "tanh", "sigmoid", "exp", "log", "neg", "abs", "sqr",
                      "sqrt", "sin", "cos", "softplus", "gelu_tanh", "cont", "scale", "clamp", "cast", "step"}
_ELEMENTWISE_BINARY = {"add", "sub", "mul", "div", "swiglu"}
# Ops over the last dim: squeezing keeps their meaning as long as that dim is not 1.
_ROW = {"sum_rows", "mean", "norm", "rms_norm", "soft_max", "cumsum", "argmax"}


def _squeezed(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(d for d in shape if d != 1)


def squeeze(stage: Stage) -> None:
    if all(n.rank <= 4 for n in stage.nodes) and all(p.rank <= 4 for p in stage.params):
        return
    logical: dict[int, TensorType] = {}  # physical stand-in -> the >4-D type it represents
    # A parameter above 4-D (a camera history [B, T, cams, C, H, W]) is its own stand-in: it
    # arrives holding its non-unit dims, which is how the emitter lays out its driver value.
    for p in stage.params:
        if p.rank > 4 and len(_squeezed(p.shape)) <= 4:
            logical[id(p)] = p.type
            p.type = TensorType(_squeezed(p.shape), p.type.dtype)

    def stand_in(b: Builder, x: Value, t: TensorType) -> Value:
        """A fresh node for x carrying logical type t (fresh, so x's own entry is untouched)."""
        y = b.reshape(x, shape=list(_squeezed(t.shape)))
        logical[id(y)] = t
        return y

    def settle(b: Builder, y: Value, t: TensorType) -> Value:
        """Physical y holds the non-unit dims of t: return t's representation."""
        if t.rank > 4:
            return stand_in(b, y, t)
        if id(y) in logical:  # a parameter stand-in may already have t's shape: never return it as is
            return b.reshape(y, shape=list(t.shape))
        return b.reshape_to(y, t.shape)

    def fix(b: Builder, n: Node) -> Value | None:
        ins = n.inputs
        squeezed_in = [id(v) in logical for v in ins]
        if n.rank <= 4 and not any(squeezed_in):
            return None
        L = [logical.get(id(v), v.type) for v in ins]

        if n.op == "reshape":
            return settle(b, ins[0], n.type)

        if n.op == "permute" and squeezed_in[0]:
            src = L[0].shape
            nonunit = [d for d in range(len(src)) if src[d] != 1]
            pos = {d: i for i, d in enumerate(nonunit)}
            phys_perm = [pos[p] for p in n.attrs["perm"] if src[p] != 1]
            y = ins[0] if phys_perm == sorted(phys_perm) else b.permute(ins[0], perm=phys_perm)
            return settle(b, y, n.type)

        if n.op == "view" and squeezed_in[0]:
            src, d = L[0].shape, n.attrs["dim"]
            if src[d] == 1:
                return settle(b, ins[0], n.type)
            phys_dim = sum(1 for k in range(d) if src[k] != 1)
            y = b.view(ins[0], dim=phys_dim, start=n.attrs["start"], length=n.attrs["length"])
            return settle(b, b.reshape_to(y, _squeezed(y.shape)), n.type)

        if n.op == "repeat":
            # Keep the output's non-unit dims; the input has size 1 wherever the output does.
            out = n.type.shape
            src = (1,) * (len(out) - L[0].rank) + L[0].shape
            keep = [k for k in range(len(out)) if out[k] != 1]
            x = b.reshape(ins[0], shape=[src[k] for k in keep])
            return settle(b, b.repeat(x, shape=[out[k] for k in keep]), n.type)

        if n.op == "concat":
            # Keep the output's non-unit dims plus the concat dim (inputs may be 1 along it).
            out, d = n.type.shape, n.attrs["dim"]
            keep = [k for k in range(len(out)) if out[k] != 1 or k == d]
            parts = [b.reshape(v, shape=[t.shape[k] for k in keep]) for v, t in zip(ins, L)]
            return settle(b, b.concat(parts[0], parts[1], dim=keep.index(d)), n.type)

        if n.op in _ELEMENTWISE_UNARY and squeezed_in[0]:
            y = b.emit(n.op, [ins[0]], n.attrs)
            return settle(b, y, n.type)

        if n.op in _ROW and len(ins) == 1 and squeezed_in[0] and L[0].shape[-1] != 1:
            return settle(b, b.emit(n.op, [ins[0]], n.attrs), n.type)

        if n.op in _ELEMENTWISE_BINARY and all(squeezed_in) and L[0] == L[1]:
            return settle(b, b.emit(n.op, list(ins), n.attrs), n.type)

        if any(squeezed_in):
            # Hand the op its logical-rank input back; layout reports it if it stays above 4-D.
            n.inputs = [b.reshape(v, shape=list(t.shape)) if s else v for v, s, t in zip(ins, squeezed_in, L)]
        return None

    rewrite(stage, fix)
