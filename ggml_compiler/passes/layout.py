"""Fit the IR to ggml's layout rules.

- ggml tensors have at most 4 dims (passes.squeeze runs first and removes what it can).
- `permute` and `view` produce strided views. Most ggml ops, and `reshape` always, need
  contiguous inputs, so a `cont` goes before any other op reading a strided view. A view is
  still contiguous, as ggml_is_contiguous sees it, when it only moves or slices around size-1
  dims: a permute that keeps the non-unit dims in order, or a slice of the outermost non-unit
  dim of a contiguous tensor. Those get no copy.
- With `strided_rows`, a strided view whose rows (innermost dim) stay contiguous needs no copy
  when it feeds mul_mat, flash_attn or set_rows, which take arbitrary strides above dim 0 (a KV
  cache read through a view, a permuted query). A mul_mat operand must also not be transposed.
  A reshape that only adds or drops size-1 dims keeps such a view a view (ggml_compiler::ops::reshape).
"""

from __future__ import annotations

from ..ir import Builder, Node, Stage, Value
from ..ir.program import CompileError
from . import rewrite

ACCEPTS_VIEWS = {"permute", "view", "cont", "cast"}  # cast is a ggml_cpy, which reads strides


def _unit_reshape(n: Node) -> bool:
    """A reshape that only adds or drops size-1 dims: a strided input stays a view."""
    return n.op == "reshape" and [d for d in n.shape if d != 1] == [d for d in n.inputs[0].shape if d != 1]


def _contiguous(n: Node, input_contiguous: bool) -> bool:
    if _unit_reshape(n):
        return input_contiguous
    if n.op == "permute":
        src = n.inputs[0].shape
        order = [p for p in n.attrs["perm"] if src[p] != 1]
        return input_contiguous and order == sorted(order)
    if n.op == "view":
        src = n.inputs[0].shape
        return input_contiguous and all(d == 1 for d in src[: n.attrs["dim"]])
    return True


ROWS_STRIDED_OK = {"mul_mat": (0, 1), "flash_attn": (0, 1, 2), "set_rows": (1,), "rope": (0,)}


def _rows_contiguous(n: Node, input_rows: bool) -> bool:
    """Whether n's innermost non-unit dim still has unit stride, given whether its input's has."""
    if n.op == "view":
        src = n.inputs[0].shape
        nonunit = [i for i, d in enumerate(src) if d != 1]
        # Slicing the innermost dim down to one element leaves the next dim innermost, strided.
        return input_rows and not (nonunit and n.attrs["dim"] == nonunit[-1] and n.attrs["length"] == 1)
    if n.op == "permute":
        src = n.inputs[0].shape
        perm = list(n.attrs["perm"])
        last = [p for p in perm if src[p] != 1]
        return input_rows and (not last or last[-1] == max(p for p in range(len(src)) if src[p] != 1))
    return input_rows


def layout(stage: Stage, strided_rows: bool = False) -> None:
    too_big = [n for n in stage.nodes if n.rank > 4] + [p for p in stage.params if p.rank > 4]
    if too_big:
        lines = [f"stage {stage.name!r}: ggml tensors have at most 4 dims"]
        for v in too_big:
            where = getattr(v, "loc", "") or getattr(v, "name", "")
            lines.append(f"  {getattr(v, 'op', 'param')} -> {v.type}  {where}")
        raise CompileError("\n".join(lines))

    strided: set[int] = set()
    rows: set[int] = set()  # strided values whose innermost dim still has unit stride
    conts: dict[int, Value] = {}  # one cont per strided value

    def fix(b: Builder, n: Node) -> None:
        if n.op not in ACCEPTS_VIEWS:
            ok = ROWS_STRIDED_OK.get(n.op, (0,) if _unit_reshape(n) else ()) if strided_rows else ()
            for i, v in enumerate(n.inputs):
                if id(v) in strided and not (i in ok and id(v) in rows):
                    if id(v) not in conts:
                        conts[id(v)] = b.cont(v)  # lands just before n
                    n.inputs[i] = conts[id(v)]
        src = n.inputs[0] if n.inputs else None
        if not _contiguous(n, src is None or id(src) not in strided):
            strided.add(id(n))
            if n.op in ("permute", "view", "reshape") and \
                    _rows_contiguous(n, src is None or id(src) not in strided or id(src) in rows):
                rows.add(id(n))
        return None

    rewrite(stage, fix)
