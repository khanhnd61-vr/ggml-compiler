"""Builder: appends type-checked nodes to a stage.

Every registered op is available as a method: `b.mul_mat(w, x)`, `b.soft_max(x, scale=0.5)`.
Positional arguments are input values, keyword arguments are op attributes.
"""

from __future__ import annotations

from typing import Any

import torch

from . import ops
from .graph import Const, Node, Stage, TensorType, Value


# Ops that may read i32 values as they are. Every other op gets its i32 inputs cast to f32
# (integers below 2**24 are exact in f32); get_rows gets float indices cast to i32 and an i32
# table cast to f32.
_I32_OK = {"get_rows", "set_rows", "rope", "cast", "reshape", "view", "permute", "cont"}


class Builder:
    def __init__(self, stage: Stage, loc: str = "") -> None:
        self.stage = stage
        self.loc = loc  # attached to every node built until changed
        self.scope = ""  # likewise: the PyTorch module path
        self.out_type: TensorType | None = None  # set by the frontend while a lowering runs
        # ids of values that were torch.bool before ggml's f32 0/1 encoding; the frontend fills it
        # (a mask argument means something else as a bool than as an additive float).
        self.bools: set[int] = set()
        # id(source tensor) -> (source tensor, Const). Holding the source keeps its id unique.
        self._consts: dict[int, tuple[torch.Tensor, Const]] = {}

    def emit(self, op: str, inputs: list[Value], attrs: dict[str, Any] | None = None) -> Node:
        d = ops.get(op)
        attrs = dict(attrs or {})
        lo, hi = d.arity
        if not lo <= len(inputs) <= hi:
            raise ops.ShapeError(f"{op}: expected {lo}..{hi} inputs, got {len(inputs)}")
        for v in inputs:
            if not isinstance(v, Value):
                raise TypeError(f"{op}: input {v!r} is not an IR value; wrap tensors with b.const()")
        if op not in _I32_OK:
            inputs = [self.emit("cast", [v], {"dtype": "f32"}) if v.type.dtype == "i32" else v for v in inputs]
        elif op == "get_rows":  # i32 indices; an i32 table is read as f32 (ggml would return i32 rows)
            table, idx = inputs
            table = self.emit("cast", [table], {"dtype": "f32"}) if table.type.dtype == "i32" else table
            idx = idx if idx.type.dtype == "i32" else self.emit("cast", [idx], {"dtype": "i32"})
            inputs = [table, idx]
        elif op == "set_rows":
            dst, src, idx = inputs
            src = self.emit("cast", [src], {"dtype": "f32"}) if src.type.dtype == "i32" else src
            idx = idx if idx.type.dtype == "i32" else self.emit("cast", [idx], {"dtype": "i32"})
            inputs = [dst, src, idx]
        try:
            t = d.infer([v.type for v in inputs], attrs)
        except ops.ShapeError as e:
            where = f" at {self.loc}" if self.loc else ""
            raise ops.ShapeError(f"{e}{where}") from None
        n = Node(op, list(inputs), attrs, t, self.loc, self.scope)
        self.stage.nodes.append(n)
        return n

    def __getattr__(self, op: str):
        if op.startswith("_") or op not in ops.OPS:
            raise AttributeError(op)
        return lambda *inputs, **attrs: self.emit(op, list(inputs), attrs)

    # --- conveniences ----------------------------------------------------------------------

    def const(self, data: torch.Tensor | float) -> Const:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(float(data))
        hit = self._consts.get(id(data))
        if hit is not None:
            return hit[1]
        t = data.detach()
        if t.dtype != torch.int32:  # floats, bools and other integers all become f32
            t = t.to(torch.float32)
        c = Const(t.contiguous())
        self._consts[id(data)] = (data, c)
        return c

    def add_scalar(self, x: Value, c: float) -> Value:
        return self.emit("scale", [x], {"s": 1.0, "b": float(c)})

    def mul_scalar(self, x: Value, c: float) -> Value:
        return self.emit("scale", [x], {"s": float(c)})

    def transpose(self, x: Value, d0: int, d1: int) -> Value:
        perm = list(range(x.rank))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return self.emit("permute", [x], {"perm": perm})

    def reshape_to(self, x: Value, shape: tuple[int, ...]) -> Value:
        if tuple(shape) == x.shape:
            return x
        return self.emit("reshape", [x], {"shape": list(shape)})
