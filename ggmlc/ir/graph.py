"""Core IR: typed values, single-output nodes, stages and programs.

Shapes are in PyTorch order (outermost dimension first). Only the emitters reverse them into
ggml's innermost-first `ne`. A dim is an int or a symbolic expression (`ir.sym`). Every node
produces exactly one value, so a Node *is* its value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from . import sym

DTYPES: dict[str, torch.dtype] = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
}


def dtype_of(t: torch.Tensor) -> str:
    for name, dt in DTYPES.items():
        if t.dtype == dt:
            return name
    raise TypeError(f"tensor dtype {t.dtype} has no ggmlc equivalent")


@dataclass(frozen=True)
class TensorType:
    shape: tuple[sym.Dim, ...]
    dtype: str = "f32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(sym.norm(d) for d in self.shape))
        if self.dtype not in DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_static(self) -> bool:
        return sym.static(self.shape)

    def numel(self) -> sym.Dim:
        n = 1
        for d in self.shape:
            n *= d
        return sym.norm(n)

    def __str__(self) -> str:
        return f"{self.dtype}[{', '.join(map(str, self.shape))}]"


class Value:
    """Anything a node can read: a stage parameter, a weight, a constant, or a node."""

    type: TensorType

    def __init__(self, type: TensorType) -> None:
        self.type = type

    @property
    def shape(self) -> tuple[sym.Dim, ...]:
        return self.type.shape

    @property
    def rank(self) -> int:
        return self.type.rank


class Param(Value):
    """Positional input of a stage."""

    def __init__(self, type: TensorType, name: str) -> None:
        super().__init__(type)
        self.name = name

    def __repr__(self) -> str:
        return f"Param({self.name}: {self.type})"


class Weight(Value):
    """A model parameter or buffer, shared by every stage of a program. `fqn` is its name on the
    root module."""

    def __init__(self, data: torch.Tensor, fqn: str) -> None:
        super().__init__(TensorType(tuple(data.shape), dtype_of(data)))
        self.data = data
        self.fqn = fqn
        self.storage = "f32"  # how the GGUF stores it; set by the dtype policy

    def __repr__(self) -> str:
        return f"Weight({self.fqn}: {self.type})"


class Const(Value):
    """A tensor computed at compile time (folded subgraphs, tables, masks)."""

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(TensorType(tuple(data.shape), dtype_of(data)))
        self.data = data

    def __repr__(self) -> str:
        return f"Const({self.type})"


class Node(Value):
    def __init__(self, op: str, inputs: list[Value], attrs: dict[str, Any], type: TensorType, loc: str = "",
                 scope: str = "") -> None:
        super().__init__(type)
        self.op = op
        self.inputs = inputs
        self.attrs = attrs
        self.loc = loc  # PyTorch source line
        self.scope = scope  # PyTorch module path, e.g. "backbone.layer1.0.conv1"

    def __repr__(self) -> str:
        return f"Node({self.op}: {self.type})"


@dataclass
class Stage:
    """One ggml graph: parameters in, outputs out, nodes in topological order."""

    name: str
    params: list[Param] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)

    def add_param(self, type: TensorType, name: str | None = None) -> Param:
        p = Param(type, name or f"a.{len(self.params)}")
        self.params.append(p)
        return p

    def users(self) -> dict[int, list[Node]]:
        """Map id(value) to the nodes that read it."""
        out: dict[int, list[Node]] = {}
        for n in self.nodes:
            for v in n.inputs:
                out.setdefault(id(v), []).append(n)
        return out

    def replace_uses(self, old: Value, new: Value) -> None:
        for n in self.nodes:
            n.inputs = [new if v is old else v for v in n.inputs]
        self.outputs = [new if v is old else v for v in self.outputs]

    def leaves(self) -> Iterable[Value]:
        """Weights and constants the stage reads, each once, in first-use order."""
        seen: set[int] = set()
        for n in self.nodes:
            for v in n.inputs:
                if isinstance(v, (Weight, Const)) and id(v) not in seen:
                    seen.add(id(v))
                    yield v
        for v in self.outputs:
            if isinstance(v, (Weight, Const)) and id(v) not in seen:
                seen.add(id(v))
                yield v

    def __str__(self) -> str:
        from .printer import format_stage

        return format_stage(self)
