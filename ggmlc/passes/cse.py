"""Common subexpression elimination (flag cse, O1): a node with the same op, the same inputs and
the same attributes as an earlier one is that node. Masks and their casts, rebuilt by every layer
of a transformer from the same input, are computed once."""

from __future__ import annotations

from typing import Any

from ..ir import Builder, Node, Stage, Value
from . import rewrite


def _key(x: Any) -> Any:
    if isinstance(x, dict):
        return tuple(sorted((k, _key(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_key(v) for v in x)
    return x


def cse(stage: Stage) -> None:
    seen: dict[tuple, Node] = {}

    def fix(b: Builder, n: Node) -> Value | None:
        try:
            key = (n.op, tuple(id(v) for v in n.inputs), _key(n.attrs), n.type)
            hash(key)
        except TypeError:
            return None
        if key in seen:
            return seen[key]
        seen[key] = n
        return None

    rewrite(stage, fix)
