"""Drop nodes nothing reads."""

from __future__ import annotations

from ..ir import Node, Stage


def dce(stage: Stage) -> None:
    live: set[int] = {id(v) for v in stage.outputs}
    keep: list[Node] = []
    for n in reversed(stage.nodes):
        if id(n) in live:
            keep.append(n)
            live.update(id(v) for v in n.inputs)
    stage.nodes = keep[::-1]
