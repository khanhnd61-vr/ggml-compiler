"""Run a stage with the ops' PyTorch references. Used to test lowering and passes without ggml."""

from __future__ import annotations

from typing import Sequence

import torch

from . import ops, sym
from .graph import Const, Param, Stage, Weight


def bind(params: Sequence[Param], shapes: Sequence[tuple[int, ...]]) -> dict[str, int]:
    """Symbol values that make each parameter's shape equal the given one."""
    return sym.solve((p.shape, shape) for p, shape in zip(params, shapes))


def run_stage(stage: Stage, args: list[torch.Tensor]) -> list[torch.Tensor]:
    if len(args) != len(stage.params):
        raise ValueError(f"stage {stage.name} takes {len(stage.params)} arguments, got {len(args)}")
    shapes = []
    for p, a in zip(stage.params, args):
        if a.dim() > 4 and a.dim() != p.rank:
            # After the squeeze pass a >4-D parameter holds its non-unit dims.
            a = a.reshape(tuple(d for d in a.shape if d != 1))
        shapes.append(tuple(a.shape))
    binding = bind(stage.params, shapes)
    env: dict[int, torch.Tensor] = {}
    for p, a, shape in zip(stage.params, args, shapes):
        if shape != sym.ev(p.shape, binding):
            raise ValueError(f"argument {p.name} has shape {tuple(a.shape)}, stage expects {p.shape}")
        env[id(p)] = a.reshape(shape)

    def get(v) -> torch.Tensor:
        if isinstance(v, (Weight, Const)):
            return v.data
        return env[id(v)]

    with torch.no_grad():
        for n in stage.nodes:
            attrs = sym.ev(n.attrs, binding) if binding else n.attrs
            env[id(n)] = ops.get(n.op).ref([get(v) for v in n.inputs], attrs)
    return [get(v) for v in stage.outputs]
