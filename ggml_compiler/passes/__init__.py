"""IR passes. `run_pipeline` takes a freshly exported program to emitter-ready tier-1 IR."""

from __future__ import annotations

from typing import Callable

from ..ir import Builder, Node, Stage, Value
from ..ir.program import ProgramIR
from ..options import Options, resolve


def rewrite(stage: Stage, fn: Callable[[Builder, Node], Value | None]) -> None:
    """Rebuild `stage`, letting `fn` replace nodes. `fn` sees each node with its inputs already
    remapped and returns a replacement value (built with the builder, or an existing value),
    or None to keep the node."""
    old, stage.nodes = stage.nodes, []
    b = Builder(stage)
    mapping: dict[int, Value] = {}
    for n in old:
        n.inputs = [mapping.get(id(v), v) for v in n.inputs]
        b.loc, b.scope = n.loc, n.scope
        r = fn(b, n)
        if r is None:
            stage.nodes.append(n)
        elif r is not n:
            mapping[id(n)] = r
    stage.outputs = [mapping.get(id(v), v) for v in stage.outputs]


def run_pipeline(prog: ProgramIR, options: Options | None = None) -> None:
    """Lower, rewrite and lay out every stage. `options` (default: O1) picks the optional
    rewrites; lowering and the rewrites ggml needs always run."""
    from .cse import cse
    from .dce import dce
    from .fold import fold, simplify
    from .fuse_attention import fuse_attention
    from .fuse_rope import fuse_rope
    from .hoist import hoist_loop_invariants
    from .layout import layout
    from .lower_tier2 import lower_tier2
    from .squeeze import squeeze
    from .state_dtype import f16_kv_cache

    from ..ir import sym

    options = options or resolve()
    sym.RANGES.clear()
    sym.RANGES.update(prog.symbols)  # the ranges every symbolic comparison below uses
    for stage in prog.stages.values():
        lower_tier2(stage, fast_gelu="fast_gelu" in options)
        simplify(stage, options)
        if "fuse_rope" in options and fuse_rope(stage):
            simplify(stage, options)
        if "flash_attention" in options:
            fuse_attention(stage, row_guard="assume_attended_rows" not in options)
            simplify(stage, options)
        if "cse" in options:
            cse(stage)
    fold(prog, options)
    if "hoist_invariants" in options:
        for stage in prog.stages.values():
            dce(stage)  # a dead reader in the loop stage would hoist its invariant inputs for nothing
        hoist_loop_invariants(prog)
    if "f16_kv_cache" in options:
        for stage in prog.stages.values():
            dce(stage)  # what attention fusion left behind would still read the cache
        f16_kv_cache(prog)
    for stage in prog.stages.values():
        dce(stage)  # dead >4-D views (a patch embedding's [N, C, T, P, P]) must not reach layout's check
        squeeze(stage)
        layout(stage, strided_rows="strided_rows" in options)
        dce(stage)
