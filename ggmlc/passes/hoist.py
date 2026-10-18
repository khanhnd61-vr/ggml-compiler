"""Loop-invariant code motion across the driver.

A stage called inside a loop often recomputes things that only depend on values the loop never
writes: SmolVLA's denoise step projects the cached prefix keys/values through the expert's
k/v projections and rebuilds the prefix attention mask on every iteration. This pass moves that
work into a new stage `<stage>.invariant`, called once right before the loop; its results become
driver values (kept on the device) that the trimmed loop stage reads as extra parameters.

Only real compute is hoisted: a chain of views over a parameter costs nothing per iteration,
and hoisting it would turn a free view into a stored copy.
"""

from __future__ import annotations

from ..ir import Node, Param, Stage
from ..ir.program import CallOp, LoopOp, ProgramIR

_FREE = {"reshape", "view", "permute"}


def hoist_loop_invariants(prog: ProgramIR) -> None:
    sites: dict[str, int] = {}
    for c in prog.calls():
        sites[c.stage] = sites.get(c.stage, 0) + 1

    for entry in prog.entries.values():
        new_driver: list[CallOp | LoopOp] = []
        for op in entry.driver:
            if isinstance(op, LoopOp):
                written = {dst for dst, _ in op.carry} | {v for v, _ in op.scalars} | {v for v, _ in op.slices} \
                    | {v for v, _ in op.counters} | {v for v, _ in op.masks}
                for c in op.body:
                    written.update(c.out)
                for c in op.body:
                    if sites[c.stage] == 1:
                        pre = _split(prog, c, written)
                        if pre is not None:
                            new_driver.append(pre)
            new_driver.append(op)
        entry.driver = new_driver


def _split(prog: ProgramIR, call: CallOp, written: set[str]) -> CallOp | None:
    stage = prog.stages[call.stage]
    invariant: set[int] = {id(p) for p, a in zip(stage.params, call.args) if a not in written}
    has_work: set[int] = set()  # invariant values with real compute somewhere behind them
    for n in stage.nodes:  # weights and constants are invariant; params and nodes as marked
        if all(id(v) in invariant or not isinstance(v, (Param, Node)) for v in n.inputs):
            invariant.add(id(n))
            if n.op not in _FREE or any(id(v) in has_work for v in n.inputs):
                has_work.add(id(n))

    if not has_work:
        return None
    # Hoisted nodes take their invariant ancestors along (free views included); if the loop
    # still reads one of those, it simply becomes a frontier value like any other.
    hoisted_ids = set(has_work)
    for n in reversed(stage.nodes):
        if id(n) in hoisted_ids:
            hoisted_ids.update(id(v) for v in n.inputs if isinstance(v, Node))
    hoisted = [n for n in stage.nodes if id(n) in hoisted_ids]
    users = stage.users()
    frontier = [n for n in hoisted
                if any(id(u) not in hoisted_ids for u in users.get(id(n), [])) or any(v is n for v in stage.outputs)]

    # The invariant stage: its own params for the invariant inputs its nodes read.
    inv = Stage(f"{stage.name}.invariant")
    remap: dict[int, Param] = {}
    inv_args: list[str] = []
    for n in hoisted:
        for i, v in enumerate(n.inputs):
            if isinstance(v, Param):
                if id(v) not in remap:
                    remap[id(v)] = inv.add_param(v.type)
                    inv_args.append(call.args[stage.params.index(v)])
                n.inputs[i] = remap[id(v)]
    inv.nodes = hoisted
    inv.outputs = list(frontier)

    # The loop stage reads the frontier values as new params.
    stage.nodes = [n for n in stage.nodes if id(n) not in hoisted_ids]
    names = []
    for k, f in enumerate(frontier):
        p = stage.add_param(f.type)
        stage.replace_uses(f, p)
        name = f"{stage.name}.invariant.{k}"
        prog.values[name] = f.type
        names.append(name)
    call.args = call.args + names
    prog.stages[inv.name] = inv
    return CallOp(inv.name, inv_args, names)

