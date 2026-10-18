"""Store f32 state in f16 where every reader can take it (flag f16_kv_cache, O2).

A KV cache is state written with set_rows and read through views by attention. When every read
ends in a cast to f16 (flash attention's K and V) or a matmul operand, the state can live in f16:
set_rows converts the new rows as it writes them, the casts disappear, and the cache takes half
the memory and bandwidth, as llama.cpp's default f16 cache does.
"""

from __future__ import annotations

from ..ir import Node, Param, Stage, TensorType, Value
from ..ir.program import ProgramIR

_VIEWS = {"view", "reshape", "permute"}


def _plan(stage: Stage, p: Param, outputs: set[int]) -> tuple[list[Node], list[Node]] | None:
    """(nodes to retype, casts to drop) if p and what derives from it can be f16, else None."""
    users = stage.users()
    retype: list[Node] = []
    casts: list[Node] = []
    todo: list[Value] = [p]
    while todo:
        v = todo.pop()
        for n in users.get(id(v), []):
            i = n.inputs.index(v)
            if n.op == "set_rows" and i == 0 or n.op in _VIEWS:
                retype.append(n)
                todo.append(n)
            elif n.op == "cast" and n.attrs["dtype"] == "f16":
                casts.append(n)
            elif n.op == "mul_mat" and i == 0:
                continue
            else:
                return None
        if id(v) in outputs and not (v is p or (isinstance(v, Node) and v.op == "set_rows")):
            return None
    return retype, casts


def f16_kv_cache(prog: ProgramIR) -> list[str]:
    """Retype eligible f32 state to f16; returns the names retyped."""
    done = []
    for name in prog.state:
        if prog.values[name].dtype != "f32":
            continue
        sites = []
        for c in prog.calls():
            st = prog.stages[c.stage]
            if name in c.args:
                p = st.params[c.args.index(name)]
                outs = {id(v) for v, o in zip(st.outputs, c.out) if o == name}
                if any(o == name for o in c.out) and not outs:
                    sites = None
                    break
                plan = _plan(st, p, outs)
                if plan is None:
                    sites = None
                    break
                sites.append((st, p, plan))
            elif name in c.out:
                sites = None  # written from scratch somewhere: keep it simple
                break
        if not sites:
            continue
        for st, p, (retype, casts) in sites:
            p.type = TensorType(p.shape, "f16")
            for n in retype:
                n.type = TensorType(n.shape, "f16")
            for c in casts:
                st.replace_uses(c, c.inputs[0])
            st.nodes = [n for n in st.nodes if not any(n is c for c in casts)]
        prog.values[name] = TensorType(prog.values[name].shape, "f16")
        done.append(name)
    return done
