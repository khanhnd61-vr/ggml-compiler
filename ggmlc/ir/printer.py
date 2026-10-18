"""Text form of a stage, for debugging and `ggmlc inspect`."""

from __future__ import annotations

from .graph import Const, Stage, Value, Weight


def format_stage(stage: Stage) -> str:
    names: dict[int, str] = {}
    for p in stage.params:
        names[id(p)] = f"%{p.name}"
    n_const = 0

    def name(v: Value) -> str:
        nonlocal n_const
        if id(v) not in names:
            if isinstance(v, Weight):
                names[id(v)] = f"@{v.fqn}"
            elif isinstance(v, Const):
                names[id(v)] = f"$c{n_const}"
                n_const += 1
            else:
                names[id(v)] = "%?"
        return names[id(v)]

    lines = [f"stage {stage.name}({', '.join(f'%{p.name}: {p.type}' for p in stage.params)}) {{"]
    for i, n in enumerate(stage.nodes):
        names[id(n)] = f"%{i}"
        args = ", ".join(name(v) for v in n.inputs)
        attrs = ", ".join(f"{k}={v}" for k, v in n.attrs.items())
        attrs = f" {{{attrs}}}" if attrs else ""
        loc = f"  # {n.loc}" if n.loc else ""
        lines.append(f"  %{i}: {n.type} = {n.op}({args}){attrs}{loc}")
    lines.append(f"  return {', '.join(name(v) for v in stage.outputs)}")
    lines.append("}")
    return "\n".join(lines)

