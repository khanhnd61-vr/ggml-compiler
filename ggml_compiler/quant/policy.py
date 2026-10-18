"""Per-tensor weight dtype policy.

Rules map glob patterns on a weight's name (its FQN on the root module) to a storage dtype;
the first matching rule wins. A tied weight (one tensor under several names) matches under any of
its names. A pattern ending in `@<op>` applies only where that op reads the weight
(`embed.weight@get_rows`, `embed.weight@mul_mat`); when a weight's readers end up wanting different
types, it is stored once per type and each reader reads its copy (llama.cpp stores a tied
embedding and output head that way). A weight can be stored in anything but `float` only where every
reader can take it: as the weight operand of a matmul or the table of an embedding lookup (any
type), or as a convolution kernel (f16).
Other weights (norm scales, biases, conv kernels, ...) stay at `float`. A quantized type also
needs the row length to be a multiple of its block size; otherwise the `fallback` types are tried
in order. Every decision lands in the compile report.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ir import Node, Value, Weight
from ..ir.program import ProgramIR
from .types import block_size, canonical


@dataclass
class DTypePolicy:
    rules: list[tuple[str, str]] = field(default_factory=list)  # (pattern, dtype)
    float: str = "f32"
    fallback: list[str] = field(default_factory=lambda: ["q4_0", "q8_0", "f16"])

    def __post_init__(self) -> None:
        self.rules = [(p, canonical(d)) for p, d in self.rules]
        self.float = canonical(self.float)
        self.fallback = [canonical(d) for d in self.fallback]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DTypePolicy":
        rules = [(r["match"], r["dtype"]) for r in d.get("rules", [])]
        kw = {k: d[k] for k in ("float", "fallback") if k in d}
        return DTypePolicy(rules=rules, **kw)

    @staticmethod
    def from_yaml(path: str | Path) -> "DTypePolicy":
        import yaml

        return DTypePolicy.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def dtype_for(self, fqn: str, op: str | None = None) -> str | None:
        """The first rule matching `fqn` (or, with `op`, `fqn@op`)."""
        for pattern, dtype in self.rules:
            if "@" in pattern:
                if op is not None and fnmatch.fnmatchcase(f"{fqn}@{op}", pattern.replace("**", "*")):
                    return dtype
            elif fnmatch.fnmatchcase(fqn, pattern.replace("**", "*")):
                return dtype
        return None


def _readers(prog: ProgramIR) -> dict[int, list[tuple[Node, int]]]:
    """For each weight, the ops that read it. A reshape that keeps the row length (a 1-D conv
    kernel viewed as 2-D) is looked through: ggml reshapes any storage type that way for free."""
    out: dict[int, list[tuple[Node, int]]] = {}
    for stage in prog.stages.values():
        users = stage.users()

        def add(w: Weight, v: Value) -> None:
            for n in users.get(id(v), []):
                for i, x in enumerate(n.inputs):
                    if x is not v:
                        continue
                    if n.op == "reshape" and n.shape[-1:] == v.shape[-1:]:
                        add(w, n)
                    else:
                        out.setdefault(id(w), []).append((n, i))

        for w in {id(x): x for n in stage.nodes for x in n.inputs if isinstance(x, Weight)}.values():
            add(w, w)
    return out


def _allowed(n: Node, i: int, dtype: str) -> bool:
    """Can reader n take a weight of this storage dtype as input i?"""
    if (n.op == "mul_mat" or n.op == "get_rows") and i == 0:
        return True
    return n.op == "conv_2d" and i == 0 and dtype in ("f32", "f16")  # kernels: f16 only


def _wanted(policy: DTypePolicy, names: list[str], op: str | None) -> str | None:
    for name in names:
        d = policy.dtype_for(name, op)
        if d is not None:
            return d
    return None


def _split_by_reader(prog: ProgramIR, policy: DTypePolicy) -> None:
    """Give a weight one copy per type its readers' `@op` rules want."""
    aliases: dict[int, list[str]] = {}
    for name, w in prog.weights.items():
        aliases.setdefault(id(w), []).append(name)
    if not any("@" in pat for pat, _ in policy.rules):
        return
    readers = _readers(prog)
    for w in list({id(x): x for s in prog.stages.values() for x in s.leaves() if isinstance(x, Weight)}.values()):
        names = [w.fqn] + [a for a in aliases.get(id(w), []) if a != w.fqn]
        uses = readers.get(id(w), [])
        by_type: dict[str | None, list[tuple[Node, int]]] = {}
        for n, i in uses:
            by_type.setdefault(_wanted(policy, names, n.op), []).append((n, i))
        if len(by_type) < 2 or any(n.inputs[i] is not w for n, i in uses):
            continue  # one type, or a reader behind a reshape (kept whole)
        for k, (dtype, group) in enumerate(by_type.items()):
            if k == 0:
                continue
            ops = sorted({n.op for n, _ in group})
            copy = Weight(w.data, f"{w.fqn}@{'+'.join(ops)}")
            copy.wanted = dtype
            prog.weights[copy.fqn] = copy
            for n, i in group:
                n.inputs[i] = copy


def apply_policy(prog: ProgramIR, policy: DTypePolicy) -> dict[str, dict[str, Any]]:
    """Set Weight.storage for every weight the program reads; returns {fqn: decision}."""
    _split_by_reader(prog, policy)
    readers = _readers(prog)
    aliases: dict[int, list[str]] = {}
    for name, w in prog.weights.items():
        aliases.setdefault(id(w), []).append(name)
    weights = {id(w): w for s in prog.stages.values() for w in s.leaves() if isinstance(w, Weight)}
    decisions = {}
    for w in weights.values():
        names = [w.fqn] + [a for a in aliases.get(id(w), []) if a != w.fqn]
        ops = sorted({n.op for n, _ in readers.get(id(w), [])})
        per_op = {_wanted(policy, names, op) for op in ops} if ops else {_wanted(policy, names, None)}
        # Readers that agree (an `@op` rule for each, all one type) give the weight that type.
        wanted = getattr(w, "wanted", None) or (per_op.pop() if len(per_op) == 1 else _wanted(policy, names, None))
        storage, note = policy.float, "float"
        if wanted is not None and wanted != policy.float:
            uses = readers.get(id(w), [])
            if not any(all(_allowed(n, i, c) for n, i in uses) for c in [wanted] + policy.fallback):
                note = f"{wanted} not allowed: read by an op that needs float"
            else:
                for cand in [wanted] + [f for f in policy.fallback if f != wanted]:
                    if w.shape[-1] % block_size(cand) == 0 and all(_allowed(n, i, cand) for n, i in uses):
                        storage = cand
                        note = "rule" if cand == wanted else f"fallback from {wanted} (row {w.shape[-1]})"
                        break
                else:
                    note = f"no type in {[wanted] + policy.fallback} fits row {w.shape[-1]}"
        w.storage = storage
        decisions[w.fqn] = {"shape": list(w.shape), "dtype": storage, "note": note}
    return decisions
