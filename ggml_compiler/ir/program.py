"""A compiled program: stages, the weights they share, and a driver over named tensors.

At this level every driver value is a single tensor (pytrees are flattened to dotted names
like `kv.3.k`), so the driver maps directly onto the runtime's value buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import sym
from .graph import Stage, TensorType, Weight


@dataclass
class CallOp:
    stage: str
    args: list[str]  # driver value names, one per stage param
    out: list[str]  # driver value names, one per stage output


@dataclass
class LoopOp:
    n: int | str  # a trip count, or the symbol that holds it
    body: list[CallOp]
    carry: list[tuple[str, str]] = field(default_factory=list)  # (loop value, initial value)
    scalars: list[tuple[str, list[float]]] = field(default_factory=list)  # (value, one entry per iteration)
    slices: list[tuple[str, str]] = field(default_factory=list)  # (loop value, source with a leading dim of n)
    counters: list[tuple[str, str]] = field(default_factory=list)  # (loop value, i32 start value): start + i
    bind: list[tuple[str, Any]] = field(default_factory=list)  # (symbol, dim expression over counter names)
    masks: list[tuple[str, str]] = field(default_factory=list)  # (loop value, counter): causal mask row
    until: list[tuple[str, list[int]]] = field(default_factory=list)  # (loop value, stop ids)
    collect: list[tuple[str, str]] = field(default_factory=list)  # (driver value, loop value): row i


@dataclass
class Entry:
    """One way to run the program: its inputs, outputs and driver ops."""

    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    driver: list[CallOp | LoopOp] = field(default_factory=list)


@dataclass
class ProgramIR:
    stages: dict[str, Stage] = field(default_factory=dict)
    weights: dict[str, Weight] = field(default_factory=dict)  # by root FQN
    values: dict[str, TensorType] = field(default_factory=dict)  # every driver value
    entries: dict[str, Entry] = field(default_factory=lambda: {"main": Entry()})  # the first is the default
    state: list[str] = field(default_factory=list)  # values that persist across runs
    symbols: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (min, max)

    # The default entry's inputs, outputs and driver.
    @property
    def main(self) -> Entry:
        return next(iter(self.entries.values()))

    @property
    def inputs(self) -> list[str]:
        return self.main.inputs

    @inputs.setter
    def inputs(self, v: list[str]) -> None:
        self.main.inputs = v

    @property
    def outputs(self) -> list[str]:
        return self.main.outputs

    @outputs.setter
    def outputs(self, v: list[str]) -> None:
        self.main.outputs = v

    @property
    def driver(self) -> list[CallOp | LoopOp]:
        return self.main.driver

    @driver.setter
    def driver(self, v: list[CallOp | LoopOp]) -> None:
        self.main.driver = v

    def all_inputs(self) -> list[str]:
        """Every entry's inputs, each once, in entry order."""
        return list(dict.fromkeys(n for e in self.entries.values() for n in e.inputs))

    def calls(self) -> list[CallOp]:
        """Every call site of every entry, loops flattened."""
        out: list[CallOp] = []
        for e in self.entries.values():
            for op in e.driver:
                out.extend(op.body if isinstance(op, LoopOp) else [op])
        return out

    @staticmethod
    def from_stage(stage: Stage) -> "ProgramIR":
        """A program that calls one stage: inputs in.0.., outputs out.0.."""
        prog = ProgramIR(stages={stage.name: stage})
        prog.inputs = [f"in.{i}" for i in range(len(stage.params))]
        prog.outputs = [f"out.{i}" for i in range(len(stage.outputs))]
        prog.values.update(zip(prog.inputs, (p.type for p in stage.params)))
        prog.values.update(zip(prog.outputs, (v.type for v in stage.outputs)))
        prog.driver = [CallOp(stage.name, list(prog.inputs), list(prog.outputs))]
        return prog

    def check(self) -> None:
        """Driver and stage signatures agree."""
        io = {n for e in self.entries.values() for n in e.inputs + e.outputs}
        if io & set(self.state):
            raise ValueError(f"state values {sorted(io & set(self.state))} cannot be entry inputs or outputs")
        for op in (op for e in self.entries.values() for op in e.driver):
            if isinstance(op, LoopOp):
                for dst, src in op.carry:
                    if self.values[dst] != self.values[src]:
                        raise ValueError(f"loop carry {dst} is {self.values[dst]}, its initial value {src} is "
                                         f"{self.values[src]}")
                n = op.n if isinstance(op.n, int) else sym.symbol(op.n)
                for dst, src in op.collect:
                    d, v = self.values[dst], self.values[src]
                    if d.shape != (sym.norm(n), *v.shape) or d.dtype != v.dtype:
                        raise ValueError(f"loop collects {src} ({v}) into {dst}, which must be ({op.n}, ...) of it, "
                                         f"it is {d}")
                for dst, src in op.slices:
                    d, s = self.values[dst], self.values[src]
                    if s.shape != (op.n, *d.shape) or s.dtype != d.dtype:
                        raise ValueError(f"loop slice {dst} is {d}; its source {src} must be ({op.n}, ...) of it, "
                                         f"it is {s}")
        for c in self.calls():
            s = self.stages[c.stage]
            if len(c.args) != len(s.params) or len(c.out) != len(s.outputs):
                raise ValueError(f"call to {c.stage}: {len(c.args)} args / {len(c.out)} results for a stage with "
                                 f"{len(s.params)} params / {len(s.outputs)} outputs")
            for name, p in zip(c.args, s.params):
                if not _same_layout(self.values[name], p.type):
                    raise ValueError(f"call to {c.stage}: value {name} is {self.values[name]}, param {p.name} is {p.type}")
            for name, v in zip(c.out, s.outputs):
                if self.values[name].numel() != v.type.numel() or self.values[name].dtype != v.type.dtype:
                    raise ValueError(f"call to {c.stage}: value {name} is {self.values[name]}, result is {v.type}")


def _same_layout(value: TensorType, param: TensorType) -> bool:
    """A driver value fits a stage parameter. A value above 4-D is stored without its size-1 dims
    and the squeeze pass gives its parameter that squeezed type."""
    if value == param:
        return True
    squeezed = tuple(d for d in value.shape if d != 1)
    return value.rank > 4 and value.dtype == param.dtype and squeezed == param.shape


class CompileError(Exception):
    """Compilation stopped. The message names every offending node and its source location."""
