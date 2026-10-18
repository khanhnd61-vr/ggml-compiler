"""User-facing description of what to compile: stages, a driver, example inputs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch


@dataclass
class Stage:
    """A function (or module) to compile into one ggml graph, with example inputs that fix every
    shape the stage does not declare dynamic. The function may read parameters of the program's
    root module.

    dynamic_shapes: one entry per example input, as torch.export takes them: None (static), or
    {dim: torch.export.Dim} for a tensor, nested like the input for a pytree. A Dim's name is
    the symbol's name across the whole program; every Dim needs a max. Derived dims (256 * Dim)
    work too.

    matmul_acc: "f32" asks every matmul of the stage to accumulate in f32 (ggml_prec_set_acc),
    where a backend would otherwise accumulate 16-bit weights' products in f16 (CUDA's cuBLAS
    path). CPU kernels accumulate in f32 anyway."""

    fn: Callable[..., Any]
    example_inputs: tuple[Any, ...]
    dynamic_shapes: Any = None
    matmul_acc: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_inputs, tuple):
            self.example_inputs = tuple(self.example_inputs)


@dataclass
class Call:
    stage: str
    args: list[str]
    out: list[str]


def call(stage: str, args: Sequence[str], out: Sequence[str]) -> Call:
    """Run one stage. `args` and `out` name driver values; a value may be a whole pytree."""
    return Call(stage, list(args), list(out))


@dataclass
class Loop:
    n: int | str
    body: list[Call]
    carry: dict[str, str] = field(default_factory=dict)
    scalars: dict[str, list[float]] = field(default_factory=dict)
    slices: dict[str, str] = field(default_factory=dict)
    counters: dict[str, str] = field(default_factory=dict)
    bind: dict[str, Callable[..., Any]] = field(default_factory=dict)
    masks: dict[str, str] = field(default_factory=dict)
    until: dict[str, list[int]] = field(default_factory=dict)
    collect: dict[str, str] = field(default_factory=dict)


def loop(n: int | str, body: Sequence[Call], carry: dict[str, str] | None = None,
         scalars: dict[str, list[float]] | None = None, slices: dict[str, str] | None = None, *,
         counters: dict[str, str] | None = None, bind: dict[str, Callable[..., Any]] | None = None,
         masks: dict[str, str] | None = None, until: dict[str, Sequence[int]] | None = None,
         collect: dict[str, str] | None = None) -> Loop:
    """Run `body` n times (n: an int, or the name of a symbol bound at run time).

    carry:    loop value -> the value it starts from; the body's outputs rebind it.
    scalars:  loop value -> one number per iteration (a table compiled into the artifact).
    slices:   loop value -> a driver value with a leading dim of n; iteration i sees row i
              (per-step noise, per-step conditioning). Rows are copied on the device.
    counters: loop value -> an i32 [1] driver value `start`; iteration i sees start + i (a token
              position while generating).
    bind:     symbol -> a function of the counters (sympy expressions) giving the symbol's value
              before each iteration (the KV length a position needs).
    masks:    loop value -> counter: an additive causal mask row, 0 up to the counter's value and
              -inf after it, in the value's shape for the current binding.
    until:    loop value -> ids: stop after an iteration whose value (one element) is one of them.
    collect:  driver value -> loop value: row i of the driver value is the loop value after
              iteration i (generated tokens). Rows past an early stop stay zero."""
    return Loop(n, list(body), dict(carry or {}), dict(scalars or {}), dict(slices or {}), dict(counters or {}),
                dict(bind or {}), dict(masks or {}), {k: [int(x) for x in v] for k, v in (until or {}).items()},
                dict(collect or {}))


@dataclass
class Driver:
    inputs: list[str]
    body: list[Call | Loop]
    outputs: list[str]
    # Example values for driver inputs. Needed for an input no stage reads directly (a loop's
    # slice source); otherwise inputs take the example of the stage argument they feed.
    examples: dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    """Several stages sharing the parameters of `root`, run by a driver.

    driver: one Driver, or several entry points by name ({"prefill": ..., "decode": ...}), each
    run on its own (`Session.run(..., entry="decode")`). The first is the default.
    state: values that persist across runs, by name, with an example (a KV cache). Drivers read
    and write them like any value; they start as zeros and `Session.reset()` zeroes them again.
    Not inputs or outputs. The compile-time reference run calls the entries once each, in order,
    with the state carried from one to the next."""

    root: torch.nn.Module | None
    stages: dict[str, Stage]
    driver: Driver | dict[str, Driver]
    # What the program fixed that a more general one would take at run time, as name -> what was
    # compiled in (e.g. {"empty_cameras": "dropped"}). Recorded in report.json and the GGUF.
    specialized: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    # Anything a caller needs to drive the artifact (an LLM's EOS ids, its max context), stored as
    # JSON in the GGUF key ggmlc.meta and read back with ggmlc_metadata / Session.metadata.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Symbols no stage input carries (a loop's trip count), as name -> (min, max, example).
    symbols: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def entries(self) -> dict[str, Driver]:
        return dict(self.driver) if isinstance(self.driver, dict) else {"main": self.driver}

    @staticmethod
    def single(
        module: torch.nn.Module,
        example_inputs: Sequence[Any],
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
    ) -> "Program":
        """One stage running module.forward."""
        example_inputs = tuple(example_inputs)
        if input_names is None:
            input_names = []
            for p in inspect.signature(module.forward).parameters.values():
                if p.kind is inspect.Parameter.VAR_POSITIONAL:
                    input_names += [f"{p.name}.{i}" for i in range(len(example_inputs) - len(input_names))]
                    break
                input_names.append(p.name)
            input_names = input_names[: len(example_inputs)]
        input_names = list(input_names)
        if len(input_names) != len(example_inputs):
            raise ValueError(f"{len(input_names)} input names for {len(example_inputs)} example inputs")
        outputs = list(output_names) if output_names else ["out"]
        return Program(
            root=module,
            stages={"forward": Stage(module, example_inputs)},
            driver=Driver(inputs=input_names, body=[call("forward", input_names, outputs)], outputs=outputs),
        )
