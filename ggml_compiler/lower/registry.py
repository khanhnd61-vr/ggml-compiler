"""Registry of source-op lowerings.

A lowering receives a Builder and the op's arguments: IR values for runtime tensors,
torch.Tensor for compile-time constants, plain Python for everything else. It returns the IR
value for the op's result. `b.out_type` holds the result type PyTorch traced, and the frontend
checks the returned value against it.

    @ggml_compiler.lowering(torch.ops.aten.mish.default)
    def mish(b, x):
        return b.mul(x, b.tanh(b.log(b.add_scalar(b.exp(x), 1.0))))

Registering an op that already has a lowering replaces it.
"""

from __future__ import annotations

from typing import Any, Callable

LOWERINGS: dict[str, Callable[..., Any]] = {}


def op_key(op: Any) -> str:
    """'aten.linear.default' for torch.ops.aten.linear.default and for the string itself."""
    return op if isinstance(op, str) else str(op)


def lowering(*ops: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        for op in ops:
            LOWERINGS[op_key(op)] = fn
        return fn

    return deco
