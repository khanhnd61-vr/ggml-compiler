"""Import models quantized with NVIDIA ModelOpt (mtq.quantize, or a restored QAT checkpoint).

For export, every enabled weight quantizer is applied once: the module's weight is replaced by its
fake-quantized value and the quantizer is switched off, so torch.export sees a plain float model
whose weights already sit on the quantization grid. Those weights are then stored in the mapped
ggml type (8-bit -> Q8_0, 4-bit -> Q4_0 by default). ggml's block scales rarely reproduce
ModelOpt's per-channel or per-group scales exactly, so the compile report gives each weight's error
against its fake-quant value. Activation quantizers are switched off and listed: ggml quantizes
activations itself inside its quantized matmuls. The model is restored afterwards.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from ..ir.program import CompileError
from ..quant.types import canonical


@dataclass
class ModelOptImport:
    """What prepare() found: per weight FQN, the source quantizer and the chosen ggml type."""

    weights: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)  # activation quantizers switched off
    scales: dict[str, torch.Tensor] = field(default_factory=dict)  # per-element grid scale s


@dataclass
class ModelOpt:
    int8: str = "q8_0"
    int4: str = "q4_0"

    def __post_init__(self) -> None:
        self.int8, self.int4 = canonical(self.int8), canonical(self.int4)

    @contextlib.contextmanager
    def prepare(self, root: torch.nn.Module, found: ModelOptImport) -> Iterator[ModelOptImport]:
        try:
            from modelopt.torch.quantization.nn import TensorQuantizer
        except ImportError as e:  # pragma: no cover
            raise CompileError("prequant=ModelOpt needs nvidia-modelopt (uv sync --extra modelopt)") from e

        restore: list[tuple[torch.nn.Module, torch.Tensor]] = []
        switched_off: list[TensorQuantizer] = []
        try:
            for name, mod in root.named_modules():
                wq = getattr(mod, "weight_quantizer", None)
                if isinstance(wq, TensorQuantizer) and wq.is_enabled:
                    bits = wq.num_bits
                    if bits not in (8, 4):
                        raise CompileError(f"{name}: {bits}-bit weight quantization is not supported "
                                           "(int8 and int4 map onto ggml types)")
                    with torch.no_grad():
                        fq = wq(mod.weight).detach().clone()
                    restore.append((mod, mod.weight.data.clone()))
                    mod.weight.data.copy_(fq)
                    wq.disable()
                    switched_off.append(wq)
                    fqn = f"{name}.weight" if name else "weight"
                    found.weights[fqn] = {"bits": bits, "block_sizes": _plain(wq.block_sizes),
                                          "dtype": self.int8 if bits == 8 else self.int4}
                    s = _grid_scale(wq, mod.weight)
                    if s is not None:
                        found.scales[fqn] = s
                for qname in ("input_quantizer", "output_quantizer"):
                    q = getattr(mod, qname, None)
                    if isinstance(q, TensorQuantizer) and q.is_enabled:
                        q.disable()
                        switched_off.append(q)
                        found.dropped.append(f"{name}.{qname}")
            yield found
        finally:
            for mod, w in restore:
                mod.weight.data.copy_(w)
            for q in switched_off:
                q.enable()


def _grid_scale(wq: Any, w: torch.Tensor) -> torch.Tensor | None:
    """Per-element scale s of the integer grid the fake-quant weight sits on (amax / maxbound),
    for per-channel and last-dim block quantizers (a shorter last block included); None for any
    other layout, which then falls back to re-encoding."""
    amax = getattr(wq, "amax", None)
    if amax is None:
        return None
    bound = 2 ** (wq.num_bits - 1) - 1
    s = amax.detach().float() / bound
    blocks = wq.block_sizes or {}
    block = blocks.get(-1) or blocks.get(w.dim() - 1)
    try:
        if not block:
            return s.expand_as(w).contiguous()
        n = w.shape[-1]
        groups = -(-n // block)  # the last group may be shorter than the block
        per_group = s.reshape(*w.shape[:-1], groups)
        return per_group[..., torch.arange(n) // block].contiguous()
    except RuntimeError:  # an amax layout this does not understand
        return None


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v
