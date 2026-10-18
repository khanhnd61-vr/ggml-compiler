"""The parity metric: max |out - ref| scaled by max(1, max |ref|)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Diff:
    name: str
    max_abs: float
    ref_range: float  # max |ref|
    scaled: float  # max_abs / max(1, ref_range)

    def ok(self, tol: float) -> bool:
        return self.scaled <= tol  # False for NaN too


def compare(outputs: Mapping[str, np.ndarray], refs: Mapping[str, np.ndarray]) -> list[Diff]:
    diffs = []
    for name, ref in refs.items():
        if name not in outputs:
            raise KeyError(f"output {name} missing")
        out = np.asarray(outputs[name], dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        if out.shape != ref.shape:
            raise ValueError(f"output {name} has shape {out.shape}, reference {ref.shape}")
        max_abs = float(np.max(np.abs(out - ref))) if ref.size else 0.0
        if not np.isfinite(out[np.isfinite(ref)]).all():
            max_abs = float("inf")  # NaN or inf where the reference is finite: never "close"
        rng = float(np.max(np.abs(ref))) if ref.size else 0.0
        diffs.append(Diff(name, max_abs, rng, max_abs / max(1.0, rng)))
    return diffs


@dataclass(frozen=True)
class LogitStats:
    """How far a language model's next-token distributions are from a reference's, as
    llama-perplexity --kl-divergence reports it: KL(ref || out) per position, and how often the
    two agree on the most likely token."""

    n: int
    kl_mean: float
    kl_p99: float
    kl_max: float
    top1: float  # fraction of positions with the same argmax

    def row(self) -> str:
        return f"KL mean {self.kl_mean:.3e}  p99 {self.kl_p99:.3e}  max {self.kl_max:.3e}  top-1 {100 * self.top1:.2f}%"


def token_stats(out: np.ndarray, ref: np.ndarray, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per position: KL(ref || out) and whether the argmax agrees. out and ref: logits [n, vocab].
    NaN or inf in `out` gives an infinite KL."""
    if out.shape != ref.shape or out.ndim != 2:
        raise ValueError(f"logits must both be [n, vocab], got {out.shape} and {ref.shape}")
    kls, same = [], []
    for i in range(0, len(ref), chunk):
        r = ref[i:i + chunk].astype(np.float64)
        o = out[i:i + chunk].astype(np.float64)
        lr = r - r.max(-1, keepdims=True)
        lr -= np.log(np.exp(lr).sum(-1, keepdims=True))
        lo = o - o.max(-1, keepdims=True)
        lo -= np.log(np.exp(lo).sum(-1, keepdims=True))
        kl = (np.exp(lr) * (lr - lo)).sum(-1)
        kls.append(np.where(np.isfinite(o).all(-1), kl, np.inf))
        same.append(r.argmax(-1) == o.argmax(-1))
    return np.concatenate(kls), np.concatenate(same)


def stats_of(kl: np.ndarray, agree: np.ndarray) -> LogitStats:
    """LogitStats over per-position KLs and argmax agreement (token_stats, pooled over texts)."""
    return LogitStats(len(kl), float(kl.mean()), float(np.percentile(kl, 99)), float(kl.max()), float(agree.mean()))


def logit_stats(out: np.ndarray, ref: np.ndarray, chunk: int = 64) -> LogitStats:
    """out and ref: logits [n_positions, n_vocab]. NaN or inf in `out` gives an infinite KL."""
    return stats_of(*token_stats(out, ref, chunk))
