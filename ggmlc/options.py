"""Optimization levels and the flags they set.

A level is a named set of flags, each level a superset of the one below:

  O0  lowering, plus the rewrites ggml needs to accept the graph (view composition, size-1
      permutes as reshapes, >4-D weight views folded). For debugging a pass.
  O1  numerics-preserving rewrites. The default, and what parity builds use.
  O2  rewrites that change numerics at the ~1e-3 level for speed.
  O3  specialization that narrows what the artifact accepts at run time (masks that leave every
      query row a key to attend to); Programs record their own (`Program.specialized`).

`flags` toggles single flags on top of a level: {"flash_attention": True, "hoist_invariants": False}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

LEVELS = ("O0", "O1", "O2", "O3")


@dataclass(frozen=True)
class Flag:
    level: str
    doc: str


FLAGS: dict[str, Flag] = {
    "simplify": Flag("O1", "arithmetic identities (x + 0, x * 1, all-zero masks); a slice of a concat reads its part"),
    "flatten_matmul": Flag("O1", "a 2-D weight times a batched activation is one GEMM"),
    "shared_kv_heads": Flag("O1", "grouped-query K/V heads are broadcast by mul_mat instead of repeated"),
    "fold_constants": Flag("O1", "nodes that read only weights and constants are computed at compile time"),
    "fold_batch_norm": Flag("O1", "a per-channel affine after a convolution folds into its kernel and bias"),
    "hoist_invariants": Flag("O1", "loop-invariant work moves into a stage that runs once before the loop"),
    "cse": Flag("O1", "identical nodes (the same op on the same inputs) are computed once"),
    "fuse_glu": Flag("O1", "silu(gate) * up, a SwiGLU MLP, is one ggml_swiglu_split"),
    "fuse_rms_norm": Flag("O1", "x * rsqrt(mean(x^2) + eps), as RMSNorm is written, is one ggml_rms_norm"),
    "strided_rows": Flag("O1", "matmuls, flash attention and row writes read strided views whose rows are "
                               "contiguous (a KV cache view) instead of a copy"),
    "flash_attention": Flag("O2", "eager attention runs as ggml_flash_attn_ext (K/V in f16)"),
    "fast_gelu": Flag("O2", "tanh-GELU is ggml_gelu everywhere (an fp16 table on CPU)"),
    "f16_kv_cache": Flag("O2", "state read only by flash attention or matmuls (a KV cache) is stored in f16"),
    "assume_attended_rows": Flag("O3", "every attention query row may attend to some key (a causal mask): "
                                       "flash attention drops the fully-masked-row guard"),
    "fuse_rope": Flag("O2", "rotary embeddings (x * cos + rotate_half(x) * sin) are one ggml_rope_ext, whose "
                            "angles move by ~1e-4 at position 1000"),
}


@dataclass(frozen=True)
class Options:
    """Resolved compile options: the level and the set of flags that are on."""

    level: str
    on: frozenset[str]

    def __contains__(self, flag: str) -> bool:
        if flag not in FLAGS:
            raise KeyError(f"unknown flag {flag!r}")
        return flag in self.on

    def to_json(self) -> dict[str, Any]:
        return {"level": self.level, "flags": {f: f in self.on for f in FLAGS}}


def _level_flags(level: str) -> set[str]:
    rank = LEVELS.index(level)
    return {f for f, d in FLAGS.items() if LEVELS.index(d.level) <= rank}


def resolve(opt: str | Options | None = None, flags: Mapping[str, bool] | None = None,
            precision: str | None = None, attention: str | None = None) -> Options:
    """Options from a level, flag overrides and the older precision / attention arguments.

    precision="exact" is O1 and "fast" is O2 (only when `opt` is not given); attention="flash" |
    "exact" sets flash_attention."""
    if isinstance(opt, Options):
        on, level = set(opt.on), opt.level
    else:
        if precision not in (None, "exact", "fast"):
            raise ValueError("precision must be 'exact' or 'fast'")
        if opt is not None and precision is not None:
            raise ValueError("pass opt or precision, not both (precision='exact' is O1, 'fast' is O2)")
        level = opt or {"fast": "O2"}.get(precision, "O1")
        if level not in LEVELS:
            raise ValueError(f"opt must be one of {', '.join(LEVELS)}, not {level!r}")
        on = _level_flags(level)
    if attention is not None:
        if attention not in ("exact", "flash"):
            raise ValueError("attention must be 'exact' or 'flash'")
        flags = {**(flags or {}), "flash_attention": attention == "flash"}
    for f, v in (flags or {}).items():
        if f not in FLAGS:
            raise ValueError(f"unknown flag {f!r}; flags are {', '.join(FLAGS)}")
        (on.add if v else on.discard)(f)
    return Options(level, frozenset(on))
