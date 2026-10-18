"""Rewrite tier-2 ops into tier-1 ops."""

from __future__ import annotations

import math

from ..ir import Builder, Node, Stage, Value
from . import rewrite


def _linear(b: Builder, n: Node) -> Value:
    x, w = n.inputs[0], n.inputs[1]
    y = b.mul_mat(w, x)
    return b.add(y, n.inputs[2]) if len(n.inputs) > 2 else y


def _layer_norm(b: Builder, n: Node) -> Value:
    rest = list(n.inputs[1:])
    y = b.norm(n.inputs[0], eps=n.attrs["eps"])
    if n.attrs.get("has_weight"):
        y = b.mul(y, rest.pop(0))
    if n.attrs.get("has_bias"):
        y = b.add(y, rest.pop(0))
    return y


def _sdpa(b: Builder, n: Node) -> Value:
    q, k, v = n.inputs[:3]
    scores = b.mul_mat(k, q)  # [..., Sq, Sk]
    if len(n.inputs) > 3:
        mask = n.inputs[3]
        if mask.rank < 2 or mask.shape[-2] != scores.shape[-2]:
            # ggml reads one mask row per score row.
            mshape = (mask.shape[:-2] if mask.rank >= 2 else ()) + scores.shape[-2:]
            mask = b.repeat(mask, shape=list(mshape))
        p = b.soft_max(scores, mask, scale=n.attrs["scale"])
    else:
        p = b.soft_max(scores, scale=n.attrs["scale"])
    vt = b.transpose(v, v.rank - 2, v.rank - 1)  # [..., Dv, Sk]
    return b.mul_mat(vt, p)  # [..., Sq, Dv]


_RULES = {"linear": _linear, "layer_norm": _layer_norm, "sdpa": _sdpa}


def lower_tier2(stage: Stage, fast_gelu: bool = False) -> None:
    rules = dict(_RULES)
    if fast_gelu:
        rules["gelu_tanh"] = lambda b, n: b.gelu(n.inputs[0])
    rewrite(stage, lambda b, n: rules[n.op](b, n) if n.op in rules else None)
