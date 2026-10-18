"""Exact Q8_0 / Q4_0 encoding of weights that already sit on a quantization grid.

A ModelOpt fake-quant weight is k * s with integer k (|k| <= 127 for int8, -8 <= k <= 7 for int4)
and one scale s per channel or per group. Q8_0 stores q * d per 32-block and Q4_0 (q - 8) * d,
so choosing d = s reproduces the grid exactly, up to rounding d to fp16 (relative 2**-11).
ggml's own encoder would pick d from the block's data (max / 127, max / -8) and land off the
grid. Used only when every 32-block shares one scale and every code is an in-range integer.
"""

from __future__ import annotations

import numpy as np

QK = 32


def grid_codes(w: np.ndarray, s: np.ndarray, lo: int, hi: int) -> np.ndarray | None:
    """Integer codes k with w = k * s, or None if w is not on that grid (or s varies in a block)."""
    if w.shape[-1] % QK:
        return None
    sb = s.reshape(-1, QK)
    if not np.all(sb == sb[:, :1]) or np.any(sb[:, 0] <= 0):
        return None
    k = w / s
    kr = np.rint(k)
    if np.abs(k - kr).max() > 1e-3 or kr.min() < lo or kr.max() > hi:
        return None
    return kr.astype(np.int32)


def encode_q8_0(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """uint8 [rows, n/32 * 34]: per block an fp16 scale and 32 int8 codes."""
    rows, n = k.shape
    kb = k.reshape(rows, n // QK, QK)
    d = s.reshape(rows, n // QK, QK)[:, :, 0].astype(np.float16)
    out = np.empty((rows, n // QK, 2 + QK), dtype=np.uint8)
    out[:, :, :2] = d.view(np.uint8).reshape(rows, n // QK, 2)
    out[:, :, 2:] = kb.astype(np.int8).view(np.uint8)
    return out.reshape(rows, -1)


def encode_q4_0(k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """uint8 [rows, n/32 * 18]: per block an fp16 scale and 16 bytes of nibbles, element j in the
    low nibble of byte j and element j + 16 in its high nibble."""
    rows, n = k.shape
    q = (k.reshape(rows, n // QK, QK) + 8).astype(np.uint8)
    d = s.reshape(rows, n // QK, QK)[:, :, 0].astype(np.float16)
    out = np.empty((rows, n // QK, 2 + QK // 2), dtype=np.uint8)
    out[:, :, :2] = d.view(np.uint8).reshape(rows, n // QK, 2)
    out[:, :, 2:] = q[:, :, : QK // 2] | (q[:, :, QK // 2:] << 4)
    return out.reshape(rows, -1)


ENCODERS = {"q8_0": (encode_q8_0, -128, 127), "q4_0": (encode_q4_0, -8, 7)}


def encode_on_grid(w: np.ndarray, s: np.ndarray, storage: str) -> np.ndarray | None:
    """Exact encoding of w (2-D view [rows, n]) on scale grid s, or None if it does not apply."""
    if storage not in ENCODERS:
        return None
    enc, lo, hi = ENCODERS[storage]
    w2 = w.reshape(-1, w.shape[-1])
    s2 = np.broadcast_to(s, w.shape).reshape(w2.shape)
    k = grid_codes(w2, s2, lo, hi)
    return None if k is None else enc(k, s2).reshape(w.shape[:-1] + (-1,))
