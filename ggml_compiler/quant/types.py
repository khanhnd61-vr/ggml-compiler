"""Weight storage types: user-facing names, ggml names, block sizes."""

from __future__ import annotations

# user name -> (ggml type name, block size)
STORAGE = {
    "f32": ("f32", 1),
    "f16": ("f16", 1),
    "bf16": ("bf16", 1),
    "q8_0": ("q8_0", 32),
    "q4_0": ("q4_0", 32),
    "q4_1": ("q4_1", 32),
    "q5_0": ("q5_0", 32),
    "q5_1": ("q5_1", 32),
    "iq4_nl": ("iq4_nl", 32),
    "q4_k": ("q4_K", 256),
    "q5_k": ("q5_K", 256),
    "q6_k": ("q6_K", 256),
}


def canonical(name: str) -> str:
    key = name.lower()
    if key not in STORAGE:
        raise ValueError(f"unknown weight dtype {name!r}; known: {', '.join(STORAGE)}")
    return key


def ggml_name(name: str) -> str:
    return STORAGE[canonical(name)][0]


def block_size(name: str) -> int:
    return STORAGE[canonical(name)][1]


def is_float(name: str) -> bool:
    return canonical(name) in ("f32", "f16", "bf16")
