"""The `vla` serving manifest: what a robot-policy host (vla.cpp's `vla-server`) needs to run a
compiled program from a camera image, an instruction, a robot state and a noise sample, with no
code written for the model.

The manifest is plain JSON under the key "vla" of `Program(metadata=...)`, so it travels in the
GGUF (`ggml_compiler.meta`) and is read back with `ggml_compiler_metadata`. It names the program's
inputs and outputs by role and carries the preprocessing the host applies to each: image size,
layout and normalization; token padding; state and action statistics. See docs/SERVING.md.

    import ggml_compiler
    from ggml_compiler import vla

    ggml_compiler.Program.single(model, example, input_names=["state", "front", "wrist"],
                                 output_names=["actions"],
                                 metadata={"vla": vla.manifest(
        images=[vla.image("front", 480, 640, mean=IMAGENET_MEAN, std=IMAGENET_STD),
                vla.image("wrist", 480, 640, mean=IMAGENET_MEAN, std=IMAGENET_STD)],
        state=vla.state("state", mean=state_mean, std=state_std),
        actions=vla.actions("actions", chunk=100, mean=action_mean, std=action_std))})
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

VERSION = 1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# SigLIP / PaliGemma convention: [0, 1] pixels to [-1, 1].
PM1_MEAN = (0.5, 0.5, 0.5)
PM1_STD = (0.5, 0.5, 0.5)

FITS = ("exact", "resize")
LAYOUTS = ("NCHW",)


def _floats(name: str, xs: Sequence[float] | None, n: int | None = None) -> list[float] | None:
    if xs is None:
        return None
    out = [float(x) for x in xs]
    if n is not None and len(out) != n:
        raise ValueError(f"{name} has {len(out)} values, expected {n}")
    return out


def image(input: str, height: int, width: int, *, mean: Sequence[float] = (0.0, 0.0, 0.0),
          std: Sequence[float] = (1.0, 1.0, 1.0), fit: str = "exact", mask: str | None = None,
          layout: str = "NCHW") -> dict[str, Any]:
    """One camera input: a `[1, 3, height, width]` float tensor, from a host image scaled to [0, 1]
    and normalized per channel as `(x - mean) / std`. `fit` says what the host does with an image of
    another size: "exact" refuses it, "resize" scales it bilinearly to height x width. `mask`, if
    the program has one, is a `[1]` input the host sets to 1 when the camera is present."""
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if height <= 0 or width <= 0:
        raise ValueError(f"image {input!r}: size {height}x{width} is not positive")
    d: dict[str, Any] = {"input": input, "height": int(height), "width": int(width), "layout": layout,
                         "fit": fit, "mean": _floats("mean", mean, 3), "std": _floats("std", std, 3)}
    if mask is not None:
        d["mask"] = mask
    return d


def lang(tokens: str, max_len: int, *, mask: str | None = None, pad_id: int = 0) -> dict[str, Any]:
    """The instruction: token ids padded with `pad_id` to `max_len` into the `[1, max_len]` int
    input `tokens`; `mask`, if any, is the matching `[1, max_len]` 0/1 input."""
    if max_len <= 0:
        raise ValueError(f"lang max_len must be positive, got {max_len}")
    d: dict[str, Any] = {"tokens": tokens, "max_len": int(max_len), "pad_id": int(pad_id)}
    if mask is not None:
        d["mask"] = mask
    return d


def _norm(role: str, d: dict[str, Any], dim: int | None, mean, std, min, max, eps: float) -> dict[str, Any]:
    if mean is not None or std is not None:
        if min is not None or max is not None:
            raise ValueError(f"{role}: give mean/std or min/max, not both")
        if mean is None or std is None:
            raise ValueError(f"{role}: mean and std go together")
        d["norm"] = "mean_std"
        d["mean"], d["std"] = _floats("mean", mean, dim), _floats("std", std, dim)
        d["eps"] = float(eps)
    elif min is not None or max is not None:
        if min is None or max is None:
            raise ValueError(f"{role}: min and max go together")
        d["norm"] = "min_max"
        d["min"], d["max"] = _floats("min", min, dim), _floats("max", max, dim)
        d["eps"] = float(eps)
    else:
        d["norm"] = "none"
    return d


def state(input: str, *, dim: int | None = None, real_dim: int | None = None, mean: Sequence[float] | None = None,
          std: Sequence[float] | None = None, min: Sequence[float] | None = None,
          max: Sequence[float] | None = None, eps: float = 1e-8) -> dict[str, Any]:
    """The robot state, a `[1, dim]` float input. The host normalizes the first `real_dim` values
    (LeRobot's `(x - mean) / (std + eps)`, or `2 (x - min) / (max - min + eps) - 1`) and zeroes the
    rest. `dim` defaults to the statistics' length; `real_dim` to `dim`."""
    if dim is None:
        for xs in (mean, std, min, max):
            if xs is not None:
                dim = len(xs)
                break
    if dim is None or dim <= 0:
        raise ValueError(f"state {input!r}: dim is needed when there are no statistics")
    real_dim = dim if real_dim is None else int(real_dim)
    if not 0 < real_dim <= dim:
        raise ValueError(f"state {input!r}: real_dim {real_dim} not in [1, {dim}]")
    d: dict[str, Any] = {"input": input, "dim": int(dim), "real_dim": real_dim}
    return _norm(f"state {input!r}", d, real_dim, mean, std, min, max, eps)


def noise(input: str) -> dict[str, Any]:
    """The initial action noise, a float input the host fills with N(0, 1) samples when the request
    carries none (its shape comes from the program)."""
    return {"input": input}


def actions(output: str, *, chunk: int, dim: int | None = None, real_dim: int | None = None,
            mean: Sequence[float] | None = None, std: Sequence[float] | None = None,
            min: Sequence[float] | None = None, max: Sequence[float] | None = None,
            eps: float = 1e-8) -> dict[str, Any]:
    """The action chunk, a `[1, chunk, dim]` float output. The host un-normalizes the first
    `real_dim` columns with the inverse of the state's formulas and returns `chunk` rows."""
    if chunk <= 0:
        raise ValueError(f"actions {output!r}: chunk must be positive, got {chunk}")
    if dim is None:
        for xs in (mean, std, min, max):
            if xs is not None:
                dim = len(xs)
                break
    if dim is None or dim <= 0:
        raise ValueError(f"actions {output!r}: dim is needed when there are no statistics")
    real_dim = dim if real_dim is None else int(real_dim)
    if not 0 < real_dim <= dim:
        raise ValueError(f"actions {output!r}: real_dim {real_dim} not in [1, {dim}]")
    d: dict[str, Any] = {"output": output, "chunk": int(chunk), "dim": int(dim), "real_dim": real_dim}
    return _norm(f"actions {output!r}", d, real_dim, mean, std, min, max, eps)


def manifest(*, images: Sequence[dict[str, Any]], actions: dict[str, Any], state: dict[str, Any] | None = None,
             lang: dict[str, Any] | None = None, noise: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble and check a version-1 manifest. Every program input the host feeds must be named
    exactly once; `check(manifest, input_names, output_names)` compares against a program."""
    m: dict[str, Any] = {"version": VERSION, "images": list(images), "actions": actions}
    if state is not None:
        m["state"] = state
    if lang is not None:
        m["lang"] = lang
    if noise is not None:
        m["noise"] = noise
    names = input_names(m)
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"manifest names inputs more than once: {dup}")
    return m


def input_names(m: dict[str, Any]) -> list[str]:
    """Every program input the manifest names, in host order."""
    names: list[str] = []
    for im in m.get("images", []):
        names.append(im["input"])
        if im.get("mask"):
            names.append(im["mask"])
    if "lang" in m:
        names.append(m["lang"]["tokens"])
        if m["lang"].get("mask"):
            names.append(m["lang"]["mask"])
    if "state" in m:
        names.append(m["state"]["input"])
    if "noise" in m:
        names.append(m["noise"]["input"])
    return names


def check(m: dict[str, Any], inputs: Sequence[str], outputs: Sequence[str]) -> None:
    """Raise if the manifest and a program's input and output names disagree: the host needs a
    role for every input, and a program input for every role."""
    named, have = set(input_names(m)), set(inputs)
    if named - have:
        raise ValueError(f"manifest names inputs the program lacks: {sorted(named - have)}")
    if have - named:
        raise ValueError(f"program inputs the manifest leaves unfilled: {sorted(have - named)}")
    if m["actions"]["output"] not in set(outputs):
        raise ValueError(f"manifest actions output {m['actions']['output']!r} is not a program output {sorted(outputs)}")
