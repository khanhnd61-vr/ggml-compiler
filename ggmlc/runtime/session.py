"""ctypes bindings to libggmlc_rt.

The library is found through $GGMLC_RT, or else in this checkout's build-cuda/ and build/
directories, in that order. A CUDA build runs on the CPU too.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_BACKENDS = {"cpu": 0, "cuda": 1, "auto": 2}
_DTYPES = {0: ("f32", np.float32), 1: ("f16", np.float16), 2: ("i32", np.int32)}
_MAX_RANK = 8


class _Options(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("device", ctypes.c_int), ("n_threads", ctypes.c_int),
                ("allow_cpu_fallback", ctypes.c_bool)]


class _TensorInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("dtype", ctypes.c_int), ("ndim", ctypes.c_int),
                ("shape", ctypes.c_int64 * _MAX_RANK), ("nbytes", ctypes.c_size_t)]


class _Stats(ctypes.Structure):
    _fields_ = [("ms_total", ctypes.c_double), ("n_calls", ctypes.c_int)]


class GgmlcRuntimeError(RuntimeError):
    """An error reported by libggmlc_rt."""


def library_path() -> Path:
    env = os.environ.get("GGMLC_RT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2]
    for d in ("build-cuda", "build"):
        p = repo / d / "libggmlc_rt.so"
        if p.exists():
            return p
    raise RuntimeError("libggmlc_rt.so not found: build the runtime (see README) or set GGMLC_RT")


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(library_path()))
    p, i, c_size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    sigs = {
        "ggmlc_version": ([], ctypes.c_char_p),
        "ggmlc_last_error": ([], ctypes.c_char_p),
        "ggmlc_default_options": ([], _Options),
        "ggmlc_has_backend": ([i], ctypes.c_bool),
        "ggmlc_load": ([ctypes.c_char_p, ctypes.POINTER(_Options)], p),
        "ggmlc_quantize": ([ctypes.c_char_p, p, ctypes.c_int64, ctypes.c_int64, p, c_size], c_size),
        "ggmlc_row_size": ([ctypes.c_char_p, ctypes.c_int64], c_size),
        "ggmlc_free": ([p], None),
        "ggmlc_backend_name": ([p], ctypes.c_char_p),
        "ggmlc_n_inputs": ([p], i),
        "ggmlc_n_outputs": ([p], i),
        "ggmlc_input_info": ([p, i, ctypes.POINTER(_TensorInfo)], i),
        "ggmlc_output_info": ([p, i, ctypes.POINTER(_TensorInfo)], i),
        "ggmlc_set_input": ([p, ctypes.c_char_p, p, c_size], i),
        "ggmlc_n_entries": ([p], i),
        "ggmlc_entry_name": ([p, i], ctypes.c_char_p),
        "ggmlc_entry_n_inputs": ([p, i], i),
        "ggmlc_entry_n_outputs": ([p, i], i),
        "ggmlc_entry_input_info": ([p, i, i, ctypes.POINTER(_TensorInfo)], i),
        "ggmlc_entry_output_info": ([p, i, i, ctypes.POINTER(_TensorInfo)], i),
        "ggmlc_run_entry": ([p, ctypes.c_char_p], i),
        "ggmlc_reset": ([p], i),
        "ggmlc_metadata": ([p, ctypes.c_char_p], ctypes.c_char_p),
        "ggmlc_n_symbols": ([p], i),
        "ggmlc_symbol_info": ([p, i, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
                               ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)], i),
        "ggmlc_set_symbol": ([p, ctypes.c_char_p, ctypes.c_int64], i),
        "ggmlc_set_input_shape": ([p, ctypes.c_char_p, i, ctypes.POINTER(ctypes.c_int64)], i),
        "ggmlc_run": ([p], i),
        "ggmlc_get_output": ([p, ctypes.c_char_p, p, c_size], i),
        "ggmlc_last_stats": ([p, ctypes.POINTER(_Stats)], i),
        "ggmlc_n_stages": ([p], i),
        "ggmlc_stage_stats": ([p, i, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_int)], i),
    }
    for name, (args, res) in sigs.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = res
    _lib = lib
    return lib


def _error() -> GgmlcRuntimeError:
    return GgmlcRuntimeError(_load().ggmlc_last_error().decode())


def version() -> str:
    return _load().ggmlc_version().decode()


def quantize(data: np.ndarray, type_name: str) -> np.ndarray:
    """Encode f32 data (rows along the last axis) in a ggml type; uint8 [rows..., row_bytes]."""
    lib = _load()
    x = np.ascontiguousarray(data, dtype=np.float32)
    n = x.shape[-1]
    rows = x.size // n
    row_bytes = lib.ggmlc_row_size(type_name.encode(), n)
    if row_bytes == 0:
        raise _error()
    out = np.empty(x.shape[:-1] + (row_bytes,), dtype=np.uint8)
    if lib.ggmlc_quantize(type_name.encode(), x.ctypes.data, rows, n, out.ctypes.data, out.nbytes) == 0:
        raise _error()
    return out


def has_backend(backend: str) -> bool:
    return bool(_load().ggmlc_has_backend(_BACKENDS[backend]))


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


class Session:
    """A loaded model. `run` takes inputs positionally or by name and returns outputs by name.

    A program with symbolic dims takes inputs of any shape it accepts: `run` binds the symbols
    from the input shapes (`symbols` lists them with their ranges). `inputs` and `outputs` show
    the shapes with every symbol at its max.

    A program with several entry points runs one per call: `run(..., entry="decode")` (default:
    the first). `entries` maps each to its (inputs, outputs). State (a KV cache) persists across
    runs until `reset()`."""

    def __init__(self, path: str | os.PathLike, backend: str = "auto", device: int = 0, n_threads: int = 0,
                 allow_cpu_fallback: bool = False) -> None:
        lib = _load()
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        opt = lib.ggmlc_default_options()
        opt.backend, opt.device, opt.n_threads, opt.allow_cpu_fallback = (
            _BACKENDS[backend], device, n_threads, allow_cpu_fallback)
        self._m = lib.ggmlc_load(os.fsencode(path), ctypes.byref(opt))
        if not self._m:
            raise _error()
        self.backend_name: str = lib.ggmlc_backend_name(self._m).decode()
        self._np: dict[str, np.dtype] = {}
        self.entries: dict[str, tuple[list[TensorInfo], list[TensorInfo]]] = {}
        for e in range(lib.ggmlc_n_entries(self._m)):
            ins = [self._info(lambda m, k, info: lib.ggmlc_entry_input_info(m, e, k, info), k)
                   for k in range(lib.ggmlc_entry_n_inputs(self._m, e))]
            outs = [self._info(lambda m, k, info: lib.ggmlc_entry_output_info(m, e, k, info), k)
                    for k in range(lib.ggmlc_entry_n_outputs(self._m, e))]
            self.entries[lib.ggmlc_entry_name(self._m, e).decode()] = (ins, outs)
        self.inputs, self.outputs = next(iter(self.entries.values()))
        self.symbols: dict[str, tuple[int, int]] = {}
        for k in range(lib.ggmlc_n_symbols(self._m)):
            name, lo, hi = ctypes.c_char_p(), ctypes.c_int64(), ctypes.c_int64()
            lib.ggmlc_symbol_info(self._m, k, ctypes.byref(name), ctypes.byref(lo), ctypes.byref(hi), None)
            self.symbols[name.value.decode()] = (lo.value, hi.value)

    @property
    def metadata(self) -> dict[str, Any]:
        """The program's own metadata (Program(metadata=...)), or {}."""
        import json

        raw = _load().ggmlc_metadata(self._m, b"ggmlc.meta")
        return json.loads(raw.decode()) if raw else {}

    def set_symbol(self, name: str, value: int) -> None:
        """Bind a symbol no input shape carries (a loop's trip count)."""
        if _load().ggmlc_set_symbol(self._m, name.encode(), int(value)) != 0:
            raise _error()

    def bound(self) -> dict[str, int]:
        """The current value of every symbol."""
        lib, out = _load(), {}
        for k in range(len(self.symbols)):
            name, v = ctypes.c_char_p(), ctypes.c_int64()
            lib.ggmlc_symbol_info(self._m, k, ctypes.byref(name), None, None, ctypes.byref(v))
            out[name.value.decode()] = v.value
        return out

    def _info(self, fn: Any, k: int) -> TensorInfo:
        info = _TensorInfo()
        if fn(self._m, k, ctypes.byref(info)) != 0:
            raise _error()
        name, (dtype, np_dtype) = info.name.decode(), _DTYPES[info.dtype]
        self._np[name] = np.dtype(np_dtype)
        return TensorInfo(name, tuple(info.shape[: info.ndim]), dtype, info.nbytes)

    def reset(self) -> None:
        """Zero the program's state (a KV cache)."""
        if _load().ggmlc_reset(self._m) != 0:
            raise _error()

    def run(self, *args: Any, entry: str | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        lib = _load()
        entry = entry or next(iter(self.entries))
        if entry not in self.entries:
            raise ValueError(f"no entry point {entry!r}; the model has {list(self.entries)}")
        inputs, outputs = self.entries[entry]
        if len(args) > len(inputs):
            raise TypeError(f"{len(args)} positional inputs for an entry with {len(inputs)} inputs")
        given = {t.name: a for t, a in zip(inputs, args)}
        for k, v in kwargs.items():
            if k in given:
                raise TypeError(f"input {k} given twice")
            given[k] = v
        names = {t.name for t in inputs}
        unknown = set(given) - names
        if unknown:
            raise TypeError(f"unknown inputs {sorted(unknown)}; the model takes {[t.name for t in inputs]}")
        missing = names - set(given)
        if missing:
            raise TypeError(f"missing inputs {sorted(missing)}")
        arrays = {t.name: _as_numpy(given[t.name], self._np[t.name]) for t in inputs}
        for t in inputs:
            x = arrays[t.name]
            if not self.symbols:
                if x.shape != t.shape:
                    raise ValueError(f"input {t.name} has shape {x.shape}, the model expects {t.shape}")
                continue
            shape = (ctypes.c_int64 * max(1, x.ndim))(*x.shape)
            if lib.ggmlc_set_input_shape(self._m, t.name.encode(), x.ndim, shape) != 0:
                raise _error()
        for t in inputs:
            x = arrays[t.name]
            if lib.ggmlc_set_input(self._m, t.name.encode(), x.ctypes.data, x.nbytes) != 0:
                raise _error()
        if lib.ggmlc_run_entry(self._m, entry.encode()) != 0:
            raise _error()
        out = {}
        if self.symbols:
            e = list(self.entries).index(entry)
            outputs = [self._info(lambda m, k, info: lib.ggmlc_entry_output_info(m, e, k, info), k)
                       for k in range(len(outputs))]
        for t in outputs:
            y = np.empty(t.shape, dtype=self._np[t.name])
            if lib.ggmlc_get_output(self._m, t.name.encode(), y.ctypes.data, y.nbytes) != 0:
                raise _error()
            out[t.name] = y
        return out

    @property
    def stats(self) -> dict[str, Any]:
        lib = _load()
        st = _Stats()
        if lib.ggmlc_last_stats(self._m, ctypes.byref(st)) != 0:
            raise _error()
        stages = {}
        for k in range(lib.ggmlc_n_stages(self._m)):
            name, ms, calls = ctypes.c_char_p(), ctypes.c_double(), ctypes.c_int()
            lib.ggmlc_stage_stats(self._m, k, ctypes.byref(name), ctypes.byref(ms), ctypes.byref(calls))
            stages[name.value.decode()] = {"ms": ms.value, "calls": calls.value}
        return {"ms_total": st.ms_total, "n_calls": st.n_calls, "stages": stages}

    def close(self) -> None:
        if getattr(self, "_m", None):
            _load().ggmlc_free(self._m)
            self._m = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _as_numpy(x: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor without importing torch here
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x).astype(dtype, copy=False))
