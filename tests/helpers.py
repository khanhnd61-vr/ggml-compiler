"""Shared test helpers: run IR or modules through the runtime and compare with PyTorch."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

from ggml_compiler.emit.gguf_runtime import write_gguf
from ggml_compiler.ir import Stage, run_stage
from ggml_compiler.ir.program import ProgramIR
from ggml_compiler.runtime import Session
from ggml_compiler.verify.parity import compare

# Parity tolerances on max|out - ref| / max(1, max|ref|). CUDA matmuls do not accumulate in
# full f32, so CUDA gets more room.
TOL = {"cpu": 1e-5, "cuda": 2e-3}


def run_stage_on(stage: Stage, inputs: list[torch.Tensor], backend: str, tmp_path: Path) -> list[np.ndarray]:
    prog = ProgramIR.from_stage(stage)
    path = tmp_path / f"{stage.name}.gguf"
    write_gguf(prog, path, {})
    with Session(path, backend=backend) as s:
        out = s.run(*inputs)
    return [out[f"out.{i}"] for i in range(len(stage.outputs))]


def assert_close(outputs: dict[str, np.ndarray], refs: dict[str, np.ndarray], tol: float) -> None:
    bad = [d for d in compare(outputs, refs) if not d.ok(tol)]
    assert not bad, "\n".join(f"{d.name}: max|d|={d.max_abs:.3g} scaled={d.scaled:.3g} > {tol}" for d in bad)


def check_stage(stage: Stage, inputs: list[torch.Tensor], backend: str, tmp_path: Path, tol: float | None = None) -> None:
    """The runtime agrees with the IR interpreter on this stage."""
    refs = run_stage(stage, inputs)
    outs = run_stage_on(stage, inputs, backend, tmp_path)
    assert_close({str(i): o for i, o in enumerate(outs)}, {str(i): r.numpy() for i, r in enumerate(refs)},
                 tol if tol is not None else TOL[backend])


def check_module(module: torch.nn.Module, inputs: tuple, backend: str, tmp_path: Path, tol: float | None = None) -> None:
    """Compile a module and check the exported IR, the IR after passes, and the runtime against
    PyTorch."""
    import ggml_compiler
    from ggml_compiler.api import Program
    from ggml_compiler.frontend.torch_export import export_program
    from ggml_compiler.passes import run_pipeline

    tol = tol if tol is not None else TOL[backend]
    with torch.no_grad():
        expected = module(*inputs)
    expected = [expected] if isinstance(expected, torch.Tensor) else list(expected)

    prog, _ = export_program(Program.single(module, inputs))
    stage = next(iter(prog.stages.values()))
    for when in ("exported", "after passes"):
        got = run_stage(stage, list(inputs))
        for g, e in zip(got, expected):
            d = (g.reshape(e.shape) - e).abs().max().item()  # a >4-D output comes back squeezed
            assert d <= 1e-5 * max(1.0, e.abs().max().item()), f"interpreter {when}: max|d|={d}"
        run_pipeline(prog)

    art = ggml_compiler.compile(module, inputs, tmp_path / "art")
    with Session(art.gguf, backend=backend) as s:
        out = s.run(*inputs)
    names = [t.name for t in s.outputs]
    assert_close(dict(zip(names, out.values())), {n: e.numpy() for n, e in zip(names, expected)}, tol)


def load_example(name: str):
    """examples/<name>/export.py as a module named <name>_example (each example has an export.py,
    so plain imports would collide in sys.modules)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "examples" / name / "export.py"
    spec = importlib.util.spec_from_file_location(f"{name}_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_generated(cpp_dir, build_dir) -> str:
    """Configure and build a generated C++ project against this checkout's runtime library;
    returns the <name>-check binary."""
    import subprocess

    from ggml_compiler.runtime import library_path

    cfg = ["cmake", "-S", str(cpp_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
           f"-DGGML_COMPILER_RUNTIME_LIB={library_path()}"]
    for cmd in (cfg, ["cmake", "--build", str(build_dir), "-j8"]):
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return str(next(build_dir.glob("*-check")))


def check_generated(art, binary, backend):
    """Run <name>-check: generated builders vs the runtime, same GGUF and inputs, bit for bit."""
    import subprocess

    res = subprocess.run([binary, str(art.gguf), str(art.out_dir / "example_inputs.bin"), backend, "4"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "DIFFERENT" not in res.stdout and "bit-identical" in res.stdout
    return res.stdout


@contextlib.contextmanager
def scratch_dir(tmp_path_factory, name: str):
    """A fresh temporary directory, deleted when the block ends. Module fixtures holding multi-GB
    artifacts use it, so a test run never keeps more than one big model on disk."""
    import shutil

    d = tmp_path_factory.mktemp(name)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
