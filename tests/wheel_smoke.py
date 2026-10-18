"""Smoke test for an installed wheel, run outside the source tree (the release workflow runs it on
every wheel): the bundled runtime loads, a small model compiles and matches PyTorch on the CPU,
and, when CMake is available, a generated C++ project builds against the installed runtime and
matches it bit for bit.

    python tests/wheel_smoke.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

import ggml_compiler
from ggml_compiler.runtime import library_path, source_checkout, version

assert source_checkout() is None, "run this against an installed wheel, outside the source tree"
print(ggml_compiler.__version__, version(), library_path())

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)).eval()
x = torch.randn(2, 16)
with tempfile.TemporaryDirectory() as d:
    emit = ("gguf", "cpp") if shutil.which("cmake") else ("gguf",)
    art = ggml_compiler.compile(model, (x,), d, input_names=["x"], output_names=["y"], emit=emit, name="smoke")
    with ggml_compiler.Session(art.gguf, backend="cpu") as s:
        y = s.run(x=x)["y"]
    err = float(np.abs(y - model(x).detach().numpy()).max())
    assert err < 1e-4, f"max|d| {err}"
    print(f"compiled and ran on the CPU: max|d| {err:.2g}")
    if "cpp" in emit:
        build = Path(d) / "build"
        for cmd in (["cmake", "-S", f"{d}/cpp", "-B", str(build)], ["cmake", "--build", str(build)]):
            subprocess.run(cmd, check=True, capture_output=True)
        res = subprocess.run([str(build / "smoke-check"), str(art.gguf), f"{d}/example_inputs.bin", "cpu"],
                             capture_output=True, text=True)
        assert res.returncode == 0 and "bit-identical" in res.stdout, res.stdout + res.stderr
        print("generated C++ against the installed runtime: bit-identical")
print("ok")
