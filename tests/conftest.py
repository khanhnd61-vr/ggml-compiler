import os
from pathlib import Path

import pytest

# Prefer the CUDA runtime build: it runs the CPU backend too, so one library covers both.
_REPO = Path(__file__).resolve().parents[1]
if "GGML_COMPILER_RT" not in os.environ and (_REPO / "build-cuda" / "libggml_compiler_rt.so").exists():
    os.environ["GGML_COMPILER_RT"] = str(_REPO / "build-cuda" / "libggml_compiler_rt.so")

from ggml_compiler.runtime import has_backend  # noqa: E402


def pytest_configure(config):
    # The end-to-end tests write multi-GB GGUFs. GGML_COMPILER_TEST_TMP puts pytest's temporary directory
    # on a roomier filesystem (pytest clears <GGML_COMPILER_TEST_TMP>/ggml-compiler-pytest at the start of a run).
    root = os.environ.get("GGML_COMPILER_TEST_TMP")
    if root and not config.option.basetemp:
        config.option.basetemp = str(Path(root) / "ggml-compiler-pytest")

BACKENDS = [pytest.param("cpu", id="cpu")]
if has_backend("cuda"):
    BACKENDS.append(pytest.param("cuda", id="cuda", marks=pytest.mark.cuda))


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
