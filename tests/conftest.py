import os
from pathlib import Path

import pytest

# Prefer the CUDA runtime build: it runs the CPU backend too, so one library covers both.
_REPO = Path(__file__).resolve().parents[1]
if "GGMLC_RT" not in os.environ and (_REPO / "build-cuda" / "libggmlc_rt.so").exists():
    os.environ["GGMLC_RT"] = str(_REPO / "build-cuda" / "libggmlc_rt.so")

from ggmlc.runtime import has_backend  # noqa: E402


def pytest_configure(config):
    # The end-to-end tests write multi-GB GGUFs. GGMLC_TEST_TMP puts pytest's temporary directory
    # on a roomier filesystem (pytest clears <GGMLC_TEST_TMP>/ggmlc-pytest at the start of a run).
    root = os.environ.get("GGMLC_TEST_TMP")
    if root and not config.option.basetemp:
        config.option.basetemp = str(Path(root) / "ggmlc-pytest")

BACKENDS = [pytest.param("cpu", id="cpu")]
if has_backend("cuda"):
    BACKENDS.append(pytest.param("cuda", id="cuda", marks=pytest.mark.cuda))


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
