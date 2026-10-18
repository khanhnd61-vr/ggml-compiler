"""M2 exit check: LeRobot ACT matches PyTorch on CPU and CUDA at the acceptance tolerances."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from helpers import assert_close, build_generated, check_generated, load_example

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the local Hugging Face cache only
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("lerobot") is None:
    pytest.skip("LeRobot not installed (uv sync --group parity)", allow_module_level=True)
act_example = load_example("act")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    try:
        program = act_example.program()
    except Exception as e:  # checkpoint not in the local cache
        pytest.skip(f"cannot load {act_example.CHECKPOINT}: {e}")
    art = ggmlc.compile(program, out_dir=tmp_path_factory.mktemp("act"))
    stage = program.stages["forward"]
    with torch.no_grad():
        ref = stage.fn(*stage.example_inputs)
    return art, stage.example_inputs, ref


def test_act_parity(compiled, backend):
    art, inputs, ref = compiled
    with Session(art.gguf, backend=backend) as s:
        out = s.run(*inputs)
    assert_close(out, {"actions": ref.numpy()}, ACCEPT[backend])


def test_act_parity_at_o0(compiled, backend, tmp_path_factory):
    """Without the optional rewrites (no batch-norm folding, no flattened matmuls) ACT still
    compiles and matches."""
    art = ggmlc.compile(act_example.program(), out_dir=tmp_path_factory.mktemp("act-o0"), opt="O0")
    assert art.report.data["stages"]["forward"]["ops"]["mul"] > 40  # the batch norms stay
    _, inputs, ref = compiled
    with Session(art.gguf, backend=backend) as s:
        out = s.run(*inputs)
    assert_close(out, {"actions": ref.numpy()}, ACCEPT[backend])


def test_act_graph_is_compact(compiled):
    ops = compiled[0].report.data["stages"]["forward"]["ops"]
    assert ops.get("concat", 0) <= 4  # list(features) + stack folded back into slices
    assert ops["conv_2d"] == 42 and ops["pool_2d"] == 2  # (20 ResNet convs + 1x1 projection) x 2 cameras
    assert ops.get("mul", 0) <= 12  # the 40 frozen batch norms are folded into the conv kernels


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    try:
        program = act_example.program()
    except Exception as e:
        pytest.skip(f"cannot load {act_example.CHECKPOINT}: {e}")
    out = tmp_path_factory.mktemp("act-cpp")
    art = ggmlc.compile(program, out_dir=out, emit=("gguf", "cpp"), name="act",
                        dtype_policy=Path(act_example.__file__).parent / "policy-cuda.yaml")
    return art, build_generated(out / "cpp", out / "build")


def test_act_generated_cpp_is_bit_identical(generated, backend):
    """M6: the generated C++ builds and matches the runtime bit for bit (f16 conv kernels, so the
    CUDA run goes through direct convolution)."""
    art, binary = generated
    assert "ggmlc::ops::conv_2d" in (art.out_dir / "cpp" / "act.cpp").read_text()
    check_generated(art, binary, backend)
