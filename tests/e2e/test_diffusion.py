"""LeRobot Diffusion Policy (lerobot/diffusion_pusht): the whole sampler as one ggmlc program.

The stage functions reproduce LeRobot's own generate_actions bit for bit (DDIM, and DDPM with
LeRobot's seeded per-step noise passed in as a loop slice), and the compiled program matches
them on CPU and CUDA.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import assert_close, build_generated, check_generated, load_example, scratch_dir

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("lerobot") is None or importlib.util.find_spec("diffusers") is None:
    pytest.skip("LeRobot with the diffusion extra not installed (uv sync)", allow_module_level=True)
dp = load_example("diffusion")


def _load(**kw):
    try:
        return dp.load(**kw)
    except Exception as e:  # checkpoint not in the local cache
        pytest.skip(f"cannot load {dp.CHECKPOINT}: {e}")


@pytest.fixture(scope="module")
def ddim(tmp_path_factory):
    policy = _load(scheduler="DDIM", num_inference_steps=10)
    with scratch_dir(tmp_path_factory, "dp-ddim") as out:
        yield policy, ggmlc.compile(dp.program_from_policy(policy), out_dir=out)


@pytest.fixture(scope="module")
def ddpm(tmp_path_factory):
    policy = _load()
    with scratch_dir(tmp_path_factory, "dp-ddpm") as out:
        yield policy, ggmlc.compile(dp.program_from_policy(policy, seed=1), out_dir=out)


def _lerobot_actions(policy, seed=1):
    inputs = dp.example_inputs(policy.config)
    batch = {"observation.state": inputs["state"], "observation.images": inputs["images"]}
    torch.manual_seed(seed)  # DDPMScheduler.step draws its noise from the global generator
    with torch.no_grad():
        return policy.diffusion.generate_actions(batch, noise=inputs["noise"]).numpy()


@pytest.mark.parametrize("which", ["ddim", "ddpm"])
def test_stages_reproduce_lerobot(which, request):
    policy, art = request.getfixturevalue(which)
    ref = np.load(art.out_dir / "reference_outputs.npz")["actions"]
    np.testing.assert_array_equal(ref, _lerobot_actions(policy))


@pytest.mark.parametrize("which", ["ddim", "ddpm"])
def test_diffusion_parity(which, request, backend):
    _, art = request.getfixturevalue(which)
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**ins)
        n = s.stats["stages"]["step"]["calls"]
    assert n == (10 if which == "ddim" else 100)
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), ACCEPT[backend])


def test_step_noise_is_an_input(ddpm, backend):
    """DDPM's per-step noise comes from the caller: other noise, other actions."""
    _, art = ddpm
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        a = s.run(**ins)["actions"]
        ins["step_noise"] = np.ascontiguousarray(ins["step_noise"][::-1])
        b = s.run(**ins)["actions"]
    assert np.abs(a - b).max() > 1e-2


def test_graph_shape(ddim):
    ops = ddim[1].report.data["stages"]["step"]["ops"]
    assert ops["conv_2d"] == 33  # every Conv1d of the U-Net, as a 2-D convolution over [N, C, 1, L]
    assert "conv_transpose_1d" not in ops  # transposed convs are a matmul + overlap-add


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    policy = _load(scheduler="DDIM", num_inference_steps=10)
    with scratch_dir(tmp_path_factory, "dp-cpp") as out:
        art = ggmlc.compile(dp.program_from_policy(policy), out_dir=out, emit=("gguf", "cpp"), name="diffusion",
                            dtype_policy=Path(dp.__file__).parent / "policy-f16.yaml")
        yield art, build_generated(out / "cpp", out / "build")


def test_f16_policy(generated, backend):
    art, _ = generated
    tensors = art.report.data["weights"]["tensors"]
    assert sum(d["dtype"] == "f16" for d in tensors.values()) > 60
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**ins)
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), 5e-3)


def test_generated_cpp_is_bit_identical(generated, backend):
    check_generated(*generated, backend)
