"""LeRobot pi0 (lerobot/pi0_libero_finetuned_v044) and pi0.5 as ggmlc programs.

Parity runs on the real weights with the language model and the expert cut to 2 layers each
(full vision tower): the same graph as the 18-layer model, small enough for the test suite. The
full-size builds are in docs/USAGE.md's examples table; GGMLC_BIG=1 runs a full-size check here too.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
from helpers import assert_close, load_example, scratch_dir

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("lerobot") is None:
    pytest.skip("LeRobot not installed (uv sync)", allow_module_level=True)
pi = load_example("pi")
CHECKPOINTS = {"pi0": "lerobot/pi0_libero_finetuned_v044", "pi05": "lerobot/pi05_libero_finetuned_v044"}


def _load(kind, **kw):
    try:
        return pi.load(CHECKPOINTS[kind], **kw)
    except Exception as e:  # checkpoint not in the local cache
        pytest.skip(f"cannot load {CHECKPOINTS[kind]}: {e}")


@pytest.fixture(scope="module", params=["pi0", "pi05"])
def shallow(request, tmp_path_factory):
    policy = _load(request.param, n_layers=2)
    with scratch_dir(tmp_path_factory, request.param) as out:  # ~5 GB each
        art = ggmlc.compile(pi.program_from_policy(policy), out_dir=out)
        yield request.param, policy, art


def test_stages_reproduce_lerobot(shallow):
    """The two stage functions plus the driver's Euler loop are LeRobot's sample_actions, and
    dropping the masked empty camera changes nothing."""
    kind, policy, art = shallow
    images, img_masks, tokens, masks, state, noise = pi.example_inputs(policy.config, empty_cameras=True)
    args = (images, img_masks, tokens, masks) + (() if kind == "pi05" else (state,))
    with torch.no_grad():
        want = policy.model.sample_actions(*args, noise=noise).numpy()
    got = np.load(art.out_dir / "reference_outputs.npz")["x"]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5 * max(1.0, np.abs(want).max()))


def test_pi_parity(shallow, backend):
    _, _, art = shallow
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**ins)
        assert s.stats["stages"]["step"]["calls"] == 10
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), ACCEPT[backend])


def test_graph_is_lean(shallow):
    """K/V heads shared through mul_mat broadcasting (no [S, 8 heads, 256] repeat), exact GELU
    as one backend-aware op, the loop-invariant prefix-key work hoisted out of the loop."""
    from ggmlc.cli import _read_gguf

    _, _, art = shallow
    program, _, _ = _read_gguf(str(art.gguf))
    for stage in program["stages"]:
        kv_repeats = [n for n in stage["nodes"] if n["op"] == "repeat" and n["ne"][0] == 256 and n["ne"][2] == 8]
        assert not kv_repeats, stage["name"]
        assert not any(n["op"] == "tanh" for n in stage["nodes"])
    stages = art.report.data["stages"]
    assert stages["prefix"]["ops"]["gelu_tanh"] > 0 and "step.invariant" in stages


@pytest.mark.skipif(not os.environ.get("GGMLC_BIG"), reason="full-size pi0: ~4 min and ~13 GB of RAM (GGMLC_BIG=1)")
def test_pi0_full_size_bf16(tmp_path, backend):
    from pathlib import Path

    policy = _load("pi0")
    art = ggmlc.compile(pi.program_from_policy(policy), out_dir=tmp_path,
                        dtype_policy=Path(pi.__file__).parent / "policy-bf16.yaml")
    del policy
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**ins)
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), 1e-2)
