"""M3 exit check: LeRobot SmolVLA (prefix + 10 Euler steps) matches PyTorch on CPU and CUDA."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import assert_close, build_generated, check_generated, load_example, scratch_dir

import ggmlc
import ggmlc.runtime
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("transformers") is None:
    pytest.skip("LeRobot with SmolVLA not installed (uv sync --group parity)", allow_module_level=True)
smolvla_example = load_example("smolvla")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    try:
        program = smolvla_example.program()
    except Exception as e:  # checkpoint not in the local cache
        pytest.skip(f"cannot load {smolvla_example.CHECKPOINT}: {e}")
    with scratch_dir(tmp_path_factory, "smolvla") as out:
        art = ggmlc.compile(program, out_dir=out)
        with np.load(out / "example_inputs.npz") as z:
            inputs = {k: z[k] for k in z.files}
        with np.load(out / "reference_outputs.npz") as z:
            refs = {k: z[k] for k in z.files}
        yield art, inputs, refs


def test_smolvla_parity(compiled, backend):
    art, inputs, refs = compiled
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**inputs)
        assert s.stats["stages"]["step"]["calls"] == 10
    assert_close(out, refs, ACCEPT[backend])


def test_smolvla_program_shape(compiled):
    art, inputs, _ = compiled
    # The expert's projections of the cached prefix K/V do not change across denoising steps: they
    # are hoisted into a stage that runs once before the loop.
    assert set(art.program.stages) == {"prefix", "step", "step.invariant"}
    assert art.report.data["stages"]["step.invariant"]["ops"]["mul_mat"] >= 16
    assert inputs["lang_tokens"].dtype == np.int32
    assert art.report.data["consts"]["bytes"] < 20e6  # zero masks dropped, repeats stored once


def test_smolvla_parity_at_o0(compiled, backend, tmp_path_factory):
    """Without the optional rewrites (and so without hoisting) SmolVLA still compiles and matches."""
    _, inputs, refs = compiled
    with scratch_dir(tmp_path_factory, "smolvla-o0") as out:
        art = ggmlc.compile(smolvla_example.program(), out_dir=out, opt="O0")
        assert set(art.program.stages) == {"prefix", "step"}
        with Session(art.gguf, backend=backend) as s:
            got = s.run(**inputs)
    assert_close(got, refs, ACCEPT[backend])


POLICIES = Path(__file__).resolve().parents[2] / "examples" / "smolvla"


@pytest.fixture(scope="module", params=["q8_0", "q4k-f16"])
def quantized(request, tmp_path_factory):
    try:
        program = smolvla_example.program()
    except Exception as e:
        pytest.skip(f"cannot load {smolvla_example.CHECKPOINT}: {e}")
    with scratch_dir(tmp_path_factory, f"smolvla-{request.param}") as out:
        art = ggmlc.compile(program, out_dir=out, dtype_policy=POLICIES / f"policy-{request.param}.yaml")
        with np.load(out / "example_inputs.npz") as z:
            inputs = {k: z[k] for k in z.files}
        with np.load(out / "reference_outputs.npz") as z:
            refs = {k: z[k] for k in z.files}
        yield art, inputs, refs


def test_smolvla_quantized(quantized, backend):
    """M5: quantized SmolVLA runs on both backends; weight and output errors are reported."""
    art, inputs, refs = quantized
    tensors = art.report.data["weights"]["tensors"]
    assert any(t["dtype"] in ("q8_0", "q4_k") for t in tensors.values())
    assert all("rel_err" in t for t in tensors.values() if t["dtype"] != "f32")
    assert art.report.data["weights"]["stored_bytes"] < 0.4 * art.report.data["weights"]["bytes"]
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**inputs)
    assert_close(out, refs, 0.1)  # vs fp32 PyTorch: Q8_0 lands near 3.5e-2, the Q4_K mix near 1.7e-2


@pytest.mark.parametrize("cfg", ["INT8_DEFAULT_CFG", "INT4_BLOCKWISE_WEIGHT_ONLY_CFG"])
def test_smolvla_modelopt_expert(cfg, tmp_path):
    """M5: SmolVLA with a ModelOpt-quantized action expert imports with a per-tensor report."""
    pytest.importorskip("modelopt.torch.quantization")
    try:
        policy = smolvla_example.load()
    except Exception as e:
        pytest.skip(f"cannot load {smolvla_example.CHECKPOINT}: {e}")
    smolvla_example.quantize_expert(policy, cfg)
    art = ggmlc.compile(smolvla_example.program_from_policy(policy), out_dir=tmp_path, prequant=ggmlc.ModelOpt())
    pre = art.report.data["prequant"]
    weights = pre["weights"]
    assert len(weights) >= 16 * 6 and all("lm_expert" in k for k in weights)
    exact = [w for w in weights.values() if w["stored"] in ("q8_0", "q4_0")]
    assert exact and all(w["rel_err_vs_fake_quant"] < 1e-3 for w in exact)  # on-grid re-encoding
    assert all(w["stored"] in ("q8_0", "q4_0", "f16") for w in weights.values())  # f16 where rows of 720 do not fit
    assert bool(pre["dropped_activation_quantizers"]) == (cfg == "INT8_DEFAULT_CFG")
    assert pre["output_error_vs_fake_quant"]["x"]["scaled"] < 0.05


def test_smolvla_generated_cpp_is_bit_identical(tmp_path):
    """M6: the generated C++ for all three stages builds and matches the runtime bit for bit."""
    try:
        program = smolvla_example.program()
    except Exception as e:
        pytest.skip(f"cannot load {smolvla_example.CHECKPOINT}: {e}")
    art = ggmlc.compile(program, out_dir=tmp_path, emit=("gguf", "cpp"), name="smolvla", precision="fast",
                        dtype_policy=POLICIES / "policy-cuda.yaml")
    src = (tmp_path / "cpp" / "smolvla.cpp").read_text()
    assert all(f"build_{s}" in src for s in ("prefix", "step", "step_invariant"))
    binary = build_generated(tmp_path / "cpp", tmp_path / "build")
    for backend in ["cpu"] + (["cuda"] if ggmlc.runtime.has_backend("cuda") else []):
        check_generated(art, binary, backend)
