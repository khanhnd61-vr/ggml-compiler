"""M7 exit check: ACT exported to ONNX compiles and matches onnxruntime within the PyTorch path's
tolerances, and agrees with the PyTorch-path build."""

import importlib.util
import os

import numpy as np
import pytest
import torch
from helpers import assert_close, load_example

import ggml_compiler
from ggml_compiler.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("lerobot") is None or importlib.util.find_spec("onnxruntime") is None:
    pytest.skip("needs LeRobot and onnxruntime", allow_module_level=True)
act_example = load_example("act")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    try:
        program = act_example.program()
    except Exception as e:
        pytest.skip(f"cannot load {act_example.CHECKPOINT}: {e}")
    stage = program.stages["forward"]
    names = program.driver.inputs
    out = tmp_path_factory.mktemp("act-onnx")
    path = out / "act.onnx"
    torch.onnx.export(stage.fn, stage.example_inputs, str(path), input_names=names, output_names=["actions"],
                      dynamo=True, external_data=False)
    art = ggml_compiler.compile(path, dict(zip(names, stage.example_inputs)), out / "gguf")
    with np.load(out / "gguf" / "reference_outputs.npz") as z:
        ort_out = {k: z[k] for k in z.files}
    with torch.no_grad():
        torch_out = stage.fn(*stage.example_inputs).numpy()
    return art, dict(zip(names, stage.example_inputs)), ort_out, torch_out


def test_act_onnx_matches_onnxruntime(compiled, backend):
    art, inputs, ort_out, torch_out = compiled
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**inputs)
    assert_close(out, ort_out, ACCEPT[backend])
    assert_close(out, {"actions": torch_out}, ACCEPT[backend])  # and PyTorch itself


def test_act_onnx_graph_matches_torch_path(compiled):
    ops = compiled[0].report.data["stages"]["forward"]["ops"]
    assert ops["conv_2d"] == 42 and ops["pool_2d"] == 2
    assert ops.get("concat", 0) <= 4  # 1200 ONNX Slice/Squeeze/Unsqueeze rows merged back into slices
