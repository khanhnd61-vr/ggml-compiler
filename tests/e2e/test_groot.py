"""LeRobot GR00T N1.7 (nvidia/GR00T-N1.7-LIBERO, libero_object, embodiment libero_sim) as ggmlc
backbone + encode + step x4.

The stage functions reproduce LeRobot's GR00TN17.get_action bit for bit. Parity runs on the real
weights with the language model cut to 4 layers, the DiT to 4 blocks and the VL self-attention to
1 (full vision tower). Full-size results are in docs/USAGE.md's examples table.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import assert_close, load_example, scratch_dir

import ggmlc
from ggmlc.runtime import Session

pytestmark = pytest.mark.slow
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}

if importlib.util.find_spec("lerobot") is None or importlib.util.find_spec("tree") is None:
    pytest.skip("LeRobot with dm-tree not installed (uv sync)", allow_module_level=True)
gr = load_example("groot")
if not Path(gr.CHECKPOINT, "config.json").exists():
    pytest.skip(f"GR00T N1.7 checkpoint not at {gr.CHECKPOINT} (GROOT_CHECKPOINT)", allow_module_level=True)


@pytest.fixture(scope="module")
def shallow(tmp_path_factory):
    with scratch_dir(tmp_path_factory, "groot") as out:  # ~4 GB
        try:
            model = gr.load(n_layers={"llm": 4, "dit": 4, "vl_self_attention": 1})
            art = ggmlc.compile(gr.program_from_model(model), out_dir=out)
        except OSError as e:  # the Cosmos-Reason2 processor files are gated on the Hub
            pytest.skip(f"GR00T processor not available: {e}")
        yield model, art


def test_stages_reproduce_lerobot(shallow):
    model, _ = shallow
    enc, state, _ = gr.example_inputs(model)
    emb = gr.embodiment_id(gr.CHECKPOINT, gr.EMBODIMENT)
    s = gr.Stages(model, enc, emb)
    torch.manual_seed(7)
    noise = torch.randn(1, model.action_head.config.action_horizon, model.action_head.action_dim)
    with torch.no_grad():
        vl, sf = s.encode(s.backbone(enc["pixel_values"], enc["input_ids"], enc["attention_mask"]), state)
        x = noise
        for t in s.timesteps():
            x = s.step(x, torch.full((1,), t), vl, sf, enc["attention_mask"])
        torch.manual_seed(7)  # get_action draws the same noise
        want = model.get_action(dict(enc, state=state, embodiment_id=torch.tensor([emb])))["action_pred"]
    np.testing.assert_array_equal(x.numpy(), want.numpy())


def test_groot_parity(shallow, backend):
    _, art = shallow
    ins = dict(np.load(art.out_dir / "example_inputs.npz"))
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**ins)
        assert s.stats["stages"]["step"]["calls"] == 4
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), ACCEPT[backend])
