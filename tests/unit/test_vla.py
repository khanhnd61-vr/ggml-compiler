"""The `vla` serving manifest: construction, validation, and the round trip through a GGUF."""

import json

import pytest
import torch
from torch import nn

import ggml_compiler
from ggml_compiler import vla


def _manifest():
    return vla.manifest(
        images=[vla.image("front", 48, 64, mean=vla.IMAGENET_MEAN, std=vla.IMAGENET_STD),
                vla.image("wrist", 48, 64, mean=vla.PM1_MEAN, std=vla.PM1_STD, fit="resize", mask="wrist_mask")],
        lang=vla.lang("tokens", 8, mask="tokens_mask", pad_id=0),
        state=vla.state("state", mean=[0.0] * 6, std=[1.0] * 6),
        noise=vla.noise("noise"),
        actions=vla.actions("actions", chunk=10, mean=[0.0] * 6, std=[1.0] * 6),
    )


def test_manifest_shape():
    m = _manifest()
    assert m["version"] == vla.VERSION
    assert [im["input"] for im in m["images"]] == ["front", "wrist"]
    assert m["images"][0]["fit"] == "exact" and "mask" not in m["images"][0]
    assert m["images"][1]["fit"] == "resize" and m["images"][1]["mask"] == "wrist_mask"
    assert m["state"] == {"input": "state", "dim": 6, "real_dim": 6, "norm": "mean_std",
                          "mean": [0.0] * 6, "std": [1.0] * 6, "eps": 1e-8}
    assert m["actions"]["chunk"] == 10 and m["actions"]["output"] == "actions"
    assert vla.input_names(m) == ["front", "wrist", "wrist_mask", "tokens", "tokens_mask", "state", "noise"]
    json.dumps(m)  # plain JSON


def test_norm_variants():
    s = vla.state("s", min=[0.0, 0.0], max=[1.0, 2.0])
    assert s["norm"] == "min_max" and s["dim"] == 2
    assert vla.state("s", dim=4)["norm"] == "none"
    assert vla.state("s", dim=8, real_dim=6, mean=[0.0] * 6, std=[1.0] * 6)["real_dim"] == 6
    with pytest.raises(ValueError, match="not both"):
        vla.state("s", mean=[0.0], std=[1.0], min=[0.0], max=[1.0])
    with pytest.raises(ValueError, match="go together"):
        vla.state("s", mean=[0.0])
    with pytest.raises(ValueError, match="expected 2"):
        vla.actions("a", chunk=3, dim=2, mean=[0.0] * 3, std=[1.0] * 3)
    with pytest.raises(ValueError, match="dim is needed"):
        vla.state("s")


def test_image_checks():
    with pytest.raises(ValueError, match="fit"):
        vla.image("i", 4, 4, fit="pad")
    with pytest.raises(ValueError, match="expected 3"):
        vla.image("i", 4, 4, mean=[0.5])
    with pytest.raises(ValueError, match="positive"):
        vla.image("i", 0, 4)


def test_duplicate_and_check():
    with pytest.raises(ValueError, match="more than once"):
        vla.manifest(images=[vla.image("x", 4, 4)], state=vla.state("x", dim=2),
                     actions=vla.actions("a", chunk=1, dim=2))
    m = _manifest()
    vla.check(m, vla.input_names(m), ["actions"])
    with pytest.raises(ValueError, match="program lacks"):
        vla.check(m, vla.input_names(m)[:-1], ["actions"])
    with pytest.raises(ValueError, match="leaves unfilled"):
        vla.check(m, vla.input_names(m) + ["extra"], ["actions"])
    with pytest.raises(ValueError, match="not a program output"):
        vla.check(m, vla.input_names(m), ["x"])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3 * 4 * 4 + 2, 6 * 3)

    def forward(self, state, front):
        return self.lin(torch.cat([front.flatten(1), state], 1)).view(1, 6, 3)


def test_round_trip(tmp_path):
    m = vla.manifest(images=[vla.image("front", 4, 4)], state=vla.state("state", dim=2),
                     actions=vla.actions("actions", chunk=6, dim=3))
    prog = ggml_compiler.Program.single(Tiny().eval(), (torch.zeros(1, 2), torch.zeros(1, 3, 4, 4)),
                                        input_names=["state", "front"], output_names=["actions"],
                                        metadata={"vla": m})
    vla.check(m, prog.driver.inputs, prog.driver.outputs)
    art = ggml_compiler.compile(prog, out_dir=tmp_path)
    with ggml_compiler.Session(art.gguf, backend="cpu") as s:
        assert s.metadata["vla"] == m


def test_check_rejects_a_program_the_manifest_does_not_cover():
    m = vla.manifest(images=[vla.image("front", 4, 4)], actions=vla.actions("actions", chunk=6, dim=3))
    prog = ggml_compiler.Program.single(Tiny().eval(), (torch.zeros(1, 2), torch.zeros(1, 3, 4, 4)),
                                        input_names=["state", "front"], output_names=["actions"],
                                        metadata={"vla": m})
    with pytest.raises(ValueError, match="unfilled"):
        ggml_compiler.compile(prog, out_dir="unused")
