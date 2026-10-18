"""Optimization levels and flags: what each level rewrites, and that every level stays correct."""

import json

import gguf
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import assert_close

import ggmlc
from ggmlc.options import resolve
from ggmlc.runtime import Session

TOL = {"cpu": 1e-5, "cuda": 2e-3}


def _repeat_kv(h, n):  # transformers' repeat_kv
    b, kv, s, d = h.shape
    return h[:, :, None].expand(b, kv, n, s, d).reshape(b, kv * n, s, d)


class Block(nn.Module):
    """Something for every O1 / O2 rewrite: conv + batch norm, a seq-first batched linear,
    grouped-query eager attention with head size 64, tanh-GELU, and an identity (x * 1)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.q, self.k, self.v = nn.Linear(64, 256), nn.Linear(64, 128), nn.Linear(64, 128)
        self.up, self.down = nn.Linear(256, 128), nn.Linear(128, 64)
        with torch.no_grad():
            self.bn.running_mean.uniform_(-0.5, 0.5)
            self.bn.running_var.uniform_(0.5, 2.0)

    def forward(self, img):
        x = self.bn(self.conv(img)).flatten(2).permute(2, 0, 1)  # seq-first [16, 1, 64]
        x = x.transpose(0, 1) * 1.0  # [1, 16, 64]
        q = self.q(x).view(1, 16, 4, 64).transpose(1, 2)
        k = self.k(x).view(1, 16, 2, 64).transpose(1, 2)
        v = self.v(x).view(1, 16, 2, 64).transpose(1, 2)
        w = (q @ _repeat_kv(k, 2).transpose(2, 3) * 0.125).softmax(-1)
        a = (w @ _repeat_kv(v, 2)).transpose(1, 2).reshape(1, 16, 256)
        return x + self.down(F.gelu(self.up(a), approximate="tanh"))


@pytest.fixture(scope="module")
def block():
    torch.manual_seed(0)
    return Block().eval(), torch.randn(1, 3, 4, 4)


def _compile(block, out, **kw):
    m, img = block
    return ggmlc.compile(m, (img,), out, input_names=["img"], output_names=["y"], **kw)


def _program_json(art) -> str:
    r = gguf.GGUFReader(art.gguf)
    f = r.fields["ggmlc.program"]
    return bytes(f.parts[f.data[0]]).decode()


def test_levels_nest():
    sets = [resolve(level).on for level in ggmlc.LEVELS]
    assert sets[0] == frozenset()
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    assert {"flash_attention", "fast_gelu"} <= sets[2] - sets[1]
    assert resolve(precision="exact") == resolve("O1") == resolve()
    assert resolve(precision="fast") == resolve("O2")
    assert "flash_attention" not in resolve("O2", attention="exact")
    assert "flash_attention" in resolve("O1", flags={"flash_attention": True})
    with pytest.raises(ValueError, match="unknown flag"):
        resolve("O1", flags={"no_such_flag": True})
    with pytest.raises(ValueError, match="not both"):
        resolve("O2", precision="exact")
    with pytest.raises(ValueError, match="opt must be"):
        resolve("O4")


@pytest.fixture(scope="module")
def arts(block, tmp_path_factory):
    out = {lv: _compile(block, tmp_path_factory.mktemp(lv), opt=lv) for lv in ggmlc.LEVELS}
    out["default"] = _compile(block, tmp_path_factory.mktemp("default"))
    out["exact"] = _compile(block, tmp_path_factory.mktemp("exact"), precision="exact")
    out["fast"] = _compile(block, tmp_path_factory.mktemp("fast"), precision="fast")
    return out


def test_aliases_are_levels(arts):
    assert _program_json(arts["default"]) == _program_json(arts["O1"]) == _program_json(arts["exact"])
    assert _program_json(arts["fast"]) == _program_json(arts["O2"])


def test_what_each_level_rewrites(arts):
    ops = {lv: arts[lv].report.data["stages"]["forward"]["ops"] for lv in ggmlc.LEVELS}
    assert ops["O0"].get("repeat") and "repeat" not in ops["O1"]  # shared_kv_heads
    assert ops["O0"]["mul"] > ops["O1"]["mul"]  # batch norm's scale and x * 1 are gone
    assert "flash_attn" in ops["O2"] and "flash_attn" not in ops["O1"]
    assert "gelu" in ops["O2"] and "gelu" not in ops["O1"]  # fast_gelu; O1 keeps the exact gelu_tanh


@pytest.mark.parametrize("level", ggmlc.LEVELS)
def test_every_level_is_correct(arts, block, level, backend):
    m, img = block
    with torch.no_grad():
        ref = m(img).numpy()
    with Session(arts[level].gguf, backend=backend) as s:
        out = s.run(img=img)
    assert_close(out, {"y": ref}, TOL[backend] if level in ("O0", "O1") else 5e-3)


def test_options_are_recorded(arts):
    meta = arts["O2"].report.data["options"]
    assert meta["level"] == "O2" and meta["flags"]["flash_attention"] and meta["specialized"] == {}
    r = gguf.GGUFReader(arts["O2"].gguf)
    f = r.fields["ggmlc.options"]
    assert json.loads(bytes(f.parts[f.data[0]]).decode()) == meta


class InvariantWork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_proj = nn.Linear(16, 4)
        self.mix = nn.Linear(4, 4)

    def step(self, x, t, ctx):
        k = torch.tanh(self.ctx_proj(ctx))  # loop-invariant
        return x + 0.1 * self.mix(x * k) * t


def _loop_program(**kw):
    torch.manual_seed(0)
    m = InvariantWork().eval()
    return ggmlc.Program(
        root=m,
        stages={"step": ggmlc.Stage(m.step, (torch.randn(2, 4), torch.ones(1), torch.randn(2, 16)))},
        driver=ggmlc.Driver(inputs=["ctx", "noise"],
                            body=[ggmlc.loop(3, carry={"x": "noise"}, scalars={"t": [1, 2, 3]},
                                             body=[ggmlc.call("step", ["x", "t", "ctx"], ["x"])])],
                            outputs=["x"]),
        **kw)


def test_flags_override_the_level(tmp_path, backend):
    art = ggmlc.compile(_loop_program(), out_dir=tmp_path / "a", flags={"hoist_invariants": False})
    assert set(art.report.data["stages"]) == {"step"}
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**dict(np.load(art.out_dir / "example_inputs.npz")))
    assert_close(out, dict(np.load(art.out_dir / "reference_outputs.npz")), TOL[backend])
    assert set(ggmlc.compile(_loop_program(), out_dir=tmp_path / "b").report.data["stages"]) == {"step", "step.invariant"}


def test_specializations_are_recorded(tmp_path):
    art = ggmlc.compile(_loop_program(specialized={"prompt_length": "12 tokens"}), out_dir=tmp_path)
    assert art.report.data["options"]["specialized"] == {"prompt_length": "12 tokens"}
