"""C++ emission: generated graph builders match the runtime bit for bit."""

import shutil

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import build_generated, check_generated
from test_driver import _flow_program

import ggmlc

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")


def r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class KitchenSink(nn.Module):
    """Most IR ops in one graph: conv/bn/pool, embedding with int inputs, masks and where,
    min/max, layer norm, eager attention (flash with precision='fast'), slices and concat."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.emb = nn.Embedding(30, 32)
        self.norm = nn.LayerNorm(32)
        self.proj = nn.Linear(32, 256)  # 4 heads of 64: a head size flash attention takes everywhere

    def forward(self, img, tok, mask):
        f = F.max_pool2d(F.relu(self.bn(self.conv(img))), 2)  # [1, 8, 4, 4]
        f = f.flatten(2).transpose(1, 2).repeat(1, 1, 4)  # [1, 16, 32]
        e = self.emb(tok)  # [1, 6, 32]
        x = self.norm(torch.cat([f, e], 1))  # [1, 22, 32]
        q = self.proj(x).view(1, 22, 4, 64).transpose(1, 2)
        w = torch.matmul(q, q.transpose(-1, -2)) * 0.125
        keep = torch.cat([torch.ones(1, 16, dtype=torch.bool), mask], 1)
        w = torch.where(keep[:, None, None, :], w, torch.finfo(torch.float32).min).softmax(-1)
        a = torch.matmul(w, q).transpose(1, 2).reshape(1, 22, 256)
        pos = torch.cumsum(mask, 1) - 1
        return F.gelu(a, approximate="tanh")[:, 2:9], a.amax(-1) - a.amin(-1), pos.float()


@pytest.fixture(scope="module")
def kitchen_sink(tmp_path_factory):
    torch.manual_seed(0)
    m = KitchenSink().eval()
    with torch.no_grad():
        m.bn.running_mean.uniform_(-1, 1)
        m.bn.running_var.uniform_(0.5, 2)
    inputs = (r(1, 3, 16, 16), torch.tensor([[1, 4, 9, 16, 25, 0]]), torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool))
    arts = {}
    for precision in ("exact", "fast"):
        out = tmp_path_factory.mktemp(f"sink-{precision}")
        art = ggmlc.compile(m, inputs, out, emit=("gguf", "cpp"), precision=precision, name="sink")
        arts[precision] = (art, build_generated(out / "cpp", out / "build"))
    return arts


@pytest.mark.parametrize("precision", ["exact", "fast"])
def test_generated_matches_runtime(kitchen_sink, precision, backend):
    art, binary = kitchen_sink[precision]
    ops = art.report.data["stages"]["forward"]["ops"]
    assert {"conv_2d", "pool_2d", "get_rows", "argmax", "concat", "view", "cumsum"} <= set(ops)
    assert ("flash_attn" in ops) == (precision == "fast")
    check_generated(art, binary, backend)


def test_generated_loop_program(backend, tmp_path):
    _, _, _, program = _flow_program()
    art = ggmlc.compile(program, out_dir=tmp_path, emit=("gguf", "cpp"), name="flow")
    src = (tmp_path / "cpp" / "flow.cpp").read_text()
    assert "build_encode" in src and "build_step" in src  # one builder per stage
    assert "enc_weight" in src  # weights named after the module tree
    out = check_generated(art, build_generated(tmp_path / "cpp", tmp_path / "build"), backend)
    assert out.count("bit-identical") == 1
