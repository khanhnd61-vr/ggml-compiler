"""ONNX frontend: models exported to ONNX compile and match onnxruntime."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import TOL, assert_close

import ggmlc
from ggmlc.runtime import Session

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _export(m, inputs, path, names, dynamic=None):
    torch.onnx.export(m.eval(), inputs, str(path), input_names=names, dynamo=True, external_data=False,
                      dynamic_shapes=dynamic)
    return path


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 4, 3, padding=1)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.bn(self.c1(x))), 2)
        return self.c2(y), F.avg_pool2d(y, 2)


class Block(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h = h
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(approximate="tanh"), nn.Linear(128, d))

    def forward(self, x, keep):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln(x)).split(d, -1)
        heads = lambda t: t.view(b, s, self.h, d // self.h).transpose(1, 2)  # noqa: E731
        w = heads(q) @ heads(k).transpose(-1, -2) / 4.0
        w = torch.where(keep[:, None, None, :], w, -1e9).softmax(-1)
        x = x + self.out((w @ heads(v)).transpose(1, 2).reshape(b, s, d))
        return x + self.mlp(x)


CASES = {
    "mlp": (lambda: nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)), lambda: (r(4, 16),), ["x"]),
    "conv": (ConvNet, lambda: (r(1, 3, 32, 32),), ["img"]),
    "block": (Block, lambda: (r(1, 10, 64), torch.tensor([[True] * 7 + [False] * 3])), ["x", "keep"]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_onnx_matches_onnxruntime(case, backend, tmp_path):
    torch.manual_seed(0)
    make, inputs, names = CASES[case]
    m = make().eval()
    if case == "conv":
        with torch.no_grad():
            m.bn.running_mean.uniform_(-1, 1)
            m.bn.running_var.uniform_(0.5, 2)
    ins = inputs()
    path = _export(m, ins, tmp_path / f"{case}.onnx", names)
    art = ggmlc.compile(path, dict(zip(names, ins)), tmp_path / "out")
    assert art.report.data["stages"]["forward"]["nodes"] > 0
    with np.load(tmp_path / "out" / "reference_outputs.npz") as z:
        refs = {k: z[k] for k in z.files}  # onnxruntime's outputs
    with Session(art.gguf, backend=backend) as s:
        out = s.run(**dict(zip(names, ins)))
    assert_close(out, refs, TOL[backend] * 10)
    with torch.no_grad():  # and the same as PyTorch itself
        want = m(*ins)
    want = [want] if isinstance(want, torch.Tensor) else list(want)
    assert_close(dict(zip(refs, out.values())), dict(zip(refs, (w.numpy() for w in want))), TOL[backend] * 10)


def test_onnx_missing_ops_are_listed(tmp_path):
    class Odd(nn.Module):
        def forward(self, x):
            return torch.erf(x) + torch.atan(x)

    path = _export(Odd(), (r(3, 4),), tmp_path / "odd.onnx", ["x"])
    with pytest.raises(ggmlc.CompileError) as e:
        ggmlc.compile(path, None, tmp_path / "out")
    assert "Erf" in str(e.value) and "Atan" in str(e.value)


def test_onnx_dynamic_input_needs_a_shape(tmp_path):
    m = nn.Linear(8, 4)
    dim = torch.export.Dim("batch", min=1, max=64)
    path = _export(m, (r(2, 8),), tmp_path / "dyn.onnx", ["x"], dynamic={"input": {0: dim}})
    with pytest.raises(ggmlc.CompileError, match="dynamic shape"):
        ggmlc.compile(path, None, tmp_path / "out")
    art = ggmlc.compile(path, None, tmp_path / "out2", input_shapes={"x": [5, 8]})
    assert art.report.data["inputs"]["x"]["shape"] == [5, 8]
