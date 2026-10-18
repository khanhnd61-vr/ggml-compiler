"""Symbolic dims: one artifact serves every size in a compiled range."""

import numpy as np
import pytest
import sympy
import torch
import torch.nn as nn
from helpers import assert_close, build_generated, check_generated
from torch.export import Dim

import ggmlc
from ggmlc.ir import sym
from ggmlc.ir.program import CompileError
from ggmlc.runtime import GgmlcRuntimeError, Session

TOL = {"cpu": 1e-5, "cuda": 3e-3}


def test_dim_expressions():
    sym.RANGES.clear()
    s, k = sym.define("s", 2, 512), sym.define("k", 1, 8)
    for d in (s, 256 * k, s + 3, 2 * s * k, sympy.floor((s + 63) / 64) * 64, sympy.Max(s, 7)):
        assert sym.from_json(sym.to_json(sym.norm(d))) == sym.norm(d)
    assert sym.bounds(256 * k - 1) == (255, 2047)
    assert sym.definitely(s + 3 <= 515) and not sym.definitely(s < 512)
    assert sym.solve([((1, s, 256 * k), (1, 7, 512))]) == {"s": 7, "k": 2}
    with pytest.raises(ValueError, match="outside"):
        sym.solve([((s,), (600,))])
    assert sym.broadcast((1, s, 64), (s, 1)) == (1, s, 64)


def _encoder():
    torch.manual_seed(0)
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), num_layers=2).eval()


def _single(model, example, dyn, **kw):
    return ggmlc.Program(root=model, stages={"forward": ggmlc.Stage(model, example, dynamic_shapes=dyn)},
                         driver=ggmlc.Driver(inputs=["x"], body=[ggmlc.call("forward", ["x"], ["y"])], outputs=["y"]),
                         **kw)


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    model = _encoder()
    prog = _single(model, (torch.randn(1, 10, 64),), ({1: Dim("seq", min=2, max=96)},))
    out = tmp_path_factory.mktemp("sym-encoder")
    art = ggmlc.compile(prog, out_dir=out, emit=("gguf", "cpp"), name="enc")
    return model, art, build_generated(out / "cpp", out / "build")


def test_encoder_any_length(encoder, backend):
    model, art, _ = encoder
    assert art.report.data["inputs"]["x"]["shape"] == [1, "seq", 64]
    assert art.report.data["symbols"] == {"seq": {"min": 2, "max": 96}}
    with Session(art.gguf, backend=backend) as s:
        assert s.symbols == {"seq": (2, 96)}
        for n in (10, 2, 33, 96, 5):
            x = torch.randn(1, n, 64)
            with torch.no_grad():
                ref = model(x).numpy()
            out = s.run(x=x)
            assert s.bound() == {"seq": n}
            assert_close(out, {"y": ref}, TOL[backend])


def test_encoder_out_of_range(encoder):
    _, art, _ = encoder
    with Session(art.gguf, backend="cpu") as s:
        with pytest.raises(GgmlcRuntimeError, match="no seq in"):
            s.run(x=np.zeros((1, 97, 64), np.float32))
        with pytest.raises(GgmlcRuntimeError, match="cannot have shape"):
            s.run(x=np.zeros((1, 10, 63), np.float32))


def test_generated_cpp_is_bit_identical(encoder, backend):
    _, art, binary = encoder
    check_generated(art, binary, backend)


class CachedAttention(nn.Module):
    """Reads the first n_kv rows of a fixed-size K/V buffer, as a decode step reads its cache: two
    symbols (new tokens s, attended rows n_kv), a symbolic slice length, and a runtime mask."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.register_buffer("k", torch.randn(1, 4, 512, 16))
        self.register_buffer("v", torch.randn(1, 4, 512, 16))

    def forward(self, x, mask):
        n_kv = mask.shape[-1]
        q = self.q(x).view(1, x.shape[1], 4, 16).transpose(1, 2)
        k, v = self.k[:, :, :n_kv], self.v[:, :, :n_kv]
        w = (q @ k.transpose(-1, -2) * 0.25 + mask).softmax(-1)
        return (w @ v).transpose(1, 2).reshape(1, x.shape[1], 64)


def test_two_symbols_and_a_symbolic_slice(tmp_path, backend):
    torch.manual_seed(0)
    m = CachedAttention().eval()
    x, mask = torch.randn(1, 3, 64), torch.randn(1, 1, 3, 256)
    blocks = Dim("blocks", min=1, max=4)  # torch raises the min to 2: n_kv in {256, 384, 512}
    prog = ggmlc.Program(
        root=m, stages={"forward": ggmlc.Stage(m, (x, mask), dynamic_shapes=({1: Dim("s", min=2, max=64)},
                                                                              {2: Dim("s", min=2, max=64),
                                                                               3: 128 * blocks}))},
        driver=ggmlc.Driver(inputs=["x", "mask"], body=[ggmlc.call("forward", ["x", "mask"], ["y"])], outputs=["y"]))
    art = ggmlc.compile(prog, out_dir=tmp_path)
    assert art.report.data["inputs"]["mask"]["shape"] == [1, 1, "s", "128*blocks"]
    with Session(art.gguf, backend=backend) as s:
        for n, kv in ((3, 256), (2, 384), (17, 512), (64, 256)):
            xi, mi = torch.randn(1, n, 64), torch.randn(1, 1, n, kv)
            with torch.no_grad():
                ref = m(xi, mi).numpy()
            assert_close(s.run(x=xi, mask=mi), {"y": ref}, TOL[backend])


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def encode(self, x):
        return torch.tanh(self.proj(x))

    def step(self, h, ctx, t):
        return h + 0.1 * t * self.proj(h * ctx)


def test_symbolic_values_cross_stages_and_loops(tmp_path, backend):
    torch.manual_seed(0)
    m = Step().eval()
    x = torch.randn(1, 6, 16)
    S = Dim("s", min=2, max=40)
    h = m.encode(x)
    prog = ggmlc.Program(
        root=m,
        stages={"encode": ggmlc.Stage(m.encode, (x,), dynamic_shapes=({1: S},)),
                "step": ggmlc.Stage(m.step, (h, h, torch.ones(1)), dynamic_shapes=({1: S}, {1: S}, None))},
        driver=ggmlc.Driver(inputs=["x"],
                            body=[ggmlc.call("encode", ["x"], ["ctx"]),
                                  ggmlc.loop(3, carry={"h": "ctx"}, scalars={"t": [1.0, 0.5, 0.25]},
                                             body=[ggmlc.call("step", ["h", "ctx", "t"], ["h"])])],
                            outputs=["h"]))
    art = ggmlc.compile(prog, out_dir=tmp_path)
    with Session(art.gguf, backend=backend) as s:
        for n in (6, 40, 2):
            xi = torch.randn(1, n, 16)
            with torch.no_grad():
                ctx = h = m.encode(xi)
                for t in (1.0, 0.5, 0.25):
                    h = m.step(h, ctx, torch.tensor([t]))
            assert_close(s.run(x=xi), {"h": h.numpy()}, TOL[backend])


def test_a_dim_needs_a_max(tmp_path):
    with pytest.raises(CompileError, match="needs a max"):
        ggmlc.compile(_single(_encoder(), (torch.randn(1, 10, 64),), ({1: Dim("seq")},)), out_dir=tmp_path)
