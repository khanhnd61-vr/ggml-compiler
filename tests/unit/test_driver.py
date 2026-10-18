"""Driver loops and integer / boolean inputs."""

import numpy as np
import torch
import torch.nn as nn
from helpers import assert_close

import ggmlc
from ggmlc.runtime import Session


def r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class Flow(nn.Module):
    """A toy flow-matching sampler: encode a context once, then Euler-integrate a velocity field."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(8, 16)
        self.vel = nn.Sequential(nn.Linear(16 + 4 + 1, 32), nn.SiLU(), nn.Linear(32, 4))

    def encode(self, ctx):
        return {"h": torch.tanh(self.enc(ctx))}

    def step(self, x, t, feats):
        tt = t.expand(x.shape[0], 1)
        v = self.vel(torch.cat([feats["h"], x, tt], -1))
        return x - 0.1 * v


def _flow_program(n=10):
    torch.manual_seed(0)
    m = Flow().eval()
    ctx, noise = r(3, 8), r(3, 4, seed=1)
    with torch.no_grad():
        feats = m.encode(ctx)
    return m, ctx, noise, ggmlc.Program(
        root=m,
        stages={"encode": ggmlc.Stage(m.encode, (ctx,)), "step": ggmlc.Stage(m.step, (noise, torch.ones(1), feats))},
        driver=ggmlc.Driver(
            inputs=["ctx", "noise"],
            body=[
                ggmlc.call("encode", ["ctx"], ["feats"]),
                ggmlc.loop(n, carry={"x": "noise"}, scalars={"t": [1.0 - i / n for i in range(n)]},
                           body=[ggmlc.call("step", ["x", "t", "feats"], ["x"])]),
            ],
            outputs=["x"],
        ),
    )


def test_loop_matches_pytorch(backend, tmp_path):
    m, ctx, noise, program = _flow_program()
    art = ggmlc.compile(program, out_dir=tmp_path)
    with torch.no_grad():
        feats, x = m.encode(ctx), noise
        for i in range(10):
            x = m.step(x, torch.full((1,), 1.0 - i / 10), feats)
    ref = np.load(tmp_path / "reference_outputs.npz")["x"]
    np.testing.assert_allclose(ref, x.numpy(), rtol=0, atol=1e-6)  # compile-time reference = eager loop
    with Session(art.gguf, backend=backend) as s:
        out = s.run(ctx=ctx, noise=noise)
        assert s.stats["n_calls"] == 11 and s.stats["stages"]["step"]["calls"] == 10
        out2 = s.run(ctx=ctx, noise=noise)  # the carry restarts from noise on every run
    tol = {"cpu": 1e-5, "cuda": 2e-3}[backend]
    assert_close(out, {"x": x.numpy()}, tol)
    assert_close(out2, {"x": x.numpy()}, tol)


class Masked(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)

    def forward(self, tokens, mask):
        e = self.emb(tokens)
        pos = torch.cumsum(mask, 1) - 1
        att = (pos[:, None, :] <= pos[:, :, None]) & mask[:, None, :]
        scores = torch.where(att, e @ e.transpose(1, 2), torch.finfo(torch.float32).min)
        return scores.softmax(-1) @ e, pos.float(), (tokens > 20).float(), tokens.ne(3).float()


def test_int_and_bool_inputs(backend, tmp_path):
    torch.manual_seed(0)
    m = Masked().eval()
    tokens = torch.tensor([[5, 17, 42, 3, 0, 0, 9]])
    mask = torch.tensor([[True, True, True, True, False, False, True]])
    art = ggmlc.compile(m, (tokens, mask), tmp_path)
    with Session(art.gguf, backend=backend) as s:
        assert [(t.name, t.dtype) for t in s.inputs] == [("tokens", "i32"), ("mask", "f32")]
        out = s.run(tokens=tokens, mask=mask)
    with torch.no_grad():
        ref = m(tokens, mask)
    assert_close(out, {f"out.{i}": v.numpy() for i, v in enumerate(ref)}, {"cpu": 1e-5, "cuda": 2e-3}[backend])


class InvariantWork(nn.Module):
    """The step reads a projection of the context that never changes inside the loop."""

    def __init__(self):
        super().__init__()
        self.ctx_proj = nn.Linear(16, 4)
        self.mix = nn.Linear(4, 4)

    def step(self, x, t, ctx):
        k = torch.tanh(self.ctx_proj(ctx))  # loop-invariant
        return x + 0.1 * self.mix(x * k) * t


def test_loop_invariants_are_hoisted(backend, tmp_path):
    torch.manual_seed(0)
    m = InvariantWork().eval()
    ctx, noise = r(2, 16), r(2, 4, seed=1)
    program = ggmlc.Program(
        root=m,
        stages={"step": ggmlc.Stage(m.step, (noise, torch.ones(1), ctx))},
        driver=ggmlc.Driver(inputs=["ctx", "noise"],
                            body=[ggmlc.loop(5, carry={"x": "noise"}, scalars={"t": [1, 2, 3, 4, 5]},
                                             body=[ggmlc.call("step", ["x", "t", "ctx"], ["x"])])],
                            outputs=["x"]))
    art = ggmlc.compile(program, out_dir=tmp_path)
    stages = art.report.data["stages"]
    assert stages["step.invariant"]["ops"].get("tanh") == 1 and "tanh" not in stages["step"]["ops"]
    with Session(art.gguf, backend=backend) as s:
        out = s.run(ctx=ctx, noise=noise)
        assert s.stats["stages"]["step.invariant"]["calls"] == 1
    ref = np.load(tmp_path / "reference_outputs.npz")
    assert_close(out, {"x": ref["x"]}, {"cpu": 1e-5, "cuda": 2e-3}[backend])


class Noisy(nn.Module):
    """A DDPM-style sampler: every step adds its own slice of a noise table."""

    def __init__(self):
        super().__init__()
        self.eps = nn.Linear(4, 4)

    def step(self, x, z, c, sigma):
        return c * (x - self.eps(x)) + sigma * z


def test_loop_slices(backend, tmp_path):
    torch.manual_seed(0)
    m = Noisy().eval()
    n = 6
    noise, zs = r(2, 4), r(n, 2, 4, seed=1)
    c, sigma = [0.9 - 0.1 * i for i in range(n)], [0.5 / (i + 1) for i in range(n)]
    program = ggmlc.Program(
        root=m,
        # Every scalar example is the same tensor object: the export must still see two inputs.
        stages={"step": ggmlc.Stage(m.step, (noise, zs[0], *[torch.ones(1)] * 2))},
        driver=ggmlc.Driver(inputs=["noise", "zs"],
                            body=[ggmlc.loop(n, carry={"x": "noise"}, scalars={"c": c, "sigma": sigma},
                                             slices={"z": "zs"},
                                             body=[ggmlc.call("step", ["x", "z", "c", "sigma"], ["x"])])],
                            outputs=["x"],
                            examples={"zs": zs}))
    art = ggmlc.compile(program, out_dir=tmp_path)
    with torch.no_grad():
        x = noise
        for i in range(n):
            x = m.step(x, zs[i], torch.full((1,), c[i]), torch.full((1,), sigma[i]))
    np.testing.assert_allclose(np.load(tmp_path / "reference_outputs.npz")["x"], x.numpy(), rtol=0, atol=1e-6)
    with Session(art.gguf, backend=backend) as s:
        out = s.run(noise=noise, zs=zs)
        out2 = s.run(noise=noise, zs=zs.flip(0).contiguous())  # different noise, different result
    assert_close(out, {"x": x.numpy()}, {"cpu": 1e-5, "cuda": 2e-3}[backend])
    assert np.abs(out2["x"] - out["x"]).max() > 1e-3


def test_aliased_example_inputs(backend, tmp_path):
    """One tensor object given for two stage inputs still makes two runtime inputs."""
    one = torch.ones(3)
    m = nn.Module()
    program = ggmlc.Program(root=m, stages={"f": ggmlc.Stage(lambda a, b: a * 2 - b, (one, one))},
                            driver=ggmlc.Driver(inputs=["a", "b"], body=[ggmlc.call("f", ["a", "b"], ["y"])],
                                                outputs=["y"]))
    art = ggmlc.compile(program, out_dir=tmp_path)
    with Session(art.gguf, backend=backend) as s:
        out = s.run(a=np.full(3, 5.0, np.float32), b=np.full(3, 1.0, np.float32))
    np.testing.assert_allclose(out["y"], np.full(3, 9.0), atol=1e-6)
