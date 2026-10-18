"""M1 exit check: a 2-layer MLP and a transformer encoder block match PyTorch on CPU and CUDA,
at the parity tolerances (1e-4 CPU, 1e-2 CUDA)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import check_module

ACCEPT = {"cpu": 1e-4, "cuda": 1e-2}


class TwoLayerMLP(nn.Module):
    def __init__(self, d_in=64, d_hidden=256, d_out=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    """Pre-norm block with separate q/k/v projections, as in most VLA backbones."""

    def __init__(self, d=128, heads=4, mlp=512):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)

        def split(t):
            return t.view(b, s, self.heads, d // self.heads).transpose(1, 2)

        a = F.scaled_dot_product_attention(split(self.q(h)), split(self.k(h)), split(self.v(h)))
        x = x + self.o(a.transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


def _seeded(module):
    torch.manual_seed(0)
    for p in module.parameters():
        nn.init.normal_(p, std=0.05)
    return module.eval()


def _x(*shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(1))


def test_two_layer_mlp(backend, tmp_path):
    check_module(_seeded(TwoLayerMLP()), (_x(4, 10, 64),), backend, tmp_path, tol=ACCEPT[backend])


def test_encoder_block(backend, tmp_path):
    check_module(_seeded(EncoderBlock()), (_x(1, 50, 128),), backend, tmp_path, tol=ACCEPT[backend])


@pytest.mark.parametrize("norm_first", [False, True])
def test_torch_transformer_encoder_layer(norm_first, backend, tmp_path):
    layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=norm_first)
    check_module(_seeded(layer), (_x(1, 50, 128),), backend, tmp_path, tol=ACCEPT[backend])


class TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TwoLayerMLP(64, 128, 64)
        self.norm = nn.LayerNorm(64)  # read by both stages
        self.head = nn.Linear(64, 8)

    def encode(self, x, extra):
        h = self.norm(self.enc(x))
        return {"h": h, "pooled": h.mean(-1) + extra["bias"]}

    def decode(self, feats):
        return self.head(self.norm(feats["h"])) * 2


def test_two_stage_program_shares_weights_and_values(backend, tmp_path):
    import ggmlc
    from helpers import assert_close
    from ggmlc.runtime import Session

    m = _seeded(TwoPart())
    x, extra = _x(1, 10, 64), {"bias": _x(1, 10)}
    with torch.no_grad():
        feats = m.encode(x, extra)
    program = ggmlc.Program(
        root=m,
        stages={"encode": ggmlc.Stage(m.encode, (x, extra)), "decode": ggmlc.Stage(m.decode, (feats,))},
        driver=ggmlc.Driver(
            inputs=["x", "extra"],
            body=[ggmlc.call("encode", ["x", "extra"], ["feats"]), ggmlc.call("decode", ["feats"], ["y"])],
            outputs=["y", "feats"],
        ),
    )
    art = ggmlc.compile(program, out_dir=tmp_path)
    assert {w.fqn for w in art.program.weights.values()} >= {"norm.weight", "head.weight", "enc.fc1.weight"}
    assert art.report.data["weights"]["count"] == 8  # norm stored once though both stages read it

    with Session(art.gguf, backend=backend) as s:
        assert [t.name for t in s.inputs] == ["x", "extra.bias"]
        out = s.run(x=x, **{"extra.bias": extra["bias"]})
        assert s.stats["n_calls"] == 2
    with torch.no_grad():
        ref = {"y": m.decode(m.encode(x, extra)), "feats.h": feats["h"], "feats.pooled": feats["pooled"]}
    assert_close(out, {k: v.numpy() for k, v in ref.items()}, ACCEPT[backend])
