"""torch.export frontend: lowering of each supported ATen op, end to end through the runtime."""

import math

import numpy as np

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import check_module

import ggml_compiler
from ggml_compiler.api import Program
from ggml_compiler.frontend.torch_export import export_program


class Fn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *xs):
        return self.fn(*xs)


def r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


CASES = {
    "scalars": (lambda x: (x * 2 + 1, 1 - x, x / 4, 3 / (x.abs() + 1), x - 0.5), (r(3, 5),)),
    "broadcast_both_ways": (lambda x, y: (x + y, x * y, x - y, y - x), (r(3, 1, 4), r(5, 1, seed=1))),
    "activations": (lambda x: (F.relu(x), F.gelu(x), F.gelu(x, approximate="tanh"), F.silu(x), torch.tanh(x),
                               torch.sigmoid(x), torch.exp(x), torch.sin(x), torch.cos(x), x.abs(), -x), (r(4, 8),)),
    "pow_sqrt_clamp": (lambda x: (x.pow(2), (x.abs() + 1).sqrt(), x.clamp(-0.5, 0.5), x.clamp(min=0.0)), (r(4, 8),)),
    "softmax_any_dim": (lambda x: (x.softmax(-1), x.softmax(1), F.softmax(x, dim=0)), (r(3, 4, 5),)),
    "reductions": (lambda x: (x.mean(-1), x.mean(-1, keepdim=True), x.sum(-1)), (r(3, 4, 5),)),
    "layout": (lambda x: (x.transpose(1, 2).reshape(3, 20), x.permute(2, 0, 1).contiguous(), x.view(12, 5),
                          x.unsqueeze(0).squeeze(0), x.flatten(1), x[:, 1:3], x[..., -2:], x.select(1, 2)),
               (r(3, 4, 5),)),
    "cat_expand": (lambda x, y: (torch.cat([x, y], 1), torch.cat([x, x, y], 1), y[:, :1].expand(3, 4, 5)),
                   (r(3, 4, 5), r(3, 2, 5, seed=1))),
    "matmul_activations": (lambda q, k: (q @ k.transpose(-1, -2), torch.bmm(q, k.transpose(1, 2)), q[0] @ k[0].T),
                           (r(2, 6, 8), r(2, 7, 8, seed=1))),
    "sdpa": (lambda q, k, v: (F.scaled_dot_product_attention(q, k, v),
                              F.scaled_dot_product_attention(q, k, v, is_causal=True),
                              F.scaled_dot_product_attention(q, k, v, scale=0.2)),
             (r(1, 2, 6, 8), r(1, 2, 6, 8, seed=1), r(1, 2, 6, 8, seed=2))),
    "sdpa_gqa": (lambda q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
                 (r(1, 4, 5, 8), r(1, 2, 7, 8, seed=1), r(1, 2, 7, 8, seed=2))),
    "constants_folded": (lambda x: x + torch.arange(5.0) * math.pi + torch.ones(4, 5).cumsum(0), (r(3, 4, 5),)),
    "mish_softplus": (lambda x: (F.mish(x * 8), F.softplus(x * 30)), (r(4, 8),)),
    "six_dim_input": (lambda x: ((x.reshape(2, 3, 4, 5) * 2)[:, :, 1:3], x.sum(-1)), (r(1, 2, 1, 3, 4, 5),)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_ops(case, backend, tmp_path):
    fn, inputs = CASES[case]
    check_module(Fn(fn), inputs, backend, tmp_path)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 16), nn.ReLU())
        self.head = nn.Linear(16, 4, bias=False)

    def forward(self, x):
        return self.head(self.net(x))


def test_mlp(backend, tmp_path):
    torch.manual_seed(0)
    check_module(MLP().eval(), (r(2, 5, 32),), backend, tmp_path)


class UNet1dBits(nn.Module):
    """Diffusion Policy's U-Net pieces: 1-D convs, group norm, Mish, transposed 1-D convs (kernel a
    multiple of the stride, not a multiple, with output padding), and 2-D group norm."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(6, 16, 5, padding=2), nn.GroupNorm(8, 16), nn.Mish())
        self.down = nn.Conv1d(16, 16, 3, 2, 1)
        self.up = nn.ConvTranspose1d(16, 16, 4, 2, 1)
        self.up_odd = nn.ConvTranspose1d(16, 8, 3, 2)
        self.up_pad = nn.ConvTranspose1d(8, 8, 5, stride=3, padding=2, output_padding=1)
        self.img = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU())

    def forward(self, x, img):
        h = self.up(self.down(self.conv(x)))
        return h, self.up_pad(self.up_odd(h)), self.img(img)


def test_unet1d_bits(backend, tmp_path):
    torch.manual_seed(0)
    check_module(UNet1dBits().eval(), (r(2, 6, 16), r(1, 3, 8, 8, seed=1)), backend, tmp_path)


def test_tied_weights_stored_once():
    lin = nn.Linear(8, 8)
    m = Fn(lambda x: lin(lin(x)))
    m.lin = lin
    prog, _ = export_program(Program.single(m, (r(2, 8),)))
    assert len({id(w) for w in prog.weights.values()}) == 2  # weight and bias


def test_buffers_are_constants():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("table", torch.arange(4.0))

        def forward(self, x):
            return x * self.table.exp()

    prog, _ = export_program(Program.single(M(), (r(3, 4),)))
    stage = prog.stages["forward"]
    assert [n.op for n in stage.nodes] == ["mul"]  # exp(table) folded at compile time
    assert prog.weights == {}


def test_missing_lowering_lists_every_op(tmp_path):
    m = Fn(lambda x: torch.erfinv(x.clamp(-0.5, 0.5)) + torch.atan(x) + torch.erfinv(x * 0.1))
    with pytest.raises(ggml_compiler.CompileError) as e:
        ggml_compiler.compile(m, (r(3, 4),), tmp_path)
    msg = str(e.value)
    assert "aten.erfinv.default" in msg and "aten.atan.default" in msg
    assert "(+1 more)" in msg  # erfinv appears twice, reported once
    assert "test_frontend.py" in msg  # points at the user's source line


def test_user_lowering(backend, tmp_path):
    from ggml_compiler.lower.registry import LOWERINGS

    @ggml_compiler.lowering(torch.ops.aten.mish.default)
    def mish(b, x):
        return b.mul(x, b.tanh(b.log(b.add_scalar(b.exp(x), 1.0))))

    try:
        check_module(Fn(F.mish), (r(4, 8),), backend, tmp_path)
    finally:
        LOWERINGS.pop("aten.mish.default", None)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=2, dilation=2, bias=False)
        self.c3 = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        y = F.max_pool2d(F.relu(self.bn(self.c1(x))), 3, 2, 1)
        return self.c3(F.relu(self.c2(y))), F.avg_pool2d(y, 2)


@pytest.mark.parametrize("batch", [1, 2])
def test_conv_bn_pool(batch, backend, tmp_path):
    torch.manual_seed(0)
    net = ConvNet().eval()
    with torch.no_grad():  # non-trivial running stats
        net.bn.running_mean.uniform_(-1, 1)
        net.bn.running_var.uniform_(0.5, 2)
    check_module(net, (r(batch, 3, 32, 40),), backend, tmp_path)


@pytest.mark.parametrize("cross", [False, True])
def test_multihead_attention_seq_first(cross, backend, tmp_path):
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(64, 8).eval()  # packed q/k/v, seq-first: 5-D intermediates
    if cross:
        m = Fn(lambda q, kv: mha(q, kv, kv, need_weights=False)[0])
        m.mha = mha
        check_module(m, (r(10, 1, 64), r(7, 1, 64, seed=1)), backend, tmp_path)
    else:
        m = Fn(lambda x: mha(x, x, x, need_weights=False)[0])
        m.mha = mha
        check_module(m, (r(10, 1, 64),), backend, tmp_path)


def test_stack_of_rows_becomes_slices(tmp_path):
    def fn(a, x):
        rows = [a] + list(x) + list(x * 2)
        return torch.stack(rows, 0)

    art = ggml_compiler.compile(Fn(fn), (r(1, 16), r(5, 1, 16, seed=1)), tmp_path)
    assert art.report.data["stages"]["forward"]["ops"].get("concat", 0) == 2
    check_module(Fn(fn), (r(1, 16), r(5, 1, 16, seed=1)), "cpu", tmp_path / "check")


def test_weight_only_ops_are_folded():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(6, 4))

        def forward(self, x):
            return x @ (self.w * 2).t()  # w * 2 and its transpose: compile time

    from ggml_compiler.passes import run_pipeline

    m = M()
    prog, _ = export_program(Program.single(m, (r(3, 4),)))
    run_pipeline(prog)
    stage = prog.stages["forward"]
    assert [n.op for n in stage.nodes] == ["mul_mat"]
    (w,) = stage.leaves()  # mul_mat(a, x) = x @ a^T, so the folded weight is w * 2 itself
    assert w.fqn.startswith("w:") and torch.allclose(w.data, m.w.detach() * 2)


def test_empty_then_fill_halves(backend, tmp_path):
    def rope_like(x, c, s):
        h = x.shape[-1] // 2
        x1, x2 = x.split(h, dim=-1)
        res = torch.empty_like(x)
        res[..., :h] = x1 * c - x2 * s
        res[..., h:] = x2 * c + x1 * s
        return res

    inputs = (r(1, 6, 2, 8), r(1, 6, 1, 4, seed=1), r(1, 6, 1, 4, seed=2))
    art = ggml_compiler.compile(Fn(rope_like), inputs, tmp_path)
    assert art.report.data["consts"]["count"] == 0  # no uninitialized buffer baked in
    assert art.report.data["stages"]["forward"]["ops"]["concat"] == 1
    check_module(Fn(rope_like), inputs, backend, tmp_path / "check")


class EagerAttention(nn.Module):
    """SigLIP / HF-style eager attention: matmul, scale, (mask), softmax, matmul."""

    def __init__(self, heads=4, kv_heads=4):
        super().__init__()
        self.h, self.kvh = heads, kv_heads

    def forward(self, q, k, v, mask):
        rep = self.h // self.kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        w = torch.matmul(q, k.transpose(2, 3)) * 0.125
        w = (w + mask).softmax(-1)
        return torch.matmul(w, v).transpose(1, 2)


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_flash_attention_fusion(kv_heads, backend, tmp_path):
    from helpers import TOL  # noqa: F401
    from ggml_compiler.runtime import Session

    m = EagerAttention(4, kv_heads)
    inputs = (r(1, 4, 9, 64), r(1, kv_heads, 9, 64, seed=1), r(1, kv_heads, 9, 64, seed=2),
              torch.full((9, 9), float("-inf")).triu(1))
    art = ggml_compiler.compile(m, inputs, tmp_path, attention="flash")
    ops = art.report.data["stages"]["forward"]["ops"]
    assert ops.get("flash_attn") == 1 and "soft_max" not in ops
    with Session(art.gguf, backend=backend) as s:
        out = s.run(*inputs)
    from helpers import assert_close

    assert_close(out, {"out": m(*inputs).numpy()}, 5e-3)  # K/V/mask read in f16


def test_dtype_policy(backend, tmp_path):
    from ggml_compiler.runtime import Session

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(40, 64)
            self.a = nn.Linear(64, 256)
            self.b = nn.Linear(256, 64)
            self.c = nn.Linear(64, 48)  # rows of 64: too short for the 256-wide K-quant blocks
            self.norm = nn.LayerNorm(64)

        def forward(self, tok):
            return self.c(self.norm(self.b(F.relu(self.a(self.emb(tok))))))

    torch.manual_seed(0)
    m = M().eval()
    tok = torch.tensor([[1, 5, 9, 39]])
    policy = ggml_compiler.DTypePolicy(rules=[("emb.*", "f16"), ("a.weight", "q8_0"), ("b.weight", "q4_k"),
                                              ("c.weight", "q4_k"), ("norm.*", "q8_0")], fallback=["q8_0"])
    art = ggml_compiler.compile(m, (tok,), tmp_path, dtype_policy=policy)
    t = art.report.data["weights"]["tensors"]
    assert t["emb.weight"]["dtype"] == "f16" and t["a.weight"]["dtype"] == "q8_0"
    assert t["b.weight"]["dtype"] == "q4_k"
    assert t["c.weight"]["dtype"] == "q8_0" and "fallback" in t["c.weight"]["note"]
    assert t["norm.weight"]["dtype"] == "f32" and "not allowed" in t["norm.weight"]["note"]
    assert art.report.data["weights"]["stored_bytes"] < art.report.data["weights"]["bytes"] / 2
    # Each stored weight decodes back to the original within its type's error.
    import gguf
    import json

    reader = gguf.GGUFReader(art.gguf)
    f = reader.fields["ggml_compiler.tensor_names"]
    fqn = json.loads(bytes(f.parts[f.data[0]]).decode())
    bound = {"emb.weight": 1e-3, "a.weight": 1e-2, "b.weight": 0.15, "c.weight": 1e-2}  # 4-bit: ~1/16 of a block
    for t in reader.tensors:
        if fqn.get(t.name) in bound:
            ref_w = dict(m.named_parameters())[fqn[t.name]].detach().numpy()
            back = gguf.quants.dequantize(t.data, t.tensor_type).reshape(ref_w.shape)
            assert np.abs(back - ref_w).max() <= bound[fqn[t.name]] * np.abs(ref_w).max(), fqn[t.name]
    with Session(art.gguf, backend=backend) as s:
        out = s.run(tok=tok)["out"]
    ref = m(tok).detach().numpy()
    assert np.abs(out - ref).max() < 0.15 * np.abs(ref).max()  # q4_K on random weights: rough


def test_report_has_quantization_errors(tmp_path):
    m = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64)).eval()
    art = ggml_compiler.compile(m, (r(2, 64),), tmp_path, dtype_policy={"rules": [{"match": "0.weight",
                                                                                   "dtype": "q8_0"},
                                                                           {"match": "1.weight", "dtype": "q4_0"}]})
    t = art.report.data["weights"]["tensors"]
    assert 0 < t["0.weight"]["rel_err"] < 1e-2 < t["1.weight"]["rel_err"] < 0.2
    assert "rel_err" not in t["0.bias"]


def test_strided_slices(backend, tmp_path):
    check_module(Fn(lambda x: (x[..., 0::2], x[..., 1::2], x[:, 1:7:3], x[::2])), (r(4, 8, 6),), backend, tmp_path)


def test_stack_through_five_dims(backend, tmp_path):
    # ACT's 2-D sine embedding: stack((x[..., 0::2].sin(), x[..., 1::2].cos()), -1) on 4-D input.
    fn = lambda x: torch.stack((x[..., 0::2].sin(), x[..., 1::2].cos()), dim=-1).flatten(3)  # noqa: E731
    check_module(Fn(fn), (r(1, 3, 5, 8),), backend, tmp_path)


def test_parity_metric_fails_on_nan():
    from ggml_compiler.verify.parity import compare

    (d,) = compare({"y": np.array([1.0, np.nan])}, {"y": np.array([1.0, 2.0])})
    assert not d.ok(1e9)


def test_flash_attention_fully_masked_rows(backend, tmp_path):
    """Padding queries see no keys: PyTorch's finfo.min mask gives them a uniform softmax. The f16
    mask must stay finite so flash attention does too, rather than produce NaN."""
    from ggml_compiler.runtime import Session

    def attn(q, k, v, keep):
        w = torch.matmul(q, k.transpose(-1, -2)) * 0.125
        w = torch.where(keep[:, None], w, torch.finfo(torch.float32).min).softmax(-1)
        return torch.matmul(w, v)

    keep = torch.ones(1, 6, 6, dtype=torch.bool)
    keep[:, 4:, :] = False  # rows 4 and 5 attend to nothing
    inputs = (r(1, 4, 6, 64), r(1, 4, 6, 64, seed=1), r(1, 4, 6, 64, seed=2), keep)
    art = ggml_compiler.compile(Fn(attn), inputs, tmp_path, precision="fast")
    assert art.report.data["stages"]["forward"]["ops"].get("flash_attn") == 1
    with Session(art.gguf, backend=backend) as s:
        out = s.run(*inputs)["out"]
    assert np.isfinite(out).all()
    from helpers import assert_close

    assert_close({"out": out}, {"out": attn(*inputs).numpy()}, 5e-3)


def _repeat_kv(h, n):  # transformers' repeat_kv
    b, kv, s, d = h.shape
    return h[:, :, None].expand(b, kv, n, s, d).reshape(b, kv * n, s, d)


def _eager_gqa(q, k, v):
    n = q.shape[1] // k.shape[1]
    w = (q @ _repeat_kv(k, n).transpose(2, 3) * 0.3).softmax(-1)
    return w @ _repeat_kv(v, n)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_repeat_kv_is_broadcast(kv_heads, backend, tmp_path):
    """HF eager attention's repeated K/V heads become ggml's mul_mat batch broadcast: no copy."""
    q, k, v = r(1, 4, 5, 8), r(1, kv_heads, 7, 8, seed=1), r(1, kv_heads, 7, 8, seed=2)
    check_module(Fn(_eager_gqa), (q, k, v), backend, tmp_path)
    art = ggml_compiler.compile(Fn(_eager_gqa), (q, k, v), tmp_path / "ops")
    assert "repeat" not in art.report.data["stages"]["forward"]["ops"]
