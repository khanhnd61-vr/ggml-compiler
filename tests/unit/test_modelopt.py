"""ModelOpt import: fake-quant weights stored as ggml quant types, with errors reported."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

import ggml_compiler
from ggml_compiler.runtime import Session

mtq = pytest.importorskip("modelopt.torch.quantization")


def _quantized(cfg_name):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 32)).eval()
    data = [torch.randn(8, 64) for _ in range(8)]
    fp32 = copy.deepcopy(m)
    q = mtq.quantize(m, getattr(mtq, cfg_name), lambda model: [model(x) for x in data])
    return fp32, q.eval(), data[0]


@pytest.mark.parametrize("cfg,stored,weight_err", [("INT8_DEFAULT_CFG", "q8_0", 1e-3),
                                                   ("INT4_BLOCKWISE_WEIGHT_ONLY_CFG", "q4_0", 1e-3)])
def test_modelopt_import(cfg, stored, weight_err, backend, tmp_path):
    fp32, q, x = _quantized(cfg)
    before = {k: v.clone() for k, v in q.state_dict().items()}
    art = ggml_compiler.compile(q, (x,), tmp_path, prequant=ggml_compiler.ModelOpt())
    pre = art.report.data["prequant"]

    assert set(pre["weights"]) == {"0.weight", "2.weight", "4.weight"}
    for w in pre["weights"].values():
        assert w["stored"] == stored and 0 <= w["rel_err_vs_fake_quant"] < weight_err
    assert bool(pre["dropped_activation_quantizers"]) == (cfg == "INT8_DEFAULT_CFG")  # W8A8 vs weight-only
    assert pre["output_error_vs_fake_quant"]["out"]["scaled"] < 0.05
    # The user's model is untouched: weights restored, quantizers back on.
    assert all(torch.equal(before[k], v) for k, v in q.state_dict().items())
    assert q[0].weight_quantizer.is_enabled

    with Session(art.gguf, backend=backend) as s:
        out = s.run(x)["out"]
    with torch.no_grad():
        ref_fq, ref_fp32 = q(x).numpy(), fp32(x).numpy()
    scale = np.abs(ref_fq).max()
    assert np.abs(out - ref_fq).max() < 0.05 * scale
    assert np.abs(out - ref_fp32).max() < 0.15 * scale  # and close to the unquantized model


def test_modelopt_weights_keep_their_type_over_policy(tmp_path):
    _, q, x = _quantized("INT4_BLOCKWISE_WEIGHT_ONLY_CFG")
    art = ggml_compiler.compile(q, (x,), tmp_path, prequant=ggml_compiler.ModelOpt(int4="q4_1"),
                                dtype_policy={"rules": [{"match": "*", "dtype": "f16"}]})
    tensors = art.report.data["weights"]["tensors"]
    assert tensors["2.weight"]["dtype"] == "q4_1"  # imported mapping wins
    assert tensors["2.bias"]["dtype"] == "f32"  # biases: float, as always


def test_grid_encoders_round_trip():
    import gguf

    from ggml_compiler.quant.encode import encode_on_grid

    rng = np.random.default_rng(0)
    for storage, lo, hi, qt in [("q8_0", -127, 127, "Q8_0"), ("q4_0", -8, 7, "Q4_0")]:
        s = np.repeat(rng.uniform(0.01, 0.1, (6, 2)).astype(np.float32), 64, axis=1)  # groups of 64
        w = rng.integers(lo, hi + 1, (6, 128)).astype(np.float32) * s
        raw = encode_on_grid(w, s, storage)
        back = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType[qt]).reshape(w.shape)
        assert np.abs(back - w).max() <= 2**-11 * np.abs(w).max()  # only the fp16 scale rounds
        assert encode_on_grid(w + 1e-3, s, storage) is None  # off the grid: not encodable exactly
