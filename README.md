# ggmlc

ggmlc compiles PyTorch and ONNX models into [ggml](https://github.com/ggml-org/ggml) programs that
run on CPU and CUDA without Python. It targets LLMs, vision-language models and the robot policies
built on them.

## Installation

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), CMake ≥ 3.22, a C++17 compiler, and
the CUDA toolkit for GPU builds.

```bash
git clone <repo> ggml-compiler && cd ggml-compiler

uv sync --all-extras              # compiler, Python bindings, ONNX and ModelOpt frontends
uv sync --no-group parity         # without the LeRobot models used by the examples and tests

# Runtime. The first configure fetches ggml (llama.cpp tag b11223) into .deps/.
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# With CUDA (the same library also runs the CPU backend):
cmake -S runtime -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda -j

uv run pytest -q -m "not slow"    # unit tests, on CPU and (if present) CUDA
```

`ggmlc.Session` and the CLI load `build-cuda/libggmlc_rt.so`, or else `build/libggmlc_rt.so`, from
this checkout. Set `GGMLC_RT` to point at a runtime elsewhere.

## Quick start

Compile a model, with example inputs that fix every shape, to ggml C++:

```python
import torch, ggmlc

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True), num_layers=4).eval()
x = torch.randn(1, 64, 256)                        # batch 1, 64 tokens

ggmlc.compile(model, (x,), "out/encoder", input_names=["x"], output_names=["y"],
              emit=("gguf", "cpp"), name="encoder")
```

```
out/encoder/
  model.gguf                 weights + compiled program
  cpp/encoder.cpp            one ggml graph builder per stage
  cpp/encoder.h              encoder_load(): the runtime's C API with these builders
  cpp/main.cpp               encoder-check: these builders vs the runtime, bit for bit
  cpp/CMakeLists.txt
  report.json, example_inputs.npz, reference_outputs.npz
```

`encoder.cpp` (excerpt) is plain ggml, with weights named after the module tree:

```cpp
// Stage 'forward': 212 nodes.
int build_forward(ggmlc_stage_build * B) {
    ggml_context * ctx = ggmlc_build_ctx(B);
    ggml_tensor * in0 = ggmlc_build_param(B, 0);
    ggml_tensor * layers_0_self_attn_in_proj_weight = ggmlc_build_weight(B, "w.0");
    // ...
    ggml_tensor * t2 = ggml_mul_mat(ctx, layers_0_self_attn_in_proj_weight, t1);
    ggml_tensor * t4 = ggml_add(ctx, t3, layers_0_self_attn_in_proj_bias);
    // ...
    ggml_tensor * t24 = ggml_mul_mat(ctx, t22, t21);
    ggml_tensor * t25 = ggml_soft_max_ext(ctx, t24, nullptr, 0.176776692f, 0.0f);
    // ...
    ggml_tensor * t209 = ggml_norm(ctx, t208, 9.99999975e-06f);
    ggml_tensor * t210 = ggml_mul(ctx, t209, layers_3_norm2_weight);
    ggml_tensor * t211 = ggml_add(ctx, t210, layers_3_norm2_bias);
    ggmlc_build_output(B, 0, t211);
    return 0;
}
```

In the generated file each line also names its PyTorch module and source line.

Build it and check it against the runtime:

```bash
cmake -S out/encoder/cpp -B out/encoder/build -DGGMLC_RUNTIME_LIB=$PWD/build-cuda/libggmlc_rt.so
cmake --build out/encoder/build -j
out/encoder/build/encoder-check out/encoder/model.gguf out/encoder/example_inputs.bin cuda
# y    bit-identical  max|d| 0
```

The GGUF also runs without the C++, from Python or the shell:

```python
with ggmlc.Session("out/encoder/model.gguf", backend="cuda") as sess:   # "cpu" | "cuda" | "auto"
    y = sess.run(x=x)["y"]
```

```bash
uv run ggmlc verify out/encoder/model.gguf --backend cpu    # against the PyTorch outputs saved at compile time
```

Language models with a KV cache (`ggmlc.causal_lm`), sizes set at run time, multi-stage models
with loops, optimization levels, weight dtypes and quantization, ModelOpt and ONNX models, and the
C API are in [docs/USAGE.md](docs/USAGE.md). [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains
how the compiler and runtime work.

## Tested models

Checked on CPU and CUDA against the reference implementations listed:

| Model | Checkpoint | Compiled with | Compared with |
|---|---|---|---|
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | `ggmlc.causal_lm` | transformers, llama.cpp |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | `ggmlc.causal_lm` | transformers, llama.cpp |
| SmolLM2 1.7B Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `ggmlc.causal_lm` | transformers, llama.cpp |
| SmolVLM2 500M Instruct | `HuggingFaceTB/SmolVLM2-500M-Instruct` | `ggmlc.causal_lm` | transformers, llama.cpp |
| Qwen3-VL 2B Instruct | `Qwen/Qwen3-VL-2B-Instruct` | `ggmlc.causal_lm` | transformers, llama.cpp |
| ACT | `khanhnd61/act_so101-multi-task-clean` | `examples/act`, also through ONNX | LeRobot, onnxruntime, vla.cpp |
| SmolVLA | `khanhnd61/smolvla_so101-multi-task-clean` | `examples/smolvla` | LeRobot, vla.cpp |
| pi0 | `lerobot/pi0_libero_finetuned_v044` | `examples/pi` | LeRobot, vla.cpp |
| pi0.5 | `lerobot/pi05_libero_finetuned_v044` | `examples/pi` | LeRobot, vla.cpp |
| GR00T N1.7 | `nvidia/GR00T-N1.7-LIBERO` (libero_object) | `examples/groot` | LeRobot, vla.cpp |
| Diffusion Policy | `lerobot/diffusion_pusht` | `examples/diffusion` | LeRobot |

## License

[Apache License 2.0](LICENSE).
