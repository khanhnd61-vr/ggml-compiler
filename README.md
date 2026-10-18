# ggml-compiler

<p align="center">
  <img src="https://raw.githubusercontent.com/khanhnd61-vr/ggml-compiler/main/assets/logo.png" alt="ggml-compiler" width="420">
</p>

ggml-compiler compiles PyTorch and ONNX models into [ggml](https://github.com/ggml-org/ggml)
programs that run on CPU and CUDA without Python. It targets LLMs, vision-language models and the
robot policies built on them.

## Installation

```bash
pip install ggml-compiler            # Linux x86-64, Python 3.10 or newer; includes the CPU runtime
pip install "ggml-compiler[llm]"     # with transformers, for ggml_compiler.causal_lm
```

The wheel's runtime is built for x86-64 CPUs with AVX2 (most since 2015). For CUDA, or a runtime
tuned to your own CPU, build it from source instead. That needs CMake 3.22 or newer, a C++17
compiler and, for CUDA, the CUDA toolkit; the build downloads ggml (llama.cpp release b11223):

```bash
pip install ggml-compiler --no-binary ggml-compiler -C cmake.define.GGML_CUDA=ON
```

Set `GGML_COMPILER_RT` to make `ggml_compiler.Session` and the CLI load a runtime library built
elsewhere.

### From a checkout

For development, the examples and the tests (Python 3.12, which the LeRobot reference models need,
and [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/khanhnd61-vr/ggml-compiler && cd ggml-compiler

uv sync --all-extras      # compiler, Python bindings, ONNX and ModelOpt frontends
uv sync --no-group parity # without the LeRobot models used by the examples and tests

# Runtime. The first configure fetches ggml (llama.cpp tag b11223) into .deps/.
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# With CUDA (the same library also runs the CPU backend):
cmake -S runtime -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda -j
```

In a checkout, `ggml_compiler.Session` and the CLI load `build-cuda/libggml_compiler_rt.so`, or
else `build/libggml_compiler_rt.so`.

## Quick start

Compile a model, with example inputs that fix every shape, to ggml C++:

```python
import torch, ggml_compiler

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
    num_layers=4
  ).eval()
x = torch.randn(1, 64, 256)

ggml_compiler.compile(model, (x,), "out/encoder", input_names=["x"], output_names=["y"],
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

`encoder.cpp` is plain ggml, with weights named after the module tree:

```cpp
int build_forward(ggml_compiler_stage_build * B) {
    ggml_context * ctx = ggml_compiler_build_ctx(B);
    ggml_tensor * in0 = ggml_compiler_build_param(B, 0);
    ggml_tensor * layers_0_self_attn_in_proj_weight = ggml_compiler_build_weight(B, "w.0");
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
    ggml_compiler_build_output(B, 0, t211);
    return 0;
}
```

In the generated file each line also names its PyTorch module and source line.
Build it and check it against the runtime:

```bash
cmake -S out/encoder/cpp -B out/encoder/build \
      -DGGML_COMPILER_RUNTIME_LIB=$PWD/build-cuda/libggml_compiler_rt.so
cmake --build out/encoder/build -j
out/encoder/build/encoder-check out/encoder/model.gguf out/encoder/example_inputs.bin cuda
```

The GGUF also runs without the C++, from Python or the shell:

```python
with ggml_compiler.Session("out/encoder/model.gguf", backend="cuda") as sess:
    y = sess.run(x=x)["y"]
```

Verify against the PyTorch outputs saved at compile time:

```bash
uv run ggml-compiler verify out/encoder/model.gguf --backend cpu
```

Language models with a KV cache (`ggml_compiler.causal_lm`), sizes set at run time, multi-stage
models with loops, optimization levels, weight dtypes and quantization, ModelOpt and ONNX models,
and the C API are in
[docs/USAGE.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/USAGE.md).
[docs/ARCHITECTURE.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/ARCHITECTURE.md)
explains how the compiler and runtime work.
[docs/MODELS.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/MODELS.md) lists the
models ggml-compiler is tested on, with their accuracy and speed against PyTorch, llama.cpp and
vla.cpp.

## License

[Apache License 2.0](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/LICENSE).
