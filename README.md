# ggml-compiler

<p align="center">
  <img src="https://raw.githubusercontent.com/khanhnd61-vr/ggml-compiler/main/assets/logo.png" alt="ggml-compiler" width="420">
</p>

ggml-compiler compiles PyTorch and ONNX models into [ggml](https://github.com/ggml-org/ggml)
programs that run on CPU and CUDA without Python. It targets LLMs, vision-language models and the
robot policies built on them.

<p align="center">
  <img src="assets/quick-start.png" alt="Side-by-side PyTorch Linear and ReLU implementation and its generated ggml C++ graph builder using ggml_mul_mat, ggml_add, and ggml_relu, in a macOS-style editor" width="800">
</p>

## Installation

### Using pip

Python 3.10+ on Linux x86-64 with AVX2. Includes the CPU runtime:

```bash
pip install ggml-compiler
pip install "ggml-compiler[llm]"  # with Transformers for LLMs and VLMs
```

### From source

For development and examples: Python 3.12, [uv](https://docs.astral.sh/uv/), CMake 3.22+, and a C++17 compiler.

```bash
git clone https://github.com/khanhnd61-vr/ggml-compiler
cd ggml-compiler
uv sync --all-extras
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For CUDA, install the CUDA toolkit and add `-DGGML_CUDA=ON` to the CMake configure command.
Run the CLI with `uv run ggml-compiler`.

See [Installation options](docs/INSTALLATION.md) for custom builds, optional dependencies, and runtime selection.

## Documentation

| Document | Contents |
|---|---|
| [examples/README.md](examples/README.md) | Examples of compiling LLMs, VLMs, and VLAs. |
| [docs/USAGE.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/USAGE.md) | API, CLI, codegen, quantization and other advance optimizations. |
| [docs/MODELS.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/MODELS.md) | List of tested models with their accuracy and speed against their references. |
| [docs/EXPORT.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/EXPORT.md) | How to write export script to compile a Pytorch model to ggml C++ source. |
| [docs/ARCHITECTURE.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/ARCHITECTURE.md) | How it works: the IR, frontends, passes, GGUF format and runtime internals. |
| [docs/OPS.md](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/docs/OPS.md) | List of supported and un-supported Pytorch/ONNX operations. |

## License

[Apache License 2.0](https://github.com/khanhnd61-vr/ggml-compiler/blob/main/LICENSE).
