# Installation options

The [README](../README.md#installation) covers the common pip and checkout installs.
This guide covers optional dependencies, custom CPU/CUDA builds, and runtime selection.

## Requirements

- Python 3.10+ for the compiler; Python 3.12 for the checkout's LeRobot examples and tests.
- The Linux x86-64 release wheel includes a CPU runtime built for x86-64-v3
  (including AVX2, FMA, and F16C), with a manylinux_2_28 baseline (glibc 2.28+).
- Source builds require CMake 3.22+ and a C++17 compiler. CUDA builds also require
  the CUDA toolkit and a compatible NVIDIA driver.

CPU and CUDA are the tested backends. Other ggml backends are not validated by this project.

## Python environments and extras

For an isolated pip installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ggml-compiler
```

Choose extras for the frontends you use:

| Extra | Installs |
|---|---|
| `llm` | Transformers for `causal_lm`, LLMs, and VLMs |
| `onnx` | ONNX, ONNX Runtime, and ONNX Script |
| `modelopt` | NVIDIA ModelOpt for importing quantized models |

For example, install LLM and ONNX support together:

```bash
python -m pip install "ggml-compiler[llm,onnx]"
```

The compiler requires PyTorch 2.11 or newer. When installing into an existing model
environment, check its pinned dependencies first. `--no-deps` leaves those packages
untouched, but you must supply all required dependencies yourself; it does not make an
older PyTorch version compatible.

## Checkout dependency groups

From the repository root, choose one of these dependency sets:

```bash
uv sync --all-extras                    # all frontends, tests, and LeRobot references
uv sync --all-extras --no-group parity  # omit LeRobot and its reference dependencies
uv sync --extra llm --no-group parity  # LLM frontend and development tools
```

`dev` and `parity` are enabled by default. The `parity` group installs the compatible
LeRobot fork used by the examples and end-to-end tests. The checkout configures uv to
install CPU PyTorch wheels; ggml's CUDA runtime is built separately and does not require
a CUDA PyTorch wheel for inference.

An editable checkout installation does not build the C++ runtime. Build it explicitly
as described below, then invoke Python and the CLI through `uv run`.

## Building a release from source with pip

To build a CPU runtime tuned to the build machine instead of using the release wheel:

```bash
python -m pip install ggml-compiler --no-binary ggml-compiler \
  -C cmake.define.GGML_NATIVE=ON
```

For CUDA, choose the target GPU's compute capability (`86` below is an example):

```bash
python -m pip install ggml-compiler --no-binary ggml-compiler \
  -C cmake.define.GGML_CUDA=ON \
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=86
```

If the package is already installed, add `--force-reinstall`; add `--no-cache-dir` to
avoid reusing a cached wheel when changing build options. Extras such as `[llm]` can
also be used with these source builds.

## Building the checkout runtime

Keep CPU and CUDA builds in separate directories:

```bash
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j
```

```bash
cmake -S runtime -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j
```

The CUDA library also supports CPU execution. For multiple GPU architectures, use a
quoted list such as `-DCMAKE_CUDA_ARCHITECTURES="86;89"`. To select a CUDA installation
explicitly, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` when configuring a new
build directory.

For a CPU build intended for another machine, set `-DGGML_NATIVE=OFF` and choose ggml's
CPU instruction-set options for the target; disabling native tuning alone does not
guarantee support for every older CPU.

### Dependency sources

The first configure downloads llama.cpp tag `b11223` into `.deps/` and builds its ggml
subdirectory. CPU and CUDA builds share the downloaded source. To store it elsewhere,
add `-DGGML_COMPILER_DEPS_DIR=/path/to/deps`.

For a runtime build without a source download, provide an existing checkout of the
pinned llama.cpp release:

```bash
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_GGML=/path/to/llama.cpp-b11223
cmake --build build -j
```

Use the llama.cpp root, which contains both `ggml/` and `vendor/`. This only avoids the
CMake source download; Python packages and model checkpoints must be available separately.
`GGML_COMPILER_LLAMA_TAG` can override the pin for development, but a different ggml
version needs compatibility testing.

## Selecting the runtime library

Python sessions and the CLI search in this order:

1. The path in `GGML_COMPILER_RT`, when set.
2. `build-cuda/libggml_compiler_rt.so` in a source checkout.
3. `build/libggml_compiler_rt.so` in a source checkout.
4. The library bundled under the installed package's `lib/` directory.

To select a custom build, set the variable before starting Python:

```bash
export GGML_COMPILER_RT=/absolute/path/to/libggml_compiler_rt.so
```

The library is loaded once per Python process. Use a runtime built from the same
version as the Python package. The paths above are for Linux; the loader uses `.dylib`
on macOS and `.dll` on Windows.

## Checking the installation

```bash
python -m pip show ggml-compiler
ggml-compiler --help
python -c 'from ggml_compiler.runtime import library_path, version; print(library_path()); print(version())'
```

In a checkout, prefix the CLI and Python commands with `uv run`. If the runtime is
missing, build it or set `GGML_COMPILER_RT`. If CUDA is unavailable, check that the
selected library was built with `GGML_CUDA=ON` and that the driver can access the GPU.

For an end-to-end compile and CPU verification without downloading model weights,
follow the [encoder-block example](../examples/README.md#compile-and-check-an-example).
