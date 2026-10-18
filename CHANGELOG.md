# Changelog

## 0.1.0 (unreleased)

First release.

- Compiles a PyTorch model (through `torch.export`) or an ONNX model into a GGUF that holds the
  weights and the compiled op graph, run by one prebuilt runtime (`libggml_compiler_rt`, a C API
  with Python bindings) on the CPU and CUDA backends of ggml. Optionally emits readable C++ that
  builds the same ggml graphs and matches the runtime bit for bit.
- Multi-stage programs with a driver: calls, fixed-count loops with per-step scalars and slices,
  entry points, persistent state, symbolic dims bound at run time, and loop-invariant hoisting.
- `ggml_compiler.causal_lm` turns a Hugging Face causal LM or VLM into prefill, decode and
  on-device greedy generation over an in-place KV cache. `ggml_compiler.Generator` drives it.
- Optimization levels `O0` to `O3` with named flags; per-tensor weight types (ggml's float and
  quantized types) through a dtype policy; `llama_cpp_policy` to run the same weight bytes as a
  llama.cpp GGUF; import of NVIDIA ModelOpt fake-quantized models.
- Tested on Qwen3 0.6B and 1.7B, SmolLM2 1.7B, DeepSeek-R1-Distill-Qwen 1.5B, Gemma 4 E2B (language
  model), SmolVLM2 500M, Qwen3-VL 2B, and the LeRobot policies ACT, SmolVLA, pi0, pi0.5, GR00T
  N1.7 and Diffusion Policy: see [docs/MODELS.md](docs/MODELS.md).
- Linux x86-64 wheel with the CPU runtime; CUDA through a source build.
