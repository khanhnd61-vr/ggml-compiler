# Tested models

Every model here compiles from its published checkpoint and is checked on CPU and CUDA: against
its PyTorch implementation in f32, and against the hand-written engine for its kind of model
(llama.cpp for language and vision-language models, vla.cpp for robot policies) at the same weight
types.

| Model | Checkpoint | Compiled with | Compared with |
|---|---|---|---|
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| SmolLM2 1.7B Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| DeepSeek-R1-Distill-Qwen 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| Gemma 4 E2B (language model) | `google/gemma-4-E2B-it` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| SmolVLM2 500M Instruct | `HuggingFaceTB/SmolVLM2-500M-Instruct` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| Qwen3-VL 2B Instruct | `Qwen/Qwen3-VL-2B-Instruct` | `ggml_compiler.causal_lm` | transformers, llama.cpp |
| ACT | `khanhnd61/act_so101-multi-task-clean` | `examples/act`, also through ONNX | LeRobot, onnxruntime, vla.cpp |
| SmolVLA | `khanhnd61/smolvla_so101-multi-task-clean` | `examples/smolvla` | LeRobot, vla.cpp |
| pi0 | `lerobot/pi0_libero_finetuned_v044` | `examples/pi` | LeRobot, vla.cpp |
| pi0.5 | `lerobot/pi05_libero_finetuned_v044` | `examples/pi` | LeRobot, vla.cpp |
| GR00T N1.7 | `nvidia/GR00T-N1.7-LIBERO` (libero_object) | `examples/groot` | LeRobot, vla.cpp |
| Diffusion Policy | `lerobot/diffusion_pusht` | `examples/diffusion` | LeRobot |

- [How the checks work](#how-the-checks-work)
- [Language models](#language-models)
- [Vision-language models](#vision-language-models)
- [Robot policies](#robot-policies)

## How the checks work

**Machines.** Two, both with CUDA 12.8 and 6 CPU threads for every CPU measurement:

- a workstation, i5-12400F (6 performance cores) and an RTX 3060 12 GB: the robot policies and
  the Qwen3 and SmolLM2 language models;
- a server, i7-14700F and an RTX 3090 24 GB, CPU runs pinned to one thread per performance core:
  the vision-language models, DeepSeek-R1-Distill and Gemma 4.

**f32 parity.** The scaled error `max|out − ref| / max(1, max|ref|)` against PyTorch in f32, with
f32 weights and exact attention. It passes at 1e-4 on CPU and 1e-2 on CUDA, whose f32 matmuls run
as TF32. Language models also have to produce transformers' greedy tokens exactly, from both the
host decoding loop and the on-device one. The end-to-end tests in `tests/e2e` run these checks.

**Against llama.cpp** (b11223, the same ggml that ggml-compiler builds on). Both engines run the
same weight bytes: llama.cpp's converter writes a BF16 GGUF from the checkpoint, `llama-quantize`
makes F16, Q8_0 and Q4_K_M from it (no importance matrix), and `ggml_compiler.llm.llama_cpp_policy`
gives each ggml-compiler weight the type llama.cpp stored it as. ggml-compiler compiles at `O3`.

- Accuracy is measured against transformers in f32, per token position: the KL divergence from
  transformers' next-token distribution, and top-1 agreement (the same most likely token).
- Language models: the first 512 tokens of `bench/llama/text.txt` (the opening of *Alice in
  Wonderland*). Greedy is the number of tokens of transformers' 64-token greedy continuation each
  engine reproduces before its first difference. Speed is llama-bench's pp512 (a 512-token prompt)
  and tg128 (128 generated tokens) with its defaults (flash attention auto, f16 KV cache, `-ngl
  99` on CUDA), against `ggml_compiler.llm.bench`, which measures the same way. Scripts:
  `bench/llm_baseline.py` and `bench/llm_ggml_compiler.py`.
- Vision-language models: seven image prompts, each an image and "Describe this image in
  detail." through the model's own processor: llama.cpp's `test-1.jpeg` and six scikit-image
  sample photos (astronaut, coffee, chelsea, rocket, hubble_deep_field, immunohistochemistry).
  transformers answers each greedily (up to 256 tokens); KL and top-1 are over those answers,
  teacher-forced, pooled over every position of the seven. Greedy is the tokens matching each
  case's first 64, summed (of 448). llama.cpp runs its own vision encoder (from the mmproj GGUF)
  on the processor's pixels. Timings are on `test-1.jpeg`: the image encode, the whole prompt,
  and 64 generated tokens, a mean of 3 after a warmup. Scripts: `bench/vlm_baseline.py` and
  `bench/vlm_ggml_compiler.py`.

**Against vla.cpp** (`vla-bench`, commit 741ae39; ACT from its `pr-38` branch, 0b3d878): the
median of 50 runs after 5 warmups, on the same inputs.

In the comparison tables, llama.cpp's or vla.cpp's number is in parentheses, and x is
ggml-compiler's time over theirs (for a speed in tokens/s, their speed over ggml-compiler's): below
1, ggml-compiler is faster. For KL, x is ggml-compiler's KL over llama.cpp's: below 1, ggml-compiler
is closer to transformers.

## Language models

f32 parity: Qwen3-0.6B's last-token logits after the 512-token text are 9.0e-7 from transformers'
on CPU and 6.4e-4 on CUDA, and its 64 greedy tokens match exactly. Qwen3-1.7B, SmolLM2-1.7B,
DeepSeek-R1-Distill-Qwen-1.5B and Gemma 4 E2B pass the same checks.

Against llama.cpp, at llama.cpp's weight types, on the workstation:

| Model | Type | Backend | KL mean | top-1 | greedy /64 | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|---|---|
| Qwen3 0.6B | BF16 | CPU | 1.06e-04 (1.06e-04) | 99.6% (99.6%) | 64 (64) | 316 (335) x1.06 | 30.5 (32.0) x1.05 |
| Qwen3 0.6B | BF16 | CUDA | 1.01e-04 (1.03e-04) | 99.4% (99.4%) | 64 (64) | 11820 (11875) x1.00 | 175.2 (195.7) x1.12 |
| Qwen3 0.6B | F16 | CPU | 2.34e-06 (2.34e-06) | 100.0% (100.0%) | 64 (64) | 318 (337) x1.06 | 30.6 (32.2) x1.05 |
| Qwen3 0.6B | F16 | CUDA | 4.61e-04 (4.29e-04) | 98.6% (98.4%) | 64 (64) | 15750 (15773) x1.00 | 176.4 (197.9) x1.12 |
| Qwen3 0.6B | Q8_0 | CPU | 6.38e-03 (6.38e-03) | 95.1% (95.1%) | 26 (26) | 484 (524) x1.08 | 54.4 (59.2) x1.09 |
| Qwen3 0.6B | Q8_0 | CUDA | 5.95e-03 (5.85e-03) | 95.1% (95.1%) | 37 (37) | 15801 (15975) x1.01 | 278.0 (330.6) x1.19 |
| Qwen3 0.6B | Q4_K_M | CPU | 2.37e-01 (2.30e-01) | 73.0% (75.4%) | 1 (1) | 557 (595) x1.07 | 80.7 (89.1) x1.10 |
| Qwen3 0.6B | Q4_K_M | CUDA | 2.41e-01 (2.53e-01) | 72.7% (71.5%) | 1 (1) | 14930 (14556) x0.97 | 342.3 (374.5) x1.09 |
| Qwen3 1.7B | BF16 | CPU | 4.09e-04 (4.09e-04) | 99.4% (99.4%) | 27 (27) | 110 (115) x1.04 | 11.1 (11.3) x1.02 |
| Qwen3 1.7B | BF16 | CUDA | 4.30e-04 (6.29e-04) | 99.4% (99.4%) | 61 (61) | 5698 (5866) x1.03 | 86.7 (89.1) x1.03 |
| Qwen3 1.7B | F16 | CPU | 1.24e-05 (1.24e-05) | 99.8% (99.8%) | 64 (64) | 111 (115) x1.04 | 11.1 (11.3) x1.02 |
| Qwen3 1.7B | F16 | CUDA | 8.88e-04 (8.74e-04) | 97.9% (98.4%) | 49 (27) | 7997 (8064) x1.01 | 87.2 (89.5) x1.03 |
| Qwen3 1.7B | Q8_0 | CPU | 2.47e-02 (2.47e-02) | 93.6% (93.6%) | 12 (12) | 187 (195) x1.05 | 20.4 (21.1) x1.03 |
| Qwen3 1.7B | Q8_0 | CUDA | 2.30e-02 (2.52e-02) | 95.1% (94.3%) | 36 (12) | 8201 (8274) x1.01 | 142.2 (153.8) x1.08 |
| Qwen3 1.7B | Q4_K_M | CPU | 6.91e-01 (7.05e-01) | 72.9% (72.9%) | 1 (6) | 229 (269) x1.18 | 32.4 (33.4) x1.03 |
| Qwen3 1.7B | Q4_K_M | CUDA | 6.12e-01 (6.30e-01) | 74.2% (74.8%) | 6 (6) | 7838 (7943) x1.01 | 191.4 (213.7) x1.12 |
| SmolLM2 1.7B | BF16 | CPU | 1.21e-04 (1.12e-04) | 99.4% (99.2%) | 64 (64) | 98 (102) x1.05 | 11.1 (11.3) x1.03 |
| SmolLM2 1.7B | BF16 | CUDA | 1.44e-04 (1.48e-04) | 99.6% (100.0%) | 64 (64) | 5070 (5225) x1.03 | 88.9 (90.7) x1.02 |
| SmolLM2 1.7B | F16 | CPU | 2.63e-06 (2.64e-06) | 99.8% (100.0%) | 64 (64) | 98 (103) x1.05 | 11.1 (11.4) x1.03 |
| SmolLM2 1.7B | F16 | CUDA | 2.25e-04 (2.37e-04) | 99.2% (99.4%) | 64 (64) | 7306 (7451) x1.02 | 89.2 (91.3) x1.02 |
| SmolLM2 1.7B | Q8_0 | CPU | 5.03e-03 (5.78e-03) | 98.2% (98.6%) | 8 (8) | 166 (175) x1.05 | 20.1 (21.1) x1.05 |
| SmolLM2 1.7B | Q8_0 | CUDA | 5.50e-03 (5.29e-03) | 97.7% (98.8%) | 7 (8) | 7388 (7516) x1.02 | 147.6 (158.8) x1.08 |
| SmolLM2 1.7B | Q4_K_M | CPU | 3.32e-01 (3.40e-01) | 84.2% (84.0%) | 1 (1) | 205 (208) x1.02 | 32.9 (35.1) x1.07 |
| SmolLM2 1.7B | Q4_K_M | CUDA | 3.52e-01 (3.52e-01) | 84.0% (84.8%) | 1 (1) | 7091 (7153) x1.01 | 206.2 (231.4) x1.12 |

- Speed: every row is within 1.25x of llama.cpp; the slowest is Qwen3-1.7B Q4_K_M's CPU prompt at
  1.18x. Generation uses the host decoding loop, one runtime call per token, as llama-bench makes
  one `llama_decode` call per token.
- KL: ggml-compiler's is at most 1.07x llama.cpp's in every row.
- Top-1: within a point of llama.cpp in 22 rows. Qwen3-0.6B Q4_K_M on CPU is 2.4 points lower, from
  llama.cpp's CPU weight repacking (interleaved Q4_K kernels that sum in another order): with
  repacking off, llama.cpp gets exactly ggml-compiler's KL (2.370e-01) and top-1 (73.05%). SmolLM2
  Q8_0 on CUDA is 1.1 points lower. Over 512 tokens a point is 5 tokens.
- The text is memorized by these models, so transformers' distribution is very peaked and the KLs
  run higher than published wikitext numbers. Both engines are measured on the same text.

On the server:

| Model | Type | Backend | KL mean | top-1 | greedy /64 | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill 1.5B | BF16 | CPU | 5.95e-05 (5.95e-05) | 99.8% (99.8%) | 64 (64) | 127 (131) x1.03 | 10.9 (11.0) x1.01 |
| DeepSeek-R1-Distill 1.5B | BF16 | CUDA | 7.11e-05 (7.11e-05) | 99.8% (99.8%) | 64 (64) | 15649 (15947) x1.02 | 189.2 (213.3) x1.13 |
| DeepSeek-R1-Distill 1.5B | F16 | CPU | 7.49e-06 (7.49e-06) | 100.0% (100.0%) | 64 (64) | 129 (135) x1.04 | 10.9 (11.1) x1.01 |
| DeepSeek-R1-Distill 1.5B | F16 | CUDA | 2.80e-04 (2.80e-04) | 99.4% (99.4%) | 64 (64) | 19696 (19742) x1.00 | 189.8 (213.4) x1.12 |
| DeepSeek-R1-Distill 1.5B | Q8_0 | CPU | 2.37e-03 (2.37e-03) | 96.9% (96.9%) | 33 (33) | 204 (222) x1.09 | 20.3 (21.1) x1.04 |
| DeepSeek-R1-Distill 1.5B | Q8_0 | CUDA | 2.40e-03 (2.40e-03) | 97.5% (97.5%) | 33 (33) | 18769 (18774) x1.00 | 276.7 (329.4) x1.19 |
| DeepSeek-R1-Distill 1.5B | Q4_K_M | CPU | 9.63e-02 (9.80e-02) | 81.2% (81.1%) | 14 (14) | 265 (254) x0.96 | 32.7 (32.9) x1.01 |
| DeepSeek-R1-Distill 1.5B | Q4_K_M | CUDA | 9.96e-02 (9.96e-02) | 82.4% (82.4%) | 14 (14) | 17545 (17508) x1.00 | 356.2 (401.0) x1.13 |
| Gemma 4 E2B | BF16 | CPU | 1.07e-03 (1.03e-03) | 99.2% (99.2%) | 64 (64) | 85 (54) x0.64 | 7.2 (6.8) x0.94 |
| Gemma 4 E2B | BF16 | CUDA | 1.20e-03 (8.50e-04) | 99.4% (99.2%) | 64 (64) | 9995 (9530) x0.95 | 125.9 (137.1) x1.09 |
| Gemma 4 E2B | F16 | CPU | 2.71e-05 (3.12e-05) | 100.0% (99.8%) | 64 (64) | 87 (55) x0.63 | 7.4 (6.9) x0.93 |
| Gemma 4 E2B | F16 | CUDA | 9.25e-04 (7.10e-04) | 99.0% (99.0%) | 64 (64) | 12426 (11709) x0.94 | 126.4 (137.4) x1.09 |
| Gemma 4 E2B | Q8_0 | CPU | 3.23e-02 (3.65e-02) | 95.5% (94.1%) | 64 (64) | 134 (140) x1.04 | 13.0 (13.3) x1.03 |
| Gemma 4 E2B | Q8_0 | CUDA | 3.68e-02 (4.01e-02) | 94.7% (95.1%) | 64 (64) | 12056 (11720) x0.97 | 182.2 (202.3) x1.11 |
| Gemma 4 E2B | Q4_K_M | CPU | 1.19e+00 (1.28e+00) | 65.2% (63.7%) | 16 (16) | 164 (165) x1.00 | 19.7 (20.7) x1.05 |
| Gemma 4 E2B | Q4_K_M | CUDA | 1.34e+00 (1.39e+00) | 63.7% (62.9%) | 16 (16) | 11482 (10780) x0.94 | 225.8 (243.7) x1.08 |

- DeepSeek-R1-Distill: the KL is llama.cpp's in every row (0.98x of it for Q4_K_M on CPU), top-1
  and greedy match llama.cpp's, and every time is within 1.19x, the slowest being CUDA generation
  through the host loop.
- Gemma 4 E2B: greedy matches llama.cpp's in every row, top-1 is within half a point or higher,
  and every time is within 1.11x; the CPU prompt runs about 1.6x faster. The KL is at most 1.04x
  llama.cpp's on CPU and at Q8_0 and Q4_K_M. At BF16 and F16 on CUDA it is 1.41x and 1.30x: there
  both engines multiply the 16-bit weights through cuBLAS with reduced-precision accumulation, and
  the ratio follows the rounding (it moved between 1.07x and 1.41x between builds that differ only
  in how the 512-wide attention heads are evaluated). With f32 accumulation in both engines
  (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`), ggml-compiler's KL is 1.04x (BF16) and 0.88x (F16) of
  llama.cpp's.

## Vision-language models

f32 parity on `test-1.jpeg`, against transformers:

| Model | Image | Prompt | Last-token logits, CPU / CUDA | Greedy |
|---|---|---|---|---|
| SmolVLM2 500M | 17 tiles of 512 x 512 | 1141 tokens | 1.6e-6 / 9.1e-4 | 64/64, host and device loop |
| Qwen3-VL 2B | 640 x 480, 300 image tokens | 316 tokens | 1.9e-6 / 9.7e-4 | 64/64, host and device loop |

Against llama.cpp, at llama.cpp's weight types for the language model and its F16 projector GGUF
for the vision encoder. SmolVLM2's Q4_K_M stores most tensors as Q5_0, because its hidden size
(960) is not a multiple of 256. SmolVLM2-500M:

| Type | Backend | KL mean | top-1 | greedy /448 | encode ms | prompt ms | tg t/s |
|---|---|---|---|---|---|---|---|
| BF16 | CPU | 5.69e-05 (5.52e-05) x1.03 | 100.0% (99.9%) | 448 (393) | 11337 (10610) x1.07 | 2501 (3112) x0.80 | 42.7 (42.2) x0.99 |
| BF16 | CUDA | 5.03e-05 (5.53e-05) x0.91 | 100.0% (99.9%) | 448 (448) | 111 (137) x0.82 | 38 (128) x0.30 | 421.2 (504.4) x1.20 |
| F16 | CPU | 1.19e-06 (7.02e-06) x0.17 | 100.0% (100.0%) | 448 (448) | 11332 (10650) x1.06 | 2461 (3083) x0.80 | 43.0 (42.7) x0.99 |
| F16 | CUDA | 1.28e-04 (1.22e-04) x1.05 | 99.7% (99.7%) | 414 (393) | 111 (137) x0.81 | 32 (115) x0.28 | 423.4 (506.0) x1.20 |
| Q8_0 | CPU | 1.97e-03 (2.01e-03) x0.98 | 97.8% (97.3%) | 234 (251) | 11328 (10613) x1.07 | 1706 (2119) x0.80 | 74.1 (72.8) x0.98 |
| Q8_0 | CUDA | 2.12e-03 (1.84e-03) x1.15 | 97.3% (97.8%) | 161 (205) | 111 (137) x0.81 | 41 (135) x0.30 | 503.2 (598.4) x1.19 |
| Q4_K_M | CPU | 3.76e-02 (3.61e-02) x1.04 | 90.1% (90.3%) | 71 (64) | 11331 (10614) x1.07 | 2252 (2609) x0.86 | 95.7 (93.5) x0.98 |
| Q4_K_M | CUDA | 3.59e-02 (3.50e-02) x1.03 | 90.7% (90.5%) | 64 (55) | 111 (137) x0.81 | 45 (137) x0.33 | 533.5 (608.3) x1.14 |

Qwen3-VL-2B:

| Type | Backend | KL mean | top-1 | greedy /448 | encode ms | prompt ms | tg t/s |
|---|---|---|---|---|---|---|---|
| BF16 | CPU | 4.66e-05 (1.16e-04) x0.40 | 99.5% (99.4%) | 432 (352) | 2716 (2676) x1.01 | 2679 (2667) x1.00 | 9.7 (9.7) x1.00 |
| BF16 | CUDA | 1.69e-03 (8.53e-04) x1.98 | 99.1% (99.1%) | 397 (342) | 31 (29) x1.07 | 30 (39) x0.76 | 189.0 (195.1) x1.03 |
| F16 | CPU | 1.92e-05 (1.01e-04) x0.19 | 99.8% (99.5%) | 387 (387) | 2721 (2669) x1.02 | 2656 (2621) x1.01 | 9.8 (9.8) x1.00 |
| F16 | CUDA | 1.85e-03 (9.93e-04) x1.86 | 98.9% (99.1%) | 397 (397) | 31 (29) x1.05 | 22 (33) x0.66 | 190.1 (194.9) x1.03 |
| Q8_0 | CPU | 2.44e-03 (2.65e-03) x0.92 | 98.1% (98.0%) | 239 (311) | 2721 (2672) x1.02 | 1688 (1577) x1.07 | 18.0 (18.0) x1.00 |
| Q8_0 | CUDA | 4.15e-03 (3.35e-03) x1.24 | 97.5% (97.5%) | 324 (340) | 30 (30) x1.01 | 24 (35) x0.67 | 269.3 (290.3) x1.08 |
| Q4_K_M | CPU | 9.75e-02 (9.55e-02) x1.02 | 86.3% (86.1%) | 69 (69) | 2733 (2705) x1.01 | 1340 (1407) x0.95 | 28.3 (28.4) x1.00 |
| Q4_K_M | CUDA | 9.46e-02 (9.34e-02) x1.01 | 86.2% (86.2%) | 64 (40) | 31 (30) x1.01 | 25 (36) x0.69 | 333.7 (360.4) x1.08 |

- Speed: every row is within 1.25x. On CUDA, ggml-compiler's prompt is 1.3x to 3.6x faster: it runs
  the whole prompt (text and image rows) in one call, where llama.cpp's multimodal helper decodes
  the text and image spans one after another. CUDA generation is 1.03x to 1.20x slower; that is the
  Python host loop (about 2.4 ms a token for SmolVLM2, against llama.cpp's 2.0).
- Top-1: within a point of llama.cpp everywhere. KL on CPU: at most 1.04x llama.cpp's.
- KL on CUDA ranges from 0.91x to 1.98x, and the ratio says little at this precision, for two
  measured reasons:
  - Batching. ggml's CUDA backend multiplies 16-bit weights through cuBLAS with f16 accumulation
    by default, and how the teacher-forced tokens are grouped into calls changes that rounding:
    with the same weights and program, regrouping them moves the pooled KL by 30% to 40% either
    way.
  - One token. Position 132 of `test-1.jpeg`'s answer, an "M" of the headline "MEN WALK ON MOON",
    where transformers is split 0.83 / 0.16, gets a large KL in every CUDA run of either engine
    (ggml-compiler 1.82, llama.cpp 0.58 on the RTX 3090 and 1.61 on an RTX 3060) and never on CPU
    (0.0016). Even pooled over Qwen3-VL's 1704 positions it is 1.07e-3 of ggml-compiler's 1.69e-3
    BF16 CUDA mean. Over Qwen3-VL's six other images, ggml-compiler is the closer of the two at
    every CUDA type: 0.77x (BF16), 0.89x (F16), 0.94x (Q8_0), 1.01x (Q4_K_M).
- With f32 cuBLAS everywhere (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`), the two agree on
  `test-1.jpeg`: KL without that token 4.06e-4 for ggml-compiler, 4.15e-4 for llama.cpp.

## Robot policies

f32 parity against LeRobot's PyTorch implementation (noise passed in as an input, so both see the
same):

| Model | Inputs | Program | CPU | CUDA |
|---|---|---|---|---|
| ACT | 2 x 480 x 640 | 1 stage | 6.9e-7 | 6.7e-4 |
| ACT via ONNX, against onnxruntime and PyTorch | 2 x 480 x 640 | 1 stage | < 1e-4 | < 1e-2 |
| SmolVLA | 3 x 512 x 512 | prefix + 10 flow steps (+ the hoisted invariant part of a step) | 2.7e-7 | 8.3e-4 |
| Diffusion Policy, DDIM 10 steps | | encode + 10 steps + finish | 1.2e-7 | 1.2e-4 |
| Diffusion Policy, DDPM 100 steps, per-step noise as an input | | encode + 100 steps + finish | 1.6e-7 | 3.1e-5 |
| pi0, full 18 + 18 layers | 2 x 224, 48 tokens | prefix + 10 flow steps | 2.5e-7 | (13 GB in f32 > 12 GB) |
| pi0, language model and expert cut to 2 layers | 2 x 224, 48 tokens | prefix + 10 flow steps | 2.6e-7 | 8.6e-5 |
| pi0.5, 2 layers | 2 x 224, 200 tokens | prefix + 10 flow steps | 1.8e-7 | 6.6e-5 |
| GR00T N1.7, full | 2 x 256, 148 tokens | backbone + encode + 4 flow steps (+ hoisted part) | 6.6e-5 | (12.6 GB in f32 > 12 GB) |
| GR00T N1.7, language model 3 layers, DiT 4, VL self-attention 1 | 2 x 256, 148 tokens | same | 1.1e-7 | 8.4e-5 |

The stage functions reproduce LeRobot's own samplers: Diffusion Policy's `generate_actions` (DDIM,
and DDPM with LeRobot's seeded noise) and GR00T N1.7's `get_action` bit for bit, pi0 and pi0.5's
`sample_actions` within 1e-5.

Full-size pi0 and pi0.5 with 16-bit weights, against the f32 reference (CPU / CUDA): pi0 BF16
1.2e-3 / 1.1e-3, F16 1.5e-4 / 5.3e-4; pi0.5 BF16 8.9e-3 / 5.0e-3, F16 1.1e-3 / 6.4e-3 (CUDA's F16
matmuls accumulate in f16). GR00T N1.7 is far more sensitive to precision, with a 16-layer Qwen3
language model whose residual activations reach about 1.5e4, then 32 DiT blocks over 4 steps: F16
1.1e-2 (CUDA) / 3.2e-2 (CPU, with ggml's fast GELU), BF16 7.3e-2 / 1.2e-1, and an f32 backbone
with an F16 action head (`examples/groot/policy-head-f16.yaml`) 5.8e-3 / 2.9e-3.

Latency against vla.cpp (inputs as `vla-bench --images N --size S --tokens T`):

| Build | ggml-compiler | vla.cpp | x |
|---|---|---|---|
| ACT 2 x 480 x 480, CPU f32 | 274 ms | 286 ms | 0.96 |
| ACT, CUDA, F16 conv kernels (`examples/act/policy-cuda.yaml`) | 10.4 ms | 9.9 ms | 1.05 |
| SmolVLA, CUDA, F16 vision (`examples/smolvla/policy-cuda.yaml`), `O2` | 120.0 ms | 147.3 ms | 0.82 |
| SmolVLA, CPU, f32 weights, `O2` | 3027 ms | 3277 ms (f32) | 0.92 |
| SmolVLA, CPU, F16 weights, `O2` | 2805 ms | 3042 ms (F16) | 0.92 |
| SmolVLA, CPU, BF16 weights, `O2` | 2820 ms | 3052 ms (BF16, its default) | 0.92 |
| pi0 2 x 224, 48 tokens, CUDA, BF16 weights | 260 ms | 239 ms (BF16 weights, f32 activations: its default) | 1.09 |
| pi0, CUDA, F16 weights, fast GELU, exact attention | 221 ms | 239 ms (205 ms with BF16 activations and flash attention) | 0.93 |
| pi0, CPU, F16 weights, fast GELU, exact attention | 9.79 s | 9.95 s (BF16; f32 11.2 s) | 0.98 |
| pi0, CPU, BF16 weights | 10.60 s | 9.95 s | 1.07 |
| pi0.5 2 x 224, 200 tokens, CUDA, BF16 weights | 309 ms | 287 ms (its default) | 1.08 |
| pi0.5, CUDA, F16 weights, fast GELU, exact attention | 262 ms | 287 ms | 0.91 |
| pi0.5, CPU, F16 weights, fast GELU, exact attention | 11.41 s | 11.38 s (its default) | 1.00 |
| pi0.5, CPU, BF16 weights | 12.33 s | 11.38 s | 1.08 |
| GR00T N1.7 2 x 256, 12 instruction tokens, CUDA, BF16 weights | 128.2 ms | 128.1 ms (BF16, its default) | 1.00 |
| GR00T N1.7, CUDA, F16 weights, fast GELU, exact attention | 109.5 ms | 128.1 ms | 0.86 |
| GR00T N1.7, CUDA, f32 backbone + F16 head | 160.0 ms | 128.1 ms | 1.25 |
| GR00T N1.7, CPU, F16 weights, fast GELU, exact attention | 3.20 s | 3.11 s (BF16, its default) | 1.03 |
| GR00T N1.7, CPU, BF16 weights | 3.37 s | 3.11 s | 1.08 |

`O2` rows use flash attention and ggml's table GELU; the others are `O1`, with ggml's fast GELU
only where the row says so. vla.cpp's
pi0.5 GGUF is `vrfai/pi05-libero-gguf`, made from `pi05_libero_base`: vla.cpp's converter refuses
the finetuned checkpoint's normalization, and the architecture and shapes are the same.

vla.cpp has no Diffusion Policy. Against PyTorch on CPU, DDIM with 10 steps takes 326 ms in
ggml-compiler with f32 weights and 199 ms with F16 (`examples/diffusion/policy-f16.yaml`), against
PyTorch's 418 ms; on CUDA, 48 ms and 29 ms. DDPM with 100 steps: 3.27 s and 1.80 s on CPU, 467
ms and 285 ms on CUDA.
