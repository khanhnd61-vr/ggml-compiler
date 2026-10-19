# Examples

These examples compile PyTorch models into ggml C++ source and a GGUF model. The generated C++
contains one ggml graph builder per stage; the GGUF contains the weights and a serialized form of
the same compiled computation for the runtime.

| Directory | Model | Generated computation |
|---|---|---|
| [`encoder_block`](encoder_block/) | Synthetic pre-norm transformer block | One forward stage |
| [`llm`](llm/) | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Stateful prefill, decode, and greedy generation |
| [`vlm`](vlm/) | [SmolVLM2-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct) | Image encoder plus stateful text generation |
| [`act`](act/) | [ACT](https://huggingface.co/khanhnd61/act_so101-multi-task-clean) | One policy forward stage |
| [`smolvla`](smolvla/) | [SmolVLA](https://huggingface.co/khanhnd61/smolvla_so101-multi-task-clean) | VLM prefix followed by 10 flow-matching steps |
| [`pi`](pi/) | [π0](https://huggingface.co/lerobot/pi0_libero_finetuned_v044) or π0.5 | PaliGemma prefix followed by 10 flow-matching steps |
| [`groot`](groot/) | GR00T N1.7 | Qwen3-VL backbone, state/VL encoder, and 4 DiT flow-matching steps |
| [`diffusion`](diffusion/) | [Diffusion Policy](https://huggingface.co/lerobot/diffusion_pusht) | Observation encoder, DDPM/DDIM loop, and action selection |

Each `export.py` exposes a zero-argument `program()` function for the CLI. The LeRobot examples
load a pretrained checkpoint, create deterministic example inputs, and preserve preprocessing and
postprocessing as caller responsibilities. They are reference exporters as well as useful starting
points for another checkpoint with the same architecture.

## Set up a checkout

The policy examples use the repository's `parity` dependency group, which installs the compatible
LeRobot fork on Python 3.12; it also provides the Transformers dependencies used by the LLM and
VLM examples. Set up the checkout with [uv](https://docs.astral.sh/uv/):

```bash
uv sync

# Build the runtime. The first configure downloads ggml into .deps/.
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For CUDA, build a CUDA runtime instead (replace `86` with the compute capability of the target
GPU):

```bash
cmake -S runtime -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j
```

The first use of a Hugging Face model downloads its checkpoint. Set the example-specific
`*_CHECKPOINT` variable to use another Hub repository or a local checkpoint with the same model
architecture.

## Compile and check an example

Start with the small example, which does not download a checkpoint:

```bash
uv run ggml-compiler compile examples/encoder_block/export.py:program \
  --out out/encoder-block --emit gguf,cpp --name encoder_block
uv run ggml-compiler verify out/encoder-block/model.gguf --backend cpu
```

Compile a policy in the same way:

```bash
uv run ggml-compiler compile examples/act/export.py:program \
  --out out/act --emit gguf,cpp --name act
uv run ggml-compiler inspect out/act/model.gguf
uv run ggml-compiler verify out/act/model.gguf --backend cpu
```

`verify` compares the GGUF runtime output with the PyTorch reference outputs captured during
compilation. Use `--backend cuda` to check a CUDA build. The other useful commands are:

```bash
uv run ggml-compiler run out/act/model.gguf --backend cpu --out actions.npz
uv run ggml-compiler bench out/act/model.gguf --backend cpu --threads 6
uv run ggml-compiler bench out/act/model.gguf --backend cuda --profile
```

By default, `run` and `bench` use the example inputs stored in the artifact. Pass
`--inputs inputs.npz` to supply your own flattened, named tensors. `inspect` reports their exact
names, shapes, and dtypes; list-valued inputs such as camera images are flattened to names such as
`images.0`.

The generated C++ can be built and checked independently of the GGUF's stored graph:

```bash
cmake -S out/act/cpp -B out/act/cpp-build \
  -DGGML_COMPILER_RUNTIME_LIB="$PWD/build/libggml_compiler_rt.so"
cmake --build out/act/cpp-build -j
out/act/cpp-build/act-check out/act/model.gguf out/act/example_inputs.bin cpu
```

Use `build-cuda/libggml_compiler_rt.so` and the final argument `cuda` for CUDA.

## Language model examples

### Qwen3 LLM

The LLM example compiles Qwen3-0.6B into `prefill` and `decode` entry points over an in-place KV
cache. It also emits a `generate` entry point that can run greedy decoding inside the runtime:

```bash
uv run ggml-compiler compile examples/llm/export.py:program \
  --out out/qwen3 --emit gguf,cpp --name qwen3 -O O3
```

The default limits are a 4096-token context and 512 new prompt tokens per prefill call. Change
them, or select another supported Hugging Face causal LM, with environment variables:

```bash
LLM_CHECKPOINT=Qwen/Qwen3-0.6B LLM_MAX_CONTEXT=2048 LLM_MAX_PROMPT=256 \
uv run ggml-compiler compile examples/llm/export.py:program \
  --out out/qwen3-2k --emit gguf,cpp --name qwen3 -O O3
```

Tokenization remains on the caller. The companion script tokenizes a prompt and uses
`ggml_compiler.Generator` to manage positions, attention masks, prompt chunks, and the KV cache:

```bash
uv run python examples/llm/generate.py out/qwen3/model.gguf \
  "The capital of France is" --backend cpu --max-new 32

# Run the greedy token loop inside the compiled runtime.
uv run python examples/llm/generate.py out/qwen3/model.gguf \
  "The capital of France is" --backend cuda --max-new 32 --device-loop
```

The generated token ids are decoded as the continuation only. Use a tokenizer's chat template
before calling `Generator.generate` when the selected checkpoint expects one.

### SmolVLM VLM

The VLM example adds `encode` and multimodal prefill entry points to the same stateful language
model structure. `encode` runs the vision tower and projector into an image buffer; subsequent
prefill calls place those image rows at the prompt's image-token positions:

```bash
uv run ggml-compiler compile examples/vlm/export.py:program \
  --out out/smolvlm --emit gguf,cpp --name smolvlm -O O3
```

The default checkpoint is SmolVLM2-500M-Instruct with a 2048-token context. The Hugging Face
processor remains on the caller and supplies the chat template, image tiling, pixel values, and
placeholder tokens. Run the included image example with:

```bash
uv run python examples/vlm/generate.py out/smolvlm/model.gguf path/to/image.jpg \
  --prompt "Describe this image in detail." --backend cpu --max-new 128
```

Set `VLM_CHECKPOINT` to another compatible SmolVLM checkpoint, or set `VLM_MAX_CONTEXT` or
`VLM_MAX_PROMPT` before compilation to change the size limits. `program(max_image_tokens=N)` can
set the image-state buffer size explicitly; when it is omitted, `causal_lm` uses the
processor-sized SmolVLM default. The artifact records the tile size, rows per tile, maximum tile
count, image token id, context limits, and EOS ids in its metadata.

## Policy examples

### ACT

```bash
ACT_CHECKPOINT=khanhnd61/act_so101-multi-task-clean \
uv run ggml-compiler compile examples/act/export.py:program \
  --out out/act --emit gguf,cpp --name act
```

The program takes normalized robot state followed by one normalized image tensor for each camera
in the checkpoint configuration, and returns a normalized action chunk. The caller performs the
same state/action normalization and image preprocessing as LeRobot. The default example uses the
training resolution, but ACT can be compiled for another camera resolution through the Python API:

```python
import ggml_compiler
from examples.act.export import program

ggml_compiler.compile(program(image_size=(240, 320)), "out/act-240x320")
```

### SmolVLA

```bash
SMOLVLA_CHECKPOINT=khanhnd61/smolvla_so101-multi-task-clean \
uv run ggml-compiler compile examples/smolvla/export.py:program \
  --out out/smolvla --emit gguf,cpp --name smolvla
```

The logical inputs are camera images, camera masks, language token ids, the language padding mask,
normalized state, and caller-generated initial action noise. The output `x` is the normalized
action chunk. Image resizing and scaling to `[-1, 1]`, tokenization, and state/action
(un)normalization remain with the caller. The prefix stage creates a VLM KV cache once; the step
stage reuses it for all ten flow-matching steps.

### π0 and π0.5

`PI_CHECKPOINT` selects the model. The checkpoint's `type` field determines whether the exporter
builds π0 or π0.5:

```bash
PI_CHECKPOINT=lerobot/pi0_libero_finetuned_v044 \
uv run ggml-compiler compile examples/pi/export.py:program \
  --out out/pi0 --emit gguf,cpp --name pi0
```

Both variants take camera images and masks, language tokens and mask, and initial action noise. π0
also takes normalized state; π0.5 puts discretized state in the prompt. Images must already be
resized to 224×224 and scaled to `[-1, 1]`. The caller also handles tokenization, padding,
discretization, and normalization.

The exporter omits checkpoint cameras named `empty_camera` because LeRobot masks them completely.
Call `program_from_policy(..., empty_cameras=True)` when an artifact must retain those inputs.
`load(..., n_layers=N)` constructs a shallow version for quick compiler development; it is not the
full policy.

### GR00T N1.7

GR00T currently expects a local checkpoint directory. Set both the directory and, when needed, the
embodiment name:

```bash
GROOT_CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_object \
GROOT_EMBODIMENT=libero_sim \
uv run ggml-compiler compile examples/groot/export.py:program \
  --out out/groot --emit gguf,cpp --name groot
```

The inputs are Qwen3-VL patchified pixels, token ids, an attention mask, normalized state, and
caller-generated initial action noise. The Qwen3-VL processor owns resizing, patchification,
tokenization, and the chat template; the caller owns state/action normalization.

The example prompt, camera count, image patch grids, prompt length, and embodiment are specialized
at compile time and recorded in the program metadata. A deployment with a different layout needs a
new artifact. Edit or call `example_inputs()` with the deployment instruction and camera count
before constructing the program. `load(..., n_layers={"llm": 4, "dit": 4,
"vl_self_attention": 1})` is available for small compiler tests.

### Diffusion Policy

```bash
DIFFUSION_CHECKPOINT=lerobot/diffusion_pusht \
uv run ggml-compiler compile examples/diffusion/export.py:program \
  --out out/diffusion --emit gguf,cpp --name diffusion
```

The program takes normalized state and camera histories, initial trajectory noise, and one noise
slice per denoising step. It returns the configured action window. The observation encoder runs
once, while scheduler coefficients are compile-time loop scalars. With DDPM, `step_noise` supplies
the random noise added at each step, keeping the artifact deterministic; the last slice is zero.

The checkpoint's scheduler and step count are the CLI defaults. Use the Python API to switch to
DDIM or change the number of inference steps:

```python
import ggml_compiler
from examples.diffusion.export import program

ggml_compiler.compile(
    program(scheduler="DDIM", num_inference_steps=10),
    "out/diffusion-ddim10",
    emit=("gguf", "cpp"),
    name="diffusion",
)
```

## Weight dtype policies

Pass a shipped YAML file with `--policy`. The policy changes how selected weights are stored and
which kernels consume them; unmatched weights remain f32 in all policies below.

```bash
uv run ggml-compiler compile examples/smolvla/export.py:program \
  --out out/smolvla-q8 --policy examples/smolvla/policy-q8_0.yaml --opt O2
```

| Example | Policy | Effect |
|---|---|---|
| ACT | [`policy-cuda.yaml`](act/policy-cuda.yaml) | F16 convolution weights in the visual backbone |
| SmolVLA | [`policy-cpu.yaml`](smolvla/policy-cpu.yaml) | BF16 GEMM weights |
| SmolVLA | [`policy-cuda.yaml`](smolvla/policy-cuda.yaml) | F16 vision tower and connector; the rest stays f32 |
| SmolVLA | [`policy-f16.yaml`](smolvla/policy-f16.yaml) | F16 GEMM weights |
| SmolVLA | [`policy-q8_0.yaml`](smolvla/policy-q8_0.yaml) | Q8_0 matmul weights and token embeddings |
| SmolVLA | [`policy-q4k-f16.yaml`](smolvla/policy-q4k-f16.yaml) | F16 vision plus Q4_K text/expert weights, with Q8_0/F16 fallback |
| π0 / π0.5 | [`policy-bf16.yaml`](pi/policy-bf16.yaml) | BF16 GEMM weights and token embeddings |
| π0 / π0.5 | [`policy-f16.yaml`](pi/policy-f16.yaml) | F16 GEMM weights and token embeddings |
| GR00T | [`policy-bf16.yaml`](groot/policy-bf16.yaml) | BF16 GEMM weights and token embeddings |
| GR00T | [`policy-f16.yaml`](groot/policy-f16.yaml) | F16 GEMM weights and token embeddings |
| GR00T | [`policy-head-f16.yaml`](groot/policy-head-f16.yaml) | F32 backbone with the action head in F16 |
| Diffusion | [`policy-f16.yaml`](diffusion/policy-f16.yaml) | F16 U-Net and ResNet encoder weights |

Establish f32 parity before choosing a reduced-precision policy. For the design of these programs,
how to adapt them to another model, and how to compare against upstream inference, see
[`docs/EXPORT.md`](../docs/EXPORT.md). The full API and CLI reference is in
[`docs/USAGE.md`](../docs/USAGE.md).
