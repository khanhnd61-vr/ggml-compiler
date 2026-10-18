# Using ggml-compiler

- [Compiling a single-stage model](#compiling-a-single-stage-model)
- [Running a compiled model](#running-a-compiled-model)
- [Multi-stage models: stages and a driver](#multi-stage-models-stages-and-a-driver)
- [Inputs, outputs and dtypes](#inputs-outputs-and-dtypes)
- [Symbolic dims](#symbolic-dims)
- [LLMs and VLMs](#llms-and-vlms)
- [Optimization levels and flags](#optimization-levels-and-flags)
- [Weight dtypes and quantization](#weight-dtypes-and-quantization)
- [Importing ModelOpt-quantized models](#importing-modelopt-quantized-models)
- [ONNX](#onnx)
- [Hand-tunable C++](#hand-tunable-c)
- [C API](#c-api)
- [CLI](#cli)
- [When an op has no ggml lowering](#when-an-op-has-no-ggml-lowering)
- [Worked examples: LeRobot policies](#worked-examples-lerobot-policies)
- [Limitations](#limitations)

## Compiling a single-stage model

A model with one forward pass compiles in one call. Wrap it so the inputs and outputs are plain
tensors or pytrees of tensors (`examples/act/export.py` is the full version of this one):

```python
import torch, ggml_compiler
from lerobot.policies.act.modeling_act import ACTPolicy

act = ACTPolicy.from_pretrained("khanhnd61/act_so101-multi-task-clean").eval().model

class ACTInfer(torch.nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, state, *images):
        actions, _ = self.act({"observation.state": state, "observation.images": list(images)})
        return actions

example = (torch.zeros(1, 6), torch.rand(1, 3, 480, 640), torch.rand(1, 3, 480, 640))
art = ggml_compiler.compile(ACTInfer(act), example, "out/act",
                    input_names=["state", "front", "wrist"], output_names=["actions"])

print(art.gguf)               # out/act/model.gguf
print(art.report.summary())   # stages, op counts, weights, inputs and outputs
```

Shapes and batch size are fixed by the example inputs. Compile again for another camera
resolution, prompt length or chunk size.

The output directory holds:

```
out/act/
  model.gguf                 weights + compiled program
  report.json                op histogram per stage, every weight's stored type and error, I/O
  example_inputs.npz         the inputs used at compile time
  reference_outputs.npz      PyTorch (or onnxruntime) outputs on them, used by `ggml-compiler verify`
```

`ggml_compiler.compile` options:

| Option | Meaning |
|---|---|
| `input_names`, `output_names` | Names of the runtime inputs and outputs (defaults: the forward's argument names, `out`) |
| `opt="O0" … "O3"` | Optimization level (default `O1`; see [below](#optimization-levels-and-flags)) |
| `flags={"name": bool}` | Turn single flags on or off on top of the level |
| `precision="exact" \| "fast"`, `attention="exact" \| "flash"` | Older spellings: `exact` is `O1`, `fast` is `O2`; `attention` sets the `flash_attention` flag |
| `dtype_policy` | Weight storage types, per tensor (YAML path, dict or `ggml_compiler.DTypePolicy`) |
| `prequant=ggml_compiler.ModelOpt()` | Import a ModelOpt-quantized model's weights |
| `emit=("gguf", "cpp")` | Also write the C++ project |
| `backends=["cpu", "cuda"]` | Load the result on these backends now, so a node a backend cannot run fails at compile time |
| `input_shapes` | ONNX only: pin dynamic input dims |

## Running a compiled model

From Python:

```python
sess = ggml_compiler.Session("out/act/model.gguf", backend="cuda", n_threads=6)  # "cpu" | "cuda" | "auto"
sess.inputs     # [TensorInfo(name="state", shape=(1, 6), dtype="f32"), ...]
out = sess.run(state=state, front=img0, wrist=img1)  # numpy or torch in, dict of numpy out
out["actions"].shape                                 # (1, 100, 6)
sess.stats      # {"ms_total": ..., "n_calls": 1, "stages": {"forward": {"ms": ..., "calls": 1}}}
sess.close()    # or use it as a context manager
```

From the shell, on the inputs saved at compile time:

```bash
ggml-compiler verify out/act/model.gguf --backend cuda  # against the PyTorch outputs saved at compile time
ggml-compiler bench  out/act/model.gguf --backend cpu --threads 6
ggml-compiler bench  out/act/model.gguf --backend cuda --profile  # every node, grouped by op and source line
```

## Multi-stage models: stages and a driver

Flow-matching and diffusion policies call an action head many times against one encoded prefix.
Instead of unrolling that loop into one huge graph, export each piece as a **stage** and describe
the loop with a **driver**. Each stage becomes its own ggml graph, built once at load. The weights
are shared, and values passed between stages, such as the prefix KV cache, stay on the device.
`examples/smolvla/export.py` is the full version of this one:

```python
import torch, ggml_compiler
from transformers import DynamicCache
from lerobot.policies.common.vla_utils import make_att_2d_masks

fm = policy.model          # LeRobot VLAFlowMatching
N, dt = 10, -1.0 / 10

def prefix(images, img_masks, lang_tokens, lang_masks, state):
    embs, pad, att = fm.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    _, cache = fm.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                                          position_ids=torch.cumsum(pad, dim=1) - 1,
                                          past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
    return [(l.keys, l.values) for l in cache.layers], pad     # stages exchange plain tensors

def step(x, t, kv, pad):
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv):
        cache.update(k, v, i)
    v = fm.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=t)
    return x + dt * v                                          # the Euler update is part of the stage

prefix_ex = (images, img_masks, lang_tokens, lang_masks, state)   # static shapes
kv_ex, pad_ex = prefix(*prefix_ex)

program = ggml_compiler.Program(
    root=fm,                                                   # owner of every parameter the stages use
    stages={"prefix": ggml_compiler.Stage(prefix, prefix_ex),
            "step": ggml_compiler.Stage(step, (noise, torch.ones(1), kv_ex, pad_ex))},
    driver=ggml_compiler.Driver(
        inputs=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
        body=[
            ggml_compiler.call("prefix", ["images", "img_masks", "lang_tokens", "lang_masks", "state"],
                               ["kv", "pad"]),
            ggml_compiler.loop(N, carry={"x": "noise"}, scalars={"t": [1.0 + i * dt for i in range(N)]},
                               body=[ggml_compiler.call("step", ["x", "t", "kv", "pad"], ["x"])]),
        ],
        outputs=["x"],
    ),
)
art = ggml_compiler.compile(program, out_dir="out/smolvla")
```

| Driver element | Meaning |
|---|---|
| `call(stage, args, out)` | Run one stage. `args` and `out` name driver values; a value may be a pytree (the KV cache, a camera list) |
| `loop(n, body, carry, scalars, slices)` | Run `body` n times. `carry` sets each loop variable's starting value; a body output with the same name rebinds it. `scalars` supplies one number per iteration (a table in the artifact). `slices` maps a loop value to a driver value with a leading dim of n: iteration i sees row i, copied on the device |
| `loop(..., counters, bind, masks, until, collect)` | Generation loops. `n` may name a symbol (`Program(symbols={"max_new": (min, max, example)})`) set per run. `counters={"pos": "start"}`: an i32 loop value equal to start + i. `bind={"kv_blocks": lambda pos: ...}`: a symbol recomputed from the counters before each iteration (the stages it shapes are rebuilt when it changes). `masks={"m": "pos"}`: an additive causal mask row, 0 up to the counter and -inf after, in the value's current shape. `until={"tok": [eos]}`: stop after an iteration whose value is one of the ids. `collect={"out": "tok"}`: row i of a driver value is the loop value after iteration i |
| `inputs` / `outputs` | The program's runtime I/O. Pytrees flatten to dotted names: `images.0`, `kv.3.1`, ... |
| `Driver(..., examples={...})` | Example values for driver inputs: required for one no stage reads directly (a slice source), optional otherwise |

A DDPM sampler adds fresh noise on every step. That noise is a program input with one slice per
step, so the artifact stays deterministic and the caller owns the random numbers
(`examples/diffusion/export.py`; the scheduler's per-step coefficients are loop scalars):

```python
sched = schedule(model)    # {"t": [...], "c0": [...], ..., "sigma": [...]}: one entry per step
ggml_compiler.loop(100, body=[ggml_compiler.call("step", ["x", "cond", "z", *sched], ["x"])],
           carry={"x": "noise"}, scalars=sched, slices={"z": "step_noise"})   # step_noise: [100, B, T, D]
```

`Stage(fn, example_inputs, dynamic_shapes=None, matmul_acc=None)`: `matmul_acc="f32"` asks every
matmul of the stage to accumulate in f32. With 16-bit weights, CUDA otherwise accumulates their
products in f16 (cuBLAS); CPU kernels accumulate in f32 anyway. It costs speed on CUDA (Qwen3-VL-2B's
F16 image encode on an RTX 3090: 30 -> 48 ms) and halves that model's F16 CUDA KL.

Work inside a loop that depends only on values the loop never changes is moved out
automatically: SmolVLA's expert projects the cached prefix K/V once, in a `step.invariant` stage
that runs before the loop, instead of on every step.

The CLI compiles a program from a Python file: `ggml-compiler compile
examples/smolvla/export.py:program --out out/smolvla`, where `program` is a `ggml_compiler.Program`
or a function returning one.

### Entry points and state

A program can have several **entry points**, each its own driver, run one at a time; and **state**,
values that persist across runs on the device. `ggml_compiler.causal_lm` is built this way:

```python
program = ggml_compiler.Program(
    root=model,
    stages={"prefill": ggml_compiler.Stage(step, prefill_ex, dynamic_shapes=...),
            "decode": ggml_compiler.Stage(step, decode_ex, dynamic_shapes=...)},
    driver={"prefill": ggml_compiler.Driver(
                inputs=["tokens", "positions", "mask"], outputs=["logits"],
                body=[ggml_compiler.call("prefill", ["tokens", "positions", "mask", "kv"], ["logits", "kv"])]),
            "decode": ggml_compiler.Driver(
                inputs=["token", "position", "mask1"], outputs=["logits1"],
                body=[ggml_compiler.call("decode", ["token", "position", "mask1", "kv"], ["logits1", "kv"])])},
    state={"kv": kv_example},                       # zeros at load; Session.reset() zeroes it again
    metadata={"eos": [151645]},                     # stored in the GGUF for callers
)

with ggml_compiler.Session(art.gguf) as s:
    s.entries                                       # {"prefill": (inputs, outputs), "decode": ...}
    s.run(tokens=..., positions=..., mask=..., entry="prefill")
    s.run(token=..., position=..., mask1=..., entry="decode")
```

- The first entry is the default (`Session.run` without `entry`, `ggml_compiler_run`).
- A stage reads state as a parameter and returns its new value, which the driver binds to the same
  name. When the stage writes it with an in-place op (`index_copy` / `index_put` of rows, which
  lower to `ggml_set_rows`), the update costs only the rows written: the runtime sees that the
  result is the state's own memory and copies nothing.
- The compile-time reference run calls every entry once, in order, with the state carried from
  one to the next; `reference_outputs.npz` keys outputs as `<entry>/<value>`.

## Inputs, outputs and dtypes

- The program starts and ends at tensors. Image resizing, tokenization, chat templates and
  anything else outside the stage functions is the caller's job. Normalization written inside a
  stage is compiled with it.
- Language prompts are runtime inputs: token ids padded to the example's length, plus a mask.
  One artifact serves every instruction of that length.
- ggml has no bool type and indexes with int32. Integer inputs are stored as `i32` and boolean
  masks as `f32` (0/1). `Session.run` converts for you; C callers read the dtype from
  `ggml_compiler_input_info`.
- Random inputs such as flow-matching noise are explicit inputs, so runs are reproducible.
- Tensors above 4-D (a camera history `[B, T, cams, C, H, W]`) are fine while all but four of
  their dims are 1: ggml stores them without the size-1 dims.
- Attention masks may be computed at run time from inputs (a padding mask), as bool or additive
  float masks.

## Symbolic dims

A dim declared dynamic stays symbolic: one artifact takes any size in the dim's range, and the
runtime builds the ggml graphs for each size it is given. Declare dims per stage input, as
`torch.export` takes them; a `Dim`'s name is the symbol's name across the program, and every
`Dim` needs a max:

```python
from torch.export import Dim

seq = Dim("seq", min=2, max=512)
program = ggml_compiler.Program(
    root=model,
    stages={"forward": ggml_compiler.Stage(model, (x,), dynamic_shapes=({1: seq},))},  # x: [1, seq, 256]
    driver=ggml_compiler.Driver(inputs=["x"], outputs=["y"],
                                body=[ggml_compiler.call("forward", ["x"], ["y"])]))
art = ggml_compiler.compile(program, out_dir="out/encoder")

with ggml_compiler.Session(art.gguf) as s:
    s.symbols                              # {"seq": (2, 512)}
    s.run(x=torch.randn(1, 37, 256))       # binds seq = 37 from the input's shape
```

- Derived dims work: `{3: 256 * Dim("blocks", min=1, max=16)}` makes that dim a multiple of 256.
  Shapes, reshapes and slices may be any integer expression of the symbols (`+`, `*`, floor
  division, modulo, min, max).
- torch.export assumes a dynamic dim is never 0 or 1 and raises such a minimum to 2. Compile a
  separate stage for size 1 (an LLM's one-token decode step).
- `Session.run` binds the symbols from the input shapes. C callers use
  `ggml_compiler_set_input_shape` or `ggml_compiler_set_symbol` (see the [C API](#c-api)).
- Driver values between stages may be symbolic too; their buffers are sized for the largest
  binding. The first run with a new binding rebuilds the graphs of the stages with symbolic
  shapes; graphs are first built with every symbol at its max, so that is the memory reserved.
- `report.json`, `ggml-compiler inspect` and the generated C++ show symbolic shapes (`[1, seq,
  256]`); the C++ builders read each symbol with `ggml_compiler_build_symbol`.

## LLMs and VLMs

`ggml_compiler.causal_lm` turns a Hugging Face causal LM (Llama, Qwen2/3, SmolLM, Gemma 4, ...) into
a program with a KV cache; `ggml_compiler.Generator` drives it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=4096, max_prompt=512),
                            out_dir="out/qwen3", opt="O3", dtype_policy="q4_k_m.yaml")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
with ggml_compiler.Session(art.gguf) as s:
    gen = ggml_compiler.Generator(s)
    ids = gen.generate(tok("The capital of France is")["input_ids"], max_new=32)   # greedy
    print(tok.decode(ids))
```

- The program has two entry points over one KV cache:
  - `prefill(tokens [1, s], positions [s], mask [1, 1, s, n_kv]) -> logits [1, 1, vocab]`, for
    2 ≤ s ≤ `max_prompt` new tokens (`logits="all"` returns every token's row instead);
  - `decode(token [1, 1], position [1], mask1 [1, 1, 1, n_kv]) -> logits1 [1, 1, vocab]`.
- The cache is [state](#entry-points-and-state). It stays on the device and is written in place
  at the given positions. Attention reads its first `n_kv` rows, a multiple of 256 between 512 and
  `max_context`, so each call costs what the context in use costs, as in llama.cpp.
- `mask` is additive: 0 where token i may attend to position j, -inf elsewhere.
- `Generator` builds positions and masks, splits long prompts into `max_prompt` chunks, and
  decodes greedily on the host. `feed(ids)` appends tokens and returns the next-token logits;
  `reset()` empties the context. Sample however you like from those logits.
- A third entry point, `generate(first [1, 1], start [1]) -> generated [max_new, 1, 1]`, runs
  greedy decoding inside the runtime: it feeds `first` at position `start`, then each argmax
  (computed on the device) back, sets the positions, masks and KV length itself, and stops at the
  model's EOS ids or after `max_new` tokens (a symbol: `Session.set_symbol("max_new", n)`). The
  tokens it produced are the first `stats["stages"]["decode_greedy"]["calls"]` rows.
  `Generator.generate(prompt, max_new, device=True)` uses it. `causal_lm(..., generate=False)`
  leaves it out.
- Tokenization and chat templates stay with the caller. The artifact records what a C caller needs
  in its metadata (`ggml_compiler_metadata(m, "ggml_compiler.meta")`, `Session.metadata`):
  `max_context`, `max_prompt`, `kv_block` (256), `min_kv`, `vocab`, `eos`, `bos`.
- `ggml_compiler.llm.bench(session, pp=512, tg=128)` measures prompt and generation tokens/s the way
  llama-bench does.
- `ggml_compiler.llm.llama_cpp_policy("model-q4_k_m.gguf", model)` is a dtype policy that gives
  every weight the type llama.cpp stored it as in that GGUF, so the two run the same weight bytes.
  For a multimodal model without `mmproj=` (below), that is the language model's weights.
- Gemma 4 (e.g. `google/gemma-4-E2B-it`) compiles its language model only: images and audio are
  not compiled yet. Its sliding-window layers need no extra input: the program masks the
  positions older than the window itself, from `positions`. Layers that reuse an earlier layer's
  K/V have no cache of their own.
- At `O3`, decoding runs with flash attention on an f16 cache, ggml's RoPE, and SwiGLU and RMSNorm
  as single ggml ops; see [optimization levels](#optimization-levels-and-flags).

### Vision-language models

`causal_lm` also takes a SmolVLM (Idefics3) or Qwen3-VL model, e.g.
`HuggingFaceTB/SmolVLM2-500M-Instruct` or `Qwen/Qwen3-VL-2B-Instruct`, and adds an image path to
the same program:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32)
art = ggml_compiler.compile(ggml_compiler.causal_lm(model, max_context=4096), out_dir="out/vlm",
                            opt="O3")

proc = AutoProcessor.from_pretrained(repo)
inp = proc(text=prompt_with_image, images=[image], return_tensors="pt")   # resizing, tiling, patches: the caller's
with ggml_compiler.Session(art.gguf) as s:
    gen = ggml_compiler.Generator(s)
    ids = gen.generate(inp["input_ids"][0].tolist(), max_new=128, pixels=inp["pixel_values"])   # SmolVLM
    ids = gen.generate(inp["input_ids"][0].tolist(), max_new=128, pixels=inp["pixel_values"],
                       grid_thw=inp["image_grid_thw"][0])                                          # Qwen3-VL
```

- Two more entry points, over an image buffer that is state like the KV cache:
  - `encode(pixels, ...)` runs the vision tower and projector and writes one row per image
    token into the buffer. It has no outputs.
  - `prefill_mm(tokens [1, s], gather [s], positions [s], ..., mask) -> logits` is `prefill` with
    input embeddings taken from `[image rows; embeddings of this call's tokens]`: `gather[i]` is
    `k` for the k-th image row and `image_rows + i` for token i itself. Image rows can be spread
    over several calls.
- `Generator.encode(pixels[, grid_thw])` runs `encode`; the image tokens fed afterwards (the
  processor's image placeholders) take its rows in order. `generate(..., pixels=...)` does both.
- `max_image_tokens` sets the buffer's size: the image tokens one `encode` can produce. The
  default is one image at the processor's defaults for SmolVLM (17 tiles, 1088 tokens) and 1024
  for Qwen3-VL (e.g. a 1024 x 1024 image). `max_image_tokens=0` compiles the language model alone.
- `llama_cpp_policy("model-q8_0.gguf", model, mmproj="mmproj-f16.gguf")` takes the vision
  weights' types from llama.cpp's projector GGUF.

SmolVLM:

- `encode(pixels [tiles, 3, H, W])` takes the processor's tiles (2 ≤ tiles ≤ max, a symbol).
  Several images go in one `encode` as their tiles concatenated. `Generator.encode` encodes a
  single tile twice and ignores the copy (a dynamic dim's minimum is 2).
- Only whole tiles are supported, which is what the processor makes: transformers' patch position
  ids are then 0 .. n-1, and its padding-image and patch-mask logic has nothing to do.
- Metadata `image`: `kind` ("tiles"), `token` (the placeholder id), `rows` (the buffer's size),
  `rows_per_tile`, `tile` ([3, H, W]) and `max_tiles`.

Qwen3-VL:

- RoPE positions are inputs of their own (`meta["rope"] == "separate"`): text after an image
  continues from the image's start position + max(grid h, w) / 2, not from its cache row. Text
  entries take one RoPE position per token (`prefill(tokens, positions, rpos, mask)`,
  `decode(token, position, rpos1, mask1)`, `generate(first, start, rstart)`), as all three
  M-RoPE streams are equal for text; `prefill_mm` takes `mrope [3, s]`, the (t, h, w) positions
  transformers' `get_rope_index` computes. `Generator` tracks them (`r_next`).
- `encode(pixels [n, 1536], pos_idx [4, n], pos_w [4, n], vpos [n, 2])` takes one image's patches as
  the processor flattens them (n = 4 * vision_tokens, a symbol). The tables that depend on the image
  grid come from the caller: `ggml_compiler.llm.qwen3_vl_tables(grid_thw, pos_side, merge)` computes
  them as transformers does (the bilinear taps into the learned position embedding, and each patch's
  row and column for the 2-D rotary embedding). One image per `encode`; videos are not supported.
- Deepstack: the features of three vision layers are three more state buffers, added to the
  hidden states after the first three decoder layers at the image tokens.
- Metadata `image`: `kind` ("qwen3_vl"), `token`, `rows`, `patch` (1536), `merge` (2) and
  `pos_side` (48).

Other vision-language models compile as stages too: a vision encoder over a fixed image size, or
a VLM prefix that hands its KV cache to a later stage. The VLA examples run PaliGemma (pi0,
pi0.5), SmolVLM (SmolVLA) and Qwen3-VL (GR00T N1.7) this way, with grouped-query attention, RoPE
and multimodal RoPE, and image-token merging, at f32 parity with PyTorch. For a VLM whose image
tokens sit at fixed positions in the prompt (one chat template, a fixed number of images of a
fixed size), compute those positions and the position ids at compile time;
`examples/groot/export.py` shows how for Qwen3-VL.

## Optimization levels and flags

An optimization level trades speed against exactness and generality. Each level is a set of
named flags, and includes the levels below it:

| Level | Adds | Use |
|---|---|---|
| `O0` | Nothing optional: lowering, plus only the rewrites ggml needs to accept the graph | Debugging a pass, or isolating a numerical difference |
| `O1` (default) | Numerics-preserving rewrites; f32 arithmetic throughout | Parity builds |
| `O2` | Rewrites that change numerics at the ~1e-3 level | Deployment |
| `O3` | Specialization that narrows what the artifact accepts at run time (masks that leave every query a key to attend to) | Deployment: LLM decoding |

| Flag | Level | Effect |
|---|---|---|
| `simplify` | O1 | Drop identities (`x + 0`, `x * 1`, all-zero softmax masks); a slice of a concat reads its part |
| `flatten_matmul` | O1 | A 2-D weight times a batched activation is one GEMM, not a batch of small ones |
| `shared_kv_heads` | O1 | Grouped-query K/V heads are broadcast by `mul_mat` instead of repeated |
| `fold_constants` | O1 | Nodes that read only weights and constants are computed at compile time |
| `fold_batch_norm` | O1 | A per-channel affine after a convolution (frozen batch norm) folds into its kernel and bias |
| `hoist_invariants` | O1 | Loop-invariant work moves into a stage that runs once before the loop |
| `cse` | O1 | Identical nodes (the same op on the same inputs: a mask every layer rebuilds) are computed once |
| `fuse_glu` | O1 | `silu(gate) * up` (a SwiGLU MLP) is one `ggml_swiglu_split` |
| `fuse_rms_norm` | O1 | RMSNorm written out (`x * rsqrt(mean(x²) + eps)`) is one `ggml_rms_norm` |
| `strided_rows` | O1 | Matmuls, flash attention, RoPE and row writes read strided views whose rows are contiguous (a KV cache view, a permuted query) without a copy |
| `flash_attention` | O2 | Eager attention runs as `ggml_flash_attn_ext`, K/V in f16 |
| `fast_gelu` | O2 | tanh-GELU is ggml's `gelu`, which reads an fp16 table on CPU |
| `f16_kv_cache` | O2 | State that only flash attention or matmuls read (a KV cache) is stored in f16 |
| `fuse_rope` | O2 | Rotary embeddings (`x * cos + rotate_half(x) * sin`) are one `ggml_rope_ext`; ggml's angles differ by ~1e-4 at position 1000 |
| `assume_attended_rows` | O3 | Every attention query may attend to some key (a causal mask): flash attention drops its fully-masked-row guard |

```python
ggml_compiler.compile(..., opt="O2")                                        # flash attention + table GELU
ggml_compiler.compile(..., opt="O2", flags={"flash_attention": False})      # table GELU, f32 attention
ggml_compiler.compile(..., flags={"hoist_invariants": False})               # O1 without loop hoisting
```

`flash_attention` fuses eager attention (`matmul → scale → [mask | where] → softmax → matmul`)
into `ggml_flash_attn_ext` for head sizes 40, 64, 72, 80, 96, 112, 128, 256 and 512. Every
backend runs the first eight; CUDA runs 512 (Gemma 4) on Turing or newer GPUs with grouped-query
heads, and elsewhere the runtime runs that attention as matmul, softmax, matmul, accumulating in
f32 (`GGML_COMPILER_LOG=debug` says so at load). Masks stay finite, and fully masked query rows
get PyTorch's uniform softmax. Attention left unfused accumulates its two matmuls in f32 too.

A level never changes weight types; that is the [dtype policy](#weight-dtypes-and-quantization)'s
job. Choices that an export script makes, such as compiling in a prompt layout, are recorded with
`Program(..., specialized={"prompt_layout": "148 tokens, 2 images"})`. `report.json` (`options`),
the GGUF (`ggml_compiler.options`) and `ggml-compiler inspect` show the level, every flag and those
specializations.

Some choices the runtime makes per backend, so one GGUF serves both:

- A convolution with an f16 kernel runs as ggml's direct implicit-GEMM convolution on a GPU when
  its output has at least 1024 positions, and as im2col + matmul otherwise.
- Exact tanh-GELU is ggml's `gelu` on a GPU (an f32 formula there) and the formula spelled out in
  ggml ops on CPU, whose `gelu` reads an fp16 table.

Start any performance investigation with `ggml-compiler bench MODEL.gguf --profile`: it times every
node and groups the cost by op and by PyTorch source line.

## Weight dtypes and quantization

By default every weight stays `f32`. A dtype policy assigns ggml types per tensor, by glob
pattern on the parameter's name (its path on the program's root module); the first matching rule
wins:

```yaml
# policy.yaml
float: f32                 # weights no rule matches, and weights that cannot take the rule's type
fallback: [q8_0, f16]      # tried in order when a type does not fit (block size, reader)
rules:
  - match: "vlm_with_expert.vlm.model.vision_model.**"
    dtype: f16
  - match: "vlm_with_expert.vlm.model.text_model.**"
    dtype: q4_k
```

```python
# dtype_policy: a YAML file, a DTypePolicy or a dict
art = ggml_compiler.compile(program, out_dir="out/smolvla-q4", dtype_policy="policy.yaml")
```

Types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl`, `q4_k`, `q5_k`,
`q6_k`, encoded by ggml's own quantizer.

- Matmul weights and embedding tables take any type; convolution kernels take `f16`. Everything
  else (norm scales, biases) stays at `float`. A quantized type also needs the row length to be a
  multiple of its block (32; 256 for the K-quants), or the fallback applies.
- A tied weight (an embedding table that is also the output head) matches a rule under any of
  its names. A pattern ending in `@<op>` applies only where that op reads the weight
  (`model.embed_tokens.weight@get_rows`, `...@mul_mat`); readers that want different types get one
  copy of the weight per type. llama.cpp stores Qwen3's tied embedding that way, as Q4_K for
  lookups and Q6_K for the head; `ggml_compiler.llm.llama_cpp_policy` writes such rules.
- `report.json` lists every weight's final type, why, and its relative error
  (`max|w − stored| / max|w|`).
- `bf16` stores a bf16 checkpoint losslessly. `f16` is as fast on CPU and CUDA and more accurate,
  because ggml converts the activations to the weight's type before the matmul (10 mantissa bits
  against 7). Deep models can be sensitive to either: keep the sensitive part in `f32` with a
  rule (`examples/groot/policy-head-f16.yaml`).

## Importing ModelOpt-quantized models

Models quantized with NVIDIA ModelOpt (`mtq.quantize`, or a restored QAT checkpoint) compile
directly:

```python
import modelopt.torch.quantization as mtq

model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
art = ggml_compiler.compile(model, example, "out/q",
                            prequant=ggml_compiler.ModelOpt(int8="q8_0", int4="q4_0"))
```

- Each enabled weight quantizer is applied once. The fake-quant weight is stored as `int8 → Q8_0`
  and `int4 → Q4_0` (configurable, `iq4_nl` and `q4_k` too).
- When the weight sits on ModelOpt's grid with one scale per 32-element block (per-channel int8,
  block-wise int4 with blocks that are multiples of 32), it is encoded *exactly*: only the scale
  rounds to fp16. Other layouts are re-encoded by ggml's quantizer and say so in the report.
- Activation quantizers are dropped and listed; ggml quantizes activations itself inside its
  quantized matmuls. The user's model is restored afterwards.
- `report.json` → `prequant` has each weight's source quantizer, stored type and error against
  its fake-quant value, plus the compiled model's output error against the fake-quant PyTorch
  model on the example inputs.

## ONNX

```python
art = ggml_compiler.compile("act.onnx", {"state": s, "front": a, "wrist": b}, "out/act-onnx")
```

```bash
ggml-compiler compile act.onnx --out out/act-onnx --input front:1x3x480x640:f32
```

Shapes must be static, either in the file or pinned with `input_shapes` / `--input`. Example
inputs are optional (seeded random data otherwise); onnxruntime runs on them for the reference
outputs. ONNX models compile to one stage, with the same optimizations as the PyTorch path.

## Hand-tunable C++

```bash
ggml-compiler compile examples/act/export.py:program --out out/act --emit gguf,cpp --name act
```

```
out/act/
  model.gguf, report.json, example_inputs.npz, reference_outputs.npz
  example_inputs.bin         the same inputs as raw bytes, for act-check
  cpp/
    act.h  act.cpp           one ggml graph builder per stage; weights named after the module tree
    main.cpp                 act-check: runtime vs compiled builders, same GGUF and inputs
    CMakeLists.txt
```

```bash
cmake -S out/act/cpp -B out/act/build && cmake --build out/act/build -j      # builds runtime + ggml too
# or link a prebuilt runtime: -DGGML_COMPILER_RUNTIME_LIB=$PWD/build-cuda/libggml_compiler_rt.so
out/act/build/act-check out/act/model.gguf out/act/example_inputs.bin cuda
# actions    bit-identical  max|d| 0
```

The generated library exposes the runtime's C API. `act_load(gguf, &opt)` replaces
`ggml_compiler_load` and builds the stage graphs from `act.cpp` instead of the GGUF's program.
Untouched, outputs are bit-identical to the runtime on the same backend and thread count. Edit
`act.cpp` to fuse or reorder by hand; `act-check` then shows how far results move.

## C API

```c
#include "ggml_compiler_rt.h"

if (!ggml_compiler_has_backend(GGML_COMPILER_BACKEND_CUDA)) { /* built without CUDA, or no device */ }

ggml_compiler_options opt = ggml_compiler_default_options();
opt.backend   = GGML_COMPILER_BACKEND_CUDA; /* GGML_COMPILER_BACKEND_CPU, GGML_COMPILER_BACKEND_AUTO */
opt.device    = 0;
opt.n_threads = 6;                          /* CPU threads; <= 0 means all */

ggml_compiler_model * m = ggml_compiler_load("out/smolvla/model.gguf", &opt);
if (!m) { fprintf(stderr, "%s\n", ggml_compiler_last_error()); return 1; }

for (int i = 0; i < ggml_compiler_n_inputs(m); i++) {
    ggml_compiler_tensor_info info;
    ggml_compiler_input_info(m, i, &info);  /* name, dtype, ndim, shape (PyTorch order), nbytes */
}

/* A program with symbols (ggml_compiler_n_symbols > 0) takes sizes per run: bind them from an input's
   shape, or by name. Infos and nbytes then follow the binding. */
const int64_t shape[2] = {1, 37};
ggml_compiler_set_input_shape(m, "lang_tokens", 2, shape);  /* or ggml_compiler_set_symbol(m, "seq", 37) */

/* Several entry points (ggml_compiler_n_entries > 1): set any entry's inputs, then run it by name. */
/* ggml_compiler_run_entry(m, "decode"); ggml_compiler_reset(m) zeroes the state (a KV cache). */
/* ggml_compiler_metadata(m, "ggml_compiler.meta") is the program's own JSON metadata (an LLM's EOS
   ids, ...). */

ggml_compiler_set_input(m, "images.0", pixels, pixels_nbytes);
ggml_compiler_set_input(m, "lang_tokens", tokens, tokens_nbytes);
/* ... remaining inputs ... */
if (ggml_compiler_run(m) != 0) { fprintf(stderr, "%s\n", ggml_compiler_last_error()); }
ggml_compiler_get_output(m, "x", actions, actions_nbytes);

ggml_compiler_stats st;
ggml_compiler_last_stats(m, &st);                /* st.ms_total, st.n_calls */
for (int i = 0; i < ggml_compiler_n_stages(m); i++) {
    const char * name; double ms; int calls;
    ggml_compiler_stage_stats(m, i, &name, &ms, &calls);
}
ggml_compiler_free(m);
```

Every function returns 0 or a non-NULL handle on success. On failure it sets
`ggml_compiler_last_error()`. Also in the header: `ggml_compiler_quantize` /
`ggml_compiler_row_size` (ggml's encoders, used by the compiler) and `ggml_compiler_load_with` with
the `ggml_compiler_build_*` functions that generated C++ uses. `build/ggml-compiler-info MODEL.gguf
[cpu|cuda]` prints a model's inputs and outputs.

## CLI

```
ggml-compiler compile SOURCE --out DIR [options]
    SOURCE                     model.onnx, or path/to/script.py:name where name is a
                               ggml_compiler.Program (or a function returning one)
    --emit gguf|cpp|gguf,cpp   default: gguf
    --name NAME                C++ project name (default: the output directory's)
    --policy FILE              weight dtype policy (YAML)
    --prequant modelopt        import ModelOpt quantizers
    -O, --opt O0|O1|O2|O3      optimization level (default O1); -O2 also works
    -f, --flag [no-]NAME       turn one flag on or off on top of the level; repeatable
    --precision exact|fast     older spelling of -O1 | -O2
    --attention exact|flash    set the flash_attention flag
    --backends cpu,cuda        load the result on these backends now, so unsupported nodes fail here
    --input NAME:SHAPE[:DTYPE] ONNX only: pin an input's shape, e.g. image:1x3x224x224:f32

ggml-compiler inspect MODEL.gguf [--tensors]    stages, driver (loops, carries, scalars, slices),
                                                I/O, op histogram, tensor types and sizes
ggml-compiler run     MODEL.gguf --out out.npz [--inputs in.npz] [--backend B] [--threads N]
ggml-compiler bench   MODEL.gguf [--inputs in.npz] [--backend B] [--threads N] [--warmup 5]
                      [--iters 50] [--profile [--top 15]]
ggml-compiler verify  MODEL.gguf [--inputs in.npz --ref ref.npz] [--backend B] [--threads N]
                      [--tol T]
                      compares with the outputs saved at compile time: max|d| / max(1, max|ref|),
                      default tolerance 1e-4 on CPU and 1e-2 on CUDA; NaN or inf always fails;
                      exit status 1 on failure
```

`run`, `bench` and `verify` default to the `example_inputs.npz` written next to the model.

## When an op has no ggml lowering

ggml-compiler first decomposes the op into ggml primitives. If that is impossible, compilation
stops, listing every missing op once with its source line. It never inserts a silent slow path:

```
ggml-compiler: error: stage 'forward': 2 op(s) have no ggml lowering
  aten.erfinv.default                      my_model.py:88 in forward (+1 more)
  aten.atan.default                        my_model.py:88 in forward
Register a lowering with @ggml_compiler.lowering, or change the model.
```

Register a lowering in terms of ops ggml-compiler already knows. A registered op is kept whole by
the exporter, so the lowering sees it:

```python
@ggml_compiler.lowering(torch.ops.aten.hardswish.default)
def hardswish(b: ggml_compiler.Builder, x):
    return b.mul(x, b.scale(b.clamp(b.add_scalar(x, 3.0), min=0.0, max=6.0), s=1 / 6))
```

The same rule applies at load time. If the chosen backend cannot run a node, `ggml_compiler_load`
fails and lists the nodes.

## Worked examples: LeRobot policies

Each `examples/<name>/export.py` builds a `ggml_compiler.Program` from a LeRobot checkpoint, with
the stage functions written so the reference run reproduces LeRobot's own inference. Latency is
against the hand-written [vla.cpp](https://github.com/VinRobotics/vla.cpp) engines on one machine
(i5-12400F with 6 threads, RTX 3060); f32 parity is the scaled error against PyTorch on CPU.
[MODELS.md](MODELS.md) has the full results.

| Policy | Example | Program | f32 parity | Latency vs vla.cpp |
|---|---|---|---|---|
| ACT | `act` | 1 stage | 6.9e-7 | 0.96× CPU, 1.05× CUDA |
| SmolVLA | `smolvla` | prefix + 10 flow steps | 2.7e-7 | 0.92× CPU, 0.82× CUDA |
| pi0 | `pi` | prefix + 10 flow steps | 2.5e-7 | 0.98× CPU, 0.93× CUDA (f16 weights) |
| pi0.5 | `pi` | prefix + 10 flow steps | 1.8e-7 (2 layers) | 1.00× CPU, 0.91× CUDA (f16 weights) |
| GR00T N1.7 | `groot` | backbone + encode + 4 DiT flow steps | 6.6e-5 | 1.03× CPU, 0.86× CUDA (f16 weights) |
| Diffusion Policy | `diffusion` | encode + 10 DDIM or 100 DDPM steps + finish | 1.6e-7 | not in vla.cpp; 0.78× PyTorch on CPU |

Notes per example:

- `act`: `program(image_size=...)` compiles for any camera resolution.
- `smolvla`: `policy-*.yaml` for f16, bf16, Q8_0 and Q4_K/F16 mixes; `quantize_expert` quantizes
  the action expert with ModelOpt.
- `pi`: `PI_CHECKPOINT` picks a pi0 or pi0.5 checkpoint. Cameras the checkpoint lists as empty
  are left out (they are masked out of attention, so the actions are the same). Its loader reads
  the checkpoint tensor by tensor; `n_layers` builds a shallow model for tests.
- `groot`: GR00T N1.7 with a Qwen3-VL backbone. The prompt layout (number and size of images,
  prompt length) is compiled in, and so is the embodiment. `policy-head-f16.yaml` keeps the
  precision-sensitive backbone in f32.
- `diffusion`: `program(scheduler="DDIM", num_inference_steps=10)` switches the scheduler; DDPM's
  per-step noise is the `step_noise` input.

## Limitations

- Sizes are static, or symbolic within a compiled range; batch size is fixed at compile time.
- On-device generation is greedy only; other sampling runs on the host from `decode`'s logits.
- Quantization covers ggml's built-in weight types; no automatic per-layer search, no ONNX QDQ
  import.
- A node the chosen backend cannot run fails the load; there is no CPU fallback.
- CPU and CUDA are tested. Other ggml backends (Metal, SYCL, Vulkan) may work but are not
  validated.
