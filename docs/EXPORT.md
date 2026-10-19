# Writing an export script

This guide is for a coding agent that has a PyTorch implementation of a model and must compile it
into ggml C++ source with ggml-compiler. It covers how to write the export script and how to show
that the compiled program computes what the PyTorch model computes.

Read two other pages first. [USAGE.md](USAGE.md) is the API reference: `Program`, `Stage`,
`Driver`, `compile`, `Session` and the CLI. [OPS.md](OPS.md) lists the operations that compile,
with workarounds for those that do not. The scripts in `examples/*/export.py` are worked examples:

| Example | Shows |
|---|---|
| `act` | One forward pass, wrapped so that inputs and outputs are plain tensors |
| `smolvla`, `pi` | A prefix stage that fills a KV cache, then a 10-step flow-matching loop |
| `groot` | Three stages; a prompt layout fixed at compile time; a patched forward that removes `.tolist()` |
| `diffusion` | Scheduler coefficients as loop scalars; per-step noise as a sliced input |

- [What to deliver](#what-to-deliver)
- [Environment](#environment)
- [Workflow](#workflow)
  1. [Read the inference path](#1-read-the-inference-path)
  2. [Draw the boundary](#2-draw-the-boundary)
  3. [Split the model into stages](#3-split-the-model-into-stages)
  4. [Write the stage functions](#4-write-the-stage-functions)
  5. [Choose example inputs](#5-choose-example-inputs)
  6. [Check the stages against the model](#6-check-the-stages-against-the-model)
  7. [Compile a cut-down model first](#7-compile-a-cut-down-model-first)
  8. [Compile the full model](#8-compile-the-full-model)
  9. [Verify](#9-verify)
- [When the compile fails](#when-the-compile-fails)
- [Template](#template)
- [Checklist](#checklist)

## What to deliver

1. **An `export.py`** that defines `program()`. The function takes no required arguments (it reads
   its configuration from keyword defaults or environment variables) and returns a
   `ggml_compiler.Program`. Then this command

   ```bash
   ggml-compiler compile export.py:program --out out/NAME --emit gguf,cpp --name NAME
   ```

   writes:

   ```
   out/NAME/
     model.gguf                 weights + the compiled program
     cpp/NAME.cpp, cpp/NAME.h   one ggml graph builder per stage
     cpp/main.cpp               NAME-check: the builders against the runtime
     cpp/CMakeLists.txt
     report.json                ops per stage, each weight's stored type, inputs and outputs
     example_inputs.npz, example_inputs.bin, reference_outputs.npz
   ```

   `NAME.cpp` is the ggml source code. Each line names the PyTorch module and source line it
   came from.

2. **Evidence that the program is correct**:
   - the stage functions reproduce the model's own inference in PyTorch ([step 6](#6-check-the-stages-against-the-model));
   - `ggml-compiler verify` passes on CPU and on CUDA ([step 9](#9-verify));
   - `NAME-check` reports `bit-identical` on both backends.

3. **The I/O contract** the caller needs. For each input and output, give its name, shape, dtype
   and units, plus the preprocessing the caller must do. Also list everything fixed at compile
   time.

## Environment

Run the export in the environment where the original model runs: its library versions define
the reference outputs. ggml-compiler needs `torch>=2.11`, `numpy`, `gguf`, `pyyaml` and `sympy`.
Install it with `pip install ggml-compiler`, which includes a CPU runtime. Alternatively, install
from a checkout whose runtime is built (`build/` or `build-cuda/`), which you need for CUDA.

Check `torch.__version__` before installing. If the environment pins other packages, install with
`--no-deps` and add the missing dependencies by hand; don't let pip silently replace the model's
torch. Don't modify an environment that other people or jobs share without asking.

## Workflow

### 1. Read the inference path

Find the function that produces the model's output in deployment, for example `select_action`,
`predict_action_chunk`, `generate`, or `forward` in eval mode. Read it, and everything it calls,
from top to bottom. Write down:

- **Preprocessing** before the network (resize, crop, scaling, normalization, tokenization,
  padding), and **post-processing** after it (de-normalization, slicing, clipping).
- **Loops**: denoising or flow-matching steps, autoregressive decoding, recurrent unrolls. Note
  each loop's trip count and its per-step values (timesteps, scheduler coefficients).
- **Randomness**: `torch.randn`, `torch.rand`, sampling.
- **Values that come from the data but are fixed in deployment**: the image size, the number of
  cameras, the prompt layout, an embodiment id, attention masks.
- **Python code that reads tensor values**: `.item()`, `.tolist()`, `if tensor...`,
  `.nonzero()`, boolean indexing. torch.export cannot trace these.
- **How the weights are loaded**, and the dtype inference runs in. The reference must use the
  numerics of deployment (usually f32, even for a bf16 checkpoint).

Run the original once on a fixed input and keep its output. Everything after this is compared
against it.

### 2. Draw the boundary

A program takes tensors in and gives tensors out. Decide what is compiled and what stays with the
caller:

| The caller keeps | The program computes |
|---|---|
| Decoding images and resizing them to the compiled size; tokenizers and chat templates | Pixel scaling and normalization (`x / 255`, `(x - mean) / std`) |
| Generating random numbers, passed in as an input | The flow-matching or diffusion updates that use that noise |
| Choosing a task, prompt or embodiment by name | Lookups by index in a table (`F.embedding`) |
| Anything whose shape depends on the data | State normalization and padding; action de-normalization |

Put as much as you can in the program, because what the program computes the caller cannot get
wrong. Example inputs fix the shapes and the batch size; [symbolic dims](USAGE.md#symbolic-dims)
give a range where one is needed.

Record what you fixed at compile time in `Program(specialized={...})`. Record what the caller
needs at run time (task names in index order, units, the image format) in
`Program(metadata={...})`. Metadata is stored in the GGUF and read back with `Session.metadata`
or `ggml_compiler_metadata`.

### 3. Split the model into stages

| Model | Program | Example |
|---|---|---|
| One forward pass | `Program.single(module, example_inputs, input_names, output_names)` | `act` |
| Encoder plus an iterative head (flow matching, DDIM) | An `encode` stage, a `step` stage in `loop(n, body, carry={"x": "noise"}, scalars={"t": [...]})`, and an optional `finish` stage | `smolvla`, `pi`, `groot` |
| Fresh noise at every step (DDPM) | `loop(..., slices={"z": "step_noise"})` over an input whose leading dim is n | `diffusion` |
| Decoder language model or VLM with a KV cache | `ggml_compiler.causal_lm`, not a hand-written driver | [USAGE.md](USAGE.md#llms-and-vlms) |

Each stage becomes one ggml graph, built once at load. Split the model where it loops, and where
one part runs a different number of times from another. Don't unroll a loop into one stage. Work
inside the loop that does not depend on the loop's values is moved out automatically, into a
`<stage>.invariant` stage. Stages pass tensors, or pytrees of tensors, to each other, and these
stay on the device.

### 4. Write the stage functions

Write a class with one method per stage. The simplest form is an `nn.Module` that holds every
module the stages call, passed as `Program(root=...)`.

- **Reuse the original modules and the original loader.** Call the upstream submodules
  (`self.head.action_encoder(...)`) instead of reimplementing them. Take them from the object the
  upstream code builds, so the weights and preprocessing are the deployed ones. If the loader
  returns a closure or a decorated function, reach the module through `inspect.unwrap(fn)` and
  `fn.__closure__` (match names in `fn.__code__.co_freevars`). Assert anything you rely on, such
  as the scaling constants a closure captured.
- **Don't edit the upstream source.** If a forward must change (to remove `.tolist()`, say),
  replace the method on the instance in the export script, as `examples/groot/export.py` does.
- **Follow the original line by line**: the same ops in the same order, the same constants, the
  same dtype. Give each deviation a comment saying why it is equivalent. Deviations include a
  rewrite for a missing op, and a step dropped because it is the identity at the compiled shape.
  Step 6 checks every one.
- **Parameters versus buffers.** `nn.Parameter`s become weights named after their path on the
  root, such as `head.model.blocks.0.attn.to_q.weight`. Those names appear in the C++ and are what
  a dtype policy matches. Buffers and plain tensor attributes become anonymous f32 constants
  (`c.0`, `c.1`, ...), and work on them is done at compile time. Store normalization statistics
  as buffers. Store a large table that you may want in f16 or quantized as a frozen
  `nn.Parameter`.
- **Fixed values are computed in `__init__`**, as Python numbers or constant tensors: embodiment
  ids, masks with a fixed layout, timestep tables, image-token ranges.
- **Stages return tensors**, or tuples, lists or dicts of them. An integer result must stay inside
  its stage ([OPS.md](OPS.md#rules-for-every-op)).
- **Batch identical work.** Two cameras through one encoder can be concatenated along the batch
  dim and run in a single pass.
- **The step stage includes the update**: `return x + dt * v`.
- **Timesteps.** If the original passes an integer timestep that every use converts to float,
  pass it as a float loop scalar. Check every use before doing this.

### 5. Choose example inputs

Example inputs fix every shape, and the reference outputs are computed on them. Use real data
when you have it: a recorded frame, a real prompt, a real robot state. Otherwise use
deterministic synthetic data in the real value range. Don't use zeros, which hide errors. Make
noise with a seed. The shapes must be the deployment shapes.

### 6. Check the stages against the model

Before compiling, run the stage functions in PyTorch the way the driver will (stage, loop of
steps, stage). Compare the result with the original inference function on the same inputs. To
make the original draw the same noise as your input, seed just before calling it:

```python
torch.manual_seed(seed)
noise = torch.randn(1, horizon, action_dim)   # the program's noise input
...
torch.manual_seed(seed)                        # the original's first draw is the same noise
want = policy.predict_action_chunk(obs)
```

This works only if nothing uses the random generator between the seed and that draw inside the
original call. Dropout in eval mode does not. Read the code to confirm.

Expect the outputs to be bit-identical. Otherwise they should differ only by float rounding
(about 1e-6 relative) for a reason you can name, such as an op you rewrote or a batched
convolution. A larger difference is a bug in the stages, so fix it before compiling. Save the
original's outputs under the program's output names, for example
`np.savez("upstream_outputs.npz", actions=want.numpy())`, for step 9.

### 7. Compile a cut-down model first

Loading and exporting a full model can take many minutes. First compile a version with only a
few layers; random weights are fine. It runs every op type, so export and lowering errors show up
in seconds:

```python
art = ggml_compiler.compile(program_from_policy(load(n_layers=2)), out_dir="/tmp/NAME-shallow",
                            emit=("gguf", "cpp"), backends=["cpu", "cuda"])
```

Most upstream configs have a layer count, and `examples/pi` and `examples/groot` take `n_layers`.
`backends=` loads the result on each backend, so a node a backend cannot run fails at compile
time. Use the [table below](#when-the-compile-fails) to fix errors, and repeat until the compile
succeeds and `verify` passes.

### 8. Compile the full model

- **Disk.** An f32 GGUF takes about 4 bytes per parameter. Write `--out` to a disk with room.
- **Memory.** The compiler writes weights one at a time, but the PyTorch model stays in memory in
  f32 for the export and the reference run. Plan for about 1.5 times the model's f32 size in RAM.
  For a model close to the machine's RAM, don't pass `backends=` in the compiling process. Load
  the GGUF in a separate process (`ggml-compiler verify`).
- **Precision.** Keep the default `opt="O1"`, whose arithmetic is f32, until parity is
  established. Speed options come after: `O2`, and f16 or quantized weights through a dtype
  policy ([USAGE.md](USAGE.md#optimization-levels-and-flags)). Policy rules match parameter names.
  A weight transformed at compile time keeps its name with a `:suffix`, so match both `**.weight`
  and `**.weight:*` (`examples/groot/policy-f16.yaml`).

Check the stage summary that `compile` prints, or `ggml-compiler inspect`. Loops should appear as
stages called n times, not as unrolled graphs.

### 9. Verify

```bash
ggml-compiler verify out/NAME/model.gguf --backend cpu     # against the PyTorch stages; tolerance 1e-4
ggml-compiler verify out/NAME/model.gguf --backend cuda    # tolerance 1e-2
cd out/NAME && ggml-compiler verify model.gguf --backend cpu --ref /path/to/upstream_outputs.npz
```

The last command compares against the original model, not your stages. The reported error is
scaled: `max|d| / max(1, max|ref|)`.

- **Expected on CPU:** about 1e-7 to 1e-6.
- **Expected on CUDA:** about 1e-4 to 1e-3. ggml's CUDA backend runs f32 matmuls as TF32, and the
  error builds up over deep models and loops.
- **If CPU error is large**, the program has a bug, often in a rewrite or a constant.
- **If only CUDA error is large**, look first at how many matmuls are chained.

Don't loosen a tolerance to make a check pass.

One example input does not test every path. Write more `(inputs, upstream outputs)` pairs so that
every discrete input is covered: every task index, every camera, every prompt. Check each pair:

```bash
ggml-compiler verify model.gguf --inputs case1_inputs.npz --ref case1_upstream.npz
```

Finally, build the generated C++ and check it against the runtime:

```bash
cmake -S out/NAME/cpp -B out/NAME/build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_COMPILER_RUNTIME_LIB=/path/to/ggml-compiler/build-cuda/libggml_compiler_rt.so
cmake --build out/NAME/build -j
out/NAME/build/NAME-check out/NAME/model.gguf out/NAME/example_inputs.bin cpu    # bit-identical  max|d| 0
out/NAME/build/NAME-check out/NAME/model.gguf out/NAME/example_inputs.bin cuda
```

Without `GGML_COMPILER_RUNTIME_LIB`, the project builds the runtime and ggml from the checkout
that generated it.

## When the compile fails

| Symptom | Cause | Fix |
|---|---|---|
| `GuardOnDataDependentSymNode: Could not guard on data-dependent expression`, raised at an `if` | Python branches on a tensor's value. torchvision's `Normalize`, for example, checks `(std == 0).any()` | Write the op out (`(x - mean) / std` with constant tensors), or decide the branch in `__init__` |
| The same error at `.tolist()`, `.item()` or `.nonzero()` | A size computed from tensor values, such as Qwen-VL's `cu_seqlens` | Compute the sizes in `__init__` from the example layout, as Python ints, and patch the forward (`_static_vision_attention` in `examples/groot/export.py`) |
| `stage 'X': N op(s) have no ggml lowering`, listing ATen ops and source lines | The op has no lowering | Look the op up in [OPS.md](OPS.md). Its workarounds include concatenating zeros instead of `F.pad`, and `x.flatten(2).mean(-1)` instead of `AdaptiveAvgPool2d(1)`. Otherwise rewrite with ops that compile, or [register a lowering](OPS.md#adding-a-lowering-for-a-missing-op) |
| `mean over dims [...] (only the last dim is supported)` | A reduction over a dim that is not the last, or over several dims | Move the dim last, or flatten the reduced dims into the last one |
| An intermediate with more than 4 dims above size 1 | ggml tensors have 4 dims | Reshape so that at most 4 dims are larger than 1 |
| A stage output's dtype fails to compile | An integer result returned from a stage | Keep it inside the stage |
| The GGUF fails to load with "program JSON does not parse" | A comparison with `inf` (`torch.logsumexp` does one) | [OPS.md, known issues](OPS.md#known-issues) |
| `ValueError: tuple.index(x): x not in tuple` while reading a closure variable | The function is wrapped by a decorator such as `@torch.no_grad()` | Call `inspect.unwrap(fn)` first |
| Step 6 passes but CPU `verify` fails | ggml computes something differently from PyTorch | Add intermediate values as extra stage outputs for the moment, compile, and compare stage by stage to find the first one that differs |

## Template

A flow-matching policy with an encoder, a step loop and a de-normalization stage. `MyPolicy`,
its attributes and the shapes are placeholders: replace them with the upstream names, and copy
each formula from the upstream code.

```python
"""<Model> as a ggml-compiler program:
`ggml-compiler compile path/to/export.py:program --out out/NAME --emit gguf,cpp --name NAME`.

encode: camera image and robot state -> conditioning tokens.
step:   one Euler step of the flow-matching head (the driver runs it `n` times).
finish: the action chunk, de-normalized.

The caller resizes the image to 224x224 (RGB, [0, 255]) and supplies standard-normal noise.
"""

import os

import numpy as np
import torch

import ggml_compiler as gc

CHECKPOINT = os.environ.get("MYMODEL_CHECKPOINT", "/path/to/checkpoint")


def load(checkpoint: str = CHECKPOINT, n_layers: int | None = None):
    """The original model, loaded by its own code, in f32 and eval mode. `n_layers` cuts the
    head for quick tests."""
    from mymodel import MyPolicy

    policy = MyPolicy.from_pretrained(checkpoint)
    if n_layers is not None:
        policy.model.head.blocks = policy.model.head.blocks[:n_layers]
    return policy.float().eval().requires_grad_(False)


class Stages(torch.nn.Module):
    """MyPolicy.predict_action_chunk, one method per stage."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy.model  # holds every parameter the stages read
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.register_buffer("action_mean", f32(policy.stats["action_mean"]))
        self.register_buffer("action_std", f32(policy.stats["action_std"]))
        self.n = policy.config.num_steps
        self.horizon = policy.config.n_action_steps

    def encode(self, image, state):
        x = (image / 255.0 - self.model.pixel_mean) / self.model.pixel_std  # MyPolicy.preprocess
        return self.model.encoder(x, state)

    def step(self, x, t, cond):
        v = self.model.head(x, t, cond)  # MyPolicy.sample, one iteration
        return x + (1.0 / self.n) * v

    def finish(self, x):
        return x[:, : self.horizon] * self.action_std + self.action_mean

    def timesteps(self) -> list[float]:
        return [i / self.n for i in range(self.n)]  # as MyPolicy.sample computes them


def example_inputs(seed: int = 0):
    frame = np.load(os.environ.get("MYMODEL_FRAME", "recorded_frame.npz"))  # real data
    image = torch.from_numpy(frame["image"]).float().permute(2, 0, 1)[None]  # [1, 3, 224, 224]
    state = torch.from_numpy(frame["state"]).float()[None]
    torch.manual_seed(seed)
    noise = torch.randn(1, 50, 7)  # [1, chunk, action_dim], as MyPolicy.sample draws it
    return image, state, noise


def program_from_policy(policy, seed: int = 0) -> gc.Program:
    s = Stages(policy).eval()
    image, state, noise = example_inputs(seed)
    with torch.no_grad():
        cond = s.encode(image, state)
    return gc.Program(
        root=s,
        stages={
            "encode": gc.Stage(s.encode, (image, state)),
            "step": gc.Stage(s.step, (noise, torch.zeros(1), cond)),
            "finish": gc.Stage(s.finish, (noise,)),
        },
        driver=gc.Driver(
            inputs=["image", "state", "noise"],
            body=[
                gc.call("encode", ["image", "state"], ["cond"]),
                gc.loop(s.n, carry={"x": "noise"}, scalars={"t": s.timesteps()},
                        body=[gc.call("step", ["x", "t", "cond"], ["x"])]),
                gc.call("finish", ["x"], ["actions"]),
            ],
            outputs=["actions"],
        ),
        specialized={"image_size": "224x224"},
        metadata={"image": "RGB, NCHW, float32 in [0, 255]", "units": "..."},
    )


def program(checkpoint: str = CHECKPOINT) -> gc.Program:
    return program_from_policy(load(checkpoint))


def check_stages(checkpoint: str = CHECKPOINT, seed: int = 0) -> None:
    """Step 6: the stages, run as the driver runs them, against the original inference."""
    policy = load(checkpoint)
    s = Stages(policy).eval()
    image, state, noise = example_inputs(seed)
    with torch.no_grad():
        cond = s.encode(image, state)
        x = noise
        for t in s.timesteps():
            x = s.step(x, torch.tensor([t]), cond)
        got = s.finish(x)
        torch.manual_seed(seed)  # the original's first draw is the same noise
        want = policy.predict_action_chunk(image, state)
    print("max|d|", (got - want).abs().max().item(), "bit-identical", torch.equal(got, want))
    np.savez("upstream_outputs.npz", actions=want.numpy())
```

## Checklist

- [ ] `ggml-compiler compile export.py:program --emit gguf,cpp` runs with `program()` called
      without arguments.
- [ ] In PyTorch, the stages match the original inference bit for bit, or differ only by rounding
      you can explain.
- [ ] `verify` passes on CPU (1e-4) and CUDA (1e-2) against the stages, and on CPU against the
      original model's saved outputs.
- [ ] Every discrete input value (task, camera, prompt) is checked at least once.
- [ ] `NAME-check` reports `bit-identical` on CPU and CUDA.
- [ ] `specialized` lists everything fixed at compile time, and `metadata` holds what the caller
      needs.
- [ ] The I/O contract is written down: names, shapes, dtypes, units and the caller's
      preprocessing.
- [ ] The upstream code is unmodified, and every deviation in the stages has a comment.
