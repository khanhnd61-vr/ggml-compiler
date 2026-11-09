# Serving a compiled policy

A compiled GGUF holds everything the runtime needs to run a model: weights, the op graphs and the
driver. It does not hold what a robot needs around the model: resizing and normalizing camera
images, padding the instruction's tokens, normalizing the state, sampling the initial noise and
un-normalizing the action chunk. Every example in this repository leaves those to "the caller".

The **serving manifest** makes a host the caller. It is a small JSON document under the key `vla`
of the program's metadata that names the program's inputs and outputs by role (camera, tokens,
state, noise, actions) and carries the preprocessing of each. A host that understands the manifest
serves any compiled policy without code written for that model.
[vla.cpp](https://github.com/VinRobotics/vla.cpp)'s `vla-server` is such a host: it detects a
ggml-compiler GGUF, reads the manifest, and answers the same requests as for its built-in models,
on every ggml backend vla.cpp builds.

- [The contract](#the-contract)
- [Writing the manifest](#writing-the-manifest)
- [Manifest reference](#manifest-reference)
- [Building the runtime inside a host](#building-the-runtime-inside-a-host)
- [Workflow](#workflow)

## The contract

Three things make up the contract between the compiler and a host, and all three are versioned:

| Part | Where | Versioned by |
|---|---|---|
| The runtime C API | `runtime/include/ggml_compiler_rt.h` | the ggml-compiler release |
| The GGUF program | `ggml_compiler.program`, `ggml_compiler.format_version` | `format_version` (the runtime reads 1 to 3) |
| The serving manifest | `ggml_compiler.meta` → `"vla"` | `vla.version` (1) |

Plus one invariant: the host and the runtime share **one ggml**. The runtime is built against the
host's `ggml` target ([below](#building-the-runtime-inside-a-host)), never loaded as a second copy
next to it. The runtime is tested with the llama.cpp tag in `runtime/CMakeLists.txt`
(`GGML_COMPILER_LLAMA_TAG`); the host's ggml has to come from the same tag.

What the manifest leaves to the program, on purpose: everything whose shape and arithmetic the
model fixes. Pixel scaling and channel normalization could run in the graph too, and
[EXPORT.md](EXPORT.md#2-decide-the-io-contract) says to prefer that. The manifest's image `mean`
and `std` exist for programs that take normalized pixels, as the LeRobot policies do; a program
that takes `[0, 1]` pixels sets them to `0` and `1`.

What the manifest does not cover, by design: tokenization (hosts take token ids), prompt templates,
image tiling, and any per-model logic. A host may offer its own extension point for those.

## Writing the manifest

`ggml_compiler.vla` builds and checks it:

```python
import ggml_compiler
from ggml_compiler import vla

manifest = vla.manifest(
    images=[vla.image("front", 480, 640, mean=vla.IMAGENET_MEAN, std=vla.IMAGENET_STD),
            vla.image("wrist", 480, 640, mean=vla.IMAGENET_MEAN, std=vla.IMAGENET_STD)],
    state=vla.state("state", mean=state_mean, std=state_std),
    actions=vla.actions("actions", chunk=100, mean=action_mean, std=action_std),
)
program = ggml_compiler.Program.single(model, example_inputs, input_names=["state", "front", "wrist"],
                                       output_names=["actions"], metadata={"vla": manifest})
```

`compile` checks a program that carries a manifest: every input of the default entry point must
be named by exactly one role, and the actions output must exist. A flow-matching policy adds
`lang=vla.lang("lang_tokens", 48, mask="lang_masks")` and `noise=vla.noise("noise")`; a camera the
program can mask takes `mask="img_masks.0"`.

[`examples/act/export.py`](../examples/act/export.py) fills the manifest from the checkpoint's
LeRobot processor pipelines (`policy_preprocessor.json` names the step whose safetensors file holds
`observation.state.mean` and the rest).

## Manifest reference

Version 1. Shapes are PyTorch order with a batch of 1. The host converts each tensor to the dtype
the program declares for that input (`f32`, `f16`, or `i32`), so a 0/1 mask may be either float or
int in the program.

```json
{
  "version": 1,
  "images": [
    {"input": "front", "height": 480, "width": 640, "layout": "NCHW", "fit": "exact",
     "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "mask": "front_mask"}
  ],
  "lang":    {"tokens": "lang_tokens", "mask": "lang_masks", "max_len": 48, "pad_id": 0},
  "state":   {"input": "state", "dim": 32, "real_dim": 6, "norm": "mean_std",
              "mean": [...], "std": [...], "eps": 1e-8},
  "noise":   {"input": "noise"},
  "actions": {"output": "actions", "chunk": 50, "dim": 32, "real_dim": 6, "norm": "mean_std",
              "mean": [...], "std": [...], "eps": 1e-8}
}
```

| Role | Program tensor | Host behaviour |
|---|---|---|
| `images[i]` | `[1, 3, height, width]` float input | The i-th camera of the request. Pixels scaled to `[0, 1]`, then `(x - mean[c]) / std[c]` per channel, laid out `NCHW`. `fit: "exact"` refuses an image of another size; `"resize"` scales it bilinearly (no antialiasing, PyTorch's `align_corners=False`). `mask`, if present, is a `[1]` input set to 1 when the camera was sent and 0 (with a zero image) when it was not; without masks the request must carry every camera. |
| `lang` | `tokens`: `[1, max_len]` int input; `mask`: `[1, max_len]` 0/1 | Token ids padded with `pad_id`; more than `max_len` tokens is an error. Absent for a policy without language (ACT). |
| `state` | `[1, dim]` float input | The request's state, `real_dim` values normalized and the rest zero. `norm`: `"mean_std"` is LeRobot's `(x - mean) / (std + eps)`; `"min_max"` is `2 (x - min) / (max - min + eps) - 1`; `"none"` passes values through. Statistics have `real_dim` values. |
| `noise` | float input of any shape | Filled with `N(0, 1)` samples when the request carries no noise; a request's noise must have the input's element count. |
| `actions` | `[1, chunk, dim]` float output | Un-normalized with the inverse of the state formulas (`y * (std + eps) + mean`, or `(y + 1) / 2 * (max - min + eps) + min`) over the first `real_dim` columns; returned as `chunk` rows of `dim`. |

Unknown keys are ignored, so a later version can add fields; a host refuses a `version` it does
not know.

## Building the runtime inside a host

`runtime/` is a CMake project that fetches llama.cpp for its ggml and links it statically, which is
right for the Python wheel and wrong inside an application that has ggml already: two copies of
ggml in one process means two backend registries and symbol interposition between them. With
`GGML_COMPILER_EXTERNAL_GGML=ON` the runtime links the enclosing project's `ggml` target instead:

```cmake
# After the host's own ggml (llama.cpp) is available as the target `ggml`:
set(GGML_COMPILER_EXTERNAL_GGML ON CACHE BOOL "" FORCE)
set(GGML_COMPILER_GGML_SOURCE_DIR ${llama_SOURCE_DIR} CACHE PATH "" FORCE)  # llama.cpp checkout: ggml/include, vendor/
set(GGML_COMPILER_EXTERNAL_GGML_TAG b11223 CACHE STRING "" FORCE)           # the tag that ggml came from
set(GGML_COMPILER_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ggml_compiler
    GIT_REPOSITORY https://github.com/khanhnd61-vr/ggml-compiler
    GIT_TAG        <release tag>
    SOURCE_SUBDIR  runtime)
FetchContent_MakeAvailable(ggml_compiler)
target_link_libraries(my_host PRIVATE ggml_compiler_rt)
```

The configure fails when `GGML_COMPILER_EXTERNAL_GGML_TAG` is not the tag the runtime was tested
with (`GGML_COMPILER_ALLOW_GGML_MISMATCH=ON` turns that into a warning). The CUDA define follows the
host's `GGML_CUDA`.

A host that picks its own device (a SYCL, Metal or OpenVINO backend the host already initializes)
loads a model onto it with `ggml_compiler_load_on`, which takes the `ggml_backend_t` and owns it
from then on:

```c
ggml_backend_t be = /* the host's backend */;
ggml_compiler_model * m = ggml_compiler_load_on("model.gguf", be, NULL, 0);
const char * meta = ggml_compiler_metadata(m, "ggml_compiler.meta");  /* JSON; "vla" is the manifest */
```

Every node of every stage is checked against the backend at load (`ggml_backend_supports_op`), so
an op the backend lacks is a named load error rather than a wrong answer.

## Workflow

For someone who wants to serve a model this project does not list:

1. Write `export.py` from the nearest example, including the manifest.
2. `ggml-compiler compile export.py:program --out out/model --emit gguf` (add `--policy` for
   reduced-precision weights once f32 parity is established).
3. `ggml-compiler verify out/model/model.gguf --backend cpu`, then on the deployment backend.
4. Hand `out/model/model.gguf` to the host. With vla.cpp:
   `vla-server out/model/model.gguf`; clients are unchanged.

[VLA_CPP.md](VLA_CPP.md) walks through these steps for CoRP, a policy neither project knew, down to
the chunk coming back from `vla-server`.

Keep `--emit cpp` for what it is for: a standalone binary that carries the model's graphs without
the JSON interpreter. The generated C++ is bit-identical to the runtime's own graphs, so it buys a
server nothing, and a per-model shared library would bring back the two-ggml problem.
