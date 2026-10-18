# ggmlc architecture

ggmlc is a compiler from PyTorch (through `torch.export`) or ONNX to ggml. It owns a small IR whose
ops sit close to ggml's, and leaves the kernels to ggml. This document describes how a model
becomes a GGUF, how the runtime executes it, and which ggml behaviours the design works around.

- [Pipeline](#pipeline)
- [Source layout](#source-layout)
- [The IR](#the-ir)
- [Frontends](#frontends)
- [Passes](#passes)
- [Weight storage and the dtype policy](#weight-storage-and-the-dtype-policy)
- [The GGUF artifact](#the-gguf-artifact)
- [The runtime](#the-runtime)
- [Generated C++](#generated-c)
- [ggml constraints the design works around](#ggml-constraints-the-design-works-around)
- [Verification](#verification)

## Pipeline

```
nn.Module + stage fns + example inputs              model.onnx
        │ torch.export, per stage                        │ onnx + shape inference
        ▼                                                ▼
  frontend/torch_export.py                       frontend/onnx.py
  (+ frontend/modelopt.py)                               │
        └───────────────► IR Program ◄───────────────────┘
                 (shared weight table, stages, driver)
                               │ passes: lower → simplify → [fuse attention] → fold → hoist
                               │         → squeeze (≤4-D) → layout (cont) → dce
                               │ dtype policy
               ┌───────────────┴────────────────┐
     emit/gguf_runtime.py                   emit/cpp.py
     model.gguf (weights + program JSON)    <name>.cpp/.h + CMakeLists.txt
               │                                │
               └──► runtime/ libggmlc_rt ◄──────┘  generated code links the same runtime
                    (C ABI: load, I/O, driver, backends)  for loading, buffers and the driver
```

A compile also runs the whole program once in PyTorch (or onnxruntime) on the example inputs and
saves those inputs and outputs next to the GGUF, so every artifact can be verified later.

## Source layout

```
ggmlc/
  api.py       Program, Stage, Driver, call, loop: what the user hands to compile
  llm.py       causal_lm (a Hugging Face LM with a KV cache; per-architecture adapters: SmolVLM and
               Qwen3-VL add encode + prefill_mm over an image-row buffer), Generator, bench,
               llama_cpp_policy, qwen3_vl_tables
  options.py   optimization levels O0-O3 and the flags they set
  compiler.py  compile(), Report, Artifact, the ModelOpt report, C++ emission entry
  ir/          graph.py (TensorType, Value, Param, Weight, Const, Node, Stage), sym.py (symbolic
               dims), ops.py (registry: arity, shape inference, PyTorch reference per op),
               builder.py, printer.py, interp.py (runs a stage with the references), program.py
               (ProgramIR, CallOp, LoopOp)
  frontend/    torch_export.py, onnx.py, modelopt.py
  lower/       aten.py (ATen -> IR), onnx.py (ONNX -> IR, mostly through aten.py), registry.py
  passes/      lower_tier2.py, fold.py (simplify and its fusions, fold), fuse_rope.py,
               fuse_attention.py, cse.py, hoist.py, state_dtype.py (f16 KV cache), squeeze.py,
               layout.py, dce.py, __init__.py (rewrite helper, the pipeline)
  quant/       types.py (storage types), policy.py (DTypePolicy, legality, fallback),
               encode.py (exact grid encoders for ModelOpt weights)
  emit/        gguf_runtime.py (program JSON, weight encoding, GGUF writer), cpp.py
  runtime/     session.py (ctypes bindings: Session, quantize, has_backend)
  verify/      parity.py (the parity metric)
  cli.py
runtime/
  include/     ggmlc_rt.h (C API, builder hooks), ggmlc_ops.h (multi-call op expansions shared
               by the runtime and generated C++)
  src/         program.cpp (JSON -> specs), ops.cpp (IR op -> ggml, with pre-checks),
               ggmlc_rt.cpp (loading, value buffers, call sites, driver, profiler, C API)
  tools/       ggmlc_info.c (plain C, keeps the header C-clean)
  CMakeLists.txt               fetches llama.cpp b11223 and builds its ggml only
```

## The IR

A **program** is a weight table, named stages, a driver, and a table of driver values (every
tensor the driver passes around, each a single tensor once pytrees are flattened to dotted names
such as `kv.3.1`).

A **stage** is an SSA graph of nodes over parameters, weights and constants. Every value has a
`TensorType(shape, dtype)`. Shapes are stored in **PyTorch order**, outermost dim first; only the
emitters reverse them into ggml's `ne[]` (innermost first) and translate dim indices,
permutations and convolution parameters (ggml's dim 0 is the width).

A dim is an int or a **symbolic** expression (`ir/sym.py`): a sympy expression over named
positive integer symbols, each with a compiled range in `ProgramIR.symbols`. Python's `==` on
expressions is structural, which is what shape checks want (`seq` is never equal to 1 or to
`seq + 1`), so most shape logic runs unchanged. Order comparisons go through `sym.definitely`,
which decides a relation for every binding in range by evaluating it at the corners of the
symbols' ranges (exact for the multilinear expressions shapes are made of). Shape inference
rejects a shape only when it is wrong for every binding; attributes (reshape targets, slice starts
and lengths) may be symbolic too, and the interpreter binds the symbols from its arguments'
shapes and evaluates them before calling the references.

Ops come in two tiers:

- **Tier 1** maps to one ggml call or a fixed expansion of a few (`ggmlc_ops.h`): `add`, `sub`,
  `mul`, `div`, the unary ops (`relu`, `gelu`, `gelu_erf`, `gelu_tanh`, `silu`, `tanh`,
  `sigmoid`, `exp`, `log`, `sqrt`, `sin`, `cos`, `softplus`, ...), `scale`, `clamp`, `step`,
  `cumsum`, `mean`, `sum_rows`, `norm`, `rms_norm`, `soft_max`, `mul_mat`, `reshape`, `permute`,
  `repeat`, `concat`, `view`, `cont`, `cast`, `get_rows`, `argmax`, `im2col`, `conv_2d`,
  `pool_2d`, `flash_attn`.
- **Tier 2** holds source-level ops the passes still want to recognise (`linear`, `layer_norm`,
  `sdpa`); `lower_tier2` rewrites them into tier 1. Emitters accept tier 1 only.

Each op registers shape/dtype inference and a PyTorch reference implementation. `ir/interp.py`
runs any stage with those references, which is how every pass is tested: interpreter output
before and after the pass must agree.

ggml has no bool type and indexes with int32. Integer values are `i32`; booleans are `f32` 0/1,
and the frontend records which values were bools so lowerings that care (an attention mask means
something else as a bool than as an additive float) can tell.

## Frontends

### torch.export

Each stage is a function or bound method using the root module's parameters. It is wrapped in a
thin `nn.Module` so `torch.export.export` sees those parameters, and they are mapped back to the
root's names by identity. Weights are deduplicated across stages by the identity of the source
tensor.

- Decompositions run down to Core ATen, except for ops with a better lowering (`linear`,
  `layer_norm`, `scaled_dot_product_attention`, and any op the user registered a lowering for),
  which are kept whole.
- Subgraphs that depend only on compile-time values (RoPE tables, sinusoidal embeddings, masks of
  static lengths) are evaluated at compile time and stored as constants. Ops that read only
  weights (a transpose, `1 + w`) become derived weights, except free views.
- Pytree inputs and outputs (a camera list, a KV cache) flatten into dotted names.
- One tensor object given as two example inputs would be traced as one input; the frontend copies
  repeated example tensors first.
- Every op without a lowering is collected, with its source line, into one error.
- `Stage(dynamic_shapes=...)` goes to `torch.export` as is. Torch names its shape symbols itself
  (`s11`); the frontend pairs each declared input dim with its placeholder and renames the symbol
  after the user's `Dim`, with torch's range for it (torch raises a minimum of 0 or 1 to 2).
  Sizes then arrive as sympy expressions in the traced metadata, `sym_size` nodes read a value's
  dim, and size arithmetic (`operator.mul`, `torch.sym_max`, ...) is evaluated symbolically.
  Nothing of symbolic size is constant-folded. A driver value takes its type from the stage
  signature it meets first, which keeps its symbolic dims.
- The driver is first run in PyTorch on the example inputs (loops included, per-iteration scalars
  and slices bound as the runtime will), which gives the reference outputs and the pytree shape of
  every driver value.

IR weights are views of the model's parameters, so a multi-GB model is not held twice. The
exception is ModelOpt import, which writes fake-quant weights into the parameters for the export
and restores them afterwards; there the weights are copies.

### ONNX

`onnx.shape_inference` with the input dims pinned gives every value a static type. Float
initializers become weights (named by the initializer, so dtype policies written for the PyTorch
path still match), scalars become constants. Nodes whose inputs are all constant are evaluated
with onnx's reference evaluator. The rest lower through `lower/onnx.py`, which mostly delegates
to the ATen lowerings so both frontends share semantics. The reference outputs come from
onnxruntime.

### ModelOpt

Before export, the importer walks the module for ModelOpt `TensorQuantizer`s. Each enabled weight
quantizer's fake-quant weight is written into the parameter and the quantizer disabled, so the
exported graph is plain float. The quantizer's scales are kept: when the weight lies on a grid
with one scale per 32-element block, it is encoded exactly into ggml's Q8_0 / Q4_0 layout.
Activation quantizers are dropped and reported.

## Passes

The optimization level (`ggmlc/options.py`) resolves to a set of flags; each optional rewrite
below names the flag that enables it. Everything else always runs: at `O0` the pipeline still
lowers, composes views, turns size-1 permutes into reshapes, folds views of weights above 4-D,
squeezes and lays out, because ggml refuses the graph otherwise.

The pipeline, per stage unless noted:

1. **lower_tier2**: `linear` → `mul_mat` + `add`; `layer_norm` → `norm` + `mul` + `add`; `sdpa` →
   `mul_mat` + `soft_max` (with mask and scale) + `mul_mat`. With `fast_gelu`, tanh-GELU becomes
   ggml's table `gelu`.
2. **simplify**: always: compose permutes and reshapes, and turn a permute that only moves size-1
   dims into a reshape (ggml refuses a transposed-looking matmul operand). `simplify`: drop
   identities (`x + 0`, `x * 1`, all-zero softmax masks) and read a slice of a concat from the
   part it lies in. `flatten_matmul`: a 2-D weight times a batched activation becomes one 2-D
   matmul. `shared_kv_heads`: drop grouped-query attention's K/V `repeat`, because `mul_mat`
   broadcasts heads the same way. `fuse_rms_norm`: `x * (1 / sqrt(mean(x²) + eps))` (and
   `x / sqrt(...)`) becomes `rms_norm`. `fuse_glu`: `silu(gate) * up` becomes `swiglu`.
3. **fuse_rope** (`fuse_rope`): `x * cos + rotate_half(x) * sin`, with cos and sin built from the
   outer product of an i32 positions value and a constant inverse-frequency vector, becomes ggml's
   NEOX `rope` on x with heads and tokens swapped (ggml wants the token dim third from the
   inside). The base is recovered from the frequencies; frequencies that are not a plain
   `base^(-2i/D)` become `freq_factors`; a scale on cos and sin becomes `attn_factor`.
4. **fuse_attention** (`flash_attention`): match eager attention (`matmul → scale →
   [mask | where] → softmax → matmul`) and rewrite it into `flash_attn` for head sizes every
   backend supports. Masks are cast to f16 (clamped to ±65504 first), and query rows that are
   fully masked get zeroed queries so their output matches PyTorch's uniform softmax, unless
   `assume_attended_rows` says no row is.
5. **cse** (`cse`): nodes with the same op, inputs and attributes merge; the mask casts and guards
   every layer rebuilds from one input are computed once.
6. **fold** (whole program): `fold_batch_norm`: fold per-channel affines after a convolution
   (frozen batch norm) into its kernel and bias. `fold_constants`: evaluate every remaining node
   that reads only weights and constants into a derived weight or constant. Views of weights stay
   views, except views of weights above 4-D, which ggml cannot hold: those are always folded.
7. **hoist** (whole program, `hoist_invariants`): loop-invariant code motion across the driver. Work in a loop body
   that depends only on values the loop never writes moves into a `<stage>.invariant` stage
   called once before the loop; its results become driver values the trimmed stage reads.
8. **f16_kv_cache** (whole program, `f16_kv_cache`): an f32 state whose every read is a cast to f16
   or a matmul operand (a KV cache under flash attention) is retyped to f16, and the casts go.
9. **dce**, then **squeeze**: values above 4-D get a stand-in holding their non-unit dims, and
   the ops touching them (reshape, permute, view, repeat, concat, elementwise and row ops) are
   rewritten against it. Stage parameters above 4-D take their squeezed type; their driver values
   use the same layout.
10. **layout**: ggml ops need contiguous inputs, except `permute`, `view`, `cont` and `cast`.
    Insert one `cont` per strided value that any other op reads. A view is still contiguous when
    it only moves or slices around size-1 dims, and gets no copy. With `strided_rows`, a strided
    view whose innermost non-unit dim keeps unit stride also feeds `mul_mat`, `flash_attn`, `rope`
    and `set_rows` (which take any strides above dim 0) and reshapes that only add or drop size-1
    dims (`ggmlc::ops::reshape` makes those a strided view) without a copy. Anything still above
    4-D fails here, naming the node.
11. **dce**.
## Weight storage and the dtype policy

Each weight has a storage type, `f32` unless a policy says otherwise. Rules match the weight's
name on the root module by glob; the first match wins. Legality decides what a weight may become:
the weight operand of `mul_mat` and the table of `get_rows` take any ggml type, a convolution
kernel takes `f16`, and anything read by another op keeps the policy's float type. A reshape that
keeps the row length (a 1-D conv kernel viewed as 2-D) is looked through. Quantized types also
need the row length to be a multiple of their block size; otherwise the policy's fallback list is
tried. The report records each weight's decision and its relative error.

## The GGUF artifact

- Metadata: `general.architecture = "ggmlc"`, `ggmlc.format_version`, `ggmlc.program` (the
  program as JSON), `ggmlc.tensor_names` (GGUF tensor name → parameter name), `ggmlc.source`
  (frontend, versions), `ggmlc.options` (the level, every flag, and the program's recorded
  specializations; the runtime does not read it), `ggmlc.meta` (the program's own metadata,
  returned by `ggmlc_metadata`).
- Tensors are named `w.<n>` for weights and `c.<n>` for constants: GGUF rejects names of 64 bytes
  or more, which module paths often exceed. Constants with the same content are stored once.
- The program JSON is already in ggml terms: per stage, its parameters (`a.<i>`), its nodes in
  topological order (`id`, `op`, sources, result type, `ne`, op parameters, PyTorch source
  location) and its outputs; the driver ops; every driver value's type, `ne` and PyTorch shape;
  and the symbols with their ranges.
- A symbolic dim is written as an expression tree: an int, a symbol name, or `[op, operand...]`
  with op one of `+`, `*`, `//` (floor), `%`, `max`, `min`. Only `ne`, shapes and a view's `start`
  can hold one.
- Format version 2 added loop slices; 3 symbolic dims, entry points (`entries`: name, inputs,
  outputs, driver) and state (`state`: value names). The runtime reads versions 1 to 3 and
  refuses others.
- The writer encodes and writes one tensor at a time.

## The runtime

`libggmlc_rt` is a C ABI over C++17, linking ggml statically.

- **Load**: parse the program, pick the backend (`cpu`, `cuda`, `auto`), upload weights and
  constants into one backend buffer, allocate every driver value in a second persistent buffer
  (zeroed, so state starts at zero), then build each driver call site's graph once and allocate it
  with `ggml_gallocr`. Every node is
  checked with `ggml_backend_supports_op`; an unsupported node fails the load and is named.
- **Ops**: `ops.cpp` maps each IR op to ggml, checking ggml's preconditions first so a bad shape
  is a named error rather than an abort. Some ops are chosen per backend at graph-build time: an
  f16 convolution kernel runs as `ggml_conv_2d_direct` on a GPU when the output is large and as
  im2col + matmul otherwise; exact tanh-GELU is `ggml_gelu` on a GPU and the formula in ggml ops
  on CPU.
- **Stages** read their arguments straight from the driver value tensors and write results back
  with `ggml_cpy` nodes appended after all other nodes, so a call can read and write the same
  value (a loop carry). Nothing returns to the host between stages.
- **Entry points and state**: each entry has its own inputs, outputs and driver;
  `ggmlc_run_entry` runs one, after checking its inputs were set. Values are global across
  entries. State values are ordinary driver values that no entry takes or returns, so they
  persist; `ggmlc_reset` zeroes them. A stage result whose memory already is its destination's (a
  `ggml_set_rows` into the state the stage read returns a view of it) gets no final copy, so a
  KV cache update writes only its new rows.
- **Driver**: `call` runs one call site's graph. `loop` copies each carry's initial value and reads
  each counter's start (4 bytes), then per iteration uploads that iteration's scalars, copies row
  i of each slice source into its loop value (device to device, through views made at load), sets
  the counters, evaluates the `bind` expressions (over the program's symbols and the counters) and
  rebinds if a symbol changed, fills the causal mask rows, runs the body, copies collected values
  into row i of their destinations, and stops if an `until` value (read back, 4 bytes) is one of
  its ids. The trip count may be a symbol.
- **Symbols**: the specs keep each symbolic dim's expression, and `ProgramSpec::bind` writes
  their values for one binding into the plain `ne` / parameter fields, which is all graph building
  reads. A driver value of symbolic size gets a buffer for its largest size (over the corners of
  its symbols' ranges) and, per binding, a view with that binding's shape. The binding starts
  with every symbol at its max, and the graphs are first built for it: that is the reservation.
  `ggmlc_set_symbol` / `ggmlc_set_input_shape` change the binding (the latter solves the symbols an
  input's shape fixes, by search over each symbol's range); the next run then rebuilds the views
  and marks every call site with symbolic shapes stale; a stale site rebuilds its graph (reusing
  its gallocr buffer) when its entry next runs it, so an entry never rebuilds another's stages.
- **Build options**: like llama.cpp's own CMake, the runtime builds ggml with tinyBLAS
  (`GGML_LLAMAFILE`) and CUDA graphs (`GGML_CUDA_GRAPHS`); ggml's standalone defaults leave both
  off, which halves CPU bf16 matmul throughput and launches every CUDA kernel separately.
- **Profiling**: with `GGMLC_PROFILE` set, every node runs as its own graph with a sync, and its
  time is logged with its op and source location; `ggmlc bench --profile` aggregates that.

## Generated C++

`emit/cpp.py` writes one builder function per stage that makes the same ggml calls in the same
order as the runtime's interpreter of the program JSON. Expansions of several calls live in
`ggmlc_ops.h`, which both sides use. The generated library registers its builders with
`ggmlc_load_with` and reuses the runtime for loading, buffers and the driver. Because ggml is
deterministic for a given graph, the outputs are bit-identical to the runtime's on the same
backend and thread count; the generated `<name>-check` binary verifies that. Symbolic dims become
C++ expressions over `const int64_t` locals read with `ggmlc_build_symbol`, and the check binary
binds the example's symbols before reading `example_inputs.bin`.

## ggml constraints the design works around

- `GGML_MAX_DIMS = 4`: the squeeze pass removes size-1 dims; anything still above 4-D fails
  compilation, naming the node.
- `ggml_mul_mat(a, b)` computes `b · aᵀ` with `a` given as `[out, in]` in PyTorch order, so
  `nn.Linear` weights need no transpose. It broadcasts `a`'s batch dims by repeat-interleave (b
  head i reads a head i // r), which is grouped-query attention's head sharing.
- ggml's layout predicates compare strides even across size-1 dims: `ggml_mul_mat` refuses an
  operand `ggml_is_transposed` calls transposed, and `ggml_scale` asserts `ggml_is_padded_1d`.
  So permutes that only move size-1 dims become reshapes, and slices give size-1 dims packed
  strides.
- Quantized types are legal only as `mul_mat`'s weight and `get_rows`' table, with rows a
  multiple of the block size.
- On CPU, `ggml_gelu` reads an fp16 table (~1e-3 error) and `tanh` is scalar; `ggml_gelu_erf` is
  exact. On CUDA, `ggml_gelu` is an exact f32 formula.
- CUDA's f32 `mul_mat` runs through TF32 (~1e-4 relative per matmul), and 16-bit weights make
  cuBLAS accumulate in f16 unless the op asks for f32 (`ggml_prec_set_acc`). CPU parity is 1e-4;
  CUDA's is 1e-2. A mul_mat node with `acc` set (`Stage(matmul_acc="f32")`) asks for f32; flash
  attention always does, as llama.cpp's graphs do.
- `ggml_flash_attn_ext` wants f16 K/V and takes head sizes {40, 64, 72, 80, 96, 112, 128, 256} on
  CUDA. Its mask is f16, so masks are clamped to ±65504 (f32 min would become -inf, and a fully
  masked row NaN).
- `ggml_conv_2d` builds its patch buffer in f16 unless the kernel is bf16, hence the runtime's own
  im2col + matmul. `ggml_conv_transpose_1d` has no padding or batch and is a slow direct loop:
  transposed 1-D convolutions lower to a matmul with the permuted kernel plus an overlap-add of
  shifted slices. 1-D convolutions run as 2-D ones over `[N, C, 1, L]`.
- One IR node can expand to several ggml tensors, so graph contexts are sized for the largest
  expansion; an overflow would make `ggml_new_tensor` return NULL in release builds.
- ggml built on its own leaves tinyBLAS (`GGML_LLAMAFILE`) and CUDA graphs (`GGML_CUDA_GRAPHS`)
  off; the runtime's CMake turns both on, as llama.cpp does. tinyBLAS doubles CPU bf16 matmul
  speed; CUDA graphs take a small model's decode step from 3.3 to 2.8 ms.
- `ggml_set_rows(dst, rows, idx)` returns a view of `dst` and writes in place along ne1; I32
  indices work. A KV cache is `[kv_heads, max_context, head_dim]` per layer (rows = positions),
  written with `index_copy` along dim 1, and read through a view of its first `n_kv` rows.
  A VLM's image rows are state the same way: `encode` writes `tiles * rows_per_tile` of them at
  `arange(n)` (a symbolic stop, ggml_arange in f32 cast to I32), and `prefill_mm` takes its input
  embeddings with one `get_rows` from `concat(image rows, embed(tokens))`. Qwen3-VL's stages call
  the decoder layers directly: text RoPE is transformers' `inv_freq @ positions` on its own
  position input (fuse_rope's pattern), M-RoPE gathers each frequency's (t, h, w) stream from a
  `[3, s]` input with a constant index (exact), and deepstack rows are gathered with a zero row for
  text tokens and added after the first decoder layers.
- `ggml_rope_ext` wants the token dim third from the inside (`[D, H, S]`) and computes each angle
  by repeated f32 multiplication, ~1e-4 from PyTorch's `pos * inv_freq` at position 1000.
- `torch.export` assumes a dynamic dim is never 0 or 1 (a Dim's min becomes 2), and fails on a
  plain Dim slicing a fixed-size buffer with guards it cannot prove; a derived `256 * Dim` works.
  It also wants `dynamic_shapes` shaped like pytree inputs; the frontend expands a None.
- `torch.export` cannot trace `.tolist()` on tensors computed from constants (Qwen-VL's patch grid
  bookkeeping); such tables are computed eagerly when the program is built.

## Verification

- The parity metric is `max|out − ref| / max(1, max|ref|)`; NaN or inf where the reference is
  finite always fails. Tolerances are 1e-4 on CPU and 1e-2 on CUDA with f32 weights.
- Unit tests check every tier-1 op against its PyTorch reference on every available backend,
  every pass through the interpreter, the frontends op by op, the driver (loops, carries, scalars,
  slices, hoisting), ModelOpt import, ONNX and the generated C++.
- End-to-end tests compile LeRobot policies and compare with PyTorch; the stage functions are
  also checked against LeRobot's own sampling loops. Models too large for a test run compile with
  the language model and action head cut to a few layers, which keeps every op and pattern of the
  full graph.
