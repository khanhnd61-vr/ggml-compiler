# Supported operations

This page lists which PyTorch modules, tensor operations and ONNX operators ggml-compiler compiles,
and what each one becomes in ggml. Every entry was compiled, run on CPU and CUDA, and compared
with PyTorch (with onnxruntime for ONNX). The checks used ggml-compiler 0.1.0, PyTorch 2.11 and
ggml from llama.cpp b11223, on an RTX 3060. Everything listed as compiling matched within 2e-6 on
CPU and 1e-3 on CUDA, in scaled error (`max|out − ref| / max(1, max|ref|)`). CUDA's bound is wider
because its f32 matmuls do not accumulate in full f32.

- [How a model reaches ggml](#how-a-model-reaches-ggml)
- [Rules for every op](#rules-for-every-op)
- [PyTorch modules](#pytorch-modules)
- [Tensor operations](#tensor-operations)
- [Model components known to compile](#model-components-known-to-compile)
- [ONNX operators](#onnx-operators)
- [Fused patterns](#fused-patterns)
- [IR ops and their ggml calls](#ir-ops-and-their-ggml-calls)
- [Adding a lowering for a missing op](#adding-a-lowering-for-a-missing-op)
- [Known issues](#known-issues)

## How a model reaches ggml

- `torch.export` traces each stage and decomposes it into Core ATen ops. A module or function
  compiles when every ATen op it decomposes into has a lowering, so most of `torch.nn` needs
  nothing of its own. For example, `nn.Hardswish` becomes a clamp and a multiply, and `nn.LSTM` is
  unrolled over its sequence length.
- Three ops are kept whole so that the fusion passes can see them: `linear`, `layer_norm` and
  `scaled_dot_product_attention`.
- Work that reads only weights and constants is computed at compile time with PyTorch, whatever
  ops it uses. This covers RoPE tables, position embeddings, masks of a static size, buffers,
  `torch.arange` and `torch.zeros`. Only ops on runtime values need a lowering.
- When ops are missing, compilation stops and lists each missing ATen op once, with its source
  line ([USAGE.md](USAGE.md#when-an-op-has-no-ggml-lowering)). You can
  [register a lowering](#adding-a-lowering-for-a-missing-op) for any op yourself.

## Rules for every op

- **At most 4 dims.** ggml tensors have four dims, and dims of size 1 don't count against that
  limit. A `[B, T, cams, C, H, W]` input with B = T = 1 is fine. An intermediate with five dims
  above 1, such as the one inside `nn.PixelShuffle`, fails and names the node.
- **f32 arithmetic.** Activations are f32. A cast to f16 or bf16 inside a stage does not round:
  `x.half().float()` gives `x`. The dtype policy sets weight storage types
  ([USAGE.md](USAGE.md#weight-dtypes-and-quantization)).
- **Integers and booleans.** Integer tensors are i32 and serve as indices. Booleans are f32 values
  of 0 and 1.
- **Integer stage outputs.** A stage output keeps the dtype it was computed in. Returning
  `argmax(...).float()` (i32 inside) or `mask.long().cumsum(-1)` (f32 inside) fails to compile.
  Inside a stage, both work.
- **Reductions.** `sum` and `mean` work over the last dim or over every element. For any other
  dim, move it last first: `x.transpose(1, -1).mean(-1)`.
- **Sizes.** Sizes are static, or symbolic within a compiled range
  ([USAGE.md](USAGE.md#symbolic-dims)). With a symbolic length, `is_causal=True` attention needs
  an explicit mask, and the `*_like` constructors fail.

## PyTorch modules

### Layers that compile

| Module | Notes |
|---|---|
| `nn.Linear` | With or without bias |
| `nn.Conv1d`, `nn.Conv2d` | Any stride, padding (`"same"` too) and dilation. `groups=1` and `padding_mode="zeros"` only |
| `nn.Conv3d` | Only with a kernel that covers the whole input, like Qwen-VL's patch embedding (one matmul). The input must reach the stage with at most 4 dims and be viewed as 5-D inside it |
| `nn.ConvTranspose1d` | `dilation=1`, `groups=1`, `output_padding <= padding`. Lowered to one matmul plus an overlap-add |
| `nn.BatchNorm1d`, `nn.BatchNorm2d` | Eval mode. After a convolution, folded into its kernel and bias (O1) |
| `nn.LayerNorm` | Over the last dim only. With or without weight and bias |
| `nn.GroupNorm` | 1-D and 2-D inputs |
| `nn.RMSNorm` | Fused into `ggml_rms_norm` (O1) |
| `nn.Embedding` | Any index shape. `padding_idx` is fine |
| `nn.MultiheadAttention` | Pass `need_weights=False` (or `average_attn_weights=False`): the default averages the attention weights over heads, a reduction ggml-compiler cannot lower, even if the weights are discarded. `kdim`, `vdim` and `attn_mask` work |
| `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.Transformer` | ReLU or GELU, `norm_first` either way |
| `nn.LSTM`, `nn.GRU`, `nn.RNN` | Unidirectional; several layers checked on `nn.LSTM`. Unrolled over the sequence length, which must be static |
| `nn.MaxPool1d`, `nn.MaxPool2d` | No `ceil_mode`, `dilation` or `return_indices` |
| `nn.AvgPool1d`, `nn.AvgPool2d` | No `ceil_mode` or `divisor_override`. With padding, `count_include_pad=True` only |
| `nn.Dropout` (eval), `nn.Identity`, `nn.Flatten`, `nn.Unflatten` | |

### Activations that compile

| Module | Notes |
|---|---|
| `nn.ReLU`, `nn.PReLU`, `nn.Threshold` | |
| `nn.GELU` | `approximate="none"` is `ggml_gelu_erf`. `"tanh"` is an f32 formula, or ggml's `gelu` at O2 (an fp16 table on CPU, ~1e-3) |
| `nn.SiLU`, `nn.Mish`, `nn.Sigmoid`, `nn.Tanh`, `nn.Tanhshrink`, `nn.Softsign` | |
| `nn.Hardswish`, `nn.Hardsigmoid` | |
| `nn.Softplus` | `beta=1` and `threshold=20` only, which ggml's `softplus` fixes |
| `nn.Softmax`, `nn.Softmin` | Any dim |
| `nn.GLU` | |

### Modules that do not compile

| Module | Missing | Workaround |
|---|---|---|
| `nn.Conv2d` with `groups > 1` (including depthwise) | Grouped convolution | |
| `nn.Conv2d` with `padding_mode` other than `"zeros"` | `aten.index.Tensor` | |
| `nn.Conv3d` (other kernels), `nn.ConvTranspose2d` | 3-D convolution, 2-D transposed convolution | |
| `nn.Bilinear` | `aten._trilinear` | |
| `nn.LayerNorm` over more than one dim | | |
| `nn.InstanceNorm2d` | `aten._native_batch_norm_legit.no_stats` | |
| `nn.LocalResponseNorm` | `aten.constant_pad_nd` | |
| `nn.EmbeddingBag` | `aten._embedding_bag` | |
| `nn.AdaptiveAvgPool1d`, `nn.AdaptiveAvgPool2d` | A reduction over two dims; `aten._adaptive_avg_pool2d` for outputs larger than 1 | Output size 1: `x.flatten(2).mean(-1)` |
| `nn.AdaptiveMaxPool2d` | `aten.adaptive_max_pool2d` | |
| `nn.Upsample` | `aten.upsample_nearest2d`, `aten.upsample_bilinear2d` | |
| `nn.PixelShuffle`, `nn.PixelUnshuffle` | A 6-D intermediate | |
| `nn.ZeroPad*`, `nn.ConstantPad*` | `aten.constant_pad_nd` | Concatenate zeros: `torch.cat([torch.zeros(...), x], dim)` |
| `nn.ReflectionPad*`, `nn.ReplicationPad*` | `aten.index.Tensor` | |
| `nn.CosineSimilarity` | `aten.linalg_vector_norm` | `x * (x.pow(2).sum(-1, keepdim=True) + eps).rsqrt()` normalizes |
| `nn.ReLU6`, `nn.Hardtanh` | `aten.hardtanh` | [A one-line lowering](#adding-a-lowering-for-a-missing-op) |
| `nn.LeakyReLU` | `aten.leaky_relu` | [A one-line lowering](#adding-a-lowering-for-a-missing-op) |
| `nn.ELU`, `nn.SELU` | `aten.elu` | |
| `nn.CELU` | `aten.expm1` | |
| `nn.LogSigmoid` | `aten.minimum`, `aten.log1p` | |
| `nn.LogSoftmax` | `aten._log_softmax` | |
| `nn.Softplus` with other `beta` or `threshold` | ggml's `softplus` fixes both | |
| `nn.LSTM`, `nn.GRU` with `bidirectional=True` | `aten.flip` | |
| `nn.MaxPool2d(ceil_mode=True)`, `nn.AvgPool2d` with `ceil_mode`, `divisor_override` or padding with `count_include_pad=False` | | |

## Tensor operations

These cover `torch.*`, `torch.nn.functional` and `Tensor` methods.

| Category | Compiles |
|---|---|
| Arithmetic | `+ - * /` on tensors and scalars with broadcasting (`alpha` too); `pow` with exponent 2, 0.5, 1, −1 or −0.5; `sqrt`, `rsqrt`, `reciprocal`, `exp`, `log`, `sin`, `cos`, `abs`, `neg`, `square`, `clamp` (min, max or both), `lerp` |
| Activations | `F.relu`, `F.gelu`, `F.silu`, `F.mish`, `torch.sigmoid`, `torch.tanh`, `F.softplus`, `F.glu`, `F.softmax` (any dim) |
| Normalization | `F.layer_norm` (last dim), `F.group_norm`, `F.rms_norm`, and RMSNorm written out (`x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`) |
| Comparison and selection | `> < >= <= == !=` on tensors or scalars; `& \| ~` and `logical_and/or/not` on booleans; `torch.where` with tensor or scalar branches; `masked_fill`; `.bool()`, `.float()`, `.to(dtype)` |
| Matmul | `@`, `matmul`, `mm`, `bmm`, `baddbmm`, `F.linear`, `torch.outer`, and `einsum` over these. Both operands at least 2-D. The right operand's batch dims broadcast against the left's, but not the other way: `x @ W` works for a batched `x` and a 2-D `W`, while a 2-D `x` times a batched `W` fails |
| Reductions | `sum` and `mean` over the last dim or over all elements; `max` and `min` over a dim (values and indices); `amax`, `amin` over one dim; `argmax`, `argmin` over one dim or flattened; `cumsum` over any dim |
| Shape | `view`, `reshape`, `flatten`, `unflatten`, `squeeze`, `unsqueeze`, `permute`, `transpose`, `.T`, `.mT`, `contiguous`, `expand`, `repeat`, `repeat_interleave`, `cat`, `stack`, `split`, `chunk`, `unbind`, `narrow`, `select`, integer indexing, slicing with any positive step, `None` indexing, `tril`, `triu` |
| Constructors | `zeros_like`, `ones_like`, `full_like` (static sizes); `torch.arange`, `zeros`, `ones`, `full` (computed at compile time; an `arange` with a symbolic end becomes `ggml_arange`) |
| Indexing | `F.embedding` (a 2-D table, any index shape); `index_select` along dim 0 of a 2-D tensor; `index_copy` and `index_put` that write rows along the second-to-last dim (`ggml_set_rows`, in place: a KV cache) |
| Attention | `F.scaled_dot_product_attention` with no mask, `is_causal=True`, or a bool or float mask that is a constant or computed at run time; `scale`; grouped-query K/V (`enable_gqa=True` or `repeat_kv`). Eager attention written out as `softmax(q @ k.mT * s + mask) @ v` compiles the same way |

Operations that do not compile:

| Operation | Missing ATen op | Workaround |
|---|---|---|
| `sum`, `mean` over a dim other than the last | | `x.transpose(d, -1).sum(-1)` |
| `var`, `std` | `aten.var.correction` | `((x - m) ** 2).mean(-1)` with `m = x.mean(-1, keepdim=True)` |
| `x.max()`, `x.min()` without a dim | `aten.max.default` | `x.flatten().amax(0)` |
| `logsumexp` | Compiles but fails to load ([known issues](#known-issues)) | `m.squeeze(-1) + (x - m).exp().sum(-1).log()` with `m = x.amax(-1, keepdim=True)` |
| `topk`, `sort` | `aten.topk`, `aten.sort` | |
| `pow` with another exponent, or a tensor exponent | `aten.pow.Tensor_Tensor` | `x * x * x` |
| `erf`, `tan`, `atan`, `floor`, `round`, `sign`, `log1p`, `expm1` | The op itself | |
| `torch.maximum`, `torch.minimum` | `aten.maximum`, `aten.minimum` | `torch.where(a > b, a, b)` |
| `torch.addmm` | `aten.addmm` | `F.linear` or `a @ b + c` |
| `matmul` with a 1-D operand | | `unsqueeze` it to 2-D |
| `F.normalize`, `torch.linalg.vector_norm`, `Tensor.norm` | `aten.linalg_vector_norm` | `x * (x.pow(2).sum(-1, keepdim=True) + eps).rsqrt()` |
| `F.log_softmax` | `aten._log_softmax` | |
| `F.pad` | `aten.constant_pad_nd`, `aten.index.Tensor` (reflect, replicate) | Constant padding: `torch.cat` with zeros |
| `F.interpolate` | `aten.upsample_*` | |
| `F.unfold` | `aten.constant_pad_nd` | |
| `flip`, `roll` | `aten.flip`, `index_select` on a non-2-D tensor | |
| `x[idx]` with a tensor index, `gather`, `scatter`, `masked_select`, `nonzero` | `aten.index.Tensor`, `aten.gather`, `aten.scatter`, ... | `F.embedding(idx, table)` for rows of a 2-D table |
| `index_select`, `index_copy`, `index_put` on other dims | | |

## Model components known to compile

The models in [MODELS.md](MODELS.md) compile from their published checkpoints and match their
PyTorch implementation. Between them, they cover these components:

- **Decoder language models** (transformers' Llama, Qwen2, Qwen3 and Gemma 4, through
  `ggml_compiler.causal_lm`):
  - RMSNorm, including Gemma's `x * pow(mean + eps, -0.5)` form;
  - rotary embeddings: plain, Qwen3-VL's M-RoPE and Gemma 4's proportional;
  - grouped-query attention, sliding-window masks, Q/K norms and logit soft-capping;
  - SwiGLU and GELU-tanh MLPs, and Gemma 4's per-layer embeddings and shared KV layers;
  - the KV cache, written with `index_copy` at runtime positions.
- **Vision encoders**: SigLIP (SmolVLM2, SmolVLA, pi0, pi0.5), and the Qwen3-VL ViT (Conv3d patch
  embedding, 2-D rotary embeddings, deepstack features).
- **Convolutional backbones**: ResNet-18 with frozen batch norm (ACT), or with GroupNorm and a
  spatial softmax (Diffusion Policy).
- **Action heads**: transformer encoder-decoders (ACT); flow-matching experts attending to a
  cached prefix (SmolVLA, pi0, pi0.5); a DiT (GR00T N1.7); and a 1-D conditional U-Net built from
  Conv1d, GroupNorm, Mish, ConvTranspose1d and FiLM (Diffusion Policy).

## ONNX operators

Each operator below was checked against onnxruntime on CPU and CUDA, as a single-node graph.
Any node whose inputs are all constants (`Shape`, `Constant`, `Range`, shape arithmetic) is
evaluated at compile time with onnx's reference evaluator, so it needs no lowering.

| Operator | Notes |
|---|---|
| `Add`, `Sub`, `Mul`, `Div` | Broadcasting |
| `Equal`, `Less`, `Greater`, `LessOrEqual`, `GreaterOrEqual`, `And`, `Or`, `Not`, `Where` | |
| `Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Neg`, `Abs`, `Sqrt`, `Sin`, `Cos`, `Reciprocal` | |
| `Gelu` | Both approximations (opset 20) |
| `Pow` | Constant exponent of 2, 0.5, 1, −1 or −0.5 |
| `Clip` | |
| `Cast` | To float or bool. A cast to f16 does not round: arithmetic stays f32 |
| `Identity`, `Dropout` | |
| `MatMul`, `Gemm` | `Gemm`: `transA`, `transB`, `alpha`, `beta`, optional `C` |
| `LayerNormalization` | Last axis only |
| `Softmax` | Any axis |
| `ReduceMean`, `ReduceSum` | Last axis only |
| `Conv` | 2-D, `group=1`, symmetric padding, `auto_pad` NOTSET or VALID |
| `MaxPool`, `AveragePool` | Symmetric padding; `MaxPool` without `ceil_mode` or dilation |
| `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`, `Transpose`, `Expand`, `Concat`, `Split`, `Slice` | |
| `Gather` | A scalar index on any axis, or a tensor of indices on axis 0 |

Any other operator stops compilation with a list of the missing ones.

## Fused patterns

The passes recognize these patterns in the traced graph and turn them into single ggml ops. Each
one belongs to a flag, and an optimization level is a set of flags
([USAGE.md](USAGE.md#optimization-levels-and-flags); details in
[ARCHITECTURE.md](ARCHITECTURE.md#passes)).

| In the model | In ggml | Flag (level) |
|---|---|---|
| `F.linear`, `nn.Linear` | `ggml_mul_mat` (+ `ggml_add` for the bias) | always |
| `F.layer_norm` | `ggml_norm` (+ `ggml_mul`, `ggml_add`) | always |
| RMSNorm, as `nn.RMSNorm` or written out | `ggml_rms_norm` | `fuse_rms_norm` (O1) |
| `F.silu(gate) * up` | `ggml_swiglu_split` | `fuse_glu` (O1) |
| A 2-D weight times a batched activation | One 2-D `ggml_mul_mat` | `flatten_matmul` (O1) |
| Grouped-query K/V heads repeated (`repeat_kv`) | `ggml_mul_mat`'s own head broadcast | `shared_kv_heads` (O1) |
| Batch norm (or any per-channel affine) after a convolution | Folded into the kernel and bias | `fold_batch_norm` (O1) |
| `F.scaled_dot_product_attention`, or eager attention | `ggml_flash_attn_ext`, with K, V and the mask in f16, for head sizes 40, 64, 72, 80, 96, 112, 128, 256 and 512 | `flash_attention` (O2) |
| transformers' rotary embedding (`x * cos + rotate_half(x) * sin`) | `ggml_rope_ext` (NEOX mode) | `fuse_rope` (O2) |
| tanh-GELU | `ggml_gelu` everywhere | `fast_gelu` (O2) |

Without `flash_attention`, attention runs as `ggml_mul_mat`, `ggml_soft_max_ext` and
`ggml_mul_mat`. Some backends cannot run flash attention for some head sizes: CUDA runs 512 only
on Turing or newer with grouped-query heads. For those, the runtime builds the node in that same
three-op form at load, so one GGUF loads everywhere.

## IR ops and their ggml calls

The compiler's IR has one op per ggml call, or per short fixed expansion. These are the building
blocks a custom lowering uses: `Builder` exposes each op as a method, as in `b.mul(x, y)` or
`b.scale(x, s=0.5, b=1.0)`. Shapes are in PyTorch order (outermost first).

| IR op | ggml | Notes |
|---|---|---|
| `add`, `sub`, `mul`, `div` | `ggml_add`, `ggml_sub`, `ggml_mul`, `ggml_div` | The second operand broadcasts into the first |
| `relu`, `silu`, `tanh`, `sigmoid`, `exp`, `log`, `neg`, `abs`, `sqr`, `sqrt`, `sin`, `cos`, `softplus`, `gelu_erf`, `gelu` | The ggml op of the same name | `gelu` is the tanh approximation (an fp16 table on CPU) |
| `gelu_tanh` | `ggml_gelu` on a GPU; the formula in ggml ops on CPU | Exact in f32 on both |
| `swiglu` | `ggml_swiglu_split` | `silu(gate) * up` |
| `scale` | `ggml_scale`, `ggml_scale_bias` | `x * s + b` |
| `step` | `ggml_step` | `x > 0` as 0 or 1 |
| `clamp` | `ggml_clamp` | |
| `cast` | `ggml_cast` | f32, f16, i32 |
| `mean`, `sum_rows`, `cumsum` | `ggml_mean`, `ggml_sum_rows`, `ggml_cumsum` | Over the last dim |
| `norm`, `rms_norm` | `ggml_norm`, `ggml_rms_norm` | Over the last dim, without affine |
| `soft_max` | `ggml_soft_max_ext` | Over the last dim, with optional mask and scale |
| `mul_mat` | `ggml_mul_mat` | `b @ a.T` with `a` as `[out, in]`. `a`'s batch dims broadcast over `b`'s, each `a` head serving consecutive `b` heads (grouped-query attention). The only op that reads quantized weights |
| `get_rows` | `ggml_get_rows` | Rows of a 2-D table at i32 indices. Reads quantized tables |
| `set_rows` | `ggml_set_rows` | Writes rows (dim −2) at i32 indices, in place |
| `argmax` | `ggml_argmax` | Over the last dim of a matrix, i32 result |
| `rope` | `ggml_rope_ext` | NEOX mode, x as `[.., S, H, D]`, i32 positions, optional `freq_factors` |
| `arange` | `ggml_arange` | Bounds may be symbolic |
| `conv_2d` | `ggml_conv_2d_direct` for an f16 kernel on a GPU with at least 1024 output positions; otherwise `ggml_im2col` + `ggml_mul_mat` | |
| `im2col` | `ggml_im2col` | Patches in f32 |
| `pool_2d` | `ggml_pool_2d` | Max or average |
| `flash_attn` | `ggml_flash_attn_ext`, f32 accumulation | Falls back to the three-op form where the backend cannot run it |
| `reshape`, `permute`, `view`, `repeat`, `concat`, `cont` | `ggml_reshape_4d`, `ggml_permute`, `ggml_view_4d`, `ggml_repeat_4d`, `ggml_concat`, `ggml_cont` | `view` is a step-1 slice along one dim |

Three source-level ops are rewritten into these before emission: `linear`, `layer_norm` and
`sdpa`. The `Builder` also has conveniences: `const(tensor)`, `add_scalar(x, c)`,
`mul_scalar(x, c)`, `transpose(x, d0, d1)` and `reshape_to(x, shape)`.

## Adding a lowering for a missing op

Register a function for the ATen op. It receives the `Builder`, then the op's arguments as
`torch.export` recorded them. Runtime tensors arrive as IR values, constant tensors as
`torch.Tensor`, and everything else as plain Python. The function returns the result's IR value,
or a list for an op with several outputs. A registered op is kept whole by the exporter, so the
lowering sees it before any decomposition. Registering an op that already has a lowering
replaces it.

```python
import torch
import ggml_compiler

@ggml_compiler.lowering(torch.ops.aten.hardtanh.default)      # nn.Hardtanh, nn.ReLU6
def hardtanh(b: ggml_compiler.Builder, x, min_val=-1.0, max_val=1.0):
    return b.clamp(x, min=float(min_val), max=float(max_val))

@ggml_compiler.lowering(torch.ops.aten.leaky_relu.default)    # nn.LeakyReLU
def leaky_relu(b: ggml_compiler.Builder, x, negative_slope=0.01):
    return b.add(b.relu(x), b.scale(b.relu(b.neg(x)), s=-float(negative_slope)))
```

Register lowerings before calling `ggml_compiler.compile`. The compile error names the ATen op to
register (`aten.hardtanh.default`). The op's schema gives its argument names and defaults:
`print(torch.ops.aten.hardtanh.default._schema)`.

## Known issues

- A comparison with an infinite scalar (`x == float("inf")`, and `torch.logsumexp`, which does
  one internally) compiles, but the GGUF then fails to load with "program JSON does not parse".
  For `logsumexp`, use the workaround in the table above.
- Converting an integer result to float as a stage output fails (see
  [rules for every op](#rules-for-every-op)).
