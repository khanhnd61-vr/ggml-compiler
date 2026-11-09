# Running a new model with vla.cpp

vla.cpp serves a model it has never heard of when that model arrives as a ggml-compiler GGUF with a
`vla` manifest ([SERVING.md](SERVING.md)). This page does it once, end to end, for a policy that
neither project knew: [CoRP](https://github.com/corp-policy/CoRP), the Compressed Representation
Policy released for LIBERO. The result is [`examples/corp/export.py`](../examples/corp/export.py),
one GGUF, and a `vla-server` that answers the same requests as for a built-in model, with no line
added to vla.cpp or to ggml-compiler.

- [What the model does around its network](#what-the-model-does-around-its-network)
- [The export script](#the-export-script)
- [The manifest](#the-manifest)
- [Compile and verify](#compile-and-verify)
- [Serve it](#serve-it)
- [What the client does](#what-the-client-does)
- [The next model](#the-next-model)

## What the model does around its network

CoRP is a DINOv2 ViT-S/14 on two 224x224 views, a T5-efficient-tiny encoder on at most 32 token
ids, a cross-attention resampler that compresses each view to 48 tokens conditioned on the pooled
instruction, a state token, and a one-block DiT that turns noise into a 12x7 action chunk in 10
Euler steps. Its `predict` is about 200 lines of PyTorch around that network, and the first job is
to read them and sort every line into one of three places:

| CoRP code | What it does | Where it goes |
|---|---|---|
| `prepare_observation` | `/255`, ImageNet mean/std, NCHW | manifest `images` |
| `LanguageEncoder.forward` | tokenizes, pads to 32, runs T5 | tokens: client; padding and mask: manifest `lang`; T5: graph |
| `Normalizer.normalize_state` | `(x - mean) / (std + 1e-6)` | manifest `state` |
| `DiTActionGenerator.sample` | `torch.randn` then 10 Euler steps | noise: manifest `noise`; steps: graph |
| `Normalizer.unnormalize_action` | min/max on columns 0-5, column 6 untouched | manifest `actions` |
| `evaluate_libero.py` | rotates the cameras 180 degrees, resizes to 224, picks the rotation vector nearest the checkpoint's angle centres | client |

The rule: arithmetic whose shapes the model fixes goes in the graph; per-tensor preprocessing the
manifest has a word for goes in the manifest; what is left stays with the client. CoRP's own
docstring says the caller applies the camera transform, so leaving the rotation to the client
follows the model, not a limitation of the manifest.

## The export script

`examples/corp/export.py` imports CoRP from a checkout (`CORP_ROOT`) rather than installing it,
loads the released checkpoint with CoRP's own `load_policy`, and wraps the modules in an
`nn.Module` whose `forward` takes tensors only:

```python
def forward(self, agentview, eye_in_hand, lang_tokens, lang_mask, state, noise):
    patches = {name: self.vision(img) for name, img in zip(self.view_names, (agentview, eye_in_hand))}
    lang, pooled = self.language(lang_tokens, lang_mask)
    z = torch.cat([self.resampler(patches, pooled), lang, self.state_encoder(state)], dim=1)
    prefix = self.gen.prepare_prefix(Representation(tokens=z, pooled_language=pooled, action_dtype=z.dtype))
    x = noise
    for step in range(self.steps):
        tau = torch.full((1,), step / float(self.steps), dtype=x.dtype)
        x = x + self.gen(x, tau, prefix) * (1.0 / float(self.steps))
    return x
```

Three things in CoRP's path do not export as written, and each has the same fix: compute it once
in `__init__`, as a constant.

- **DINOv2's position embedding** is a 37x37 grid that `interpolate_pos_encoding` resizes with
  bicubic `F.interpolate`, an op ggml has no kernel for. The image size is fixed, so the wrapper
  calls that method once and keeps the 16x16 result as a buffer; `vision()` is the ViT's forward
  with the buffer added in place of the call.
- **T5's relative position bias** is built from bucketed distances with `log`, `min` and integer
  casts. The token count is fixed at 32, so `compute_bias(32, 32)` runs once; the padding mask is
  added to it at run time and the blocks are called with `position_bias=` set.
- **The masked mean over the sequence** was `tokens.sum(dim=1)`. ggml reduces over the last dim,
  so it becomes `tokens.transpose(1, 2).sum(dim=-1)`. The compiler names the op and the line when
  it meets one of these: `sum_rows over dims [1] (only the last dim is supported) at export.py:90`.

Everything else exported unchanged: the resampler's `nn.MultiheadAttention` (CoRP already passes
`need_weights=False`), the DiT's `scaled_dot_product_attention` with its constant prefix mask, the
sinusoidal time embedding on a constant `tau`, the unrolled loop.

Before compiling, check the wrapper against the model it replaces. The reference is CoRP's own
modules called the way `predict` calls them, with the example noise instead of `torch.randn`:

```
wrapper vs CoRP (normalized chunk): max|d| = 0.0
```

Bit-identical, which is what replacing glue with constants should give. A wrapper that is only
close is hiding a mistake in it or a real numerical difference, and this is the cheapest place
to find out which.

## The manifest

`serving_manifest()` reads the statistics from the checkpoint's `Normalizer`:

```python
vla.manifest(
    images=[vla.image(name, 224, 224, mean=vla.IMAGENET_MEAN, std=vla.IMAGENET_STD)
            for name in ("agentview", "eye_in_hand")],
    lang=vla.lang("lang_tokens", 32, mask="lang_mask", pad_id=0),
    state=vla.state("state", mean=norm.state_mean.tolist(), std=norm.state_std.tolist(), eps=1e-6),
    noise=vla.noise("noise"),
    actions=vla.actions("actions", chunk=12,
                        min=[*norm.action_min[:6].tolist(), -1.0], max=[*norm.action_max[:6].tolist(), 1.0],
                        eps=0.0),
)
```

Two details matter. CoRP's z-score uses `eps=1e-6` and its min/max un-normalization uses none,
so both are set rather than left at the helper's default. And the gripper column is not
un-normalized at all: the manifest has one rule per role, so column 6 gets `min=-1, max=1`, which
makes `(y + 1) / 2 * 2 - 1` the identity to within one float rounding. Applying the manifest's
formula to the reference chunk and comparing with CoRP's `unnormalize_action` gives `1.2e-7`.

`compile` checks the manifest against the program: every input named by exactly one role, the
`actions` output present with `chunk x dim` values.

## Compile and verify

CoRP's checkpoint is a Git LFS file (228 MB; `checkpoints/SHA256SUMS` has its digest):

```bash
git clone https://github.com/corp-policy/CoRP && git -C CoRP lfs pull
export CORP_ROOT=$PWD/CoRP
uv run ggml-compiler compile examples/corp/export.py:program --out out/corp --emit gguf
uv run ggml-compiler verify out/corp/model.gguf --backend cpu
uv run ggml-compiler verify out/corp/model.gguf --backend cuda
```

CoRP needs nothing installed: the checkout goes on `sys.path`, and its checkpoint loads under the
torch and transformers versions ggml-compiler already uses. The compile reports one stage of 2261
nodes over 261 weights (174 MB); `verify` runs the GGUF through the runtime on the compile's own
example inputs and compares with the PyTorch output saved beside it:

```
PASS: worst scaled error 7.082e-07, tolerance 1e-04 (CPU)
PASS: worst scaled error 6.079e-04, tolerance 1e-02 (CUDA0)
```

## Serve it

vla.cpp builds the runtime against its own ggml when `VLA_COMPILED` is on, which is the default,
so any vla.cpp binary from a tree with that option serves the GGUF. A smoke test with `vla-cli`
(the instruction goes in as token ids; the server has no tokenizer):

```bash
./build/vla-cli --ckpt out/corp/model.gguf --image agentview.png --image eye_in_hand.png \
    --tokens 474,8,3047,30,8,3829,1 --state -0.05,0.02,1.05,3,0.1,-0.2,0.04,-0.04 --pretty
```

```
vla: arch = compiled (ggml-compiler)
vla(compiled): backend = CPU (12 threads)
vla(compiled): ggml_compiler_rt 0.1.0 (ggml b11223) on CPU; 2 camera(s), instruction, state, noise, chunk 12 x 7 action dims
vla(compiled): view 0 'agentview' 224x224
vla(compiled): view 1 'eye_in_hand' 224x224
-0.283761 0.218862 -0.114784 -0.047822 -0.00925654 -0.137066 0.992453
...
```

Then the server, and a client sending the same uint8 views, token ids, raw state and the noise
the reference used (`PredictRequest.noise`; without it the server samples its own):

```bash
./build/vla-server out/corp/model.gguf
```

```
vla-server: loaded. chunk_size=12  action_dim=7  n_lang=32 ...
vla-server: bound to tcp://*:5555. ready.
```

The chunk that comes back agrees with CoRP's own `predict` to `7.2e-7` (scaled), the same figure
`verify` gave, because vla.cpp's compiled arch runs the same graphs on the same ggml. Latency on
the machine used here (i5-12400F, 12 threads; RTX 3060):

| | ms per chunk |
|---|--:|
| vla-server, CPU | 125 |
| CoRP's PyTorch `predict`, CPU | 73 |
| ggml-compiler runtime, CUDA | 9.8 |

PyTorch wins on this CPU: oneDNN's f32 convolutions and GEMMs beat ggml's at batch 1, and this is
an `O1` f32 compile. `-O2` (flash attention, table GELU) and a 16-bit weight policy are the
usual next steps once parity is in hand; the CUDA number is what a deployment would see.

## What the client does

Everything the table at the top assigned to the client, which is also everything CoRP's own
`evaluate_libero.py` does before calling `predict`:

```python
# cameras: LIBERO renders 256x256; CoRP rotates 180 degrees, then resizes with PIL (bilinear)
image = np.asarray(obs["agentview_image"], np.uint8)[::-1, ::-1]
image = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))

# instruction: T5 token ids with </s>, from the tokenizer in CoRP/corp/assets/t5_efficient_tiny
ids = T5TokenizerFast.from_pretrained(f"{CORP_ROOT}/corp/assets/t5_efficient_tiny")(text)["input_ids"]

# state: position, rotation vector wrapped to within pi of the checkpoint's angle centres, gripper
rotvec = Rotation.from_quat(obs["robot0_eef_quat"]).as_rotvec()
for i in range(3):
    while rotvec[i] - centre[i] > np.pi: rotvec[i] -= 2 * np.pi
    while centre[i] - rotvec[i] > np.pi: rotvec[i] += 2 * np.pi
state = np.concatenate([obs["robot0_eef_pos"], rotvec, obs["robot0_gripper_qpos"]])
```

The angle centres are the checkpoint's state mean, which the manifest carries, so a client reads
them from the GGUF (`ggml_compiler.Session(path).metadata["vla"]["state"]["mean"][3:6]`) rather
than from the checkpoint. The manifest's `fit: exact` is deliberate: the host's bilinear resize is
PyTorch's, without antialiasing, and CoRP was evaluated with PIL's, so the client resizes.

A manifest version 2 could take over the rotation (an image `rotate` field) and the untouched
gripper column (a per-column `passthrough`), and the client would shrink to tokenizing and the
rotation-vector choice. Those are the two additions CoRP argues for.

## The next model

The same steps, in the same order, for any policy:

1. Read its `predict` and sort every line into graph, manifest or client. Anything the manifest
   has no word for and the graph cannot hold is a client responsibility, and should be written
   down as one.
2. Write `export.py` from the nearest example. Replace Python glue with tensors; replace ops the
   compiler refuses with constants computed in `__init__` when their inputs are fixed; reduce over
   the last dim. Check the wrapper against the model's own forward before compiling, with the
   noise as an input so both see the same.
3. Fill the manifest from the checkpoint's statistics, with the model's own `eps` values.
4. `compile`, `verify` on CPU and on the deployment backend.
5. Hand the GGUF to `vla-server`, and compare one chunk against the reference through the server,
   with the client's real preprocessing in the loop.

What this path cannot do yet, so that it is clear before the work starts: dynamic image grids (the
compile fixes the camera resolution), state carried across calls (the runtime supports it; the
serving protocol does not expose a reset), ops without a lowering ([OPS.md](OPS.md) lists them),
and per-model host logic beyond the manifest's roles.
