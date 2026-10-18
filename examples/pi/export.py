"""LeRobot pi0 and pi0.5 as two-stage ggmlc programs with a 10-step flow-matching loop:
`ggmlc compile examples/pi/export.py:program --out out/pi0` (PI_CHECKPOINT picks the checkpoint;
its config says which of the two it is).

prefix: camera images (SigLIP), image masks, language tokens and mask -> the PaliGemma key/value
        cache and the prefix padding mask.
step:   one Euler step: the Gemma action expert reads the noisy actions and the timestep, attends
        to the cached prefix, and predicts the velocity. pi0 also reads the state as a token;
        pi0.5 has the (discretized) state in its prompt and conditions the expert's adaptive
        RMSNorms on the timestep instead.
Image resizing to 224x224 and [-1, 1] scaling, tokenization, state padding / discretization and
(un)normalization stay with the caller, as in LeRobot's processors.
"""

import contextlib
import gc
import os

import torch

import ggmlc

CHECKPOINT = os.environ.get("PI_CHECKPOINT", "lerobot/pi0_libero_finetuned_v044")


def _classes(checkpoint: str):
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    path = Path(checkpoint) / "config.json" if Path(checkpoint).is_dir() else hf_hub_download(checkpoint, "config.json")
    kind = json.loads(Path(path).read_text())["type"]
    if kind == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Config, PI0Policy
    if kind == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Config, PI05Policy
    raise ValueError(f"{checkpoint} is a {kind!r} checkpoint, not pi0 or pi05")


class _Key:
    """Stands in for a checkpoint tensor while LeRobot's key fix-ups rename the keys."""

    def __init__(self, name: str) -> None:
        self.name = name

    def clone(self) -> "_Key":
        return self


def load(checkpoint: str = CHECKPOINT, n_layers: int | None = None):
    """Load a pi0 / pi0.5 checkpoint for compilation: torch.compile and gradient checkpointing off,
    every weight in f32 (the checkpoint's bf16 values, exactly). `n_layers` builds the language
    model and the expert with only their first layers: the same graph, small enough for quick
    parity tests.

    LeRobot's from_pretrained randomly initializes the model, reads the whole checkpoint, then
    copies it in (200 s and 14 GB peak for 3B parameters). This builds the model without
    initializing it, drops the unused LM head (2 GB in f32), and copies the checkpoint in one
    tensor at a time, so the f32 model is all that is ever held."""
    import importlib
    from pathlib import Path

    from safetensors import safe_open
    from transformers.initialization import no_init_weights
    from transformers.utils import cached_file

    config_cls, policy_cls = _classes(checkpoint)
    config = config_cls.from_pretrained(checkpoint)
    config.compile_model = False
    config.gradient_checkpointing = False
    config.dtype = "float32"

    modeling = importlib.import_module(policy_cls.__module__)
    depth = contextlib.nullcontext()
    if n_layers is not None:
        full = modeling.get_gemma_config

        def shallow(variant):
            c = full(variant)
            c.depth = n_layers
            return c

        depth = _patched(modeling, "get_gemma_config", shallow)
    with no_init_weights(), depth:
        policy = policy_cls(config).eval()
    policy.model.paligemma_with_expert.paligemma.lm_head = None

    path = Path(checkpoint) / "model.safetensors" if Path(checkpoint).is_dir() else cached_file(checkpoint, "model.safetensors")
    with safe_open(str(path), framework="pt") as f:
        fixed = policy._fix_pytorch_state_dict_keys({k: _Key(k) for k in f.keys()}, policy.config)
        source = {(k if k.startswith("model.") else f"model.{k}"): v.name for k, v in fixed.items()}
        params = policy.state_dict()
        missing = [k for k in params if k not in source]
        if missing:
            raise KeyError(f"{checkpoint} lacks {len(missing)} weights, e.g. {missing[:3]}")
        with torch.no_grad():
            for k, t in params.items():
                t.copy_(f.get_tensor(source[k]))
    gc.collect()
    return policy


@contextlib.contextmanager
def _patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


class Stages:
    """The two stage functions over one PI0Pytorch / PI05Pytorch model."""

    def __init__(self, model):
        self.m = model
        self.pi05 = type(model).__name__ == "PI05Pytorch"
        self.n = model.config.num_inference_steps
        self.dt = -1.0 / self.n

    def prefix(self, images, img_masks, lang_tokens, lang_masks):
        from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d

        m = self.m
        embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, cache = m.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        # The HF DynamicCache crosses the stage boundary as plain tensors: (keys, values) per layer.
        return [(layer.keys, layer.values) for layer in cache.layers], pad

    def step(self, x, t, kv, pad, state=None):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(k, v, i)
        self.m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        kw = {} if self.pi05 else {"state": state}
        v = self.m.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=t, **kw)
        return x + self.dt * v

    def timesteps(self) -> list[float]:
        return [1.0 + i * self.dt for i in range(self.n)]


def example_inputs(config, n_tokens: int = 12, empty_cameras: bool = False):
    """Static shapes from the config, deterministic contents.

    LeRobot feeds each of the checkpoint's `empty_cameras` as an image of -1 with a false mask.
    Its tokens are masked out of attention and add nothing to the position ids, so leaving them
    out gives the same actions for a third less vision and prefix work (vla.cpp does the same).
    empty_cameras=True keeps them."""
    g = torch.Generator().manual_seed(0)
    cams = [c for c in config.image_features if empty_cameras or "empty_camera" not in c]
    h, w = config.image_resolution
    images, img_masks = [], []
    for c in cams:
        empty = "empty_camera" in c
        images.append(-torch.ones(1, 3, h, w) if empty else torch.rand(1, 3, h, w, generator=g) * 2 - 1)
        img_masks.append(torch.tensor([not empty]))
    n_tok = config.tokenizer_max_length
    lang_tokens = torch.randint(0, 1000, (1, n_tok), generator=g)
    lang_masks = torch.zeros(1, n_tok, dtype=torch.bool)
    lang_masks[:, :n_tokens] = True  # an instruction, the rest padding
    state = torch.randn(1, config.max_state_dim, generator=g) * 0.5
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=g)
    return images, img_masks, lang_tokens, lang_masks, state, noise


def program_from_policy(policy, empty_cameras: bool = False, n_tokens: int = 12) -> ggmlc.Program:
    s = Stages(policy.model)
    images, img_masks, lang_tokens, lang_masks, state, noise = example_inputs(
        policy.config, n_tokens=n_tokens, empty_cameras=empty_cameras)
    prefix_ex = (images, img_masks, lang_tokens, lang_masks)
    with torch.no_grad():
        kv, pad = s.prefix(*prefix_ex)
    step_args = ["x", "t", "kv", "pad"] + ([] if s.pi05 else ["state"])
    step_ex = (noise, torch.ones(1), kv, pad) + (() if s.pi05 else (state,))
    return ggmlc.Program(
        root=policy.model,
        stages={"prefix": ggmlc.Stage(s.prefix, prefix_ex), "step": ggmlc.Stage(s.step, step_ex)},
        driver=ggmlc.Driver(
            inputs=["images", "img_masks", "lang_tokens", "lang_masks"] + ([] if s.pi05 else ["state"]) + ["noise"],
            body=[
                ggmlc.call("prefix", ["images", "img_masks", "lang_tokens", "lang_masks"], ["kv", "pad"]),
                ggmlc.loop(s.n, carry={"x": "noise"}, scalars={"t": s.timesteps()},
                           body=[ggmlc.call("step", step_args, ["x"])]),
            ],
            outputs=["x"],
        ),
        specialized={} if empty_cameras else {"empty_cameras": "dropped (LeRobot feeds them as -1 images, masked out)"},
    )


def program(checkpoint: str = CHECKPOINT, n_layers: int | None = None) -> ggmlc.Program:
    return program_from_policy(load(checkpoint, n_layers))
