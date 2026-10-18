"""LeRobot SmolVLA as a two-stage ggml-compiler program with a 10-step denoising loop:
`ggml-compiler compile examples/smolvla/export.py:program --out out/smolvla`.

prefix: images, image masks, language tokens and mask, state -> VLM key/value cache and prefix
        padding mask.
step:   one flow-matching step: velocity from the action expert, then the Euler update.
Image resizing/normalization, tokenization and state/action (un)normalization stay with the
caller, as in LeRobot's processors.
"""

import os

import torch

import ggml_compiler

CHECKPOINT = os.environ.get("SMOLVLA_CHECKPOINT", "khanhnd61/smolvla_so101-multi-task-clean")


def load(checkpoint: str = CHECKPOINT):
    """Load a SmolVLA checkpoint. Config fields this LeRobot does not know are dropped when they
    are empty (checkpoints from forks with layer pruning carry `prune_vlm_layers: []`)."""
    import dataclasses
    import json
    import tempfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = Path(checkpoint) / "config.json" if Path(checkpoint).is_dir() else hf_hub_download(checkpoint, "config.json")
    raw = json.loads(Path(path).read_text())
    known = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
    for key in [k for k in raw if k not in known]:
        if raw[key] not in ([], {}, None, 0, False):
            raise ValueError(f"{checkpoint}: config field {key}={raw[key]!r} is not supported by this LeRobot")
        del raw[key]
    raw["load_vlm_weights"] = False  # every weight comes from the checkpoint
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(raw))
        config = SmolVLAConfig.from_pretrained(d)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).eval()
    # LeRobot keeps the VLM's layers in bf16 and casts activations to match. Run everything in f32
    # (the weights keep their bf16-rounded values) so PyTorch and ggml-compiler compute the same thing.
    return policy.float()


class Stages:
    """The two stage functions over one VLAFlowMatching model."""

    def __init__(self, fm):
        self.fm = fm
        self.n = fm.config.num_steps
        self.dt = -1.0 / self.n

    def prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        fm = self.fm
        embs, pad, att = fm.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        _, cache = fm.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, att),
            position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        # The HF DynamicCache crosses the stage boundary as plain tensors: (keys, values) per layer.
        return [(layer.keys, layer.values) for layer in cache.layers], pad

    def step(self, x, t, kv, pad):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(k, v, i)
        v = self.fm.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=t)
        return x + self.dt * v

    def timesteps(self) -> list[float]:
        return [1.0 + i * self.dt for i in range(self.n)]


def example_inputs(config):
    """Static shapes from the config; deterministic contents."""
    g = torch.Generator().manual_seed(0)
    cams = list(config.image_features)
    h, w = config.resize_imgs_with_padding
    images = [torch.rand(1, 3, h, w, generator=g) * 2 - 1 for _ in cams]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in cams]
    n_tok = config.tokenizer_max_length
    lang_tokens = torch.randint(0, 1000, (1, n_tok), generator=g)
    lang_masks = torch.zeros(1, n_tok, dtype=torch.bool)
    lang_masks[:, :12] = True  # a 12-token instruction, the rest padding
    state = torch.randn(1, config.max_state_dim, generator=g) * 0.5
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, generator=g)
    return images, img_masks, lang_tokens, lang_masks, state, noise


def program(checkpoint: str = CHECKPOINT) -> ggml_compiler.Program:
    return program_from_policy(load(checkpoint))

def quantize_expert(policy, cfg_name: str = "INT8_DEFAULT_CFG"):
    """Quantize only the action expert with NVIDIA ModelOpt, calibrating on one example sample.
    Compile the result with `ggml_compiler.compile(..., prequant=ggml_compiler.ModelOpt())`."""
    import copy

    import modelopt.torch.quantization as mtq

    cfg = copy.deepcopy(getattr(mtq, cfg_name))
    entries = [{"quantizer_name": "*", "enable": False}]
    for e in cfg["quant_cfg"]:
        name = e.get("quantizer_name", "")
        if name in ("*weight_quantizer", "*input_quantizer"):
            entries.append({**e, "quantizer_name": "*lm_expert*" + name[1:]})
    cfg["quant_cfg"] = entries

    s = Stages(policy.model)
    images, img_masks, lang_tokens, lang_masks, state, noise = example_inputs(policy.config)

    def calibrate(model):
        with torch.no_grad():
            kv, pad = s.prefix(images, img_masks, lang_tokens, lang_masks, state)
            x = noise
            for t in s.timesteps():
                x = s.step(x, torch.full((1,), t), kv, pad)

    mtq.quantize(policy.model, cfg, calibrate)
    return policy


def program_from_policy(policy) -> ggml_compiler.Program:
    """Like program(), for an already loaded (and possibly quantized) policy."""
    s = Stages(policy.model)
    images, img_masks, lang_tokens, lang_masks, state, noise = example_inputs(policy.config)
    prefix_ex = (images, img_masks, lang_tokens, lang_masks, state)
    with torch.no_grad():
        kv, pad = s.prefix(*prefix_ex)
    return ggml_compiler.Program(
        root=policy.model,
        stages={
            "prefix": ggml_compiler.Stage(s.prefix, prefix_ex),
            "step": ggml_compiler.Stage(s.step, (noise, torch.ones(1), kv, pad)),
        },
        driver=ggml_compiler.Driver(
            inputs=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
            body=[
                ggml_compiler.call("prefix", ["images", "img_masks", "lang_tokens", "lang_masks", "state"], ["kv",
                                                                                                             "pad"]),
                ggml_compiler.loop(s.n, carry={"x": "noise"}, scalars={"t": s.timesteps()},
                                   body=[ggml_compiler.call("step", ["x", "t", "kv", "pad"], ["x"])]),
            ],
            outputs=["x"],
        ),
    )
