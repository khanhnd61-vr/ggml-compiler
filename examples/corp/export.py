"""CoRP (Compressed Representation Policy, LIBERO release) as a ggml-compiler program:
`CORP_ROOT=/path/to/CoRP ggml-compiler compile examples/corp/export.py:program --out out/corp --emit gguf`.

A model this repository did not know before, compiled from its own checkout and served by vla.cpp
through the `vla` manifest with no code added to either project (docs/VLA_CPP.md walks through it).

The compiled program is CoRP's `predict` with its Python glue removed: DINOv2 ViT-S/14 on each
224x224 view, the T5-efficient-tiny encoder on 32 token ids, the cross-attention resampler, the
state token, then the one-block DiT integrated with 10 Euler steps from the caller's noise. In and
out are what CoRP's `prepare_observation` and `Normalizer` produce and consume: ImageNet-normalized
pixels, the z-scored 8-dim state, and the chunk in [-1, 1] before `unnormalize_action`. The
manifest carries those statistics so the host does the normalization. Two things stay with the
client, as they do in CoRP's own evaluation script: the 180-degree camera rotation and the choice
of the rotation vector closest to the checkpoint's angle centres.
"""

import os
import sys

import torch
from torch import nn

import ggml_compiler
from ggml_compiler import vla

CORP_ROOT = os.environ.get("CORP_ROOT", "CoRP")
CHECKPOINT = os.environ.get("CORP_CHECKPOINT", os.path.join(CORP_ROOT, "checkpoints", "corp_libero.pt"))
INSTRUCTION = "put the bowl on the plate"


def load(root: str = CORP_ROOT, checkpoint: str = CHECKPOINT):
    """The released policy, from a CoRP checkout whose Git LFS checkpoint has been fetched."""
    if root not in sys.path:
        sys.path.insert(0, root)
    from corp.checkpoint import load_policy

    return load_policy(checkpoint)


class CoRPInfer(nn.Module):
    """CoRP's forward on plain tensors, with the two pieces torch.export cannot trace replaced by
    constants computed here: DINOv2's bicubic interpolation of its 37x37 position grid to the
    compiled image size, and T5's relative position bias for the fixed token count."""

    def __init__(self, policy):
        super().__init__()
        rx = policy.representation_extractor
        self.view_names = tuple(rx.view_names)
        size = policy.config["image_size"]
        self.vit = rx.visual_encoder.backbone
        with torch.no_grad():
            n = (size // 14) ** 2
            pos = self.vit.interpolate_pos_encoding(torch.zeros(1, n + 1, 384), size, size)
            self.register_buffer("pos_embed", pos.clone())
            lang = rx.language_encoder
            self.t5 = lang.backbone.encoder
            self.lang_norm, self.lang_proj = lang.norm, lang.projection
            self.max_len = lang.max_length
            attn = self.t5.block[0].layer[0].SelfAttention
            self.register_buffer("rel_bias", attn.compute_bias(self.max_len, self.max_len).clone())
        self.resampler = rx.cross_attention_resampler
        self.state_encoder = rx.state_encoder
        self.gen = policy.action_generator
        self.steps = self.gen.num_sample_steps

    def vision(self, image):
        vit = self.vit
        tokens = vit.patch_embed(image)
        tokens = torch.cat([vit.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1) + self.pos_embed
        for block in vit.blocks:
            tokens = block(tokens)
        return vit.norm(tokens)[:, 1:]

    def language(self, tokens, mask):
        h = self.t5.embed_tokens(tokens)
        pad = (1.0 - mask.to(h.dtype)) * torch.finfo(h.dtype).min
        bias = self.rel_bias + pad[:, None, None, :]
        for block in self.t5.block:
            h = block(h, position_bias=bias)[0]
        h = self.t5.final_layer_norm(h)
        tokens = self.lang_proj(self.lang_norm(h))
        m = mask.to(tokens.dtype)
        tokens = tokens * m.unsqueeze(-1)
        # CoRP's masked mean over the sequence, summed over the last dim (the one ggml reduces).
        return tokens, tokens.transpose(1, 2).sum(dim=-1) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(self, agentview, eye_in_hand, lang_tokens, lang_mask, state, noise):
        from corp.representation.extractor import Representation

        patches = {name: self.vision(img) for name, img in zip(self.view_names, (agentview, eye_in_hand), strict=True)}
        lang, pooled = self.language(lang_tokens, lang_mask)
        z = torch.cat([self.resampler(patches, pooled), lang, self.state_encoder(state)], dim=1)
        prefix = self.gen.prepare_prefix(Representation(tokens=z, pooled_language=pooled, action_dtype=z.dtype))
        x = noise
        for step in range(self.steps):
            tau = torch.full((1,), step / float(self.steps), dtype=x.dtype)
            x = x + self.gen(x, tau, prefix) * (1.0 / float(self.steps))
        return x


def tokenize(policy, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """What a client sends: T5 token ids (with `</s>`) padded to `max_length`, and their mask."""
    enc = policy.representation_extractor.language_encoder
    out = enc.tokenizer([text], padding="max_length", truncation=True, max_length=enc.max_length, return_tensors="pt")
    return out["input_ids"].to(torch.int32), out["attention_mask"].to(torch.int32)


def raw_state() -> torch.Tensor:
    """A plausible LIBERO state: end-effector position, rotation vector, gripper finger positions."""
    return torch.tensor([[-0.05, 0.02, 1.05, 3.0, 0.1, -0.2, 0.04, -0.04]], dtype=torch.float32)


def example_images(policy) -> list[torch.Tensor]:
    """Deterministic uint8 views, as a camera would send them: pixel (x + 2y + 40c + 17v) & 255."""
    size = policy.config["image_size"]
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return [torch.stack([(x + 2 * y + 40 * c + 17 * v) & 255 for c in range(3)]).to(torch.uint8)
            for v in range(len(policy.config["view_names"]))]


def example_inputs(policy) -> tuple[torch.Tensor, ...]:
    """The program's inputs for `raw_state()`, `INSTRUCTION` and `example_images()`, after the
    preprocessing the manifest describes, plus the noise of a fixed seed."""
    mean = torch.tensor(vla.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(vla.IMAGENET_STD).view(3, 1, 1)
    images = [((img.float() / 255 - mean) / std).unsqueeze(0) for img in example_images(policy)]
    tokens, mask = tokenize(policy, INSTRUCTION)
    state = policy.normalizer.normalize_state(raw_state())
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(1, policy.config["action_horizon"], policy.config["action_dim"], generator=g)
    return (*images, tokens, mask, state, noise)


def serving_manifest(policy) -> dict:
    """CoRP's preprocessing as manifest roles. The state is z-scored with eps 1e-6; the first six
    action columns are min/max un-normalized with no eps; the gripper column passes through, which
    `min_max` with min -1 and max 1 expresses exactly."""
    cfg, norm = policy.config, policy.normalizer
    size = cfg["image_size"]
    return vla.manifest(
        images=[vla.image(name, size, size, mean=vla.IMAGENET_MEAN, std=vla.IMAGENET_STD) for name in cfg["view_names"]],
        lang=vla.lang("lang_tokens", policy.representation_extractor.language_encoder.max_length,
                      mask="lang_mask", pad_id=0),
        state=vla.state("state", mean=norm.state_mean.tolist(), std=norm.state_std.tolist(), eps=1e-6),
        noise=vla.noise("noise"),
        actions=vla.actions("actions", chunk=cfg["action_horizon"],
                            min=[*norm.action_min[:6].tolist(), -1.0], max=[*norm.action_max[:6].tolist(), 1.0],
                            eps=0.0),
    )


def program(root: str = CORP_ROOT, checkpoint: str = CHECKPOINT) -> ggml_compiler.Program:
    policy = load(root, checkpoint)
    return ggml_compiler.Program.single(
        CoRPInfer(policy).eval(), example_inputs(policy),
        input_names=[*policy.config["view_names"], "lang_tokens", "lang_mask", "state", "noise"],
        output_names=["actions"], metadata={"vla": serving_manifest(policy)})
