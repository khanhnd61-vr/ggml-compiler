"""LeRobot GR00T N1.7 as a three-stage ggml-compiler program with a 4-step flow-matching loop:
`ggml-compiler compile examples/groot/export.py:program --out out/groot`.

backbone: Qwen3-VL (Cosmos-Reason2-2B) vision tower on the processor's patches, then the
          language model's first `select_layer` layers -> the pre-norm hidden states.
encode:   VL layer norm + self-attention over those states, and the state encoder.
step:     one Euler step of the DiT action head (alternating cross-attention to image and text
          tokens).

The prompt layout is fixed at compile time: the number of images, their patch grid, where the
image tokens sit and the prompt length. That fixes the multimodal RoPE ids, the image-token
ranges and the vision tower's per-image attention, which are compiled as constants. Token ids,
the attention mask, pixels and the state are runtime inputs. GR00T's VL self-attention has no
padding mask, so a prompt of another length needs another artifact (LeRobot runs batch 1
unpadded). The embodiment (its per-embodiment encoder/decoder weights) is fixed too.

The Qwen3-VL processor (resize, patchify, tokenize, chat template) and state/action
normalization stay with the caller, as in LeRobot's processors.
"""

import os
import types

import torch

import ggml_compiler

CHECKPOINT = os.environ.get("GROOT_CHECKPOINT", "/mnt/khanh/hf_data/torch/nvidia__GR00T-N1.7-LIBERO/libero_object")
EMBODIMENT = os.environ.get("GROOT_EMBODIMENT", "libero_sim")


def load(checkpoint: str = CHECKPOINT, n_layers: dict[str, int] | None = None):
    """The GR00T N1.7 model (LeRobot's GR00TN17, what GrootPolicy wraps) in f32 and eval mode.

    `n_layers` builds it shallower for quick tests, e.g. {"llm": 4, "dit": 4,
    "vl_self_attention": 1}; the vision tower stays whole. Only the kept layers are read from the
    checkpoint (the full model is 3.1B parameters, 12.6 GB in f32)."""
    import json
    from pathlib import Path

    from lerobot.policies.groot.groot_n1_7 import GR00TN17, _tie_unused_qwen_lm_head

    cfg = json.loads((Path(checkpoint) / "config.json").read_text())
    overrides = {}
    if n_layers:
        if "llm" in n_layers:
            overrides["select_layer"] = n_layers["llm"]
        if "dit" in n_layers:
            overrides["diffusion_model_cfg"] = {**cfg["diffusion_model_cfg"], "num_layers": n_layers["dit"]}
        if "vl_self_attention" in n_layers:
            overrides["vl_self_attention_cfg"] = {**cfg["vl_self_attention_cfg"],
                                                  "num_layers": n_layers["vl_self_attention"]}
    model = GR00TN17.from_pretrained(checkpoint, transformers_loading_kwargs={"trust_remote_code": True},
                                     use_flash_attention=False, **overrides)
    _tie_unused_qwen_lm_head(model.backbone.model)
    return model.float().eval()


def embodiment_id(checkpoint: str, embodiment: str) -> int:
    import json
    from pathlib import Path

    return int(json.loads((Path(checkpoint) / "embodiment_id.json").read_text())[embodiment])


def processor_inputs(images, instruction: str):
    """What LeRobot's GR00T N1.7 preprocessor hands the model for one sample: the Qwen3-VL chat
    template over the camera images (PIL) and the instruction."""
    from lerobot.policies.groot.processor_groot import _build_n1_7_processor

    proc = _build_n1_7_processor()
    conv = [{"role": "user", "content": [*[{"type": "image", "image": im} for im in images],
                                         {"type": "text", "text": instruction}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    return dict(proc(text=[text], images=list(images), return_tensors="pt", padding=True))


def _static_vision_attention(lengths: list[int]):
    """Qwen3VLVisionAttention.forward with the per-image split sizes as Python ints. HF computes
    them from cu_seqlens with .tolist(), which torch.export cannot trace."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    def forward(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        n = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        outs = [torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=self.scaling)
                for qi, ki, vi in zip(q.split(lengths, 2), k.split(lengths, 2), v.split(lengths, 2))]
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(n, -1)
        return self.proj(out)

    return forward


class Stages:
    def __init__(self, model, example: dict[str, torch.Tensor], emb_id: int):
        self.m = model
        qwen = model.backbone.model.model  # Qwen3VLModel
        self.visual, self.lm = qwen.visual, qwen.language_model
        head = model.action_head
        self.head = head
        self.n = head.num_inference_timesteps
        self.emb = torch.tensor([emb_id])

        grid = example["image_grid_thw"]
        ids = example["input_ids"][0]
        self.img = ids == model.backbone.model.config.image_token_id  # (L,) compile-time layout
        idx = self.img.nonzero().flatten().tolist()
        self.ranges = []  # contiguous runs of image tokens
        for i in idx:
            if self.ranges and self.ranges[-1][1] == i:
                self.ranges[-1][1] = i + 1
            else:
                self.ranges.append([i, i + 1])
        with torch.no_grad():
            self.pos_embeds = self.visual.fast_pos_embed_interpolate(grid)
            rot = self.visual.rot_pos_emb(grid).reshape(self.pos_embeds.shape[0], -1)
            emb = torch.cat((rot, rot), dim=-1)
            self.vis_pe = (emb.cos(), emb.sin())
            # GR00T N1.7 (backbone._ensure_legacy_qwen3_position_ids): 3 M-RoPE rows plus a text row.
            pos = qwen.compute_3d_position_ids(input_ids=example["input_ids"], inputs_embeds=None,
                                               image_grid_thw=grid, attention_mask=example["attention_mask"],
                                               mm_token_type_ids=example["mm_token_type_ids"])
            qwen.rope_deltas = None
            self.text_pos, pos3 = pos[:1][0], pos
            self.text_pe = self.lm.rotary_emb(torch.zeros(1, ids.shape[0], 1), pos3)
        lengths = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
        for blk in self.visual.blocks:
            blk.attn.forward = types.MethodType(_static_vision_attention(lengths), blk.attn)

    def _place(self, base: torch.Tensor, rows: torch.Tensor, add: bool) -> torch.Tensor:
        """base (1, L, D) with `rows` (one per image token, in order) written into (or added to)
        the image-token positions: masked_scatter and HF's deepstack update, at fixed positions."""
        parts, last, k = [], 0, 0
        for a, e in self.ranges:
            parts.append(base[:, last:a])
            r = rows[k: k + e - a].unsqueeze(0)
            parts.append(base[:, a:e] + r if add else r)
            last, k = e, k + e - a
        parts.append(base[:, last:])
        return torch.cat(parts, dim=1)

    def backbone(self, pixel_values, input_ids, attention_mask):
        v = self.visual
        h = v.patch_embed(pixel_values) + self.pos_embeds
        deepstack = []
        for i, blk in enumerate(v.blocks):
            h = blk(h, cu_seqlens=None, position_embeddings=self.vis_pe)
            if i in v.deepstack_visual_indexes:
                deepstack.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(i)](h))
        image_embeds = v.merger(h)

        x = self._place(self.lm.embed_tokens(input_ids), image_embeds, add=False)
        n = input_ids.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool).tril()
        allowed = causal[None, None] & (attention_mask[:, None, None, :] > 0)
        mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        for i, layer in enumerate(self.lm.layers):
            x = layer(x, attention_mask=mask, position_ids=self.text_pos, position_embeddings=self.text_pe)
            features = x  # N1.7's head reads the last kept layer's own output: before the final norm,
            if i < len(deepstack):  # and before a deepstack addition after it
                x = self._place(x, deepstack[i], add=True)
        return features

    def encode(self, features, state):
        feats = self.head.vl_self_attention(self.head.vlln(features))
        state_features = self.head.state_encoder(state.view(state.shape[0], 1, -1), self.emb)
        return feats, state_features

    def step(self, x, t, vl, state_features, attention_mask):
        head = self.head
        ts = t  # the bucketed timestep, as a float: every use of it casts to float first
        action_features = head.action_encoder(x, ts, self.emb)
        if head.config.add_pos_embed:
            pos = torch.arange(action_features.shape[1])
            action_features = action_features + head.position_embedding(pos).unsqueeze(0)
        sa = torch.cat((state_features, action_features), dim=1)
        out = head.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts, image_mask=self.img[None],
                         backbone_attention_mask=attention_mask > 0)
        pred = head.action_decoder(out, self.emb)
        return x + (1.0 / self.n) * pred[:, -head.config.action_horizon:]

    def timesteps(self) -> list[float]:
        h = self.head
        return [float(int(i / float(self.n) * h.num_timestep_buckets)) for i in range(self.n)]


def example_inputs(model, instruction: str = "pick up the alphabet soup and place it in the basket",
                   cameras: int = 2, seed: int = 0):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    images = [Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)) for _ in range(cameras)]
    enc = processor_inputs(images, instruction)
    head = model.action_head
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(1, head.config.state_history_length, head.config.max_state_dim, generator=g) * 0.5
    noise = torch.randn(1, head.config.action_horizon, head.action_dim, generator=g)
    return enc, state, noise


def program_from_model(model, checkpoint: str = CHECKPOINT, embodiment: str = EMBODIMENT) -> ggml_compiler.Program:
    enc, state, noise = example_inputs(model)
    s = Stages(model, enc, embodiment_id(checkpoint, embodiment))
    bb_ex = (enc["pixel_values"], enc["input_ids"], enc["attention_mask"])
    with torch.no_grad():
        feats = s.backbone(*bb_ex)
        vl, sf = s.encode(feats, state)
    return ggml_compiler.Program(
        root=model,
        stages={
            "backbone": ggml_compiler.Stage(s.backbone, bb_ex),
            "encode": ggml_compiler.Stage(s.encode, (feats, state)),
            "step": ggml_compiler.Stage(s.step, (noise, torch.zeros(1), vl, sf, enc["attention_mask"])),
        },
        driver=ggml_compiler.Driver(
            inputs=["pixel_values", "input_ids", "attention_mask", "state", "noise"],
            body=[
                ggml_compiler.call("backbone", ["pixel_values", "input_ids", "attention_mask"], ["features"]),
                ggml_compiler.call("encode", ["features", "state"], ["vl", "state_features"]),
                ggml_compiler.loop(
                    s.n, carry={"x": "noise"}, scalars={"t": s.timesteps()},
                    body=[ggml_compiler.call("step", ["x", "t", "vl", "state_features", "attention_mask"], ["x"])]),
            ],
            outputs=["x"],
        ),
        specialized={
            "prompt_layout": f"{enc['input_ids'].shape[1]} tokens, {len(enc['image_grid_thw'])} images, "
                             f"patch grids {enc['image_grid_thw'].tolist()}",
            "embodiment": embodiment,
        },
    )


def program(checkpoint: str = CHECKPOINT, embodiment: str = EMBODIMENT) -> ggml_compiler.Program:
    return program_from_model(load(checkpoint), checkpoint, embodiment)
