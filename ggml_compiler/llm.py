"""Causal language models: a Hugging Face model as a ggml-compiler program with a KV cache.

    program = ggml_compiler.causal_lm(model, max_context=4096)
    art = ggml_compiler.compile(program, out_dir="out/qwen3")
    with ggml_compiler.Session(art.gguf) as s:
        gen = ggml_compiler.Generator(s)
        ids = gen.generate(prompt_ids, max_new=64)

The program has two entry points over one KV cache, which is state: it lives on the device, is
written in place at runtime positions (ggml_set_rows), and persists across runs.

- prefill(tokens [1, s], positions [s], mask [1, 1, s, n_kv]) -> logits [1, 1, vocab]
  (or [1, s, vocab] with logits="all"): s new tokens, 2 <= s <= max_prompt.
- decode(token [1, 1], position [1], mask1 [1, 1, 1, n_kv]) -> logits1 [1, 1, vocab].

A vision-language model (SmolVLM / Idefics3) adds an image buffer, also state, and two entries:

- encode(pixels [tiles, 3, H, W]) -> nothing: the vision tower and projector write one row per
  image token into the buffer (rows 0 .. tiles * rows_per_tile), 2 <= tiles <= max_tiles.
- prefill_mm(tokens [1, s], gather [s], positions, mask) -> logits: as prefill, but token i's
  input embedding is row gather[i] of [image rows; embeddings of this call's tokens], i.e.
  k for the k-th image row and image_rows + i for token i itself.

Attention reads the first n_kv cache rows, n_kv a multiple of 256 (from 512 to max_context: a
symbol, so each call's graph matches the context in use, as in llama.cpp). The mask is additive:
0 where token i may attend to position j, -inf elsewhere. `Generator` builds positions and
masks, chunks long prompts, and samples greedily; C callers do the same with the metadata the
artifact records (ggml_compiler.meta: max_context, max_prompt, kv_block, eos).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.export import Dim

from . import api

KV_BLOCK = 256


def _cache_class():
    from transformers.cache_utils import Cache

    class FunctionalCache(Cache):
        """A KV cache as plain tensors, one [kv_heads, max_context, head_dim] per layer for K and
        for V. update() writes the new rows at `positions` (index_copy: ggml_set_rows) and returns
        the first n_kv rows for attention; `out` holds the updated tensors to return. `slots` maps
        a layer to its K/V pair when not every layer has one (layers that share another's K/V)."""

        def __init__(self, kv: Sequence[torch.Tensor], positions: torch.Tensor, n_kv: Any,
                     slots: dict[int, int] | None = None) -> None:
            self.kv, self.out, self.positions, self.n_kv = list(kv), list(kv), positions, n_kv
            self.slots = slots or {}
            self.layers = []

        def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, *args: Any, **kwargs: Any):
            read = []
            slot = self.slots.get(layer_idx, layer_idx)
            for j, x in enumerate((key, value)):
                c = self.kv[2 * slot + j]
                c = c.index_copy(1, self.positions, x[0].to(c.dtype))
                self.out[2 * layer_idx + j] = c
                read.append(c[:, : self.n_kv][None].to(x.dtype))
            return read[0], read[1]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return 0

        @property
        def is_compileable(self) -> bool:
            return True

    return FunctionalCache


class _LM(torch.nn.Module):
    """Stage functions for a causal LM whose RoPE positions are its cache positions (Llama, Qwen3,
    SmolLM). Subclasses add an image path through `images`."""

    rope = False  # True: text entries take RoPE positions (`rpos`) apart from cache positions

    def __init__(self, model: torch.nn.Module, keep: int, max_image_tokens: int | None) -> None:
        super().__init__()
        self.model = model
        object.__setattr__(self, "_cache", _cache_class())
        self.keep = keep
        self.slots: dict[int, int] = {}  # layer -> K/V pair, set by causal_lm from kv_layout

    def kv_layout(self, config: Any) -> list[tuple[int, int, int]]:
        """(layer, kv_heads, head_dim) of each layer with K and V in the cache, in cache order."""
        layers, kv_heads, head_dim = _geometry(config)
        return [(i, kv_heads, head_dim) for i in range(layers)]

    def _lm(self, keep: int, positions: torch.Tensor, mask: torch.Tensor, kv: list[torch.Tensor],
            tokens: torch.Tensor | None = None, embeds: torch.Tensor | None = None):
        cache = self._cache(kv, positions, mask.shape[-1], self.slots)
        out = self.model(input_ids=tokens, inputs_embeds=embeds, position_ids=positions[None], attention_mask=mask,
                         past_key_values=cache, use_cache=True, logits_to_keep=keep)
        return out.logits, cache.out

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor, kv: list[torch.Tensor]):
        return self._lm(self.keep, positions, mask, kv, tokens=tokens)

    def greedy(self, token: torch.Tensor, position: torch.Tensor, mask: torch.Tensor, kv: list[torch.Tensor]):
        """One decode step that returns the most likely next token instead of logits."""
        logits, out = self._lm(1, position, mask, kv, tokens=token)
        return logits[:, -1:].argmax(-1), out

    def images(self, b: "_Build") -> None:
        """Add the image entries (encode, prefill_mm), their state and metadata to b."""

    def _merge(self, tokens: torch.Tensor, gather: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Input embeddings mixing image rows and tokens: row gather[i] of [image rows; token
        embeddings]. One get_rows, as transformers' masked_scatter / where()."""
        emb = self.model.get_input_embeddings()(tokens)[0]
        return torch.nn.functional.embedding(gather, torch.cat([rows, emb]))[None]


class _SmolVLM(_LM):
    """SmolVLM / Idefics3: SigLIP over square tiles, pixel shuffle, a linear projector.

    Only whole tiles, which is what the processor makes (each image is resized to a grid of full
    tiles plus one downscaled global tile): transformers' bucketized patch position ids are then
    0 .. n-1, and its padding-image and patch-mask logic has nothing to do."""

    def __init__(self, model: torch.nn.Module, keep: int, max_image_tokens: int | None) -> None:
        super().__init__(model, keep, max_image_tokens)
        cfg = model.config
        vc = cfg.vision_config
        self.tile = (3, vc.image_size, vc.image_size)
        self.rows_per_tile = (vc.image_size // vc.patch_size) ** 2 // cfg.scale_factor ** 2
        # default: one image at the processor's defaults, a 4 x 4 grid + the global tile
        self.max_tiles = 17 if max_image_tokens is None else max_image_tokens // self.rows_per_tile

    def encode(self, pixels: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Image features into the first rows of the image buffer (index_copy: ggml_set_rows)."""
        m = self.model.model
        vm = m.vision_model
        e = vm.embeddings
        x = e.patch_embedding(pixels).flatten(2).transpose(1, 2) + e.position_embedding.weight
        x = vm.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state
        x = m.connector(vm.post_layernorm(x))
        f = x.reshape(-1, x.shape[-1])
        return rows.index_copy(0, torch.arange(f.shape[0]), f)

    def step_mm(self, tokens: torch.Tensor, gather: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor,
                kv: list[torch.Tensor], rows: torch.Tensor):
        return self._lm(self.keep, positions, mask, kv, embeds=self._merge(tokens, gather, rows))

    def images(self, b: "_Build") -> None:
        if self.max_tiles < 3:
            raise ValueError(f"room for {self.max_tiles} tiles: at least 3 are needed "
                             "(torch.export makes a dynamic dim's min 2)")
        n_rows = self.max_tiles * self.rows_per_tile
        rows = torch.zeros(n_rows, b.hidden)
        b.state["image_rows"] = rows
        b.stages["encode"] = api.Stage(self.encode, (torch.randn(2, *self.tile), rows),
                                       dynamic_shapes=({0: Dim("tiles", min=2, max=self.max_tiles)}, None))
        tokens, positions, mask, kv = b.prefill_ex
        b.stages["prefill_mm"] = api.Stage(self.step_mm, (tokens, b.gather_ex(n_rows), positions, mask, kv, rows),
                                           dynamic_shapes=({1: b.S}, {0: b.S}, {0: b.S}, {2: b.S, 3: b.N}, None, None))
        b.drivers["encode"] = api.Driver(inputs=["pixels"], outputs=[],
                                         body=[api.call("encode", ["pixels", "image_rows"], ["image_rows"])])
        b.drivers["prefill_mm"] = api.Driver(
            inputs=["tokens", "gather", "positions", "mask"], outputs=["logits"],
            body=[api.call("prefill_mm", ["tokens", "gather", "positions", "mask", "kv", "image_rows"], ["logits", "kv"])])
        b.metadata["image"] = {"kind": "tiles", "token": self.model.config.image_token_id, "rows": n_rows,
                               "rows_per_tile": self.rows_per_tile, "tile": list(self.tile), "max_tiles": self.max_tiles}


class _Qwen3VL(_LM):
    """Qwen3-VL: multimodal RoPE and deepstack.

    The stages call the decoder layers directly. RoPE positions are their own input: text after an
    image continues from the image's start + max(grid h, w), not from its cache row. Text entries
    take one position per token (the three M-RoPE streams are then equal, and the angles are
    transformers' own `inv_freq @ positions`, which fuse_rope recognizes); prefill_mm takes all
    three ([3, s]) and gathers each frequency's stream, as transformers' interleaved M-RoPE does.
    Deepstack: the vision tower's features from three of its layers are added to the hidden
    states after the first three decoder layers, at the image tokens (three more state buffers,
    gathered with a zero row for text tokens).

    Vision: one image per encode, `pixels` [n, C*T*P*P] as the processor flattens them, n = 4 *
    vision_tokens (the 2 x 2 merge). The tables that depend on the image grid come from the
    caller, as `Generator.encode` computes them with transformers' own code: pos_idx / pos_w
    [4, n], the bilinear interpolation of the learned position embedding, and vpos [n, 2], each
    patch's (row, col) for the 2-D rotary embedding."""

    rope = True

    def __init__(self, model: torch.nn.Module, keep: int, max_image_tokens: int | None) -> None:
        super().__init__(model, keep, max_image_tokens)
        qm = model.model
        self.lm, self.visual = qm.language_model, qm.visual
        rot = self.lm.rotary_emb
        sec = rot.mrope_section
        axis = torch.zeros(rot.inv_freq.shape[0], dtype=torch.long)
        for d, off in ((1, 1), (2, 2)):  # apply_interleaved_mrope: [T H W T H W ... T T T T]
            axis[off: sec[d] * 3: 3] = d
        object.__setattr__(self, "_axis", axis)
        self.max_rows = 1024 if max_image_tokens is None else max_image_tokens
        self.n_deep = len(self.visual.deepstack_visual_indexes)

    def _decoder(self, keep: int, x: torch.Tensor, positions: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                 mask: torch.Tensor, kv: list[torch.Tensor], deep: list[torch.Tensor] | None = None):
        cache = self._cache(kv, positions, mask.shape[-1], self.slots)
        for i, layer in enumerate(self.lm.layers):
            x = layer(x, position_embeddings=rope, attention_mask=mask, past_key_values=cache)
            if deep is not None and i < len(deep):
                x = x + deep[i]
        x = self.lm.norm(x)
        return self.model.lm_head(x[:, -keep:] if keep else x), cache.out

    def _cos_sin(self, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat((freqs, freqs), dim=-1)
        a = self.lm.rotary_emb.attention_scaling
        return emb.cos() * a, emb.sin() * a

    def _text_rope(self, rpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.lm.rotary_emb.inv_freq
        return self._cos_sin((inv[None, :, None].float() @ rpos[None, None, :].float()).transpose(1, 2))

    def _mrope(self, mrope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.lm.rotary_emb.inv_freq
        streams = torch.nn.functional.embedding(self._axis, mrope.float())  # [D/2, s]: each frequency's stream
        return self._cos_sin((streams.transpose(0, 1) * inv)[None])

    def step(self, tokens, positions, rpos, mask, kv):  # type: ignore[override]
        return self._decoder(self.keep, self.lm.embed_tokens(tokens), positions, self._text_rope(rpos), mask, kv)

    def greedy(self, token, position, rpos, mask, kv):  # type: ignore[override]
        logits, out = self._decoder(1, self.lm.embed_tokens(token), position, self._text_rope(rpos), mask, kv)
        return logits[:, -1:].argmax(-1), out

    def step_mm(self, tokens, gather, positions, mrope, mask, kv, rows, deep):
        x = self._merge(tokens, gather, rows)
        g = gather.clamp(max=self.max_rows)  # text tokens read the zero row after the image rows
        add = [torch.nn.functional.embedding(g, torch.cat([d, d.new_zeros(1, d.shape[1])]))[None] for d in deep]
        return self._decoder(self.keep, x, positions, self._mrope(mrope), mask, kv, add)

    def encode(self, pixels, pos_idx, pos_w, vpos, rows, deep):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

        v = self.visual
        n = pixels.shape[0]
        pe = [torch.nn.functional.embedding(pos_idx[k], v.pos_embed.weight) * pos_w[k][:, None] for k in range(4)]
        h = v.patch_embed(pixels) + (pe[0] + pe[1] + pe[2] + pe[3])
        rot = (vpos.float()[:, :, None] * v.rotary_pos_emb.inv_freq).flatten(1)
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        feats = []
        for i, blk in enumerate(v.blocks):
            a = blk.attn
            q, k, val = a.qkv(blk.norm1(h)).reshape(n, 3, a.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q, k, val = (t.transpose(0, 1)[None] for t in (q, k, val))
            o = torch.nn.functional.scaled_dot_product_attention(q, k, val, scale=a.scaling)
            h = h + a.proj(o[0].transpose(0, 1).reshape(n, -1))
            h = h + blk.mlp(blk.norm2(h))
            if i in v.deepstack_visual_indexes:
                feats.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(i)](h))
        merged = v.merger(h)
        idx = torch.arange(merged.shape[0])
        return rows.index_copy(0, idx, merged), [d.index_copy(0, idx, f) for d, f in zip(deep, feats)]

    def images(self, b: "_Build") -> None:
        vc = self.model.config.vision_config
        merge = vc.spatial_merge_size ** 2
        patch = vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2
        rows = torch.zeros(self.max_rows, b.hidden)
        deep = [torch.zeros(self.max_rows, b.hidden) for _ in range(self.n_deep)]
        b.state["image_rows"], b.state["deep"] = rows, deep
        side = int(vc.num_position_embeddings ** 0.5)
        m = 16  # example: a 4 x 4 merged grid (8 x 8 patches)
        n = merge * m
        ex = (torch.randn(n, patch), torch.randint(0, side * side, (4, n)), torch.rand(4, n),
              torch.randint(0, 8, (n, 2)), rows, deep)
        P = merge * Dim("vision_tokens", min=2, max=self.max_rows)
        b.stages["encode"] = api.Stage(self.encode, ex, dynamic_shapes=({0: P}, {1: P}, {1: P}, {0: P}, None, None))
        tokens, positions, rpos, mask, kv = b.prefill_ex
        mrope = torch.stack([rpos, rpos + 1, rpos + 2])
        b.stages["prefill_mm"] = api.Stage(
            self.step_mm, (tokens, b.gather_ex(self.max_rows), positions, mrope, mask, kv, rows, deep),
            dynamic_shapes=({1: b.S}, {0: b.S}, {0: b.S}, {1: b.S}, {2: b.S, 3: b.N}, None, None, None))
        b.drivers["encode"] = api.Driver(
            inputs=["pixels", "pos_idx", "pos_w", "vpos"], outputs=[],
            body=[api.call("encode", ["pixels", "pos_idx", "pos_w", "vpos", "image_rows", "deep"], ["image_rows", "deep"])])
        b.drivers["prefill_mm"] = api.Driver(
            inputs=["tokens", "gather", "positions", "mrope", "mask"], outputs=["logits"],
            body=[api.call("prefill_mm", ["tokens", "gather", "positions", "mrope", "mask", "kv", "image_rows", "deep"],
                           ["logits", "kv"])])
        b.metadata["image"] = {"kind": "qwen3_vl", "token": self.model.config.image_token_id, "rows": self.max_rows,
                               "patch": patch, "merge": vc.spatial_merge_size, "pos_side": side}


class _Gemma4(_LM):
    """Gemma 4, text only (from Gemma4ForCausalLM, or the language model of
    Gemma4ForConditionalGeneration).

    Sliding-window layers see the last `sliding_window` positions: their mask is the caller's
    causal mask with older positions masked out, made in the graph from the positions, so the
    caller still passes one mask. Layers from num_kv_shared_layers on reuse an earlier layer's K/V
    and have no cache of their own; full-attention layers have their own head size
    (global_head_dim). The per-layer embeddings are read from the token ids, as in transformers."""

    def __init__(self, model: torch.nn.Module, keep: int, max_image_tokens: int | None) -> None:
        super().__init__(model, keep, max_image_tokens)
        inner = model.model
        self.text = getattr(inner, "language_model", inner)
        cfg = model.config.get_text_config()
        self.window, self.softcap = cfg.sliding_window, cfg.final_logit_softcapping

    def kv_layout(self, config: Any) -> list[tuple[int, int, int]]:
        return [(i, a.k_proj.out_features // a.head_dim, a.head_dim)
                for i, a in enumerate(layer.self_attn for layer in self.text.layers) if not a.is_kv_shared_layer]

    def _lm(self, keep, positions, mask, kv, tokens=None, embeds=None):  # type: ignore[override]
        cache = self._cache(kv, positions, mask.shape[-1], self.slots)
        old = positions[:, None] - torch.arange(mask.shape[-1]) >= self.window  # [s, n_kv]
        masks = {"full_attention": mask, "sliding_attention": mask + torch.where(old, float("-inf"), 0.0)}
        h = self.text(input_ids=tokens, inputs_embeds=embeds, attention_mask=masks, position_ids=positions[None],
                      past_key_values=cache, use_cache=True).last_hidden_state
        logits = self.model.lm_head(h[:, -keep:] if keep else h)
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits, cache.out


_ADAPTERS = {"smolvlm": _SmolVLM, "idefics3": _SmolVLM, "qwen3_vl": _Qwen3VL, "gemma4": _Gemma4,
             "gemma4_text": _Gemma4}


class _Build:
    """What causal_lm assembles: stages, drivers, state, metadata, and the shared examples."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def gather_ex(self, n_rows: int) -> torch.Tensor:
        g = torch.arange(n_rows, n_rows + self.s)
        g[1:3] = torch.tensor([0, 1])
        return g


def _geometry(config: Any) -> tuple[int, int, int]:
    layers = config.num_hidden_layers
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return layers, kv_heads, head_dim


def causal_mask(n_past: int, s: int, n_kv: int) -> np.ndarray:
    """[1, 1, s, n_kv]: token i (at position n_past + i) attends to positions 0 .. n_past + i."""
    rows = n_past + np.arange(s)[:, None]
    return np.where(np.arange(n_kv)[None, :] <= rows, 0.0, -np.inf).astype(np.float32)[None, None]


def causal_lm(model: torch.nn.Module, max_context: int = 4096, max_prompt: int = 512,
              kv_dtype: torch.dtype = torch.float32, logits: str = "last", generate: bool = True,
              max_image_tokens: int | None = None) -> api.Program:
    """A Hugging Face causal LM (eager attention, fp32) as a ggml-compiler Program with prefill and decode
    entry points over a KV cache, and (generate=True) a `generate` entry that decodes greedily on
    the device. For a vision-language model (SmolVLM, Qwen3-VL), also `encode` and `prefill_mm`
    for up to max_image_tokens image tokens in the context (default: one image at the
    processor's defaults for SmolVLM, 1024 for Qwen3-VL; 0 compiles the language model alone).
    See the module docstring for the signatures."""
    if max_context % KV_BLOCK or max_context < 3 * KV_BLOCK:
        # n_kv = 256 * kv_blocks with kv_blocks >= 2 (torch.export makes a dynamic dim's min 2),
        # and a Dim needs max > min.
        raise ValueError(f"max_context must be a multiple of {KV_BLOCK}, at least {3 * KV_BLOCK}")
    if logits not in ("last", "all"):
        raise ValueError("logits must be 'last' or 'all'")
    model.eval()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")  # the attention ggml-compiler matches and fuses, in every sub-model
    elif getattr(model.config, "_attn_implementation", "eager") != "eager":
        model.config._attn_implementation = "eager"
    config = model.config.get_text_config()
    lm = _ADAPTERS.get(getattr(model.config, "model_type", None), _LM)(model, 1 if logits == "last" else 0,
                                                                       max_image_tokens)
    layout = lm.kv_layout(config)
    lm.slots = {layer: i for i, (layer, _, _) in enumerate(layout)}
    kv = [torch.zeros(h, max_context, d, dtype=kv_dtype) for _, h, d in layout for _ in range(2)]

    s, n_kv = 5, 2 * KV_BLOCK
    vocab = config.vocab_size
    S = Dim("s", min=2, max=max_prompt)
    N = KV_BLOCK * Dim("kv_blocks", min=2, max=max_context // KV_BLOCK)
    rp = lm.rope  # the text entries' RoPE positions, after their cache positions
    prefill_ex = (torch.randint(0, vocab, (1, s)), torch.arange(s)) + ((torch.arange(s),) if rp else ()) \
        + (torch.from_numpy(causal_mask(0, s, n_kv)), kv)
    decode_ex = (torch.randint(0, vocab, (1, 1)), torch.tensor([s])) + ((torch.tensor([s]),) if rp else ()) \
        + (torch.from_numpy(causal_mask(s, 1, n_kv)), kv)
    prefill_dyn = ({1: S}, {0: S}) + (({0: S},) if rp else ()) + ({2: S, 3: N}, None)
    decode_dyn = (None, None) + ((None,) if rp else ()) + ({3: N}, None)
    eos = getattr(config, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    gen = getattr(model, "generation_config", None)
    if gen is not None and gen.eos_token_id is not None:
        g = gen.eos_token_id
        eos = sorted(set(eos) | set([g] if isinstance(g, int) else g))
    stages = {
        "prefill": api.Stage(lm.step, prefill_ex, dynamic_shapes=prefill_dyn),
        "decode": api.Stage(lm.step, decode_ex, dynamic_shapes=decode_dyn),
    }
    pre_in = ["tokens", "positions"] + (["rpos"] if rp else []) + ["mask"]
    dec_in = ["token", "position"] + (["rpos1"] if rp else []) + ["mask1"]
    drivers = {
        "prefill": api.Driver(inputs=pre_in, body=[api.call("prefill", pre_in + ["kv"], ["logits", "kv"])],
                              outputs=["logits"]),
        "decode": api.Driver(inputs=dec_in, body=[api.call("decode", dec_in + ["kv"], ["logits1", "kv"])],
                             outputs=["logits1"]),
    }
    symbols = {}
    if generate:
        import sympy

        # generate(first [1, 1], start [1] [, rstart [1]]) -> generated [max_new, 1, 1]: feed `first`
        # at position `start`, then each argmax back, until an EOS or max_new tokens (a symbol set
        # per call).
        blocks = max_context // KV_BLOCK
        stages["decode_greedy"] = api.Stage(lm.greedy, decode_ex, dynamic_shapes=decode_dyn)
        counters = {"pos": "start", **({"rp": "rstart"} if rp else {})}
        drivers["generate"] = api.Driver(
            inputs=["first", "start"] + (["rstart"] if rp else []),
            body=[api.loop("max_new", body=[api.call("decode_greedy", ["tok", *counters, "gmask", "kv"], ["tok", "kv"])],
                           carry={"tok": "first"}, counters=counters,
                           bind={"kv_blocks": lambda pos, rp=None: sympy.Max(2, sympy.Min(blocks, sympy.floor(pos / KV_BLOCK) + 1))},
                           masks={"gmask": "pos"}, until={"tok": eos} if eos else None, collect={"generated": "tok"})],
            outputs=["generated"],
            examples={"start": torch.tensor([s + 1], dtype=torch.int32),
                      **({"rstart": torch.tensor([s + 1], dtype=torch.int32)} if rp else {})})
        symbols["max_new"] = (1, max_context, 3)
    metadata = {"kind": "causal_lm", "max_context": max_context, "max_prompt": max_prompt, "kv_block": KV_BLOCK,
                "min_kv": 2 * KV_BLOCK, "vocab": vocab, "eos": eos, "logits": logits,
                "bos": getattr(config, "bos_token_id", None), "rope": "separate" if rp else "positions"}
    b = _Build(stages=stages, drivers=drivers, state={"kv": kv}, metadata=metadata, S=S, N=N, s=s,
               prefill_ex=prefill_ex, hidden=config.hidden_size)
    if max_image_tokens != 0:
        lm.images(b)
    return api.Program(root=model, stages=stages, driver=drivers, symbols=symbols, state=b.state, metadata=metadata)


def qwen3_vl_tables(grid_thw: Sequence[int], pos_side: int, merge: int) -> dict[str, np.ndarray]:
    """Qwen3-VL's grid-dependent encode inputs for one image of grid (t, h, w) patches, computed as
    transformers' Qwen3VLVisionModel does (fast_pos_embed_interpolate, rot_pos_emb), in the
    processor's patch order: pos_idx / pos_w [4, n] (bilinear taps into the learned position
    embedding) and vpos [n, 2] (each patch's row and column)."""
    t, h, w = (int(x) for x in grid_thw)
    hi, wi = torch.linspace(0, pos_side - 1, h), torch.linspace(0, pos_side - 1, w)
    hf, wf = hi.int(), wi.int()
    hc, wc = (hf + 1).clip(max=pos_side - 1), (wf + 1).clip(max=pos_side - 1)
    dh, dw = hi - hf, wi - wf
    bf, bc = hf * pos_side, hc * pos_side
    idx = torch.stack([(bf[:, None] + wf[None]).flatten(), (bf[:, None] + wc[None]).flatten(),
                       (bc[:, None] + wf[None]).flatten(), (bc[:, None] + wc[None]).flatten()])
    wts = torch.stack([((1 - dh)[:, None] * (1 - dw)[None]).flatten(), ((1 - dh)[:, None] * dw[None]).flatten(),
                       (dh[:, None] * (1 - dw)[None]).flatten(), (dh[:, None] * dw[None]).flatten()])
    # raster order -> the processor's 2 x 2 merge blocks
    perm = torch.arange(h * w).view(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
    rows = (torch.arange(h // merge)[:, None, None, None] * merge + torch.arange(merge)[None, None, :, None])
    cols = (torch.arange(w // merge)[None, :, None, None] * merge + torch.arange(merge)[None, None, None, :])
    shape = (h // merge, w // merge, merge, merge)
    vpos = torch.stack([rows.expand(shape).reshape(-1), cols.expand(shape).reshape(-1)], dim=-1)
    return {"pos_idx": idx[:, perm].repeat(1, t).numpy().astype(np.int32),
            "pos_w": wts[:, perm].repeat(1, t).numpy().astype(np.float32),
            "vpos": vpos.repeat(t, 1).numpy().astype(np.int32)}


class Generator:
    """Drives a compiled causal_lm on the host: positions, masks, prompt chunks, greedy decoding."""

    def __init__(self, session: Any) -> None:
        meta = session.metadata
        if meta.get("kind") != "causal_lm":
            raise ValueError("the artifact was not compiled from ggml_compiler.causal_lm")
        self.s, self.meta = session, meta
        self.reset(state=False)

    def reset(self, state: bool = True) -> None:
        if state:
            self.s.reset()
        self.n_past = 0
        self.r_next = 0  # the next text token's RoPE position (Qwen3-VL: not n_past after an image)
        self._rows = self._next_row = 0  # image rows encoded, and the next one an image token takes
        self._grid: tuple[int, int] | None = None  # Qwen3-VL: the encoded image's merged grid
        self._img_start = 0

    def encode(self, pixels: Any, grid_thw: Sequence[int] | None = None) -> int:
        """Encode an image; the image tokens fed next take its rows in order. Returns the number of
        rows (image tokens) encoded.

        SmolVLM: `pixels` [tiles, 3, H, W] as the processor makes them (or [1, tiles, ...]);
        several images' tiles concatenated. Qwen3-VL: one image's `pixels` [n, C*T*P*P] as the
        processor flattens them, with its `grid_thw` (the processor's image_grid_thw row)."""
        img = self.meta.get("image")
        if not img:
            raise ValueError("the artifact was compiled without images")
        px = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32))
        if img["kind"] == "qwen3_vl":
            if grid_thw is None:
                raise ValueError("Qwen3-VL: pass the image's grid_thw")
            g = [int(x) for x in np.asarray(grid_thw).reshape(-1)]
            m = img["merge"]
            if len(g) != 3 or g[0] != 1 or px.shape != (g[0] * g[1] * g[2], img["patch"]) or g[1] % m or g[2] % m:
                raise ValueError(f"pixels {px.shape} with grid {g}: want one image, [t*h*w, {img['patch']}], t = 1")
            rows = px.shape[0] // (m * m)
            if rows > img["rows"]:
                raise ValueError(f"{rows} image tokens: compiled for at most {img['rows']}")
            self.s.run(pixels=px, **qwen3_vl_tables(g, img["pos_side"], m), entry="encode")
            self._grid = (g[1] // m, g[2] // m)
        else:
            while px.ndim > 4 and px.shape[0] == 1:
                px = px[0]
            tiles = px.shape[0]
            if px.ndim != 4 or list(px.shape[1:]) != img["tile"] or not 1 <= tiles <= img["max_tiles"]:
                raise ValueError(f"pixels {px.shape}: want [1..{img['max_tiles']}, {', '.join(map(str, img['tile']))}]")
            # torch.export's dynamic dims start at 2: one tile is encoded twice and the copy ignored.
            self.s.run(pixels=np.concatenate([px, px]) if tiles == 1 else px, entry="encode")
            rows = tiles * img["rows_per_tile"]
        self._rows, self._next_row = rows, 0
        return rows

    def _n_kv(self, end: int) -> int:
        block = self.meta["kv_block"]
        return max(self.meta["min_kv"], -(-end // block) * block)

    def _rope(self, is_img: np.ndarray) -> np.ndarray:
        """[3, n] RoPE positions (Qwen3-VL's t, h, w) of the next n tokens; advances r_next."""
        out = np.empty((3, len(is_img)), np.int32)
        k = self._next_row
        for i, im in enumerate(is_img):
            if im:
                if k == 0:
                    self._img_start = self.r_next
                gh, gw = self._grid
                st = self._img_start
                out[:, i] = (st, st + k // gw, st + k % gw)
                k += 1
                if k == self._rows:  # text after the image continues from start + max(h, w)
                    self.r_next = st + max(gh, gw)
            else:
                out[:, i] = self.r_next
                self.r_next += 1
        return out

    def _run(self, ids: Sequence[int]) -> np.ndarray:
        n = len(ids)
        if self.n_past + n > self.meta["max_context"]:
            raise ValueError(f"context full: {self.n_past} + {n} tokens > max_context {self.meta['max_context']}")
        n_kv = self._n_kv(self.n_past + n)
        pos = np.arange(self.n_past, self.n_past + n, dtype=np.int32)
        mask = causal_mask(self.n_past, n, n_kv)
        tok = np.asarray(ids, dtype=np.int32)[None]
        img = self.meta.get("image")
        is_img = tok[0] == img["token"] if img else np.zeros(n, bool)
        sep = self.meta.get("rope") == "separate"
        if is_img.any():
            k = int(is_img.sum())
            if self._next_row + k > self._rows:
                raise ValueError(f"{self._next_row + k} image tokens for {self._rows} encoded rows: call encode() first")
            if n == 1:
                raise ValueError("a lone image token: feed image tokens with their neighbours")
            gather = img["rows"] + np.arange(n, dtype=np.int32)
            gather[is_img] = np.arange(self._next_row, self._next_row + k, dtype=np.int32)
            rope = {"mrope": self._rope(is_img)} if sep else {}
            self._next_row += k
            out = self.s.run(tokens=tok, gather=gather, positions=pos, mask=mask, **rope, entry="prefill_mm")["logits"]
        else:
            rpos = self._rope(is_img)[0] if sep else None
            if n == 1:
                rope = {"rpos1": rpos} if sep else {}
                out = self.s.run(token=tok, position=pos, mask1=mask, **rope, entry="decode")["logits1"]
            else:
                rope = {"rpos": rpos} if sep else {}
                out = self.s.run(tokens=tok, positions=pos, mask=mask, **rope, entry="prefill")["logits"]
        self.n_past += n
        if not sep:
            self.r_next = self.n_past
        return out[0]

    def feed(self, ids: Sequence[int]) -> np.ndarray:
        """Append tokens to the context; returns the logits after the last one ([vocab]), or with
        logits="all" the logits of every token fed ([n, vocab])."""
        ids = list(ids)
        step = self.meta["max_prompt"]
        rows = []
        i = 0
        while i < len(ids):
            n = min(step, len(ids) - i)
            if n == 1 or len(ids) - i - n == 1:  # never leave a lone token for a prefill of one
                n = 1 if n == 1 else n - 1
            rows.append(self._run(ids[i:i + n]))
            i += n
        out = np.concatenate(rows, axis=0)
        return out if self.meta["logits"] == "all" else out[-1]

    def generate(self, prompt: Sequence[int], max_new: int, stop: Sequence[int] | None = None,
                 device: bool = False, pixels: Any = None, grid_thw: Sequence[int] | None = None) -> list[int]:
        """Greedy continuation of `prompt` from an empty context, stopping after max_new tokens or
        at an end-of-sequence id (default: the model's). device=True runs the decoding loop in the
        runtime (the `generate` entry; it stops at the model's EOS ids only). `pixels` (and
        Qwen3-VL's `grid_thw`): the prompt's image (see encode)."""
        self.reset()
        if pixels is not None:
            self.encode(pixels, grid_thw)
        stop = set(self.meta["eos"] if stop is None else stop)
        logits = self.feed(prompt)
        if device:
            if "generate" not in self.s.entries:
                raise ValueError("compiled without the generate entry")
            first = int(np.argmax(logits if logits.ndim == 1 else logits[-1]))
            if first in stop or max_new == 1:
                return [first]
            room = self.meta["max_context"] - self.n_past  # positions left in the cache
            self.s.set_symbol("max_new", max(1, min(max_new - 1, room)))
            rope = {"rstart": np.array([self.r_next], np.int32)} if self.meta.get("rope") == "separate" else {}
            out = self.s.run(first=np.array([[first]], np.int32), start=np.array([self.n_past], np.int32), **rope,
                             entry="generate")["generated"].reshape(-1)
            n = self.s.stats["stages"]["decode_greedy"]["calls"]
            self.n_past += n
            self.r_next += n
            return [first] + out[:n].tolist()
        out: list[int] = []
        for _ in range(max_new):
            tok = int(np.argmax(logits if logits.ndim == 1 else logits[-1]))
            out.append(tok)
            if tok in stop or len(out) == max_new:
                break
            logits = self.feed([tok])
        return out


def bench(session: Any, pp: int = 512, tg: int = 128, reps: int = 5, seed: int = 0,
          device: bool = False) -> dict[str, float]:
    """Tokens per second as llama-bench measures them: pp = one prefill of `pp` tokens from an empty
    context, tg = `tg` single-token decodes from an empty context; mean over `reps` runs after a
    warmup of each. tg includes the host loop (positions, masks, Python), or with device=True
    runs the `generate` entry (an early EOS ends a run sooner; its rate counts the tokens made)."""
    import time

    g = Generator(session)
    rng = np.random.default_rng(seed)
    vocab = g.meta["vocab"]
    out: dict[str, float] = {}
    if pp:
        times = []
        for r in range(reps + 1):
            ids = rng.integers(0, vocab, pp).tolist()
            g.reset()
            t = time.perf_counter()
            g.feed(ids)
            if r:
                times.append(time.perf_counter() - t)
        out[f"pp{pp}"] = pp / float(np.mean(times))
    if tg and device:
        rates = []
        session.set_symbol("max_new", tg)
        for r in range(reps + 1):
            g.reset()
            first = np.array([[int(rng.integers(0, vocab))]], np.int32)
            t = time.perf_counter()
            session.run(first=first, start=np.array([0], np.int32), entry="generate")
            dt = time.perf_counter() - t
            if r:
                rates.append(session.stats["stages"]["decode_greedy"]["calls"] / dt)
        out[f"tg{tg}"] = float(np.mean(rates))
    elif tg:
        times = []
        for r in range(reps + 1):
            g.reset()
            tok = int(rng.integers(0, vocab))
            t = time.perf_counter()
            for _ in range(tg):
                logits = g.feed([tok])
                tok = int(np.argmax(logits))
            if r:
                times.append(time.perf_counter() - t)
        out[f"tg{tg}"] = tg / float(np.mean(times))
    return out


def _llama_cpp_names(fqn: str, config: Any) -> tuple[list[str], list[str]]:
    """(HF-style names for gguf-py's name map, GGUF tensor names) llama.cpp's converter may have
    written `fqn` as. It drops a VLM's "text_model." (SmolVLM) and "model.language_model." /
    "model.visual." prefixes (Qwen3-VL), and renames Qwen3-VL's patch embedding, merger and
    deepstack mergers by hand (conversion/qwen3vl.py)."""
    import re

    hf = [fqn, fqn.replace("text_model.", ""), fqn.replace("model.language_model.", "model."),
          fqn.replace("model.visual.", "visual.")]
    direct = []
    v = fqn.replace("model.visual.", "visual.")
    if v.startswith("visual.patch_embed.proj."):
        direct.append("v.patch_embd." + v.rsplit(".", 1)[1])
    m = re.fullmatch(r"visual\.merger\.(norm|linear_fc1|linear_fc2)\.(weight|bias)", v)
    if m:
        direct.append({"norm": "v.post_ln", "linear_fc1": "mm.0", "linear_fc2": "mm.2"}[m[1]] + "." + m[2])
    m = re.fullmatch(r"visual\.deepstack_merger_list\.(\d+)\.(norm|linear_fc1|linear_fc2)\.(weight|bias)", v)
    if m:
        layer = config.vision_config.deepstack_visual_indexes[int(m[1])]
        direct.append(f"v.deepstack.{layer}.{ {'norm': 'norm', 'linear_fc1': 'fc1', 'linear_fc2': 'fc2'}[m[2]]}.{m[3]}")
    return list(dict.fromkeys(hf)), direct


def llama_cpp_policy(gguf_path: str, model: torch.nn.Module, mmproj: str | None = None) -> Any:
    """A DTypePolicy giving each weight the type llama.cpp stored it as in `gguf_path` (a GGUF made
    by llama.cpp's converter and llama-quantize), so both run the same weight bytes. Names are
    matched with gguf-py's own HF -> llama.cpp tensor name map. For a vision-language model,
    `mmproj` is the vision projector GGUF (convert_hf_to_gguf.py --mmproj) giving the vision
    weights' types; without it only the language model's weights get rules."""
    import gguf

    from .quant import DTypePolicy

    def tables(path: str, blocks: int) -> tuple[Any, dict[str, str]]:
        reader = gguf.GGUFReader(path)
        arch_name = bytes(reader.fields["general.architecture"].parts[-1]).decode()
        arch = gguf.MODEL_ARCH.MMPROJ if arch_name == "clip" else \
            next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch_name)
        return gguf.get_tensor_name_map(arch, blocks), {t.name: t.tensor_type.name.lower() for t in reader.tensors}

    cfg = model.config
    sources = [tables(gguf_path, cfg.get_text_config().num_hidden_layers)]
    params = list(model.named_parameters(remove_duplicate=False))
    if mmproj is not None:
        vc = cfg.vision_config
        sources.append(tables(mmproj, getattr(vc, "num_hidden_layers", None) or vc.depth))
    else:  # the language model alone: a multimodal model's vision and audio towers are not in gguf_path
        keep = {id(p) for m in (model.get_decoder(), model.get_output_embeddings()) if m is not None
                for p in m.parameters()}
        params = [(n, p) for n, p in params if id(p) in keep]
    rules, missing = [], []
    for fqn, _ in params:
        hf, direct = _llama_cpp_names(fqn, cfg)
        found = None
        for tmap, stored in sources:
            names = direct + [tmap.get_name(c, try_suffixes=(".weight", ".bias")) for c in hf]
            for name in filter(None, names):
                dtype = stored.get(name) or (stored.get("token_embd.weight") if name == "output.weight" else None)
                if dtype is not None:
                    found = name, dtype
                    break
            if found:
                break
        if found is None:
            missing.append(fqn)
            continue
        name, dtype = found
        # The embedding table and the output head may be one tied tensor here and two in the GGUF
        # (Qwen3's converter writes both): give each reader its own type.
        op = {"token_embd.weight": "@get_rows", "output.weight": "@mul_mat"}.get(name, "")
        rules.append((fqn + op, dtype))
    if missing:
        where = gguf_path + (f" / {mmproj}" if mmproj else "")
        raise ValueError(f"no llama.cpp tensor for {missing[:5]} ({len(missing)} in all) in {where}")
    return DTypePolicy(rules=rules, float="f32", fallback=[])
