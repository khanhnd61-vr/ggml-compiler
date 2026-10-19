"""SmolVLM2 as ggml C++ with image encode, prefill, decode, and generation entry points.

Compile with:
    ggml-compiler compile examples/vlm/export.py:program --out out/smolvlm \
        --emit gguf,cpp --name smolvlm -O O3

The Hugging Face processor stays with the caller: it applies the chat template, resizes and tiles
the image, and produces the image placeholder tokens consumed by the compiled model.
"""

import os

import torch

import ggml_compiler

CHECKPOINT = os.environ.get("VLM_CHECKPOINT", "HuggingFaceTB/SmolVLM2-500M-Instruct")
MAX_CONTEXT = int(os.environ.get("VLM_MAX_CONTEXT", "2048"))
MAX_PROMPT = int(os.environ.get("VLM_MAX_PROMPT", "512"))


def load(checkpoint: str = CHECKPOINT):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


def program(
    checkpoint: str = CHECKPOINT,
    max_context: int = MAX_CONTEXT,
    max_prompt: int = MAX_PROMPT,
    max_image_tokens: int | None = None,
) -> ggml_compiler.Program:
    return ggml_compiler.causal_lm(
        load(checkpoint),
        max_context=max_context,
        max_prompt=max_prompt,
        max_image_tokens=max_image_tokens,
    )
