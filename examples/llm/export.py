"""Qwen3 as ggml C++ with prefill, decode, and on-device greedy generation entry points.

Compile with:
    ggml-compiler compile examples/llm/export.py:program --out out/qwen3 \
        --emit gguf,cpp --name qwen3 -O O3

Tokenization and chat templates stay with the caller. The compiled KV cache supports prompts up
to LLM_MAX_PROMPT tokens per prefill call and LLM_MAX_CONTEXT tokens in all.
"""

import os

import torch

import ggml_compiler

CHECKPOINT = os.environ.get("LLM_CHECKPOINT", "Qwen/Qwen3-0.6B")
MAX_CONTEXT = int(os.environ.get("LLM_MAX_CONTEXT", "4096"))
MAX_PROMPT = int(os.environ.get("LLM_MAX_PROMPT", "512"))


def load(checkpoint: str = CHECKPOINT):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


def program(
    checkpoint: str = CHECKPOINT,
    max_context: int = MAX_CONTEXT,
    max_prompt: int = MAX_PROMPT,
) -> ggml_compiler.Program:
    return ggml_compiler.causal_lm(
        load(checkpoint),
        max_context=max_context,
        max_prompt=max_prompt,
    )
