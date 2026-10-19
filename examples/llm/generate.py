"""Generate text with a model compiled by examples/llm/export.py."""

import argparse
import os

from transformers import AutoTokenizer

import ggml_compiler

CHECKPOINT = os.environ.get("LLM_CHECKPOINT", "Qwen/Qwen3-0.6B")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="compiled model.gguf")
    p.add_argument("prompt")
    p.add_argument("--checkpoint", default=CHECKPOINT, help="tokenizer checkpoint")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--device-loop", action="store_true", help="run greedy decoding inside the runtime")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt_ids = tokenizer(args.prompt)["input_ids"]
    with ggml_compiler.Session(args.model, backend=args.backend) as session:
        output_ids = ggml_compiler.Generator(session).generate(
            prompt_ids,
            max_new=args.max_new,
            device=args.device_loop,
        )
    print(tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
