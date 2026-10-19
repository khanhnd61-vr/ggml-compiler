"""Describe an image with a model compiled by examples/vlm/export.py."""

import argparse
import os

from PIL import Image
from transformers import AutoProcessor

import ggml_compiler

CHECKPOINT = os.environ.get("VLM_CHECKPOINT", "HuggingFaceTB/SmolVLM2-500M-Instruct")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="compiled model.gguf")
    p.add_argument("image")
    p.add_argument("--prompt", default="Describe this image in detail.")
    p.add_argument("--checkpoint", default=CHECKPOINT, help="processor checkpoint")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--device-loop", action="store_true", help="run greedy decoding inside the runtime")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=[Image.open(args.image).convert("RGB")],
        return_tensors="pt",
    )
    prompt_ids = inputs["input_ids"][0].tolist()

    with ggml_compiler.Session(args.model, backend=args.backend) as session:
        output_ids = ggml_compiler.Generator(session).generate(
            prompt_ids,
            max_new=args.max_new,
            pixels=inputs["pixel_values"],
            device=args.device_loop,
        )
    print(processor.decode(output_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
