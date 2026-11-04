# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert the old preformatted LATR parquet prompts to V1 message lists."""

import argparse
import re
from pathlib import Path

# Preserve old base-model prompt text byte-for-byte instead of applying a new chat template.
PASSTHROUGH_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def convert_prompt(value, prompt_mode="preformatted"):
    if isinstance(value, str):
        if prompt_mode == "chat":
            # The old base/inst-qwen dataset contains an already rendered conversation.
            match = re.search(r"(?:\nUser: |<\|im_start\|>user\n)(.*?)(?:\nAssistant:|<\|im_end\|>)", value, re.DOTALL)
            if match is None:
                raise ValueError("Cannot extract a user question from this old prompt; use preformatted mode")
            value = match.group(1).strip()
        return [{"role": "user", "content": value}]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError("Expected a preformatted string or a list of chat messages")


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prompt-mode", choices=("preformatted", "chat"), default="preformatted")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("Write converted data to a separate directory")
    args.destination.mkdir(parents=True, exist_ok=True)
    files = sorted(args.source.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files in {args.source}")
    for source in files:
        target = args.destination / source.name
        if target.exists():
            raise FileExistsError(target)
        dataset = load_dataset("parquet", data_files=str(source), split="train")
        dataset = dataset.map(lambda row: {"prompt": convert_prompt(row["prompt"], args.prompt_mode)})
        dataset.to_parquet(str(target))
        print(f"Converted {source} -> {target}: {len(dataset)} rows")


if __name__ == "__main__":
    main()
