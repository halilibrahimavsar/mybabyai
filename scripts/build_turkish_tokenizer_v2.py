#!/usr/bin/env python3
"""
Build a Turkish-optimized tokenizer package for CodeMind.

Base: ytu-ce-cosmos/Turkish-GPT2-large (ByteLevel BPE)
Adds:
- Prompt/control tokens used by src/core/prompting
- Language tags (tr/python/dart/javascript)
- Minimal code-structure tokens used across the repo

This script is intentionally deterministic and safe to re-run.
"""

from __future__ import annotations

from pathlib import Path


BASE_TOKENIZER = "ytu-ce-cosmos/Turkish-GPT2-large"

# Keep this list small and practical: repeated prompt markers should be single tokens.
ADDITIONAL_SPECIAL_TOKENS = [
    "<|tr|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|context|>",
    "<|/context|>",
    "<|answer|>",
    "<|code|>",
    "<|comment|>",
    "<|docstring|>",
    "<|function|>",
    "<|class|>",
    "<|error|>",
    "<|fix|>",
    "<|explain|>",
    "<|question|>",
    # language tags
    "<|python|>",
    "<|dart|>",
    "<|javascript|>",
    # misc compatibility token used in older local tokenizers
    "▁",
]


def main() -> None:
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="codemind/checkpoints/tokenizer_tr_v2",
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="model_max_length to write into tokenizer_config.json",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=BASE_TOKENIZER,
        help="Base tokenizer on HF Hub (or local path)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    # Core tokens (these should be single tokens in the final vocab)
    core = ["<|pad|>", "<|eos|>", "<|unk|>"]
    tok.add_tokens(core, special_tokens=True)
    tok.add_tokens(ADDITIONAL_SPECIAL_TOKENS, special_tokens=True)

    # Make sure the tokenizer config points to our canonical choices
    if "<|pad|>" in tok.get_vocab():
        tok.pad_token = "<|pad|>"
    if "<|unk|>" in tok.get_vocab():
        tok.unk_token = "<|unk|>"
    if "<|eos|>" in tok.get_vocab():
        tok.eos_token = "<|eos|>"
    if "<|endoftext|>" in tok.get_vocab():
        tok.bos_token = "<|endoftext|>"

    tok.model_max_length = int(args.max_length)

    tok.save_pretrained(str(out_dir))

    print(f"Saved tokenizer to: {out_dir}")
    print(f"Base: {args.base}")
    print(f"Vocab size: {len(tok)}")
    print(
        "Key IDs:",
        {
            "bos": tok.bos_token_id,
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
            "unk": tok.unk_token_id,
        },
    )


if __name__ == "__main__":
    main()

