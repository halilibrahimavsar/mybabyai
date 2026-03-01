"""
Canonical prompt tokens and builders for CodeMind.

This module is the single source of truth for prompt templates used by
training, inference, and data preparation code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PromptTokens:
    system: str = "<|system|>"
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    context: str = "<|context|>"
    context_end: str = "<|/context|>"
    eos: str = "<|eos|>"
    answer: str = "<|answer|>"


TOKENS = PromptTokens()

LEGACY_PROMPT_MARKERS = (
    "<|assistant|:",
    "<|assistant|/>",
)

CORRUPTED_PROMPT_LITERALS = (
    "erusform",
    "ƭ",
)


def normalize_prompt_text(text: str) -> str:
    """Normalize known legacy markers to canonical prompt tokens."""
    normalized = text.replace("<|assistant|/>", TOKENS.assistant)
    normalized = normalized.replace("<|assistant|:", TOKENS.assistant)
    normalized = re.sub(r"<\|assistant\|(?!>)", TOKENS.assistant, normalized)
    return normalized


def validate_prompt_template(text: str) -> None:
    """
    Raise when prompt text contains known invalid/corrupted template markers.
    """
    for marker in LEGACY_PROMPT_MARKERS:
        if marker in text:
            raise ValueError(
                f"Non-canonical prompt marker detected: {marker!r}. "
                f"Use {TOKENS.assistant!r}."
            )
    if re.search(r"<\|assistant\|(?!>)", text):
        raise ValueError(
            f"Non-canonical prompt marker detected: '<|assistant|' (missing '>'). "
            f"Use {TOKENS.assistant!r}."
        )
    for bad in CORRUPTED_PROMPT_LITERALS:
        if bad in text:
            raise ValueError(f"Corrupted prompt literal detected: {bad!r}.")


def build_chat_prompt(
    user_input: str,
    system_prompt: str,
    context: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    history_turns: int = 5,
    append_assistant_token: bool = True,
) -> str:
    parts = [f"{TOKENS.system}\n{system_prompt.strip()}"]

    if context:
        parts.append(f"\n\n{TOKENS.context}\n{context.strip()}\n{TOKENS.context_end}")

    if history:
        for turn in history[-history_turns:]:
            user_text = normalize_prompt_text(turn.get("user", "")).strip()
            assistant_text = normalize_prompt_text(turn.get("assistant", "")).strip()
            parts.append(f"\n{TOKENS.user}\n{user_text}")
            parts.append(f"\n{TOKENS.assistant}\n{assistant_text}")

    parts.append(f"\n{TOKENS.user}\n{normalize_prompt_text(user_input).strip()}")
    if append_assistant_token:
        parts.append(f"\n{TOKENS.assistant}\n")

    prompt = "".join(parts)
    validate_prompt_template(prompt)
    return prompt


def build_instruction_prompt(
    user: str,
    assistant: Optional[str] = None,
    language: str = "general",
    include_eos: bool = True,
) -> str:
    segments: List[str] = []

    language = (language or "general").strip().lower()
    if language and language != "general":
        segments.append(f"<|{language}|>")

    segments.append(f"{TOKENS.user}\n{normalize_prompt_text(user).strip()}")
    segments.append(f"\n{TOKENS.assistant}\n")

    if assistant is not None:
        segments.append(normalize_prompt_text(assistant).strip())

    if include_eos:
        segments.append(TOKENS.eos)

    prompt = "".join(segments)
    validate_prompt_template(prompt)
    return prompt


def build_codemind_code_prompt(prompt: str, language: str = "python") -> str:
    """Code-only prompt format used by base CodeMind adapter."""
    formatted = f"<|{language}|>{normalize_prompt_text(prompt).strip()}\n{TOKENS.answer}"
    validate_prompt_template(formatted)
    return formatted


def extract_assistant_response(text: str) -> str:
    normalized = normalize_prompt_text(text)
    if TOKENS.assistant in normalized:
        normalized = normalized.split(TOKENS.assistant)[-1]
    if TOKENS.answer in normalized:
        normalized = normalized.split(TOKENS.answer)[-1]
    if TOKENS.eos in normalized:
        normalized = normalized.split(TOKENS.eos)[0]
    return normalized.strip()
