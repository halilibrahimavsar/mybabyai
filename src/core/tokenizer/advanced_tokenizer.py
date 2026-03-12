"""
Advanced Code Tokenizer for CodeMind

Uses HuggingFace tokenizers library for proper BPE encoding
with code-specific optimizations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        trainers,
        processors,
        decoders,
    )
    from tokenizers.models import BPE, WordPiece, Unigram
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
    from tokenizers.pre_tokenizers import (
        ByteLevel,
        Whitespace,
        Split,
        Digits,
        Punctuation,
        Sequence,
    )

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

import torch


PYTHON_TOKENS = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "list",
    "dict",
    "set",
    "tuple",
    "open",
    "self",
    "cls",
    "__init__",
    "__main__",
    "__name__",
]

DART_TOKENS = [
    "abstract",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "covariant",
    "default",
    "deferred",
    "do",
    "dynamic",
    "else",
    "enum",
    "export",
    "extends",
    "extension",
    "external",
    "factory",
    "false",
    "final",
    "finally",
    "for",
    "Function",
    "get",
    "hide",
    "if",
    "implements",
    "import",
    "in",
    "interface",
    "is",
    "late",
    "library",
    "mixin",
    "new",
    "null",
    "on",
    "operator",
    "part",
    "required",
    "rethrow",
    "return",
    "set",
    "show",
    "static",
    "super",
    "switch",
    "sync",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "Widget",
    "StatelessWidget",
    "StatefulWidget",
    "State",
    "BuildContext",
    "Container",
    "Row",
    "Column",
    "Scaffold",
    "AppBar",
    "Text",
    "ListView",
    "GestureDetector",
    "onPressed",
    "child",
    "children",
    "padding",
    "margin",
]

JAVASCRIPT_TOKENS = [
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "else",
    "export",
    "extends",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "new",
    "return",
    "super",
    "switch",
    "this",
    "throw",
    "try",
    "typeof",
    "var",
    "void",
    "while",
    "with",
    "yield",
    "let",
    "static",
    "async",
    "await",
    "of",
    "null",
    "undefined",
    "true",
    "false",
    "console",
    "log",
    "const",
    "let",
    "async",
    "await",
    "Promise",
    "async",
    "fetch",
    "then",
    "catch",
]

CODE_SPECIAL_TOKENS = [
    "<|tr|>",
    "<|python|>",
    "<|dart|>",
    "<|javascript|>",
    "<|code|>",
    "<|comment|>",
    "<|docstring|>",
    "<|function|>",
    "<|class|>",
    "<|variable|>",
    "<|import|>",
    "<|error|>",
    "<|fix|>",
    "<|explain|>",
    "<|question|>",
    "<|answer|>",
    "<|context|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|pad|>",
    "<|eos|>",
    "<|unk|>",
    "<|mask|>",
]
LEGACY_ASSISTANT_TOKEN = "<|assistant" + "|"


class AdvancedCodeTokenizer:
    """
    Advanced BPE tokenizer optimized for code.

    Features:
    - Code-aware pre-tokenization
    - Language-specific special tokens
    - Efficient vocabulary
    - Fast encoding/decoding
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        min_frequency: int = 2,
        add_prefix_space: bool = True,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.add_prefix_space = add_prefix_space

        self._tokenizer: Optional[Tokenizer] = None
        self._vocab: Dict[str, int] = {}
        self._vocab_inverse: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {}

        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Initialize base vocabulary with code tokens."""
        self._base_tokens = (
            CODE_SPECIAL_TOKENS + PYTHON_TOKENS + DART_TOKENS + JAVASCRIPT_TOKENS
        )
        self._base_tokens = list(dict.fromkeys(self._base_tokens))

    def train(
        self,
        files: List[str],
        save_path: Optional[str] = None,
    ) -> "AdvancedCodeTokenizer":
        """
        Train tokenizer on code files.

        Args:
            files: List of file paths to train on
            save_path: Path to save tokenizer

        Returns:
            Self for chaining
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "tokenizers library not installed. Run: pip install tokenizers"
            )

        self._tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=CODE_SPECIAL_TOKENS,
            show_progress=True,
        )

        self._tokenizer.train(files, trainer)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="$A <|eos|>",
            special_tokens=[("<|eos|>", self._tokenizer.token_to_id("<|eos|>"))],
        )

        self._tokenizer.decoder = decoders.ByteLevel()

        self._build_vocab_cache()

        if save_path:
            self.save(save_path)

        return self

    def train_from_texts(
        self,
        texts: List[str],
        save_path: Optional[str] = None,
    ) -> "AdvancedCodeTokenizer":
        """Train tokenizer from list of texts."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "train.txt"
            with open(tmp_file, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n")

            self.train([str(tmp_file)], save_path)

        return self

    def _build_vocab_cache(self) -> None:
        """Build vocabulary cache for fast lookup."""
        if self._tokenizer is None:
            return

        self._vocab = self._tokenizer.get_vocab()
        self._vocab_inverse = {v: k for k, v in self._vocab.items()}
        self.special_tokens = {
            "pad": self._vocab.get("<|pad|>", 0),
            "eos": self._vocab.get("<|eos|>", 1),
            "unk": self._vocab.get("<|unk|>", 2),
            "assistant": self._vocab.get(
                "<|assistant|>", self._vocab.get(LEGACY_ASSISTANT_TOKEN, 0)
            ),
            "user": self._vocab.get("<|user|>", 0),
            "system": self._vocab.get("<|system|>", 0),
        }

    def __call__(
        self,
        text: Union[str, List[str]],
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compatibility method for Transformers-like usage."""
        if isinstance(text, list):
            input_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
            # Simple truncation for list of texts
            if truncation and max_length:
                input_ids = [ids[:max_length] for ids in input_ids]
        else:
            input_ids = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation and max_length:
                input_ids = input_ids[:max_length]

        result = {"input_ids": input_ids}
        if return_attention_mask:
            if isinstance(text, list):
                result["attention_mask"] = [[1] * len(ids) for ids in input_ids]
            else:
                result["attention_mask"] = [1] * len(input_ids)
        
        # return_tensors logic could be added here if needed, but for our datasets it's not required
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        language: Optional[str] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            language: Programming language hint

        Returns:
            List of token IDs
        """
        if self._tokenizer is None:
            return self._encode_fallback(text, add_special_tokens, language)

        if language:
            lang_token = f"<|{language}|>"
            if lang_token in self._vocab:
                text = f"{lang_token}{text}"

        encoding = self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        )

        return encoding.ids

    def _encode_fallback(
        self,
        text: str,
        add_special_tokens: bool,
        language: Optional[str],
    ) -> List[int]:
        """Fallback encoding when tokenizer not trained."""
        tokens = []

        if add_special_tokens and language:
            lang_token = f"<|{language}|>"
            if lang_token in self._base_tokens:
                tokens.append(self._base_tokens.index(lang_token))

        words = text.split()
        for word in words:
            if word in self._vocab:
                tokens.append(self._vocab[word])
            else:
                for char in word:
                    tokens.append(self._vocab.get(char, self._vocab.get("<|unk|>", 0)))

        if add_special_tokens:
            eos_idx = self._vocab.get("<|eos|>", len(self._base_tokens) - 2)
            tokens.append(eos_idx)

        return tokens

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self._tokenizer is None:
            return self._decode_fallback(ids, skip_special_tokens)

        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def _decode_fallback(
        self,
        ids: List[int],
        skip_special_tokens: bool,
    ) -> str:
        """Fallback decoding."""
        tokens = []
        special_ids = {i for i, t in enumerate(self._base_tokens) if t.startswith("<|")}

        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue

            token = self._vocab_inverse.get(id_, "")
            if token:
                tokens.append(token)

        text = "".join(tokens)
        text = text.replace("Ġ", " ").replace("▁", " ")

        return text.strip()

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 2048,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode texts.

        Args:
            texts: List of texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            Dict with input_ids and attention_mask
        """
        all_ids = []

        for text in texts:
            ids = self.encode(text, add_special_tokens=True)

            if truncation and len(ids) > max_length:
                ids = ids[:max_length]

            all_ids.append(ids)

        if padding:
            max_len = max(len(ids) for ids in all_ids)
            attention_mask = []

            padded_ids = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

            all_ids = padded_ids
        else:
            attention_mask = [[1] * len(ids) for ids in all_ids]

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def save(self, path: str) -> None:
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._tokenizer is not None:
            self._tokenizer.save(str(path / "tokenizer.json"))

        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "add_prefix_space": self.add_prefix_space,
            "base_tokens": self._base_tokens,
        }

        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AdvancedCodeTokenizer":
        """Load tokenizer from directory."""
        path = Path(path)

        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            add_prefix_space=config.get("add_prefix_space", True),
        )

        instance._base_tokens = config.get("base_tokens", CODE_SPECIAL_TOKENS)

        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            instance._vocab = json.load(f)

        instance._vocab_inverse = {v: k for k, v in instance._vocab.items()}
        instance.special_tokens = {
            "pad": instance._vocab.get("<|pad|>", 0),
            "eos": instance._vocab.get("<|eos|>", 1),
            "unk": instance._vocab.get("<|unk|>", 2),
            "assistant": instance._vocab.get(
                "<|assistant|>", instance._vocab.get(LEGACY_ASSISTANT_TOKEN, 0)
            ),
            "user": instance._vocab.get("<|user|>", 0),
            "system": instance._vocab.get("<|system|>", 0),
        }

        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists() and TOKENIZERS_AVAILABLE:
            instance._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return instance

    @property
    def vocab_size_actual(self) -> int:
        """Actual vocabulary size."""
        return len(self._vocab) if self._vocab else self.vocab_size

    @property
    def pad_token_id(self) -> int:
        """PAD token ID."""
        return self._vocab.get("<|pad|>", 0)

    @property
    def eos_token_id(self) -> int:
        """EOS token ID."""
        return self._vocab.get("<|eos|>", 1)

    @property
    def unk_token_id(self) -> int:
        """UNK token ID."""
        return self._vocab.get("<|unk|>", 2)

    @property
    def mask_token_id(self) -> int:
        """MASK token ID."""
        return self._vocab.get("<|mask|>", 3)

    def __len__(self) -> int:
        return self.vocab_size_actual

    def __repr__(self) -> str:
        return f"AdvancedCodeTokenizer(vocab_size={self.vocab_size_actual})"


def create_tokenizer(
    vocab_size: int = 16384,
    training_files: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> AdvancedCodeTokenizer:
    """
    Create and optionally train a tokenizer.

    Args:
        vocab_size: Target vocabulary size
        training_files: Files to train on
        save_path: Path to save tokenizer

    Returns:
        Trained tokenizer
    """
    tokenizer = AdvancedCodeTokenizer(vocab_size=vocab_size)

    if training_files:
        tokenizer.train(training_files, save_path)

    return tokenizer
