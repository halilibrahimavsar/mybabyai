"""
Code-Aware BPE Tokenizer for CodeMind

Special tokenizer that understands:
- Python syntax
- Dart/Flutter syntax
- JavaScript syntax
- Code structure (indentation, brackets, etc.)
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
import pickle

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
    from tokenizers.implementations import BaseTokenizer
    from tokenizers.models import BPE

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


PYTHON_KEYWORDS = [
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
]

PYTHON_BUILTINS = [
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "compile",
    "complex",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "eval",
    "exec",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "hex",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "memoryview",
    "min",
    "next",
    "object",
    "oct",
    "open",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
]

DART_KEYWORDS = [
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
]

FLUTTER_WIDGETS = [
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
    "Image",
    "ListView",
    "GridView",
    "Stack",
    "Positioned",
    "Expanded",
    "Flexible",
    "Padding",
    "Margin",
    "EdgeInsets",
    "Decoration",
    "BoxDecoration",
    "MaterialApp",
    "CupertinoApp",
    "Navigator",
    "Route",
    "Theme",
    "Color",
    "TextStyle",
    "FontWeight",
    "Alignment",
    "BorderRadius",
    "BoxShadow",
]

JAVASCRIPT_KEYWORDS = [
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
]

CODE_PATTERNS = {
    "python": {
        "function_def": r"def\s+\w+\s*\([^)]*\)\s*:",
        "class_def": r"class\s+\w+(\([^)]*\))?\s*:",
        "import": r"(from\s+\S+\s+)?import\s+.*",
        "decorator": r"@\w+(\([^)]*\))?",
        "docstring": r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
        "comment": r"#.*$",
        "fstring": r'f"[^"]*"|f\'[^\']*\'',
    },
    "dart": {
        "class_def": r"class\s+\w+(\s+extends\s+\w+)?(\s+with\s+\w+)?\s*\{",
        "function_def": r"(\w+\s+)?\w+\s*\([^)]*\)\s*(async\s*)?\{",
        "widget": r"class\s+\w+\s+extends\s+(Stateless|Stateful)Widget",
        "import": r"import\s+['\"][^'\"]+['\"];",
        "annotation": r"@\w+",
        "comment": r"//.*$|/\*[\s\S]*?\*/",
    },
    "javascript": {
        "function_def": r"function\s+\w+\s*\([^)]*\)|=>\s*\{",
        "arrow": r"\([^)]*\)\s*=>",
        "class_def": r"class\s+\w+(\s+extends\s+\w+)?\s*\{",
        "import": r"import\s+.*\s+from\s+['\"][^'\"]+['\"]",
        "export": r"export\s+(default\s+)?",
        "comment": r"//.*$|/\*[\s\S]*?\*/",
        "template": r"`[^`]*`",
    },
}

SPECIAL_TOKENS = [
    "<|python|>",
    "<|dart|>",
    "<|javascript|>",
    "<|code|>",
    "<|comment|>",
    "<|docstring|>",
    "<|function|>",
    "<|class|>",
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
]


class CodeTokenizer:
    def __init__(
        self,
        vocab_size: int = 16384,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        self.tokenizer: Optional[Any] = None
        self.vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # HuggingFace compatibility attributes
        self.padding_side = "right"
        self.model_max_length = 2048 # Default for CodeMind
        self.is_fast = False

        self._init_vocab()

    def __call__(
        self,
        text: Union[str, List[str]],
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        batch_input_ids = []
        batch_attention_mask = []

        # HF standard padding can be "max_length", True, "longest"
        should_pad = padding is True or padding == "max_length"

        for t in text_list:
            ids = self.encode(t, add_special_tokens=True)
            
            # Truncation
            if truncation and max_length is not None:
                ids = ids[:max_length]

            # Attention mask
            attention_mask = [1] * len(ids)

            # Padding
            if should_pad and max_length is not None:
                pad_len = max_length - len(ids)
                if pad_len > 0:
                    ids += [self.pad_token_id] * pad_len
                    attention_mask += [0] * pad_len

            batch_input_ids.append(ids)
            batch_attention_mask.append(attention_mask)

        # Handle return_tensors
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor(batch_input_ids)
            attention_mask = torch.tensor(batch_attention_mask)
        else:
            input_ids = batch_input_ids
            attention_mask = batch_attention_mask

        # If a single string was passed, results should still be batch-like for TextDataset squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    @property
    def pad_token(self):
        return "<|pad|>"

    @property
    def pad_token_id(self):
        return self.vocab.get("<|pad|>", 0)

    @property
    def eos_token(self):
        return "<|eos|>"

    @property
    def eos_token_id(self):
        return self.vocab.get("<|eos|>", 0)

    @property
    def unk_token(self):
        return "<|unk|>"

    @property
    def unk_token_id(self):
        return self.vocab.get("<|unk|>", 0)

    def pad(
        self,
        encoded_inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Pad a single dictionary or a list of dictionaries (batch).
        This is required by HF DataCollator.
        """
        if isinstance(encoded_inputs, dict):
            # Single item, usually already padded by __call__ but just in case
            return encoded_inputs

        batch = encoded_inputs
        if not batch:
            return {}

        # Determine max length
        if max_length is None:
            max_length = max(len(x["input_ids"]) for x in batch)
        
        if pad_to_multiple_of is not None:
            max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        padded_batch = {
            "input_ids": [],
            "attention_mask": [],
        }
        
        # Some collators might pass labels too
        if "labels" in batch[0]:
            padded_batch["labels"] = []

        for item in batch:
            ids = list(item["input_ids"])
            attn = list(item.get("attention_mask", [1] * len(ids)))
            
            diff = max_length - len(ids)
            if diff > 0:
                ids += [self.pad_token_id] * diff
                attn += [0] * diff
            elif diff < 0:
                ids = ids[:max_length]
                attn = attn[:max_length]
                
            padded_batch["input_ids"].append(ids)
            padded_batch["attention_mask"].append(attn)
            
            if "labels" in padded_batch:
                labels = list(item["labels"])
                if diff > 0:
                    labels += [-100] * diff # Standard HF ignore index
                else:
                    labels = labels[:max_length]
                padded_batch["labels"].append(labels)

        if return_tensors == "pt":
            import torch
            return {k: torch.tensor(v) for k, v in padded_batch.items()}
            
        return padded_batch

    def _init_vocab(self) -> None:
        all_keywords = (
            PYTHON_KEYWORDS
            + PYTHON_BUILTINS
            + DART_KEYWORDS
            + FLUTTER_WIDGETS
            + JAVASCRIPT_KEYWORDS
        )

        self.vocab = {}
        idx = 0

        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1

        for keyword in all_keywords:
            token = f"▁{keyword}"
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1

        code_tokens = [
            "▁{",
            "▁}",
            "▁[",
            "▁]",
            "▁(",
            "▁)",
            "▁:",
            "▁=",
            "▁==",
            "▁!=",
            "▁<",
            "▁>",
            "▁<=",
            "▁>=",
            "▁+",
            "▁-",
            "▁*",
            "▁/",
            "▁//",
            "▁/*",
            "▁*/",
            "▁->",
            "▁=>",
            "▁self",
            "▁this",
            "▁True",
            "▁False",
            "▁None",
            "▁null",
            "▁print",
            "▁return",
            "▁if",
            "▁else",
            "▁for",
            "▁while",
            "▁class",
            "▁def",
            "▁function",
            "▁const",
            "▁let",
            "▁var",
        ]

        for token in code_tokens:
            if token not in self.vocab:
                self.vocab[token] = idx
                idx += 1

        self.initial_vocab = self.vocab.copy()

    def train(self, files: List[str], save_path: Optional[str] = None) -> None:
        if not TOKENIZERS_AVAILABLE:
            print("Warning: 'tokenizers' library not available. Using simple BPE.")
            self._train_simple_bpe(files)
        else:
            self._train_with_tokenizers_library(files)

        if save_path:
            self.save(save_path)

    def _train_simple_bpe(self, files: List[str]) -> None:
        word_freqs: Counter = Counter()

        for filepath in files:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    words = self._pre_tokenize(line.strip())
                    for word in words:
                        word_freqs[word] += 1

        vocab = dict(self.initial_vocab)
        splits: Dict[str, List[str]] = {}

        for word in word_freqs:
            splits[word] = list(word)

        while len(vocab) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs(splits, word_freqs)

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            new_token = best_pair[0] + best_pair[1].lstrip("▁")
            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            splits = self._merge_pair(*best_pair, splits)

            if len(vocab) >= self.vocab_size:
                break

        self.vocab = vocab
        self._build_tokenizer()

    def _compute_pair_freqs(
        self, splits: Dict[str, List[str]], word_freqs: Counter
    ) -> Dict[Tuple[str, str], int]:
        pair_freqs: Counter = Counter()

        for word, freq in word_freqs.items():
            split = splits.get(word, [word])
            if len(split) < 2:
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq

        return dict(pair_freqs)

    def _merge_pair(
        self, a: str, b: str, splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        new_splits = {}

        for word, split in splits.items():
            new_split = []
            i = 0

            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(a + b.lstrip("▁"))
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1

            new_splits[word] = new_split

        return new_splits

    def _train_with_tokenizers_library(self, files: List[str]) -> None:
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        special_tokens = self.special_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=special_tokens,
        )

        tokenizer.train(files, trainer)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()

        self.tokenizer = tokenizer
        self.vocab = tokenizer.get_vocab()

    def _pre_tokenize(self, text: str) -> List[str]:
        pattern = r"""
            (?P<special><\|[^|]+\|>) |
            (?P<string>\"[^\"]*\"|\'[^\']*\'|`[^`]*`) |
            (?P<number>\d+\.?\d*) |
            (?P<word>[a-zA-Z_][a-zA-Z0-9_]*) |
            (?P<op>[+\-*/=<>!&|^~@#$%?:;,\.]+) |
            (?P<bracket>[\(\)\[\]\{\}]) |
            (?P<whitespace>\s+) |
            (?P<other>.)
        """

        tokens = []
        for match in re.finditer(pattern, text, re.VERBOSE):
            token = match.group()
            if token.strip():
                tokens.append(f"▁{token}" if not token.startswith("▁") else token)

        return tokens

    def _build_tokenizer(self) -> None:
        pass

    def encode(
        self, text: str, add_special_tokens: bool = True, lang: Optional[str] = None
    ) -> List[int]:
        if self.tokenizer is not None:
            result = self.tokenizer.encode(text)
            return result.ids

        tokens = self._tokenize_simple(text)

        ids = []

        if add_special_tokens and lang:
            lang_token = f"<|{lang}|>"
            if lang_token in self.vocab:
                ids.append(self.vocab[lang_token])

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab.get("<|unk|>", 0))

        if add_special_tokens:
            eos = self.vocab.get("<|eos|>")
            if eos is not None:
                ids.append(eos)

        return ids

    def _tokenize_simple(self, text: str) -> List[str]:
        tokens = []

        for match in re.finditer(r"\S+", text):
            word = match.group()
            tokens.append(f"▁{word}")

        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.tokenizer is not None:
            text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            return text

        id_to_token = {v: k for k, v in self.vocab.items()}

        tokens = []
        for id_ in ids:
            token = id_to_token.get(id_, "")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        text = "".join(tokens)
        text = text.replace("▁", " ")

        return text.strip()

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is not None:
            self.tokenizer.save(str(path / "tokenizer.json"))

        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.save(save_directory)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        path = Path(path)

        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            special_tokens=config["special_tokens"],
        )

        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            instance.vocab = json.load(f)

        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists() and TOKENIZERS_AVAILABLE:
            instance.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            instance.tokenizer.decoder = decoders.ByteLevel()

        return instance

    @property
    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get("<|unk|>", 0))

    def id_to_token(self, id_: int) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return id_to_token.get(id_, "<|unk|>")


def create_code_tokenizer(
    vocab_size: int = 16384,
    training_files: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> CodeTokenizer:
    tokenizer = CodeTokenizer(vocab_size=vocab_size)

    if training_files:
        tokenizer.train(training_files, save_path)

    return tokenizer
