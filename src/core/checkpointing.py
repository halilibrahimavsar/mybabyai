"""
Checkpoint metadata helpers for CodeMind.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, Iterable, Optional


@dataclass
class CheckpointMetadata:
    architecture_version: str
    vocab_size: int
    tokenizer_type: str
    special_tokens_hash: str
    created_at_utc: str
    model_config: Dict[str, Any]


def _token_values(tokens: Any) -> Iterable[str]:
    if isinstance(tokens, dict):
        return [str(k) for k in sorted(tokens.keys())]
    if isinstance(tokens, (list, tuple, set)):
        return [str(t) for t in tokens]
    return []


def compute_special_tokens_hash(tokens: Any) -> str:
    values = sorted(_token_values(tokens))
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
    return digest[:16]


def build_checkpoint_metadata(
    *,
    model_config: Dict[str, Any],
    tokenizer: Optional[Any],
    tokenizer_type: str,
    architecture_version: str = "codemind-v2",
) -> CheckpointMetadata:
    vocab_size = int(model_config.get("vocab_size", 0))
    if tokenizer is not None and hasattr(tokenizer, "vocab_size_actual"):
        vocab_size = int(tokenizer.vocab_size_actual)

    special_tokens: Any = []
    if tokenizer is not None and hasattr(tokenizer, "special_tokens"):
        special_tokens = getattr(tokenizer, "special_tokens")
    elif tokenizer is not None and hasattr(tokenizer, "special_tokens_map"):
        special_tokens = getattr(tokenizer, "special_tokens_map")

    return CheckpointMetadata(
        architecture_version=architecture_version,
        vocab_size=vocab_size,
        tokenizer_type=tokenizer_type,
        special_tokens_hash=compute_special_tokens_hash(special_tokens),
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        model_config=dict(model_config),
    )


def attach_checkpoint_metadata(
    checkpoint: Dict[str, Any], metadata: CheckpointMetadata
) -> Dict[str, Any]:
    checkpoint["checkpoint_metadata"] = asdict(metadata)
    return checkpoint


def extract_checkpoint_metadata(checkpoint: Dict[str, Any]) -> Optional[CheckpointMetadata]:
    raw = checkpoint.get("checkpoint_metadata")
    if not isinstance(raw, dict):
        return None

    try:
        return CheckpointMetadata(
            architecture_version=str(raw.get("architecture_version", "")),
            vocab_size=int(raw.get("vocab_size", 0)),
            tokenizer_type=str(raw.get("tokenizer_type", "")),
            special_tokens_hash=str(raw.get("special_tokens_hash", "")),
            created_at_utc=str(raw.get("created_at_utc", "")),
            model_config=dict(raw.get("model_config", {})),
        )
    except Exception:
        return None
