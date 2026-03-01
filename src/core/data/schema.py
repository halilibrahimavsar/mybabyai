"""
Canonical dataset schema and normalization helpers for CodeMind.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict
import random


class CanonicalSample(TypedDict, total=False):
    system: str
    user: str
    assistant: str
    language: str
    task_type: str
    metadata: Dict[str, Any]


DEFAULT_LANGUAGE = "general"
DEFAULT_TASK_TYPE = "chat"


def infer_task_type(user_text: str, assistant_text: str) -> str:
    blob = f"{user_text} {assistant_text}".lower()
    if any(token in blob for token in ("debug", "error", "traceback", "hata", "fix")):
        return "debug"
    if any(token in blob for token in ("test", "unit test", "pytest")):
        return "test"
    if any(token in blob for token in ("refactor", "improve", "optimize")):
        return "refactor"
    if any(token in blob for token in ("explain", "neden", "niye", "acikla")):
        return "explain"
    if any(token in blob for token in ("function", "class", "write code", "kod yaz")):
        return "code_gen"
    return DEFAULT_TASK_TYPE


def normalize_sample(
    sample: Dict[str, Any],
    *,
    default_language: str = DEFAULT_LANGUAGE,
    source: Optional[str] = None,
) -> Optional[CanonicalSample]:
    user = (
        sample.get("user")
        or sample.get("instruction")
        or sample.get("prompt")
        or sample.get("input")
        or ""
    )
    assistant = (
        sample.get("assistant")
        or sample.get("output")
        or sample.get("response")
        or ""
    )
    system = sample.get("system", "")
    language = (sample.get("language") or default_language or DEFAULT_LANGUAGE).strip()
    task_type = (sample.get("task_type") or "").strip()
    metadata = dict(sample.get("metadata") or {})

    user = str(user).strip()
    assistant = str(assistant).strip()
    if not user or not assistant:
        return None

    if not task_type:
        task_type = infer_task_type(user, assistant)

    if source:
        metadata["source"] = source

    normalized: CanonicalSample = {
        "user": user,
        "assistant": assistant,
        "language": language,
        "task_type": task_type,
        "metadata": metadata,
    }

    system = str(system).strip()
    if system:
        normalized["system"] = system

    return normalized


def validate_sample(sample: CanonicalSample) -> Tuple[bool, str]:
    required = ("user", "assistant", "language", "task_type", "metadata")
    for key in required:
        if key not in sample:
            return False, f"missing required key: {key}"

    if not isinstance(sample["user"], str) or not sample["user"].strip():
        return False, "invalid user"
    if not isinstance(sample["assistant"], str) or not sample["assistant"].strip():
        return False, "invalid assistant"
    if not isinstance(sample["language"], str) or not sample["language"].strip():
        return False, "invalid language"
    if not isinstance(sample["task_type"], str) or not sample["task_type"].strip():
        return False, "invalid task_type"
    if not isinstance(sample["metadata"], dict):
        return False, "invalid metadata"
    return True, ""


def normalize_dataset(
    samples: Iterable[Dict[str, Any]],
    *,
    default_language: str = DEFAULT_LANGUAGE,
    source: Optional[str] = None,
) -> List[CanonicalSample]:
    normalized: List[CanonicalSample] = []
    for sample in samples:
        item = normalize_sample(
            sample, default_language=default_language, source=source
        )
        if item is None:
            continue
        ok, _ = validate_sample(item)
        if ok:
            normalized.append(item)
    return normalized


def stratified_split(
    dataset: List[CanonicalSample],
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[CanonicalSample]]:
    if not dataset:
        return {"train": [], "val": [], "test": []}

    by_bucket: Dict[Tuple[str, str], List[CanonicalSample]] = {}
    for sample in dataset:
        bucket = (sample["language"], sample["task_type"])
        by_bucket.setdefault(bucket, []).append(sample)

    rng = random.Random(seed)
    train: List[CanonicalSample] = []
    val: List[CanonicalSample] = []
    test: List[CanonicalSample] = []

    for bucket_samples in by_bucket.values():
        rng.shuffle(bucket_samples)
        n = len(bucket_samples)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = max(n - n_val - n_test, 0)

        train.extend(bucket_samples[:n_train])
        val.extend(bucket_samples[n_train : n_train + n_val])
        test.extend(bucket_samples[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}
