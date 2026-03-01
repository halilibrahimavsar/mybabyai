"""
Shared Utilities Module

Common functions used across the codebase:
- Device detection
- Path setup
- Configuration helpers
- Constants
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import torch

DeviceType = Literal["cuda", "cpu", "mps"]


def get_device() -> DeviceType:
    """
    Detect the best available device for computation.

    Returns:
        DeviceType: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_path(file_path: str | Path) -> Path:
    """
    Setup sys.path to include project root.

    Args:
        file_path: Path to the current file (usually __file__)

    Returns:
        Path: Project root directory
    """
    root = Path(file_path).resolve()

    while root.name != "mybabyai" and root.parent != root:
        root = root.parent

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return root


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path: The ensured directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_memory_requirements(
    num_params: int, dtype: str = "float16"
) -> Dict[str, float]:
    """
    Calculate memory requirements for a model.

    Args:
        num_params: Number of parameters
        dtype: Data type ("float16", "float32", "int8")

    Returns:
        Dict with memory requirements in GB
    """
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }

    bpb = bytes_per_param.get(dtype, 2)
    model_size_gb = (num_params * bpb) / (1024**3)

    return {
        "model_size_gb": model_size_gb,
        "training_gb": model_size_gb * 4,
        "inference_gb": model_size_gb * 1.2,
        "recommended_ram_gb": model_size_gb * 6,
    }


class ModelConstants:
    """Constants for model configurations."""

    VOCAB_SIZE_SMALL = 8192
    VOCAB_SIZE_MEDIUM = 16384
    VOCAB_SIZE_LARGE = 32768

    MAX_LENGTH_DEFAULT = 2048
    MAX_LENGTH_LONG = 4096

    TEMPERATURE_DEFAULT = 0.7
    TOP_P_DEFAULT = 0.95
    TOP_K_DEFAULT = 50

    SEED = 42


class TrainingConstants:
    """Constants for training configurations."""

    BATCH_SIZE_SMALL = 2
    BATCH_SIZE_MEDIUM = 4
    BATCH_SIZE_LARGE = 8

    LEARNING_RATE_DEFAULT = 3e-4
    LEARNING_RATE_LOW = 1e-4
    LEARNING_RATE_HIGH = 1e-3

    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    GRADIENT_ACCUMULATION_STEPS = 4


class UIConstants:
    """Constants for UI configurations."""

    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    SIDEBAR_WIDTH = 250

    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_SMALL = 10
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14

    COLOR_PRIMARY = "#6366f1"
    COLOR_SECONDARY = "#10b981"
    COLOR_ERROR = "#ef4444"
    COLOR_WARNING = "#f59e0b"
    COLOR_BACKGROUND = "#1e1e1e"
    COLOR_SURFACE = "#252525"
    COLOR_SURFACE_LIGHT = "#3d3d3d"
