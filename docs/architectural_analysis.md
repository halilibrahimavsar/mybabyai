# Architectural Analysis: Fragmentation Report

## Overview
The MyBabyAI project currently exhibits a "dual-world" architecture where the `codemind/` and `src/` directories operate as parallel ecosystems with overlapping responsibilities. This fragmentation leads to redundant logic, maintenance challenges, and potential inconsistencies.

## 1. Redundant Training Ecosystems
There are at least four different entry points for training, leading to confusion about which is the "source of truth":
- **`src/core/trainer.py`**: The primary trainer used by the GUI (`TrainerWidget`). It is integrated with `ModelManager`.
- **`codemind/training/train.py`**: A parallel training implementation featuring the `CodeMindTrainer` class and `train_codemind` function.
- **`codemind/scripts/train_instruction_model.py`**: A script specifically for instruction tuning.
- **`codemind/scripts/quick_train.py`**: A simplified training script.

**Risk**: Changes made to the GUI trainer logic (e.g., stopping, resuming, metrics) do not reflect in the `codemind/` scripts. For example, `src.core.trainer.LoRATrainer` and `codemind.training.train.CodeMindTrainer` implement separate training loops.

## 2. Model Loading Inconsistencies
- **`src/core/model_manager.py`**: Designed as a singleton-like manager for the application state, but currently hardcoded to `"CodeMind-125M"`.
- **`codemind/model/base_model.py`**: Defines a `CodeMindModel` class which handles its own architecture and loading logic independently of `ModelManager`.

**Risk**: The GUI uses `ModelManager` while other tools use the `codemind.model` classes directly. If `ModelManager` is updated to support more models but the `codemind/` scripts aren't, the system becomes inconsistent.

**Risk**: The GUI might be using `ModelManager` while other tools use the `codemind.model` classes directly, leading to different behaviors in quantization, device allocation, or LoRA configuration.

## 3. Data Processing & RAG Overlap
- **`src/data/`**: Handles dataset loading and downloading for the GUI.
- **`codemind/scripts/`**: Contains numerous scripts for data collection (`collect_data.py`), normalization (`normalize_data.py`), and instruction preparation (`prepare_instruction_data.py`).
- **`src/core/chat.py` vs `codemind/scripts/chat_rag.py`**: Both implement some form of chat or RAG logic.

**Risk**: Data prepared by `codemind` scripts might not be fully compatible with the `src.data` loader expectations without manual intervention.

## 4. Configuration Split
The project uses `src/utils/config.py` as a centralized config, but many scripts in `codemind/` use hardcoded defaults or their own command-line arguments, bypassing the application's configuration system.

## Summary Recommendation
The project should decide on a single source of truth. Ideally:
1.  Move all core logic from `codemind/` into `src/core/`.
2.  Refactor standalone scripts to use the same `src.core` modules (e.g., `src.core.trainer` and `src.core.model_manager`).
3.  Unify the configuration management across both directories.
