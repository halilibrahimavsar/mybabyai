import time
import math
import psutil
import torch
from typing import Callable
from transformers import TrainerCallback


class UIProgressCallback(TrainerCallback):
    """Bridges HF Trainer progress to a callable for UI updates."""

    def __init__(self, progress_fn):
        super().__init__()
        self.progress_fn = progress_fn
        self.start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress_fn and state.max_steps > 0:
            if self.start_time is None:
                self.start_time = time.time()

            logs = logs or {}
            elapsed_time = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed_time if elapsed_time > 0 else 0

            remaining_steps = state.max_steps - state.global_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ram_percent = ram.percent

            gpu_alloc_gb = 0.0
            gpu_reserved_gb = 0.0
            gpu_max_alloc_gb = 0.0
            if torch.cuda.is_available():
                gpu_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                gpu_max_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            self.progress_fn({
                "current_step": state.global_step,
                "max_steps": state.max_steps,
                "progress_percent": state.global_step / state.max_steps * 100,
                "epoch": state.epoch,
                "loss": logs.get("loss", 0),
                "learning_rate": logs.get("learning_rate", 0),
                "grad_norm": logs.get("grad_norm", 0),
                "elapsed_time": elapsed_time,
                "eta": eta,
                "speed": steps_per_sec,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "gpu_alloc_gb": gpu_alloc_gb,
                "gpu_reserved_gb": gpu_reserved_gb,
                "gpu_max_alloc_gb": gpu_max_alloc_gb,
            })


class NotebookProgressCallback(TrainerCallback):
    """Plain-text progress logger for Jupyter / Colab notebooks.

    Every 10 steps (plus the very first and last), prints a clean block
    showing all key training metrics in a simple, scrollable format.
    No Rich tables, no live updates — just plain stdout lines.
    """

    def __init__(self):
        super().__init__()
        self.start_time = None

    # ── lifecycle hooks ────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("=" * 60)
        print("  🚀  CodeMind Eğitimi Başladı")
        print("=" * 60)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print("=" * 60)
        print(f"  ✅  Eğitim Tamamlandı  —  Toplam süre: {elapsed / 60:.1f} dk")
        print("=" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not (logs and "loss" in logs):
            return

        # Log every 10 steps + first + last
        is_milestone = (
            state.global_step % 10 == 0
            or state.global_step == 1
            or state.global_step == state.max_steps
        )
        if not is_milestone:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        speed = state.global_step / elapsed if elapsed > 0 else 0
        remaining = state.max_steps - state.global_step
        eta_sec = remaining / speed if speed > 0 else 0

        loss_val = logs.get("loss", 0.0)
        grad_val = logs.get("grad_norm", 0.0)
        lr_val = logs.get("learning_rate", 0.0)

        # Perplexity — cap at 20 to avoid overflow
        if isinstance(loss_val, (int, float)) and math.isfinite(loss_val):
            ppl_str = f"{math.exp(min(float(loss_val), 20.0)):.2f}"
        else:
            ppl_str = "N/A"

        # Hardware
        cpu_pct = psutil.cpu_percent()
        ram_pct = psutil.virtual_memory().percent
        gpu_line = ""
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_line = f"  GPU    : {gpu_alloc:.2f} GB / {gpu_total:.1f} GB"

        # ASCII progress bar (30 chars wide)
        pct = state.global_step / state.max_steps if state.max_steps > 0 else 0
        filled = int(pct * 30)
        bar = "█" * filled + "░" * (30 - filled)

        print("─" * 60)
        print(f"  Step   : {state.global_step} / {state.max_steps}  [{bar}]  {pct * 100:.1f}%")
        print(f"  Epoch  : {state.epoch:.3f}")
        print(f"  Loss   : {loss_val:.6f}")
        print(f"  Grad   : {grad_val:.4f}")
        print(f"  Perplex: {ppl_str}")
        print(f"  LR     : {lr_val:.2e}")
        print(f"  Speed  : {speed:.2f} steps/s")
        print(f"  Time   : {elapsed / 60:.1f} dk geçti  |  ETA: {eta_sec / 60:.1f} dk")
        print(f"  CPU    : {cpu_pct:.1f}%  |  RAM: {ram_pct:.1f}%")
        if gpu_line:
            print(gpu_line)


class StopCallback(TrainerCallback):
    """Callback to stop training gracefully."""

    def __init__(self, check_fn: Callable[[], bool]):
        super().__init__()
        self.check_fn = check_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.check_fn():
            control.should_training_stop = True
            control.should_save_model = True
