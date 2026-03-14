import time
import psutil
import torch
import math
from typing import Callable, Dict, Any
from transformers import TrainerCallback

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
            
            # Simple ETA calculation
            remaining_steps = state.max_steps - state.global_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # Hardware stats
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
    """Live-updating progress tracker for Jupyter/Colab notebooks."""

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.last_log_time = 0
        self.metrics_history = []
        self.rich_available = RICH_AVAILABLE
        if self.rich_available:
            self.console = Console()
            self.live = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.rich_available:
            self.console.print("\n[bold magenta]CodeMind Training Started[/bold magenta]")
            # Create a reusable header table to show column names once
            header = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            header.add_column("Step", width=12)
            header.add_column("Epoch", width=8)
            header.add_column("Loss", width=10)
            header.add_column("PPL", width=10)
            header.add_column("Grad", width=10)
            header.add_column("LR", width=10)
            header.add_column("Speed", width=12)
            header.add_column("ETA", width=10)
            self.console.print(header)
            self.console.print("-" * 90)

    def on_train_end(self, args, state, control, **kwargs):
        if self.rich_available:
            self.console.print("\n[bold green]Training Completed Successfully![/bold green]")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.metrics_history.append(logs.copy())
            
            # Update every 10 steps as requested, or the very first/last log
            is_milestone = (state.global_step % 10 == 0) or (state.global_step == 1) or (state.global_step == state.max_steps)
            
            if is_milestone:
                if self.rich_available:
                    row = self._generate_row(state, logs)
                    self.console.print(row)
                else:
                    # Fallback to simple print
                    ppl = math.exp(min(logs.get("loss", 0), 20))
                    print(f"Step: {state.global_step} | Loss: {logs.get('loss', 0):.4f} | PPL: {ppl:.2f}")

    def _generate_row(self, state, logs):
        """Generates a single-row table for consistent column alignment."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        speed = state.global_step / elapsed if elapsed > 0 else 0
        eta = (state.max_steps - state.global_step) / speed if speed > 0 else 0
        
        loss_val = logs.get("loss", 0)
        ppl_display = "N/A"
        if isinstance(loss_val, (int, float)) and math.isfinite(loss_val):
            capped = min(float(loss_val), 20.0)
            ppl = math.exp(capped)
            ppl_display = f"{ppl:.2f}"

        row_table = Table(show_header=False, box=None, padding=(0, 1))
        row_table.add_column("Step", width=12)
        row_table.add_column("Epoch", width=8)
        row_table.add_column("Loss", width=10)
        row_table.add_column("PPL", width=10)
        row_table.add_column("Grad", width=10)
        row_table.add_column("LR", width=10)
        row_table.add_column("Speed", width=12)
        row_table.add_column("ETA", width=10)

        row_table.add_row(
            f"{state.global_step}/{state.max_steps}",
            f"{state.epoch:.2f}",
            f"[bold yellow]{loss_val:.4f}[/]",
            ppl_display,
            f"{logs.get('grad_norm', 0):.2f}",
            f"{logs.get('learning_rate', 0):.1e}",
            f"{speed:.1f} s/s",
            f"{eta/60:.1f}m"
        )

        return row_table

class StopCallback(TrainerCallback):
    """Callback to stop training gracefully."""

    def __init__(self, check_fn: Callable[[], bool]):
        super().__init__()
        self.check_fn = check_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.check_fn():
            control.should_training_stop = True
            control.should_save_model = True
