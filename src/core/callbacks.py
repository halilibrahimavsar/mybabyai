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
            self.live = Live(self._generate_table(state, {}), refresh_per_second=1, console=self.console)
            self.live.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.live:
            self.live.stop()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.metrics_history.append(logs.copy())
            if self.live:
                self.live.update(self._generate_table(state, logs))
            else:
                # Fallback to simple print if rich not available
                print(f"Step: {state.global_step}/{state.max_steps} | Loss: {logs.get('loss', 0):.4f} | LR: {logs.get('learning_rate', 0):.2e}")

    def _generate_table(self, state, logs):
        if not self.rich_available:
            return None

        elapsed = time.time() - self.start_time if self.start_time else 0
        speed = state.global_step / elapsed if elapsed > 0 else 0
        eta = (state.max_steps - state.global_step) / speed if speed > 0 else 0
        
        # Hardware
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu_alloc = 0.0
        gpu_reserved = 0.0
        gpu_max_alloc = 0.0
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_max_alloc = torch.cuda.max_memory_allocated() / (1024**3)

        loss_val = logs.get("loss", None)
        ppl_display = "N/A"
        if isinstance(loss_val, (int, float)) and math.isfinite(loss_val):
            # Avoid inf/overflow for large loss; perplexity beyond exp(20) is not informative in UI.
            capped = min(float(loss_val), 20.0)
            ppl = math.exp(capped)
            ppl_display = f">{ppl:.2f}" if loss_val > 20.0 else f"{ppl:.2f}"

        table = Table(title="CodeMind Training Progress", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Step", f"{state.global_step}/{state.max_steps}")
        table.add_row("Epoch", f"{state.epoch:.2f}")
        table.add_row("Loss", f"[bold yellow]{logs.get('loss', 0):.4f}[/]")
        table.add_row("Learning Rate", f"{logs.get('learning_rate', 0):.2e}")
        table.add_row("Perplexity", ppl_display)
        table.add_row("---", "---")
        table.add_row("Speed", f"{speed:.2f} steps/s")
        table.add_row("Elapsed", f"{elapsed/60:.1f} min")
        table.add_row("ETA", f"[bold cyan]{eta/60:.1f} min[/]")
        table.add_row("---", "---")
        table.add_row("CPU", f"{cpu}%")
        table.add_row("RAM", f"{ram}%")
        table.add_row("GPU Mem (alloc)", f"{gpu_alloc:.1f} GB")
        table.add_row("GPU Mem (rsvd)", f"{gpu_reserved:.1f} GB")
        table.add_row("GPU Mem (max)", f"{gpu_max_alloc:.1f} GB")

        return table

class StopCallback(TrainerCallback):
    """Callback to stop training gracefully."""

    def __init__(self, check_fn: Callable[[], bool]):
        super().__init__()
        self.check_fn = check_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.check_fn():
            control.should_training_stop = True
            control.should_save_model = True
