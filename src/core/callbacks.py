import time
import math
import psutil
import torch
import html as _html
from typing import Callable, Optional
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


try:
    from transformers.utils.notebook import NotebookProgressCallback as HFNotebookCallback

    class EnhancedNotebookCallback(HFNotebookCallback):
        """
        Extends Hugging Face's default NotebookProgressCallback.
        We use this specific class type to find and enrich the tracker 
        from within CustomTrainer.
        """
        pass
except ImportError:
    class EnhancedNotebookCallback(TrainerCallback):
        pass

class NotebookProgressBarCallback(TrainerCallback):
    """Notebook-only progress bar (no metrics table)."""

    def __init__(self, width: int = 300):
        super().__init__()
        self.width = int(width)
        self._pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            from transformers.utils.notebook import NotebookProgressBar
        except Exception:
            return

        total = int(getattr(state, "max_steps", 0) or 0)
        if total <= 0:
            total = int(getattr(args, "max_steps", 0) or 0)
        if total <= 0:
            return

        self._pbar = NotebookProgressBar(total, width=self.width)
        try:
            self._pbar.update(0, force_update=True)
        except Exception:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        if self._pbar is None:
            return
        try:
            epoch = getattr(state, "epoch", None)
            if epoch is None:
                comment = None
            else:
                epoch_str = (
                    int(epoch) if int(epoch) == epoch else f"{epoch:.2f}"
                )
                num_epochs = getattr(state, "num_train_epochs", None)
                if num_epochs is None:
                    num_epochs = getattr(args, "num_train_epochs", None)
                comment = (
                    f"Epoch {epoch_str}/{int(num_epochs)}"
                    if num_epochs is not None
                    else f"Epoch {epoch_str}"
                )
            self._pbar.update(int(state.global_step) + 1, comment=comment)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar is None:
            return
        try:
            self._pbar.update(int(getattr(self._pbar, "total", 0) or 0), force_update=True)
        except Exception:
            pass

class CompactNotebookMetricsCallback(TrainerCallback):
    """
    Single-line, table-like HTML metrics display for notebook environments.

    Uses an in-place `display_id` update so the output stays one row (no growing table).
    """

    def __init__(self, max_loss_for_ppl: float = 80.0, append_lines: bool = False):
        super().__init__()
        self.max_loss_for_ppl = float(max_loss_for_ppl)
        self.append_lines = bool(append_lines)
        self._start_time: Optional[float] = None
        self._handle = None
        self._header_shown = False

    @staticmethod
    def _fmt_eta(seconds: Optional[float]) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _world_size(args) -> int:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_world_size())
        except Exception:
            pass
        return int(getattr(args, "world_size", 1) or 1)

    def _ensure_display(self) -> None:
        if self.append_lines:
            return
        if self._handle is not None:
            return
        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            return

        header = "| step -- | loss -- | ppl -- | grad -- | lr -- | tok/s -- | gpu -- | cpu/ram --/-- | eta --:-- |"
        style = (
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
            "\"Liberation Mono\", \"Courier New\", monospace; "
            "white-space: pre; line-height: 1.25;"
        )
        self._handle = display(
            HTML(f"<div style='{style}'>{_html.escape(header)}</div>"),
            display_id=True,
        )

    def _show_header_if_needed(self) -> None:
        if self._header_shown:
            return
        header = "| step -- | loss -- | ppl -- | grad -- | lr -- | tok/s -- | gpu -- | cpu/ram --/-- | eta --:-- |"
        self._header_shown = True
        try:
            from IPython.display import HTML, display  # type: ignore

            style = (
                "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
                "\"Liberation Mono\", \"Courier New\", monospace; "
                "white-space: pre; line-height: 1.25;"
            )
            display(HTML(f"<div style='{style}'>{_html.escape(header)}</div>"))
        except Exception:
            print(header)

    def on_train_begin(self, args, state, control, **kwargs):
        if self._start_time is None:
            self._start_time = time.time()
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        if self.append_lines:
            self._show_header_if_needed()
        else:
            self._ensure_display()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if self._start_time is None:
            self._start_time = time.time()
        if self.append_lines:
            self._show_header_if_needed()
        else:
            self._ensure_display()
            if self._handle is None:
                return

        step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)

        loss = logs.get("loss", None)
        loss_f = float(loss) if isinstance(loss, (int, float)) and math.isfinite(loss) else float("nan")

        # Prefer Trainer-provided grad norm if present; otherwise keep blank.
        grad = logs.get("grad_norm", logs.get("Grad", None))
        grad_f = float(grad) if isinstance(grad, (int, float)) and math.isfinite(grad) else float("nan")

        lr = logs.get("learning_rate", logs.get("LR", None))
        lr_f = float(lr) if isinstance(lr, (int, float)) and math.isfinite(lr) else float("nan")

        ppl = float("nan")
        if math.isfinite(loss_f):
            try:
                ppl = math.exp(min(loss_f, self.max_loss_for_ppl))
            except OverflowError:
                ppl = float("inf")
            except Exception:
                ppl = float("nan")

        elapsed = time.time() - self._start_time
        steps_per_sec = (step / elapsed) if elapsed > 0 else 0.0
        eta = None
        if steps_per_sec > 0 and max_steps > 0:
            eta = (max_steps - step) / steps_per_sec

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        gpu_alloc_gb = float("nan")
        if torch.cuda.is_available():
            try:
                gpu_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            except Exception:
                gpu_alloc_gb = float("nan")

        # Approximate tokens/sec using max_length and effective batch per optimizer step.
        max_len = int(getattr(args, "codemind_max_length", 0) or 0)
        if max_len <= 0:
            model = kwargs.get("model", None)
            try:
                if model is not None and hasattr(model, "config"):
                    max_len = int(getattr(model.config, "max_position_embeddings", 0) or 0)
            except Exception:
                max_len = 0
        if max_len <= 0:
            max_len = 0
        per_device_bs = int(getattr(args, "per_device_train_batch_size", 0) or 0)
        grad_acc = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        world_size = self._world_size(args)
        tokens_per_step = per_device_bs * grad_acc * world_size * max_len
        tok_s = tokens_per_step * steps_per_sec if tokens_per_step > 0 else float("nan")

        def _fmt_float(val: float, width: int, prec: int) -> str:
            if not math.isfinite(val):
                return " " * (width - 2) + "--"
            return f"{val:{width}.{prec}f}"

        def _fmt_sci(val: float, width: int) -> str:
            if not math.isfinite(val):
                return " " * (width - 2) + "--"
            return f"{val:{width}.2e}"

        def _fmt_ppl(val: float, width: int) -> str:
            if not math.isfinite(val):
                return " " * (width - 2) + "--"
            if val >= 1e6:
                s = f"{val:.2e}"
                return f"{s:>{width}}"
            return f"{val:{width}.2f}"

        line = (
            f"| step {step:>6d} |"
            f" loss {_fmt_float(loss_f, 7, 4).strip():>7} |"
            f" ppl {_fmt_ppl(ppl, 11).strip():>11} |"
            f" grad {_fmt_float(grad_f, 7, 3).strip():>7} |"
            f" lr {_fmt_sci(lr_f, 10).strip():>10} |"
            f" tok/s {_fmt_float(tok_s, 8, 0).strip():>8} |"
            f" gpu {_fmt_float(gpu_alloc_gb, 6, 1).strip():>6} |"
            f" cpu/ram {cpu:>3.0f}%/{ram:>3.0f}% |"
            f" eta {self._fmt_eta(eta):>8} |"
        )

        if self.append_lines:
            try:
                from IPython.display import HTML, display  # type: ignore

                style = (
                    "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
                    "\"Liberation Mono\", \"Courier New\", monospace; "
                    "white-space: pre; line-height: 1.25;"
                )
                display(HTML(f"<div style='{style}'>{_html.escape(line)}</div>"))
            except Exception:
                print(line)
        else:
            try:
                from IPython.display import HTML  # type: ignore

                style = (
                    "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
                    "\"Liberation Mono\", \"Courier New\", monospace; "
                    "white-space: pre; line-height: 1.25;"
                )
                self._handle.update(HTML(f"<div style='{style}'>{_html.escape(line)}</div>"))
            except Exception:
                print(line)

class StopCallback(TrainerCallback):
    """Callback to stop training gracefully."""

    def __init__(self, check_fn: Callable[[], bool]):
        super().__init__()
        self.check_fn = check_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.check_fn():
            control.should_training_stop = True
            control.should_save_model = True
