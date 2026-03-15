import time
import math
import psutil
import torch
import html as _html
from collections import deque
from typing import Callable, Deque, Optional
from transformers import TrainerCallback


class UIProgressCallback(TrainerCallback):
    """Bridges HF Trainer progress to a callable for UI updates."""

    def __init__(self, progress_fn):
        super().__init__()
        self.progress_fn = progress_fn
        self.start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress_fn and state.max_steps > 0:
            if getattr(self, "start_time", None) is None:
                self.start_time = time.time()
                self.start_step = getattr(state, "global_step", 0)

            logs = logs or {}
            elapsed_time = time.time() - self.start_time
            step_diff = state.global_step - self.start_step
            steps_per_sec = step_diff / elapsed_time if elapsed_time > 0 else 0

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

class CompactNotebookMetricsCallback(TrainerCallback):
    """
    Single-line, table-like HTML metrics display for notebook environments.

    Uses an in-place `display_id` update so the output stays one row (no growing table).
    """

    def __init__(
        self,
        max_loss_for_ppl: float = 80.0,
        append_lines: bool = False,
        max_lines: int = 200,
        show_progress: bool = True,
    ):
        super().__init__()
        self.max_loss_for_ppl = float(max_loss_for_ppl)
        self.append_lines = bool(append_lines)
        self.max_lines = int(max_lines)
        self.show_progress = bool(show_progress)
        self._start_time: Optional[float] = None
        self._start_step: int = 0
        self._handle = None
        self._header_shown = False
        self._lines: Deque[str] = deque(maxlen=max(1, self.max_lines))
        self._header = "|  STEP  | LOSS    | PPL         | GRAD    | LR         | TOK/S    | GPU    | CPU/RAM   | ETA      |" + "\n" + ("-" * 100) + "\n"

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
        if self._handle is not None:
            return
        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            return

        header = self._header
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
        header = self._header
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

    def _render(self, args, state, line: Optional[str] = None) -> None:
        if line is not None:
            self._lines.append(line)

        max_steps = int(getattr(state, "max_steps", 0) or 0)
        step = int(getattr(state, "global_step", 0) or 0)
        pct = (step / max_steps * 100.0) if (self.show_progress and max_steps > 0) else None

        style = (
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
            "\"Liberation Mono\", \"Courier New\", monospace; "
            "white-space: pre; line-height: 1.25;"
        )

        bar_html = ""
        if pct is not None and math.isfinite(pct):
            pct = max(0.0, min(100.0, pct))
            bar_outer = "width: 320px; height: 10px; border: 1px solid #bbb; background: #f5f5f5; display: inline-block; vertical-align: middle;"
            bar_inner = f"width: {pct:.2f}%; height: 10px; background: #4c8bf5;"

            epoch_suffix = ""
            try:
                epoch_val = getattr(state, "epoch", None)
                epoch_cur = math.ceil(epoch_val) if isinstance(epoch_val, (int, float)) and math.isfinite(epoch_val) else None

                total_val = getattr(state, "num_train_epochs", None)
                if not (isinstance(total_val, (int, float)) and math.isfinite(total_val)):
                    total_val = getattr(args, "num_train_epochs", None)
                epoch_total = int(total_val) if isinstance(total_val, (int, float)) and math.isfinite(total_val) else None

                if epoch_cur is not None and epoch_total is not None:
                    if epoch_total > 1_000_000:  # Detect sys.maxsize / streaming
                        epoch_suffix = f"  step {step}/{max_steps}" if max_steps > 0 else f"  step {step}"
                    else:
                        epoch_suffix = f"  epoch {epoch_cur}/{epoch_total}"
            except Exception:
                epoch_suffix = ""

            bar_html = (
                f"<div style='margin: 2px 0 6px 0;'>"
                f"<span style='{style}'>Progress {pct:6.2f}%{_html.escape(epoch_suffix)}</span> "
                f"<span style='{bar_outer}'><span style='display:block; {bar_inner}'></span></span>"
                f"</div>"
            )

        body = self._header + "\n" + "\n".join(self._lines)
        html = f"<div style='{style}'>{bar_html}{_html.escape(body)}</div>"

        if self._handle is None:
            try:
                from IPython.display import HTML, display  # type: ignore

                self._handle = display(HTML(html), display_id=True)
            except Exception:
                # Non-notebook fallback: just print the latest line.
                if line is not None:
                    print(line)
            return

        try:
            from IPython.display import HTML  # type: ignore

            self._handle.update(HTML(html))
        except Exception:
            if line is not None:
                print(line)

    def on_train_begin(self, args, state, control, **kwargs):
        if self._start_time is None:
            self._start_time = time.time()
            self._start_step = getattr(state, "global_step", 0)
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self._ensure_display()
        if self.append_lines:
            self._lines.clear()
        self._render(args, state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if self._start_time is None:
            self._start_time = time.time()
            self._start_step = getattr(state, "global_step", 0)
        self._ensure_display()
        if self._handle is None:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)

        loss = logs.get("loss", None)
        loss_f = float(loss) if isinstance(loss, (int, float)) and math.isfinite(loss) else float("nan")

        # Re-implement normalization for PPL/Loss display because HF logs summed loss in this env
        grad_acc = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        vocab_size = 50257
        try:
            model = kwargs.get("model", None)
            if model is not None and hasattr(model, "config"):
                vocab_size = int(getattr(model.config, "vocab_size", 50257) or 50257)
        except Exception:
            pass

        norm_div = 1
        if grad_acc > 1 and math.isfinite(loss_f):
            expected = math.log(max(2, vocab_size))
            if loss_f > expected * 1.5:
                norm_div = grad_acc

        loss_display = (loss_f / norm_div) if (math.isfinite(loss_f) and norm_div > 1) else loss_f

        # Prefer Trainer-provided grad norm if present; otherwise keep blank.
        grad = logs.get("grad_norm", logs.get("Grad", None))
        grad_f = float(grad) if isinstance(grad, (int, float)) and math.isfinite(grad) else float("nan")

        lr = logs.get("learning_rate", logs.get("LR", None))
        lr_f = float(lr) if isinstance(lr, (int, float)) and math.isfinite(lr) else float("nan")

        ppl = float("nan")
        if math.isfinite(loss_display):
            try:
                ppl = math.exp(min(loss_display, self.max_loss_for_ppl))
            except OverflowError:
                ppl = float("inf")
            except Exception:
                ppl = float("nan")

        elapsed = time.time() - self._start_time
        step_diff = step - self._start_step
        steps_per_sec = (step_diff / elapsed) if elapsed > 0 else 0.0
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
            f"| {step:>6d} |" # STEP
            f" {_fmt_float(loss_display, 7, 4).strip():>7} |" # LOSS
            f" {_fmt_ppl(ppl, 11).strip():>11} |" # PPL
            f" {_fmt_float(grad_f, 7, 3).strip():>7} |" # GRAD
            f" {_fmt_sci(lr_f, 10).strip():>10} |" # LR
            f" {_fmt_float(tok_s, 8, 0).strip():>8} |" # TOK/S
            f" {_fmt_float(gpu_alloc_gb, 6, 1).strip():>6} |" # GPU
            f" {cpu:>3.0f}%/{ram:>3.0f}% |" # CPU/RAM
            f" {self._fmt_eta(eta):>8} |" # ETA
        )

        self._render(args, state, line=line if self.append_lines else None)
        if not self.append_lines:
            # In non-append mode, keep a single-row output.
            self._lines.clear()
            self._render(args, state, line=line)

    def on_step_end(self, args, state, control, **kwargs):
        # Update progress bar every step without appending a new log line
        if self._handle is not None:
             self._render(args, state)

class StopCallback(TrainerCallback):
    """Callback to stop training gracefully."""

    def __init__(self, check_fn: Callable[[], bool]):
        super().__init__()
        self.check_fn = check_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.check_fn():
            control.should_training_stop = True
            control.should_save_model = True
