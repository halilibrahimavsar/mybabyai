import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


console = Console()
_loggers: dict = {}


def setup_logger(
    name: str = "mybabyai", level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    rich_format = "%(message)s"
    rich_handler.setFormatter(logging.Formatter(rich_format))
    logger.addHandler(rich_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "mybabyai") -> logging.Logger:
    if name not in _loggers:
        base_dir = Path(__file__).parent.parent.parent
        log_file = base_dir / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}.log"
        return setup_logger(name, log_file=str(log_file))
    return _loggers[name]


class LoggerMixin:
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger
