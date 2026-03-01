from typing import Any
from PyQt6.QtCore import QThread, pyqtSignal
from src.core.trainer import LoRATrainer

class TrainingThread(QThread):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(
        self, 
        trainer: LoRATrainer, 
        data: Any, 
        config: dict,
        mode: str = "conversations" # "conversations", "texts", "urls"
    ):
        super().__init__()
        self.trainer = trainer
        self.data = data
        self.config = config
        self.mode = mode

    def run(self) -> None:
        try:
            self.log.emit(f"Eğitim başlatılıyor (Mod: {self.mode})...")

            # Wire HF Trainer progress to the Qt signal
            self.trainer.progress_callback = lambda data: self.progress.emit(data)

            if self.mode == "pool":
                metrics = self.trainer.train_from_pool(
                    self.data, **self.config
                )
            elif self.mode == "conversations":
                metrics = self.trainer.train_from_conversations(
                    self.data, **self.config
                )
            elif self.mode == "texts":
                metrics = self.trainer.train_from_texts(
                    self.data, **self.config
                )
            elif self.mode == "urls":
                metrics = self.trainer.train_from_urls(
                    self.data, **self.config
                )
            else:
                raise ValueError(f"Geçersiz eğitim modu: {self.mode}")

            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))
