from typing import List
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt6.QtCore import Qt, QPointF

class StatsDashboard(QWidget):
    """A custom widget to plot training metrics like loss."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.steps: List[int] = []
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")

    def add_data(self, step: int, loss: float, lr: float, grad_norm: float = 0):
        self.steps.append(step)
        self.loss_history.append(loss)
        self.lr_history.append(lr)
        self.grad_norm_history.append(grad_norm)
        self.update()

    def clear(self):
        self.steps = []
        self.loss_history = []
        self.lr_history = []
        self.grad_norm_history = []
        self.update()

    def paintEvent(self, event):
        if not self.loss_history or len(self.loss_history) < 2:
            painter = QPainter(self)
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Veri bekleniyor...")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 40

        # Draw axis
        painter.setPen(QPen(QColor("#444444"), 2))
        painter.drawLine(margin, h - margin, w - margin, h - margin) # X
        painter.drawLine(margin, margin, margin, h - margin) # Y

        # Scaling
        max_loss = max(self.loss_history) if self.loss_history else 1.0
        min_loss = min(self.loss_history) if self.loss_history else 0.0
        range_loss = max_loss - min_loss if max_loss != min_loss else 1.0
        
        max_step = max(self.steps) if self.steps else 1.0
        min_step = min(self.steps) if self.steps else 0.0
        range_step = max_step - min_step if max_step != min_step else 1.0

        def to_screen(step, loss):
            x = margin + (step - min_step) / range_step * (w - 2 * margin)
            # Flip Y since 0 is top
            y = (h - margin) - (loss - min_loss) / range_loss * (h - 2 * margin)
            return QPointF(x, y)

        # Draw Loss Curve
        path = QPolygonF()
        for s, l in zip(self.steps, self.loss_history):
            path.append(to_screen(s, l))

        painter.setPen(QPen(QColor("#6366f1"), 2))
        painter.drawPolyline(path)

        # Labels
        painter.setPen(QColor("#ffffff"))
        painter.drawText(margin, h - 10, f"{int(min_step)}")
        painter.drawText(w - margin - 20, h - 10, f"{int(max_step)}")
        painter.drawText(5, h - margin, f"{min_loss:.2f}")
        
        # Legend
        painter.setPen(QPen(QColor("#6366f1"), 4))
        painter.drawLine(w - 120, 20, w - 100, 20)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(w - 95, 25, "Loss")
        
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, "Eğitim İstatistikleri")
