"""
Training Studio — redesigned as a single tab-free page.

Layout:
  ┌─ Header ─────────────────────────────────────────────────────┐
  ├─ Body ────────────────────────────────────────────────────────┤
  │  Left panel (35%)           Right panel (65%)                │
  │  ─ Dataset Pool             ─ Live Loss Graph                │
  │  ─ Hyperparameters          ─ Metric cards                   │
  │                             ─ Log (collapsible)              │
  ├─ Bottom toolbar ─────────────────────────────────────────────┤
  │  [Başlat] [Duraklat] [Durdur]   Step ETA GPU               │
  └───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PyQt6.QtCore import Qt, QThread, QTimer, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.trainer import LoRATrainer
from src.data.database import Database
from src.data.dataset_loader import DatasetLoader
from src.data.dataset_downloader import DatasetDownloader
from src.gui.dialogs.download_dialog import DownloadDatasetDialog, DownloadThread
from src.gui.dialogs.crawl_dialog import CrawlURLDialog
from src.gui.threads.training_thread import TrainingThread


# ---------------------------------------------------------------------------
# Mini live-loss graph widget
# ---------------------------------------------------------------------------

class LossGraph(QWidget):
    """Simple polyline loss graph rendered with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._losses: List[float] = []
        self.setMinimumHeight(200)
        self.setStyleSheet("background: transparent;")

    def add_loss(self, loss: float) -> None:
        self._losses.append(loss)
        # Keep last 300 points to avoid unbounded growth
        if len(self._losses) > 300:
            self._losses = self._losses[-300:]
        self.update()

    def reset(self) -> None:
        self._losses = []
        self.update()

    # ---- painEvent ----
    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not self._losses:
            painter = QPainter(self)
            painter.setPen(QPen(QColor("#334155")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Eğitim başladığında grafik burada görünecek")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 46, 12, 12, 28

        # Grid
        grid_pen = QPen(QColor("#1a2235"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)
        for i in range(1, 5):
            y = pad_t + (h - pad_t - pad_b) * i // 4
            painter.drawLine(pad_l, y, w - pad_r, y)

        # Y axis labels
        min_loss = min(self._losses)
        max_loss = max(self._losses)
        if max_loss == min_loss:
            max_loss = min_loss + 0.01

        label_pen = QPen(QColor("#475569"))
        painter.setPen(label_pen)
        painter.setFont(painter.font())
        for i in range(5):
            val = max_loss - (max_loss - min_loss) * i / 4
            y = pad_t + (h - pad_t - pad_b) * i // 4
            painter.drawText(0, y - 8, pad_l - 4, 16, Qt.AlignmentFlag.AlignRight, f"{val:.3f}")

        # Loss polyline
        pts: List[QPointF] = []
        n = len(self._losses)
        for i, loss in enumerate(self._losses):
            x = pad_l + (w - pad_l - pad_r) * i / max(n - 1, 1)
            y = pad_t + (h - pad_t - pad_b) * (1 - (loss - min_loss) / (max_loss - min_loss))
            pts.append(QPointF(x, y))

        line_pen = QPen(QColor("#3b82f6"))
        line_pen.setWidth(2)
        painter.setPen(line_pen)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])


# ---------------------------------------------------------------------------
# Dataset Chip
# ---------------------------------------------------------------------------

class DatasetChip(QFrame):
    """A removable chip representing one dataset in the pool."""

    remove_requested = pyqtSignal(object)  # self

    _ICONS = {"hf": "🤗", "file": "📁", "url": "🌐", "conversations": "💬"}

    def __init__(self, dataset: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QFrame {
                background: #1e3a5f;
                border: 1px solid #2563eb;
                border-radius: 20px;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(6)

        icon = self._ICONS.get(dataset.get("type", "file"), "📄")
        lbl = QLabel(f"{icon} {dataset.get('name', 'Veri Seti')[:28]}")
        lbl.setStyleSheet("font-size: 12px; color: #bfdbfe;")
        lay.addWidget(lbl, 1)

        count_str = ""
        if "samples" in dataset:
            count_str = f"  {dataset['samples']:,} örnek"
        if count_str:
            cnt = QLabel(count_str)
            cnt.setStyleSheet("font-size: 11px; color: #93c5fd;")
            lay.addWidget(cnt)

        rm = QPushButton("×")
        rm.setFixedSize(22, 22)
        rm.setCursor(Qt.CursorShape.PointingHandCursor)
        rm.setStyleSheet("""
            QPushButton {
                background: #1e40af;
                color: #93c5fd;
                border-radius: 11px;
                font-size: 14px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover { background: #dc2626; color: #fca5a5; }
        """)
        rm.clicked.connect(lambda: self.remove_requested.emit(self))
        lay.addWidget(rm)


# ---------------------------------------------------------------------------
# Training Studio
# ---------------------------------------------------------------------------

class TrainerWidget(QWidget):
    """The main Training Studio page."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.config = Config()
        self.logger = get_logger("trainer_widget")

        self.trainer: Optional[LoRATrainer] = None
        self.database: Optional[Database] = None
        self.dataset_loader = DatasetLoader()
        self.dataset_downloader = DatasetDownloader()
        self.training_thread: Optional[TrainingThread] = None
        self.download_thread: Optional[DownloadThread] = None

        self.dataset_pool: List[Dict[str, Any]] = []
        self._chip_widgets: List[DatasetChip] = []
        self._is_training = False
        self._start_time: Optional[float] = None

        self._setup_ui()

    # -----------------------------------------------------------------------
    # Dependency injection
    # -----------------------------------------------------------------------

    def set_trainer(self, trainer: LoRATrainer) -> None:
        self.trainer = trainer

    def set_database(self, db: Database) -> None:
        self.database = db

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──
        header = QFrame()
        header.setFixedHeight(62)
        header.setStyleSheet("QFrame { background: #0d1424; border-bottom: 1px solid #1a2235; }")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)
        h_lay.addWidget(QLabel("🎯  Eğitim Stüdyosu", styleSheet="font-size: 18px; font-weight: 700; color: #e2e8f0;"))
        h_lay.addStretch()

        self._phase_badge = QLabel("Hazır")
        self._phase_badge.setStyleSheet("""
            QLabel {
                background: #1e3a5f;
                color: #93c5fd;
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        h_lay.addWidget(self._phase_badge)
        root.addWidget(header)

        # ── Body (left + right) ──
        body_scroll = QScrollArea()
        body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        body_scroll.setWidgetResizable(True)
        body_scroll.setStyleSheet("QScrollArea { background: #0a0f1e; }")

        body = QWidget()
        body.setStyleSheet("background: #0a0f1e;")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(16, 16, 16, 16)
        body_lay.setSpacing(16)

        body_lay.addLayout(self._build_left_panel(), 35)
        body_lay.addLayout(self._build_right_panel(), 65)

        body_scroll.setWidget(body)
        root.addWidget(body_scroll, 1)

        # ── Bottom toolbar ──
        root.addWidget(self._build_toolbar())

    # -------------------------------------------------------------------

    def _build_left_panel(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(14)

        # ─ Dataset Pool ─
        pool_frame = QFrame()
        pool_frame.setStyleSheet(self._frame_style())
        pf = QVBoxLayout(pool_frame)
        pf.setContentsMargins(14, 14, 14, 14)
        pf.setSpacing(10)

        pool_header = QHBoxLayout()
        pool_header.addWidget(QLabel("Veri Havuzu", styleSheet="font-size: 14px; font-weight: 700; color: #e2e8f0;"))
        pool_header.addStretch()
        self._pool_count = QLabel("0 veri seti")
        self._pool_count.setStyleSheet("font-size: 11px; color: #64748b;")
        pool_header.addWidget(self._pool_count)
        pf.addLayout(pool_header)

        # Chips scroll area
        chip_scroll = QScrollArea()
        chip_scroll.setFrameShape(QFrame.Shape.NoFrame)
        chip_scroll.setWidgetResizable(True)
        chip_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        chip_scroll.setFixedHeight(190)

        self._chips_container = QWidget()
        self._chips_container.setStyleSheet("background: transparent;")
        self._chips_layout = QVBoxLayout(self._chips_container)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        self._chips_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._empty_pool_lbl = QLabel("Henüz veri seti eklenmedi.\nAşağıdaki butonlarla ekleyin.")
        self._empty_pool_lbl.setStyleSheet("font-size: 12px; color: #475569;")
        self._empty_pool_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chips_layout.addWidget(self._empty_pool_lbl)

        chip_scroll.setWidget(self._chips_container)
        pf.addWidget(chip_scroll, 1)

        # Add dataset buttons
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        for label, slot in [
            ("📁 Dosya", self._add_file_dataset),
            ("🤗 HF", self._add_hf_dataset),
            ("💬 Sohbet", self._add_conversations_dataset),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._btn_style("#1e3a5f"))
            btn.clicked.connect(slot)
            add_row.addWidget(btn)
        pf.addLayout(add_row)

        add_row2 = QHBoxLayout()
        add_row2.setSpacing(6)
        for label, slot in [
            ("🌐 URL", self._add_url_dataset),
            ("🗑 Havuzu Temizle", self._clear_pool),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._btn_style("#374151"))
            btn.clicked.connect(slot)
            add_row2.addWidget(btn)
        pf.addLayout(add_row2)

        lay.addWidget(pool_frame)

        # ─ Hyperparameters ─
        hp_frame = QFrame()
        hp_frame.setStyleSheet(self._frame_style())
        hf = QVBoxLayout(hp_frame)
        hf.setContentsMargins(14, 14, 14, 14)
        hf.setSpacing(10)
        hf.addWidget(QLabel("Hiperparametreler", styleSheet="font-size: 14px; font-weight: 700; color: #e2e8f0;"))

        def _row(label: str, widget: QWidget) -> QHBoxLayout:
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 12px; color: #94a3b8; min-width: 100px;")
            r.addWidget(lbl)
            r.addWidget(widget, 1)
            return r

        def _spin(lo, hi, val, step=1) -> QSpinBox:
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setStyleSheet(self._spinbox_style())
            return s

        def _dspin(lo, hi, val, decimals=4) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setStyleSheet(self._spinbox_style())
            return s

        # Training type selector
        self._training_type_combo = QComboBox()
        self._training_type_combo.addItem("LoRA (Verimli)", "lora")
        self._training_type_combo.addItem("Full (Tam Parametre)", "full")
        self._training_type_combo.setStyleSheet(self._spinbox_style())
        self._training_type_combo.currentIndexChanged.connect(self._on_training_type_changed)

        self._epochs_spin = _spin(1, 100, self.config.get("training.num_epochs", 3))
        self._batch_spin = _spin(1, 32, self.config.get("training.batch_size", 4))
        self._lr_dspin = _dspin(0.00001, 0.1, self.config.get("training.learning_rate", 2e-4))
        self._lora_r_spin = _spin(1, 128, self.config.get("model.lora.r", 16))
        self._max_steps_spin = _spin(-1, 100000, self.config.get("training.max_steps", -1))

        for lbl, wgt in [
            ("Eğitim Tipi", self._training_type_combo),
            ("Epoch", self._epochs_spin),
            ("Batch Boyutu", self._batch_spin),
            ("Öğr. Hızı", self._lr_dspin),
            ("LoRA Rank (r)", self._lora_r_spin),
            ("Maks. Adım (-1=∞)", self._max_steps_spin),
        ]:
            hf.addLayout(_row(lbl, wgt))

        lay.addWidget(hp_frame)
        lay.addStretch()
        return lay

    # -------------------------------------------------------------------

    def _build_right_panel(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(14)

        # ─ Metric cards ─
        cards = QHBoxLayout()
        cards.setSpacing(10)

        def _mcard(title: str, init: str, accent: str) -> Tuple[QFrame, QLabel]:
            f = QFrame()
            f.setStyleSheet(f"""
                QFrame {{
                    background: #0f172a;
                    border: 1px solid #1e3a5f;
                    border-left: 3px solid {accent};
                    border-radius: 10px;
                }}
            """)
            fl = QVBoxLayout(f)
            fl.setContentsMargins(12, 8, 12, 8)
            fl.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet("font-size: 11px; color: #475569;")
            fl.addWidget(t)
            v = QLabel(init)
            v.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {accent};")
            fl.addWidget(v)
            return f, v

        self._loss_card, self._loss_val = _mcard("Loss", "—", "#3b82f6")
        self._step_card, self._step_val = _mcard("Adım", "0 / —", "#8b5cf6")
        self._eta_card, self._eta_val = _mcard("Tahmini Süre", "—", "#06b6d4")
        self._gpu_card, self._gpu_val = _mcard("GPU Belleği", "— GB", "#22c55e")

        for c in [self._loss_card, self._step_card, self._eta_card, self._gpu_card]:
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            c.setFixedHeight(72)
            cards.addWidget(c)
        lay.addLayout(cards)

        # ─ Loss graph ─
        graph_frame = QFrame()
        graph_frame.setStyleSheet("""
            QFrame {
                background: #0f172a;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
            }
        """)
        gf = QVBoxLayout(graph_frame)
        gf.setContentsMargins(10, 10, 10, 10)

        graph_header = QHBoxLayout()
        graph_header.addWidget(QLabel("Canlı Loss Grafiği", styleSheet="font-size: 13px; font-weight: 600; color: #cbd5e1;"))
        graph_header.addStretch()
        self._graph_step_lbl = QLabel("adım: 0")
        self._graph_step_lbl.setStyleSheet("font-size: 11px; color: #475569;")
        graph_header.addWidget(self._graph_step_lbl)
        gf.addLayout(graph_header)

        self._loss_graph = LossGraph()
        self._loss_graph.setMinimumHeight(220)
        gf.addWidget(self._loss_graph, 1)

        lay.addWidget(graph_frame, 1)

        # ─ Log ─
        log_frame = QFrame()
        log_frame.setStyleSheet(self._frame_style())
        lf = QVBoxLayout(log_frame)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.setSpacing(0)

        log_header = QHBoxLayout()
        log_header.setContentsMargins(12, 8, 12, 8)
        log_header.addWidget(QLabel("Eğitim Logu", styleSheet="font-size: 12px; font-weight: 600; color: #64748b;"))
        log_header.addStretch()
        clr_btn = QPushButton("Temizle")
        clr_btn.setStyleSheet(self._btn_style("#374151", height=24, font_px=11))
        clr_btn.clicked.connect(lambda: self._log_text.clear())
        log_header.addWidget(clr_btn)
        lf.addLayout(log_header)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFixedHeight(140)
        self._log_text.setStyleSheet("""
            QTextEdit {
                background: #070d1a;
                border: none;
                border-radius: 0;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                color: #94a3b8;
                padding: 8px;
            }
        """)
        lf.addWidget(self._log_text)
        lay.addWidget(log_frame)

        return lay

    # -------------------------------------------------------------------

    def _build_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setFixedHeight(58)
        toolbar.setStyleSheet("""
            QFrame {
                background: #080d18;
                border-top: 1px solid #1a2235;
            }
        """)
        lay = QHBoxLayout(toolbar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        self._start_btn = QPushButton("▶  Eğitimi Başlat")
        self._start_btn.setFixedHeight(38)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(self._btn_style("#16a34a", height=38))
        self._start_btn.clicked.connect(self._start_training)
        lay.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Durdur")
        self._stop_btn.setFixedHeight(38)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(self._btn_style("#b91c1c", height=38))
        self._stop_btn.clicked.connect(self._stop_training)
        lay.addWidget(self._stop_btn)

        lay.addStretch()

        for label, metric in [
            ("Loss:", "_loss_val"), ("Adım:", "_step_val")
        ]:
            lay.addWidget(QLabel(label, styleSheet="font-size: 12px; color: #475569;"))
            ref = getattr(self, metric)
            # We use separate smaller stat labels in toolbar instead
        # compact GPU readout in toolbar
        self._tb_gpu = QLabel("GPU: — GB")
        self._tb_gpu.setStyleSheet("font-size: 11px; color: #4b5563;")
        lay.addWidget(self._tb_gpu)

        # Auto-refresh GPU every 5s
        self._gpu_timer = QTimer()
        self._gpu_timer.timeout.connect(self._refresh_gpu)
        self._gpu_timer.start(5000)

        return toolbar

    # -------------------------------------------------------------------
    # Dataset pool management
    # -------------------------------------------------------------------

    def _add_chip(self, dataset: Dict[str, Any]) -> None:
        chip = DatasetChip(dataset)
        chip.remove_requested.connect(self._remove_chip)
        self._chip_widgets.append(chip)
        self._chips_layout.addWidget(chip)
        self.dataset_pool.append(dataset)
        self._empty_pool_lbl.setVisible(False)
        self._pool_count.setText(f"{len(self.dataset_pool)} veri seti")

    def _remove_chip(self, chip: DatasetChip) -> None:
        if chip in self._chip_widgets:
            self._chip_widgets.remove(chip)
            if chip.dataset in self.dataset_pool:
                self.dataset_pool.remove(chip.dataset)
        chip.deleteLater()
        self._empty_pool_lbl.setVisible(len(self.dataset_pool) == 0)
        self._pool_count.setText(f"{len(self.dataset_pool)} veri seti")

    def _clear_pool(self) -> None:
        for chip in list(self._chip_widgets):
            chip.deleteLater()
        self._chip_widgets.clear()
        self.dataset_pool.clear()
        self._empty_pool_lbl.setVisible(True)
        self._pool_count.setText("0 veri seti")

    def _add_file_dataset(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Veri Dosyası Seç", "",
            "Desteklenen (*.json *.jsonl *.csv *.txt *.parquet);;Tümü (*)"
        )
        for p in paths:
            self._add_chip({"type": "file", "path": p, "name": Path(p).stem})

    def _add_hf_dataset(self) -> None:
        dlg = DownloadDatasetDialog(self.dataset_downloader, self)
        if dlg.exec():
            key = dlg.selected_key
            max_samples = dlg.max_samples
            if key:
                # Determine display name
                if key.startswith("custom:"):
                    hf_id = key.replace("custom:", "").strip()
                    name = hf_id
                    dataset_key = hf_id
                else:
                    info = self.dataset_downloader.READY_DATASETS.get(key, {})
                    name = info.get("name", key)
                    dataset_key = key
                self._add_chip({
                    "type": "huggingface",
                    "name": name,
                    "dataset_key": dataset_key,
                    "max_samples": max_samples,
                })

    def _add_conversations_dataset(self) -> None:
        if not self.database:
            QMessageBox.warning(self, "Uyarı", "Veritabanı bağlantısı yok.")
            return
        convs = self.database.get_all_conversations()
        if not convs:
            QMessageBox.information(self, "Bilgi", "Henüz sohbet kaydı yok.")
            return
        self._add_chip({
            "type": "conversations",
            "name": f"Sohbet Geçmişi ({len(convs)} konuşma)",
            "conversations": convs,
        })

    def _add_url_dataset(self) -> None:
        dlg = CrawlURLDialog(self)
        if dlg.exec():
            urls = dlg.get_urls()
            depth = dlg.get_crawl_depth()
            if urls:
                name = urls[0][:35] + ("..." if len(urls[0]) > 35 else "")
                if len(urls) > 1:
                    name += f" (+{len(urls)-1})"
                if depth > 0:
                    name += f" [derinlik:{depth}]"
                self._add_chip({
                    "type": "urls",
                    "data": urls,
                    "name": name,
                    "crawl_depth": depth,
                })

    def _on_training_type_changed(self) -> None:
        is_lora = self._training_type_combo.currentData() == "lora"
        self._lora_r_spin.setEnabled(is_lora)

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def _collect_training_config(self) -> Dict[str, Any]:
        return {
            "num_epochs": self._epochs_spin.value(),
            "batch_size": self._batch_spin.value(),
            "learning_rate": self._lr_dspin.value(),
            "lora_r": self._lora_r_spin.value(),
            "max_steps": self._max_steps_spin.value() if self._max_steps_spin.value() != -1 else None,
            "training_type": self._training_type_combo.currentData(),
        }

    def _start_training(self) -> None:
        if not self.trainer:
            QMessageBox.warning(self, "Uyarı", "Model yüklenmeden eğitim başlatılamaz.")
            return
        if not self.dataset_pool:
            QMessageBox.warning(self, "Uyarı", "Veri havuzu boş. Lütfen veri seti ekleyin.")
            return

        self._is_training = True
        self._start_time = time.time()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._phase_badge.setText("⚡ Eğitim Devam Ediyor")
        self._phase_badge.setStyleSheet(
            "QLabel { background: #16a34a; color: #86efac; border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 600; }"
        )
        self._loss_graph.reset()
        self._log("Eğitim başlatılıyor...")

        cfg = self._collect_training_config()
        self.training_thread = TrainingThread(
            self.trainer, self.dataset_pool, cfg, mode="pool"
        )
        self.training_thread.progress.connect(self._on_progress)
        self.training_thread.finished.connect(self._on_training_done)
        self.training_thread.error.connect(self._on_training_error)
        self.training_thread.log.connect(self._log)
        self.training_thread.start()

    def _stop_training(self) -> None:
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.requestInterruption()
            self.training_thread.wait(2000)
        self._finish_training("Eğitim durduruldu.")

    def _on_progress(self, data: Dict[str, Any]) -> None:
        loss = data.get("loss")
        step = data.get("step", 0)
        total = data.get("total_steps") or data.get("num_steps", 0)

        if loss is not None:
            self._loss_val.setText(f"{loss:.4f}")
            self._loss_graph.add_loss(float(loss))

        step_str = f"{step} / {total}" if total else str(step)
        self._step_val.setText(step_str)
        self._graph_step_lbl.setText(f"adım: {step}")

        if self._start_time and step > 0 and total:
            elapsed = time.time() - self._start_time
            estimated_total = elapsed / step * total
            remaining = max(0, estimated_total - elapsed)
            m, s = divmod(int(remaining), 60)
            h, m = divmod(m, 60)
            self._eta_val.setText(f"{h}s {m}d {s}s" if h else f"{m}d {s}s")

    def _on_training_done(self, metrics: Dict[str, Any]) -> None:
        self._log(f"✓ Eğitim tamamlandı. Son loss: {metrics.get('train_loss', '?'):.4f}")
        self._finish_training("Eğitim tamamlandı.")

    def _on_training_error(self, msg: str) -> None:
        self._log(f"✗ Hata: {msg}")
        self._finish_training(f"Hata: {msg}")
        QMessageBox.critical(self, "Eğitim Hatası", msg)

    def _finish_training(self, message: str = "") -> None:
        self._is_training = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._phase_badge.setText("Hazır")
        self._phase_badge.setStyleSheet(
            "QLabel { background: #1e3a5f; color: #93c5fd; border-radius: 10px; padding: 3px 10px; font-size: 11px; font-weight: 600; }"
        )
        if message:
            self.main_window.update_status(message)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{ts}] {msg}")

    def _refresh_gpu(self) -> None:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            s = f"GPU: {used:.1f}/{total:.0f} GB"
            self._gpu_val.setText(f"{used:.1f} / {total:.0f} GB")
        else:
            s = "GPU: CPU modu"
        self._tb_gpu.setText(s)

    # -------------------------------------------------------------------
    # Styles
    # -------------------------------------------------------------------

    @staticmethod
    def _frame_style() -> str:
        return """
            QFrame {
                background: #0f172a;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
            }
        """

    @staticmethod
    def _btn_style(bg: str, height: int = 32, font_px: int = 12) -> str:
        return f"""
            QPushButton {{
                background: {bg};
                border: none;
                border-radius: 7px;
                height: {height}px;
                font-size: {font_px}px;
                font-weight: 600;
                color: #f1f5f9;
                padding: 0 12px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: #1e293b; color: #475569; }}
        """

    @staticmethod
    def _spinbox_style() -> str:
        return """
            QSpinBox, QDoubleSpinBox {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 8px;
                color: #e2e8f0;
                font-size: 12px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background: #334155;
                border-radius: 3px;
            }
        """
