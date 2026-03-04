"""
Model Hub — Unified checkpoint & LoRA adapter management page.

Displays:
  - Currently loaded model info card (name, parameters, VRAM)
  - Unified checkpoint list (both .pt and HuggingFace LoRA formats)
  - One-click Load / Merge LoRA / Night Shift buttons
  - Generation parameter sliders
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.utils.config import Config
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _MergeThread(QThread):
    finished = pyqtSignal(str)  # path
    error = pyqtSignal(str)

    def __init__(self, model_manager):
        super().__init__()
        self.mm = model_manager

    def run(self):
        try:
            path = self.mm.merge_lora_into_base()
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))


class _LoadThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_manager, checkpoint_path: str):
        super().__init__()
        self.mm = model_manager
        self.path = checkpoint_path

    def run(self):
        try:
            self.mm.load_model(self.path)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Small card widget
# ---------------------------------------------------------------------------

def _card(title: str, value: str, accent: str = "#3b82f6") -> QFrame:
    """Creates a small glassmorphism stat card."""
    frame = QFrame()
    frame.setFixedHeight(72)
    frame.setStyleSheet(f"""
        QFrame {{
            background: #111827;
            border: 1px solid #1e3a5f;
            border-left: 3px solid {accent};
            border-radius: 10px;
        }}
    """)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(12, 8, 12, 8)
    lay.setSpacing(2)

    ttl = QLabel(title)
    ttl.setStyleSheet("font-size: 11px; color: #64748b; font-weight: 500;")
    lay.addWidget(ttl)

    val = QLabel(value)
    val.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {accent};")
    lay.addWidget(val)
    frame._value_label = val  # type: ignore[attr-defined]
    return frame


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------

class ModelHubWidget(QWidget):
    """Full-page model management hub."""

    load_requested = pyqtSignal(str)  # checkpoint path

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.config = Config()
        self.logger = get_logger("model_hub")
        self._mm = None  # model_manager (injected after load)
        self._trainer = None
        self._memory = None
        self._merge_thread: Optional[_MergeThread] = None
        self._load_thread: Optional[_LoadThread] = None

        self._setup_ui()

    # -------------------------------------------------------------------
    # Dependency injection  (called by MainWindow)
    # -------------------------------------------------------------------

    def set_model_manager(self, mm) -> None:
        self._mm = mm
        self.refresh()

    def set_trainer(self, trainer) -> None:
        self._trainer = trainer

    def set_memory_manager(self, mem) -> None:
        self._memory = mem

    # -------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Header ----
        header = QFrame()
        header.setFixedHeight(62)
        header.setStyleSheet("""
            QFrame {
                background: #0d1424;
                border-bottom: 1px solid #1a2235;
            }
        """)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)
        title_lbl = QLabel("📡  Model Hub")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #e2e8f0;")
        h_lay.addWidget(title_lbl)
        h_lay.addStretch()

        self._refresh_btn = QPushButton("↻ Yenile")
        self._refresh_btn.clicked.connect(self.refresh)
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.setStyleSheet(self._btn_style("#1e3a5f"))
        h_lay.addWidget(self._refresh_btn)
        root.addWidget(header)

        # ---- Body (scrollable) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #0a0f1e; }")

        body = QWidget()
        body.setStyleSheet("background: #0a0f1e;")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(20, 20, 20, 20)
        body_lay.setSpacing(20)

        # -- Stat cards row --
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        self._card_name = _card("Yüklü Model", "Yüklenmedi", "#64748b")
        self._card_params = _card("Parametre Sayısı", "—", "#8b5cf6")
        self._card_status = _card("Durum", "Boşta", "#64748b")
        self._card_vram = _card("VRAM Kullanımı", "— GB", "#06b6d4")

        for c in [self._card_name, self._card_params, self._card_status, self._card_vram]:
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cards_row.addWidget(c)

        body_lay.addLayout(cards_row)

        # -- Two-column layout: checkpoints | controls --
        split = QHBoxLayout()
        split.setSpacing(16)
        split.addLayout(self._build_checkpoint_panel(), 3)
        split.addLayout(self._build_controls_panel(), 2)
        body_lay.addLayout(split)

        body_lay.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    # -------------------------------------------------------------------

    def _build_checkpoint_panel(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(10)

        title = QLabel("💾  Checkpoint Listesi")
        title.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;"
        )
        lay.addWidget(title)

        hint = QLabel("Her iki format da görüntülenir: .pt dosyaları ve LoRA adaptörler")
        hint.setStyleSheet("font-size: 11px; color: #475569;")
        lay.addWidget(hint)

        self._ckpt_list = QListWidget()
        self._ckpt_list.setStyleSheet("""
            QListWidget {
                background: #0f172a;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
                padding: 4px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 10px 12px;
                color: #cbd5e1;
                border: 1px solid transparent;
            }
            QListWidget::item:hover {
                background: #1e293b;
                border-color: #334155;
            }
            QListWidget::item:selected {
                background: #1e3a5f;
                border-color: #3b82f6;
                color: #e2e8f0;
            }
        """)
        self._ckpt_list.setMinimumHeight(280)
        self._ckpt_list.itemDoubleClicked.connect(self._on_checkpoint_double_clicked)
        lay.addWidget(self._ckpt_list, 1)

        btn_row = QHBoxLayout()
        self._load_ckpt_btn = QPushButton("▶ Seçiliyi Yükle")
        self._load_ckpt_btn.clicked.connect(self._load_selected_checkpoint)
        self._load_ckpt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_ckpt_btn.setStyleSheet(self._btn_style("#2563eb"))

        btn_row.addWidget(self._load_ckpt_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        return lay

    # -------------------------------------------------------------------

    def _build_controls_panel(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.setSpacing(14)

        # ---- Actions ----
        act_title = QLabel("⚡  Hızlı İşlemler")
        act_title.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;"
        )
        lay.addWidget(act_title)

        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background: #0f172a;
                border: 1px solid #1e3a5f;
                border-radius: 10px;
            }
        """)
        af = QVBoxLayout(actions_frame)
        af.setContentsMargins(14, 14, 14, 14)
        af.setSpacing(10)

        self._merge_btn = QPushButton("🔀  LoRA → Base Model Birleştir")
        self._merge_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._merge_btn.clicked.connect(self._merge_lora)
        self._merge_btn.setToolTip(
            "Aktif LoRA adaptörü base ağırlıklarına birleştirir.\n"
            "Sonuç daha hızlı inference sağlar."
        )
        self._merge_btn.setStyleSheet(self._btn_style("#7c3aed"))
        af.addWidget(self._merge_btn)

        self._nightshift_btn = QPushButton("🌙  Gece Vardiyasını Başlat")
        self._nightshift_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nightshift_btn.clicked.connect(self._trigger_night_shift)
        self._nightshift_btn.setToolTip(
            "System 2 başarılı deneyimlerini alır ve\n"
            "LoRA ile System 1'e aktarır (arka planda)."
        )
        self._nightshift_btn.setStyleSheet(self._btn_style("#0f766e"))
        af.addWidget(self._nightshift_btn)

        self._new_model_btn = QPushButton("🆕  Sıfır Model Oluştur")
        self._new_model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_model_btn.clicked.connect(self._create_fresh_model)
        self._new_model_btn.setStyleSheet(self._btn_style("#374151"))
        af.addWidget(self._new_model_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setStyleSheet("""
            QProgressBar { background: #1e293b; border-radius: 2px; }
            QProgressBar::chunk { background: #3b82f6; border-radius: 2px; }
        """)
        af.addWidget(self._progress)

        self._action_status = QLabel("")
        self._action_status.setStyleSheet("font-size: 11px; color: #94a3b8;")
        af.addWidget(self._action_status)
        lay.addWidget(actions_frame)

        # ---- Generation Settings ----
        gen_title = QLabel("🎛️  Üretim Ayarları")
        gen_title.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 4px;"
        )
        lay.addWidget(gen_title)

        gen_frame = QFrame()
        gen_frame.setStyleSheet(actions_frame.styleSheet())
        gf = QVBoxLayout(gen_frame)
        gf.setContentsMargins(14, 14, 14, 14)
        gf.setSpacing(10)

        # Temperature
        temp_lbl = QLabel("Sıcaklık (Temperature)")
        temp_lbl.setStyleSheet("font-size: 12px; color: #94a3b8;")
        gf.addWidget(temp_lbl)
        temp_row = QHBoxLayout()
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(1, 200)
        self._temp_slider.setValue(int(self.config.get("generation.temperature", 0.7) * 100))
        self._temp_value_lbl = QLabel(f"{self._temp_slider.value() / 100:.2f}")
        self._temp_value_lbl.setStyleSheet("color: #3b82f6; font-weight: 600; min-width: 34px;")
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_value_lbl.setText(f"{v / 100:.2f}")
        )
        self._temp_slider.setStyleSheet(self._slider_style())
        temp_row.addWidget(self._temp_slider)
        temp_row.addWidget(self._temp_value_lbl)
        gf.addLayout(temp_row)

        # Top-P
        top_p_lbl = QLabel("Top-P (Nucleus Sampling)")
        top_p_lbl.setStyleSheet(temp_lbl.styleSheet())
        gf.addWidget(top_p_lbl)
        top_p_row = QHBoxLayout()
        self._top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self._top_p_slider.setRange(1, 100)
        self._top_p_slider.setValue(int(self.config.get("generation.top_p", 0.9) * 100))
        self._top_p_value_lbl = QLabel(f"{self._top_p_slider.value() / 100:.2f}")
        self._top_p_value_lbl.setStyleSheet(self._temp_value_lbl.styleSheet())
        self._top_p_slider.valueChanged.connect(
            lambda v: self._top_p_value_lbl.setText(f"{v / 100:.2f}")
        )
        self._top_p_slider.setStyleSheet(self._slider_style())
        top_p_row.addWidget(self._top_p_slider)
        top_p_row.addWidget(self._top_p_value_lbl)
        gf.addLayout(top_p_row)

        # Max new tokens
        max_tok_lbl = QLabel("Maks. Token")
        max_tok_lbl.setStyleSheet(temp_lbl.styleSheet())
        gf.addWidget(max_tok_lbl)
        self._max_tok_spin = QSpinBox()
        self._max_tok_spin.setRange(64, 4096)
        self._max_tok_spin.setValue(self.config.get("generation.max_new_tokens", 512))
        self._max_tok_spin.setStyleSheet("""
            QSpinBox {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 8px;
                color: #e2e8f0;
            }
        """)
        gf.addWidget(self._max_tok_spin)

        # Top-K
        top_k_lbl = QLabel("Top-K")
        top_k_lbl.setStyleSheet(temp_lbl.styleSheet())
        gf.addWidget(top_k_lbl)
        self._top_k_spin = QSpinBox()
        self._top_k_spin.setRange(1, 100)
        self._top_k_spin.setValue(self.config.get("generation.top_k", 50))
        self._top_k_spin.setStyleSheet(self._max_tok_spin.styleSheet())
        gf.addWidget(self._top_k_spin)

        # Repetition Penalty
        rep_pen_lbl = QLabel("Tekrar Cezası (Repetition Penalty)")
        rep_pen_lbl.setStyleSheet(temp_lbl.styleSheet())
        gf.addWidget(rep_pen_lbl)
        self._rep_pen_spin = QDoubleSpinBox()
        self._rep_pen_spin.setRange(1.0, 2.0)
        self._rep_pen_spin.setSingleStep(0.05)
        self._rep_pen_spin.setValue(self.config.get("generation.repetition_penalty", 1.1))
        self._rep_pen_spin.setStyleSheet(self._max_tok_spin.styleSheet().replace("QSpinBox", "QDoubleSpinBox"))
        gf.addWidget(self._rep_pen_spin)

        # System Prompt
        sys_lbl = QLabel("Sistem Promptu")
        sys_lbl.setStyleSheet(temp_lbl.styleSheet())
        gf.addWidget(sys_lbl)
        from PyQt6.QtWidgets import QLineEdit
        self._sys_prompt_edit = QLineEdit()
        self._sys_prompt_edit.setText(self.config.get("inference.system_prompt", "Sen yardımsever, bilgili ve nazik bir AI asistanısın."))
        self._sys_prompt_edit.setStyleSheet(self._max_tok_spin.styleSheet().replace("QSpinBox", "QLineEdit"))
        gf.addWidget(self._sys_prompt_edit)

        apply_gen_btn = QPushButton("Ayarları Uygula")
        apply_gen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_gen_btn.clicked.connect(self._apply_gen_settings)
        apply_gen_btn.setStyleSheet(self._btn_style("#1e3a5f"))
        gf.addWidget(apply_gen_btn)

        lay.addWidget(gen_frame)
        lay.addStretch()

        return lay

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def refresh(self) -> None:
        """Refresh model info cards and checkpoint list."""
        self._refresh_model_cards()
        self._refresh_checkpoint_list()

    def _refresh_model_cards(self) -> None:
        if self._mm is None or self._mm.model is None:
            self._card_name._value_label.setText("Yüklenmedi")
            self._card_name._value_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #ef4444;")
            self._card_params._value_label.setText("—")
            self._card_status._value_label.setText("Boşta")
            self._card_vram._value_label.setText("— GB")
            return

        info = self._mm.get_model_info()
        name = str(info.get("model_name", self._mm.model_name))[:22]
        total_p = info.get("total_parameters", 0)
        param_str = f"{total_p / 1e6:.0f}M" if total_p < 1e9 else f"{total_p / 1e9:.1f}B"
        status = info.get("status", "—")

        self._card_name._value_label.setText(name)
        self._card_name._value_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #3b82f6;")
        self._card_params._value_label.setText(param_str)
        self._card_status._value_label.setText(status)

        vram_str = "— GB"
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            vram_str = f"{used:.1f} / {total:.0f} GB"
        self._card_vram._value_label.setText(vram_str)

    def _refresh_checkpoint_list(self) -> None:
        self._ckpt_list.clear()
        if self._mm is None:
            self._ckpt_list.addItem("Model manager yüklenmedi.")
            return

        checkpoints = self._mm.list_checkpoints()
        if not checkpoints:
            item = QListWidgetItem("Checkpoint bulunamadı.")
            item.setForeground(Qt.GlobalColor.gray)
            self._ckpt_list.addItem(item)
            return

        for ckpt in checkpoints:
            fmt_icon = "💙" if ckpt["format"] == "pt" else "🔌"
            mod_time = time.strftime("%d.%m.%Y %H:%M", time.localtime(ckpt["modified"]))
            text = f"{fmt_icon}  {ckpt['name']}\n      {ckpt['size_mb']} MB  ·  {mod_time}  ·  [{ckpt['format']}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, ckpt["path"])
            self._ckpt_list.addItem(item)

    def _on_checkpoint_double_clicked(self, item: QListWidgetItem) -> None:
        self._load_selected_checkpoint()

    def _load_selected_checkpoint(self) -> None:
        item = self._ckpt_list.currentItem()
        if not item:
            QMessageBox.information(self, "Bilgi", "Lütfen bir checkpoint seçin.")
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        reply = QMessageBox.question(
            self,
            "Checkpoint Yükle",
            f"Şu checkpoint yüklensin mi?\n\n{path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_busy(True, "Checkpoint yükleniyor...")
        if self._mm is None:
            self._mm = self.main_window.model_manager
        self._load_thread = _LoadThread(self._mm, path)
        self._load_thread.finished.connect(self._on_load_finished)
        self._load_thread.error.connect(self._on_action_error)
        self._load_thread.start()

    def _on_load_finished(self) -> None:
        self._set_busy(False)
        self.refresh()
        self.main_window._on_model_loaded()
        self._action_status.setText("✓ Checkpoint başarıyla yüklendi.")

    def _merge_lora(self) -> None:
        if self._mm is None or self._mm.model is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir model yükleyin.")
            return

        reply = QMessageBox.question(
            self,
            "LoRA Birleştir",
            "LoRA adaptörü base model ağırlıklarına kalıcı olarak birleştirip\n"
            "yeni bir .pt checkpoint olarak kaydedilecek.\n\nDevam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_busy(True, "LoRA birleştiriliyor...")
        self._merge_thread = _MergeThread(self._mm)
        self._merge_thread.finished.connect(self._on_merge_finished)
        self._merge_thread.error.connect(self._on_action_error)
        self._merge_thread.start()

    def _on_merge_finished(self, path: str) -> None:
        self._set_busy(False)
        self.refresh()
        self._action_status.setText(f"✓ Birleştirme tamamlandı: {Path(path).name}")
        QMessageBox.information(
            self,
            "Başarılı",
            f"Model başarıyla birleştirildi ve kaydedildi:\n{path}",
        )

    def _trigger_night_shift(self) -> None:
        if self._mm is None or self._trainer is None or self._memory is None:
            QMessageBox.warning(
                self,
                "Uyarı",
                "NightShift için model, trainer ve memory manager'ın yüklenmiş olması gerekiyor.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Gece Vardiyası",
            "System 2 deneyimleri LoRA ile System 1'e aktarılacak.\n"
            "Bu işlem arka planda çalışır. Devam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._mm.trigger_night_shift(
            background=True,
            memory_manager=self._memory,
            trainer=self._trainer,
        )
        self._action_status.setText("🌙 Gece vardiyası arka planda çalışıyor...")

    def _create_fresh_model(self) -> None:
        if self.main_window.model_manager is None:
            self.main_window.model_manager = __import__(
                "src.core.model_manager", fromlist=["ModelManager"]
            ).ModelManager(self.config)
            self._mm = self.main_window.model_manager

        self._set_busy(True, "Sıfır model oluşturuluyor...")
        try:
            self._mm.load_fresh_model("350M-MoE")
            self.main_window._on_model_loaded()
            self._action_status.setText("✓ 350M-MoE sıfır model oluşturuldu.")
        except Exception as e:
            self._action_status.setText(f"✗ Hata: {e}")
        finally:
            self._set_busy(False)
        self.refresh()

    def _apply_gen_settings(self) -> None:
        temperature = self._temp_slider.value() / 100
        top_p = self._top_p_slider.value() / 100
        max_new_tokens = self._max_tok_spin.value()
        top_k = self._top_k_spin.value()
        repetition_penalty = self._rep_pen_spin.value()
        system_prompt = self._sys_prompt_edit.text()

        self.config.set("generation.temperature", temperature)
        self.config.set("generation.top_p", top_p)
        self.config.set("generation.max_new_tokens", max_new_tokens)
        self.config.set("generation.top_k", top_k)
        self.config.set("generation.repetition_penalty", repetition_penalty)
        self.config.set("inference.system_prompt", system_prompt)
        self.config.save()

        # BUG-3 Fix: Push new values to the live InferenceEngine instance
        if hasattr(self, "main_window") and hasattr(self.main_window, "inference_engine") and self.main_window.inference_engine:
            self.main_window.inference_engine.update_settings(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                system_prompt=system_prompt,
            )
            self._action_status.setText("✓ Ayarlar kaydedildi ve canlı olarak uygulandı.")
        else:
            self._action_status.setText("✓ Ayarlar kaydedildi (model yüklenince geçerli olacak).")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _on_action_error(self, msg: str) -> None:
        self._set_busy(False)
        self._action_status.setText(f"✗ Hata: {msg[:80]}")
        QMessageBox.critical(self, "Hata", msg)

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self._progress.setVisible(busy)
        self._action_status.setText(msg)
        for btn in [self._merge_btn, self._nightshift_btn, self._load_ckpt_btn, self._new_model_btn]:
            btn.setEnabled(not busy)

    @staticmethod
    def _btn_style(bg: str) -> str:
        return f"""
            QPushButton {{
                background: {bg};
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 600;
                color: #f1f5f9;
                text-align: left;
            }}
            QPushButton:hover {{ background: {bg}cc; }}
            QPushButton:disabled {{ background: #1e293b; color: #475569; }}
        """

    @staticmethod
    def _slider_style() -> str:
        return """
            QSlider::groove:horizontal {
                height: 4px;
                background: #1e3a5f;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                background: #3b82f6;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 2px;
            }
        """
