import sys
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QTabWidget,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt


from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.core.memory import MemoryManager


class SettingsWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.config = Config()
        self.logger = get_logger("settings_widget")

        self.model_manager: Optional[ModelManager] = None
        self.memory_manager: Optional[MemoryManager] = None

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("⚙️ Ayarlar")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #6366f1;
            }
        """)

        model_tab = self._create_model_tab()
        tabs.addTab(model_tab, "🤖 Uygulama & Model")

        memory_tab = self._create_memory_tab()
        tabs.addTab(memory_tab, "🧠 Hafıza")

        about_tab = self._create_about_tab()
        tabs.addTab(about_tab, "ℹ️ Hakkında")

        layout.addWidget(tabs)

        save_btn = QPushButton("💾 Ayarları Kaydet")
        save_btn.clicked.connect(self._save_settings)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        layout.addWidget(save_btn)

    def _create_model_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # App & Hardware Settings
        hardware_group = QGroupBox("Donanım Ayarları")
        hardware_group.setStyleSheet(self._group_style())
        hardware_layout = QFormLayout(hardware_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        hardware_layout.addRow("Cihaz:", self.device_combo)

        self.quantize_check = QCheckBox("4-bit Quantization (QLoRA)")
        self.quantize_check.setChecked(True)
        hardware_layout.addRow("", self.quantize_check)

        self.max_memory_edit = QLineEdit("8GB")
        hardware_layout.addRow("Maksimum Bellek:", self.max_memory_edit)

        layout.addWidget(hardware_group)

        # Removed redundant model loading buttons and info - Now handled by Model Hub
        layout.addStretch()
        return widget

    def _create_memory_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        memory_group = QGroupBox("Vektör Hafıza")
        memory_group.setStyleSheet(self._group_style())
        memory_layout = QFormLayout(memory_group)

        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.setEditable(True)
        self.embedding_model_combo.addItems(
            [
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/all-MiniLM-L6-v2",
                "intfloat/multilingual-e5-small",
            ]
        )
        memory_layout.addRow("Embedding Modeli:", self.embedding_model_combo)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(128, 2048)
        self.chunk_size_spin.setValue(512)
        memory_layout.addRow("Chunk Boyutu:", self.chunk_size_spin)

        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 256)
        self.chunk_overlap_spin.setValue(50)
        memory_layout.addRow("Chunk Örtüşmesi:", self.chunk_overlap_spin)

        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(10, 1000)
        self.max_history_spin.setValue(100)
        memory_layout.addRow("Maksimum Geçmiş:", self.max_history_spin)

        layout.addWidget(memory_group)

        doc_group = QGroupBox("Belge Yönetimi")
        doc_group.setStyleSheet(self._group_style())
        doc_layout = QVBoxLayout(doc_group)

        add_doc_btn = QPushButton("📄 Belge Ekle")
        add_doc_btn.clicked.connect(self._add_document)
        add_doc_btn.setStyleSheet(self._button_style())
        doc_layout.addWidget(add_doc_btn)

        clear_memory_btn = QPushButton("🗑️ Hafızayı Temizle")
        clear_memory_btn.clicked.connect(self._clear_memory)
        clear_memory_btn.setStyleSheet(self._button_style("#ef4444"))
        doc_layout.addWidget(clear_memory_btn)

        self.memory_stats_label = QLabel()
        self.memory_stats_label.setStyleSheet("padding: 10px;")
        doc_layout.addWidget(self.memory_stats_label)

        layout.addWidget(doc_group)

        self._update_memory_stats()

        layout.addStretch()
        return widget

    def _create_about_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        about_group = QGroupBox("Hakkında")
        about_group.setStyleSheet(self._group_style())
        about_layout = QVBoxLayout(about_group)

        info_text = QLabel(f"""
        <h2>{self.config.app_name}</h2>
        <p><b>Versiyon:</b> {self.config.app_version}</p>
        <br>
        <p>Bu uygulama, az veriyle hızlı öğrenme kabiliyetine sahip 
        bir AI asistanıdır. LoRA (Low-Rank Adaptation) teknolojisi 
        kullanılarak büyük dil modelleri kişiselleştirilebilir.</p>
        <br>
        <p><b>Özellikler:</b></p>
        <ul>
            <li>Çok dilli destek</li>
            <li>LoRA/QLoRA ile verimli fine-tuning</li>
            <li>RAG tabanlı hafıza sistemi</li>
            <li>Streaming yanıt üretimi</li>
            <li>GUI tabanlı kolay kullanım</li>
        </ul>
        <br>
        <p><b>Gereksinimler:</b></p>
        <ul>
            <li>Python 3.10+</li>
            <li>CUDA destekli GPU (önerilir)</li>
            <li>Minimum 8GB RAM</li>
        </ul>
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 10px;")
        about_layout.addWidget(info_text)

        layout.addWidget(about_group)
        layout.addStretch()
        return widget

    def _refresh_model_list(self) -> None:
        self.model_combo.clear()
        
        # Add basic aliases
        self.model_combo.addItems([
            "CodeMind-125M",
            "microsoft/phi-1_5",
            "microsoft/phi-2"
        ])

        # Scan for local checkpoints in codemind directory
        base_dir = Path(self.config.base_dir)
        codemind_dir = base_dir / "codemind"
        
        checkpoints_dirs = ["checkpoints", "checkpoints_instruct"]
        
        for d in checkpoints_dirs:
            p = codemind_dir / d
            if p.exists():
                for checkpoint in p.glob("*.pt"):
                    # Use relative path for cleaner UI
                    rel_path = checkpoint.relative_to(base_dir)
                    self.model_combo.addItem(str(rel_path))

    def _group_style(self) -> str:
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

    def _button_style(self, color: str = "#6366f1") -> str:
        return f"""
            QPushButton {{
                background-color: {color};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
        """

    def set_model_manager(self, manager: ModelManager) -> None:
        self.model_manager = manager

    def set_memory_manager(self, manager: MemoryManager) -> None:
        self.memory_manager = manager
        self._update_memory_stats()

    def _load_settings(self) -> None:
        self.device_combo.setCurrentText(self.config.get("model.device", "auto"))
        self.quantize_check.setChecked(self.config.get("model.load_in_4bit", True))
        self.max_memory_edit.setText(self.config.get("model.max_memory", "8GB"))

        self.embedding_model_combo.setEditText(
            self.config.get("memory.embedding_model", "")
        )
        self.chunk_size_spin.setValue(self.config.get("memory.chunk_size", 512))
        self.chunk_overlap_spin.setValue(self.config.get("memory.chunk_overlap", 50))
        self.max_history_spin.setValue(self.config.get("memory.max_history", 100))

    def _save_settings(self) -> None:
        self.config.set("model.device", self.device_combo.currentText())
        self.config.set("model.load_in_4bit", self.quantize_check.isChecked())
        self.config.set("model.max_memory", self.max_memory_edit.text())

        self.config.set(
            "memory.embedding_model", self.embedding_model_combo.currentText()
        )
        self.config.set("memory.chunk_size", self.chunk_size_spin.value())
        self.config.set("memory.chunk_overlap", self.chunk_overlap_spin.value())
        self.config.set("memory.max_history", self.max_history_spin.value())

        self.config.save()
        QMessageBox.information(self, "Başarılı", "Ayarlar kaydedildi!")

    def _add_document(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Belge Seç", "", "Text Files (*.txt *.md);;All Files (*)"
        )

        if filepath and self.memory_manager:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                filename = Path(filepath).name
                self.memory_manager.add_document(content, {"filename": filename})
                self._update_memory_stats()
                QMessageBox.information(self, "Başarılı", "Belge eklendi!")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Belge eklenemedi: {e}")

    def _clear_memory(self) -> None:
        reply = QMessageBox.question(
            self,
            "Onay",
            "Tüm hafıza silinecek. Emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes and self.memory_manager:
            self.memory_manager.clear_documents()
            self.memory_manager.clear_conversations()
            self._update_memory_stats()

    def _update_memory_stats(self) -> None:
        if self.memory_manager:
            stats = self.memory_manager.get_stats()
            self.memory_stats_label.setText(
                f"Belge Sayısı: {stats.get('document_count', 0)}\n"
                f"Konuşma Sayısı: {stats.get('conversation_count', 0)}"
            )
        else:
            self.memory_stats_label.setText("Hafıza yöneticisi yüklenmedi")
