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
        tabs.addTab(model_tab, "🤖 Model")

        inference_tab = self._create_inference_tab()
        tabs.addTab(inference_tab, "💬 Çıkarım")

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

        model_group = QGroupBox("Model Yapılandırması")
        model_group.setStyleSheet(self._group_style())
        model_layout = QFormLayout(model_group)

        self.model_label = QLabel("CodeMind-125M (Local)")
        model_layout.addRow("Varsayılan Model:", self.model_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        model_layout.addRow("Cihaz:", self.device_combo)

        self.quantize_check = QCheckBox("4-bit Quantization (QLoRA)")
        self.quantize_check.setChecked(True)
        model_layout.addRow("", self.quantize_check)

        self.max_memory_edit = QLineEdit("8GB")
        model_layout.addRow("Maksimum Bellek:", self.max_memory_edit)

        layout.addWidget(model_group)

        lora_group = QGroupBox("LoRA Yapılandırması")
        lora_group.setStyleSheet(self._group_style())
        lora_layout = QFormLayout(lora_group)

        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(1, 128)
        self.lora_r_spin.setValue(16)
        lora_layout.addRow("r (Rank):", self.lora_r_spin)

        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(1, 256)
        self.lora_alpha_spin.setValue(32)
        lora_layout.addRow("Alpha:", self.lora_alpha_spin)

        self.lora_dropout_spin = QDoubleSpinBox()
        self.lora_dropout_spin.setRange(0.0, 0.5)
        self.lora_dropout_spin.setValue(0.05)
        lora_layout.addRow("Dropout:", self.lora_dropout_spin)

        layout.addWidget(lora_group)

        model_info_group = QGroupBox("Model Bilgileri")
        model_info_group.setStyleSheet(self._group_style())
        info_layout = QVBoxLayout(model_info_group)

        self.model_info_label = QLabel("Model yüklenmedi")
        self.model_info_label.setWordWrap(True)
        info_layout.addWidget(self.model_info_label)

        load_model_btn = QPushButton("Model Yükle")
        load_model_btn.clicked.connect(self._load_model)
        load_model_btn.setStyleSheet(self._button_style())
        info_layout.addWidget(load_model_btn)

        unload_model_btn = QPushButton("Model Kaldır")
        unload_model_btn.clicked.connect(self._unload_model)
        unload_model_btn.setStyleSheet(self._button_style("#ef4444"))
        info_layout.addWidget(unload_model_btn)

        layout.addWidget(model_info_group)

        layout.addStretch()
        return widget

    def _create_inference_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        gen_group = QGroupBox("Üretim Parametreleri")
        gen_group.setStyleSheet(self._group_style())
        gen_layout = QFormLayout(gen_group)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 8192)
        self.max_tokens_spin.setValue(2048)
        gen_layout.addRow("Maksimum Token:", self.max_tokens_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        gen_layout.addRow("Temperature:", self.temp_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.95)
        gen_layout.addRow("Top P:", self.top_p_spin)

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(50)
        gen_layout.addRow("Top K:", self.top_k_spin)

        self.rep_penalty_spin = QDoubleSpinBox()
        self.rep_penalty_spin.setRange(1.0, 2.0)
        self.rep_penalty_spin.setSingleStep(0.05)
        self.rep_penalty_spin.setValue(1.1)
        gen_layout.addRow("Repetition Penalty:", self.rep_penalty_spin)

        layout.addWidget(gen_group)

        system_group = QGroupBox("Sistem Promptu")
        system_group.setStyleSheet(self._group_style())
        system_layout = QVBoxLayout(system_group)

        self.system_prompt_edit = QLineEdit()
        self.system_prompt_edit.setPlaceholderText("Sistem promptu...")
        self.system_prompt_edit.setText(
            "Sen yardımsever, bilgili ve nazik bir AI asistanısın."
        )
        system_layout.addWidget(self.system_prompt_edit)

        layout.addWidget(system_group)

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
        self._update_model_info()

    def set_memory_manager(self, manager: MemoryManager) -> None:
        self.memory_manager = manager
        self._update_memory_stats()

    def _load_settings(self) -> None:
        pass # Default model always CodeMind
        self.device_combo.setCurrentText(self.config.get("model.device", "auto"))
        self.quantize_check.setChecked(self.config.get("model.load_in_4bit", True))
        self.max_memory_edit.setText(self.config.get("model.max_memory", "8GB"))

        self.lora_r_spin.setValue(self.config.get("model.lora.r", 16))
        self.lora_alpha_spin.setValue(self.config.get("model.lora.lora_alpha", 32))
        self.lora_dropout_spin.setValue(
            self.config.get("model.lora.lora_dropout", 0.05)
        )

        self.max_tokens_spin.setValue(self.config.get("inference.max_new_tokens", 2048))
        self.temp_spin.setValue(self.config.get("inference.temperature", 0.7))
        self.top_p_spin.setValue(self.config.get("inference.top_p", 0.95))
        self.top_k_spin.setValue(self.config.get("inference.top_k", 50))

        self.embedding_model_combo.setEditText(
            self.config.get("memory.embedding_model", "")
        )
        self.chunk_size_spin.setValue(self.config.get("memory.chunk_size", 512))
        self.chunk_overlap_spin.setValue(self.config.get("memory.chunk_overlap", 50))
        self.max_history_spin.setValue(self.config.get("memory.max_history", 100))

    def _save_settings(self) -> None:
        self.config.set("model.default_model", "CodeMind-125M")
        self.config.set("model.device", self.device_combo.currentText())
        self.config.set("model.load_in_4bit", self.quantize_check.isChecked())
        self.config.set("model.max_memory", self.max_memory_edit.text())

        self.config.set("model.lora.r", self.lora_r_spin.value())
        self.config.set("model.lora.lora_alpha", self.lora_alpha_spin.value())
        self.config.set("model.lora.lora_dropout", self.lora_dropout_spin.value())

        self.config.set("inference.max_new_tokens", self.max_tokens_spin.value())
        self.config.set("inference.temperature", self.temp_spin.value())
        self.config.set("inference.top_p", self.top_p_spin.value())
        self.config.set("inference.top_k", self.top_k_spin.value())

        self.config.set(
            "memory.embedding_model", self.embedding_model_combo.currentText()
        )
        self.config.set("memory.chunk_size", self.chunk_size_spin.value())
        self.config.set("memory.chunk_overlap", self.chunk_overlap_spin.value())
        self.config.set("memory.max_history", self.max_history_spin.value())

        self.config.save()
        QMessageBox.information(self, "Başarılı", "Ayarlar kaydedildi!")

    def _load_model(self) -> None:
        self.main_window.load_model("CodeMind-125M")
        self._update_model_info()

    def _unload_model(self) -> None:
        self.main_window.unload_model()
        self._update_model_info()

    def _update_model_info(self) -> None:
        if self.model_manager:
            info = self.model_manager.get_model_info()
            if info.get("status") == "not_loaded":
                self.model_info_label.setText("Model yüklenmedi")
            else:
                self.model_info_label.setText(
                    f"Model: {info.get('model_name', 'Bilinmiyor')}\n"
                    f"Cihaz: {info.get('device', 'Bilinmiyor')}\n"
                    f"Toplam Parametre: {info.get('total_parameters', 0):,}\n"
                    f"Eğitilebilir: {info.get('trainable_parameters', 0):,}\n"
                    f"Eğitilebilir Yüzde: {info.get('trainable_percentage', '0%')}\n"
                    f"Quantized: {info.get('quantized', False)}"
                )
        else:
            self.model_info_label.setText("Model yüklenmedi")

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
