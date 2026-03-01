import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QProgressBar,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTabWidget,
    QFormLayout,
    QDialog,
    QComboBox,
    QDialogButtonBox,
    QProgressDialog,
    QFrame,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF


from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.trainer import LoRATrainer
from src.data.database import Database
from src.data.dataset_loader import DatasetLoader
from src.data.dataset_downloader import DatasetDownloader


from src.gui.widgets.stats_dashboard import StatsDashboard
from src.gui.dialogs.download_dialog import DownloadDatasetDialog, DownloadThread
from src.gui.threads.training_thread import TrainingThread
from src.gui.dialogs.crawl_dialog import CrawlURLDialog


class TrainerWidget(QWidget):
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

        self._load_search_results: List[Dict[str, Any]] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("🎯 Model Eğitimi")
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

        data_tab = self._create_data_tab()
        tabs.addTab(data_tab, "📁 Veri")

        config_tab = self._create_config_tab()
        tabs.addTab(config_tab, "⚙️ Yapılandırma")

        # Auto-enable resume if model is already fine-tuned
        if hasattr(self.main_window, "model_manager") and getattr(self.main_window.model_manager, "is_fine_tuned", False):
            self.resume_check.setChecked(True)
            self.logger.info("Mevcut fine-tuned model tespit edildi, 'Resume' otomatik aktifleştirildi.")

        monitor_tab = self._create_monitor_tab()
        tabs.addTab(monitor_tab, "📊 İlerleme")

        stats_tab = self._create_stats_tab()
        tabs.addTab(stats_tab, "📈 Grafikler")

        layout.addWidget(tabs)

    def _create_data_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        load_group = QGroupBox("Veri Yükle & Kaynak")
        load_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        load_layout = QVBoxLayout(load_group)

        # Mode Selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("📂 Veri Tipi:"))
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItem("💬 Karşılıklı Konuşma (JSON/JSONL/CSV)", "conversations")
        self.data_type_combo.addItem("📄 Düz Metin (TXT/MD/Web)", "texts")
        self.data_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                min-width: 250px;
            }
        """)
        mode_layout.addWidget(self.data_type_combo)
        mode_layout.addStretch()
        load_layout.addLayout(mode_layout)

        # Quick Source Selection
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("🚀 Hızlı Kaynak:"))
        self.quick_source_combo = QComboBox()
        self.quick_source_combo.addItem("--- Manuel Seçim ---", None)
        
        # --- CATEGORY: WEB ---
        self.quick_source_combo.addItem("🌐 [WEB] Wikipedia (Türkçe)", {"type": "url", "data": ["https://tr.wikipedia.org/wiki/Yapay_zeka", "https://tr.wikipedia.org/wiki/Makine_öğrenimi"]})
        self.quick_source_combo.addItem("🌐 [WEB] Wikipedia (İngilizce)", {"type": "url", "data": ["https://en.wikipedia.org/wiki/Artificial_intelligence", "https://en.wikipedia.org/wiki/Machine_learning"]})
        
        # --- CATEGORY: TURKISH ---
        self.quick_source_combo.addItem("📚 [TR] Kaliteli Talimatlar (merve)", {"type": "dataset", "data": "turkish_instructions_merve"})
        self.quick_source_combo.addItem("📚 [TR] Alpaca Çeviri Seti", {"type": "dataset", "data": "turkish_alpaca"})
        self.quick_source_combo.addItem("📚 [TR] Çeşitli Görevler", {"type": "dataset", "data": "tr_instruction_dataset"})
        self.quick_source_combo.addItem("📰 [TR] Haberler (Resmi Dil)", {"type": "dataset", "data": "turkish_news_70k"})
        self.quick_source_combo.addItem("🌐 [TR] Devasa Web Kültürü (Örnek)", {"type": "dataset", "data": "culturax_tr_sample"})

        # --- CATEGORY: ENGLISH ---
        self.quick_source_combo.addItem("📚 [EN] SlimOrca (GPT-4)", {"type": "dataset", "data": "slim_orca"})
        self.quick_source_combo.addItem("💬 [EN] UltraChat 200k", {"type": "dataset", "data": "ultrachat_200k"})
        self.quick_source_combo.addItem("📚 [EN] Dolly 15k (Temiz)", {"type": "dataset", "data": "dolly_15k"})
        self.quick_source_combo.addItem("📚 [EN] WizardLM (Zor Mantık)", {"type": "dataset", "data": "wizardlm_70k"})
        self.quick_source_combo.addItem("🧮 [EN] GSM8K (Matematik)", {"type": "dataset", "data": "gsm8k_math"})
        
        # --- CATEGORY: CODE & OTHER ---
        self.quick_source_combo.addItem("💻 [CODE] CodeAlpaca 20k", {"type": "dataset", "data": "code_alpaca_20k"})
        self.quick_source_combo.addItem("🐍 [CODE] Python Talimatları", {"type": "dataset", "data": "python_instructions"})
        self.quick_source_combo.addItem("💙 [CODE] Flutter/Dart Stack", {"type": "dataset", "data": "flutter_dart_stack"})
        self.quick_source_combo.addItem("🎭 [TEXT] Tiny Shakespeare", {"type": "dataset", "data": "tiny_shakespeare"})
        self.quick_source_combo.addItem("🔄 [LANG] EN-TR Paralel Veri", {"type": "dataset", "data": "en_tr_parallel"})
        
        # --- CATEGORY: ARCHIVES ---
        self.quick_source_combo.addItem("📦 [ZIP] Örnek Zip Dosyası", {"type": "dataset", "data": "sample_archive_zip"})
        self.quick_source_combo.addItem("📦 [TAR] Örnek Tar.GZ Dosyası", {"type": "dataset", "data": "turkish_sample_archive"})

        self.quick_source_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 5px;
                min-width: 300px;
                color: #e2e8f0;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #e2e8f0;
                selection-background-color: #334155;
            }
        """)
        self.quick_source_combo.currentIndexChanged.connect(self._on_quick_source_changed)
        quick_layout.addWidget(self.quick_source_combo)
        quick_layout.addStretch()
        load_layout.addLayout(quick_layout)

        btn_layout = QHBoxLayout()

        load_file_btn = QPushButton("📄 Dosyadan Yükle")
        load_file_btn.clicked.connect(self._load_from_file)
        load_file_btn.setStyleSheet(self._button_style())
        btn_layout.addWidget(load_file_btn)

        load_dir_btn = QPushButton("📁 Dizinden Yükle")
        load_dir_btn.clicked.connect(self._load_from_directory)
        load_dir_btn.setStyleSheet(self._button_style())
        btn_layout.addWidget(load_dir_btn)

        download_btn = QPushButton("⬇️ Veriseti İndir")
        download_btn.clicked.connect(self._open_download_dialog)
        download_btn.setStyleSheet(self._button_style("#8b5cf6"))
        btn_layout.addWidget(download_btn)

        crawl_btn = QPushButton("🌐 Web'den Çek")
        crawl_btn.clicked.connect(self._open_crawl_dialog)
        crawl_btn.setStyleSheet(self._button_style("#06b6d4"))
        btn_layout.addWidget(crawl_btn)

        clear_btn = QPushButton("🗑️ Havuzu Temizle")
        clear_btn.clicked.connect(self._clear_dataset)
        clear_btn.setStyleSheet(self._button_style("#ef4444"))
        btn_layout.addWidget(clear_btn)

        load_layout.addLayout(btn_layout)

        self.dataset_stats_label = QLabel("Yüklenen toplam veri: 0 kayıt")
        self.dataset_stats_label.setStyleSheet("padding: 10px; color: #888;")
        load_layout.addWidget(self.dataset_stats_label)

        layout.addWidget(load_group)

        preview_group = QGroupBox("Veri Havuzu & Önizleme")
        preview_group.setStyleSheet(load_group.styleSheet())
        preview_layout = QVBoxLayout(preview_group)

        self.pool_list = QListWidget()
        self.pool_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                max-height: 100px;
            }
        """)
        preview_layout.addWidget(self.pool_list)

        pool_btn_layout = QHBoxLayout()
        self.remove_pool_btn = QPushButton("❌ Seçili Olanı Çıkar")
        self.remove_pool_btn.clicked.connect(self._remove_from_pool)
        self.remove_pool_btn.setStyleSheet(self._button_style("#ef4444"))
        pool_btn_layout.addWidget(self.remove_pool_btn)
        pool_btn_layout.addStretch()
        preview_layout.addLayout(pool_btn_layout)

        preview_layout.addWidget(QLabel("Son Eklenen Veri Önizleme (İlk 50 Kayıt):"))

        self.preview_list = QListWidget()
        self.preview_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
        """)
        preview_layout.addWidget(self.preview_list)

        layout.addWidget(preview_group)

        return widget

    def _create_config_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form_group = QGroupBox("Eğitim Parametreleri")
        form_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        form_layout = QFormLayout(form_group)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(self.config.get("training.num_train_epochs", 3))
        form_layout.addRow("Epoch Sayısı:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(
            self.config.get("training.per_device_train_batch_size", 8)
        )
        form_layout.addRow("Batch Size:", self.batch_size_spin)

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 16)
        self.grad_accum_spin.setValue(
            self.config.get("training.gradient_accumulation_steps", 2)
        )
        form_layout.addRow("Gradient Accumulation:", self.grad_accum_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(self.config.get("training.learning_rate", 2e-4))
        form_layout.addRow("Learning Rate:", self.lr_spin)

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 1000)
        self.warmup_spin.setValue(self.config.get("training.warmup_steps", 100))
        form_layout.addRow("Warmup Steps:", self.warmup_spin)

        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(64, 4096)
        self.max_length_spin.setSingleStep(64)
        self.max_length_spin.setValue(
            self.config.get(
                "training.max_length",
                self.config.get("memory.chunk_size", 512),
            )
        )
        form_layout.addRow("Max Sequence Length:", self.max_length_spin)

        self.pack_sequences_check = QCheckBox("Enable Sequence Packing")
        self.pack_sequences_check.setChecked(
            self.config.get("training.pack_sequences", True)
        )
        form_layout.addRow("Packing:", self.pack_sequences_check)

        self.output_dir_edit = QLineEdit(
            self.config.get("training.output_dir", "models/fine_tuned")
        )
        form_layout.addRow("Çıktı Dizini:", self.output_dir_edit)

        self.resume_check = QCheckBox("Kaldığı Yerden Devam Et (Resume)")
        self.resume_check.setChecked(False)
        self.resume_check.setToolTip("Eğer bu klasörde daha önce bir eğitim yapıldıysa en son checkpoint'ten devam eder.")
        form_layout.addRow("Devam Et:", self.resume_check)

        layout.addWidget(form_group)

        lora_group = QGroupBox("LoRA Parametreleri")
        lora_group.setStyleSheet(form_group.styleSheet())
        lora_layout = QFormLayout(lora_group)

        self.lora_r_spin = QSpinBox()
        self.lora_r_spin.setRange(1, 128)
        self.lora_r_spin.setValue(self.config.get("model.lora.r", 16))
        lora_layout.addRow("LoRA r:", self.lora_r_spin)

        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(1, 256)
        self.lora_alpha_spin.setValue(self.config.get("model.lora.lora_alpha", 32))
        lora_layout.addRow("LoRA Alpha:", self.lora_alpha_spin)

        self.lora_dropout_spin = QDoubleSpinBox()
        self.lora_dropout_spin.setRange(0.0, 0.5)
        self.lora_dropout_spin.setDecimals(2)
        self.lora_dropout_spin.setValue(
            self.config.get("model.lora.lora_dropout", 0.05)
        )
        lora_layout.addRow("LoRA Dropout:", self.lora_dropout_spin)

        layout.addWidget(lora_group)

        layout.addStretch()

        return widget

    def _create_monitor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        progress_group = QGroupBox("Eğitim İlerlemesi")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Eğitim başlatılmadı")
        self.progress_label.setStyleSheet("padding: 10px; font-size: 14px;")
        progress_layout.addWidget(self.progress_label)
        
        import torch
        import psutil
        device_type = "CPU"
        if torch.cuda.is_available():
            device_type = f"GPU ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "MPS (Apple Silicon)"
            
        ram_total = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count(logical=False)

        self.hw_info_label = QLabel(f"🖥️ Donanım: {device_type} | Toplam RAM: {ram_total:.1f} GB | CPU Çekirdekleri: {cpu_cores}")
        self.hw_info_label.setStyleSheet("padding: 5px 10px; font-size: 13px; color: #6366f1; font-weight: bold;")
        progress_layout.addWidget(self.hw_info_label)

        self.sys_stats_label = QLabel("Sistem: CPU 0% | RAM 0% | GPU 0 GB")
        self.sys_stats_label.setStyleSheet("padding: 5px 10px; font-size: 13px; color: #a8a8a8;")
        progress_layout.addWidget(self.sys_stats_label)

        layout.addWidget(progress_group)

        log_group = QGroupBox("Eğitim Günlüğü")
        log_group.setStyleSheet(progress_group.styleSheet())
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                font-family: monospace;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Eğitimi Başlat")
        self.start_btn.clicked.connect(self._start_training)
        self.start_btn.setStyleSheet(self._button_style("#10b981"))
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Durdur")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._button_style("#ef4444"))
        btn_layout.addWidget(self.stop_btn)

        self.save_ckpt_btn = QPushButton("💾 Checkpoint Kaydet")
        self.save_ckpt_btn.clicked.connect(self._save_checkpoint)
        self.save_ckpt_btn.setEnabled(False)
        self.save_ckpt_btn.setStyleSheet(self._button_style("#3b82f6"))
        btn_layout.addWidget(self.save_ckpt_btn)

        layout.addLayout(btn_layout)

        return widget

    def _create_stats_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.stats_dashboard = StatsDashboard()
        layout.addWidget(self.stats_dashboard)

        # LR Plot or other info could go here
        info_label = QLabel("Mavi çizgi: Loss (Kayıp) değerini gösterir.\nGrafik eğitim ilerledikçe güncellenir.")
        info_label.setStyleSheet("color: #888; padding: 10px;")
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

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
            QPushButton:disabled {{
                background-color: #4d4d4d;
            }}
        """

    def _save_checkpoint(self):
        if not self.trainer:
            return
            
        try:
            self._log("Manuel olarak model checkpoint'i kaydediliyor...")
            from src.core.checkpointing import build_checkpoint_metadata, attach_checkpoint_metadata
            import torch
            import time
            
            checkpoint_dir = self.config.get_path("codemind.checkpoint_dir", "codemind/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            state_dict = self.main_window.model_manager.model.state_dict()
            checkpoint = {"model_state_dict": state_dict}
            
            config_dict = getattr(self.main_window.model_manager.model, "config", None)
            config_data = config_dict.to_dict() if config_dict else {}
            
            metadata = build_checkpoint_metadata(
                model_config=config_data,
                tokenizer=self.main_window.model_manager.tokenizer,
                tokenizer_type="pretrained",
                architecture_version="codemind-v2"
            )
            checkpoint = attach_checkpoint_metadata(checkpoint, metadata)
            
            timestamp = int(time.time())
            path = checkpoint_dir / f"model_manual_{timestamp}.pt"
            torch.save(checkpoint, path)
            
            self._log(f"Checkpoint başarıyla kaydedildi: {path.name}")
        except Exception as e:
            self._log(f"Checkpoint kaydedilemedi: {e}")

    def set_trainer(self, trainer: LoRATrainer) -> None:
        self.trainer = trainer
        self.save_ckpt_btn.setEnabled(True)

    def set_database(self, database: Database) -> None:
        self.database = database

    def _on_quick_source_changed(self) -> None:
        source_data = self.quick_source_combo.currentData()
        if not source_data:
            return

        source_type = source_data["type"]
        data = source_data["data"]

        self.quick_source_combo.setCurrentIndex(0)

        if source_type == "url":
            self._add_to_pool("Hızlı Web URL'leri", "urls", data)
            self._log(f"Hızlı kaynak havuza eklendi: {len(data)} adet URL (Web)")
        elif source_type == "dataset":
            self._start_quick_download(data)

    def _add_to_pool(self, name: str, data_type: str, data: Any):
        self.dataset_pool.append({
            "name": name,
            "type": data_type,
            "data": data
        })
        self._update_dataset_ui()

    def _remove_from_pool(self) -> None:
        current_row = self.pool_list.currentRow()
        if current_row >= 0:
            removed = self.dataset_pool.pop(current_row)
            self._log(f"Havuzdan çıkarıldı: {removed['name']}")
            self._update_dataset_ui()

    def _start_quick_download(self, dataset_key: str) -> None:
        max_samples = 5000
        self._log(f"Hızlı indirme başlatılıyor: {dataset_key}")
        
        self.progress_dialog = QProgressDialog(f"'{dataset_key}' İndiriliyor...\nLütfen Bekleyin.", "İptal", 0, 0, self)
        self.progress_dialog.setWindowTitle("İndiriliyor")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        dataset_name = dataset_key
        if dataset_key in self.dataset_downloader.READY_DATASETS:
            dataset_name = self.dataset_downloader.READY_DATASETS[dataset_key]["name"]

        self.download_thread = DownloadThread(self.dataset_downloader, dataset_key, max_samples)
        self.download_thread.finished.connect(lambda data: self._on_download_finished(data, dataset_name))
        self.download_thread.error.connect(self._on_download_error)
        self.download_thread.start()

    def _open_crawl_dialog(self) -> None:
        dialog = CrawlURLDialog(self)
        if dialog.exec():
            urls = dialog.urls
            self._add_to_pool(f"Web'den Çekilen URL'ler ({len(urls)} adet)", "urls", urls)
            self._log(f"Web urls havuza eklendi: {len(urls)} adet URL")

    def _load_from_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Veri Dosyası Seç",
            "",
            "Desteklenen Formatlar (*.json *.jsonl *.csv *.txt *.md);;All Files (*)",
        )

        if filepath:
            try:
                data_type = self.data_type_combo.currentData()
                if data_type == "texts":
                    # Load as raw text chunks
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Split by triple dash or just take the whole thing
                    if "---" in content:
                        loaded_data = [s.strip() for s in content.split("---") if s.strip()]
                    else:
                        loaded_data = [content.strip()]
                else:
                    loaded_data = self.dataset_loader.load_from_file(filepath)
                
                self._add_to_pool(Path(filepath).name, data_type, loaded_data)
                self._log(f"Dosya havuza eklendi: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya yüklenemedi: {e}")

    def _load_from_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Veri Dizini Seç")

        if directory:
            try:
                data_type = self.data_type_combo.currentData()
                loaded_data = []
                if data_type == "texts":
                    path = Path(directory)
                    for filepath in path.glob("**/*"):
                        if filepath.is_file() and filepath.suffix.lower() in [".txt", ".md", ".py", ".js", ".c", ".cpp"]:
                            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                                loaded_data.append(f.read().strip())
                else:
                    loaded_data = self.dataset_loader.load_from_directory(directory)
                
                self._add_to_pool(f"Dizin: {Path(directory).name}", data_type, loaded_data)
                self._log(f"Dizin havuza eklendi: {directory}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dizin yüklenemedi: {e}")

    def _clear_dataset(self) -> None:
        self.dataset_pool = []
        self._update_dataset_ui()
        self._log("Veri havuzu temizlendi")

    def _update_dataset_ui(self, is_urls: bool = False) -> None:
        total_items = 0
        self.pool_list.clear()
        
        for item in self.dataset_pool:
            count = len(item["data"])
            total_items += count
            self.pool_list.addItem(f"{item['name']} ({count} kayıt) - [{item['type']}]")

        self.dataset_stats_label.setText(
            f"Yüklenen toplam veri: {total_items} kayıt (Çoklu-Veri Havuzu)"
        )

        self.preview_list.clear()
        if self.dataset_pool:
            last_data = self.dataset_pool[-1]["data"]
            for i, d_item in enumerate(last_data[:50]):
                if isinstance(d_item, dict):
                    item_text = (
                        f"User: {d_item.get('user', '')[:50]}... | Assistant: {d_item.get('assistant', '')[:50]}..."
                    )
                else:
                    item_text = f"Text/URL: {str(d_item)[:100]}..."
                self.preview_list.addItem(item_text)

    def _start_training(self) -> None:
        if not self.trainer:
            QMessageBox.warning(self, "Uyarı", "Model yüklenmemiş!")
            return

        if not self.dataset_pool:
            QMessageBox.warning(self, "Uyarı", "Veri havuzu boş!")
            return

        training_config = {
            "num_train_epochs": self.epochs_spin.value(),
            "per_device_train_batch_size": self.batch_size_spin.value(),
            "gradient_accumulation_steps": self.grad_accum_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "warmup_steps": self.warmup_spin.value(),
            "max_length": self.max_length_spin.value(),
            "pack_sequences": self.pack_sequences_check.isChecked(),
            "output_dir": self.output_dir_edit.text(),
            "resume_from_checkpoint": self.resume_check.isChecked(),
        }

        self.training_thread = TrainingThread(
            self.trainer, self.dataset_pool, training_config, mode="pool"
        )
        self.training_thread.log.connect(self._log)
        self.training_thread.progress.connect(self._update_progress)
        self.training_thread.finished.connect(self._training_finished)
        self.training_thread.error.connect(self._training_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_ckpt_btn.setEnabled(False)
        self.stats_dashboard.clear()
        self.training_thread.start()
        self._log("Eğitim başlatıldı")

    def _stop_training(self) -> None:
        if self.training_thread and self.training_thread.isRunning():
            self.trainer.stop_training()
            self._log("Eğitim durdurma isteği gönderildi (bir sonraki adımda duracak)...")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Durduruluyor...")
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("⏹️ Durdur")

    def _update_progress(self, progress: dict) -> None:
        if "progress_percent" in progress:
            self.progress_bar.setValue(int(progress["progress_percent"]))

        step = progress.get('current_step', 0)
        max_steps = progress.get('max_steps', 0)
        loss = progress.get('loss', 0)
        epoch = progress.get('epoch', 0)
        lr = progress.get('learning_rate', 0)
        grad_norm = progress.get('grad_norm', 0)
        speed = progress.get('speed', 0)
        eta = progress.get('eta', 0)

        # Format ETA
        eta_str = "..."
        if eta > 0:
            if eta > 3600:
                eta_str = f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
            elif eta > 60:
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
            else:
                eta_str = f"{int(eta)}s"

        cpu = progress.get('cpu_percent', 0)
        ram = progress.get('ram_percent', 0)
        gpu = progress.get('gpu_mem_gb', 0)

        self.progress_label.setText(
            f"Adım: {step}/{max_steps} | Epoch: {epoch:.2f} | Loss: {loss:.4f} | "
            f"Grad Norm: {grad_norm:.4f} | LR: {lr:.2e} | ETA: {eta_str} | Hız: {speed:.2f} step/s"
        )
        self.sys_stats_label.setText(f"Sistem: CPU {cpu:.1f}% | RAM {ram:.1f}% | GPU VRAM {gpu:.2f} GB")
        
        self.stats_dashboard.add_data(step, loss, lr, grad_norm)
        self._log(f"Step {step}/{max_steps} | Loss: {loss:.4f} | Grad: {grad_norm:.4f} | Hız: {speed:.2f} s/s")

    def _training_finished(self, metrics: dict) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_ckpt_btn.setEnabled(True)
        self.stop_btn.setText("⏹️ Durdur")
        self.progress_bar.setValue(100)
        self.progress_label.setText("Eğitim tamamlandı veya durduruldu!")
        if self.trainer and getattr(self.trainer, "should_stop", False):
             self._log("Eğitim kullanıcı tarafından durduruldu.")
        else:
             self._log(f"Eğitim tamamlandı: {metrics}")
             QMessageBox.information(self, "Başarılı", "Eğitim başarıyla tamamlandı!")

    def _training_error(self, error: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_ckpt_btn.setEnabled(True)
        self.stop_btn.setText("⏹️ Durdur")
        self._log(f"Hata: {error}")
        QMessageBox.critical(self, "Hata", f"Eğitim hatası: {error}")

    def _log(self, message: str) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _open_download_dialog(self) -> None:
        dialog = DownloadDatasetDialog(self.dataset_downloader, self)
        if dialog.exec():
            dataset_key = dialog.selected_key
            max_samples = dialog.max_samples
            
            dataset_name = dataset_key
            if dataset_key in self.dataset_downloader.READY_DATASETS:
                dataset_name = self.dataset_downloader.READY_DATASETS[dataset_key]["name"]

            self._log(f"İndirme başlatılıyor: {dataset_name} (Maks: {max_samples} örnek)")
            
            self.progress_dialog = QProgressDialog("Veriseti İndiriliyor...\nLütfen Bekleyin.", "İptal", 0, 0, self)
            self.progress_dialog.setWindowTitle("İndiriliyor")
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setCancelButton(None)
            self.progress_dialog.show()

            self.download_thread = DownloadThread(self.dataset_downloader, dataset_key, max_samples)
            self.download_thread.finished.connect(lambda data: self._on_download_finished(data, dataset_name))
            self.download_thread.error.connect(self._on_download_error)
            self.download_thread.start()

    def _on_download_finished(self, data: List[Dict[str, str]], name: str) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
            
        data_type = self.data_type_combo.currentData()
        self._add_to_pool(name, data_type, data)
        self._log(f"İndirme tamamlandı: {name} ({len(data)} kayıt)")
        
        if self.download_thread:
            self.download_thread.deleteLater()
            self.download_thread = None

    def _on_download_error(self, error: str) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
            
        self._log(f"İndirme hatası: {error}")
        QMessageBox.critical(self, "Hata", f"Veriseti indirilemedi:\n{error}")
        
        if self.download_thread:
            self.download_thread.deleteLater()
            self.download_thread = None
