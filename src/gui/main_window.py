import sys
from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QPushButton,
    QLabel,
    QFrame,
    QStatusBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal


from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.core.memory import MemoryManager
from src.core.inference import InferenceEngine
from src.core.trainer import LoRATrainer
from src.core.agent import (
    AgentCoworker, ToolRegistry, RepoSearchTool, ReadFileTool,
    RunTestsTool, CommandTool, RAGSearchTool, TrainModelTool
)
from src.data.database import Database
from src.gui.chat_widget import ChatWidget
from src.gui.trainer_widget import TrainerWidget
from src.gui.settings_widget import SettingsWidget
from src.gui.widgets.agent_widget import AgentWidget


class ModelLoadThread(QThread):
    """Loads model in a background thread to keep the UI responsive."""
    success = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_manager: ModelManager, config: Config):
        super().__init__()
        self.model_manager = model_manager
        self.config = config

    def run(self) -> None:
        try:
            self.model_manager.load_model("CodeMind-125M")
            self.success.emit()
        except Exception as e:
            self.error.emit(str(e))


class SidebarButton(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
                color: #d1d5db;
            }
            QPushButton:hover {
                background-color: #1f2937;
                border-color: #334155;
            }
            QPushButton:checked {
                background-color: #1d4ed8;
                border-color: #3b82f6;
                color: #f8fafc;
                font-weight: 600;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.config = Config()
        self.logger = get_logger("main_window")

        self.model_manager: Optional[ModelManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.trainer: Optional[LoRATrainer] = None
        self.database: Optional[Database] = None

        self.model_loaded = False

        self._setup_window()
        self._setup_ui()
        self._initialize_components()

    def _setup_window(self) -> None:
        window_size = self.config.get("gui.window_size", [1200, 800])
        self.setWindowTitle(f"{self.config.app_name} v{self.config.app_version}")
        self.setMinimumSize(QSize(800, 600))
        self.resize(window_size[0], window_size[1])

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            QStatusBar {
                background-color: #111827;
                border-top: 1px solid #1f2937;
                color: #94a3b8;
            }
        """)

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        content = self._create_content_area()
        main_layout.addWidget(content, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border-right: 1px solid #1f2937;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(6)

        title_label = QLabel(self.config.app_name)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 21px;
                font-weight: bold;
                color: #60a5fa;
                padding: 4px 8px;
            }
        """)
        layout.addWidget(title_label)

        subtitle_label = QLabel("Yerel AI Asistanı")
        subtitle_label.setStyleSheet(
            "QLabel { font-size: 11px; color: #94a3b8; padding: 0 8px 10px 8px; }"
        )
        layout.addWidget(subtitle_label)

        layout.addSpacing(10)

        self.chat_btn = SidebarButton("💬  Sohbet")
        self.chat_btn.setChecked(True)
        self.chat_btn.clicked.connect(lambda: self._switch_page(0))
        layout.addWidget(self.chat_btn)

        self.trainer_btn = SidebarButton("🎯  Eğitim")
        self.trainer_btn.clicked.connect(lambda: self._switch_page(1))
        layout.addWidget(self.trainer_btn)

        self.settings_btn = SidebarButton("⚙️  Ayarlar")
        self.settings_btn.clicked.connect(lambda: self._switch_page(2))
        layout.addWidget(self.settings_btn)

        self.agent_btn = SidebarButton("🤖  Agent")
        self.agent_btn.clicked.connect(lambda: self._switch_page(3))
        layout.addWidget(self.agent_btn)

        layout.addStretch()

        self.model_status_label = QLabel()
        self.model_status_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #fca5a5;
                background-color: #1f2937;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.model_status_label)
        self._set_model_status("Model: Yüklenmedi", "#fca5a5")

        return sidebar

    def _create_content_area(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()

        self.chat_widget = ChatWidget(self)
        self.trainer_widget = TrainerWidget(self)
        self.settings_widget = SettingsWidget(self)
        
        # Agent widget needs coworker which we init in _initialize_components
        self.agent_widget_container = QWidget()
        self.agent_layout = QVBoxLayout(self.agent_widget_container)
        self.agent_layout.setContentsMargins(0, 0, 0, 0)

        self.stack.addWidget(self.chat_widget)      # Index 0
        self.stack.addWidget(self.trainer_widget)   # Index 1
        self.stack.addWidget(self.settings_widget)  # Index 2
        self.stack.addWidget(self.agent_widget_container) # Index 3

        layout.addWidget(self.stack)

        return content

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

        self.chat_btn.setChecked(index == 0)
        self.trainer_btn.setChecked(index == 1)
        self.settings_btn.setChecked(index == 2)
        self.agent_btn.setChecked(index == 3)

    def _initialize_components(self) -> None:
        self.database = Database(self.config)
        self.memory_manager = MemoryManager(self.config)

        self.chat_widget.set_database(self.database)
        self.trainer_widget.set_database(self.database)
        self.settings_widget.set_memory_manager(self.memory_manager)

        # Initialize Agent
        registry = ToolRegistry()
        project_root = Path(self.config.base_dir)
        registry.register(RepoSearchTool(project_root))
        registry.register(ReadFileTool(project_root))
        registry.register(RunTestsTool(project_root))
        registry.register(CommandTool(project_root))
        registry.register(RAGSearchTool(self.memory_manager))
        # Note: TrainModelTool is registered, but we can't cleanly hook it directly to trainer widget here without complex plumbing, or just pass a callback that triggers the trainer tab.
        registry.register(TrainModelTool(training_callback=lambda dataset: self._switch_page(1)))
        
        # We don't have model_manager loaded yet for LLM planning. 
        # Inference engine will be supplied later in _on_model_loaded.
        self.agent_coworker = AgentCoworker(registry=registry)
        self.agent_widget = AgentWidget(self, self.agent_coworker)
        self.agent_layout.addWidget(self.agent_widget)

        if self.config.get("app.auto_load_model", True):
            self.load_model()

    def load_model(self, model_name: Optional[str] = None) -> bool:
        try:
            self.status_bar.showMessage("Model yükleniyor...")
            self._set_model_status("Model: Yükleniyor...", "#fde68a")

            if self.model_manager is None:
                self.model_manager = ModelManager(self.config)

            self._model_load_thread = ModelLoadThread(self.model_manager, self.config)
            self._model_load_thread.success.connect(self._on_model_loaded)
            self._model_load_thread.error.connect(self._on_model_load_error)
            self._model_load_thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            self._set_model_status("Model: Hata", "#fca5a5")
            self.status_bar.showMessage(f"Hata: {str(e)}")
            QMessageBox.critical(
                self,
                "Model Yükleme Hatası",
                f"Model yüklenirken hata oluştu:\n{str(e)}",
            )
            return False

    def _on_model_loaded(self) -> None:
        self.inference_engine = InferenceEngine(
            self.model_manager, self.memory_manager, self.config
        )
        self.trainer = LoRATrainer(self.model_manager, self.config)
        self.model_loaded = True
        self._set_model_status("Model: Hazır", "#86efac")
        self.status_bar.showMessage("Model başarıyla yüklendi")

        self.chat_widget.set_inference_engine(self.inference_engine)
        self.trainer_widget.set_trainer(self.trainer)
        if hasattr(self, 'agent_coworker'):
            self.agent_coworker.inference_engine = self.inference_engine
        self.settings_widget.set_model_manager(self.model_manager)

    def _on_model_load_error(self, error_msg: str) -> None:
        self.logger.error(f"Model yükleme hatası: {error_msg}")
        self._set_model_status("Model: Hata", "#fca5a5")
        self.status_bar.showMessage(f"Hata: {error_msg}")
        QMessageBox.warning(
            self,
            "Model Yükleme Uyarısı",
            f"Model yüklenemedi. Eğitim sekmesini kullanarak model oluşturabilirsiniz.\n\nHata: {error_msg}",
        )

    def unload_model(self) -> None:
        if self.model_manager:
            self.model_manager.unload_model()
            self.model_manager = None
            self.inference_engine = None
            self.trainer = None
            self.model_loaded = False
            self._set_model_status("Model: Yüklenmedi", "#fca5a5")

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self,
            "Çıkış",
            "Uygulamayı kapatmak istediğinize emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.unload_model()
            event.accept()
        else:
            event.ignore()

    def update_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

    def _set_model_status(self, text: str, color: str) -> None:
        self.model_status_label.setText(text)
        self.model_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 11px;
                color: {color};
                background-color: #1f2937;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px;
            }}
        """)
