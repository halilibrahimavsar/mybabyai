"""
Main application window — redesigned with a slim 60px icon sidebar
and 5 pages: Chat, Training Studio, Model Hub, Agent, Settings.
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.core.memory import MemoryManager
from src.core.inference import InferenceEngine
from src.core.trainer import LoRATrainer
from src.core.agent import (
    AgentCoworker,
    ToolRegistry,
    RepoSearchTool,
    ReadFileTool,
    RunTestsTool,
    CommandTool,
    RAGSearchTool,
    TrainModelTool,
)
from src.data.database import Database
from src.gui.chat_widget import ChatWidget
from src.gui.trainer_widget import TrainerWidget
from src.gui.settings_widget import SettingsWidget
from src.gui.widgets.agent_widget import AgentWidget


# ---------------------------------------------------------------------------
# Background model loading thread
# ---------------------------------------------------------------------------

class ModelLoadThread(QThread):
    """Loads model in a background thread to keep the UI responsive."""

    success = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        model_manager: ModelManager,
        config: Config,
        model_name: Optional[str] = None,
    ):
        super().__init__()
        self.model_manager = model_manager
        self.config = config
        self.model_name = model_name

    def run(self) -> None:
        try:
            self.model_manager.load_model(self.model_name)
            self.success.emit()
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Slim Icon Sidebar Button
# ---------------------------------------------------------------------------

class NavButton(QPushButton):
    """A square icon-only navigation button for the slim sidebar."""

    STYLE = """
        QPushButton {{
            background: transparent;
            border: none;
            border-radius: 12px;
            font-size: {icon_size}px;
            color: #64748b;
            padding: 0;
        }}
        QPushButton:hover {{
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
        }}
        QPushButton:checked {{
            background: rgba(59, 130, 246, 0.25);
            color: #3b82f6;
        }}
    """

    def __init__(self, icon: str, tooltip: str, parent=None):
        super().__init__(icon, parent)
        self.setCheckable(True)
        self.setFixedSize(52, 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setStyleSheet(self.STYLE.format(icon_size=22))


# ---------------------------------------------------------------------------
# Slim Sidebar Widget
# ---------------------------------------------------------------------------

class SlimSidebar(QWidget):
    """60px icon-only left sidebar with nav buttons and model status."""

    page_changed = pyqtSignal(int)

    _PAGES = [
        ("💬", "Sohbet (Chat)"),
        ("🎯", "Eğitim Stüdyosu"),
        ("📡", "Model Hub"),
        ("🤖", "Ajan (Agent)"),
        ("⚙️", "Ayarlar"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(64)
        self.setStyleSheet("""
            SlimSidebar {
                background-color: #080d18;
                border-right: 1px solid #1a2235;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 12, 6, 12)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # App logo / icon
        logo = QLabel("🧠")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedHeight(48)
        logo.setStyleSheet("font-size: 26px; color: #3b82f6; padding-bottom: 8px;")
        layout.addWidget(logo)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #1a2235;")
        layout.addWidget(sep)
        layout.addSpacing(4)

        # Nav buttons
        self._buttons: list[NavButton] = []
        for i, (icon, tip) in enumerate(self._PAGES):
            btn = NavButton(icon, tip)
            btn.clicked.connect(lambda checked, idx=i: self._on_page_clicked(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()

        # Model status dot
        self._status_dot = QLabel("●")
        self._status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_dot.setFixedHeight(28)
        self._status_dot.setToolTip("Model durumu")
        self._set_status_color("#4b5563")  # Gray = not loaded
        layout.addWidget(self._status_dot)

        # Activate first page
        self._buttons[0].setChecked(True)

    # -----------------------------------------------------------------------

    def _update_button_states(self, index: int) -> None:
        """Update visual state of nav buttons without emitting signals."""
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)

    def _on_page_clicked(self, index: int) -> None:
        """Called by user click — updates state and emits signal."""
        self._update_button_states(index)
        self.page_changed.emit(index)

    def set_active_page(self, index: int) -> None:
        """Called programmatically — updates state but does NOT emit signal."""
        self._update_button_states(index)

    def set_model_status(self, status: str, model_name: str = "") -> None:
        """
        status: 'loading' | 'ready' | 'error' | 'idle'
        model_name: e.g. 'CodeMind-350M-MoE'
        """
        colors = {
            "loading": "#fbbf24",
            "ready":   "#22c55e",
            "error":   "#ef4444",
            "idle":    "#4b5563",
        }
        color = colors.get(status, "#4b5563")
        self._set_status_color(color)
        
        name_display = f" ({model_name})" if model_name else ""
        tip_map = {
            "loading": f"Model yükleniyor...{name_display}",
            "ready":   f"Model hazır ✓{name_display}",
            "error":   f"Model yüklenemedi ✗{name_display}",
            "idle":    "Model yüklenmedi",
        }
        self._status_dot.setToolTip(tip_map.get(status, ""))
        
        # Show model name label if available
        if not hasattr(self, "_model_name_label"):
            from PyQt6.QtWidgets import QLabel
            self._model_name_label = QLabel("")
            self._model_name_label.setStyleSheet("color: #94a3b8; font-size: 9px;")
            # Insert after status dot
            self.layout().addWidget(self._model_name_label)
        
        if model_name and status == "ready":
            self._model_name_label.setText(model_name)
            self._model_name_label.setStyleSheet("color: #22c55e; font-size: 9px; font-weight: bold;")
        elif status == "loading" and model_name:
            self._model_name_label.setText(f"{model_name}...")
            self._model_name_label.setStyleSheet("color: #fbbf24; font-size: 9px;")
        else:
            self._model_name_label.setText("")

    def _set_status_color(self, color: str) -> None:
        self._status_dot.setStyleSheet(f"font-size: 18px; color: {color};")


# ---------------------------------------------------------------------------
# Glassy header used at the top of every page
# ---------------------------------------------------------------------------

class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self.setStyleSheet("""
            PageHeader {
                background: #0d1424;
                border-bottom: 1px solid #1a2235;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #e2e8f0;")
        lay.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet("font-size: 12px; color: #64748b;")
            lay.addWidget(sub_lbl)

        lay.addStretch()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Window setup
    # -----------------------------------------------------------------------

    def _setup_window(self) -> None:
        w, h = self.config.get("gui.window_size", [1280, 820])
        self.setWindowTitle(f"{self.config.app_name}")
        self.setMinimumSize(QSize(900, 640))
        self.resize(w, h)

        # Application-wide dark stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0a0f1e;
                color: #e2e8f0;
                font-family: 'Segoe UI', 'SF Pro Display', system-ui, sans-serif;
            }
            QScrollBar:vertical {
                background: #0f172a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1e3a5f;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #2563eb;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QStatusBar {
                background: #080d18;
                border-top: 1px solid #1a2235;
                color: #4b5563;
                font-size: 11px;
            }
            QToolTip {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
        """)

    # -----------------------------------------------------------------------
    # UI layout
    # -----------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Slim sidebar
        self.sidebar = SlimSidebar()
        self.sidebar.page_changed.connect(self._switch_page)
        root.addWidget(self.sidebar)

        # Content area (pages)
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background: #0a0f1e; }")
        root.addWidget(self.stack, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")

        # Pages will be added in _initialize_components()

    # -----------------------------------------------------------------------
    # Component initialization
    # -----------------------------------------------------------------------

    def _initialize_components(self) -> None:
        self.database = Database(self.config)
        self.memory_manager = MemoryManager(self.config)

        # === Page 0 — Chat ===
        self.chat_widget = ChatWidget(self)
        self.chat_widget.set_database(self.database)
        self.stack.addWidget(self.chat_widget)

        # === Page 1 — Training Studio ===
        self.trainer_widget = TrainerWidget(self)
        self.trainer_widget.set_database(self.database)
        self.stack.addWidget(self.trainer_widget)

        # === Page 2 — Model Hub ===
        from src.gui.model_hub_widget import ModelHubWidget
        self.model_hub_widget = ModelHubWidget(self)
        self.stack.addWidget(self.model_hub_widget)

        # === Page 3 — Agent ===
        registry = ToolRegistry()
        project_root = Path(self.config.base_dir)
        registry.register(RepoSearchTool(project_root))
        registry.register(ReadFileTool(project_root))
        registry.register(RunTestsTool(project_root))
        registry.register(CommandTool(project_root))
        registry.register(RAGSearchTool(self.memory_manager))
        registry.register(
            TrainModelTool(training_callback=lambda _: self._switch_page(1))
        )
        self.agent_coworker = AgentCoworker(registry=registry)
        self.agent_widget = AgentWidget(self, self.agent_coworker)
        self.stack.addWidget(self.agent_widget)

        # === Page 4 — Settings ===
        self.settings_widget = SettingsWidget(self)
        self.settings_widget.set_memory_manager(self.memory_manager)
        self.stack.addWidget(self.settings_widget)

        # Auto-load model
        if self.config.get("app.auto_load_model", True):
            self.load_model()

    # -----------------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------------

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.sidebar.set_active_page(index)

        # Refresh Model Hub when navigated to
        if index == 2 and hasattr(self, "model_hub_widget"):
            self.model_hub_widget.refresh()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def load_model(self, model_name: Optional[str] = None) -> bool:
        try:
            self.status_bar.showMessage("Model yükleniyor...")
            self.sidebar.set_model_status("loading", model_name)

            if self.model_manager is None:
                self.model_manager = ModelManager(self.config)

            self._model_load_thread = ModelLoadThread(
                self.model_manager, self.config, model_name
            )
            self._model_load_thread.success.connect(self._on_model_loaded)
            self._model_load_thread.error.connect(self._on_model_load_error)
            self._model_load_thread.start()
            return True

        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            self.sidebar.set_model_status("error")
            self.status_bar.showMessage(f"Hata: {str(e)}")
            QMessageBox.critical(self, "Model Yükleme Hatası", str(e))
            return False

    def _on_model_loaded(self) -> None:
        self.inference_engine = InferenceEngine(
            self.model_manager, self.memory_manager, self.config
        )
        self.trainer = LoRATrainer(self.model_manager, self.config)
        self.model_loaded = True

        self.sidebar.set_model_status("ready", self.model_manager.model_name)
        self.status_bar.showMessage("Model hazır ✓")

        self.chat_widget.set_inference_engine(self.inference_engine)
        self.trainer_widget.set_trainer(self.trainer)
        if hasattr(self, "agent_coworker"):
            self.agent_coworker.inference_engine = self.inference_engine
        self.settings_widget.set_model_manager(self.model_manager)

        if hasattr(self, "model_hub_widget"):
            self.model_hub_widget.set_model_manager(self.model_manager)
            self.model_hub_widget.set_trainer(self.trainer)
            self.model_hub_widget.set_memory_manager(self.memory_manager)
            self.model_hub_widget.refresh()

    def _on_model_load_error(self, error_msg: str) -> None:
        # Guard: truncate very long messages to avoid further recursion in string formatting
        short_msg = error_msg[:500] if len(error_msg) > 500 else error_msg

        try:
            self.logger.error(f"Model yükleme hatası: {short_msg}")
        except Exception:  # noqa: BLE001 — logging itself must not crash
            pass

        try:
            self.sidebar.set_model_status("error")
            self.status_bar.showMessage(f"Hata: {short_msg[:120]}")
        except Exception:  # noqa: BLE001
            pass

        if "out of memory" in short_msg.lower() or "oom" in short_msg.lower():
            hint = (
                "GPU belleği yetersiz (CUDA OOM).\n\n"
                "• Model CPU üzerinde yüklenmeye çalışıldı — lütfen bekleyin.\n"
                "• Alternatif olarak config.yaml dosyasında 'model.device: cpu' ayarlayın.\n"
                "• Ya da daha küçük bir model mimarisi seçin."
            )
        elif "recursion" in short_msg.lower():
            hint = (
                "Python recursion hatası — genellikle CUDA OOM'un yan etkisidir.\n"
                "Uygulamayı yeniden başlatın ve model.device: cpu ayarlayın."
            )
        elif "FileNotFoundError" in short_msg or "bulunamadı" in short_msg.lower():
            hint = (
                "Model dosyası bulunamadı. "
                "Model Hub sayfasından yeni bir model oluşturun veya mevcut bir checkpoint seçin."
            )
        else:
            hint = "Model yüklenemedi. Checkpoint uygunluğunu kontrol edin."

        try:
            QMessageBox.warning(self, "Model Yükleme Uyarısı", f"{hint}\n\nHata: {short_msg}")
        except Exception:  # noqa: BLE001
            pass

    # -----------------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------------

    def unload_model(self) -> None:
        if self.model_manager:
            self.model_manager.unload_model()
            self.model_manager = None
            self.inference_engine = None
            self.trainer = None
            self.model_loaded = False
            self.sidebar.set_model_status("idle")

    def update_status(self, message: str) -> None:
        self.status_bar.showMessage(message)

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
