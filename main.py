#!/usr/bin/env python3
"""
MyBabyAI - Eğitilebilir AI Masaüstü Uygulaması

Bu uygulama, az veriyle hızlı öğrenme kabiliyetine sahip
bir AI asistanıdır. LoRA teknolojisi kullanılarak kişiselleştirilebilir.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"


def check_dependencies() -> bool:
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")

    if missing:
        print("=" * 50)
        print("Eksik bağımlılıklar tespit edildi!")
        print("=" * 50)
        print(f"Eksik paketler: {', '.join(missing)}")
        print("\nYüklemek için şu komutu çalıştırın:")
        print(f"  pip install {' '.join(missing)}")
        print("\nveya tüm bağımlılıkları yükleyin:")
        print("  pip install -r requirements.txt")
        print("=" * 50)
        return False

    return True


def main():
    if not check_dependencies():
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont

    from src.utils.config import Config
    from src.utils.logger import setup_logger
    from src.gui.main_window import MainWindow

    config = Config()

    if not config.validate():
        print("Yapılandırma doğrulama hatası. Lütfen config.yaml dosyasını kontrol edin.")
        sys.exit(1)

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = setup_logger("mybabyai", log_file=str(log_dir / "app.log"))

    logger.info(f"{config.app_name} v{config.app_version} başlatılıyor...")

    app = QApplication(sys.argv)
    app.setApplicationName(config.app_name)
    app.setApplicationVersion(config.app_version)

    font_family = config.get("gui.font_family", "Segoe UI")
    font_size = config.get("gui.font_size", 11)
    app.setFont(QFont(font_family, font_size))

    app.setStyle("Fusion")

    dark_palette = """
    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    QMenuBar {
        background-color: #252525;
    }
    QMenuBar::item:selected {
        background-color: #3d3d3d;
    }
    QMenu {
        background-color: #252525;
        border: 1px solid #3d3d3d;
    }
    QMenu::item:selected {
        background-color: #6366f1;
    }
    QScrollBar:vertical {
        background-color: #252525;
        width: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background-color: #4d4d4d;
        border-radius: 6px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #6366f1;
    }
    QScrollBar:horizontal {
        background-color: #252525;
        height: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal {
        background-color: #4d4d4d;
        border-radius: 6px;
        min-width: 20px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #6366f1;
    }
    QToolTip {
        background-color: #3d3d3d;
        color: #e0e0e0;
        border: 1px solid #6366f1;
        border-radius: 4px;
        padding: 5px;
    }
    """
    app.setStyleSheet(app.styleSheet() + dark_palette)

    window = MainWindow()
    window.show()

    logger.info("Uygulama başlatıldı")

    exit_code = app.exec()

    logger.info(f"Uygulama kapatılıyor (exit code: {exit_code})")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
