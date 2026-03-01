import os
from typing import List, Dict
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QLineEdit, 
    QSpinBox, QDialogButtonBox, QMessageBox
)

from src.data.dataset_downloader import DatasetDownloader

class DownloadThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, downloader: DatasetDownloader, dataset_key: str, max_samples: int):
        super().__init__()
        self.downloader = downloader
        self.dataset_key = dataset_key
        self.max_samples = max_samples

    def run(self):
        try:
            if self.dataset_key.startswith(("http://", "https://")):
                # Direct URL download and extraction
                path = self.downloader.download_url_and_extract(self.dataset_key)
                
                # Load the data from the extracted path
                from src.data.dataset_loader import DatasetLoader
                loader = DatasetLoader()
                
                if os.path.isdir(path):
                    data = loader.load_from_directory(path)
                else:
                    data = loader.load_from_file(path)
            elif self.dataset_key.startswith("custom:"):
                hf_id = self.dataset_key.replace("custom:", "").strip()
                data = self.downloader.download_custom(hf_id, max_samples=self.max_samples)
            else:
                data = self.downloader.download_dataset(self.dataset_key, max_samples=self.max_samples)
            
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class DownloadDatasetDialog(QDialog):
    def __init__(self, downloader: DatasetDownloader, parent=None):
        super().__init__(parent)
        self.downloader = downloader
        self.selected_key = None
        self.max_samples = 10000
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("İnternetten Veriseti İndir")
        self.resize(500, 300)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("1. Bir Hazır Veriseti Seçin veya Özel Girmek İçin 'Özel (HuggingFace)' Seçin:"))
        
        self.dataset_combo = QComboBox()
        self.datasets = self.downloader.list_available_datasets()
        
        self.dataset_combo.addItem("--- Hazır Verisetleri ---", userData="")
        for key, info in self.datasets.items():
            display_name = f"{info['name']} ({info.get('size', '')}) - {', '.join(info.get('languages', []))}"
            self.dataset_combo.addItem(display_name, userData=key)
            
        self.dataset_combo.addItem("--- Özel (HuggingFace ID) ---", userData="custom")
        layout.addWidget(self.dataset_combo)

        self.custom_hf_input = QLineEdit()
        self.custom_hf_input.setPlaceholderText("Örn: tatsu-lab/alpaca")
        self.custom_hf_input.setEnabled(False)
        layout.addWidget(self.custom_hf_input)
        
        self.dataset_combo.currentIndexChanged.connect(self._on_combo_changed)

        layout.addWidget(QLabel("2. Maksimum Örnek Sayısı (Çok yüksek değerler indirmeyi uzatır):"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 5000000)
        self.samples_spin.setSingleStep(1000)
        self.samples_spin.setValue(10000)
        layout.addWidget(self.samples_spin)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_combo_changed(self):
        user_data = self.dataset_combo.currentData()
        if user_data == "custom":
            self.custom_hf_input.setEnabled(True)
        else:
            self.custom_hf_input.setEnabled(False)
            self.custom_hf_input.clear()

    def accept(self):
        user_data = self.dataset_combo.currentData()
        if not user_data:
            QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir seçim yapın.")
            return
            
        if user_data == "custom":
            hf_id = self.custom_hf_input.text().strip()
            if not hf_id:
                QMessageBox.warning(self, "Uyarı", "Lütfen bir HuggingFace ID girin.")
                return
            self.selected_key = f"custom:{hf_id}"
        else:
            self.selected_key = user_data
            
        self.max_samples = self.samples_spin.value()
        super().accept()
