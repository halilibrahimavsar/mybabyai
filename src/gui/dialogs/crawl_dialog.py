from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QDialogButtonBox, QMessageBox
)

class CrawlURLDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.urls = []
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Web'den Veri Çek (Wikipedia & Diğer)")
        self.resize(600, 400)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Eğitilecek URL'leri girin (Her satıra bir tane):"))
        
        self.url_edit = QTextEdit()
        self.url_edit.setPlaceholderText("https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning")
        layout.addWidget(self.url_edit)

        layout.addWidget(QLabel("Not: Sadece ana metin içeriği çekilecek, reklamlar ve tablolar ayıklanacaktır."))

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def accept(self):
        text = self.url_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir URL girin.")
            return
            
        self.urls = [line.strip() for line in text.splitlines() if line.strip().startswith("http")]
        
        if not self.urls:
            QMessageBox.warning(self, "Uyarı", "Lütfen geçerli HTTP/HTTPS URL'leri girin.")
            return
            
        super().accept()
