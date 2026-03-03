from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QDialogButtonBox, QMessageBox, QSpinBox, QCheckBox,
)


class CrawlURLDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.urls: list[str] = []
        self.crawl_depth: int = 0
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Web'den Veri Çek (Wikipedia & Diğer)")
        self.resize(600, 450)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Eğitilecek URL'leri girin (Her satıra bir tane):"))

        self.url_edit = QTextEdit()
        self.url_edit.setPlaceholderText(
            "https://en.wikipedia.org/wiki/Artificial_intelligence\n"
            "https://en.wikipedia.org/wiki/Machine_learning"
        )
        layout.addWidget(self.url_edit)

        # --- Sub-crawl depth ---
        depth_row = QHBoxLayout()
        self._sub_crawl_cb = QCheckBox("Alt sayfa tarama (sub-crawl)")
        self._sub_crawl_cb.setToolTip(
            "Aktif edilirse, verilen URL'lerdeki bağlantılar da taranır."
        )
        depth_row.addWidget(self._sub_crawl_cb)

        depth_row.addWidget(QLabel("Derinlik:"))
        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(1, 3)
        self._depth_spin.setValue(1)
        self._depth_spin.setEnabled(False)
        depth_row.addWidget(self._depth_spin)
        depth_row.addStretch()
        layout.addLayout(depth_row)

        self._sub_crawl_cb.toggled.connect(self._depth_spin.setEnabled)

        layout.addWidget(
            QLabel(
                "Not: Sadece ana metin içeriği çekilecek, reklamlar ve tablolar "
                "ayıklanacaktır. Alt sayfa taramada en fazla 50 sayfa ziyaret edilir."
            )
        )

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def accept(self):
        text = self.url_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir URL girin.")
            return

        self.urls = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("http")
        ]

        if not self.urls:
            QMessageBox.warning(
                self, "Uyarı", "Lütfen geçerli HTTP/HTTPS URL'leri girin."
            )
            return

        self.crawl_depth = (
            self._depth_spin.value() if self._sub_crawl_cb.isChecked() else 0
        )
        super().accept()

    # --- Public accessors (backward-compatible) ---

    def get_url(self) -> str:
        """Return the first URL (for single-URL callers)."""
        return self.urls[0] if self.urls else ""

    def get_urls(self) -> list[str]:
        """Return all entered URLs."""
        return self.urls

    def get_crawl_depth(self) -> int:
        """0 = no sub-crawl, 1-3 = follow links at given depth."""
        return self.crawl_depth
