import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QPoint, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


from src.core.inference import InferenceEngine
from src.data.database import Conversation, Database


class ChatInputTextEdit(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event) -> None:
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MessageBubble(QFrame):
    def __init__(self, text: str, is_user: bool = True, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.timestamp = datetime.now()
        self._setup_ui(text)

    def _setup_ui(self, text: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        role_label = QLabel("Sen" if self.is_user else "MyBabyAI")
        role_label.setStyleSheet(
            f"""
            QLabel {{
                font-size: 11px;
                font-weight: 700;
                color: {"#60a5fa" if self.is_user else "#4ade80"};
            }}
            """
        )
        header_layout.addWidget(role_label)

        header_layout.addStretch()

        self.time_label = QLabel(self.timestamp.strftime("%H:%M"))
        self.time_label.setStyleSheet("QLabel { font-size: 11px; color: #9ca3af; }")
        header_layout.addWidget(self.time_label)

        layout.addLayout(header_layout)

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_label.setStyleSheet(
            "QLabel { font-size: 13px; color: #e5e7eb; line-height: 1.45; }"
        )
        layout.addWidget(self.text_label)

        bg_color = "#28375f" if self.is_user else "#243830"
        border_color = "#3b82f6" if self.is_user else "#22c55e"
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            """
        )

    def set_text(self, text: str) -> None:
        self.text_label.setText(text)


class GenerateThread(QThread):
    text_generated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    mode_selected = pyqtSignal(str, int)

    def __init__(
        self,
        inference_engine: InferenceEngine,
        user_input: str,
        history: List[Dict[str, str]],
        stream: bool = True,
    ):
        super().__init__()
        self.inference_engine = inference_engine
        self.user_input = user_input
        self.history = history
        self.stream = stream

    def run(self) -> None:
        try:
            def mode_callback(mode: str, simulations: int):
                self.mode_selected.emit(mode, simulations)

            if self.stream:
                full_response = ""
                for text in self.inference_engine.generate_stream(
                    self.user_input, 
                    use_memory=True, 
                    history=self.history,
                    mode_callback=mode_callback,
                ):
                    if self.isInterruptionRequested():
                        break
                    full_response += text
                    self.text_generated.emit(text)
                self.finished.emit(full_response)
            else:
                response = self.inference_engine.generate(
                    self.user_input, 
                    use_memory=True, 
                    history=self.history,
                    mode_callback=mode_callback,
                )
                self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))


class ChatWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.inference_engine: Optional[InferenceEngine] = None
        self.database: Optional[Database] = None

        self.current_conversation: Optional[Conversation] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.generate_thread: Optional[GenerateThread] = None
        self.current_bubble: Optional[MessageBubble] = None
        self.current_response = ""
        self.pending_user_input = ""
        self.is_generating = False

        # Cognitive mode override (None = auto-routing)
        self._COGNITIVE_MODES = [
            ("auto",              "Oto ✨",               "#1e3a5f", "#93c5fd"),
            ("system_1",          "System 1 ⚡",           "#1e3a5f", "#93c5fd"),
            ("system_2_plan",     "Plan 📝",              "#3b186e", "#d8b4fe"),
            ("system_2_deepthink","DeepThink 🧠",         "#4a1942", "#f0abfc"),
            ("system_2_agent",    "Agent 🤖",             "#164e28", "#86efac"),
        ]
        self._cog_mode_index = 0  # starts on Auto

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: #30384d;
                width: 1px;
            }
            """
        )

        sidebar = self._create_sidebar()
        splitter.addWidget(sidebar)

        chat_area = self._create_chat_area()
        splitter.addWidget(chat_area)

        splitter.setSizes([260, 940])
        layout.addWidget(splitter)
        self._update_send_button_state()

    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar_width = self.main_window.config.get("gui.sidebar_width", 260)
        sidebar.setFixedWidth(max(220, min(340, int(sidebar_width))))
        sidebar.setStyleSheet("background-color: #181e2c;")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        sidebar_title = QLabel("Sohbet Geçmişi")
        sidebar_title.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: 600; color: #cdd6f4; }"
        )
        layout.addWidget(sidebar_title)

        new_chat_btn = QPushButton("Yeni Sohbet")
        new_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_chat_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2563eb;
                border: 1px solid #3b82f6;
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 13px;
                font-weight: 600;
                color: #f8fafc;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            """
        )
        new_chat_btn.clicked.connect(self._new_conversation)
        layout.addWidget(new_chat_btn)

        self.conversation_list = QListWidget()
        self.conversation_list.setStyleSheet(
            """
            QListWidget {
                background-color: #111827;
                border: 1px solid #2f3a4f;
                border-radius: 10px;
                padding: 2px;
            }
            QListWidget::item {
                border: 1px solid transparent;
                border-radius: 8px;
                margin: 2px 0;
                padding: 10px 9px;
                color: #d1d5db;
            }
            QListWidget::item:hover {
                background-color: #1f2937;
                border-color: #334155;
            }
            QListWidget::item:selected {
                background-color: #24364d;
                border-color: #3b82f6;
            }
            """
        )
        self.conversation_list.itemClicked.connect(self._load_conversation)
        self.conversation_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.conversation_list.customContextMenuRequested.connect(
            self._show_context_menu
        )
        layout.addWidget(self.conversation_list, 1)

        self.export_btn = QPushButton("Sohbeti Dışa Aktar")
        self.export_btn.setEnabled(False)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 9px;
                color: #cbd5e1;
            }
            QPushButton:hover:enabled {
                background-color: #1e293b;
                border-color: #475569;
            }
            QPushButton:disabled {
                border-color: #263041;
                color: #64748b;
            }
            """
        )
        self.export_btn.clicked.connect(self._export_conversation)
        layout.addWidget(self.export_btn)

        shortcuts_label = QLabel("Enter: Gönder | Shift+Enter: Yeni satır")
        shortcuts_label.setStyleSheet("QLabel { font-size: 11px; color: #94a3b8; }")
        layout.addWidget(shortcuts_label)

        return sidebar

    def _create_chat_area(self) -> QWidget:
        chat_widget = QWidget()
        layout = QVBoxLayout(chat_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            """
            QFrame {
                background-color: #151c2b;
                border-bottom: 1px solid #2f3a4f;
            }
            """
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12)
        header_layout.setSpacing(2)

        self.chat_title_label = QLabel("Yeni Sohbet")
        self.chat_title_label.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: 700; color: #e5e7eb; }"
        )
        header_layout.addWidget(self.chat_title_label)

        # Second row: subtitle + cognitive mode badge
        subtitle_row = QHBoxLayout()
        subtitle_row.setContentsMargins(0, 0, 0, 0)

        self.chat_subtitle_label = QLabel("Bir mesaj yazarak sohbeti başlatın.")
        self.chat_subtitle_label.setStyleSheet(
            "QLabel { font-size: 12px; color: #94a3b8; }"
        )
        subtitle_row.addWidget(self.chat_subtitle_label)
        subtitle_row.addStretch()

        # Cognitive mode badge (clickable to cycle modes)
        self._cog_badge = QPushButton("Oto ✨")
        self._cog_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cog_badge.setStyleSheet(
            """
            QPushButton {
                background: #1e3a5f;
                color: #93c5fd;
                border: none;
                border-radius: 9px;
                padding: 2px 9px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #264a73;
            }
            """
        )
        self._cog_badge.setToolTip(
            "Bilişsel mod seçici (tıkla değiştir):\n"
            "Oto ✨ — Otomatik yönlendirme\n"
            "System 1 ⚡ — Hızlı sezgisel yanıt\n"
            "Plan 📝 — Adım adım düşünme\n"
            "DeepThink 🧠 — Derin MCTS muhakemesi\n"
            "Agent 🤖 — Otonom ajan modu"
        )
        self._cog_badge.clicked.connect(self._cycle_cognitive_mode)
        subtitle_row.addWidget(self._cog_badge)

        header_layout.addLayout(subtitle_row)
        layout.addWidget(header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #0f172a;
            }
            """
        )

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_layout.setContentsMargins(0, 14, 0, 14)
        self.messages_layout.setSpacing(8)

        self.empty_state = self._create_empty_state()
        self.messages_layout.addWidget(self.empty_state)

        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area, 1)

        input_area = self._create_input_area()
        layout.addWidget(input_area)

        return chat_widget

    def _create_empty_state(self) -> QFrame:
        empty = QFrame()
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(40, 30, 40, 30)
        empty_layout.setSpacing(8)

        title = QLabel("Sohbete Hazır")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "QLabel { font-size: 18px; font-weight: 700; color: #e2e8f0; }"
        )
        empty_layout.addWidget(title)

        subtitle = QLabel(
            "Modele bir soru sorun veya bir görev verin.\nYanıtlar burada mesaj balonları şeklinde görünecek."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("QLabel { font-size: 12px; color: #94a3b8; }")
        subtitle.setWordWrap(True)
        empty_layout.addWidget(subtitle)

        empty.setStyleSheet(
            """
            QFrame {
                background-color: #111827;
                border: 1px dashed #334155;
                border-radius: 12px;
                margin: 12px 20px;
            }
            """
        )
        return empty

    def _create_input_area(self) -> QWidget:
        input_widget = QWidget()
        input_widget.setStyleSheet(
            """
            QWidget {
                background-color: #151c2b;
                border-top: 1px solid #2f3a4f;
            }
            """
        )

        layout = QVBoxLayout(input_widget)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(8)

        self.input_field = ChatInputTextEdit()
        self.input_field.setAcceptRichText(False)
        self.input_field.setPlaceholderText("Mesajınızı yazın...")
        self.input_field.setFixedHeight(46)
        self.input_field.send_requested.connect(self._send_message)
        self.input_field.textChanged.connect(self._on_input_changed)
        self.input_field.setStyleSheet(
            """
            QTextEdit {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 13px;
                color: #e5e7eb;
            }
            QTextEdit:focus {
                border-color: #3b82f6;
            }
            """
        )
        layout.addWidget(self.input_field)

        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        helper_label = QLabel("Enter ile gönder, Shift+Enter ile yeni satır.")
        helper_label.setStyleSheet("QLabel { font-size: 11px; color: #94a3b8; }")
        actions_layout.addWidget(helper_label)
        actions_layout.addStretch()

        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setVisible(False)
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #b91c1c;
                border: 1px solid #ef4444;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
                color: #fee2e2;
            }
            QPushButton:hover {
                background-color: #991b1b;
            }
            """
        )
        actions_layout.addWidget(self.stop_btn)

        self.send_btn = QPushButton("Gönder")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2563eb;
                border: 1px solid #3b82f6;
                border-radius: 8px;
                padding: 8px 18px;
                font-weight: 700;
                color: #f8fafc;
            }
            QPushButton:hover:enabled {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #1f2937;
                border-color: #334155;
                color: #64748b;
            }
            """
        )
        actions_layout.addWidget(self.send_btn)

        layout.addLayout(actions_layout)
        return input_widget

    def _on_input_changed(self) -> None:
        self._resize_input_box()
        self._update_send_button_state()

    def _resize_input_box(self) -> None:
        content_height = int(self.input_field.document().size().height()) + 16
        self.input_field.setFixedHeight(max(46, min(130, content_height)))

    def _update_send_button_state(self) -> None:
        has_text = bool(self.input_field.toPlainText().strip())
        self.send_btn.setEnabled(has_text and not self.is_generating)

    def _set_empty_state_visible(self, is_visible: bool) -> None:
        self.empty_state.setVisible(is_visible)

    def _update_conversation_header(self) -> None:
        if self.current_conversation:
            self.chat_title_label.setText(self.current_conversation.title)
            turn_count = len(self.conversation_history)
            self.chat_subtitle_label.setText(
                f"{turn_count} tur konuşma geçmişi yüklendi."
            )
        else:
            self.chat_title_label.setText("Yeni Sohbet")
            self.chat_subtitle_label.setText("Bir mesaj yazarak sohbeti başlatın.")

        self.export_btn.setEnabled(self.current_conversation is not None)

    def _add_message_bubble(self, text: str, is_user: bool) -> MessageBubble:
        self._set_empty_state_visible(False)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(14, 2, 14, 2)
        row_layout.setSpacing(8)

        bubble = MessageBubble(text, is_user=is_user)
        bubble.setMaximumWidth(760)

        if is_user:
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch()

        self.messages_layout.addWidget(row)
        return bubble

    def set_inference_engine(self, engine: InferenceEngine) -> None:
        self.inference_engine = engine

    def set_cognitive_mode(self, mode: str, simulations: int = 0) -> None:
        """Update the cognitive mode badge to show which mode is active.

        Called by the GenerateThread's mode_selected signal.

        Args:
            mode: 'system_1' | 'system_2_plan' | 'system_2_deepthink' | 'system_2_agent'
            simulations: Number of MCTS simulations (only shown for System 2)
        """
        mode_labels = {
            "system_1":           ("System 1 ⚡",      "#1e3a5f", "#93c5fd"),
            "system_2_plan":      (f"Plan 📝 {simulations} sim.",    "#3b186e", "#d8b4fe"),
            "system_2_deepthink": (f"DeepThink 🧠 {simulations} sim.", "#4a1942", "#f0abfc"),
            "system_2_agent":     (f"Agent 🤖 {simulations} sim.",     "#164e28", "#86efac"),
        }
        label, bg, fg = mode_labels.get(mode, ("System 1 ⚡", "#1e3a5f", "#93c5fd"))
        self._cog_badge.setText(label)
        self._cog_badge.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: none;
                border-radius: 9px;
                padding: 2px 9px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {bg};
                filter: brightness(1.2);
            }}
            """
        )

    def _cycle_cognitive_mode(self) -> None:
        """Cycle through cognitive modes on badge click."""
        self._cog_mode_index = (self._cog_mode_index + 1) % len(self._COGNITIVE_MODES)
        mode_key, label, bg, fg = self._COGNITIVE_MODES[self._cog_mode_index]

        self._cog_badge.setText(label)
        self._cog_badge.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: none;
                border-radius: 9px;
                padding: 2px 9px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {bg};
            }}
            """
        )

        # Apply to inference engine router
        if self.inference_engine:
            if mode_key == "auto":
                self.inference_engine.router.force_mode = None
            else:
                from src.core.cognitive.modes import CognitiveMode
                mode_enum = CognitiveMode(mode_key)
                self.inference_engine.router.force_mode = mode_enum

    def set_database(self, database: Database) -> None:
        self.database = database
        self._load_conversations_list()

    def _new_conversation(self) -> None:
        if not self.database:
            return

        self.current_conversation = self.database.create_conversation()
        self.conversation_history = []
        self.pending_user_input = ""
        self._clear_messages()
        self._load_conversations_list()
        self._update_conversation_header()
        self.input_field.setFocus()

    def _load_conversation(self, item: QListWidgetItem) -> None:
        if not self.database:
            return

        conv_id = item.data(Qt.ItemDataRole.UserRole)
        self.current_conversation = self.database.get_conversation(conv_id)

        if not self.current_conversation:
            return

        self._clear_messages()
        messages = self.database.get_messages(conv_id)
        self.conversation_history = []

        for msg in messages:
            self._add_message_bubble(msg.content, msg.role == "user")
            if msg.role == "user":
                self.conversation_history.append({"user": msg.content, "assistant": ""})
            elif self.conversation_history:
                self.conversation_history[-1]["assistant"] = msg.content

        self._set_empty_state_visible(len(messages) == 0)
        self._update_conversation_header()
        self._scroll_to_bottom()

    def _load_conversations_list(self) -> None:
        if not self.database:
            return

        selected_id = self.current_conversation.id if self.current_conversation else None

        self.conversation_list.clear()
        for conv in self.database.get_all_conversations():
            title = (conv.title or "Yeni Sohbet").strip()
            short_title = title if len(title) <= 32 else f"{title[:29]}..."
            updated_at = (
                conv.updated_at.strftime("%d.%m %H:%M") if conv.updated_at else ""
            )
            item = QListWidgetItem(f"{short_title}\n{updated_at}")
            item.setToolTip(title)
            item.setData(Qt.ItemDataRole.UserRole, conv.id)
            self.conversation_list.addItem(item)
            if selected_id == conv.id:
                self.conversation_list.setCurrentItem(item)

    def _send_message(self) -> None:
        if self.is_generating:
            return

        text = self.input_field.toPlainText().strip()
        if not text:
            return

        if not self.inference_engine:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir model yükleyin!")
            return

        if not self.current_conversation and self.database:
            self.current_conversation = self.database.create_conversation(text[:50])
            self._load_conversations_list()
            self._update_conversation_header()

        self._add_message_bubble(text, is_user=True)

        if self.database and self.current_conversation:
            self.database.add_message(self.current_conversation.id, "user", text)

        self.pending_user_input = text
        self.input_field.clear()
        self._on_input_changed()
        self._scroll_to_bottom()
        self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        self.is_generating = True
        self.stop_btn.setVisible(True)
        self.input_field.setEnabled(False)
        self._update_send_button_state()

        self.current_response = ""
        self.current_bubble = self._add_message_bubble("Yanıt hazırlanıyor...", is_user=False)

        self.generate_thread = GenerateThread(
            self.inference_engine, text, self.conversation_history, stream=True
        )
        self.generate_thread.text_generated.connect(self._on_text_generated)
        self.generate_thread.finished.connect(self._on_generation_finished)
        self.generate_thread.error.connect(self._on_generation_error)
        self.generate_thread.mode_selected.connect(self._on_mode_selected)
        self.generate_thread.start()

    def _on_mode_selected(self, mode: str, simulations: int) -> None:
        self.set_cognitive_mode(mode, simulations)

    def _on_text_generated(self, text: str) -> None:
        self.current_response += text
        if not self.current_bubble:
            self.current_bubble = self._add_message_bubble("", is_user=False)
        self.current_bubble.set_text(self.current_response or "...")
        self._scroll_to_bottom()

    def _on_generation_finished(self, response: str) -> None:
        response = response.strip()
        if self.current_bubble:
            self.current_bubble.set_text(response or "(Boş yanıt)")

        if self.database and self.current_conversation:
            self.database.add_message(self.current_conversation.id, "assistant", response)

        if self.pending_user_input:
            self.conversation_history.append(
                {"user": self.pending_user_input, "assistant": response}
            )
            self.pending_user_input = ""

        self._load_conversations_list()
        self._update_conversation_header()
        self._finish_generation()

    def _on_generation_error(self, error: str) -> None:
        if self.current_bubble:
            self.current_bubble.set_text("Yanıt üretilirken bir hata oluştu.")

        self.pending_user_input = ""
        self._finish_generation()
        QMessageBox.critical(self, "Hata", f"Üretim hatası: {error}")

    def _finish_generation(self) -> None:
        self.is_generating = False
        self.stop_btn.setVisible(False)
        self.input_field.setEnabled(True)
        self.generate_thread = None
        self.current_bubble = None
        self._update_send_button_state()
        self.input_field.setFocus()

    def _stop_generation(self) -> None:
        if self.generate_thread and self.generate_thread.isRunning():
            self.generate_thread.requestInterruption()
            if not self.generate_thread.wait(800):
                self.generate_thread.terminate()
                self.generate_thread.wait()

        if self.current_bubble and not self.current_response:
            self.current_bubble.set_text("Yanıt durduruldu.")

        if (
            self.current_response.strip()
            and self.database
            and self.current_conversation
            and self.pending_user_input
        ):
            response = self.current_response.strip()
            self.database.add_message(self.current_conversation.id, "assistant", response)
            self.conversation_history.append(
                {"user": self.pending_user_input, "assistant": response}
            )
            self.pending_user_input = ""
            self._load_conversations_list()
            self._update_conversation_header()

        self.pending_user_input = ""
        self._finish_generation()

    def _clear_messages(self) -> None:
        while self.messages_layout.count():
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.empty_state = self._create_empty_state()
        self.messages_layout.addWidget(self.empty_state)
        self._set_empty_state_visible(True)

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            60,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

    def _export_conversation(self) -> None:
        if not self.current_conversation or not self.database:
            QMessageBox.information(self, "Bilgi", "Dışa aktarılacak sohbet yok.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Sohbeti Dışa Aktar",
            f"conversation_{self.current_conversation.id}.json",
            "JSON Files (*.json);;Text Files (*.txt)",
        )

        if not filepath:
            return

        format_type = "json" if filepath.endswith(".json") else "txt"
        content = self.database.export_conversation(
            self.current_conversation.id, format_type
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        QMessageBox.information(self, "Başarılı", "Sohbet dışa aktarıldı.")

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.conversation_list.itemAt(position)
        if not item:
            return

        menu = QMenu(self)
        delete_action = QAction("Sil", self)
        delete_action.triggered.connect(lambda: self._delete_conversation(item))
        menu.addAction(delete_action)
        menu.exec(self.conversation_list.mapToGlobal(position))

    def _delete_conversation(self, item: QListWidgetItem) -> None:
        if not self.database:
            return

        conv_id = item.data(Qt.ItemDataRole.UserRole)
        conv_title = item.text().split("\n")[0]

        reply = QMessageBox.question(
            self,
            "Sohbeti Sil",
            f"'{conv_title}' sohbetini silmek istediğinize emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        if not self.database.delete_conversation(conv_id):
            QMessageBox.critical(self, "Hata", "Sohbet silinemedi.")
            return

        if self.current_conversation and self.current_conversation.id == conv_id:
            self.current_conversation = None
            self.conversation_history = []
            self.pending_user_input = ""
            self._clear_messages()
            self._update_conversation_header()

        self._load_conversations_list()
