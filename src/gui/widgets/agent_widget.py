import sys
from datetime import datetime
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QScrollArea,
    QLabel,
    QFrame,
)

from src.core.agent import AgentCoworker
from src.core.inference import InferenceEngine

class AgentWorkerThread(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, coworker: AgentCoworker, query: str):
        super().__init__()
        self.coworker = coworker
        self.query = query
        
    def run(self):
        try:
            result = self.coworker.run(self.query)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"error": str(e), "query": self.query})

class AgentMessageBubble(QFrame):
    def __init__(self, text: str, is_user: bool = True, is_tool: bool = False, is_thought: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.is_tool = is_tool
        self.is_thought = is_thought
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        
        container = QFrame()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 10, 12, 10)
        
        if is_user:
            container.setStyleSheet("""
                QFrame {
                    background-color: #2563eb;
                    border-radius: 12px;
                }
            """)
            layout.addStretch()
            layout.addWidget(container)
        elif is_tool:
            container.setStyleSheet("""
                QFrame {
                    background-color: #374151;
                    border: 1px solid #4b5563;
                    border-radius: 8px;
                }
            """)
            layout.addWidget(container)
            layout.addStretch()
        elif self.is_thought:
            container.setStyleSheet("""
                QFrame {
                    background-color: #3b186e;
                    border: 1px solid #581c87;
                    border-radius: 12px;
                }
            """)
            layout.addWidget(container)
            layout.addStretch()
        else:
            container.setStyleSheet("""
                QFrame {
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 12px;
                }
            """)
            layout.addWidget(container)
            layout.addStretch()
            
        title_text = "Siz"
        if not is_user:
            if is_tool:
                title_text = "🔧 Agent Aracı"
            elif self.is_thought:
                title_text = "🧠 MCTS Düşünce Zinciri"
            else:
                title_text = "🤖 Agent"

        header = QLabel(title_text)
        header.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        container_layout.addWidget(header)
        
        msg_label = QLabel(text)
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_label.setStyleSheet("color: white; font-size: 13px;")
        container_layout.addWidget(msg_label)

class AgentWidget(QWidget):
    def __init__(self, main_window, agent_coworker: AgentCoworker, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.agent = agent_coworker
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        header = QFrame()
        header.setStyleSheet("background-color: #1e293b; border-bottom: 1px solid #334155;")
        header_layout = QHBoxLayout(header)
        title = QLabel("🤖 AI Agent (Otonom Görevler)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        layout.addWidget(header)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #0f172a; }")
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: #0f172a;")
        self.chat_layout = QVBoxLayout(self.scroll_content)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(10)
        
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)
        
        input_container = QFrame()
        input_container.setStyleSheet("background-color: #1e293b; border-top: 1px solid #334155;")
        input_layout = QHBoxLayout(input_container)
        
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Agent'a bir görev verin (örn: modelin testlerini çalıştır, memory'de X'i ara)...")
        self.input_edit.setFixedHeight(60)
        self.input_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: white;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        input_layout.addWidget(self.input_edit)
        
        self.send_btn = QPushButton("Gönder")
        self.send_btn.clicked.connect(self._on_send)
        self.send_btn.setFixedSize(80, 40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #1e3a8a; color: #94a3b8; }
        """)
        input_layout.addWidget(self.send_btn)
        
        layout.addWidget(input_container)

    def _on_send(self):
        query = self.input_edit.toPlainText().strip()
        if not query:
            return
            
        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self._add_message(query, is_user=True)
        
        self.worker = AgentWorkerThread(self.agent, query)
        self.worker.finished.connect(self._on_agent_finished)
        self.worker.start()
        
    def _on_agent_finished(self, result: dict):
        self.send_btn.setEnabled(True)
        if "error" in result:
            self._add_message(f"Hata: {result['error']}", is_user=False)
            return
            
        planned = result.get("planned_calls", [])
        trace = result.get("trace", [])
        thought_process = result.get("thought_process", "")
        
        if thought_process:
            self._add_message(thought_process, is_user=False, is_thought=True)
        
        if not planned:
            self._add_message("Agent bu görev için plan yapamadı veya çalıştıracak araç bulamadı.", is_user=False)
            return

        for call in trace:
            tool = call["tool_name"]
            ok = "Başarılı" if call["ok"] else "Başarısız"
            output = call["result"]
            err = call.get("error", "")
            
            msg = f"Araç: {tool}\nSonuç: {ok}"
            if output:
                # truncate long output
                if len(output) > 500:
                    output = output[:500] + "... (devamı kesildi)"
                msg += f"\nÇıktı:\n{output}"
            if err:
                msg += f"\nHata:\n{err}"
                
            self._add_message(msg, is_user=False, is_tool=True)
            
        success = result.get("success", False)
        status_msg = "Görev başarıyla tamamlandı." if success else "Görev tamamlanamadı veya bazı araçlar hata verdi."
        self._add_message(status_msg, is_user=False)

    def _add_message(self, text: str, is_user: bool, is_tool: bool = False, is_thought: bool = False):
        bubble = AgentMessageBubble(text, is_user, is_tool, is_thought)
        self.chat_layout.addWidget(bubble)
        # Scroll to bottom
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
