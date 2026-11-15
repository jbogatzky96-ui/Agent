"""Always-on-top overlay window (PyQt6). The engine runs in another thread and talks to it via Qt signals."""
from __future__ import annotations

from html import escape

from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import UIConfig
from .llm import AnswerStats

STATUS_COLORS = {"listening": "#3fb950", "answering": "#58a6ff", "error": "#f85149"}

STYLE = """
#panel { background: rgba(22, 24, 29, 238); border: 1px solid rgba(255,255,255,30); border-radius: 12px; }
QLabel { color: #e6edf3; }
#title { color: #8b949e; font-weight: 600; }
#hint, #meta, #live { color: #8b949e; }
#question { color: #ffd479; font-weight: 600; }
QTextBrowser { background: transparent; border: none; color: #e6edf3; }
QPushButton#close { background: transparent; color: #8b949e; border: none; font-size: 15px; padding: 0 4px; }
QPushButton#close:hover { color: #f85149; }
"""


class QtSink(QObject):
    """Implements engine.UiSink. Signals emitted from the engine thread are queued onto the GUI thread."""

    status_sig = pyqtSignal(str, str)
    question_sig = pyqtSignal(str, str)
    delta_sig = pyqtSignal(str)
    done_sig = pyqtSignal(object, str)
    utterance_sig = pyqtSignal(str, str)
    partial_sig = pyqtSignal(str, str)
    toggle_sig = pyqtSignal()

    def status(self, state: str, detail: str = "") -> None:
        self.status_sig.emit(state, detail)

    def question(self, text: str, trigger: str) -> None:
        self.question_sig.emit(text, trigger)

    def answer_delta(self, text: str) -> None:
        self.delta_sig.emit(text)

    def answer_done(self, stats: AnswerStats | None, error: str = "") -> None:
        self.done_sig.emit(stats, error)

    def utterance(self, speaker: str, text: str) -> None:
        self.utterance_sig.emit(speaker, text)

    def partial(self, speaker: str, text: str) -> None:
        self.partial_sig.emit(speaker, text)


class Overlay(QWidget):
    def __init__(self, cfg: UIConfig, sink: QtSink) -> None:
        super().__init__()
        self.cfg = cfg
        self._answer = ""
        self._drag_from: QPoint | None = None
        self._recent: list[str] = []

        self.setWindowTitle("Call Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(cfg.opacity)
        self.resize(cfg.width, cfg.height)
        font = QFont("Segoe UI", cfg.font_size)
        self.setFont(font)
        self.setStyleSheet(STYLE)

        panel = QFrame(objectName="panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.dot = QLabel("●")
        self.title = QLabel("Call Assistant", objectName="title")
        self.state_label = QLabel("", objectName="hint")
        close = QPushButton("×", objectName="close")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(QApplication.instance().quit)
        header.addWidget(self.dot)
        header.addWidget(self.title)
        header.addWidget(self.state_label, 1)
        header.addWidget(close)
        layout.addLayout(header)

        self.question_label = QLabel(
            f"Warte auf eine Frage … ({cfg.hotkey_answer.upper()} = jetzt antworten)", objectName="question"
        )
        self.question_label.setWordWrap(True)
        layout.addWidget(self.question_label)

        self.answer_view = QTextBrowser()
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.answer_view, 1)

        self.live_label = QLabel("", objectName="live")
        self.live_label.setWordWrap(True)
        self.live_label.setMaximumHeight(int(cfg.font_size * 4.2))
        layout.addWidget(self.live_label)

        footer = QHBoxLayout()
        self.meta_label = QLabel(
            f"{cfg.hotkey_toggle.upper()} ein-/ausblenden · Fenster ziehen zum Verschieben", objectName="meta"
        )
        footer.addWidget(self.meta_label, 1)
        footer.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        layout.addLayout(footer)

        sink.status_sig.connect(self.on_status)
        sink.question_sig.connect(self.on_question)
        sink.delta_sig.connect(self.on_delta)
        sink.done_sig.connect(self.on_done)
        sink.utterance_sig.connect(self.on_utterance)
        sink.partial_sig.connect(self.on_partial)
        sink.toggle_sig.connect(lambda: self.setVisible(not self.isVisible()))
        self.on_status("listening", "Starte …")

    # ---- slots ---------------------------------------------------------------------------------

    def on_status(self, state: str, detail: str) -> None:
        self.dot.setStyleSheet(f"color: {STATUS_COLORS.get(state, '#8b949e')};")
        self.state_label.setText(detail)

    def on_question(self, text: str, trigger: str) -> None:
        prefix = "Frage" if trigger == "auto" else "Hotkey"
        self.question_label.setText(f"{prefix}: {text}")
        self._answer = ""
        self.answer_view.setMarkdown("")

    def on_delta(self, text: str) -> None:
        self._answer += text
        self.answer_view.setMarkdown(self._answer)

    def on_done(self, stats: AnswerStats | None, error: str) -> None:
        if error:
            self.on_status("error", "Fehler")
            self.answer_view.setMarkdown(f"**{error}**")
            return
        if stats is None:
            return
        if stats.refused:
            self.answer_view.setMarkdown(self._answer + "\n\n*(Anfrage vom Modell abgelehnt)*")
        first = f"{stats.first_token_s:.1f}s" if stats.first_token_s is not None else "–"
        self.meta_label.setText(f"{stats.model} · erste Worte nach {first} · {stats.output_tokens} Tokens")

    def on_utterance(self, speaker: str, text: str) -> None:
        self._recent = (self._recent + [f"<b>{escape(speaker)}:</b> {escape(text)}"])[-2:]
        self.live_label.setText("<br>".join(self._recent))

    def on_partial(self, speaker: str, text: str) -> None:
        lines = self._recent[-1:] + [f"<b>{escape(speaker)}:</b> <i>{escape(text)}</i>"]
        self.live_label.setText("<br>".join(lines))

    # ---- dragging ------------------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
