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
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import UIConfig
from .llm import AnswerStats

STATUS_COLORS = {"listening": "#3fb950", "answering": "#58a6ff", "error": "#f85149"}
SPEAKER_COLORS = ["#79c0ff", "#d2a8ff"]
MAX_TRANSCRIPT_LINES = 300

STYLE = """
#panel { background: rgba(22, 24, 29, 238); border: 1px solid rgba(255,255,255,30); border-radius: 12px; }
QLabel { color: #e6edf3; }
#title { color: #8b949e; font-weight: 600; }
#hint, #meta, #section { color: #8b949e; }
#section { font-size: 9pt; text-transform: uppercase; }
#question { color: #ffd479; font-weight: 600; }
QTextBrowser { background: transparent; border: none; color: #e6edf3; }
#transcript { background: rgba(255,255,255,8); border-radius: 6px; padding: 4px; }
QSplitter::handle { background: rgba(255,255,255,25); height: 1px; }
QProgressBar { background: rgba(255,255,255,20); border: none; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #3fb950; border-radius: 3px; }
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
    detection_sig = pyqtSignal(str, bool, str)
    levels_sig = pyqtSignal(dict)
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

    def detection(self, text: str, is_question: bool, question: str) -> None:
        self.detection_sig.emit(text, is_question, question)

    def levels(self, levels: dict[str, float]) -> None:
        self.levels_sig.emit(levels)


class Overlay(QWidget):
    def __init__(self, cfg: UIConfig, sink: QtSink) -> None:
        super().__init__()
        self.cfg = cfg
        self._answer = ""
        self._drag_from: QPoint | None = None
        self._lines: list[str] = []
        self._partials: dict[str, str] = {}
        self._speaker_colors: dict[str, str] = {}
        self._meters: dict[str, QProgressBar] = {}

        self.setWindowTitle("Call Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(cfg.opacity)
        self.resize(cfg.width, cfg.height)
        self.setFont(QFont("Segoe UI", cfg.font_size))
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

        self.meter_row = QHBoxLayout()
        self.meter_row.setSpacing(8)
        layout.addLayout(self.meter_row)

        # Answer on top, full transcript below; the user can drag the divider.
        answer_box = QWidget()
        answer_layout = QVBoxLayout(answer_box)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        self.question_label = QLabel(
            f"Warte auf eine Frage … ({cfg.hotkey_answer.upper()} = jetzt antworten)", objectName="question"
        )
        self.question_label.setWordWrap(True)
        self.answer_view = QTextBrowser()
        answer_layout.addWidget(self.question_label)
        answer_layout.addWidget(self.answer_view, 1)

        transcript_box = QWidget()
        transcript_layout = QVBoxLayout(transcript_box)
        transcript_layout.setContentsMargins(0, 4, 0, 0)
        transcript_layout.addWidget(QLabel("Transkript", objectName="section"))
        self.transcript_view = QTextBrowser(objectName="transcript")
        transcript_layout.addWidget(self.transcript_view, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(answer_box)
        splitter.addWidget(transcript_box)
        splitter.setSizes([cfg.height * 55 // 100, cfg.height * 45 // 100])
        layout.addWidget(splitter, 1)

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
        sink.detection_sig.connect(self.on_detection)
        sink.levels_sig.connect(self.on_levels)
        sink.toggle_sig.connect(lambda: self.setVisible(not self.isVisible()))
        self.on_status("listening", "Starte …")

    # ---- helpers -------------------------------------------------------------------------------

    def _color(self, speaker: str) -> str:
        if speaker not in self._speaker_colors:
            self._speaker_colors[speaker] = SPEAKER_COLORS[len(self._speaker_colors) % len(SPEAKER_COLORS)]
        return self._speaker_colors[speaker]

    def _render_transcript(self) -> None:
        bar = self.transcript_view.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4  # don't yank the view while the user scrolls back
        partial = [
            f'<span style="color:{self._color(s)}">{escape(s)}:</span> <i style="color:#8b949e">{escape(t)} …</i>'
            for s, t in self._partials.items()
            if t
        ]
        self.transcript_view.setHtml("<br>".join(self._lines + partial))
        if at_bottom:
            bar.setValue(bar.maximum())

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
        self._partials.pop(speaker, None)
        self._lines.append(f'<span style="color:{self._color(speaker)}"><b>{escape(speaker)}:</b></span> {escape(text)}')
        del self._lines[:-MAX_TRANSCRIPT_LINES]
        self._render_transcript()

    def on_partial(self, speaker: str, text: str) -> None:
        self._partials[speaker] = text
        self._render_transcript()

    def on_detection(self, text: str, is_question: bool, question: str) -> None:
        if is_question:
            self._lines.append(f'<span style="color:#ffd479">&nbsp;&nbsp;↳ Frage erkannt: {escape(question)}</span>')
            self._render_transcript()
        else:
            self.state_label.setText("Höre zu · letzter Satz: keine Frage")

    def on_levels(self, levels: dict) -> None:
        for name, level in levels.items():
            if name not in self._meters:
                label = QLabel(name, objectName="meta")
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {self._color(name)}; }}")
                self.meter_row.addWidget(label)
                self.meter_row.addWidget(bar, 1)
                self._meters[name] = bar
            # Speech RMS is typically 0.01-0.2; a square-root curve makes quiet input visible.
            self._meters[name].setValue(min(100, int((level ** 0.5) * 220)))

    # ---- dragging ------------------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_from)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
