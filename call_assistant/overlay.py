"""Always-on-top overlay window (PyQt6). The engine runs in another thread and talks to it via Qt signals."""
from __future__ import annotations

from datetime import datetime
from html import escape

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .config import UIConfig
from .llm import AnswerStats

STATUS_COLORS = {"listening": "#3fb950", "answering": "#58a6ff", "error": "#f85149"}
SPEAKER_COLORS = ["#79c0ff", "#d2a8ff"]
CARD_STYLES = {
    "question": ("FRAGE", "#ffd479"),
    "info": ("INFO", "#58a6ff"),
    "answer": ("ANTWORT", "#3fb950"),
}
MAX_CARDS = 200
MAX_TRANSCRIPT_LINES = 300

STYLE = """
#panel { background: #16181d; }
QLabel { color: #e6edf3; }
#title { color: #8b949e; font-weight: 600; }
#hint, #meta, #section, #empty { color: #8b949e; }
#section { font-size: 9pt; }
QScrollArea, #feed { background: transparent; border: none; }
QTextBrowser { background: rgba(255,255,255,8); border: none; border-radius: 6px; color: #e6edf3; padding: 4px; }
QSplitter::handle { background: rgba(255,255,255,25); height: 1px; }
QProgressBar { background: rgba(255,255,255,20); border: none; border-radius: 3px; max-height: 6px; }
QPushButton#jump { background: #58a6ff; color: #0d1117; border: none; border-radius: 10px; padding: 3px 10px;
                   font-weight: 600; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,50); border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


class QtSink(QObject):
    """Implements engine.UiSink. Signals emitted from the engine thread are queued onto the GUI thread."""

    status_sig = pyqtSignal(str, str)
    utterance_sig = pyqtSignal(str, str)
    partial_sig = pyqtSignal(str, str)
    levels_sig = pyqtSignal(dict)
    card_start_sig = pyqtSignal(int, str, str)
    card_delta_sig = pyqtSignal(int, str)
    card_done_sig = pyqtSignal(int, object, str)
    toggle_sig = pyqtSignal()

    def status(self, state: str, detail: str = "") -> None:
        self.status_sig.emit(state, detail)

    def utterance(self, speaker: str, text: str) -> None:
        self.utterance_sig.emit(speaker, text)

    def partial(self, speaker: str, text: str) -> None:
        self.partial_sig.emit(speaker, text)

    def levels(self, levels: dict[str, float]) -> None:
        self.levels_sig.emit(levels)

    def card_start(self, card_id: int, kind: str, title: str) -> None:
        self.card_start_sig.emit(card_id, kind, title)

    def card_delta(self, card_id: int, text: str) -> None:
        self.card_delta_sig.emit(card_id, text)

    def card_done(self, card_id: int, stats: AnswerStats | None, error: str = "") -> None:
        self.card_done_sig.emit(card_id, stats, error)


class Card(QFrame):
    def __init__(self, kind: str, title: str) -> None:
        super().__init__()
        label, color = CARD_STYLES.get(kind, ("INFO", "#58a6ff"))
        self.setStyleSheet(
            f"Card {{ background: rgba(255,255,255,10); border-left: 3px solid {color}; border-radius: 6px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        head = QLabel(
            f'<span style="color:#8b949e">{datetime.now():%H:%M:%S}</span> '
            f'<b style="color:{color}">{label}</b> {escape(title)}'
        )
        head.setWordWrap(True)
        self.body = QLabel("")
        self.body.setTextFormat(Qt.TextFormat.MarkdownText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meta = QLabel("", objectName="meta")
        self.meta.hide()
        layout.addWidget(head)
        layout.addWidget(self.body)
        layout.addWidget(self.meta)
        self.text = ""

    def append(self, text: str) -> None:
        self.text += text
        self.body.setText(self.text)

    def finish(self, stats: AnswerStats | None, error: str) -> None:
        if error:
            self.append(f"\n\n*{error}*")
        elif stats:
            first = f"{stats.first_token_s:.1f}s" if stats.first_token_s is not None else "–"
            refused = " · vom Modell abgelehnt" if stats.refused else ""
            entry = f"{stats.model} {first}{refused}"
            self.meta.setText(f"{self.meta.text()} · {entry}" if self.meta.text() else entry)
            self.meta.show()


class Overlay(QWidget):
    def __init__(self, cfg: UIConfig, sink: QtSink) -> None:
        super().__init__()
        self.cfg = cfg
        self._lines: list[str] = []
        self._partials: dict[str, str] = {}
        self._speaker_colors: dict[str, str] = {}
        self._meters: dict[str, QProgressBar] = {}
        self._cards: dict[int, Card] = {}
        self._follow = True
        self._unseen = 0

        self.setWindowTitle("Call Assistant")
        # Normal Windows frame: title bar with minimize / maximize / close, resizable at the edges,
        # taskbar entry so a minimized window can be restored. Stays on top of the call window.
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowOpacity(cfg.opacity)
        self.setMinimumSize(320, 300)
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
        header.addWidget(self.dot)
        header.addWidget(self.title)
        header.addWidget(self.state_label, 1)
        layout.addLayout(header)

        self.meter_row = QHBoxLayout()
        self.meter_row.setSpacing(8)
        layout.addLayout(self.meter_row)

        # --- card feed (scrollable, newest at the bottom, follows unless the user scrolled up) ---
        feed_box = QWidget()
        feed_layout = QVBoxLayout(feed_box)
        feed_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        feed = QWidget(objectName="feed")
        self.cards_layout = QVBoxLayout(feed)
        self.cards_layout.setContentsMargins(0, 0, 4, 0)
        self.cards_layout.setSpacing(6)
        self.empty_label = QLabel(
            f"Noch keine Karten. Sie erscheinen, sobald im Gespräch etwas Relevantes fällt.\n"
            f"{cfg.hotkey_answer.upper()} = ausführliche Antwort jetzt.",
            objectName="empty",
        )
        self.empty_label.setWordWrap(True)
        self.cards_layout.addWidget(self.empty_label)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(feed)
        feed_layout.addWidget(self.scroll, 1)

        self.jump = QPushButton("", objectName="jump")
        self.jump.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump.clicked.connect(self._jump_to_latest)
        self.jump.hide()
        feed_layout.addWidget(self.jump, 0, Qt.AlignmentFlag.AlignHCenter)

        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(self._on_range_changed)

        # --- transcript ---
        transcript_box = QWidget()
        transcript_layout = QVBoxLayout(transcript_box)
        transcript_layout.setContentsMargins(0, 4, 0, 0)
        transcript_layout.addWidget(QLabel("TRANSKRIPT", objectName="section"))
        self.transcript_view = QTextBrowser()
        transcript_layout.addWidget(self.transcript_view, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(feed_box)
        splitter.addWidget(transcript_box)
        splitter.setSizes([cfg.height * 68 // 100, cfg.height * 32 // 100])
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.meta_label = QLabel(f"{cfg.hotkey_toggle.upper()} ein-/ausblenden", objectName="meta")
        footer.addWidget(self.meta_label, 1)
        layout.addLayout(footer)

        sink.status_sig.connect(self.on_status)
        sink.utterance_sig.connect(self.on_utterance)
        sink.partial_sig.connect(self.on_partial)
        sink.levels_sig.connect(self.on_levels)
        sink.card_start_sig.connect(self.on_card_start)
        sink.card_delta_sig.connect(self.on_card_delta)
        sink.card_done_sig.connect(self.on_card_done)
        sink.toggle_sig.connect(self.toggle)
        self.on_status("listening", "Starte …")

    # ---- feed scrolling ------------------------------------------------------------------------

    def _on_scrolled(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        self._follow = value >= bar.maximum() - 8
        if self._follow:
            self._unseen = 0
            self.jump.hide()

    def _on_range_changed(self, _min: int, maximum: int) -> None:
        if self._follow:
            self.scroll.verticalScrollBar().setValue(maximum)

    def _jump_to_latest(self) -> None:
        self._follow = True
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

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

    def on_card_start(self, card_id: int, kind: str, title: str) -> None:
        self.empty_label.hide()
        card = Card(kind, title)
        self._cards[card_id] = card
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)  # before the stretch
        while len(self._cards) > MAX_CARDS:
            oldest = min(self._cards)
            self._cards.pop(oldest).deleteLater()
        if not self._follow:
            self._unseen += 1
            self.jump.setText(f"↓ {self._unseen} neue Karte{'n' if self._unseen > 1 else ''}")
            self.jump.show()

    def on_card_delta(self, card_id: int, text: str) -> None:
        if card := self._cards.get(card_id):
            card.append(text)

    def on_card_done(self, card_id: int, stats: AnswerStats | None, error: str) -> None:
        if card := self._cards.get(card_id):
            card.finish(stats, error)
        if stats:
            first = f"{stats.first_token_s:.1f}s" if stats.first_token_s is not None else "–"
            self.meta_label.setText(f"Letzte Karte: {stats.model} · erste Worte nach {first}")

    def on_utterance(self, speaker: str, text: str) -> None:
        self._partials.pop(speaker, None)
        self._lines.append(f'<span style="color:{self._color(speaker)}"><b>{escape(speaker)}:</b></span> {escape(text)}')
        del self._lines[:-MAX_TRANSCRIPT_LINES]
        self._render_transcript()

    def on_partial(self, speaker: str, text: str) -> None:
        self._partials[speaker] = text
        self._render_transcript()

    def on_levels(self, levels: dict) -> None:
        for name, level in levels.items():
            if name not in self._meters:
                label = QLabel(name, objectName="meta")
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {self._color(name)}; border-radius: 3px; }}")
                self.meter_row.addWidget(label)
                self.meter_row.addWidget(bar, 1)
                self._meters[name] = bar
            # Speech RMS is typically 0.01-0.2; a square-root curve makes quiet input visible.
            self._meters[name].setValue(min(100, int((level ** 0.5) * 220)))

    # ---- window ------------------------------------------------------------------------------

    def toggle(self) -> None:
        """Hotkey: hide when visible, otherwise bring back (also from minimized) without stealing focus."""
        if self.isVisible() and not self.isMinimized():
            self.hide()
        else:
            self.showNormal()
            self.raise_()
