"""Orchestrates audio -> STT -> transcript -> live cards (feed) and hotkey answers, all on one asyncio loop."""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from .config import AppConfig
from .llm import AnswerStats, Assistant, CardHeader
from .protocol import CardRecord, SessionRecord, save_protocol
from .transcript import Transcript, Turn

log = logging.getLogger(__name__)

KIND_LABELS = {"question": "FRAGE", "info": "INFO", "answer": "ANTWORT"}


class UiSink(Protocol):
    def status(self, state: str, detail: str = "") -> None: ...
    def utterance(self, speaker: str, text: str) -> None: ...
    def partial(self, speaker: str, text: str) -> None: ...
    def levels(self, levels: dict[str, float]) -> None: ...
    def card_start(self, card_id: int, kind: str, title: str) -> None: ...
    def card_delta(self, card_id: int, text: str) -> None: ...
    def card_done(self, card_id: int, stats: AnswerStats | None, error: str = "") -> None: ...
    def session(self, state: str, detail: str = "") -> None: ...  # "idle" | "running" | "saving"
    def protocol_saved(self, path: str) -> None: ...


class Engine:
    def __init__(
        self,
        cfg: AppConfig,
        ui: UiSink,
        assistant: Assistant,
        simulate: Path | None = None,
        context_files: Callable[[], list[str]] = lambda: [],
    ) -> None:
        self.cfg = cfg
        self.ui = ui
        self.assistant = assistant
        self.simulate = simulate
        self.context_files = context_files
        self.loop: asyncio.AbstractEventLoop | None = None
        self._main_task: asyncio.Task | None = None
        self._card_ids = itertools.count(1)
        self._session_task: asyncio.Task | None = None
        self._stopping = False
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        save = None
        if self.cfg.save_transcript:
            save = Path(self.cfg.transcript_dir) / f"call_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.transcript = Transcript(save)
        self.session_started = datetime.now()
        self._streams: list = []
        self._answer_task: asyncio.Task | None = None
        self._feed_event = asyncio.Event()
        self._feed_busy = False
        self._fed_turn_id = 0  # turns up to this id are covered by a card (or were skipped by the model)
        self._recent_cards: deque[str] = deque(maxlen=6)
        self._deepen_tasks: set[asyncio.Task] = set()
        self._cards: dict[int, CardRecord] = {}

    @property
    def running(self) -> bool:
        return self._session_task is not None and not self._stopping

    # ---- thread-safe entry points (hotkeys, UI) -------------------------------------------------

    def request_answer(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._on_hotkey)

    def request_toggle(self) -> None:
        """Start if idle, stop (and write the protocol) if running. Safe to call from any thread."""
        if self.loop:
            self.loop.call_soon_threadsafe(self._toggle)

    def stop(self) -> None:
        if self.loop and self._main_task:
            self.loop.call_soon_threadsafe(self._main_task.cancel)

    # ---- pipeline ------------------------------------------------------------------------------

    async def run(self, autostart: bool = False) -> None:
        self.loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self.ui.session("idle", "Bereit")
        try:
            if self.simulate:
                self._start_session()
                while self._session_task is not None:
                    await asyncio.sleep(0.05)
            else:
                if autostart:
                    self._start_session()
                await asyncio.Event().wait()  # idle until the app quits; sessions are started via request_toggle
        except asyncio.CancelledError:
            pass
        finally:
            if self._session_task is not None:  # app closed or Ctrl+C during a session: still write the protocol
                await self._stop_session()

    # ---- sessions ------------------------------------------------------------------------------

    def _toggle(self) -> None:
        if self._stopping:
            return
        if self._session_task is None:
            self._start_session()
        else:
            asyncio.get_running_loop().create_task(self._stop_session())

    def _start_session(self) -> None:
        self._reset_session_state()
        log.info("Session started")
        self.ui.session("running", "Höre zu")
        self._session_task = asyncio.get_running_loop().create_task(self._session_body())
        self._session_task.add_done_callback(self._on_session_ended)

    def _on_session_ended(self, task: asyncio.Task) -> None:
        # The body ends on its own when the simulation is over or audio/STT failed; finish the session then.
        if self._stopping or task.cancelled():
            return
        if exc := task.exception():
            log.error("Session aborted: %s", exc)
            self.ui.status("error", f"Abgebrochen: {exc}")
        asyncio.get_running_loop().create_task(self._stop_session())

    async def _session_body(self) -> None:
        feed = asyncio.create_task(self._feed_loop()) if self.cfg.llm.feed_enabled else None
        try:
            if self.simulate:
                await self._run_simulation(self.simulate)
            else:
                await self._run_live()
        finally:
            if feed:
                feed.cancel()
                await asyncio.gather(feed, return_exceptions=True)

    async def _stop_session(self) -> None:
        if self._session_task is None or self._stopping:
            return
        self._stopping = True
        try:
            self.ui.session("saving", "Beende Aufnahme …")
            for stream in self._streams:
                stream.flush()  # keep the sentence that was still being spoken
            self._session_task.cancel()
            await asyncio.gather(self._session_task, return_exceptions=True)

            # Give answers that are still being written a moment so they end up in the protocol.
            pending = [t for t in (self._answer_task, *self._deepen_tasks) if t and not t.done()]
            if pending:
                _, still_running = await asyncio.wait(pending, timeout=15)
                for task in still_running:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            ended = datetime.now()
            log.info("Session stopped")

            if not self.transcript.turns:
                self.ui.session("idle", "Gestoppt · nichts aufgenommen, kein Protokoll")
                return
            self.ui.session("saving", "Protokoll wird erstellt …")
            path = await self._write_protocol(ended)
            self.ui.protocol_saved(str(path))
            self.ui.session("idle", "Protokoll gespeichert")
        except Exception as exc:
            log.exception("Could not finish the session")
            self.ui.session("idle", f"Protokoll fehlgeschlagen: {exc}")
        finally:
            self._session_task = None
            self._stopping = False

    async def _write_protocol(self, ended: datetime) -> Path:
        summary, note = None, ""
        if self.cfg.protocol_summary:
            try:
                summary = await self.assistant.summarize(self.transcript.render())
            except Exception as exc:  # the transcript must be saved even if the summary fails
                log.error("Summary failed: %s", exc)
                note = f"Fehler bei der Erstellung ({exc})"
        else:
            note = "in der Konfiguration deaktiviert"
        record = SessionRecord(
            started=self.session_started,
            ended=ended,
            turns=list(self.transcript.turns),
            cards=list(self._cards.values()),
            context_files=self.context_files(),
        )
        return save_protocol(record, summary, self.cfg.protocol_dir, docx=self.cfg.protocol_docx, summary_note=note)

    # ---- card bookkeeping: every card goes to the UI and into the protocol ----------------------

    def _card_start(self, kind: str, title: str) -> int:
        card_id = next(self._card_ids)
        self._cards[card_id] = CardRecord(card_id, kind, title)
        self.ui.card_start(card_id, kind, title)
        return card_id

    def _card_delta(self, card_id: int, text: str) -> None:
        if card := self._cards.get(card_id):
            card.text += text
        self.ui.card_delta(card_id, text)

    async def _run_live(self) -> None:
        from .audio import AudioCapture, _pyaudio, find_loopback, find_mic
        from .stt import DeepgramStream

        key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set (see .env.example)")

        pyaudio = _pyaudio()
        pa = pyaudio.PyAudio()
        captures: list[AudioCapture] = []
        try:
            sources = [(self.cfg.them_label, find_loopback(pa, self.cfg.audio.loopback_device))]
            if self.cfg.audio.capture_mic:
                sources.append((self.cfg.me_label, find_mic(pa, self.cfg.audio.mic_device)))

            for speaker, device in sources:
                queue: asyncio.Queue = asyncio.Queue(maxsize=200)  # ~10 s at 50 ms chunks
                capture = AudioCapture(pa, device, self.loop, queue, self.cfg.audio.chunk_ms)
                stream = DeepgramStream(
                    speaker, key, self.cfg.deepgram, device.sample_rate, queue,
                    on_utterance=self._on_utterance,
                    on_partial=self.ui.partial,
                    on_status=lambda msg: self.ui.status("listening", msg),
                )
                captures.append(capture)
                self._streams.append(stream)

            for capture in captures:
                capture.start()
            self.ui.status("listening", "Höre zu")
            labels = [speaker for speaker, _ in sources]
            await asyncio.gather(*(s.run() for s in self._streams), self._meter(dict(zip(labels, captures))))
        finally:
            for capture in captures:
                capture.stop()
            pa.terminate()

    async def _meter(self, captures: dict) -> None:
        silent_since = {name: 0.0 for name in captures}
        loop = asyncio.get_running_loop()
        while True:
            levels = {name: cap.level for name, cap in captures.items()}
            self.ui.levels(levels)
            now = loop.time()
            for name, level in levels.items():
                if level > 0.002:
                    silent_since[name] = now
                elif silent_since[name] and now - silent_since[name] > 30:
                    log.warning("No audio from '%s' for 30 s. Wrong device? Try --list-devices.", name)
                    silent_since[name] = now
            await asyncio.sleep(0.2)

    async def _run_simulation(self, path: Path) -> None:
        """Replay a scripted conversation; lets you test cards and answers without audio or Deepgram.

        Script lines: "them: <text>", "me: <text>", "pause <seconds>", "hotkey", "# comment".
        """
        labels = {"them": self.cfg.them_label, "me": self.cfg.me_label}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "hotkey":
                self._on_hotkey()
                continue
            if m := re.fullmatch(r"pause\s+([\d.]+)", line):
                await asyncio.sleep(float(m.group(1)))
                continue
            who, _, text = line.partition(":")
            speaker = labels.get(who.strip().lower())
            if speaker is None:
                raise ValueError(f"Bad simulation line: {raw!r}")
            await asyncio.sleep(min(0.25 * len(text.split()), 4.0))  # rough speaking time
            self._on_utterance(speaker, text.strip())
        await asyncio.sleep(0.1)
        while (
            self._feed_event.is_set()
            or self._feed_busy
            or self._deepen_tasks
            or (self._answer_task and not self._answer_task.done())
        ):
            await asyncio.sleep(0.05)

    # ---- events --------------------------------------------------------------------------------

    def _on_utterance(self, speaker: str, text: str) -> None:
        turn = self.transcript.add(speaker, text)
        log.info("%s: %s", speaker, turn.text)
        self.ui.utterance(speaker, turn.text)
        self._feed_event.set()

    def _on_hotkey(self) -> None:
        if not self.running:
            self.ui.status("listening", "Erst Start drücken, dann antwortet der Hotkey")
            return
        for stream in self._streams:
            stream.flush()  # include the sentence that is still being spoken
        if self._answer_task and not self._answer_task.done():
            self._answer_task.cancel()
        self._answer_task = asyncio.get_running_loop().create_task(self._hotkey_answer())

    # ---- feed: one card (or none) per new segment ---------------------------------------------

    def _pending_segment(self) -> list[Turn] | None:
        new = [t for t in self.transcript.turns if t.id > self._fed_turn_id]
        triggers = [t for t in new if t.speaker == self.cfg.them_label or self.cfg.llm.feed_on_my_speech]
        if not triggers:
            return None
        words = sum(len(t.text.split()) for t in triggers)
        if words < self.cfg.llm.feed_min_words and not any("?" in t.text for t in triggers):
            return None  # keep collecting; the next utterance re-checks
        return new

    async def _feed_loop(self) -> None:
        # Cards are generated one after another. Utterances that arrive meanwhile are batched into
        # the next card, so the feed never falls behind the conversation by more than one card.
        while True:
            await self._feed_event.wait()
            self._feed_event.clear()
            segment = self._pending_segment()
            if not segment:
                continue
            self._fed_turn_id = segment[-1].id
            self._feed_busy = True
            try:
                await self._make_card(segment)
            finally:
                self._feed_busy = False

    async def _make_card(self, segment: list[Turn]) -> None:
        llm = self.cfg.llm
        earlier_turns = [t for t in self.transcript.turns if t.id < segment[0].id]
        earlier = "\n".join(t.render() for t in earlier_turns)[-llm.feed_transcript_chars:]
        new = "\n".join(t.render() for t in segment)
        card_id = None
        header: CardHeader | None = None
        body: list[str] = []
        try:
            async for item in self.assistant.stream_card(earlier, new, list(self._recent_cards)):
                if isinstance(item, CardHeader):
                    header = item
                    card_id = self._card_start(item.kind, item.title)
                    self._recent_cards.append(f"{KIND_LABELS[item.kind]}: {item.title}")
                    log.info("Card %s: %s", KIND_LABELS[item.kind], item.title)
                elif isinstance(item, AnswerStats):
                    self.ui.card_done(card_id, item)
                    self._log_stats("Card", item)
                    if header and header.kind == "question" and self.cfg.llm.deepen_questions and not item.refused:
                        self._start_deepen(card_id, header.title, "".join(body))
                else:
                    body.append(item)
                    self._card_delta(card_id, item)
            if card_id is None:
                log.info("Feed: nothing to add (SKIP)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Card failed: %s", exc)
            if card_id is not None:
                self.ui.card_done(card_id, None, error=f"Fehler: {exc}")
            else:
                self.ui.status("error", f"Live-Karte fehlgeschlagen: {type(exc).__name__}")

    # ---- second stage: depth for question cards ------------------------------------------------

    def _start_deepen(self, card_id: int, question: str, short_answer: str) -> None:
        task = asyncio.get_running_loop().create_task(self._deepen(card_id, question, short_answer))
        self._deepen_tasks.add(task)
        task.add_done_callback(self._deepen_tasks.discard)

    async def _deepen(self, card_id: int, question: str, short_answer: str) -> None:
        # Runs in parallel to the feed, so new cards keep coming while the stronger model writes.
        transcript = self.transcript.render(max_chars=self.cfg.llm.answer_transcript_chars)
        started = False
        try:
            async for item in self.assistant.stream_deepen(transcript, question, short_answer):
                if isinstance(item, AnswerStats):
                    self.ui.card_done(card_id, item)
                    self._log_stats("Deepen", item)
                    continue
                if not started:
                    started = True
                    self._card_delta(card_id, "\n\n**▸ Ausführlich zum Ablesen:**\n\n")
                self._card_delta(card_id, item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Deepen failed: %s", exc)
            self.ui.card_done(card_id, None, error=f"Vertiefung fehlgeschlagen: {exc}")

    # ---- hotkey: thorough answer -----------------------------------------------------------------

    async def _hotkey_answer(self) -> None:
        log.info("Hotkey answer requested")
        card_id = self._card_start("answer", "Ausführliche Antwort (Hotkey)")
        self.ui.status("answering", "Antworte …")
        transcript = self.transcript.render(max_chars=self.cfg.llm.answer_transcript_chars)
        try:
            async for item in self.assistant.stream_answer(transcript):
                if isinstance(item, AnswerStats):
                    self.ui.card_done(card_id, item)
                    self._log_stats("Answer", item)
                else:
                    self._card_delta(card_id, item)
        except asyncio.CancelledError:
            self.ui.card_done(card_id, None, error="abgebrochen")
            raise
        except Exception as exc:
            log.error("Answer failed: %s", exc)
            self.ui.card_done(card_id, None, error=f"Fehler: {exc}")
        finally:
            self.ui.status("listening", "Höre zu")

    @staticmethod
    def _log_stats(what: str, stats: AnswerStats) -> None:
        log.info(
            "%s: %s, first token %.2fs, total %.2fs, in %d (cached %d), out %d",
            what, stats.model, stats.first_token_s or -1, stats.total_s,
            stats.input_tokens, stats.cached_tokens, stats.output_tokens,
        )
