"""Orchestrates audio -> STT -> transcript -> question detection -> answer, all on one asyncio loop."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Protocol

import anthropic

from .config import AppConfig
from .llm import AnswerStats, Assistant
from .transcript import Transcript, Turn

log = logging.getLogger(__name__)


class UiSink(Protocol):
    def status(self, state: str, detail: str = "") -> None: ...
    def question(self, text: str, trigger: str) -> None: ...
    def answer_delta(self, text: str) -> None: ...
    def answer_done(self, stats: AnswerStats | None, error: str = "") -> None: ...
    def utterance(self, speaker: str, text: str) -> None: ...
    def partial(self, speaker: str, text: str) -> None: ...


class Engine:
    def __init__(self, cfg: AppConfig, ui: UiSink, assistant: Assistant, simulate: Path | None = None) -> None:
        self.cfg = cfg
        self.ui = ui
        self.assistant = assistant
        self.simulate = simulate
        save = None
        if cfg.save_transcript:
            save = Path(cfg.transcript_dir) / f"call_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.transcript = Transcript(save)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._streams: list = []
        self._answer_task: asyncio.Task | None = None
        self._answered_turn_id = 0
        self._main_task: asyncio.Task | None = None
        self._pending_detections = 0

    # ---- thread-safe entry points (hotkeys, UI) -------------------------------------------------

    def request_answer(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._on_hotkey)

    def stop(self) -> None:
        if self.loop and self._main_task:
            self.loop.call_soon_threadsafe(self._main_task.cancel)

    # ---- pipeline ------------------------------------------------------------------------------

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self.ui.status("listening", "Höre zu")
        try:
            if self.simulate:
                await self._run_simulation(self.simulate)
                if self._answer_task:
                    await asyncio.gather(self._answer_task, return_exceptions=True)
            else:
                await self._run_live()
        except asyncio.CancelledError:
            pass
        finally:
            if self._answer_task:
                self._answer_task.cancel()

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
            await asyncio.gather(*(s.run() for s in self._streams))
        finally:
            for capture in captures:
                capture.stop()
            pa.terminate()

    async def _run_simulation(self, path: Path) -> None:
        """Replay a scripted conversation; lets you test detection and answers without audio or Deepgram.

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
        await asyncio.sleep(0.5)
        while self._pending_detections:
            await asyncio.sleep(0.1)

    # ---- events --------------------------------------------------------------------------------

    def _on_utterance(self, speaker: str, text: str) -> None:
        turn = self.transcript.add(speaker, text)
        self.ui.utterance(speaker, turn.text)
        if (
            speaker == self.cfg.them_label
            and self.cfg.llm.detection_enabled
            and (len(turn.text.split()) >= self.cfg.llm.min_question_words or "?" in turn.text)
        ):
            self._pending_detections += 1
            asyncio.get_running_loop().create_task(self._detect(turn))

    async def _detect(self, turn: Turn) -> None:
        try:
            recent = self.transcript.render(last_n=self.cfg.llm.detector_window_turns)
            result = await self.assistant.check_question(recent)
        except anthropic.APIError as exc:
            log.warning("Question detection failed: %s", exc)
            return
        finally:
            self._pending_detections -= 1
        if result is None or not result.is_question:
            return
        if turn.id <= self._answered_turn_id:  # a newer question is already being answered
            return
        self._answered_turn_id = turn.id
        self._start_answer(result.question or turn.text, "auto")

    def _on_hotkey(self) -> None:
        for stream in self._streams:
            stream.flush()  # include the sentence that is still being spoken
        self._answered_turn_id = self.transcript.turns[-1].id if self.transcript.turns else 0
        self._start_answer(None, "hotkey")

    def _start_answer(self, question: str | None, trigger: str) -> None:
        if self._answer_task and not self._answer_task.done():
            self._answer_task.cancel()
        self.ui.question(question or "(Hotkey: letzte offene Frage)", trigger)
        self._answer_task = asyncio.get_running_loop().create_task(self._answer(question, trigger))

    async def _answer(self, question: str | None, trigger: str) -> None:
        self.ui.status("answering", "Antworte …")
        transcript = self.transcript.render(max_chars=self.cfg.llm.answer_transcript_chars)
        try:
            stats = None
            async for item in self.assistant.stream_answer(transcript, question, trigger):
                if isinstance(item, AnswerStats):
                    stats = item
                else:
                    self.ui.answer_delta(item)
            self.ui.answer_done(stats)
            if stats:
                log.info(
                    "Answer: %s, first token %.2fs, total %.2fs, in %d (cached %d), out %d",
                    stats.model, stats.first_token_s or -1, stats.total_s,
                    stats.input_tokens, stats.cached_tokens, stats.output_tokens,
                )
        except asyncio.CancelledError:
            raise
        except anthropic.APIError as exc:
            log.error("Answer failed: %s", exc)
            self.ui.answer_done(None, error=f"API-Fehler: {exc}")
        finally:
            self.ui.status("listening", "Höre zu")
