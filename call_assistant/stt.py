"""Deepgram live transcription over a raw WebSocket (no SDK, so SDK version changes can't break it)."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable
from urllib.parse import urlencode

import websockets

from .config import DeepgramConfig

log = logging.getLogger(__name__)

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
KEEPALIVE_AFTER_S = 4.0  # Deepgram closes the socket after ~10 s without audio; loopback is silent when nobody talks


class UtteranceAssembler:
    """Turns Deepgram's interim/final result stream into complete utterances.

    Deepgram sends interim results (is_final=false) that get revised, then final segments
    (is_final=true). An utterance ends at speech_final=true (pause detected) or at an
    UtteranceEnd message (fallback when the pause detection does not fire, e.g. background noise).
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def feed(self, msg: dict) -> tuple[str | None, str | None]:
        """Return (finished_utterance, current_partial_text)."""
        kind = msg.get("type")
        if kind == "UtteranceEnd":
            return self.flush(), None
        if kind != "Results":
            return None, None

        alts = msg.get("channel", {}).get("alternatives") or [{}]
        text = (alts[0].get("transcript") or "").strip()
        if not msg.get("is_final"):
            return None, " ".join(self._parts + [text]).strip() or None
        if text:
            self._parts.append(text)
        if msg.get("speech_final"):
            return self.flush(), None
        return None, " ".join(self._parts) or None

    def flush(self) -> str | None:
        text = " ".join(self._parts).strip()
        self._parts.clear()
        return text or None


def build_url(cfg: DeepgramConfig, sample_rate: int) -> str:
    params: list[tuple[str, str]] = [
        ("model", cfg.model),
        ("language", cfg.language),
        ("encoding", "linear16"),
        ("sample_rate", str(sample_rate)),
        ("channels", "1"),
        ("interim_results", "true"),
        ("punctuate", "true"),
        ("smart_format", "true"),
        ("endpointing", str(cfg.endpointing_ms)),
        ("utterance_end_ms", str(cfg.utterance_end_ms)),
        ("vad_events", "true"),
    ]
    params += [("keyterm", term) for term in cfg.keyterms]
    return f"{DEEPGRAM_URL}?{urlencode(params)}"


class DeepgramStream:
    def __init__(
        self,
        speaker: str,
        api_key: str,
        cfg: DeepgramConfig,
        sample_rate: int,
        audio: asyncio.Queue,
        on_utterance: Callable[[str, str], None],
        on_partial: Callable[[str, str], None],
        on_status: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self.speaker = speaker
        self._api_key = api_key
        self._url = build_url(cfg, sample_rate)
        self._audio = audio
        self._on_utterance = on_utterance
        self._on_partial = on_partial
        self._on_status = on_status
        self.assembler = UtteranceAssembler()

    def flush(self) -> None:
        """Emit whatever has been recognized so far, e.g. when the hotkey is pressed mid-sentence."""
        text = self.assembler.flush()
        if text:
            self._on_utterance(self.speaker, text)

    async def run(self) -> None:
        delay = 1.0
        while True:
            try:
                async with websockets.connect(
                    self._url,
                    additional_headers={"Authorization": f"Token {self._api_key}"},
                    max_size=None,
                ) as ws:
                    log.info("Deepgram connected (%s)", self.speaker)
                    self._on_status(f"STT verbunden ({self.speaker})")
                    delay = 1.0
                    await self._session(ws)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.InvalidStatus as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    self._on_status("Deepgram API key ungültig")
                    raise RuntimeError(f"Deepgram rejected the API key (HTTP {status})") from exc
                log.warning("Deepgram HTTP %s (%s), retry in %.0fs", status, self.speaker, delay)
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                log.warning("Deepgram connection lost (%s): %s, retry in %.0fs", self.speaker, exc, delay)
            self._on_status(f"STT getrennt ({self.speaker}), neuer Versuch …")
            self.flush()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    async def _session(self, ws) -> None:
        async def sender() -> None:
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(self._audio.get(), timeout=KEEPALIVE_AFTER_S)
                    except TimeoutError:
                        await ws.send(json.dumps({"type": "KeepAlive"}))
                        continue
                    await ws.send(chunk)
            except asyncio.CancelledError:
                try:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception:
                    pass
                raise

        async def receiver() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                done, partial = self.assembler.feed(msg)
                if done:
                    self._on_utterance(self.speaker, done)
                elif partial:
                    self._on_partial(self.speaker, partial)

        tasks = {asyncio.create_task(sender()), asyncio.create_task(receiver())}
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception():
                raise task.exception()
