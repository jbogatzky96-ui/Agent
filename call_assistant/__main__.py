from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from .config import AppConfig, load_config
from .context import load_context
from .llm import AnswerStats

log = logging.getLogger("call_assistant")


class ConsoleSink:
    """UiSink for --no-ui. Cards are printed when complete, so a hotkey answer and a feed card never interleave."""

    LABELS = {"question": "FRAGE", "info": "INFO", "answer": "ANTWORT"}

    def __init__(self) -> None:
        self._cards: dict[int, list[str]] = {}

    def status(self, state: str, detail: str = "") -> None:
        log.debug("status %s %s", state, detail)

    def utterance(self, speaker: str, text: str) -> None:
        pass  # already printed by the engine's log

    def partial(self, speaker: str, text: str) -> None:
        pass

    def levels(self, levels: dict[str, float]) -> None:
        pass

    def card_start(self, card_id: int, kind: str, title: str) -> None:
        self._cards[card_id] = [f"[{self.LABELS.get(kind, kind)}] {title}\n"]

    def card_delta(self, card_id: int, text: str) -> None:
        self._cards.setdefault(card_id, []).append(text)

    def card_done(self, card_id: int, stats: AnswerStats | None, error: str = "") -> None:
        text = "".join(self._cards.pop(card_id, []))
        meta = f"[{error}]" if error else ""
        if stats:
            first = f"{stats.first_token_s:.1f}s" if stats.first_token_s is not None else "-"
            meta = f"[{stats.model}, erste Worte nach {first}{' (abgelehnt)' if stats.refused else ''}]"
        print(f"\n{'-' * 60}\n{text.strip()}\n{meta}\n{'-' * 60}", flush=True)


def register_hotkeys(bindings: dict[str, callable]) -> None:
    try:
        import keyboard

        for combo, callback in bindings.items():
            keyboard.add_hotkey(combo, callback)
        log.info("Hotkeys: %s", ", ".join(bindings))
    except Exception as exc:  # e.g. not on Windows, or missing permissions on Linux
        log.warning("Global hotkeys unavailable (%s)", exc)


def check_keys(cfg: AppConfig, simulate: bool) -> None:
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        missing.append("ANTHROPIC_API_KEY")
    if not simulate and not os.environ.get("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if missing:
        sys.exit(f"Missing: {', '.join(missing)}. Copy .env.example to .env and fill it in.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="call_assistant", description="Live call assistant")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml (default: ./config.toml)")
    parser.add_argument("--context", help="Folder with briefing.md and documents (overrides config)")
    parser.add_argument("--simulate", type=Path, help="Replay a scripted conversation instead of live audio")
    parser.add_argument("--no-ui", action="store_true", help="Print to the terminal instead of the overlay")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--log-file", type=Path, help="Also write the log (incl. transcript) to this file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        handlers=handlers,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "websockets", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_devices:
        from .audio import list_devices

        list_devices()
        return

    load_dotenv()
    cfg = load_config(args.config)
    check_keys(cfg, simulate=args.simulate is not None)

    context_text = load_context(args.context or cfg.context_dir)
    log.info("Context: %d characters (~%d tokens)", len(context_text), len(context_text) // 4)

    from .engine import Engine
    from .llm import Assistant

    assistant = Assistant(cfg, context_text)

    if args.no_ui:
        engine = Engine(cfg, ConsoleSink(), assistant, simulate=args.simulate)
        register_hotkeys({cfg.ui.hotkey_answer: engine.request_answer})
        try:
            asyncio.run(engine.run())
        except KeyboardInterrupt:
            pass
        return

    from PyQt6.QtWidgets import QApplication

    from .overlay import Overlay, QtSink

    app = QApplication(sys.argv)
    sink = QtSink()
    overlay = Overlay(cfg.ui, sink)
    engine = Engine(cfg, sink, assistant, simulate=args.simulate)

    def run_engine() -> None:
        try:
            asyncio.run(engine.run())
        except Exception as exc:
            log.exception("Engine stopped")
            sink.status("error", str(exc))

    register_hotkeys({cfg.ui.hotkey_answer: engine.request_answer, cfg.ui.hotkey_toggle: sink.toggle_sig.emit})
    thread = threading.Thread(target=run_engine, name="engine", daemon=True)
    thread.start()
    overlay.show()
    code = app.exec()
    engine.stop()
    thread.join(timeout=3)
    sys.exit(code)


if __name__ == "__main__":
    main()
