from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    capture_mic: bool = True
    # Substring of the device name; empty means the Windows default device.
    loopback_device: str = ""
    mic_device: str = ""
    chunk_ms: int = 50


@dataclass
class DeepgramConfig:
    model: str = "nova-3"
    language: str = "de"
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000
    # Domain terms (names, products, abbreviations) that the recognizer should prefer.
    keyterms: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    # Continuous feed: after each new segment of the conversation, one short card (or nothing).
    feed_enabled: bool = True
    feed_model: str = "claude-haiku-5-5"
    feed_effort: str = "low"
    feed_max_tokens: int = 1500
    feed_min_words: int = 6  # shorter segments are collected until enough was said (or a "?" appears)
    feed_on_my_speech: bool = False  # also create cards after you speak
    feed_transcript_chars: int = 20000
    # Hotkey: thorough answer with the stronger model.
    answer_model: str = "claude-opus-5-5"
    answer_effort: str = "low"
    answer_max_tokens: int = 4000
    answer_transcript_chars: int = 60000
    use_refusal_fallback: bool = True


@dataclass
class UIConfig:
    opacity: float = 0.93
    font_size: int = 11
    width: int = 480
    height: int = 600
    hotkey_answer: str = "ctrl+alt+a"
    hotkey_toggle: str = "ctrl+alt+h"


@dataclass
class AppConfig:
    answer_language: str = "Deutsch"
    my_name: str = ""
    me_label: str = "Ich"
    them_label: str = "Gegenüber"
    context_dir: str = "context"
    save_transcript: bool = False
    transcript_dir: str = "transcripts"
    audio: AudioConfig = field(default_factory=AudioConfig)
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Keys from earlier versions; ignored so an old config.toml keeps working.
REMOVED_KEYS = {
    "llm.detection_enabled", "llm.detector_model", "llm.detector_effort",
    "llm.detector_window_turns", "llm.min_question_words",
}


def _apply(target: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if f"{path}{key}" in REMOVED_KEYS:
            log.warning("Config key %s%s is no longer used and ignored", path, key)
            continue
        if key not in known:
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Config section [{path}{key}] must be a table")
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None) -> AppConfig:
    cfg = AppConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    with p.open("rb") as fh:
        _apply(cfg, tomllib.load(fh))
    return cfg
