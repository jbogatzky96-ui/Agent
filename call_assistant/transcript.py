from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Turn:
    id: int
    speaker: str
    text: str
    ts: datetime = field(default_factory=datetime.now)

    def render(self) -> str:
        return f"[{self.ts:%H:%M:%S}] {self.speaker}: {self.text}"


class Transcript:
    def __init__(self, save_path: Path | None = None) -> None:
        self.turns: list[Turn] = []
        self._ids = itertools.count(1)
        self._save_path = save_path
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, speaker: str, text: str) -> Turn:
        turn = Turn(next(self._ids), speaker, text.strip())
        self.turns.append(turn)
        if self._save_path:
            with self._save_path.open("a", encoding="utf-8") as fh:
                fh.write(turn.render() + "\n")
        return turn

    def render(self, *, last_n: int | None = None, max_chars: int | None = None) -> str:
        turns = self.turns[-last_n:] if last_n else self.turns
        lines = [t.render() for t in turns]
        if max_chars is not None:
            # Keep the most recent lines that fit; the newest context matters most.
            kept, size = [], 0
            for line in reversed(lines):
                size += len(line) + 1
                if size > max_chars:
                    break
                kept.append(line)
            lines = list(reversed(kept))
        return "\n".join(lines)
