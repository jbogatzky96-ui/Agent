from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}
BRIEFING_NAMES = ("briefing.md", "briefing.txt")
NOTES_NAME = "notizen.md"


def _read(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    raise ValueError(f"Unsupported file type: {path}")


def load_context(directory: str | Path) -> str:
    """Concatenate briefing + documents into one block for the system prompt.

    The briefing goes first because it carries the most specific information about the call.
    Files whose name starts with "_" are skipped, so you can park documents without deleting them.
    """
    root = Path(directory)
    if not root.is_dir():
        return ""

    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("_")),
        # briefing first, then the notes typed into the window, then everything else
        key=lambda p: (p.name.lower() not in BRIEFING_NAMES, p.name.lower() != NOTES_NAME, str(p).lower()),
    )
    parts = []
    for path in files:
        try:
            text = _read(path).strip()
        except Exception as exc:  # a broken PDF must not stop the call
            log.warning("Could not read %s: %s", path, exc)
            continue
        if text:
            rel = path.relative_to(root).as_posix()
            parts.append(f'<document name="{rel}">\n{text}\n</document>')
            log.info("Context loaded: %s (%d chars)", rel, len(text))
    return "\n\n".join(parts)


class ContextStore:
    """The context folder, plus the operations the window offers: notes text field and adding files."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted(
            p for p in self.dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("_")
        )

    def load(self) -> str:
        return load_context(self.dir)

    def notes(self) -> str:
        path = self.dir / NOTES_NAME
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def save_notes(self, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / NOTES_NAME
        if text.strip():
            path.write_text(text.strip() + "\n", encoding="utf-8")
        elif path.exists():
            path.unlink()

    def add_files(self, paths: list[str | Path]) -> list[str]:
        """Copy supported files into the folder; returns the names that were added."""
        import shutil

        self.dir.mkdir(parents=True, exist_ok=True)
        added = []
        for src in map(Path, paths):
            if src.is_file() and src.suffix.lower() in SUPPORTED:
                if src.resolve().parent != self.dir.resolve():
                    shutil.copy2(src, self.dir / src.name)
                added.append(src.name)
        return added

    def remove(self, name: str) -> None:
        path = self.dir / name
        if path.is_file() and path.resolve().parent == self.dir.resolve():
            path.unlink()

    @staticmethod
    def summary(text: str, n_files: int) -> str:
        words = len(text.split())
        return f"{n_files} Datei{'en' if n_files != 1 else ''} · ~{words:,} Wörter".replace(",", ".")
