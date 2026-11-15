from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}
BRIEFING_NAMES = ("briefing.md", "briefing.txt")


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
        key=lambda p: (p.name.lower() not in BRIEFING_NAMES, str(p).lower()),
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
