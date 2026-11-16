"""Session minutes: summary (from the model), full transcript and all cards, saved as Markdown and Word."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .transcript import Turn

log = logging.getLogger(__name__)

KIND_LABELS = {"question": "FRAGE", "info": "INFO", "answer": "ANTWORT"}


@dataclass
class CardRecord:
    id: int
    kind: str
    title: str
    time: datetime = field(default_factory=datetime.now)
    text: str = ""


@dataclass
class SessionRecord:
    started: datetime
    ended: datetime
    turns: list[Turn]
    cards: list[CardRecord]
    context_files: list[str]


def render_markdown(rec: SessionRecord, summary: str | None, summary_note: str = "") -> str:
    minutes = max(1, round((rec.ended - rec.started).total_seconds() / 60))
    lines = [
        "# Gesprächsprotokoll",
        "",
        f"**Datum:** {rec.started:%d.%m.%Y} · **Beginn:** {rec.started:%H:%M} · **Ende:** {rec.ended:%H:%M} · "
        f"**Dauer:** {minutes} min",
    ]
    if rec.context_files:
        lines.append(f"**Kontext:** {', '.join(rec.context_files)}")
    lines += ["", "## Zusammenfassung (KI-generiert, bitte prüfen)", ""]
    if summary:
        lines.append(summary.strip())
    else:
        lines.append(f"*Keine Zusammenfassung: {summary_note or 'nicht erstellt'}.*")
    lines += ["", "## Transkript", ""]
    lines += [f"**[{t.ts:%H:%M:%S}] {t.speaker}:** {t.text}" for t in rec.turns] or ["*Leer.*"]
    lines += ["", "## Karten des Assistenten", ""]
    if not rec.cards:
        lines.append("*Keine Karten.*")
    for card in rec.cards:
        title = f" · {card.title}" if card.title else ""
        lines += [f"### {card.time:%H:%M:%S} · {KIND_LABELS.get(card.kind, card.kind)}{title}", "", card.text.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _add_runs(paragraph, text: str) -> None:
    # **bold** segments alternate with normal text; single *italics* markers are dropped
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            paragraph.add_run(part.replace("*", "")).bold = i % 2 == 1


def write_docx(markdown: str, path: Path) -> None:
    import docx

    doc = docx.Document()
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=0)
        elif re.match(r"^[-*] ", stripped):
            _add_runs(doc.add_paragraph(style="List Bullet"), stripped[2:])
        else:
            _add_runs(doc.add_paragraph(), stripped)
    doc.save(path)


def _free_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    n = 2
    while path.exists():
        path = directory / f"{stem}_{n}{suffix}"
        n += 1
    return path


def save_protocol(
    rec: SessionRecord, summary: str | None, directory: str | Path, docx: bool = True, summary_note: str = ""
) -> Path:
    """Write the minutes; returns the file to open (Word if written, otherwise Markdown)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(rec, summary, summary_note)
    md_path = _free_path(directory, f"Protokoll_{rec.started:%Y-%m-%d_%H-%M}", ".md")
    md_path.write_text(markdown, encoding="utf-8")
    log.info("Protocol saved: %s", md_path)
    if docx:
        try:
            docx_path = md_path.with_suffix(".docx")
            write_docx(markdown, docx_path)
            log.info("Protocol saved: %s", docx_path)
            return docx_path
        except Exception as exc:  # the Markdown file is already there; Word is a convenience
            log.warning("Could not write Word protocol: %s", exc)
    return md_path
