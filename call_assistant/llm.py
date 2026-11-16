from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator

import anthropic

from .config import AppConfig

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


FEED_SYSTEM = """\
You are a silent live assistant during a call. "{me}" is the user you work for{name_hint}; "{them}" is everyone \
else on the call. The transcript comes from speech recognition and may contain misrecognized words; interpret charitably.

After each new segment of the conversation, decide what helps the user most right now and reply in exactly one \
of three forms. The first line must start with FRAGE:, INFO: or be exactly SKIP.

1. The new segment contains a question or request the user may need to respond to (also when asked to the group):
FRAGE: <the question, short and self-contained>
**<one sentence the user can say out loud>**
- at most 3 short bullets with facts, numbers or arguments

2. Otherwise, the segment mentions a topic, term, company, number, claim or decision where background knowledge \
helps the user follow or contribute:
INFO: <topic in 2-5 words>
- at most 3 short bullets of background, or a good question the user could ask

3. Otherwise (small talk, filler, nothing new), or when a card would repeat one of the recent cards without adding \
anything: SKIP

Rules:
- Write in {language}. At most 70 words. No preamble, no restating the transcript.
- Prefer the briefing and documents below and their exact numbers. Mark general knowledge you are not sure about \
with "(unsicher)". A wrong confident number in a live call is worse than none.
- When unsure whether something was a question, use FRAGE: a missed question costs the user more than an extra card."""


ANSWER_SYSTEM = """\
You are a live assistant for a person in an ongoing call. "{me}" is that person{name_hint}; "{them}" is everyone else. \
The user pressed a hotkey because they want a thorough suggestion right now.

Format, in {language}:
- First line: the core answer as one short sentence in **bold**, phrased so the user can say it directly.
- Then at most 5 short bullet points with supporting facts, numbers, arguments or a good follow-up question.
- No preamble, no restating the question, no closing remarks. Stay under about 150 words.

Content rules:
- Answer the most recent open question or request from "{them}". If there is none, give 2 to 3 useful talking points \
on the current topic.
- Base the answer on the briefing and documents below where they cover the topic, and prefer their exact numbers.
- If the documents do not cover it and you are not confident, say so briefly and mark uncertain statements with \
"(unsicher)".
- The transcript is machine-generated and may contain misrecognized words; interpret charitably."""


@dataclass
class AnswerStats:
    model: str
    first_token_s: float | None
    total_s: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    refused: bool = False


@dataclass
class CardHeader:
    kind: str  # "question" | "info" | "answer"
    title: str


_HEADER = re.compile(r"^[\s*#_>]*(SKIP|FRAGE|INFO)\b[\s*_:]*(.*?)[\s*_]*$", re.IGNORECASE)


def parse_header(line: str) -> CardHeader | None:
    """Parse the first line of a feed reply. Returns kind "skip" for SKIP, None if the line has no marker."""
    m = _HEADER.match(line.strip())
    if not m:
        return None
    marker = m.group(1).upper()
    kind = {"SKIP": "skip", "FRAGE": "question", "INFO": "info"}[marker]
    return CardHeader(kind, m.group(2).strip())


def _name_hint(cfg: AppConfig) -> str:
    return f" (name: {cfg.my_name})" if cfg.my_name else ""


class Assistant:
    def __init__(self, cfg: AppConfig, context_text: str, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.client = client or anthropic.AsyncAnthropic()
        fmt = dict(me=cfg.me_label, them=cfg.them_label, language=cfg.answer_language, name_hint=_name_hint(cfg))
        context: list[dict] = []
        if context_text:
            # Instructions + context are identical on every request, so they form a cacheable prefix.
            context.append(
                {
                    "type": "text",
                    "text": f"<briefing_and_documents>\n{context_text}\n</briefing_and_documents>",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            )
        self._feed_system = [{"type": "text", "text": FEED_SYSTEM.format(**fmt)}, *context]
        self._answer_system = [{"type": "text", "text": ANSWER_SYSTEM.format(**fmt)}, *context]

    async def _stream(
        self, model: str, effort: str, max_tokens: int, system: list[dict], user: str
    ) -> AsyncIterator[str | AnswerStats]:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            output_config={"effort": effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Haiku has no server-side refusal fallback; the parameter would be a no-op there.
        if self.cfg.llm.use_refusal_fallback and "haiku" not in model:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")

        start = time.perf_counter()
        first: float | None = None
        async with self.client.beta.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if first is None:
                    first = time.perf_counter() - start
                yield text
            final = await stream.get_final_message()

        usage = final.usage
        yield AnswerStats(
            model=final.model,
            first_token_s=first,
            total_s=time.perf_counter() - start,
            input_tokens=usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0),
            cached_tokens=usage.cache_read_input_tokens or 0,
            output_tokens=usage.output_tokens,
            refused=final.stop_reason == "refusal",
        )

    async def stream_card(
        self, earlier: str, segment: str, recent_cards: list[str]
    ) -> AsyncIterator[CardHeader | str | AnswerStats]:
        """Yield a CardHeader, then body text deltas, then AnswerStats. Yields nothing for SKIP."""
        llm = self.cfg.llm
        recent = "\n".join(recent_cards) or "(none yet)"
        user = (
            f"<earlier_transcript>\n{earlier or '(start of call)'}\n</earlier_transcript>\n\n"
            f"<recent_cards>\n{recent}\n</recent_cards>\n\n"
            f"<new_segment>\n{segment}\n</new_segment>"
        )
        gen = self._stream(llm.feed_model, llm.feed_effort, llm.feed_max_tokens, self._feed_system, user)
        buffer = ""
        header_sent = False
        try:
            async for item in gen:
                if isinstance(item, AnswerStats):
                    if not header_sent:  # reply without a line break, e.g. just "SKIP"
                        header = parse_header(buffer)
                        if header is None or header.kind == "skip":
                            return
                        yield header
                    yield item
                    return
                if header_sent:
                    yield item
                    continue
                buffer = (buffer + item).lstrip()  # tolerate leading blank lines before the marker
                if "\n" not in buffer and len(buffer) < 200:
                    continue
                first_line, _, rest = buffer.partition("\n")
                header = parse_header(first_line)
                if header and header.kind == "skip":
                    return  # closing the generator closes the HTTP stream, so no tokens are wasted
                if header is None:  # the model ignored the format; show the text anyway
                    header, rest = CardHeader("info", ""), buffer
                header_sent = True
                yield header
                if rest.strip():
                    yield rest.lstrip("\n")
        finally:
            await gen.aclose()

    async def stream_answer(self, transcript: str) -> AsyncIterator[str | AnswerStats]:
        """Thorough answer for the hotkey: text deltas, then AnswerStats."""
        llm = self.cfg.llm
        user = f"<transcript>\n{transcript}\n</transcript>"
        async for item in self._stream(llm.answer_model, llm.answer_effort, llm.answer_max_tokens, self._answer_system, user):
            yield item
