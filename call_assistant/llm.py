from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

import anthropic
from pydantic import BaseModel, Field

from .config import AppConfig

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class QuestionCheck(BaseModel):
    is_question: bool = Field(description="True if the latest utterance needs an answer or contribution from the user now.")
    question: str = Field(description="Self-contained reformulation of the question, empty if is_question is false.")


DETECTOR_SYSTEM = """\
You monitor the live transcript of a call. "{me}" is the user you work for{name_hint}; "{them}" is everyone else \
on the call (all remote participants share this label). The transcript comes from speech recognition and may contain errors.

Decide whether the LATEST utterance by "{them}" contains a genuine question or request for information, an opinion, \
a decision or a number that the user may need to answer or contribute to right now.

Count as NOT a question: greetings and small talk ("Wie geht's?"), technical checks ("Hört ihr mich?", \
"Sieht man meinen Bildschirm?"), rhetorical questions the speaker answers themselves, questions clearly addressed \
to another named person, and fragments without a recognizable question.

If it is a question, write `question` as a self-contained reformulation in {language}: resolve references \
("das", "die Zahl von vorhin") using the earlier turns, and keep it short. Otherwise leave `question` empty."""


ANSWER_SYSTEM = """\
You are a live assistant for a person in an ongoing call. "{me}" is that person{name_hint}; "{them}" is everyone else. \
The user reads your text on a small overlay while still listening, so they need something they can say out loud \
within seconds.

Format, in {language}:
- First line: the core answer as one short sentence in **bold**, phrased so the user can say it directly.
- Then at most 4 short bullet points with supporting facts, numbers, arguments or a good follow-up question.
- No preamble, no restating the question, no closing remarks. Stay under about 120 words.

Content rules:
- Base the answer on the briefing and documents below where they cover the topic, and prefer their exact numbers.
- If the documents do not cover it and you are not confident, say so briefly and mark uncertain statements with \
"(unsicher)". A wrong confident number in a live call is worse than an honest "das prüfe ich nach".
- The transcript is machine-generated and may contain misrecognized words; interpret charitably.
- If the request comes from a hotkey and there is no clear open question, give the user 2 to 3 useful talking points \
on the current topic instead."""


@dataclass
class AnswerStats:
    model: str
    first_token_s: float | None
    total_s: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    refused: bool = False


def _name_hint(cfg: AppConfig) -> str:
    return f' (name: {cfg.my_name})' if cfg.my_name else ""


class Assistant:
    def __init__(self, cfg: AppConfig, context_text: str, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.cfg = cfg
        self.client = client or anthropic.AsyncAnthropic()
        fmt = dict(me=cfg.me_label, them=cfg.them_label, language=cfg.answer_language, name_hint=_name_hint(cfg))
        self._detector_system = DETECTOR_SYSTEM.format(**fmt)
        # Instructions + context are identical on every request, so they form a cacheable prefix.
        self._answer_system: list[dict] = [{"type": "text", "text": ANSWER_SYSTEM.format(**fmt)}]
        if context_text:
            self._answer_system.append(
                {
                    "type": "text",
                    "text": f"<briefing_and_documents>\n{context_text}\n</briefing_and_documents>",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            )

    async def check_question(self, recent_transcript: str) -> QuestionCheck | None:
        llm = self.cfg.llm
        response = await self.client.messages.parse(
            model=llm.detector_model,
            max_tokens=2048,
            output_config={"effort": llm.detector_effort},
            system=self._detector_system,
            messages=[{"role": "user", "content": f"<transcript>\n{recent_transcript}\n</transcript>"}],
            output_format=QuestionCheck,
        )
        if response.stop_reason != "end_turn":
            log.warning("Detector stopped with %s", response.stop_reason)
            return None
        return response.parsed_output

    async def stream_answer(self, transcript: str, question: str | None, trigger: str) -> AsyncIterator[str | AnswerStats]:
        """Yield text deltas, then one AnswerStats as the last item."""
        llm = self.cfg.llm
        if question:
            ask = f"<question source=\"{trigger}\">\n{question}\n</question>"
        else:
            ask = (
                f'<question source="{trigger}">\nThe user pressed the hotkey. Answer the most recent open question '
                f'from "{self.cfg.them_label}" in the transcript.\n</question>'
            )
        kwargs: dict = dict(
            model=llm.answer_model,
            max_tokens=llm.answer_max_tokens,
            output_config={"effort": llm.answer_effort},
            system=self._answer_system,
            messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>\n\n{ask}"}],
        )
        if llm.use_refusal_fallback:
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
