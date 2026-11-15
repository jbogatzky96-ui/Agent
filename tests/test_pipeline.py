import asyncio
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from call_assistant.config import AppConfig, load_config
from call_assistant.context import load_context
from call_assistant.engine import Engine
from call_assistant.llm import AnswerStats, Assistant, QuestionCheck
from call_assistant.stt import UtteranceAssembler, build_url
from call_assistant.transcript import Transcript

ROOT = Path(__file__).resolve().parent.parent


def result(text, is_final=False, speech_final=False):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text}]},
    }


def test_assembler_joins_final_segments_until_speech_final():
    a = UtteranceAssembler()
    assert a.feed(result("Wie hoch")) == (None, "Wie hoch")
    assert a.feed(result("Wie hoch war", is_final=True)) == (None, "Wie hoch war")
    assert a.feed(result("der Umsatz")) == (None, "Wie hoch war der Umsatz")
    assert a.feed(result("der Umsatz?", is_final=True, speech_final=True)) == ("Wie hoch war der Umsatz?", None)
    assert a.flush() is None


def test_assembler_flushes_on_utterance_end_and_ignores_empty_finals():
    a = UtteranceAssembler()
    a.feed(result("", is_final=True))
    a.feed(result("Okay danke", is_final=True))
    assert a.feed({"type": "UtteranceEnd"}) == ("Okay danke", None)
    assert a.feed({"type": "UtteranceEnd"}) == (None, None)
    assert a.feed({"type": "Metadata"}) == (None, None)


def test_build_url_contains_german_model_and_keyterms():
    cfg = AppConfig().deepgram
    cfg.keyterms = ["EBITDA", "Müller GmbH"]
    q = parse_qs(urlparse(build_url(cfg, 48000)).query)
    assert q["model"] == ["nova-3"] and q["language"] == ["de"]
    assert q["sample_rate"] == ["48000"] and q["encoding"] == ["linear16"]
    assert q["keyterm"] == ["EBITDA", "Müller GmbH"]


def test_transcript_max_chars_keeps_newest_lines():
    t = Transcript()
    for i in range(10):
        t.add("Gegenüber", f"Satz {i}")
    rendered = t.render(max_chars=60)
    assert rendered.endswith("Satz 9") and "Satz 0" not in rendered
    assert t.render(last_n=2).count("\n") == 1


def test_example_config_matches_defaults():
    assert load_config(ROOT / "config.example.toml") == AppConfig()


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[llm]\nanswer_modle = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="llm.answer_modle"):
        load_config(p)


def test_context_puts_briefing_first_and_skips_underscore_files(tmp_path):
    (tmp_path / "a_notes.md").write_text("Notizen", encoding="utf-8")
    (tmp_path / "briefing.md").write_text("Briefing", encoding="utf-8")
    (tmp_path / "_old.md").write_text("alt", encoding="utf-8")
    text = load_context(tmp_path)
    assert text.index("briefing.md") < text.index("a_notes.md")
    assert "alt" not in text


class FakeAssistant:
    def __init__(self):
        self.answers = []

    async def check_question(self, recent):
        last = recent.splitlines()[-1]
        await asyncio.sleep(0.01)
        return QuestionCheck(is_question="?" in last, question=last.split(": ", 1)[1] if "?" in last else "")

    async def stream_answer(self, transcript, question, trigger):
        self.answers.append((question, trigger, transcript))
        for word in ["**Antwort**", " auf ", str(question)]:
            yield word
        yield AnswerStats("fake", 0.1, 0.2, 10, 0, 3)


class RecordingSink:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args, **kw: self.events.append((name, args))


def test_engine_simulation_answers_detected_questions(tmp_path):
    script = tmp_path / "sim.txt"
    script.write_text(
        "# comment\nthem: Hallo zusammen\nme: Hallo\nthem: Wie hoch war der Umsatz?\npause 0.2\nhotkey\npause 0.2\n",
        encoding="utf-8",
    )
    cfg = AppConfig()
    fake, sink = FakeAssistant(), RecordingSink()
    engine = Engine(cfg, sink, fake, simulate=script)
    asyncio.run(engine.run())

    triggers = [(q, t) for q, t, _ in fake.answers]
    assert ("Wie hoch war der Umsatz?", "auto") in triggers
    assert (None, "hotkey") in triggers
    assert "Gegenüber: Wie hoch war der Umsatz?" in fake.answers[0][2]
    assert any(name == "answer_done" for name, _ in sink.events)


def test_assistant_builds_cacheable_system_prompt():
    a = Assistant(AppConfig(), "Doku", client=object())
    assert a._answer_system[-1]["cache_control"]["type"] == "ephemeral"
    assert "Doku" in a._answer_system[-1]["text"]
    assert len(Assistant(AppConfig(), "", client=object())._answer_system) == 1


def test_deepgram_stream_against_fake_server(monkeypatch):
    import json

    import websockets

    from call_assistant import stt

    monkeypatch.setattr(stt, "KEEPALIVE_AFTER_S", 0.1)
    received = {"audio": b"", "keepalive": 0, "auth": None}

    async def handler(ws):
        received["auth"] = ws.request.headers.get("Authorization")
        await ws.send(json.dumps(result("Wie hoch", is_final=True)))
        await ws.send(json.dumps(result("ist das Budget?", is_final=True, speech_final=True)))
        async for msg in ws:
            if isinstance(msg, bytes):
                received["audio"] += msg
            elif json.loads(msg)["type"] == "KeepAlive":
                received["keepalive"] += 1

    async def scenario():
        utterances = []
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            queue = asyncio.Queue()
            stream = stt.DeepgramStream(
                "Gegenüber", "secret", AppConfig().deepgram, 16000, queue,
                on_utterance=lambda s, t: utterances.append((s, t)), on_partial=lambda s, t: None,
            )
            stream._url = f"ws://127.0.0.1:{port}/v1/listen"
            task = asyncio.create_task(stream.run())
            await queue.put(b"\x01\x02" * 100)
            await asyncio.sleep(0.5)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return utterances

    utterances = asyncio.run(scenario())
    assert utterances == [("Gegenüber", "Wie hoch ist das Budget?")]
    assert received["auth"] == "Token secret"
    assert received["audio"] == b"\x01\x02" * 100
    assert received["keepalive"] >= 2
