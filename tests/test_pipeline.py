import asyncio
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from call_assistant.config import AppConfig, load_config
from call_assistant.context import load_context
from call_assistant.engine import Engine
from call_assistant.llm import AnswerStats, Assistant, CardHeader, parse_header
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
    """Feed: a FRAGE card when the segment contains '?', INFO when it mentions 'ISO', otherwise SKIP."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.segments = []
        self.answers = 0
        self.deepened = []

    async def stream_card(self, earlier, segment, recent_cards):
        self.segments.append(segment)
        await asyncio.sleep(self.delay)
        if "?" in segment:
            yield CardHeader("question", segment.splitlines()[-1].split(": ", 1)[1])
        elif "ISO" in segment:
            yield CardHeader("info", "ISO 27001")
        else:
            return
        yield "**Vorschlag**"
        yield AnswerStats("fake", 0.1, 0.2, 10, 0, 3)

    async def stream_deepen(self, transcript, question, short_answer):
        self.deepened.append((question, short_answer))
        yield "- Begründung mit Zahl"
        yield AnswerStats("fake-opus", 2.0, 3.0, 10, 0, 5)

    async def summarize(self, transcript):
        return "### Zusammenfassung\nEs ging um den **Umsatz**.\n### Offene Punkte und nächste Schritte\n- Zahl nachliefern"

    async def stream_answer(self, transcript):
        self.answers += 1
        yield "**Ausführlich**"
        yield AnswerStats("fake-opus", 0.5, 1.0, 10, 0, 3)


class RecordingSink:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        return lambda *args, **kw: self.events.append((name, args))

    def cards(self):
        return [args[1:] for name, args in self.events if name == "card_start"]


def run_sim(tmp_path, script, assistant=None, cfg=None):
    path = tmp_path / "sim.txt"
    path.write_text(script, encoding="utf-8")
    cfg = cfg or AppConfig()
    cfg.protocol_dir = str(tmp_path / "protokolle")
    assistant, sink = assistant or FakeAssistant(), RecordingSink()
    asyncio.run(Engine(cfg, sink, assistant, simulate=path, context_files=lambda: ["briefing.md"]).run())
    return assistant, sink


def test_feed_creates_question_and_info_cards_and_skips_small_talk(tmp_path):
    fake, sink = run_sim(
        tmp_path,
        "them: Hallo zusammen, schön dass es heute klappt\npause 0.2\n"
        "them: Wir stehen vor der ISO 27001 Zertifizierung im Frühjahr\npause 0.2\n"
        "them: Wie hoch war der Umsatz?\npause 0.2\nhotkey\npause 0.2\n",
    )
    assert sink.cards() == [
        ("info", "ISO 27001"),
        ("question", "Wie hoch war der Umsatz?"),
        ("answer", "Ausführliche Antwort (Hotkey)"),
    ]
    assert len(fake.segments) == 3  # small talk was checked too, the model just answered SKIP
    assert fake.answers == 1
    done = [args for name, args in sink.events if name == "card_done"]
    assert len(done) == 4 and all(args[1] is not None for args in done)  # question card finishes twice


def test_question_cards_get_deepened_in_the_same_card(tmp_path):
    fake, sink = run_sim(
        tmp_path,
        "them: Wir stehen vor der ISO 27001 Zertifizierung im Frühjahr\npause 0.2\n"
        "them: Wie hoch war der Umsatz?\npause 0.2\n",
    )
    assert fake.deepened == [("Wie hoch war der Umsatz?", "**Vorschlag**")]  # only the FRAGE card
    question_id = next(args[0] for name, args in sink.events if name == "card_start" and args[1] == "question")
    deltas = [args[1] for name, args in sink.events if name == "card_delta" and args[0] == question_id]
    assert deltas == ["**Vorschlag**", "\n\n**▸ Ausführlich zum Ablesen:**\n\n", "- Begründung mit Zahl"]
    models = [args[1].model for name, args in sink.events if name == "card_done" and args[0] == question_id]
    assert models == ["fake", "fake-opus"]

    cfg = AppConfig()
    cfg.llm.deepen_questions = False
    fake, _ = run_sim(tmp_path, "them: Wie hoch war der Umsatz?\npause 0.2\n", cfg=cfg)
    assert fake.deepened == []


def test_feed_collects_short_utterances_and_ignores_my_speech(tmp_path):
    fake, _ = run_sim(tmp_path, "them: Ja genau\nme: Das sehe ich auch so, absolut richtig\nthem: und dann noch das Budget\npause 0.2\n")
    assert len(fake.segments) == 1
    assert "Ja genau" in fake.segments[0] and "Budget" in fake.segments[0]


def test_feed_batches_utterances_that_arrive_while_a_card_is_generating():
    fake = FakeAssistant(delay=0.3)
    engine = Engine(AppConfig(), RecordingSink(), fake)

    async def scenario():
        feed = asyncio.create_task(engine._feed_loop())
        for i in range(5):
            engine._on_utterance("Gegenüber", f"Das ist Satz Nummer {i} mit genug Wörtern")
            await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)
        feed.cancel()

    asyncio.run(scenario())
    # first sentence starts a card at once; the four that arrive while it runs end up in one or two more
    assert 1 < len(fake.segments) < 5
    assert "Nummer 0" in fake.segments[0]
    joined = "\n".join(fake.segments)
    assert all(joined.count(f"Nummer {i}") == 1 for i in range(5))


def test_parse_header_variants():
    assert parse_header("FRAGE: Wie hoch ist das Budget?") == CardHeader("question", "Wie hoch ist das Budget?")
    assert parse_header("**INFO:** ISO 27001") == CardHeader("info", "ISO 27001")
    assert parse_header("## Frage: Preis") == CardHeader("question", "Preis")
    assert parse_header("SKIP").kind == "skip"
    assert parse_header("Der Umsatz lag bei 184.000 €") is None


def stream_card_with(chunks):
    a = Assistant(AppConfig(), "", client=object())

    async def fake_stream(*args, **kwargs):
        for c in chunks:
            yield c
        yield AnswerStats("m", 0.1, 0.2, 1, 0, 1)

    a._stream = fake_stream

    async def collect():
        return [item async for item in a.stream_card("", "x", [])]

    return asyncio.run(collect())


def test_stream_card_splits_header_and_body_across_chunks():
    items = stream_card_with(["\nFRA", "GE: Budget?\n**Ja", ", 3 %**"])
    assert items[0] == CardHeader("question", "Budget?")
    assert "".join(i for i in items if isinstance(i, str)) == "**Ja, 3 %**"
    assert isinstance(items[-1], AnswerStats)


def test_stream_card_skip_yields_nothing_and_unformatted_text_becomes_info():
    assert stream_card_with(["SKIP"]) == []
    assert stream_card_with(["SKI", "P\n"]) == []
    items = stream_card_with(["Einfach ein Hinweis ohne Marker\nzweite Zeile"])
    assert items[0] == CardHeader("info", "")
    assert "Einfach ein Hinweis" in "".join(i for i in items if isinstance(i, str))


def test_old_config_keys_are_ignored(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[llm]\ndetection_enabled = true\ndetector_model = 'x'\nfeed_min_words = 4\n", encoding="utf-8")
    assert load_config(p).llm.feed_min_words == 4


def test_haiku_feed_requests_skip_refusal_fallback():
    captured = {}

    class FakeStream:
        def __init__(self, **kw):
            captured.update(kw)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def text_stream(self):
            async def gen():
                yield "SKIP"
            return gen()

        async def get_final_message(self):
            class U:
                input_tokens = output_tokens = 1
                cache_read_input_tokens = cache_creation_input_tokens = 0

            class M:
                model, stop_reason, usage = "claude-haiku-5-5", "end_turn", U()

            return M()

    class Client:
        class beta:
            class messages:
                stream = FakeStream

    a = Assistant(AppConfig(), "Doku", client=Client())

    async def go():
        return [i async for i in a.stream_card("", "them: hi", [])]

    assert asyncio.run(go()) == []
    assert captured["model"] == "claude-haiku-5-5" and "fallbacks" not in captured
    assert captured["system"][-1]["cache_control"]["type"] == "ephemeral"



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


def test_audio_level_meter_and_decay(monkeypatch):
    import numpy as np

    from call_assistant import audio

    class FakePyAudio:
        paContinue = 0

    monkeypatch.setattr(audio, "_pyaudio", lambda: FakePyAudio)
    loop = asyncio.new_event_loop()
    cap = audio.AudioCapture(None, audio.Device(0, "x", 2, 16000), loop, asyncio.Queue(maxsize=2), 50)
    stereo = (np.ones(1600, dtype=np.int16) * 3277).tobytes()  # 800 frames x 2 ch at ~0.1 full scale
    cap._callback(stereo, 800, None, 0)
    assert 0.09 < cap.level < 0.11
    monkeypatch.setattr(audio.time, "monotonic", lambda: 1e12)
    assert cap.level == 0.0
    loop.close()


def test_context_store_notes_files_and_order(tmp_path):
    from call_assistant.context import ContextStore

    ctx = tmp_path / "context"
    store = ContextStore(ctx)
    store.save_notes("1. Einleitung\n2. Methode")
    (ctx / "briefing.md").write_text("Briefing", encoding="utf-8")
    src = tmp_path / "Gliederung.txt"
    src.write_text("Kapitel 1", encoding="utf-8")
    (tmp_path / "bild.png").write_bytes(b"x")
    assert store.add_files([src, tmp_path / "bild.png"]) == ["Gliederung.txt"]

    text = store.load()
    assert text.index("briefing.md") < text.index("notizen.md") < text.index("Gliederung.txt")
    assert store.notes().startswith("1. Einleitung")
    assert ContextStore.summary(text, len(store.files())) .startswith("3 Dateien")

    store.remove("Gliederung.txt")
    store.save_notes("   ")
    assert [p.name for p in store.files()] == ["briefing.md"]


def test_assistant_context_can_be_swapped_and_prompts_ask_for_spoken_prose():
    a = Assistant(AppConfig(), "", client=object())
    assert len(a._feed_system) == 1
    a.set_context("Gliederung: 1. Einleitung")
    for system in (a._feed_system, a._answer_system, a._deepen_system):
        assert "Gliederung: 1. Einleitung" in system[-1]["text"]
        assert "read it out loud" in system[0]["text"] and "no lists" in system[0]["text"]
        assert "outline (Gliederung)" in system[0]["text"]


def test_session_writes_protocol_with_summary_transcript_and_cards(tmp_path):
    import docx

    fake, sink = run_sim(
        tmp_path,
        "them: Wir stehen vor der ISO 27001 Zertifizierung im Frühjahr\npause 0.2\n"
        "them: Wie hoch war der Umsatz?\npause 0.2\n",
    )
    states = [args[0] for name, args in sink.events if name == "session"]
    assert states[0] == "idle" and "running" in states and states[-1] == "idle"
    saved = [args[0] for name, args in sink.events if name == "protocol_saved"]
    assert len(saved) == 1 and saved[0].endswith(".docx")

    md = Path(saved[0]).with_suffix(".md").read_text(encoding="utf-8")
    assert md.startswith("# Gesprächsprotokoll")
    assert "Es ging um den **Umsatz**." in md and "**Kontext:** briefing.md" in md
    assert "Gegenüber:** Wie hoch war der Umsatz?" in md
    assert "· FRAGE · Wie hoch war der Umsatz?" in md and "Ausführlich zum Ablesen" in md
    assert "· INFO · ISO 27001" in md

    text = "\n".join(p.text for p in docx.Document(saved[0]).paragraphs)
    assert "Gesprächsprotokoll" in text and "Zahl nachliefern" in text and "**" not in text


def test_protocol_is_saved_even_if_summary_fails_and_skipped_when_empty(tmp_path):
    class NoSummary(FakeAssistant):
        async def summarize(self, transcript):
            raise RuntimeError("API down")

    _, sink = run_sim(tmp_path, "them: Wie hoch war der Umsatz?\npause 0.2\n", assistant=NoSummary())
    saved = [args[0] for name, args in sink.events if name == "protocol_saved"]
    md = Path(saved[0]).with_suffix(".md").read_text(encoding="utf-8")
    assert "Keine Zusammenfassung: Fehler bei der Erstellung (API down)" in md
    assert "Wie hoch war der Umsatz?" in md

    _, sink = run_sim(tmp_path, "pause 0.1\n")
    assert not [1 for name, _ in sink.events if name == "protocol_saved"]
    assert "kein Protokoll" in [args for name, args in sink.events if name == "session"][-1][1]


def test_engine_start_stop_toggle_and_hotkey_only_while_running(tmp_path):
    cfg = AppConfig()
    cfg.protocol_dir = str(tmp_path / "p")
    fake, sink = FakeAssistant(), RecordingSink()
    engine = Engine(cfg, sink, fake)

    async def no_audio():
        await asyncio.Event().wait()

    engine._run_live = no_audio  # stands in for microphone + Deepgram

    async def scenario():
        main = asyncio.create_task(engine.run())
        await asyncio.sleep(0.05)
        engine._on_hotkey()  # idle: ignored
        engine._toggle()  # start
        await asyncio.sleep(0.05)
        assert engine.running
        engine._on_utterance("Gegenüber", "Wie hoch war der Umsatz?")
        await asyncio.sleep(0.3)
        engine._toggle()  # stop -> protocol
        while engine._session_task is not None:
            await asyncio.sleep(0.05)
        engine._toggle()  # a second session starts fresh
        await asyncio.sleep(0.05)
        assert engine.running and engine.transcript.turns == []
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(scenario())
    assert fake.answers == 0
    assert len([1 for name, _ in sink.events if name == "protocol_saved"]) == 1  # 2nd session was empty
