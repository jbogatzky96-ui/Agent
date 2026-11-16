# Call Assistant

Listens to a video call from outside (Zoom, Teams, Meet or anything else that plays audio on your PC), transcribes it live in German and keeps a running feed of short cards in a small always-on-top window: an answer you can read out loud when someone asks something (FRAGE), background on topics, terms and numbers that come up (INFO). A hotkey gives you a thorough answer on demand (ANTWORT).

```
speakers (loopback) ─┐                         ┌─ live feed after every segment (Claude Haiku 5.5):
                     ├─ Deepgram nova-3 (de) ──┤    FRAGE card, INFO card, or nothing
microphone ──────────┘     transcript          └─ hotkey: thorough answer (Claude Opus 5.5)
                                                  both use your briefing/docs ─> overlay
```

For FRAGE cards a second stage starts automatically: Claude Opus 5.5 appends a thorough answer ("▸ Ausführlich") to the same card a few seconds later, with reasoning, numbers, likely objections and follow-up questions. You get something to say within about a second and substance shortly after. Turn it off with `deepen_questions = false`; card length is set with `feed_max_words`.

There is no separate "is this a question?" step that could miss something. After every segment the other side says, the model sees the conversation plus your briefing and decides itself whether a card helps. Short fragments are collected until enough was said, and segments that arrive while a card is being written are combined into the next card, so the feed never falls behind.

No bot joins the call. The app records what your headset plays (WASAPI loopback) and your microphone, as two separate streams, so it knows who said what without speaker diarization.

## Setup (Windows 10/11)

Requires Python 3.11 or newer ([python.org](https://www.python.org/downloads/), tick "Add python.exe to PATH").

Double-click `setup.bat`. It creates a virtual environment, installs everything and copies the templates (`.env`, `config.toml`, `context/`). Then put your keys into `.env`.

Manual alternative:

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
copy config.example.toml config.toml
xcopy /E /I context.example context
```

API keys: [console.anthropic.com](https://console.anthropic.com) and [console.deepgram.com](https://console.deepgram.com).

## Before a call: give it context

Easiest way: click **Kontext** at the top of the window. Paste the outline (Gliederung), agenda or notes into the text field and add documents with "Dateien hinzufügen" or by dragging them onto the window. Changes apply immediately, also during a call. The assistant uses an outline to understand which point is being discussed.

Behind the scenes everything lives in the `context/` folder (the text field is saved as `notizen.md`), so you can also manage it directly:

Put everything the assistant should know into `context/`:

- `briefing.md`: participants, goal, key numbers, expected questions and your position. It is read first and treated as the most reliable source.
- Any `.pdf`, `.docx`, `.md`, `.txt` files (offers, reports, product sheets). Prefix a file with `_` to exclude it.
- Names and technical terms that speech recognition tends to get wrong go into `keyterms` in `config.toml`.

The context is sent with every answer and cached for an hour, so repeated answers are cheap.

## Run

First test without audio or Deepgram (only uses the Anthropic API):

```powershell
run.bat --simulate examples\simulation.txt --context context.example --no-ui
```

Then live:

```powershell
run.bat                 # overlay window
run.bat --no-ui         # terminal only
run.bat --list-devices  # if the wrong headset/mic is picked up
run.bat --log-file debug.log  # also write everything to a file (contains the transcript!)
```

The overlay shows a level bar per audio source. If the bar for "Gegenüber" stays flat while someone talks, the app is listening to the wrong output device: check the Windows default playback device or set `loopback_device` in `config.toml`. Under "Transkript" you see everything that was recognized. The terminal window logs every utterance, every card and every SKIP.

The card feed scrolls like a chat: new cards appear at the bottom and the view follows them. When you scroll up to re-read something, the view stays where you are and a "↓ neue Karten" button takes you back to the latest card. The divider between cards and transcript can be dragged.

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+A` | Thorough answer now with the stronger model (the most recent open question, including a sentence still being spoken) |
| `Ctrl+Alt+H` | Show/hide the overlay |

It is a normal Windows window that stays on top: move it by the title bar, resize it at the edges, minimize/maximize/close with the buttons at the top right.

## Practical notes

- **Use headphones.** With speakers, your microphone picks up the other participants and everything appears twice in the transcript.
- **Latency:** expect about 1 to 2 s until the sentence is final plus 1 to 2 s until the first words of the answer. Set `answer_effort = "medium"` for more thorough answers at the cost of speed.
- **Costs (rough, check current pricing):** two Deepgram streams per call minute (set `capture_mic = false` to halve that); the Haiku feed costs roughly €0.10-0.20 per call hour, the Opus second stage for questions roughly €1-2 per hour; one Opus hotkey answer with a few thousand context tokens costs a few cents. If the cards are too shallow, set `feed_model = "claude-sonnet-5-5"` (roughly €1.5-2 per hour).
- If an answer request fails with a 400 error mentioning `fallbacks`, set `use_refusal_fallback = false` in `config.toml`.

## Legal

Transcribing other people requires their consent in Germany (§ 201 StGB for recordings; the audio is processed by Deepgram and Anthropic, which is GDPR-relevant). Tell the participants. `save_transcript` is off by default.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

Module overview: `audio.py` (WASAPI capture), `stt.py` (Deepgram WebSocket), `transcript.py`, `context.py`, `llm.py` (prompts, feed cards, hotkey answer), `engine.py` (asyncio pipeline, simulation), `overlay.py` (PyQt6), `__main__.py` (CLI).
