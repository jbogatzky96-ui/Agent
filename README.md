# Call Assistant

Listens to a video call from outside (Zoom, Teams, Meet or anything else that plays audio on your PC), transcribes it live in German, detects questions from the other participants and shows a suggested answer in a small always-on-top window.

```
speakers (loopback) ─┐                         ┌─ question detector (Claude Haiku 5.5)
                     ├─ Deepgram nova-3 (de) ──┤
microphone ──────────┘     transcript          └─ answer (Claude Opus 5.5 + briefing/docs) ─> overlay
```

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
```

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+A` | Answer now (the most recent open question, including a sentence still being spoken) |
| `Ctrl+Alt+H` | Show/hide the overlay |

Drag the window anywhere; resize at the bottom right corner.

## Practical notes

- **Use headphones.** With speakers, your microphone picks up the other participants and everything appears twice in the transcript.
- **Latency:** expect about 1 to 2 s until the sentence is final plus 1 to 2 s until the first words of the answer. Set `answer_effort = "medium"` for more thorough answers at the cost of speed.
- **Costs (rough, check current pricing):** two Deepgram streams per call minute (set `capture_mic = false` to halve that); detection with Haiku is a fraction of a cent per question; one Opus answer with a few thousand context tokens costs a few cents.
- If an answer request fails with a 400 error mentioning `fallbacks`, set `use_refusal_fallback = false` in `config.toml`.

## Legal

Transcribing other people requires their consent in Germany (§ 201 StGB for recordings; the audio is processed by Deepgram and Anthropic, which is GDPR-relevant). Tell the participants. `save_transcript` is off by default.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

Module overview: `audio.py` (WASAPI capture), `stt.py` (Deepgram WebSocket), `transcript.py`, `context.py`, `llm.py` (prompts, detector, answer), `engine.py` (asyncio pipeline, simulation), `overlay.py` (PyQt6), `__main__.py` (CLI).
