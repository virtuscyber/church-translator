# Church Live Translation

Real-time Ukrainian → English translation for church services with biblical language styling. Audio flows from microphone through AI translation and out to speakers or Dante network.

## How It Works

```
Microphone (Ukrainian audio) → Whisper STT → GPT-4o Translation → ElevenLabs TTS → Speakers / Dante
```

1. **Captures** the preacher's Ukrainian audio from a microphone or Dante virtual soundcard
2. **Transcribes** using OpenAI's `gpt-4o-transcribe` (best Ukrainian accuracy)
3. **Translates** Ukrainian → English with biblical vocabulary (GPT-4o + custom prompt)
4. **Speaks** the English translation via ElevenLabs (warm, natural voice)
5. **Outputs** to local speakers and/or Dante network for listeners' headphones

**Expected latency: 10-15 seconds** | **Cost: ~$2-3/hour**

---

## Quick Start (Recommended)

No Docker, no terminal commands. Any tech volunteer can do this.

### 1. Download

```
git clone https://github.com/virtuscyber/church-translator.git
```

Or download and extract the ZIP from GitHub.

### 2. Launch

- **Windows:** Double-click `start.bat`
- **macOS:** Double-click `start.command`
- **Linux:** Double-click `start.sh` (or run `./start.sh`)

The launcher automatically:
- Checks that Python 3.11+ and ffmpeg are installed
- Creates a virtual environment and installs dependencies (first time only)
- Starts the dashboard and opens your browser

### 3. Setup Wizard

The browser opens to the dashboard. On first run, a setup wizard walks you through:
- Entering your **OpenAI API key** (required)
- Entering your **ElevenLabs API key** (optional — falls back to OpenAI TTS)
- Choosing source and target languages

### 4. Start Translating

Click **Start Live Translation**, and the system begins listening, translating, and speaking in real time.

---

## Alternative: Docker

If you prefer Docker:

```bash
git clone https://github.com/virtuscyber/church-translator.git
cd church-translator
docker compose up -d
```

Open **http://localhost:8085** in your browser. The setup wizard handles configuration.

```bash
docker compose logs -f          # Watch logs
docker compose down             # Stop
docker compose up -d --build    # Rebuild after updates
```

---

## One-Line Install Script

For a native install on a fresh machine:

```bash
curl -sL https://raw.githubusercontent.com/virtuscyber/church-translator/main/install.sh | bash
```

The script detects your OS (macOS, Ubuntu/Debian, Fedora, Arch), installs prerequisites, and sets everything up.

---

## Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **ffmpeg** — for audio processing
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: [Download](https://ffmpeg.org/download.html) and add to PATH
- **OpenAI API key** — [Get one](https://platform.openai.com/api-keys)
- **ElevenLabs API key** (optional) — [Get one](https://elevenlabs.io/)

The dashboard shows a health check panel on load — it tells you exactly what's missing and how to fix it.

---

## Dante / AES67 Configuration

For network audio output to Dante:

1. Open **Dante Controller** on the network
2. Route the preacher's microphone channel to Dante Via on the translation PC
3. In **Dante Via**, route that channel to the app's input
4. Route the app's output back through Dante Via to the translation output channel
5. Connect translation output to listener headphones/speakers in Dante Controller

Set `output.mode` to `"dante"` or `"both"` in `config.yaml` to enable AES67 multicast output.

## Configuration

Edit `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `audio.input_device` | system default | Input device name or index |
| `audio.output_device` | system default | Output device name or index |
| `audio.chunk_duration_sec` | 8 | Seconds of audio per processing chunk |
| `transcription.model` | gpt-4o-transcribe | OpenAI STT model |
| `translation.model` | gpt-4o | Translation model |
| `synthesis.provider` | elevenlabs | "elevenlabs" or "openai" |
| `output.mode` | sounddevice | "sounddevice", "dante", or "both" |

## Biblical Language

The translation prompt ensures biblical vocabulary:
- "foolish" not "stupid"
- "brethren" not "guys"
- "transgression" not "mistake"
- "congregation" not "crowd"
- Scripture references preserved exactly

See `prompts/biblical_translator.txt` to customize.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Microphone  │────▶│  Whisper STT │────▶│  GPT-4o      │────▶│  ElevenLabs  │────▶│  Speakers /  │
│  (Ukrainian) │     │  Transcribe  │     │  Translation  │     │  TTS         │     │  Dante Out   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

- **VAD smart chunking** — Detects sentence boundaries for natural translation flow
- **Streaming pipeline** — Overlapped processing for ~2s per-chunk latency
- **AES67 multicast** — Native network audio output, no Dante Via required
- **Web dashboard** — Real-time transcript, health checks, device/language selection

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

33 tests covering imports, config, dashboard API, pipeline, launcher, AES67, and OS scripts.

---

## License

MIT

---

**Developed by [Virtus Cybersecurity](https://virtuscyber.com) — Bogdan Salamakha**
