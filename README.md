# Church Live Translation 🎙️✝️

Real-time Ukrainian → English translation for church services with biblical language styling. Audio flows from Dante network through AI translation and back to Dante.

## How It Works

```
Dante (Ukrainian audio) → Whisper STT → GPT-4o Translation → ElevenLabs TTS → Dante (English audio)
```

1. **Captures** the preacher's Ukrainian audio from Dante via virtual soundcard
2. **Transcribes** using OpenAI's `gpt-4o-transcribe` (best Ukrainian accuracy)
3. **Translates** Ukrainian → English with biblical vocabulary (GPT-4o + custom prompt)
4. **Speaks** the English translation via ElevenLabs (warm, natural voice)
5. **Outputs** back to Dante network for listeners' headphones/speakers

**Expected latency: 10-15 seconds** | **Cost: ~$2-3/hour**

---

## 🚀 Quick Start (Docker — Recommended)

The fastest way to get running. You just need [Docker](https://docs.docker.com/get-docker/) installed.

```bash
git clone https://github.com/virtuscyber/church-translator.git
cd church-translator
docker compose up -d
```

Open **http://localhost:8085** in your browser. The setup wizard will walk you through entering your API keys.

**That's it!** Your data (API keys, config, recordings) persists across restarts.

```bash
# Useful commands
docker compose logs -f          # Watch logs
docker compose down             # Stop
docker compose up -d --build    # Rebuild after updates
```

---

## 📦 One-Line Install Script

For a native install (no Docker), run:

```bash
curl -sL https://raw.githubusercontent.com/virtuscyber/church-translator/main/install.sh | bash
```

Or with Docker:

```bash
curl -sL https://raw.githubusercontent.com/virtuscyber/church-translator/main/install.sh | bash -s -- --docker
```

The script will:
- Detect your OS (macOS, Ubuntu/Debian, Fedora, Arch)
- Install Python 3.11+, ffmpeg, and other prerequisites
- Clone the repo, create a virtualenv, install dependencies
- Optionally set up auto-start (systemd on Linux)

---

## 🔧 Manual Install

### Prerequisites

- Python 3.11+
- ffmpeg
- OpenAI API key
- ElevenLabs API key
- [Dante Via](https://www.getdante.com/products/software-essentials/dante-via/) ($50) or Dante Virtual Soundcard ($30) *(for Dante audio routing)*

### Setup

```bash
git clone https://github.com/virtuscyber/church-translator.git
cd church-translator

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the dashboard
python dashboard/server.py
```

Open **http://localhost:8085** — the setup wizard handles API key configuration.

---

## ⛪ Dante Configuration

1. Open **Dante Controller** on the network
2. Route the preacher's microphone channel to Dante Via on the translation PC
3. In **Dante Via**, route that channel to the app's input
4. Route the app's output back through Dante Via to the translation output channel
5. Connect translation output to listener headphones/speakers in Dante Controller

## Configuration

Edit `config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `audio.input_device` | system default | Dante input device name or index |
| `audio.output_device` | system default | Dante output device name or index |
| `audio.chunk_duration_sec` | 8 | Seconds of audio per processing chunk |
| `transcription.model` | gpt-4o-transcribe | OpenAI STT model |
| `translation.model` | gpt-4o | Translation model |
| `synthesis.provider` | elevenlabs | "elevenlabs" or "openai" |

## Biblical Language

The translation prompt ensures biblical vocabulary:
- "foolish" not "stupid"
- "brethren" not "guys"
- "transgression" not "mistake"
- "congregation" not "crowd"
- Scripture references preserved exactly

See `prompts/biblical_translator.txt` to customize.

## Architecture

See [PLAN.md](PLAN.md) for the full technical plan, cost estimates, and roadmap.
