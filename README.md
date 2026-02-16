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

## Quick Start

### Prerequisites

- Python 3.11+
- [Dante Via](https://www.getdante.com/products/software-essentials/dante-via/) ($50) or Dante Virtual Soundcard ($30)
- OpenAI API key
- ElevenLabs API key

### Setup

```bash
# Clone and install
git clone <repo-url>
cd church-translator
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Find your audio devices
python scripts/list_devices.py

# Edit config.yaml with your Dante device names
# Set input_device and output_device

# Run
python -m src.main
```

### Dante Configuration

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
