# Church Live Translation System — Comprehensive Plan

## Overview

Real-time Ukrainian → English audio translation for church services, with biblical language styling. Audio flows from Dante network → translation pipeline → back to Dante network.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│ Dante Input  │────▶│  Audio       │────▶│  Translation │────▶│  Text-to-    │────▶│ Dante Output│
│ (Ukrainian   │     │  Capture +   │     │  (GPT-4o)    │     │  Speech      │     │ (English    │
│  preacher)   │     │  STT (Whisper)│    │  + Biblical  │     │  (ElevenLabs)│     │  audio)     │
└─────────────┘     └──────────────┘     │  styling     │     └──────────────┘     └─────────────┘
                                         └──────────────┘
```

### Pipeline Stages

| Stage | Technology | Latency | Notes |
|-------|-----------|---------|-------|
| **1. Audio Capture** | Dante Virtual Soundcard (DVS) or Dante Via → system audio device | ~5ms | DVS appears as ASIO/WDM device on Windows |
| **2. Speech-to-Text** | OpenAI `gpt-4o-transcribe` API | ~2-5s | Best Ukrainian support; chunked uploads every 5-10s of speech |
| **3. Translation + Styling** | OpenAI GPT-4o | ~1-3s | Ukrainian→English + biblical tone via system prompt |
| **4. Text-to-Speech** | ElevenLabs Streaming API | ~1-2s | Low-latency streaming; warm, authoritative voice |
| **5. Audio Output** | Play to Dante Virtual Soundcard output channel | ~5ms | Routes back into Dante network |

**Total expected latency: 5-15 seconds** (well under the 1-minute tolerance)

## Component Details

### 1. Dante Audio I/O

**Option A: Dante Virtual Soundcard (DVS)** — $30 license, appears as standard audio device on Windows/Mac. Best for simplicity.

**Option B: Dante Via** — $50 license, more flexible routing. Can route any app's audio to/from Dante.

**Recommendation: Dante Via** — more flexible, lets us pick which app routes where without ASIO complexity.

With Dante Via installed:
- **Input**: Dante Via routes the preacher's Dante channel to a virtual audio input that Python can capture via `sounddevice` or `pyaudio`
- **Output**: Python plays translated audio to a virtual output that Dante Via routes back to the Dante network

### 2. Speech-to-Text (Ukrainian → Text)

**API: OpenAI `gpt-4o-transcribe`**
- Superior multilingual accuracy vs whisper-1
- Supports Ukrainian natively
- Chunked approach: buffer 5-10 seconds of audio, send as a chunk
- Use Voice Activity Detection (VAD) to detect sentence boundaries for natural chunking

**Alternative: OpenAI Realtime API with `gpt-4o-transcribe`**
- WebSocket-based streaming transcription
- Lower latency (incremental results)
- Good for v2 optimization

### 3. Translation + Biblical Styling

**API: OpenAI GPT-4o (chat completions)**

System prompt (key piece):
```
You are a live translator for a Christian church service. Translate Ukrainian to English.

CRITICAL RULES:
- Use biblical/liturgical English vocabulary:
  - "foolish" not "stupid"
  - "wicked" not "evil" (when describing people)
  - "righteous" not "good" (in moral contexts)
  - "transgression" or "sin" not "mistake" (for moral failings)
  - "brethren" or "brothers and sisters" not "guys" or "folks"
  - "the Lord" not "God" (when context is devotional)
  - "grace" and "mercy" preferred over secular equivalents
  - "scripture" or "the Word" not "the Bible" (in reverent contexts)
  - "congregation" not "audience" or "crowd"
  - "prayer" not "wish" or "hope" (in spiritual contexts)
- Maintain the preacher's emphasis and emotion
- Keep sentences flowing naturally for text-to-speech
- Preserve scripture references exactly (e.g., "John 3:16")
- Do NOT add content — translate faithfully with biblical vocabulary
- Output ONLY the English translation, nothing else
```

### 4. Text-to-Speech (English)

**Primary: ElevenLabs Streaming TTS**
- Ultra-low latency streaming mode
- Clone or select a warm, authoritative male/female voice
- Model: `eleven_turbo_v2_5` (fastest) or `eleven_multilingual_v2` (highest quality)
- Stream audio chunks directly to output device

**Fallback: OpenAI `gpt-4o-mini-tts`**
- Good quality, steerable voice
- Higher latency than ElevenLabs streaming
- Use as backup if ElevenLabs has issues

### 5. Audio Output

Play generated audio to Dante Via's virtual output device → routes to Dante network → church speakers/headphones/translation receivers.

## Language Choice: **Python**

**Why Python for this project:**
- 1-minute latency tolerance makes Python's speed perfectly fine
- Best SDK support: `openai`, `elevenlabs`, `sounddevice` all have mature Python libs
- Rapid iteration — this is a specialized tool, not a high-throughput service
- Easy for church tech volunteers to understand/maintain
- `sounddevice` + `numpy` handle real-time audio capture/playback well

Go/Rust would add compilation complexity for zero meaningful benefit — the bottleneck is API round-trips (seconds), not compute (microseconds).

## Project Structure

```
church-translator/
├── README.md
├── PLAN.md
├── requirements.txt
├── .env.example
├── config.yaml                 # Voice settings, chunk size, device selection
├── src/
│   ├── __init__.py
│   ├── main.py                 # Entry point — orchestrates pipeline
│   ├── audio_capture.py        # Dante/system audio input capture
│   ├── audio_playback.py       # Audio output to Dante/system device
│   ├── transcriber.py          # OpenAI STT (Ukrainian → text)
│   ├── translator.py           # GPT-4o translation + biblical styling
│   ├── synthesizer.py          # ElevenLabs/OpenAI TTS
│   ├── pipeline.py             # Async pipeline coordinator
│   ├── vad.py                  # Voice Activity Detection for smart chunking
│   └── config.py               # Configuration loader
├── prompts/
│   └── biblical_translator.txt # System prompt for translation
├── scripts/
│   ├── list_devices.py         # List available audio devices
│   ├── test_capture.py         # Test audio capture
│   ├── test_tts.py             # Test TTS output
│   └── install_windows.ps1     # Windows setup script
└── tests/
    ├── test_translator.py
    ├── test_vad.py
    └── test_pipeline.py
```

## Configuration

```yaml
# config.yaml
audio:
  input_device: "Dante Via Input"    # or device index
  output_device: "Dante Via Output"  # or device index
  sample_rate: 48000                  # Dante standard
  channels: 1                         # Mono for speech
  chunk_duration_sec: 8               # Audio chunk size for STT

transcription:
  model: "gpt-4o-transcribe"
  language: "uk"                      # Ukrainian ISO 639-1

translation:
  model: "gpt-4o"
  temperature: 0.3                    # Low creativity, faithful translation
  prompt_file: "prompts/biblical_translator.txt"

synthesis:
  provider: "elevenlabs"              # or "openai"
  elevenlabs:
    model: "eleven_turbo_v2_5"
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam — warm male voice
    stability: 0.7
    similarity_boost: 0.8
  openai:
    model: "gpt-4o-mini-tts"
    voice: "onyx"                     # Deep male voice

pipeline:
  max_latency_sec: 30
  overlap_sec: 1                      # Overlap between chunks for context
  buffer_silence_sec: 2               # Silence before flushing buffer
```

## Dependencies

```
openai>=1.60.0
elevenlabs>=1.15.0
sounddevice>=0.5.0
numpy>=1.26.0
pyyaml>=6.0
webrtcvad>=2.0.10        # Voice activity detection
python-dotenv>=1.0.0
```

## Deployment

### Windows Setup
1. Install Python 3.11+ from python.org
2. Install Dante Via (from Audinate, $50 license)
3. Configure Dante Via: route preacher channel → app input, app output → translation channel
4. Clone repo, `pip install -r requirements.txt`
5. Copy `.env.example` → `.env`, add API keys
6. Edit `config.yaml` for device names
7. Run `python scripts/list_devices.py` to verify devices
8. Run `python src/main.py`

### Linux Setup
1. Install Python 3.11+
2. Option A: Dante Virtual Soundcard (if licensed for Linux)
3. Option B: AES67 mode on Dante device → Linux AES67 kernel driver → ALSA device
4. Same steps 4-8 as Windows

## Cost Estimate (per service)

| Component | Rate | Per Hour (est.) |
|-----------|------|-----------------|
| `gpt-4o-transcribe` | $0.006/min audio | $0.36/hr |
| GPT-4o translation | ~$0.01/1K tokens | ~$0.50/hr (est. 50K tokens/hr) |
| ElevenLabs TTS | $0.30/1K chars (Creator plan) | ~$1.50/hr |
| **Total** | | **~$2.36/hr** |

A typical 1.5hr service ≈ **$3.50**. Very affordable.

## Phases

### Phase 1: Core Pipeline (MVP) — 2-3 days
- [ ] Audio capture from system device
- [ ] Chunked STT with OpenAI
- [ ] Translation with biblical prompt
- [ ] TTS with ElevenLabs
- [ ] Audio playback to system device
- [ ] Basic CLI with start/stop

### Phase 2: Intelligence — 1-2 days
- [ ] VAD-based smart chunking (sentence boundaries)
- [ ] Context windowing (send previous sentence for translation continuity)
- [ ] Silence detection (pause TTS during silence)
- [ ] Overlap handling (no repeated words)

### Phase 3: Production Hardening — 1-2 days
- [ ] Auto-reconnect on API failures
- [ ] Fallback TTS (OpenAI if ElevenLabs fails)
- [ ] Audio level monitoring (warn if input too quiet/loud)
- [ ] Simple GUI (optional — tkinter or web dashboard)
- [ ] Logging and diagnostics

### Phase 4: Polish — 1 day
- [ ] Windows installer script
- [ ] Config validation on startup
- [ ] Latency monitoring dashboard
- [ ] Documentation for church tech team
