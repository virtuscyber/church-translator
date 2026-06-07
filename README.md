# Church Live Translation

Real-time multilingual translation for church services with biblical language styling. Audio flows from microphone through AI-powered speech recognition, translation, and synthesis — out to speakers, Dante network, or AES67 multicast.

## How It Works

```
Microphone → VAD Chunking → Whisper STT → GPT-4o Translation → Streaming TTS → Speakers / Dante / AES67
```

1. **Captures** the speaker's audio from a microphone, Dante Via, or VBCable
2. **Detects speech** using VAD (Voice Activity Detection) with smart sentence-boundary splitting
3. **Transcribes** using OpenAI's `gpt-4o-transcribe` with speculative early transcription
4. **Translates** to one or more target languages (GPT-4o + custom biblical prompt)
5. **Speaks** the translation via streaming ElevenLabs or OpenAI TTS
6. **Outputs** to local speakers, Dante Via, and/or AES67 multicast — simultaneously

### Supported Languages

Ukrainian, Russian, English, Spanish, Portuguese, French, German, Korean, Mandarin Chinese, Arabic, Polish, Romanian, Italian, Japanese, Hindi — with multi-language simultaneous output.

---

## Performance

| Metric | Value |
|--------|-------|
| **Typical end-to-end latency** | **5–8 seconds** |
| **Worst case latency** | ~12 seconds |
| **Estimated cost** | ~$2–3/hour |

### Optimization Pipeline

The system uses four layered optimizations to minimize latency:

| Optimization | How It Works | Savings |
|---|---|---|
| **VAD smart chunking** | Splits audio on natural speech pauses instead of fixed intervals. Smart force-splitting at max duration finds the best silence gap instead of cutting mid-word. | Faster, higher quality chunks |
| **Speculative STT** | Starts transcribing a preview snapshot while the speaker is still talking. If STT finishes before the VAD chunk completes, the transcription step is skipped entirely. | ~1–2s per chunk |
| **Streaming TTS** | Plays audio chunks as they arrive from the TTS API (~300ms for first audio) instead of waiting for full synthesis. | ~2–3s per chunk |
| **Concurrent processing** | Up to 3 chunks process STT + translate + TTS simultaneously, with an ordered playback system that ensures audio plays in the correct sequence. | Eliminates gaps between translations |

```
Pipeline visualization (concurrent mode):

Chunk 1: [STT][translate][TTS→slot1→🔊🔊🔊]
Chunk 2:   [STT][translate][TTS→slot2→buffer] → [🔊🔊🔊]
Chunk 3:       [STT][translate][TTS→slot3]        → [🔊🔊]
                 ↑ all running simultaneously
```

---

## Quick Start

No Docker, no terminal commands. Any tech volunteer can run this.

### 1. Download

```bash
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

The browser opens to the dashboard at **http://localhost:8085**. On first run, a setup wizard walks you through:
- Entering your **OpenAI API key** (required)
- Entering your **ElevenLabs API key** (optional — falls back to OpenAI TTS)
- Choosing source and target languages

### 4. Start Translating

Click **Start Live Translation**, and the system begins listening, translating, and speaking in real time.

### Adjusting settings live

The transcript and live controls stay front-and-center; settings, system health, and
the file-test tool are tucked into collapsible panels below. Change a setting and an
**Unsaved changes** indicator appears — click **Apply Live** to push it into the
running translation instantly with no audio gap:

- **Hot-swapped live:** translation/STT/TTS model, ElevenLabs voice, custom vocabulary,
  source/target language.
- **Quick auto-restart:** changing the microphone or speaker briefly restarts the audio
  stream (an open device can't be switched in place), then resumes automatically.

When nothing is running, **Apply** simply saves your settings for the next session.

### Reliability during a service

- **API hiccups** (rate limits, brief network drops) are retried automatically with
  backoff and per-request timeouts. A sustained failure (bad key, exhausted quota)
  is surfaced in the dashboard instead of silently dropping audio.
- **Microphone drop-outs** are detected within seconds — the dashboard alerts you and
  automatically re-opens the device when it comes back.
- **Export the transcript** any time with the **⬇ TXT** / **⬇ SRT** buttons on the
  transcript panel (handy for service records or captions).

---

## Alternative: Docker

> **Note:** Docker is for the dashboard and file-test workflow. Live microphone
> capture does **not** work in a container without explicit audio-device passthrough
> (`--device /dev/snd` plus a PulseAudio/ALSA bridge). For live translation, run
> natively (see Quick Start).

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

Detects your OS (macOS, Ubuntu/Debian, Fedora, Arch), installs prerequisites, and sets everything up.

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

## Audio Output Options

### Local Speakers (Default)

Set `output.mode: "sounddevice"` in `config.yaml`. Audio plays through the system's default output device or a specified device.

### Dante Via (Recommended for Dante Networks)

The simplest way to get translation audio onto a Dante network:

1. Install **Dante Via** on the translation laptop
2. Create a **Transmit** route in Dante Via
3. Set the output device in `config.yaml` to the Dante Via virtual output:
   ```yaml
   output:
     mode: "sounddevice"
     output_device: "Dante Via Transmit"
   ```
4. In **Dante Controller**, route the Dante Via device to your receivers (Williams Sound headsets, speakers, etc.)

### AES67 Multicast (Direct, No Dante Via)

For direct AES67 output without Dante Via:

1. Set output mode in `config.yaml`:
   ```yaml
   output:
     mode: "both"          # "sounddevice", "dante", or "both"
     stream_name: "Church Translation EN"
     multicast_address: "239.69.0.1"
     port: 5004
     ttl: 32
   ```
2. The app broadcasts a continuous AES67 RTP multicast stream with SAP/SDP announcements
3. In **Dante Controller**, enable AES67 mode and PTPv2 on the receiving device
4. The stream appears automatically — route it to your outputs

**Multi-language AES67:** Each target language gets its own multicast stream (`.1`, `.2`, `.3`...) with separate SAP announcements.

**Troubleshooting AES67:**
- Run `python scripts/diagnose_aes67.py` to verify multicast is working
- Run `python scripts/diagnose_dante.py` for Dante network diagnostics
- Check Windows Firewall — multicast UDP is commonly blocked
- Ensure the translation laptop and Dante devices are on the same VLAN/subnet

---

## Configuration

Edit `config.yaml`:

### Audio

| Setting | Default | Description |
|---------|---------|-------------|
| `audio.input_device` | system default | Input device name or index |
| `audio.output_device` | system default | Output device name or index |
| `audio.sample_rate` | 48000 | Audio sample rate (Hz) |

### Pipeline / VAD

| Setting | Default | Description |
|---------|---------|-------------|
| `pipeline.use_vad` | true | Use voice activity detection (recommended) |
| `pipeline.vad_aggressiveness` | 2 | Noise filtering 0–3 (2 = typical church, 3 = noisy) |
| `pipeline.min_chunk_sec` | 2.0 | Minimum speech before emitting a chunk |
| `pipeline.max_chunk_sec` | 8.0 | Force-split even during continuous speech |
| `pipeline.silence_threshold_sec` | 0.6 | Silence duration to trigger chunk boundary |
| `pipeline.context_sentences` | 2 | Previous sentences fed as translation context |

### Models

| Setting | Default | Description |
|---------|---------|-------------|
| `transcription.model` | gpt-4o-transcribe | OpenAI STT model |
| `translation.model` | gpt-4o | Translation model |
| `synthesis.provider` | elevenlabs | `"elevenlabs"` or `"openai"` |

### Reliability / Anti-Hallucination

Speech-to-text models invent confident text when fed silence, noise, or music
("thank you for watching", subtitle credits, looping phrases). These guards
suppress that so phantom sentences never get spoken aloud:

| Setting | Default | Description |
|---------|---------|-------------|
| `transcription.temperature` | 0.0 | Deterministic STT; lowest hallucination |
| `transcription.gate_silence` | true | Skip near-silent chunks before STT (the biggest lever) |
| `transcription.silence_peak` | 0.008 | Peak amplitude (0–1) below which a chunk is treated as silence. Lower it if quiet speech is being dropped |
| `transcription.min_duration_sec` | 0.4 | Drop chunks shorter than this |
| `transcription.filter_hallucinations` | true | Drop known artifacts and repetition loops (STT **and** translation) |
| `translation.temperature` | 0.0 | Faithful, deterministic translation |
| `translation.filter_hallucinations` | true | Reject junk input and hallucinated output; keep it out of the context window |

If legitimate quiet speech is being skipped, lower `silence_peak` (e.g. `0.004`)
or set `gate_silence: false`.

### Output

| Setting | Default | Description |
|---------|---------|-------------|
| `output.mode` | sounddevice | `"sounddevice"`, `"dante"`, or `"both"` |
| `output.stream_name` | Church Translation EN | AES67 stream name (visible in Dante Controller) |
| `output.multicast_address` | 239.69.0.1 | AES67 multicast address |
| `output.port` | 5004 | AES67 RTP port |

### Tuning Latency vs Quality

```yaml
# Lower latency (faster but may split mid-sentence)
pipeline:
  min_chunk_sec: 1.5
  max_chunk_sec: 5.0
  silence_threshold_sec: 0.4

# Higher quality (longer chunks = better translation context)
pipeline:
  min_chunk_sec: 3.0
  max_chunk_sec: 10.0
  silence_threshold_sec: 0.8
```

---

## Biblical Language

The translation prompt is **fidelity-first**: it translates only what is
actually said and never invents, completes, or embellishes content. Reverent,
liturgical vocabulary is applied only where it fits naturally and does not
change the meaning:
- "brethren" for an address to the congregation
- "Scripture" / "the Word", "the Lord", "grace", "mercy", "repentance"
- Scripture references and proper nouns preserved exactly
- Empty output when the audio contains no real speech

See `prompts/biblical_translator.txt` to customize the translation style.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Microphone  │───▶│  VAD Smart  │───▶│ Speculative │───▶│  GPT-4o     │───▶│  Streaming   │
│  / Dante Via │    │  Chunking   │    │  STT        │    │  Translation │    │  TTS         │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────┬───────┘
                                                                                    │
                   ┌────────────────────────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  Ordered Playback   │───▶ Local Speakers
        │  (concurrent slots) │───▶ Dante Via
        │                     │───▶ AES67 Multicast
        └─────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **VAD Chunker** | `src/vad_chunker.py` | Energy-based VAD with adaptive noise floor, smart force-splitting at silence gaps |
| **VAD Capture** | `src/vad_capture.py` | Async audio capture with preview snapshots for speculative STT |
| **Transcriber** | `src/transcriber.py` | OpenAI Whisper STT wrapper |
| **Translator** | `src/translator.py` | GPT-4o translation with context windowing |
| **Synthesizer** | `src/synthesizer.py` | Streaming TTS via ElevenLabs or OpenAI (batch + stream modes) |
| **AES67 Sender** | `src/aes67_output.py` | Continuous RTP multicast with SAP/SDP announcements |
| **Audio Playback** | `src/audio_playback.py` | Local speaker output via sounddevice |
| **Dashboard** | `dashboard/server.py` | Web UI, API, pipeline orchestration, WebSocket live updates |

### Diagnostic Tools

| Script | Description |
|--------|-------------|
| `scripts/diagnose_aes67.py` | Verify multicast send/receive, check firewall, inspect network interfaces |
| `scripts/diagnose_dante.py` | Dante/AES67 network diagnostics |
| `scripts/list_devices.py` | List available audio input/output devices |
| `scripts/test_aes67.py` | Test AES67 stream output |

---

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

Tests cover imports, config, dashboard API, VAD pipeline, anti-hallucination filters (silence gating, artifact and repetition-loop detection), audio device handling, WebSocket error handling, launcher scripts, and AES67 output.

---

## License

MIT

---

**Developed by [Virtus Cybersecurity](https://virtuscyber.com) — Bogdan Salamakha**
