<div align="center">

# ⛪ Church Live Translation

**Real-time, multilingual sermon translation with biblical language styling.**

Microphone → AI speech recognition → translation → natural speech — out to local speakers, Dante, or AES67 multicast, all at once.

[![CI](https://github.com/virtuscyber/church-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/virtuscyber/church-translator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Mode](https://img.shields.io/badge/mode-real--time-orange)
![Output](https://img.shields.io/badge/output-Speakers%20%7C%20Dante%20%7C%20AES67-8a2be2)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Contents

- [What it does](#-what-it-does)
- [How it works](#-how-it-works)
- [Capabilities](#-capabilities)
- [Voice models](#-voice-models)
- [Quick start](#-quick-start)
- [Using the dashboard](#-using-the-dashboard)
- [Audio output (Speakers / Dante / AES67)](#-audio-output)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Testing](#-testing)

---

## ✨ What it does

A volunteer points a microphone at the preacher, opens a browser, and clicks **Start**. The congregation hears a natural-sounding translation in their language seconds later — through headsets, speakers, or a Dante/AES67 audio network.

- 🎙️ **Any audio source** — USB mic, laptop mic, Dante Via, or VBCable
- 🌍 **15 languages**, with simultaneous multi-language output
- 📖 **Fidelity-first biblical styling** — translates only what is said, never invents
- 🖥️ **Zero-terminal setup** — double-click a launcher, finish a browser wizard
- 🔊 **Pro audio out** — local speakers, Dante Via, and drift-free AES67 multicast

---

## 🔄 How it works

```
                          ┌──────────────────────── two STT modes ────────────────────────┐
  🎙️  Microphone ─────────┤                                                                │
  (USB / Dante Via)       │   ⚡ Streaming   → persistent WebSocket, live interim words    │
                          │   📦 Chunked     → VAD splits on pauses + speculative STT      │
                          └───────────────────────────────┬───────────────────────────────┘
                                                           ▼
                                            🧠  Speech-to-text
                                   (Scribe v2 · Deepgram Nova-3 · OpenAI)
                                                           ▼
                                       🌐  GPT-4o translation (+ biblical prompt)
                                                           ▼
                                        🔊  Streaming TTS (ElevenLabs Flash v2.5)
                                                           ▼
                                ┌──────────────────────────┴──────────────────────────┐
                                ▼                          ▼                           ▼
                          🔈 Local speakers          🎚️ Dante Via              📡 AES67 multicast
```

The default stack is a **full ElevenLabs pipeline**: Scribe v2 STT → GPT-4o translation → Flash v2.5 TTS — every stage swappable from the dashboard.

### Supported languages

Ukrainian · Russian · English · Spanish · Portuguese · French · German · Korean · Mandarin · Arabic · Polish · Romanian · Italian · Japanese · Hindi

---

## 🚀 Capabilities

| Area | What you get |
|---|---|
| **Two STT modes** | **⚡ True streaming** (words appear as the preacher speaks) or **📦 chunked** (VAD + speculative STT). Switch per service. |
| **3 streaming engines** | Deepgram Nova-3, ElevenLabs Scribe v2 Realtime, OpenAI `gpt-realtime-whisper` — behind one interface. |
| **3 STT providers (chunked)** | ElevenLabs Scribe v2 (best Ukrainian accuracy), Deepgram Nova-3, OpenAI `gpt-4o-transcribe` — with automatic OpenAI fallback. |
| **Anti-hallucination** | Silence gating, known-artifact filtering, repetition-loop detection, source-language anchoring — so dead air never becomes phantom text. |
| **Live tuning** | Adjust quality, VAD, reliability, TTS speed, and AES67 settings **mid-service** with no restart. |
| **Resilience** | Bounded retries with backoff, per-request timeouts, and sustained-failure alerts surfaced in the dashboard. |
| **Mic-drop recovery** | A watchdog detects a stalled/unplugged device within seconds and auto-reconnects. |
| **Transcript export** | One-click **TXT** / **SRT** download for service records or captions. |
| **Pro audio output** | Local speakers, Dante Via, and a **drift-free AES67** RTP multicast (SAP/SDP announced). |
| **Multi-language out** | One source → several target-language streams simultaneously. |

---

## 🎛️ Voice models

The defaults track the best current real-time models, and every choice is a dropdown in the dashboard.

| Stage | Default | Alternatives |
|---|---|---|
| **Speech-to-text** | **ElevenLabs Scribe v2** (≤5% Ukrainian WER) | Deepgram Nova-3 · OpenAI `gpt-4o-transcribe` *(OpenAI is the auto-fallback)* |
| **Streaming STT** | **Deepgram Nova-3** | ElevenLabs Scribe v2 Realtime · OpenAI `gpt-realtime-whisper` |
| **Translation** | **GPT-4o** | `gpt-4o-mini` · `gpt-4.1-mini` · `gpt-4.1-nano` |
| **Text-to-speech** | **ElevenLabs Flash v2.5** (~75 ms) | `eleven_v3` (most expressive) · Multilingual v2 · Turbo v2.5 · OpenAI `gpt-4o-mini-tts` |

> 💡 **⚡ True streaming** holds a socket open so the provider does the endpointing and **interim words show live**. Turn it off (or pick a chunked provider) to trade streaming latency for top accuracy. **TTS speed** is tunable live (~1.1–1.15 keeps the translation pacing the speaker).

---

## ⚡ Quick start

No Docker, no terminal. Any tech volunteer can run this.

**1. Get the code**

```bash
git clone https://github.com/virtuscyber/church-translator.git
```

**2. Launch**

| OS | Action |
|---|---|
| 🪟 Windows | double-click `start.bat` |
| 🍎 macOS | double-click `start.command` |
| 🐧 Linux | double-click `start.sh` (or `./start.sh`) |

The launcher checks Python 3.11+ and ffmpeg, creates a virtualenv, installs dependencies (first run only), starts the dashboard, and opens your browser.

**3. Finish the wizard** at **http://localhost:8085**

- 🔑 **OpenAI API key** — required (translation + fallback)
- 🔑 **ElevenLabs API key** — recommended (Scribe v2 STT + Flash TTS)
- 🔑 **Deepgram API key** — optional (`DEEPGRAM_API_KEY`, for Deepgram STT/streaming)
- 🌐 Source + target languages

**4. Click _Start Live Translation_** — it begins listening, translating, and speaking in real time.

<details>
<summary><b>Prerequisites (handled by the launcher / wizard)</b></summary>

- **Python 3.11+** — <https://www.python.org/downloads/>
- **ffmpeg** — `brew install ffmpeg` · `sudo apt install ffmpeg` · [Windows build](https://ffmpeg.org/download.html)
- **OpenAI API key** — <https://platform.openai.com/api-keys>
- **ElevenLabs API key** *(optional)* — <https://elevenlabs.io/>
- **Deepgram API key** *(optional)* — <https://deepgram.com/>

The dashboard's health panel tells you exactly what's missing and how to fix it.

</details>

<details>
<summary><b>Alternatives: Docker & one-line install</b></summary>

**Docker** (dashboard + file-test only — live mic needs native audio passthrough):

```bash
git clone https://github.com/virtuscyber/church-translator.git
cd church-translator
docker compose up -d           # → http://localhost:8085
docker compose logs -f         # watch logs
docker compose down            # stop
```

**One-line native install** (macOS, Ubuntu/Debian, Fedora, Arch):

```bash
curl -sL https://raw.githubusercontent.com/virtuscyber/church-translator/main/install.sh | bash
```

</details>

---

## 🖥️ Using the dashboard

The live transcript and controls stay front-and-center; **Settings**, **System health**, and the **file-test** tool are tucked into collapsible panels.

- **Apply live** — change the model, voice, vocabulary, or language and push it into the running translation with **no audio gap**. Mic/speaker changes do a quick seamless restart.
- **⚡ Live interim line** — in streaming mode, words appear greyed-in as they're recognized, then settle into the transcript.
- **🔧 Advanced tuning** — silence gate & threshold, VAD aggressiveness, chunk timing, hallucination filter, temperatures, **TTS speed**, API timeout/retries, mic-watchdog, and AES67 output — all applied instantly.
- **⬇ Export** — download the transcript as **TXT** or **SRT** at any time.

---

## 🔊 Audio output

Set `output.mode` to `sounddevice`, `dante`, or `both`.

<details open>
<summary><b>🔈 Local speakers</b> (default)</summary>

`output.mode: "sounddevice"` — plays through the system default or a chosen output device.

</details>

<details>
<summary><b>🎚️ Dante Via</b> (simplest for Dante networks)</summary>

1. Install **Dante Via** on the translation laptop and create a **Transmit** route.
2. Point the output device at it:
   ```yaml
   output:
     mode: "sounddevice"
     output_device: "Dante Via Transmit"
   ```
3. In **Dante Controller**, route the Dante Via device to your receivers (Williams Sound headsets, speakers, etc.).

</details>

<details>
<summary><b>📡 AES67 multicast</b> (direct, no Dante Via)</summary>

```yaml
output:
  mode: "both"          # "sounddevice", "dante", or "both"
  stream_name: "Church Translation EN"
  multicast_address: "239.69.0.1"
  port: 5004
  ttl: 32
```

The app broadcasts a continuous AES67 RTP stream with SAP/SDP announcements, **paced from an absolute clock so the 48 kHz rate stays drift-free** over a long service. In **Dante Controller**, enable AES67 + PTPv2 on the receiver and route the stream.

> Because this is a software source, set the receiver's latency/link-offset to **≥ 5 ms** to absorb scheduler jitter. Multi-language output uses one multicast stream per language (`.1`, `.2`, …).

**Troubleshooting:** `python scripts/diagnose_aes67.py` (multicast send/receive) · `python scripts/diagnose_dante.py` (Dante network) · check Windows Firewall (multicast UDP is often blocked) · keep all devices on the same VLAN/subnet.

</details>

---

## ⚙️ Configuration

Everything below lives in `config.yaml` (or the dashboard). Tables show the most useful knobs.

<details>
<summary><b>Speech-to-text & streaming</b></summary>

| Setting | Default | Description |
|---|---|---|
| `transcription.provider` | `elevenlabs` | `elevenlabs` (Scribe v2) · `deepgram` (Nova-3) · `openai` |
| `transcription.streaming` | `true` | True-streaming WebSocket path when the provider supports it |
| `transcription.language` | `uk` | Source language (ISO 639-1) |
| `transcription.elevenlabs_model` | `scribe_v2` | ElevenLabs chunked STT model |
| `transcription.deepgram_model` | `nova-3` | Deepgram model |
| `transcription.elevenlabs_realtime_model` | `scribe_v2_realtime` | ElevenLabs streaming model |
| `transcription.openai_realtime_model` | `gpt-realtime-whisper` | OpenAI streaming model |
| `transcription.model` | `gpt-4o-transcribe` | OpenAI chunked STT (and fallback) |

</details>

<details>
<summary><b>Translation & synthesis</b></summary>

| Setting | Default | Description |
|---|---|---|
| `translation.model` | `gpt-4o` | Translation model |
| `translation.temperature` | `0.0` | Faithful, deterministic translation |
| `synthesis.provider` | `elevenlabs` | `elevenlabs` or `openai` |
| `synthesis.speed` | `1.0` | Playback speed (ElevenLabs ~0.7–1.2; try 1.1–1.15) |
| `synthesis.elevenlabs.model` | `eleven_flash_v2_5` | TTS model |
| `synthesis.elevenlabs.voice_id` | Adam | ElevenLabs voice |

</details>

<details>
<summary><b>Pipeline / VAD (chunked mode)</b></summary>

| Setting | Default | Description |
|---|---|---|
| `pipeline.vad_aggressiveness` | `2` | Noise filtering 0–3 (2 = typical church, 3 = noisy) |
| `pipeline.min_chunk_sec` | `2.0` | Minimum speech before emitting a chunk |
| `pipeline.max_chunk_sec` | `8.0` | Force-split during continuous speech |
| `pipeline.silence_threshold_sec` | `0.6` | Silence that triggers a chunk boundary |
| `pipeline.context_sentences` | `2` | Previous sentences fed as translation context |

```yaml
# Lower latency (may split mid-sentence)        # Higher quality (more context)
pipeline:                                        pipeline:
  min_chunk_sec: 1.5                               min_chunk_sec: 3.0
  max_chunk_sec: 5.0                               max_chunk_sec: 10.0
  silence_threshold_sec: 0.4                       silence_threshold_sec: 0.8
```

</details>

<details>
<summary><b>Anti-hallucination</b></summary>

STT models invent confident text when fed silence, noise, or music ("thank you for watching", subtitle credits, looping phrases). These guards suppress that so phantom sentences are never spoken:

| Setting | Default | Description |
|---|---|---|
| `transcription.gate_silence` | `true` | Skip near-silent chunks before STT (**biggest lever**) |
| `transcription.silence_peak` | `0.008` | Peak (0–1) below which a chunk is "silence" — lower if quiet speech is dropped |
| `transcription.min_duration_sec` | `0.4` | Drop chunks shorter than this |
| `transcription.filter_hallucinations` | `true` | Drop known artifacts + repetition loops (STT **and** translation) |
| `transcription.temperature` | `0.0` | Deterministic STT; lowest hallucination |

</details>

---

## 🧩 Architecture

```
🎙️ Mic / Dante Via
        │
        ├── ⚡ Streaming  → streaming_stt.py (WebSocket) ──┐
        └── 📦 Chunked    → vad_capture.py → vad_chunker.py┤
                                                          ▼
                              transcriber.py  (Scribe v2 / Deepgram / OpenAI, + fallback)
                                                          ▼
                              translator.py   (GPT-4o + biblical prompt, context window)
                                                          ▼
                              synthesizer.py  (streaming Flash v2.5 / OpenAI TTS)
                                                          ▼
                              ordered playback (concurrent slots, in-sequence audio)
                                                          ▼
              audio_playback.py 🔈     ·     aes67_output.py 📡 (drift-free RTP + SAP/SDP)
```

| Component | File | Role |
|---|---|---|
| **Streaming STT** | `src/streaming_stt.py` | WebSocket engines (Deepgram / ElevenLabs Realtime / OpenAI) behind one interface |
| **VAD chunker** | `src/vad_chunker.py` | Energy VAD with adaptive noise floor + smart silence-gap splitting |
| **VAD capture** | `src/vad_capture.py` | Async capture, preview snapshots, raw-PCM streaming tap, device watchdog |
| **Transcriber** | `src/transcriber.py` | Multi-provider chunked STT with anti-hallucination + OpenAI fallback |
| **Translator** | `src/translator.py` | GPT-4o translation with rolling context |
| **Synthesizer** | `src/synthesizer.py` | Streaming/batch TTS (ElevenLabs + OpenAI) with retry/fallback |
| **AES67 sender** | `src/aes67_output.py` | Drift-free RTP multicast with SAP/SDP announcements |
| **Hallucination** | `src/hallucination.py` | Silence gate, artifact filter, repetition detection |
| **Dashboard** | `dashboard/server.py` | Web UI, API, pipeline orchestration, live WebSocket updates |

**Diagnostics:** `scripts/diagnose_aes67.py` · `scripts/diagnose_dante.py` · `scripts/list_devices.py` · `scripts/test_aes67.py`

---

## 📖 Biblical language

The translation prompt is **fidelity-first**: it renders only what is actually said and never invents, completes, or embellishes. Reverent vocabulary is applied only where it fits naturally — "brethren", "Scripture"/"the Word", "the Lord", "grace", "mercy", "repentance" — with Scripture references and proper nouns preserved exactly, and empty output when there's no real speech. Customize it in [`prompts/biblical_translator.txt`](prompts/biblical_translator.txt).

---

## 🧪 Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

Covers config, the dashboard API, the VAD pipeline, anti-hallucination filters, multi-provider STT + fallback, the three streaming engines, synthesis, live tuning, device recovery, transcript export, and AES67 output.

---

<div align="center">

**Developed by [Virtus Cybersecurity](https://virtuscyber.com) — Bogdan Salamakha**

MIT License

</div>
