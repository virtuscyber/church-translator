# Church Live Translator — Product Roadmap

_Last updated: 2026-02-16_

## Vision
An open-source, self-hosted live translation system for churches. Any church can deploy it with their own API keys, select their audio source (Dante, USB mic, system audio), pick source/target languages, and get real-time translated audio + subtitles for their congregation.

---

## Phase 1 — Core Pipeline ✅ DONE
- [x] Ukrainian → English translation pipeline
- [x] OpenAI STT (gpt-4o-transcribe)
- [x] GPT-4o translation with biblical context prompt
- [x] ElevenLabs + OpenAI TTS
- [x] VAD smart chunking (sentence-boundary detection)
- [x] Streaming pipeline (4.2x speedup, ~2s per-chunk latency)
- [x] Web dashboard with dark theme
- [x] File-based testing (`scripts/test_file.py`)

## Phase 2 — Critical Hardening ✅ DONE
- [x] Fix streaming deadlock on dropped sequences
- [x] Dashboard auth (API key) + localhost binding
- [x] Skip TTS in test mode (no wasted credits)
- [x] API key hygiene (.gitignore, .env.example)

---

## Phase 3 — Open Source Ready 🔨 IN PROGRESS

### 3.1 Audio Source/Output Selector (#1) 🔨
- [ ] Enumerate available audio input devices (system, Dante virtual soundcard, USB mic)
- [ ] Enumerate available audio output devices (Dante output, speakers, headphones)
- [ ] Dropdown selectors in dashboard UI
- [ ] "Test" button — plays a short tone through selected output
- [ ] Persist selection across sessions (saved to config)
- [ ] Fallback: if selected device disappears, show warning + revert to default

### 3.2 Language Selector (#2) 🔨
- [ ] Source language dropdown (Ukrainian, Spanish, Korean, Mandarin, Russian, Portuguese, French, Arabic, etc.)
- [ ] Target language dropdown (same options)
- [ ] Auto-update STT model/prompt based on source language
- [ ] Custom vocabulary/prompt field (biblical terms, church-specific names)
- [ ] Language pair saved to config

### 3.3 First-Run API Key Wizard (#4) 🔨
- [ ] On first launch (no .env), show setup wizard in dashboard
- [ ] Step 1: Enter OpenAI API key → test connectivity → save
- [ ] Step 2: Enter ElevenLabs API key (optional) → test → save
- [ ] Step 3: Select default language pair
- [ ] Keys stored in local `.env` (never transmitted)
- [ ] Settings page to update keys later
- [ ] Clear instructions + links to get API keys

---

## Phase 4 — Voice & Quality

### 4.1 Voice Selection (#3)
- [ ] List available ElevenLabs voices in UI
- [ ] Preview button (short sample of each voice)
- [ ] OpenAI TTS voice options as fallback
- [ ] Support for cloned voices (upload audio sample)
- [ ] Voice selection saved per language pair

### 4.2 Latency Tuning Panel (#9)
- [ ] Quality ↔ Speed slider
- [ ] Controls: chunk size, VAD sensitivity, model selection
- [ ] Real-time latency graph in dashboard
- [ ] Presets: "Sunday Service" / "Bible Study" / "Conference" (#10)

---

## Phase 5 — Production Features

### 5.1 Live Transcript Display (#5)
- [ ] Side-by-side original + translation (upgrade existing)
- [ ] Export to SRT/VTT subtitles
- [ ] Integration guide for OBS, vMix, ProPresenter overlay
- [ ] Searchable transcript during live session

### 5.2 Cost Estimator (#6)
- [ ] Estimated $/hour based on current settings (before starting)
- [ ] Running cost counter during live session
- [ ] Session summary: total cost, chunks processed, duration
- [ ] Monthly cost projection

### 5.3 Multi-Output Routing (#7)
- [ ] Route translated audio to multiple devices simultaneously
- [ ] Per-output volume control
- [ ] Use case: earpieces + overflow room + livestream mix

### 5.4 Session Recording & Logs (#8)
- [ ] Auto-save transcript + translated audio per session
- [ ] Browse past sessions in dashboard
- [ ] Export for church archives (PDF transcript, audio files)
- [ ] Session metadata: date, language pair, duration, cost

---

## Phase 6 — Polish & Community

### 6.1 Deployment & Docs
- [ ] One-click install script (Docker or native)
- [ ] Raspberry Pi support (budget-friendly hardware)
- [ ] Comprehensive README with screenshots
- [ ] Video tutorial: "Set up live translation for your church in 15 minutes"
- [ ] Troubleshooting guide (Dante setup, ffmpeg, API issues)

### 6.2 Reliability
- [ ] Retry/backoff on API failures (STT, translation, TTS)
- [ ] Circuit breaker pattern
- [ ] Health monitoring + alerting
- [ ] Graceful degradation (if TTS fails, still show transcript)

### 6.3 Community
- [ ] GitHub release with changelog
- [ ] Contributing guide
- [ ] Feature request / issue templates
- [ ] Discord or Discussions for church tech teams

---

## Priority Order
1. **Phase 3** — Audio selector, language selector, API key wizard (minimum for other churches)
2. **Phase 4** — Voice selection, quality tuning
3. **Phase 5** — Transcripts/subtitles, cost tracking, multi-output
4. **Phase 6** — Deployment, reliability, community
