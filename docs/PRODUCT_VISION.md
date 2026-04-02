# Babel — Real-Time Translation for Houses of Worship

> *"Now the whole earth had one language and the same words."* — Genesis 11:1

**Product Vision Document**
**Date:** March 15, 2026
**Author:** Virtus Cybersecurity LLC / Dev Squad
**Status:** Research & Ideation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Market Landscape](#2-market-landscape)
3. [Hardware Device](#3-hardware-device)
4. [Cloud Infrastructure](#4-cloud-infrastructure)
5. [Mobile App](#5-mobile-app)
6. [System Architecture](#6-system-architecture)
7. [Business Model](#7-business-model)
8. [Development Roadmap](#8-development-roadmap)
9. [Open Questions](#9-open-questions)

---

## 1. Executive Summary

**The Problem:** Multilingual churches and nonprofits need affordable, reliable real-time audio translation. Current solutions are either too expensive ($50–$100+/hour for platforms like KUDO and Wordly), too complex (requiring trained interpreters), or too limited (text-only captions on phones, no audio output).

**The Opportunity:** We've already built a working translation pipeline (`church-translator`) that delivers 5–8 second latency at ~$2–3/hour using OpenAI STT, GPT-4o translation, and ElevenLabs TTS. The next step is to productize this into a turnkey service: a **physical device** that plugs into any sound system + a **mobile app** for listeners + a **cloud backend** that handles the AI pipeline.

**The Vision:** A church tech volunteer opens a box, plugs the device into their soundboard's aux output, connects to Wi-Fi, and within minutes the congregation is hearing real-time translated audio on their phones. No interpreters needed. No software to install on church computers. No ongoing technical maintenance.

---

## 2. Market Landscape

### Existing Competitors

| Product | Model | Audio Output | Pricing | Church-Specific | Key Limitation |
|---------|-------|-------------|---------|-----------------|----------------|
| **KUDO AI** | Cloud SaaS | ✅ AI voice + human interpreters | ~$100/hr; yearly license | ✅ Houses of Worship page | Expensive, complex admin UI, steep for small churches |
| **Wordly** | Cloud SaaS | ✅ AI voice | Per-hour packages, nonprofit discounts | ✅ Church Translation page | High cost, no dedicated hardware, BYOD-only |
| **LiveVoice** | Cloud SaaS | ✅ AI voice + live audio relay | From $10/day | ✅ Church page | Limited voice quality, browser-based only |
| **Interprefy** | Cloud + human interpreters | ✅ Human voice | Custom quotes | ❌ Enterprise focus | Requires human interpreters, very expensive |
| **Polyglossia** | Cloud SaaS | ✅ AI voice | More affordable | ✅ | Newer, limited reviews, unclear reliability |
| **OneAccord** | Cloud SaaS | ✅ AI voice | Mid-range | ✅ | Voice output quality concerns |
| **Timekettle X1** | Hardware hub | ✅ Via earbuds | $699 device + subscription | ❌ Business meetings | 1:1 interpretation, not broadcast |
| **Traditional** | Human interpreters | ✅ | $50–150/hr per language | ❌ | Expensive, scheduling complexity, limited languages |

### Market Gaps We Can Fill

1. **No turnkey hardware solution** — Every competitor is software-only (BYOD). Churches must figure out audio routing themselves. No one ships a plug-and-play device.
2. **Price** — KUDO/Wordly are $50–100+/hr. Our pipeline runs at $2–3/hr. Even with margins, we can be dramatically cheaper.
3. **Biblical language quality** — Our custom translation prompt produces natural, reverent translations. Competitors use generic models.
4. **Voice consistency** — We can offer consistent, recognizable voices per-language (pastor's "English voice" is always the same). Competitors often produce robotic or inconsistent output.
5. **No phone required for primary output** — Our device can output translated audio directly to speakers/headsets, not just phone screens.

### Target Customers

- **Primary:** Multilingual churches (50–5,000 members), especially immigrant/diaspora communities
- **Secondary:** Nonprofits, community centers, conferences, municipal meetings
- **Tertiary:** Schools, hospitals with multilingual populations

---

## 3. Hardware Device

### Requirements

The device must:
- Accept audio input from any sound system (XLR, 1/4" TRS, 3.5mm, or Dante/AES67)
- Connect to the internet (Wi-Fi or Ethernet)
- Stream audio to the cloud for processing
- Receive translated audio back and output it (to the mobile app, and optionally to local speakers/transmitters)
- Be small, quiet, rack-mountable or shelf-stable
- Require zero ongoing maintenance from church staff
- Auto-update firmware OTA
- Boot and connect automatically on power-up

### Option A: Raspberry Pi-Based (Recommended for MVP)

**Platform:** Raspberry Pi 5 (or Compute Module 5)

| Component | Specification | Est. Cost |
|-----------|--------------|-----------|
| Raspberry Pi 5 (4GB) | Quad-core Arm Cortex-A76, Wi-Fi 5, BT 5.0, Gigabit Ethernet | $60 |
| Audio HAT (e.g., HiFiBerry DAC+ ADC Pro) | Balanced XLR/TRS input + line output, 192kHz/24-bit | $65 |
| Custom carrier board (future) | Audio I/O, status LEDs, power management | $15–25 at scale |
| Enclosure (injection-molded, custom) | Compact, ventilated, rack-ear option | $8–15 at scale |
| SD card (industrial grade, 32GB) | Endurance-rated for continuous operation | $12 |
| Power supply (USB-C PD, 27W) | Reliable, surge-protected | $15 |
| **Total BOM (prototype)** | | **~$175** |
| **Total BOM (at 1,000 units)** | With custom PCB and bulk pricing | **~$85–110** |

**Software stack on device:**
- Minimal Linux (DietPi or custom Yocto image)
- Python agent: captures audio, streams to cloud via WebSocket, plays back translated audio
- Auto-provisioning: device boots → connects to Wi-Fi (configured via mobile app BLE pairing) → registers with cloud → ready
- Watchdog + health reporting to cloud dashboard
- OTA update system (Mender.io or custom)

**Pros:** Rich Linux ecosystem, proven audio HATs, sufficient CPU for local VAD/preprocessing, fast time-to-market, easy to develop on.

**Cons:** Pi availability can be spotty (improving), higher power draw than embedded, overkill CPU for what's essentially an audio relay.

### Option B: ESP32-S3 Based (Cost-Optimized)

For a future cost-optimized version:

| Component | Specification | Est. Cost |
|-----------|--------------|-----------|
| ESP32-S3-WROOM-1 module | Dual-core 240MHz, Wi-Fi, BLE 5, 8MB PSRAM | $4 |
| Audio codec (e.g., ES8388 or WM8960) | I2S stereo ADC/DAC, line in/out | $2 |
| Custom PCB + connectors | XLR/TRS input, 3.5mm output, Ethernet (optional), status LEDs | $8–12 |
| Enclosure | Compact injection-molded | $5–8 |
| Power supply | 5V USB-C or barrel jack | $5 |
| **Total BOM (at 1,000 units)** | | **~$30–45** |

**Pros:** Dramatically cheaper, lower power, smaller form factor, Wi-Fi built-in, good I2S audio support.

**Cons:** Limited processing power (all AI must be cloud-side), tighter memory constraints, harder to debug, longer development cycle, no native Ethernet without add-on.

### Option C: Off-the-Shelf + Software

Use an existing device as the "hardware":
- **Raspberry Pi 5 in a generic case** with a USB audio interface — quickest to market
- **Intel NUC / mini PC** — overkill but zero hardware risk
- **Android TV box / Fire TV Stick** — cheap but limited audio I/O

**Recommendation:** Start with **Option A** (Raspberry Pi 5 + audio HAT) for the first 100–500 units. This gets us to market fast with a reliable, developer-friendly platform. Transition to **Option B** (custom ESP32 board) at scale (1,000+ units) to reduce unit cost. Have a manufacturer like **PCBWay, Seeed Studio, or JLCPCB** do the custom PCB + assembly for the production version.

### Contract Manufacturing Partners

| Manufacturer | Specialty | MOQ | Location |
|-------------|-----------|-----|----------|
| **Seeed Studio** | IoT devices, Raspberry Pi ecosystem, Fusion PCBA | 50+ | Shenzhen, CN |
| **PCBWay** | PCB fab + assembly, small batch friendly | 5+ | Shenzhen, CN |
| **JLCPCB** | Cheapest PCB fab, decent assembly | 2+ | Shenzhen, CN |
| **Makerfabs** | Custom embedded products, ESP32 specialty | 100+ | Shenzhen, CN |
| **US-based:** MacroFab, Tempo Automation | Domestic manufacturing, higher cost | Varies | TX / CA |

---

## 4. Cloud Infrastructure

### Architecture Overview

```
┌──────────────┐     WebSocket      ┌─────────────────────────────────────┐
│  Babel       │◄──────────────────►│          Cloud Backend              │
│  Device      │   audio upstream   │                                     │
│  (on-site)   │   audio downstream │  ┌─────────┐  ┌──────────┐        │
└──────────────┘                    │  │ STT     │→ │ Translate │        │
                                    │  │ (Whisper│  │ (GPT-4o) │        │
       ┌──────────┐   WebSocket     │  │  / DG)  │  └────┬─────┘        │
       │ Mobile   │◄───────────────►│  └─────────┘       │              │
       │ App      │  translated     │                ┌────▼─────┐       │
       │ (users)  │  audio stream   │                │ TTS      │       │
       └──────────┘                 │                │(ElevenLabs│       │
                                    │                │ / OpenAI) │       │
                                    │                └──────────┘       │
                                    │                                     │
                                    │  ┌─────────────────────────────┐   │
                                    │  │ Session Manager             │   │
                                    │  │ - Device registry           │   │
                                    │  │ - Listener connections      │   │
                                    │  │ - Language routing          │   │
                                    │  │ - Usage metering            │   │
                                    │  └─────────────────────────────┘   │
                                    └─────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **API Gateway** | AWS API Gateway or Cloudflare Workers | WebSocket support, global edge |
| **Compute** | AWS ECS Fargate or Fly.io | Auto-scaling, pay-per-use, GPU not needed (AI is API-based) |
| **WebSocket Server** | Node.js or Python (FastAPI + WebSockets) | Real-time bidirectional audio streaming |
| **AI Pipeline** | OpenAI Whisper (STT) → GPT-4o (translate) → ElevenLabs (TTS) | Our proven stack from `church-translator` |
| **Audio Delivery** | WebSocket binary frames or WebRTC | Low-latency audio to mobile clients |
| **Database** | PostgreSQL (Supabase or RDS) | Organizations, devices, subscriptions, usage |
| **Auth** | Clerk or Auth0 | Org-level auth, device provisioning tokens |
| **Monitoring** | Datadog or Grafana Cloud | Device health, pipeline latency, error rates |
| **CDN / Edge** | Cloudflare | Static assets, edge caching, DDoS protection |

### AI Provider Strategy

| Service | Primary | Fallback | Estimated Cost |
|---------|---------|----------|---------------|
| **STT** | OpenAI `gpt-4o-transcribe` | Deepgram Nova-3 | $0.006/min |
| **Translation** | OpenAI `gpt-4o` | Anthropic Claude | $0.01–0.03/min |
| **TTS** | ElevenLabs Turbo v2.5 | OpenAI `gpt-4o-mini-tts` | $0.02–0.04/min |
| **Total per minute** | | | **~$0.04–0.07/min ($2.40–4.20/hr)** |

### Scaling Considerations

- Each active session (1 device streaming) needs ~1 WebSocket connection upstream + N downstream (1 per listener language)
- AI API calls are the bottleneck, not compute — scale is limited by provider rate limits
- At 100 concurrent churches: ~100 STT streams, ~200–500 TTS streams (multi-language)
- Estimated cloud cost per church per month (4 services × 1.5hrs): ~$15–25 in AI API costs

### Multi-Tenancy

```
Organization (Church)
├── Subscription (plan, billing)
├── Devices[] (registered hardware)
│   ├── Device 1 (main sanctuary)
│   └── Device 2 (youth room)
├── Sessions[] (active translations)
│   ├── Session (Sunday AM service)
│   │   ├── Source: Ukrainian
│   │   ├── Targets: [English, Spanish, Russian]
│   │   └── Listeners: 47 connected
│   └── Session (Wednesday Bible Study)
├── Settings
│   ├── Voice preferences per language
│   ├── Glossary (custom terms, names)
│   └── Biblical translation style
└── Usage / Billing
```

---

## 5. Mobile App

### Platform

**React Native** (cross-platform iOS + Android) with Expo for rapid development.

**Why React Native over Flutter:**
- Larger hiring pool and community
- Better audio streaming libraries (`react-native-live-audio-stream`, `expo-av`)
- JavaScript/TypeScript aligns with our web backend
- Expo EAS for simplified builds and OTA updates

### Core Features

#### For Listeners (Congregation Members)

1. **Join a Session** — Scan QR code displayed on church screen or enter a room code
2. **Select Language** — Choose from available target languages
3. **Listen** — Translated audio streams to phone speaker or connected earbuds/headphones
4. **Read Along** — Optional live captions/subtitles in selected language
5. **Offline Mode** — Download sermon recordings with translations after service
6. **Volume & Playback** — Independent volume control, latency buffer adjustment

#### For Admins (Church Tech Team)

1. **Device Management** — Pair new devices, monitor status, configure audio inputs
2. **Session Control** — Start/stop translation sessions, select source/target languages
3. **Dashboard** — Live listener count, latency metrics, usage/cost tracking
4. **Settings** — Custom glossary (names, places, theological terms), voice selection, translation style
5. **Billing** — View usage, manage subscription, download invoices

### UX Flow (Listener)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Open App   │ →  │  Scan QR /   │ →  │   Select     │ →  │  Listening   │
│  or tap     │    │  Enter Code  │    │   Language    │    │  (streaming) │
│  notification│    │              │    │   🇬🇧 🇪🇸 🇺🇦  │    │  🔊 ▶ ━━━━   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### UX Flow (Admin)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Login      │ →  │  Dashboard   │ →  │  Start       │ →  │  Live View   │
│  (church    │    │  Devices: ✅  │    │  Session     │    │  47 listeners│
│   account)  │    │  Plan: Pro   │    │  UK → EN,ES  │    │  Latency: 6s │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Tech Stack (Mobile)

| Layer | Technology |
|-------|-----------|
| **Framework** | React Native + Expo (SDK 52+) |
| **Audio Playback** | `expo-av` for streaming, `react-native-audio-api` for low-latency |
| **WebSocket** | Native WebSocket API with reconnection logic |
| **State Management** | Zustand or Jotai (lightweight) |
| **UI Components** | Tamagui or NativeWind (Tailwind for RN) |
| **Navigation** | Expo Router (file-based) |
| **Push Notifications** | Expo Notifications (service starting alerts) |
| **BLE (device pairing)** | `react-native-ble-plx` |
| **Analytics** | PostHog or Mixpanel |

### Audio Streaming Protocol

The app receives translated audio via WebSocket:

```
1. App connects: wss://api.babel.church/listen?session=abc&lang=en
2. Server sends binary frames: Opus-encoded audio chunks (20ms frames)
3. App decodes and plays through audio buffer with jitter compensation
4. Heartbeat every 5s to maintain connection
5. Auto-reconnect with exponential backoff on disconnect
```

**Why Opus over MP3/AAC:**
- Lower latency (no codec startup delay)
- Better quality at low bitrates (24–48 kbps for speech)
- Open standard, no licensing
- Native WebRTC compatibility if we upgrade later

---

## 6. System Architecture

### End-to-End Flow

```
                    ON-SITE                                    CLOUD
┌──────────────────────────────┐     ┌──────────────────────────────────────────┐
│                              │     │                                          │
│  Sound System                │     │  1. Receive audio stream                 │
│  (pastor mic)                │     │  2. VAD + chunking (or on-device)        │
│       │                      │     │  3. STT (Whisper/Deepgram)               │
│       ▼                      │     │  4. Translate (GPT-4o)                   │
│  ┌──────────┐                │     │  5. TTS per target language (ElevenLabs) │
│  │  Babel   │  ── Wi-Fi ──────────►│  6. Stream translated audio to listeners │
│  │  Device  │                │     │                                          │
│  └──────────┘                │     └──────────────────────────────────────────┘
│       │                      │                    │
│  (optional local output      │                    │ WebSocket
│   to speakers/transmitters)  │                    ▼
│                              │     ┌──────────────────────────┐
└──────────────────────────────┘     │  Mobile App (listeners)  │
                                     │  - Stream translated     │
                                     │    audio per language     │
                                     │  - Optional captions     │
                                     └──────────────────────────┘
```

### Latency Budget

| Stage | Time | Notes |
|-------|------|-------|
| Audio capture + VAD chunk | 2–4s | Wait for speech pause or max chunk |
| Network upload | 0.1–0.3s | Compressed audio, small chunks |
| STT (Whisper) | 0.5–1.5s | With speculative early transcription |
| Translation (GPT-4o) | 0.3–0.8s | Short text, streaming response |
| TTS (ElevenLabs streaming) | 0.3–0.5s | Time to first audio byte |
| Network download to app | 0.1–0.3s | Opus stream, small frames |
| **Total end-to-end** | **3.3–7.4s** | **Target: under 8 seconds** |

### Reliability

- **Device → Cloud:** WebSocket with automatic reconnection + local audio buffering (30s ring buffer on device so no audio is lost during brief disconnects)
- **Cloud → App:** WebSocket with jitter buffer (500ms) + reconnection
- **AI Provider Failover:** If OpenAI is down → fall back to Deepgram (STT) + Claude (translation) + OpenAI TTS
- **Offline Fallback:** Device stores audio locally; cloud processes backlog when connection restores (for recording, not live translation)

---

## 7. Business Model

### Pricing Tiers

| Plan | Price | Includes | Target |
|------|-------|----------|--------|
| **Starter** | $29/mo | 1 device, 2 languages, 20 hrs/mo | Small church (<200 members) |
| **Growth** | $79/mo | 2 devices, 5 languages, unlimited hours | Mid-size church (200–1,000) |
| **Pro** | $149/mo | 5 devices, unlimited languages, priority support, custom voices | Large church (1,000+) |
| **Enterprise** | Custom | Unlimited devices, SLA, dedicated support, on-prem option | Mega-church, denomination-wide |

### Hardware Pricing

| Model | Price | Includes |
|-------|-------|----------|
| **Babel One** (Pi 5 based) | $299 | Device + 3 months Starter plan |
| **Babel Pro** (custom PCB, future) | $199 | Device + 1 month any plan |
| **BYOD Kit** (software-only) | $0 | Use with own Raspberry Pi + audio HAT |

### Unit Economics (at Growth plan, $79/mo)

| Item | Monthly Cost |
|------|-------------|
| AI API costs (est. 30 hrs usage) | ~$15 |
| Cloud infrastructure (allocated) | ~$5 |
| Support (allocated) | ~$3 |
| **Gross margin** | **~$56 (71%)** |

### Revenue Projections

| Milestone | Churches | MRR | ARR |
|-----------|---------|-----|-----|
| Launch + 6 months | 50 | $3,950 | $47K |
| Year 1 | 200 | $15,800 | $190K |
| Year 2 | 1,000 | $79,000 | $948K |
| Year 3 | 5,000 | $395,000 | $4.7M |

There are an estimated **380,000+ churches in the US alone**, with ~35% serving multilingual communities. Even capturing 1% of the addressable market = 1,300+ subscribers.

---

## 8. Development Roadmap

### Phase 1: MVP (Months 1–3)
**Goal:** Working hardware + cloud + app for 10 beta churches

- [ ] Refactor `church-translator` pipeline into cloud-deployable microservices
- [ ] Build device agent (Pi 5): audio capture, WebSocket streaming, auto-provisioning
- [ ] Build minimal React Native app: join session, select language, listen
- [ ] Deploy cloud backend (Fly.io or AWS): session management, AI pipeline, audio relay
- [ ] Admin web dashboard: device management, session control
- [ ] 10 beta devices assembled and shipped to pilot churches

### Phase 2: Polish & Launch (Months 4–6)
**Goal:** Public launch with paid subscriptions

- [ ] App store submissions (iOS + Android)
- [ ] Stripe billing integration
- [ ] Custom glossary / terminology management
- [ ] Voice selection and preview
- [ ] QR code session joining
- [ ] Usage analytics and cost dashboard
- [ ] Marketing site and documentation
- [ ] FCC/CE compliance testing for device (if custom PCB)

### Phase 3: Scale (Months 7–12)
**Goal:** 200+ churches, custom hardware

- [ ] Custom PCB design (ESP32-S3 or CM5-based)
- [ ] Manufacturing partnership (Seeed Studio or similar)
- [ ] Multi-language simultaneous output
- [ ] Recording + on-demand playback
- [ ] Sermon archive with searchable translations
- [ ] API for integrations (church management software, live streaming)
- [ ] WebRTC upgrade for sub-3-second latency

### Phase 4: Expand (Year 2+)
**Goal:** Beyond churches

- [ ] Nonprofit / conference / municipal meeting support
- [ ] On-premise deployment option (all processing on local GPU server)
- [ ] Live captioning display (for projection screens)
- [ ] Sign language avatar integration
- [ ] White-label / OEM partnerships

---

## 9. Open Questions

### Technical
1. **Audio codec for mobile streaming** — Opus is ideal but may need native modules for React Native playback. Evaluate `react-native-audio-api` vs. raw WebAudio.
2. **WebSocket vs. WebRTC** — WebSocket is simpler to build and debug; WebRTC offers lower latency. Start with WebSocket, upgrade later?
3. **On-device VAD vs. cloud VAD** — Running VAD on the Pi reduces upstream bandwidth and latency. Cloud VAD is simpler but adds ~1s of network round-trip.
4. **Multi-language TTS fan-out** — When translating to 5 languages simultaneously, that's 5× the TTS API calls. Need to design for parallel execution and cost management.

### Hardware
5. **Custom PCB timeline** — At what volume does a custom board make financial sense? Likely 500+ units.
6. **Audio input flexibility** — Do we need XLR (pro) or is 3.5mm/USB sufficient for most churches? Survey needed.
7. **Dante/AES67 support** — Important for large churches with networked audio. Add as Pro feature?
8. **Enclosure design** — Rack-mount (1U?) vs. desktop "puck" vs. wall-mount. What do church tech teams prefer?

### Business
9. **Nonprofit pricing** — Should there be a free tier for churches under 100 members?
10. **Denomination partnerships** — Could we partner with denominations (e.g., Assemblies of God, SBC) for bulk deployment?
11. **Grant funding** — Many churches receive technology grants. Can we facilitate grant applications?
12. **White-label** — Would existing church AV companies (e.g., Shure, QSC, Allen & Heath) want to resell this?

### Regulatory
13. **FCC certification** — Required for any device with Wi-Fi/BLE radio. The Pi module is already FCC-certified, but a custom board needs its own certification (~$5–15K).
14. **Data privacy** — Audio is being processed in the cloud. Need clear privacy policy, GDPR compliance if serving EU churches, and option for no-recording mode.

---

## Appendix A: Competitive Positioning Matrix

```
                        LOW COST ◄─────────────────────────► HIGH COST
                            │                                    │
    SOFTWARE-ONLY           │                                    │
         ▲                  │    LiveVoice                       │
         │                  │    ($10/day)      Polyglossia      │
         │                  │                                    │
         │                  │         ★ Babel (app)              │  Wordly
         │                  │                                    │  ($50-100/hr)
         │                  │                                    │
         │                  │                                    │  KUDO
         │                  │                                    │  ($100/hr)
         │                  │                                    │
    HARDWARE                │                                    │
    INCLUDED                │                                    │
         │                  │    ★ Babel (device + app)          │
         │                  │    ($29-149/mo + $299 device)      │
         │                  │                                    │
         ▼                  │              Timekettle X1         │  Interprefy
                            │              ($699 + sub)          │  (custom)
                            │                                    │
```

**Our positioning:** The only solution that combines dedicated hardware + mobile app + affordable AI-powered translation. We sit in the "hardware included, low cost" quadrant where no competitor exists today.

---

## Appendix B: Working Product Name Options

| Name | Domain Available? | Notes |
|------|------------------|-------|
| **Babel** | babel.church — check | Biblical reference (Tower of Babel), instant recognition |
| **Pentecost** | pentecost.app — check | "They heard in their own language" (Acts 2) |
| **Tongues** | tongues.io — check | Biblical, but may have negative connotations |
| **OneVoice** | onevoice.church — check | Inclusive, clear meaning |
| **Unison** | unison.church — check | Together, one sound |

---

*This document is a living artifact. As research continues and decisions are made, it will be updated with specifics on hardware sourcing, cloud architecture details, and mobile app wireframes.*
