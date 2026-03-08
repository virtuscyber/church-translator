#!/usr/bin/env python3
"""Dashboard web server for Church Live Translation.

Serves the dashboard UI and provides WebSocket endpoints for real-time
translation status, transcript updates, and pipeline control.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

from aiohttp import web

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Config Persistence ──────────────────────────────────────────

CONFIG_PATH = PROJECT_ROOT / "config.json"

SUPPORTED_LANGUAGES = [
    {"code": "uk", "name": "Ukrainian"},
    {"code": "ru", "name": "Russian"},
    {"code": "es", "name": "Spanish"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "ko", "name": "Korean"},
    {"code": "zh", "name": "Mandarin Chinese"},
    {"code": "ar", "name": "Arabic"},
    {"code": "pl", "name": "Polish"},
    {"code": "ro", "name": "Romanian"},
    {"code": "it", "name": "Italian"},
    {"code": "ja", "name": "Japanese"},
    {"code": "hi", "name": "Hindi"},
    {"code": "en", "name": "English"},
]


def load_saved_config() -> dict:
    """Load saved config from config.json."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load config.json, using defaults")
    return {}


def save_config(data: dict):
    """Save config to config.json (merges with existing)."""
    existing = load_saved_config()
    existing.update(data)
    CONFIG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ── Auth & CORS Middleware ──────────────────────────────────────

DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")
CORS_ORIGIN = os.getenv("DASHBOARD_CORS_ORIGIN", "")

# Paths that don't require auth
_PUBLIC_PATHS = {"/", "/ws"}


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Require Bearer token for API routes when DASHBOARD_API_KEY is set."""
    if DASHBOARD_API_KEY and request.path not in _PUBLIC_PATHS:
        # Allow setup endpoints without auth (needed for first-run wizard)
        if not request.path.startswith("/api/setup"):
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {DASHBOARD_API_KEY}":
                return web.json_response({"error": "Unauthorized"}, status=401)
    response = await handler(request)
    origin = CORS_ORIGIN or "http://localhost:8085"
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Global State ────────────────────────────────────────────────

class AppState:
    """Shared state between dashboard and pipeline."""
    running: bool = False
    live_running: bool = False
    connected_clients: list = []
    transcript: list[dict] = []
    stats: dict = {
        "chunks_processed": 0,
        "avg_latency": 0.0,
        "total_runtime": 0.0,
        "status": "stopped",
    }
    start_time: float = 0.0
    live_pipeline = None
    live_task = None
    live_capture = None
    live_playback = None
    live_aes67 = None

state = AppState()


async def broadcast(msg: dict):
    """Send a message to all connected WebSocket clients."""
    data = json.dumps(msg)
    dead = []
    for ws in state.connected_clients:
        try:
            await ws.send_str(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.connected_clients.remove(ws)


# ── HTTP Routes ─────────────────────────────────────────────────

async def index(request):
    """Serve the dashboard HTML."""
    html_path = Path(__file__).parent / "index.html"
    return web.FileResponse(html_path)


async def api_status(request):
    """Get current pipeline status."""
    return web.json_response({
        "status": state.stats["status"],
        "chunks_processed": state.stats["chunks_processed"],
        "avg_latency": state.stats["avg_latency"],
        "total_runtime": time.time() - state.start_time if state.running else state.stats["total_runtime"],
        "transcript_count": len(state.transcript),
    })


async def api_transcript(request):
    """Get full transcript."""
    return web.json_response({"transcript": state.transcript})


async def api_start(request):
    """Start the translation pipeline."""
    if state.running:
        return web.json_response({"error": "Already running"}, status=400)

    state.running = True
    state.start_time = time.time()
    state.stats["status"] = "running"
    state.transcript = []
    state.stats["chunks_processed"] = 0

    await broadcast({"type": "status", "status": "running"})
    return web.json_response({"status": "started"})


async def api_stop(request):
    """Stop the translation pipeline."""
    state.running = False
    state.stats["status"] = "stopped"
    state.stats["total_runtime"] = time.time() - state.start_time

    await broadcast({"type": "status", "status": "stopped"})
    return web.json_response({"status": "stopped"})


async def api_test_file(request):
    """Run a test translation on an uploaded or specified file."""
    data = await request.json()
    file_path = data.get("file_path", "")

    if not file_path or not Path(file_path).exists():
        return web.json_response({"error": "File not found"}, status=400)

    state.running = True
    state.start_time = time.time()
    state.stats["status"] = "running"
    state.transcript = []
    state.stats["chunks_processed"] = 0

    await broadcast({"type": "status", "status": "running"})
    asyncio.create_task(_run_file_test(file_path))

    return web.json_response({"status": "started", "file": file_path})


# ── Feature 1: Audio Device API ────────────────────────────────

async def api_devices(request):
    """List available audio input and output devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        result = {"input": [], "output": []}
        for i, dev in enumerate(devices):
            entry = {"index": i, "name": dev["name"], "sample_rate": dev["default_samplerate"]}
            if dev["max_input_channels"] > 0:
                result["input"].append(entry)
            if dev["max_output_channels"] > 0:
                result["output"].append(entry)
        return web.json_response(result)
    except ImportError:
        return web.json_response({"input": [], "output": [], "error": "sounddevice not installed"})
    except Exception as e:
        return web.json_response({"input": [], "output": [], "error": str(e)})


async def api_test_output(request):
    """Play a short test tone through the selected output device."""
    data = await request.json()
    device_index = data.get("device_index")

    try:
        import sounddevice as sd
        import numpy as np

        sr = 44100
        duration = 0.5
        freq = 880  # A5 note
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Pleasant bell-like tone with envelope
        tone = 0.3 * np.sin(2 * math.pi * freq * t) * np.exp(-3 * t)

        device = int(device_index) if device_index is not None else None
        sd.play(tone.astype(np.float32), samplerate=sr, device=device)
        sd.wait()
        return web.json_response({"ok": True})
    except ImportError:
        return web.json_response({"ok": False, "error": "sounddevice or numpy not installed"}, status=500)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


# ── Feature 1 & 2: Config API ──────────────────────────────────

async def api_get_config(request):
    """Get saved device/language config."""
    cfg = load_saved_config()
    return web.json_response(cfg)


async def api_save_config(request):
    """Save device/language config."""
    data = await request.json()
    # Only allow safe keys
    allowed = {"input_device", "output_device", "source_language", "target_language", "target_languages", "custom_vocabulary", "elevenlabs_voice_id", "stt_model", "translation_model", "tts_model", "multi_language_mode"}
    filtered = {k: v for k, v in data.items() if k in allowed}
    save_config(filtered)
    return web.json_response({"ok": True})


async def api_languages(request):
    """List supported languages."""
    return web.json_response({"languages": SUPPORTED_LANGUAGES})


# ── Voice Selection API ────────────────────────────────────────

async def api_voices(request):
    """List available ElevenLabs voices."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        return web.json_response({"voices": [], "error": "ElevenLabs API key not configured"})

    try:
        import aiohttp as _aiohttp
        async with _aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
                timeout=_aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return web.json_response({"voices": [], "error": f"ElevenLabs API error: {resp.status}"})
                data = await resp.json()
                voices = []
                for v in data.get("voices", []):
                    voices.append({
                        "voice_id": v["voice_id"],
                        "name": v["name"],
                        "category": v.get("category", ""),
                        "description": v.get("description", ""),
                        "preview_url": v.get("preview_url", ""),
                        "labels": v.get("labels", {}),
                    })
                # Sort: premade first, then alphabetical
                voices.sort(key=lambda x: (0 if x["category"] == "premade" else 1, x["name"]))
                return web.json_response({"voices": voices})
    except Exception as e:
        logger.error("Failed to fetch voices: %s", e)
        return web.json_response({"voices": [], "error": str(e)})


# ── Feature 3: Setup Wizard API ────────────────────────────────

async def api_setup_status(request):
    """Check if API keys are configured."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    has_openai = bool(openai_key and not openai_key.startswith("sk-your"))
    has_elevenlabs = bool(elevenlabs_key and elevenlabs_key != "your-elevenlabs-key-here")
    return web.json_response({
        "configured": has_openai,
        "has_openai": has_openai,
        "has_elevenlabs": has_elevenlabs,
    })


async def api_setup_test_openai(request):
    """Test OpenAI API key by making a simple API call."""
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return web.json_response({"ok": False, "error": "No API key provided"}, status=400)

    try:
        import aiohttp as aio
        async with aio.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aio.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return web.json_response({"ok": True})
                else:
                    body = await resp.json()
                    return web.json_response({
                        "ok": False,
                        "error": body.get("error", {}).get("message", f"HTTP {resp.status}"),
                    })
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def api_setup_test_elevenlabs(request):
    """Test ElevenLabs API key."""
    data = await request.json()
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return web.json_response({"ok": False, "error": "No API key provided"}, status=400)

    try:
        import aiohttp as aio
        async with aio.ClientSession() as session:
            async with session.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key},
                timeout=aio.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return web.json_response({"ok": True})
                else:
                    return web.json_response({"ok": False, "error": f"HTTP {resp.status}"})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def api_setup_save(request):
    """Save API keys to .env and config to config.json."""
    data = await request.json()

    # Write .env file (keys only)
    env_path = PROJECT_ROOT / ".env"
    env_lines = []

    # Preserve existing env vars
    existing_env = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key = line.split("=", 1)[0]
                existing_env[key] = line

    # Update keys
    if data.get("openai_api_key"):
        existing_env["OPENAI_API_KEY"] = f"OPENAI_API_KEY={data['openai_api_key']}"
    if data.get("elevenlabs_api_key"):
        existing_env["ELEVENLABS_API_KEY"] = f"ELEVENLABS_API_KEY={data['elevenlabs_api_key']}"

    env_path.write_text("\n".join(existing_env.values()) + "\n", encoding="utf-8")

    # Save device/language config
    config_data = {}
    for key in ("source_language", "target_language", "input_device", "output_device", "custom_vocabulary"):
        if key in data:
            config_data[key] = data[key]
    if config_data:
        save_config(config_data)

    return web.json_response({"ok": True})


# ── File Test Pipeline ──────────────────────────────────────────

async def _run_file_test(file_path: str):
    """Background task to run file translation and stream results."""
    try:
        from src.config import load_config
        from src.transcriber import Transcriber
        from src.translator import Translator
        from src.vad_chunker import FileVADChunker
        import subprocess
        import tempfile

        config = load_config()

        # Apply saved language config
        saved = load_saved_config()
        src_lang = saved.get("source_language", config.transcription.language)
        tgt_lang = saved.get("target_language", "en")
        custom_vocab = saved.get("custom_vocabulary", "")

        transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=src_lang,
        )

        # Build system prompt with language and custom vocabulary
        system_prompt = config.translation.system_prompt
        if custom_vocab:
            system_prompt += f"\n\nCustom vocabulary and terms to use:\n{custom_vocab}"

        translator = Translator(
            api_key=config.openai_api_key,
            system_prompt=system_prompt,
            model=config.translation.model,
            temperature=config.translation.temperature,
            context_sentences=config.pipeline.context_sentences,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", "-y", tmp_path],
            capture_output=True, check=True,
        )

        vad = FileVADChunker(
            aggressiveness=config.pipeline.vad_aggressiveness,
            min_chunk_sec=config.pipeline.min_chunk_sec,
            max_chunk_sec=config.pipeline.max_chunk_sec,
            silence_threshold_sec=config.pipeline.silence_threshold_sec,
        )
        chunks = vad.chunk_file(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        await broadcast({"type": "info", "message": f"Processing {len(chunks)} chunks..."})

        # Find language names for display
        lang_names = {l["code"]: l["name"] for l in SUPPORTED_LANGUAGES}
        src_name = lang_names.get(src_lang, src_lang)
        tgt_name = lang_names.get(tgt_lang, tgt_lang)

        total_latency = 0.0

        for i, wav_chunk in enumerate(chunks):
            if not state.running:
                break

            t0 = time.time()

            src_text = await transcriber.transcribe(wav_chunk)
            if not src_text:
                continue

            await broadcast({
                "type": "stt",
                "seq": i + 1,
                "total": len(chunks),
                "text": src_text,
            })

            tgt_text = await translator.translate(src_text)
            if not tgt_text:
                continue

            latency = time.time() - t0
            total_latency += latency
            state.stats["chunks_processed"] = i + 1
            state.stats["avg_latency"] = total_latency / (i + 1)

            entry = {
                "seq": i + 1,
                "source": src_text,
                "translated": tgt_text,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "latency": round(latency, 1),
                "timestamp": time.time(),
                # Keep backward compat
                "ukrainian": src_text,
                "english": tgt_text,
            }
            state.transcript.append(entry)

            await broadcast({
                "type": "translation",
                "entry": entry,
                "progress": (i + 1) / len(chunks),
                "stats": {
                    "chunks_processed": i + 1,
                    "total_chunks": len(chunks),
                    "avg_latency": round(state.stats["avg_latency"], 1),
                },
            })

        state.running = False
        state.stats["status"] = "stopped"
        state.stats["total_runtime"] = time.time() - state.start_time

        await broadcast({
            "type": "status",
            "status": "completed",
            "stats": {
                "chunks_processed": state.stats["chunks_processed"],
                "avg_latency": round(state.stats["avg_latency"], 1),
                "total_runtime": round(state.stats["total_runtime"], 1),
            },
        })

    except Exception as e:
        logger.error("File test failed: %s", e, exc_info=True)
        state.running = False
        state.stats["status"] = "error"
        await broadcast({"type": "error", "message": str(e)})


# ── Live Translation Pipeline ──────────────────────────────────

async def api_start_live(request):
    """Start live mic → translate → speak pipeline."""
    if state.running or state.live_running:
        return web.json_response({"error": "Already running"}, status=400)

    state.live_running = True
    state.running = True
    state.start_time = time.time()
    state.stats["status"] = "live"
    state.transcript = []
    state.stats["chunks_processed"] = 0
    state.stats["avg_latency"] = 0.0

    await broadcast({"type": "status", "status": "live"})
    state.live_task = asyncio.create_task(_run_live_pipeline())

    return web.json_response({"status": "started", "mode": "live"})


async def api_stop_live(request):
    """Stop the live translation pipeline."""
    if not state.live_running:
        return web.json_response({"error": "Not running"}, status=400)

    state.live_running = False
    state.running = False

    # Stop the audio capture (microphone) immediately
    if state.live_capture:
        try:
            await state.live_capture.stop()
            logger.info("Live capture stopped")
        except Exception as e:
            logger.warning("Error stopping capture: %s", e)
        state.live_capture = None

    # Stop AES67 output(s)
    if state.live_aes67:
        for a67 in (state.live_aes67 if isinstance(state.live_aes67, list) else [state.live_aes67]):
            try:
                a67.stop()
            except Exception as e:
                logger.warning("Error stopping AES67: %s", e)
        state.live_aes67 = None

    # Stop the pipeline (if set)
    if state.live_pipeline:
        try:
            await state.live_pipeline.stop()
        except Exception as e:
            logger.warning("Error stopping pipeline: %s", e)
        state.live_pipeline = None

    # Cancel the background task
    if state.live_task and not state.live_task.done():
        state.live_task.cancel()
        try:
            await state.live_task
        except (asyncio.CancelledError, Exception):
            pass
        state.live_task = None

    state.live_playback = None

    state.stats["status"] = "stopped"
    state.stats["total_runtime"] = time.time() - state.start_time

    await broadcast({
        "type": "status",
        "status": "stopped",
        "stats": {
            "chunks_processed": state.stats["chunks_processed"],
            "avg_latency": round(state.stats["avg_latency"], 1),
            "total_runtime": round(state.stats["total_runtime"], 1),
        },
    })
    return web.json_response({"status": "stopped"})


async def _run_live_pipeline():
    """Background task running the live translation pipeline with dashboard integration."""
    try:
        from src.config import load_config
        from src.transcriber import Transcriber
        from src.translator import Translator
        from src.synthesizer import Synthesizer
        from src.vad_capture import VADAudioCapture
        from src.audio_playback import AudioPlayback

        config = load_config()

        # Apply saved config
        saved = load_saved_config()
        src_lang = saved.get("source_language", config.transcription.language)
        tgt_lang = saved.get("target_language", "en")
        custom_vocab = saved.get("custom_vocabulary", "")
        multi_mode = saved.get("multi_language_mode", False)
        target_langs = saved.get("target_languages", [tgt_lang])
        if not multi_mode:
            target_langs = [tgt_lang]

        # Override audio devices from saved config
        input_dev = saved.get("input_device")
        output_dev = saved.get("output_device")
        if input_dev is not None and input_dev != "":
            config.audio.input_device = int(input_dev)
        if output_dev is not None and output_dev != "":
            config.audio.output_device = int(output_dev)

        # Override ElevenLabs voice if user selected one
        saved_voice_id = saved.get("elevenlabs_voice_id")
        if saved_voice_id:
            config.synthesis.elevenlabs.voice_id = saved_voice_id

        # Override models from saved config
        saved_stt_model = saved.get("stt_model")
        if saved_stt_model:
            config.transcription.model = saved_stt_model
        saved_translation_model = saved.get("translation_model")
        if saved_translation_model:
            config.translation.model = saved_translation_model
        saved_tts_model = saved.get("tts_model")
        if saved_tts_model:
            config.synthesis.elevenlabs.model = saved_tts_model

        # Initialize components
        capture = VADAudioCapture(
            device=config.audio.input_device,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            vad_aggressiveness=config.pipeline.vad_aggressiveness,
            min_chunk_sec=config.pipeline.min_chunk_sec,
            max_chunk_sec=config.pipeline.max_chunk_sec,
            silence_threshold_sec=config.pipeline.silence_threshold_sec,
        )

        transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=src_lang,
        )

        system_prompt = config.translation.system_prompt
        if custom_vocab:
            system_prompt += f"\n\nCustom vocabulary and terms to use:\n{custom_vocab}"

        # Build per-language output channels
        # Each target language gets its own translator, synthesizer, playback queue, and optional AES67 stream
        lang_names = {l["code"]: l["name"] for l in SUPPORTED_LANGUAGES}
        lang_channels = {}  # {lang_code: {translator, synthesizer, playback_queue, aes67, playback_task}}

        # Base AES67 multicast: 239.69.0.1 for first lang, .2 for second, etc.
        base_multicast = config.output.multicast_address.rsplit(".", 1)
        base_multicast_prefix = base_multicast[0] + "."
        base_multicast_last = int(base_multicast[1])
        base_port = config.output.port

        for i, lang_code in enumerate(target_langs):
            lang_name = lang_names.get(lang_code, lang_code)

            # Each language gets its own translator with language-specific prompt
            lang_system_prompt = system_prompt.replace(
                "English", lang_name
            ) if "English" in system_prompt else system_prompt + f"\n\nTranslate to {lang_name}."

            lang_translator = Translator(
                api_key=config.openai_api_key,
                system_prompt=lang_system_prompt,
                model=config.translation.model,
                temperature=config.translation.temperature,
                context_sentences=config.pipeline.context_sentences,
            )

            lang_synthesizer = Synthesizer(
                provider=config.synthesis.provider,
                openai_api_key=config.openai_api_key,
                elevenlabs_api_key=config.elevenlabs_api_key,
                elevenlabs_voice_id=config.synthesis.elevenlabs.voice_id,
                elevenlabs_model=config.synthesis.elevenlabs.model,
                elevenlabs_stability=config.synthesis.elevenlabs.stability,
                elevenlabs_similarity=config.synthesis.elevenlabs.similarity_boost,
                openai_model=config.synthesis.openai.model,
                openai_voice=config.synthesis.openai.voice,
            )

            # AES67: each language gets its own multicast stream
            lang_aes67 = None
            if config.output.mode in ("dante", "both"):
                from src.aes67_output import AES67Sender
                lang_aes67 = AES67Sender(
                    stream_name=f"Church Translation {lang_code.upper()}",
                    multicast_addr=f"{base_multicast_prefix}{base_multicast_last + i}",
                    port=base_port + (i * 2),
                    ttl=config.output.ttl,
                )
                lang_aes67.start()

            lang_channels[lang_code] = {
                "name": lang_name,
                "translator": lang_translator,
                "synthesizer": lang_synthesizer,
                "aes67": lang_aes67,
                "playback_queue": asyncio.Queue(),
            }

        # Primary playback (first language goes to local speakers)
        playback = AudioPlayback(
            device=config.audio.output_device,
            sample_rate=24000,
            channels=1,
        )
        primary_lang = target_langs[0] if target_langs else tgt_lang

        # Store references so api_stop_live can stop them
        state.live_capture = capture
        state.live_playback = playback
        state.live_aes67 = [ch["aes67"] for ch in lang_channels.values() if ch["aes67"]]

        await capture.start()

        mode_desc = "Multi-language" if multi_mode else "Single-language"
        lang_list = ", ".join(lang_names.get(l, l) for l in target_langs)
        await broadcast({"type": "info", "message": f"🎙️ {mode_desc} mode — listening ({lang_list})..."})

        total_latency = 0.0
        chunk_count = 0

        # Playback workers — one per language
        playback_workers = []
        playback_active = True

        async def _playback_worker(lang_code):
            """Plays TTS audio for a specific language channel."""
            ch = lang_channels[lang_code]
            q = ch["playback_queue"]
            is_primary = (lang_code == primary_lang)
            while playback_active or not q.empty():
                try:
                    audio_bytes = await asyncio.wait_for(q.get(), timeout=0.5)
                    if audio_bytes is None:
                        break
                    play_tasks = []
                    # Primary language plays through local speakers
                    if is_primary:
                        play_tasks.append(playback.play(audio_bytes))
                    # AES67 output for this language's stream
                    if ch["aes67"]:
                        play_tasks.append(ch["aes67"].play(audio_bytes))
                    if play_tasks:
                        await asyncio.gather(*play_tasks)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning("Playback error (%s): %s", lang_code, e)

        for lang_code in target_langs:
            task = asyncio.create_task(_playback_worker(lang_code))
            playback_workers.append(task)

        async def _process_chunk(wav_bytes, seq):
            """Process a single chunk: STT → fan-out to all languages → TTS (queued)."""
            nonlocal total_latency
            t0 = time.time()

            # STT (shared — one transcription for all languages)
            src_text = await transcriber.transcribe(wav_bytes)
            if not src_text:
                return

            await broadcast({
                "type": "stt",
                "seq": seq,
                "total": 0,
                "text": src_text,
            })

            # Fan-out: translate + TTS for each target language in parallel
            async def _translate_lang(lang_code):
                ch = lang_channels[lang_code]
                tgt_text = await ch["translator"].translate(src_text)
                if not tgt_text:
                    return None
                # TTS and queue for playback
                audio_bytes = await ch["synthesizer"].synthesize(tgt_text)
                if audio_bytes:
                    await ch["playback_queue"].put(audio_bytes)
                return {"lang": lang_code, "text": tgt_text}

            results = await asyncio.gather(*[_translate_lang(lc) for lc in target_langs], return_exceptions=True)

            latency = time.time() - t0
            total_latency += latency
            state.stats["chunks_processed"] = seq
            state.stats["avg_latency"] = total_latency / seq

            # Broadcast translations for all languages
            for r in results:
                if isinstance(r, dict):
                    entry = {
                        "seq": seq,
                        "source": src_text,
                        "translated": r["text"],
                        "source_lang": src_lang,
                        "target_lang": r["lang"],
                        "latency": round(latency, 1),
                        "timestamp": time.time(),
                    }
                    state.transcript.append(entry)
                    await broadcast({
                        "type": "translation",
                        "entry": entry,
                        "progress": 0,
                        "stats": {
                            "chunks_processed": seq,
                            "total_chunks": 0,
                            "avg_latency": round(state.stats["avg_latency"], 1),
                        },
                    })

        # Main loop: capture overlaps with processing
        # We process chunks concurrently (up to 2 at a time) so capture never stalls
        processing_tasks = set()
        MAX_CONCURRENT = 2

        while state.live_running:
            try:
                wav_bytes = await capture.get_chunk()
            except Exception:
                if not state.live_running:
                    break
                continue
            if wav_bytes is None:
                continue

            chunk_count += 1
            task = asyncio.create_task(_process_chunk(wav_bytes, chunk_count))
            processing_tasks.add(task)
            task.add_done_callback(processing_tasks.discard)

            # If at max concurrency, wait for one to finish before accepting next
            if len(processing_tasks) >= MAX_CONCURRENT:
                done, _ = await asyncio.wait(processing_tasks, return_when=asyncio.FIRST_COMPLETED)
                processing_tasks -= done

        # Wait for in-flight processing to finish
        if processing_tasks:
            await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Signal all playback workers to stop after draining queues
        playback_active = False
        for lang_code in target_langs:
            await lang_channels[lang_code]["playback_queue"].put(None)
        await asyncio.gather(*playback_workers, return_exceptions=True)

        # Cleanup
        await capture.stop()
        for ch in lang_channels.values():
            if ch["aes67"]:
                ch["aes67"].stop()

    except asyncio.CancelledError:
        logger.info("Live pipeline cancelled")
    except Exception as e:
        logger.error("Live pipeline error: %s", e, exc_info=True)
        await broadcast({"type": "error", "message": f"Live translation error: {e}"})
    finally:
        # Ensure all audio resources are released
        if state.live_capture:
            try:
                await state.live_capture.stop()
            except Exception:
                pass
            state.live_capture = None
        if state.live_aes67:
            aes67_list = state.live_aes67 if isinstance(state.live_aes67, list) else [state.live_aes67]
            for a67 in aes67_list:
                try:
                    a67.stop()
                except Exception:
                    pass
            state.live_aes67 = None
        state.live_playback = None
        state.live_running = False
        state.running = False
        state.stats["status"] = "stopped"
        state.stats["total_runtime"] = time.time() - state.start_time
        state.live_pipeline = None
        logger.info("Live pipeline fully stopped and cleaned up")


# ── Health Check ───────────────────────────────────────────────

async def api_health(request):
    """Return system health checks as JSON."""
    import shutil as _shutil
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(PROJECT_ROOT / ".env", override=True)

    checks = {}

    # Python version
    v = sys.version_info
    checks["python"] = {
        "ok": v.major >= 3 and v.minor >= 11,
        "version": f"{v.major}.{v.minor}.{v.micro}",
        "detail": f"Python {v.major}.{v.minor}.{v.micro}" + ("" if v.minor >= 11 else " (need 3.11+)"),
    }

    # ffmpeg
    checks["ffmpeg"] = {
        "ok": _shutil.which("ffmpeg") is not None,
        "detail": "Installed" if _shutil.which("ffmpeg") else "Not found — needed for audio processing",
    }

    # API keys
    openai_key = os.getenv("OPENAI_API_KEY", "")
    has_openai = bool(openai_key and not openai_key.startswith("sk-your"))
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    has_elevenlabs = bool(elevenlabs_key and elevenlabs_key != "your-elevenlabs-key-here")
    checks["api_keys"] = {
        "ok": has_openai,
        "has_openai": has_openai,
        "has_elevenlabs": has_elevenlabs,
        "detail": ("OpenAI: configured" if has_openai else "OpenAI: NOT configured")
                  + " | "
                  + ("ElevenLabs: configured" if has_elevenlabs else "ElevenLabs: not configured (optional)"),
    }

    # Audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_count = sum(1 for d in devices if d["max_input_channels"] > 0)
        output_count = sum(1 for d in devices if d["max_output_channels"] > 0)
        checks["audio"] = {
            "ok": input_count > 0 and output_count > 0,
            "inputs": input_count,
            "outputs": output_count,
            "detail": f"{input_count} input(s), {output_count} output(s)",
        }
    except Exception as e:
        checks["audio"] = {
            "ok": False,
            "inputs": 0,
            "outputs": 0,
            "detail": f"Audio system error: {e}",
        }

    # Network (for AES67)
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        checks["network"] = {
            "ok": True,
            "ip": local_ip,
            "detail": f"Network available ({local_ip})",
        }
    except Exception:
        checks["network"] = {
            "ok": False,
            "ip": None,
            "detail": "No network connection",
        }

    all_ok = all(c["ok"] for c in checks.values())
    return web.json_response({"healthy": all_ok, "checks": checks})


# ── WebSocket ───────────────────────────────────────────────────

async def websocket_handler(request):
    """WebSocket endpoint for real-time updates."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state.connected_clients.append(ws)
    logger.info("Dashboard client connected (%d total)", len(state.connected_clients))

    await ws.send_json({
        "type": "init",
        "status": state.stats["status"],
        "transcript": state.transcript,
        "stats": state.stats,
    })

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("action") == "ping":
                    await ws.send_json({"type": "pong"})
            elif msg.type == web.WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
    finally:
        state.connected_clients.remove(ws)
        logger.info("Dashboard client disconnected (%d remaining)", len(state.connected_clients))

    return ws


# ── App Setup ───────────────────────────────────────────────────

def create_app():
    app = web.Application(middlewares=[auth_middleware])
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/transcript", api_transcript)
    app.router.add_post("/api/start", api_start)
    app.router.add_post("/api/stop", api_stop)
    app.router.add_post("/api/test-file", api_test_file)
    # Feature 1: Audio devices
    app.router.add_get("/api/devices", api_devices)
    app.router.add_post("/api/test-output", api_test_output)
    # Feature 1 & 2: Config
    app.router.add_get("/api/config", api_get_config)
    app.router.add_post("/api/config", api_save_config)
    app.router.add_get("/api/settings", api_get_config)
    app.router.add_post("/api/settings", api_save_config)
    app.router.add_get("/api/languages", api_languages)
    app.router.add_get("/api/voices", api_voices)
    # Feature 3: Setup wizard
    app.router.add_get("/api/setup/status", api_setup_status)
    app.router.add_post("/api/setup/test-openai", api_setup_test_openai)
    app.router.add_post("/api/setup/test-elevenlabs", api_setup_test_elevenlabs)
    app.router.add_post("/api/setup/save", api_setup_save)
    # Live translation
    app.router.add_post("/api/start-live", api_start_live)
    app.router.add_post("/api/stop-live", api_stop_live)
    # Health check
    app.router.add_get("/api/health", api_health)
    # Static assets
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static/", static_path)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    port = int(os.getenv("DASHBOARD_PORT", "8085"))
    app = create_app()
    logger.info("Dashboard starting on http://localhost:%d", port)
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    web.run_app(app, host=host, port=port)
