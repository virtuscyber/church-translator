#!/usr/bin/env python3
"""Dashboard web server for Church Live Translation.

Serves the dashboard UI and provides WebSocket endpoints for real-time
translation status, transcript updates, and pipeline control.
"""

from __future__ import annotations

import asyncio
import contextlib
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

    def __init__(self):
        self.running: bool = False
        self.live_running: bool = False
        self.connected_clients: list = []
        self.transcript: list[dict] = []
        self.stats: dict = {
            "chunks_processed": 0,
            "avg_latency": 0.0,
            "total_runtime": 0.0,
            "status": "stopped",
        }
        self.start_time: float = 0.0
        self.live_pipeline = None
        self.live_task = None
        self.live_capture = None
        self.live_playback = None
        self.live_aes67 = None
        self.audio_monitor_lock = asyncio.Lock()

state = AppState()


def _discard_client(ws) -> None:
    """Remove a WebSocket from the client list if it is still present."""
    with contextlib.suppress(ValueError):
        state.connected_clients.remove(ws)


async def broadcast(msg: dict):
    """Send a message to all connected WebSocket clients."""
    data = json.dumps(msg)
    dead = []
    for ws in list(state.connected_clients):
        try:
            await ws.send_str(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _discard_client(ws)


def _input_device_entries(sd) -> list[dict]:
    """Return normalized metadata for every input-capable audio device."""
    entries = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            sr = int(dev.get("default_samplerate") or 48000)
            entries.append({
                "index": i,
                "name": dev["name"],
                "sr": sr,
                "channels": min(int(dev["max_input_channels"]), 1),
            })
    return entries


def _sample_input_device_level(sd, np, *, device_index: int, sample_rate: int, channels: int, duration: float) -> tuple[float, float]:
    """Capture a short blocking sample from an input device and return peak/rms dB."""
    frames = max(1, int(sample_rate * duration))
    sd.check_input_settings(
        device=device_index,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )

    with sd.InputStream(
        device=device_index,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=frames,
    ) as stream:
        recording, overflowed = stream.read(frames)

    if overflowed:
        logger.debug("Audio level sample overflowed for device %s", device_index)

    if recording is None or len(recording) == 0:
        return -100.0, -100.0

    audio = recording[:, 0] if getattr(recording, "ndim", 1) > 1 else recording
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak_db = 20 * np.log10(peak) if peak > 1e-10 else -100.0
    rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100.0
    return float(peak_db), float(rms_db)


def _log_live_chunk_result(done_task: asyncio.Task) -> None:
    """Consume background task exceptions so they don't surface as unhandled."""
    if done_task.cancelled():
        return
    exc = done_task.exception()
    if exc:
        logger.error(
            "Live chunk processing failed: %s",
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


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


# ── Audio Level Monitor ────────────────────────────────────────

async def api_audio_levels(request):
    """Sample audio levels from all input devices simultaneously.

    Returns a JSON array of {index, name, level_db, level_pct, has_audio}.
    Samples each device for ~200ms. Devices with audio show higher levels.
    """
    try:
        import sounddevice as sd
        import numpy as np

        input_devices = _input_device_entries(sd)

        results = []
        loop = asyncio.get_running_loop()

        def _sample_device(dev_index, sr, channels):
            """Record a short sample from one device and return its levels."""
            try:
                peak_db, _ = _sample_input_device_level(
                    sd,
                    np,
                    device_index=dev_index,
                    sample_rate=sr,
                    channels=channels,
                    duration=0.2,
                )
                return peak_db
            except Exception as exc:
                return exc

        async with state.audio_monitor_lock:
            for dev in input_devices:
                result = await loop.run_in_executor(None, _sample_device, dev["index"], dev["sr"], dev["channels"])
                if isinstance(result, Exception):
                    logger.debug("Audio level scan failed for device %s: %s", dev["index"], result)
                    results.append({
                        "index": dev["index"],
                        "name": dev["name"],
                        "level_db": None,
                        "level_pct": 0,
                        "has_audio": False,
                        "error": True,
                    })
                    continue

                level_db = result
                pct = max(0, min(100, int((level_db + 60) / 60 * 100)))
                results.append({
                    "index": dev["index"],
                    "name": dev["name"],
                    "level_db": round(level_db, 1),
                    "level_pct": pct,
                    "has_audio": level_db > -40,
                })

        return web.json_response({"devices": results})
    except ImportError:
        return web.json_response({"devices": [], "error": "sounddevice not installed"})
    except Exception as e:
        return web.json_response({"devices": [], "error": str(e)})


async def api_audio_level_single(request):
    """Get audio level for a single device (faster, for continuous monitoring)."""
    try:
        import sounddevice as sd
        import numpy as np

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        dev_index = data.get("device_index")
        if dev_index is None:
            return web.json_response({"error": "device_index required"}, status=400)

        try:
            dev_index = int(dev_index)
        except (TypeError, ValueError):
            return web.json_response({"error": "device_index must be an integer"}, status=400)

        try:
            dev_info = sd.query_devices(dev_index)
        except Exception as exc:
            return web.json_response({"error": f"Invalid audio device: {exc}"}, status=400)

        if dev_info["max_input_channels"] <= 0:
            return web.json_response({"error": "Selected device is not an input device"}, status=400)

        sr = int(dev_info.get("default_samplerate") or 48000)
        channels = min(int(dev_info["max_input_channels"]), 1)

        loop = asyncio.get_running_loop()

        def _sample():
            return _sample_input_device_level(
                sd,
                np,
                device_index=dev_index,
                sample_rate=sr,
                channels=channels,
                duration=0.1,
            )

        async with state.audio_monitor_lock:
            peak_db, rms_db = await loop.run_in_executor(None, _sample)
        pct = max(0, min(100, int((peak_db + 60) / 60 * 100)))

        return web.json_response({
            "index": dev_index,
            "peak_db": round(peak_db, 1),
            "rms_db": round(rms_db, 1),
            "level_pct": pct,
            "has_audio": peak_db > -40,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ── Feature 1 & 2: Config API ──────────────────────────────────

async def api_get_config(request):
    """Get saved device/language config."""
    cfg = load_saved_config()
    return web.json_response(cfg)


async def api_save_config(request):
    """Save device/language config."""
    data = await request.json()
    # Only allow safe keys
    allowed = {"input_device", "output_device", "source_language", "target_language", "custom_vocabulary", "elevenlabs_voice_id", "stt_model", "translation_model", "tts_model"}
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
    tmp_path = None
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
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


async def _stop_live_pipeline(*, notify_clients: bool) -> None:
    """Stop all live-translation resources and optionally broadcast status."""
    state.live_running = False
    state.running = False

    if state.live_capture:
        try:
            await state.live_capture.stop()
            logger.info("Live capture stopped")
        except Exception as e:
            logger.warning("Error stopping capture: %s", e)
        state.live_capture = None

    if state.live_aes67:
        try:
            state.live_aes67.stop()
        except Exception as e:
            logger.warning("Error stopping AES67: %s", e)
        state.live_aes67 = None

    if state.live_pipeline:
        try:
            await state.live_pipeline.stop()
        except Exception as e:
            logger.warning("Error stopping pipeline: %s", e)
        state.live_pipeline = None

    if state.live_task:
        if not state.live_task.done():
            state.live_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await state.live_task
        state.live_task = None

    state.live_playback = None
    state.stats["status"] = "stopped"
    state.stats["total_runtime"] = time.time() - state.start_time if state.start_time else 0.0

    if notify_clients:
        await broadcast({
            "type": "status",
            "status": "stopped",
            "stats": {
                "chunks_processed": state.stats["chunks_processed"],
                "avg_latency": round(state.stats["avg_latency"], 1),
                "total_runtime": round(state.stats["total_runtime"], 1),
            },
        })


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

    await _stop_live_pipeline(notify_clients=True)
    return web.json_response({"status": "stopped"})


async def _run_live_pipeline():
    """Background task running the live translation pipeline with dashboard integration."""
    playback_queue = None
    playback_task = None
    processing_tasks = set()
    playback_active = False
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
            enable_preview=True,
            preview_after_sec=config.pipeline.min_chunk_sec,
        )

        transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=src_lang,
        )

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

        synthesizer = Synthesizer(
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

        playback = AudioPlayback(
            device=config.audio.output_device,
            sample_rate=24000,
            channels=1,
        )

        # Optional AES67
        aes67 = None
        if config.output.mode in ("dante", "both"):
            from src.aes67_output import AES67Sender
            aes67 = AES67Sender(
                stream_name=config.output.stream_name,
                multicast_addr=config.output.multicast_address,
                port=config.output.port,
                ttl=config.output.ttl,
            )
            aes67.start()

        # Store references so api_stop_live can stop them
        state.live_capture = capture
        state.live_playback = playback
        state.live_aes67 = aes67

        await capture.start()
        await broadcast({"type": "info", "message": "Microphone active — listening for speech..."})

        lang_names = {l["code"]: l["name"] for l in SUPPORTED_LANGUAGES}
        src_name = lang_names.get(src_lang, src_lang)
        tgt_name = lang_names.get(tgt_lang, tgt_lang)
        total_latency = 0.0
        chunk_count = 0

        # ── Ordered concurrent playback system ──────────────────────
        # Each chunk gets a "slot" (asyncio.Queue). Chunks process
        # concurrently (STT + translate + TTS), but audio plays back
        # in strict sequence order to prevent garbled output.
        #
        # Chunk 1: [STT][translate][TTS→slot1: 🔊🔊🔊]
        # Chunk 2:   [STT][translate][TTS→slot2: buffer]  →  [🔊🔊🔊]
        #                   ↑ concurrent!                     ↑ plays after slot1
        playback_slots: dict[int, asyncio.Queue] = {}
        playback_active = True
        next_slot_ready = asyncio.Event()

        async def _playback_worker():
            """Drains playback slots in order: slot 1, then 2, then 3..."""
            next_seq = 1
            while playback_active or playback_slots:
                # Wait for the next sequence's slot to appear
                if next_seq not in playback_slots:
                    next_slot_ready.clear()
                    try:
                        await asyncio.wait_for(next_slot_ready.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if next_seq not in playback_slots:
                        continue

                slot = playback_slots[next_seq]
                logger.info("Playback: starting slot %d", next_seq)

                # Drain all audio chunks from this slot
                while True:
                    try:
                        audio_bytes = await asyncio.wait_for(slot.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        if not playback_active:
                            break
                        continue
                    if audio_bytes is None:  # End-of-slot marker
                        break
                    play_tasks = [playback.play(audio_bytes)]
                    if aes67:
                        play_tasks.append(aes67.play(audio_bytes))
                    try:
                        await asyncio.gather(*play_tasks)
                    except Exception as e:
                        logger.warning("Playback error (slot %d): %s", next_seq, e)

                # Slot done — advance to next
                del playback_slots[next_seq]
                logger.info("Playback: slot %d complete, advancing", next_seq)
                next_seq += 1

        playback_task = asyncio.create_task(_playback_worker())

        async def _process_chunk(wav_bytes, seq, pre_transcribed=None):
            """Process a single chunk: STT → Translate → TTS → slot.
            
            Runs concurrently with other chunks. STT and translation happen
            in parallel across chunks, but TTS audio is routed to a per-chunk
            slot that the playback worker drains in order.
            """
            nonlocal total_latency
            t0 = time.time()

            # Reserve our playback slot FIRST so ordering is guaranteed
            slot = asyncio.Queue()
            playback_slots[seq] = slot
            next_slot_ready.set()  # Wake playback worker if waiting

            try:
                # STT — use speculative result if available
                if pre_transcribed:
                    src_text = pre_transcribed
                    logger.info("[Chunk %d] Using speculative STT result", seq)
                else:
                    src_text = await transcriber.transcribe(wav_bytes)
                if not src_text:
                    return

                await broadcast({
                    "type": "stt",
                    "seq": seq,
                    "total": 0,
                    "text": src_text,
                })

                # Translate
                tgt_text = await translator.translate(src_text)
                if not tgt_text:
                    return

                latency = time.time() - t0
                total_latency += latency
                state.stats["chunks_processed"] = seq
                state.stats["avg_latency"] = total_latency / seq

                entry = {
                    "seq": seq,
                    "source": src_text,
                    "translated": tgt_text,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "latency": round(latency, 1),
                    "timestamp": time.time(),
                    "ukrainian": src_text,
                    "english": tgt_text,
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

                # TTS — stream chunks into our slot
                # Playback worker will drain this slot when it's our turn
                tts_started = False
                async for audio_chunk in synthesizer.synthesize_stream(tgt_text):
                    if not tts_started:
                        tts_time_to_first = time.time() - t0 - latency
                        logger.info("[Chunk %d] TTS first audio in %.2fs", seq, tts_time_to_first)
                        tts_started = True
                    await slot.put(audio_chunk)

            finally:
                # Always send end marker so playback worker advances
                await slot.put(None)

        # Main loop with speculative STT:
        # 1. When VAD emits a "preview", start STT speculatively
        # 2. When "final" arrives, use speculative result if ready, or transcribe full chunk
        # This overlaps STT processing with VAD capture, saving 1-2s
        MAX_CONCURRENT = 3  # Process up to 3 chunks simultaneously
        # (STT + translate run concurrently; playback is serialized in order)
        speculative_stt_task: Optional[asyncio.Task] = None
        speculative_stt_result: Optional[str] = None

        async def _speculative_transcribe(wav_bytes: bytes):
            """Run STT on preview audio in background."""
            nonlocal speculative_stt_result
            try:
                result = await transcriber.transcribe(wav_bytes)
                speculative_stt_result = result
                logger.info("Speculative STT complete: %s", 
                           (result[:60] + "...") if result and len(result) > 60 else result)
            except Exception as e:
                logger.warning("Speculative STT failed: %s", e)
                speculative_stt_result = None

        while state.live_running:
            try:
                tagged = await capture.get_chunk()
            except Exception:
                if not state.live_running:
                    break
                continue
            if tagged is None:
                continue

            tag, wav_bytes = tagged

            if tag == "preview":
                # Start speculative STT in background — don't block capture
                if speculative_stt_task is None or speculative_stt_task.done():
                    speculative_stt_result = None
                    speculative_stt_task = asyncio.create_task(
                        _speculative_transcribe(wav_bytes)
                    )
                    logger.info("Speculative STT started on preview chunk")
                continue

            # tag == "final" — process the complete chunk
            chunk_count += 1

            # Check if speculative STT already has our answer
            pre_transcribed = None
            if speculative_stt_task is not None:
                if speculative_stt_task.done() and speculative_stt_result:
                    pre_transcribed = speculative_stt_result
                    logger.info("Using speculative STT result (saved STT wait time!)")
                else:
                    # Cancel pending speculative — we'll transcribe the full chunk
                    if not speculative_stt_task.done():
                        speculative_stt_task.cancel()
                        logger.info("Speculative STT not ready, transcribing full chunk")
                speculative_stt_task = None
                speculative_stt_result = None

            task = asyncio.create_task(
                _process_chunk(wav_bytes, chunk_count, pre_transcribed=pre_transcribed)
            )
            processing_tasks.add(task)
            task.add_done_callback(processing_tasks.discard)
            task.add_done_callback(_log_live_chunk_result)

            # If at max concurrency, wait for one to finish before accepting next
            if len(processing_tasks) >= MAX_CONCURRENT:
                done, _ = await asyncio.wait(processing_tasks, return_when=asyncio.FIRST_COMPLETED)
                processing_tasks -= done

        # Wait for in-flight processing to finish
        if processing_tasks:
            await asyncio.gather(*processing_tasks, return_exceptions=True)

    except asyncio.CancelledError:
        logger.info("Live pipeline cancelled")
    except Exception as e:
        logger.error("Live pipeline error: %s", e, exc_info=True)
        await broadcast({"type": "error", "message": f"Live translation error: {e}"})
    finally:
        if processing_tasks:
            for task in list(processing_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Signal playback worker to stop
        playback_active = False
        next_slot_ready.set()  # Wake it up if waiting
        if playback_task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await playback_task

        # Ensure all audio resources are released
        if state.live_capture:
            try:
                await state.live_capture.stop()
            except Exception:
                pass
            state.live_capture = None
        if state.live_aes67:
            try:
                state.live_aes67.stop()
            except Exception:
                pass
            state.live_aes67 = None
        state.live_playback = None
        state.live_running = False
        state.running = False
        state.stats["status"] = "stopped"
        state.stats["total_runtime"] = time.time() - state.start_time
        state.live_pipeline = None
        state.live_task = None
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
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid WebSocket payload"})
                    continue
                if data.get("action") == "ping":
                    await ws.send_json({"type": "pong"})
            elif msg.type == web.WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
    finally:
        _discard_client(ws)
        logger.info("Dashboard client disconnected (%d remaining)", len(state.connected_clients))

    return ws


async def on_shutdown(app):
    """Release live audio resources and close WebSockets on server shutdown."""
    if state.live_running or state.live_task:
        await _stop_live_pipeline(notify_clients=False)

    for ws in list(state.connected_clients):
        with contextlib.suppress(Exception):
            await ws.close()
        _discard_client(ws)


# ── App Setup ───────────────────────────────────────────────────

def create_app():
    app = web.Application(middlewares=[auth_middleware])
    app.on_shutdown.append(on_shutdown)
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
    app.router.add_get("/api/audio-levels", api_audio_levels)
    app.router.add_post("/api/audio-level", api_audio_level_single)
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
