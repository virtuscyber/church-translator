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
    connected_clients: list = []
    transcript: list[dict] = []
    stats: dict = {
        "chunks_processed": 0,
        "avg_latency": 0.0,
        "total_runtime": 0.0,
        "status": "stopped",
    }
    start_time: float = 0.0

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
    allowed = {"input_device", "output_device", "source_language", "target_language", "custom_vocabulary"}
    filtered = {k: v for k, v in data.items() if k in allowed}
    save_config(filtered)
    return web.json_response({"ok": True})


async def api_languages(request):
    """List supported languages."""
    return web.json_response({"languages": SUPPORTED_LANGUAGES})


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
    app.router.add_get("/api/languages", api_languages)
    # Feature 3: Setup wizard
    app.router.add_get("/api/setup/status", api_setup_status)
    app.router.add_post("/api/setup/test-openai", api_setup_test_openai)
    app.router.add_post("/api/setup/test-elevenlabs", api_setup_test_elevenlabs)
    app.router.add_post("/api/setup/save", api_setup_save)
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
