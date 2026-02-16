#!/usr/bin/env python3
"""Dashboard web server for Church Live Translation.

Serves the dashboard UI and provides WebSocket endpoints for real-time
translation status, transcript updates, and pipeline control.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from aiohttp import web

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ── Auth & CORS Middleware ──────────────────────────────────────

DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "")
CORS_ORIGIN = os.getenv("DASHBOARD_CORS_ORIGIN", "")

# Paths that don't require auth (static assets, websocket handled separately)
_PUBLIC_PATHS = {"/", "/ws"}


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Require Bearer token for API routes when DASHBOARD_API_KEY is set."""
    if DASHBOARD_API_KEY and request.path not in _PUBLIC_PATHS:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {DASHBOARD_API_KEY}":
            return web.json_response({"error": "Unauthorized"}, status=401)
    response = await handler(request)
    # CORS headers
    origin = CORS_ORIGIN or "http://localhost:8080"
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Global state
class AppState:
    """Shared state between dashboard and pipeline."""
    running: bool = False
    connected_clients: list = []
    transcript: list[dict] = []
    stats: dict = {
        "chunks_processed": 0,
        "avg_latency": 0.0,
        "total_runtime": 0.0,
        "status": "stopped",  # stopped, running, error
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
    
    # Run test in background
    asyncio.create_task(_run_file_test(file_path))
    
    return web.json_response({"status": "started", "file": file_path})


async def _run_file_test(file_path: str):
    """Background task to run file translation and stream results."""
    try:
        from src.config import load_config
        from src.transcriber import Transcriber
        from src.translator import Translator
        from src.vad_chunker import FileVADChunker
        import subprocess
        import tempfile
        import wave
        import io

        config = load_config()
        
        transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=config.transcription.language,
        )
        translator = Translator(
            api_key=config.openai_api_key,
            system_prompt=config.translation.system_prompt,
            model=config.translation.model,
            temperature=config.translation.temperature,
            context_sentences=config.pipeline.context_sentences,
        )
        # Convert and chunk
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
        
        total_latency = 0.0
        
        for i, wav_chunk in enumerate(chunks):
            if not state.running:
                break
            
            t0 = time.time()
            
            # STT
            uk_text = await transcriber.transcribe(wav_chunk)
            if not uk_text:
                continue
            
            await broadcast({
                "type": "stt",
                "seq": i + 1,
                "total": len(chunks),
                "text": uk_text,
            })
            
            # Translate
            en_text = await translator.translate(uk_text)
            if not en_text:
                continue
            
            # Skip TTS in test mode to avoid wasting credits
            # Audio is not played or returned in dashboard test pipeline
            
            latency = time.time() - t0
            total_latency += latency
            state.stats["chunks_processed"] = i + 1
            state.stats["avg_latency"] = total_latency / (i + 1)
            
            # Build transcript entry
            entry = {
                "seq": i + 1,
                "ukrainian": uk_text,
                "english": en_text,
                "latency": round(latency, 1),
                "timestamp": time.time(),
            }
            state.transcript.append(entry)
            
            # Stream to dashboard
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
    
    # Send current state
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
    # Serve static assets
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static/", static_path)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    app = create_app()
    logger.info("Dashboard starting on http://localhost:%d", port)
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    web.run_app(app, host=host, port=port)
