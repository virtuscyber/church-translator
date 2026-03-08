from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def fake_sounddevice_module():
    devices = [
        {
            "name": "USB Mic",
            "default_samplerate": 48000,
            "max_input_channels": 1,
            "max_output_channels": 0,
        },
        {
            "name": "Main Speakers",
            "default_samplerate": 48000,
            "max_input_channels": 0,
            "max_output_channels": 2,
        },
    ]
    return SimpleNamespace(query_devices=lambda: devices)


@pytest.fixture(autouse=True)
def reset_dashboard_state(monkeypatch, tmp_path):
    from dashboard import server

    server.CONFIG_PATH = tmp_path / "config.json"
    server.DASHBOARD_API_KEY = ""
    server.CORS_ORIGIN = ""
    server.state.running = False
    server.state.live_running = False
    server.state.connected_clients = []
    server.state.transcript = []
    server.state.stats = {
        "chunks_processed": 0,
        "avg_latency": 0.0,
        "total_runtime": 0.0,
        "status": "stopped",
    }
    server.state.start_time = 0.0
    server.state.live_pipeline = None
    server.state.live_task = None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    yield
    if server.state.live_task and not server.state.live_task.done():
        server.state.live_task.cancel()
