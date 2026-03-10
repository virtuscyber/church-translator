from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


class DummyRequest:
    def __init__(self, path: str, data: dict | None = None, headers: dict | None = None, query: dict | None = None):
        self.path = path
        self._data = data or {}
        self.headers = headers or {}
        self.query = query or {}

    async def json(self):
        return self._data


def decode_json_response(response):
    return json.loads(response.text)


def make_fake_monitor_sounddevice():
    devices = [
        {
            "name": "Hot Mic",
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
        {
            "name": "Broken Mic",
            "default_samplerate": 48000,
            "max_input_channels": 1,
            "max_output_channels": 0,
        },
    ]

    class FakeInputStream:
        def __init__(self, *, device, samplerate, channels, dtype, blocksize):
            self.device = device
            self.blocksize = blocksize

        def __enter__(self):
            if self.device == 2:
                raise RuntimeError("device busy")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, frames):
            if self.device == 0:
                audio = np.full((frames, 1), 0.5, dtype=np.float32)
            else:
                audio = np.zeros((frames, 1), dtype=np.float32)
            return audio, False

    def query_devices(index=None):
        if index is None:
            return devices
        return devices[index]

    def check_input_settings(*, device, **kwargs):
        if device == 1:
            raise RuntimeError("not an input")

    return SimpleNamespace(
        query_devices=query_devices,
        check_input_settings=check_input_settings,
        InputStream=FakeInputStream,
    )


@pytest.mark.asyncio
async def test_dashboard_root_serves_html():
    from dashboard import server

    response = await server.index(DummyRequest("/"))

    assert response.status == 200
    assert response._path.name == "index.html"


@pytest.mark.asyncio
async def test_health_endpoint_returns_expected_shape(monkeypatch, fake_sounddevice_module):
    from dashboard import server

    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice_module)

    response = await server.api_health(DummyRequest("/api/health"))

    assert response.status == 200
    payload = decode_json_response(response)
    assert "healthy" in payload
    assert set(payload["checks"]) == {"python", "ffmpeg", "api_keys", "audio", "network"}
    assert payload["checks"]["python"]["ok"] is True
    assert payload["checks"]["audio"]["inputs"] == 1
    assert payload["checks"]["audio"]["outputs"] == 1


@pytest.mark.asyncio
async def test_devices_endpoint_lists_audio_devices(monkeypatch, fake_sounddevice_module):
    from dashboard import server

    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice_module)

    response = await server.api_devices(DummyRequest("/api/devices"))

    assert response.status == 200
    payload = decode_json_response(response)
    assert payload["input"] == [{"index": 0, "name": "USB Mic", "sample_rate": 48000}]
    assert payload["output"] == [{"index": 1, "name": "Main Speakers", "sample_rate": 48000}]


@pytest.mark.asyncio
async def test_audio_level_single_samples_selected_input(monkeypatch):
    from dashboard import server

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_monitor_sounddevice())

    response = await server.api_audio_level_single(
        DummyRequest("/api/audio-level", data={"device_index": 0})
    )

    assert response.status == 200
    payload = decode_json_response(response)
    assert payload["index"] == 0
    assert payload["has_audio"] is True
    assert payload["peak_db"] == -6.0
    assert payload["level_pct"] == 90


@pytest.mark.asyncio
async def test_audio_level_single_rejects_output_only_device(monkeypatch):
    from dashboard import server

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_monitor_sounddevice())

    response = await server.api_audio_level_single(
        DummyRequest("/api/audio-level", data={"device_index": 1})
    )

    assert response.status == 400
    assert decode_json_response(response)["error"] == "Selected device is not an input device"


@pytest.mark.asyncio
async def test_audio_levels_marks_failed_devices_without_crashing(monkeypatch):
    from dashboard import server

    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_monitor_sounddevice())

    response = await server.api_audio_levels(DummyRequest("/api/audio-levels"))

    assert response.status == 200
    payload = decode_json_response(response)
    assert payload["devices"][0]["name"] == "Hot Mic"
    assert payload["devices"][0]["has_audio"] is True
    assert payload["devices"][1]["name"] == "Broken Mic"
    assert payload["devices"][1]["error"] is True


@pytest.mark.asyncio
async def test_settings_round_trip_uses_alias_routes():
    from dashboard import server

    app = server.create_app()
    paths = {route.resource.canonical for route in app.router.routes()}
    assert "/api/settings" in paths

    initial = await server.api_get_config(DummyRequest("/api/settings"))
    assert initial.status == 200
    assert decode_json_response(initial) == {}

    save = await server.api_save_config(
        DummyRequest(
            "/api/settings",
            data={
                "input_device": 1,
                "output_device": 2,
                "source_language": "uk",
                "target_language": "en",
                "custom_vocabulary": "Psalm, grace",
                "ignored": "value",
            },
        )
    )
    assert save.status == 200
    assert decode_json_response(save) == {"ok": True}

    loaded = await server.api_get_config(DummyRequest("/api/settings"))
    assert loaded.status == 200
    assert decode_json_response(loaded) == {
        "input_device": 1,
        "output_device": 2,
        "source_language": "uk",
        "target_language": "en",
        "custom_vocabulary": "Psalm, grace",
    }


@pytest.mark.asyncio
async def test_live_start_and_stop_are_graceful_without_real_services(monkeypatch):
    from dashboard import server

    async def fake_run_live_pipeline():
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(server, "_run_live_pipeline", fake_run_live_pipeline)

    start = await server.api_start_live(DummyRequest("/api/start-live"))
    assert start.status == 200
    assert decode_json_response(start) == {"status": "started", "mode": "live"}

    stop = await server.api_stop_live(DummyRequest("/api/stop-live"))
    assert stop.status == 200
    assert decode_json_response(stop)["status"] == "stopped"


@pytest.mark.asyncio
async def test_test_file_rejects_when_pipeline_is_busy(tmp_path):
    from dashboard import server

    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"fake")
    server.state.live_running = True

    response = await server.api_test_file(
        DummyRequest("/api/test-file", data={"file_path": str(wav_path)})
    )

    assert response.status == 400
    assert decode_json_response(response) == {"error": "Already running"}


@pytest.mark.asyncio
async def test_auth_middleware_allows_setup_only_before_configuration(monkeypatch):
    from aiohttp import web
    from dashboard import server

    server.DASHBOARD_API_KEY = "secret"

    async def handler(request):
        return web.json_response({"ok": True})

    monkeypatch.setattr(server, "_has_configured_openai_key", lambda: False)
    allowed = await server.auth_middleware(DummyRequest("/api/setup/status"), handler)
    assert allowed.status == 200

    monkeypatch.setattr(server, "_has_configured_openai_key", lambda: True)
    blocked = await server.auth_middleware(DummyRequest("/api/setup/status"), handler)
    assert blocked.status == 401


@pytest.mark.asyncio
async def test_auth_middleware_accepts_websocket_query_token():
    from aiohttp import web
    from dashboard import server

    server.DASHBOARD_API_KEY = "secret"

    async def handler(request):
        return web.json_response({"ok": True})

    response = await server.auth_middleware(
        DummyRequest("/ws", query={"token": "secret"}),
        handler,
    )

    assert response.status == 200


@pytest.mark.asyncio
async def test_websocket_connects_and_responds_to_ping(monkeypatch):
    from dashboard import server

    class FakeMessage:
        def __init__(self, msg_type, data):
            self.type = msg_type
            self.data = data

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self._messages = [
                FakeMessage(server.web.WSMsgType.TEXT, json.dumps({"action": "ping"})),
            ]

        async def prepare(self, request):
            return self

        async def send_json(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            raise StopAsyncIteration

    fake_ws = FakeWebSocket()
    monkeypatch.setattr(server.web, "WebSocketResponse", lambda: fake_ws)

    response = await server.websocket_handler(DummyRequest("/ws"))

    assert response is fake_ws
    assert fake_ws.sent[0]["type"] == "init"
    assert fake_ws.sent[1] == {"type": "pong"}
    assert server.state.connected_clients == []


@pytest.mark.asyncio
async def test_websocket_rejects_invalid_json_without_crashing(monkeypatch):
    from dashboard import server

    class FakeMessage:
        def __init__(self, msg_type, data):
            self.type = msg_type
            self.data = data

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self._messages = [
                FakeMessage(server.web.WSMsgType.TEXT, "{not-json"),
            ]

        async def prepare(self, request):
            return self

        async def send_json(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._messages:
                return self._messages.pop(0)
            raise StopAsyncIteration

    fake_ws = FakeWebSocket()
    monkeypatch.setattr(server.web, "WebSocketResponse", lambda: fake_ws)

    response = await server.websocket_handler(DummyRequest("/ws"))

    assert response is fake_ws
    assert fake_ws.sent[1] == {"type": "error", "message": "Invalid WebSocket payload"}
    assert server.state.connected_clients == []
