from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest


class DummyRequest:
    def __init__(self, path: str, data: dict | None = None, headers: dict | None = None):
        self.path = path
        self._data = data or {}
        self.headers = headers or {}

    async def json(self):
        return self._data


def decode_json_response(response):
    return json.loads(response.text)


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
async def test_settings_round_trip_preserves_multi_language_model_fields():
    from dashboard import server

    save = await server.api_save_config(
        DummyRequest(
            "/api/settings",
            data={
                "source_language": "uk",
                "target_language": "en",
                "target_languages": ["en", "ru", "pl"],
                "stt_model": "gpt-4o-transcribe",
                "translation_model": "gpt-4o",
                "tts_model": "eleven_turbo_v2_5",
                "multi_language_mode": True,
            },
        )
    )

    assert save.status == 200
    assert decode_json_response(save) == {"ok": True}

    loaded = await server.api_get_config(DummyRequest("/api/settings"))

    assert loaded.status == 200
    assert decode_json_response(loaded) == {
        "source_language": "uk",
        "target_language": "en",
        "target_languages": ["en", "ru", "pl"],
        "stt_model": "gpt-4o-transcribe",
        "translation_model": "gpt-4o",
        "tts_model": "eleven_turbo_v2_5",
        "multi_language_mode": True,
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
