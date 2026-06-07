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
    assert len(payload["input"]) == 1 and len(payload["output"]) == 1
    mic = payload["input"][0]
    assert mic["index"] == 0 and mic["name"] == "USB Mic" and mic["sample_rate"] == 48000
    assert "fingerprint" in mic and mic["remembered"] is False
    spk = payload["output"][0]
    assert spk["index"] == 1 and spk["name"] == "Main Speakers" and spk["sample_rate"] == 48000


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


# ── Live-apply endpoint ───────────────────────────────────────────────

class _FakeTranscriber:
    def __init__(self):
        self.model = "old-stt"
        self.language = "uk"


class _FakeTranslator:
    def __init__(self):
        self.model = "old-trans"
        self.system_prompt = "base"
        self.source_language = "Ukrainian"
        self.target_language = "English"


class _FakeSynth:
    def __init__(self):
        self.el_voice_id = "old-voice"
        self.el_model = "old-tts"


@pytest.mark.asyncio
async def test_apply_persists_when_not_live(monkeypatch):
    from dashboard import server

    response = await server.api_apply(
        DummyRequest("/api/apply", data={"translation_model": "gpt-4o-mini"})
    )
    payload = decode_json_response(response)
    assert payload["ok"] is True
    assert payload["live"] is False
    assert payload["applied"] == []
    assert server.load_saved_config()["translation_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_apply_hot_swaps_running_components(monkeypatch):
    from dashboard import server

    tr, tl, sy = _FakeTranscriber(), _FakeTranslator(), _FakeSynth()
    server.state.live_running = True
    server.state.live_transcriber = tr
    server.state.live_translator = tl
    server.state.live_synthesizer = sy
    server.state.live_settings = {"input_device": "1", "output_device": "2"}
    monkeypatch.setattr(server, "broadcast", lambda msg: asyncio.sleep(0))

    response = await server.api_apply(DummyRequest("/api/apply", data={
        "stt_model": "new-stt",
        "translation_model": "new-trans",
        "tts_model": "new-tts",
        "elevenlabs_voice_id": "new-voice",
        "target_language": "es",
    }))
    payload = decode_json_response(response)

    assert payload["live"] is True
    assert payload["restart_needed"] is False
    assert tr.model == "new-stt"
    assert tl.model == "new-trans"
    assert tl.target_language == "Spanish"
    assert sy.el_voice_id == "new-voice"
    assert sy.el_model == "new-tts"
    assert set(payload["applied"]) >= {"STT model", "translation model", "TTS model", "voice", "target language"}


@pytest.mark.asyncio
async def test_apply_flags_restart_on_device_change(monkeypatch):
    from dashboard import server

    server.state.live_running = True
    server.state.live_transcriber = _FakeTranscriber()
    server.state.live_translator = _FakeTranslator()
    server.state.live_synthesizer = _FakeSynth()
    server.state.live_settings = {"input_device": "1", "output_device": "2"}
    monkeypatch.setattr(server, "broadcast", lambda msg: asyncio.sleep(0))

    response = await server.api_apply(
        DummyRequest("/api/apply", data={"input_device": "5"})
    )
    payload = decode_json_response(response)

    assert payload["restart_needed"] is True
    assert "device" in payload["restart_reason"]


# ── Transcript export ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_transcript_txt_and_srt():
    from dashboard import server

    server.state.transcript = [
        {"seq": 1, "source": "Привіт", "translated": "Hello", "timestamp": 1000.0},
        {"seq": 2, "source": "світ", "translated": "World", "timestamp": 1003.0},
    ]

    txt = await server.api_export_transcript(DummyRequest("/api/transcript/export", query={"format": "txt"}))
    assert txt.status == 200
    assert "Hello" in txt.text and "[#1]" in txt.text
    assert txt.headers["Content-Disposition"].endswith('transcript.txt"')

    srt = await server.api_export_transcript(DummyRequest("/api/transcript/export", query={"format": "srt"}))
    assert srt.status == 200
    assert "-->" in srt.text and "World" in srt.text
    assert "00:00:00,000 --> 00:00:03,000" in srt.text


# ── Live tuning ───────────────────────────────────────────────────────

class _FakeOpenAIClient:
    def __init__(self):
        self.opts = None

    def with_options(self, **kw):
        new = _FakeOpenAIClient()
        new.opts = kw
        return new


class _FakeComponent:
    """Stands in for Transcriber/Translator with the attrs tuning touches."""
    def __init__(self):
        self.client = _FakeOpenAIClient()
        self.gate_silence = True
        self.silence_peak = 0.008
        self.min_duration_sec = 0.4
        self.filter_hallucinations = True
        self.temperature = 0.0


class _FakeSynth:
    def __init__(self):
        self.timeout = 30.0
        self.max_retries = 2


class _FakeCapture:
    def __init__(self):
        self.chunking_calls = []

    def update_chunking(self, **kw):
        self.chunking_calls.append(kw)


@pytest.mark.asyncio
async def test_tuning_endpoint_returns_effective_values():
    from dashboard import server

    resp = await server.api_tuning(DummyRequest("/api/tuning"))
    assert resp.status == 200
    payload = decode_json_response(resp)
    # A representative key from each group is present.
    for key in ("silence_peak", "vad_aggressiveness", "api_timeout",
                "mic_watchdog_sec", "output_mode", "aes67_port"):
        assert key in payload


def test_coerce_tuning_clamps_and_types():
    from dashboard import server

    out = server._coerce_tuning({
        "silence_peak": "0.99",          # clamped to 0.2
        "max_retries": "10",             # clamped to 6, int
        "vad_aggressiveness": 5,         # clamped to 3
        "gate_silence": 0,               # -> bool False
        "mic_watchdog_sec": 1.0,         # clamped up to 2.0
    })
    assert out["silence_peak"] == 0.2
    assert out["max_retries"] == 6 and isinstance(out["max_retries"], int)
    assert out["vad_aggressiveness"] == 3
    assert out["gate_silence"] is False
    assert out["mic_watchdog_sec"] == 2.0


def test_apply_tuning_hot_swaps_into_live_components():
    from dashboard import server

    transcriber = _FakeComponent()
    translator = _FakeComponent()
    synth = _FakeSynth()
    capture = _FakeCapture()
    server.state.live_transcriber = transcriber
    server.state.live_translator = translator
    server.state.live_synthesizer = synth
    server.state.live_capture = capture
    server.state.live_tuning = {}
    try:
        data = server._coerce_tuning({
            "gate_silence": False,
            "silence_peak": 0.02,
            "stt_temperature": 0.3,
            "translation_temperature": 0.5,
            "filter_hallucinations": False,
            "api_timeout": 12.0,
            "max_retries": 4,
            "mic_watchdog_sec": 9.0,
            "vad_aggressiveness": 3,
            "min_chunk_sec": 2.5,
        })
        applied = server._apply_tuning_to_live(data)

        assert transcriber.gate_silence is False
        assert transcriber.silence_peak == 0.02
        assert transcriber.temperature == 0.3
        assert translator.temperature == 0.5
        assert transcriber.filter_hallucinations is False
        assert translator.filter_hallucinations is False
        # API options pushed through with_options / synth attrs
        assert transcriber.client.opts == {"timeout": 12.0, "max_retries": 4}
        assert synth.timeout == 12.0 and synth.max_retries == 4
        # VAD update mapped aggressiveness + passed min_chunk_sec
        assert capture.chunking_calls == [{"aggressiveness": 3, "min_chunk_sec": 2.5}]
        # Watchdog threshold landed in shared tuning snapshot
        assert server.state.live_tuning["mic_watchdog_sec"] == 9.0
        assert "silence gate" in applied and "VAD/chunking" in applied
    finally:
        server.state.live_transcriber = None
        server.state.live_translator = None
        server.state.live_synthesizer = None
        server.state.live_capture = None
        server.state.live_tuning = {}


@pytest.mark.asyncio
async def test_apply_aes67_off_stops_sender():
    from dashboard import server

    class _FakeSender:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    sender = _FakeSender()
    server.state.live_aes67 = sender
    try:
        labels = await server._apply_aes67_to_live({"output_mode": "sounddevice"})
        assert sender.stopped is True
        assert server.state.live_aes67 is None
        assert labels == ["AES67 output (off)"]
    finally:
        server.state.live_aes67 = None


@pytest.mark.asyncio
async def test_apply_aes67_dante_starts_sender(monkeypatch):
    import src.aes67_output as aes_mod
    from dashboard import server

    created = {}

    class _FakeSender:
        def __init__(self, *, stream_name, multicast_addr, port, ttl):
            created.update(stream_name=stream_name, multicast_addr=multicast_addr, port=port, ttl=ttl)
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            pass

    monkeypatch.setattr(aes_mod, "AES67Sender", _FakeSender)
    server.state.live_aes67 = None
    try:
        labels = await server._apply_aes67_to_live({
            "output_mode": "dante",
            "aes67_stream_name": "Test Stream",
            "aes67_port": 5006,
        })
        assert server.state.live_aes67 is not None
        assert server.state.live_aes67.started is True
        assert created["stream_name"] == "Test Stream" and created["port"] == 5006
        assert labels == ["AES67 output (dante)"]
    finally:
        server.state.live_aes67 = None


@pytest.mark.asyncio
async def test_apply_skips_aes67_restart_when_unchanged(monkeypatch):
    from dashboard import server

    calls = {"n": 0}

    async def fake_apply_aes67(data):
        calls["n"] += 1
        return ["AES67 output (dante)"]

    monkeypatch.setattr(server, "_apply_aes67_to_live", fake_apply_aes67)

    server.state.live_running = True
    server.state.live_settings = {
        "output_mode": "dante", "aes67_stream_name": "S",
        "aes67_multicast": "239.69.0.1", "aes67_port": 5004, "aes67_ttl": 32,
    }
    server.state.live_tuning = {}
    try:
        # Re-send identical AES params -> no restart.
        await server.api_apply(DummyRequest("/api/apply", data={
            "output_mode": "dante", "aes67_stream_name": "S",
            "aes67_multicast": "239.69.0.1", "aes67_port": 5004, "aes67_ttl": 32,
        }))
        assert calls["n"] == 0

        # Change the port -> restart fires once.
        await server.api_apply(DummyRequest("/api/apply", data={
            "output_mode": "dante", "aes67_stream_name": "S",
            "aes67_multicast": "239.69.0.1", "aes67_port": 5005, "aes67_ttl": 32,
        }))
        assert calls["n"] == 1
    finally:
        server.state.live_running = False
        server.state.live_settings = {}
        server.state.live_tuning = {}


# ── TTS speed + STT provider live settings ────────────────────────────

@pytest.mark.asyncio
async def test_tuning_includes_tts_speed():
    from dashboard import server
    resp = await server.api_tuning(DummyRequest("/api/tuning"))
    payload = decode_json_response(resp)
    assert "tts_speed" in payload


def test_coerce_clamps_tts_speed():
    from dashboard import server
    assert server._coerce_tuning({"tts_speed": 5.0})["tts_speed"] == 1.2
    assert server._coerce_tuning({"tts_speed": 0.1})["tts_speed"] == 0.7


def test_apply_tuning_sets_synth_speed():
    from dashboard import server

    class _Synth:
        def __init__(self):
            self.speed = 1.0

    synth = _Synth()
    server.state.live_synthesizer = synth
    server.state.live_tuning = {}
    try:
        applied = server._apply_tuning_to_live(server._coerce_tuning({"tts_speed": 1.15}))
        assert synth.speed == 1.15
        assert "TTS speed" in applied
    finally:
        server.state.live_synthesizer = None
        server.state.live_tuning = {}


def test_apply_stt_provider_hot_swap():
    from dashboard import server

    class _T:
        def __init__(self):
            self.provider = "openai"

    t = _T()
    server.state.live_transcriber = t
    try:
        applied = server._apply_to_live_components({"stt_provider": "elevenlabs"})
        assert t.provider == "elevenlabs"
        assert "STT provider" in applied
    finally:
        server.state.live_transcriber = None


# ── Deepgram API key + per-provider STT model ─────────────────────────

@pytest.mark.asyncio
async def test_setup_save_writes_deepgram_key(monkeypatch, tmp_path):
    import os
    from dashboard import server

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    try:
        resp = await server.api_setup_save(
            DummyRequest("/api/setup/save", data={"deepgram_api_key": "dg-123"})
        )
        assert decode_json_response(resp) == {"ok": True}
        assert "DEEPGRAM_API_KEY=dg-123" in (tmp_path / ".env").read_text()
        assert os.environ["DEEPGRAM_API_KEY"] == "dg-123"

        status = await server.api_setup_status(DummyRequest("/api/setup/status"))
        assert decode_json_response(status)["has_deepgram"] is True
    finally:
        os.environ.pop("DEEPGRAM_API_KEY", None)


@pytest.mark.asyncio
async def test_tuning_exposes_per_provider_stt_models():
    from dashboard import server
    payload = decode_json_response(await server.api_tuning(DummyRequest("/api/tuning")))
    assert "deepgram_model" in payload and "elevenlabs_model" in payload


def test_apply_deepgram_and_elevenlabs_model_hot_swap():
    from dashboard import server

    class _T:
        def __init__(self):
            self.deepgram_model = "nova-3"
            self.elevenlabs_model = "scribe_v2"

    t = _T()
    server.state.live_transcriber = t
    try:
        applied = server._apply_to_live_components({
            "deepgram_model": "nova-2", "elevenlabs_model": "scribe_v2",
        })
        assert t.deepgram_model == "nova-2"
        assert "Deepgram model" in applied and "ElevenLabs STT model" in applied
    finally:
        server.state.live_transcriber = None


@pytest.mark.asyncio
async def test_setup_save_clickthrough_preserves_existing_keys(monkeypatch, tmp_path):
    """Wizard click-through (all keys blank) must not clobber or write blanks."""
    import os
    from dashboard import server

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-existing\n")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    try:
        resp = await server.api_setup_save(DummyRequest("/api/setup/save", data={
            "openai_api_key": "", "elevenlabs_api_key": "", "deepgram_api_key": "",
            "source_language": "uk", "target_language": "en",
        }))
        assert decode_json_response(resp) == {"ok": True}
        env = (tmp_path / ".env").read_text()
        assert "OPENAI_API_KEY=sk-existing" in env   # preserved
        assert "DEEPGRAM_API_KEY" not in env          # blank not written
        assert "DEEPGRAM_API_KEY" not in os.environ
    finally:
        os.environ.pop("DEEPGRAM_API_KEY", None)


# ── Endpoint smoke test (no GET endpoint should 5xx) ──────────────────

@pytest.mark.asyncio
async def test_get_endpoints_do_not_5xx(monkeypatch, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from dashboard import server

    # Isolate persistence + avoid external calls.
    monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice_for_smoke())

    client = TestClient(TestServer(server.create_app()))
    await client.start_server()
    try:
        paths = [
            "/",
            "/api/languages",
            "/api/tuning",
            "/api/transcript",
            "/api/transcript/export?format=txt",
            "/api/transcript/export?format=srt",
            "/api/config",
            "/api/settings",
            "/api/setup/status",
            "/api/health",
            "/api/devices",
        ]
        for path in paths:
            resp = await client.get(path)
            assert resp.status < 500, f"{path} returned {resp.status}"
    finally:
        await client.close()


def fake_sounddevice_for_smoke():
    """Minimal sounddevice stand-in so /api/health and /api/devices don't error."""
    devices = [
        {"name": "Mic", "default_samplerate": 48000, "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Speaker", "default_samplerate": 48000, "max_input_channels": 0, "max_output_channels": 2},
    ]

    def query_devices(index=None):
        return devices if index is None else devices[index]

    return SimpleNamespace(
        query_devices=query_devices,
        check_input_settings=lambda **k: None,
        check_output_settings=lambda **k: None,
        default=SimpleNamespace(device=(0, 1)),
    )
