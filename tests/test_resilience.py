from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.synthesizer import Synthesizer
from src.transcriber import Transcriber
from src.translator import Translator


def _wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── Transcriber error signal ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_transcriber_sets_last_error_on_api_failure():
    t = Transcriber(api_key="x", gate_silence=False)
    t.client = MagicMock()
    t.client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("boom"))

    result = await t.transcribe(b"not-really-wav")

    assert result is None
    assert t.last_error is not None and "boom" in t.last_error


@pytest.mark.asyncio
async def test_transcriber_no_error_on_gated_silence():
    t = Transcriber(api_key="x")  # gate_silence on by default
    t.client = MagicMock()
    t.client.audio.transcriptions.create = AsyncMock(return_value="should not be called")

    result = await t.transcribe(_wav(np.zeros(16000, dtype=np.float32)))

    assert result is None
    assert t.last_error is None
    assert t.client.audio.transcriptions.create.await_count == 0


# ── Translator error signal ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_translator_sets_last_error_on_api_failure():
    tr = Translator(api_key="x", system_prompt="sp")
    tr.client = MagicMock()
    tr.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))

    result = await tr.translate("Привіт")

    assert result is None
    assert tr.last_error is not None and "api down" in tr.last_error


@pytest.mark.asyncio
async def test_translator_no_error_on_filtered_junk():
    tr = Translator(api_key="x", system_prompt="sp")
    tr.client = MagicMock()
    tr.client.chat.completions.create = AsyncMock(return_value="should not be called")

    result = await tr.translate("Thanks for watching")  # known hallucination phrase

    assert result is None
    assert tr.last_error is None
    assert tr.client.chat.completions.create.await_count == 0


# ── Translator conversation-history context ───────────────────────────

def _completion(text: str):
    msg = MagicMock()
    msg.choices = [MagicMock()]
    msg.choices[0].message.content = text
    return msg


@pytest.mark.asyncio
async def test_translator_replays_history_as_conversation_turns():
    tr = Translator(api_key="x", system_prompt="sp", context_sentences=2)
    tr.client = MagicMock()
    tr.client.chat.completions.create = AsyncMock(
        side_effect=[_completion("First."), _completion("Second.")]
    )

    assert await tr.translate("Перший") == "First."
    assert await tr.translate("Другий") == "Second."

    # The second call must carry the first chunk as a real user/assistant pair.
    messages = tr.client.chat.completions.create.await_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "Перший" in messages[1]["content"]
    assert messages[2]["content"] == "First."
    assert "Другий" in messages[3]["content"]
    # The newest request must be the only untranslated turn.
    assert "Перший" not in messages[3]["content"]


@pytest.mark.asyncio
async def test_translator_history_is_bounded_and_resettable():
    tr = Translator(api_key="x", system_prompt="sp", context_sentences=2)
    tr.client = MagicMock()
    tr.client.chat.completions.create = AsyncMock(
        side_effect=[_completion(f"T{i}") for i in range(5)]
    )

    for i in range(4):
        await tr.translate(f"Речення номер {i}")

    messages = tr.client.chat.completions.create.await_args.kwargs["messages"]
    # maxlen=2 → system + 2 pairs + new request = 6 messages, oldest aged out.
    assert len(messages) == 6
    assert all("Речення номер 0" not in m["content"] for m in messages)

    tr.reset_context()
    await tr.translate("Нове речення")
    messages = tr.client.chat.completions.create.await_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]


# ── Synthesizer retry + error signal ──────────────────────────────────

@pytest.mark.asyncio
async def test_synthesizer_retries_then_records_error():
    s = Synthesizer(provider="openai", max_retries=1, timeout=5.0)
    calls = {"n": 0}

    async def boom(text):
        calls["n"] += 1
        raise RuntimeError("tts fail")

    s._synthesize_openai = boom

    result = await s.synthesize("hello")

    assert result is None
    assert s.last_error is not None and "tts fail" in s.last_error
    assert calls["n"] == 2  # initial attempt + 1 retry


@pytest.mark.asyncio
async def test_synthesizer_clears_error_on_success():
    s = Synthesizer(provider="openai")
    s.last_error = "stale"

    async def ok(text):
        return b"pcm"

    s._synthesize_openai = ok

    result = await s.synthesize("hello")

    assert result == b"pcm"
    assert s.last_error is None


# ── Device-drop recovery mechanism ────────────────────────────────────

@pytest.mark.asyncio
async def test_vad_capture_restart_reopens_stream(monkeypatch):
    import src.vad_capture as vc

    opened = {"count": 0}

    class FakeStream:
        def __init__(self, **kw):
            pass

        def start(self):
            opened["count"] += 1

        def stop(self):
            pass

        def close(self):
            pass

    class FakeSD:
        def InputStream(self, **kw):
            return FakeStream(**kw)

    monkeypatch.setattr(vc, "_load_sounddevice", lambda: FakeSD())

    cap = vc.VADAudioCapture(device=0, sample_rate=16000)
    await cap.start()
    assert opened["count"] == 1
    assert cap.seconds_since_audio() >= 0.0

    await cap.restart()
    assert opened["count"] == 2  # stream re-opened


# ── STT language anchoring (Ukrainian, not Polish/Russian) ────────────

def test_transcriber_derives_ukrainian_anchor_prompt():
    from src.transcriber import Transcriber, stt_anchor_prompt

    t = Transcriber(api_key="x", language="uk")
    assert t.language == "uk"
    assert t.prompt and "українською" in t.prompt
    assert stt_anchor_prompt("UK") == stt_anchor_prompt("uk")  # case-insensitive
    assert stt_anchor_prompt("xx") is None  # unknown language -> no prompt


def test_explicit_prompt_overrides_language_anchor():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", language="uk", prompt="custom church terms")
    assert t.prompt == "custom church terms"


@pytest.mark.asyncio
async def test_transcribe_passes_language_and_prompt_to_api():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", language="uk", gate_silence=False)
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return "Слава Богу"

    t.client = MagicMock()
    t.client.audio.transcriptions.create = AsyncMock(side_effect=fake_create)

    out = await t.transcribe(b"fake-wav-bytes")
    assert out == "Слава Богу"
    assert captured["language"] == "uk"
    assert "українською" in captured["prompt"]  # Ukrainian anchor reached the API


# ── ElevenLabs Scribe v2 STT provider + fallback ──────────────────────

@pytest.mark.asyncio
async def test_transcriber_uses_elevenlabs_when_provider_set():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", provider="elevenlabs", elevenlabs_api_key="k", gate_silence=False)

    async def fake_el(_wav):
        return "Слава Богу"

    t._transcribe_elevenlabs = fake_el
    t._transcribe_openai = AsyncMock(side_effect=AssertionError("OpenAI should not be called"))

    out = await t.transcribe(b"wav")
    assert out == "Слава Богу"
    assert t.last_error is None


@pytest.mark.asyncio
async def test_transcriber_falls_back_to_openai_when_elevenlabs_fails():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", provider="elevenlabs", elevenlabs_api_key="k", gate_silence=False)

    async def boom(_wav):
        raise RuntimeError("scribe 500")

    async def ok(_wav):
        return "fallback text"

    t._transcribe_elevenlabs = boom
    t._transcribe_openai = ok

    out = await t.transcribe(b"wav")
    assert out == "fallback text"
    assert t.last_error is None  # recovered via fallback


@pytest.mark.asyncio
async def test_transcriber_reports_error_when_all_providers_fail():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", provider="elevenlabs", elevenlabs_api_key="k", gate_silence=False)

    async def boom(_wav):
        raise RuntimeError("down")

    t._transcribe_elevenlabs = boom
    t._transcribe_openai = boom

    out = await t.transcribe(b"wav")
    assert out is None
    assert t.last_error and "down" in t.last_error


# ── Synthesizer speed ─────────────────────────────────────────────────

def test_synthesizer_defaults_to_flash_and_accepts_speed():
    from src.synthesizer import Synthesizer

    s = Synthesizer(provider="elevenlabs", speed=1.15)
    assert s.el_model == "eleven_flash_v2_5"
    assert s.speed == 1.15


@pytest.mark.asyncio
async def test_transcribe_elevenlabs_builds_correct_request(monkeypatch):
    import aiohttp
    from src.transcriber import Transcriber

    captured = {}

    class FakeResp:
        status = 200
        async def text(self):
            return ""
        async def json(self):
            return {"text": "Слава Богу"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def post(self, url, data=None, headers=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            return FakeResp()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)

    t = Transcriber(api_key="x", provider="elevenlabs", elevenlabs_api_key="EL_KEY",
                    elevenlabs_model="scribe_v2", language="uk", gate_silence=False)
    out = await t.transcribe(b"wav-bytes")

    assert out == "Слава Богу"
    assert captured["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert captured["headers"]["xi-api-key"] == "EL_KEY"
    # model_id + language_code were added to the multipart form
    field_names = {f[0].get("name") for f in captured["data"]._fields}
    assert "model_id" in field_names and "language_code" in field_names and "file" in field_names


# ── ElevenLabs TTS over REST (streaming + prosody continuity) ─────────

def _fake_tts_http(monkeypatch, captured: dict, pcm: bytes = b"\x00" * 9600):
    """Patch aiohttp so ElevenLabs TTS calls hit a canned PCM response."""
    import aiohttp

    class FakeContent:
        def iter_chunked(self, size):
            async def gen():
                for i in range(0, len(pcm), size):
                    yield pcm[i:i + size]
            return gen()

    class FakeResp:
        status = 200
        content = FakeContent()
        async def text(self):
            return ""
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def post(self, url, params=None, json=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["json"] = json
            captured["headers"] = headers
            return FakeResp()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)


@pytest.mark.asyncio
async def test_tts_elevenlabs_builds_correct_request(monkeypatch):
    captured = {}
    _fake_tts_http(monkeypatch, captured)

    s = Synthesizer(provider="elevenlabs", elevenlabs_api_key="EL_KEY",
                    elevenlabs_voice_id="voice123", speed=1.1)
    audio = await s.synthesize("Glory to God.")

    assert audio == b"\x00" * 9600
    assert captured["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice123/stream"
    assert captured["params"] == {"output_format": "pcm_24000"}
    assert captured["headers"]["xi-api-key"] == "EL_KEY"
    body = captured["json"]
    assert body["text"] == "Glory to God."
    assert body["model_id"] == "eleven_flash_v2_5"
    assert body["voice_settings"]["speed"] == 1.1
    # First utterance — no prosody context yet.
    assert "previous_text" not in body


@pytest.mark.asyncio
async def test_tts_elevenlabs_carries_previous_text_for_prosody(monkeypatch):
    captured = {}
    _fake_tts_http(monkeypatch, captured)

    s = Synthesizer(provider="elevenlabs", elevenlabs_api_key="k")
    await s.synthesize("First sentence.")
    await s.synthesize("Second sentence.")

    assert captured["json"]["previous_text"] == "First sentence."


@pytest.mark.asyncio
async def test_tts_elevenlabs_stream_yields_chunks(monkeypatch):
    captured = {}
    _fake_tts_http(monkeypatch, captured, pcm=b"\x01" * 5000)

    s = Synthesizer(provider="elevenlabs", elevenlabs_api_key="k")
    chunks = [c async for c in s.synthesize_stream("Hello")]

    assert b"".join(chunks) == b"\x01" * 5000
    assert len(chunks) >= 2  # streamed, not one blob
    assert s.last_error is None


@pytest.mark.asyncio
async def test_tts_stream_falls_back_to_openai_batch(monkeypatch):
    s = Synthesizer(provider="elevenlabs", elevenlabs_api_key="k")

    async def boom(text):
        raise RuntimeError("el down")
        yield b""  # pragma: no cover — makes this an async generator

    async def openai_ok(text):
        return b"OPENAI-PCM"

    s._stream_elevenlabs = boom
    s._synthesize_openai = openai_ok

    chunks = [c async for c in s.synthesize_stream("Hello")]
    assert chunks == [b"OPENAI-PCM"]
    assert s.last_error is None


# ── Deepgram Nova-3 STT provider ──────────────────────────────────────

@pytest.mark.asyncio
async def test_transcriber_uses_deepgram_when_provider_set():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", provider="deepgram", deepgram_api_key="d", gate_silence=False)

    async def fake_dg(_wav):
        return "Слава Богу"

    t._transcribe_deepgram = fake_dg
    t._transcribe_openai = AsyncMock(side_effect=AssertionError("OpenAI should not be called"))

    out = await t.transcribe(b"wav")
    assert out == "Слава Богу"
    assert t.last_error is None


@pytest.mark.asyncio
async def test_transcriber_deepgram_falls_back_to_openai():
    from src.transcriber import Transcriber

    t = Transcriber(api_key="x", provider="deepgram", deepgram_api_key="d", gate_silence=False)

    async def boom(_wav):
        raise RuntimeError("deepgram 503")

    async def ok(_wav):
        return "fallback"

    t._transcribe_deepgram = boom
    t._transcribe_openai = ok

    out = await t.transcribe(b"wav")
    assert out == "fallback"
    assert t.last_error is None


@pytest.mark.asyncio
async def test_transcribe_deepgram_builds_correct_request(monkeypatch):
    import aiohttp
    from src.transcriber import Transcriber

    captured = {}

    class FakeResp:
        status = 200
        async def text(self):
            return ""
        async def json(self):
            return {"results": {"channels": [{"alternatives": [{"transcript": "Слава Богу"}]}]}}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def post(self, url, params=None, data=None, headers=None):
            captured.update(url=url, params=params, data=data, headers=headers)
            return FakeResp()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)

    t = Transcriber(api_key="x", provider="deepgram", deepgram_api_key="DG_KEY",
                    deepgram_model="nova-3", language="uk", gate_silence=False)
    out = await t.transcribe(b"wav-bytes")

    assert out == "Слава Богу"
    assert captured["url"] == "https://api.deepgram.com/v1/listen"
    assert captured["headers"]["Authorization"] == "Token DG_KEY"
    assert captured["params"]["model"] == "nova-3"
    assert captured["params"]["language"] == "uk"
    assert captured["data"] == b"wav-bytes"


# ── Streaming: raw PCM tap on VADAudioCapture ─────────────────────────

@pytest.mark.asyncio
async def test_vad_capture_raw_listener_forwards_pcm(monkeypatch):
    import numpy as np
    import src.vad_capture as vc

    cap = vc.VADAudioCapture(device=0, sample_rate=48000)
    cap._loop = asyncio.get_running_loop()

    received = []
    cap.set_raw_listener(received.append)

    # Simulate a sounddevice callback with a half-scale tone.
    frames = np.full((16, 1), 0.5, dtype=np.float32)
    cap._audio_callback(frames, 16, None, None)
    await asyncio.sleep(0)  # let call_soon_threadsafe run

    assert len(received) == 1
    pcm = np.frombuffer(received[0], dtype="<i2")
    assert len(pcm) == 16
    assert abs(int(pcm[0]) - int(0.5 * 32767)) <= 1  # float32 -> int16 LE
    # With a raw listener set, the VAD chunker must NOT also emit chunks.
    assert cap._chunk_queue.qsize() == 0
