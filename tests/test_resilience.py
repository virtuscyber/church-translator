from __future__ import annotations

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
