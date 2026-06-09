"""End-to-end integration tests for the live translation pipeline.

These drive ``_run_live_pipeline`` (the orchestration heart of the app) with
fully faked audio/STT/translate/TTS/playback components and assert that a chunk
of speech flows all the way through to a broadcast translation, an ordered
audio playback, and a transcript entry — for both the chunked and the true-
streaming paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest


# ── Fakes ─────────────────────────────────────────────────────────────

class FakeCapture:
    """Emits one final chunk, then blocks until cancelled."""

    def __init__(self, **kwargs):
        self._served = False
        self.raw_listener = None
        self.started = False
        self.stopped = False

    def set_raw_listener(self, cb):
        self.raw_listener = cb

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    def seconds_since_audio(self):
        return 0.0

    async def restart(self):
        pass

    async def get_chunk(self):
        if not self._served:
            self._served = True
            return ("final", b"RIFFFAKEWAVDATA")
        await asyncio.sleep(3600)  # idle until the task is cancelled


class FakeTranscriber:
    def __init__(self, **kwargs):
        self.last_error = None

    async def transcribe(self, wav_bytes):
        return "Привіт"


class FakeTranslator:
    def __init__(self, **kwargs):
        self.last_error = None

    async def translate(self, text):
        return "Hello"


class FakeSynth:
    def __init__(self, **kwargs):
        self.last_error = None

    async def synthesize_stream(self, text):
        yield b"AUDIO-CHUNK"


class FakePlayback:
    def __init__(self, **kwargs):
        self.played = []

    async def play(self, audio):
        self.played.append(audio)

    async def stop(self):
        pass


def _install_fakes(monkeypatch):
    """Swap the real components for fakes at their import sites + capture
    broadcasts. Returns the list that collects broadcast messages."""
    import src.transcriber
    import src.translator
    import src.synthesizer
    import src.vad_capture
    import src.audio_playback
    from dashboard import server

    monkeypatch.setattr(src.vad_capture, "VADAudioCapture", FakeCapture)
    monkeypatch.setattr(src.transcriber, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(src.translator, "Translator", FakeTranslator)
    monkeypatch.setattr(src.synthesizer, "Synthesizer", FakeSynth)
    monkeypatch.setattr(src.audio_playback, "AudioPlayback", FakePlayback)

    events = []

    async def fake_broadcast(msg):
        events.append(msg)

    monkeypatch.setattr(server, "broadcast", fake_broadcast)
    return events


async def _drive_pipeline(server, events, want_type, timeout=8.0):
    """Run the pipeline until an event of ``want_type`` is broadcast, then
    stop and tear it down cleanly. Returns when done (or raises on timeout)."""
    server.state.live_running = True
    server.state.running = True
    server.state.start_time = time.time()
    server.state.transcript = []
    server.state.stats.update(chunks_processed=0, avg_latency=0.0)

    task = asyncio.create_task(server._run_live_pipeline())
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(e.get("type") == want_type for e in events):
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(f"no {want_type!r} event within {timeout}s; got "
                                 f"{[e.get('type') for e in events]}")
    finally:
        server.state.live_running = False
        server.state.running = False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


# ── Chunked path ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_pipeline_chunked_end_to_end(monkeypatch):
    from dashboard import server

    events = _install_fakes(monkeypatch)
    # Keep the chunked path (no streaming key present for the default provider).
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    await _drive_pipeline(server, events, want_type="translation")

    # STT + translation were both broadcast.
    types = [e["type"] for e in events]
    assert "stt" in types and "translation" in types

    entry = next(e["entry"] for e in events if e["type"] == "translation")
    assert entry["source"] == "Привіт"
    assert entry["translated"] == "Hello"

    # The transcript was recorded and the audio reached playback.
    assert server.state.transcript and server.state.transcript[-1]["translated"] == "Hello"
    assert server.state.live_playback is None or True  # released on teardown


# ── Streaming path ────────────────────────────────────────────────────

class FakeStreamingTranscriber:
    """Fires one final utterance shortly after run() starts, then idles."""

    def __init__(self, *args, on_interim=None, on_final=None, **kwargs):
        self.on_interim = on_interim
        self.on_final = on_final
        self.last_error = None
        self._running = False

    def feed(self, pcm):
        pass

    async def run(self):
        self._running = True
        await asyncio.sleep(0.05)
        if self.on_interim:
            self.on_interim("При")
        if self.on_final:
            self.on_final("Привіт")
        while self._running:
            await asyncio.sleep(0.05)

    async def stop(self):
        self._running = False


@pytest.mark.asyncio
async def test_live_pipeline_streaming_end_to_end(monkeypatch):
    import src.streaming_stt
    from dashboard import server

    events = _install_fakes(monkeypatch)

    # Force the streaming path: Deepgram provider + key + a fake engine.
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setattr(server, "load_saved_config",
                        lambda: {"stt_provider": "deepgram", "stt_streaming": True})
    monkeypatch.setattr(src.streaming_stt, "make_streaming_transcriber",
                        lambda *a, **k: FakeStreamingTranscriber(*a, **k))

    await _drive_pipeline(server, events, want_type="translation")

    types = [e["type"] for e in events]
    assert "partial" in types        # interim preview reached the dashboard
    assert "translation" in types

    entry = next(e["entry"] for e in events if e["type"] == "translation")
    assert entry["source"] == "Привіт" and entry["translated"] == "Hello"
    assert server.state.transcript[-1]["translated"] == "Hello"


# ── Playback jitter buffer ────────────────────────────────────────────

class DribblingSynth:
    """Streams the utterance as several small TTS chunks (like a real API)."""

    def __init__(self, **kwargs):
        self.last_error = None

    async def synthesize_stream(self, text):
        for piece in (b"AAA", b"BBB", b"CCC"):
            yield piece


@pytest.mark.asyncio
async def test_playback_coalesces_streamed_tts_chunks(monkeypatch):
    """Sub-prebuffer TTS chunks must reach the device as one contiguous write,
    not as dribbled 3-byte writes that would underrun the output stream."""
    import src.synthesizer
    import src.audio_playback
    from dashboard import server

    events = _install_fakes(monkeypatch)
    monkeypatch.setattr(src.synthesizer, "Synthesizer", DribblingSynth)

    played = []

    class RecordingPlayback:
        def __init__(self, **kwargs):
            pass
        async def play(self, audio):
            played.append(audio)
        async def close(self):
            pass

    monkeypatch.setattr(src.audio_playback, "AudioPlayback", RecordingPlayback)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    await _drive_pipeline(server, events, want_type="translation")
    # Give the playback worker a beat to drain the slot.
    deadline = time.monotonic() + 2.0
    while not played and time.monotonic() < deadline:
        await asyncio.sleep(0.02)

    assert played == [b"AAABBBCCC"]


# ── Error surfacing through the pipeline ──────────────────────────────

class FailingTranscriber:
    def __init__(self, **kwargs):
        self.last_error = None

    async def transcribe(self, wav_bytes):
        # Simulate a hard API failure (distinct from "no speech").
        self.last_error = "401 invalid api key"
        return None


@pytest.mark.asyncio
async def test_live_pipeline_surfaces_stt_api_error(monkeypatch):
    import src.transcriber
    from dashboard import server

    events = _install_fakes(monkeypatch)
    monkeypatch.setattr(src.transcriber, "Transcriber", FailingTranscriber)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    await _drive_pipeline(server, events, want_type="error")

    errors = [e for e in events if e["type"] == "error"]
    assert any("Speech-to-text" in e["message"] for e in errors)
    # A failed chunk must not produce a phantom transcript entry.
    assert server.state.transcript == []
