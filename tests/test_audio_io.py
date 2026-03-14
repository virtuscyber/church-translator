from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from src.audio_capture import AudioCapture
from src.audio_playback import AudioPlayback


def test_pcm_to_wav_clips_float_samples():
    capture = AudioCapture.__new__(AudioCapture)
    capture.channels = 1
    capture.sample_rate = 48000

    wav_bytes = capture._pcm_to_wav(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))

    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    assert samples.tolist() == [-32767, -32767, 0, 32767, 32767]


@pytest.mark.asyncio
async def test_audio_playback_logs_failures(monkeypatch):
    """Playback errors are logged (not raised) so the pipeline continues."""
    class FakeOutputStream:
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def write(self, *args, **kwargs):
            raise RuntimeError('device failure')

    class FakeSoundDevice:
        def check_output_settings(self, **kw):
            pass
        def query_devices(self, *a, **kw):
            return {"default_samplerate": 24000}
        def OutputStream(self, **kw):
            return FakeOutputStream()

    monkeypatch.setattr('src.audio_playback._load_sounddevice', lambda: FakeSoundDevice())

    playback = AudioPlayback()
    # Should not raise — errors are caught and logged
    await playback.play(np.array([0, 100, -100], dtype=np.int16).tobytes())
    await playback.close()


@pytest.mark.asyncio
async def test_audio_playback_resamples(monkeypatch):
    """When device doesn't support source rate, playback resamples."""
    written_data = []

    class FakeOutputStream:
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def write(self, data):
            written_data.append(data)

    class FakeSoundDevice:
        def check_output_settings(self, **kw):
            if kw.get("samplerate") == 24000.0:
                raise RuntimeError("Unsupported sample rate")
        def query_devices(self, *a, **kw):
            return {"default_samplerate": 48000}
        def OutputStream(self, **kw):
            assert kw["samplerate"] == 48000.0
            return FakeOutputStream()

    monkeypatch.setattr('src.audio_playback._load_sounddevice', lambda: FakeSoundDevice())

    playback = AudioPlayback(sample_rate=24000)
    # 4 samples at 24kHz → should become ~8 samples at 48kHz (2x ratio)
    pcm = np.array([1000, 2000, 3000, 4000], dtype=np.int16).tobytes()
    await playback.play(pcm)
    await playback.close()

    assert len(written_data) == 1
    # Resampled from 4 → 8 samples (2x ratio)
    assert written_data[0].shape[0] == 8
