"""Tests for the shared anti-aliased resampler.

The point of this module (vs. the bare linear interpolation it replaced) is
that downsampling must not alias: a tone above the target Nyquist has to be
strongly attenuated, while in-band speech content passes through unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio_resample import (
    StreamingResampler,
    resample_f32,
    resample_int16_bytes,
)


def _tone(freq: float, rate: int, seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# ── One-shot conversion ───────────────────────────────────────────────

def test_passthrough_when_rates_equal():
    audio = _tone(440, 24000)
    out = resample_f32(audio, 24000, 24000)
    assert out is audio or np.array_equal(out, audio)


def test_output_length_matches_ratio():
    audio = np.zeros(48000, dtype=np.float32)
    assert resample_f32(audio, 48000, 24000).size == 24000
    assert resample_f32(audio, 24000, 48000).size == 96000
    assert resample_f32(audio, 48000, 44100).size == 44100


def test_in_band_tone_survives_downsampling():
    # 1 kHz is well inside the 12 kHz Nyquist of the 24 kHz target.
    audio = _tone(1000, 48000)
    out = resample_f32(audio, 48000, 24000)
    # Compare steady-state RMS (skip filter edges).
    assert _rms(out[100:-100]) == pytest.approx(_rms(audio[100:-100]), rel=0.06)


def test_above_nyquist_tone_is_attenuated():
    # 15 kHz at 48 kHz would alias to 9 kHz after a naive 2:1 decimation.
    audio = _tone(15000, 48000)
    out = resample_f32(audio, 48000, 24000)
    in_rms = _rms(audio)
    out_rms = _rms(out[200:-200])
    # The anti-alias filter must knock it down by at least 20 dB.
    assert out_rms < in_rms * 0.1


def test_int16_bytes_round_trip():
    pcm = (_tone(440, 48000, 0.1) * 16000).astype("<i2").tobytes()
    out = resample_int16_bytes(pcm, 48000, 24000)
    assert len(out) == len(pcm) // 2
    assert resample_int16_bytes(pcm, 48000, 48000) is pcm


# ── Streaming conversion ──────────────────────────────────────────────

def test_streaming_matches_one_shot_in_steady_state():
    """Feeding frame-by-frame must produce a seamless stream (no per-frame
    edge artifacts) — its spectrum-free middle should match one-shot output."""
    audio = _tone(1000, 48000, 1.0)
    rs = StreamingResampler(48000, 24000)
    pieces = [rs.process_f32(chunk) for chunk in np.array_split(audio, 37)]
    streamed = np.concatenate(pieces)

    # Same length as the input warrants (±1 from phase bookkeeping).
    assert abs(streamed.size - 24000) <= 2

    # A pure tone must stay a pure tone: check there are no discontinuities by
    # verifying the maximum sample-to-sample jump matches a clean 1 kHz tone.
    max_step_clean = np.max(np.abs(np.diff(resample_f32(audio, 48000, 24000))))
    max_step_streamed = np.max(np.abs(np.diff(streamed[100:-100])))
    assert max_step_streamed <= max_step_clean * 1.1


def test_streaming_above_nyquist_tone_is_attenuated():
    audio = _tone(15000, 48000, 0.5)
    rs = StreamingResampler(48000, 24000)
    out = np.concatenate([rs.process_f32(c) for c in np.array_split(audio, 20)])
    assert _rms(out[200:]) < _rms(audio) * 0.1


def test_streaming_passthrough_and_tiny_frames():
    rs = StreamingResampler(48000, 48000)
    frame = _tone(440, 48000, 0.01)
    assert np.array_equal(rs.process_f32(frame), frame)

    # Downsampling fed one sample at a time must never crash and must keep
    # the overall output count right.
    rs2 = StreamingResampler(48000, 24000)
    total = sum(rs2.process_f32(np.array([0.1], dtype=np.float32)).size for _ in range(480))
    assert abs(total - 240) <= 2


def test_streaming_int16_bytes_interface():
    rs = StreamingResampler(48000, 24000)
    pcm = (np.ones(4800, dtype=np.float32) * 1000).astype("<i2").tobytes()
    out = rs.process_int16_bytes(pcm)
    assert len(out) % 2 == 0
    assert abs(len(out) // 2 - 2400) <= 2
