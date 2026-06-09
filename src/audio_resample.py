"""Shared audio resampling with anti-aliasing.

Every rate conversion in the app used to be bare linear interpolation. That is
acceptable for upsampling speech, but *downsampling* without a low-pass filter
folds everything above the target Nyquist back into the audible band
(aliasing) — which both sounds bad and measurably hurts STT accuracy on the
48 kHz → 24 kHz feed to OpenAI's realtime API.

Two entry points:

- :func:`resample_f32` — one-shot conversion of a complete buffer (playback,
  AES67). Applies a windowed-sinc FIR low-pass before any downsampling.
- :class:`StreamingResampler` — stateful conversion for continuous PCM streams
  (streaming STT). Carries filter history and fractional phase across calls so
  frame boundaries are seamless, with no per-frame edge artifacts.

Pure numpy; no scipy dependency.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# 63 symmetric taps (linear phase): ~0.65 ms group delay at 48 kHz and roughly
# -50 dB stopband with a Hamming window — plenty for speech.
_FIR_TAPS = 63


@lru_cache(maxsize=8)
def _lowpass_kernel(cutoff: float) -> np.ndarray:
    """Windowed-sinc low-pass FIR kernel.

    ``cutoff`` is normalized to the *source* sample rate (0 < cutoff < 0.5).
    """
    n = np.arange(_FIR_TAPS) - (_FIR_TAPS - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    kernel *= np.hamming(_FIR_TAPS)
    kernel /= kernel.sum()  # unity gain at DC
    return kernel.astype(np.float32)


def _anti_alias_cutoff(src_rate: int, dst_rate: int) -> float:
    # Slightly below the target Nyquist so the transition band stays out of it.
    return round(0.45 * dst_rate / src_rate, 4)


def _linear_interp(audio: np.ndarray, n_out: int) -> np.ndarray:
    idx = np.linspace(0, audio.size - 1, n_out)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, audio.size - 1)
    frac = (idx - lo).astype(np.float32)
    return (audio[lo] * (1.0 - frac) + audio[hi] * frac).astype(np.float32)


def resample_f32(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a complete float32 mono buffer from ``src_rate`` to ``dst_rate``.

    Downsampling is low-pass filtered first so no aliasing lands in band.
    Output length is ``round(len * dst_rate / src_rate)``.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    if dst_rate < src_rate:
        kernel = _lowpass_kernel(_anti_alias_cutoff(src_rate, dst_rate))
        audio = np.convolve(audio, kernel, mode="same")
    n_out = max(1, int(round(audio.size * dst_rate / src_rate)))
    return _linear_interp(audio, n_out)


def resample_int16_bytes(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample int16-LE mono PCM bytes (one-shot, anti-aliased)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    out = resample_f32(audio, src_rate, dst_rate)
    return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


class StreamingResampler:
    """Stateful resampler for continuous int16-LE mono PCM streams.

    Feeding a long stream frame-by-frame produces (within float tolerance) the
    same samples as converting it in one shot: the FIR filter keeps the last
    ``taps - 1`` raw input samples as history between calls, and the
    interpolator carries its fractional read position, so there are no
    boundary discontinuities. Total latency is the filter's half-kernel group
    delay (~0.65 ms at 48 kHz).
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._step = src_rate / dst_rate
        self._kernel = (
            _lowpass_kernel(_anti_alias_cutoff(src_rate, dst_rate))
            if dst_rate < src_rate
            else None
        )
        # Raw input tail so the convolution has full context at frame starts.
        self._hist = (
            np.zeros(_FIR_TAPS - 1, dtype=np.float32) if self._kernel is not None else None
        )
        # Last filtered sample, so interpolation can straddle the frame seam.
        self._carry = np.zeros(0, dtype=np.float32)
        # Fractional read position into [carry + filtered frame].
        self._pos = 0.0

    def process_f32(self, frame: np.ndarray) -> np.ndarray:
        """Convert one float32 mono frame; returns the next output samples."""
        frame = np.asarray(frame, dtype=np.float32)
        if self.src_rate == self.dst_rate:
            return frame
        if frame.size == 0:
            return frame

        if self._kernel is not None:
            x = np.concatenate([self._hist, frame])
            self._hist = x[-(_FIR_TAPS - 1):]
            filtered = np.convolve(x, self._kernel, mode="valid")
        else:
            filtered = frame

        stream = np.concatenate([self._carry, filtered])
        last = stream.size - 1  # highest index usable for interpolation
        if self._pos > last:
            # Not enough new samples yet to emit anything.
            self._carry = stream[-1:]
            self._pos -= last
            return np.zeros(0, dtype=np.float32)

        n_out = int(np.floor((last - self._pos) / self._step)) + 1
        idx = self._pos + self._step * np.arange(n_out)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, last)
        frac = (idx - lo).astype(np.float32)
        out = (stream[lo] * (1.0 - frac) + stream[hi] * frac).astype(np.float32)

        # Re-anchor: the next call's stream starts at this stream's last sample.
        self._pos = (idx[-1] + self._step) - last
        self._carry = stream[-1:]
        return out

    def process_int16_bytes(self, pcm: bytes) -> bytes:
        """Convert one int16-LE mono PCM frame (bytes in, bytes out)."""
        if self.src_rate == self.dst_rate or not pcm:
            return pcm
        frame = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        out = self.process_f32(frame)
        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()
