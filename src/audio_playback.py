"""Audio playback to system/Dante output device.

Uses a persistent OutputStream for gapless playback during live translation.
Each call to play() writes into the stream without reopening the device,
eliminating the inter-chunk gaps and device-reopen overhead that caused
translated audio to be inaudible despite test tones working fine.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_sounddevice():
    """Import sounddevice lazily so module import works without PortAudio."""
    import sounddevice as sd
    return sd


class AudioPlayback:
    """Plays audio bytes to an output device using a persistent stream.

    The stream opens on first play() and stays open until close().
    This avoids the overhead (and audible gaps) of creating a new
    OutputStream for every 50 ms TTS chunk.
    """

    def __init__(
        self,
        device: Optional[str | int] = None,
        sample_rate: int = 24000,  # TTS PCM sample rate (ElevenLabs/OpenAI default)
        channels: int = 1,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels

        self._stream = None
        self._stream_sr: int = 0  # Actual device sample rate (may differ from self.sample_rate)
        self._lock = threading.Lock()
        self._sd = None

    def _ensure_stream(self):
        """Open a persistent OutputStream if one isn't already running.

        If the device doesn't support the TTS sample rate (e.g. 24 kHz),
        we open it at the device's default rate and resample in play().
        """
        if self._stream is not None:
            return

        sd = _load_sounddevice()
        self._sd = sd

        # Determine what sample rate the device actually supports.
        # Many USB speakers only do 44100 or 48000.
        target_sr = self.sample_rate
        try:
            sd.check_output_settings(
                device=self.device,
                samplerate=float(target_sr),
                channels=self.channels,
                dtype="float32",
            )
            device_sr = target_sr
        except Exception:
            # Fall back to the device's default sample rate
            info = sd.query_devices(self.device, "output")
            device_sr = int(info["default_samplerate"])
            logger.info(
                "Output device doesn't support %d Hz — using device default %d Hz (will resample)",
                target_sr, device_sr,
            )

        self._stream_sr = device_sr
        self._stream = sd.OutputStream(
            device=self.device,
            samplerate=float(device_sr),
            channels=self.channels,
            dtype="float32",
            blocksize=0,  # Let PortAudio choose optimal block size
        )
        self._stream.start()
        logger.info(
            "Opened persistent output stream: device=%s, rate=%d Hz",
            self.device, device_sr,
        )

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Simple linear-interpolation resample (good enough for speech)."""
        if from_sr == to_sr:
            return audio
        ratio = to_sr / from_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def play(self, pcm_bytes: bytes, sample_rate: Optional[int] = None):
        """Play raw PCM int16 audio bytes through the persistent stream.

        Args:
            pcm_bytes: Raw PCM audio (int16, mono).
            sample_rate: Override source sample rate if different from default.
        """
        src_rate = sample_rate or self.sample_rate

        if len(pcm_bytes) < 2:
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._play_sync, pcm_bytes, src_rate)
        except Exception as e:
            logger.error("Playback failed: %s", e)

    def _play_sync(self, pcm_bytes: bytes, src_rate: int):
        """Synchronous write into the persistent output stream."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

        if len(audio) == 0:
            return

        with self._lock:
            self._ensure_stream()

            # Resample if device rate differs from source rate
            if self._stream_sr != src_rate:
                audio = self._resample(audio, src_rate, self._stream_sr)

            # Write the full buffer — OutputStream.write() blocks until
            # all samples are accepted, giving us back-pressure flow control.
            self._stream.write(audio.reshape(-1, 1) if self.channels == 1 else audio)

    async def close(self):
        """Close the persistent output stream (call when pipeline stops)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._close_sync)

    def _close_sync(self):
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error closing output stream: %s", e)
                self._stream = None
                self._stream_sr = 0
                logger.info("Output stream closed")
