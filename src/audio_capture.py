"""Audio capture from system/Dante audio device."""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_sounddevice():
    """Import sounddevice lazily so module import works without PortAudio."""
    import sounddevice as sd

    return sd


class AudioCapture:
    """Captures audio from an input device and yields PCM chunks."""

    def __init__(
        self,
        device: Optional[str | int] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        chunk_duration_sec: float = 8.0,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_sec = chunk_duration_sec
        self._stream = None
        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._running = False

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_sec)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("Audio capture status: %s", status)
        with self._buffer_lock:
            self._buffer.append(indata.copy())

    async def start(self):
        """Start capturing audio."""
        logger.info(
            "Starting audio capture: device=%s, rate=%d, channels=%d",
            self.device,
            self.sample_rate,
            self.channels,
        )
        sd = _load_sounddevice()
        self._running = True
        with self._buffer_lock:
            self._buffer = []
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()

    async def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped.")

    async def get_chunk(self) -> Optional[bytes]:
        """Wait for a full chunk of audio and return as WAV bytes."""
        while self._running:
            with self._buffer_lock:
                total_samples = sum(b.shape[0] for b in self._buffer)
                if total_samples >= self.chunk_samples:
                    all_audio = np.concatenate(self._buffer, axis=0)
                    chunk = all_audio[: self.chunk_samples]
                    remainder = all_audio[self.chunk_samples :]
                    self._buffer = [remainder] if len(remainder) > 0 else []
                    return self._pcm_to_wav(chunk)
            await asyncio.sleep(0.1)
        return None

    def _pcm_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert float32 PCM numpy array to WAV bytes."""
        # Convert float32 [-1, 1] to int16
        pcm_int16 = (audio * 32767).astype(np.int16)
        
        buf = io.BytesIO()
        # WAV header
        num_samples = pcm_int16.shape[0]
        data_size = num_samples * self.channels * 2  # 2 bytes per int16 sample
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))  # chunk size
        buf.write(struct.pack("<H", 1))   # PCM format
        buf.write(struct.pack("<H", self.channels))
        buf.write(struct.pack("<I", self.sample_rate))
        buf.write(struct.pack("<I", self.sample_rate * self.channels * 2))  # byte rate
        buf.write(struct.pack("<H", self.channels * 2))  # block align
        buf.write(struct.pack("<H", 16))  # bits per sample
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm_int16.tobytes())
        
        return buf.getvalue()

    def get_rms(self) -> float:
        """Get current RMS level of buffer (for monitoring)."""
        with self._buffer_lock:
            if not self._buffer:
                return 0.0
            recent = self._buffer[-1] if self._buffer else np.zeros(1)
        return float(np.sqrt(np.mean(recent ** 2)))
