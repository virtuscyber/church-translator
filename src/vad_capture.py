"""VAD-aware audio capture — replaces fixed-duration chunking with speech-boundary detection."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator, Optional

import numpy as np

from .vad_chunker import VADChunker

logger = logging.getLogger(__name__)


def _load_sounddevice():
    """Import sounddevice lazily so module import works without PortAudio."""
    import sounddevice as sd

    return sd


class VADAudioCapture:
    """Captures audio and yields speech-bounded chunks using VAD.
    
    Unlike the original AudioCapture which yields fixed 8s chunks,
    this yields variable-length chunks split on natural speech pauses.
    """

    def __init__(
        self,
        device: Optional[str | int] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        vad_aggressiveness: int = 2,
        min_chunk_sec: float = 3.0,
        max_chunk_sec: float = 15.0,
        silence_threshold_sec: float = 0.8,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        
        self._chunker = VADChunker(
            aggressiveness=vad_aggressiveness,
            min_chunk_sec=min_chunk_sec,
            max_chunk_sec=max_chunk_sec,
            silence_threshold_sec=silence_threshold_sec,
            input_sample_rate=sample_rate,
        )

        self._stream = None
        self._chunk_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunker_lock = threading.Lock()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called from audio thread — feed to VAD chunker."""
        if status:
            logger.warning("Audio status: %s", status)
        
        # VAD chunker expects mono float32
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        with self._chunker_lock:
            chunks = self._chunker.feed(audio)
        for chunk in chunks:
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(self._chunk_queue.put_nowait, chunk)
                except RuntimeError:
                    logger.debug("Event loop closed before queued VAD chunk could be delivered")

    async def start(self):
        """Start VAD-aware audio capture."""
        self._loop = asyncio.get_running_loop()
        self._running = True
        
        logger.info(
            "Starting VAD capture: device=%s, rate=%d",
            self.device, self.sample_rate,
        )

        sd = _load_sounddevice()
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
        """Stop capture and flush remaining audio."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None

        # Flush remaining audio
        with self._chunker_lock:
            final = self._chunker.flush()
        if final:
            self._chunk_queue.put_nowait(final)
        
        logger.info("VAD capture stopped.")

    async def get_chunk(self) -> Optional[bytes]:
        """Wait for the next speech-bounded chunk.
        
        Returns WAV bytes when a speech segment is detected,
        or None if capture has stopped.
        """
        while self._running or not self._chunk_queue.empty():
            try:
                chunk = await asyncio.wait_for(self._chunk_queue.get(), timeout=0.5)
                return chunk
            except asyncio.TimeoutError:
                continue
        return None
