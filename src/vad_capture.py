"""VAD-aware audio capture — replaces fixed-duration chunking with speech-boundary detection."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
        enable_preview: bool = False,
        preview_after_sec: float = 2.0,
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
            enable_preview=enable_preview,
            preview_after_sec=preview_after_sec,
        )

        self._stream = None
        # Queue holds (tag, wav_bytes) tuples: tag is "preview" or "final"
        self._chunk_queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunker_lock = threading.Lock()
        # Monotonic timestamp of the last audio callback. A healthy input
        # stream fires continuously (even in silence), so a stale value means
        # the device has stalled or been unplugged.
        self._last_frame_monotonic: float = 0.0
        # Optional raw-PCM tap for true-streaming STT. When set, the callback
        # forwards int16-LE mono frames to it (on the event loop) and skips the
        # VAD chunker entirely — the streaming provider does its own endpointing.
        self._raw_listener: Optional[callable] = None

    def set_raw_listener(self, callback) -> None:
        """Forward raw int16-LE mono PCM frames to ``callback`` instead of
        emitting VAD-bounded chunks. Pass ``None`` to restore chunking."""
        self._raw_listener = callback

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called from audio thread — feed to VAD chunker."""
        if status:
            logger.warning("Audio status: %s", status)

        self._last_frame_monotonic = time.monotonic()

        # VAD chunker expects mono float32
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Streaming mode: forward raw int16 PCM to the listener (on the loop)
        # and bypass the VAD chunker.
        if self._raw_listener is not None:
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(self._raw_listener, pcm16)
                except RuntimeError:
                    logger.debug("Event loop closed before raw PCM frame could be delivered")
            return

        with self._chunker_lock:
            tagged_chunks = self._chunker.feed(audio)
        for tagged_chunk in tagged_chunks:
            if self._loop:
                try:
                    self._loop.call_soon_threadsafe(self._chunk_queue.put_nowait, tagged_chunk)
                except RuntimeError:
                    logger.debug("Event loop closed before queued VAD chunk could be delivered")

    def _open_stream(self):
        """Open and start the PortAudio input stream (used by start/restart)."""
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
        self._last_frame_monotonic = time.monotonic()

    async def start(self):
        """Start VAD-aware audio capture."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        logger.info(
            "Starting VAD capture: device=%s, rate=%d",
            self.device, self.sample_rate,
        )

        self._open_stream()

    def update_chunking(self, **kwargs) -> None:
        """Live-update VAD/chunking parameters (aggressiveness, min/max chunk
        seconds, silence threshold). Held under the chunker lock so it can't
        race the audio thread mid-``feed()``."""
        with self._chunker_lock:
            self._chunker.update_settings(**kwargs)

    def seconds_since_audio(self) -> float:
        """Seconds since the last audio callback (0 if not started yet)."""
        if not self._running or self._last_frame_monotonic == 0.0:
            return 0.0
        return time.monotonic() - self._last_frame_monotonic

    async def restart(self):
        """Re-open the input stream after a stall/disconnect.

        Raises if the device can't be re-opened (e.g. truly unplugged) so the
        caller can surface the failure to the operator.
        """
        logger.warning("Restarting audio capture stream (device=%s)", self.device)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing stalled stream: %s", e)
            self._stream = None
        self._open_stream()
        logger.info("Audio capture stream restarted")

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
            self._chunk_queue.put_nowait(("final", final))
        
        logger.info("VAD capture stopped.")

    async def get_chunk(self) -> Optional[tuple[str, bytes]]:
        """Wait for the next speech-bounded chunk.
        
        Returns:
            (tag, wav_bytes) where tag is "preview" or "final",
            or None if capture has stopped.
        """
        while self._running or not self._chunk_queue.empty():
            try:
                tagged = await asyncio.wait_for(self._chunk_queue.get(), timeout=0.5)
                return tagged
            except asyncio.TimeoutError:
                continue
        return None
