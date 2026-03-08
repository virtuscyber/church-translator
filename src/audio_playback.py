"""Audio playback to system/Dante output device."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_sounddevice():
    """Import sounddevice lazily so module import works without PortAudio."""
    import sounddevice as sd

    return sd


class AudioPlayback:
    """Plays audio bytes to an output device."""

    def __init__(
        self,
        device: Optional[str | int] = None,
        sample_rate: int = 24000,  # ElevenLabs PCM default
        channels: int = 1,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels

    async def play(self, pcm_bytes: bytes, sample_rate: Optional[int] = None):
        """Play raw PCM int16 audio bytes.
        
        Args:
            pcm_bytes: Raw PCM audio (int16, mono).
            sample_rate: Override sample rate if different from default.
        """
        rate = sample_rate or self.sample_rate
        
        try:
            # Convert bytes to numpy float32
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            if len(audio) == 0:
                logger.warning("Empty audio data, skipping playback.")
                return

            logger.info("Playing %d samples (%.1fs) at %dHz", len(audio), len(audio) / rate, rate)

            # Play synchronously in executor to not block event loop
            loop = asyncio.get_event_loop()
            sd = _load_sounddevice()
            await loop.run_in_executor(
                None,
                lambda: sd.play(audio, samplerate=rate, device=self.device, blocking=True),
            )

        except Exception as e:
            logger.error("Playback failed: %s", e)
