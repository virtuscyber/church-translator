"""Speech-to-text using OpenAI's transcription API."""

from __future__ import annotations

import io
import logging
from typing import Optional

from openai import AsyncOpenAI

from .hallucination import is_probably_silence, sanitize_transcript

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes Ukrainian audio to Ukrainian text using OpenAI.

    Includes two anti-hallucination guards that are critical for live use:
    a near-silence gate that skips non-speech chunks before they reach the
    API (the dominant source of phantom transcripts), and a post-filter that
    drops known model artifacts and repetition loops.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-transcribe",
        language: str = "uk",
        temperature: float = 0.0,
        gate_silence: bool = True,
        silence_peak: float = 0.008,
        min_duration_sec: float = 0.4,
        filter_hallucinations: bool = True,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.language = language
        self.temperature = temperature
        self.gate_silence = gate_silence
        self.silence_peak = silence_peak
        self.min_duration_sec = min_duration_sec
        self.filter_hallucinations = filter_hallucinations

    async def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe WAV audio bytes to Ukrainian text.

        Args:
            wav_bytes: WAV-formatted audio bytes.

        Returns:
            Transcribed Ukrainian text, or None if the chunk is non-speech,
            empty, or a detected hallucination.
        """
        # Gate near-silent / too-short chunks BEFORE hitting the API. This is
        # the single biggest lever against STT hallucination: models invent
        # confident text when given audio with no real speech.
        if self.gate_silence and is_probably_silence(
            wav_bytes,
            min_duration_sec=self.min_duration_sec,
            silence_peak=self.silence_peak,
        ):
            logger.debug("Skipping near-silent chunk (no speech detected).")
            return None

        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "chunk.wav"

            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                response_format="text",
                temperature=self.temperature,
            )

            text = response.strip() if isinstance(response, str) else response.text.strip()

            if not text:
                logger.debug("Empty transcription result.")
                return None

            if self.filter_hallucinations:
                text = sanitize_transcript(text, source="STT")
                if not text:
                    return None

            logger.info("Transcribed: %s", text[:80] + ("..." if len(text) > 80 else ""))
            return text

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return None
