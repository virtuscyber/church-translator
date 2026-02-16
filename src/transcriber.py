"""Speech-to-text using OpenAI's transcription API."""

from __future__ import annotations

import io
import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes Ukrainian audio to Ukrainian text using OpenAI."""

    def __init__(self, api_key: str, model: str = "gpt-4o-transcribe", language: str = "uk"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe WAV audio bytes to Ukrainian text.
        
        Args:
            wav_bytes: WAV-formatted audio bytes.
            
        Returns:
            Transcribed Ukrainian text, or None if empty/failed.
        """
        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "chunk.wav"

            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=self.language,
                response_format="text",
            )

            text = response.strip() if isinstance(response, str) else response.text.strip()
            
            if not text:
                logger.debug("Empty transcription result.")
                return None

            logger.info("Transcribed: %s", text[:80] + ("..." if len(text) > 80 else ""))
            return text

        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return None
