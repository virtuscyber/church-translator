"""Speech-to-text using OpenAI or ElevenLabs Scribe v2.

Supports both providers:
- OpenAI: gpt-4o-transcribe (default)
- ElevenLabs: Scribe v2 (scribe_v2) — 90+ languages including Ukrainian
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribes Ukrainian audio to Ukrainian text."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        elevenlabs_api_key: str = "",
        model: str = "gpt-4o-transcribe",
        language: str = "uk",
    ):
        self.provider = provider
        self.openai_api_key = api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.model = model
        self.language = language

        if provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        # ElevenLabs uses aiohttp per-request (no persistent client needed)

    async def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe WAV audio bytes to Ukrainian text.

        Args:
            wav_bytes: WAV-formatted audio bytes.

        Returns:
            Transcribed Ukrainian text, or None if empty/failed.
        """
        try:
            if self.provider == "elevenlabs":
                return await self._transcribe_elevenlabs(wav_bytes)
            else:
                return await self._transcribe_openai(wav_bytes)
        except Exception as e:
            logger.error("Transcription failed (%s): %s", self.provider, e)
            # Fallback: if ElevenLabs fails, try OpenAI
            if self.provider == "elevenlabs" and self.openai_api_key:
                logger.info("Falling back to OpenAI transcription...")
                try:
                    return await self._transcribe_openai(wav_bytes)
                except Exception as e2:
                    logger.error("OpenAI fallback also failed: %s", e2)
            return None

    async def _transcribe_openai(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe using OpenAI's transcription API."""
        if not hasattr(self, 'client'):
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.openai_api_key)

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
            logger.debug("Empty transcription result (OpenAI).")
            return None

        logger.info("Transcribed (OpenAI): %s", text[:80] + ("..." if len(text) > 80 else ""))
        return text

    async def _transcribe_elevenlabs(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe using ElevenLabs Scribe v2 API.

        Uses the REST endpoint: POST https://api.elevenlabs.io/v1/speech-to-text
        with multipart form data.
        """
        import aiohttp

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": self.elevenlabs_api_key,
        }

        # Build multipart form data
        form = aiohttp.FormData()
        form.add_field("model_id", "scribe_v2")
        form.add_field("language_code", self.language)
        form.add_field("tag_audio_events", "false")
        form.add_field(
            "file",
            wav_bytes,
            filename="chunk.wav",
            content_type="audio/wav",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"ElevenLabs STT returned {resp.status}: {error_text}"
                    )

                result = await resp.json()

        # Extract text from response
        text = result.get("text", "").strip()

        if not text:
            logger.debug("Empty transcription result (ElevenLabs).")
            return None

        logger.info(
            "Transcribed (ElevenLabs): %s",
            text[:80] + ("..." if len(text) > 80 else ""),
        )
        return text
