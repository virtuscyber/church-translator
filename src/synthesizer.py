"""Text-to-speech synthesis using ElevenLabs or OpenAI."""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Synthesizer:
    """Converts English text to speech audio."""

    def __init__(
        self,
        provider: str = "elevenlabs",
        openai_api_key: str = "",
        elevenlabs_api_key: str = "",
        elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB",
        elevenlabs_model: str = "eleven_turbo_v2_5",
        elevenlabs_stability: float = 0.7,
        elevenlabs_similarity: float = 0.8,
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "onyx",
    ):
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        # ElevenLabs settings
        self.el_voice_id = elevenlabs_voice_id
        self.el_model = elevenlabs_model
        self.el_stability = elevenlabs_stability
        self.el_similarity = elevenlabs_similarity
        # OpenAI settings
        self.oai_model = openai_model
        self.oai_voice = openai_voice

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to audio bytes (WAV/MP3).
        
        Args:
            text: English text to speak.
            
        Returns:
            Audio bytes, or None on failure.
        """
        try:
            if self.provider == "elevenlabs":
                return await self._synthesize_elevenlabs(text)
            else:
                return await self._synthesize_openai(text)
        except Exception as e:
            logger.error("Synthesis failed with %s: %s", self.provider, e)
            # Fallback
            if self.provider == "elevenlabs":
                logger.info("Falling back to OpenAI TTS...")
                try:
                    return await self._synthesize_openai(text)
                except Exception as e2:
                    logger.error("Fallback also failed: %s", e2)
            return None

    async def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Synthesize using ElevenLabs API."""
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=self.elevenlabs_api_key)

        audio_iter = client.text_to_speech.convert(
            voice_id=self.el_voice_id,
            text=text,
            model_id=self.el_model,
            voice_settings={
                "stability": self.el_stability,
                "similarity_boost": self.el_similarity,
            },
            output_format="pcm_24000",
        )

        # Collect all chunks — handle both sync and async iterators
        chunks = []
        if hasattr(audio_iter, '__aiter__'):
            async for chunk in audio_iter:
                chunks.append(chunk)
        elif hasattr(audio_iter, '__iter__'):
            for chunk in audio_iter:
                chunks.append(chunk)
        else:
            # It's a coroutine that needs awaiting first
            result = await audio_iter
            if hasattr(result, '__aiter__'):
                async for chunk in result:
                    chunks.append(chunk)
            elif hasattr(result, '__iter__'):
                for chunk in result:
                    chunks.append(chunk)
            else:
                chunks.append(result)
        
        return b"".join(chunks)

    async def _synthesize_openai(self, text: str) -> bytes:
        """Synthesize using OpenAI TTS API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        response = await client.audio.speech.create(
            model=self.oai_model,
            voice=self.oai_voice,
            input=text,
            response_format="pcm",
            speed=1.0,
        )

        return response.content
