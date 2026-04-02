"""Text-to-speech synthesis using ElevenLabs or OpenAI.

Supports both batch mode (returns all audio at once) and streaming mode
(yields audio chunks as they arrive from the API for lower latency).
"""

from __future__ import annotations

import io
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Minimum bytes to yield in streaming mode — small enough for low latency,
# large enough to avoid excessive overhead. ~50ms of PCM 24kHz mono 16-bit.
_STREAM_MIN_CHUNK_BYTES = 2400  # 50ms at 24kHz × 2 bytes


class Synthesizer:
    """Converts translated text to speech audio."""

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
        speed: float = 1.0,
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
        # Speed
        self.speed = speed

    # ── Batch mode (original interface, unchanged) ────────────────────

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to audio bytes (all at once).
        
        Args:
            text: Text to speak.
            
        Returns:
            Complete audio bytes (PCM), or None on failure.
        """
        try:
            if self.provider == "elevenlabs":
                return await self._synthesize_elevenlabs(text)
            else:
                return await self._synthesize_openai(text)
        except Exception as e:
            logger.error("Synthesis failed with %s: %s", self.provider, e)
            if self.provider == "elevenlabs":
                logger.info("Falling back to OpenAI TTS...")
                try:
                    return await self._synthesize_openai(text)
                except Exception as e2:
                    logger.error("Fallback also failed: %s", e2)
            return None

    # ── Streaming mode (new — yields chunks as they arrive) ──────────

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks as they arrive from the TTS API.
        
        Yields PCM audio bytes in chunks (~50-200ms each) as they're
        received from the provider. First chunk arrives much sooner
        than waiting for the full synthesis to complete.
        
        Args:
            text: Text to speak.
            
        Yields:
            PCM audio byte chunks.
        """
        try:
            if self.provider == "elevenlabs":
                async for chunk in self._stream_elevenlabs(text):
                    yield chunk
            else:
                async for chunk in self._stream_openai(text):
                    yield chunk
        except Exception as e:
            logger.error("Streaming synthesis failed with %s: %s", self.provider, e)
            # Fallback to batch mode — still yields one chunk
            if self.provider == "elevenlabs":
                logger.info("Stream failed, falling back to OpenAI batch TTS...")
                try:
                    audio = await self._synthesize_openai(text)
                    if audio:
                        yield audio
                except Exception as e2:
                    logger.error("Fallback also failed: %s", e2)

    async def _stream_elevenlabs(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM chunks from ElevenLabs API."""
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=self.elevenlabs_api_key)

        convert_kwargs = dict(
            voice_id=self.el_voice_id,
            text=text,
            model_id=self.el_model,
            voice_settings={
                "stability": self.el_stability,
                "similarity_boost": self.el_similarity,
            },
            output_format="pcm_24000",
        )
        if self.speed != 1.0:
            convert_kwargs["speed"] = self.speed

        audio_iter = client.text_to_speech.convert(**convert_kwargs)

        buffer = bytearray()
        first_chunk = True

        if hasattr(audio_iter, '__aiter__'):
            async for chunk in audio_iter:
                buffer.extend(chunk)
                if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                    if first_chunk:
                        logger.info("TTS stream: first chunk ready (%d bytes)", len(buffer))
                        first_chunk = False
                    yield bytes(buffer)
                    buffer.clear()
        elif hasattr(audio_iter, '__iter__'):
            for chunk in audio_iter:
                buffer.extend(chunk)
                if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                    if first_chunk:
                        logger.info("TTS stream: first chunk ready (%d bytes)", len(buffer))
                        first_chunk = False
                    yield bytes(buffer)
                    buffer.clear()
        else:
            # Coroutine — await it first
            result = await audio_iter
            if hasattr(result, '__aiter__'):
                async for chunk in result:
                    buffer.extend(chunk)
                    if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                        if first_chunk:
                            logger.info("TTS stream: first chunk ready (%d bytes)", len(buffer))
                            first_chunk = False
                        yield bytes(buffer)
                        buffer.clear()
            elif hasattr(result, '__iter__'):
                for chunk in result:
                    buffer.extend(chunk)
                    if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                        if first_chunk:
                            logger.info("TTS stream: first chunk ready (%d bytes)", len(buffer))
                            first_chunk = False
                        yield bytes(buffer)
                        buffer.clear()
            else:
                buffer.extend(result)

        # Flush remaining
        if buffer:
            yield bytes(buffer)

    async def _stream_openai(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM chunks from OpenAI TTS API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        # Use streaming response
        response = await client.audio.speech.create(
            model=self.oai_model,
            voice=self.oai_voice,
            input=text,
            response_format="pcm",
            speed=self.speed,
        )

        # OpenAI returns the full response — check if we can stream it
        if hasattr(response, 'iter_bytes'):
            buffer = bytearray()
            first_chunk = True
            for chunk in response.iter_bytes(chunk_size=_STREAM_MIN_CHUNK_BYTES):
                buffer.extend(chunk)
                if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                    if first_chunk:
                        logger.info("TTS stream (OpenAI): first chunk ready (%d bytes)", len(buffer))
                        first_chunk = False
                    yield bytes(buffer)
                    buffer.clear()
            if buffer:
                yield bytes(buffer)
        elif hasattr(response, 'aiter_bytes'):
            buffer = bytearray()
            first_chunk = True
            async for chunk in response.aiter_bytes(chunk_size=_STREAM_MIN_CHUNK_BYTES):
                buffer.extend(chunk)
                if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                    if first_chunk:
                        logger.info("TTS stream (OpenAI): first chunk ready (%d bytes)", len(buffer))
                        first_chunk = False
                    yield bytes(buffer)
                    buffer.clear()
            if buffer:
                yield bytes(buffer)
        else:
            # Fallback — return full content as one chunk
            yield response.content

    # ── Batch implementations (unchanged) ─────────────────────────────

    async def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Synthesize using ElevenLabs API (batch — collects all chunks)."""
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=self.elevenlabs_api_key)

        convert_kwargs = dict(
            voice_id=self.el_voice_id,
            text=text,
            model_id=self.el_model,
            voice_settings={
                "stability": self.el_stability,
                "similarity_boost": self.el_similarity,
            },
            output_format="pcm_24000",
        )
        if self.speed != 1.0:
            convert_kwargs["speed"] = self.speed

        audio_iter = client.text_to_speech.convert(**convert_kwargs)

        chunks = []
        if hasattr(audio_iter, '__aiter__'):
            async for chunk in audio_iter:
                chunks.append(chunk)
        elif hasattr(audio_iter, '__iter__'):
            for chunk in audio_iter:
                chunks.append(chunk)
        else:
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
        """Synthesize using OpenAI TTS API (batch)."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key)

        response = await client.audio.speech.create(
            model=self.oai_model,
            voice=self.oai_voice,
            input=text,
            response_format="pcm",
            speed=self.speed,
        )

        return response.content
