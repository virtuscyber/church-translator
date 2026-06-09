"""Text-to-speech synthesis using ElevenLabs or OpenAI.

Supports both batch mode (returns all audio at once) and streaming mode
(yields audio chunks as they arrive from the API for lower latency).

ElevenLabs is called over its REST streaming endpoint directly (no SDK), which
lets us pass ``previous_text`` — the previously spoken sentence — so prosody
carries across utterances instead of every chunk restarting cold. Both
providers return raw PCM mono 16-bit at 24 kHz.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# Minimum bytes to yield in streaming mode — small enough for low latency,
# large enough to avoid excessive overhead. ~50ms of PCM 24kHz mono 16-bit.
_STREAM_MIN_CHUNK_BYTES = 2400  # 50ms at 24kHz × 2 bytes

# How much trailing text to send as ElevenLabs prosody context. Longer adds
# request weight without improving continuity.
_PREVIOUS_TEXT_MAX_CHARS = 500

_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


class Synthesizer:
    """Converts translated text to speech audio."""

    def __init__(
        self,
        provider: str = "elevenlabs",
        openai_api_key: str = "",
        elevenlabs_api_key: str = "",
        elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB",
        elevenlabs_model: str = "eleven_flash_v2_5",
        elevenlabs_stability: float = 0.7,
        elevenlabs_similarity: float = 0.8,
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "onyx",
        timeout: float = 30.0,
        max_retries: int = 2,
        speed: float = 1.0,
    ):
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        # Playback speed (ElevenLabs ~0.7-1.2, OpenAI 0.25-4.0).
        self.speed = speed
        # ElevenLabs settings
        self.el_voice_id = elevenlabs_voice_id
        self.el_model = elevenlabs_model
        self.el_stability = elevenlabs_stability
        self.el_similarity = elevenlabs_similarity
        # OpenAI settings
        self.oai_model = openai_model
        self.oai_voice = openai_voice
        # Resilience
        self.timeout = timeout
        self.max_retries = max_retries
        # Error string when synthesis ultimately fails, else None.
        self.last_error: Optional[str] = None
        # Last successfully synthesized text — ElevenLabs prosody context.
        self._previous_text: str = ""

    # ── Batch mode ─────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to audio bytes (all at once).

        Args:
            text: Text to speak.

        Returns:
            Complete audio bytes (PCM), or None on failure.
        """
        self.last_error = None
        last_exc: Optional[Exception] = None

        # Retry the configured provider with exponential backoff on transient
        # failures before giving up.
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "elevenlabs":
                    return await asyncio.wait_for(self._synthesize_elevenlabs(text), self.timeout)
                return await asyncio.wait_for(self._synthesize_openai(text), self.timeout)
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Synthesis attempt %d/%d failed (%s): %s",
                    attempt + 1, self.max_retries + 1, self.provider, e,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))

        # All retries exhausted — fall back to OpenAI if we were on ElevenLabs.
        if self.provider == "elevenlabs":
            logger.info("Falling back to OpenAI TTS...")
            try:
                return await asyncio.wait_for(self._synthesize_openai(text), self.timeout)
            except Exception as e2:
                last_exc = e2
                logger.error("Fallback also failed: %s", e2)

        self.last_error = str(last_exc) if last_exc else "synthesis failed"
        return None

    # ── Streaming mode (yields chunks as they arrive) ──────────────────

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
        self.last_error = None
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
                        return
                except Exception as e2:
                    logger.error("Fallback also failed: %s", e2)
                    self.last_error = str(e2)
                    return
            self.last_error = str(e)

    # ── ElevenLabs (REST streaming endpoint) ──────────────────────────

    def _elevenlabs_payload(self, text: str) -> dict:
        payload = {
            "text": text,
            "model_id": self.el_model,
            "voice_settings": {
                "stability": self.el_stability,
                "similarity_boost": self.el_similarity,
                "speed": self.speed,
            },
        }
        # Prosody continuity: tell the model what was just spoken so this
        # utterance's intonation continues the sermon instead of restarting.
        if self._previous_text:
            payload["previous_text"] = self._previous_text[-_PREVIOUS_TEXT_MAX_CHARS:]
        return payload

    async def _stream_elevenlabs(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM chunks from the ElevenLabs REST API. Raises on error."""
        import aiohttp

        url = _ELEVENLABS_TTS_URL.format(voice_id=self.el_voice_id)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        first_chunk = True

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                params={"output_format": "pcm_24000"},
                json=self._elevenlabs_payload(text),
                headers={"xi-api-key": self.elevenlabs_api_key},
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"ElevenLabs TTS HTTP {resp.status}: {body[:200]}")

                buffer = bytearray()
                async for chunk in resp.content.iter_chunked(_STREAM_MIN_CHUNK_BYTES):
                    buffer.extend(chunk)
                    if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                        if first_chunk:
                            logger.info("TTS stream: first chunk ready (%d bytes)", len(buffer))
                            first_chunk = False
                        yield bytes(buffer)
                        buffer.clear()
                if buffer:
                    yield bytes(buffer)

        self._previous_text = text

    async def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Synthesize using ElevenLabs (batch — collects the stream)."""
        chunks = [chunk async for chunk in self._stream_elevenlabs(text)]
        return b"".join(chunks)

    # ── OpenAI ─────────────────────────────────────────────────────────

    async def _stream_openai(self, text: str) -> AsyncIterator[bytes]:
        """Stream PCM chunks from the OpenAI TTS API as they're generated."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key, timeout=self.timeout, max_retries=self.max_retries)

        async with client.audio.speech.with_streaming_response.create(
            model=self.oai_model,
            voice=self.oai_voice,
            input=text,
            response_format="pcm",
            speed=self.speed,
        ) as response:
            buffer = bytearray()
            first_chunk = True
            async for chunk in response.iter_bytes(chunk_size=_STREAM_MIN_CHUNK_BYTES):
                buffer.extend(chunk)
                if len(buffer) >= _STREAM_MIN_CHUNK_BYTES:
                    if first_chunk:
                        logger.info("TTS stream (OpenAI): first chunk ready (%d bytes)", len(buffer))
                        first_chunk = False
                    yield bytes(buffer)
                    buffer.clear()
            if buffer:
                yield bytes(buffer)

    async def _synthesize_openai(self, text: str) -> bytes:
        """Synthesize using OpenAI TTS API (batch)."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.openai_api_key, timeout=self.timeout, max_retries=self.max_retries)

        response = await client.audio.speech.create(
            model=self.oai_model,
            voice=self.oai_voice,
            input=text,
            response_format="pcm",
            speed=self.speed,
        )

        return response.content
