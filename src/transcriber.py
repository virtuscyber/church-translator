"""Speech-to-text using OpenAI's transcription API."""

from __future__ import annotations

import io
import logging
from typing import Optional

from openai import AsyncOpenAI

from .hallucination import is_probably_silence, sanitize_transcript

logger = logging.getLogger(__name__)


# A short phrase in the SOURCE language steers the ASR toward that language and
# domain. Ukrainian is routinely mis-detected as Polish or Russian on short or
# noisy chunks because the `language` flag is only a weak hint the model can
# override. Anchoring with a Ukrainian church phrase biases it far harder
# toward Ukrainian (and primes the religious vocabulary). Keyed by ISO-639-1.
_STT_LANGUAGE_PROMPTS = {
    "uk": "Проповідь українською мовою у християнській церкві.",
    "ru": "Проповедь на русском языке в христианской церкви.",
    "pl": "Kazanie po polsku w kościele chrześcijańskim.",
}


def stt_anchor_prompt(language_code: str) -> Optional[str]:
    """Return a source-language biasing prompt for the STT call, or None."""
    return _STT_LANGUAGE_PROMPTS.get((language_code or "").strip().lower())


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
        timeout: float = 30.0,
        max_retries: int = 2,
        prompt: Optional[str] = None,
        provider: str = "openai",
        elevenlabs_api_key: str = "",
        elevenlabs_model: str = "scribe_v2",
        deepgram_api_key: str = "",
        deepgram_model: str = "nova-3",
    ):
        # Client-level timeout + retries give automatic exponential backoff on
        # transient failures (429/5xx/network) so a blip doesn't drop a chunk.
        # The OpenAI client is always created so it can serve as a fallback even
        # when ElevenLabs is the primary provider.
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.language = (language or "uk").strip().lower()
        # Source-language anchor prompt — explicit override wins, otherwise
        # derive one from the language so Ukrainian isn't read as Polish/Russian.
        self.prompt = prompt if prompt is not None else stt_anchor_prompt(self.language)
        self.temperature = temperature
        self.gate_silence = gate_silence
        self.silence_peak = silence_peak
        self.min_duration_sec = min_duration_sec
        self.filter_hallucinations = filter_hallucinations
        self.timeout = timeout
        # STT provider: "openai", "elevenlabs" (Scribe v2 — best Ukrainian WER),
        # or "deepgram" (Nova-3). Non-OpenAI providers fall back to OpenAI.
        self.provider = provider
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_model = elevenlabs_model
        self.deepgram_api_key = deepgram_api_key
        self.deepgram_model = deepgram_model
        # Set to the error string when a transcription raises, else None. Lets
        # callers tell "API failed" apart from "no speech" (both return None).
        self.last_error: Optional[str] = None

    async def transcribe(self, wav_bytes: bytes) -> Optional[str]:
        """Transcribe WAV audio bytes to source-language text.

        Args:
            wav_bytes: WAV-formatted audio bytes.

        Returns:
            Transcribed text, or None if the chunk is non-speech, empty, or a
            detected hallucination.
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

        self.last_error = None
        text = await self._run_stt(wav_bytes)
        if not text:
            return None

        if self.filter_hallucinations:
            text = sanitize_transcript(text, source="STT")
            if not text:
                return None

        logger.info("Transcribed: %s", text[:80] + ("..." if len(text) > 80 else ""))
        return text

    async def _run_stt(self, wav_bytes: bytes) -> Optional[str]:
        """Run STT through the configured provider, falling back to OpenAI.

        Returns the raw transcript text (may be empty), or None on hard failure
        (in which case ``last_error`` is set).
        """
        # Non-OpenAI providers always fall back to OpenAI on failure.
        order = {
            "openai": ["openai"],
            "elevenlabs": ["elevenlabs", "openai"],
            "deepgram": ["deepgram", "openai"],
        }
        providers = order.get(self.provider, ["openai"])
        last_exc: Optional[Exception] = None
        for i, prov in enumerate(providers):
            try:
                if prov == "elevenlabs":
                    return await self._transcribe_elevenlabs(wav_bytes)
                if prov == "deepgram":
                    return await self._transcribe_deepgram(wav_bytes)
                return await self._transcribe_openai(wav_bytes)
            except Exception as e:
                last_exc = e
                logger.warning("%s STT failed: %s", prov, e)
                if i + 1 < len(providers):
                    logger.info("Falling back to OpenAI STT...")
        self.last_error = str(last_exc) if last_exc else "transcription failed"
        return None

    async def _transcribe_openai(self, wav_bytes: bytes) -> str:
        """Transcribe one chunk via OpenAI. Raises on API error."""
        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "chunk.wav"

        # The source-language prompt strongly anchors the detected language;
        # only send it when we have one so other languages are unaffected.
        extra = {"prompt": self.prompt} if self.prompt else {}
        response = await self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            language=self.language,
            response_format="text",
            temperature=self.temperature,
            **extra,
        )
        return response.strip() if isinstance(response, str) else response.text.strip()

    async def _transcribe_elevenlabs(self, wav_bytes: bytes) -> str:
        """Transcribe one chunk via ElevenLabs Scribe v2 (REST). Raises on error.

        Scribe v2 has the best Ukrainian word-error-rate of the available
        models; ``language_code`` enforces the source language so it isn't
        mis-detected as Polish/Russian.
        """
        import aiohttp

        url = "https://api.elevenlabs.io/v1/speech-to-text"
        form = aiohttp.FormData()
        form.add_field("model_id", self.elevenlabs_model)
        if self.language:
            form.add_field("language_code", self.language)
        form.add_field("file", wav_bytes, filename="chunk.wav", content_type="audio/wav")

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, data=form, headers={"xi-api-key": self.elevenlabs_api_key}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"ElevenLabs STT HTTP {resp.status}: {body[:200]}")
                data = await resp.json()
        return (data.get("text") or "").strip()

    async def _transcribe_deepgram(self, wav_bytes: bytes) -> str:
        """Transcribe one chunk via Deepgram (pre-recorded REST). Raises on error.

        Deepgram Nova-3 supports Ukrainian; ``language`` enforces the source so
        it isn't mis-detected. Audio is sent as raw WAV bytes.
        """
        import aiohttp

        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": self.deepgram_model,
            "smart_format": "true",
            "punctuate": "true",
        }
        if self.language:
            params["language"] = self.language

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        headers = {
            "Authorization": f"Token {self.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, params=params, data=wav_bytes, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Deepgram STT HTTP {resp.status}: {body[:200]}")
                data = await resp.json()
        try:
            return (
                data["results"]["channels"][0]["alternatives"][0]["transcript"] or ""
            ).strip()
        except (KeyError, IndexError, TypeError):
            return ""

