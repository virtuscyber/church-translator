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
    ):
        # Client-level timeout + retries give automatic exponential backoff on
        # transient failures (429/5xx/network) so a blip doesn't drop a chunk.
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
        # Set to the error string when a transcription raises, else None. Lets
        # callers tell "API failed" apart from "no speech" (both return None).
        self.last_error: Optional[str] = None

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

        self.last_error = None
        try:
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
            self.last_error = str(e)
            logger.error("Transcription failed: %s", e)
            return None
