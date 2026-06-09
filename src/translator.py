"""Ukrainian → English translation with biblical styling via GPT-4o."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from openai import AsyncOpenAI

from .hallucination import sanitize_transcript

logger = logging.getLogger(__name__)


class Translator:
    """Translates Ukrainian text to biblically-styled English."""

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        context_sentences: int = 2,
        filter_hallucinations: bool = True,
        source_language: str = "Ukrainian",
        target_language: str = "English",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        # Client-level timeout + retries give automatic backoff on transient
        # failures so a single API blip doesn't silently drop a translation.
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.filter_hallucinations = filter_hallucinations
        self.source_language = source_language
        self.target_language = target_language
        # Error string when a translation raises, else None (see Transcriber).
        self.last_error: Optional[str] = None
        # Sliding window of (source, translation) pairs, replayed as real
        # user/assistant turns so the model keeps terminology, pronouns, and
        # tone consistent across chunks without re-translating old text.
        self._history: deque[tuple[str, str]] = deque(maxlen=context_sentences)

    def reset_context(self) -> None:
        """Forget previous chunks (call when the language pair changes)."""
        self._history.clear()

    def _request_text(self, text: str) -> str:
        return (
            f"Translate the following {self.source_language} speech to "
            f"{self.target_language}:\n\n{text}"
        )

    async def translate(self, ukrainian_text: str) -> Optional[str]:
        """Translate source-language text to the target language with biblical tone.

        Maintains a sliding window of previous (source, translation) pairs for
        context continuity. Junk input is rejected up front, and the model's
        output is filtered for hallucination artifacts before being returned or
        stored as context — so a single bad chunk cannot poison later
        translations.

        Args:
            ukrainian_text: Ukrainian text to translate.

        Returns:
            English translation with biblical vocabulary, or None on failure or
            when the input/output is non-speech.
        """
        # Don't translate junk — empty strings, stray punctuation, or
        # hallucination phrases that slipped through STT.
        clean_input = (
            sanitize_transcript(ukrainian_text, source="STT")
            if self.filter_hallucinations
            else (ukrainian_text or "").strip()
        )
        if not clean_input:
            logger.debug("Translation skipped: empty or non-speech input.")
            return None

        self.last_error = None
        try:
            # Replay recent chunks as genuine conversation turns. The model
            # sees exactly how the previous source text was rendered, which is
            # both a stronger continuity signal and far less likely to be
            # re-translated than context pasted into the request itself.
            messages = [{"role": "system", "content": self.system_prompt}]
            for prev_src, prev_tgt in self._history:
                messages.append({"role": "user", "content": self._request_text(prev_src)})
                messages.append({"role": "assistant", "content": prev_tgt})
            messages.append({"role": "user", "content": self._request_text(clean_input)})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000,
            )

            english_text = response.choices[0].message.content.strip()

            if not english_text:
                logger.debug("Empty translation result.")
                return None

            if self.filter_hallucinations:
                english_text = sanitize_transcript(english_text, source="translation")
                if not english_text:
                    return None

            # Store only clean pairs for context continuity.
            self._history.append((clean_input, english_text))

            logger.info("Translated: %s", english_text[:80] + ("..." if len(english_text) > 80 else ""))
            return english_text

        except Exception as e:
            self.last_error = str(e)
            logger.error("Translation failed: %s", e)
            return None
