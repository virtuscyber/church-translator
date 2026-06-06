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
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.filter_hallucinations = filter_hallucinations
        self._context: deque[str] = deque(maxlen=context_sentences)

    async def translate(self, ukrainian_text: str) -> Optional[str]:
        """Translate Ukrainian text to English with biblical tone.

        Maintains a sliding window of previous translations for context
        continuity. Junk input is rejected up front, and the model's output is
        filtered for hallucination artifacts before being returned or stored as
        context — so a single bad chunk cannot poison later translations.

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

        try:
            # Provide previous translations purely as reference for continuity.
            # The system prompt instructs the model never to re-translate or
            # continue this block.
            context_block = ""
            if self._context:
                prev = " ".join(self._context)
                context_block = (
                    f"\n\n[Reference only — previously translated, do NOT repeat "
                    f"or continue: {prev}]"
                )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Translate the following Ukrainian speech to English:{context_block}\n\n{clean_input}",
                },
            ]

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

            # Store only clean output for context continuity.
            self._context.append(english_text)

            logger.info("Translated: %s", english_text[:80] + ("..." if len(english_text) > 80 else ""))
            return english_text

        except Exception as e:
            logger.error("Translation failed: %s", e)
            return None
