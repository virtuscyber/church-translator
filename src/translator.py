"""Ukrainian → English translation with biblical styling via GPT-4o."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Translator:
    """Translates Ukrainian text to biblically-styled English."""

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        context_sentences: int = 2,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self._context: deque[str] = deque(maxlen=context_sentences)

    async def translate(self, ukrainian_text: str) -> Optional[str]:
        """Translate Ukrainian text to English with biblical tone.
        
        Maintains a sliding window of previous translations for context continuity.
        
        Args:
            ukrainian_text: Ukrainian text to translate.
            
        Returns:
            English translation with biblical vocabulary, or None on failure.
        """
        try:
            # Build context from previous translations
            context_block = ""
            if self._context:
                prev = " ".join(self._context)
                context_block = f"\n\n[Previous translation for context: {prev}]"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Translate the following Ukrainian speech to English:{context_block}\n\n{ukrainian_text}",
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

            # Store for context continuity
            self._context.append(english_text)

            logger.info("Translated: %s", english_text[:80] + ("..." if len(english_text) > 80 else ""))
            return english_text

        except Exception as e:
            logger.error("Translation failed: %s", e)
            return None
