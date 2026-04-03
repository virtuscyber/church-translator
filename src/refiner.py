"""Post-translation text refinement — cleans up translated text before TTS.

Sits between the Translator and Synthesizer in the pipeline:

    STT → Translate → **Refine** → TTS

Handles:
- Filler words and verbal tics ("um", "uh", "you know", "like")
- False starts and backtracks ("We should— actually, let me say it this way")
- Redundant repetitions from the speaker rephrasing the same idea
- Awkward word-for-word translation artifacts
- Smoothing into natural spoken English suitable for TTS output

Uses a lightweight GPT call with a focused prompt. The refinement is
deliberately conservative — it preserves meaning and tone, only cleaning
up delivery artifacts that would sound unnatural through a speaker.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_REFINER_SYSTEM_PROMPT = """\
You are a live speech refinement engine for a church translation system. \
You receive translated English text from a sermon that was originally spoken \
in another language. Your job is to clean the text so it sounds natural and \
clear when read aloud by a text-to-speech system.

REFINEMENT RULES:
1. Remove filler words: "um", "uh", "well", "you know", "like", "so", \
   "I mean", "right", "okay", "actually" (when used as filler, not emphasis).
2. Clean up false starts and backtracks: when the speaker started a thought, \
   abandoned it, and rephrased — keep ONLY the final version.
   Example: "We need to— actually what I want to say is we must pray" → \
   "We must pray"
3. Remove redundant self-corrections: "It's not about money, I mean it's \
   not about wealth" → "It's not about wealth"
4. Smooth awkward phrasing from literal translation into natural spoken English, \
   while preserving biblical/liturgical vocabulary.
5. Preserve the speaker's meaning, emphasis, and emotional tone EXACTLY.
6. Preserve all scripture references, proper nouns, and theological terms.
7. Do NOT add content, commentary, or interpretation.
8. Do NOT change the theological meaning or soften/strengthen statements.
9. Keep the output concise — every word should earn its place in the listener's ear.
10. If the text is already clean, return it unchanged.

CONTEXT: You will be given recent previous translations for continuity. Use them \
to understand what the speaker has been talking about — this helps identify \
when a new chunk is a continuation vs. a rephrase of the same idea.

OUTPUT: Only the refined English text. No notes, no explanations."""


class Refiner:
    """Refines translated text before TTS synthesis.

    Lightweight GPT call that cleans up filler, false starts, and
    awkward phrasing while preserving meaning and biblical tone.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        context_window: int = 3,
        enabled: bool = True,
    ):
        """Initialize the refiner.

        Args:
            api_key: OpenAI API key.
            model: Model to use (gpt-4o-mini is fast and cheap enough
                   for this lightweight task — ~10-50 tokens in/out).
            temperature: Low temperature for consistent, conservative edits.
            context_window: Number of previous refined outputs to keep
                           for continuity context.
            enabled: When False, refine() passes text through unchanged.
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.enabled = enabled
        self._context: deque[str] = deque(maxlen=context_window)

    async def refine(self, translated_text: str) -> str:
        """Refine translated text for TTS output.

        Args:
            translated_text: English text from the translator.

        Returns:
            Cleaned text ready for TTS. Returns original text on
            failure or when refinement is disabled.
        """
        if not self.enabled or not translated_text:
            return translated_text

        try:
            # Build context from recent outputs
            context_block = ""
            if self._context:
                prev = " | ".join(self._context)
                context_block = f"\n\n[Recent context — what the speaker has been saying: {prev}]"

            messages = [
                {"role": "system", "content": _REFINER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Refine this translated sermon text for TTS:{context_block}\n\n{translated_text}",
                },
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000,
            )

            refined = response.choices[0].message.content.strip()

            if not refined:
                logger.warning("Refiner returned empty text, using original")
                self._context.append(translated_text)
                return translated_text

            # Log the diff for debugging
            if refined != translated_text:
                logger.info(
                    "Refined: %r → %r",
                    translated_text[:60] + ("..." if len(translated_text) > 60 else ""),
                    refined[:60] + ("..." if len(refined) > 60 else ""),
                )
            else:
                logger.debug("Refiner: text unchanged")

            self._context.append(refined)
            return refined

        except Exception as e:
            logger.error("Refinement failed: %s (passing through original text)", e)
            self._context.append(translated_text)
            return translated_text
