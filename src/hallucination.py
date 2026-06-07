"""Anti-hallucination filters for the STT → translation pipeline.

Speech-to-text models in the Whisper family (including ``gpt-4o-transcribe``)
reliably *invent* text when fed non-speech audio — silence, breaths, room
noise, music, or applause. In a church setting the classic artifacts are
"thank you for watching", "subscribe to the channel", subtitle credits, or a
single word/phrase looped many times. Those phantom transcripts then get
faithfully translated and spoken aloud, which is exactly the unreliability we
want to remove.

This module centralizes three cheap, deterministic guards used by both the
transcriber and the translator:

1. ``analyze_wav`` / ``is_probably_silence`` — gate near-silent audio *before*
   it ever reaches the STT API (saves cost and removes the #1 hallucination
   source).
2. ``is_hallucination_phrase`` — reject transcripts that match well-known
   model artifacts.
3. ``has_runaway_repetition`` — reject degenerate repetition loops.

``sanitize_transcript`` ties the text guards together and returns a cleaned
string, or ``None`` when the text should be dropped.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Audio silence gate ───────────────────────────────────────────────

def analyze_wav(wav_bytes: bytes) -> Optional[tuple[float, float, float]]:
    """Return (duration_sec, peak_amplitude, rms_amplitude) for WAV bytes.

    Amplitudes are normalized to the 0.0–1.0 range. Returns ``None`` if the
    audio cannot be parsed, so callers can fail open (proceed to STT) rather
    than dropping a chunk on a parse error.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate() or 1
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return None

    if not raw:
        return 0.0, 0.0, 0.0

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None

    if samples.size == 0:
        return 0.0, 0.0, 0.0

    duration = samples.size / sample_rate
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return duration, peak, rms


def is_probably_silence(
    wav_bytes: bytes,
    *,
    min_duration_sec: float = 0.4,
    silence_peak: float = 0.008,
) -> bool:
    """Return True if a chunk is too short or too quiet to contain real speech.

    Gating on *peak* (not average) amplitude means a chunk with even a brief
    burst of genuine speech passes through — only chunks that are entirely
    low-level noise/silence are dropped, which is precisely the case that
    triggers STT hallucination.
    """
    stats = analyze_wav(wav_bytes)
    if stats is None:
        return False  # Fail open — let STT decide.
    duration, peak, _rms = stats
    if duration < min_duration_sec:
        return True
    if peak < silence_peak:
        return True
    return False


# ── Text guards ──────────────────────────────────────────────────────

# Normalized (lowercased, punctuation-stripped) transcripts that match these
# exactly are model artifacts, not speech. Covers Ukrainian, Russian, and the
# English fallbacks the models emit even on non-English audio.
_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        # English (Whisper's most common phantom captions)
        "thank you",
        "thank you for watching",
        "thanks for watching",
        "thank you for watching this video",
        "please subscribe",
        "subscribe to my channel",
        "subscribe to the channel",
        "don't forget to subscribe",
        "like and subscribe",
        "see you next time",
        "see you in the next video",
        "bye",
        "bye bye",
        "you",
        "the end",
        "music",
        "applause",
        "silence",
        # Ukrainian
        "дякую за перегляд",
        "дякую за увагу",
        "дякую",
        "дякую вам за перегляд",
        "підписуйтесь на канал",
        "підписуйтеся на канал",
        "субтитри",
        "субтитри створено спільнотою",
        "продовження далі",
        "до зустрічі",
        "редактор субтитрів",
        # Russian
        "спасибо за просмотр",
        "спасибо за внимание",
        "спасибо",
        "подписывайтесь на канал",
        "подпишитесь на канал",
        "субтитры",
        "субтитры сделал",
        "субтитры создавал",
        "продолжение следует",
        "до новых встреч",
    }
)


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation/symbols, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    # Remove anything that isn't a letter, digit, or whitespace.
    text = "".join(
        ch for ch in text
        if ch.isalnum() or ch.isspace()
    )
    return re.sub(r"\s+", " ", text).strip()


def is_hallucination_phrase(text: str) -> bool:
    """Return True if the whole transcript is a known model artifact."""
    norm = _normalize(text)
    if not norm:
        return True
    return norm in _HALLUCINATION_PHRASES


def has_runaway_repetition(text: str, *, min_repeats: int = 4) -> bool:
    """Detect degenerate repetition loops that STT models fall into.

    Catches three patterns:
    - a single word repeated consecutively (``так так так так``),
    - a short phrase repeated consecutively (``glory glory glory ...``),
    - very low lexical diversity over a longer transcript.
    """
    tokens = _normalize(text).split()
    if len(tokens) < min_repeats:
        return False

    # 1. Consecutive identical-token run.
    run = max_run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    if max_run >= min_repeats:
        return True

    # 2. Consecutive repeated phrase (window of 2–4 tokens).
    for size in (2, 3, 4):
        if len(tokens) < size * min_repeats:
            continue
        i = 0
        while i + size <= len(tokens):
            window = tokens[i:i + size]
            reps = 1
            j = i + size
            while tokens[j:j + size] == window:
                reps += 1
                j += size
            if reps >= min_repeats:
                return True
            i += 1

    # 3. Pathologically low lexical diversity.
    if len(tokens) >= 8 and len(set(tokens)) / len(tokens) < 0.3:
        return True

    return False


def sanitize_transcript(text: Optional[str], *, source: str = "STT") -> Optional[str]:
    """Clean a transcript/translation, or return ``None`` if it should be dropped.

    Args:
        text: Raw model output.
        source: Label used in log messages ("STT" or "translation").

    Returns:
        The stripped text, or ``None`` when it is empty, a known hallucination
        phrase, or a runaway repetition loop.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    if is_hallucination_phrase(stripped):
        logger.warning("%s dropped (hallucination phrase): %r", source, stripped[:80])
        return None

    if has_runaway_repetition(stripped):
        logger.warning("%s dropped (repetition loop): %r", source, stripped[:80])
        return None

    return stripped
