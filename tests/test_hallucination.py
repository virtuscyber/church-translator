from __future__ import annotations

import io
import wave

import numpy as np

from src.hallucination import (
    analyze_wav,
    has_runaway_repetition,
    is_hallucination_phrase,
    is_probably_silence,
    sanitize_transcript,
)


def _make_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── Audio gate ───────────────────────────────────────────────────────

def test_silence_chunk_is_gated():
    silence = np.zeros(16000, dtype=np.float32)  # 1s of pure silence
    assert is_probably_silence(_make_wav(silence)) is True


def test_low_noise_chunk_is_gated():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.0012, 16000).astype(np.float32)  # noise floor, peak < 0.008
    assert is_probably_silence(_make_wav(noise)) is True


def test_speech_level_chunk_passes():
    t = np.linspace(0, 1, 16000, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # clear signal
    assert is_probably_silence(_make_wav(tone)) is False


def test_too_short_chunk_is_gated():
    t = np.linspace(0, 0.2, 3200, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # loud but 0.2s
    assert is_probably_silence(_make_wav(tone)) is True


def test_brief_speech_in_mostly_silence_passes():
    audio = np.zeros(16000, dtype=np.float32)
    audio[4000:6000] = 0.4  # a short genuine burst — must not be dropped
    assert is_probably_silence(_make_wav(audio)) is False


def test_analyze_wav_handles_garbage_bytes():
    assert analyze_wav(b"not a wav file") is None


def test_unparsable_audio_fails_open():
    # If we can't read the audio, don't silently drop a possibly-good chunk.
    assert is_probably_silence(b"garbage") is False


# ── Phrase filter ────────────────────────────────────────────────────

def test_known_english_hallucination_rejected():
    assert is_hallucination_phrase("Thank you for watching!") is True
    assert is_hallucination_phrase("Please subscribe") is True


def test_known_ukrainian_hallucination_rejected():
    assert is_hallucination_phrase("Дякую за перегляд") is True
    assert is_hallucination_phrase("Субтитри") is True


def test_known_russian_hallucination_rejected():
    assert is_hallucination_phrase("Спасибо за просмотр.") is True


def test_real_sentence_not_flagged_as_phrase():
    assert is_hallucination_phrase("The Lord is my shepherd.") is False
    assert is_hallucination_phrase("Браття, помолимося разом.") is False


# ── Repetition detection ─────────────────────────────────────────────

def test_repeated_word_loop_detected():
    assert has_runaway_repetition("так так так так так так") is True


def test_repeated_phrase_loop_detected():
    assert has_runaway_repetition("glory to god glory to god glory to god glory to god") is True


def test_low_diversity_detected():
    assert has_runaway_repetition("amen amen amen praise amen amen amen amen") is True


def test_normal_sentence_not_flagged_as_repetition():
    assert has_runaway_repetition("The Lord bless you and keep you this day") is False


def test_short_text_not_flagged():
    assert has_runaway_repetition("Glory to God") is False


# ── sanitize_transcript ──────────────────────────────────────────────

def test_sanitize_passes_real_text():
    assert sanitize_transcript("  The Lord is good.  ") == "The Lord is good."


def test_sanitize_drops_empty():
    assert sanitize_transcript("") is None
    assert sanitize_transcript("   ") is None
    assert sanitize_transcript(None) is None


def test_sanitize_drops_hallucination_and_loops():
    assert sanitize_transcript("Thanks for watching") is None
    assert sanitize_transcript("ла ла ла ла ла ла") is None
