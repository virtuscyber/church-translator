from __future__ import annotations

import numpy as np

from src.vad_chunker import FRAME_MS, VADChunker


def test_vad_chunker_uses_bounded_preroll_before_speech():
    chunker = VADChunker(input_sample_rate=48000)
    chunker.vad.is_speech = lambda frame: False

    frame_samples = int(chunker.input_sample_rate * FRAME_MS / 1000)
    silence = np.zeros(frame_samples * (chunker._preroll_frame_limit + 5), dtype=np.float32)
    chunker.feed(silence)

    assert chunker._audio_buffer == []
    assert len(chunker._preroll_buffer) == chunker._preroll_frame_limit
    assert chunker._buffered_seconds == 0.0


def test_vad_chunker_promotes_preroll_once_speech_starts():
    chunker = VADChunker(input_sample_rate=48000)
    responses = iter([False, False, True])
    chunker.vad.is_speech = lambda frame: next(responses)

    frame_samples = int(chunker.input_sample_rate * FRAME_MS / 1000)
    audio = np.ones(frame_samples * 3, dtype=np.float32)
    chunker.feed(audio)

    assert chunker._speech_started is True
    assert len(chunker._audio_buffer) == 3
    assert len(chunker._preroll_buffer) == 0
    assert chunker._buffered_seconds > 0
