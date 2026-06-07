from __future__ import annotations

import asyncio
import json

import pytest

from src.streaming_stt import DeepgramStreamingTranscriber


def _results(transcript, *, is_final=False, speech_final=False):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript}]},
    }


def _make(**kw):
    finals, interims = [], []
    t = DeepgramStreamingTranscriber(
        "key",
        on_interim=interims.append,
        on_final=finals.append,
        **kw,
    )
    return t, finals, interims


def test_interim_results_drive_preview_not_final():
    t, finals, interims = _make()
    t._handle(_results("слава", is_final=False))
    assert interims == ["слава"]
    assert finals == []


def test_is_final_buffers_until_speech_final():
    t, finals, interims = _make()
    t._handle(_results("слава", is_final=True, speech_final=False))
    assert finals == []  # buffered, utterance not done
    t._handle(_results("богу", is_final=True, speech_final=True))
    assert finals == ["слава богу"]  # flushed on speech_final
    # buffer cleared afterwards
    assert t._final_buffer == []


def test_utterance_end_flushes_buffer():
    t, finals, _ = _make()
    t._handle(_results("привіт світ", is_final=True, speech_final=False))
    t._handle({"type": "UtteranceEnd"})
    assert finals == ["привіт світ"]


def test_interim_preview_includes_buffered_finals():
    t, finals, interims = _make()
    t._handle(_results("слава", is_final=True, speech_final=False))
    t._handle(_results("богу", is_final=False))
    assert interims[-1] == "слава богу"  # finalized words + current guess


def test_empty_transcript_ignored():
    t, finals, interims = _make()
    t._handle(_results("   ", is_final=True, speech_final=True))
    assert finals == [] and interims == []


def test_hallucination_filtered_from_final():
    t, finals, _ = _make(filter_hallucinations=True)
    # A known artifact phrase should be stripped before on_final fires.
    t._handle(_results("Thanks for watching", is_final=True, speech_final=True))
    assert finals == []


def test_build_url_has_streaming_params():
    t, _, _ = _make(model="nova-3", language="uk", sample_rate=48000)
    url = t._build_url()
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    for token in ("model=nova-3", "language=uk", "encoding=linear16",
                  "sample_rate=48000", "interim_results=true", "endpointing=300"):
        assert token in url


def test_feed_is_dropped_when_not_running():
    t, _, _ = _make()
    t.feed(b"\x00\x00")  # not running -> ignored
    assert t._audio_q.qsize() == 0
    t._running = True
    t.feed(b"\x00\x00")
    assert t._audio_q.qsize() == 1


@pytest.mark.asyncio
async def test_receiver_dispatches_messages_then_stops():
    import aiohttp

    t, finals, _ = _make()

    class Msg:
        def __init__(self, mtype, data=""):
            self.type = mtype
            self.data = data

    class FakeWS:
        def __init__(self, msgs):
            self._msgs = msgs

        def __aiter__(self):
            self._it = iter(self._msgs)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    msgs = [
        Msg(aiohttp.WSMsgType.TEXT, json.dumps(_results("слава богу", is_final=True, speech_final=True))),
        Msg(aiohttp.WSMsgType.CLOSED),
    ]
    await t._receiver(FakeWS(msgs))
    assert finals == ["слава богу"]


@pytest.mark.asyncio
async def test_stop_unblocks_and_clears_running():
    t, _, _ = _make()
    t._running = True
    await t.stop()
    assert t._running is False
    assert t._audio_q.qsize() == 1  # sentinel to wake the sender


# ── ElevenLabs Scribe v2 Realtime ─────────────────────────────────────

def _make_el(**kw):
    from src.streaming_stt import ElevenLabsStreamingTranscriber
    finals, interims = [], []
    t = ElevenLabsStreamingTranscriber("k", on_interim=interims.append, on_final=finals.append, **kw)
    return t, finals, interims


def test_elevenlabs_partial_and_committed():
    t, finals, interims = _make_el()
    t._handle({"message_type": "partial_transcript", "text": "слава"})
    assert interims == ["слава"] and finals == []
    t._handle({"message_type": "final_transcript", "text": "слава богу"})
    assert finals == ["слава богу"]


def test_elevenlabs_accepts_alternate_committed_names():
    for mtype in ("committed_transcript", "transcript", "committed"):
        t, finals, _ = _make_el()
        t._handle({"message_type": mtype, "text": "амінь"})
        assert finals == ["амінь"], mtype


def test_elevenlabs_url_and_headers():
    t, _, _ = _make_el(model="scribe_v2_realtime", language="uk")
    url = t._url()
    assert url.startswith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")
    assert "model_id=scribe_v2_realtime" in url and "language_code=uk" in url
    assert t._headers()["xi-api-key"] == "k"


@pytest.mark.asyncio
async def test_elevenlabs_send_audio_is_base64_json():
    t, _, _ = _make_el(sample_rate=16000)
    sent = {}

    class WS:
        async def send_str(self, s):
            sent.update(json.loads(s))

    await t._send_audio(WS(), b"\x01\x02\x03\x04")
    assert sent["message_type"] == "input_audio_chunk"
    assert sent["sample_rate"] == 16000
    import base64
    assert base64.b64decode(sent["audio_base_64"]) == b"\x01\x02\x03\x04"


# ── OpenAI Realtime (gpt-realtime-whisper) ────────────────────────────

def _make_oai(**kw):
    from src.streaming_stt import OpenAIRealtimeTranscriber
    finals, interims = [], []
    t = OpenAIRealtimeTranscriber("k", on_interim=interims.append, on_final=finals.append, **kw)
    return t, finals, interims


def test_openai_delta_accumulates_then_completes():
    t, finals, interims = _make_oai()
    t._handle({"type": "conversation.item.input_audio_transcription.delta", "delta": "слава"})
    t._handle({"type": "conversation.item.input_audio_transcription.delta", "delta": " богу"})
    assert interims[-1] == "слава богу"
    t._handle({"type": "conversation.item.input_audio_transcription.completed", "transcript": "слава богу"})
    assert finals == ["слава богу"]
    assert t._delta_buffer == ""  # reset after completion


def test_openai_error_sets_last_error():
    t, _, _ = _make_oai()
    t._handle({"type": "error", "error": {"message": "bad session"}})
    assert t.last_error == "bad session"


def test_openai_headers_and_url():
    t, _, _ = _make_oai()
    assert t._url() == "wss://api.openai.com/v1/realtime?intent=transcription"
    h = t._headers()
    assert h["Authorization"] == "Bearer k" and h["OpenAI-Beta"] == "realtime=v1"


@pytest.mark.asyncio
async def test_openai_resamples_to_24k_and_sends_append():
    import numpy as np
    t, _, _ = _make_oai(sample_rate=48000)
    sent = {}

    class WS:
        async def send_str(self, s):
            sent.update(json.loads(s))

    # 48 samples @48k -> ~24 samples @24k
    pcm = np.full(48, 1000, dtype="<i2").tobytes()
    await t._send_audio(WS(), pcm)
    assert sent["type"] == "input_audio_buffer.append"
    import base64
    out = np.frombuffer(base64.b64decode(sent["audio"]), dtype="<i2")
    assert 22 <= len(out) <= 26  # halved sample count


# ── Factory ───────────────────────────────────────────────────────────

def test_factory_builds_each_engine():
    from src.streaming_stt import (
        make_streaming_transcriber, DeepgramStreamingTranscriber,
        ElevenLabsStreamingTranscriber, OpenAIRealtimeTranscriber,
    )
    common = dict(api_key="k", language="uk", sample_rate=48000)
    assert isinstance(make_streaming_transcriber("deepgram", model="nova-3", **common), DeepgramStreamingTranscriber)
    assert isinstance(make_streaming_transcriber("elevenlabs", model="scribe_v2_realtime", **common), ElevenLabsStreamingTranscriber)
    assert isinstance(make_streaming_transcriber("openai", model="gpt-realtime-whisper", **common), OpenAIRealtimeTranscriber)
    with pytest.raises(ValueError):
        make_streaming_transcriber("nope", model="x", **common)
