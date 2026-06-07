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
