from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.live_smoke as ls


def _cfg(openai="", elevenlabs="", deepgram=""):
    return SimpleNamespace(
        openai_api_key=openai,
        elevenlabs_api_key=elevenlabs,
        deepgram_api_key=deepgram,
        translation=SimpleNamespace(model="gpt-4o"),
        transcription=SimpleNamespace(
            model="gpt-4o-transcribe", elevenlabs_model="scribe_v2",
            deepgram_model="nova-3", elevenlabs_realtime_model="scribe_v2_realtime",
            openai_realtime_model="gpt-realtime-whisper",
        ),
        synthesis=SimpleNamespace(
            elevenlabs=SimpleNamespace(voice_id="v", model="eleven_flash_v2_5"),
            openai=SimpleNamespace(model="gpt-4o-mini-tts", voice="onyx"),
        ),
    )


def _patch_all_pass(monkeypatch):
    async def fake_tr(cfg):
        return True, "Glory to God"

    async def fake_syn(cfg, provider):
        return b"\x00\x01" * 1000

    async def fake_chunk(cfg, provider, wav):
        return True, "glory to god"

    async def fake_stream(cfg, provider, pcm):
        return True, "glory to god"

    monkeypatch.setattr(ls, "_check_translation", fake_tr)
    monkeypatch.setattr(ls, "_synthesize", fake_syn)
    monkeypatch.setattr(ls, "_check_stt_chunked", fake_chunk)
    monkeypatch.setattr(ls, "_check_stt_streaming", fake_stream)


async def _collect(cfg):
    events = []

    async def emit(e):
        events.append(e)

    summary = await ls.run_smoke(emit, cfg=cfg)
    return summary, events


@pytest.mark.asyncio
async def test_run_smoke_all_pass(monkeypatch):
    _patch_all_pass(monkeypatch)
    summary, events = await _collect(_cfg("sk-real", "el-real", "dg-real"))

    assert summary == {"passed": 8, "failed": 0, "skipped": 0}  # transl+tts+3 chunk+3 stream
    assert events[-1]["phase"] == "done"
    results = [e for e in events if e["phase"] == "result"]
    assert len(results) == 8 and all(r["status"] == "pass" for r in results)
    # every result is preceded by a start with the same id
    starts = {e["id"] for e in events if e["phase"] == "start"}
    assert {r["id"] for r in results} <= starts


@pytest.mark.asyncio
async def test_run_smoke_skips_providers_without_keys(monkeypatch):
    _patch_all_pass(monkeypatch)
    # Only OpenAI configured.
    summary, events = await _collect(_cfg(openai="sk-real"))

    # passed: translation, tts(openai), chunked-openai, streaming-openai
    # skipped: chunked el+dg, streaming dg+el
    assert summary["passed"] == 4 and summary["failed"] == 0 and summary["skipped"] == 4
    skipped = {e["id"] for e in events if e.get("status") == "skip"}
    assert {"stt_chunk_elevenlabs", "stt_chunk_deepgram",
            "stt_stream_deepgram", "stt_stream_elevenlabs"} == skipped


@pytest.mark.asyncio
async def test_run_smoke_no_keys_all_skip(monkeypatch):
    # No fakes needed: every check should be skipped before calling out.
    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        return True, "should not run"

    monkeypatch.setattr(ls, "_check_translation", boom)
    monkeypatch.setattr(ls, "_synthesize", boom)
    monkeypatch.setattr(ls, "_check_stt_chunked", boom)
    monkeypatch.setattr(ls, "_check_stt_streaming", boom)

    summary, events = await _collect(_cfg())

    assert summary["passed"] == 0 and summary["failed"] == 0 and summary["skipped"] == 8
    assert called["n"] == 0  # nothing actually called a provider
    assert all(e.get("status", "skip") == "skip" for e in events if e["phase"] == "result")


@pytest.mark.asyncio
async def test_run_smoke_reports_failure(monkeypatch):
    _patch_all_pass(monkeypatch)

    async def bad_translation(cfg):
        raise RuntimeError("401 invalid key")

    monkeypatch.setattr(ls, "_check_translation", bad_translation)
    summary, events = await _collect(_cfg("sk-real"))

    assert summary["failed"] == 1
    tr = next(e for e in events if e["phase"] == "result" and e["id"] == "translation")
    assert tr["status"] == "fail" and "401" in tr["detail"]
