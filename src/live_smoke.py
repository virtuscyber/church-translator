"""Opt-in *live* smoke checks against the real provider APIs.

Unlike the unit/integration tests (which fake every provider), this performs a
single real round-trip per configured provider so you can confirm your keys,
models, and the streaming sockets actually work before a service:

  1. OpenAI translation       — text round-trip
  2. Text-to-speech           — synthesize a known phrase (produces audio for #3/#4)
  3. Chunked STT round-trip   — transcribe that audio (per provider with a key)
  4. Streaming STT round-trip — stream that audio over the WebSocket and get a
                                final transcript (per streaming provider with a key)

Checks for providers without a key are reported as ``skip`` (never fail), so it
degrades gracefully. Drive it from the CLI (``scripts/smoke_live.py``) or the
dashboard (the "Live API test" panel), both of which consume the same events.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import wave
from typing import Awaitable, Callable, Optional

# The phrase we synthesize and then transcribe back. English so any STT
# provider/language handles it; short to keep the round-trip cheap.
SMOKE_TEXT = "Glory to God. Amen."
_TTS_RATE = 24000  # ElevenLabs pcm_24000 / OpenAI pcm are both 24 kHz mono 16-bit

EmitFn = Callable[[dict], Awaitable[None]]


def _has_key(value: str) -> bool:
    v = (value or "").strip().lower()
    return bool(v) and not v.startswith(("your-", "sk-your"))


def _pcm_to_wav(pcm: bytes, rate: int = _TTS_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def run_smoke(emit: EmitFn, cfg=None) -> dict:
    """Run the live checks, calling ``emit`` for each ``start``/``result`` event.

    Returns a summary dict ``{"passed", "failed", "skipped"}``.
    """
    if cfg is None:
        from src.config import load_config
        cfg = load_config()

    summary = {"passed": 0, "failed": 0, "skipped": 0}

    async def check(check_id: str, label: str, make_coro, *, skip_reason: Optional[str] = None):
        await emit({"phase": "start", "id": check_id, "label": label})
        if skip_reason:
            summary["skipped"] += 1
            await emit({"phase": "result", "id": check_id, "label": label,
                        "status": "skip", "detail": skip_reason, "elapsed": 0.0})
            return None
        t0 = time.monotonic()
        try:
            ok, detail = await asyncio.wait_for(make_coro(), timeout=40.0)
            status = "pass" if ok else "fail"
            summary["passed" if ok else "failed"] += 1
        except asyncio.TimeoutError:
            status, detail = "fail", "timed out"
            summary["failed"] += 1
        except Exception as e:  # noqa: BLE001
            status, detail = "fail", str(e)[:200]
            summary["failed"] += 1
        await emit({"phase": "result", "id": check_id, "label": label,
                    "status": status, "detail": detail,
                    "elapsed": round(time.monotonic() - t0, 2)})
        return status == "pass"

    has_openai = _has_key(cfg.openai_api_key)
    has_eleven = _has_key(cfg.elevenlabs_api_key)
    has_deepgram = _has_key(getattr(cfg, "deepgram_api_key", ""))

    # 1. Translation ---------------------------------------------------
    await check(
        "translation", "OpenAI translation",
        lambda: _check_translation(cfg),
        skip_reason=None if has_openai else "no OpenAI API key",
    )

    # 2. TTS (also produces the audio reused by the STT round-trips) ----
    tts_provider = "elevenlabs" if has_eleven else ("openai" if has_openai else None)
    audio_pcm: Optional[bytes] = None
    if tts_provider:
        await emit({"phase": "start", "id": "tts", "label": f"Text-to-speech ({tts_provider})"})
        t0 = time.monotonic()
        try:
            audio_pcm = await asyncio.wait_for(_synthesize(cfg, tts_provider), timeout=40.0)
            ok = bool(audio_pcm and len(audio_pcm) > 1000)
            summary["passed" if ok else "failed"] += 1
            await emit({"phase": "result", "id": "tts", "label": f"Text-to-speech ({tts_provider})",
                        "status": "pass" if ok else "fail",
                        "detail": f"{len(audio_pcm or b'')} bytes of audio",
                        "elapsed": round(time.monotonic() - t0, 2)})
            if not ok:
                audio_pcm = None
        except Exception as e:  # noqa: BLE001
            summary["failed"] += 1
            audio_pcm = None
            await emit({"phase": "result", "id": "tts", "label": f"Text-to-speech ({tts_provider})",
                        "status": "fail", "detail": str(e)[:200], "elapsed": round(time.monotonic() - t0, 2)})
    else:
        summary["skipped"] += 1
        await emit({"phase": "start", "id": "tts", "label": "Text-to-speech"})
        await emit({"phase": "result", "id": "tts", "label": "Text-to-speech",
                    "status": "skip", "detail": "no TTS provider key", "elapsed": 0.0})

    wav = _pcm_to_wav(audio_pcm) if audio_pcm else None
    no_audio = None if wav else "no TTS audio to transcribe"

    # 3. Chunked STT round-trip per provider with a key ----------------
    provider_keys = {"openai": has_openai, "elevenlabs": has_eleven, "deepgram": has_deepgram}
    for provider, present in provider_keys.items():
        reason = (f"no {provider} key" if not present else no_audio)
        await check(
            f"stt_chunk_{provider}", f"Chunked STT — {provider}",
            lambda p=provider: _check_stt_chunked(cfg, p, wav),
            skip_reason=reason,
        )

    # 4. Streaming STT round-trip per streaming provider with a key -----
    from src.streaming_stt import STREAMING_PROVIDERS
    for provider in ("deepgram", "elevenlabs", "openai"):
        if provider not in STREAMING_PROVIDERS:
            continue
        reason = (f"no {provider} key" if not provider_keys[provider] else no_audio)
        await check(
            f"stt_stream_{provider}", f"Streaming STT — {provider}",
            lambda p=provider: _check_stt_streaming(cfg, p, audio_pcm),
            skip_reason=reason,
        )

    await emit({"phase": "done", **summary})
    return summary


# ── Individual checks ─────────────────────────────────────────────────

async def _check_translation(cfg):
    from src.translator import Translator
    tr = Translator(
        api_key=cfg.openai_api_key,
        system_prompt="You are a translator. Translate the user's text to English.",
        model=cfg.translation.model,
        source_language="Ukrainian",
        target_language="English",
        filter_hallucinations=False,
    )
    out = await tr.translate("Слава Богу")
    return bool(out and out.strip()), (out or "no output")[:120]


async def _synthesize(cfg, provider) -> bytes:
    from src.synthesizer import Synthesizer
    s = Synthesizer(
        provider=provider,
        openai_api_key=cfg.openai_api_key,
        elevenlabs_api_key=cfg.elevenlabs_api_key,
        elevenlabs_voice_id=cfg.synthesis.elevenlabs.voice_id,
        elevenlabs_model=cfg.synthesis.elevenlabs.model,
        openai_model=cfg.synthesis.openai.model,
        openai_voice=cfg.synthesis.openai.voice,
    )
    return await s.synthesize(SMOKE_TEXT) or b""


def _stt_creds(cfg, provider):
    return {
        "openai": (cfg.openai_api_key, cfg.transcription.model),
        "elevenlabs": (cfg.elevenlabs_api_key, cfg.transcription.elevenlabs_model),
        "deepgram": (getattr(cfg, "deepgram_api_key", ""), cfg.transcription.deepgram_model),
    }[provider]


async def _check_stt_chunked(cfg, provider, wav: bytes):
    from src.transcriber import Transcriber
    t = Transcriber(
        api_key=cfg.openai_api_key,
        language="en",  # we transcribe the English phrase we synthesized
        provider=provider,
        elevenlabs_api_key=cfg.elevenlabs_api_key,
        elevenlabs_model=cfg.transcription.elevenlabs_model,
        deepgram_api_key=getattr(cfg, "deepgram_api_key", ""),
        deepgram_model=cfg.transcription.deepgram_model,
        gate_silence=False,
        filter_hallucinations=False,
    )
    out = await t.transcribe(wav)
    return bool(out and out.strip()), (out or "no transcript")[:120]


async def _check_stt_streaming(cfg, provider, pcm: bytes):
    from src.streaming_stt import make_streaming_transcriber

    key, _ = _stt_creds(cfg, provider)
    model = {
        "deepgram": cfg.transcription.deepgram_model,
        "elevenlabs": cfg.transcription.elevenlabs_realtime_model,
        "openai": cfg.transcription.openai_realtime_model,
    }[provider]

    finals: list[str] = []
    stt = make_streaming_transcriber(
        provider, api_key=key, model=model, language="en",
        sample_rate=_TTS_RATE, filter_hallucinations=False,
        on_final=finals.append,
    )
    run_task = asyncio.create_task(stt.run())
    try:
        # Feed the audio in ~20 ms frames at real-time pace, then a bit of
        # trailing silence so the provider's endpointing fires.
        frame = int(_TTS_RATE * 0.02) * 2  # bytes per 20 ms
        for i in range(0, len(pcm), frame):
            stt.feed(pcm[i:i + frame])
            await asyncio.sleep(0.02)
        silence = b"\x00" * frame
        for _ in range(50):  # ~1 s of silence to trigger an utterance end
            stt.feed(silence)
            await asyncio.sleep(0.02)
        # Wait up to ~8 s for a final transcript.
        deadline = time.monotonic() + 8.0
        while not finals and time.monotonic() < deadline:
            if stt.last_error:
                return False, f"socket error: {stt.last_error}"[:160]
            await asyncio.sleep(0.05)
    finally:
        await stt.stop()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run_task
    return bool(finals), (finals[0] if finals else "no final transcript")[:120]
