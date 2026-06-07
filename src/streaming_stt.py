"""True streaming speech-to-text over WebSockets.

Three interchangeable engines behind one interface:

- :class:`DeepgramStreamingTranscriber`   — Deepgram live (raw linear16 frames)
- :class:`ElevenLabsStreamingTranscriber` — Scribe v2 Realtime (base64 chunks)
- :class:`OpenAIRealtimeTranscriber`      — gpt-realtime-whisper (Realtime API)

Each keeps a persistent socket open, streams PCM continuously, and lets the
provider do the endpointing. They emit **interim** transcripts (for a live
on-screen preview) and **final** transcripts (one per finished utterance) via
the ``on_interim`` / ``on_final`` callbacks. Audio is fed as int16-LE mono PCM
at ``in_sample_rate``; engines that need a different rate resample internally.

Build one with :func:`make_streaming_transcriber`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Callable, Optional
from urllib.parse import urlencode

import numpy as np

from .hallucination import sanitize_transcript

logger = logging.getLogger(__name__)

# Send a keepalive if no audio has been forwarded for this long, so the socket
# isn't closed during silence.
_KEEPALIVE_IDLE_SEC = 5.0


class StreamingTranscriber:
    """Base class: socket lifecycle, reconnect, and the emit/resample helpers.

    Subclasses implement the provider wire protocol via the hooks at the bottom.
    """

    def __init__(
        self,
        *,
        language: str = "uk",
        in_sample_rate: int = 48000,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        filter_hallucinations: bool = True,
        max_reconnects: int = 5,
    ):
        self.language = (language or "uk").strip().lower()
        self.in_sample_rate = in_sample_rate
        self.on_interim = on_interim
        self.on_final = on_final
        self.filter_hallucinations = filter_hallucinations
        self.max_reconnects = max_reconnects

        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self.last_error: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def feed(self, pcm: bytes) -> None:
        """Queue an int16-LE mono PCM frame (non-blocking; dropped if stopped)."""
        if self._running and pcm:
            self._audio_q.put_nowait(pcm)

    async def run(self) -> None:
        """Own the socket for the session, reconnecting on unexpected drops."""
        self._running = True
        attempts = 0
        while self._running:
            try:
                await self._session_once()
                attempts = 0  # clean close resets the backoff
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - surface + reconnect
                self.last_error = str(e)
                logger.error("%s stream error: %s", type(self).__name__, e)
                attempts += 1
                if not self._running or attempts > self.max_reconnects:
                    break
                await asyncio.sleep(min(0.5 * 2 ** (attempts - 1), 8.0))
        self._running = False

    async def stop(self) -> None:
        """Stop streaming and unblock the sender."""
        self._running = False
        self._audio_q.put_nowait(b"")  # wake the sender so it can exit

    # ── Shared helpers ────────────────────────────────────────────────

    def _fire_interim(self, text: str) -> None:
        text = (text or "").strip()
        if text and self.on_interim:
            self.on_interim(text)

    def _fire_final(self, text: str) -> None:
        text = (text or "").strip()
        if self.filter_hallucinations:
            text = sanitize_transcript(text, source="STT")
        if text and self.on_final:
            self.on_final(text)

    def _resample(self, pcm: bytes, out_rate: int) -> bytes:
        """Resample int16-LE mono PCM from ``in_sample_rate`` to ``out_rate``."""
        if out_rate == self.in_sample_rate or not pcm:
            return pcm
        a = np.frombuffer(pcm, dtype="<i2")
        if a.size == 0:
            return b""
        n_out = max(1, int(round(a.size * out_rate / self.in_sample_rate)))
        idx = np.linspace(0, a.size - 1, n_out)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, a.size - 1)
        frac = idx - lo
        out = (a[lo] * (1.0 - frac) + a[hi] * frac).astype("<i2")
        return out.tobytes()

    # ── Socket loop (shared) ──────────────────────────────────────────

    async def _session_once(self) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                self._url(), headers=self._headers(), heartbeat=None
            ) as ws:
                logger.info("%s connected", type(self).__name__)
                self.last_error = None
                await self._on_open(ws)
                sender = asyncio.create_task(self._sender(ws))
                try:
                    await self._receiver(ws)
                finally:
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await sender

    async def _sender(self, ws) -> None:
        while self._running:
            try:
                pcm = await asyncio.wait_for(self._audio_q.get(), timeout=_KEEPALIVE_IDLE_SEC)
            except asyncio.TimeoutError:
                await self._send_keepalive(ws)
                continue
            if pcm:
                await self._send_audio(ws, pcm)
        with contextlib.suppress(Exception):
            await self._send_close(ws)

    async def _receiver(self, ws) -> None:
        import aiohttp

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    self._handle(json.loads(msg.data))
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    logger.debug("Skipping malformed message: %s", e)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    # ── Provider hooks (override) ─────────────────────────────────────

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict:
        raise NotImplementedError

    async def _on_open(self, ws) -> None:
        """Send any session-configuration message after connect."""
        return None

    async def _send_audio(self, ws, pcm: bytes) -> None:
        raise NotImplementedError

    async def _send_keepalive(self, ws) -> None:
        return None

    async def _send_close(self, ws) -> None:
        return None

    def _handle(self, data: dict) -> None:
        raise NotImplementedError


class DeepgramStreamingTranscriber(StreamingTranscriber):
    """Deepgram live: raw linear16 frames, is_final/speech_final endpointing."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        language: str = "uk",
        sample_rate: int = 48000,
        channels: int = 1,
        endpointing_ms: int = 300,
        utterance_end_ms: int = 1000,
        interim_results: bool = True,
        filter_hallucinations: bool = True,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        max_reconnects: int = 5,
    ):
        super().__init__(
            language=language,
            in_sample_rate=sample_rate,
            on_interim=on_interim,
            on_final=on_final,
            filter_hallucinations=filter_hallucinations,
            max_reconnects=max_reconnects,
        )
        self.api_key = api_key
        self.model = model
        self.channels = channels
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.interim_results = interim_results
        self._final_buffer: list[str] = []

    def _url(self) -> str:
        params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": self.in_sample_rate,
            "channels": self.channels,
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "true" if self.interim_results else "false",
            "endpointing": self.endpointing_ms,
            "utterance_end_ms": self.utterance_end_ms,
            "vad_events": "true",
        }
        if self.language:
            params["language"] = self.language
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    # Backwards-compatible alias (used in tests).
    _build_url = _url

    def _headers(self) -> dict:
        return {"Authorization": f"Token {self.api_key}"}

    async def _send_audio(self, ws, pcm: bytes) -> None:
        await ws.send_bytes(pcm)  # native rate, no resample

    async def _send_keepalive(self, ws) -> None:
        await ws.send_str(json.dumps({"type": "KeepAlive"}))

    async def _send_close(self, ws) -> None:
        await ws.send_str(json.dumps({"type": "CloseStream"}))

    def _handle(self, data: dict) -> None:
        msg_type = data.get("type")
        if msg_type == "Results":
            alts = data.get("channel", {}).get("alternatives", [])
            text = ((alts[0].get("transcript") if alts else "") or "").strip()
            if not text:
                return
            if data.get("is_final"):
                self._final_buffer.append(text)
                if data.get("speech_final"):
                    self._flush()
            else:
                self._fire_interim(" ".join(self._final_buffer + [text]))
        elif msg_type == "UtteranceEnd":
            self._flush()

    def _flush(self) -> None:
        if not self._final_buffer:
            return
        text = " ".join(self._final_buffer).strip()
        self._final_buffer = []
        self._fire_final(text)


class ElevenLabsStreamingTranscriber(StreamingTranscriber):
    """ElevenLabs Scribe v2 Realtime: base64 chunks, partial/committed transcripts."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "scribe_v2_realtime",
        language: str = "uk",
        sample_rate: int = 48000,
        filter_hallucinations: bool = True,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        max_reconnects: int = 5,
    ):
        super().__init__(
            language=language,
            in_sample_rate=sample_rate,
            on_interim=on_interim,
            on_final=on_final,
            filter_hallucinations=filter_hallucinations,
            max_reconnects=max_reconnects,
        )
        self.api_key = api_key
        self.model = model

    def _url(self) -> str:
        params = {"model_id": self.model, "commit_strategy": "vad"}
        if self.language:
            params["language_code"] = self.language
        return f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{urlencode(params)}"

    def _headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    async def _send_audio(self, ws, pcm: bytes) -> None:
        await ws.send_str(json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": self.in_sample_rate,
        }))

    def _handle(self, data: dict) -> None:
        mtype = data.get("message_type")
        text = (data.get("text") or "").strip()
        if mtype == "partial_transcript":
            self._fire_interim(text)
        # The committed/final message name has varied across previews — accept
        # the documented and likely-adjacent spellings.
        elif mtype in ("final_transcript", "committed_transcript", "transcript", "committed"):
            self._fire_final(text)


class OpenAIRealtimeTranscriber(StreamingTranscriber):
    """OpenAI Realtime transcription (gpt-realtime-whisper).

    Requires 24 kHz pcm16; audio is resampled from the capture rate. Interim
    text is accumulated from transcription deltas; the completed event carries
    the full final transcript.
    """

    _OUT_RATE = 24000

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-realtime-whisper",
        language: str = "uk",
        sample_rate: int = 48000,
        filter_hallucinations: bool = True,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        max_reconnects: int = 5,
    ):
        super().__init__(
            language=language,
            in_sample_rate=sample_rate,
            on_interim=on_interim,
            on_final=on_final,
            filter_hallucinations=filter_hallucinations,
            max_reconnects=max_reconnects,
        )
        self.api_key = api_key
        self.model = model
        self._delta_buffer = ""

    def _url(self) -> str:
        return "wss://api.openai.com/v1/realtime?intent=transcription"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}

    async def _on_open(self, ws) -> None:
        transcription = {"model": self.model}
        if self.language:
            transcription["language"] = self.language
        await ws.send_str(json.dumps({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": transcription,
                "turn_detection": {"type": "server_vad"},
            },
        }))

    async def _send_audio(self, ws, pcm: bytes) -> None:
        pcm24 = self._resample(pcm, self._OUT_RATE)
        await ws.send_str(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm24).decode("ascii"),
        }))

    def _handle(self, data: dict) -> None:
        mtype = data.get("type")
        if mtype == "conversation.item.input_audio_transcription.delta":
            self._delta_buffer = (self._delta_buffer + (data.get("delta") or "")).strip()
            self._fire_interim(self._delta_buffer)
        elif mtype == "conversation.item.input_audio_transcription.completed":
            text = data.get("transcript") or self._delta_buffer
            self._delta_buffer = ""
            self._fire_final(text)
        elif mtype == "error":
            err = data.get("error") or {}
            self.last_error = err.get("message") or json.dumps(err)
            logger.warning("OpenAI realtime error: %s", self.last_error)


# Providers that support a true-streaming engine.
STREAMING_PROVIDERS = {"deepgram", "elevenlabs", "openai"}


def make_streaming_transcriber(
    provider: str,
    *,
    api_key: str,
    model: str,
    language: str,
    sample_rate: int,
    filter_hallucinations: bool = True,
    on_interim: Optional[Callable[[str], None]] = None,
    on_final: Optional[Callable[[str], None]] = None,
) -> StreamingTranscriber:
    """Construct the streaming transcriber for ``provider``."""
    common = dict(
        language=language,
        sample_rate=sample_rate,
        filter_hallucinations=filter_hallucinations,
        on_interim=on_interim,
        on_final=on_final,
    )
    if provider == "deepgram":
        return DeepgramStreamingTranscriber(api_key, model=model, **common)
    if provider == "elevenlabs":
        return ElevenLabsStreamingTranscriber(api_key, model=model, **common)
    if provider == "openai":
        return OpenAIRealtimeTranscriber(api_key, model=model, **common)
    raise ValueError(f"Provider {provider!r} has no streaming engine")
