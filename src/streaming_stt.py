"""True streaming speech-to-text via the Deepgram live WebSocket API.

Unlike the chunk-based :class:`~src.transcriber.Transcriber`, this keeps a
persistent socket open, streams raw PCM continuously, and lets Deepgram do the
endpointing. It emits **interim** transcripts (for a live on-screen preview)
and **final** transcripts (one per finished utterance) which the pipeline then
translates and speaks.

Audio is expected as linear16 (signed 16-bit little-endian) mono PCM at
``sample_rate``. The caller pushes frames with :meth:`feed`; a background task
owns the socket lifecycle and reconnects on unexpected drops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Callable, Optional
from urllib.parse import urlencode

from .hallucination import sanitize_transcript

logger = logging.getLogger(__name__)

_DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
# Send a KeepAlive if no audio has been forwarded for this long, so Deepgram
# doesn't close the socket during silence.
_KEEPALIVE_IDLE_SEC = 5.0


class DeepgramStreamingTranscriber:
    """Streaming STT client for Deepgram's live WebSocket endpoint."""

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
        self.api_key = api_key
        self.model = model
        self.language = (language or "uk").strip().lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.interim_results = interim_results
        self.filter_hallucinations = filter_hallucinations
        self.on_interim = on_interim
        self.on_final = on_final
        self.max_reconnects = max_reconnects

        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._final_buffer: list[str] = []
        self.last_error: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def feed(self, pcm: bytes) -> None:
        """Queue a PCM frame for streaming (non-blocking, drops if stopped)."""
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
                logger.error("Deepgram stream error: %s", e)
                attempts += 1
                if not self._running or attempts > self.max_reconnects:
                    break
                await asyncio.sleep(min(0.5 * 2 ** (attempts - 1), 8.0))
        self._running = False

    async def stop(self) -> None:
        """Stop streaming and unblock the sender."""
        self._running = False
        self._audio_q.put_nowait(b"")  # wake the sender so it can exit

    # ── URL ───────────────────────────────────────────────────────────

    def _build_url(self) -> str:
        params = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
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
        return f"{_DEEPGRAM_WS_URL}?{urlencode(params)}"

    # ── Session ───────────────────────────────────────────────────────

    async def _session_once(self) -> None:
        import aiohttp

        url = self._build_url()
        headers = {"Authorization": f"Token {self.api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, headers=headers, heartbeat=None) as ws:
                logger.info("Deepgram stream connected (%s/%s)", self.model, self.language)
                self.last_error = None
                sender = asyncio.create_task(self._sender(ws))
                try:
                    await self._receiver(ws)
                finally:
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await sender

    async def _sender(self, ws) -> None:
        """Forward queued PCM frames; KeepAlive during silence."""
        while self._running:
            try:
                pcm = await asyncio.wait_for(self._audio_q.get(), timeout=_KEEPALIVE_IDLE_SEC)
            except asyncio.TimeoutError:
                await ws.send_str(json.dumps({"type": "KeepAlive"}))
                continue
            if pcm:
                await ws.send_bytes(pcm)
        # Graceful close — let Deepgram flush any final result.
        with contextlib.suppress(Exception):
            await ws.send_str(json.dumps({"type": "CloseStream"}))

    async def _receiver(self, ws) -> None:
        import aiohttp

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    self._handle(json.loads(msg.data))
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    logger.debug("Skipping malformed Deepgram message: %s", e)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    # ── Message handling (pure — unit tested directly) ────────────────

    def _handle(self, data: dict) -> None:
        """Process one decoded Deepgram message, firing interim/final hooks."""
        msg_type = data.get("type")

        if msg_type == "Results":
            alts = data.get("channel", {}).get("alternatives", [])
            text = (alts[0].get("transcript") if alts else "") or ""
            text = text.strip()
            if not text:
                return
            if data.get("is_final"):
                self._final_buffer.append(text)
                if data.get("speech_final"):
                    self._emit_final()
            elif self.on_interim:
                # Live preview = already-finalized words + the current guess.
                self.on_interim(" ".join(self._final_buffer + [text]).strip())

        elif msg_type == "UtteranceEnd":
            # Endpoint reached — flush whatever finals we've accumulated.
            self._emit_final()

    def _emit_final(self) -> None:
        if not self._final_buffer:
            return
        text = " ".join(self._final_buffer).strip()
        self._final_buffer = []
        if self.filter_hallucinations:
            text = sanitize_transcript(text, source="STT")
        if text and self.on_final:
            self.on_final(text)
