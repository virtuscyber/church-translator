"""AES67/Dante-compatible multicast RTP audio output.

Sends L24/48kHz audio via RTP multicast and announces the stream via SAP/SDP
so it appears in Dante Controller and other AES67-compatible receivers.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# AES67 constants
_AES67_SAMPLE_RATE = 48000
_AES67_BIT_DEPTH = 24          # L24 linear PCM
_AES67_CHANNELS = 1            # Mono for speech translation
_AES67_PACKET_TIME_MS = 1      # 1ms packet time (AES67 standard)
_SAMPLES_PER_PACKET = _AES67_SAMPLE_RATE * _AES67_PACKET_TIME_MS // 1000  # 48

# RTP constants
_RTP_VERSION = 2
_RTP_PAYLOAD_TYPE = 97         # Dynamic payload type for L24/48000
_RTP_HEADER_SIZE = 12

# SAP constants
_SAP_VERSION = 1
_SAP_INTERVAL_SEC = 30


def _build_sdp(
    session_name: str,
    multicast_addr: str,
    port: int,
    origin_addr: str,
    session_id: int,
) -> str:
    """Build SDP description for an AES67 audio stream."""
    return (
        "v=0\r\n"
        f"o=- {session_id} 1 IN IP4 {origin_addr}\r\n"
        f"s={session_name}\r\n"
        f"c=IN IP4 {multicast_addr}/32\r\n"
        "t=0 0\r\n"
        f"m=audio {port} RTP/AVP {_RTP_PAYLOAD_TYPE}\r\n"
        f"a=rtpmap:{_RTP_PAYLOAD_TYPE} L24/{_AES67_SAMPLE_RATE}/{_AES67_CHANNELS}\r\n"
        f"a=ptime:{_AES67_PACKET_TIME_MS}\r\n"
        "a=sendonly\r\n"
        f"a=clock-domain:PTPv2 0\r\n"
    )


def _build_sap_packet(
    origin_addr: str,
    msg_id_hash: int,
    sdp: str,
    delete: bool = False,
) -> bytes:
    """Build a SAP announcement packet (RFC 2974)."""
    # SAP header: V=1, A=0 (IPv4), R=0, T=delete, auth_len=0, msg_id_hash
    flags = (_SAP_VERSION << 5)
    if delete:
        flags |= 0x04  # T bit = deletion
    header = struct.pack(
        "!BBH4s",
        flags,
        0,                                       # auth_len
        msg_id_hash & 0xFFFF,
        socket.inet_aton(origin_addr),
    )
    payload_type = b"application/sdp\x00"
    return header + payload_type + sdp.encode("utf-8")


def _resample_24k_to_48k(pcm_int16: np.ndarray) -> np.ndarray:
    """Resample 24kHz int16 audio to 48kHz using linear interpolation.

    For speech content, linear interpolation is sufficient and avoids
    the scipy dependency. The 2x ratio makes this straightforward.
    """
    n = len(pcm_int16)
    if n == 0:
        return np.array([], dtype=np.float64)
    # 2x upsampling: insert interpolated sample between each pair
    out = np.empty(n * 2, dtype=np.float64)
    samples = pcm_int16.astype(np.float64)
    out[0::2] = samples
    out[1::2] = np.empty(n, dtype=np.float64)
    # Interpolate: average of current and next sample
    out[1:-1:2] = (samples[:-1] + samples[1:]) / 2.0
    # Last interpolated sample: repeat last value
    out[-1] = samples[-1]
    return out


def _float_to_l24(samples: np.ndarray) -> bytes:
    """Convert float64 samples (int16 range) to L24 big-endian bytes.

    Input samples are in int16 range (-32768..32767). L24 range is -8388608..8388607.
    We scale by 256 (shift left 8 bits) to fill the 24-bit range.
    """
    # Scale from int16 range to int24 range
    scaled = np.clip(samples * 256.0, -8388608, 8388607).astype(np.int32)
    # Pack each sample as 3 bytes big-endian
    result = bytearray(len(scaled) * 3)
    for i, s in enumerate(scaled):
        val = int(s) & 0xFFFFFF  # Mask to 24-bit unsigned representation
        result[i * 3] = (val >> 16) & 0xFF
        result[i * 3 + 1] = (val >> 8) & 0xFF
        result[i * 3 + 2] = val & 0xFF
    return bytes(result)


def _float_to_l24_fast(samples: np.ndarray) -> bytes:
    """Vectorized L24 conversion using numpy — much faster for large buffers."""
    scaled = np.clip(samples * 256.0, -8388608, 8388607).astype(np.int32)
    # Convert negative values to unsigned 24-bit representation
    unsigned = scaled & 0xFFFFFF
    # Extract 3 bytes per sample, big-endian
    b0 = ((unsigned >> 16) & 0xFF).astype(np.uint8)
    b1 = ((unsigned >> 8) & 0xFF).astype(np.uint8)
    b2 = (unsigned & 0xFF).astype(np.uint8)
    # Interleave into [b0, b1, b2, b0, b1, b2, ...]
    interleaved = np.empty(len(samples) * 3, dtype=np.uint8)
    interleaved[0::3] = b0
    interleaved[1::3] = b1
    interleaved[2::3] = b2
    return interleaved.tobytes()


class AES67Sender:
    """Sends audio as an AES67/Dante-compatible RTP multicast stream.

    Provides the same async ``play(pcm_bytes, sample_rate)`` interface as
    :class:`AudioPlayback` so it can be used as a drop-in replacement
    in the translation pipeline.

    AES67 requires a **continuous** packet stream — even during silence.
    A background thread sends silence at 1ms intervals, and ``play()``
    injects real audio into the stream buffer.
    """

    def __init__(
        self,
        stream_name: str = "Church Translation EN",
        multicast_addr: str = "239.69.0.1",
        port: int = 5004,
        ttl: int = 32,
    ):
        self.stream_name = stream_name
        self.multicast_addr = multicast_addr
        self.port = port
        self.ttl = ttl

        self._rtp_seq = 0
        self._rtp_ts = 0
        self._rtp_ssrc = struct.unpack("!I", struct.pack("!I", hash(stream_name) & 0xFFFFFFFF))[0]
        self._session_id = int(time.time())

        self._rtp_sock: Optional[socket.socket] = None
        self._sap_sock: Optional[socket.socket] = None
        self._sap_thread: Optional[threading.Thread] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._running = False
        self._origin_addr = "0.0.0.0"

        # Audio buffer for continuous streaming — lock-protected ring buffer
        self._audio_lock = threading.Lock()
        self._audio_buffer = bytearray()  # L24 bytes ready to send
        # Pre-compute one packet of silence (48 samples × 3 bytes = 144 bytes of zeros)
        self._silence_packet = b'\x00' * (_SAMPLES_PER_PACKET * 3)

    def start(self):
        """Open sockets and begin SAP announcements."""
        if self._running:
            return

        # Detect local IP for SAP origin
        self._origin_addr = self._get_local_ip()

        # RTP multicast socket
        self._rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._rtp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        # Bind multicast to the correct interface (critical on Windows with multiple NICs)
        self._rtp_sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
            socket.inet_aton(self._origin_addr),
        )

        # SAP multicast socket (SAP uses 239.255.255.255:9875)
        self._sap_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sap_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        self._sap_sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
            socket.inet_aton(self._origin_addr),
        )

        self._running = True

        # Start continuous RTP stream thread (sends silence when no audio)
        self._stream_thread = threading.Thread(target=self._continuous_stream_loop, daemon=True)
        self._stream_thread.start()

        # Start SAP announcement thread
        self._sap_thread = threading.Thread(target=self._sap_loop, daemon=True)
        self._sap_thread.start()

        logger.info(
            "AES67 sender started: %s @ %s:%d (SSRC=%08X) [continuous stream]",
            self.stream_name, self.multicast_addr, self.port, self._rtp_ssrc,
        )

    def stop(self):
        """Send SAP deletion and close sockets."""
        if not self._running:
            return

        self._running = False

        # Send SAP deletion announcement
        if self._sap_sock:
            try:
                sdp = _build_sdp(
                    self.stream_name, self.multicast_addr, self.port,
                    self._origin_addr, self._session_id,
                )
                pkt = _build_sap_packet(
                    self._origin_addr, self._session_id, sdp, delete=True,
                )
                self._sap_sock.sendto(pkt, ("239.255.255.255", 9875))
            except OSError:
                pass
            self._sap_sock.close()
            self._sap_sock = None

        if self._rtp_sock:
            self._rtp_sock.close()
            self._rtp_sock = None

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        if self._sap_thread:
            self._sap_thread.join(timeout=2.0)
            self._sap_thread = None

        logger.info("AES67 sender stopped.")

    async def play(self, pcm_bytes: bytes, sample_rate: Optional[int] = None):
        """Queue PCM int16 audio into the continuous AES67 stream.

        Matches the :class:`AudioPlayback` interface. Accepts raw PCM int16 bytes
        (from ElevenLabs/OpenAI TTS), resamples to 48kHz, converts to L24,
        and appends to the stream buffer. The background thread handles
        real-time pacing — this method returns immediately.

        Args:
            pcm_bytes: Raw PCM audio (int16, mono).
            sample_rate: Source sample rate (default 24000 — ElevenLabs PCM).
        """
        if not self._running or not self._rtp_sock:
            logger.warning("AES67 sender not started, skipping.")
            return

        src_rate = sample_rate or 24000
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        if len(audio_int16) == 0:
            return

        # Resample to 48kHz if needed
        if src_rate == 24000:
            audio_48k = _resample_24k_to_48k(audio_int16)
        elif src_rate == _AES67_SAMPLE_RATE:
            audio_48k = audio_int16.astype(np.float64)
        else:
            # Generic resampling via linear interpolation
            ratio = _AES67_SAMPLE_RATE / src_rate
            n_out = int(len(audio_int16) * ratio)
            indices = np.arange(n_out) / ratio
            idx_floor = np.floor(indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, len(audio_int16) - 1)
            frac = indices - idx_floor
            src = audio_int16.astype(np.float64)
            audio_48k = src[idx_floor] * (1 - frac) + src[idx_ceil] * frac

        # Convert to L24 bytes and append to buffer
        l24_bytes = _float_to_l24_fast(audio_48k)

        logger.info(
            "AES67: queued %d samples (%.1fs) into stream buffer",
            len(audio_48k),
            len(audio_48k) / _AES67_SAMPLE_RATE,
        )

        with self._audio_lock:
            self._audio_buffer.extend(l24_bytes)

    def _continuous_stream_loop(self):
        """Send RTP packets continuously at 1ms intervals (runs in daemon thread).

        When audio is in the buffer, sends real audio. Otherwise sends silence.
        This keeps the AES67 stream alive so Dante Controller never drops it.
        """
        bytes_per_packet = _SAMPLES_PER_PACKET * 3  # 48 samples × 3 bytes = 144
        packet_interval = _AES67_PACKET_TIME_MS / 1000.0  # 1ms

        t_start = time.monotonic()
        packet_count = 0

        logger.info("AES67 continuous stream started (1ms packet cadence)")

        while self._running:
            # Get audio from buffer, or use silence
            with self._audio_lock:
                if len(self._audio_buffer) >= bytes_per_packet:
                    payload = bytes(self._audio_buffer[:bytes_per_packet])
                    del self._audio_buffer[:bytes_per_packet]
                else:
                    payload = self._silence_packet

            # Build and send RTP packet
            # Set marker bit on first packet and after silence→audio transitions
            header = self._build_rtp_header(marker=(packet_count == 0))
            packet = header + payload

            try:
                self._rtp_sock.sendto(packet, (self.multicast_addr, self.port))
            except OSError as e:
                if self._running:
                    logger.error("RTP send failed: %s", e)
                break

            self._rtp_seq = (self._rtp_seq + 1) & 0xFFFF
            self._rtp_ts = (self._rtp_ts + _SAMPLES_PER_PACKET) & 0xFFFFFFFF
            packet_count += 1

            # Pace to real-time
            expected_time = t_start + packet_count * packet_interval
            now = time.monotonic()
            if expected_time > now:
                time.sleep(expected_time - now)
            elif now - expected_time > 0.01:
                # We've fallen behind by >10ms — reset timing to avoid burst
                t_start = now
                packet_count = 0

        logger.info("AES67 continuous stream stopped (sent %d packets)", packet_count)

    def _build_rtp_header(self, marker: bool = False) -> bytes:
        """Build a 12-byte RTP header (RFC 3550)."""
        # Byte 0: V=2, P=0, X=0, CC=0
        byte0 = (_RTP_VERSION << 6)
        # Byte 1: M + PT
        byte1 = _RTP_PAYLOAD_TYPE
        if marker:
            byte1 |= 0x80
        return struct.pack(
            "!BBHII",
            byte0,
            byte1,
            self._rtp_seq,
            self._rtp_ts,
            self._rtp_ssrc,
        )

    def _sap_loop(self):
        """Periodically send SAP announcements (runs in daemon thread)."""
        sdp = _build_sdp(
            self.stream_name, self.multicast_addr, self.port,
            self._origin_addr, self._session_id,
        )
        pkt = _build_sap_packet(self._origin_addr, self._session_id, sdp)

        while self._running:
            try:
                self._sap_sock.sendto(pkt, ("239.255.255.255", 9875))
                logger.debug("SAP announcement sent for '%s'", self.stream_name)
            except OSError:
                if not self._running:
                    break
                logger.warning("SAP announcement send failed")

            # Sleep in small intervals so we can exit quickly
            for _ in range(int(_SAP_INTERVAL_SEC * 10)):
                if not self._running:
                    break
                time.sleep(0.1)

    @staticmethod
    def _get_local_ip() -> str:
        """Get local IP address used for outbound connections."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("239.69.0.1", 5004))
            addr = s.getsockname()[0]
            s.close()
            return addr
        except OSError:
            return "127.0.0.1"
