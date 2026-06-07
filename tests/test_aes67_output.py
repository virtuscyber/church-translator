from __future__ import annotations

import struct
import threading
import time

from src.aes67_output import (
    _SAMPLES_PER_PACKET,
    AES67Sender,
    _build_sap_packet,
    _build_sdp,
)
from src.config import load_config


def test_aes67_sender_initialization_and_config_parsing(project_root):
    cfg = load_config(str(project_root / "config.yaml"))
    sender = AES67Sender(
        stream_name=cfg.output.stream_name,
        multicast_addr=cfg.output.multicast_address,
        port=cfg.output.port,
        ttl=cfg.output.ttl,
    )

    assert sender.stream_name == "Church Translation EN"
    assert sender.multicast_addr == "239.69.0.1"
    assert sender.port == 5004
    assert sender.ttl == 32


def test_sdp_contains_expected_aes67_fields():
    sdp = _build_sdp(
        session_name="Church Translation EN",
        multicast_addr="239.69.0.1",
        port=5004,
        origin_addr="192.168.1.10",
        session_id=1234,
    )

    assert "s=Church Translation EN" in sdp
    assert "c=IN IP4 239.69.0.1/32" in sdp
    assert "m=audio 5004 RTP/AVP 97" in sdp
    assert "a=rtpmap:97 L24/48000/1" in sdp
    assert "a=clock-domain:PTPv2 0" in sdp


def test_sap_packet_contains_sdp_payload_and_delete_flag():
    sdp = _build_sdp("Church Translation EN", "239.69.0.1", 5004, "127.0.0.1", 4321)

    normal_packet = _build_sap_packet("127.0.0.1", 4321, sdp)
    delete_packet = _build_sap_packet("127.0.0.1", 4321, sdp, delete=True)

    normal_flags = struct.unpack("!B", normal_packet[:1])[0]
    delete_flags = struct.unpack("!B", delete_packet[:1])[0]

    assert b"application/sdp\x00" in normal_packet
    assert sdp.encode("utf-8") in normal_packet
    assert normal_flags & 0x04 == 0
    assert delete_flags & 0x04 == 0x04


class _FakeSocket:
    """Captures packets passed to sendto instead of touching the network."""

    def __init__(self):
        self.packets: list[bytes] = []

    def sendto(self, data, addr):
        self.packets.append(bytes(data))


def _parse_rtp(packet: bytes):
    byte1, seq, ts = struct.unpack("!BHI", packet[1:8])
    return {
        "marker": bool(byte1 & 0x80),
        "payload_type": byte1 & 0x7F,
        "seq": seq,
        "ts": ts,
        "payload": packet[12:],
    }


def test_continuous_stream_paces_in_realtime_with_correct_rtp_fields():
    """The pacer should emit ~1 packet/ms with drift-free, contiguous RTP."""
    sender = AES67Sender()
    fake = _FakeSocket()
    sender._rtp_sock = fake
    # Pre-load ~5 packets of (non-silent) audio so a talkspurt starts immediately.
    sender._audio_buffer = bytearray(b"\x01\x02\x03" * (_SAMPLES_PER_PACKET * 5))
    sender._running = True

    thread = threading.Thread(target=sender._continuous_stream_loop)
    thread.start()
    time.sleep(0.3)
    sender._running = False
    thread.join(timeout=2.0)

    packets = [_parse_rtp(p) for p in fake.packets]
    assert len(packets) > 0

    # Real-time rate: ~300 packets in 0.3s. Wide bounds tolerate loaded CI.
    assert 120 <= len(packets) <= 600

    # Sequence numbers are contiguous (mod 2^16) and payload type is L24/97.
    for prev, cur in zip(packets, packets[1:]):
        assert cur["seq"] == (prev["seq"] + 1) & 0xFFFF
        assert cur["payload_type"] == 97
        # Timestamp advances exactly one packet (48 samples), no drift/reset.
        assert cur["ts"] == (prev["ts"] + _SAMPLES_PER_PACKET) & 0xFFFFFFFF

    # Marker bit flags the start of the talkspurt: the first audio packet
    # (buffer was non-silent at start) carries it; silence packets do not.
    assert packets[0]["marker"] is True
    assert sum(p["marker"] for p in packets) >= 1


def test_continuous_stream_sends_silence_when_buffer_empty():
    """With no audio queued, the stream still emits silence packets to keep
    the AES67 stream alive (no marker bit on silence)."""
    sender = AES67Sender()
    fake = _FakeSocket()
    sender._rtp_sock = fake
    sender._running = True

    thread = threading.Thread(target=sender._continuous_stream_loop)
    thread.start()
    time.sleep(0.1)
    sender._running = False
    thread.join(timeout=2.0)

    packets = [_parse_rtp(p) for p in fake.packets]
    assert len(packets) > 0
    # All silence: zero payloads and no marker bits.
    assert all(not p["marker"] for p in packets)
    assert all(p["payload"] == b"\x00" * (_SAMPLES_PER_PACKET * 3) for p in packets)
