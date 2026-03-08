from __future__ import annotations

import struct

from src.aes67_output import AES67Sender, _build_sap_packet, _build_sdp
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
