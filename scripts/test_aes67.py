#!/usr/bin/env python3
"""Test AES67 output by streaming a 440Hz sine wave.

Run this script and check Dante Controller to verify the stream
'Church Translation EN' appears as a discoverable AES67 source.

Usage:
    python scripts/test_aes67.py [--duration 10] [--freq 440] [--addr 239.69.0.1]
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aes67_output import AES67Sender


def generate_tone(freq: float, duration: float, sample_rate: int = 24000) -> bytes:
    """Generate a sine wave tone as PCM int16 bytes."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return samples.tobytes()


async def main():
    parser = argparse.ArgumentParser(description="Test AES67 multicast output")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--freq", type=float, default=440.0, help="Tone frequency in Hz")
    parser.add_argument("--addr", type=str, default="239.69.0.1", help="Multicast address")
    parser.add_argument("--port", type=int, default=5004, help="RTP port")
    parser.add_argument("--name", type=str, default="Church Translation EN", help="Stream name")
    args = parser.parse_args()

    sender = AES67Sender(
        stream_name=args.name,
        multicast_addr=args.addr,
        port=args.port,
    )

    print(f"Starting AES67 stream: '{args.name}' @ {args.addr}:{args.port}")
    print(f"Generating {args.freq}Hz tone for {args.duration}s")
    print("Check Dante Controller for the stream. Press Ctrl+C to stop early.\n")

    sender.start()

    try:
        # Generate and stream the tone
        pcm_bytes = generate_tone(args.freq, args.duration)
        print(f"Streaming {len(pcm_bytes)} bytes ({args.duration}s of 24kHz int16 audio)...")
        print("Audio will be resampled to 48kHz L24 for AES67 output.\n")
        await sender.play(pcm_bytes)
        print("Tone playback complete.")

        # Keep SAP announcements running so the stream stays visible
        print("SAP announcements still active. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sender.stop()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
