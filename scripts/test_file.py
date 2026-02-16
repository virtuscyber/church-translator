#!/usr/bin/env python3
"""Test the translation pipeline with an audio file instead of live capture.

Usage:
    python scripts/test_file.py path/to/ukrainian_sermon.mp3 [--output output.mp3]

Supports: mp3, wav, m4a, ogg, flac, webm (anything ffmpeg/OpenAI accepts).
Chunks the file into segments, runs STT → translate → TTS on each,
and writes the combined English audio + a transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import struct
import sys
import time
import wave
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.transcriber import Transcriber
from src.translator import Translator
from src.synthesizer import Synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def chunk_audio_file(file_path: str, chunk_sec: float = 8.0) -> list[bytes]:
    """Split an audio file into WAV chunks for the transcription API.
    
    Uses pydub for format flexibility, falls back to wave for plain .wav files.
    Returns a list of WAV-formatted byte buffers.
    """
    path = Path(file_path)
    
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(path))
        # Normalize to mono 16-bit 16kHz (good for STT)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    except ImportError:
        # Fallback: only works for .wav files
        if path.suffix.lower() != ".wav":
            logger.error("Install pydub + ffmpeg to handle non-WAV files: pip install pydub")
            sys.exit(1)
        with wave.open(str(path), "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(wf.getnframes())
        # Wrap raw PCM into AudioSegment-like handling
        from pydub import AudioSegment
        audio = AudioSegment(
            data=raw,
            sample_width=params.sampwidth,
            frame_rate=params.framerate,
            channels=params.nchannels,
        ).set_channels(1).set_frame_rate(16000).set_sample_width(2)

    chunk_ms = int(chunk_sec * 1000)
    chunks = []
    
    for start_ms in range(0, len(audio), chunk_ms):
        segment = audio[start_ms : start_ms + chunk_ms]
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        chunks.append(buf.getvalue())
    
    logger.info(
        "Split %s into %d chunks of ~%.1fs each (total %.1fs)",
        path.name, len(chunks), chunk_sec, len(audio) / 1000.0,
    )
    return chunks


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def run_test(input_file: str, output_file: str, chunk_sec: float):
    config = load_config()

    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    # Init components
    transcriber = Transcriber(
        api_key=config.openai_api_key,
        model=config.transcription.model,
        language=config.transcription.language,
    )
    translator = Translator(
        api_key=config.openai_api_key,
        system_prompt=config.translation.system_prompt,
        model=config.translation.model,
        temperature=config.translation.temperature,
        context_sentences=config.pipeline.context_sentences,
    )
    synthesizer = Synthesizer(
        provider=config.synthesis.provider,
        openai_api_key=config.openai_api_key,
        elevenlabs_api_key=config.elevenlabs_api_key,
        elevenlabs_voice_id=config.synthesis.elevenlabs.voice_id,
        elevenlabs_model=config.synthesis.elevenlabs.model,
        elevenlabs_stability=config.synthesis.elevenlabs.stability,
        elevenlabs_similarity=config.synthesis.elevenlabs.similarity,
        openai_model=config.synthesis.openai.model,
        openai_voice=config.synthesis.openai.voice,
    )

    # Chunk the file
    chunks = chunk_audio_file(input_file, chunk_sec)

    all_audio: list[bytes] = []
    transcript_lines: list[str] = []
    total_start = time.time()

    for i, wav_chunk in enumerate(chunks):
        chunk_start = time.time()
        logger.info("━━━ Chunk %d/%d ━━━", i + 1, len(chunks))

        # 1. Transcribe (Ukrainian)
        t0 = time.time()
        uk_text = await transcriber.transcribe(wav_chunk)
        stt_time = time.time() - t0

        if not uk_text:
            logger.warning("  ⏭ No speech detected, skipping chunk")
            continue

        logger.info("  🇺🇦 STT (%.1fs): %s", stt_time, uk_text)

        # 2. Translate to English
        t0 = time.time()
        en_text = await translator.translate(uk_text)
        translate_time = time.time() - t0

        if not en_text:
            logger.warning("  ⏭ Translation empty, skipping")
            continue

        logger.info("  🇬🇧 Translate (%.1fs): %s", translate_time, en_text)

        # 3. Synthesize English audio
        t0 = time.time()
        audio_bytes = await synthesizer.synthesize(en_text)
        tts_time = time.time() - t0

        if audio_bytes:
            all_audio.append(audio_bytes)
            logger.info("  🔊 TTS (%.1fs): %d bytes", tts_time, len(audio_bytes))
        else:
            logger.warning("  ⚠️ TTS failed for this chunk")

        chunk_total = time.time() - chunk_start
        logger.info("  ⏱ Chunk total: %.1fs (STT %.1f + Translate %.1f + TTS %.1f)",
                     chunk_total, stt_time, translate_time, tts_time)

        transcript_lines.append(f"[Chunk {i+1}]")
        transcript_lines.append(f"  UK: {uk_text}")
        transcript_lines.append(f"  EN: {en_text}")
        transcript_lines.append("")

    total_time = time.time() - total_start

    # Write combined audio output
    if all_audio:
        combined_pcm = b"".join(all_audio)
        # PCM from ElevenLabs/OpenAI is 24kHz 16-bit mono
        wav_data = pcm_to_wav(combined_pcm, sample_rate=24000)
        
        output_path = Path(output_file)
        output_path.write_bytes(wav_data)
        logger.info("✅ Output audio: %s (%.1f MB)", output_path, len(wav_data) / 1e6)
    else:
        logger.warning("No audio generated.")

    # Write transcript
    transcript_path = Path(output_file).with_suffix(".txt")
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    logger.info("📝 Transcript: %s", transcript_path)

    # Summary
    logger.info("━━━ Summary ━━━")
    logger.info("  Input: %s", input_file)
    logger.info("  Chunks processed: %d/%d", len(transcript_lines) // 4, len(chunks))
    logger.info("  Total time: %.1fs", total_time)
    logger.info("  Output: %s + %s", output_file, transcript_path)


def main():
    parser = argparse.ArgumentParser(description="Test church translator with an audio file")
    parser.add_argument("input", help="Path to Ukrainian audio file (mp3, wav, m4a, etc.)")
    parser.add_argument("--output", "-o", default="output/test_translation.wav",
                        help="Output audio path (default: output/test_translation.wav)")
    parser.add_argument("--chunk-sec", "-c", type=float, default=8.0,
                        help="Chunk duration in seconds (default: 8.0)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("File not found: %s", args.input)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_test(args.input, args.output, args.chunk_sec))


if __name__ == "__main__":
    main()
