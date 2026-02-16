"""Streaming translation pipeline — overlaps all stages for minimum latency.

Architecture:
    Capture → [stt_queue] → STT Worker → [translate_queue] → Translate Worker 
    → [tts_queue] → TTS Worker → [playback_queue] → Playback Worker

Each stage runs as an independent async task. While chunk N is being spoken,
chunk N+1 is already being transcribed, and chunk N+2 is being captured.
This cuts perceived latency from ~5s sequential to ~2-3s pipelined.

The playback queue ensures audio is played in order even though upstream
stages may complete out of order (we use sequence numbers).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .transcriber import Transcriber
from .translator import Translator
from .synthesizer import Synthesizer

logger = logging.getLogger(__name__)


@dataclass
class ChunkState:
    """Tracks a single chunk through the pipeline stages."""
    seq: int
    created_at: float = field(default_factory=time.monotonic)
    wav_bytes: bytes = b""
    ukrainian_text: str = ""
    english_text: str = ""
    audio_bytes: bytes = b""
    # Timing
    t_captured: float = 0.0
    t_stt_done: float = 0.0
    t_translate_done: float = 0.0
    t_tts_done: float = 0.0
    t_played: float = 0.0


class StreamingPipeline:
    """Streaming translation pipeline with overlapped stages.
    
    All 4 processing stages run concurrently:
    1. STT worker: pulls audio chunks, transcribes to Ukrainian text
    2. Translate worker: translates Ukrainian → biblical English  
    3. TTS worker: synthesizes English text to audio
    4. Playback worker: plays audio in sequence order
    
    This means while playing chunk N, we're already transcribing chunk N+2.
    """

    def __init__(self, config: Config):
        self.config = config
        self._running = False
        self._seq = 0

        # Queues between stages
        self._stt_queue: asyncio.Queue[ChunkState] = asyncio.Queue(maxsize=3)
        self._translate_queue: asyncio.Queue[ChunkState] = asyncio.Queue(maxsize=3)
        self._tts_queue: asyncio.Queue[ChunkState] = asyncio.Queue(maxsize=3)
        self._playback_queue: asyncio.Queue[ChunkState] = asyncio.Queue(maxsize=5)

        # Initialize capture (lazy import — requires PortAudio)
        use_vad = getattr(config.pipeline, 'use_vad', True)
        if use_vad:
            from .vad_capture import VADAudioCapture
            self.capture = VADAudioCapture(
                device=config.audio.input_device,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                vad_aggressiveness=getattr(config.pipeline, 'vad_aggressiveness', 2),
                min_chunk_sec=getattr(config.pipeline, 'min_chunk_sec', 3.0),
                max_chunk_sec=getattr(config.pipeline, 'max_chunk_sec', 15.0),
                silence_threshold_sec=getattr(config.pipeline, 'silence_threshold_sec', 0.8),
            )
        else:
            from .audio_capture import AudioCapture
            self.capture = AudioCapture(
                device=config.audio.input_device,
                sample_rate=config.audio.sample_rate,
                channels=config.audio.channels,
                chunk_duration_sec=config.audio.chunk_duration_sec,
            )

        # Processing components
        self.transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=config.transcription.language,
        )
        self.translator = Translator(
            api_key=config.openai_api_key,
            system_prompt=config.translation.system_prompt,
            model=config.translation.model,
            temperature=config.translation.temperature,
            context_sentences=config.pipeline.context_sentences,
        )
        self.synthesizer = Synthesizer(
            provider=config.synthesis.provider,
            openai_api_key=config.openai_api_key,
            elevenlabs_api_key=config.elevenlabs_api_key,
            elevenlabs_voice_id=config.synthesis.elevenlabs.voice_id,
            elevenlabs_model=config.synthesis.elevenlabs.model,
            elevenlabs_stability=config.synthesis.elevenlabs.stability,
            elevenlabs_similarity=config.synthesis.elevenlabs.similarity_boost,
            openai_model=config.synthesis.openai.model,
            openai_voice=config.synthesis.openai.voice,
        )
        from .audio_playback import AudioPlayback
        self.playback = AudioPlayback(
            device=config.audio.output_device,
            sample_rate=24000,
            channels=1,
        )

        # Stats
        self._chunks_processed = 0
        self._total_e2e_latency = 0.0
        self._pipeline_start_time = 0.0

    async def start(self):
        """Start the streaming pipeline with all workers."""
        logger.info("=" * 60)
        logger.info("Church Live Translation — STREAMING MODE")
        logger.info("Ukrainian → English (Biblical)")
        logger.info("Pipeline: Capture → STT → Translate → TTS → Play (overlapped)")
        logger.info("=" * 60)

        self._running = True
        self._pipeline_start_time = time.monotonic()
        await self.capture.start()

        # Launch all workers concurrently
        workers = [
            asyncio.create_task(self._capture_worker(), name="capture"),
            asyncio.create_task(self._stt_worker(), name="stt"),
            asyncio.create_task(self._translate_worker(), name="translate"),
            asyncio.create_task(self._tts_worker(), name="tts"),
            asyncio.create_task(self._playback_worker(), name="playback"),
        ]

        try:
            # Wait for any worker to finish (usually due to stop/error)
            done, pending = await asyncio.wait(
                workers, return_when=asyncio.FIRST_EXCEPTION
            )
            # Re-raise any exceptions
            for task in done:
                if task.exception():
                    raise task.exception()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as e:
            logger.error("Pipeline error: %s", e)
        finally:
            await self.stop()
            # Cancel remaining workers
            for task in workers:
                if not task.done():
                    task.cancel()

    async def stop(self):
        """Stop the streaming pipeline gracefully."""
        self._running = False
        await self.capture.stop()

        # Send poison pills to drain queues
        sentinel = ChunkState(seq=-1)
        for q in [self._stt_queue, self._translate_queue, 
                  self._tts_queue, self._playback_queue]:
            try:
                q.put_nowait(sentinel)
            except asyncio.QueueFull:
                pass

        if self._chunks_processed > 0:
            avg_latency = self._total_e2e_latency / self._chunks_processed
            logger.info(
                "━━━ Session Stats ━━━\n"
                "  Chunks: %d\n"
                "  Avg end-to-end latency: %.1fs\n"
                "  Total runtime: %.0fs",
                self._chunks_processed,
                avg_latency,
                time.monotonic() - self._pipeline_start_time,
            )
        logger.info("Streaming pipeline stopped.")

    # ── Workers ──────────────────────────────────────────────────

    async def _capture_worker(self):
        """Captures audio chunks and feeds them to STT queue."""
        logger.info("🎤 Capture worker started")
        while self._running:
            wav_bytes = await self.capture.get_chunk()
            if wav_bytes is None:
                continue
            
            self._seq += 1
            chunk = ChunkState(seq=self._seq, wav_bytes=wav_bytes)
            chunk.t_captured = time.monotonic()
            
            await self._stt_queue.put(chunk)
            logger.debug("Captured chunk #%d → STT queue", chunk.seq)

    async def _stt_worker(self):
        """Transcribes audio chunks to Ukrainian text."""
        logger.info("🇺🇦 STT worker started")
        while self._running:
            chunk = await self._stt_queue.get()
            if chunk.seq == -1:  # Poison pill
                await self._translate_queue.put(chunk)
                break

            t0 = time.monotonic()
            text = await self.transcriber.transcribe(chunk.wav_bytes)
            chunk.t_stt_done = time.monotonic()
            
            if not text:
                logger.debug("Chunk #%d: no speech detected, dropping", chunk.seq)
                continue

            chunk.ukrainian_text = text
            stt_time = chunk.t_stt_done - t0
            logger.info(
                "🇺🇦 #%d STT (%.1fs): %s",
                chunk.seq, stt_time,
                text[:80] + ("..." if len(text) > 80 else ""),
            )
            
            await self._translate_queue.put(chunk)

    async def _translate_worker(self):
        """Translates Ukrainian text to biblical English."""
        logger.info("🇬🇧 Translate worker started")
        while self._running:
            chunk = await self._translate_queue.get()
            if chunk.seq == -1:
                await self._tts_queue.put(chunk)
                break

            t0 = time.monotonic()
            text = await self.translator.translate(chunk.ukrainian_text)
            chunk.t_translate_done = time.monotonic()
            
            if not text:
                logger.warning("Chunk #%d: translation empty, dropping", chunk.seq)
                continue

            chunk.english_text = text
            translate_time = chunk.t_translate_done - t0
            logger.info(
                "🇬🇧 #%d Translate (%.1fs): %s",
                chunk.seq, translate_time,
                text[:80] + ("..." if len(text) > 80 else ""),
            )
            
            await self._tts_queue.put(chunk)

    async def _tts_worker(self):
        """Synthesizes English text to audio."""
        logger.info("🔊 TTS worker started")
        while self._running:
            chunk = await self._tts_queue.get()
            if chunk.seq == -1:
                await self._playback_queue.put(chunk)
                break

            t0 = time.monotonic()
            audio = await self.synthesizer.synthesize(chunk.english_text)
            chunk.t_tts_done = time.monotonic()
            
            if not audio:
                logger.warning("Chunk #%d: TTS failed, dropping", chunk.seq)
                continue

            chunk.audio_bytes = audio
            tts_time = chunk.t_tts_done - t0
            logger.info(
                "🔊 #%d TTS (%.1fs): %d bytes",
                chunk.seq, tts_time, len(audio),
            )
            
            await self._playback_queue.put(chunk)

    async def _playback_worker(self):
        """Plays audio chunks in order, skipping missing sequences after timeout."""
        logger.info("▶️ Playback worker started")
        next_seq = 1
        buffer: dict[int, ChunkState] = {}
        _SEQ_TIMEOUT = 5.0  # seconds to wait for a missing sequence before skipping
        
        while self._running:
            # If next_seq not in buffer, wait with a timeout then skip gaps
            if next_seq not in buffer:
                try:
                    chunk = await asyncio.wait_for(
                        self._playback_queue.get(), timeout=_SEQ_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    # Skip missing sequences — advance to the lowest buffered seq
                    if buffer:
                        skipped_to = min(buffer)
                        logger.warning(
                            "⏭️ Seq #%d–#%d missing after %.1fs, skipping to #%d",
                            next_seq, skipped_to - 1, _SEQ_TIMEOUT, skipped_to,
                        )
                        next_seq = skipped_to
                    continue
                else:
                    if chunk.seq == -1:
                        break
                    buffer[chunk.seq] = chunk
            else:
                # Drain any additional chunks already available
                try:
                    chunk = self._playback_queue.get_nowait()
                    if chunk.seq == -1:
                        # Play remaining buffered before exiting
                        while next_seq in buffer:
                            c = buffer.pop(next_seq)
                            next_seq += 1
                            if c.audio_bytes:
                                await self.playback.play(c.audio_bytes)
                        break
                    buffer[chunk.seq] = chunk
                except asyncio.QueueEmpty:
                    pass
            
            # Play in order
            while next_seq in buffer:
                c = buffer.pop(next_seq)
                next_seq += 1
                
                if c.audio_bytes:
                    await self.playback.play(c.audio_bytes)
                    c.t_played = time.monotonic()
                    
                    # Stats
                    e2e = c.t_played - c.t_captured
                    pipeline_time = c.t_played - c.t_stt_done  # Excluding capture wait
                    self._chunks_processed += 1
                    self._total_e2e_latency += pipeline_time
                    
                    logger.info(
                        "▶️ #%d PLAYED | Pipeline: %.1fs (STT %.1f + Trans %.1f + TTS %.1f) | E2E: %.1fs",
                        c.seq,
                        pipeline_time,
                        c.t_stt_done - c.t_captured,
                        c.t_translate_done - c.t_stt_done,
                        c.t_tts_done - c.t_translate_done,
                        e2e,
                    )


class StreamingFileTest:
    """Test the streaming pipeline with a file — simulates real-time overlapped processing.
    
    Instead of sequential chunk processing, runs STT/translate/TTS concurrently
    on different chunks, just like the live pipeline would.
    """

    def __init__(self, config: Config):
        self.config = config
        self.transcriber = Transcriber(
            api_key=config.openai_api_key,
            model=config.transcription.model,
            language=config.transcription.language,
        )
        self.translator = Translator(
            api_key=config.openai_api_key,
            system_prompt=config.translation.system_prompt,
            model=config.translation.model,
            temperature=config.translation.temperature,
            context_sentences=config.pipeline.context_sentences,
        )
        self.synthesizer = Synthesizer(
            provider=config.synthesis.provider,
            openai_api_key=config.openai_api_key,
            elevenlabs_api_key=config.elevenlabs_api_key,
            elevenlabs_voice_id=config.synthesis.elevenlabs.voice_id,
            elevenlabs_model=config.synthesis.elevenlabs.model,
            elevenlabs_stability=config.synthesis.elevenlabs.stability,
            elevenlabs_similarity=config.synthesis.elevenlabs.similarity_boost,
            openai_model=config.synthesis.openai.model,
            openai_voice=config.synthesis.openai.voice,
        )

    async def process_chunks(self, wav_chunks: list[bytes]) -> list[ChunkState]:
        """Process chunks with pipelined concurrency.
        
        Launches up to `concurrency` chunks simultaneously across different
        pipeline stages. Returns completed ChunkStates in order.
        """
        results: dict[int, ChunkState] = {}
        semaphore = asyncio.Semaphore(3)  # Max 3 chunks in-flight

        async def process_one(seq: int, wav_bytes: bytes):
            async with semaphore:
                chunk = ChunkState(seq=seq, wav_bytes=wav_bytes)
                chunk.t_captured = time.monotonic()

                # STT
                text = await self.transcriber.transcribe(wav_bytes)
                chunk.t_stt_done = time.monotonic()
                if not text:
                    return
                chunk.ukrainian_text = text
                stt_time = chunk.t_stt_done - chunk.t_captured
                logger.info("🇺🇦 #%d STT (%.1fs): %s", seq, stt_time, text[:80])

                # Translate
                text = await self.translator.translate(chunk.ukrainian_text)
                chunk.t_translate_done = time.monotonic()
                if not text:
                    return
                chunk.english_text = text
                trans_time = chunk.t_translate_done - chunk.t_stt_done
                logger.info("🇬🇧 #%d Translate (%.1fs): %s", seq, trans_time, text[:80])

                # TTS
                audio = await self.synthesizer.synthesize(chunk.english_text)
                chunk.t_tts_done = time.monotonic()
                if not audio:
                    return
                chunk.audio_bytes = audio
                tts_time = chunk.t_tts_done - chunk.t_translate_done
                total = chunk.t_tts_done - chunk.t_captured
                logger.info(
                    "🔊 #%d TTS (%.1fs) | Total: %.1fs (STT %.1f + Trans %.1f + TTS %.1f)",
                    seq, tts_time, total, stt_time, trans_time, tts_time,
                )

                results[seq] = chunk

        # Launch all chunks with bounded concurrency
        tasks = [
            asyncio.create_task(process_one(i + 1, wav))
            for i, wav in enumerate(wav_chunks)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Return in order
        return [results[i] for i in sorted(results.keys())]
