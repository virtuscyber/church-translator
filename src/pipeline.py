"""Async translation pipeline — orchestrates capture → STT → translate → TTS → playback."""

from __future__ import annotations

import asyncio
import logging
import time

from .audio_capture import AudioCapture
from .audio_playback import AudioPlayback
from .config import Config
from .transcriber import Transcriber
from .translator import Translator
from .synthesizer import Synthesizer

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Main pipeline: captures Ukrainian audio, translates, speaks English."""

    def __init__(self, config: Config):
        self.config = config
        self._running = False

        # Initialize components
        self.capture = AudioCapture(
            device=config.audio.input_device,
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            chunk_duration_sec=config.audio.chunk_duration_sec,
        )

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

        # Output at 24kHz for ElevenLabs PCM, 24kHz for OpenAI PCM
        self.playback = AudioPlayback(
            device=config.audio.output_device,
            sample_rate=24000,
            channels=1,
        )

        # Stats
        self._chunks_processed = 0
        self._total_latency = 0.0

    async def start(self):
        """Start the translation pipeline."""
        logger.info("=" * 60)
        logger.info("Church Live Translation — Starting")
        logger.info("Ukrainian → English (Biblical)")
        logger.info("=" * 60)

        self._running = True
        await self.capture.start()

        try:
            while self._running:
                await self._process_one_chunk()
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the pipeline gracefully."""
        self._running = False
        await self.capture.stop()
        
        if self._chunks_processed > 0:
            avg_latency = self._total_latency / self._chunks_processed
            logger.info(
                "Session stats: %d chunks, avg latency %.1fs",
                self._chunks_processed,
                avg_latency,
            )
        logger.info("Pipeline stopped.")

    async def _process_one_chunk(self):
        """Process a single audio chunk through the full pipeline."""
        t0 = time.monotonic()

        # 1. Capture audio chunk
        wav_bytes = await self.capture.get_chunk()
        if wav_bytes is None:
            return

        t_capture = time.monotonic()

        # 2. Transcribe (Ukrainian audio → Ukrainian text)
        ukrainian_text = await self.transcriber.transcribe(wav_bytes)
        if not ukrainian_text:
            logger.debug("No speech detected in chunk, skipping.")
            return

        t_stt = time.monotonic()

        # 3. Translate (Ukrainian text → English text with biblical style)
        english_text = await self.translator.translate(ukrainian_text)
        if not english_text:
            logger.warning("Translation returned empty, skipping.")
            return

        t_translate = time.monotonic()

        # 4. Synthesize (English text → audio)
        audio_bytes = await self.synthesizer.synthesize(english_text)
        if not audio_bytes:
            logger.warning("Synthesis returned empty, skipping.")
            return

        t_tts = time.monotonic()

        # 5. Play translated audio
        await self.playback.play(audio_bytes)

        t_done = time.monotonic()

        # Stats
        latency = t_done - t0 - self.config.audio.chunk_duration_sec  # Subtract capture wait
        self._chunks_processed += 1
        self._total_latency += max(0, latency)

        logger.info(
            "Chunk #%d | STT: %.1fs | Translate: %.1fs | TTS: %.1fs | Total pipeline: %.1fs",
            self._chunks_processed,
            t_stt - t_capture,
            t_translate - t_stt,
            t_tts - t_translate,
            t_done - t_capture,
        )
