"""Church Live Translation — Main entry point."""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import load_config
from .pipeline import TranslationPipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def validate_config(config):
    """Validate configuration before starting."""
    errors = []

    if not config.openai_api_key:
        errors.append("OPENAI_API_KEY not set in .env")

    if config.synthesis.provider == "elevenlabs" and not config.elevenlabs_api_key:
        errors.append("ELEVENLABS_API_KEY not set in .env (required for ElevenLabs provider)")

    if errors:
        for e in errors:
            logging.error("Config error: %s", e)
        sys.exit(1)


def main():
    setup_logging()
    logger = logging.getLogger("church-translator")

    logger.info("Loading configuration...")
    config = load_config()
    validate_config(config)

    logger.info("Configuration loaded:")
    logger.info("  Input device:  %s", config.audio.input_device or "system default")
    logger.info("  Output device: %s", config.audio.output_device or "system default")
    logger.info("  STT model:     %s", config.transcription.model)
    logger.info("  Translation:   %s", config.translation.model)
    logger.info("  TTS provider:  %s", config.synthesis.provider)
    logger.info("  Chunk size:    %.1fs", config.audio.chunk_duration_sec)

    pipeline = TranslationPipeline(config)

    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
