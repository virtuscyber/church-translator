"""Configuration loader for church-translator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class AudioConfig:
    input_device: Optional[str | int] = None
    output_device: Optional[str | int] = None
    sample_rate: int = 48000
    channels: int = 1
    chunk_duration_sec: float = 8.0


@dataclass
class TranscriptionConfig:
    model: str = "gpt-4o-transcribe"
    language: str = "uk"


@dataclass
class TranslationConfig:
    model: str = "gpt-4o"
    temperature: float = 0.3
    prompt_file: str = "prompts/biblical_translator.txt"
    _system_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        if not self._system_prompt:
            prompt_path = Path(__file__).parent.parent / self.prompt_file
            self._system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        return self._system_prompt


@dataclass
class ElevenLabsConfig:
    model: str = "eleven_turbo_v2_5"
    voice_id: str = "pNInz6obpgDQGcFmaJgB"
    stability: float = 0.7
    similarity_boost: float = 0.8


@dataclass
class OpenAITTSConfig:
    model: str = "gpt-4o-mini-tts"
    voice: str = "onyx"


@dataclass
class SynthesisConfig:
    provider: str = "elevenlabs"
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    openai: OpenAITTSConfig = field(default_factory=OpenAITTSConfig)


@dataclass
class PipelineConfig:
    overlap_sec: float = 1.0
    buffer_silence_sec: float = 2.0
    context_sentences: int = 2
    # VAD settings (Phase 2)
    use_vad: bool = True
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive filtering
    min_chunk_sec: float = 3.0   # Minimum speech segment duration
    max_chunk_sec: float = 15.0  # Maximum before force-split
    silence_threshold_sec: float = 0.8  # Silence duration to trigger split


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    openai_api_key: str = ""
    elevenlabs_api_key: str = ""


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and environment variables."""
    load_dotenv()

    cfg = Config()

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "audio" in raw:
            for k, v in raw["audio"].items():
                if hasattr(cfg.audio, k):
                    setattr(cfg.audio, k, v)

        if "transcription" in raw:
            for k, v in raw["transcription"].items():
                if hasattr(cfg.transcription, k):
                    setattr(cfg.transcription, k, v)

        if "translation" in raw:
            for k, v in raw["translation"].items():
                if hasattr(cfg.translation, k):
                    setattr(cfg.translation, k, v)

        if "synthesis" in raw:
            s = raw["synthesis"]
            cfg.synthesis.provider = s.get("provider", cfg.synthesis.provider)
            if "elevenlabs" in s:
                for k, v in s["elevenlabs"].items():
                    if hasattr(cfg.synthesis.elevenlabs, k):
                        setattr(cfg.synthesis.elevenlabs, k, v)
            if "openai" in s:
                for k, v in s["openai"].items():
                    if hasattr(cfg.synthesis.openai, k):
                        setattr(cfg.synthesis.openai, k, v)

        if "pipeline" in raw:
            for k, v in raw["pipeline"].items():
                if hasattr(cfg.pipeline, k):
                    setattr(cfg.pipeline, k, v)

    cfg.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    cfg.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    return cfg
