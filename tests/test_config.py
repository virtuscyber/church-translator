from __future__ import annotations

from pathlib import Path

from src.config import Config, OutputConfig, load_config


def test_load_config_parses_yaml_defaults_and_env(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    config_path.write_text(
        "\n".join(
            [
                "audio:",
                "  input_device: 7",
                "translation:",
                "  model: custom-model",
                "synthesis:",
                "  provider: openai",
                "  openai:",
                "    voice: alloy",
                "output:",
                "  mode: both",
                "  stream_name: Sanctuary EN",
                "  multicast_address: 239.69.1.2",
                "  port: 6000",
                "pipeline:",
                "  context_sentences: 4",
            ]
        ),
        encoding="utf-8",
    )
    def fake_load_dotenv():
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-eleven")

    monkeypatch.setattr("src.config.load_dotenv", fake_load_dotenv)

    cfg = load_config(str(config_path))

    assert cfg.audio.input_device == 7
    assert cfg.audio.sample_rate == 48000
    assert cfg.translation.model == "custom-model"
    assert cfg.translation.temperature == 0.3
    assert cfg.synthesis.provider == "openai"
    assert cfg.synthesis.openai.voice == "alloy"
    assert cfg.pipeline.context_sentences == 4
    assert cfg.output.mode == "both"
    assert cfg.output.stream_name == "Sanctuary EN"
    assert cfg.output.multicast_address == "239.69.1.2"
    assert cfg.output.port == 6000
    assert cfg.openai_api_key == "test-openai"
    assert cfg.elevenlabs_api_key == "test-eleven"


def test_load_config_returns_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)

    cfg = load_config(str(tmp_path / "missing.yaml"))

    assert isinstance(cfg, Config)
    assert cfg.audio.sample_rate == 48000
    assert cfg.output.mode == "sounddevice"
    assert cfg.openai_api_key == ""
    assert cfg.elevenlabs_api_key == ""


def test_output_config_defaults():
    output = OutputConfig()

    assert output.mode == "sounddevice"
    assert output.stream_name == "Church Translation EN"
    assert output.multicast_address == "239.69.0.1"
    assert output.port == 5004
    assert output.ttl == 32
