from __future__ import annotations

import asyncio

import pytest

from src.config import Config


class LivePipelineRecorder:
    def __init__(self):
        self.capture_init = None
        self.capture_started = 0
        self.capture_stopped = 0
        self.transcriber_init = None
        self.transcribe_calls: list[bytes] = []
        self.translator_inits: list[dict] = []
        self.translate_calls: list[tuple[str, str]] = []
        self.synth_inits: list[dict] = []
        self.synthesize_calls: list[tuple[str, str]] = []
        self.playback_init = None
        self.playback_calls: list[bytes] = []
        self.aes67_inits: list[dict] = []
        self.aes67_start_calls: list[str] = []
        self.aes67_play_calls: list[tuple[str, bytes]] = []
        self.aes67_stop_calls: list[str] = []
        self.queue_instances: list[asyncio.Queue] = []
        self.queue_puts: list[tuple[int, bytes | None]] = []
        self.gather_calls: list[list[str | None]] = []
        self.broadcasts: list[dict] = []


def _build_config(output_mode: str = "both") -> Config:
    cfg = Config()
    cfg.output.mode = output_mode
    cfg.output.multicast_address = "239.69.0.1"
    cfg.output.port = 5004
    cfg.translation._system_prompt = "Translate this sermon from Ukrainian to English."
    return cfg


async def _run_live_pipeline_once(
    monkeypatch,
    saved_config: dict,
    *,
    output_mode: str = "both",
    fail_languages: set[str] | None = None,
):
    from dashboard import server

    fail_languages = fail_languages or set()
    recorder = LivePipelineRecorder()
    cfg = _build_config(output_mode=output_mode)
    original_queue = asyncio.Queue
    original_gather = asyncio.gather

    lang_name_map = {lang["name"]: lang["code"] for lang in server.SUPPORTED_LANGUAGES}

    async def fake_broadcast(message):
        recorder.broadcasts.append(message)

    class RecordingQueue(original_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorder.queue_instances.append(self)

        async def put(self, item):
            recorder.queue_puts.append((id(self), item))
            await super().put(item)

    async def recording_gather(*aws, **kwargs):
        recorder.gather_calls.append(
            [
                getattr(getattr(awaitable, "cr_code", None), "co_name", None)
                for awaitable in aws
            ]
        )
        return await original_gather(*aws, **kwargs)

    class FakeCapture:
        def __init__(self, **kwargs):
            recorder.capture_init = kwargs
            self.calls = 0

        async def start(self):
            recorder.capture_started += 1

        async def stop(self):
            recorder.capture_stopped += 1

        async def get_chunk(self):
            self.calls += 1
            if self.calls == 1:
                server.state.live_running = False
                return b"wav-chunk-1"
            return None

    class FakeTranscriber:
        def __init__(self, **kwargs):
            recorder.transcriber_init = kwargs

        async def transcribe(self, wav_bytes):
            recorder.transcribe_calls.append(wav_bytes)
            return "Mir vam"

    class FakeTranslator:
        def __init__(self, **kwargs):
            system_prompt = kwargs["system_prompt"]
            lang_code = next(
                (
                    code
                    for lang_name, code in lang_name_map.items()
                    if f"to {lang_name}" in system_prompt
                ),
                "en",
            )
            self.lang_code = lang_code
            recorder.translator_inits.append({"lang_code": lang_code, **kwargs})

        async def translate(self, text):
            recorder.translate_calls.append((self.lang_code, text))
            await asyncio.sleep(0)
            if self.lang_code in fail_languages:
                raise RuntimeError(f"{self.lang_code} translator failed")
            return f"{self.lang_code}:{text}"

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            recorder.synth_inits.append(kwargs)

        async def synthesize(self, text):
            lang_code = text.split(":", 1)[0]
            recorder.synthesize_calls.append((lang_code, text))
            return f"audio-{lang_code}".encode()

    class FakePlayback:
        def __init__(self, **kwargs):
            recorder.playback_init = kwargs

        async def play(self, audio_bytes):
            recorder.playback_calls.append(audio_bytes)

    class FakeAES67:
        def __init__(self, **kwargs):
            self.stream_name = kwargs["stream_name"]
            recorder.aes67_inits.append(kwargs)

        def start(self):
            recorder.aes67_start_calls.append(self.stream_name)

        def stop(self):
            recorder.aes67_stop_calls.append(self.stream_name)

        async def play(self, audio_bytes):
            recorder.aes67_play_calls.append((self.stream_name, audio_bytes))

    monkeypatch.setattr(server, "broadcast", fake_broadcast)
    monkeypatch.setattr(server, "load_saved_config", lambda: saved_config)
    monkeypatch.setattr(server.asyncio, "Queue", RecordingQueue)
    monkeypatch.setattr(server.asyncio, "gather", recording_gather)
    monkeypatch.setattr("src.config.load_config", lambda: cfg)
    monkeypatch.setattr("src.vad_capture.VADAudioCapture", FakeCapture)
    monkeypatch.setattr("src.transcriber.Transcriber", FakeTranscriber)
    monkeypatch.setattr("src.translator.Translator", FakeTranslator)
    monkeypatch.setattr("src.synthesizer.Synthesizer", FakeSynthesizer)
    monkeypatch.setattr("src.audio_playback.AudioPlayback", FakePlayback)
    monkeypatch.setattr("src.aes67_output.AES67Sender", FakeAES67)

    server.state.live_running = True
    server.state.running = True
    await server._run_live_pipeline()
    return recorder


def test_save_and_load_config_preserve_multi_language_fields():
    from dashboard import server

    server.save_config(
        {
            "multi_language_mode": True,
            "target_languages": ["en", "ru", "pl"],
            "target_language": "en",
            "stt_model": "gpt-4o-transcribe",
        }
    )

    loaded = server.load_saved_config()

    assert loaded["multi_language_mode"] is True
    assert isinstance(loaded["multi_language_mode"], bool)
    assert loaded["target_languages"] == ["en", "ru", "pl"]
    assert all(isinstance(code, str) for code in loaded["target_languages"])
    assert loaded["target_language"] == "en"


def test_save_config_merges_multi_language_update_into_existing_single_target():
    from dashboard import server

    server.save_config({"target_language": "en"})
    server.save_config({"multi_language_mode": False, "target_languages": ["en"]})

    loaded = server.load_saved_config()

    assert loaded == {
        "target_language": "en",
        "multi_language_mode": False,
        "target_languages": ["en"],
    }


@pytest.mark.asyncio
async def test_multi_language_pipeline_fans_out_per_language_and_routes_outputs(monkeypatch):
    recorder = await _run_live_pipeline_once(
        monkeypatch,
        {
            "source_language": "uk",
            "target_language": "en",
            "multi_language_mode": True,
            "target_languages": ["en", "ru", "pl"],
        },
    )

    assert recorder.transcribe_calls == [b"wav-chunk-1"]
    assert [item["lang_code"] for item in recorder.translator_inits] == ["en", "ru", "pl"]
    assert len(recorder.synth_inits) == 3
    assert len({id(queue) for queue in recorder.queue_instances}) == 3

    assert any(call == ["_translate_lang", "_translate_lang", "_translate_lang"] for call in recorder.gather_calls)

    assert [item["stream_name"] for item in recorder.aes67_inits] == [
        "Church Translation EN",
        "Church Translation RU",
        "Church Translation PL",
    ]
    assert [item["multicast_addr"] for item in recorder.aes67_inits] == [
        "239.69.0.1",
        "239.69.0.2",
        "239.69.0.3",
    ]
    assert [item["port"] for item in recorder.aes67_inits] == [5004, 5006, 5008]

    translator_prompts = {item["lang_code"]: item["system_prompt"] for item in recorder.translator_inits}
    assert translator_prompts["en"] == "Translate this sermon from Ukrainian to English."
    assert translator_prompts["ru"] == "Translate this sermon from Ukrainian to Russian."
    assert translator_prompts["pl"] == "Translate this sermon from Ukrainian to Polish."

    non_sentinel_puts = [(queue_id, item) for queue_id, item in recorder.queue_puts if item is not None]
    assert len(non_sentinel_puts) == 3
    assert len({queue_id for queue_id, _ in non_sentinel_puts}) == 3
    assert {item for _, item in non_sentinel_puts} == {b"audio-en", b"audio-ru", b"audio-pl"}

    assert recorder.playback_calls == [b"audio-en"]
    assert recorder.aes67_play_calls == [
        ("Church Translation EN", b"audio-en"),
        ("Church Translation RU", b"audio-ru"),
        ("Church Translation PL", b"audio-pl"),
    ]


@pytest.mark.asyncio
async def test_multi_language_pipeline_isolates_translation_failures(monkeypatch):
    recorder = await _run_live_pipeline_once(
        monkeypatch,
        {
            "source_language": "uk",
            "target_language": "en",
            "multi_language_mode": True,
            "target_languages": ["en", "ru", "pl"],
        },
        fail_languages={"ru"},
    )

    assert recorder.transcribe_calls == [b"wav-chunk-1"]
    assert recorder.translate_calls == [("en", "Mir vam"), ("ru", "Mir vam"), ("pl", "Mir vam")]
    assert recorder.synthesize_calls == [
        ("en", "en:Mir vam"),
        ("pl", "pl:Mir vam"),
    ]
    assert recorder.playback_calls == [b"audio-en"]
    assert recorder.aes67_play_calls == [
        ("Church Translation EN", b"audio-en"),
        ("Church Translation PL", b"audio-pl"),
    ]

    translation_entries = [
        message["entry"]
        for message in recorder.broadcasts
        if message.get("type") == "translation"
    ]
    assert {entry["target_lang"] for entry in translation_entries} == {"en", "pl"}


@pytest.mark.asyncio
async def test_single_language_mode_ignores_multi_targets_and_falls_back_to_primary_target(monkeypatch):
    recorder = await _run_live_pipeline_once(
        monkeypatch,
        {
            "source_language": "uk",
            "target_language": "en",
            "multi_language_mode": False,
            "target_languages": [],
        },
        output_mode="sounddevice",
    )

    assert recorder.transcribe_calls == [b"wav-chunk-1"]
    assert [item["lang_code"] for item in recorder.translator_inits] == ["en"]
    assert len(recorder.synth_inits) == 1
    assert len(recorder.queue_instances) == 1
    assert recorder.aes67_inits == []
    assert recorder.playback_calls == [b"audio-en"]
    assert recorder.synthesize_calls == [("en", "en:Mir vam")]
