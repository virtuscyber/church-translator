from __future__ import annotations

import pytest

from src.config import Config


class Recorder:
    def __init__(self):
        self.calls = []


@pytest.mark.asyncio
async def test_translation_pipeline_wires_components_and_processes_audio(monkeypatch):
    import src.pipeline as pipeline_module

    recorder = Recorder()

    class FakeCapture:
        def __init__(self, **kwargs):
            recorder.calls.append(("capture_init", kwargs))

        async def get_chunk(self):
            recorder.calls.append(("get_chunk", None))
            return ("final", b"wav-data")

        async def start(self):
            recorder.calls.append(("capture_start", None))

        async def stop(self):
            recorder.calls.append(("capture_stop", None))

    class FakeTranscriber:
        def __init__(self, **kwargs):
            recorder.calls.append(("transcriber_init", kwargs))

        async def transcribe(self, wav_bytes):
            recorder.calls.append(("transcribe", wav_bytes))
            return "Slava Bohu"

    class FakeTranslator:
        def __init__(self, **kwargs):
            recorder.calls.append(("translator_init", kwargs))

        async def translate(self, text):
            recorder.calls.append(("translate", text))
            return "Glory to God"

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            recorder.calls.append(("synth_init", kwargs))

        async def synthesize(self, text):
            recorder.calls.append(("synthesize", text))
            return b"pcm-data"

    class FakePlayback:
        def __init__(self, **kwargs):
            recorder.calls.append(("playback_init", kwargs))

        async def play(self, audio_bytes):
            recorder.calls.append(("playback_play", audio_bytes))

    class FakeAES67:
        def __init__(self, **kwargs):
            recorder.calls.append(("aes67_init", kwargs))

        async def play(self, audio_bytes):
            recorder.calls.append(("aes67_play", audio_bytes))

        def start(self):
            recorder.calls.append(("aes67_start", None))

        def stop(self):
            recorder.calls.append(("aes67_stop", None))

    monkeypatch.setattr(pipeline_module, "VADAudioCapture", FakeCapture)
    monkeypatch.setattr(pipeline_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline_module, "Translator", FakeTranslator)
    monkeypatch.setattr(pipeline_module, "Synthesizer", FakeSynthesizer)
    monkeypatch.setattr(pipeline_module, "AudioPlayback", FakePlayback)
    monkeypatch.setattr("src.aes67_output.AES67Sender", FakeAES67)

    cfg = Config()
    cfg.pipeline.use_vad = True
    cfg.output.mode = "both"

    pipeline = pipeline_module.TranslationPipeline(cfg)
    await pipeline._process_one_chunk()

    assert pipeline.capture.__class__ is FakeCapture
    assert pipeline.playback.__class__ is FakePlayback
    assert pipeline.aes67.__class__ is FakeAES67
    assert ("transcribe", b"wav-data") in recorder.calls
    assert ("translate", "Slava Bohu") in recorder.calls
    assert ("synthesize", "Glory to God") in recorder.calls
    assert ("playback_play", b"pcm-data") in recorder.calls
    assert ("aes67_play", b"pcm-data") in recorder.calls


def test_streaming_pipeline_uses_fixed_chunk_capture_when_vad_disabled(monkeypatch):
    import src.streaming_pipeline as streaming_module

    recorder = Recorder()

    class FakeCapture:
        def __init__(self, **kwargs):
            recorder.calls.append(("capture_init", kwargs))

    class FakeTranscriber:
        def __init__(self, **kwargs):
            recorder.calls.append(("transcriber_init", kwargs))

    class FakeTranslator:
        def __init__(self, **kwargs):
            recorder.calls.append(("translator_init", kwargs))

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            recorder.calls.append(("synth_init", kwargs))

    class FakeAES67:
        def __init__(self, **kwargs):
            recorder.calls.append(("aes67_init", kwargs))

    monkeypatch.setattr("src.audio_capture.AudioCapture", FakeCapture)
    monkeypatch.setattr(streaming_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(streaming_module, "Translator", FakeTranslator)
    monkeypatch.setattr(streaming_module, "Synthesizer", FakeSynthesizer)
    monkeypatch.setattr("src.aes67_output.AES67Sender", FakeAES67)

    cfg = Config()
    cfg.pipeline.use_vad = False
    cfg.output.mode = "dante"

    pipeline = streaming_module.StreamingPipeline(cfg)

    assert pipeline.capture.__class__ is FakeCapture
    assert pipeline.playback is None
    assert pipeline.aes67.__class__ is FakeAES67
    assert any(name == "capture_init" for name, _ in recorder.calls)
    assert any(name == "transcriber_init" for name, _ in recorder.calls)
    assert any(name == "translator_init" for name, _ in recorder.calls)
    assert any(name == "synth_init" for name, _ in recorder.calls)
    assert any(name == "aes67_init" for name, _ in recorder.calls)


@pytest.mark.asyncio
async def test_streaming_pipeline_capture_worker_unpacks_vad_chunks(monkeypatch):
    import src.streaming_pipeline as streaming_module

    class FakeCapture:
        def __init__(self, **kwargs):
            self.calls = 0

        async def get_chunk(self):
            self.calls += 1
            if self.calls == 1:
                return ("preview", b"preview-wav")
            pipeline._running = False
            return ("final", b"final-wav")

    class FakeTranscriber:
        def __init__(self, **kwargs):
            pass

    class FakeTranslator:
        def __init__(self, **kwargs):
            pass

    class FakeSynthesizer:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("src.vad_capture.VADAudioCapture", FakeCapture)
    monkeypatch.setattr(streaming_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(streaming_module, "Translator", FakeTranslator)
    monkeypatch.setattr(streaming_module, "Synthesizer", FakeSynthesizer)

    cfg = Config()
    cfg.pipeline.use_vad = True

    pipeline = streaming_module.StreamingPipeline(cfg)
    pipeline._running = True

    await pipeline._capture_worker()

    queued = await pipeline._stt_queue.get()
    assert queued.wav_bytes == b"final-wav"
