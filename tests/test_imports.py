from __future__ import annotations

import importlib
import sys

import pytest


MODULES = [
    "src.config",
    "src.pipeline",
    "src.streaming_pipeline",
    "src.aes67_output",
    "src.transcriber",
    "src.translator",
    "src.synthesizer",
    "src.audio_capture",
    "src.audio_playback",
    "src.vad_capture",
    "dashboard.server",
    "run",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_modules_import_without_errors(module_name):
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    assert module is not None
