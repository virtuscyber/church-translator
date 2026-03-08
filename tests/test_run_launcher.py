from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import run


def test_check_python_passes_on_python_311_plus(monkeypatch):
    messages = []

    monkeypatch.setattr(run, "info", messages.append)
    monkeypatch.setattr(run, "fail", lambda message: (_ for _ in ()).throw(AssertionError(message)))

    run.check_python()

    assert any(message.startswith("Python ") for message in messages)


def test_check_ffmpeg_passes_when_installed(monkeypatch):
    messages = []

    monkeypatch.setattr(run, "info", messages.append)
    monkeypatch.setattr(run, "fail", lambda message: (_ for _ in ()).throw(AssertionError(message)))

    run.check_ffmpeg()

    assert "ffmpeg — OK" in messages


def test_ensure_venv_skips_creation_when_python_exists(monkeypatch, tmp_path):
    python_in_venv = tmp_path / "venv-python"
    python_in_venv.write_text("", encoding="utf-8")
    messages = []

    monkeypatch.setattr(run, "get_python_in_venv", lambda: str(python_in_venv))
    monkeypatch.setattr(run, "info", messages.append)

    run.ensure_venv()

    assert messages == ["Virtual environment — OK"]


def test_ensure_venv_creates_environment_when_missing(monkeypatch, tmp_path):
    python_in_venv = tmp_path / "venv-python"
    messages = []
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append((cmd, check, capture_output, text))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run, "get_python_in_venv", lambda: str(python_in_venv))
    monkeypatch.setattr(run, "info", messages.append)
    monkeypatch.setattr(run.subprocess, "run", fake_run)

    run.ensure_venv()

    assert calls == [([run.sys.executable, "-m", "venv", str(run.VENV_DIR)], True, True, True)]
    assert messages == [
        "Creating virtual environment (first time only)...",
        "Virtual environment created",
    ]


def test_ensure_dependencies_skips_install_when_import_check_passes(monkeypatch):
    calls = []
    messages = []

    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        calls.append((cmd, capture_output, text, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run, "get_python_in_venv", lambda: "/tmp/fake-python")
    monkeypatch.setattr(run, "get_pip_in_venv", lambda: "/tmp/fake-pip")
    monkeypatch.setattr(run, "info", messages.append)
    monkeypatch.setattr(run.subprocess, "run", fake_run)

    run.ensure_dependencies()

    assert len(calls) == 1
    assert calls[0][0] == ["/tmp/fake-python", "-c", "import aiohttp; import openai; import sounddevice"]
    assert messages == ["Dependencies — OK"]


def test_ensure_dependencies_installs_when_import_check_fails(monkeypatch):
    calls = []
    messages = []

    def fake_run(cmd, capture_output=False, text=False, check=False, cwd=None, **kwargs):
        calls.append((cmd, capture_output, text, check, cwd))
        if len(calls) == 1:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run, "get_python_in_venv", lambda: "/tmp/fake-python")
    monkeypatch.setattr(run, "get_pip_in_venv", lambda: "/tmp/fake-pip")
    monkeypatch.setattr(run, "info", messages.append)
    monkeypatch.setattr(run.subprocess, "run", fake_run)

    run.ensure_dependencies()

    assert calls[1] == (
        ["/tmp/fake-pip", "install", "-r", str(run.REQUIREMENTS)],
        False,
        False,
        True,
        str(run.PROJECT_ROOT),
    )
    assert messages == [
        "Installing dependencies (this may take a minute)...",
        "Dependencies installed",
    ]
