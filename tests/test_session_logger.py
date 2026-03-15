"""Tests for session transcript logger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.session_logger import SessionLogger, list_sessions, get_session, LOGS_DIR


@pytest.fixture(autouse=True)
def clean_logs(tmp_path, monkeypatch):
    """Redirect logs to a temp directory so tests don't pollute the real logs/."""
    test_logs = tmp_path / "logs"
    test_logs.mkdir()
    monkeypatch.setattr("src.session_logger.LOGS_DIR", test_logs)
    return test_logs


def test_session_logger_creates_log_file(clean_logs):
    """Logger creates a JSON file on first chunk."""
    logger = SessionLogger(mode="live", config={"source_language": "uk", "target_language": "en"})
    logger.log_chunk(seq=1, stt="Привіт", translation="Hello", tts_text="Hello", latency_sec=2.5)

    assert logger.file_path is not None
    assert logger.file_path.exists()

    data = json.loads(logger.file_path.read_text())
    assert data["mode"] == "live"
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["stt"] == "Привіт"
    assert data["chunks"][0]["translation"] == "Hello"
    assert data["chunks"][0]["tts_text"] == "Hello"
    assert data["chunks"][0]["latency_sec"] == 2.5


def test_session_logger_multiple_chunks(clean_logs):
    """Multiple chunks accumulate in the same file."""
    logger = SessionLogger(mode="file")
    logger.log_chunk(seq=1, stt="One", translation="One EN", tts_text="One EN")
    logger.log_chunk(seq=2, stt="Two", translation="Two EN", tts_text="Two EN")
    logger.log_chunk(seq=3, stt="Three", translation="Three EN", tts_text="Three EN")

    data = json.loads(logger.file_path.read_text())
    assert len(data["chunks"]) == 3
    assert data["chunks"][2]["seq"] == 3


def test_session_logger_records_refinement(clean_logs):
    """When refined text differs from translation, both are recorded."""
    logger = SessionLogger(mode="live")
    logger.log_chunk(
        seq=1,
        stt="Ну так от",
        translation="Well, so like, you know",
        refined="Brothers and sisters",
        tts_text="Brothers and sisters",
    )

    data = json.loads(logger.file_path.read_text())
    chunk = data["chunks"][0]
    assert chunk["translation"] == "Well, so like, you know"
    assert chunk["refined"] == "Brothers and sisters"
    assert chunk["tts_text"] == "Brothers and sisters"


def test_session_logger_no_refined_when_unchanged(clean_logs):
    """When refined text equals translation, refined field is omitted."""
    logger = SessionLogger(mode="live")
    logger.log_chunk(
        seq=1,
        stt="Слава Богу",
        translation="Glory to God",
        refined="Glory to God",  # Same as translation
        tts_text="Glory to God",
    )

    data = json.loads(logger.file_path.read_text())
    assert "refined" not in data["chunks"][0]


def test_session_logger_finish(clean_logs):
    """finish() sets ended_at timestamp."""
    logger = SessionLogger(mode="live")
    logger.log_chunk(seq=1, stt="Test", translation="Test", tts_text="Test")
    logger.finish()

    data = json.loads(logger.file_path.read_text())
    assert data["ended_at"] is not None


def test_list_sessions(clean_logs):
    """list_sessions returns metadata sorted newest first."""
    # Create two sessions
    l1 = SessionLogger(mode="live")
    l1.log_chunk(seq=1, stt="A", translation="A", tts_text="A")
    l1.finish()

    import time
    time.sleep(1.1)  # Ensure different second-level timestamps for filenames

    l2 = SessionLogger(mode="file")
    l2.log_chunk(seq=1, stt="B", translation="B", tts_text="B")
    l2.log_chunk(seq=2, stt="C", translation="C", tts_text="C")
    l2.finish()

    sessions = list_sessions()
    assert len(sessions) == 2
    # Newest first
    assert sessions[0]["mode"] == "file"
    assert sessions[0]["chunk_count"] == 2
    assert sessions[1]["mode"] == "live"
    assert sessions[1]["chunk_count"] == 1


def test_get_session(clean_logs):
    """get_session loads full session data by ID."""
    logger = SessionLogger(mode="live")
    logger.log_chunk(seq=1, stt="Test", translation="Test", tts_text="Test")
    logger.finish()

    data = get_session(logger.session_id)
    assert data is not None
    assert data["session_id"] == logger.session_id
    assert len(data["chunks"]) == 1


def test_get_session_not_found(clean_logs):
    """get_session returns None for unknown ID."""
    assert get_session("nonexistent_2099-01-01_00-00-00") is None
