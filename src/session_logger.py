"""Session transcript logger — persists STT, translation, and TTS text side-by-side.

Each live or file-test session writes a timestamped JSON log file to ``logs/``.
The log captures every pipeline stage for each chunk so you can review
STT accuracy, translation quality, and refinement effects side-by-side
while tuning the system.

Log format (one JSON file per session):
{
  "session_id": "2026-03-14_22-08-39",
  "started_at": "2026-03-14T22:08:39-04:00",
  "ended_at": "...",
  "mode": "live" | "file",
  "config": { source_lang, target_lang, stt_model, ... },
  "chunks": [
    {
      "seq": 1,
      "timestamp": "...",
      "stt":         "Братья и сёстры, мы должны...",
      "translation": "Brothers and sisters, um, we must...",
      "refined":     "Brothers and sisters, we must...",
      "tts_text":    "Brothers and sisters, we must...",
      "latency_sec": 5.2
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"


class SessionLogger:
    """Accumulates chunk data during a session and writes to disk."""

    def __init__(
        self,
        mode: str = "live",
        config: Optional[dict] = None,
    ):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.started_at = datetime.now().astimezone().isoformat()
        self.ended_at: Optional[str] = None
        self.mode = mode
        self.config = config or {}
        self.chunks: list[dict] = []
        self._file_path: Optional[Path] = None

    def log_chunk(
        self,
        seq: int,
        stt: str = "",
        translation: str = "",
        refined: Optional[str] = None,
        tts_text: str = "",
        latency_sec: float = 0.0,
        source_lang: str = "",
        target_lang: str = "",
    ):
        """Record one chunk's pipeline outputs."""
        entry = {
            "seq": seq,
            "timestamp": datetime.now().astimezone().isoformat(),
            "stt": stt,
            "translation": translation,
            "tts_text": tts_text,
            "latency_sec": round(latency_sec, 2),
        }
        # Only include refined field when refinement actually changed the text
        if refined is not None and refined != translation:
            entry["refined"] = refined

        if source_lang:
            entry["source_lang"] = source_lang
        if target_lang:
            entry["target_lang"] = target_lang

        self.chunks.append(entry)

        # Auto-save after each chunk so data isn't lost on crash
        self._save()

    def finish(self):
        """Mark the session as complete and write final log."""
        self.ended_at = datetime.now().astimezone().isoformat()
        self._save()
        logger.info(
            "Session log saved: %s (%d chunks)",
            self._file_path, len(self.chunks),
        )

    def _save(self):
        """Write current state to disk."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if self._file_path is None:
            self._file_path = LOGS_DIR / f"session_{self.session_id}.json"

        data = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "mode": self.mode,
            "config": self.config,
            "chunks": self.chunks,
        }

        self._file_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def file_path(self) -> Optional[Path]:
        return self._file_path


def list_sessions(limit: int = 50) -> list[dict]:
    """Return metadata for recent sessions (newest first)."""
    if not LOGS_DIR.exists():
        return []

    files = sorted(LOGS_DIR.glob("session_*.json"), reverse=True)[:limit]
    sessions = []

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", f.stem),
                "started_at": data.get("started_at", ""),
                "ended_at": data.get("ended_at"),
                "mode": data.get("mode", "unknown"),
                "chunk_count": len(data.get("chunks", [])),
                "config": {
                    "source_lang": data.get("config", {}).get("source_language", ""),
                    "target_lang": data.get("config", {}).get("target_language", ""),
                },
                "filename": f.name,
            })
        except Exception as e:
            logger.warning("Failed to read session log %s: %s", f.name, e)

    return sessions


def get_session(session_id: str) -> Optional[dict]:
    """Load a full session log by ID."""
    path = LOGS_DIR / f"session_{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load session %s: %s", session_id, e)
        return None
