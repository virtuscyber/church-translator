#!/usr/bin/env python3
"""Opt-in LIVE smoke test against the real provider APIs.

Runs one real round-trip per configured provider (translation, TTS, chunked
STT, and streaming STT) so you can confirm your keys, models, and the streaming
sockets work before a service. Providers without a key are skipped — never
failed.

    python scripts/smoke_live.py

Exit code 0 if nothing failed, 1 otherwise. This costs a few cents in API usage.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.live_smoke import run_smoke  # noqa: E402

_ICON = {"pass": "✅", "fail": "❌", "skip": "⚪"}


async def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    print("\n  Live API smoke test — one real round-trip per configured provider\n")

    async def emit(ev):
        phase = ev.get("phase")
        if phase == "start":
            print(f"  … {ev['label']}", end="\r", flush=True)
        elif phase == "result":
            icon = _ICON.get(ev["status"], "•")
            extra = f" — {ev['detail']}" if ev.get("detail") else ""
            secs = f" ({ev['elapsed']}s)" if ev.get("elapsed") else ""
            print(f"  {icon} {ev['label']}{secs}{extra}".ljust(80))
        elif phase == "done":
            print(f"\n  Summary: {ev['passed']} passed · {ev['failed']} failed · {ev['skipped']} skipped\n")

    summary = await run_smoke(emit)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
