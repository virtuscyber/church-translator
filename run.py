#!/usr/bin/env python3
"""Church Live Translation — Turnkey Launcher.

Double-click (or run from terminal) to start the translation dashboard.
Handles venv creation, dependency installation, and browser launch automatically.
"""

import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
PORT = int(os.environ.get("DASHBOARD_PORT", "8085"))

# ── Pretty output ──────────────────────────────────────────────

def info(msg):
    print(f"  [+] {msg}")

def warn(msg):
    print(f"  [!] {msg}")

def fail(msg):
    print(f"\n  [ERROR] {msg}\n")
    if platform.system() == "Windows":
        input("  Press Enter to close...")
    sys.exit(1)

def banner():
    print()
    print("  ===================================")
    print("    Church Live Translation")
    print("    Real-time sermon translation")
    print("  ===================================")
    print()

# ── Checks ─────────────────────────────────────────────────────

def check_python():
    """Ensure Python >= 3.11."""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        fail(
            f"Python 3.11 or newer is required (you have {v.major}.{v.minor}).\n"
            "  Download it from https://www.python.org/downloads/"
        )
    info(f"Python {v.major}.{v.minor}.{v.micro} — OK")

def check_ffmpeg():
    """Ensure ffmpeg is installed."""
    if shutil.which("ffmpeg"):
        info("ffmpeg — OK")
        return
    system = platform.system()
    if system == "Darwin":
        hint = "Install with:  brew install ffmpeg"
    elif system == "Windows":
        hint = "Download from https://ffmpeg.org/download.html and add to PATH"
    else:
        hint = "Install with:  sudo apt install ffmpeg  (or your package manager)"
    fail(f"ffmpeg is not installed (needed for audio processing).\n  {hint}")

# ── Virtual Environment ────────────────────────────────────────

def get_python_in_venv():
    """Return path to python inside the venv."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")

def get_pip_in_venv():
    """Return path to pip inside the venv."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    return str(VENV_DIR / "bin" / "pip")

def ensure_venv():
    """Create virtual environment if it doesn't exist."""
    python_path = get_python_in_venv()
    if Path(python_path).exists():
        info("Virtual environment — OK")
        return

    info("Creating virtual environment (first time only)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            check=True, capture_output=True, text=True,
        )
        info("Virtual environment created")
    except subprocess.CalledProcessError as e:
        fail(
            "Could not create virtual environment.\n"
            f"  {e.stderr.strip()}"
        )

def ensure_dependencies():
    """Install dependencies if needed."""
    pip = get_pip_in_venv()
    python = get_python_in_venv()

    # Quick check: try importing a key dependency
    result = subprocess.run(
        [python, "-c", "import aiohttp; import openai; import sounddevice"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        info("Dependencies — OK")
        return

    info("Installing dependencies (this may take a minute)...")
    try:
        subprocess.run(
            [pip, "install", "-r", str(REQUIREMENTS)],
            check=True,
            cwd=str(PROJECT_ROOT),
            # Show output so user sees progress
        )
        info("Dependencies installed")
    except subprocess.CalledProcessError:
        fail(
            "Failed to install dependencies.\n"
            "  Try running manually:\n"
            f"  {pip} install -r requirements.txt"
        )

# ── Launch ─────────────────────────────────────────────────────

def open_browser():
    """Open browser after a short delay to let server start."""
    time.sleep(1.5)
    url = f"http://localhost:{PORT}"
    info(f"Opening browser: {url}")
    webbrowser.open(url)

def start_server():
    """Start the dashboard server using the venv Python."""
    python = get_python_in_venv()
    server_script = str(PROJECT_ROOT / "dashboard" / "server.py")

    info(f"Starting dashboard on http://localhost:{PORT}")
    info("Press Ctrl+C to stop\n")

    # Open browser in a background thread
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        proc = subprocess.run(
            [python, server_script],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        print("\n")
        info("Shutting down... goodbye!")
        sys.exit(0)

# ── Main ───────────────────────────────────────────────────────

def main():
    # Change to project root so relative paths work
    os.chdir(PROJECT_ROOT)

    banner()
    check_python()
    check_ffmpeg()
    ensure_venv()
    ensure_dependencies()
    print()
    start_server()

if __name__ == "__main__":
    main()
