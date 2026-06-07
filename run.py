#!/usr/bin/env python3
"""Church Live Translation — Turnkey Launcher.

Double-click (or run from terminal) to start the translation dashboard.
Handles venv creation, dependency installation, and browser launch automatically.
"""

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / "venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
PORT = int(os.environ.get("DASHBOARD_PORT", "8085"))
PORT_FORCED = "DASHBOARD_PORT" in os.environ  # user pinned a specific port
HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
# Probe loopback even if bound to 0.0.0.0 — that's where a local instance lives.
PROBE_HOST = "127.0.0.1" if HOST in ("0.0.0.0", "::", "") else HOST

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

# ── Port handling ──────────────────────────────────────────────

def port_in_use(port):
    """True if something is already listening on PROBE_HOST:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((PROBE_HOST, port)) == 0

def is_our_dashboard(port):
    """True if the thing on this port answers like our dashboard's health API."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{PROBE_HOST}:{port}/api/health", timeout=1.5) as r:
            return "healthy" in json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return False

def resolve_port():
    """Pick the port to run on, working around an already-occupied one.

    Returns (port, status) where status is one of:
      "free"    — preferred port is available
      "ours"    — our dashboard is already running there (just open it)
      "alt"     — preferred port busy; found a free one nearby
      "blocked" — busy and we shouldn't/can't move (forced port, or none free)
    """
    if not port_in_use(PORT):
        return PORT, "free"
    if is_our_dashboard(PORT):
        return PORT, "ours"
    if PORT_FORCED:
        return PORT, "blocked"
    for candidate in range(PORT + 1, PORT + 21):
        if not port_in_use(candidate):
            return candidate, "alt"
    return PORT, "blocked"

# ── Launch ─────────────────────────────────────────────────────

def open_browser(port):
    """Open browser after a short delay to let server start."""
    time.sleep(1.5)
    url = f"http://localhost:{port}"
    info(f"Opening browser: {url}")
    webbrowser.open(url)

def start_server(port):
    """Start the dashboard server using the venv Python on the given port."""
    python = get_python_in_venv()
    server_script = str(PROJECT_ROOT / "dashboard" / "server.py")

    info(f"Starting dashboard on http://localhost:{port}")
    info("Press Ctrl+C to stop\n")

    # Pass the resolved port to the server so both agree on it.
    env = {**os.environ, "DASHBOARD_PORT": str(port)}

    # Open browser in a background thread
    import threading
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    try:
        subprocess.run(
            [python, server_script],
            cwd=str(PROJECT_ROOT),
            env=env,
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

    port, status = resolve_port()
    if status == "ours":
        # A dashboard is already running here — just open it, don't start a 2nd.
        info(f"Dashboard is already running on http://localhost:{PORT} — opening it.")
        info("(To run a fresh copy, close the other one first, or set DASHBOARD_PORT.)")
        open_browser(PORT)
        return
    if status == "blocked":
        why = (
            f"you set DASHBOARD_PORT={PORT}, but it's already taken"
            if PORT_FORCED else
            f"port {PORT} and the next 20 ports are all in use"
        )
        fail(
            f"Can't start the dashboard — {why}.\n"
            "  Another program (or a leftover copy of this app) is using the port.\n"
            "  Close it and try again, or pick a free port:\n"
            "      Windows:  set DASHBOARD_PORT=8090 && python run.py\n"
            "      macOS/Linux:  DASHBOARD_PORT=8090 python run.py"
        )
    if status == "alt":
        warn(f"Port {PORT} is already in use — starting on {port} instead.")

    start_server(port)

if __name__ == "__main__":
    main()
