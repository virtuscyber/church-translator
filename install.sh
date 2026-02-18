#!/usr/bin/env bash
set -e

# ── Church Live Translator — Install Script ──────────────────────
# Works on macOS, Ubuntu/Debian, Fedora/RHEL, Arch Linux
# Usage:
#   curl -sL <raw-url>/install.sh | bash
#   ./install.sh              # native install
#   ./install.sh --docker     # Docker install

REPO_URL="https://github.com/virtuscyber/church-translator.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/church-translator}"
BRANCH="main"

# ── Colors ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Detect OS ────────────────────────────────────────────────────
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint) OS="debian" ;;
            fedora|rhel|centos|rocky|alma) OS="fedora" ;;
            arch|manjaro|endeavouros) OS="arch" ;;
            *) OS="unknown" ;;
        esac
    else
        OS="unknown"
    fi
    info "Detected OS: $OS"
}

# ── Install system packages ──────────────────────────────────────
install_deps() {
    case "$OS" in
        macos)
            if ! command -v brew &>/dev/null; then
                warn "Homebrew not found. Installing..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.11 ffmpeg git 2>/dev/null || true
            ;;
        debian)
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3 python3-pip python3-venv ffmpeg git
            ;;
        fedora)
            sudo dnf install -y python3 python3-pip ffmpeg git
            ;;
        arch)
            sudo pacman -Sy --noconfirm python python-pip ffmpeg git
            ;;
        *)
            warn "Unknown OS — please install Python 3.11+, pip, ffmpeg, and git manually."
            ;;
    esac
}

# ── Check prerequisites ──────────────────────────────────────────
check_prereqs() {
    local missing=0
    for cmd in python3 pip3 ffmpeg git; do
        if ! command -v "$cmd" &>/dev/null; then
            # pip3 might be pip
            if [[ "$cmd" == "pip3" ]] && command -v pip &>/dev/null; then
                continue
            fi
            warn "Missing: $cmd"
            missing=1
        fi
    done

    # Check Python version >= 3.11
    local pyver
    pyver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
    if python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        info "Python $pyver ✓"
    else
        warn "Python 3.11+ required (found $pyver)"
        missing=1
    fi

    if [ "$missing" -eq 1 ]; then
        info "Installing missing dependencies..."
        install_deps
    fi
}

# ── Docker install ───────────────────────────────────────────────
docker_install() {
    if ! command -v docker &>/dev/null; then
        fail "Docker not found. Install Docker first: https://docs.docker.com/get-docker/"
    fi

    info "Installing via Docker..."

    if [ -d "$INSTALL_DIR/.git" ]; then
        info "Updating existing repo..."
        cd "$INSTALL_DIR" && git pull origin "$BRANCH"
    else
        info "Cloning repository..."
        git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Create .env if missing
    if [ ! -f .env ] && [ -f .env.example ]; then
        cp .env.example .env
        warn "Created .env from .env.example — edit it or use the web setup wizard."
    fi

    # Create output dir
    mkdir -p output

    info "Building Docker image..."
    docker compose build

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Church Translator — Docker Install Complete!   ${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Start:   cd $INSTALL_DIR && docker compose up -d"
    echo "  Open:    http://localhost:8085"
    echo "  Logs:    docker compose logs -f"
    echo "  Stop:    docker compose down"
    echo ""
    exit 0
}

# ── Native install ───────────────────────────────────────────────
native_install() {
    detect_os
    check_prereqs

    # Clone or update
    if [ -d "$INSTALL_DIR/.git" ]; then
        info "Updating existing repo..."
        cd "$INSTALL_DIR" && git pull origin "$BRANCH"
    else
        info "Cloning repository..."
        git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    # Virtual environment
    if [ ! -d venv ]; then
        info "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate

    info "Installing Python dependencies..."
    pip install -q -r requirements.txt

    # Create .env if missing
    if [ ! -f .env ] && [ -f .env.example ]; then
        cp .env.example .env
        warn "Created .env from .env.example — the web UI will guide you through setup."
    fi

    mkdir -p output

    # Systemd service (Linux only)
    if [[ "$OS" != "macos" ]] && command -v systemctl &>/dev/null; then
        echo ""
        read -rp "Install as systemd service (auto-start on boot)? [y/N] " yn
        if [[ "$yn" =~ ^[Yy]$ ]]; then
            local svc="/etc/systemd/system/church-translator.service"
            sudo cp deploy/church-translator.service "$svc"
            sudo sed -i "s|__USER__|$USER|g" "$svc"
            sudo sed -i "s|__INSTALL_DIR__|$INSTALL_DIR|g" "$svc"
            sudo systemctl daemon-reload
            sudo systemctl enable church-translator
            info "Systemd service installed. Start with: sudo systemctl start church-translator"
        fi
    fi

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Church Translator — Install Complete!          ${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Start:   cd $INSTALL_DIR && source venv/bin/activate && python dashboard/server.py"
    echo "  Open:    http://localhost:8085"
    echo ""
    if [[ "$OS" == "macos" ]]; then
        echo "  💡 macOS auto-start: Add to Login Items or create a launchd plist."
        echo "     See: https://support.apple.com/guide/mac-help/open-items-automatically-when-you-log-in"
        echo ""
    fi
}

# ── Main ─────────────────────────────────────────────────────────
echo ""
echo "🎙️ Church Live Translator — Installer"
echo "────────────────────────────────────────"
echo ""

if [[ "$1" == "--docker" ]]; then
    docker_install
else
    native_install
fi
