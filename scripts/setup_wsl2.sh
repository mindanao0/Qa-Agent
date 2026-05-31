#!/usr/bin/env bash
# ==============================================================================
#  setup_wsl2.sh — Full environment setup for WSL2 Ubuntu 22.04
#
#  Installs:
#    - System packages (curl, wget, git, build-essential, etc.)
#    - Ollama (localhost:11434)
#    - Python 3.11 via deadsnakes PPA
#    - uv package manager
#    - Project Python dependencies (uv sync)
#    - Playwright Chromium browser + system deps
#    - Required ~/.qa-agent/ subdirectories
#
#  Usage (from WSL2 Ubuntu terminal):
#    chmod +x scripts/setup_wsl2.sh
#    ./scripts/setup_wsl2.sh
# ==============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Guard: must run in WSL2 ────────────────────────────────────────────────────
check_wsl2() {
    if ! grep -qi "microsoft" /proc/version 2>/dev/null; then
        warn "This script is designed for WSL2 Ubuntu. Proceeding anyway, but some steps may fail."
    else
        success "Running inside WSL2"
    fi
}

# ── System packages ────────────────────────────────────────────────────────────
install_system_deps() {
    info "Updating apt package index…"
    sudo apt-get update -qq

    info "Installing system dependencies…"
    sudo apt-get install -y --no-install-recommends \
        curl wget git ca-certificates gnupg lsb-release \
        build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
        software-properties-common \
        xvfb libgbm1 libasound2 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
        libxrandr2 libgtk-3-0 libnss3 libx11-xcb1 libxcb-dri3-0 \
        libxtst6 libpango-1.0-0 libpangocairo-1.0-0

    success "System packages installed"
}

# ── Ollama ─────────────────────────────────────────────────────────────────────
install_ollama() {
    if command -v ollama &>/dev/null; then
        success "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
        return 0
    fi
    info "Installing Ollama…"
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed"

    # Start Ollama service in background if not already running
    if ! pgrep -x ollama &>/dev/null; then
        info "Starting Ollama service in background…"
        nohup ollama serve >/tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        sleep 3
        if kill -0 "$OLLAMA_PID" 2>/dev/null; then
            success "Ollama service started (PID $OLLAMA_PID)"
        else
            warn "Ollama may not have started — check /tmp/ollama.log"
        fi
    else
        success "Ollama service already running"
    fi
}

# ── Python 3.11 via deadsnakes PPA ─────────────────────────────────────────────
install_python311() {
    if command -v python3.11 &>/dev/null; then
        success "Python 3.11 already installed: $(python3.11 --version)"
        return 0
    fi
    info "Adding deadsnakes PPA and installing Python 3.11…"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
    success "Python 3.11 installed: $(python3.11 --version)"
}

# ── uv package manager ─────────────────────────────────────────────────────────
install_uv() {
    if command -v uv &>/dev/null; then
        success "uv already installed: $(uv --version)"
        return 0
    fi
    info "Installing uv package manager…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv shell integration
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        success "uv installed: $(uv --version)"
    else
        warn "uv may require a shell restart. Trying ~/.local/bin/uv…"
        if [ -x "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
            success "uv found at ~/.local/bin/uv"
        else
            die "uv installation failed. See https://docs.astral.sh/uv/"
        fi
    fi
}

# ── Project dependencies via uv ────────────────────────────────────────────────
install_project_deps() {
    local PROJECT_DIR
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    info "Installing project dependencies from: $PROJECT_DIR"
    cd "$PROJECT_DIR"

    if [ ! -f pyproject.toml ]; then
        die "pyproject.toml not found in $PROJECT_DIR"
    fi

    # Sync core deps (excludes optional finetune extras to avoid CUDA requirement)
    uv sync --no-dev
    success "Project dependencies installed"
}

# ── Playwright browsers ────────────────────────────────────────────────────────
install_playwright() {
    info "Installing Playwright Chromium browser and system dependencies…"
    local PROJECT_DIR
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$PROJECT_DIR"

    uv run playwright install chromium --with-deps
    success "Playwright Chromium installed"
}

# ── Required directories ───────────────────────────────────────────────────────
create_directories() {
    info "Creating ~/.qa-agent directory structure…"
    local dirs=(
        "$HOME/.qa-agent/vector_db"
        "$HOME/.qa-agent/sessions"
        "$HOME/.qa-agent/datasets"
        "$HOME/.qa-agent/checkpoints"
        "$HOME/.qa-agent/auth"
        "$HOME/.qa-agent/exports"
    )
    for d in "${dirs[@]}"; do
        mkdir -p "$d"
        success "  $d"
    done
}

# ── Verify Ollama is responding ────────────────────────────────────────────────
verify_ollama() {
    info "Verifying Ollama API is reachable at http://localhost:11434…"
    local max_attempts=10
    local attempt=0
    while ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            warn "Ollama did not respond after $max_attempts attempts."
            warn "Start it manually: ollama serve"
            return 1
        fi
        info "  Waiting for Ollama… (attempt $attempt/$max_attempts)"
        sleep 3
    done
    success "Ollama API is responding"
}

# ── Copy .env.example if no .env exists ───────────────────────────────────────
setup_env() {
    local PROJECT_DIR
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        warn ".env created from .env.example — edit it to set real credentials."
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Universal Local AI QA Agent — WSL2 Setup  ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
    echo ""

    check_wsl2
    install_system_deps
    install_ollama
    install_python311
    install_uv
    install_project_deps
    install_playwright
    create_directories
    setup_env
    verify_ollama

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Setup Complete ✓                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env with your application credentials"
    echo "  2. Pull required models:"
    echo "       ./scripts/pull_models.sh"
    echo "  3. Generate a test:"
    echo "       python main.py --mode generate \\"
    echo "         --requirement 'test HRM login' \\"
    echo "         --url http://localhost:3000 --role admin"
}

main "$@"
