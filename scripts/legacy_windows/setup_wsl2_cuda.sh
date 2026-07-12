#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# WSL2 CUDA + Unsloth setup for QLoRA fine-tuning of qa-agent on a 6 GB GPU.
#
# Run inside WSL2 Ubuntu (one-time):
#     bash scripts/setup_wsl2_cuda.sh
#
# This script:
#   1. Verifies WSL2 sees the host GPU (nvidia-smi).
#   2. Installs CUDA toolkit 12.4 from NVIDIA's WSL repo.
#   3. Installs Python 3.11 + uv.
#   4. Creates a dedicated training venv at ~/.qa-finetune-env.
#   5. Installs PyTorch (cu124) + Unsloth + trl/peft/accelerate/bitsandbytes.
#   6. Verifies CUDA is reachable from Python.
#   7. Copies the project's finetune/ directory and dataset into ~/qa-finetune.
#
# It is idempotent — re-running on a partially configured system is safe.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

readonly VENV_DIR="${HOME}/.qa-finetune-env"
readonly QA_HOME="${HOME}/qa-finetune"
readonly PROJECT_DIR="/mnt/d/Code/qa-agent"
readonly DATASETS_SRC="/mnt/c/Users/Da00/.qa-agent/datasets"

# ── Pretty helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC}    $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()  { echo -e "${RED}[err]${NC}   $*" >&2; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { err "Missing command: $1"; exit 1; }
}

# ── Step 1 — Verify WSL2 sees the GPU ────────────────────────────────────────
log 'Step 1/8 — Verifying GPU access from WSL2…'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found in WSL2."
    err "Install NVIDIA driver on the Windows host, then enable WSL GPU support."
    err "See: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute"
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    err "GPU not visible in WSL2. Enable in Windows: nvidia-smi -pm 1"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ok 'GPU reachable.'

# ── Step 2 — Install CUDA toolkit 12.4 (WSL2 Ubuntu) ─────────────────────────
log 'Step 2/8 — Installing CUDA toolkit 12.4 (skip if already installed)…'
if dpkg -s cuda-toolkit-12-4 >/dev/null 2>&1; then
    ok 'cuda-toolkit-12-4 already installed.'
else
    require_cmd wget
    require_cmd sudo
    tmpdir="$(mktemp -d)"
    pushd "${tmpdir}" >/dev/null
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-12-4
    popd >/dev/null
    rm -rf "${tmpdir}"
    ok 'CUDA toolkit 12.4 installed.'
fi

# ── Step 3 — Install Python 3.11 + uv ────────────────────────────────────────
log 'Step 3/8 — Installing Python 3.11 + uv…'
if ! command -v python3.11 >/dev/null 2>&1; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
    ok 'Python 3.11 installed.'
else
    ok 'python3.11 already present.'
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "${HOME}/.bashrc" 2>/dev/null || true
    export PATH="${HOME}/.local/bin:${PATH}"
    ok 'uv installed.'
else
    ok 'uv already present.'
fi

# ── Step 4 — Create training venv ────────────────────────────────────────────
log "Step 4/8 — Creating venv at ${VENV_DIR}…"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3.11 -m venv "${VENV_DIR}"
    ok 'venv created.'
else
    ok 'venv already exists.'
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null
ok "Active venv: $(python -c 'import sys; print(sys.executable)')"

# ── Step 5 — PyTorch (cu124) ─────────────────────────────────────────────────
log 'Step 5/8 — Installing PyTorch 2.4.0 (cu124)…'
if ! python -c 'import torch' >/dev/null 2>&1; then
    pip install --quiet torch==2.4.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    ok 'torch installed.'
else
    ok 'torch already installed — skipping.'
fi

# ── Step 6 — Unsloth + TRL + bitsandbytes ────────────────────────────────────
log 'Step 6/8 — Installing Unsloth + helpers…'
if ! python -c 'import unsloth' >/dev/null 2>&1; then
    pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --quiet --no-deps trl peft accelerate bitsandbytes
    pip install --quiet transformers datasets sentencepiece protobuf
    ok 'Unsloth + TRL installed.'
else
    ok 'unsloth already installed — skipping.'
fi

# ── Step 7 — Verify CUDA from Python ─────────────────────────────────────────
log 'Step 7/8 — Verifying CUDA is reachable from Python…'
python - <<'PY'
import torch
ok = torch.cuda.is_available()
if not ok:
    raise SystemExit("CUDA: False — driver / cuda-toolkit / torch mismatch")
props = torch.cuda.get_device_properties(0)
print(f"CUDA: True, Device: {props.name}, VRAM: {props.total_memory / 1e9:.1f}GB")
PY
ok 'CUDA reachable.'

# ── Step 8 — Stage project files into WSL2 home ──────────────────────────────
log "Step 8/8 — Staging project files into ${QA_HOME}…"
mkdir -p "${QA_HOME}/finetune" "${QA_HOME}/datasets" "${HOME}/.qa-agent/checkpoints" "${HOME}/.qa-agent/gguf"

if [[ -d "${PROJECT_DIR}/src/finetune" ]]; then
    cp -r "${PROJECT_DIR}/src/finetune/." "${QA_HOME}/finetune/"
    ok "finetune sources copied -> ${QA_HOME}/finetune"
else
    warn "Project finetune dir not found at ${PROJECT_DIR}/src/finetune — skipping."
fi

if [[ -d "${DATASETS_SRC}" ]]; then
    cp -r "${DATASETS_SRC}/." "${QA_HOME}/datasets/"
    ok "Datasets copied  -> ${QA_HOME}/datasets"
else
    warn "Dataset dir not found at ${DATASETS_SRC}."
    warn "Run 'uv run python main.py --mode finetune --step generate' on Windows first."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ''
ok 'WSL2 CUDA setup complete.'
echo "  Run training with: bash scripts/run_training_wsl2.sh"
