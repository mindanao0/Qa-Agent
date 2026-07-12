#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Step 4 — WSL2 fine-tune pipeline (6 GB VRAM safe).
#
# Run inside WSL2 Ubuntu:
#     bash scripts/wsl2_finetune_pipeline.sh
#
# What it does:
#   1. Check GPU visible in WSL2.
#   2. Create/reuse a dedicated Python 3.11 venv with CUDA torch + Unsloth.
#   3. Train with src/finetune/wsl2_trainer.py → models/finetune_output/final/
#   4. Export GGUF Q4_K_M → models/qwen2.5-coder-finetuned/
#   5. Print next steps (register + eval on Windows).
#
# Prereq: NVIDIA driver on the Windows host, WSL2 GPU passthrough enabled.
#   https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
# PROJ_DIR resolves to the repo root (script lives in <root>/scripts/), with an
# env override; no longer hardcoded to the Windows /mnt/d path.
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${HOME}/.qa-finetune-env"
TRAINER="${PROJ_DIR}/src/finetune/wsl2_trainer.py"
DATASET="${PROJ_DIR}/data/training/train.jsonl"
CKPT_DIR="${PROJ_DIR}/models/finetune_output"
GGUF_DIR="${PROJ_DIR}/models/qwen2.5-coder-finetuned"
GGUF_FINAL="${GGUF_DIR}/qwen2.5-coder-finetuned.Q4_K_M.gguf"
MAX_STEPS="${MAX_STEPS:-100}"
# 7B only per project mandate (never the 1.5b/3b presets); override via env if needed.
MODEL_SIZE="${MODEL_SIZE:-7b}"

# ── Helpers ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[pipeline]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC}       $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}     $*"; }
err()  { echo -e "${RED}[err]${NC}      $*" >&2; }

# ── Step 1 — GPU check ───────────────────────────────────────────────────────
log "Step 1/4 — GPU check"
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Enable WSL2 GPU support:"
    err "  https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
ok "GPU visible."

# ── Step 2 — Python venv ─────────────────────────────────────────────────────
log "Step 2/4 — Python env (${VENV_DIR})"

# Install uv (handles Python + venv without OS package requirements)
if ! command -v uv &>/dev/null; then
    log "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# Ensure C compiler present (triton compiles CUDA utils at first import)
if ! command -v gcc &>/dev/null; then
    log "  Installing build-essential (gcc needed by triton)..."
    sudo apt-get install -y build-essential 2>&1 | grep -E "^(Get|Inst|Setting|Processing|done)" || true
    command -v gcc &>/dev/null || { err "gcc still missing after apt-get — run: sudo apt-get install -y build-essential"; exit 1; }
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    rm -rf "${VENV_DIR}"   # remove any broken partial venv
    log "  Installing Python 3.11 via uv..."
    uv python install 3.11
    log "  Creating venv..."
    uv venv "${VENV_DIR}" --python 3.11
    ok "  venv created."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log "  Python: $(python --version)"

# PyTorch with CUDA 12.4 (only if not already a CUDA build)
if ! python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" &>/dev/null; then
    log "  Installing PyTorch cu124..."
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
fi
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
if [[ "${CUDA_OK}" != "True" ]]; then
    err "CUDA not available in Python after install — check driver/toolkit."
    exit 1
fi
ok "  torch $(python -c 'import torch; print(torch.__version__)') | CUDA ${CUDA_OK}"

# Unsloth + training stack
if ! python -c "import unsloth" &>/dev/null; then
    log "  Installing Unsloth + training stack (first time, ~5 min)..."
    uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    uv pip install --no-deps trl peft accelerate bitsandbytes
    uv pip install transformers datasets sentencepiece protobuf pydantic setuptools wheel
    # torchao 0.17.0 (pulled by unsloth) requires torch >= 2.7 API (register_constant).
    # Downgrade to a version compatible with torch 2.6.0.
    uv pip install "torchao==0.9.0"
fi
ok "  Unsloth ready."

# Ensure torchao is compatible with torch 2.6 (0.17.0 requires torch>=2.7 API)
TORCHAO_VER=$(python -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "none")
if [[ "${TORCHAO_VER}" != "0.9.0" ]]; then
    log "  Pinning torchao to 0.9.0 (found ${TORCHAO_VER}, incompatible with torch 2.6)..."
    uv pip install "torchao==0.9.0" --quiet
fi
# Ensure setuptools present (triton needs it at runtime)
python -c "import setuptools" 2>/dev/null || uv pip install setuptools wheel --quiet

# Free VRAM guard
python - <<'PY'
import sys, torch
free, total = torch.cuda.mem_get_info()
free_gb = free / 1e9
print(f"  VRAM: {free_gb:.2f} GB free / {total/1e9:.1f} GB total")
if free_gb < 4.5:
    print(f"ERROR: Need ≥4.5 GB free, have {free_gb:.2f} GB. Close GPU apps.")
    sys.exit(1)
PY
ok "  VRAM sufficient."

# ── Step 3 — Train ───────────────────────────────────────────────────────────
log "Step 3/4 — Training (max_steps=${MAX_STEPS})"
if [[ ! -f "${DATASET}" ]]; then
    err "Dataset not found: ${DATASET}"
    err "Run on Windows first: uv run python scripts/collect_training_data.py"
    exit 1
fi
COUNT=$(wc -l < "${DATASET}")
ok "  Dataset: ${COUNT} examples"
mkdir -p "${CKPT_DIR}"

python "${TRAINER}" \
    --model "${MODEL_SIZE}" \
    --dataset "${DATASET}" \
    --output  "${CKPT_DIR}" \
    --max-steps "${MAX_STEPS}"
ok "Training complete → ${CKPT_DIR}/final"

# ── Step 4 — Export GGUF ─────────────────────────────────────────────────────
log "Step 4/4 — Export GGUF Q4_K_M"
mkdir -p "${GGUF_DIR}"

python "${TRAINER}" \
    --model "${MODEL_SIZE}" \
    --export-only \
    --checkpoint "${CKPT_DIR}/final" \
    --output     "${GGUF_DIR}"

# wsl2_trainer.py writes qa-agent-coder-q4_k_m.gguf; rename to the
# name that create_modelfile.py expects.
WSL_GGUF="${GGUF_DIR}/qa-agent-coder-q4_k_m.gguf"
if [[ -f "${WSL_GGUF}" ]] && [[ "${WSL_GGUF}" != "${GGUF_FINAL}" ]]; then
    mv -v "${WSL_GGUF}" "${GGUF_FINAL}"
fi
# Copy the Modelfile too (wsl2_trainer writes it alongside the GGUF)
WSL_MF="${GGUF_DIR}/Modelfile"
if [[ ! -f "${WSL_MF}" ]]; then
    warn "Modelfile not generated by wsl2_trainer — create_modelfile.py will write it."
fi
ok "GGUF → ${GGUF_FINAL}"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
ok "WSL2 training pipeline complete."
echo ""
echo "  Next — run on Windows (PowerShell / cmd):"
echo "    uv run python scripts/create_modelfile.py   # register qa-agent-finetuned"
echo "    uv run python scripts/eval_finetune.py      # → models/eval_results.json"
