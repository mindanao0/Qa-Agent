#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QLoRA training runner for WSL2 (UPDATE 4).
#
# Prerequisite: run scripts/setup_wsl2_cuda.sh once.
#
# This script:
#   1. Activates the training venv at ~/.qa-finetune-env.
#   2. Verifies the dataset exists and warns if it has < 200 examples.
#   3. Verifies free VRAM ≥ 5.5 GB.
#   4. Runs wsl2_trainer.py to train QLoRA adapters.
#   5. After training, calls wsl2_trainer.py --export-only to write GGUF + Modelfile.
#   6. Copies the GGUF + Modelfile to the Windows-side D:\Code\qa-agent\models\.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

readonly VENV_DIR="${HOME}/.qa-finetune-env"
readonly QA_HOME="${HOME}/qa-finetune"
readonly TRAINER="${QA_HOME}/finetune/wsl2_trainer.py"
readonly DATASET="${QA_HOME}/datasets/synthetic_universal.jsonl"
readonly CKPT_DIR="${HOME}/.qa-agent/checkpoints"
readonly GGUF_DIR="${HOME}/.qa-agent/gguf"
readonly WIN_MODELS_DIR="/mnt/d/Code/qa-agent/models"
readonly MAX_STEPS="${MAX_STEPS:-300}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[run]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ok]${NC}   $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[err]${NC}  $*" >&2; }

# ── Step 1 — Activate venv ───────────────────────────────────────────────────
log 'Step 1/6 — Activating training venv…'
if [[ ! -d "${VENV_DIR}" ]]; then
    err "venv not found at ${VENV_DIR}. Run: bash scripts/setup_wsl2_cuda.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
ok  "venv active: $(python -c 'import sys; print(sys.executable)')"

if [[ ! -f "${TRAINER}" ]]; then
    err "Trainer not found at ${TRAINER}. Re-run setup_wsl2_cuda.sh to stage files."
    exit 1
fi

# ── Step 2 — Dataset check ───────────────────────────────────────────────────
log "Step 2/6 — Verifying dataset at ${DATASET}…"
if [[ ! -f "${DATASET}" ]]; then
    err "Dataset not found: ${DATASET}"
    err "Generate it on Windows: uv run python main.py --mode finetune --step generate"
    exit 1
fi
COUNT=$(wc -l < "${DATASET}" | tr -d ' ')
ok "Dataset has ${COUNT} examples."
if (( COUNT < 200 )); then
    warn "Only ${COUNT} examples (recommended 500+)."
    read -r -p "Continue? (y/n) " answer
    if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
        err 'Aborted by user.'
        exit 1
    fi
fi

# ── Step 3 — Free VRAM check ─────────────────────────────────────────────────
log 'Step 3/6 — Checking free VRAM…'
python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
free_gb, total_gb = free / 1e9, total / 1e9
print(f"Free VRAM: {free_gb:.2f} GB / {total_gb:.1f} GB total")
if free_gb < 5.5:
    print("ERROR: Need 5.5GB free VRAM. Close other GPU apps and retry.")
    sys.exit(1)
PY
ok 'VRAM OK.'

# ── Step 4 — Train ───────────────────────────────────────────────────────────
log "Step 4/6 — Training (max-steps=${MAX_STEPS})…"
mkdir -p "${CKPT_DIR}"
python3 "${TRAINER}" \
    --dataset "${DATASET}" \
    --output  "${CKPT_DIR}" \
    --max-steps "${MAX_STEPS}"
ok "Training complete — checkpoint at ${CKPT_DIR}/final"

# ── Step 5 — Export GGUF ─────────────────────────────────────────────────────
log 'Step 5/6 — Exporting GGUF + Modelfile…'
mkdir -p "${GGUF_DIR}"
python3 "${TRAINER}" \
    --export-only \
    --checkpoint "${CKPT_DIR}/final" \
    --output     "${GGUF_DIR}"
ok "GGUF exported to ${GGUF_DIR}"

# ── Step 6 — Copy to Windows ─────────────────────────────────────────────────
log "Step 6/6 — Copying GGUF + Modelfile to ${WIN_MODELS_DIR}…"
if [[ ! -d "${WIN_MODELS_DIR}" ]]; then
    if ! mkdir -p "${WIN_MODELS_DIR}" 2>/dev/null; then
        warn "Could not create ${WIN_MODELS_DIR}. Create it on Windows first."
    fi
fi
GGUF_SRC="${GGUF_DIR}/qa-agent-coder-q4_k_m.gguf"
MF_SRC="${GGUF_DIR}/Modelfile"
if [[ -f "${GGUF_SRC}" ]]; then
    cp -v "${GGUF_SRC}" "${WIN_MODELS_DIR}/"
else
    err "GGUF file not found at ${GGUF_SRC}"
    exit 1
fi
if [[ -f "${MF_SRC}" ]]; then
    cp -v "${MF_SRC}" "${WIN_MODELS_DIR}/"
else
    warn "Modelfile missing at ${MF_SRC} — register manually."
fi

echo ''
ok 'WSL2 training pipeline complete.'
echo 'GGUF copied to D:\Code\qa-agent\models\ — register in Ollama from Windows PowerShell:'
echo '    powershell -ExecutionPolicy Bypass -File scripts\register_ollama_windows.ps1'
