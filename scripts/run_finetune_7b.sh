#!/usr/bin/env bash
# run_finetune_7b.sh — fine-tune the 7B model on collected QA training data.
#
# WHY HEADLESS: the 7B 4-bit base resides ~5.20GB on a 6GB GTX 1660 Ti (5.61GB
# usable). A single bitsandbytes dequant buffer (130MB) at the first training
# step needs more headroom than the GUI desktop leaves free — it OOMs by ~1MB
# while GNOME (gnome-shell ~89MB) + the GUI terminal (ptyxis ~46MB) are running.
# Those can't be killed from inside a GUI-hosted shell (ptyxis hosts the session).
#
# HOW TO RUN (frees the ~135MB the GUI holds):
#   1. Switch to a text console:  Ctrl+Alt+F3  (log in), OR SSH in from another machine.
#   2. Stop the graphical session:  sudo systemctl isolate multi-user.target
#      (this stops GNOME/gnome-shell → frees its VRAM; reverse later with
#       `sudo systemctl isolate graphical.target`).
#   3. cd /home/da00/code/Qa-Agent && bash scripts/run_finetune_7b.sh
#
# Honors the 7B-only mandate (never 1.5b/3b). Dataset + code are already prepared.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"
PY="${FINETUNE_PY:-$HOME/.qa-finetune-env/bin/python}"
DATASET="data/training/train.jsonl"
CKPT_DIR="models/finetune_output"
GGUF_DIR="models/qwen2.5-coder-finetuned"
MAX_STEPS="${MAX_STEPS:-120}"

echo "[finetune] freeing GPU: stopping ollama"
pkill -x ollama 2>/dev/null || true
sleep 2
echo "[finetune] VRAM now:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[finetune] training 7B (max_steps=$MAX_STEPS, dataset=$(wc -l < "$DATASET") examples)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" src/finetune/wsl2_trainer.py \
    --model 7b \
    --dataset "$DATASET" \
    --output  "$CKPT_DIR" \
    --max-steps "$MAX_STEPS"

echo "[finetune] exporting GGUF Q4_K_M → $GGUF_DIR"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" src/finetune/wsl2_trainer.py \
    --model 7b \
    --export-only \
    --checkpoint "$CKPT_DIR/final" \
    --output     "$GGUF_DIR"

echo "[finetune] DONE. Register in Ollama with:  uv run python scripts/create_modelfile.py"
echo "[finetune] Then restart the GUI:  sudo systemctl isolate graphical.target"
