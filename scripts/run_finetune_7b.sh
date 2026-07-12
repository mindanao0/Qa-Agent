#!/usr/bin/env bash
# run_finetune_7b.sh — fine-tune the 7B model on collected QA training data.
#
# WHY HEADLESS: 7B on the GTX 1660 Ti 6GB (5.61GB usable) needs nearly all VRAM.
# Normally launched via scripts/finetune_headless.sh (stops the GUI, streams the
# log to journald, restores the GUI after). Running this directly works too if
# you freed the VRAM yourself (Ctrl+Alt+F3 → sudo systemctl isolate multi-user.target).
#
# Env knobs (all optional):
#   MAX_STEPS=N        default 60
#   CPU_OFFLOAD=0|1    default 1 — 7B does NOT fit all-on-GPU on this card
#   GPU_BUDGET_GIB=X   default 2.9 (validated; >3.2 the trainer warns + clamps)
#   RESUME=auto|PATH   continue from the latest/explicit checkpoint
#   LOG_STEPS / SAVE_STEPS  forwarded to the trainer via env
#   SKIP_EXPORT=1      stop after training (adapter saved at models/finetune_output/final)
#   USE_LEGACY_EXPORT=1  old unsloth all-on-GPU merge+GGUF (OOMs if GUI is up)
#
# Honors the 7B-only mandate (never 1.5b/3b). Dataset + code are already prepared.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"
PY="${FINETUNE_PY:-$HOME/.qa-finetune-env/bin/python}"
DATASET="data/training/train.jsonl"
CKPT_DIR="models/finetune_output"
GGUF_DIR="models/qwen2.5-coder-finetuned"
MAX_STEPS="${MAX_STEPS:-60}"
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
GPU_BUDGET_GIB="${GPU_BUDGET_GIB:-2.9}"
RESUME="${RESUME:-}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
USE_LEGACY_EXPORT="${USE_LEGACY_EXPORT:-0}"

OFFLOAD_ARGS=()
if [[ "$CPU_OFFLOAD" == "1" ]]; then
  OFFLOAD_ARGS=(--cpu-offload --gpu-budget-gib "$GPU_BUDGET_GIB")
fi
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

echo "[finetune] freeing GPU: stopping ollama (user processes; the headless launcher also stops the system service)"
pkill -x ollama 2>/dev/null || true
sleep 2
echo "[finetune] VRAM now:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "[finetune] training 7B (max_steps=$MAX_STEPS, cpu_offload=$CPU_OFFLOAD, resume='${RESUME:-fresh}', dataset=$(wc -l < "$DATASET") examples)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" src/finetune/wsl2_trainer.py \
    --model 7b \
    --dataset "$DATASET" \
    --output  "$CKPT_DIR" \
    --max-steps "$MAX_STEPS" \
    "${OFFLOAD_ARGS[@]}" \
    "${RESUME_ARGS[@]}"

if [[ "$SKIP_EXPORT" == "1" ]]; then
  echo "[finetune] SKIP_EXPORT=1 — done (adapter at $CKPT_DIR/final; export later with scripts/export_adapter_gguf.sh)"
  exit 0
fi

if [[ "$USE_LEGACY_EXPORT" == "1" ]]; then
  echo "[finetune] exporting GGUF via LEGACY unsloth path (all-on-GPU — OOMs if the GUI is up)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" src/finetune/wsl2_trainer.py \
      --model 7b \
      --export-only \
      --checkpoint "$CKPT_DIR/final" \
      --output     "$GGUF_DIR"
  echo "[finetune] Register in Ollama with:  uv run python scripts/create_modelfile.py"
else
  echo "[finetune] exporting LoRA adapter → GGUF (CPU-only, zero VRAM)"
  bash scripts/export_adapter_gguf.sh "$CKPT_DIR/final" "$GGUF_DIR"
fi

echo "[finetune] DONE. If the GUI is still off:  sudo systemctl isolate graphical.target"
