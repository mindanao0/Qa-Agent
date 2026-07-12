#!/usr/bin/env bash
# export_adapter_gguf.sh — แปลง LoRA adapter → GGUF แล้ว register เข้า Ollama แบบ ADAPTER
#
# ทำไมมีสคริปต์นี้: ทาง export เดิม (wsl2_trainer.py --export-only) ใช้ unsloth
#   โหลดโมเดลทั้งใบขึ้น GPU เพื่อ merge → OOM พลาด ~34MB บนการ์ด 6GB ถ้า GUI เปิดอยู่
#   ทางนี้แปลงแค่ "ตัว adapter" (ไม่กี่ MB) บน CPU ล้วน — ไม่แตะ VRAM เลย แล้วให้
#   Ollama ประกอบร่างเอง (FROM base + ADAPTER lora.gguf)
#
# ข้อจำกัดที่รู้: adapter ถูกเทรนเทียบกับ base ที่ quantize แบบ nf4 แต่ Ollama จะ
#   ทาบมันบน base q4_K_M — mismatch เล็กน้อยเป็นเรื่องปกติของ QLoRA→GGUF workflow
#
# Usage:
#   bash scripts/export_adapter_gguf.sh [ADAPTER_DIR] [OUT_DIR]
#   default: ADAPTER_DIR=models/finetune_output/final, OUT_DIR=models/qwen2.5-coder-finetuned
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER_DIR="${1:-$PROJ_DIR/models/finetune_output/final}"
OUT_DIR="${2:-$PROJ_DIR/models/qwen2.5-coder-finetuned}"
PY="${FINETUNE_PY:-$HOME/.qa-finetune-env/bin/python}"
LLAMA_CPP="${LLAMA_CPP_DIR:-$HOME/.cache/qa-agent/llama.cpp}"
BASE_HF_REPO="Qwen/Qwen2.5-Coder-7B-Instruct"     # base ที่ adapter ถูกเทรนมา
OLLAMA_BASE="${OLLAMA_BASE:-qwen2.5-coder:7b}"     # ต้องเป็นตระกูลเดียวกับ BASE_HF_REPO
MODEL_TAG="${MODEL_TAG:-qa-agent-finetuned}"
ADAPTER_GGUF_NAME="qa-agent-adapter.gguf"

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "✗ adapter ไม่พบ: $ADAPTER_DIR/adapter_model.safetensors" >&2
  echo "  (ต้องเทรนก่อน หรือชี้ path ให้ถูก: bash $0 <adapter_dir> [out_dir])" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# ── 1) llama.cpp scripts (clone ตื้น ๆ ครั้งเดียว — ใช้แค่ convert script + gguf-py) ──
if [[ ! -f "$LLAMA_CPP/convert_lora_to_gguf.py" ]]; then
  echo "[export] llama.cpp ยังไม่มี → clone (ครั้งเดียว) → $LLAMA_CPP"
  mkdir -p "$(dirname "$LLAMA_CPP")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
fi

# ── 2) หา base snapshot จาก HF cache (โหลดไว้แล้วตอนเทรน — ไม่ใช้เน็ต) ─────────
BASE_SNAP="$("$PY" - "$BASE_HF_REPO" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PYEOF
)"
echo "[export] base snapshot: $BASE_SNAP"

# ── 3) แปลง LoRA → GGUF (CPU ล้วน ใช้เวลาไม่กี่วินาที) ────────────────────────
# PYTHONPATH ชี้ gguf-py ของ llama.cpp เอง กัน version mismatch กับ gguf ใน venv
echo "[export] converting adapter → $OUT_DIR/$ADAPTER_GGUF_NAME"
PYTHONPATH="$LLAMA_CPP/gguf-py" "$PY" "$LLAMA_CPP/convert_lora_to_gguf.py" \
  --base "$BASE_SNAP" \
  --outtype f16 \
  --outfile "$OUT_DIR/$ADAPTER_GGUF_NAME" \
  "$ADAPTER_DIR"

# ── 4) Modelfile แบบ ADAPTER ──────────────────────────────────────────────────
cat > "$OUT_DIR/Modelfile.adapter" <<EOF
FROM $OLLAMA_BASE
ADAPTER ./$ADAPTER_GGUF_NAME
SYSTEM """You are an expert QA automation engineer. You write pytest-playwright tests using only semantic locators: get_by_role, get_by_label, get_by_text, get_by_test_id. You never use CSS selectors or XPath. You always use expect() for assertions. Function names must start with test_."""
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
EOF
echo "[export] Modelfile written: $OUT_DIR/Modelfile.adapter"

# REGISTER=0 → แปลงอย่างเดียว ไม่แตะ Ollama (เช่นอยากตรวจไฟล์ก่อน)
if [[ "${REGISTER:-1}" != "1" ]]; then
  echo "[export] REGISTER=0 — ข้ามขั้น register; ทำทีหลัง: (cd $OUT_DIR && ollama create $MODEL_TAG -f Modelfile.adapter)"
  exit 0
fi

# ── 5) register (ทำได้เฉพาะตอน daemon รันอยู่ — ตอน headless daemon ถูกปิด) ────
if ! ollama list >/dev/null 2>&1; then
  cat <<MSG
[export] adapter GGUF พร้อมแล้ว แต่ ollama daemon ยังไม่รัน — register ทีหลังด้วย:
  ollama serve > /tmp/ollama.log 2>&1 &
  sleep 3
  cd "$OUT_DIR" && ollama create $MODEL_TAG -f Modelfile.adapter
  ollama run $MODEL_TAG "Generate a pytest test for a login page"
MSG
  exit 0
fi

if ! ollama show "$OLLAMA_BASE" >/dev/null 2>&1; then
  echo "[export] base '$OLLAMA_BASE' ยังไม่มีใน Ollama → pull (~4.7GB ครั้งเดียว)"
  ollama pull "$OLLAMA_BASE"
fi

(cd "$OUT_DIR" && ollama create "$MODEL_TAG" -f Modelfile.adapter)
echo "[export] DONE → ทดสอบ: ollama run $MODEL_TAG \"Generate a pytest test for a login page\""
