#!/usr/bin/env bash
# run_after_finetune.sh — รอให้ fine-tune จบ แล้วค่อยรัน multi-site eval ต่ออัตโนมัติ
#
# แทนที่ run_continue_12h.sh / run_continue_r2.sh เดิม (สองไฟล์นั้นผูก regex กับ
# run เก่าที่จบไปแล้ว และ pgrep ใน r2 ใช้ \| ซึ่งไม่ใช่ alternation ของ ERE —
# เงื่อนไขไม่เคยทำงานจริง) ตัวนี้เช็คสถานะตรง ๆ: unit qa-finetune หรือ process
# ของ trainer ยังอยู่ = ยังเทรนอยู่
#
# ollama บนเครื่องนี้ไม่ใช่ systemd service (is-enabled → not-found) —
# run_finetune_7b.sh pkill มันตอนเริ่มเทรนแล้ว "ไม่มีอะไรปลุกกลับ" สคริปต์นี้
# จึงสตาร์ทเองถ้ายังดับ แล้ว register adapter ล่าสุดเป็น qa-agent-finetuned ก่อน eval
# (รอบ validation 2026-07-12 export สำเร็จแต่ข้าม register เพราะ daemon ดับ)
#
# Usage:
#   bash scripts/run_after_finetune.sh            # รอเทรนจบ → eval 6 ชม. (base model)
#   bash scripts/run_after_finetune.sh 3          # รอเทรนจบ → eval 3 ชม.
#   LLM_MODEL=qa-agent-finetuned bash scripts/run_after_finetune.sh 6
#                                                 # eval ด้วยโมเดล fine-tuned (A/B กับ base)
#
# คำเตือน: ถ้าจะให้ "รอ" ข้ามช่วงเทรน อย่ารันจาก terminal ใน GUI — ตอนเทรนเริ่ม
# GUI ถูก isolate ทิ้งทั้ง session สคริปต์ตายไปด้วย ให้ nohup จาก TTY (Ctrl+Alt+F3)
# หรือง่ายสุด: รอเทรนจบแล้วค่อยรันตรง ๆ (เคสนั้นข้ามการรอ เริ่ม eval ทันที)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EVAL_HOURS="${1:-6}"
MODELFILE_DIR="models/qwen2.5-coder-finetuned"
MODELFILE="Modelfile.adapter"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

finetune_running() {
    systemctl is-active --quiet qa-finetune 2>/dev/null && return 0
    pgrep -f "wsl2_trainer.py" >/dev/null 2>&1 && return 0
    return 1
}

ollama_up() { ollama list >/dev/null 2>&1; }

ensure_ollama() {
    if ollama_up; then
        log "ollama รันอยู่แล้ว"
        return 0
    fi
    log "ollama ยังไม่รัน — สตาร์ทเอง (log → /tmp/ollama.log)"
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    local i
    for i in $(seq 1 30); do
        sleep 1
        if ollama_up; then
            log "ollama พร้อม (รอ ${i}s)"
            return 0
        fi
    done
    log "✗ ollama ไม่ตอบภายใน 30s — ดู /tmp/ollama.log"
    return 1
}

register_adapter() {
    if [[ ! -f "${MODELFILE_DIR}/${MODELFILE}" ]]; then
        log "ไม่พบ ${MODELFILE_DIR}/${MODELFILE} — ข้าม register (ยังไม่เคย export adapter)"
        return 0
    fi
    # cd ก่อน: ADAPTER ใน Modelfile เป็น relative path (./qa-agent-adapter.gguf)
    # ollama create ทับ tag เดิม → idempotent รันซ้ำได้ ได้ adapter ล่าสุดเสมอ
    if (cd "${MODELFILE_DIR}" && ollama create qa-agent-finetuned -f "${MODELFILE}"); then
        log "register แล้ว: qa-agent-finetuned (adapter ล่าสุดจาก ${MODELFILE_DIR})"
    else
        log "✗ ollama create ล้มเหลว (base qwen2.5-coder:7b หายจาก ollama?) — eval ด้วย base ยังไปต่อได้"
    fi
}

if ! finetune_running; then
    log "ไม่พบ fine-tune ที่กำลังรัน — ไปขั้น eval เลย"
else
    log "fine-tune กำลังรันอยู่ — รอ (เช็คทุก 60 วิ; ดูเทรนสด: journalctl -u qa-finetune -f)"
    while finetune_running; do
        sleep 60
    done
    log "fine-tune จบแล้ว — รอ 30 วิให้ GUI กลับมาก่อน"
    sleep 30
fi

ensure_ollama || { log "ยกเลิก eval — ollama ไม่ขึ้น ทุก LLM call จะพังเปล่า ๆ"; exit 1; }
register_adapter

# ถ้าสั่ง eval ด้วยโมเดลที่ไม่มีจริง ให้ตายตรงนี้ ไม่ใช่เผาเวลา eval หลายชั่วโมงฟรี
if [[ -n "${LLM_MODEL:-}" ]] && ! ollama list | grep -q "^${LLM_MODEL}"; then
    log "✗ LLM_MODEL='${LLM_MODEL}' ไม่มีใน ollama (ollama list) — ยกเลิก"
    exit 1
fi

log "เริ่ม multi-site eval (${EVAL_HOURS}h) — model: ${LLM_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M (base ตาม config)}"
exec bash scripts/run_multi_site.sh "$EVAL_HOURS"
