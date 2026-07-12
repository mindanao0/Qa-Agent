#!/usr/bin/env bash
# run_after_finetune.sh — รอให้ fine-tune จบ แล้วค่อยรัน multi-site eval ต่ออัตโนมัติ
#
# แทนที่ run_continue_12h.sh / run_continue_r2.sh เดิม (สองไฟล์นั้นผูก regex กับ
# run เก่าที่จบไปแล้ว และ pgrep ใน r2 ใช้ \| ซึ่งไม่ใช่ alternation ของ ERE —
# เงื่อนไขไม่เคยทำงานจริง) ตัวนี้เช็คสถานะตรง ๆ: unit qa-finetune หรือ process
# ของ trainer ยังอยู่ = ยังเทรนอยู่
#
# Usage:
#   bash scripts/run_after_finetune.sh            # รอเทรนจบ → eval 6 ชม.
#   bash scripts/run_after_finetune.sh 3          # รอเทรนจบ → eval 3 ชม.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EVAL_HOURS="${1:-6}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

finetune_running() {
    systemctl is-active --quiet qa-finetune 2>/dev/null && return 0
    pgrep -f "wsl2_trainer.py" >/dev/null 2>&1 && return 0
    return 1
}

if ! finetune_running; then
    log "ไม่พบ fine-tune ที่กำลังรัน — เริ่ม eval เลย"
else
    log "fine-tune กำลังรันอยู่ — รอ (เช็คทุก 60 วิ; ดูเทรนสด: journalctl -u qa-finetune -f)"
    while finetune_running; do
        sleep 60
    done
    log "fine-tune จบแล้ว — รอ 30 วิให้ GUI/ollama กลับมาก่อน"
    sleep 30
fi

log "เริ่ม multi-site eval (${EVAL_HOURS}h)"
exec bash scripts/run_multi_site.sh "$EVAL_HOURS"
