#!/usr/bin/env bash
# finetune_headless.sh — รัน fine-tune 7B แบบไม่ต้องเข้า TTY login เอง
#
# ทำไมต้อง headless: บน GTX 1660 Ti 6GB (usable ~5.61GB) การเทรน 7B ต้องการ VRAM
#   เกือบทั้งใบ — ต้องปิด GUI (gnome-shell ฯลฯ) เพื่อคืน VRAM ก่อนเริ่ม
#
# วิธีทำงาน: สร้าง transient unit (qa-finetune) ผ่าน systemd-run ที่ผูกกับ init
#   ไม่ใช่ session → ปิด GUI ได้โดยงานไม่ดับตาม → เทรนเสร็จเปิด GUI คืนอัตโนมัติ
#   payload จริงอยู่ใน scripts/finetune_headless_inner.sh (ไฟล์จริง ไม่ใช่ inline
#   string — กันบั๊ก systemd expand ${VAR} ที่ทำ gate พังคืน 2026-06-27)
#
# วิธีรัน (จาก terminal ใน GUI ได้เลย):
#   ตรวจความพร้อมก่อน (ไม่ต้อง sudo, ไม่แตะ GUI):
#       PREFLIGHT=1 bash scripts/finetune_headless.sh
#   รันทดสอบสั้น:
#       sudo MAX_STEPS=20 bash scripts/finetune_headless.sh
#   รันจริง + ต่อจาก checkpoint ล่าสุดถ้ามี:
#       sudo MAX_STEPS=300 RESUME=auto bash scripts/finetune_headless.sh
#
# ค่า env ที่ปรับได้: MAX_STEPS, RESUME (auto|<path>), CPU_OFFLOAD (default 1),
#   GPU_BUDGET_GIB (default 2.9 — ค่า validated; >3.2 trainer จะเตือน+clamp),
#   LOG_STEPS (default 1), SAVE_STEPS (default 20), RESTORE_GUI (default 1)
#
# หลังสั่ง: จอจะดับไปหน้า TTY (ปกติ) — ดูสด: Ctrl+Alt+F3 → journalctl -u qa-finetune -f
#   ตอนนี้ journal เห็นทุกบรรทัดแล้ว (step/loss/ETA ทุก ~4 นาที + VRAM ทุก 30 วิ)
#   ถ้าจอเงียบเกิน 5 นาที = ผิดปกติจริง (ไม่ใช่แบบเดิมที่เงียบทั้ง 18 ชม.)
set -euo pipefail

PROJ_DIR="/home/da00/code/Qa-Agent"   # เครื่องนี้ single-user — hardcode ตั้งใจ
RUN_USER="da00"
INNER="${PROJ_DIR}/scripts/finetune_headless_inner.sh"

MAX_STEPS="${MAX_STEPS:-60}"
RESTORE_GUI="${RESTORE_GUI:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"       # 7B บน 6GB ต้องใช้ offload path — default เปิด
GPU_BUDGET_GIB="${GPU_BUDGET_GIB:-2.9}"
RESUME="${RESUME:-}"
LOG_STEPS="${LOG_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-20}"
VRAM_FREE_TARGET_MB="${VRAM_FREE_TARGET_MB:-5700}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${PROJ_DIR}/models/finetune_output/headless_run_${TS}.log"
CKPT_DIR="${PROJ_DIR}/models/finetune_output"
PY="${FINETUNE_PY:-/home/${RUN_USER}/.qa-finetune-env/bin/python}"

mkdir -p "${CKPT_DIR}"

# ── preflight: เช็คทุกอย่างก่อนแตะ GUI — พังตรงไหนบอกตรงนั้น ────────────────
fail=0
if [[ ! -x "${PY}" ]]; then
  echo "✗ finetune python ไม่พบ/รันไม่ได้: ${PY}" >&2; fail=1
fi
if [[ ! -s "${PROJ_DIR}/data/training/train.jsonl" ]]; then
  echo "✗ dataset ว่างหรือไม่พบ: data/training/train.jsonl" >&2; fail=1
fi
if [[ ! -f "${INNER}" ]]; then
  echo "✗ inner script ไม่พบ: ${INNER}" >&2; fail=1
fi
FREE_DISK_GB="$(df --output=avail -BG "${PROJ_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ "${FREE_DISK_GB:-0}" -lt 5 ]]; then
  echo "✗ disk ว่าง ${FREE_DISK_GB:-?}GB (< 5GB) — checkpoint จะเขียนไม่พอ" >&2; fail=1
fi
VRAM_NOW="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
if [[ -z "${VRAM_NOW}" ]]; then
  echo "✗ nvidia-smi อ่านค่าไม่ได้ — driver มีปัญหา ห้ามเริ่มเทรน" >&2; fail=1
fi
# มี checkpoint ค้างแต่ไม่ได้ขอ resume → trainer จะ abort เอง เตือนตั้งแต่ตรงนี้เลย
if [[ -z "${RESUME}" ]] && compgen -G "${CKPT_DIR}/checkpoint-*" > /dev/null; then
  echo "✗ พบ checkpoint ค้างใน ${CKPT_DIR} แต่ไม่ได้ตั้ง RESUME" >&2
  echo "  ต่อจากเดิม:  sudo RESUME=auto MAX_STEPS=${MAX_STEPS} bash scripts/finetune_headless.sh" >&2
  echo "  เริ่มใหม่:    ย้าย checkpoint-* ออกก่อน (เช่นเข้าโฟลเดอร์ archive_*)" >&2
  fail=1
fi
if (( fail != 0 )); then
  echo "preflight ไม่ผ่าน — ยังไม่แตะ GUI, ยกเลิก" >&2
  exit 1
fi

EST_MIN=$(( MAX_STEPS * 4 ))
echo "✓ preflight ผ่าน: dataset $(wc -l < "${PROJ_DIR}/data/training/train.jsonl") บรรทัด | disk ${FREE_DISK_GB}GB | VRAM free ตอนนี้ ${VRAM_NOW}MB (หลังปิด GUI ต้อง >${VRAM_FREE_TARGET_MB}MB)"
echo "  แผน: MAX_STEPS=${MAX_STEPS} (~${EST_MIN} นาที ≈ $(( EST_MIN / 60 )) ชม. ที่ ~4 นาที/step) | RESUME='${RESUME:-<fresh>}' | budget ${GPU_BUDGET_GIB}GiB | checkpoint ทุก ${SAVE_STEPS} steps"

if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  echo "PREFLIGHT=1 — จบแค่ตรวจความพร้อม ไม่ launch"
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "ต้องรันด้วย sudo:  sudo MAX_STEPS=${MAX_STEPS} bash scripts/finetune_headless.sh" >&2
  exit 1
fi

echo "=== launching detached transient unit 'qa-finetune' ==="
echo "    training log -> ${LOG}"

# --collect: เก็บ unit ทิ้งเมื่อจบ
# IgnoreOnIsolate=yes: กันไม่ให้ `systemctl isolate` หยุด unit เทรนของเราเอง
# --expand-environment=no: กัน systemd expand ${VAR} ใน ExecStart (บั๊ก 2026-06-27)
# config ส่งผ่าน --setenv → inner script อ่านจาก env ตามปกติ
systemd-run --collect --unit=qa-finetune --service-type=oneshot \
  --property=IgnoreOnIsolate=yes \
  --expand-environment=no \
  --setenv=PROJ_DIR="${PROJ_DIR}" \
  --setenv=RUN_USER="${RUN_USER}" \
  --setenv=LOG="${LOG}" \
  --setenv=MAX_STEPS="${MAX_STEPS}" \
  --setenv=RESTORE_GUI="${RESTORE_GUI}" \
  --setenv=CPU_OFFLOAD="${CPU_OFFLOAD}" \
  --setenv=GPU_BUDGET_GIB="${GPU_BUDGET_GIB}" \
  --setenv=RESUME="${RESUME}" \
  --setenv=LOG_STEPS="${LOG_STEPS}" \
  --setenv=SAVE_STEPS="${SAVE_STEPS}" \
  --setenv=VRAM_FREE_TARGET_MB="${VRAM_FREE_TARGET_MB}" \
  --setenv=FINETUNE_SEQ_LEN="${FINETUNE_SEQ_LEN:-}" \
  --description="QA Agent 7B fine-tune (headless)" \
  /usr/bin/bash "${INNER}"

cat <<MSG

✅ สั่งงานแล้ว — อีก ~3-5 วิจอ GUI จะดับไปหน้า TTY (ปกติ! การเทรนทำงานเบื้องหลัง)

ดูความคืบหน้า (ตอนนี้ journal เห็นทุกบรรทัดแล้ว ไม่เงียบเหมือนก่อน):
  • กด Ctrl+Alt+F3 → login → journalctl -u qa-finetune -f
  • หรือดูไฟล์:  tail -f ${LOG}
  • ถ้าเงียบเกิน ~5 นาทีค่อยถือว่าผิดปกติ (มี VRAM heartbeat ทุก 30 วิ)

หยุดกลางทางแบบไม่เสียงาน (มี checkpoint ทุก ${SAVE_STEPS} steps):
  sudo systemctl stop qa-finetune     # แล้วรอบหน้าใส่ RESUME=auto เพื่อเทรนต่อ

เทรนเสร็จ: GUI กลับมาเอง (RESTORE_GUI=1) | เช็คผล: ls -lt ${CKPT_DIR}/
MSG
