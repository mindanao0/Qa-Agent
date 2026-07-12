#!/usr/bin/env bash
# finetune_headless_inner.sh — payload ที่รันข้างใน transient unit (root)
# ห้ามเรียกไฟล์นี้ตรง ๆ — ให้ scripts/finetune_headless.sh เป็นคน launch ผ่าน systemd-run
#
# ทำไมต้องเป็น "ไฟล์จริง" ไม่ใช่ inline string ผ่าน `bash -c`:
#   systemd จะ expand ${VAR} ใน ExecStart ของ transient unit เองก่อน bash ได้เห็น
#   (พิสูจน์จาก journal คืน 2026-06-27: "Referenced but unset environment variable
#   evaluates to an empty string: FREE, READY" → loop เช็ค VRAM ตาบอด + gate พังแบบ
#   fail-open แล้วหลุดไปเทรนทั้งที่ตรวจอะไรไม่สำเร็จเลย) สคริปต์ที่เป็นไฟล์จริง
#   systemd ไม่ยุ่งกับ $ ข้างใน และ launcher ยังส่ง --expand-environment=no ซ้ำอีกชั้น
set -uo pipefail

# ── config มาจาก systemd-run --setenv (launcher เป็นคนตั้ง) ─────────────────
PROJ_DIR="${PROJ_DIR:?PROJ_DIR not set — launch via scripts/finetune_headless.sh}"
RUN_USER="${RUN_USER:?RUN_USER not set}"
LOG="${LOG:?LOG not set}"
MAX_STEPS="${MAX_STEPS:-60}"
RESTORE_GUI="${RESTORE_GUI:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-1}"
GPU_BUDGET_GIB="${GPU_BUDGET_GIB:-2.9}"
RESUME="${RESUME:-}"
LOG_STEPS="${LOG_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-20}"
VRAM_FREE_TARGET_MB="${VRAM_FREE_TARGET_MB:-5700}"
VRAM_WAIT_MAX_S="${VRAM_WAIT_MAX_S:-120}"

# stdout/stderr ของ unit ไป journald เสมอ → `journalctl -u qa-finetune -f` เห็นทุกบรรทัด
say() { echo "[headless $(date +%T)] $*"; }

restore_gui() {
  if [ "${RESTORE_GUI}" = "1" ]; then
    say "restoring GUI (isolate graphical.target)..."
    systemctl isolate graphical.target || true
  fi
}

say "stopping GUI (isolate multi-user.target)..."
systemctl isolate multi-user.target
# เผื่อ display-manager ไม่ได้ผูกกับ graphical.target ตรง ๆ
systemctl stop display-manager.service gdm.service 2>/dev/null || true
# ollama อาจรันเป็น system service — pkill ฝั่ง user ฆ่าไม่ได้ ต้องหยุดจาก root ตรงนี้
systemctl stop ollama.service 2>/dev/null || true
# เครื่องนี้ ollama ปกติรันเป็น user process (nohup ollama serve) ซึ่ง "รอด" การปิด GUI
# (logind KillUserProcesses=no) และ run_finetune_7b.sh ค่อย pkill หลัง gate — สายไป:
# ถ้ามีโมเดลค้างใน VRAM (keep_alive) ตอนสั่งรัน gate จะเห็น VRAM ไม่พอแล้ว abort ฟรี
# → ฆ่าตรงนี้ก่อนวัด VRAM (root ฆ่า process ของ user ได้)
pkill -x ollama 2>/dev/null || true
sleep 2

# ── VRAM gate: fail-CLOSED ────────────────────────────────────────────────────
# เงื่อนไขเดียวที่ยอมให้เทรน: อ่านค่า "ตัวเลข" ได้จริง และเกิน target จริง
# อ่านไม่ได้ / ไม่ถึง target / หมดเวลา → abort + คืน GUI (ไม่ฝืนเทรนเด็ดขาด)
READY=0
for i in $(seq 1 "${VRAM_WAIT_MAX_S}"); do
  FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
  if [ -n "${FREE}" ]; then
    say "VRAM free=${FREE}MB (target>${VRAM_FREE_TARGET_MB}MB) [${i}/${VRAM_WAIT_MAX_S}]"
    if [ "${FREE}" -gt "${VRAM_FREE_TARGET_MB}" ]; then
      READY=1
      break
    fi
  else
    say "VRAM read empty (driver transitioning) — waiting... [${i}/${VRAM_WAIT_MAX_S}]"
  fi
  sleep 1
done

if [ "${READY}" != "1" ]; then
  say "ABORT: VRAM never freed above ${VRAM_FREE_TARGET_MB}MB within ${VRAM_WAIT_MAX_S}s — NOT starting training" | tee -a "${LOG}"
  say "hint: ดูว่าใครถือ VRAM อยู่: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  restore_gui
  exit 1
fi

say "VRAM confirmed free — starting training (MAX_STEPS=${MAX_STEPS}, CPU_OFFLOAD=${CPU_OFFLOAD}, GPU_BUDGET_GIB=${GPU_BUDGET_GIB}, RESUME='${RESUME}', LOG_STEPS=${LOG_STEPS}, SAVE_STEPS=${SAVE_STEPS}) as ${RUN_USER}"
say "log file → ${LOG} (บรรทัดเดียวกันขึ้น journalctl -u qa-finetune -f สด ๆ ด้วย — จอไม่เงียบอีกแล้ว)"

# สร้างไฟล์ log เป็นของ RUN_USER (เดิม root เป็นเจ้าของ ทำให้ user ลบ/rotate ไม่ได้)
install -o "${RUN_USER}" -g "${RUN_USER}" -m 0644 /dev/null "${LOG}" 2>/dev/null || true

# tee: เขียนทั้งไฟล์และ stdout(journald) — บั๊กเดิมคือ redirect ลงไฟล์อย่างเดียว
# ทำให้ journal เงียบสนิทตลอดการเทรน จนถูกเข้าใจว่าเครื่องค้างแล้วโดนกดปิด
runuser -l "${RUN_USER}" -c "cd '${PROJ_DIR}' && \
  MAX_STEPS='${MAX_STEPS}' CPU_OFFLOAD='${CPU_OFFLOAD}' GPU_BUDGET_GIB='${GPU_BUDGET_GIB}' \
  RESUME='${RESUME}' LOG_STEPS='${LOG_STEPS}' SAVE_STEPS='${SAVE_STEPS}' \
  FINETUNE_SEQ_LEN='${FINETUNE_SEQ_LEN:-}' \
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/run_finetune_7b.sh" 2>&1 | tee -a "${LOG}"
RC="${PIPESTATUS[0]}"

say "training finished rc=${RC}" | tee -a "${LOG}"
restore_gui
exit "${RC}"
