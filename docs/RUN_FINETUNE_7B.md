# คู่มือรัน Fine-tune 7B (ฉบับปรับปรุง 2026-07-12 — headless + resume)

> เป้าหมาย: เทรน `qwen2.5-coder:7b` บนข้อมูล QA (`data/training/train.jsonl` —
> **3,641 examples หลัง dedupe**) → export adapter GGUF → register เข้า Ollama
> เป็นโมเดล `qa-agent-finetuned`
>
> **สิ่งที่เปลี่ยนจากคู่มือเดิม:** ตอนนี้ทุกอย่างรันผ่าน `scripts/finetune_headless.sh`
> ตัวเดียว — มันปิด GUI ให้เอง, ตรวจ VRAM แบบ fail-closed, สตรีม log ขึ้น
> `journalctl` สด ๆ (บทเรียนคืน 27 มิ.ย.: จอเงียบ ≠ ค้าง แต่คราวนี้จอไม่เงียบแล้ว),
> เทรนเสร็จเปิด GUI คืน และ**เทรนต่อจาก checkpoint ได้** (`RESUME=auto`) —
> โดนปิดเครื่อง/ไฟดับ เสียงานแค่ ≤ ~80 นาที ไม่ใช่ทั้งหมด

---

## ⚠️ ตัวเลขที่ควรรู้ก่อนตัดสินใจ

- ความเร็ว: **~4 นาที/step** (CPU-offload path บน GTX 1660 Ti 6GB)
- `MAX_STEPS=20` (รอบทดสอบ) ≈ **1.5 ชม.** | `MAX_STEPS=300` ≈ **18–19 ชม.**
- 1 step = 16 examples (batch 1 × grad_accum 16) → 300 steps ≈ 4,800 examples
- checkpoint ทุก **20 steps** (~80 นาที) — ปรับได้ด้วย `SAVE_STEPS`
- ตัวกรองความยาว (วัดจริง 2026-07-12 หลัง dedupe):
  seq=256 เก็บข้อมูลได้ **21.8%** | seq=320 → 67.2% | **seq=384 → 92.6%**
  → รอบทดสอบ 20 steps ควรลอง `FINETUNE_SEQ_LEN=384` ถ้า VRAM ไหว จะได้ใช้ข้อมูลเกือบครบ

---

## ✅ STEP 0 — ตรวจความพร้อม (ไม่ต้อง sudo, GUI ไม่ดับ)

```bash
cd ~/code/Qa-Agent
PREFLIGHT=1 bash scripts/finetune_headless.sh
```

มันเช็คให้: python env, dataset, disk, nvidia-smi, checkpoint ค้าง — พังข้อไหนบอกตรงนั้น
ผ่านแล้วจะสรุปแผน (จำนวน step, เวลาโดยประมาณ, budget) ให้ดูก่อน

## 🧪 STEP 1 — รอบทดสอบ 20 steps (~1.5 ชม.)

```bash
sudo MAX_STEPS=20 FINETUNE_SEQ_LEN=384 bash scripts/finetune_headless.sh
```

- จอจะดับไปหน้า TTY ภายใน ~5 วิ = **ปกติ** (GUI ถูกปิดเพื่อคืน VRAM)
- ดูสด: กด `Ctrl+Alt+F3` → login → `journalctl -u qa-finetune -f`
  จะเห็น `Step N/20 | Loss: ... | VRAM: ... | ETA ...` ทุก ~4 นาที + VRAM ทุก 30 วิ
  **ถ้าเงียบเกิน ~5 นาทีค่อยถือว่าผิดปกติ** (ของเดิมเงียบทั้งวันเพราะ log ลงไฟล์อย่างเดียว)
- เทรนจบ GUI กลับมาเอง — ถ้าระหว่างนั้น VRAM watchdog เตือน/หยุด แปลว่า seq=384 ตึงไป
  → รอบจริงใช้ `FINETUNE_SEQ_LEN=320` หรือไม่ใส่ (=256)

## 🚀 STEP 2 — รอบจริง 300 steps (~18–19 ชม.)

```bash
sudo MAX_STEPS=300 RESUME=auto FINETUNE_SEQ_LEN=384 bash scripts/finetune_headless.sh
# (FINETUNE_SEQ_LEN ตามผลรอบทดสอบ; RESUME=auto = ต่อจาก checkpoint ล่าสุดถ้ามี)
```

- **หยุดกลางทางแบบไม่เสียงาน:** `sudo systemctl stop qa-finetune`
  แล้วรอบหน้าใส่ `RESUME=auto` — มันเทรนต่อจาก checkpoint ล่าสุด (ทุก 20 steps)
- ไฟดับ/เครื่องรีบูตเอง → เหมือนกัน: สั่งคำสั่งเดิมพร้อม `RESUME=auto`
- อยากเริ่มใหม่หมดจริง ๆ → ย้าย `models/finetune_output/checkpoint-*` ออกก่อน
  (สคริปต์จะ**ปฏิเสธ**การรันทับ checkpoint เก่าถ้าไม่ได้บอก RESUME — กันงานหายเงียบ ๆ)

## 📦 STEP 3 — Export + Register (อัตโนมัติ / ทำเองก็ได้)

เทรนจบสคริปต์จะ export ให้เอง (แบบใหม่: **CPU ล้วน ไม่ใช้ VRAM** — แปลงเฉพาะ
adapter ~2MB ด้วย llama.cpp แล้วให้ Ollama ประกอบ `FROM qwen2.5-coder:7b + ADAPTER`)
แต่ตอนนั้น ollama daemon ถูกปิดอยู่ มันจะพิมพ์คำสั่ง register ทิ้งไว้ให้ ทำต่อ:

```bash
ollama serve > /tmp/ollama.log 2>&1 &
sleep 3
cd ~/code/Qa-Agent/models/qwen2.5-coder-finetuned && ollama create qa-agent-finetuned -f Modelfile.adapter

# ทดสอบ
ollama run qa-agent-finetuned "Generate a pytest test for a login page"
```

หรือรัน export เองทีหลังจาก checkpoint ไหนก็ได้:

```bash
bash scripts/export_adapter_gguf.sh                     # ใช้ models/finetune_output/final
bash scripts/export_adapter_gguf.sh <adapter_dir>       # ระบุเอง
```

> ทางเก่า (unsloth merge ทั้งโมเดลบน GPU) ยังอยู่หลัง `USE_LEGACY_EXPORT=1`
> แต่มัน OOM ถ้า GUI เปิด — ไม่แนะนำแล้ว

## 🔁 (ทางเลือก) เทรนเสร็จให้รัน multi-site eval ต่ออัตโนมัติ

เปิดอีก terminal (หรือ tmux) ก่อนสั่งเทรน:

```bash
bash scripts/run_after_finetune.sh 6   # รอเทรนจบ → eval 14 sites 6 ชม.
```

---

## 🆘 แก้ปัญหา

| อาการ | ความหมาย / วิธีแก้ |
|---|---|
| `ABORT: VRAM never freed above 5700MB` | GUI/process อื่นยังถือ VRAM หลังปิด GUI แล้ว — `nvidia-smi` ดูใครถือ; นี่คือ gate ทำงานถูกต้อง (ของเดิมจะฝืนเทรนแล้วพัง) |
| `Found existing checkpoint ... but --resume was not given` | มีงานเก่าค้าง — ใส่ `RESUME=auto` เพื่อต่อ หรือย้าย checkpoint ออกเพื่อเริ่มใหม่ |
| `stop_reason=vram_critical (watchdog)` | VRAM หมดจริง 2 รอบติด — ลด `FINETUNE_SEQ_LEN` หรือใช้ budget 2.9 (default) |
| journalctl เงียบเกิน 5 นาที | ผิดปกติจริง — `systemctl status qa-finetune` + `tail models/finetune_output/headless_run_*.log` |
| อยากยกเลิก | `sudo systemctl stop qa-finetune` (GUI จะถูกเปิดคืนโดย unit; ถ้าไม่ ใช้ `sudo systemctl isolate graphical.target`) |
| trainer เตือน clamp budget | ตั้งใจ: >3.2GiB คือโซน OOM-spike ที่วัดมาแล้ว และเร็วขึ้นแค่ ~6% — ใช้ 2.9 ตาม default |

## 📁 ผลลัพธ์อยู่ที่ไหน

- Checkpoint (เทรนต่อได้): `models/finetune_output/checkpoint-*/`
- Adapter สุดท้าย: `models/finetune_output/final/`
- Adapter GGUF + Modelfile: `models/qwen2.5-coder-finetuned/`
- โมเดลใน Ollama: `qa-agent-finetuned` (`ollama list`)
- Log ต่อรอบ: `models/finetune_output/headless_run_<timestamp>.log` (+ `journalctl -u qa-finetune`)
- ของเก่าคืน 27 มิ.ย. (validation 6 steps): `models/finetune_output/archive_20260627_7b_validation/`

## 🧯 ภาคผนวก — รันเองแบบ manual ไม่ผ่าน systemd (แทบไม่จำเป็นแล้ว)

```bash
# Ctrl+Alt+F3 → login
sudo systemctl isolate multi-user.target
cd ~/code/Qa-Agent && MAX_STEPS=300 RESUME=auto bash scripts/run_finetune_7b.sh
sudo systemctl isolate graphical.target
```
