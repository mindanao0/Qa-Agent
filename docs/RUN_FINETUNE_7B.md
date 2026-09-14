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

- ความเร็ว: **~4.5–6 นาที/step** ที่ seq=384 (วัดจริงรอบ validation 2026-07-12: 275–350s/step)
- `MAX_STEPS=20` (รอบทดสอบ) ≈ **2 ชม.** | `MAX_STEPS=300` ≈ **23–29 ชม.**
  (`RESUME=auto` ต่อจาก checkpoint-20 ของรอบทดสอบ → เหลือ ~21–27 ชม.)
- 1 step = 16 examples (batch 1 × grad_accum 16) → 300 steps ≈ 4,800 examples (~1.3 epoch)
- checkpoint ทุก **20 steps** (~1.5–2 ชม.) — ปรับได้ด้วย `SAVE_STEPS`
- **seq=384 เป็นค่า default ของ preset 7b แล้ว** (validation 2026-07-12 ผ่าน: VRAM เหลือ
  ต่ำสุด 0.83GB, watchdog ไม่สะดุด, เก็บข้อมูล 93.3%) — seq-fit หลัง dedupe:
  256 → 21.8% | 320 → 67.2% | **384 → 92.6%**
  `FINETUNE_SEQ_LEN` เหลือไว้สำหรับทดลองเท่านั้น และ**ห้าม RESUME ข้ามค่า seq ที่ต่างกัน**
  (dataset ถูกกรองใหม่ตามความยาว → เทรนต่อบนข้อมูลคนละชุดแบบเงียบ ๆ)

---

## 📋 เช็คลิสต์วันรันจริง (นัด 2026-07-16 — audit ครบ 2026-07-12)

**ก่อนกด (~5 นาที):**
1. `train.jsonl` ต้องไม่ถูกแก้หลัง 12 ก.ค. (sha256 อ้างอิง: `3903477...568ab`, 3641 บรรทัด)
   — **ห้ามรัน `run_multi_site.sh` ก่อนวันรัน** (มัน append ลง train.jsonl); ถ้าเผลอรันไป
   preflight จะ fail เอง → ต้องเปลี่ยนเป็นเริ่ม fresh (ย้าย checkpoint-* เข้า archive_*)
2. ห้ามอัปเดต `~/.qa-finetune-env` และอย่าอัปเกรด nvidia driver ก่อน/ระหว่างรัน
3. `PREFLIGHT=1 RESUME=auto MAX_STEPS=300 bash scripts/finetune_headless.sh` → ต้องเขียวทุกข้อ
4. save งานทุกอย่างแล้วปิดแอป — GUI จะถูกปิดทิ้งทั้ง session ตอนเริ่มเทรน

**กด:** `sudo MAX_STEPS=300 RESUME=auto bash scripts/finetune_headless.sh`
— ไม่ต้องใส่ env อื่น (seq=384 เป็น default แล้ว) | จอดับไป TTY = ปกติ |
โดนขัดกลางทาง (ไฟดับ/รีบูต) → สั่งคำสั่งเดิมซ้ำ เทรนต่อจาก checkpoint ล่าสุด

**เช็ค 2 บรรทัดนี้ใน ~2 นาทีแรก (สำคัญที่สุด):** ตอน resume ต้องเห็น
```
[resume] restored 88 LoRA tensors from .../checkpoint-20 (adapter='default', ||lora_B||=6.1516)
[resume] LR re-synced to the current schedule at step 20: 0.000e+00 → 1.994e-04
```
ไม่เห็น = โค้ดที่รันไม่มี fix → kill ทันที (`sudo systemctl stop qa-finetune`) แล้วเรียกคนดู
จากนั้นที่ **step 21** บรรทัด dict ของ HF ต้องอ่านได้ `'learning_rate': 0.000199…`
(ถ้าเป็น `0.0` หรือ `~5e-06` = LR ตายจริง → kill)

> **ประวัติ (2026-07-14):** เส้นทาง resume เคย **พังจริง** — รัน 2 ครั้ง (21:39, 21:41) ตายใน 2 วินาที
> ด้วย `ValueError: weight is on the meta device` (PEFT `load_adapter()` สั่ง `dispatch_model()` ใหม่
> บนโมเดลที่ CPU-offload อยู่) แล้ว trap คืน GUI → เด้งไปหน้า login ทันที ดูเหมือน "รันแล้ว logout"
> แก้แล้วใน `wsl2_trainer.py` (`restore_lora_adapter` + `resync_lr_after_resume`) และ**ทดสอบสดกับ
> checkpoint-20 ตัวจริงผ่านแล้ว** — รอบซ้อม 40 steps ที่เคยแนะนำจึงไม่จำเป็นอีก

**เทรนจบ:** `bash scripts/run_after_finetune.sh 6` (start ollama + register + eval ให้ครบ)
หรือ A/B ด้วยโมเดลใหม่: `LLM_MODEL=qa-agent-finetuned bash scripts/run_after_finetune.sh 6`

---

## ✅ STEP 0 — ตรวจความพร้อม (ไม่ต้อง sudo, GUI ไม่ดับ)

```bash
cd ~/code/Qa-Agent
PREFLIGHT=1 bash scripts/finetune_headless.sh
```

มันเช็คให้: python env, dataset, disk, nvidia-smi, checkpoint ค้าง — พังข้อไหนบอกตรงนั้น
ผ่านแล้วจะสรุปแผน (จำนวน step, เวลาโดยประมาณ, budget) ให้ดูก่อน

## 🧪 STEP 1 — รอบทดสอบ 20 steps (~2 ชม.) — ✅ ผ่านแล้ว 2026-07-12

```bash
sudo MAX_STEPS=20 bash scripts/finetune_headless.sh
```

- จอจะดับไปหน้า TTY ภายใน ~5 วิ = **ปกติ** (GUI ถูกปิดเพื่อคืน VRAM)
- ดูสด: กด `Ctrl+Alt+F3` → login → `journalctl -u qa-finetune -f`
  จะเห็น `Step N/20 | Loss: ... | VRAM: ... | ETA ...` ทุก ~4 นาที + VRAM ทุก 30 วิ
  **ถ้าเงียบเกิน ~5 นาทีค่อยถือว่าผิดปกติ** (ของเดิมเงียบทั้งวันเพราะ log ลงไฟล์อย่างเดียว)
- เทรนจบ GUI กลับมาเอง — ถ้าอนาคต watchdog เตือน/หยุดที่ seq=384 (เช่น dataset โตขึ้น)
  → เริ่มรอบทดสอบใหม่แบบ fresh ด้วย `FINETUNE_SEQ_LEN=320` (ห้าม resume ข้ามค่า seq)
- รอบ 2026-07-12 ผ่านจริง: 20/20 steps, loss 2.22→1.95, VRAM เหลือต่ำสุด 0.83GB

## 🚀 STEP 2 — รอบจริง 300 steps (~21–27 ชม. เมื่อต่อจาก checkpoint-20)

```bash
sudo MAX_STEPS=300 RESUME=auto bash scripts/finetune_headless.sh
# (RESUME=auto = ต่อจาก checkpoint ล่าสุดถ้ามี — เช่น checkpoint-20 ของรอบทดสอบ
#  ซึ่งเทรนด้วย seq/dataset เดียวกัน จึงต่อได้เลย ไม่เสีย 20 step แรกฟรี)
```

- **หยุดกลางทางแบบไม่เสียงาน:** `sudo systemctl stop qa-finetune`
  แล้วรอบหน้าใส่ `RESUME=auto` — มันเทรนต่อจาก checkpoint ล่าสุด (ทุก 20 steps)
- ไฟดับ/เครื่องรีบูตเอง → เหมือนกัน: สั่งคำสั่งเดิมพร้อม `RESUME=auto`
- อยากเริ่มใหม่หมดจริง ๆ → ย้าย `models/finetune_output/checkpoint-*` ออกก่อน
  (สคริปต์จะ**ปฏิเสธ**การรันทับ checkpoint เก่าถ้าไม่ได้บอก RESUME — กันงานหายเงียบ ๆ)

## 📦 STEP 3 — Export + Register (อัตโนมัติ / ทำเองก็ได้)

เทรนจบสคริปต์จะ export ให้เอง (แบบใหม่: **CPU ล้วน ไม่ใช้ VRAM** — แปลงเฉพาะ
adapter ~2MB ด้วย llama.cpp แล้วให้ Ollama ประกอบ `FROM qwen2.5-coder:7b + ADAPTER`)
แต่ตอนนั้น ollama daemon ถูกปิดอยู่ (โดน pkill ตอนเริ่มเทรน และ**ไม่ฟื้นเอง** —
เครื่องนี้ ollama ไม่ใช่ systemd service) — ขั้น register มี 2 ทาง:

```bash
# ทางสะดวก: สคริปต์ post-run จัดการให้ (start ollama + register + eval ต่อ)
bash scripts/run_after_finetune.sh 6

# หรือทำเอง:
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

สคริปต์นี้จัดการให้ครบ: รอเทรนจบ → **start ollama เอง** (ไม่มีใครปลุกให้หลังเทรน) →
**register adapter ล่าสุดเป็น `qa-agent-finetuned`** → รัน eval 14 sites:

```bash
bash scripts/run_after_finetune.sh 6                          # eval ด้วย base model
LLM_MODEL=qa-agent-finetuned bash scripts/run_after_finetune.sh 6   # eval ด้วยโมเดล fine-tuned (A/B)
```

⚠️ ถ้าจะให้มัน "รอ" ข้ามช่วงเทรน ห้ามรันจาก terminal ใน GUI — ตอนเทรนเริ่ม GUI ถูก
isolate ทิ้งทั้ง session สคริปต์ตายไปด้วย ให้ `nohup` จาก TTY (Ctrl+Alt+F3) หรือ
ง่ายสุดคือรอเทรนจบแล้วค่อยรัน (มันข้ามการรอ เริ่ม eval ทันที)

---

## 🆘 แก้ปัญหา

| อาการ | ความหมาย / วิธีแก้ |
|---|---|
| `ABORT: VRAM never freed above 5700MB` | GUI/process อื่นยังถือ VRAM หลังปิด GUI แล้ว — `nvidia-smi` ดูใครถือ; นี่คือ gate ทำงานถูกต้อง (ของเดิมจะฝืนเทรนแล้วพัง) |
| `Found existing checkpoint ... but --resume was not given` | มีงานเก่าค้าง — ใส่ `RESUME=auto` เพื่อต่อ หรือย้าย checkpoint ออกเพื่อเริ่มใหม่ |
| `stop_reason=vram_critical (watchdog)` | VRAM หมดจริง 2 รอบติด — ใช้ budget 2.9 (default) หรือลด `FINETUNE_SEQ_LEN` **แบบเริ่ม fresh เท่านั้น** (resume ข้าม seq = dataset คนละชุด) |
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
