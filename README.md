# QA Agent — Universal Local AI QA Agent

> ระบบ AI สำหรับสร้างและรัน Playwright test อัตโนมัติ โดยใช้ภาษาธรรมดาเป็น input และทำงานได้ทั้งหมดบนเครื่อง local โดยไม่ต้องส่งข้อมูลออกไปภายนอก

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Playwright](https://img.shields.io/badge/Playwright-1.49%2B-green)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## สารบัญ

1. [ระบบทำอะไรได้บ้าง](#ระบบทำอะไรได้บ้าง)
2. [ความต้องการของระบบ](#ความต้องการของระบบ)
3. [การติดตั้ง](#การติดตั้ง)
4. [วิธีใช้งาน](#วิธีใช้งาน)
5. [ดู Allure Report](#ดู-allure-report)
6. [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
7. [การตั้งค่า](#การตั้งค่า)
8. [RBAC — การจัดการ Role](#rbac--การจัดการ-role)
9. [การแก้ปัญหาที่พบบ่อย](#การแก้ปัญหาที่พบบ่อย)
10. [ข้อจำกัดของระบบ](#ข้อจำกัดของระบบ)
11. [License](#license)

---

## ระบบทำอะไรได้บ้าง

QA Agent มี 4 mode การทำงาน:

| Mode | สิ่งที่ทำ |
|------|-----------|
| **Generate** | รับ requirement ภาษาธรรมดา → วางแผน test → สร้าง Playwright Python code อัตโนมัติ |
| **Run** | รัน test script ผ่าน pytest + Self-Healing อัตโนมัติเมื่อ locator พัง |
| **Heal** | ซ่อม locator ที่พังโดยใช้ AI วิเคราะห์ Accessibility Tree (AxTree) ของหน้าเว็บ |
| **Finetune** | สร้าง synthetic dataset → fine-tune โมเดลด้วย QLoRA 4-bit บน GPU local |

### Architecture Overview

```
                    ┌─────────────────────────────────────────────────────┐
                    │              QA Agent Pipeline                      │
                    └─────────────────────────────────────────────────────┘

  ภาษาธรรมดา
  "test login page"
        │
        ▼
  ┌─────────────┐     TestPlan JSON    ┌───────────────┐    Python Code
  │  Planner    │ ──────────────────► │   Generator   │ ──────────────►
  │   Agent     │                     │    Agent      │
  │ (Ollama LLM)│                     │ (Ollama LLM)  │
  └─────────────┘                     └───────────────┘
        ▲                                                       │
        │                                                       ▼
        │                                            ┌──────────────────┐
        │                                            │  Playwright      │
        │              AxTree + Error                │  Executor        │
        │         ◄──────────────────────────────── │  (pytest)        │
        │                                            └──────────────────┘
        │                                                       │ FAIL
        │                                                       ▼
        │                                            ┌──────────────────┐
        │                                            │  Healer Agent    │
        └────────────────────────────────────────── │  Phase 1: Fuzzy  │
          healed locator                             │  Phase 2: AI     │
                                                     │  Phase 3: VLM    │
                                                     └──────────────────┘
                                                               │ PASS
                                                               ▼
                                                    ┌──────────────────┐
                                                    │  Allure Report   │
                                                    │  + Session JSON  │
                                                    └──────────────────┘
```

### Self-Healing ทำงานอย่างไร

เมื่อ test พัง ระบบจะไม่หยุดทันที แต่จะพยายามซ่อม locator อัตโนมัติใน 3 phase:

```
test FAILED
     │
     ├─► Phase 1: Fuzzy Match (Jaro-Winkler ≥ 0.85)
     │         ถ้าพบ locator คล้ายกันใน AxTree → ใช้เลย (เร็ว)
     │
     ├─► Phase 2: AI Analysis (AxTree → LLM → new locator)
     │         LLM วิเคราะห์ว่า element ไปอยู่ที่ไหน → confidence ≥ 0.50
     │
     └─► Phase 3: VLM Fallback (CPU-based vision model)
               สำหรับ page ที่ AxTree ไม่เพียงพอ (density < 0.40)
```

---

## ความต้องการของระบบ

| Component | ขั้นต่ำ | แนะนำ |
|-----------|---------|-------|
| OS | Windows 11 (WSL2) หรือ Ubuntu 22.04 | Ubuntu 22.04 บน WSL2 |
| Python | 3.11 | 3.11 หรือ 3.12 |
| VRAM (GPU) | 6 GB | 8 GB+ |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB ว่าง | 40 GB+ (สำหรับ models + datasets) |
| Ollama | latest | latest |
| Node.js | ไม่จำเป็น | ไม่จำเป็น |
| CUDA | 11.8+ (สำหรับ Finetune เท่านั้น) | 12.1+ |

> **หมายเหตุ:** Generate, Run, Heal mode ทำงานได้โดยไม่ต้องมี GPU — Ollama รัน LLM ผ่าน CPU ได้ (ช้ากว่า) Finetune mode ต้องการ GPU เท่านั้น

---

## การติดตั้ง

### Step 1: ติดตั้ง Ollama

```powershell
# Windows — ดาวน์โหลด installer จาก
# https://ollama.com/download/windows
# แล้วติดตั้งตามขั้นตอนปกติ

# ตรวจสอบว่าติดตั้งสำเร็จ
ollama --version
# ควรเห็น: ollama version 0.x.x
```

หลังติดตั้ง Ollama จะรัน background service อัตโนมัติบน `http://localhost:11434`  
ถ้าไม่รันอัตโนมัติ ให้เปิด PowerShell แล้วรัน:

```powershell
ollama serve
```

---

### Step 2: Pull โมเดลที่จำเป็น

ระบบใช้โมเดล 2 ตัว:

| โมเดล | หน้าที่ | ขนาด | VRAM ที่ใช้ |
|-------|---------|------|------------|
| `qwen2.5-coder:7b-instruct-q4_K_M` | สร้าง test plan และ code | ~4.5 GB | ~4.5 GB |
| `nomic-embed-text` | แปลง document เป็น vector (RAG) | ~274 MB | ~300 MB |

```powershell
# Pull โมเดลหลัก (ใช้เวลา 10-20 นาที ขึ้นอยู่กับความเร็ว internet)
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Pull embedding model (เร็วกว่า ~2-5 นาที)
ollama pull nomic-embed-text

# ตรวจสอบว่า pull สำเร็จ
ollama list
```

ผลลัพธ์ที่ควรเห็น:

```
NAME                                    ID              SIZE    MODIFIED
qwen2.5-coder:7b-instruct-q4_K_M       xxxxxxxx        4.7 GB  x minutes ago
nomic-embed-text:latest                 xxxxxxxx        274 MB  x minutes ago
```

---

### Step 3: Clone และ Setup โปรเจกต์

```powershell
# เข้าไปที่โฟลเดอร์ที่ต้องการ
cd $HOME

# Clone โปรเจกต์
git clone <repo-url> qa-agent
cd qa-agent

# ติดตั้ง uv (package manager) ถ้ายังไม่มี
pip install uv

# ติดตั้ง Python dependencies ทั้งหมด
uv sync

# ติดตั้ง Playwright browser (Chromium)
uv run playwright install chromium --with-deps
```

> **ถ้าเจอ error `Failed to hardlink files`** ให้รันก่อน:
> ```powershell
> $env:UV_LINK_MODE = "copy"
> uv sync
> ```

---

### Step 4: ตั้งค่า Environment Variables

```powershell
# คัดลอก template
Copy-Item .env.example .env

# แก้ไขค่าใน .env ตามระบบของคุณ
notepad .env
```

ค่าสำคัญใน `.env`:

```env
# URL ของ Ollama (ปกติไม่ต้องเปลี่ยน)
OLLAMA_BASE_URL=http://localhost:11434

# URL ของ application ที่ต้องการ test
APP_BASE_URL=http://localhost:3000

# Credentials สำหรับแต่ละ role (ใส่ตามระบบของคุณ)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=yourpassword
```

---

### Step 5: ตรวจสอบการติดตั้ง

```powershell
# ทดสอบ Ollama ตอบได้ไหม
curl.exe -s -X POST http://localhost:11434/api/generate `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen2.5-coder:7b-instruct-q4_K_M","prompt":"say ok","stream":false}' `
  | python -c "import sys,json; print(json.load(sys.stdin)['response'])"
```

ผลลัพธ์ที่ควรเห็น:

```
ok
```

```powershell
# ทดสอบ Python imports
uv run python -c "from src.llm.adapter import OllamaAdapter; print('LLM OK')"
uv run python -c "from src.agents.graph import build_graph; print('Graph OK')"
uv run python -c "from src.agents.generator import GeneratorAgent; print('Generator OK')"
```

ผลลัพธ์ที่ควรเห็น:

```
LLM OK
Graph OK
Generator OK
```

---

## วิธีใช้งาน

### 🟢 Generate Mode — สร้าง Test อัตโนมัติ

**ระบบทำงานอย่างไร:**

1. **Planner Agent** รับ requirement → คิดวิเคราะห์ → สร้าง `TestPlan` (JSON) ที่มี steps, edge cases, RBAC scenarios
2. **Generator Agent** รับ `TestPlan` → สร้าง Playwright Python code ในรูปแบบ pytest-playwright
3. **Executor** รัน code ที่สร้างขึ้นผ่าน pytest เพื่อตรวจสอบว่า syntax ถูกต้อง
4. บันทึกไฟล์ใน `tests/generated/`

```powershell
uv run python main.py `
  --mode generate `
  --requirement "test login page with correct and incorrect credentials" `
  --url https://the-internet.herokuapp.com/login `
  --role admin
```

**Parameters:**

| Parameter | คำอธิบาย | ค่า default | ตัวอย่าง |
|-----------|-----------|------------|---------|
| `--requirement` | สิ่งที่ต้องการ test (ภาษาอะไรก็ได้) | *(จำเป็น)* | `"test payroll calculation"` |
| `--url` | URL ของหน้าที่ต้องการ test | `APP_BASE_URL` จาก `.env` | `https://app.com/login` |
| `--role` | Role สำหรับ RBAC authentication | `admin` | `admin`, `hr_manager`, `employee` |
| `--domain` | Domain hint ช่วย Planner วางแผน (optional) | `general` | `hrm`, `payroll`, `general` |
| `--model` | Ollama model ที่ใช้ | `qwen2.5-coder:7b-instruct-q4_K_M` | ชื่อ model ใน ollama |
| `--output-dir` | โฟลเดอร์ที่บันทึก script | `tests/generated` | `tests/my_tests` |
| `--max-retries` | จำนวนครั้งที่ retry เมื่อ fail | `3` | `5` |

**ตัวอย่าง log output ที่ควรเห็น:**

```
┌───────────────────────────── qa-agent generate ─────────────────────────────┐
│ Generating test script                                                       │
│ Requirement: test login page with correct and incorrect credentials          │
│ URL: https://the-internet.herokuapp.com/login  Role: admin  Domain: general  │
└──────────────────────────────────────────────────────────────────────────────┘
21:10:49 | INFO     | BrowserManager started | headless=True max_contexts=3
21:10:49 | INFO     | ▶ planner_node
21:11:41 | INFO     | PlannerAgent: plan ready | title='Login Feature Test Plan' steps=5 complexity='low'
21:11:41 | INFO     | ▶ generator_node
21:11:41 | INFO     | GeneratorAgent attempt 1/3 (two-pass)
21:12:29 | INFO     | GeneratorAgent: valid Python generated
21:12:29 | INFO     | ▶ executor_node

✓ Script saved: tests\generated\test_login_with_correct_credentials.py
```

**ตัวอย่าง code ที่ระบบสร้างขึ้น:**

```python
import pytest
from playwright.sync_api import Page, expect

def test_login_with_correct_credentials(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('tomsmith')
    page.get_by_label('Password').fill('SuperSecretPassword!')
    page.get_by_role('button', name='Login').click()
    expect(page).to_have_url('https://the-internet.herokuapp.com/secure')

def test_login_with_incorrect_credentials(page: Page) -> None:
    page.goto('https://the-internet.herokuapp.com/login')
    page.get_by_label('Username').fill('tomsmith')
    page.get_by_label('Password').fill('wrongpassword')
    page.get_by_role('button', name='Login').click()
    expect(page.get_by_text('Your password is invalid!')).to_be_visible()
```

---

### 🔵 Run Mode — รัน Test + Self-Healing

**ระบบทำงานอย่างไร:**

1. โหลด test script ที่ระบุ
2. รันผ่าน `pytest` พร้อม Playwright
3. ถ้า test พัง → **Healer Agent** วิเคราะห์ error message
4. ถ้าเป็น locator error → ดึง live AxTree จากเบราว์เซอร์ → AI หา locator ใหม่ → patch code → retry อัตโนมัติ
5. retry สูงสุด 3 ครั้ง → บันทึกผลใน Allure

```powershell
uv run python main.py `
  --mode run `
  --script tests\generated\test_generated.py
```

**Parameters:**

| Parameter | คำอธิบาย | ค่า default |
|-----------|-----------|------------|
| `--script` | path ของ test script | *(จำเป็น)* |
| `--alluredir` | โฟลเดอร์บันทึก Allure results | `allure-results` |
| `--max-retries` | จำนวนครั้ง Self-Healing สูงสุด | `3` |
| `--url` | URL ที่ใช้เปิดหน้าเว็บสำหรับ Self-Healing | `APP_BASE_URL` จาก `.env` |
| `--role` | Role ที่ใช้ login ตอน Self-Healing | `admin` |

**Self-Healing Loop รายละเอียด:**

```
Retry 1/3:
  Phase 1 (Fuzzy Match):
    - สแกน AxTree หา element ที่ชื่อคล้าย locator เดิม
    - ใช้ Jaro-Winkler distance (threshold: 0.85)
    - ถ้า confidence ≥ 0.85 → ใช้เลย (ไม่เรียก LLM)
  Phase 2 (AI):
    - ส่ง AxTree + locator ที่พัง → LLM วิเคราะห์
    - LLM เลือก locator ใหม่ที่ semantic ใกล้เคียง
    - ถ้า confidence ≥ 0.50 → patch code → retry

Retry 2/3: [ทำซ้ำ]
Retry 3/3: [ทำซ้ำ]

หลังจาก 3 ครั้ง → reporter_node บันทึกผลสุดท้าย
```

**ตัวอย่าง log output ที่ควรเห็น:**

```
┌─────────────────────────────── qa-agent run ────────────────────────────────┐
│ Executing test script                                                        │
│ Script: tests\generated\test_generated.py                                    │
└──────────────────────────────────────────────────────────────────────────────┘
collected 5 items

tests/generated/test_generated.py::test_login_with_correct_credentials PASSED  [ 20%]
tests/generated/test_generated.py::test_login_with_incorrect_credentials FAILED [ 40%]

21:17:06 | INFO    | ▶ healer_node
21:17:06 | INFO    | CodeHealerAgent: healing locator="page.get_by_text('Your username is invalid!')"
21:17:09 | INFO    | Phase 1 confidence 0.0000 < 0.85 — escalating to Phase 2 (AI)
21:17:21 | INFO    | ai_heal: result | healed="page.get_by_role('alert')" confidence=0.9000
21:17:21 | INFO    | Phase 2 (AI) succeeded | confidence=0.9000

tests/generated/test_generated.py::test_login_with_empty_username_and_password PASSED [ 60%]
tests/generated/test_generated.py::test_login_with_empty_username_and_correct_password PASSED [ 80%]
tests/generated/test_generated.py::test_login_with_correct_username_and_empty_password PASSED [100%]

1 failed, 4 passed in 26.55s
```

---

### 🟡 Heal Mode — ซ่อม Locator ที่พัง

ใช้สำหรับซ่อม locator ใน `locators/locators.json` ที่บันทึกไว้ก่อนหน้า  
เหมาะสำหรับกรณีที่ UI ของ application เปลี่ยนแปลงและ locator เดิมพังหลายตัวพร้อมกัน

```powershell
uv run python main.py `
  --mode heal `
  --url https://your-app.com `
  --locator-file locators\locators.json
```

**Parameters:**

| Parameter | คำอธิบาย | ค่า default |
|-----------|-----------|------------|
| `--url` | Base URL ของ application | `APP_BASE_URL` จาก `.env` |
| `--locator-file` | path ของ locators.json | `locators/locators.json` |
| `--role` | Role ที่ใช้ login | `admin` |

**ตัวอย่าง output:**

```
┌──────────────────────────── qa-agent heal ──────────────────────────────────┐
│ Re-validating locators                                                       │
│ Locator file: locators\locators.json  Base URL: https://app.com  Entries: 8  │
└──────────────────────────────────────────────────────────────────────────────┘
  Healing page.get_by_label('Username')…          ✓ confidence=0.97 method=fuzzy
  Healing page.get_by_role('button', name='Login')… ✓ confidence=0.92 method=ai
  Healing page.get_by_text('Dashboard')…          ⚠ low confidence=0.62

┌─────────────────┬───────┐
│ Status          │ Count │
├─────────────────┼───────┤
│ Healed (≥0.85)  │ 7     │
│ Low / failed    │ 1     │
└─────────────────┴───────┘
```

---

### 🔴 Finetune Mode — Fine-tune โมเดลด้วยข้อมูลของตัวเอง

> ⚠️ **ต้องการ GPU ที่มี VRAM ว่างอย่างน้อย 5.5 GB** และต้องติดตั้ง `unsloth` แยกต่างหาก:
> ```powershell
> pip install "unsloth[colab-new]"
> ```

**ขั้นตอนการทำงาน:**

1. สร้าง **Synthetic Dataset** (JSONL ChatML format) จาก 7 domain seeds:
   - `hrm_login`, `payroll_calculation`, `leave_request`, `employee_profile`
   - `rbac_permission_check`, `thai_tax_calculation`, `overtime_calculation`
2. สร้างตัวอย่างประมาณ 500 รายการ (happy path / negative / RBAC boundary)
3. **Fine-tune** ด้วย Unsloth QLoRA 4-bit บน `Qwen2.5-Coder-7B-Instruct`
4. **Export** เป็น GGUF Q4_K_M
5. **Register** ใน Ollama เป็น `qa-agent-coder:latest`

```powershell
# สร้าง dataset + fine-tune ในคำสั่งเดียว
uv run python main.py `
  --mode finetune `
  --dataset ~/.qa-agent/datasets/synthetic.jsonl `
  --generate-data

# Fine-tune จาก dataset ที่มีอยู่แล้ว + export GGUF
uv run python main.py `
  --mode finetune `
  --dataset ~/.qa-agent/datasets/synthetic.jsonl `
  --export-gguf
```

**Parameters:**

| Parameter | คำอธิบาย | ค่า default |
|-----------|-----------|------------|
| `--dataset` | path ของ JSONL dataset | `~/.qa-agent/datasets/synthetic.jsonl` |
| `--checkpoint-dir` | โฟลเดอร์บันทึก checkpoint | `~/.qa-agent/checkpoints` |
| `--generate-data` | สร้าง dataset ก่อน fine-tune | `False` |
| `--export-gguf` | Export และ register ใน Ollama หลัง train | `False` |

---

### 🟣 Continuous Mode — Explore/Generate/Execute/Heal แบบวนหลายรอบ

> เป็นคนละ entry point จาก `main.py --mode ...` (เหมือน Universal QA Agent) — เรียกผ่าน
> `python -m src.continuous` โดยตรง เพราะเป็น orchestration mode คู่ขนาน (multi-cycle
> loop ที่คง browser context เดิมข้ามรอบ + LangGraph checkpoint) ไม่ใช่ phase หนึ่งของ
> pipeline เดิม

วนลูป N รอบของ crawl (state ใหม่) → สร้าง web+code test → execute → self-heal →
checkpoint (LangGraph `AsyncSqliteSaver`, resume ได้ด้วย `--run-id` เดิม) จนกว่าจะถึง
`--max-cycles`, coverage plateau, หรือ memory limit (`StopConditionEvaluator`)

```bash
# TodoMVC-shaped app (profile เริ่มต้น — action plan เขียนเฉพาะ TodoMVC DOM)
uv run python -m src.continuous --url https://demo.playwright.dev/todomvc/#/ --max-cycles 5

# เว็บไซต์อื่น ๆ ที่ไม่ใช่ TodoMVC — ใช้ profile generic (delegate ไปที่ UniversalQAAgent
# หนึ่งรอบเดียว ไม่ใช่ loop จริง ๆ — ดู caveat ใน `python -m src.continuous --help`)
uv run python -m src.continuous --url https://your-app.com --profile generic --max-pages 30
```

**Parameters:**

| Parameter | คำอธิบาย | ค่า default |
|-----------|-----------|------------|
| `--url` | Target URL (จำเป็น) | — |
| `--profile` | `todomvc` (loop จริง, TodoMVC-shaped) หรือ `generic` (single-pass ทุกเว็บ) | `todomvc` |
| `--max-cycles` | จำนวนรอบสูงสุด, `0` = ไม่จำกัด | `10` |
| `--run-id` | id ของ run / checkpoint thread — ใส่ id เดิมซ้ำเพื่อ resume | สุ่มใหม่ |
| `--output-dir` | โฟลเดอร์เก็บ checkpoint/SFG/audit/episodic ของ run นี้ | `reports/continuous/<run-id>/` |

ผลลัพธ์แต่ละ run แยกต่อ `--run-id` ใต้ `reports/continuous/` (ไม่ commit เข้า git —
คนละที่กับ `audit/sprint11/` ซึ่งเป็น artifact ของ gate ที่ปิดแล้วและต้องไม่ถูกเขียนทับ)

---

## ดู Allure Report

ต้องติดตั้ง Allure CLI ก่อน: [allurereport.org/docs/install](https://allurereport.org/docs/install/)

```powershell
# เปิด interactive report (เปิด browser อัตโนมัติ)
allure serve allure-results

# หรือ export เป็น HTML static
allure generate allure-results -o allure-report --clean
Start-Process allure-report\index.html
```

**Report แสดงข้อมูลอะไรบ้าง:**

| ส่วน | ข้อมูลที่แสดง |
|------|--------------|
| Timeline | ลำดับการรัน test แต่ละ function |
| Suites | Pass/Fail statistics แยกตาม test file |
| Behaviors | จัดกลุ่มตาม feature / scenario |
| Self-Healing Events | before/after locator ที่ถูกซ่อม พร้อม confidence score |
| AxTree Snapshots | Accessibility Tree ของหน้าเว็บ ณ เวลาที่ test พัง |
| AI Reasoning | เหตุผลที่ LLM เลือก locator ใหม่ |

---

## โครงสร้างโปรเจกต์

```
qa-agent/
├── src/
│   ├── llm/
│   │   ├── adapter.py           # OllamaAdapter — เชื่อมต่อ Ollama API พร้อม retry + VRAM guard
│   │   ├── structured.py        # Pydantic schemas + JSON repair pipeline (enforce_json_output)
│   │   └── prompt_templates.py  # PTCF/RTF prompt templates สำหรับทุก agent
│   ├── browser/
│   │   ├── manager.py           # BrowserManager — จัดการ Playwright browser lifecycle
│   │   ├── resource_filter.py   # บล็อก image/media/font/stylesheet เพื่อประหยัด RAM
│   │   ├── auth_manager.py      # จัดการ storageState RBAC ต่อ role (API login เท่านั้น)
│   │   └── ax_extractor.py      # ดึงและ prune Accessibility Tree จาก page
│   ├── healing/
│   │   ├── engine.py            # Self-Healing orchestrator (Phase 1→2→3)
│   │   ├── fuzzy_matcher.py     # Phase 1: Jaro-Winkler fuzzy matching
│   │   └── ai_healer.py         # Phase 2: AI วิเคราะห์ AxTree → locator ใหม่
│   ├── rag/
│   │   ├── store.py             # LanceDB vector store (local, no server)
│   │   ├── ingestion.py         # Document ingestion + semantic chunking
│   │   └── retriever.py         # Hybrid search: 70% semantic + 30% BM25
│   ├── agents/
│   │   ├── graph.py             # LangGraph StateGraph — orchestrate ทุก node
│   │   ├── planner.py           # PlannerAgent: requirement → TestPlan
│   │   ├── generator.py         # GeneratorAgent: TestPlan → Playwright code (two-pass)
│   │   └── healer.py            # CodeHealerAgent: error + AxTree → patched code
│   ├── data/
│   │   ├── synthetic_gen.py     # สร้าง synthetic QA dataset (JSONL ChatML)
│   │   └── pii_masker.py        # Mask PII ก่อนส่ง LLM (PDPA compliance)
│   ├── finetune/
│   │   ├── trainer.py           # Unsloth QLoRA 4-bit fine-tuning pipeline
│   │   └── export.py            # Merge LoRA → GGUF → register ใน Ollama
│   └── reporting/
│       └── allure_reporter.py   # Allure integration: attach AxTree, screenshot, reasoning
├── config/
│   ├── agent.yaml               # ทุก parameter ที่ปรับได้ (timeout, threshold, model)
│   └── roles.yaml               # RBAC role definitions + storageState paths
├── locators/
│   └── locators.json            # คลัง locator ที่ถูกซ่อมแล้ว (atomic read/write)
├── tests/
│   ├── generated/               # test script ที่ระบบสร้างขึ้น (Generate mode output)
│   └── example_hrm_payroll.py   # ตัวอย่าง test สำหรับระบบ Thai HRM/Payroll
├── scripts/
│   ├── setup_wsl2.sh            # ติดตั้งทุกอย่างบน WSL2 Ubuntu ในคำสั่งเดียว
│   └── pull_models.sh           # Pull และ verify Ollama models
├── main.py                      # CLI entry point (--mode generate|run|finetune|heal)
├── pyproject.toml               # Project config + dependencies (uv)
├── .env.example                 # Template environment variables
└── README.md                    # ไฟล์นี้
```

---

## การตั้งค่า

ไฟล์ `config/agent.yaml` รวมทุก parameter ที่ปรับได้:

```yaml
llm:
  model: "qwen2.5-coder:7b-instruct-q4_K_M"  # โมเดลหลักสำหรับ generate code
  embedding_model: "nomic-embed-text"          # โมเดลสำหรับ RAG embeddings
  base_url: "http://localhost:11434"           # Ollama API endpoint
  temperature: 0.1         # ต่ำ = deterministic (แนะนำสำหรับ QA)
  max_tokens: 4096         # ความยาวสูงสุดของ output (ลดถ้า VRAM ไม่พอ)
  semaphore_limit: 2       # จำนวน LLM call พร้อมกันสูงสุด (ป้องกัน VRAM OOM)
  vram_buffer_mb: 500      # VRAM ที่ต้องว่างก่อนเรียก LLM (MB)
  request_timeout_sec: 120 # timeout ต่อ request
  retry_attempts: 3        # จำนวน retry เมื่อ LLM call ล้มเหลว
  retry_min_wait_sec: 2    # รอขั้นต่ำระหว่าง retry
  retry_max_wait_sec: 10   # รอสูงสุดระหว่าง retry (exponential backoff)

browser:
  headless: true           # true = ไม่แสดง browser window (แนะนำสำหรับ CI)
  timeout_ms: 15000        # default timeout ทุก action (milliseconds)
  max_contexts: 3          # browser context พร้อมกันสูงสุด (จำกัดตาม RAM)
  launch_args:             # Chrome flags สำหรับ WSL2/headless
    - "--disable-dev-shm-usage"  # สำคัญมากสำหรับ WSL2
    - "--no-sandbox"
    - "--disable-gpu"
  blocked_resources:       # ประเภท resource ที่บล็อก (ประหยัด bandwidth + RAM)
    - "image"
    - "media"
    - "font"
    - "stylesheet"

healing:
  fuzzy_threshold: 0.85    # Jaro-Winkler threshold สำหรับ Phase 1 auto-accept
  ai_threshold: 0.50       # confidence ขั้นต่ำที่ยอมรับจาก AI (Phase 2)
  vlm_threshold: 0.40      # semantic density ต่ำกว่านี้ → trigger VLM (Phase 3)
  max_retries: 3           # จำนวน healing retry สูงสุด
  locator_file: "locators/locators.json"  # ไฟล์เก็บ locator ที่ซ่อมแล้ว

rag:
  top_k: 5                 # จำนวน document chunk ที่ดึงมาต่อ query
  semantic_weight: 0.7     # น้ำหนักของ vector search (70%)
  bm25_weight: 0.3         # น้ำหนักของ BM25 keyword search (30%)
  max_context_tokens: 2000 # token สูงสุดที่ส่ง LLM จาก RAG context

axtree:
  max_nodes: 200           # จำนวน node สูงสุดใน AxTree ที่ส่ง LLM (prune ส่วนเกิน)
  semantic_density_threshold: 0.4  # ต่ำกว่านี้ = หน้าไม่มี interactive element → VLM

reporting:
  allure_results_dir: "allure-results"  # โฟลเดอร์ output ของ Allure
  attach_axtree: true      # แนบ AxTree snapshot ใน report
  attach_screenshots: true # แนบ screenshot เมื่อ test พัง
  attach_reasoning: true   # แนบ AI reasoning log ใน report
```

---

## RBAC — การจัดการ Role

ระบบรองรับ 5 role ที่กำหนดใน `config/roles.yaml`:

| Role | สิทธิ์ที่มี | ใช้สำหรับ test |
|------|------------|---------------|
| `admin` | ทุก permission รวมถึง `rbac.manage`, `system.config` | ทดสอบ admin features และ full workflow |
| `hr_manager` | `employee.read/write`, `leave.approve`, `reports.read` | ทดสอบการสร้าง/แก้ไขพนักงาน |
| `payroll_officer` | `payroll.read/write/run`, `tax.calculate` | ทดสอบการคำนวณเงินเดือนและภาษี |
| `employee` | `profile.read`, `leave.request`, `payslip.read_own` | ทดสอบ RBAC boundary (ต้องเข้าไม่ได้) |
| `auditor` | `audit_log.read`, `reports.read/export` (read-only ทั้งหมด) | ทดสอบ audit trail |

**Authentication:** ระบบ login ผ่าน API (POST request) เท่านั้น — ไม่มีการ login ผ่าน UI  
`storageState` ถูกบันทึกใน `~/.qa-agent/auth/<role>.json` และ refresh อัตโนมัติเมื่อ token หมดอายุ

### วิธีเพิ่ม Role ใหม่

เปิด `config/roles.yaml` แล้วเพิ่ม block ใหม่:

```yaml
roles:
  # เพิ่ม role ใหม่ตรงนี้
  branch_manager:
    username: "${BRANCH_MANAGER_USERNAME}"
    password: "${BRANCH_MANAGER_PASSWORD}"
    storage_state_path: "~/.qa-agent/auth/branch_manager.json"
    login_payload_template:
      username: "${BRANCH_MANAGER_USERNAME}"
      password: "${BRANCH_MANAGER_PASSWORD}"
      role: "branch_manager"
    permissions:
      - "employee.read"
      - "reports.read"
      - "leave.approve"
```

จากนั้นเพิ่ม credentials ใน `.env`:

```env
BRANCH_MANAGER_USERNAME=branch_mgr
BRANCH_MANAGER_PASSWORD=yourpassword
```

---

## การแก้ปัญหาที่พบบ่อย

| Error Message | สาเหตุ | วิธีแก้ |
|---------------|--------|---------|
| `Connection refused localhost:11434` | Ollama server ไม่ได้รัน | เปิด PowerShell ใหม่แล้วรัน `ollama serve` |
| `CUDA out of memory` | VRAM ไม่พอสำหรับ model | ลด `llm.max_tokens` ใน `config/agent.yaml` จาก 4096 เป็น 2048 |
| `collected 0 items` (exit code 5) | ไม่มี function ที่ขึ้นต้นด้วย `test_` | รัน generate mode ใหม่ — ระบบจะ regenerate อัตโนมัติ |
| `unrecognized arguments: --timeout` | `pytest-timeout` ไม่ได้ติดตั้ง | ลบ `--timeout` flag ออกจาก pytest command |
| `Could not parse LLM output as PlaywrightScript` | LLM ส่ง JSON ที่ parse ไม่ได้ | ระบบ retry อัตโนมัติ 3 ครั้ง — ถ้ายังพัง ลอง reduce `max_tokens` |
| `No module named 'xxx'` | dependencies ไม่ครบ | รัน `uv sync` |
| `Failed to hardlink files` | uv cache คนละ drive | รัน `$env:UV_LINK_MODE = "copy"` ก่อน `uv sync` |
| `VRAM guard: XXX MB free, need 500MB` | VRAM ไม่พอ รอ | ระบบรอ 5 วินาทีแล้ว retry อัตโนมัติ — ปกติหายเอง |
| `PytestConfigWarning: Unknown config option: asyncio_mode` | pytest-asyncio config ไม่ match version | warning นี้ไม่กระทบการรัน sync pytest-playwright tests — ไม่ต้องแก้ |
| `playwright._impl._errors.Error: browserType.launch: ...` | Playwright browser ไม่ได้ติดตั้ง | รัน `uv run playwright install chromium --with-deps` |
| `AuthError: token field 'access_token' not found` | Auth endpoint response format ต่างจาก config | แก้ `auth.token_field` ใน `config/roles.yaml` ให้ตรงกับ API response |
| `Phase 2 (AI) succeeded` แต่ test ยังพัง | Healed locator ไม่ตรงกับ DOM จริง | หน้าเว็บอาจโหลดช้า — เพิ่ม `browser.timeout_ms` ใน agent.yaml |

---

## ข้อจำกัดของระบบ

- **Self-Healing ซ่อมได้เฉพาะ locator** — ถ้า UI เปลี่ยน flow หรือ business logic เปลี่ยน ต้อง generate test ใหม่
- **6 GB VRAM รัน VLM + coding model พร้อมกันไม่ได้** — VLM (Phase 3) จะ activate เฉพาะเมื่อ coding model ไม่ได้รัน
- **Finetune mode ต้องการ VRAM ว่าง 5.5 GB** — ปิด Ollama service ก่อนรัน finetune (`taskkill /f /im ollama.exe`)
- **AxTree ถูก prune เหลือ 200 nodes** — หน้าเว็บที่ซับซ้อนมากอาจ prune element ที่ต้องการออกไป (เพิ่ม `axtree.max_nodes` ได้ แต่ใช้ token มากขึ้น)
- **Context window จำกัด** — RAG context ถูกจำกัดที่ 2000 tokens ต่อ LLM call เพื่อให้เหลือ token สำหรับ code generation
- **ไม่รองรับ SPA ที่ใช้ Shadow DOM** — AxTree ของ Shadow DOM ไม่ถูก expose ผ่าน `page.aria_snapshot()`
- **API-only authentication** — AuthManager ใช้ POST API login เท่านั้น ไม่รองรับ OAuth2 / SSO / MFA ที่ต้องการ UI interaction
- **Windows path บน native Python** — ถ้ารันบน Windows (ไม่ใช่ WSL2) path `~/.qa-agent/` จะ map เป็น `C:\Users\<user>\.qa-agent\`

---

## Fine-tuning on WSL2 (6GB VRAM)

The complete pipeline for fine-tuning the qa-agent coder model on a 6 GB GPU.
All steps are required the first time; later runs reuse the venv and the
TDR registry fix.

**Step 1 — Fix GPU timeout (Windows PowerShell as Admin, ONE TIME ONLY):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fix_wddm_tdr.ps1
# Restart PC when prompted
```

This raises the WDDM TDR timeout from 2 s to 60 s; without it the GPU driver
will reset mid-training (~2 minutes in) and the run will crash.

**Step 2 — Generate the synthetic dataset (Windows PowerShell):**

```powershell
uv run python main.py --mode finetune --step generate
# Expected: 500–800 examples across 13 domains, ~30–60 min
# Dataset lands at C:\Users\<you>\.qa-agent\datasets\synthetic_universal.jsonl
```

A per-domain summary table is printed at the end so you can confirm coverage.

**Step 3 — Set up WSL2 CUDA (WSL2 terminal, ONE TIME ONLY):**

```bash
bash scripts/setup_wsl2_cuda.sh
# Expected: ~10–15 min — installs CUDA 12.4, PyTorch (cu124), Unsloth
```

This creates a dedicated venv at `~/.qa-finetune-env` (kept separate from the
Windows project venv) and stages the trainer + dataset under `~/qa-finetune/`.

**Step 4 — Run training (WSL2 terminal):**

```bash
bash scripts/run_training_wsl2.sh
# Expected: 1–2 hours — 300 steps QLoRA on Qwen2.5-Coder-7B (r=8, q/v_proj only)
```

The VRAM watchdog logs usage every 30 s and forces a graceful stop if free
VRAM drops below 200 MB. Checkpoints land under `~/.qa-agent/checkpoints/`,
and the GGUF + Modelfile end up at `D:\Code\qa-agent\models\` for the Windows
side to pick up.

**Step 5 — Register the model in Ollama (Windows PowerShell):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_ollama_windows.ps1
```

This calls `ollama create qa-agent-coder -f Modelfile`, smoke-tests the model
with a short prompt, then rewrites `config\agent.yaml` so `llm.model` points
at the new tag.

**Step 6 — Verify:**

```powershell
ollama list   # should show qa-agent-coder
uv run python main.py --mode generate `
    --requirement "test checkout flow" `
    --url http://localhost:3000 --role admin
# Output should be produced by qa-agent-coder, not the stock Qwen2.5-Coder
```

---

## License

```
MIT License

Copyright (c) 2026 QA Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
