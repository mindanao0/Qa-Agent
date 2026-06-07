## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Cleanup only — ห้าม modify Sprint logic, measure scripts, หรือ test files
BASELINE: 15/15 sprints PASS, audit_chain_valid=true

## OBJECTIVE
GOAL: ทำความสะอาด 2 จุดก่อนเข้า fine-tune pipeline:
      (1) pyproject.toml deprecated field
      (2) Semaphore standardize → Semaphore(1) ทุกที่

---
## TASK 1 — pyproject.toml: fix deprecated field

FIND in pyproject.toml:
    [tool.uv.dev-dependencies]

REPLACE WITH:
    [dependency-groups]
    dev = [...]

Reference: https://docs.astral.sh/uv/concepts/dependencies/#dev-dependencies
uv ≥ 0.4.0 ใช้ dependency-groups แทน tool.uv.dev-dependencies

VERIFY:
    uv sync
    # ต้องไม่มี DeprecationWarning เกี่ยวกับ tool.uv.dev-dependencies
    # exit 0

---
## TASK 2 — Semaphore audit + standardize to Semaphore(1)

STEP 1 — inventory semaphore declarations ทั้ง project:
    grep -rn "Semaphore" src/ --include="*.py"
    # print ทุก match: file:line:content

STEP 2 — categorize แต่ละ match:
    A) Semaphore(1)  → CORRECT — leave untouched
    B) Semaphore(2)  → NEEDS FIX — change to Semaphore(1)
    C) Semaphore(N>2)→ NEEDS FIX — change to Semaphore(1)
    D) Semaphore ที่ไม่ใช่ Ollama (เช่น browser workers, parse workers)
                     → SKIP — parallel semaphores ไม่ต้องแก้

STEP 3 — ระบุว่า Semaphore ไหน guard Ollama calls:
    ดูว่า semaphore ถูกใช้กับ:
    - adapter.generate() / adapter.embed()
    - httpx call ไปยัง localhost:11434
    - OllamaAdapter method ใดๆ
    → ถ้าใช่ = Ollama semaphore → ต้องเป็น Semaphore(1)

STEP 4 — fix Semaphore(2) ใน src/llm/adapter.py:
    BEFORE: _inference_semaphore = asyncio.Semaphore(2)
    AFTER:  _inference_semaphore = asyncio.Semaphore(1)

    ADD comment above:
    # INVARIANT: must remain Semaphore(1) — 6GB VRAM GTX 1660 Ti
    # concurrent Ollama calls cause VRAM contention and OOM
    # do NOT increase without hardware upgrade

STEP 5 — scan for any other Semaphore(2+) guarding Ollama:
    ถ้าพบใน files อื่น → fix เป็น Semaphore(1) เช่นกัน
    ถ้าพบ Semaphore ที่ไม่ใช่ Ollama (เช่น _PARSE_SEMAPHORE(4) ใน js_ast_parser.py)
    → SKIP (parallel file parsing, ไม่ใช่ VRAM)

---
## TASK 3 — Dataset preparation scaffold

สร้างไฟล์เหล่านี้ (empty scaffold สำหรับ fine-tune pipeline):

### scripts/collect_training_data.py
# Collects training examples from Sprint run logs → JSONL format
# Schema per example:
#
# class TrainingExample(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     example_id:   str        # sha256[:10]
#     source:       str        # "sprint6" | "sprint9" | "sprint13" | etc.
#     prompt:       str        # input to LLM
#     completion:   str        # expected LLM output
#     quality:      float      # 0.0–1.0 (1.0 = test passed, 0.0 = judge rejected)
#     metadata:     dict       # sprint, func_id, test_type, etc.
#
# Sources to mine (append-only, read-only):
#   Sprint 6:  src/codetest/generator.py generation logs
#              → (prompt, completion) pairs where test passed pytest
#   Sprint 9:  src/codetest/js_generator.py
#              → (prompt, completion) pairs where Vitest passed
#   Sprint 13: src/fuzzer/schema_inferrer.py inference logs
#              → (endpoint_traces, inferred_schema) pairs
#   Sprint 14: src/pbt/invariant_extractor.py
#              → (function_spec, invariant) pairs
#
# Output: data/training/raw_examples.jsonl
# Target: 500+ examples (count after collection)
#
# NOTE: scaffold only — implement collection logic per source
# DO NOT run yet (logs may not exist in expected format)

### scripts/validate_dataset.py
# Validates raw_examples.jsonl before fine-tuning
#
# Checks:
#   1. Total examples ≥ 500
#   2. quality distribution: mean ≥ 0.70
#   3. source diversity: ≥ 3 distinct sources
#   4. No duplicate example_ids
#   5. prompt + completion both non-empty
#   6. No BLOCKED_ACTION_PATTERNS in completions
#
# Output: data/training/validation_report.json
# Exit 1 if any check fails

### data/training/.gitkeep
# Create empty directory (training data not committed to git)

### .gitignore additions:
# data/training/*.jsonl
# data/training/*.json
# !data/training/.gitkeep

---
## TASK 4 — WDDM TDR scaffold (read-only prep, NO registry writes yet)

สร้างไฟล์นี้เป็น documentation + dry-run script:

### scripts/wddm_tdr_fix.py
# WDDM TDR registry fix for GPU training stability
# WARNING: modifies Windows registry — run only when ready for fine-tune
# DRY RUN by default (--apply flag required to write)
#
# Registry keys to set (HKLM\System\CurrentControlSet\Control\GraphicsDrivers):
#   TdrLevel  = 3   (REG_DWORD) — recover without reboot
#   TdrDelay  = 60  (REG_DWORD) — 60s before TDR timeout (default=2s)
#   TdrDdiDelay = 60 (REG_DWORD)
#
# Script logic:
#   if "--apply" not in sys.argv:
#       print("DRY RUN — current values:")
#       # read and print current registry values
#       print("Run with --apply to write changes")
#       sys.exit(0)
#   else:
#       # write registry values
#       # print confirmation
#       print("WDDM TDR fix applied — reboot required")
#
# Use winreg module (stdlib, Windows only)
# pathlib.Path for any file operations
# NO external dependencies

---
## VERIFICATION

รัน checks ทั้งหมด:

# Check 1: no deprecation warning
uv sync 2>&1 | grep -i "deprecat"
# Expected: 0 lines

# Check 2: Semaphore(1) only for Ollama
grep -rn "Semaphore" src/ --include="*.py"
# Expected: Ollama semaphores = Semaphore(1) ทุกตัว

# Check 3: dataset scaffold exists
uv run python -c "
import pathlib
files = [
    'scripts/collect_training_data.py',
    'scripts/validate_dataset.py',
    'scripts/wddm_tdr_fix.py',
    'data/training/.gitkeep',
]
for f in files:
    p = pathlib.Path(f)
    print(f'{'OK' if p.exists() else 'MISSING'}: {f}')
"

# Check 4: sprint results still intact
uv run python -c "
import json, pathlib
for n in range(4, 16):
    p = pathlib.Path(f'audit/sprint{n}/sprint{n}_results.json')
    if p.exists():
        d = json.loads(p.read_text())
        print(f'Sprint {n}: {d.get(f\"sprint{n}_status\")} | regression={d.get(\"regression\")}')
    else:
        print(f'Sprint {n}: FILE NOT FOUND')
"
# Expected: ทุก sprint = PASS | regression=false

# Check 5: wddm dry run (safe — read only)
uv run python scripts/wddm_tdr_fix.py
# Expected: prints current TDR values, no writes

## OUTPUT CONTRACT
DONE WHEN:
- [ ] uv sync ไม่มี deprecation warning
- [ ] Semaphore(1) ทุก Ollama path (print inventory ก่อน+หลัง)
- [ ] scaffold files ครบ 4 ไฟล์
- [ ] sprint 4–15 results ทุกอันยัง PASS
- [ ] wddm dry run exit 0

## CLEANUP REPORT format:
PHASE 2 CLEANUP REPORT
======================
pyproject deprecated field: FIXED | ALREADY_CLEAN
semaphore_inventory_before: [list of Semaphore(N) found]
semaphore_inventory_after:  [list — all Ollama = Semaphore(1)]
dataset_scaffold:           CREATED
wddm_dry_run:               OK | ERROR
  current TdrLevel:   <value>
  current TdrDelay:   <value>
  current TdrDdiDelay: <value>
sprint_results:             15/15 PASS