## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Housekeeping only — fix legacy violations found by audit
ห้าม modify Sprint 4–10 source files (sfg, grounder, crawler, compiler,
repair, planner, hypothesis, executor, ast_parser, generator, judge,
spa/*, shadow/*, observability/*, race/*, fuzzer/*, js_*)
ห้าม modify audit/sprint*/measure_*.py

## OBJECTIVE
GOAL: แก้ 3 กลุ่มปัญหาที่ audit พบ ให้ codebase สะอาดก่อนเริ่ม Sprint 11

---
## TASK 1 — pyproject.toml: add missing deps + verify lock

# 1a. Add jsonschema (ใช้อยู่ใน Sprint 10 แต่ยังไม่อยู่ใน pyproject)
uv add jsonschema

# 1b. ตรวจ pyproject.toml ว่ามี deps ครบสำหรับทุก sprint:
#     ถ้าขาด dep ใดใน list นี้ให้ uv add ทันที:
REQUIRED_DEPS = [
    "lancedb",
    "pydantic>=2.0",
    "playwright",
    "langgraph",
    "sentence-transformers",
    "opentelemetry-sdk",
    "opentelemetry-api",
    "pyarrow",
    "jsonschema",
    "defusedxml",
    "pytest-asyncio",
    "pytest",
    "rich",
]
# ตรวจด้วย: uv run python -c "import <pkg>" แต่ละตัว
# ถ้า ImportError → uv add <pkg>

# 1c. หลัง uv add ทุกตัวแล้ว:
uv lock
uv sync
# ตรวจว่า uv.lock updated และ uv sync exit 0

---
## TASK 2 — Fix page.accessibility violation

FILE: src/core/sfg_engine.py line ~53

FIND pattern:
    await page.accessibility.snapshot()

REPLACE WITH CDP equivalent:
    cdp = await page.context.new_cdp_session(page)
    try:
        ax_tree = await cdp.send("Accessibility.getFullAXTree")
    finally:
        await cdp.detach()
    # ax_tree["nodes"] replaces snapshot() output

IMPORTANT: ถ้า sfg_engine.py ใช้ snapshot() output ใน dict format
ให้ map fields:
    snapshot key "role"  → ax_tree node "role"]["value"]
    snapshot key "name"  → ax_tree node["name"]["value"]
    snapshot key "value" → ax_tree node["value"]["value"]

หลังแก้:
    grep -r "page.accessibility" src/
    → ต้องได้ 0 matches (ยกเว้น comment บรรทัดที่ขึ้นต้นด้วย #)

---
## TASK 3 — Fix bare Ollama HTTP calls (no semaphore)

TARGET FILES (6 ไฟล์):
    agents/generator_agent.py      line ~90
    agents/planner_agent.py        line ~75
    agents/healer_agent.py         line ~79
    agents/observer_driver/driver_agent.py  line ~141
    src/core/sfg_engine.py         line ~157
    finetune/export.py             line ~222

PATTERN TO FIND in each file:
    httpx.post("http://localhost:11434/...")
    httpx.AsyncClient().post(...)
    requests.post("http://localhost:11434/...")

FIX PATTERN — replace direct HTTP with OllamaAdapter:

    # BEFORE (example):
    response = await httpx.AsyncClient().post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5-coder:7b-instruct-q4_K_M", "prompt": prompt}
    )
    result = response.json()["response"]

    # AFTER:
    from src.core.adapter import OllamaAdapter
    _adapter = OllamaAdapter()  # singleton — if already instantiated, reuse
    result = await _adapter.generate(prompt)

SINGLETON PATTERN — ถ้าในไฟล์นั้นมี class:
    # เพิ่ม _adapter เป็น instance variable ใน __init__
    self._adapter = OllamaAdapter()
    # แล้วใช้ self._adapter.generate() แทน

ถ้าในไฟล์นั้นเป็น standalone function:
    # module-level singleton
    _adapter = OllamaAdapter()

VERIFY OllamaAdapter has generate() + embed() methods:
    type src/core/adapter.py
    # ต้องมี:
    #   async def generate(self, prompt: str, **kwargs) -> str
    #   async def embed(self, text: str) -> list[float]
    #   _inference_semaphore: asyncio.Semaphore  (Semaphore(1) or (2))
    # ถ้า method ขาด → เพิ่มใน adapter.py ก่อน

---
## TASK 4 — Fix grounder import path

AUDIT FINDING:
    spec says: src/contractskill/grounder.py → FILE NOT FOUND
    actual:    src/perception/grounder.py

STEP 1 — ตรวจว่า import path ถูกต้องทุกที่:
    grep -r "from src.contractskill.grounder" src/
    grep -r "from .grounder" src/contractskill/

STEP 2 — ถ้ามี import ที่ผิด path ให้แก้เป็น:
    from src.perception.grounder import Grounder

STEP 3 — เพิ่ม re-export ใน src/contractskill/__init__.py:
    from src.perception.grounder import Grounder
    __all__ = ["Grounder", ...]
    # เพื่อให้ import จาก contractskill ก็ยังได้ (backward compat)

STEP 4 — verify:
    uv run python -c "from src.contractskill import Grounder; g = Grounder(); print('OK')"
    → ต้องได้ OK

---
## TASK 5 — Verification Suite

รัน checks ทั้งหมดนี้แล้ว print ผลทุกบรรทัด:

# Check 1: no page.accessibility
grep -rn "page\.accessibility" src/
# Expected: 0 results (comments excluded)

# Check 2: no bare Ollama HTTP
grep -rn "localhost:11434" src/
# Expected: 0 results นอกจาก src/core/adapter.py เท่านั้น

# Check 3: Grounder import
uv run python -c "
from src.contractskill import Grounder
g = Grounder()
print(f'Grounder OK: {g}')
"

# Check 4: jsonschema importable
uv run python -c "import jsonschema; print(f'jsonschema {jsonschema.__version__} OK')"

# Check 5: all sprint results still intact (regression check)
uv run python -c "
import json, pathlib
sprints = range(4, 11)
for n in sprints:
    p = pathlib.Path(f'audit/sprint{n}/sprint{n}_results.json')
    if p.exists():
        d = json.loads(p.read_text())
        status = d.get(f'sprint{n}_status', 'UNKNOWN')
        reg = d.get('regression', '?')
        print(f'Sprint {n}: {status} | regression={reg}')
    else:
        print(f'Sprint {n}: FILE NOT FOUND')
"
# Expected: ทุก sprint = PASS | regression=false

# Check 6: uv sync clean
uv sync --frozen
# Expected: exit 0

---
## OUTPUT CONTRACT
DONE WHEN:
- [ ] uv.lock updated, uv sync exit 0
- [ ] grep page.accessibility → 0 matches
- [ ] grep localhost:11434 → only adapter.py
- [ ] Grounder() instantiates without error
- [ ] jsonschema importable
- [ ] Sprint 4–10 results.json ทุกไฟล์ยัง PASS
- [ ] Print summary:

HOUSEKEEPING REPORT
===================
pyproject.toml:     UPDATED / deps added: [list]
page.accessibility: FIXED / was at sfg_engine.py:53
bare HTTP calls:    FIXED / N files updated
grounder path:      FIXED / re-export added
uv sync:            OK
all sprints:        PASS ✅

## MUST NOT
- ห้าม modify audit/sprint*/measure_*.py
- ห้าม modify sprint 4–10 core files
- ห้าม downgrade pydantic < 2.0
- ห้าม remove asyncio.Semaphore(1) จาก adapter.py
- ห้าม ใช้ os.path string concat (pathlib.Path เท่านั้น)