## CONTEXT
PROJECT: D:\Code\qa-agent — Production-Grade Universal Local AI QA Agent
PYTHON: 3.13 + uv | OS: Windows 11 native
WORKING DIR: D:\Code\qa-agent
SCOPE: Read-only audit — ห้าม modify source files ใดๆ ทั้งสิ้น

## OBJECTIVE
GOAL: ตรวจสอบความถูกต้องของ implementation จริงใน Sprint 4–10 ว่า
measurement scripts วัดของจริงหรือ count artifacts/mocks

## REQUIREMENTS

### PHASE 1 — File Inventory (type แต่ละไฟล์จริงๆ)
อ่านและ print full content ของไฟล์เหล่านี้ทั้งหมด:

SPRINT 4:
- src/contractskill/sfg.py
- src/contractskill/grounder.py
- src/contractskill/crawler.py
- src/contractskill/compiler.py
- src/contractskill/repair.py
- audit/sprint4/measure_sprint4.py

SPRINT 5:
- src/explorer/planner.py
- src/explorer/hypothesis.py
- src/explorer/executor.py
- audit/sprint5/measure_sprint5.py

SPRINT 6:
- src/codetest/ast_parser.py
- src/codetest/generator.py
- src/codetest/judge.py
- src/codetest/executor.py
- audit/sprint6/measure_sprint6.py

SPRINT 7:
- src/spa/hydration_guard.py
- src/spa/route_tracker.py
- src/shadow/extractor.py
- src/shadow/locator_builder.py
- audit/sprint7/measure_sprint7.py

SPRINT 8:
- src/observability/tracer.py
- src/observability/structured_logger.py
- src/observability/audit_chain.py
- src/observability/metrics.py
- .github/workflows/qa_agent.yml
- audit/sprint8/measure_sprint8.py

SPRINT 9:
- src/codetest/js_ast_parser.py
- src/codetest/js_generator.py
- src/codetest/js_judge.py
- src/codetest/js_executor.py
- scripts/ast_walker.js
- audit/sprint9/measure_sprint9.py

SPRINT 10:
- src/race/swarm.py
- src/race/detector.py
- src/fuzzer/api_fuzzer.py
- src/fuzzer/fuzz_vectors.py
- audit/sprint10/measure_sprint10.py

### PHASE 2 — Integrity Checks (ตรวจ 6 จุดต่อ sprint)

สำหรับทุก sprint ให้ตรวจ pattern เหล่านี้และ print ผล TRUE/FALSE:

CHECK_1 — Gate values มาจาก real execution หรือ hardcode?
  ค้นหา pattern ใน measure_*.py:
  - hardcoded return values เช่น `return {"pass_rate": 1.0}`
  - `result = X if condition else GATE_THRESHOLD`
  - ค่าที่ตรงกับ gate threshold เป๊ะโดยไม่มี computation

CHECK_2 — asyncio.Semaphore(1) ครบทุก Ollama call?
  ค้นหาใน src/**/*.py ทุกไฟล์:
  - `requests.post` หรือ `httpx` ไปยัง `localhost:11434` โดยไม่มี semaphore
  - `async def` ที่ call ollama แต่ไม่มี `async with self._sem`

CHECK_3 — page.accessibility ถูกใช้ไหม? (ห้ามใช้)
  grep ใน src/**/*.py:
  - `page.accessibility`
  - `await page.accessibility`
  print ทุก match พร้อม filename:line

CHECK_4 — os.path string concat ถูกใช้ไหม? (ต้องใช้ pathlib.Path)
  grep ใน src/**/*.py:
  - `os.path.join`
  - `os.path.exists`
  - string path concat เช่น `"src/" + filename`
  print ทุก match

CHECK_5 — Judge มี 4 checks เท่านั้นไหม?
  อ่าน judge.py ของ sprint 6 และ js_judge.py ของ sprint 9:
  - นับจำนวน check conditions จริงๆ ใน code
  - print: "Sprint 6 judge checks: N" และ "Sprint 9 judge checks: N"

CHECK_6 — measurement วัด real behavior หรือ schema validation เท่านั้น?
  ตรวจว่า executor ของแต่ละ sprint:
  - Sprint 5: HypothesisExecutor → await page.goto() จริงไหม หรือแค่ validate Pydantic schema?
  - Sprint 7: ShadowDOMExtractor → CDP getFlattenedDocument จริงไหม หรือ mock data?
  - Sprint 10: RaceConditionSwarm → asyncio.Barrier จริงไหม หรือ asyncio.sleep?
  - Sprint 10: conflict_found logic → hash comparison จริงไหม หรือ always True?

### PHASE 3 — Suspicious Pattern Report

หลังจาก Phase 1+2 ให้ generate รายงานในรูปแบบนี้:
=== SPRINT INTEGRITY REPORT ===
Generated: {datetime}
Project: D:\Code\qa-agent
SPRINT 4
CHECK_1 hardcode gate:     CLEAN / SUSPICIOUS — {evidence}
CHECK_2 semaphore:         CLEAN / MISSING — {files affected}
CHECK_3 page.accessibility: CLEAN / VIOLATION — {location}
CHECK_4 os.path:           CLEAN / VIOLATION — {location}
CHECK_5 judge checks:      4 checks ✅ / {N} checks ⚠️
CHECK_6 real execution:    REAL / MOCK — {evidence}
VERDICT: TRUSTED / NEEDS_RETEST
[repeat for Sprint 5–10]
=== SUMMARY ===
Sprints fully trusted:     [list]
Sprints needing retest:    [list]
Critical violations found: [list]

### PHASE 4 — Retest ถ้าพบปัญหา

ถ้า PHASE 2 พบ CHECK_1 = SUSPICIOUS หรือ CHECK_6 = MOCK ใน sprint ใด:

1. print คำเตือน:
   "⚠️ SPRINT {N}: measure script may not reflect real behavior"
   "Evidence: {specific line numbers and code}"

2. สำหรับ Sprint 10 โดยเฉพาะ ให้รันคำสั่งนี้แล้ว print raw output:
   uv run python -c "
   import src.race.swarm as s
   import inspect
   print(inspect.getsource(s.RaceConditionSwarm.run))
   "

   uv run python -c "
   import src.race.detector as d
   import inspect
   print(inspect.getsource(d.ConflictDetector.analyze))
   "

3. ถ้า conflict_found logic = hash comparison ที่ always True บน ToDoMVC
   print: "SPRINT 10 RACE: FALSE POSITIVE — isolated BrowserContext cannot share state"

## MUST NOT
- ห้าม modify ไฟล์ใดๆ ทั้งสิ้น (read-only audit)
- ห้าม rerun measure scripts (อ่าน source เท่านั้น)
- ห้าม skip ไฟล์ที่หาไม่เจอ — ถ้าไม่มีให้ print "FILE NOT FOUND: {path}"

## OUTPUT CONTRACT
DONE WHEN:
- [ ] ทุกไฟล์ใน PHASE 1 ถูกอ่านและแสดง content
- [ ] ทุก CHECK ใน PHASE 2 มีผล TRUE/FALSE พร้อม evidence
- [ ] SPRINT INTEGRITY REPORT ครบทุก sprint 4–10
- [ ] Sprint ที่ NEEDS_RETEST ระบุ specific evidence (file:line)