#!/usr/bin/env python
"""Golden-set evaluation harness for the Universal test planner.

Drives the REAL src.universal_qa.test_planner.UniversalTestPlanner against the
20 curated pages in eval/golden_set.jsonl and reports three metrics per planner
configuration (baseline / +grammar / +few-shot / +RAG):

  json_valid_rate   first-pass rate at which the planner's LLM calls produce a
                    schema-valid _FuncResponse with NO repair retry. This is the
                    raw structured-output reliability that Phase-1 grammar
                    constraint is meant to fix ("JSON พัง / markdown leak").
                    Measured identically for every config; the only thing that
                    changes per phase is the intervention under test.

  coverage_score    mean over pages of (expected_scenarios covered /
                    expected_scenarios). A scenario counts as covered when any of
                    its bilingual keywords appears in any LLM-generated test for
                    the page. Scenario-based (not raw count) so it does not
                    saturate and can rise when RAG surfaces missed behaviours.

  assertion_quality fraction of LLM-generated tests whose assertion checks a
                    CONCRETE state/backend signal (a specific URL reached, a
                    validation/error message, a content/count/state change, an
                    HTTP/WCAG check) rather than a generic "page responds / loads
                    / is visible". Heuristic classifier below (TH + EN).

Why "LLM-generated tests" for the headline: grammar / few-shot / RAG only change
the LLM planning path (functional / negative / edge). The planner's rule-based
template tests (accessibility / security / form_validation / broken_link /
error_page / search / logout) are constant across phases, so they are scored and
reported SEPARATELY (rule_* diagnostics) to keep the headline sensitive to the
actual interventions. With flows=[] in the synthetic NavigationMap, every
type=="functional" test is LLM-derived.

Honesty: every number here comes from real Ollama generations against the pinned
production model (config/agent.yaml). Nothing is mocked or dry-run.

Run:
  .venv/bin/python eval/run_eval.py --mode baseline --label baseline --out eval/baseline.json
  .venv/bin/python eval/run_eval.py --mode baseline --limit 2          # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.universal_qa.explorer.nav_map import (  # noqa: E402
    ExploredAction, ExploredPage, NavigationMap,
)
from src.universal_qa.models import TestCase  # noqa: E402
from src.universal_qa.test_planner import UniversalTestPlanner  # noqa: E402

# type=="functional" == LLM-generated (flows are empty so no rule-based functional)
_LLM_TYPE = "functional"

# ──────────────────────────────────────────────────────────────────────────────
# Assertion-quality classifier  (does the test assert CONCRETE state?)
# ──────────────────────────────────────────────────────────────────────────────
_URL_RE = re.compile(r'https?://[^\s"\']+')

# STRONG concrete signals — specific, checkable state/backend assertions.
_STRONG_CONCRETE = [
    # validation / error messages
    "error", "validation", "ผิดพลาด", "แจ้งเตือน", "required", "invalid",
    "ไม่ถูกต้อง", "ต้องกรอก", "ห้ามว่าง", "ไม่ตรงกัน", "already", "มีอยู่แล้ว",
    "incorrect", "could not be verified", "ถูกล็อก", "locked", "ถูกใช้แล้ว",
    "taken", "ซ้ำ", "did not match",
    # specific result / content
    "ปรากฏข้อความ", "แสดงข้อความ", "verify text", "ผลการค้นหา", "complete",
    "created", "submitted", "thanks for", "บันทึกสำเร็จ", "สร้างบัญชี",
    # count / quantity
    "จำนวน", "count", "badge", "เพิ่มขึ้น", "ลดลง", "ตะกร้า", "row", "แถว",
    # state / persistence / navigation target
    "checked", "unchecked", "selected", "ถูกเลือก", "ติ๊ก", "redirect",
    "เข้าสู่ระบบ", "logged in", "ยอดเงิน", "ยอดคงเหลือ", "balance", "dashboard",
    "overview", "accounts", "บัญชี", "selected value",
    # backend / http
    "http", "status code", " 200", " 404", "response code", "backend",
    "database", "api ", "endpoint",
    # accessibility rule
    "wcag", "accessible name", "alt text", "aria",
]

# GENERIC phrases — NOT a concrete assertion (page responds / loads / visible).
_GENERIC_RE = [
    re.compile(r"ตอบสนอง.*ได้ถูกต้อง"),
    re.compile(r"ตอบสนองต่อการคลิก"),
    re.compile(r"โหลด(ได้|สำเร็จ|หน้า|เสร็จ)"),
    re.compile(r"หน้าเว็บโหลด"),
    re.compile(r"แสดงขึ้นมา"),
    re.compile(r"ชื่อหน้าแสดง"),
    re.compile(r"ทำงาน(ได้|ถูกต้อง)"),
    re.compile(r"\bvisible\b"),
    re.compile(r"tobevisible"),
    re.compile(r"is displayed"),
    re.compile(r"displayed correctly"),
    re.compile(r"works? (correctly|as expected)"),
    re.compile(r"responds? (correctly|to)"),
]


def classify_assertion(tc: TestCase) -> tuple[bool, str]:
    """Return (is_concrete, reason)."""
    eo = tc.expected_outcome or ""
    steps_txt = " ".join(tc.steps or [])
    low = (eo + " \n " + steps_txt).lower()

    # 1) a specific full URL in the expected outcome == a redirect/navigation assertion
    if _URL_RE.search(eo):
        return True, "specific-url"
    # 2) strong concrete token anywhere (assertion or a concrete verify step)
    for kw in _STRONG_CONCRETE:
        if kw in low:
            return True, f"concrete:{kw.strip()}"
    # 3) explicitly generic phrasing
    for rx in _GENERIC_RE:
        if rx.search(low):
            return False, "generic"
    # 4) conservative default: no concrete signal found
    return False, "no-concrete-signal"


# ──────────────────────────────────────────────────────────────────────────────
# Scenario coverage
# ──────────────────────────────────────────────────────────────────────────────
def scenario_covered(scenario: dict, tests: list[TestCase]) -> bool:
    kws = [k.lower() for k in scenario.get("keywords_any", [])]
    if not kws:
        return False
    for tc in tests:
        blob = (tc.title + " " + tc.expected_outcome + " " + " ".join(tc.steps)).lower()
        if any(k in blob for k in kws):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Instrumented structured client — records first-pass schema validity
# ──────────────────────────────────────────────────────────────────────────────
class RecordingClient:
    """Wrap any client exposing async create_structured(); count first-pass validity."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.calls = 0
        self.valid = 0
        self.errors: list[str] = []

    async def create_structured(self, prompt, response_model, temperature: float = 0.0):
        self.calls += 1
        try:
            result = await self.inner.create_structured(prompt, response_model, temperature)
            self.valid += 1
            return result
        except Exception as exc:  # noqa: BLE001 — planner falls back; we record the miss
            if len(self.errors) < 20:
                self.errors.append(repr(exc)[:200])
            raise

    async def close(self) -> None:
        if hasattr(self.inner, "close"):
            await self.inner.close()


def build_client(mode: str):
    """Construct the phase-specific structured client (wrapped for recording)."""
    if mode == "baseline":
        # First-pass (no repair retry) instructor JSON mode against the pinned 7B.
        from src.llm.instructor_client import InstructorClient
        return RecordingClient(InstructorClient(max_retries=1))
    if mode == "grammar":
        # Phase 1 — Ollama schema-constrained decoding (format=<json schema>).
        from src.llm.grammar_client import GrammarConstrainedClient  # added in Phase 1
        return RecordingClient(GrammarConstrainedClient())
    if mode in ("fewshot", "rag"):
        # Phases 2/3 are CUMULATIVE on top of grammar (Phase 1 adopted): grammar
        # client + few-shot / RAG planner changes (toggled via env in run()).
        from src.llm.grammar_client import GrammarConstrainedClient
        return RecordingClient(GrammarConstrainedClient())
    raise ValueError(f"unknown mode: {mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Golden page -> synthetic NavigationMap (no live crawl; fixed input for all phases)
# ──────────────────────────────────────────────────────────────────────────────
def page_to_navmap(page: dict) -> NavigationMap:
    url = page["url"]
    sp = urlsplit(url)
    base_url = f"{sp.scheme}://{sp.netloc}"
    actions = [
        ExploredAction(
            page_url=url,
            action_label=e["name"],
            element_role=e.get("role"),
            element_name=e["name"],
            leads_to_url=e.get("leads_to_url"),
            is_destructive=False,
        )
        for e in page["elements"]
    ]
    ep = ExploredPage(
        url=url, title=page["title"], pam_content=page["pam_content"], actions=actions,
    )
    return NavigationMap(
        base_url=base_url, pages=[ep], flows=[],
        explored_at_iso=datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-page evaluation
# ──────────────────────────────────────────────────────────────────────────────
async def eval_page(page: dict, mode: str) -> dict:
    client = build_client(mode)
    planner = UniversalTestPlanner()
    # Inject the instrumented / phase-specific structured client without modifying
    # production code (the planner reads self._client for every LLM call).
    planner._client = client  # noqa: SLF001 — deliberate eval-harness override
    nav = page_to_navmap(page)

    t0 = time.monotonic()
    try:
        cases: list[TestCase] = await planner.plan_from_map(nav)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"plan_from_map failed for {page['page_id']}: {exc!r}")
        cases = []
    elapsed = time.monotonic() - t0

    llm_cases = [c for c in cases if c.type == _LLM_TYPE]
    rule_cases = [c for c in cases if c.type != _LLM_TYPE]

    # assertion quality
    llm_concrete = [classify_assertion(c) for c in llm_cases]
    n_concrete = sum(1 for ok, _ in llm_concrete if ok)
    rule_concrete = sum(1 for c in rule_cases if classify_assertion(c)[0])

    # scenario coverage (headline = vs LLM tests; diagnostic = vs all tests)
    scenarios = page["expected_scenarios"]
    covered_llm = [s["id"] for s in scenarios if scenario_covered(s, llm_cases)]
    covered_all = [s["id"] for s in scenarios if scenario_covered(s, cases)]

    await client.close()

    return {
        "page_id": page["page_id"],
        "site": page["site"],
        "url": page["url"],
        "elapsed_s": round(elapsed, 1),
        "llm_calls": client.calls,
        "llm_calls_valid": client.valid,
        "n_llm_tests": len(llm_cases),
        "n_rule_tests": len(rule_cases),
        "n_concrete_llm": n_concrete,
        "n_concrete_rule": rule_concrete,
        "expected_test_count": page["expected_test_count"],
        "n_scenarios": len(scenarios),
        "covered_scenarios_llm": covered_llm,
        "covered_scenarios_all": covered_all,
        "page_json_valid_rate": round(client.valid / client.calls, 4) if client.calls else None,
        "page_coverage_llm": round(len(covered_llm) / len(scenarios), 4) if scenarios else None,
        "page_assertion_quality": round(n_concrete / len(llm_cases), 4) if llm_cases else None,
        "errors": client.errors,
        # a few sample classifications for auditability
        "sample_tests": [
            {"title": c.title, "expected_outcome": c.expected_outcome,
             "concrete": classify_assertion(c)[0], "reason": classify_assertion(c)[1]}
            for c in llm_cases[:4]
        ],
    }


async def run(golden_path: Path, mode: str, label: str, limit: int | None) -> dict:
    import os
    # Cumulative phases: fewshot = grammar + few-shot; rag = grammar + few-shot + RAG.
    # Set explicitly (not relying on config) so eval modes are deterministic.
    os.environ["TEST_PLANNER_FEWSHOT"] = "1" if mode in ("fewshot", "rag") else "0"
    os.environ["TEST_PLANNER_RAG"] = "1" if mode == "rag" else "0"

    pages = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        pages = pages[:limit]

    logger.info(f"Evaluating {len(pages)} pages | mode={mode} label={label}")
    per_page = []
    for i, page in enumerate(pages, 1):
        logger.info(f"[{i}/{len(pages)}] {page['page_id']}")
        per_page.append(await eval_page(page, mode))

    # ── aggregate ──
    tot_calls = sum(p["llm_calls"] for p in per_page)
    tot_valid = sum(p["llm_calls_valid"] for p in per_page)
    tot_llm = sum(p["n_llm_tests"] for p in per_page)
    tot_concrete = sum(p["n_concrete_llm"] for p in per_page)
    tot_scen = sum(p["n_scenarios"] for p in per_page)
    tot_cov_llm = sum(len(p["covered_scenarios_llm"]) for p in per_page)
    tot_cov_all = sum(len(p["covered_scenarios_all"]) for p in per_page)
    tot_rule = sum(p["n_rule_tests"] for p in per_page)
    tot_rule_concrete = sum(p["n_concrete_rule"] for p in per_page)

    # coverage_score = mean of per-page scenario-coverage (equal weight per page)
    page_cov = [p["page_coverage_llm"] for p in per_page if p["page_coverage_llm"] is not None]
    coverage_score = round(sum(page_cov) / len(page_cov), 4) if page_cov else 0.0

    summary = {
        "label": label,
        "mode": mode,
        "model": "qwen2.5-coder:7b-instruct-q4_K_M",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pages": len(per_page),
        "metrics": {
            "json_valid_rate": round(tot_valid / tot_calls, 4) if tot_calls else None,
            "coverage_score": coverage_score,
            "assertion_quality": round(tot_concrete / tot_llm, 4) if tot_llm else None,
        },
        "diagnostics": {
            "total_llm_calls": tot_calls,
            "total_llm_calls_valid": tot_valid,
            "total_llm_tests": tot_llm,
            "total_rule_tests": tot_rule,
            "avg_llm_tests_per_page": round(tot_llm / len(per_page), 2) if per_page else 0,
            "scenario_coverage_micro_llm": round(tot_cov_llm / tot_scen, 4) if tot_scen else None,
            "scenario_coverage_micro_all": round(tot_cov_all / tot_scen, 4) if tot_scen else None,
            "assertion_quality_rule_tests": round(tot_rule_concrete / tot_rule, 4) if tot_rule else None,
            "total_scenarios": tot_scen,
            "total_eval_seconds": round(sum(p["elapsed_s"] for p in per_page), 1),
        },
        "per_page": per_page,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden_set.jsonl"))
    ap.add_argument("--mode", default="baseline",
                    choices=["baseline", "grammar", "fewshot", "rag"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="eval only first N pages (smoke)")
    args = ap.parse_args()

    label = args.label or args.mode
    summary = asyncio.run(run(Path(args.golden), args.mode, label, args.limit))

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["diagnostics"], indent=2, ensure_ascii=False))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
