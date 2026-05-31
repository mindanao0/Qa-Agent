"""Sprint 3 measurement harness — BFT generator + adaptive routing.

Runs a golden 10-case dataset through the BFT generator twice:
  pass_1 (cold cache) — drives LOW tier across the board
  pass_2 (warm cache) — should escalate hits to MED/HIGH

Produces ``audit/sprint3/sprint3_results.json`` with the schema defined in
the Sprint 3 spec.

Dry mode (default): emits synthetic zeroed metrics so CI can run without
a live Ollama instance.  ``--live`` actually drives the local Ollama model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Embedded fallback dataset (10 cases). Self-contained — no test fixtures.
GOLDEN_DATASET: list[dict[str, str]] = [
    {
        "url": "https://example.com/login",
        "requirement": "Verify login form has email and password fields",
        "domain": "authentication",
    },
    {
        "url": "https://example.com/dashboard",
        "requirement": "Verify dashboard loads with welcome banner",
        "domain": "dashboard_analytics",
    },
    {
        "url": "https://example.com/users",
        "requirement": "Verify user list table renders at least one row",
        "domain": "crud_operations",
    },
    {
        "url": "https://example.com/users/new",
        "requirement": "Verify create-user form has Save button",
        "domain": "crud_operations",
    },
    {
        "url": "https://example.com/settings",
        "requirement": "Verify Settings page has Save Changes button",
        "domain": "settings_configuration",
    },
    {
        "url": "https://example.com/billing",
        "requirement": "Verify billing page shows invoice list",
        "domain": "finance_banking",
    },
    {
        "url": "https://example.com/profile",
        "requirement": "Verify profile shows display name field",
        "domain": "hrm_payroll",
    },
    {
        "url": "https://example.com/reports",
        "requirement": "Verify reports page has download CSV link",
        "domain": "dashboard_analytics",
    },
    {
        "url": "https://example.com/notifications",
        "requirement": "Verify notifications dropdown shows unread count",
        "domain": "notifications_messaging",
    },
    {
        "url": "https://example.com/admin",
        "requirement": "Verify admin panel requires admin role",
        "domain": "rbac_permissions",
    },
]


SPRINT2_BASELINE_LATENCY_MS = 76899


def _check_ollama() -> None:
    """Verify Ollama is reachable before the live measurement. Exits on failure."""
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    except Exception as exc:
        print(f"ERROR: Ollama not reachable: {exc!r}. Start Ollama first.")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"ERROR: Ollama returned HTTP {resp.status_code}. Start Ollama first.")
        sys.exit(1)
    try:
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        models = []
    print(f"Ollama OK — models: {models}", flush=True)


def _empty_pass_metrics(pass_label: str, n: int) -> dict[str, Any]:
    return {
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_size": n,
        "pass_label": pass_label,
        "bft_metrics": {
            "consensus_reached_rate": 0.0,
            "consensus_failed_rate": 0.0,
            "judge_approved_rate": 0.0,
            "judge_rejected_rate": 0.0,
            "avg_bft_latency_ms": 0,
            "node_agreement_distribution": {
                "3_of_3": 0,
                "2_of_3": 0,
                "1_of_3": 0,
                "0_of_3": 0,
            },
        },
        "routing_metrics": {
            "HIGH_tier_pct": 0.0,
            "MED_tier_pct": 0.0,
            "LOW_tier_pct": 0.0,
            "avg_latency_by_tier": {"HIGH": 0, "MED": 0, "LOW": 0},
        },
        "quality_metrics": {
            "first_run_pass_rate": 0.0,
            "false_pass_reduction_vs_sprint2": 0.0,
        },
        "latency_metrics": {
            "avg_generation_ms": 0,
            "worst_case_full_bft_ms": 0,
            "sprint2_baseline_ms": SPRINT2_BASELINE_LATENCY_MS,
        },
    }


async def _run_one_case_dry(case: dict[str, str]) -> dict[str, Any]:
    """Synthetic per-case result for dry mode."""
    return {
        "tier": "LOW",
        "bft_status": "CONSENSUS_REACHED",
        "latency_ms": 0,
        "passed": True,
        "agreement": "2_of_3",
        "judge_approved": True,
    }


async def _run_one_case_live(
    case: dict[str, str],
    router: Any,
    case_idx: int,
    total: int,
) -> dict[str, Any]:
    """Drive the real BFT pipeline through Ollama for a single case."""
    # Local imports keep dry mode runnable without these deps in path.
    from src.agents.bft_generator import bft_generator_node
    from src.llm.adapter import OllamaAdapter
    from src.llm.instructor_client import InstructorClient
    from src.llm.judge_client import JudgeClient

    # Diagnostic — classify tier and announce start so we can see live progress.
    pre_tier = router.classify(case["url"], case["requirement"], case["domain"])
    pre_tier_name = pre_tier.value if hasattr(pre_tier, "value") else str(pre_tier)
    n_generators = {"HIGH": 0, "MED": 1, "LOW": 3}.get(pre_tier_name, 3)
    if pre_tier_name == "HIGH":
        gen_desc = "cached (no LLM call)"
    else:
        gen_desc = f"{n_generators} serial generator{'s' if n_generators != 1 else ''}"
    print(
        f"[BFT] Starting case {case_idx + 1}/{total}, tier={pre_tier_name}, {gen_desc}...",
        flush=True,
    )

    adapter = OllamaAdapter()
    bft_instr = InstructorClient(
        base_url=adapter.base_url, model=adapter.model, max_retries=1
    )
    judge_instr = InstructorClient(
        base_url=adapter.base_url, model=adapter.model, max_retries=1
    )
    judge = JudgeClient(judge_instr)

    state = {
        "url": case["url"],
        "requirement": case["requirement"],
        "domain": case["domain"],
        "test_plan": {
            "title": case["requirement"],
            "requirement_summary": case["requirement"],
            "estimated_complexity": "low",
            "domain": case["domain"],
            "steps": [
                {
                    "step_number": 1,
                    "description": case["requirement"],
                    "action": "visit",
                    "expected_result": "Page loads",
                    "role": "admin",
                    "preconditions": [],
                }
            ],
        },
    }

    t0 = time.monotonic()
    try:
        result = await bft_generator_node(
            state,
            instructor_client=bft_instr,
            judge_client=judge,
            router=router,
        )
    except Exception as exc:  # pragma: no cover — live path
        result = {
            "bft_status": "CONSENSUS_FAILED",
            "bft_node_results": [],
            "bft_confidence_tier": "LOW",
            "script": None,
            "error": repr(exc),
        }
    latency_ms = (time.monotonic() - t0) * 1000

    passed = result.get("bft_status") == "CONSENSUS_REACHED"

    try:
        router.record_result(
            url=case["url"],
            requirement=case["requirement"],
            domain=case["domain"],
            passed=passed,
            generated_code=(
                (result.get("script") or {}).get("code")
                if isinstance(result.get("script"), dict)
                else None
            ),
        )
    except Exception:
        pass

    node_results = result.get("bft_node_results") or []
    pass_count = sum(
        1 for r in node_results if isinstance(r, str) and r.endswith(":PASS")
    )
    agreement = {
        3: "3_of_3",
        2: "2_of_3",
        1: "1_of_3",
        0: "0_of_3",
    }.get(pass_count, "0_of_3")

    try:
        await adapter.close()
    except Exception:
        pass
    try:
        await bft_instr.close()
    except Exception:
        pass
    try:
        await judge_instr.close()
    except Exception:
        pass

    script_obj = result.get("script")
    generated_code = (
        script_obj.get("code", "") if isinstance(script_obj, dict) else ""
    ) or ""

    return {
        "tier": result.get("bft_confidence_tier", "LOW"),
        "bft_status": result.get("bft_status"),
        "latency_ms": latency_ms,
        "passed": passed,
        "agreement": agreement,
        "judge_approved": result.get("bft_status") != "JUDGE_REJECTED",
        "generated_code": generated_code,
    }


async def _run_one_case_original_generator(
    case: dict[str, str],
    case_idx: int,
    total: int,
) -> dict[str, Any]:
    """Drive Sprint 2 GeneratorAgent directly, bypassing bft_generator_node."""
    from src.agents.generator import GeneratorAgent
    from src.llm.adapter import OllamaAdapter
    from src.llm.instructor_client import InstructorClient
    from src.llm.judge_client import JudgeClient

    print(f"[ORIG] Starting case {case_idx + 1}/{total}...", flush=True)

    adapter = OllamaAdapter()
    instr = InstructorClient(base_url=adapter.base_url, model=adapter.model, max_retries=1)
    judge_instr = InstructorClient(base_url=adapter.base_url, model=adapter.model, max_retries=1)
    judge = JudgeClient(judge_instr)

    state = {
        "url": case["url"],
        "requirement": case["requirement"],
        "domain": case["domain"],
        "test_plan": {
            "title": case["requirement"],
            "requirement_summary": case["requirement"],
            "estimated_complexity": "low",
            "domain": case["domain"],
            "steps": [
                {
                    "step_number": 1,
                    "description": case["requirement"],
                    "action": "visit",
                    "expected_result": "Page loads",
                    "role": "admin",
                    "preconditions": [],
                }
            ],
        },
    }

    t0 = time.monotonic()
    bft_status = "CONSENSUS_FAILED"
    judge_approved = False
    try:
        generator = GeneratorAgent(adapter=adapter, instructor_client=instr)
        script = await generator.generate(state=state)
        verdict = await judge.evaluate(
            generated_code=script.code,
            requirement=case["requirement"],
            domain=case["domain"],
        )
        judge_approved = verdict.approved
        bft_status = "CONSENSUS_REACHED" if verdict.approved else "JUDGE_REJECTED"
    except Exception as exc:
        print(f"[ORIG] case {case_idx + 1} error: {exc!r}", flush=True)
    latency_ms = (time.monotonic() - t0) * 1000

    try:
        await adapter.close()
    except Exception:
        pass
    try:
        await instr.close()
    except Exception:
        pass
    try:
        await judge_instr.close()
    except Exception:
        pass

    return {
        "tier": "MED",
        "bft_status": bft_status,
        "latency_ms": latency_ms,
        "passed": bft_status == "CONSENSUS_REACHED",
        "agreement": "1_of_3",
        "judge_approved": judge_approved,
    }


async def _run_pass(
    label: str,
    live: bool,
    router: Any | None,
    use_original_generator: bool = False,
) -> dict[str, Any]:
    n = len(GOLDEN_DATASET)
    metrics = _empty_pass_metrics(label, n)

    per_case: list[dict[str, Any]] = []
    for i, case in enumerate(GOLDEN_DATASET):
        if live and use_original_generator:
            result = await _run_one_case_original_generator(case, i, n)
        elif live and router is not None:
            result = await _run_one_case_live(case, router, i, n)
        else:
            result = await _run_one_case_dry(case)
        per_case.append(result)
        if not result["passed"]:
            print(f"\n[FAIL CASE {i+1}]")
            print(f"  requirement: {case.get('requirement_text', case.get('requirement', ''))[:100]}")
            print(f"  judge_approved: {result.get('judge_approved')}")
            print(f"  rejection_reason: {result.get('rejection_reason', 'N/A')}")
            print(f"  issues_found: {result.get('issues_found', [])}")
            print(f"  generated_code_preview: {result.get('generated_code','')[:200]}", flush=True)

    # ── Aggregate BFT metrics ─────────────────────────────────────────────
    consensus_reached = sum(
        1 for c in per_case if c["bft_status"] == "CONSENSUS_REACHED"
    )
    consensus_failed = sum(
        1 for c in per_case if c["bft_status"] == "CONSENSUS_FAILED"
    )
    judge_rejected = sum(
        1 for c in per_case if c["bft_status"] == "JUDGE_REJECTED"
    )
    judge_approved = sum(1 for c in per_case if c["judge_approved"])

    bm = metrics["bft_metrics"]
    bm["consensus_reached_rate"] = consensus_reached / n if n else 0.0
    bm["consensus_failed_rate"] = consensus_failed / n if n else 0.0
    bm["judge_approved_rate"] = judge_approved / n if n else 0.0
    bm["judge_rejected_rate"] = judge_rejected / n if n else 0.0
    bm["avg_bft_latency_ms"] = (
        int(sum(c["latency_ms"] for c in per_case) / n) if n else 0
    )
    for c in per_case:
        bm["node_agreement_distribution"][c["agreement"]] += 1

    # ── Routing metrics ──────────────────────────────────────────────────
    rm = metrics["routing_metrics"]
    for tier_name in ("HIGH", "MED", "LOW"):
        cases_in_tier = [c for c in per_case if c["tier"] == tier_name]
        rm[f"{tier_name}_tier_pct"] = len(cases_in_tier) / n if n else 0.0
        rm["avg_latency_by_tier"][tier_name] = (
            int(sum(c["latency_ms"] for c in cases_in_tier) / len(cases_in_tier))
            if cases_in_tier
            else 0
        )

    # ── Quality metrics ──────────────────────────────────────────────────
    qm = metrics["quality_metrics"]
    qm["first_run_pass_rate"] = (
        sum(1 for c in per_case if c["passed"]) / n if n else 0.0
    )
    # Sprint 2 false-pass count is not part of this harness's inputs;
    # baseline injection (when available) goes here in a future revision.
    qm["false_pass_reduction_vs_sprint2"] = 0.0

    # ── Latency metrics ──────────────────────────────────────────────────
    lm = metrics["latency_metrics"]
    lm["avg_generation_ms"] = bm["avg_bft_latency_ms"]
    lm["worst_case_full_bft_ms"] = max(
        (c["latency_ms"] for c in per_case if c["tier"] == "LOW"),
        default=0,
    )

    metrics["per_case"] = per_case
    return metrics


async def main(live: bool, use_original_generator: bool = False) -> None:
    router: Any | None = None
    if live:
        _check_ollama()
        # Lazy import — keeps dry mode runnable without sqlite path setup quirks.
        from src.routing.adaptive_router import AdaptiveRouter
        router = AdaptiveRouter()

    pass_1 = await _run_pass("pass_1 (cold cache)", live, router, use_original_generator)

    if live and router is not None and not use_original_generator:
        db_path = router._db_path
        try:
            with sqlite3.connect(str(db_path)) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM url_test_cache"
                ).fetchone()[0]
            print(f"[CACHE] Before pass 2: {count} entries in url_test_cache", flush=True)
        except Exception as exc:
            print(f"[CACHE] Could not read {db_path}: {exc!r}", flush=True)

    pass_2 = await _run_pass("pass_2 (warm cache)", live, router, use_original_generator)

    output: dict[str, Any] = {
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(GOLDEN_DATASET),
        "mode": "live" if live else "dry",
        "all_passes": [pass_1, pass_2],
        # Headline = pass_1 (cold) for the SPRINT 3 acceptance gate
        "pass_label": pass_1["pass_label"],
        "bft_metrics": pass_1["bft_metrics"],
        # Routing headline blends cold + warm: HIGH/MED come from warm,
        # LOW from cold (since warm should have escalated everything).
        "routing_metrics": {
            "HIGH_tier_pct": pass_2["routing_metrics"]["HIGH_tier_pct"],
            "MED_tier_pct": pass_2["routing_metrics"]["MED_tier_pct"],
            "LOW_tier_pct": pass_1["routing_metrics"]["LOW_tier_pct"],
            "avg_latency_by_tier": {
                "HIGH": pass_2["routing_metrics"]["avg_latency_by_tier"]["HIGH"],
                "MED": pass_2["routing_metrics"]["avg_latency_by_tier"]["MED"],
                "LOW": pass_1["routing_metrics"]["avg_latency_by_tier"]["LOW"],
            },
        },
        "quality_metrics": pass_1["quality_metrics"],
        "latency_metrics": pass_1["latency_metrics"],
    }

    out_path = Path(__file__).parent / "sprint3_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        f"Mode={output['mode']} "
        f"consensus_reached_rate={output['bft_metrics']['consensus_reached_rate']:.2f} "
        f"first_run_pass_rate={output['quality_metrics']['first_run_pass_rate']:.2f}"
    )

    if live:
        print("SPRINT 3 LIVE MEASUREMENT COMPLETE")
        print(
            f"Pass 1 avg_bft_latency: {pass_1['bft_metrics']['avg_bft_latency_ms']}ms | "
            f"Pass 2 HIGH_tier_pct: {pass_2['routing_metrics']['HIGH_tier_pct'] * 100:.0f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Drive real Ollama (slow). Without this flag, dry-mode synthetic metrics are emitted.",
    )
    parser.add_argument(
        "--use-original-generator",
        action="store_true",
        help="Bypass bft_generator_node; route directly to Sprint 2 GeneratorAgent for diagnosis.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.live, args.use_original_generator))
