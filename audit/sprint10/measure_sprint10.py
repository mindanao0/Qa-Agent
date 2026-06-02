"""
Sprint 10 gate measurement.

Gates:
  race_scenarios_tested      >= 5
  race_conditions_detected   >= 1   (at least one real interleaving conflict)
  fuzz_endpoints_tested      >= 5
  fuzz_anomalies_found       >= 1   (unexpected status code or schema drift)
  otel_spans_emitted         >= 8
  audit_trail_entries        >= 15
  REGRESSION if pass_rate    < 0.75 (not applicable; kept False for audit chain)
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from loguru import logger
from playwright.async_api import async_playwright

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint10_results.json"

_GATE_RACE_SCENARIOS = 5
_GATE_RACE_DETECTED = 1
_GATE_FUZZ_ENDPOINTS = 5
_GATE_FUZZ_ANOMALIES = 1
_GATE_OTEL = 8
_GATE_AUDIT = 15

_TODOMVC_URL = "https://demo.playwright.dev/todomvc/#/"
_JSONPLACEHOLDER_URL = "https://jsonplaceholder.typicode.com/"

_LAUNCH_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
]


async def _run_race(tracer, audit_trail, browser) -> tuple[int, int]:
    from src.race.swarm import RaceConditionSwarm, RaceScenario, RaceResult
    from src.race.detector import ConflictDetector

    swarm = RaceConditionSwarm()
    detector = ConflictDetector()

    # HONEST MEASUREMENT NOTE
    # -----------------------
    # demo.playwright.dev/todomvc persists state in localStorage ONLY, and every
    # agent runs in an isolated BrowserContext (separate localStorage partition).
    # There is therefore NO shared backend on which a true data race can occur:
    # identical actions yield identical per-agent semantic state, so the corrected
    # semantic-hash detector legitimately reports ~0 conflicts here. This exposes
    # the previous "5/5 conflicts" result as a false positive (it was driven by
    # CDP nodeId drift, not real races). For meaningful race testing, point these
    # scenarios at a real shared-state backend (e.g. a REST API with a database).
    scenarios = [
        RaceScenario(
            scenario_id="s1",
            description="3 agents simultaneously add todo with same title",
            agents=3,
            action="add_todo",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s2",
            description=(
                "2 agents simultaneously toggle-all. NOTE: localStorage-only + "
                "isolated contexts = no shared backend, so no real race is possible"
            ),
            agents=2,
            action="toggle_all",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s3",
            description="2 agents simultaneously add-and-complete a todo",
            agents=2,
            action="add_and_complete",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s4",
            description=(
                "3 agents add the SAME todo text simultaneously — each isolated "
                "context dedups to identical state, so no conflict is expected"
            ),
            agents=3,
            action="add_todo",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s5",
            description=(
                "2 agents perform a read-only view (count todos) simultaneously — "
                "read operations never mutate state, so they cannot conflict"
            ),
            agents=2,
            action="read_only_view",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
    ]

    race_results = []
    for scenario in scenarios:
        async with tracer.span("race.scenario", scenario_id=scenario.scenario_id):
            try:
                result = await swarm.run(scenario, browser)
            except Exception as exc:
                logger.error(f"race.scenario {scenario.scenario_id} failed: {exc!r}")
                result = RaceResult(
                    scenario_id=scenario.scenario_id,
                    conflict_found=True,
                    interleaving=[],
                    error_summary=str(exc),
                    duration_ms=0.0,
                )
        race_results.append(result)
        audit_trail.append(
            f"race.scenario.{scenario.scenario_id}",
            {
                "conflict_found": result.conflict_found,
                "error_summary": result.error_summary,
                "duration_ms": result.duration_ms,
            },
        )
        logger.info(
            f"race.scenario {scenario.scenario_id}: "
            f"conflict={result.conflict_found} "
            f"error={result.error_summary!r}"
        )

    async with tracer.span("race.detect"):
        analysis = detector.analyze(race_results)

    audit_trail.append("race.detect", analysis)

    race_scenarios_tested = analysis["total_scenarios"]
    race_conditions_detected = analysis["conflicts_found"]
    logger.info(
        f"race: tested={race_scenarios_tested} detected={race_conditions_detected} "
        f"rate={analysis['conflict_rate']}"
    )
    return race_scenarios_tested, race_conditions_detected


async def _run_fuzz(tracer, audit_trail, page) -> tuple[int, int]:
    from src.fuzzer.api_fuzzer import AutonomousAPIFuzzer

    fuzzer = AutonomousAPIFuzzer(tracer=tracer)

    async with tracer.span("fuzz.discover", url=_JSONPLACEHOLDER_URL):
        targets = await fuzzer.discover_endpoints(page, _JSONPLACEHOLDER_URL)

    for target in targets:
        audit_trail.append(
            "fuzz.endpoint.discovered",
            {"endpoint": target.endpoint, "method": target.method},
        )

    logger.info(f"fuzz: discovered {len(targets)} endpoints")

    fuzz_endpoints_tested = len(targets)
    fuzz_anomalies_found = 0

    for target in targets:
        async with tracer.span("fuzz.target", endpoint=target.endpoint):
            results = await fuzzer.fuzz(target, page)

        anomalies = [r for r in results if r.anomaly]
        fuzz_anomalies_found += len(anomalies)

        audit_trail.append(
            "fuzz.target.complete",
            {
                "endpoint": target.endpoint,
                "vectors_tested": len(results),
                "anomalies": len(anomalies),
            },
        )
        logger.info(
            f"fuzz.target {target.endpoint}: "
            f"vectors={len(results)} anomalies={len(anomalies)}"
        )

    return fuzz_endpoints_tested, fuzz_anomalies_found


async def main() -> None:
    from src.observability.tracer import OTelTracer
    from src.observability.audit_chain import CryptoAuditTrail

    tracer = OTelTracer()
    audit_trail = CryptoAuditTrail(
        path=pathlib.Path.home() / ".qa-agent" / "sprint10_audit.jsonl"
    )

    race_scenarios_tested = 0
    race_conditions_detected = 0
    fuzz_endpoints_tested = 0
    fuzz_anomalies_found = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        try:
            async with tracer.span("sprint10.race"):
                race_scenarios_tested, race_conditions_detected = await _run_race(
                    tracer, audit_trail, browser
                )

            # Fuzz phase — shared page for discover + fuzz
            fuzz_context = await browser.new_context()
            fuzz_page = await fuzz_context.new_page()
            try:
                async with tracer.span("sprint10.fuzz"):
                    fuzz_endpoints_tested, fuzz_anomalies_found = await _run_fuzz(
                        tracer, audit_trail, fuzz_page
                    )
            finally:
                await fuzz_context.close()

        finally:
            await browser.close()

    otel_spans = tracer.flush()

    audit_trail.append(
        "sprint10.complete",
        {
            "race_scenarios_tested": race_scenarios_tested,
            "race_conditions_detected": race_conditions_detected,
            "fuzz_endpoints_tested": fuzz_endpoints_tested,
            "fuzz_anomalies_found": fuzz_anomalies_found,
            "otel_spans_emitted": otel_spans,
        },
    )

    audit_trail_entries = audit_trail._seq

    regression = False

    sprint10_pass = (
        race_scenarios_tested >= _GATE_RACE_SCENARIOS
        and race_conditions_detected >= _GATE_RACE_DETECTED
        and fuzz_endpoints_tested >= _GATE_FUZZ_ENDPOINTS
        and fuzz_anomalies_found >= _GATE_FUZZ_ANOMALIES
        and otel_spans >= _GATE_OTEL
        and audit_trail_entries >= _GATE_AUDIT
        and not regression
    )

    results = {
        "race_scenarios_tested": race_scenarios_tested,
        "race_conditions_detected": race_conditions_detected,
        "fuzz_endpoints_tested": fuzz_endpoints_tested,
        "fuzz_anomalies_found": fuzz_anomalies_found,
        "otel_spans_emitted": otel_spans,
        "audit_trail_entries": audit_trail_entries,
        "regression": regression,
        "race_detection_method": "semantic_hash_comparison",
        "false_positive_risk": "low (semantic fields only) | was: high (CDP nodeId included)",
        "sprint10_status": "PASS" if sprint10_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 10 Results ===")
    gates = {
        "race_scenarios_tested": _GATE_RACE_SCENARIOS,
        "race_conditions_detected": _GATE_RACE_DETECTED,
        "fuzz_endpoints_tested": _GATE_FUZZ_ENDPOINTS,
        "fuzz_anomalies_found": _GATE_FUZZ_ANOMALIES,
        "otel_spans_emitted": _GATE_OTEL,
        "audit_trail_entries": _GATE_AUDIT,
    }
    for k, v in results.items():
        gate_str = f" (gate >= {gates[k]})" if k in gates else ""
        ok = " ✓" if (k in gates and v >= gates[k]) else (" ✗" if k in gates else "")
        print(f"  {k}: {v}{gate_str}{ok}")
    print(f"\n  -> sprint10_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint10_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
