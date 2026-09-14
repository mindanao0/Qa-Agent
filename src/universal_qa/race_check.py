"""Race-condition swarm testing wired into UniversalQAAgent (production, opt-in).

Builds a scenario set from the real pages discovered by Phase 2/3 and runs it
through the Sprint 10/12 ``RaceConditionSwarm`` in an isolated browser
session, returning a JSON-serializable report. Mirrors the
``src/universal_qa/coverage/crosscheck.py`` pattern (PR #5): a self-contained
async entry point that owns its own browser lifecycle and never touches the
agent's main ``page``/``context``.

Safety scoping (see CLAUDE.md "Sprint 12" notes and the PR description) —
read this before changing ``allow_destructive``:

  Sprint 12's genuine conflict scenarios (concurrent same-username register)
  were hand-built for ONE known target (a Conduit REST API) with a known,
  safe-to-race request body, and required explicit user authorization for
  the resulting throwaway ``race_<random>`` accounts. UniversalQAAgent runs
  against ARBITRARY third-party sites it has never seen before — there is no
  safe, generic way to synthesize an analogous stateful/destructive race
  scenario (what body to POST? what does "safe to duplicate" even mean on an
  unknown site?) without risking real side effects: duplicate signups, spam
  submissions, accidental purchases, etc. That is exactly the hazard Sprint
  12 needed sign-off for, and there is no user in the loop to authorize it
  during an automated multi-site eval run.

  So: the ONLY scenario set this module ever builds is concurrent, read-only
  ``http_get`` requests against distinct already-discovered page URLs — safe
  because GET is not expected to mutate state, and because Sprint 10/11
  already established that an honest 0-conflicts result on read-only
  scenarios is a valid, non-vacuous outcome (see CLAUDE.md Sprint 10 notes).

  ``allow_destructive`` is accepted and threaded through (mirroring
  ``ExplorerConfig.allow_destructive``) so the calling code has a real gate
  to plumb, but it currently changes no behaviour here — it only produces a
  log line explaining why. This is a deliberate safety decision, not a
  placeholder bug: implementing genuine destructive-scenario synthesis for
  arbitrary sites would need site-specific knowledge (and almost certainly a
  human-authorized target/credential set) this generic production path does
  not have.
"""
from __future__ import annotations

from typing import Iterable

from loguru import logger
from playwright.async_api import Browser, async_playwright

from src.race.detector import ConflictDetector
from src.race.interleaving_recorder import InterleavingRecorder
from src.race.swarm import RaceConditionSwarm, RaceResult, RaceScenario

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]


def build_safe_scenarios(
    urls: Iterable[str],
    *,
    agents_per_scenario: int = 2,
    overlap_ms: int = 200,
    max_scenarios: int = 5,
) -> list[RaceScenario]:
    """Build read-only concurrent-GET race scenarios from discovered URLs.

    Deduplicates (order-preserving), caps at ``max_scenarios``, and clamps
    ``agents_per_scenario`` to >= 2 (a race needs at least 2 concurrent
    agents). Every scenario uses ``action="http_get"`` and
    ``expected_safe=True`` — see the module docstring for why this is the
    only scenario shape produced here.
    """
    agents = max(2, agents_per_scenario)
    seen: set[str] = set()
    scenarios: list[RaceScenario] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        scenarios.append(RaceScenario(
            scenario_id=f"race_get_{len(scenarios) + 1}",
            description=f"{agents} agents concurrently GET {url}",
            agents=agents,
            action="http_get",
            target_url=url,
            overlap_ms=overlap_ms,
            expected_safe=True,
        ))
        if len(scenarios) >= max_scenarios:
            break
    return scenarios


async def run_scenarios(scenarios: list[RaceScenario], browser: Browser) -> dict:
    """Run pre-built scenarios against an already-open ``Browser``.

    Never raises: a scenario that errors (navigation failure, timeout, ...)
    is captured as a conflict-found result with an ``error_summary``, the
    same posture ``measure_sprint12.main`` takes, so one bad URL never
    aborts the rest of the check.
    """
    swarm = RaceConditionSwarm()
    detector = ConflictDetector()
    results: list[RaceResult] = []
    patterns: list[str] = []

    for scenario in scenarios:
        recorder = InterleavingRecorder()
        try:
            result = await swarm.run(scenario, browser, recorder)
        except Exception as exc:
            logger.warning(f"race_check: scenario {scenario.scenario_id} failed: {exc!r}")
            result = RaceResult(
                scenario_id=scenario.scenario_id, conflict_found=True,
                interleaving=[], error_summary=str(exc), duration_ms=0.0,
            )
        results.append(result)
        pattern = recorder.pattern()
        if pattern:
            patterns.append(f"{scenario.scenario_id}: {pattern}")

    analysis = detector.analyze(results)
    return {
        "scenarios_tested": analysis["total_scenarios"],
        "conflicts_found": analysis["conflicts_found"],
        "conflict_rate": analysis["conflict_rate"],
        "scenario_set": "read_only_safe",
        "results": [r.model_dump() for r in results],
        "interleaving_patterns": patterns,
    }


async def run_race_check(
    urls: Iterable[str],
    *,
    headless: bool = True,
    agents_per_scenario: int = 2,
    overlap_ms: int = 200,
    max_scenarios: int = 5,
    allow_destructive: bool = False,
) -> dict:
    """Build the safe scenario set and run it in a fresh, isolated browser.

    ``allow_destructive`` is accepted for forward-compatible parity with
    ``ExplorerConfig.allow_destructive`` but has no effect yet — see the
    module docstring.
    """
    if allow_destructive:
        logger.warning(
            "race_check: allow_destructive=True has no effect yet — there is no "
            "safe, generic way to synthesize destructive race scenarios against "
            "an arbitrary third-party site. Only read-only concurrent GET "
            "scenarios are run. See src/universal_qa/race_check.py."
        )

    scenarios = build_safe_scenarios(
        urls,
        agents_per_scenario=agents_per_scenario,
        overlap_ms=overlap_ms,
        max_scenarios=max_scenarios,
    )
    if not scenarios:
        return {
            "scenarios_tested": 0, "conflicts_found": 0, "conflict_rate": 0.0,
            "scenario_set": "read_only_safe", "results": [], "interleaving_patterns": [],
            "note": "no URLs available to race",
        }

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
        try:
            report = await run_scenarios(scenarios, browser)
        finally:
            await browser.close()

    report["allow_destructive_requested"] = allow_destructive
    return report


__all__ = ["build_safe_scenarios", "run_race_check", "run_scenarios"]
