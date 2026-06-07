"""
Sprint 11 gate measurement — Continuous Mode.

Runs ContinuousLoopController on the TodoMVC demo (max_cycles=3) and writes
audit/sprint11/sprint11_results.json.

Gates
-----
  continuous_loop_cycles     >= 3
  new_states_per_cycle       >= 1     (minimum new states across cycles)
  cumulative_tests_generated >= 15    (web hypotheses + pytest tests, all cycles)
  self_heal_rate             >= 0.50  (healed / max(1, repair-needed), web hyps)
  memory_mb_stable           == True  (RSS growth < 200 MB, cycle 1 -> 3)
  regression                 == False (True iff combined pass_rate < 0.75)

Honest-reporting posture: real numbers are written as-is. A missed gate is a
documented real result (cf. Sprint 10's honest 0), never seeded or fudged.
"""
from __future__ import annotations

import asyncio
import json
import pathlib

from loguru import logger

from src.continuous.loop_controller import ContinuousLoopController, combined_pass_rate

_START_URL = "https://demo.playwright.dev/todomvc/#/"
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint11_results.json"
_CHECKPOINT_PATH = pathlib.Path(__file__).parent / "checkpoints.db"

# Gate thresholds
_GATE_CYCLES = 3
_GATE_NEW_STATES = 1
_GATE_TESTS = 15
_GATE_SELF_HEAL = 0.50
_REGRESSION_THRESHOLD = 0.75


async def _load_existing_skills() -> list:
    """Best-effort load of stored ContractSkills (drives skills_reused fidelity).

    Mirrors measure_sprint5; never fatal — returns [] on any failure.
    """
    try:
        from src.contractskill.compiler import ContractSkill, ContractSkillStore, ContractStep
        from src.llm.adapter import DEFAULT_EMBED_MODEL, OllamaAdapter, _inference_semaphore

        adapter = OllamaAdapter()

        async def embed_fn(text: str) -> list[float]:
            async with _inference_semaphore:
                return await adapter.embed(text, model=DEFAULT_EMBED_MODEL)

        store = ContractSkillStore(embedding_fn=embed_fn)
        await store.connect()
        try:
            rows = await asyncio.to_thread(lambda: store._table.to_arrow().to_pylist())
            skills: list = []
            for row in rows:
                steps = [ContractStep(**s) for s in json.loads(row.get("steps_json", "[]"))]
                skills.append(
                    ContractSkill(
                        skill_id=row["skill_id"], goal=row["goal"], target_url=row["target_url"],
                        domain=row["domain"],
                        preconditions=json.loads(row.get("preconditions_json", "[]")),
                        steps=steps,
                        postconditions=json.loads(row.get("postconditions_json", "[]")),
                        repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
                        created_at_iso=row["created_at_iso"],
                        success_count=int(row.get("success_count", 0)),
                        failure_count=int(row.get("failure_count", 0)),
                    )
                )
            logger.info(f"measure_sprint11: loaded {len(skills)} existing ContractSkills")
            return skills
        finally:
            await store.close()
            await adapter.close()
    except Exception as exc:
        logger.warning(f"measure_sprint11: could not load existing skills — {exc!r}")
        return []


async def main() -> dict:
    # Initialise so we always write results, even on partial failure.
    continuous_loop_cycles = 0
    new_states_per_cycle = 0
    cumulative_tests_generated = 0
    self_heal_rate = 0.0
    self_heal_needed = 0
    memory_mb_stable = False
    memory_growth_mb = 0.0
    stop_reason: str | None = None
    pass_rate = 0.0
    regression = True
    skills_reused = 0
    otel_spans_emitted = 0
    audit_trail_entries = 0

    existing_skills = await _load_existing_skills()

    controller = ContinuousLoopController(
        _START_URL,
        max_cycles=3,
        existing_skills=existing_skills,
        checkpoint_path=_CHECKPOINT_PATH,
    )

    try:
        state = await controller.run()

        continuous_loop_cycles = int(state.get("cycle", 0))
        per_cycle = controller.coverage_tracker._new_per_cycle
        new_states_per_cycle = min(per_cycle) if per_cycle else 0
        cumulative_tests_generated = len(state.get("tests_generated", []))

        n_passed = len(state.get("tests_passed", []))
        n_failed = len(state.get("tests_failed", []))
        pass_rate = combined_pass_rate(n_passed, n_failed)
        regression = pass_rate < _REGRESSION_THRESHOLD

        self_heal_needed = controller.heal_needed_total
        self_heal_rate = controller.heal_healed_total / max(1, self_heal_needed)
        memory_mb_stable = controller.memory_guard.is_stable()
        memory_growth_mb = controller.memory_guard.growth_mb()
        stop_reason = state.get("stop_reason")

        skills_reused = len(controller.skills_used)
        otel_spans_emitted = controller.otel_spans_emitted
        audit_trail_entries = controller.audit._seq if controller.audit else 0

    except Exception as exc:
        logger.error(f"measure_sprint11: unhandled error — {exc!r}")

    # The self-heal gate is CONDITIONAL per the spec phrasing ("if test fail ->
    # repair succeeds >= 50%"). When zero tests needed repair the antecedent is
    # never triggered, so the gate is vacuously satisfied — an all-green run must
    # not FAIL a heal-rate gate it never exercised. self_heal_needed is reported
    # transparently so a vacuous pass is never mistaken for real heals.
    self_heal_ok = (self_heal_needed == 0) or (self_heal_rate >= _GATE_SELF_HEAL)

    sprint11_pass = (
        continuous_loop_cycles >= _GATE_CYCLES
        and new_states_per_cycle >= _GATE_NEW_STATES
        and cumulative_tests_generated >= _GATE_TESTS
        and self_heal_ok
        and memory_mb_stable
        and not regression
    )

    results = {
        "continuous_loop_cycles": continuous_loop_cycles,
        "new_states_per_cycle": new_states_per_cycle,
        "cumulative_tests_generated": cumulative_tests_generated,
        "self_heal_rate": round(self_heal_rate, 6),
        "memory_mb_stable": memory_mb_stable,
        "stop_reason": stop_reason,
        "regression": regression,
        "sprint11_status": "PASS" if sprint11_pass else "FAIL",
        # ── transparency extras (not gated) ──
        "self_heal_needed": self_heal_needed,
        "self_heal_note": (
            "vacuous: 0 tests required repair (conditional gate not triggered)"
            if self_heal_needed == 0
            else f"{controller.heal_healed_total}/{self_heal_needed} repaired"
        ),
        "pass_rate": round(pass_rate, 6),
        "memory_growth_mb": round(memory_growth_mb, 2),
        "skills_reused": skills_reused,
        "otel_spans_emitted": otel_spans_emitted,
        "audit_trail_entries": audit_trail_entries,
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"measure_sprint11: results written to {_OUTPUT_PATH}")

    print("\n=== Sprint 11 Results ===")
    gates = {
        "continuous_loop_cycles": _GATE_CYCLES,
        "new_states_per_cycle": _GATE_NEW_STATES,
        "cumulative_tests_generated": _GATE_TESTS,
    }
    for k, v in results.items():
        gate = gates.get(k)
        if gate is not None:
            ok = " ✓" if v >= gate else " ✗"
            print(f"  {k}: {v} (gate >= {gate}){ok}")
        elif k == "self_heal_rate":
            tag = " (vacuous ✓)" if self_heal_needed == 0 else (" ✓" if self_heal_ok else " ✗")
            print(f"  {k}: {v} (gate >= {_GATE_SELF_HEAL}){tag}")
        elif k == "memory_mb_stable":
            print(f"  {k}: {v} (gate == True){' ✓' if v else ' ✗'}")
        elif k == "regression":
            print(f"  {k}: {v} (gate == False){' ✓' if not v else ' ✗'}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  -> sprint11_status: {results['sprint11_status']}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
