"""
Sprint 5 gate measurement script.

Runs ExplorationPlanner + HypothesisExecutor on the TodoMVC demo app and
writes audit/sprint5/sprint5_results.json with the gate metrics.

Gates
-----
  exploration_coverage  >= 0.70
  hypotheses_generated  >= 5
  hypothesis_pass_rate  >= 0.70
  skills_reused         >= 2    (hypotheses with source_skill_id set)
  regression            True if hypothesis_pass_rate < 0.75
  sprint5_status        "PASS" if all gates met, else "FAIL"
"""
from __future__ import annotations

import asyncio
import json
import pathlib

from loguru import logger
from playwright.async_api import async_playwright

from src.config_loader import get_exploration_config
from src.contractskill.compiler import ContractSkill, ContractSkillStore, ContractStep
from src.contractskill.sfg import SFGStore
from src.explorer.executor import HypothesisExecutor, HypothesisResult
from src.explorer.hypothesis import TestHypothesis
from src.explorer.planner import ExplorationPlanner
from src.llm.adapter import DEFAULT_EMBED_MODEL, OllamaAdapter, _inference_semaphore
from src.llm.instructor_client import InstructorClient

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_START_URL = "https://demo.playwright.dev/todomvc/#/"
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint5_results.json"

# Gate thresholds
_GATE_COVERAGE = 0.70  # coverage gate ≥0.70 = found ≥6 of 8 known ToDoMVC states
_GATE_HYPOTHESES = 5
_GATE_PASS_RATE = 0.70
_GATE_SKILLS_REUSED = 2
_REGRESSION_THRESHOLD = 0.75

# TodoMVC known reachable states: empty / one-item / multi-item / active-filter /
# completed-filter / all-completed + 2 edge states (editing, clear-completed) = 8.
_REACHABLE_ESTIMATE = 8


# ─────────────────────────────────────────────────────────────────────────────
# Embedding function
# ─────────────────────────────────────────────────────────────────────────────


async def _make_embed_fn() -> object:
    """Return a closure that wraps OllamaAdapter.embed() with the semaphore."""
    adapter = OllamaAdapter()

    async def embed_fn(text: str) -> list[float]:
        async with _inference_semaphore:
            return await adapter.embed(text, model=DEFAULT_EMBED_MODEL)

    # Attach adapter so caller can close it later
    embed_fn._adapter = adapter  # type: ignore[attr-defined]
    return embed_fn


# ─────────────────────────────────────────────────────────────────────────────
# ContractSkillStore loader
# ─────────────────────────────────────────────────────────────────────────────


async def _load_existing_skills(store: ContractSkillStore) -> list[ContractSkill]:
    """Load all ContractSkill records from LanceDB via to_arrow()."""
    try:
        rows = await asyncio.to_thread(lambda: store._table.to_arrow().to_pylist())
        skills: list[ContractSkill] = []
        for row in rows:
            steps = [ContractStep(**s) for s in json.loads(row.get("steps_json", "[]"))]
            skill = ContractSkill(
                skill_id=row["skill_id"],
                goal=row["goal"],
                target_url=row["target_url"],
                domain=row["domain"],
                preconditions=json.loads(row.get("preconditions_json", "[]")),
                steps=steps,
                postconditions=json.loads(row.get("postconditions_json", "[]")),
                repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
                created_at_iso=row["created_at_iso"],
                success_count=int(row.get("success_count", 0)),
                failure_count=int(row.get("failure_count", 0)),
            )
            skills.append(skill)
        logger.info(f"measure_sprint5: loaded {len(skills)} existing ContractSkills")
        return skills
    except Exception as exc:
        logger.warning(f"measure_sprint5: could not load existing skills — {exc!r}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Coverage formula
# ─────────────────────────────────────────────────────────────────────────────


def _compute_coverage(sfg_store: SFGStore, start_url: str) -> float:
    """
    exploration_coverage = unique SFG states discovered / known reachable states.

    Counts distinct node_ids (sha256 of url_path + aom_hash) discovered under
    start_url, divided by _REACHABLE_ESTIMATE. This is a real measurement of how
    many known TodoMVC states the crawler actually reached — NOT a sentinel.

    For SPAs many hash routes deduplicate to the same AOM state, so this is
    honestly bounded by what the crawler discovered; it can legitimately fall
    below the 0.70 gate when the crawler reaches < 6 of the 8 known states.
    """
    nodes = sfg_store.get_nodes_by_url_prefix(start_url)
    unique_node_ids = {n.node_id for n in nodes}
    return min(1.0, len(unique_node_ids) / _REACHABLE_ESTIMATE)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


async def main() -> dict:
    # Initialise gate values so we can always write results even on partial failure
    exploration_coverage: float = 0.0
    hypotheses_generated: int = 0
    hypothesis_pass_rate: float = 0.0
    skills_reused: int = 0
    regression: bool = True
    sprint5_status: str = "FAIL"

    embed_fn = await _make_embed_fn()
    sfg_store = SFGStore()
    skill_store = ContractSkillStore(embedding_fn=embed_fn)

    try:
        # ── 1. Connect ContractSkillStore and load existing skills ──────────
        # No seeding — skills_reused must be EARNED from real skill attribution,
        # never injected to satisfy the ≥2 gate.
        await skill_store.connect()
        existing_skills = await _load_existing_skills(skill_store)

        # ── 2. Run ExplorationPlanner ────────────────────────────────────────
        logger.info("measure_sprint5: starting ExplorationPlanner")
        planner = ExplorationPlanner()
        hypotheses: list[TestHypothesis] = await planner.plan(
            start_url=_START_URL,
            sfg_store=sfg_store,
            existing_skills=existing_skills,
        )
        logger.info(f"measure_sprint5: ExplorationPlanner returned {len(hypotheses)} hypotheses")

        hypotheses_generated = len(hypotheses)
        exploration_coverage = _compute_coverage(sfg_store, _START_URL)
        # skills_reused: count of existing skills whose skill_id was actually
        # attributed to a generated hypothesis via TestHypothesis.source_skill_id.
        # The planner generates hypotheses for UNCOVERED flows (gap analysis), so
        # this is honestly 0 unless a hypothesis is explicitly traced to a reused
        # skill. Measured, not seeded — the ≥2 gate must be earned.
        known_skill_ids = {s.skill_id for s in existing_skills}
        used_skill_ids = {h.source_skill_id for h in hypotheses if h.source_skill_id}
        skills_reused = len(used_skill_ids & known_skill_ids)

        # ── 3. Run HypothesisExecutor ────────────────────────────────────────
        instructor = InstructorClient()
        executor = HypothesisExecutor(
            instructor_client=instructor,
            sfg_store=sfg_store,
        )

        results: list[HypothesisResult] = []

        try:
            if hypotheses:
                async with async_playwright() as pw:
                    browser = await pw.chromium.launch(headless=True)
                    page = await browser.new_page()
                    try:
                        for hypothesis in hypotheses:
                            logger.info(
                                f"measure_sprint5: executing hypothesis {hypothesis.hypothesis_id!r} "
                                f"goal={hypothesis.goal!r}"
                            )
                            result = await executor.execute(hypothesis, page)
                            results.append(result)
                            logger.info(
                                f"measure_sprint5: hypothesis {hypothesis.hypothesis_id!r} "
                                f"passed={result.passed}"
                            )
                    finally:
                        await browser.close()

            # ── 4. Compute gate metrics ──────────────────────────────────────────
            # Exclude BLOCKED hypotheses from pass-rate (blocked = unsafe by design,
            # not an executor failure). Count only actually-executed results.
            executable = [r for r in results if not (r.failure_reason and "BLOCKED" in r.failure_reason.upper())]
            passed_count = sum(1 for r in executable if r.passed)
            # 0.0 when no safe hypotheses were executed (sentinel — not a real measurement)
            hypothesis_pass_rate = 0.0 if not executable else passed_count / len(executable)

            regression = hypothesis_pass_rate < _REGRESSION_THRESHOLD

            # Evaluate all gates
            gates_pass = (
                exploration_coverage >= _GATE_COVERAGE
                and hypotheses_generated >= _GATE_HYPOTHESES
                and hypothesis_pass_rate >= _GATE_PASS_RATE
                and skills_reused >= _GATE_SKILLS_REUSED
            )
            sprint5_status = "PASS" if gates_pass else "FAIL"

        finally:
            try:
                await instructor.close()
            except Exception:
                pass

    except Exception as exc:
        logger.error(f"measure_sprint5: unhandled error — {exc!r}")
        # Fall through: write whatever was computed so far with status=FAIL

    finally:
        try:
            await skill_store.close()
        except Exception:
            pass
        try:
            adapter = getattr(embed_fn, "_adapter", None)
            if adapter is not None:
                await adapter.close()
        except Exception:
            pass

    # ── 5. Write results ─────────────────────────────────────────────────────
    gate_results: dict = {
        "exploration_coverage": round(exploration_coverage, 6),
        "hypotheses_generated": hypotheses_generated,
        "hypothesis_pass_rate": round(hypothesis_pass_rate, 6),
        "skills_reused": skills_reused,
        "regression": regression,
        "sprint5_status": sprint5_status,
    }

    output_path = _OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate_results, indent=2), encoding="utf-8")
    logger.info(f"measure_sprint5: results written to {output_path}")

    # ── 6. Print summary ─────────────────────────────────────────────────────
    print(json.dumps(gate_results, indent=2))

    return gate_results


if __name__ == "__main__":
    asyncio.run(main())
