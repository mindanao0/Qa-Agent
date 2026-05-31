# tests/test_instructor_client.py
"""
Integration tests for InstructorClient.

These tests hit a live Ollama instance.
Run with: uv run pytest tests/test_instructor_client.py -v -m integration
Pre-requisite: Ollama serving qwen2.5-coder:7b-instruct-q4_K_M at localhost:11434.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.llm.schemas import HealedLocator, PlaywrightScript, TestPlan, TestStep


# ── Helpers ───────────────────────────────────────────────────────────────────


def _levenshtein(s1: str, s2: str) -> int:
    """Simple O(n*m) Levenshtein distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> InstructorClient:
    return InstructorClient(max_retries=3)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simple_plan_generation_returns_valid_pydantic_model(
    client: InstructorClient,
) -> None:
    """
    Given a simple requirement, create_structured() must return a TestPlan
    instance (not a dict, not a str) with at least one step.
    """
    prompt = (
        "Generate a test plan for: verify the login button is visible on the home page. "
        "Target URL: https://example.com. Role: admin. Domain: authentication."
    )
    result = await client.create_structured(prompt, TestPlan)

    assert isinstance(result, TestPlan), f"Expected TestPlan, got {type(result)}"
    assert isinstance(result.steps, list)
    assert len(result.steps) >= 1, "TestPlan must have at least one step"
    assert isinstance(result.steps[0], TestStep)
    assert result.estimated_complexity in ("low", "medium", "high")
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_schema_triggers_retry_then_succeeds(
    client: InstructorClient,
) -> None:
    """
    Instructor should retry when the model returns output that partially
    misses required fields.  We verify success after retries, not that
    retries happened (Ollama is usually good enough to pass on first try
    for simple schemas; this test confirms retries don't break anything).
    """
    prompt = (
        "Produce a HealedLocator for this failure: "
        "locator page.get_by_role('button', name='Submit') timed out on https://example.com. "
        "Suggest a replacement using get_by_role or get_by_label."
    )
    result = await client.create_structured(
        prompt,
        HealedLocator,
        temperature=0.0,
    )

    assert isinstance(result, HealedLocator), f"Expected HealedLocator, got {type(result)}"
    assert result.method in ("fuzzy", "ai", "vlm")
    assert 0.0 <= result.confidence <= 1.0
    assert result.healed  # non-empty
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semaphore_blocks_concurrent_calls() -> None:
    """
    With a Semaphore(1), three concurrent create_structured() calls must
    execute serially.  Total wall time >= 2x the shortest individual call.
    """
    sem = asyncio.Semaphore(1)
    client = InstructorClient(semaphore=sem, max_retries=1)
    prompt = "Generate a minimal test plan. Target URL: https://example.com. One step only."

    call_durations: list[float] = []

    async def timed_call() -> None:
        t0 = time.monotonic()
        await client.create_structured(prompt, TestPlan)
        call_durations.append(time.monotonic() - t0)

    wall_start = time.monotonic()
    await asyncio.gather(timed_call(), timed_call(), timed_call())
    wall_elapsed = time.monotonic() - wall_start

    assert len(call_durations) == 3
    min_individual = min(call_durations)
    # With Semaphore(1), wall time must be >= 2x the fastest single call
    assert wall_elapsed >= 2 * min_individual, (
        f"Semaphore not enforcing serialisation: "
        f"wall={wall_elapsed:.2f}s min_individual={min_individual:.2f}s"
    )
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_temperature_zero_is_deterministic(client: InstructorClient) -> None:
    """
    Same prompt twice at temperature=0.0.  The two responses should be
    byte-identical or very close (Levenshtein distance <= 5 between the
    JSON-serialised outputs).
    """
    prompt = (
        "Generate a test plan for: verify the search button is visible. "
        "Target URL: https://playwright.dev/. Role: admin."
    )

    r1 = await client.create_structured(prompt, TestPlan, temperature=0.0)
    r2 = await client.create_structured(prompt, TestPlan, temperature=0.0)

    s1 = r1.model_dump_json()
    s2 = r2.model_dump_json()

    dist = _levenshtein(s1, s2)
    assert dist <= 5, (
        f"Responses not deterministic at T=0 (Levenshtein={dist}):\n"
        f"  Run 1: {s1[:200]}\n"
        f"  Run 2: {s2[:200]}"
    )
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_path_still_works_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When STRUCTURED_OUTPUT_ENGINE=legacy_repair, PlannerAgent must still
    produce a valid TestPlan using enforce_json_output (no instructor).
    """
    import os
    monkeypatch.setenv("STRUCTURED_OUTPUT_ENGINE", "legacy_repair")

    # Force cache clear so config_loader picks up the env var
    from src import config_loader
    config_loader._load_yaml.cache_clear()

    from src.config_loader import get_structured_output_engine
    assert get_structured_output_engine() == "legacy_repair"

    from src.llm.adapter import OllamaAdapter
    from src.agents.planner import PlannerAgent

    adapter = OllamaAdapter()
    planner = PlannerAgent(adapter=adapter)

    plan = await planner.plan(
        requirement="verify the search button is visible",
        url="https://playwright.dev/",
    )

    assert isinstance(plan, TestPlan)
    assert len(plan.steps) >= 1

    await adapter.close()
