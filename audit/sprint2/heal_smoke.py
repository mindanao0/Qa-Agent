"""
Sprint 2 Heal Smoke Test
========================

Proves the FUZZY_HIT -> save_healed_experience round-trip end-to-end against
a live Playwright browser + LanceDB, per the recommendation in
audit/sprint2/SPRINT2_FINAL_LOG.md section 7.

Schema note
-----------
The original spec proposed a flat CSS-style entry
({"selector": "[data-testid='todo-input']", ...}).  The actual
src/healing/fuzzy_matcher.py FuzzyMatcher scores the JSON *keys*
(not a "selector" field) and reads entry["healed"] for the returned
candidate.  src/healing/ai_healer.py._eval_locator only parses Playwright
DSL ("get_by_*"), not raw CSS.  This script therefore seeds an entry whose
key is the Playwright DSL form ``page.get_by_test_id("todo-input")`` and
whose ``healed`` value is a locator known to work on TodoMVC.  The intent
of the spec (HIGH fuzzy hit -> save -> retrieve) is preserved.

Exits 0 on success, non-zero on first failed assertion.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure project root on sys.path when run from arbitrary CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filelock import FileLock
from playwright.async_api import TimeoutError as PWTimeoutError
from playwright.async_api import async_playwright

from src.healing.ai_healer import ActionContext, AIHealer
from src.healing.fuzzy_matcher import FuzzyMatcher
from src.healing.state_validator import StateValidator
from src.llm.adapter import OllamaAdapter
from src.memory.episodic_store import EpisodicStore
from src.perception.aom_extractor import AOMExtractor


REPO_PATH = PROJECT_ROOT / "locators" / "locators.json"
TODOMVC_URL = "https://demo.playwright.dev/todomvc"

REAL_KEY = 'page.get_by_test_id("todo-input")'
BROKEN_LOCATOR = 'page.get_by_test_id("todo-input-BROKEN")'
EXPECTED_HEALED = "page.get_by_role('textbox', name='What needs to be done?')"


def ensure_seed() -> float:
    """Idempotent: seed the locator repo if the key is missing.  Returns JW score."""
    lock = FileLock(str(REPO_PATH.with_suffix(".lock")), timeout=10)
    with lock:
        data = json.loads(REPO_PATH.read_text(encoding="utf-8"))
        if REAL_KEY not in data:
            data[REAL_KEY] = {
                "original": REAL_KEY,
                "healed": EXPECTED_HEALED,
                "confidence": 0.9,
                "method": "smoke_test_seed",
                "url_pattern": TODOMVC_URL,
                "heal_count": 0,
                "last_updated": None,
                "reasoning": "Sprint 2 Heal Smoke Test seed",
                "role": "textbox",
                "accessible_name": "What needs to be done?",
            }
            tmp = REPO_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, REPO_PATH)

    fm = FuzzyMatcher(locator_repo_path=REPO_PATH)
    return fm._jaro_winkler(BROKEN_LOCATOR, REAL_KEY)


async def main() -> int:
    print("=" * 72)
    print("SPRINT 2 HEAL SMOKE TEST")
    print("=" * 72)

    # ── Pre-flight: seed + JW check ─────────────────────────────────────────
    jw_score = ensure_seed()
    print(f"[pre] JW similarity({BROKEN_LOCATOR!r}, {REAL_KEY!r}) = {jw_score:.4f}")
    assert jw_score >= 0.85, f"JW score {jw_score:.4f} < 0.85 - adjust the pair"

    fm_check = FuzzyMatcher(locator_repo_path=REPO_PATH)
    candidates = fm_check.find_candidates(BROKEN_LOCATOR, 0.85)
    assert len(candidates) >= 1, "FuzzyMatcher returned no HIGH candidates"
    assert candidates[0].confidence_tier == "HIGH"
    print(f"[pre] FuzzyMatcher HIGH candidates: {len(candidates)}")
    print(f"[pre] top candidate healed: {candidates[0].candidate_locator}")

    # ── Wire dependencies ───────────────────────────────────────────────────
    adapter = OllamaAdapter()

    async def embed_fn(text: str) -> list[float]:
        return await adapter.embed(text)

    store = EpisodicStore(embedding_fn=embed_fn)
    await store.connect()

    fuzzy = FuzzyMatcher(locator_repo_path=REPO_PATH)
    aom = AOMExtractor()
    healer = AIHealer(
        fuzzy_matcher=fuzzy,
        aom_extractor=aom,
        episodic_store=store,
        session_id="heal_smoke_session",
    )
    validator = StateValidator()

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    page = await browser.new_page()

    try:
        # ── STEP 2: live heal cycle ─────────────────────────────────────────
        await page.goto(TODOMVC_URL, wait_until="domcontentloaded")
        await page.wait_for_load_state("networkidle", timeout=10_000)

        pre = await validator.capture_pre_state(page)
        print(f"[step2] pre-state captured | url={pre.url!r}")

        # Simulate a locator failure - the broken test_id does not exist.
        broken_failed = False
        try:
            await page.get_by_test_id("todo-input-BROKEN").wait_for(
                state="visible", timeout=5_000
            )
        except PWTimeoutError:
            broken_failed = True
            print("[step2] expected TimeoutError on broken locator - ok")

        assert broken_failed, "Broken locator unexpectedly resolved - smoke test invalid"

        action_context = ActionContext(
            action_type="fill",
            domain="crud",
            page_url=page.url,
            element_description="todo input box",
        )
        result = await healer.heal(
            page=page,
            failed_locator=BROKEN_LOCATOR,
            action_context=action_context,
        )

        print(f"[step2] HealResult.strategy       = {result.strategy}")
        print(f"[step2] HealResult.locator        = {result.locator}")
        print(f"[step2] HealResult.attempt_count  = {result.attempt_count}")
        print(f"[step2] HealResult.latency_ms     = {result.latency_ms}")
        print(f"[step2] HealResult.failure_sig    = {result.failure_signature}")

        assert result.strategy == "FUZZY_HIT", (
            f"Expected strategy=FUZZY_HIT, got {result.strategy}"
        )
        assert result.locator == EXPECTED_HEALED, (
            f"Expected locator={EXPECTED_HEALED!r}, got {result.locator!r}"
        )

        # Verify a row exists in healed_experiences via retrieve_for_planning.
        hits = await store.retrieve_for_planning(
            query_text=f"{BROKEN_LOCATOR} fill",
            domain="crud",
            failure_signature=result.failure_signature,
            limit=5,
        )
        healed_hits = [h for h in hits if h.memory_type == "healed_experience"]
        assert len(healed_hits) >= 1, (
            f"No healed_experience rows retrievable; got {len(hits)} hits total"
        )
        print(
            f"HEAL SMOKE: FUZZY_HIT confirmed, healed_experiences count={len(healed_hits)}"
        )

        # ── STEP 3: cross-session retrieval ─────────────────────────────────
        await store.close()
        store2 = EpisodicStore(embedding_fn=embed_fn)
        await store2.connect()
        hits2 = await store2.retrieve_for_planning(
            query_text=f"{BROKEN_LOCATOR} fill",
            domain="crud",
            failure_signature=result.failure_signature,
            limit=5,
        )
        healed_hits2 = [h for h in hits2 if h.memory_type == "healed_experience"]
        assert len(healed_hits2) >= 1, (
            "Cross-session retrieval returned no healed_experience rows"
        )
        print("MEMORY PERSISTENCE: cross-session retrieval confirmed")
        await store2.close()

    finally:
        await page.close()
        await browser.close()
        await pw.stop()
        try:
            await adapter.close()
        except Exception:
            pass

    print("=" * 72)
    print("SMOKE TEST: PASS")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
