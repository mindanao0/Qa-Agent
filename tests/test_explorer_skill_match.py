"""TDD for ExplorationPlanner skill attribution (Sprint 5 FIX 2).

`_match_skill` attributes a generated hypothesis to an existing ContractSkill by
genuine keyword overlap of their goal statements (no LLM call, no seeding). This
is what lets `skills_reused` be EARNED rather than injected.
"""
from __future__ import annotations

from src.contractskill.compiler import ContractSkill
from src.explorer.planner import _match_skill, _meaningful_words


def _skill(skill_id: str, goal: str) -> ContractSkill:
    return ContractSkill(
        skill_id=skill_id,
        goal=goal,
        target_url="https://demo.playwright.dev/todomvc/#/",
        domain="web_exploration",
        preconditions=[],
        steps=[],
        postconditions=[],
        repair_operators=[],
        created_at_iso="2026-06-02T00:00:00+00:00",
    )


# ── _meaningful_words ────────────────────────────────────────────────────────


def test_meaningful_words_drops_stopwords_and_short_tokens():
    # "to" (len 2) and "the"/"a" (stopwords) are dropped; content words kept.
    assert _meaningful_words("Add a todo to the list") == {"add", "todo", "list"}


def test_meaningful_words_is_case_insensitive():
    assert _meaningful_words("Complete TODO") == _meaningful_words("complete todo")


# ── _match_skill ─────────────────────────────────────────────────────────────


def test_match_skill_empty_list_returns_none():
    assert _match_skill("Add a todo item", []) is None


def test_match_skill_no_overlap_returns_none():
    skills = [_skill("k1", "Add a todo to the list")]
    assert _match_skill("Delete user account permanently", skills) is None


def test_match_skill_strong_overlap_returns_skill_id():
    skills = [_skill("k1", "Add a todo to the list")]
    # All meaningful skill words (add, todo, list) appear in the hypothesis goal.
    assert _match_skill("Add a new todo item to the list", skills) == "k1"


def test_match_skill_returns_best_overlap_among_many():
    skills = [
        _skill("k_complete", "Complete a todo"),   # shares only "todo" (ratio 0.5)
        _skill("k_add", "Add a todo to the list"),  # shares add/todo/list (ratio 1.0)
    ]
    # Both clear the threshold; the BEST overlap (k_add) must win, not the first.
    assert _match_skill("Add a brand new todo to the list", skills) == "k_add"


def test_match_skill_ignores_shared_stopwords_only():
    # The two goals share only stopwords / short tokens -> no genuine overlap.
    skills = [_skill("k1", "A an the is are")]
    assert _match_skill("To be or in on of", skills) is None


def test_match_skill_morphological_variants_match():
    # "marking ... completed" is genuinely the same intent as "mark ... complete".
    # Exact-word overlap misses it; prefix/stem relation must catch it so the
    # "complete" skill is reused, not just the "add" skill.
    skills = [_skill("k_complete", "User can mark a todo as complete")]
    assert (
        _match_skill("Check that marking a todo as completed updates its status", skills)
        == "k_complete"
    )


def test_match_skill_two_distinct_skills_get_reused():
    # Distinct hypotheses attribute to distinct skills -> skills_reused == 2.
    skills = [
        _skill("k_add", "User can add a new todo item"),
        _skill("k_complete", "User can mark a todo as complete"),
    ]
    add_hit = _match_skill("Verify that adding a new todo item updates the list", skills)
    complete_hit = _match_skill("Check that marking a todo as completed works", skills)
    assert {add_hit, complete_hit} == {"k_add", "k_complete"}


def test_match_skill_prefix_requires_min_length():
    # Short 3-letter tokens must NOT spuriously relate via prefix ("car" vs "care").
    skills = [_skill("k", "car wash service done")]
    assert _match_skill("care about cards today", skills) is None
