"""Tests for the new FuzzyMatcher class (Sprint 2, Cluster S2-A).

Note: these tests cover only the NEW FuzzyMatcher class — the legacy
fuzzy_match() function tests live elsewhere.
"""
import json

from src.healing.fuzzy_matcher import FuzzyCandidate, FuzzyMatcher


def _make_repo(tmp_path, entries: dict) -> "pathlib.Path":  # type: ignore[name-defined]
    repo = tmp_path / "locators.json"
    repo.write_text(json.dumps(entries))
    return repo


def test_exact_match_scores_1_0(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "page.get_by_role('button', name='Login')": {
                "original": "page.get_by_role('button', name='Login')",
                "healed": "page.get_by_role(\"button\", name=\"Login\")",
                "confidence": 1.0,
                "method": "fuzzy",
                "url_pattern": "https://example.com",
                "heal_count": 1,
                "last_updated": "2026-05-20T14:32:28.025876+00:00",
                "reasoning": "",
            }
        },
    )
    fm = FuzzyMatcher(repo)
    s = fm._jaro_winkler(
        "page.get_by_role('button', name='Login')",
        "page.get_by_role('button', name='Login')",
    )
    assert s == 1.0


def test_similar_css_selector_scores_above_085(tmp_path):
    fm = FuzzyMatcher(tmp_path / "empty.json")
    s = fm._jaro_winkler(
        "page.get_by_role('button', name='Login')",
        "page.get_by_role(\"button\", name=\"Login\")",
    )
    assert s > 0.85


def test_dissimilar_strings_score_below_060(tmp_path):
    fm = FuzzyMatcher(tmp_path / "empty.json")
    s = fm._jaro_winkler(
        "foobar",
        "page.get_by_test_id('totally-different')",
    )
    assert s < 0.60


def test_jaro_winkler_known_value(tmp_path):
    fm = FuzzyMatcher(tmp_path / "empty.json")
    assert abs(fm._jaro_winkler("MARTHA", "MARHTA") - 0.9611) < 0.0005


def test_find_candidates_returns_sorted_by_score(tmp_path):
    repo = _make_repo(
        tmp_path,
        {
            "page.get_by_role('button', name='Login')": {
                "original": "page.get_by_role('button', name='Login')",
                "healed": "page.get_by_role(\"button\", name=\"Login\")",
                "confidence": 1.0,
                "method": "fuzzy",
                "url_pattern": "https://example.com",
                "heal_count": 1,
                "last_updated": "2026-05-20T14:32:28.025876+00:00",
                "reasoning": "",
            },
            "page.get_by_role('button', name='Submit')": {
                "original": "page.get_by_role('button', name='Submit')",
                "healed": "page.get_by_role(\"button\", name=\"Submit\")",
                "confidence": 0.9,
                "method": "fuzzy",
                "url_pattern": "https://example.com",
                "heal_count": 1,
                "last_updated": "2026-05-20T14:32:28.025876+00:00",
                "reasoning": "",
            },
            "page.get_by_test_id('totally-different-thing')": {
                "original": "page.get_by_test_id('totally-different-thing')",
                "healed": "page.get_by_test_id(\"totally-different-thing\")",
                "confidence": 0.9,
                "method": "fuzzy",
                "url_pattern": "https://example.com",
                "heal_count": 1,
                "last_updated": "2026-05-20T14:32:28.025876+00:00",
                "reasoning": "",
            },
        },
    )
    fm = FuzzyMatcher(repo)
    # Use a low threshold so we get MEDIUM candidates too
    results = fm.find_candidates(
        "page.get_by_role('button', name='Login')", threshold=0.0
    )
    assert len(results) >= 2
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True)
    # First result is the exact-key match (the Login entry)
    assert results[0].candidate_locator == "page.get_by_role(\"button\", name=\"Login\")"
    assert results[0].similarity_score == 1.0
    assert results[0].confidence_tier == "HIGH"
    assert isinstance(results[0], FuzzyCandidate)


def test_empty_repo_returns_empty_list(tmp_path):
    fm = FuzzyMatcher(tmp_path / "nonexistent.json")
    assert fm.find_candidates("any") == []
