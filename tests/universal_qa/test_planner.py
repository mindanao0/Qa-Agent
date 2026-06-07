import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.models import TestCase
from src.universal_qa.test_planner import UniversalTestPlanner


def _make_node(url: str, title: str, pam: str, tags: list[str]):
    node = MagicMock()
    node.url = url
    node.page_title = title
    node.pam_content = pam
    node.coverage_tags = tags
    return node


def test_plan_accessibility_one_per_node():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    nodes = [
        _make_node("https://x.com/login", "Login", "login form", ["form"]),
        _make_node("https://x.com/about", "About", "static text", []),
    ]
    cases = planner._plan_accessibility(nodes, "https://x.com")
    assert len(cases) == 2
    assert all(tc.type == "accessibility" for tc in cases)
    assert all(isinstance(tc, TestCase) for tc in cases)


def test_plan_security_only_form_nodes():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    nodes = [
        _make_node("https://x.com/login", "Login", "login form", ["form"]),
        _make_node("https://x.com/about", "About", "static text", []),
    ]
    cases = planner._plan_security(nodes, "https://x.com")
    assert len(cases) == 2  # XSS + SQLi only for the form node
    assert all(tc.type == "security" for tc in cases)


def test_plan_sorts_by_priority():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    cases = [
        TestCase(title="low", type="functional", priority="low",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
        TestCase(title="high", type="functional", priority="high",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
        TestCase(title="med", type="functional", priority="medium",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
    ]
    sorted_cases = planner._sort_by_priority(cases)
    assert [tc.priority for tc in sorted_cases] == ["high", "medium", "low"]


def test_plan_security_titles_include_type():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    nodes = [_make_node("https://x.com/reg", "Register", "form", ["form"])]
    cases = planner._plan_security(nodes, "https://x.com")
    titles = [tc.title.lower() for tc in cases]
    assert any("xss" in t for t in titles)
    assert any("sql" in t for t in titles)
