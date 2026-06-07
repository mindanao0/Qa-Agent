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


from src.universal_qa.explorer.nav_map import (
    ExploredAction, ExploredPage, NavigationFlow, NavigationMap,
)


def test_action_to_step_navigate():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    a = ExploredAction(page_url="https://x.com/a", action_label="Home",
                       element_role="link", element_name="Home",
                       leads_to_url="https://x.com/home")
    step = planner._action_to_step(a)
    assert "https://x.com/home" in step
    assert step.startswith("เปิดหน้า")


def test_action_to_step_button_click():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    a = ExploredAction(page_url="https://x.com/a", action_label="Buy",
                       element_role="button", element_name="Checkout")
    step = planner._action_to_step(a)
    assert "Checkout" in step
    assert "คลิกปุ่ม" in step


def test_plan_flows_builds_test_case():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    steps = [
        ExploredAction(page_url="https://x.com/inv", action_label="Add to cart",
                       element_role="button", element_name="Add to cart"),
        ExploredAction(page_url="https://x.com/inv", action_label="Cart",
                       element_role="link", element_name="Cart",
                       leads_to_url="https://x.com/cart"),
    ]
    flow = NavigationFlow(flow_id="f1", name="checkout_flow", steps=steps,
                          start_url="https://x.com/inv", end_url="https://x.com/cart")
    cases = planner._plan_flows([flow])
    assert len(cases) == 1
    assert cases[0].type == "functional"
    assert cases[0].priority == "high"
    assert len(cases[0].steps) == 2
    assert "https://x.com/cart" in cases[0].expected_outcome
