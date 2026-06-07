import pytest
from pydantic import ValidationError
from src.universal_qa.explorer.nav_map import (
    ElementCandidate, ExploredAction, ExploredPage, NavigationFlow,
    NavigationMap, ExplorerConfig,
)


def test_element_candidate_minimal():
    ec = ElementCandidate(label="Add to cart", priority=1)
    assert ec.role is None
    assert ec.is_in_iframe is False
    assert ec.is_in_shadow is False


def test_explored_action_defaults():
    a = ExploredAction(page_url="https://x.com/a", action_label="Cart")
    assert a.leads_to_url is None
    assert a.leads_to_modal is False
    assert a.state_change is None
    assert a.is_destructive is False


def test_explored_action_rejects_extra():
    with pytest.raises(ValidationError):
        ExploredAction(page_url="https://x.com", action_label="X", bogus=1)


def test_explored_page_defaults():
    p = ExploredPage(url="https://x.com/p", title="P", pam_content="...", actions=[])
    assert p.state_snapshot is None
    assert p.requires_path == []


def test_navigation_flow_holds_steps():
    a = ExploredAction(page_url="https://x.com/a", action_label="Go")
    flow = NavigationFlow(flow_id="f1", name="checkout", steps=[a],
                          start_url="https://x.com/a", end_url="https://x.com/b")
    assert len(flow.steps) == 1


def test_navigation_map_aggregates():
    nm = NavigationMap(base_url="https://x.com", pages=[], flows=[],
                       explored_at_iso="2026-06-07T00:00:00Z")
    assert nm.pages == []
    assert nm.flows == []


def test_explorer_config_defaults():
    cfg = ExplorerConfig()
    assert cfg.max_pages == 50
    assert cfg.explore_timeout_min == 5
    assert cfg.max_depth == 4
    assert cfg.allow_destructive is False
    assert cfg.max_visits_per_url == 2
