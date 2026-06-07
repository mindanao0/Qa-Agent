import pytest
from unittest.mock import AsyncMock
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.nav_map import ElementCandidate


def test_classify_priority_nav():
    assert ElementScanner._classify_priority("nav", "Home") == 0
    assert ElementScanner._classify_priority("banner", "Logo") == 0


def test_classify_priority_form_button():
    assert ElementScanner._classify_priority("button", "Submit") == 1


def test_classify_priority_other():
    assert ElementScanner._classify_priority("link", "Random") == 2


def test_raw_to_candidates_sorts_by_priority():
    scanner = ElementScanner()
    raw = [
        {"label": "Random link", "role": "link", "name": "Random",
         "selector": "a.x", "container": "main", "is_in_iframe": False,
         "is_in_shadow": False},
        {"label": "Home", "role": "link", "name": "Home",
         "selector": "a.home", "container": "nav", "is_in_iframe": False,
         "is_in_shadow": False},
    ]
    cands = scanner._raw_to_candidates(raw)
    assert isinstance(cands[0], ElementCandidate)
    # nav (priority 0) must sort before other (priority 2)
    assert cands[0].label == "Home"


@pytest.mark.asyncio
async def test_scan_returns_candidates():
    scanner = ElementScanner()
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=[
        {"label": "Buy", "role": "button", "name": "Buy", "selector": "button.buy",
         "container": "form", "is_in_iframe": False, "is_in_shadow": False},
    ])
    page.frames = []
    cands = await scanner.scan(page)
    assert len(cands) == 1
    assert cands[0].label == "Buy"
    assert cands[0].priority == 1
