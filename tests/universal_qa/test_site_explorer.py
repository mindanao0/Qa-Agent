import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.explorer.site_explorer import SiteExplorer, _element_key, _is_destructive
from src.universal_qa.explorer.nav_map import ExplorerConfig, ExploredAction, NavigationMap


def test_element_key_stable():
    k1 = _element_key("https://x.com/a", "button", "Buy")
    k2 = _element_key("https://x.com/a", "button", "Buy")
    assert k1 == k2


def test_is_destructive_matches_blocked():
    assert _is_destructive("Delete account") is True
    assert _is_destructive("Logout") is True
    assert _is_destructive("View products") is False


def test_build_flow_from_path():
    explorer = SiteExplorer.__new__(SiteExplorer)
    steps = [
        ExploredAction(page_url="https://x.com/a", action_label="Add",
                       leads_to_url="https://x.com/a"),
        ExploredAction(page_url="https://x.com/a", action_label="Cart",
                       leads_to_url="https://x.com/cart"),
    ]
    flow = explorer._build_flow(steps, idx=0)
    assert flow.start_url == "https://x.com/a"
    assert flow.end_url == "https://x.com/cart"
    assert len(flow.steps) == 2


@pytest.mark.asyncio
async def test_explore_records_navigation(tmp_path):
    cfg = ExplorerConfig(max_pages=1, max_depth=1)
    auth = MagicMock()
    explorer = SiteExplorer(auth=auth, config=cfg)

    page = AsyncMock()
    page.url = "https://x.com/start"
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Start")
    page.evaluate = AsyncMock(return_value="{}")  # localStorage snapshot
    page.wait_for_load_state = AsyncMock()

    # No interactive elements found → page recorded with no actions
    with patch.object(explorer._scanner, "scan", new=AsyncMock(return_value=[])), \
         patch.object(explorer, "_record_page_node", new=AsyncMock(
             return_value=("Start", "pam content"))), \
         patch.object(explorer._guard, "is_session_lost", new=AsyncMock(return_value=False)):
        nav_map = await explorer.explore(page, ["https://x.com/start"])

    assert isinstance(nav_map, NavigationMap)
    assert len(nav_map.pages) == 1
    assert nav_map.pages[0].url == "https://x.com/start"
