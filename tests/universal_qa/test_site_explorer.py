import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.explorer.site_explorer import SiteExplorer, _element_key, _is_destructive
from src.universal_qa.explorer.nav_map import ExplorerConfig, ExploredAction, ExploredPage, NavigationMap


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


# ── Fix 2: query-param dedup ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explore_queues_all_query_param_variants():
    """URLs ที่มี query params ต่างกันควรถูก explore ได้ถึง max_visits_per_url ครั้ง
    ไม่ใช่แค่ 1 ครั้งเพราะ base path เดียวกัน."""
    cfg = ExplorerConfig(max_pages=10, max_depth=2, max_visits_per_url=5)
    auth = MagicMock()
    explorer = SiteExplorer(auth=auth, config=cfg)

    page = AsyncMock()
    page.url = "https://x.com/inventory.html"

    explored_urls: list[str] = []

    async def fake_explore_page(p, url, depth, base_domain, visited_actions):
        explored_urls.append(url)
        ep = ExploredPage(url=url, title="T", pam_content="", actions=[
            ExploredAction(page_url=url, action_label="P1",
                           element_role="link", leads_to_url="https://x.com/item.html?id=1"),
            ExploredAction(page_url=url, action_label="P2",
                           element_role="link", leads_to_url="https://x.com/item.html?id=2"),
        ])
        new_urls = ["https://x.com/item.html?id=1", "https://x.com/item.html?id=2"]
        return ep, new_urls, []

    with patch.object(explorer, "_explore_page", side_effect=fake_explore_page):
        await explorer.explore(page, ["https://x.com/inventory.html"])

    assert "https://x.com/item.html?id=1" in explored_urls, "?id=1 ต้องถูก explore"
    assert "https://x.com/item.html?id=2" in explored_urls, "?id=2 ต้องถูก explore"


# ── Fix 3: flow detection ─────────────────────────────────────────────────────

def test_detect_flows_one_per_unique_destination():
    """_detect_flows ต้องสร้าง flow แยกสำหรับแต่ละ URL ปลายทางที่ไม่ซ้ำกัน."""
    explorer = SiteExplorer.__new__(SiteExplorer)
    actions = [
        ExploredAction(page_url="https://x.com/inv", action_label="Product 1",
                       element_role="link", leads_to_url="https://x.com/item.html?id=1"),
        ExploredAction(page_url="https://x.com/inv", action_label="Product 2",
                       element_role="link", leads_to_url="https://x.com/item.html?id=2"),
        ExploredAction(page_url="https://x.com/inv", action_label="Product 3",
                       element_role="link", leads_to_url="https://x.com/item.html?id=3"),
    ]
    flows = explorer._detect_flows(actions)

    end_urls = {f.end_url for f in flows}
    assert "https://x.com/item.html?id=1" in end_urls, "flow ไปยัง ?id=1 ต้องมี"
    assert "https://x.com/item.html?id=2" in end_urls, "flow ไปยัง ?id=2 ต้องมี"
    assert "https://x.com/item.html?id=3" in end_urls, "flow ไปยัง ?id=3 ต้องมี"
    assert len(flows) >= 3, f"ต้องมีอย่างน้อย 3 flows แต่ได้ {len(flows)}"


def test_detect_flows_no_duplicates():
    """action ที่นำไปยัง URL เดิม 2 ครั้ง ต้องไม่สร้าง flow ซ้ำ."""
    explorer = SiteExplorer.__new__(SiteExplorer)
    actions = [
        ExploredAction(page_url="https://x.com/inv", action_label="Cart",
                       element_role="link", leads_to_url="https://x.com/cart"),
        ExploredAction(page_url="https://x.com/inv", action_label="Cart icon",
                       element_role="link", leads_to_url="https://x.com/cart"),
    ]
    flows = explorer._detect_flows(actions)
    cart_flows = [f for f in flows if f.end_url == "https://x.com/cart"]
    assert len(cart_flows) == 1, "URL ปลายทางซ้ำกัน ต้องได้ flow เดียว"


def test_detect_flows_adds_multistep_flow_when_multiple_nav():
    """เมื่อมี nav actions ≥2 ต้องมี multi-step flow รวมด้วย."""
    explorer = SiteExplorer.__new__(SiteExplorer)
    actions = [
        ExploredAction(page_url="https://x.com/inv", action_label="Item",
                       element_role="link", leads_to_url="https://x.com/item.html?id=1"),
        ExploredAction(page_url="https://x.com/inv", action_label="Cart",
                       element_role="link", leads_to_url="https://x.com/cart"),
    ]
    flows = explorer._detect_flows(actions)
    multi = [f for f in flows if len(f.steps) >= 2]
    assert len(multi) >= 1, "ต้องมี multi-step flow อย่างน้อย 1 ตัว"
