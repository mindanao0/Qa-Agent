import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.site_discovery import SiteDiscovery


def test_site_discovery_default_limits():
    sd = SiteDiscovery()
    assert sd.max_pages == 50
    assert sd.max_depth == 6


def test_site_discovery_custom_limits():
    sd = SiteDiscovery(max_pages=20, max_depth=3)
    assert sd.max_pages == 20
    assert sd.max_depth == 3


async def test_discover_returns_sfg_store(tmp_path):
    sd = SiteDiscovery(max_pages=2, max_depth=1)

    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[])  # no links found

    mock_node = MagicMock()
    mock_node.node_id = "abc123"

    with patch.object(sd, "_visit_and_record", return_value=mock_node):
        store = await sd.discover(mock_page, "https://example.com",
                                   db_path=tmp_path / "sfg.db")

    assert store is not None
    # discover was called once (the seed url)
    mock_page.goto.assert_called_once()
