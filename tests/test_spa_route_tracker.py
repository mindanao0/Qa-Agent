# tests/test_spa_route_tracker.py
from __future__ import annotations

import pydantic
import pytest
from unittest.mock import AsyncMock

from src.spa.route_tracker import RouteEvent, SPARouteTracker


@pytest.mark.asyncio
async def test_attach_injects_history_override():
    page = AsyncMock()
    tracker = SPARouteTracker()
    await tracker.attach(page)

    page.evaluate.assert_called_once()
    script: str = page.evaluate.call_args[0][0]
    assert "pushState" in script
    assert "replaceState" in script
    assert "hashchange" in script
    assert "__spa_route_events__" in script
    assert "__spa_prev_url__" in script


@pytest.mark.asyncio
async def test_flush_returns_route_events():
    page = AsyncMock()
    page.evaluate.return_value = [
        {
            "from_url": "http://example.com/#/",
            "to_url": "http://example.com/#/active",
            "trigger": "hashchange",
            "timestamp": 1717000000.0,
        }
    ]
    tracker = SPARouteTracker()
    events = await tracker.flush(page)

    assert len(events) == 1
    assert isinstance(events[0], RouteEvent)
    assert events[0].trigger == "hashchange"
    assert events[0].from_url == "http://example.com/#/"
    assert events[0].to_url == "http://example.com/#/active"


@pytest.mark.asyncio
async def test_flush_clears_events_on_next_read():
    page = AsyncMock()
    page.evaluate.return_value = []
    tracker = SPARouteTracker()
    events = await tracker.flush(page)
    assert events == []

    flush_script: str = page.evaluate.call_args[0][0]
    assert "window.__spa_route_events__ = []" in flush_script


@pytest.mark.asyncio
async def test_flush_multiple_events():
    page = AsyncMock()
    page.evaluate.return_value = [
        {"from_url": "a", "to_url": "b", "trigger": "pushState", "timestamp": 1.0},
        {"from_url": "b", "to_url": "c", "trigger": "replaceState", "timestamp": 2.0},
        {"from_url": "c", "to_url": "d", "trigger": "popstate", "timestamp": 3.0},
    ]
    tracker = SPARouteTracker()
    events = await tracker.flush(page)

    assert len(events) == 3
    assert events[0].trigger == "pushState"
    assert events[1].trigger == "replaceState"
    assert events[2].trigger == "popstate"


def test_route_event_model_extra_forbidden():
    with pytest.raises((pydantic.ValidationError, TypeError)):
        RouteEvent(
            from_url="a",
            to_url="b",
            trigger="pushState",
            timestamp=1.0,
            extra_field="nope",
        )
