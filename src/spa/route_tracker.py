# src/spa/route_tracker.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class RouteEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_url: str
    to_url: str
    trigger: Literal["pushState", "replaceState", "hashchange", "popstate"]
    timestamp: float


class SPARouteTracker:
    """No required constructor args."""

    async def attach(self, page: Any) -> None:
        """Inject JS listeners for history.pushState/replaceState + hashchange/popstate."""
        await page.evaluate(
            """
            () => {
                window.__spa_route_events__ = [];
                const _push = history.pushState.bind(history);
                const _replace = history.replaceState.bind(history);

                history.pushState = function(state, title, url) {
                    const from = location.href;
                    _push(state, title, url);
                    window.__spa_route_events__.push({
                        from_url: from,
                        to_url: location.href,
                        trigger: 'pushState',
                        timestamp: Date.now() / 1000
                    });
                };

                history.replaceState = function(state, title, url) {
                    const from = location.href;
                    _replace(state, title, url);
                    window.__spa_route_events__.push({
                        from_url: from,
                        to_url: location.href,
                        trigger: 'replaceState',
                        timestamp: Date.now() / 1000
                    });
                };

                window.addEventListener('hashchange', (e) => {
                    window.__spa_route_events__.push({
                        from_url: e.oldURL,
                        to_url: e.newURL,
                        trigger: 'hashchange',
                        timestamp: Date.now() / 1000
                    });
                });

                window.addEventListener('popstate', () => {
                    window.__spa_route_events__.push({
                        from_url: document.referrer || location.href,
                        to_url: location.href,
                        trigger: 'popstate',
                        timestamp: Date.now() / 1000
                    });
                });
            }
            """
        )

    async def flush(self, page: Any) -> list[RouteEvent]:
        """Read and clear window.__spa_route_events__."""
        events: list[dict] = await page.evaluate(
            """
            () => {
                const evts = window.__spa_route_events__ || [];
                window.__spa_route_events__ = [];
                return evts;
            }
            """
        )
        return [RouteEvent(**e) for e in events]


__all__ = ["RouteEvent", "SPARouteTracker"]
