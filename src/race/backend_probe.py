"""BackendProbe — Sprint 12.

Verifies a target exposes a real, stateful shared backend (a Conduit-style REST
API returning real data) BEFORE the race swarm runs. Frontend-independent: the
mirror may be API-only, so the probe also issues direct ``/api/tags`` and
``/api/articles`` requests via ``page.request`` (CDP-routed).
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict


class BackendProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    has_real_backend: bool
    api_endpoints: list[str]   # captured / confirmed during probe
    probe_note: str | None


def _has_real_data(payload: Any) -> bool:
    """True if a parsed JSON API response carries real (non-empty) Conduit data.

    Distinguishes a real Conduit backend (non-empty ``tags`` / ``articles`` /
    positive ``articlesCount``) from an empty store or a non-Conduit shape
    (e.g. jsonplaceholder's ``{"id": ...}``).
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("tags"):
        return True
    if payload.get("articles"):
        return True
    count = payload.get("articlesCount")
    if isinstance(count, int) and count > 0:
        return True
    return False


class BackendProbe:
    """No required constructor args."""

    async def probe(self, url: str, page: Page) -> BackendProbeResult:
        base = url.rstrip("/")
        captured: list[str] = []

        def _on_request(req: Any) -> None:
            if "/api/" in req.url:
                captured.append(req.url)

        page.on("request", _on_request)

        try:
            await page.goto(url, timeout=15_000)
        except Exception as exc:
            logger.debug(f"BackendProbe: goto {url!r} failed (API-only mirror?): {exc!r}")

        endpoints: list[str] = []
        notes: list[str] = []
        has_real = False

        for ep in ("/api/tags", "/api/articles?limit=1"):
            try:
                resp = await page.request.get(base + ep, timeout=10_000)
                if resp.ok:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = None
                    if _has_real_data(data):
                        has_real = True
                        endpoints.append(ep)
                    else:
                        notes.append(f"{ep}->empty")
                else:
                    notes.append(f"{ep}->{resp.status}")
            except Exception as exc:
                notes.append(f"{ep}->err:{type(exc).__name__}")

        for c in captured:
            if c not in endpoints:
                endpoints.append(c)

        result = BackendProbeResult(
            url=url,
            has_real_backend=has_real,
            api_endpoints=endpoints,
            probe_note=("; ".join(notes) if notes else None),
        )
        logger.info(
            f"BackendProbe: url={url!r} has_real_backend={has_real} "
            f"endpoints={endpoints} note={result.probe_note!r}"
        )
        return result


__all__ = ["BackendProbe", "BackendProbeResult", "_has_real_data"]
