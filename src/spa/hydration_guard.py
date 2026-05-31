# src/spa/hydration_guard.py
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


class HydrationGuard:
    """No required constructor args."""

    async def wait_stable(self, page: Any, timeout_ms: int = 5000) -> None:
        """
        Wait until:
          1. No pending fetch/XHR (networkidle2-equivalent via CDP)
          2. No React/Vue/Angular pending state
          3. document.readyState == "complete"

        CDP only — no page.wait_for_load_state("networkidle").
        """
        client = await page.context.new_cdp_session(page)
        try:
            await client.send("Network.enable")
            pending_ids: set[str] = set()

            def _on_request(params: dict) -> None:
                pending_ids.add(params.get("requestId", ""))

            def _on_done(params: dict) -> None:
                pending_ids.discard(params.get("requestId", ""))

            client.on("Network.requestWillBeSent", _on_request)
            client.on("Network.loadingFinished", _on_done)
            client.on("Network.loadingFailed", _on_done)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_ms / 1000
            idle_since: float | None = None

            # Wait for document.readyState == "complete" first
            while loop.time() < deadline:
                try:
                    state = await page.evaluate("document.readyState")
                except Exception:
                    state = "loading"
                if state == "complete":
                    break
                await asyncio.sleep(0.05)

            # Wait for network idle (500 ms of no pending requests)
            while loop.time() < deadline:
                await asyncio.sleep(0.05)
                if not pending_ids:
                    if idle_since is None:
                        idle_since = loop.time()
                    elif loop.time() - idle_since >= 0.5:
                        break
                else:
                    idle_since = None

        finally:
            await client.detach()

        # Framework-specific flush — best-effort, never raises
        try:
            framework = await self.detect_framework(page)
            await self._flush_framework(page, framework)
        except Exception as exc:
            logger.debug(f"HydrationGuard: framework flush skipped ({exc!r})")

    async def detect_framework(self, page: Any) -> str:
        """Returns: "react" | "vue" | "angular" | "unknown" """
        return await page.evaluate(
            """
            () => {
                if (
                    window.__REACT_FIBER__ !== undefined ||
                    window._reactRootContainer !== undefined ||
                    document.querySelector('[data-reactroot]') !== null
                ) { return 'react'; }
                if (window.__vue_app__ !== undefined || window.Vue !== undefined) {
                    return 'vue';
                }
                if (typeof window.getAllAngularTestabilities === 'function') {
                    return 'angular';
                }
                return 'unknown';
            }
            """
        )

    async def _flush_framework(self, page: Any, framework: str) -> None:
        if framework == "react":
            await page.evaluate(
                "() => new Promise(resolve => setTimeout(resolve, 100))"
            )
        elif framework == "vue":
            await page.evaluate(
                """
                () => new Promise(resolve => {
                    if (window.__vue_app__?.config?.globalProperties?.$nextTick) {
                        window.__vue_app__.config.globalProperties.$nextTick(resolve);
                    } else {
                        setTimeout(resolve, 100);
                    }
                })
                """
            )
        elif framework == "angular":
            await page.evaluate(
                """
                () => new Promise(resolve => {
                    const tbs = (typeof window.getAllAngularTestabilities === 'function')
                        ? window.getAllAngularTestabilities()
                        : [];
                    if (tbs.length === 0) { resolve(); return; }
                    let n = tbs.length;
                    tbs.forEach(t => t.whenStable(() => { if (--n === 0) resolve(); }));
                })
                """
            )
        else:
            await asyncio.sleep(0.1)


__all__ = ["HydrationGuard"]
