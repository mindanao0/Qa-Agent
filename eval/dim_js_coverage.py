"""Coverage dimension: real client-side JS code execution.

playwright-python has no `page.coverage` — that helper only exists in
Playwright for Node; the Python bindings never wrapped it (verified against
playwright==1.62.0's generated API: no `coverage` attribute on Page/Frame).
This module talks to the same underlying Chrome DevTools Protocol domain
that helper uses (Profiler) via `context.new_cdp_session()` instead.

Reports function-level JS coverage (% of profiled functions that executed
at least once) — coarser than V8's byte-precise ranges, but exact at
function granularity and needs no extra CDP round-trips (no
Debugger.getScriptSource). Verified live against a local fixture with one
called + one uncalled function + the executing top-level script body:
js_functions_total=3, js_functions_executed=2 (66.67%) — matches exactly.

CSS.startRuleUsageTracking/stopRuleUsageTracking was tried for a matching
CSS-rule-usage metric and dropped: live against a 2-rule stylesheet (one
class present in the DOM, one not) it returned only 1 ruleUsage entry
total — not even a used:false entry for the second rule — so
"rules_used / rules_total" would be silently wrong (denominator too low).
Not shipped until that's understood; JS-only for now.

Chromium only; answers a different question from every path-based
dimension here — not "which pages did we visit" but "how much of the
shipped JS did the session actually execute".
"""
from __future__ import annotations

from playwright.async_api import BrowserContext, Page

from eval.coverage_dimensions import DimensionResult


class JSCoverageSession:
    def __init__(self, cdp) -> None:
        self._cdp = cdp

    @classmethod
    async def start(cls, context: BrowserContext, page: Page) -> "JSCoverageSession":
        cdp = await context.new_cdp_session(page)
        await cdp.send("Profiler.enable")
        await cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
        return cls(cdp)

    async def stop(self, target_id: str) -> DimensionResult:
        js_result = await self._cdp.send("Profiler.takePreciseCoverage")
        await self._cdp.send("Profiler.stopPreciseCoverage")

        scripts = js_result.get("result", [])
        total_fns = executed_fns = 0
        for script in scripts:
            for fn in script.get("functions", []):
                total_fns += 1
                if any(r.get("count", 0) > 0 for r in fn.get("ranges", [])):
                    executed_fns += 1
        js_pct = round(100.0 * executed_fns / total_fns, 2) if total_fns else None

        return DimensionResult(
            method="js_coverage",
            target_id=target_id,
            paths=set(),
            meta={
                "js_scripts_profiled": len(scripts),
                "js_functions_total": total_fns,
                "js_functions_executed": executed_fns,
                "js_pct_functions_executed": js_pct,
            },
        )
