# src/universal_qa/agent.py
from __future__ import annotations

import asyncio
import json
import pathlib
from urllib.parse import quote, urlparse

from loguru import logger
from playwright.async_api import async_playwright

from src.cache.semantic_cache import SemanticCache
from src.config_loader import (
    get_api_fuzz_config,
    get_semantic_cache_config,
    get_use_api_fuzz,
    get_use_semantic_cache,
)
from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.explorer.nav_map import ExploredPage, NavigationMap
from src.universal_qa.models import TestResult
from src.universal_qa.reporters.html import HTMLReporter
from src.universal_qa.reporters.terminal import TerminalReporter
from src.universal_qa.site_discovery import SiteDiscovery
from src.universal_qa.test_planner import UniversalTestPlanner
from src.universal_qa.test_runner import UniversalTestRunner

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]


class UniversalQAAgent:
    """Orchestrates all 4 phases: Discover → Plan → Execute → Report."""

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        max_pages: int = 50,
        explore_timeout: int = 5,
        max_depth: int = 4,
        allow_destructive: bool = False,
        headless: bool = True,
        output_dir: pathlib.Path | None = None,
        enable_coverage_crosscheck: bool = False,
        enable_api_fuzz: bool = False,
    ) -> None:
        self._url = url
        self._max_pages = max_pages
        self._headless = headless
        self._output_dir = output_dir or pathlib.Path("reports")
        self._enable_coverage_crosscheck = enable_coverage_crosscheck
        # Config-file opt-in (get_use_api_fuzz) OR an explicit constructor/CLI
        # flag turns the phase on — mirrors llm.structured_output_engine's
        # dual env/config gate. Default (both False) is zero behavior change.
        self._enable_api_fuzz = bool(enable_api_fuzz) or get_use_api_fuzz()
        self._auth = AuthManager(username=username, password=password)
        self._discovery = SiteDiscovery(max_pages=max_pages)
        _semantic_cache: SemanticCache | None = None
        if get_use_semantic_cache():
            _cache_cfg = get_semantic_cache_config()
            _semantic_cache = SemanticCache(
                db_path=pathlib.Path(
                    _cache_cfg.get("db_path", "~/.qa-agent/semantic_cache.lance")
                ).expanduser(),
                hit_threshold=float(_cache_cfg.get("hit_threshold", 0.95)),
                guided_threshold=float(_cache_cfg.get("guided_threshold", 0.50)),
            )
        self._planner = UniversalTestPlanner(cache=_semantic_cache)
        self._terminal = TerminalReporter()
        from src.universal_qa.explorer.nav_map import ExplorerConfig
        self._explorer_cfg = ExplorerConfig(
            max_pages=max_pages,
            explore_timeout_min=explore_timeout,
            max_depth=max_depth,
            allow_destructive=allow_destructive,
        )

    async def run(self) -> list[TestResult]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self._headless, args=_LAUNCH_ARGS
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            try:
                logger.info(f"UniversalQAAgent: starting run on {self._url}")

                # Phase 1: Navigate + Auth
                await page.goto(self._url, wait_until="domcontentloaded", timeout=30_000)
                await self._auth.setup(page)
                # รอให้ SPA router อัปเดต URL หลัง auth (Angular/React redirect ช้ากว่า DOM)
                await page.wait_for_timeout(1_500)

                # Phase 2: Discover URLs (fast)
                discover_url = page.url
                logger.info(f"Phase 2: site discovery from {discover_url}")
                sfg_store = await self._discovery.discover(page, discover_url)
                discovered_urls = [
                    n.url for n in sfg_store.get_nodes_by_url_prefix(
                        f"{urlparse(discover_url).scheme}://{urlparse(discover_url).netloc}"
                    )
                ]
                if discover_url not in discovered_urls:
                    discovered_urls.insert(0, discover_url)

                # Phase 3: Explore (thorough) → NavigationMap
                logger.info("Phase 3: interaction-based exploration")
                from src.universal_qa.explorer.site_explorer import SiteExplorer
                explorer = SiteExplorer(
                    auth=self._auth, config=self._explorer_cfg,
                    client=self._planner._client,
                )
                nav_map = await explorer.explore(page, discovered_urls)
                logger.info(
                    f"  Explored {len(nav_map.pages)} pages, {len(nav_map.flows)} flows"
                )

                if self._enable_coverage_crosscheck:
                    await self._run_coverage_crosscheck(discover_url)

                if self._enable_api_fuzz:
                    await self._run_api_fuzz(discover_url)

                # Supplement nav_map with Phase 2 pages not reached by Phase 3
                _explored_urls = {p.url.split("?")[0].split("#")[0] for p in nav_map.pages}
                _extra_pages = []
                for _node in sfg_store.get_nodes_by_url_prefix(
                    f"{urlparse(discover_url).scheme}://{urlparse(discover_url).netloc}"
                ):
                    _nurl = _node.url.split("?")[0].split("#")[0]
                    if _nurl not in _explored_urls:
                        _extra_pages.append(ExploredPage(
                            url=_node.url,
                            title=_node.page_title,
                            pam_content=_node.pam_content,
                            actions=[],
                        ))
                if _extra_pages:
                    nav_map = NavigationMap(
                        base_url=nav_map.base_url,
                        pages=nav_map.pages + _extra_pages,
                        flows=nav_map.flows,
                        explored_at_iso=nav_map.explored_at_iso,
                    )
                    logger.info(
                        f"  +{len(_extra_pages)} Phase 2 pages → total {len(nav_map.pages)}"
                    )

                # Phase 4: Plan from NavigationMap
                logger.info("Phase 4: generating test cases")
                test_cases = await self._planner.plan_from_map(nav_map)
                logger.info(f"  {len(test_cases)} test cases generated")

                # Phase 4 (E2E): เพิ่ม E2E flows ถ้ามี credentials
                if self._auth._username and self._auth._password:
                    from src.universal_qa.e2e_planner import E2EFlowPlanner
                    _e2e_planner = E2EFlowPlanner(self._planner._client)
                    _e2e_cases = await _e2e_planner.plan(
                        nav_map,
                        username=self._auth._username,
                        password=self._auth._password,
                    )
                    if _e2e_cases:
                        test_cases = test_cases + _e2e_cases
                        logger.info(f"  +{len(_e2e_cases)} E2E flows → total {len(test_cases)}")

                # Phase 4: Execute + Report
                logger.info("Phase 4: executing test cases")
                screenshot_dir = self._output_dir / "screenshots"
                runner = UniversalTestRunner(
                    sfg_store=sfg_store,
                    screenshot_dir=screenshot_dir,
                    terminal_reporter=self._terminal,
                )
                results = await runner.run(test_cases, page)

                # Final reports
                self._terminal.report_summary(results)
                html_reporter = HTMLReporter(output_dir=self._output_dir)
                report_path = html_reporter.generate(results)
                logger.info(f"HTML report: {report_path}")

                return results

            finally:
                await browser.close()

    async def _run_coverage_crosscheck(self, seed_url: str) -> None:
        """Optional post-exploration stage: cross-check the SFG crawl against
        8 independent coverage dimensions (static-declared, JS execution,
        viewport A/B, session-state A/B, keyboard-walk, monkey-walk, declared
        API surface) and write the merged report. Runs its own isolated
        browser session(s) — never shares `page` with the main flow — so a
        failure here never breaks the run. See eval/FINDINGS_explore.md.
        """
        from src.universal_qa.coverage.crosscheck import run_full_crosscheck

        target_id = urlparse(seed_url).netloc or seed_url
        try:
            report = await run_full_crosscheck(
                target_id=target_id, seed_url=seed_url,
                headless=self._headless,
                safe_mode=not self._explorer_cfg.allow_destructive,
            )
        except Exception as exc:
            logger.warning(f"UniversalQAAgent: coverage crosscheck failed — {exc!r}")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "coverage_crosscheck.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"  Coverage crosscheck: {report.get('seen_by_all_count', '?')}/"
            f"{report.get('total_distinct_paths', '?')} paths seen by every dimension "
            f"→ {out_path}"
        )

    async def _run_api_fuzz(self, seed_url: str) -> None:
        """Optional post-exploration stage: discover REST endpoints the site
        actually calls, infer their schema from real captured traffic (Sprint 13
        ``SchemaInferrer``), generate typed fuzz vectors (``AdvancedVectorGenerator``),
        and send them against the REAL target to surface server-side anomalies —
        5xx on malformed input, response-schema drift (``AnomalyClassifier``).
        Runs its own isolated browser session — never shares `page`/`context`
        with the main flow — so a failure here never breaks the run.

        Deliberately conservative for a phase that fuzzes a real third-party
        site: only GET endpoints that were actually observed on the wire are
        fuzzed (no POST/PUT/DELETE is ever synthesized here), on top of the
        fuzzer's own BLOCKED_ACTION_PATTERNS enforcement (delete/remove/
        transfer/payment/password — never weakened) and its built-in
        ≤2 req/sec spacing + consecutive-5xx circuit breaker. See
        docs/specs/"Sprint 13 Advanced API Fuzzing.md" / src/fuzzer/.
        """
        from src.fuzzer.anomaly_classifier import AnomalyClassifier
        from src.fuzzer.api_fuzzer import AutonomousAPIFuzzer, FuzzRequest
        from src.fuzzer.schema_inferrer import SchemaInferrer
        from src.fuzzer.vector_generator import AdvancedVectorGenerator, BLOCKED_ACTION_PATTERNS

        cfg = get_api_fuzz_config()
        max_endpoints = int(cfg.get("max_endpoints", 5))
        max_vectors = int(cfg.get("max_vectors_per_field", 5))

        report: dict = {
            "seed_url": seed_url,
            "endpoints_discovered": 0,
            "endpoints_fuzzed": 0,
            "vectors_sent": 0,
            "anomalies_found": 0,
            "anomaly_breakdown": {},
            "results": [],
        }
        try:
            sem = asyncio.Semaphore(1)
            base = f"{urlparse(seed_url).scheme}://{urlparse(seed_url).netloc}"

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=self._headless, args=_LAUNCH_ARGS)
                try:
                    ctx = await browser.new_context()
                    page = await ctx.new_page()
                    try:
                        inferrer = SchemaInferrer()
                        generator = AdvancedVectorGenerator()
                        classifier = AnomalyClassifier()
                        fuzzer = AutonomousAPIFuzzer()

                        traces = await inferrer.capture(page, seed_url)
                        schemas = await inferrer.infer(traces, sem)
                        report["endpoints_discovered"] = len(schemas)

                        for schema in schemas[:max_endpoints]:
                            if schema.method != "GET":
                                continue  # read-only scope: never mutate a real 3rd-party site
                            if BLOCKED_ACTION_PATTERNS.search(schema.endpoint):
                                continue

                            field_vectors = await generator.generate_fields(schema, sem, max_vectors)
                            if not field_vectors:
                                continue  # nothing safely fuzzable without guessing params

                            requests: list[FuzzRequest] = []
                            for field, vectors in field_vectors.items():
                                sep = "&" if "?" in schema.endpoint else "?"
                                for v in vectors:
                                    url = f"{base}{schema.endpoint}{sep}{field}={quote(v, safe='')}"
                                    requests.append(FuzzRequest(vector=v, method="GET", url=url, body=None))
                            if not requests:
                                continue

                            results = await fuzzer.fuzz_endpoint(
                                f"GET {schema.endpoint}", requests, page, classifier,
                                schema.response_schema or None,
                            )
                            report["endpoints_fuzzed"] += 1
                            report["vectors_sent"] += len(requests)
                            for r in results:
                                if r.anomaly and not r.false_positive:
                                    report["anomalies_found"] += 1
                                    atype = r.anomaly_type.value if r.anomaly_type else "?"
                                    report["anomaly_breakdown"][atype] = (
                                        report["anomaly_breakdown"].get(atype, 0) + 1
                                    )
                                    report["results"].append({
                                        "endpoint": r.endpoint,
                                        "vector": r.vector,
                                        "status_code": r.status_code,
                                        "anomaly_type": atype,
                                        "duration_ms": r.duration_ms,
                                    })
                    finally:
                        await ctx.close()
                finally:
                    await browser.close()
        except Exception as exc:
            logger.warning(f"UniversalQAAgent: api fuzz phase failed — {exc!r}")
            report["error"] = repr(exc)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "api_fuzz_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"  API fuzz: {report['endpoints_fuzzed']}/{report['endpoints_discovered']} GET "
            f"endpoints fuzzed, {report['anomalies_found']} anomalies → {out_path}"
        )


__all__ = ["UniversalQAAgent"]
