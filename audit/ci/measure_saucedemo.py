"""
measure_saucedemo.py — Full QA agent pipeline on https://www.saucedemo.com/

Phases:
  1. Continuous exploratory loop — 3 cycles, SFGCrawler state recording
  2. Targeted scenarios — happy path (standard_user), bug hunting (problem_user),
                          negative test (locked_out_user)
  3. API fuzzing — AutonomousAPIFuzzer (SauceDemo is localStorage-only; 0 XHR/fetch expected)
  4. Report — write audit/ci/saucedemo_report.md + saucedemo_results.json

Run: uv run python -m audit.ci.measure_saucedemo
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from playwright.async_api import Browser, Page, async_playwright

from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import SFGStore
from src.observability.audit_chain import CryptoAuditTrail
from src.observability.tracer import OTelTracer
from src.perception.grounder import Grounder

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_TARGET_URL = "https://www.saucedemo.com/"
_INVENTORY_URL = "https://www.saucedemo.com/inventory.html"
_CART_URL = "https://www.saucedemo.com/cart.html"
_STANDARD = ("standard_user", "secret_sauce")
_PROBLEM = ("problem_user", "secret_sauce")
_LOCKED = ("locked_out_user", "secret_sauce")

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
_AUDIT_DIR = pathlib.Path("audit/ci")
_RESULTS_PATH = _AUDIT_DIR / "saucedemo_results.json"
_REPORT_PATH = _AUDIT_DIR / "saucedemo_report.md"
_AUDIT_PATH = _AUDIT_DIR / "saucedemo_audit.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Browser helpers (get_by_role / get_by_label / get_by_text / get_by_test_id only)
# ─────────────────────────────────────────────────────────────────────────────

async def _login(page: Page, username: str, password: str) -> tuple[bool, str | None]:
    """Navigate to SauceDemo and log in. Returns (success, error_message_or_None).

    SauceDemo has no <label> elements — uses data-test attributes and placeholders.
    get_by_test_id works after pw.selectors.set_test_id_attribute("data-test").
    """
    await page.goto(_TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
    try:
        await page.get_by_test_id("username").fill(username, timeout=5_000)
        await page.get_by_test_id("password").fill(password, timeout=5_000)
        await page.get_by_test_id("login-button").click(timeout=5_000)
    except Exception as exc:
        return False, f"login form error: {exc!r}"

    # Detect locked-out / error message
    try:
        err_el = page.get_by_text("Epic sadface", exact=False)
        await err_el.wait_for(state="visible", timeout=2_500)
        full_text = await err_el.inner_text()
        return False, full_text.strip()
    except Exception:
        pass

    # Confirm we reached inventory
    try:
        await page.wait_for_url("**/inventory.html", timeout=8_000)
        return True, None
    except Exception:
        return False, f"did not reach inventory (url={page.url!r})"


async def _logout(page: Page) -> None:
    try:
        await page.get_by_role("button", name="Open Menu").click(timeout=3_000)
        await page.get_by_role("link", name="Logout").click(timeout=3_000)
        await page.wait_for_url(_TARGET_URL, timeout=5_000)
    except Exception:
        await page.goto(_TARGET_URL, wait_until="domcontentloaded", timeout=15_000)


async def _safe(coro, label: str = "") -> bool:
    try:
        await coro
        return True
    except Exception as exc:
        logger.warning(f"{label or 'action'} failed: {exc!r}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Continuous exploratory loop (3 cycles)
# ─────────────────────────────────────────────────────────────────────────────

async def run_phase1(
    browser: Browser,
    crawler: SFGCrawler,
    tracer: OTelTracer,
    audit: CryptoAuditTrail,
) -> dict[str, Any]:
    context = await browser.new_context()
    page = await context.new_page()

    all_node_ids: set[str] = set()
    new_per_cycle: list[int] = []
    cycles_completed = 0

    async def record() -> bool:
        try:
            node, _ = await crawler._visit_node(page, None)
            if node.node_id not in all_node_ids:
                all_node_ids.add(node.node_id)
                return True
        except Exception as exc:
            logger.warning(f"_visit_node failed: {exc!r}")
        return False

    # Deepening exploration plans (role/label/text locators only)
    plans: list[list] = [
        # Cycle 1: login → inventory → product detail → back
        [
            ("login", _STANDARD),
            "record",
            ("role_link", "Sauce Labs Backpack"),
            "record",
            "go_back",
            "record",
        ],
        # Cycle 2: login → add to cart → view cart
        [
            ("login", _STANDARD),
            ("role_button", "Add to cart"),   # first item
            "record",
            "goto_cart",
            "record",
        ],
        # Cycle 3: login → add to cart → checkout step 1 → step 2
        [
            ("login", _STANDARD),
            ("role_button", "Add to cart"),
            "goto_cart",
            ("role_button", "Checkout"),
            "record",
            "fill_checkout_info",
            "record",
            ("test_id", "continue"),
            "record",
        ],
    ]

    for cycle_idx, plan in enumerate(plans, 1):
        logger.info(f"[phase1] cycle {cycle_idx}")
        new_this_cycle = 0
        async with tracer.span("saucedemo.explore", cycle=cycle_idx):
            for step in plan:
                if step == "record":
                    was_new = await record()
                    if was_new:
                        new_this_cycle += 1

                elif step == "go_back":
                    await _safe(page.go_back(timeout=15_000), "go_back")
                    await _safe(page.wait_for_load_state("domcontentloaded", timeout=8_000), "load_state")

                elif step == "goto_cart":
                    await _safe(
                        page.goto(_CART_URL, wait_until="domcontentloaded", timeout=15_000),
                        "goto_cart",
                    )

                elif step == "fill_checkout_info":
                    try:
                        await page.get_by_test_id("firstName").fill("Test", timeout=3_000)
                        await page.get_by_test_id("lastName").fill("User", timeout=3_000)
                        await page.get_by_test_id("postalCode").fill("12345", timeout=3_000)
                    except Exception as exc:
                        logger.warning(f"fill_checkout_info failed: {exc!r}")

                elif isinstance(step, tuple):
                    action, arg = step
                    if action == "login":
                        username, password = arg
                        ok, err = await _login(page, username, password)
                        if not ok:
                            logger.warning(f"[cycle {cycle_idx}] login failed: {err}")
                            break
                    elif action == "role_link":
                        await _safe(
                            page.get_by_role("link", name=arg).first.click(timeout=10_000),
                            f"link:{arg}",
                        )
                    elif action == "role_button":
                        await _safe(
                            page.get_by_role("button", name=arg).first.click(timeout=10_000),
                            f"button:{arg}",
                        )
                    elif action == "test_id":
                        await _safe(
                            page.get_by_test_id(arg).click(timeout=10_000),
                            f"test_id:{arg}",
                        )

        new_per_cycle.append(new_this_cycle)
        cycles_completed += 1
        audit.append("saucedemo.explore.cycle", {
            "cycle": cycle_idx,
            "new_states": new_this_cycle,
            "total_states": len(all_node_ids),
        })

        try:
            await _logout(page)
        except Exception:
            await page.goto(_TARGET_URL, wait_until="domcontentloaded", timeout=15_000)

    await context.close()
    return {
        "cycles_completed": cycles_completed,
        "states_discovered": len(all_node_ids),
        "new_per_cycle": new_per_cycle,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2A — Happy path (standard_user)
# ─────────────────────────────────────────────────────────────────────────────

async def _scenario_happy_path(browser: Browser, tracer: OTelTracer) -> dict[str, Any]:
    context = await browser.new_context()
    page = await context.new_page()
    errors: list[str] = []
    steps_done = 0

    async def step(label: str, coro) -> bool:
        nonlocal steps_done
        try:
            await coro
            steps_done += 1
            return True
        except Exception as exc:
            errors.append(f"{label}: {exc!r}")
            return False

    try:
        async with tracer.span("saucedemo.happy_path"):
            ok, err = await _login(page, *_STANDARD)
            if not ok:
                return {"name": "happy_path_standard_user", "passed": False,
                        "steps_done": 0, "error": err}
            steps_done += 1  # login

            # Verify product list loaded
            btns = await page.get_by_role("button", name="Add to cart").all()
            if not btns:
                errors.append("no Add-to-cart buttons found on inventory page")

            # Add first item
            await step("add_to_cart",
                       page.get_by_role("button", name="Add to cart").first.click(timeout=10_000))

            # Navigate to cart
            await step("goto_cart",
                       page.goto(_CART_URL, wait_until="domcontentloaded", timeout=15_000))

            # Verify cart has at least one item
            cart_items = await page.get_by_role("listitem").all()
            if not cart_items:
                errors.append("cart appears empty after adding item")

            # Checkout step 1
            if await step("click_checkout",
                          page.get_by_role("button", name="Checkout").click(timeout=10_000)):
                try:
                    await page.wait_for_url("**/checkout-step-one.html", timeout=8_000)
                except Exception as exc:
                    errors.append(f"checkout-step-one: {exc!r}")

            # Fill checkout info via data-test attributes (no <label> elements in SauceDemo)
            try:
                await page.get_by_test_id("firstName").fill("Test", timeout=3_000)
                await page.get_by_test_id("lastName").fill("User", timeout=3_000)
                await page.get_by_test_id("postalCode").fill("12345", timeout=3_000)
                steps_done += 1
            except Exception as exc:
                errors.append(f"fill_info: {exc!r}")

            # Continue to step 2 (SauceDemo: <input type="submit" data-test="continue">)
            if await step("click_continue",
                          page.get_by_test_id("continue").click(timeout=10_000)):
                try:
                    await page.wait_for_url("**/checkout-step-two.html", timeout=8_000)
                except Exception as exc:
                    errors.append(f"checkout-step-two: {exc!r}")

            # Finish order
            if await step("click_finish",
                          page.get_by_role("button", name="Finish").click(timeout=10_000)):
                try:
                    await page.wait_for_url("**/checkout-complete.html", timeout=8_000)
                    steps_done += 1
                except Exception as exc:
                    errors.append(f"checkout-complete: {exc!r}")
    finally:
        await context.close()

    return {
        "name": "happy_path_standard_user",
        "passed": len(errors) == 0,
        "steps_done": steps_done,
        "error": "; ".join(errors) if errors else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2B — Bug hunting (problem_user)
# ─────────────────────────────────────────────────────────────────────────────

async def _scenario_problem_user(browser: Browser, tracer: OTelTracer) -> dict[str, Any]:
    """Detect known problem_user bugs. Returns {tests, bugs}."""
    context = await browser.new_context()
    page = await context.new_page()
    bugs: list[str] = []
    console_errors: list[str] = []
    tests: list[dict] = []

    page.on(
        "console",
        lambda msg: console_errors.append(f"[{msg.type}] {msg.text}")
        if msg.type in ("error", "warning")
        else None,
    )

    try:
        async with tracer.span("saucedemo.problem_user"):
            ok, err = await _login(page, *_PROBLEM)
            tests.append({"name": "problem_user_login", "passed": ok,
                          "error": err if not ok else None})
            if not ok:
                bugs.append(f"problem_user login failed unexpectedly: {err}")
                return {"tests": tests, "bugs": bugs}

            # ── Bug 1: Product images should be distinct ─────────────────────
            try:
                imgs = await page.get_by_role("img").all()
                product_srcs: list[str] = []
                for img in imgs:
                    src = await img.get_attribute("src")
                    alt = await img.get_attribute("alt")
                    if src and "/static/media/" in src and alt:
                        product_srcs.append(src)

                unique_srcs = set(product_srcs)
                image_bug = len(product_srcs) > 1 and len(unique_srcs) == 1
                tests.append({
                    "name": "problem_user_image_diversity",
                    "passed": not image_bug,
                    "detail": f"{len(product_srcs)} product images, {len(unique_srcs)} unique src(s)",
                })
                if image_bug:
                    bugs.append(
                        f"problem_user: all {len(product_srcs)} product images share the "
                        f"same broken src ({next(iter(unique_srcs))})"
                    )
            except Exception as exc:
                tests.append({"name": "problem_user_image_diversity",
                              "passed": False, "error": str(exc)})

            # ── Bug 2: Sort Name Z→A should reorder products ─────────────────
            try:
                # Collect product names before sorting
                links = await page.get_by_role("link").filter(has_text="Sauce Labs").all()
                names_before = [await a.inner_text() for a in links]

                sort_box = page.get_by_role("combobox")
                await sort_box.select_option(label="Name (Z to A)", timeout=5_000)
                await page.wait_for_timeout(800)

                links = await page.get_by_role("link").filter(has_text="Sauce Labs").all()
                names_after = [await a.inner_text() for a in links]

                sort_bug = len(names_before) > 1 and names_before == names_after
                tests.append({
                    "name": "problem_user_sort_z_to_a",
                    "passed": not sort_bug,
                    "detail": f"before={names_before[:2]}, after={names_after[:2]}",
                })
                if sort_bug:
                    bugs.append(
                        "problem_user: sort 'Name (Z to A)' does not reorder products "
                        f"(order unchanged: {names_before[:2]})"
                    )
            except Exception as exc:
                tests.append({"name": "problem_user_sort_z_to_a",
                              "passed": False, "error": str(exc)})

            # ── Bug 3: Add-to-cart — compare Remove-button count before/after each click
            # Avoids stale-locator issue: re-query all() each iteration instead of
            # holding references across React re-renders.
            try:
                await page.goto(_INVENTORY_URL, wait_until="domcontentloaded", timeout=15_000)
                total_items = len(await page.get_by_role("button", name="Add to cart").all())
                failed_indices: list[int] = []

                for idx in range(total_items):
                    # Re-query each iteration — first remaining "Add to cart" is next item
                    adds_before = await page.get_by_role("button", name="Add to cart").all()
                    removes_before = len(await page.get_by_role("button", name="Remove").all())
                    if not adds_before:
                        break
                    try:
                        await adds_before[0].click(timeout=5_000)
                        await page.wait_for_timeout(400)
                        removes_after = len(await page.get_by_role("button", name="Remove").all())
                        if removes_after <= removes_before:
                            # Remove count didn't increase → click had no effect
                            failed_indices.append(idx)
                    except Exception:
                        failed_indices.append(idx)

                cart_bug = len(failed_indices) > 0
                tests.append({
                    "name": "problem_user_add_to_cart_all",
                    "passed": not cart_bug,
                    "detail": (
                        f"{len(failed_indices)}/{total_items} items failed (indices {failed_indices})"
                        if cart_bug
                        else f"all {total_items} items added successfully"
                    ),
                })
                if cart_bug:
                    bugs.append(
                        f"problem_user: add-to-cart failed for {len(failed_indices)}/{total_items} "
                        f"item(s) (indices {failed_indices})"
                    )
            except Exception as exc:
                tests.append({"name": "problem_user_add_to_cart_all",
                              "passed": False, "error": str(exc)})

            # ── Console errors ────────────────────────────────────────────────
            if console_errors:
                sample = console_errors[0][:120]
                bugs.append(
                    f"problem_user: {len(console_errors)} JS console error(s) "
                    f"— e.g. {sample}"
                )
    finally:
        await context.close()

    return {"tests": tests, "bugs": bugs}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2C — Negative test (locked_out_user)
# ─────────────────────────────────────────────────────────────────────────────

async def _scenario_locked_out(browser: Browser, tracer: OTelTracer) -> dict[str, Any]:
    context = await browser.new_context()
    page = await context.new_page()
    try:
        async with tracer.span("saucedemo.locked_out"):
            ok, err_text = await _login(page, *_LOCKED)
            if not ok and err_text and "locked out" in err_text.lower():
                return {
                    "name": "locked_out_user_negative",
                    "passed": True,
                    "detail": f"Got expected error: {err_text[:100]}",
                }
            return {
                "name": "locked_out_user_negative",
                "passed": False,
                "error": f"Expected locked-out error; got ok={ok}, msg={err_text!r}",
            }
    finally:
        await context.close()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Orchestrate scenarios
# ─────────────────────────────────────────────────────────────────────────────

async def run_phase2(
    browser: Browser,
    tracer: OTelTracer,
    audit: CryptoAuditTrail,
) -> dict[str, Any]:
    happy = await _scenario_happy_path(browser, tracer)
    problem = await _scenario_problem_user(browser, tracer)
    locked = await _scenario_locked_out(browser, tracer)

    all_tests = [happy, locked, *problem["tests"]]
    bugs_found = problem["bugs"]
    tests_passed = sum(1 for t in all_tests if t.get("passed", False))

    audit.append("saucedemo.phase2", {
        "tests_generated": len(all_tests),
        "tests_passed": tests_passed,
        "bugs_found": bugs_found,
    })

    return {
        "tests_generated": len(all_tests),
        "tests_passed": tests_passed,
        "bugs_found": bugs_found,
        "details": {
            "happy_path": happy,
            "problem_user": problem,
            "locked_out": locked,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — API Fuzzing
# ─────────────────────────────────────────────────────────────────────────────

_FUZZER_PROBE_PATHS = frozenset(["/todos", "/posts", "/users", "/albums", "/comments"])
_STATIC_EXTS = frozenset([".js", ".css", ".png", ".jpg", ".ico", ".svg",
                           ".woff", ".woff2", ".gif", ".map", ".webp"])


async def run_phase3(
    browser: Browser,
    tracer: OTelTracer,
    audit: CryptoAuditTrail,
) -> dict[str, Any]:
    """Capture REAL XHR/fetch traffic while navigating the app.

    Deliberately does NOT use discover_endpoints() — that method injects its
    own probe fetches (/todos, /posts …) and then counts them as discovered
    endpoints, which is misleading for a localStorage-only app like SauceDemo.
    Instead we listen passively while driving real user flows.
    """
    context = await browser.new_context()
    page = await context.new_page()
    captured_urls: list[str] = []
    anomalies: list[str] = []
    js_errors: list[str] = []

    page.on("pageerror", lambda err: js_errors.append(str(err)))

    def _on_request(request) -> None:
        if request.resource_type not in ("xhr", "fetch"):
            return
        url: str = request.url
        # Skip static assets and the fuzzer's own known probe paths
        from urllib.parse import urlparse
        path = urlparse(url).path
        if any(path.endswith(ext) for ext in _STATIC_EXTS):
            return
        if path in _FUZZER_PROBE_PATHS:
            return
        if url not in captured_urls:
            captured_urls.append(url)

    page.on("request", _on_request)

    try:
        async with tracer.span("saucedemo.fuzz"):
            # Drive a real user journey to surface any XHR/fetch calls
            await _login(page, *_STANDARD)
            await page.goto(_INVENTORY_URL, wait_until="domcontentloaded", timeout=15_000)
            # Add an item (might trigger a request)
            await _safe(
                page.get_by_role("button", name="Add to cart").first.click(timeout=5_000),
                "add_to_cart_fuzz",
            )
            await page.goto(_CART_URL, wait_until="domcontentloaded", timeout=15_000)
            await page.goto(_TARGET_URL, wait_until="domcontentloaded", timeout=15_000)
            await page.wait_for_timeout(2_000)

        page.remove_listener("request", _on_request)

        endpoints_discovered = len(captured_urls)
        if captured_urls:
            logger.info(f"phase3: {endpoints_discovered} real XHR/fetch endpoint(s): {captured_urls}")
        else:
            logger.info("phase3: 0 real XHR/fetch endpoints — SauceDemo is localStorage-only")

        # JS page errors are anomalies
        if js_errors:
            anomalies.extend(f"js_page_error: {e[:120]}" for e in js_errors[:3])

        audit.append("saucedemo.phase3", {
            "endpoints_discovered": endpoints_discovered,
            "real_endpoints": captured_urls,
            "anomalies": len(anomalies),
        })
    finally:
        await context.close()

    note = (
        "SauceDemo is localStorage-only — no real XHR/fetch traffic observed"
        if not captured_urls else None
    )
    return {
        "endpoints_discovered": endpoints_discovered,
        "real_endpoints": captured_urls,
        "anomalies": anomalies,
        "note": note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Report
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(results: dict[str, Any]) -> pathlib.Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    p1 = results["phase1"]
    p2 = results["phase2"]
    p3 = results["phase3"]
    bugs = p2["bugs_found"]
    anomalies = p3["anomalies"]

    lines = [
        "# SauceDemo QA Agent Report",
        "",
        f"**Generated:** {now}  ",
        "**Target:** https://www.saucedemo.com/  ",
        "**Users tested:** standard_user, problem_user, locked_out_user  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cycles completed | {p1['cycles_completed']} |",
        f"| States discovered | {p1['states_discovered']} |",
        f"| Tests generated | {p2['tests_generated']} |",
        f"| Tests passed | {p2['tests_passed']} |",
        f"| Tests failed | {p2['tests_generated'] - p2['tests_passed']} |",
        f"| Bugs found | {len(bugs)} |",
        f"| API anomalies | {len(anomalies)} |",
        f"| Fuzz endpoints | {p3['endpoints_discovered']} |",
        "",
        "## Phase 1 — Exploratory Loop",
        "",
        f"Ran {p1['cycles_completed']} cycles of SFG-crawling against the SauceDemo SPA.",
        f"Recorded **{p1['states_discovered']} distinct UI states** (SFG nodes).",
        f"New states per cycle: {p1.get('new_per_cycle', [])}",
        "",
        "Cycles covered:",
        "- Cycle 1: Login → Inventory → Product detail → Back",
        "- Cycle 2: Login → Add to cart → Cart view",
        "- Cycle 3: Login → Add to cart → Checkout step 1 → Step 2",
        "",
        "## Phase 2 — Targeted Scenarios",
        "",
    ]

    details = p2.get("details", {})

    # Happy path
    hp = details.get("happy_path", {})
    hp_icon = "✅" if hp.get("passed") else "❌"
    lines += [
        f"### Scenario A: Happy Path (standard_user) {hp_icon}",
        "",
        f"Steps completed: {hp.get('steps_done', '?')}",
    ]
    if hp.get("error"):
        lines += [f"Errors: `{hp['error']}`", ""]
    else:
        lines += [
            "Flow: Login → Add to cart → Checkout → Complete order",
            "Result: Full checkout flow completed successfully.",
            "",
        ]

    # Problem user
    pb = details.get("problem_user", {})
    pb_tests = pb.get("tests", [])
    lines += ["### Scenario B: Bug Hunting (problem_user)", ""]
    for t in pb_tests:
        icon = "✅" if t.get("passed") else "❌"
        detail = t.get("detail") or t.get("error") or ""
        lines.append(f"- {icon} **{t['name']}** — {detail}")
    lines.append("")

    # Locked out
    lo = details.get("locked_out", {})
    lo_icon = "✅" if lo.get("passed") else "❌"
    lines += [
        f"### Scenario C: Negative Test (locked_out_user) {lo_icon}",
        "",
        lo.get("detail") or lo.get("error") or "",
        "",
    ]

    # Bugs
    lines += ["## Bugs Found", ""]
    if bugs:
        for bug in bugs:
            lines.append(f"- {bug}")
    else:
        lines.append("No bugs detected.")
    lines.append("")

    # Phase 3
    real_eps = p3.get("real_endpoints", [])
    lines += [
        "## Phase 3 — API Fuzzing",
        "",
        f"Real XHR/fetch endpoints observed (passive capture, no probe injection): **{p3['endpoints_discovered']}**",
    ]
    if real_eps:
        for ep in real_eps:
            lines.append(f"- `{ep}`")
    if p3.get("note"):
        lines.append(f"*{p3['note']}*")
    if anomalies:
        lines += ["", "Anomalies:"]
        for a in anomalies:
            lines.append(f"- {a}")
    else:
        lines.append("")
        lines.append("No network anomalies detected.")
    lines.append("")

    content = "\n".join(lines)
    _REPORT_PATH.write_text(content, encoding="utf-8")
    return _REPORT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> dict[str, Any]:
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    tracer = OTelTracer()
    audit = CryptoAuditTrail(path=_AUDIT_PATH)

    async with async_playwright() as pw:
        # Configure data-test as the testId attribute for SauceDemo
        pw.selectors.set_test_id_attribute("data-test")

        browser = await pw.chromium.launch(
    headless=True,
    args=_LAUNCH_ARGS,
    executable_path="/usr/bin/chromium-browser",
)

        # Setup SFG store + crawler for Phase 1
        sfg_dir = pathlib.Path(tempfile.mkdtemp(prefix="saucedemo_sfg_"))
        sfg_store = SFGStore(db_path=sfg_dir / "sfg.db")
        grounder = Grounder()
        crawler = SFGCrawler(sfg_store, grounder, CrawlerConfig())

        logger.info("═══ Phase 1: Exploratory Loop ═══")
        t0 = time.perf_counter()
        p1 = await run_phase1(browser, crawler, tracer, audit)
        logger.info(f"Phase 1 done in {time.perf_counter()-t0:.1f}s: {p1}")

        logger.info("═══ Phase 2: Targeted Scenarios ═══")
        t0 = time.perf_counter()
        p2 = await run_phase2(browser, tracer, audit)
        logger.info(f"Phase 2 done in {time.perf_counter()-t0:.1f}s: tests={p2['tests_generated']}, passed={p2['tests_passed']}")

        logger.info("═══ Phase 3: API Fuzzing ═══")
        t0 = time.perf_counter()
        p3 = await run_phase3(browser, tracer, audit)
        logger.info(f"Phase 3 done in {time.perf_counter()-t0:.1f}s: endpoints={p3['endpoints_discovered']}")

        await browser.close()

    otel_spans = tracer.flush()

    results = {
        "phase1": p1,
        "phase2": p2,
        "phase3": p3,
        "otel_spans_emitted": otel_spans,
        "audit_chain_valid": audit.verify(),
    }

    # Phase 4: Report
    report_path = _write_report(results)
    logger.info(f"Report written to {report_path}")

    output = {
        "target": "saucedemo",
        "cycles_completed": p1["cycles_completed"],
        "states_discovered": p1["states_discovered"],
        "tests_generated": p2["tests_generated"],
        "tests_passed": p2["tests_passed"],
        "bugs_found": p2["bugs_found"],
        "anomalies": p3["anomalies"],
        "report_path": str(report_path),
    }

    _RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results JSON written to {_RESULTS_PATH}")
    return output


if __name__ == "__main__":
    result = asyncio.run(main())
    print("\n" + "═" * 60)
    print(json.dumps(result, indent=2))
