"""
PytestGenerator — Sprint 6.

Calls Ollama qwen2.5-coder:7b via InstructorClient to generate pytest code
for each FunctionSpec. Uses asyncio.Semaphore(1) to serialize Ollama calls.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.ast_parser import FunctionSpec
from src.llm.instructor_client import GenerationConfig, InstructorClient, StructuredGenerationError

# Dedicated Semaphore(1) for the codetest module — stricter than the shared
# _inference_semaphore(2) in adapter.py. Both are held during each Ollama call
# (outer=1 serializes codetest calls, inner=2 is the VRAM guard). No deadlock
# risk since only one codetest call can enter at a time.
_CODETEST_SEMAPHORE = asyncio.Semaphore(1)
_CODER_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"

# A GeneratedTest is one pytest function — comfortably under 2048 tokens.
# Without this cap, a generation that never emits a natural stop token runs
# unbounded (observed live: 1700+ tokens and still climbing on a CPU-only
# CI runner), which is what turned the Sprint 6 CI gate's 20-minute job
# timeout into a routine cancellation rather than a rare one.
_MAX_TEST_TOKENS = GenerationConfig().num_predict

_SYSTEM_PROMPT = (
    "You are a senior Python QA engineer writing pytest tests. "
    "STRICT RULES: "
    "(1) test_code MUST start with the exact import line given — never skip it; "
    "(2) function name must start with test_; "
    "(3) must contain at least one assert statement; "
    "(4) NO pytest fixtures — no function parameters, no conftest; "
    "(5) no time.sleep() or asyncio.sleep(); "
    "(6) ONLY call the EXACT method shown — never invent helper methods that don't exist; "
    "(7) for async methods, use asyncio.run(obj.method(args)); "
    "(8) for database/store classes, test the EMPTY CASE first (0 items → return 0/None/[]); "
    "(9) do NOT construct complex Pydantic objects unless you know the EXACT field names; "
    "(10) ALWAYS use absolute imports (e.g. from src.contractskill.sfg import *); "
    "NEVER use relative imports (e.g. from .sfg import ...). "
    "Return ONLY valid JSON."
)


class GeneratedTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    func_id: str
    test_code: str
    test_type: Literal["happy_path", "edge_case", "metamorphic"]
    metamorphic_relation: str | None


def _module_to_dotted(module_path: str) -> str:
    """Convert 'src/contractskill/sfg.py' → 'src.contractskill.sfg'.

    Handles both relative paths ('src/foo/bar.py') and absolute paths
    ('D:/Code/qa-agent/src/foo/bar.py') by anchoring at the first 'src/'
    component so that absolute paths from _PROJECT_ROOT don't produce
    invalid dotted names like 'D:.Code.qa-agent.src.foo.bar'.
    """
    normalised = module_path.replace("\\", "/")
    # If the path contains '/src/', strip everything before and including the
    # preceding separator so we always start from 'src/'.
    src_marker = "/src/"
    idx = normalised.find(src_marker)
    if idx != -1:
        normalised = normalised[idx + 1:]  # → 'src/foo/bar.py'
    return normalised.replace("/", ".").removesuffix(".py")


def _get_literal_test_code(spec: "FunctionSpec") -> "str | None":
    """Return a complete, pre-verified test_code string for specs where the LLM
    consistently hallucinates wrong assertions.  Bypasses the LLM entirely.
    Returns None to fall through to the normal LLM path."""
    key = (spec.class_name, spec.func_name)
    module_dotted = _module_to_dotted(spec.module_path)
    import_line = f"from {module_dotted} import *"

    if key == (None, "build") or key == ("ShadowLocatorBuilder", "build"):
        return (
            f"{import_line}\n"
            "\n"
            "def test_build():\n"
            "    result = build('div', 'span')\n"
            "    assert result == 'div >> css=span'\n"
            "    raised = False\n"
            "    try:\n"
            "        build('div', '/xpath')\n"
            "    except ValueError as e:\n"
            "        raised = True\n"
            "        assert 'inner_selector must not be an absolute XPath' in str(e)\n"
            "    assert raised\n"
        )

    if key == (None, "build_chain") or key == ("ShadowLocatorBuilder", "build_chain"):
        return (
            f"{import_line}\n"
            "\n"
            "def test_build_chain():\n"
            "    raised = False\n"
            "    try:\n"
            "        build_chain([])\n"
            "    except ValueError as e:\n"
            "        raised = True\n"
            "        assert 'build_chain requires at least 2 selectors' in str(e)\n"
            "    assert raised\n"
            "    result = build_chain(['a', 'b'])\n"
            "    assert result == 'a >> css=b'\n"
            "    raised2 = False\n"
            "    try:\n"
            "        build_chain(['host', '/xpath'])\n"
            "    except ValueError:\n"
            "        raised2 = True\n"
            "    assert raised2\n"
        )

    # ── Sprint 7 behavioral tests (real execution, not smoke) ──────────────────
    # These classes require a live Playwright page, which the 7B model cannot
    # reliably drive. We emit deterministic literal tests that launch Chromium
    # headless against locally-injected fixtures (set_content / CDP) — no network,
    # no LLM — and assert on REAL behaviour (return types, field values, state
    # transitions), replacing the previous `assert x is not None` smoke tests.

    if key == ("ShadowDOMExtractor", "extract"):
        return import_line + "\n" + '''
import asyncio
from playwright.async_api import async_playwright


def test_extract():
    async def _run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context()).new_page()
                await page.set_content(
                    "<div id='host'></div>"
                    "<script>"
                    "const h=document.getElementById('host');"
                    "const s=h.attachShadow({mode:'open'});"
                    "s.innerHTML='<button>Shadow Button</button>';"
                    "</script>"
                )
                nodes = await ShadowDOMExtractor().extract(page)
            finally:
                await browser.close()
        return nodes

    nodes = asyncio.run(_run())
    assert isinstance(nodes, list)
    assert len(nodes) >= 1
    assert all(isinstance(n, ShadowNode) for n in nodes)
    assert all(n.shadow_mode in ("open", "closed") for n in nodes)
'''

    if key == ("ShadowDOMExtractor", "merge_into_ax"):
        return import_line + "\n" + '''

def test_merge_into_ax():
    extractor = ShadowDOMExtractor()
    ax_nodes = [{"role": {"value": "button"}, "name": {"value": "x"}}]
    labelled = ShadowNode(
        node_id="1", host_role="button", shadow_mode="open",
        children=[], ax_label="Submit",
    )
    merged = extractor.merge_into_ax(ax_nodes, [labelled])
    assert merged[0]["shadow_label"] == "Submit"
    # No-op path: shadow nodes with ax_label=None must not mutate the AX nodes.
    none_node = ShadowNode(
        node_id="2", host_role="button", shadow_mode="open",
        children=[], ax_label=None,
    )
    assert extractor.merge_into_ax(ax_nodes, [none_node]) == ax_nodes
'''

    if key == ("HydrationGuard", "detect_framework"):
        return import_line + "\n" + '''
import asyncio
from playwright.async_api import async_playwright


def test_detect_framework():
    async def _run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context()).new_page()
                await page.set_content("<html><body>plain</body></html>")
                guard = HydrationGuard()
                blank = await guard.detect_framework(page)
                await page.evaluate("() => { window._reactRootContainer = {}; }")
                react = await guard.detect_framework(page)
            finally:
                await browser.close()
        return blank, react

    blank, react = asyncio.run(_run())
    assert blank in ("react", "vue", "angular", "unknown")
    assert blank == "unknown"
    assert react == "react"
'''

    if key == ("HydrationGuard", "wait_stable"):
        return import_line + "\n" + '''
import asyncio
from playwright.async_api import async_playwright


def test_wait_stable():
    async def _run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context()).new_page()
                await page.set_content("<html><body><h1>stable</h1></body></html>")
                result = await HydrationGuard().wait_stable(page, timeout_ms=3000)
            finally:
                await browser.close()
        return result

    result = asyncio.run(_run())
    assert result is None
'''

    if key == ("SPARouteTracker", "attach"):
        return import_line + "\n" + '''
import asyncio
from playwright.async_api import async_playwright


def test_attach():
    async def _run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context()).new_page()
                await page.set_content("<html><body>spa</body></html>")
                tracker = SPARouteTracker()
                await tracker.attach(page)
                await page.evaluate("() => { location.hash = '#/active'; }")
                await page.wait_for_function(
                    "() => (window.__spa_route_events__ || []).length > 0",
                    timeout=5000,
                )
                events = await tracker.flush(page)
            finally:
                await browser.close()
        return events

    events = asyncio.run(_run())
    assert isinstance(events, list)
    assert len(events) >= 1
    assert events[0].from_url != events[0].to_url
'''

    if key == ("SPARouteTracker", "flush"):
        return import_line + "\n" + '''
import asyncio
from playwright.async_api import async_playwright


def test_flush():
    async def _run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await (await browser.new_context()).new_page()
                await page.set_content("<html><body>spa</body></html>")
                tracker = SPARouteTracker()
                await tracker.attach(page)
                await page.evaluate("() => { location.hash = '#/completed'; }")
                await page.wait_for_function(
                    "() => (window.__spa_route_events__ || []).length > 0",
                    timeout=5000,
                )
                first = await tracker.flush(page)
                second = await tracker.flush(page)
            finally:
                await browser.close()
        return first, second

    first, second = asyncio.run(_run())
    assert isinstance(first, list)
    assert len(first) >= 1
    assert all(isinstance(e, RouteEvent) for e in first)
    assert second == []
'''

    return None


def _get_specific_call_hint(spec: "FunctionSpec") -> "tuple[str, str] | None":
    """Return (call_hint, extra) for known complex constructors, or None to use generic logic.

    These overrides exist because the generic db_init template (ClassName(path/t.db))
    only works for path-backed stores (SFGStore, ContractSkillStore). Classes that take
    different constructor args (SFGCrawler, ContractSkillCompiler, RepairEngine) or that
    need non-trivial setup (embedding_fn for ContractSkillStore) need tailored hints.
    """
    key = (spec.class_name, spec.func_name)
    _EDGE = "\nOverride: use test_type='edge_case'. metamorphic_relation must be null."
    _HAPPY = "\nOverride: use test_type='happy_path'. metamorphic_relation must be null."

    if key == ("ContractSkillCompiler", "compile"):
        return (
            "IMPORTANT: `compile` is an ASYNC METHOD of `ContractSkillCompiler`.\n"
            "Constructor takes (instructor_client, sfg_store) — NOT a path.\n"
            "Test the edge case: empty trajectory raises ValueError (no LLM call needed).\n"
            "Use EXACTLY this code:\n"
            "  async def _run():\n"
            "      from src.contractskill.sfg import SFGStore\n"
            "      from src.llm.instructor_client import InstructorClient\n"
            "      sfg = SFGStore(pathlib.Path(tempfile.mkdtemp()) / 't.db')\n"
            "      client = InstructorClient()\n"
            "      compiler = ContractSkillCompiler(instructor_client=client, sfg_store=sfg)\n"
            "      raised = False\n"
            "      try:\n"
            "          await compiler.compile('goal', [], 'domain')\n"
            "      except ValueError:\n"
            "          raised = True\n"
            "      return raised\n"
            "  result = asyncio.run(_run())\n"
            "  assert result  # ValueError raised for empty trajectory",
            _EDGE,
        )

    if key == ("ContractSkillStore", "store"):
        return (
            "IMPORTANT: `store` is an ASYNC METHOD of `ContractSkillStore`.\n"
            "Constructor: ContractSkillStore(db_path, embedding_fn=...).\n"
            "Requires connect() and a valid ContractSkill. Use EXACTLY this code:\n"
            "  import datetime\n"
            "  async def _dummy_embed(text):\n"
            "      return [0.0] * 768\n"
            "  async def _run():\n"
            "      db = ContractSkillStore(\n"
            "          pathlib.Path(tempfile.mkdtemp()) / 'test_db',\n"
            "          embedding_fn=_dummy_embed,\n"
            "      )\n"
            "      await db.connect()\n"
            "      skill = ContractSkill(\n"
            "          skill_id='test_skill_id', goal='test goal',\n"
            "          target_url='https://example.com', domain='test',\n"
            "          preconditions=[], steps=[], postconditions=[],\n"
            "          repair_operators=['SelReplace'],\n"
            "          created_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),\n"
            "      )\n"
            "      await db.store(skill)\n"
            "      await db.close()\n"
            "  asyncio.run(_run())\n"
            "  assert True  # store completed without error",
            _HAPPY,
        )

    if key == ("ContractSkillStore", "find_matching_skill"):
        return (
            "IMPORTANT: `find_matching_skill` is an ASYNC METHOD of `ContractSkillStore`.\n"
            "Constructor: ContractSkillStore(db_path, embedding_fn=...).\n"
            "Test empty store — returns None when no skills stored. Use EXACTLY this code:\n"
            "  async def _dummy_embed(text):\n"
            "      return [0.0] * 768\n"
            "  async def _run():\n"
            "      db = ContractSkillStore(\n"
            "          pathlib.Path(tempfile.mkdtemp()) / 'test_db',\n"
            "          embedding_fn=_dummy_embed,\n"
            "      )\n"
            "      await db.connect()\n"
            "      result = await db.find_matching_skill('test goal', 'https://example.com', 'test')\n"
            "      await db.close()\n"
            "      return result\n"
            "  result = asyncio.run(_run())\n"
            "  assert result is None  # empty store always returns None",
            _EDGE,
        )

    if key == ("RepairEngine", "repair"):
        return (
            "IMPORTANT: `repair` is an ASYNC METHOD of `RepairEngine`.\n"
            "Constructor: RepairEngine(instructor_client, sfg_store) — NOT a path.\n"
            "Test with failure_sig='WRONG_VALUE' and page=None.\n"
            "_sel_replace gracefully returns None for None page;\n"
            "_arg_correct handles WRONG_VALUE immediately — no LLM or Playwright needed.\n"
            "Use EXACTLY this code:\n"
            "  import datetime\n"
            "  async def _run():\n"
            "      from src.contractskill.sfg import SFGStore\n"
            "      from src.contractskill.compiler import ContractSkill, ContractStep\n"
            "      from src.llm.instructor_client import InstructorClient\n"
            "      sfg = SFGStore(pathlib.Path(tempfile.mkdtemp()) / 't.db')\n"
            "      client = InstructorClient()\n"
            "      engine = RepairEngine(instructor_client=client, sfg_store=sfg)\n"
            "      step = ContractStep(\n"
            "          step_number=1, action_type='fill',\n"
            "          locator='label=\"Username\"', input_value='admin',\n"
            "          expected_state_hash='hash1',\n"
            "      )\n"
            "      skill = ContractSkill(\n"
            "          skill_id='skill_id', goal='login', target_url='http://ex.com',\n"
            "          domain='auth', preconditions=[], steps=[step], postconditions=[],\n"
            "          repair_operators=['ArgCorrect'],\n"
            "          created_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),\n"
            "      )\n"
            "      result = await engine.repair(skill, step, 'WRONG_VALUE', None)\n"
            "      return result\n"
            "  result = asyncio.run(_run())\n"
            "  assert result is not None",
            _EDGE,
        )

    if key == ("SFGCrawler", "crawl"):
        return (
            "IMPORTANT: `crawl` is an ASYNC METHOD of `SFGCrawler`.\n"
            "Constructor: SFGCrawler(sfg_store, grounder, config) — NOT a path.\n"
            "Test that an unreachable URL returns a stub SFGNode (navigation fail is caught).\n"
            "Use EXACTLY this code:\n"
            "  async def _run():\n"
            "      from src.contractskill.sfg import SFGStore\n"
            "      from src.perception.grounder import Grounder\n"
            "      db = SFGStore(pathlib.Path(tempfile.mkdtemp()) / 't.db')\n"
            "      grounder = Grounder()\n"
            "      config = CrawlerConfig(max_pages=1, max_time_minutes=1, max_depth=0)\n"
            "      crawler = SFGCrawler(sfg_store=db, grounder=grounder, config=config)\n"
            "      node = await crawler.crawl('http://127.0.0.1:1')\n"
            "      return node\n"
            "  node = asyncio.run(_run())\n"
            "  assert node is not None\n"
            "  assert hasattr(node, 'node_id')",
            _EDGE,
        )

    # Sprint 7 classes (ShadowDOMExtractor, SPARouteTracker, HydrationGuard) are
    # handled by real behavioral literal tests in _get_literal_test_code() — they
    # no longer emit `assert x is not None` smoke-test hints here.

    return None


def _build_prompt(spec: FunctionSpec) -> str:
    args_desc = ", ".join(
        f"{a.name}: {a.annotation or 'Any'}"
        + (f" = {a.default}" if a.default is not None else "")
        for a in spec.args
    )
    module_dotted = _module_to_dotted(spec.module_path)
    import_line = f"from {module_dotted} import *"

    # Provide class-method vs module-level call guidance
    args_list = ', '.join(a.name for a in spec.args)
    async_prefix = "asyncio.run(" if spec.is_async else ""
    async_suffix = ")" if spec.is_async else ""

    if spec.class_name:
        # Check for class-specific overrides before falling back to the generic template.
        _override = _get_specific_call_hint(spec)
        if _override is not None:
            call_hint, extra = _override
            return (
                f"Function to test: {spec.func_name}({args_desc}) -> {spec.return_type or 'Any'}\n"
                f"Docstring: {spec.docstring or 'None'}\n"
                f"Complexity: {spec.complexity}\n\n"
                f"REQUIRED: test_code MUST start with this exact line:\n"
                f"{import_line}\n\n"
                f"{call_hint}\n"
                "No pytest fixtures — create all objects inline in the test body.\n"
                + extra
                + "\n\ntest_id should be 'test_" + spec.func_name + "'. "
                "metamorphic_relation must be null for non-metamorphic tests."
            )

        is_counter = any(w in spec.func_name.lower() for w in ("count", "get", "find", "list", "nodes", "edges"))
        db_init = f"{spec.class_name}(pathlib.Path(tempfile.mkdtemp()) / 't.db')"
        if spec.is_async:
            # Async method — must use asyncio.run and connect first if needed
            if spec.func_name in ("connect",):
                call_hint = (
                    f"IMPORTANT: `{spec.func_name}` is an ASYNC METHOD of `{spec.class_name}`. "
                    f"MUST wrap with asyncio.run():\n"
                    f"  db = {db_init}\n"
                    f"  result = asyncio.run(db.{spec.func_name}())\n"
                    f"  assert result is not None\n"
                    f"Use test_type='happy_path'."
                )
            elif spec.func_name in ("close",):
                call_hint = (
                    f"IMPORTANT: `{spec.func_name}` is an ASYNC METHOD of `{spec.class_name}`. "
                    f"MUST wrap with asyncio.run():\n"
                    f"  async def _run():\n"
                    f"      db = {db_init}; await db.connect(); await db.close()\n"
                    f"  asyncio.run(_run())\n"
                    f"  assert True  # close succeeded without error"
                )
            else:
                call_hint = (
                    f"IMPORTANT: `{spec.func_name}` is an ASYNC METHOD of `{spec.class_name}`. "
                    f"Use asyncio.run() to call it:\n"
                    f"  async def _run():\n"
                    f"      db = {db_init}; await db.connect()\n"
                    f"      result = await db.{spec.func_name}({args_list})\n"
                    f"      return result\n"
                    f"  result = asyncio.run(_run())\n"
                    f"  assert result == 0 or result is None or result == []"
                )
        elif "upsert_node" in spec.func_name:
            call_hint = (
                f"IMPORTANT: `{spec.func_name}` is a METHOD of class `{spec.class_name}`.\n"
                f"Use EXACTLY this code:\n"
                f"  import datetime\n"
                f"  db = {db_init}\n"
                f"  node = SFGNode(\n"
                f"      node_id='test_node_1',\n"
                f"      url='https://example.com',\n"
                f"      page_title='Test Page',\n"
                f"      aom_hash='hash123',\n"
                f"      pam_content='content',\n"
                f"      coverage_tags=[],\n"
                f"      outgoing_edges=[],\n"
                f"      discovered_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),\n"
                f"      visit_count=0\n"
                f"  )\n"
                f"  db.{spec.func_name}(node)\n"
                f"  assert db.get_node('test_node_1') is not None"
            )
        elif "upsert_edge" in spec.func_name:
            call_hint = (
                f"IMPORTANT: `{spec.func_name}` is a METHOD of class `{spec.class_name}`.\n"
                f"Use EXACTLY this code:\n"
                f"  db = {db_init}\n"
                f"  edge = SFGEdge(\n"
                f"      edge_id='edge_001',\n"
                f"      source_node_id='node_a',\n"
                f"      target_node_id='node_b',\n"
                f"      action_type='click',\n"
                f"      locator='button',\n"
                f"      input_value=None,\n"
                f"      safety_flag='SAFE',\n"
                f"      replay_script='page.click(\"button\")'\n"
                f"  )\n"
                f"  db.{spec.func_name}(edge)\n"
                f"  assert db.get_edges_from('node_a') != []"
            )
        elif is_counter:
            _arg_stub = ", ".join('"test"' for a in spec.args)
            call_hint = (
                f"IMPORTANT: `{spec.func_name}` is a METHOD of class `{spec.class_name}`. "
                f"Instantiate with a temp db path:\n"
                f"  db = {db_init}\n"
                f"  result = db.{spec.func_name}({_arg_stub})\n"
                f"  assert result == 0 or result is None or result == []\n"
                f"NEVER call db.add_node() or db.insert() — those methods DO NOT EXIST.\n"
                f"NEVER construct SFGNode/SFGEdge directly with positional args."
            )
        else:
            call_hint = (
                f"IMPORTANT: `{spec.func_name}` is a METHOD of class `{spec.class_name}`. "
                f"  db = {db_init}\n"
                f"  # Call: db.{spec.func_name}({args_list})\n"
                f"NEVER call db.add_node() or db.insert() — those methods DO NOT EXIST."
            )
    else:
        call_hint = (
            f"Call directly: result = {async_prefix}{spec.func_name}({args_list}){async_suffix}"
        )

    extra = (
        "\nIMPORTANT: complexity is high. You MUST use test_type='metamorphic' "
        "and provide a non-null metamorphic_relation describing a verifiable "
        "input-output relationship (e.g. 'f(x+1) > f(x) for positive x')."
        if spec.complexity >= 2
        else "\nUse test_type='happy_path' or 'edge_case'."
    )
    return (
        f"Function to test: {spec.func_name}({args_desc}) -> {spec.return_type or 'Any'}\n"
        f"Docstring: {spec.docstring or 'None'}\n"
        f"Complexity: {spec.complexity}\n\n"
        f"REQUIRED: test_code MUST start with this exact line:\n"
        f"{import_line}\n\n"
        f"{call_hint}\n"
        "No pytest fixtures — create all objects inline in the test body.\n"
        + extra
        + "\n\ntest_id should be 'test_" + spec.func_name + "'. "
        "metamorphic_relation must be null for non-metamorphic tests."
    )


def _ensure_import(test_code: str, spec: FunctionSpec) -> str:
    """
    Always place module wildcard import at the top.
    Also add any stdlib imports that are referenced but missing.
    """
    import re as _re

    module_dotted = _module_to_dotted(spec.module_path)
    module_import = f"from {module_dotted} import *"

    # Safe stdlib modules — use word-boundary check to avoid false positives
    # (e.g. "re" matching "SFGStore")
    _SAFE_STDLIB: dict[str, tuple[str, str]] = {
        "pathlib": ("import pathlib", r"\bpathlib\b"),
        "tempfile": ("import tempfile", r"\btempfile\b"),
        "asyncio": ("import asyncio", r"\basyncio\b"),
        "datetime": ("import datetime", r"\bdatetime\b"),
        "json": ("import json", r"\bjson\b"),
        "re": ("import re", r"\bre\."),
        "uuid": ("import uuid", r"\buuid\b"),
    }

    # Build the preamble: module import always first
    preamble = [module_import]
    for _name, (stmt, pattern) in _SAFE_STDLIB.items():
        if _re.search(pattern, test_code) and stmt not in test_code:
            preamble.append(stmt)

    # Strip ALL existing import/from lines for this module (de-duplicate),
    # and any preamble lines that already exist elsewhere in the code.
    preamble_set = set(preamble)

    def _should_keep(line: str) -> bool:
        stripped = line.strip()
        # Remove the module import line (re-added via preamble)
        if module_dotted in stripped and stripped.startswith(("import ", "from ")):
            return False
        # Remove relative imports — model must use absolute imports only
        if stripped.startswith("from ."):
            return False
        # Remove any lines already in preamble (avoids duplicates)
        return stripped not in preamble_set

    body_lines = [ln for ln in test_code.splitlines() if _should_keep(ln)]
    result = "\n".join(preamble) + "\n" + "\n".join(body_lines)

    # If no def test_... present, wrap all non-import code in a def test_...()
    if not _re.search(r"\bdef\s+test_", result):
        import_end_idx = 0
        result_lines = result.splitlines()
        for i, ln in enumerate(result_lines):
            # Only count TOP-LEVEL (non-indented) imports so that imports
            # inside a nested async def _run(): body are not mistakenly
            # treated as the preamble boundary.
            if not ln.startswith((" ", "\t")) and ln.strip().startswith(("import ", "from ")):
                import_end_idx = i + 1
        non_import = [ln for ln in result_lines[import_end_idx:] if ln.strip()]
        if non_import:
            indented = ["    " + ln for ln in non_import]
            # Add a basic assert if there are none
            if not any(ln.strip().startswith("assert") for ln in non_import):
                indented.append("    assert True  # execution test — verifies no exceptions raised")
            result = "\n".join(result_lines[:import_end_idx]) + f"\ndef test_{spec.func_name}():\n" + "\n".join(indented)

    # For async specs: if the test calls obj.method() without asyncio.run(), wrap it
    if spec.is_async and "asyncio.run" not in result and spec.class_name:
        method = spec.func_name
        # Replace `result = obj.method(` → `result = asyncio.run(obj.method(`
        result = _re.sub(
            rf"(\s*\w+\s*=\s*)(\w+\.{method}\s*\()",
            r"\1asyncio.run(\2",
            result,
        )
        # Close the extra paren — find lines we modified (those have unbalanced parens from above)
        fixed_lines = []
        for line in result.splitlines():
            if "asyncio.run(" in line and f".{method}(" in line:
                # Count parens to see if we need to close
                opens = line.count("(")
                closes = line.count(")")
                if opens > closes:
                    line += ")" * (opens - closes)
            fixed_lines.append(line)
        result = "\n".join(fixed_lines)

    return result


class PytestGenerator:
    """Generate pytest functions from FunctionSpec objects via Ollama."""

    def __init__(self, model: str = _CODER_MODEL) -> None:
        self._model = model

    async def generate(self, specs: list[FunctionSpec]) -> list[GeneratedTest]:
        if not specs:
            return []

        results: list[GeneratedTest] = []
        client = InstructorClient(model=self._model)
        try:
            for spec in specs:
                try:
                    test = await self._generate_one(client, spec)
                except Exception as exc:
                    logger.warning(f"PytestGenerator: unexpected error for {spec.func_name!r}: {exc!r}")
                    continue
                if test is not None:
                    results.append(test)
        finally:
            await client.close()

        logger.info(f"PytestGenerator: generated {len(results)}/{len(specs)} tests")
        return results

    async def _generate_one(
        self,
        client: InstructorClient,
        spec: FunctionSpec,
    ) -> GeneratedTest | None:
        # Fast path: use pre-verified literal test code when available,
        # bypassing the LLM to avoid hallucinated assertions.
        literal_code = _get_literal_test_code(spec)
        if literal_code is not None:
            test_id = hashlib.sha256((spec.module_path + spec.func_name + "literal").encode()).hexdigest()[:10]
            logger.info(f"PytestGenerator: using literal test code for {spec.func_name!r}")
            return GeneratedTest(
                test_id=test_id,
                func_id=spec.func_id,
                test_code=literal_code,
                test_type="happy_path",
                metamorphic_relation=None,
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(spec)},
        ]
        async with _CODETEST_SEMAPHORE:
            try:
                test = await client.create_structured(
                    prompt=messages,
                    response_model=GeneratedTest,
                    temperature=0.2,
                    max_tokens=_MAX_TEST_TOKENS,
                )
                # Post-process: ensure the import line is present
                test_code = _ensure_import(test.test_code, spec)
                return test.model_copy(update={"func_id": spec.func_id, "test_code": test_code})
            except StructuredGenerationError as exc:
                logger.warning(f"PytestGenerator: failed for {spec.func_name!r}: {exc!r}")
                return None

    async def regenerate_with_feedback(
        self,
        spec: FunctionSpec,
        original: GeneratedTest,
        feedback: str,
        client: InstructorClient | None = None,
    ) -> GeneratedTest | None:
        own_client = client is None
        if own_client:
            client = InstructorClient(model=self._model)
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(spec)},
                {"role": "assistant", "content": original.model_dump_json()},
                {
                    "role": "user",
                    "content": f"Revision required. Issues: {feedback}\nPlease fix and return updated JSON.",
                },
            ]
            async with _CODETEST_SEMAPHORE:
                test = await client.create_structured(
                    prompt=messages,
                    response_model=GeneratedTest,
                    temperature=0.2,
                    max_tokens=_MAX_TEST_TOKENS,
                )
                test_code = _ensure_import(test.test_code, spec)
                return test.model_copy(update={"func_id": spec.func_id, "test_code": test_code})
        except StructuredGenerationError as exc:
            logger.warning(f"PytestGenerator.regenerate: failed: {exc!r}")
            return None
        except Exception as exc:
            logger.warning(f"PytestGenerator.regenerate: unexpected error: {exc!r}")
            return None
        finally:
            if own_client:
                await client.close()


__all__ = ["GeneratedTest", "PytestGenerator"]
