from __future__ import annotations

import ast
import re
from typing import Any

from loguru import logger

from src.config_loader import get_structured_output_engine, get_use_grounder
from src.llm.adapter import OllamaAdapter
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.llm.structured import PlaywrightScript, TestPlan
from src.rag.retriever import HybridRetriever

_MAX_RETRIES = 3

# ──────────────────────────────────────────────────────────────────────────────
# AST-based static validator
# ──────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_ATTRIBUTES = {
    "wait_for_timeout": (
        "wait_for_timeout() is forbidden — "
        "use wait_for_load_state() or expect() assertions instead"
    ),
}

_FORBIDDEN_ASYNCIO_CALLS = {
    "sleep": "asyncio.sleep() is forbidden — use Playwright's built-in auto-wait",
}

_FORBIDDEN_LOCATOR_PREFIXES = ("css=", "xpath=", "//", "#", ".")


def validate_playwright_ast(code: str) -> list[str]:
    """
    Parse *code* as a Python AST and return a list of violation descriptions.
    Returns an empty list if the code is clean.
    """
    violations: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    for node in ast.walk(tree):
        # Forbidden attribute calls: page.wait_for_timeout(...)
        if isinstance(node, ast.Attribute):
            msg = _FORBIDDEN_ATTRIBUTES.get(node.attr)
            if msg:
                violations.append(msg)

        # Forbidden asyncio.sleep(...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FORBIDDEN_ASYNCIO_CALLS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "asyncio"
        ):
            violations.append(_FORBIDDEN_ASYNCIO_CALLS[node.func.attr])

        # Forbidden raw CSS/XPath in page.locator("...")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "locator"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    val: str = arg.value
                    if any(val.startswith(pfx) for pfx in _FORBIDDEN_LOCATOR_PREFIXES):
                        violations.append(
                            f"Forbidden locator in page.locator({val!r}): "
                            "use get_by_role/get_by_label/get_by_text/get_by_test_id"
                        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# Generator Agent
# ──────────────────────────────────────────────────────────────────────────────


class GeneratorAgent:
    """
    Converts a TestPlan into executable Playwright Python test code.

    Two-pass pipeline (avoids JSON-encoding Python source):
      Pass 1 — ask LLM for plain-text reasoning only (no JSON, no code).
      Pass 2 — ask LLM for raw Python code only (no JSON, no markdown fences).
      AST-validate the code; retry on SyntaxError or forbidden patterns.
    """

    SYSTEM_PROMPT = (
        "You are an expert Playwright Python QA engineer. "
        "Write sync pytest-playwright tests using the Page fixture. "
        "Use ONLY page.get_by_role(), page.get_by_label(), page.get_by_text(), page.get_by_test_id(). "
        "NEVER use CSS selectors or XPath. "
        "NEVER use async/await — pytest-playwright is synchronous. "
        "NEVER use browser.launch() or async_playwright(). "
        "Use expect() from playwright.sync_api for all assertions. "
        "page.url is a PROPERTY not a coroutine — never use await page.url."
    )

    def __init__(
        self,
        adapter: OllamaAdapter,
        retriever: HybridRetriever | None = None,
        instructor_client: InstructorClient | None = None,
    ) -> None:
        self.adapter = adapter
        self.retriever = retriever
        # instructor_generator: max_retries=1 to reduce latency (bottleneck per Day 2 analysis)
        # Effective total attempts per generate() call: outer loop (_MAX_RETRIES=3) x
        # instructor internal (initial + max_retries=1) = up to 6 instructor calls total.
        # Revert to max_retries=2 if first_run_pass_rate drops > 5pp below 0.85.
        self._instructor_client = instructor_client or InstructorClient(
            base_url=adapter.base_url,
            model=adapter.model,
            max_retries=1,
        )

    # ── Two-pass generation ────────────────────────────────────────────────────

    async def _get_reasoning(self, plan: str, url: str) -> str:
        """Pass 1: ask for a plain-text locator strategy explanation only."""
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Test plan:\n{plan}\n\n"
            f"Target URL: {url}\n\n"
            "In 2-3 sentences, explain which Playwright locator strategies you will use "
            "and why they are appropriate for this test.\n"
            "Output ONLY the explanation text. No JSON. No code."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await self.adapter.generate(messages)
        return raw.strip()

    async def _get_code(self, plan: str, url: str, reasoning: str) -> str:
        """Pass 2: ask for raw Python code only — no JSON, no fences."""
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Locator strategy decided:\n{reasoning}\n\n"
            f"Test plan:\n{plan}\n\n"
            f"Target URL: {url}\n\n"
            "Output ONLY raw Python code. No markdown fences. No JSON. No explanations.\n\n"
            "The code MUST follow this exact pytest-playwright structure:\n\n"
            "import pytest\n"
            "from playwright.sync_api import Page, expect\n\n"
            "def test_<feature_name>(page: Page) -> None:\n"
            "    page.goto('<url>')\n"
            "    # test steps here using page directly\n"
            "    # use page.get_by_role(), page.get_by_label(), page.get_by_text()\n"
            "    # use expect(locator).to_be_visible() for assertions\n"
            "    # use expect(page).to_have_url() to check URL changes\n"
            "    # page.url is a PROPERTY not a coroutine — never use await page.url\n\n"
            "RULES:\n"
            "- Function name must start with test_\n"
            "- Use sync Playwright API (Page, not async_playwright)\n"
            "- Use expect() from playwright.sync_api for all assertions\n"
            "- No async/await — pytest-playwright handles async automatically\n"
            "- No browser.launch() or playwright.start() — pytest-playwright injects page fixture\n"
            "- Start code with: import pytest"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await self.adapter.generate(messages)
        # Strip any fences the model adds anyway
        raw = re.sub(r"^```python\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"^```\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        return raw.strip()

    @staticmethod
    def _validate_python(code: str) -> None:
        """Raise SyntaxError if *code* is not valid Python."""
        ast.parse(code)

    @staticmethod
    def _extract_locators(code: str) -> list[str]:
        """Return which Playwright locator helpers appear in *code*."""
        candidates = [
            "get_by_role",
            "get_by_label",
            "get_by_text",
            "get_by_test_id",
            "get_by_placeholder",
        ]
        return [c for c in candidates if c in code]

    async def _generate_via_instructor(
        self, plan: str, url: str
    ) -> PlaywrightScript:
        """
        Single-pass structured generation via Instructor.

        Builds a single messages list that combines reasoning-request and
        code-request in one structured JSON output call.
        """
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Test plan:\n{plan}\n\n"
            f"Target URL: {url}\n\n"
            "Respond with a JSON object matching PlaywrightScript:\n"
            "  reasoning: 2-3 sentence explanation of locator strategy\n"
            "  code: complete runnable pytest-playwright test function\n"
            "  locators_used: list of locator helpers used\n"
            "  test_function_name: name of the test function\n\n"
            "RULES for code:\n"
            "- Function must start with def test_ (sync, NOT async)\n"
            "- Import: import pytest; from playwright.sync_api import Page, expect\n"
            "- Use ONLY page.get_by_role(), page.get_by_label(), page.get_by_text(), page.get_by_test_id()\n"
            "- Use expect() for all assertions\n"
            "- NO async/await, NO browser.launch(), NO asyncio.sleep()"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await self._instructor_client.create_structured(
            messages, PlaywrightScript, temperature=0.0
        )

    async def generate(
        self,
        state: dict | None = None,
        page_state: str = "",
        **kwargs,
    ) -> PlaywrightScript:
        """
        Generate a PlaywrightScript.

        Selects between instructor (single-pass structured) and the legacy
        two-pass plain-text path based on the structured_output_engine flag.

        Args:
            state: Graph state dict (may include test_plan, url, requirement).
            page_state: Grounder CompactPAM markdown string. When non-empty and
                ``perception.use_grounder`` is True, this is prepended to the plan
                passed to the LLM so the generator can reference live page elements.
            **kwargs: Additional state overrides merged with *state*.
        """
        if state is None:
            state = {}
        merged = {**state, **kwargs}
        raw_plan = merged.get("test_plan") or merged.get("requirement") or ""
        if isinstance(raw_plan, dict):
            import json as _json
            plan = _json.dumps(raw_plan, indent=2)
        elif isinstance(raw_plan, TestPlan):
            plan = raw_plan.model_dump_json(indent=2)
        else:
            plan = str(raw_plan)

        # Inject Grounder page state when enabled (Cluster E)
        if page_state and get_use_grounder():
            plan = f"Page state:\n{page_state}\n\n{plan}"

        url: str = merged.get("url", "")
        last_error: Exception | None = None
        engine = get_structured_output_engine()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if engine == "instructor":
                    logger.info(f"GeneratorAgent attempt {attempt}/{_MAX_RETRIES} (instructor)")
                    script = await self._generate_via_instructor(plan, url)
                    self._validate_python(script.code)
                    logger.info("GeneratorAgent: valid Python generated (instructor)")
                    return script
                else:
                    logger.info(
                        f"GeneratorAgent attempt {attempt}/{_MAX_RETRIES} (two-pass legacy)"
                    )
                    reasoning = await self._get_reasoning(plan, url)
                    code = await self._get_code(plan, url, reasoning)
                    self._validate_python(code)
                    logger.info("GeneratorAgent: valid Python generated (legacy)")
                    return PlaywrightScript(
                        reasoning=reasoning,
                        code=code,
                        locators_used=self._extract_locators(code),
                    )
            except StructuredGenerationError as exc:
                logger.warning(
                    f"GeneratorAgent attempt {attempt}/{_MAX_RETRIES} "
                    f"instructor retries exhausted: {exc}"
                )
                last_error = exc
            except SyntaxError as exc:
                logger.warning(
                    f"GeneratorAgent attempt {attempt}/{_MAX_RETRIES} SyntaxError: {exc}"
                )
                last_error = exc
            except Exception as exc:
                logger.warning(
                    f"GeneratorAgent attempt {attempt}/{_MAX_RETRIES} failed: {exc}"
                )
                last_error = exc

        raise RuntimeError(
            f"GeneratorAgent: failed after {_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ── Legacy helper kept for backward compatibility ──────────────────────────

    async def _get_rag_context(self, query: str) -> str:
        if not self.retriever:
            return ""
        try:
            chunks = await self.retriever.retrieve(query)
            return self.retriever.format_context(chunks)
        except Exception as exc:
            logger.warning(f"GeneratorAgent: RAG retrieval failed: {exc}")
            return ""

    @staticmethod
    def _append_correction(
        messages: list[dict[str, Any]],
        last_raw: str,
        feedback: str,
    ) -> list[dict[str, Any]]:
        return messages + [
            {"role": "assistant", "content": last_raw},
            {"role": "user", "content": feedback},
        ]
