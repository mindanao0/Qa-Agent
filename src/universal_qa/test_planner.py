from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.contractskill.sfg import SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.universal_qa.models import TestCase

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_XSS_PAYLOAD = "<script>alert('xss')</script>"
_SQLI_PAYLOAD = "' OR '1'='1"


class _FuncItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    priority: Literal["high", "medium", "low"] = "medium"
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str] = Field(..., min_length=1)
    expected_outcome: str


class _FuncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_cases: list[_FuncItem]


class UniversalTestPlanner:
    """Generates TestCase list from SFGStore nodes.

    Functional: LLM analyses each form-node's PAM content.
    Accessibility: rule-based, 1 TestCase per node.
    Security: template XSS + SQLi per form-node.
    """

    def __init__(self) -> None:
        self._client = InstructorClient()

    async def plan(self, sfg_store: SFGStore, start_url: str) -> list[TestCase]:
        nodes = sfg_store.get_nodes_by_url_prefix(start_url)
        if not nodes:
            logger.warning("UniversalTestPlanner: no SFG nodes found for %s", start_url)
            return []

        functional = await self._plan_functional(nodes, start_url)
        accessibility = self._plan_accessibility(nodes, start_url)
        security = self._plan_security(nodes, start_url)

        all_cases = functional + accessibility + security
        return self._sort_by_priority(all_cases)

    async def _plan_functional(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        form_nodes = [n for n in nodes if "form" in n.coverage_tags]
        if not form_nodes:
            form_nodes = nodes[:5]  # fallback: use first 5 pages

        for node in form_nodes[:10]:  # cap at 10 to avoid LLM overload
            prompt = (
                f"You are a QA engineer. Given this web page, write 2-3 test cases "
                f"(at least 1 happy-path + 1 negative) as JSON.\n\n"
                f"URL: {node.url}\n"
                f"Title: {node.page_title}\n"
                f"Page content summary:\n{node.pam_content[:800]}\n\n"
                f"Return JSON with field 'test_cases': list of objects each having "
                f"title, priority (high/medium/low), preconditions (list), "
                f"steps (list of strings, min 1), expected_outcome."
            )
            try:
                response: _FuncResponse = await self._client.create_structured(
                    prompt, _FuncResponse, temperature=0.1
                )
                for item in response.test_cases:
                    results.append(TestCase(
                        title=item.title,
                        type="functional",
                        priority=item.priority,
                        preconditions=item.preconditions,
                        steps=item.steps,
                        expected_outcome=item.expected_outcome,
                        source_url=node.url,
                    ))
            except (StructuredGenerationError, Exception) as exc:
                logger.warning(f"UniversalTestPlanner: LLM failed for {node.url}: {exc!r}")
                results.append(TestCase(
                    title=f"Verify {node.page_title} loads",
                    type="functional",
                    priority="medium",
                    steps=[f"Navigate to {node.url}", "Verify page title is present"],
                    expected_outcome="Page loads without error",
                    source_url=node.url,
                ))
        return results

    def _plan_accessibility(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        return [
            TestCase(
                title=f"Accessibility: {node.page_title}",
                type="accessibility",
                priority="medium",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    "Check all interactive elements have accessible names",
                    "Check all images have alt text",
                    "Check no decorative roles on interactive elements",
                ],
                expected_outcome="No WCAG violations found",
                source_url=node.url,
            )
            for node in nodes
        ]

    def _plan_security(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        for node in nodes:
            if "form" not in node.coverage_tags:
                continue
            results.append(TestCase(
                title=f"XSS injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    f"Fill all text inputs with XSS payload: {_XSS_PAYLOAD}",
                    "Submit the form",
                    "Verify payload is not executed",
                ],
                expected_outcome="Page does not execute the script payload",
                source_url=node.url,
            ))
            results.append(TestCase(
                title=f"SQL injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    f"Fill all text inputs with SQLi payload: {_SQLI_PAYLOAD}",
                    "Submit the form",
                    "Verify no SQL error is exposed",
                ],
                expected_outcome="Page does not expose SQL errors or unintended data",
                source_url=node.url,
            ))
        return results

    @staticmethod
    def _sort_by_priority(cases: list[TestCase]) -> list[TestCase]:
        return sorted(cases, key=lambda tc: _PRIORITY_ORDER.get(tc.priority, 1))


__all__ = ["UniversalTestPlanner"]
