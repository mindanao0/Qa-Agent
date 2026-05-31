from __future__ import annotations

import hashlib
from typing import Any, TYPE_CHECKING

from loguru import logger

from src.config_loader import get_structured_output_engine, get_use_grounder
from src.data.synthetic_gen import detect_domain
from src.llm.adapter import OllamaAdapter
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.llm.prompt_templates import PlannerPromptTemplate, get_domain_hint
from src.llm.structured import TestPlan, enforce_json_output
from src.rag.retriever import HybridRetriever

if TYPE_CHECKING:
    from src.memory.episodic_store import EpisodicStore
    from src.contractskill.compiler import ContractSkillStore

_MAX_RETRIES = 3  # legacy repair loop only; instructor max_retries set per-instance in __init__
_AUTO_DETECT_SENTINEL = "general"


class PlannerAgent:
    """
    Converts a natural-language requirement into a structured TestPlan.

    UPDATE 3 — RAG-enhanced planning:
      Before calling the LLM, the planner pulls three layers of context from
      the hybrid retriever (when available) and merges them into the prompt:

        1. Page structure from prior crawls (via get_page_context(url)).
        2. Relevant Playwright API docs (search filtered to playwright_docs).
        3. Similar test examples / docs filtered by the detected domain.

      The planner still auto-detects the domain via detect_domain() and
      injects it into both the prompt and the returned TestPlan.
    """

    def __init__(
        self,
        adapter: OllamaAdapter,
        retriever: HybridRetriever | None = None,
        instructor_client: InstructorClient | None = None,
        episodic_store: "EpisodicStore | None" = None,
        contract_skill_store: "ContractSkillStore | None" = None,
    ) -> None:
        self.adapter = adapter
        self.retriever = retriever
        self.episodic_store = episodic_store
        self.contract_skill_store = contract_skill_store
        # instructor_planner: max_retries=3 (explicit — planning benefits from correction loops)
        self._instructor_client = instructor_client or InstructorClient(
            base_url=adapter.base_url,
            model=adapter.model,
            max_retries=3,
        )

    async def plan(
        self,
        requirement: str,
        url: str,
        role: str = "admin",
        domain: str = _AUTO_DETECT_SENTINEL,
        page_state: str = "",
    ) -> TestPlan:
        # ── Step 1: Auto-detect domain when not explicitly provided ──────────
        if not domain or domain == _AUTO_DETECT_SENTINEL:
            domain = await detect_domain(
                requirement=requirement,
                url=url,
                adapter=self.adapter,
            )
        logger.info(f"PlannerAgent: detected domain={domain}")

        # ── Step 2: Pull layered RAG context (UPDATE 3) ──────────────────────
        rag_context = await self._build_rag_context(
            requirement=requirement,
            url=url,
            domain=domain,
        )

        # ── Step 2a: Inject memory-based locator policy (S2-D) ──────────────
        memory_block = ""
        if self.episodic_store is not None:
            try:
                # Use a generic failure_signature derived from URL since
                # planner has no specific failure yet.
                from urllib.parse import urlparse
                approximate_sig = hashlib.sha256(
                    urlparse(url).path.encode()
                ).hexdigest()[:16]
                policy = await self.episodic_store.evolve_locator_policy(
                    domain, approximate_sig
                )
                if policy.preferred_strategies or policy.avoid_strategies:
                    lines = ["## Memory-Based Strategy"]
                    if policy.preferred_strategies:
                        lines.append(
                            f"Preferred (proven): {policy.preferred_strategies[:3]}"
                        )
                    if policy.avoid_strategies:
                        lines.append(
                            f"Avoid (known-fail): {policy.avoid_strategies[:3]}"
                        )
                    memory_block = "\n".join(lines)
            except Exception as exc:
                logger.warning(f"PlannerAgent: memory injection failed: {exc}")

        if memory_block:
            rag_context = (rag_context or "") + "\n\n" + memory_block

        # ── Step 2b: Inject Grounder page state when enabled (Cluster E) ────
        # TODO: PlannerPromptTemplate.user does not yet have a {page_state} placeholder.
        # Until the template is updated, page_state is prepended to rag_context so the
        # LLM still sees live page context.  When the template gains {page_state}, pass
        # it as a separate kwarg to PlannerPromptTemplate.to_messages() instead.
        if page_state and get_use_grounder():
            page_state_block = f"## Live Page State (Grounder)\n{page_state}"
            rag_context = (
                page_state_block + "\n\n" + (rag_context or "(no RAG context available)")
            )

        # ── Step 2c: Inject ContractSkill when available (S4-D) ─────────────
        if self.contract_skill_store is not None:
            try:
                skill = await self.contract_skill_store.find_matching_skill(
                    goal=requirement, url=url, domain=domain
                )
                if skill is not None:
                    contract_context = (
                        f"\n## ContractSkill Available\n"
                        f"Proven trajectory for this goal (success_count={skill.success_count}):\n"
                        f"Steps: {[s.action_type + ' ' + s.locator for s in skill.steps[:5]]}\n"
                        f"Postconditions: {skill.postconditions}\n"
                        f"Use this trajectory as the preferred plan.\n"
                    )
                    rag_context = (rag_context or "") + contract_context
                    logger.info(
                        f"PlannerAgent: ContractSkill injected | "
                        f"skill_id={skill.skill_id[:12]} success_count={skill.success_count}"
                    )
            except Exception as exc:
                logger.warning(f"PlannerAgent: ContractSkill injection failed: {exc}")

        # ── Step 3: Render the universal planner prompt ──────────────────────
        domain_hint = get_domain_hint(domain)
        messages = PlannerPromptTemplate.to_messages(
            requirement=requirement,
            url=url,
            role=role,
            domain=domain,
            domain_hint=domain_hint,
            rag_context=rag_context or "(no RAG context available)",
        )

        # ── Step 4: Call the LLM — instructor or legacy path ─────────────────
        engine = get_structured_output_engine()
        last_raw: str = ""
        last_error: Exception | None = None

        if engine == "instructor":
            try:
                plan = await self._instructor_client.create_structured(
                    messages, TestPlan, temperature=0.0
                )
                if not plan.domain or plan.domain == "crud_operations":
                    plan = plan.model_copy(update={"domain": domain})
                elif plan.domain != domain:
                    logger.debug(
                        f"PlannerAgent: LLM returned domain={plan.domain!r}, "
                        f"overriding to detected={domain!r}"
                    )
                    plan = plan.model_copy(update={"domain": domain})
                logger.info(
                    f"PlannerAgent: plan ready (instructor) | title={plan.title!r} "
                    f"domain={plan.domain!r} steps={len(plan.steps)} "
                    f"complexity={plan.estimated_complexity!r}"
                )
                return plan
            except StructuredGenerationError as exc:
                logger.error(f"PlannerAgent: instructor path failed: {exc}; raising")
                raise RuntimeError(
                    f"PlannerAgent: instructor failed after retries — {exc}"
                ) from exc
        else:
            # Legacy 3-stage repair path
            for attempt in range(_MAX_RETRIES):
                try:
                    last_raw = await self.adapter.generate(messages)
                    plan = enforce_json_output(TestPlan, last_raw)
                    if not plan.domain or plan.domain == "crud_operations":
                        plan = plan.model_copy(update={"domain": domain})
                    elif plan.domain != domain:
                        logger.debug(
                            f"PlannerAgent: LLM returned domain={plan.domain!r}, "
                            f"overriding to detected={domain!r}"
                        )
                        plan = plan.model_copy(update={"domain": domain})
                    logger.info(
                        f"PlannerAgent: plan ready (legacy) | title={plan.title!r} "
                        f"domain={plan.domain!r} steps={len(plan.steps)} "
                        f"complexity={plan.estimated_complexity!r}"
                    )
                    return plan
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        f"PlannerAgent attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}"
                    )
                    if attempt < _MAX_RETRIES - 1:
                        messages.append({"role": "assistant", "content": last_raw})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your response could not be parsed as a TestPlan: {exc}\n"
                                "Output ONLY valid JSON matching the schema. "
                                "Do not include any text outside the JSON object."
                            ),
                        })

            raise RuntimeError(
                f"PlannerAgent: failed after {_MAX_RETRIES} attempts. "
                f"Last error: {last_error}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # RAG context assembly (UPDATE 3)
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_rag_context(
        self,
        requirement: str,
        url: str,
        domain: str,
    ) -> str:
        """
        Compose three RAG sections — page context, Playwright API context,
        and domain examples — into a single labelled block.  Each section is
        skipped (with a [unavailable] marker) when retrieval fails or returns
        nothing.
        """
        if not self.retriever:
            return ""

        # Pull each layer separately so we can report chunk counts.
        page_chunks = await self._safe_search(
            self.retriever.search(
                query=f"page structure for {url}",
                sources=["url_crawl"],
                top_k=3,
            ),
            "url_crawl",
        )
        api_chunks = await self._safe_search(
            self.retriever.search(
                query=requirement,
                sources=["playwright_docs"],
                top_k=3,
            ),
            "playwright_docs",
        )
        domain_chunks = await self._safe_search(
            self.retriever.search(
                query=f"{domain} {requirement}",
                domain=domain,
                top_k=3,
            ),
            "domain_examples",
        )

        # Spec-mandated log format: "RAG context: X chunks retrieved from …"
        logger.info(
            f"RAG context: {len(page_chunks)} chunks retrieved from url_crawl"
        )
        logger.info(
            f"RAG context: {len(api_chunks)} chunks retrieved from playwright_docs"
        )
        logger.info(
            f"RAG context: {len(domain_chunks)} chunks retrieved from "
            f"domain_examples (domain={domain})"
        )

        page_context = self.retriever.format_context(page_chunks) if page_chunks else ""
        api_context = self.retriever.format_context(api_chunks) if api_chunks else ""
        domain_examples = (
            self.retriever.format_context(domain_chunks) if domain_chunks else ""
        )

        parts: list[str] = []
        parts.append("## Page structure from crawl")
        parts.append(page_context or "(no crawled context for this URL)")
        parts.append("\n## Relevant Playwright APIs")
        parts.append(api_context or "(no Playwright docs ingested yet)")
        parts.append("\n## Similar test examples / domain notes")
        parts.append(domain_examples or "(no domain examples found)")
        return "\n".join(parts)

    # ── small async helpers ──────────────────────────────────────────────────

    @staticmethod
    async def _safe_search(coro: Any, label: str) -> list:
        """Await a retriever.search() coroutine; return [] on exception."""
        try:
            return await coro
        except Exception as exc:
            logger.warning(f"PlannerAgent RAG {label}: retrieval failed ({exc})")
            return []
