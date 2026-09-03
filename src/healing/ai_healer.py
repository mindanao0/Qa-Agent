# src/healing/ai_healer.py
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Literal
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.browser.ax_extractor import extract_axtree, prune_axtree
from src.config_loader import get_structured_output_engine
from src.llm.adapter import OllamaAdapter
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.llm.prompt_templates import HealerPromptTemplate
from src.llm.structured import HealedLocator, enforce_json_output


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 / Cluster S2-D — class-based AIHealer schemas
# ─────────────────────────────────────────────────────────────────────────────


class ActionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str           # click, fill, select, etc.
    domain: str
    page_url: str
    element_description: str   # human-readable


class HealResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["FUZZY_HIT","AOM_HIT","MEMORY_HIT","QUARANTINE"]
    locator: str | None
    attempt_count: int
    latency_ms: int
    failure_signature: str

_MAX_AXTREE_NODES = 200
_MAX_KNOWN_LOCATORS = 20


async def ai_heal(
    page: Page,
    failed_locator: str,
    action: str,
    error_message: str,
    axtree: str | None = None,
    known_locators: dict[str, Any] | None = None,
    adapter: OllamaAdapter | None = None,
) -> HealedLocator:
    """
    Phase 2 self-healing: use the LLM to recover a broken locator via AxTree
    context.

    Args:
        page:            Current Playwright page (used to re-capture AxTree if
                         *axtree* is None).
        failed_locator:  The Playwright locator string that raised an error.
        action:          The action attempted (e.g. "click", "fill", "check").
        error_message:   The exception message from the failed action.
        axtree:          Pre-captured pruned AxTree; if None it is captured here.
        known_locators:  Dict of previously healed locators for this URL.
        adapter:         OllamaAdapter instance; a new one is created if None.

    Returns:
        HealedLocator with method="ai".  Confidence reflects the LLM's
        self-reported certainty; callers should escalate to VLM if < 0.5.
    """
    _owns_adapter = adapter is None
    if adapter is None:
        adapter = OllamaAdapter()

    instructor_client: InstructorClient | None = None

    try:
        if not axtree:
            raw = await extract_axtree(page)
            axtree = await prune_axtree(raw, max_nodes=_MAX_AXTREE_NODES)

        page_url = page.url

        trimmed_locators: dict[str, Any] = {}
        if known_locators:
            items = list(known_locators.items())[:_MAX_KNOWN_LOCATORS]
            trimmed_locators = dict(items)

        messages = HealerPromptTemplate.to_messages(
            failed_locator=failed_locator,
            action=action,
            error_message=error_message,
            page_url=page_url,
            axtree=axtree,
            known_locators=json.dumps(trimmed_locators, indent=2) if trimmed_locators else "{}",
            max_nodes=_MAX_AXTREE_NODES,
        )

        logger.info(
            f"ai_heal: calling LLM | locator={failed_locator!r} "
            f"action={action!r} url={page_url!r}"
        )

        engine = get_structured_output_engine()

        if engine == "instructor":
            # instructor_healer: max_retries=3 (healing accuracy matters more than latency)
            instructor_client = InstructorClient(
                base_url=adapter.base_url,
                model=adapter.model,
                max_retries=3,
            )
            try:
                healed = await instructor_client.create_structured(
                    messages, HealedLocator, temperature=0.0
                )
                # Enforce method="ai" via model_copy (immutable-safe pattern)
                healed = healed.model_copy(update={"method": "ai"})
            except StructuredGenerationError as exc:
                logger.warning(
                    f"ai_heal: instructor path failed ({exc}); falling back to legacy"
                )
                raw_response = await adapter.generate(messages)
                healed = enforce_json_output(HealedLocator, raw_response)
                healed = healed.model_copy(update={"method": "ai"})
        else:
            raw_response = await adapter.generate(messages)
            healed = enforce_json_output(HealedLocator, raw_response)
            # Enforce method tag — use model_copy for immutability safety
            healed = healed.model_copy(update={"method": "ai"})

        logger.info(
            f"ai_heal: result | healed={healed.healed!r} "
            f"confidence={healed.confidence:.4f}"
        )
        return healed

    except Exception as exc:
        logger.error(f"ai_heal: failed — {exc}")
        return HealedLocator(
            reasoning=f"AI healing failed with error: {exc}",
            original=failed_locator,
            healed=failed_locator,
            confidence=0.0,
            method="ai",
        )
    finally:
        if instructor_client is not None:
            await instructor_client.close()
        if _owns_adapter:
            await adapter.close()


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 / Cluster S2-D — AIHealer (3-attempt pipeline)
# ─────────────────────────────────────────────────────────────────────────────

# Heavy imports kept inside the class file but lazy where possible so the
# legacy ai_heal() path above stays untouched.
from src.healing.fuzzy_matcher import FuzzyMatcher  # noqa: E402
from src.memory.episodic_store import (  # noqa: E402
    EpisodicStore,
    HealedExperience,
    PostMortem,
)
from src.perception.aom_extractor import AOMExtractor, AOMNode, AOMSnapshot  # noqa: E402


_LOCATOR_PARSE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "role",
        re.compile(
            r'get_by_role\s*\(\s*["\'](?P<role>[^"\']+)["\']'
            r'(?:.*?name\s*=\s*["\'](?P<name>[^"\']+)["\'])?',
            re.DOTALL,
        ),
    ),
    ("label", re.compile(r'get_by_label\s*\(\s*["\'](?P<name>[^"\']+)["\']')),
    ("text", re.compile(r'get_by_text\s*\(\s*["\'](?P<name>[^"\']+)["\']')),
    ("placeholder", re.compile(r'get_by_placeholder\s*\(\s*["\'](?P<name>[^"\']+)["\']')),
    ("test_id", re.compile(r'get_by_test_id\s*\(\s*["\'](?P<name>[^"\']+)["\']')),
]


def _parse_failed_locator(locator_str: str) -> tuple[str, str]:
    """Extract (role, text) from a Playwright DSL locator string.

    Returns ("", "") if nothing recognisable found.  For non-role strategies
    the role is left empty and the text is the captured value.
    """
    for kind, pattern in _LOCATOR_PARSE_PATTERNS:
        m = pattern.search(locator_str)
        if not m:
            continue
        gd = m.groupdict()
        if kind == "role":
            return (gd.get("role") or "", gd.get("name") or "")
        return ("", gd.get("name") or "")
    return ("", "")


def _eval_locator(page: Page, locator_str: str):
    """Convert a Playwright DSL string to a real Locator (or None).

    Mirrors HealingEngine._eval_locator so the AIHealer never needs to import
    the legacy engine module.
    """
    m = re.search(
        r'get_by_role\s*\(\s*["\'](\w+)["\'](?:.*?name\s*=\s*["\']([^"\']+)["\'])?',
        locator_str,
        re.DOTALL,
    )
    if m:
        role, name = m.group(1), m.group(2)
        return page.get_by_role(role, name=name) if name else page.get_by_role(role)  # type: ignore[arg-type]

    m = re.search(r'get_by_label\s*\(\s*["\']([^"\']+)["\']', locator_str)
    if m:
        return page.get_by_label(m.group(1))

    m = re.search(r'get_by_text\s*\(\s*["\']([^"\']+)["\']', locator_str)
    if m:
        return page.get_by_text(m.group(1))

    m = re.search(r'get_by_test_id\s*\(\s*["\']([^"\']+)["\']', locator_str)
    if m:
        return page.get_by_test_id(m.group(1))

    m = re.search(r'get_by_placeholder\s*\(\s*["\']([^"\']+)["\']', locator_str)
    if m:
        return page.get_by_placeholder(m.group(1))

    return None


def _flatten_aom(node: AOMNode) -> list[AOMNode]:
    out = [node]
    for c in node.children:
        out.extend(_flatten_aom(c))
    return out


def _name_overlap(a: str, b: str) -> float:
    """Jaccard-style word overlap normalised by min word-count."""
    wa = {w for w in re.findall(r"\w+", (a or "").lower()) if w}
    wb = {w for w in re.findall(r"\w+", (b or "").lower()) if w}
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    return len(inter) / max(1, min(len(wa), len(wb)))


class AIHealer:
    """Sprint 2 self-healer.

    Three-attempt pipeline:
      1) Fuzzy match against the locator repo (FuzzyMatcher).
      2) AOM re-discovery (current page accessibility tree).
      3) Episodic memory replay (EpisodicStore).
    If all fail → save a PostMortem and return ``QUARANTINE``.

    The class is non-raising: every external call is wrapped in try/except so
    a single broken layer cannot abort the whole heal.
    """

    def __init__(
        self,
        fuzzy_matcher: FuzzyMatcher,
        aom_extractor: AOMExtractor,
        episodic_store: EpisodicStore,
        locator_synthesizer: Any = None,  # kept for future use
        session_id: str = "",
    ) -> None:
        self.fuzzy_matcher = fuzzy_matcher
        self.aom_extractor = aom_extractor
        self.episodic_store = episodic_store
        self.locator_synthesizer = locator_synthesizer
        self.session_id = session_id

    # ── Public API ────────────────────────────────────────────────────────

    async def heal(
        self,
        page: Page,
        failed_locator: str,
        action_context: ActionContext,
    ) -> HealResult:
        t0 = monotonic()
        failure_sig = self._compute_failure_signature(
            failed_locator, action_context.page_url, action_context.action_type
        )
        attempt_count = 0

        # ── Attempt 1: Fuzzy ──────────────────────────────────────────────
        attempt_count = 1
        try:
            candidates = self.fuzzy_matcher.find_candidates(
                failed_locator, threshold=0.85
            )
        except Exception as exc:
            logger.warning(f"AIHealer: fuzzy lookup failed: {exc}")
            candidates = []

        for c in candidates[:3]:
            if c.confidence_tier != "HIGH":
                continue
            if await self._try_locator(page, c.candidate_locator):
                await self._save_heal(
                    failure_sig,
                    failed_locator,
                    c.candidate_locator,
                    action_context,
                    impact=0.5,
                    root_cause="fuzzy_match",
                )
                return HealResult(
                    strategy="FUZZY_HIT",
                    locator=c.candidate_locator,
                    attempt_count=attempt_count,
                    latency_ms=int((monotonic() - t0) * 1000),
                    failure_signature=failure_sig,
                )

        # ── Attempt 2: AOM re-discovery ───────────────────────────────────
        attempt_count = 2
        try:
            aom = await self.aom_extractor.extract(page)
            candidate_node = self._match_aom_node(aom, failed_locator)
            if candidate_node is not None:
                new_locator = self._synth_locator_from_aom_node(candidate_node)
                if new_locator and await self._try_locator(page, new_locator):
                    await self._save_heal(
                        failure_sig,
                        failed_locator,
                        new_locator,
                        action_context,
                        impact=0.7,
                        root_cause="aom_rediscovery",
                    )
                    return HealResult(
                        strategy="AOM_HIT",
                        locator=new_locator,
                        attempt_count=attempt_count,
                        latency_ms=int((monotonic() - t0) * 1000),
                        failure_signature=failure_sig,
                    )
        except Exception as exc:
            logger.warning(f"AIHealer: AOM attempt failed: {exc}")

        # ── Attempt 3: Memory lookup ──────────────────────────────────────
        attempt_count = 3
        try:
            memories = await self.episodic_store.retrieve_for_planning(
                query_text=f"{failed_locator} {action_context.action_type}",
                domain=action_context.domain,
                failure_signature=failure_sig,
                limit=3,
            )
            for m in memories:
                if m.confidence < 0.70:
                    continue
                if m.memory_type != "healed_experience":
                    continue
                if await self._try_locator(page, m.strategy):
                    await self._save_heal(
                        failure_sig,
                        failed_locator,
                        m.strategy,
                        action_context,
                        impact=0.9,
                        root_cause="memory_replay",
                    )
                    return HealResult(
                        strategy="MEMORY_HIT",
                        locator=m.strategy,
                        attempt_count=attempt_count,
                        latency_ms=int((monotonic() - t0) * 1000),
                        failure_signature=failure_sig,
                    )
        except Exception as exc:
            logger.warning(f"AIHealer: memory attempt failed: {exc}")

        # ── All failed → quarantine + post_mortem ─────────────────────────
        try:
            await self._save_post_mortem(
                failure_sig, failed_locator, action_context, attempt_count
            )
        except Exception as exc:
            logger.error(f"AIHealer: failed to save post-mortem: {exc}")

        return HealResult(
            strategy="QUARANTINE",
            locator=None,
            attempt_count=attempt_count,
            latency_ms=int((monotonic() - t0) * 1000),
            failure_signature=failure_sig,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_failure_signature(
        failed_locator: str, url: str, action_type: str
    ) -> str:
        try:
            path = urlparse(url).path or ""
        except Exception:
            path = ""
        raw = f"{action_type}|{path}|{failed_locator}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    async def _try_locator(page: Page, locator_str: str) -> bool:
        """Best-effort visibility probe — returns False on any error."""
        try:
            element = _eval_locator(page, locator_str)
            if element is None:
                return False
            first = element.first
            await first.wait_for(state="visible", timeout=2_000)
            return True
        except Exception as exc:
            logger.debug(f"AIHealer._try_locator: {locator_str!r} not visible: {exc}")
            return False

    @staticmethod
    def _match_aom_node(
        aom_snapshot: AOMSnapshot, failed_locator: str
    ) -> AOMNode | None:
        parsed_role, parsed_text = _parse_failed_locator(failed_locator)
        if not parsed_role and not parsed_text:
            return None
        nodes = _flatten_aom(aom_snapshot.root_node)
        for n in nodes:
            if parsed_role and n.role.lower() != parsed_role.lower():
                continue
            if not parsed_text:
                # role-only locator: take the first role match
                return n
            if _name_overlap(n.name, parsed_text) >= 0.5:
                return n
        return None

    @staticmethod
    def _synth_locator_from_aom_node(node: AOMNode) -> str:
        role = (node.role or "").strip()
        name = (node.name or "").strip().replace('"', '\\"')
        if not role:
            return ""
        if name:
            return f'page.get_by_role("{role}", name="{name}")'
        return f'page.get_by_role("{role}")'

    async def _save_heal(
        self,
        failure_sig: str,
        failed_locator: str,
        winning_strategy: str,
        action_context: ActionContext,
        impact: float,
        root_cause: str,
    ) -> None:
        try:
            text = f"{failure_sig} {winning_strategy}"
            vector = await self._embed(text)
            exp = HealedExperience(
                memory_id=uuid.uuid4().hex,
                session_id=self.session_id,
                domain=action_context.domain,
                page_url=action_context.page_url,
                page_type="",
                failure_signature=failure_sig,
                bad_strategy=failed_locator,
                winning_strategy=winning_strategy,
                root_cause=root_cause[:100],
                confidence=0.7,
                impact_score=float(impact),
                created_at_iso=datetime.now(timezone.utc).isoformat(),
                vector=vector,
            )
            await self.episodic_store.save_healed_experience(exp)
        except Exception as exc:
            logger.warning(f"AIHealer._save_heal failed: {exc}")

    async def _save_post_mortem(
        self,
        failure_sig: str,
        failed_locator: str,
        action_context: ActionContext,
        attempt_count: int,
    ) -> None:
        error_message = "exhausted_3_attempts"
        text = f"{failure_sig} {error_message}"
        vector = await self._embed(text)
        pm = PostMortem(
            memory_id=uuid.uuid4().hex,
            session_id=self.session_id,
            attempt_id=int(attempt_count),
            domain=action_context.domain,
            page_url=action_context.page_url,
            failure_signature=failure_sig,
            current_plan=action_context.element_description,
            locator_candidates=[failed_locator],
            error_message=error_message,
            dom_snapshot_ref="",
            confidence=0.0,
            impact_score=0.5,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            vector=vector,
        )
        await self.episodic_store.save_post_mortem(pm)

    async def _embed(self, text: str) -> list[float]:
        """Use the store's embedding function; gracefully degrade to []."""
        try:
            # Prefer the public-ish _embed (validates dim) when configured
            return await self.episodic_store._embed(text)
        except Exception as exc:
            logger.debug(f"AIHealer._embed: falling back ({exc})")
            fn = getattr(self.episodic_store, "_embedding_fn", None) or getattr(
                self.episodic_store, "embedding_fn", None
            )
            if fn is None:
                return []
            try:
                vec = await fn(text)
                return [float(v) for v in vec]
            except Exception as exc2:
                logger.warning(f"AIHealer._embed: embedding_fn failed: {exc2}")
                return []
