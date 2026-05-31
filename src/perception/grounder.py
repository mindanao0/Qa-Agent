# src/perception/grounder.py
"""
Grounder — Layer 4 / Entry point for the Universal DOM Compression Pipeline.

Orchestrates AOMExtractor → DOMPruner → SemanticCompactor with:
- Graceful fallback when AOM is sparse or fails
- Budget-aware re-compaction (25 → 15 → 10 items)
- JSONL audit metric emission
- Safety net: NEVER raises — returns emergency PAM on complete failure (R2.1)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock
from loguru import logger
from playwright.async_api import Page

from src.perception.aom_extractor import (
    AOMExtractor,
    AOMSnapshot,
    AOMSparseError,
    is_aom_sparse,
)
from src.perception.dom_pruner import DOMPruner, PrunedDOMSnapshot
from src.perception.semantic_compactor import CompactPAM, SemanticCompactor

# Audit output path (relative to project root → three levels up from this file)
_METRICS_PATH = (
    Path(__file__).parent.parent.parent / "audit" / "phase0" / "grounder_metrics.jsonl"
)


# ── Metric emission ────────────────────────────────────────────────────────────


def emit_grounder_metric(
    url: str,
    source: str,
    tokens: int,
    latency_ms: int,
    budget_overflow: bool,
) -> None:
    """Write one JSONL line to audit/phase0/grounder_metrics.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "source": source,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "budget_overflow": budget_overflow,
    }
    lock_path = _METRICS_PATH.with_suffix(".lock")
    try:
        with FileLock(str(lock_path), timeout=2):
            _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_METRICS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"emit_grounder_metric: failed to write metrics: {exc}")


# ── Grounder ──────────────────────────────────────────────────────────────────


class Grounder:
    """
    Entry point for the Universal DOM Compression Pipeline.

    Orchestrates extraction, pruning, and compaction with graceful degradation.

    NEVER raises — always returns a CompactPAM (possibly source='failure').
    When all perception layers fail, returns an emergency PAM with URL + title
    so the LLM at least knows where it is.

    Usage::

        grounder = Grounder()
        pam = await grounder.ground(page, context_budget_tokens=1000)
    """

    async def ground(
        self,
        page: Page,
        context_budget_tokens: int = 1000,
    ) -> CompactPAM:
        """
        Extract a CompactPAM from *page*, respecting *context_budget_tokens*.

        Never raises. On complete perception failure, returns an emergency PAM
        with source='failure' containing the page URL and title.

        Extraction order:
        1. AOMExtractor (best-effort, never raises from this method)
        2. DOMPruner if AOM failed/sparse (best-effort, never raises)
        3. SemanticCompactor at max_items=25→15→10 if over budget
        4. Emergency PAM if all layers fail (source='failure')
        """
        start_time = time.monotonic()
        url: str = page.url
        errors: list[str] = []

        # ── Layer 1: AOM extraction (best-effort) ─────────────────────────────
        aom: AOMSnapshot | None = None
        try:
            aom_result = await AOMExtractor().extract(page)
            if is_aom_sparse(aom_result):
                logger.warning(
                    f"Grounder: AOM is sparse ({aom_result.node_count} nodes) for {url}, "
                    "discarding AOM"
                )
                errors.append("aom_sparse")
                aom = None
            else:
                aom = aom_result
        except AOMSparseError as exc:
            logger.warning(f"Grounder: AOMSparseError for {url}: {exc}")
            errors.append(f"aom_sparse:{exc.node_count}")
            aom = None
        except Exception as exc:
            logger.warning(f"Grounder: AOMExtractor failed for {url}: {exc}")
            errors.append(f"aom_failed:{type(exc).__name__}:{str(exc)[:80]}")
            aom = None

        # ── Layer 2: DOM pruning (best-effort, only when AOM unavailable) ─────
        dom: PrunedDOMSnapshot | None = None
        if aom is None:
            try:
                dom = await DOMPruner().prune(page)
            except Exception as exc:
                logger.warning(f"Grounder: DOMPruner failed for {url}: {exc}")
                errors.append(f"dom_failed:{type(exc).__name__}:{str(exc)[:80]}")
                dom = None

        # ── Layer 3: Complete failure — return emergency PAM, do not raise ─────
        if aom is None and dom is None:
            errors.append("all_layers_failed")
            logger.error(
                f"Grounder: all perception layers failed for {url} — returning emergency PAM. "
                f"errors={errors}"
            )
            emergency_pam = self._emergency_pam(page, errors)
            latency_ms = int((time.monotonic() - start_time) * 1000)
            emit_grounder_metric(
                url=url,
                source="failure",
                tokens=emergency_pam.estimated_tokens,
                latency_ms=latency_ms,
                budget_overflow=False,
            )
            return emergency_pam

        # ── Layer 4: Compaction with budget-aware retry ───────────────────────
        compactor = SemanticCompactor()
        best_pam: CompactPAM | None = None

        for max_items in (25, 15, 10):
            try:
                candidate = compactor.compact(aom, dom, max_items=max_items)
                # Keep the best (lowest-token) result
                if best_pam is None or candidate.estimated_tokens < best_pam.estimated_tokens:
                    best_pam = candidate
                if candidate.estimated_tokens <= context_budget_tokens:
                    break
                if max_items == 10:
                    logger.warning(
                        f"Grounder: budget overflow at max_items=10 "
                        f"(estimated_tokens={candidate.estimated_tokens} > budget={context_budget_tokens}), "
                        "returning best available result"
                    )
            except Exception as exc:
                logger.warning(f"Grounder: compaction failed at max_items={max_items}: {exc}")
                errors.append(f"compaction_failed:{type(exc).__name__}")
                continue

        # All compaction attempts failed — return emergency PAM
        if best_pam is None:
            errors.append("all_compaction_failed")
            logger.error(f"Grounder: all compaction attempts failed for {url}")
            emergency_pam = self._emergency_pam(page, errors)
            latency_ms = int((time.monotonic() - start_time) * 1000)
            emit_grounder_metric(
                url=url,
                source="failure",
                tokens=emergency_pam.estimated_tokens,
                latency_ms=latency_ms,
                budget_overflow=False,
            )
            return emergency_pam

        pam = best_pam

        # ── Metrics ───────────────────────────────────────────────────────────
        latency_ms = int((time.monotonic() - start_time) * 1000)
        emit_grounder_metric(
            url=url,
            source=pam.source,
            tokens=pam.estimated_tokens,
            latency_ms=latency_ms,
            budget_overflow=pam.estimated_tokens > context_budget_tokens,
        )

        logger.info(
            f"Grounder: source={pam.source} tokens={pam.estimated_tokens} "
            f"latency={latency_ms}ms url={url}"
        )

        return pam

    def _emergency_pam(self, page: Page, errors: list[str]) -> CompactPAM:
        """
        Last-resort minimum-viable PAM when all perception layers fail.
        Surfaces page title + URL only. Better than nothing — the LLM at
        least knows where it is and can attempt a best-effort plan.
        """
        from src.perception.token_counter import estimate_tokens

        try:
            title = page.title()
        except Exception:
            title = "(unknown)"

        content = (
            f"## Page State (perception unavailable)\n"
            f"- URL: {page.url}\n"
            f"- title: {title}\n"
            f"- errors: {', '.join(errors)}\n"
            f"- Note: full DOM extraction unavailable, plan with care\n"
        )

        return CompactPAM(
            format="md",
            content=content,
            controls_count=0,
            forms_count=0,
            lists_count=0,
            estimated_tokens=estimate_tokens(content),
            source="failure",
            dropped_nodes=0,
        )
