"""Deterministic Observer→Driver feedback distillation.

Pure Python — NO random, NO clock/time, NO LLM. Given the :class:`ObserverReport`
items produced by a *single* Driver step, collapse them into at most three advisory
directives the Driver can read on its *next* planning step. Same inputs always give
the same output, so the hint that rides the checkpointed :class:`AgentState` between
steps is reproducible and testable offline.

Two facts about the source data drive the policy (both verified against the concrete
observers in ``src/agents/observer_driver/observers/``):

* ``Severity`` is a ``Literal[str]`` with **no** natural order → ranked explicitly via
  :data:`_SEVERITY_RANK`.
* Severity is **report-level** (one per :class:`ObserverReport`, but ``findings`` is a
  list) → every finding inherits the severity of its containing report.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.agents.observer_driver.state import (
    DriverFeedback,
    ObserverReport,
    Severity,
)

# ``Severity`` is a Literal[str] with NO comparison order — rank it here.
_SEVERITY_RANK: dict[str, int] = {"info": 0, "warn": 1, "critical": 2}

# Minimum rank a finding must reach to be surfaced. "info" (rank 0) is dropped,
# which conveniently discards the observers' filler "no ... detected" lines.
_MIN_RANK: int = _SEVERITY_RANK["warn"]

# Never surface more than this many directives in a single hint.
MAX_DIRECTIVES: int = 3


def finding_hash(observer_name: str, finding: str) -> str:
    """Stable 12-hex-char id for one ``(observer, finding)`` pair.

    Deterministic (SHA-1 of ``"<observer_name>:<finding>"``); the same finding from
    the same observer always hashes identically, which is what lets
    :attr:`AgentState.acted_finding_hashes` suppress re-nagging across steps.
    """
    return hashlib.sha1(f"{observer_name}:{finding}".encode()).hexdigest()[:12]


@dataclass(frozen=True)
class _Candidate:
    """One finding lifted to candidate status, carrying its report's severity."""

    finding: str
    severity: Severity
    hash_id: str


def distill_feedback(
    reports: list[ObserverReport],
    step: int,
    already_acted: frozenset[str],
) -> DriverFeedback | None:
    """Collapse one step's ObserverReports into a :class:`DriverFeedback` (or None).

    Policy (fully deterministic):

    1. Expand every report into per-finding candidates; each candidate inherits the
       **report-level** severity of its containing report.
    2. Drop candidates below ``"warn"`` (rank < :data:`_MIN_RANK`) — this discards
       ``"info"`` filler such as ``"no security issues detected"``.
    3. Drop candidates whose ``hash_id`` is in *already_acted* (surfaced on an earlier
       step) and de-duplicate within this call by ``hash_id`` (first occurrence wins).
    4. If nothing new survives → return ``None``.
    5. Stable-sort survivors by severity rank **descending** (encounter order is
       preserved within an equal-severity tier).
    6. Surface at most :data:`MAX_DIRECTIVES`; ``top_severity`` is the highest severity
       among the surfaced candidates, and ``finding_hashes`` are exactly the surfaced
       ones (so only surfaced findings get recorded — the rest can surface later).
    """
    # 1. Expand reports → per-finding candidates (report-level severity).
    candidates: list[_Candidate] = []
    for report in reports:
        for finding in report.findings:
            candidates.append(
                _Candidate(
                    finding=finding,
                    severity=report.severity,
                    hash_id=finding_hash(report.observer_name, finding),
                )
            )

    # 2-3. Filter sub-"warn", already-surfaced, and in-call duplicates.
    survivors: list[_Candidate] = []
    seen: set[str] = set()
    for c in candidates:
        if _SEVERITY_RANK[c.severity] < _MIN_RANK:
            continue
        if c.hash_id in already_acted or c.hash_id in seen:
            continue
        seen.add(c.hash_id)
        survivors.append(c)

    # 4. Nothing new to say.
    if not survivors:
        return None

    # 5. Stable high→low sort. A NEGATIVE key (not reverse=True) makes the intent
    #    explicit: ascending on -rank == descending on rank, and sorted() is stable
    #    so encounter order is preserved within an equal-severity tier.
    survivors.sort(key=lambda c: -_SEVERITY_RANK[c.severity])

    # 6. Surface at most MAX_DIRECTIVES.
    selected = survivors[:MAX_DIRECTIVES]
    # survivors is sorted severity-descending, so the first surfaced candidate
    # already carries the highest severity — no need to re-scan the slice.
    top_severity: Severity = selected[0].severity
    return DriverFeedback(
        from_step=step,
        directives=[c.finding for c in selected],
        top_severity=top_severity,
        finding_hashes=[c.hash_id for c in selected],
    )


__all__ = ["MAX_DIRECTIVES", "distill_feedback", "finding_hash"]
