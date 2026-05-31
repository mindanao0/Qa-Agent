"""Accessibility Observer — audits ARIA hygiene after each Driver action.

For every ``driver_action`` trace event the observer:
  1. Snapshots the current accessibility tree.
  2. Diffs against the previously captured snapshot.
  3. Flags missing accessible names on interactive nodes and decorative roles
     (``none`` / ``presentation``) attached to elements that look interactive.
"""
from __future__ import annotations

import re
import time

from playwright.async_api import Page

from src.agents.observer_driver.observers.base_observer import BaseObserver
from src.agents.observer_driver.state import ObserverReport, Severity
from src.agents.observer_driver.trace_bus import TraceEvent

_INTERACTIVE_ROLES = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "checkbox",
        "radio",
        "switch",
        "slider",
        "spinbutton",
        "menuitem",
        "tab",
        "treeitem",
    }
)

_DECORATIVE_ROLES = frozenset({"none", "presentation"})

# Matches "  - <role> [\"name\"] [attrs]" lines in aria_snapshot YAML.
_NODE_RE = re.compile(
    r"^\s*-\s+(?P<role>[\w-]+)"
    r"(?:\s+\"(?P<name>[^\"]*)\")?"
    r"(?:\s+\[(?P<attrs>[^\]]*)\])?",
)


class AccessibilityObserver(BaseObserver):
    """Detect missing accessible names and improper decorative roles."""

    def __init__(self, queue) -> None:  # type: ignore[no-untyped-def]
        super().__init__(name="accessibility", queue=queue)
        self._previous_snapshot: str = ""

    async def _audit(
        self, event: TraceEvent, page: Page
    ) -> ObserverReport | None:
        if event.event_type != "driver_action":
            return None

        try:
            current = await page.aria_snapshot()
        except Exception:
            current = ""

        findings = self._scan(current)
        diff_added, diff_removed = self._diff(self._previous_snapshot, current)
        self._previous_snapshot = current

        if diff_added or diff_removed:
            findings.append(
                f"AXTree delta: +{diff_added} nodes / -{diff_removed} nodes"
            )

        severity: Severity = "info"
        if any(f.startswith("missing accessible name") for f in findings):
            severity = "warn"
        if any(f.startswith("decorative role") for f in findings):
            severity = "warn"

        return ObserverReport(
            observer_name=self.name,
            findings=findings or ["no a11y issues detected"],
            severity=severity,
            timestamp_ms=int(time.time() * 1000),
        )

    @staticmethod
    def _scan(snapshot: str) -> list[str]:
        findings: list[str] = []
        missing_name_count = 0
        decorative_interactive_count = 0

        for line in snapshot.splitlines():
            match = _NODE_RE.match(line)
            if not match:
                continue
            role = (match.group("role") or "").lower()
            name = match.group("name") or ""
            attrs = match.group("attrs") or ""

            if role in _INTERACTIVE_ROLES and not name.strip():
                missing_name_count += 1
            if (
                role in _DECORATIVE_ROLES
                and ("onclick" in attrs.lower() or "tabindex" in attrs.lower())
            ):
                decorative_interactive_count += 1

        if missing_name_count:
            findings.append(
                f"missing accessible name on {missing_name_count} interactive node(s)"
            )
        if decorative_interactive_count:
            findings.append(
                f"decorative role on {decorative_interactive_count} interactive-looking node(s)"
            )
        return findings

    @staticmethod
    def _diff(previous: str, current: str) -> tuple[int, int]:
        prev_lines = set(previous.splitlines())
        curr_lines = set(current.splitlines())
        added = len(curr_lines - prev_lines)
        removed = len(prev_lines - curr_lines)
        return added, removed


__all__ = ["AccessibilityObserver"]
