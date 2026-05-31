import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Literal

import jellyfish
from filelock import FileLock
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.llm.structured import HealedLocator

# Thresholds
FUZZY_AUTO_ACCEPT = 0.85
FUZZY_AI_FALLBACK = 0.50

# Playwright locator patterns for extraction
_LOCATOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("get_by_role", re.compile(
        r'get_by_role\s*\(\s*["\'](?P<role>[^"\']+)["\']'
        r'(?:.*?name\s*=\s*["\'](?P<name>[^"\']+)["\'])?',
        re.DOTALL,
    )),
    ("get_by_label", re.compile(
        r'get_by_label\s*\(\s*["\'](?P<label>[^"\']+)["\']'
    )),
    ("get_by_text", re.compile(
        r'get_by_text\s*\(\s*["\'](?P<text>[^"\']+)["\']'
    )),
    ("get_by_test_id", re.compile(
        r'get_by_test_id\s*\(\s*["\'](?P<testid>[^"\']+)["\']'
    )),
    ("get_by_placeholder", re.compile(
        r'get_by_placeholder\s*\(\s*["\'](?P<placeholder>[^"\']+)["\']'
    )),
]

# AxTree element line pattern
_NODE_RE = re.compile(
    r"^(?P<indent>\s*)-\s+(?P<role>\w[\w-]*)(?:\s+\"(?P<name>[^\"]*)\")?"
    r"(?:\s+\[(?P<attrs>[^\]]*)\])?:?\s*$"
)
_TESTID_RE = re.compile(r'data-testid="?([^",\]]+)"?', re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r'placeholder="?([^",\]]+)"?', re.IGNORECASE)


@dataclass
class LocatorAttributes:
    """Parsed attributes from a Playwright locator string."""
    strategy: str          # e.g. "get_by_role"
    role: str = ""
    name: str = ""
    label: str = ""
    text: str = ""
    testid: str = ""
    placeholder: str = ""

    @property
    def primary_text(self) -> str:
        """Best string to fuzzy-match against element names in AxTree."""
        return self.name or self.label or self.text or self.testid or self.placeholder


@dataclass
class AXElement:
    """Element extracted from the pruned AxTree for matching."""
    role: str
    name: str
    testid: str
    placeholder: str
    raw_line: str

    def candidate_texts(self) -> list[str]:
        texts = []
        if self.name:
            texts.append(self.name)
        if self.testid:
            texts.append(self.testid)
        if self.placeholder:
            texts.append(self.placeholder)
        return texts

    def to_playwright_locator(self) -> str:
        """Generate a Playwright locator expression for this element."""
        if self.testid:
            return f'page.get_by_test_id("{self.testid}")'
        if self.role and self.name:
            return f'page.get_by_role("{self.role}", name="{self.name}")'
        if self.name:
            return f'page.get_by_text("{self.name}")'
        if self.role:
            return f'page.get_by_role("{self.role}")'
        return ""


def _parse_locator(locator_str: str) -> LocatorAttributes | None:
    """Extract structured attributes from a Playwright locator string."""
    for strategy, pattern in _LOCATOR_PATTERNS:
        m = pattern.search(locator_str)
        if not m:
            continue
        gd = m.groupdict()
        return LocatorAttributes(
            strategy=strategy,
            role=gd.get("role") or "",
            name=gd.get("name") or "",
            label=gd.get("label") or "",
            text=gd.get("text") or "",
            testid=gd.get("testid") or "",
            placeholder=gd.get("placeholder") or "",
        )
    return None


def _extract_elements(axtree: str) -> list[AXElement]:
    """Parse all named / interactive elements from the pruned AxTree."""
    elements: list[AXElement] = []
    for line in axtree.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue
        role = m.group("role") or ""
        name = m.group("name") or ""
        attrs = m.group("attrs") or ""
        testid_m = _TESTID_RE.search(attrs)
        placeholder_m = _PLACEHOLDER_RE.search(attrs)
        elem = AXElement(
            role=role,
            name=name,
            testid=testid_m.group(1) if testid_m else "",
            placeholder=placeholder_m.group(1) if placeholder_m else "",
            raw_line=line,
        )
        # Only include elements that have something to match against
        if elem.candidate_texts() or role:
            elements.append(elem)
    return elements


def _score_candidate(
    locator_attrs: LocatorAttributes,
    element: AXElement,
) -> float:
    """
    Compute a composite Jaro-Winkler similarity score between the locator
    attributes and an AxTree element.

    Weights:
    - text similarity (name/label/text vs element name): 60%
    - role match (exact): 40%
    """
    # Role bonus: if roles match exactly, add 0.4 weight
    role_score = 0.0
    if locator_attrs.role and element.role:
        role_score = 1.0 if locator_attrs.role.lower() == element.role.lower() else 0.0

    # Text similarity: compare primary_text against every candidate text in the element
    target = locator_attrs.primary_text.strip().lower()
    if not target:
        # No text to match on — rely solely on role
        return role_score * 0.4

    text_score = 0.0
    for candidate in element.candidate_texts():
        s = jellyfish.jaro_winkler_similarity(target, candidate.strip().lower())
        text_score = max(text_score, s)

    composite = text_score * 0.6 + role_score * 0.4
    return round(composite, 4)


def fuzzy_match(
    failed_locator: str,
    axtree: str,
    threshold: float = FUZZY_AUTO_ACCEPT,
) -> HealedLocator | None:
    """
    Phase 1 self-healing: Jaro-Winkler fuzzy match of a failed locator against
    the current page AxTree.

    Returns a HealedLocator if best confidence >= threshold, else None.
    Callers should fall through to AI healing if this returns None.
    """
    if not axtree:
        logger.warning("fuzzy_match: empty AxTree — cannot match")
        return None

    locator_attrs = _parse_locator(failed_locator)
    if not locator_attrs:
        logger.debug(f"fuzzy_match: could not parse locator: {failed_locator!r}")
        return None

    elements = _extract_elements(axtree)
    if not elements:
        logger.debug("fuzzy_match: no elements extracted from AxTree")
        return None

    best_score = 0.0
    best_elem: AXElement | None = None

    for elem in elements:
        score = _score_candidate(locator_attrs, elem)
        if score > best_score:
            best_score = score
            best_elem = elem

    if best_elem is None or best_score == 0.0:
        return None

    healed_str = best_elem.to_playwright_locator()
    if not healed_str:
        return None

    reasoning = (
        f"Fuzzy match: original={failed_locator!r} strategy={locator_attrs.strategy!r} "
        f"primary_text={locator_attrs.primary_text!r} → "
        f"best_element role={best_elem.role!r} name={best_elem.name!r} "
        f"score={best_score:.4f}"
    )
    logger.debug(reasoning)

    result = HealedLocator(
        reasoning=reasoning,
        original=failed_locator,
        healed=healed_str,
        confidence=best_score,
        method="fuzzy",
    )

    if best_score >= threshold:
        logger.info(
            f"fuzzy_match: auto-accept | confidence={best_score:.4f} "
            f"healed={healed_str!r}"
        )
        return result

    if best_score >= FUZZY_AI_FALLBACK:
        logger.info(
            f"fuzzy_match: AI fallback zone | confidence={best_score:.4f} — "
            "returning partial result for AI phase"
        )
        return result  # caller uses confidence to decide whether to escalate

    logger.debug(f"fuzzy_match: below AI fallback threshold ({best_score:.4f}) — returning None")
    return None


def score_locator_against_axtree(
    locator_str: str,
    axtree: str,
) -> dict[str, Any]:
    """
    Diagnostic helper: return full score table for a locator vs all AxTree elements.
    Useful for debugging and logging.
    """
    locator_attrs = _parse_locator(locator_str)
    if not locator_attrs:
        return {"error": "Could not parse locator", "locator": locator_str}

    elements = _extract_elements(axtree)
    scores = []
    for elem in elements:
        scores.append({
            "role": elem.role,
            "name": elem.name,
            "testid": elem.testid,
            "score": _score_candidate(locator_attrs, elem),
            "healed": elem.to_playwright_locator(),
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {
        "locator": locator_str,
        "parsed": {
            "strategy": locator_attrs.strategy,
            "role": locator_attrs.role,
            "primary_text": locator_attrs.primary_text,
        },
        "top_matches": scores[:5],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 2 — Cluster S2-A
#
# New FuzzyMatcher class: searches a JSON repository of previously-known
# locators using a self-implemented Jaro-Winkler similarity.  The legacy
# `fuzzy_match()` / `score_locator_against_axtree()` functions above are
# preserved unchanged for the current healer engine; Cluster D will rewire
# the healer to use this class instead.
#
# Strict implementation rules:
#   - Jaro-Winkler is computed from scratch (no jellyfish / rapidfuzz here).
#   - Repository read is wrapped in a FileLock matching the engine's pattern.
#   - find_candidates() never raises — it returns an empty list on any error.
# ──────────────────────────────────────────────────────────────────────────────


class FuzzyCandidate(BaseModel):
    """A previously-known healed locator scored against a failed locator."""

    model_config = ConfigDict(extra="forbid")

    candidate_locator: str
    similarity_score: float
    source_file: str
    last_used_iso: str | None
    confidence_tier: Literal["HIGH", "MEDIUM"]


class FuzzyMatcher:
    """
    Fuzzy matcher over a JSON locator repository.

    The corpus is the KEYS of the JSON dict (each a Playwright locator
    expression).  We score each key against the failed locator using a
    self-implemented Jaro-Winkler.  The returned ``candidate_locator`` is
    the entry's ``healed`` value — i.e. the locator that previously worked.

    Tiers:
        HIGH   ≥ 0.85
        MEDIUM 0.60 – 0.85
        below 0.60 → discarded

    ``threshold`` semantics:
        - threshold ≥ 0.85 → only return HIGH candidates above threshold.
        - threshold < 0.85 → return all candidates (HIGH + MEDIUM) above
          threshold (still discards anything below 0.60).
    """

    _HIGH_THRESHOLD = 0.85
    _MEDIUM_THRESHOLD = 0.60

    def __init__(self, locator_repo_path: pathlib.Path):
        self.locator_repo_path = pathlib.Path(locator_repo_path)
        self._lock_path = self.locator_repo_path.with_suffix(".lock")
        # Simple mtime-keyed cache (avoids re-reading on every call).
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: float | None = None

    # ── public API ────────────────────────────────────────────────────────

    def find_candidates(
        self,
        failed_locator: str,
        threshold: float = 0.85,
    ) -> list[FuzzyCandidate]:
        repo = self._load_repo()
        if not repo:
            return []

        results: list[FuzzyCandidate] = []
        source = str(self.locator_repo_path)

        for key, entry in repo.items():
            if not isinstance(entry, dict):
                continue
            healed = entry.get("healed")
            if not isinstance(healed, str) or not healed:
                continue

            score = self._jaro_winkler(failed_locator, key)
            if score < self._MEDIUM_THRESHOLD:
                continue

            tier: Literal["HIGH", "MEDIUM"] = (
                "HIGH" if score >= self._HIGH_THRESHOLD else "MEDIUM"
            )

            # Threshold filtering
            if threshold >= self._HIGH_THRESHOLD:
                # Only HIGH candidates above the (high) threshold survive
                if tier != "HIGH" or score < threshold:
                    continue
            else:
                if score < threshold:
                    continue

            last_used = entry.get("last_updated")
            if last_used is not None and not isinstance(last_used, str):
                last_used = None

            results.append(
                FuzzyCandidate(
                    candidate_locator=healed,
                    similarity_score=score,
                    source_file=source,
                    last_used_iso=last_used,
                    confidence_tier=tier,
                )
            )

        results.sort(key=lambda c: c.similarity_score, reverse=True)
        return results

    # ── Jaro-Winkler (self-implemented) ──────────────────────────────────

    def _jaro_winkler(self, s1: str, s2: str) -> float:
        if s1 == s2:
            # Trivially identical (covers both empty-string case and equal-string case)
            return 1.0 if s1 else 0.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Matching window: floor(max(len1, len2) / 2) - 1  (clamped to ≥ 0)
        match_window = max(0, (max(len1, len2) // 2) - 1)

        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0

        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            for j in range(start, end):
                if s2_matches[j]:
                    continue
                if s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Transpositions: walk matched chars in order from s1 and s2 in parallel
        t = 0
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
        transpositions = t / 2.0

        m = float(matches)
        jaro = (m / len1 + m / len2 + (m - transpositions) / m) / 3.0

        # Winkler prefix boost (p up to 4, scaling factor 0.1)
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1.0 - jaro)

    # ── internal helpers ──────────────────────────────────────────────────

    def _load_repo(self) -> dict[str, Any]:
        path = self.locator_repo_path
        if not path.exists():
            return {}

        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            logger.warning(f"FuzzyMatcher: could not stat {path}: {exc}")
            return {}

        if self._cache is not None and self._cache_mtime == mtime:
            return self._cache

        try:
            lock = FileLock(str(self._lock_path), timeout=10)
            with lock:
                text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as exc:
            logger.warning(
                f"FuzzyMatcher: could not read locator repo {path}: {exc}"
            )
            return {}

        if not isinstance(data, dict):
            logger.warning(
                f"FuzzyMatcher: locator repo {path} is not a JSON object — ignoring"
            )
            return {}

        self._cache = data
        self._cache_mtime = mtime
        return data
