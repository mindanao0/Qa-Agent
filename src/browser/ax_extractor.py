import re
from dataclasses import dataclass

from loguru import logger
from playwright.async_api import Page

# Roles considered interactive for semantic density scoring
_INTERACTIVE_ROLES: frozenset[str] = frozenset({
    "button", "link", "textbox", "searchbox", "combobox", "listbox",
    "checkbox", "radio", "switch", "slider", "spinbutton", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "tab", "treeitem",
    "gridcell", "columnheader", "rowheader",
})

# Roles that are purely decorative / non-semantic — prune these
_DECORATIVE_ROLES: frozenset[str] = frozenset({
    "none", "presentation", "separator", "generic",
})

# Rough chars-per-token estimate for English/code mixed content
_CHARS_PER_TOKEN = 4
_TOKEN_WARN_THRESHOLD = 4000

# Node pattern: `  - role "name" [attrs]` or `  - role:`
_NODE_RE = re.compile(
    r"^(?P<indent>\s*)-\s+(?P<role>\w[\w-]*)(?:\s+\"(?P<name>[^\"]*)\")?(?:\s+\[(?P<attrs>[^\]]*)\])?:?\s*$"
)
# aria-hidden attribute pattern inside the [...] block
_ARIA_HIDDEN_RE = re.compile(r"aria-hidden(?:\s*:\s*true)?", re.IGNORECASE)
# data-testid in attrs
_TESTID_RE = re.compile(r'data-testid="?([^",\]]+)"?', re.IGNORECASE)


@dataclass
class AXNode:
    indent: int
    role: str
    name: str
    attrs: str
    raw_line: str

    @property
    def is_interactive(self) -> bool:
        return self.role.lower() in _INTERACTIVE_ROLES

    @property
    def is_decorative(self) -> bool:
        return self.role.lower() in _DECORATIVE_ROLES or _ARIA_HIDDEN_RE.search(self.attrs) is not None

    @property
    def testid(self) -> str:
        m = _TESTID_RE.search(self.attrs)
        return m.group(1) if m else ""


def _parse_nodes(raw_yaml: str) -> list[AXNode]:
    nodes: list[AXNode] = []
    for line in raw_yaml.splitlines():
        if not line.strip():
            continue
        m = _NODE_RE.match(line)
        if m:
            nodes.append(
                AXNode(
                    indent=len(m.group("indent")),
                    role=m.group("role") or "",
                    name=m.group("name") or "",
                    attrs=m.group("attrs") or "",
                    raw_line=line,
                )
            )
    return nodes


async def extract_axtree(page: Page) -> str:
    """
    Capture the Accessibility Tree of the current page via aria_snapshot().
    Returns raw YAML string; call prune_axtree() before sending to an LLM.
    """
    try:
        snapshot: str = await page.aria_snapshot()
        logger.debug(f"AxTree extracted | raw_lines={len(snapshot.splitlines())}")
        return snapshot
    except Exception as exc:
        logger.error(f"extract_axtree failed: {exc}")
        return ""


async def prune_axtree(raw_yaml: str, max_nodes: int = 200) -> str:
    """
    Remove decorative / aria-hidden nodes and cap the tree at max_nodes lines.

    Strategy:
    1. Drop nodes whose role is in _DECORATIVE_ROLES or that carry aria-hidden.
    2. Keep all interactive nodes unconditionally.
    3. Keep structural/semantic container nodes (landmark roles).
    4. If still over max_nodes, keep the first max_nodes retained lines (breadth-first
       order preserves document structure over deep-leaf decoration).

    Returns a pruned YAML string safe for LLM ingestion.
    """
    if not raw_yaml:
        return ""

    nodes = _parse_nodes(raw_yaml)

    _STRUCTURAL_ROLES: frozenset[str] = frozenset({
        "document", "banner", "navigation", "main", "complementary",
        "contentinfo", "region", "form", "search", "article", "section",
        "dialog", "alertdialog", "heading", "list", "listitem",
        "table", "row", "grid", "rowgroup", "tree", "group",
        "toolbar", "menu", "menubar", "tablist", "tabpanel",
    })

    kept: list[str] = []
    for node in nodes:
        if node.is_decorative:
            continue
        if node.is_interactive or node.role.lower() in _STRUCTURAL_ROLES:
            kept.append(node.raw_line)
        elif node.name:
            # Named node even if not obviously interactive — might be label or text
            kept.append(node.raw_line)

    if len(kept) > max_nodes:
        logger.debug(
            f"prune_axtree: trimmed {len(kept)} → {max_nodes} nodes"
        )
        kept = kept[:max_nodes]

    pruned = "\n".join(kept)
    _warn_token_budget(pruned)
    return pruned


def compute_semantic_density(axtree: str) -> float:
    """
    Ratio of interactive nodes to total named/role nodes.

    Returns a float in [0, 1].  Values below 0.4 trigger VLM fallback signal.
    """
    if not axtree:
        return 0.0

    nodes = _parse_nodes(axtree)
    if not nodes:
        return 0.0

    total = sum(1 for n in nodes if n.role and n.role.lower() not in {"document"})
    interactive = sum(1 for n in nodes if n.is_interactive)

    if total == 0:
        return 0.0

    density = interactive / total
    logger.debug(
        f"Semantic density: {density:.3f} "
        f"(interactive={interactive}, total={total})"
    )
    return round(density, 4)


def needs_vlm_fallback(axtree: str, threshold: float = 0.4) -> bool:
    """
    Return True when semantic density is below threshold, signalling that
    the AxTree alone is insufficient and a VLM should be activated.
    Does NOT call the VLM — only emits the signal.
    """
    density = compute_semantic_density(axtree)
    if density < threshold:
        logger.warning(
            f"AxTree semantic density {density:.3f} < {threshold} — VLM fallback required"
        )
        return True
    return False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _warn_token_budget(text: str) -> None:
    tokens = estimate_tokens(text)
    if tokens > _TOKEN_WARN_THRESHOLD:
        logger.warning(
            f"AxTree token estimate {tokens} exceeds {_TOKEN_WARN_THRESHOLD} — "
            "consider reducing max_nodes or further pruning before LLM call"
        )
