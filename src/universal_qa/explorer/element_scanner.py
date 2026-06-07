from __future__ import annotations

from loguru import logger
from playwright.async_api import Page

from src.universal_qa.explorer.nav_map import ElementCandidate

# Roles/containers that indicate primary navigation.
_NAV_CONTAINERS = frozenset({"nav", "banner", "header", "navigation", "menu"})
_FORM_ROLES = frozenset({"button"})

# JS that collects interactive elements from main doc + shadow roots.
_SCAN_JS = """
() => {
    const out = [];
    const seen = new Set();
    function describe(el, container) {
        const role = el.getAttribute('role')
            || ({BUTTON:'button', A:'link', INPUT:'textbox', SELECT:'combobox'})[el.tagName] || null;
        const name = (el.getAttribute('aria-label')
            || el.textContent || el.value || '').trim().slice(0, 60);
        const label = name || role || el.tagName.toLowerCase();
        if (!label || seen.has(container + '|' + label)) return;
        seen.add(container + '|' + label);
        const sel = el.id ? '#' + el.id
            : (el.className && typeof el.className === 'string' && el.className.trim()
                ? el.tagName.toLowerCase() + '.' + el.className.trim().split(/\\s+/)[0]
                : el.tagName.toLowerCase());
        out.push({label, role, name: name || null, selector: sel,
                  container, is_in_iframe: false, is_in_shadow: container === 'shadow'});
    }
    const SEL = 'a, button, [role=button], [role=link], input[type=submit], select, [onclick]';
    function containerOf(el) {
        let p = el;
        while (p) {
            const tag = (p.tagName || '').toLowerCase();
            const role = p.getAttribute && p.getAttribute('role');
            if (tag === 'nav' || tag === 'header' || role === 'navigation' || role === 'banner') return 'nav';
            if (tag === 'form') return 'form';
            p = p.parentElement;
        }
        return 'main';
    }
    document.querySelectorAll(SEL).forEach(el => describe(el, containerOf(el)));
    // shadow DOM piercing
    document.querySelectorAll('*').forEach(host => {
        if (host.shadowRoot) {
            host.shadowRoot.querySelectorAll(SEL).forEach(el => describe(el, 'shadow'));
        }
    });
    return out.slice(0, 40);
}
"""


class ElementScanner:
    """Scans a page for interactive elements (main doc + shadow + same-origin iframes)."""

    @staticmethod
    def _classify_priority(container_or_role: str, label: str) -> int:
        c = (container_or_role or "").lower()
        if c in _NAV_CONTAINERS:
            return 0
        if c in _FORM_ROLES or c == "form":
            return 1
        return 2

    def _raw_to_candidates(self, raw: list[dict]) -> list[ElementCandidate]:
        cands: list[ElementCandidate] = []
        for r in raw:
            container = r.get("container", "main")
            role = r.get("role")
            # priority by container first, then role
            prio = self._classify_priority(container, r.get("label", ""))
            if prio == 2 and role:
                prio = self._classify_priority(role, r.get("label", ""))
            cands.append(ElementCandidate(
                label=r.get("label", ""),
                role=role,
                name=r.get("name"),
                selector=r.get("selector"),
                is_in_iframe=bool(r.get("is_in_iframe")),
                is_in_shadow=bool(r.get("is_in_shadow")),
                priority=prio,
            ))
        cands.sort(key=lambda c: c.priority)
        return cands

    async def scan(self, page: Page) -> list[ElementCandidate]:
        try:
            raw: list[dict] = await page.evaluate(_SCAN_JS)
        except Exception as exc:
            logger.warning(f"ElementScanner: main scan failed — {exc!r}")
            raw = []

        # same-origin iframes
        for frame in getattr(page, "frames", []):
            try:
                if frame == page.main_frame:
                    continue
                furl = frame.url
                if not furl or not furl.startswith("http"):
                    continue
                fraw = await frame.evaluate(_SCAN_JS)
                for item in fraw:
                    item["is_in_iframe"] = True
                raw.extend(fraw)
            except Exception:
                continue

        return self._raw_to_candidates(raw)


__all__ = ["ElementScanner"]
