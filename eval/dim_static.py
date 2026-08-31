"""Coverage dimension: declared-not-observed.

Reads sitemap.xml, robots.txt, and route patterns grepped out of the site's
own JS bundles — what the site SAYS it has, independent of what any crawler
managed to click its way into. Zero LLM calls, zero DOM interaction; pure
HTTP fetch + text parsing.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree as ET

import httpx

from eval.coverage_dimensions import DimensionResult

# React/Vue Router `path: "/x"` or JSX `<Route path="/x">`; Angular
# `loadChildren: () => import(...).then(m => m.X) ` lazy routes carry a
# sibling `path: "/x"` too, matched by the same first pattern.
_ROUTE_PATTERNS = [
    re.compile(r'path\s*:\s*["\']([/][a-zA-Z0-9_\-/:]*)["\']'),
    re.compile(r'\bpath=["\']([/][a-zA-Z0-9_\-/:]*)["\']'),
]
_ASSET_JS_RE = re.compile(r'<script[^>]+src=["\']([^"\']+\.js[^"\']*)["\']', re.I)
_UA = "Mozilla/5.0 (qa-agent coverage probe)"


async def fetch_sitemap_paths(client: httpx.AsyncClient, base: str) -> set[str]:
    paths: set[str] = set()
    for candidate in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            r = await client.get(urljoin(base, candidate), timeout=10.0)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            continue
        for el in root.iter():
            if el.tag.endswith("loc") and el.text:
                paths.add(urlsplit(el.text.strip()).path or "/")
    return paths


async def fetch_robots_disallow(client: httpx.AsyncClient, base: str) -> set[str]:
    try:
        r = await client.get(urljoin(base, "/robots.txt"), timeout=10.0)
    except Exception:  # noqa: BLE001
        return set()
    if r.status_code != 200:
        return set()
    out: set[str] = set()
    for line in r.text.splitlines():
        line = line.strip()
        if line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path and path != "/":
                out.add(path)
    return out


def extract_routes_from_js(js_text: str) -> set[str]:
    found: set[str] = set()
    for pat in _ROUTE_PATTERNS:
        for m in pat.finditer(js_text):
            p = m.group(1)
            if p and p != "/" and len(p) < 120:
                found.add(p)
    return found


async def fetch_js_bundle_routes(client: httpx.AsyncClient, base: str, html: str,
                                  max_bundles: int = 6, max_bytes: int = 3_000_000) -> set[str]:
    scripts = _ASSET_JS_RE.findall(html)[:max_bundles]
    routes: set[str] = set()
    for src in scripts:
        try:
            r = await client.get(urljoin(base, src), timeout=15.0)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        routes |= extract_routes_from_js(r.text[:max_bytes])
    return routes


async def collect(target_id: str, seed_url: str) -> DimensionResult:
    base = f"{urlsplit(seed_url).scheme}://{urlsplit(seed_url).netloc}"
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": _UA}) as client:
        sitemap_paths = await fetch_sitemap_paths(client, base)
        disallow_paths = await fetch_robots_disallow(client, base)
        try:
            home = await client.get(seed_url, timeout=15.0)
            html = home.text if home.status_code == 200 else ""
        except Exception:  # noqa: BLE001
            html = ""
        bundle_routes = await fetch_js_bundle_routes(client, base, html) if html else set()

    return DimensionResult(
        method="static_declared",
        target_id=target_id,
        paths=sitemap_paths | bundle_routes,
        meta={
            "sitemap_found": bool(sitemap_paths),
            "sitemap_count": len(sitemap_paths),
            "robots_disallow_count": len(disallow_paths),
            "robots_disallow_paths": sorted(disallow_paths),
            "js_bundle_route_count": len(bundle_routes),
        },
    )
