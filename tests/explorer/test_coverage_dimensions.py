"""Tests for eval.coverage_dimensions.merge_dimensions and eval.dim_static —
the pure-logic and network-parsing pieces of the coverage cross-check (no
live network, no browser, no Ollama; httpx.MockTransport stands in for the
real site)."""
import httpx
import pytest

from eval.coverage_dimensions import DimensionResult, merge_dimensions
from eval.dim_static import extract_routes_from_js, fetch_robots_disallow, fetch_sitemap_paths


# ─── merge_dimensions ────────────────────────────────────────────────────

def test_merge_no_path_dimensions_returns_scalar_only():
    scalar = DimensionResult(method="js_css_coverage", target_id="t",
                              paths=set(), meta={"js_pct_executed": 42.0})
    out = merge_dimensions([scalar])
    assert out["path_dimensions"] == []
    assert out["scalar_metrics"] == {"js_css_coverage": {"js_pct_executed": 42.0}}


def test_merge_single_dimension_everything_is_seen_by_all_and_exclusive():
    d = DimensionResult(method="sfg_crawl", target_id="t", paths={"/a", "/b"})
    out = merge_dimensions([d])
    assert out["total_distinct_paths"] == 2
    assert out["seen_by_all_count"] == 2
    assert out["exclusive_per_method"]["sfg_crawl"] == ["/a", "/b"]


def test_merge_two_dimensions_overlap_and_exclusive_are_correct():
    crawl = DimensionResult(method="sfg_crawl", target_id="t", paths={"/a", "/b", "/checkout"})
    static = DimensionResult(method="static_declared", target_id="t", paths={"/a", "/b", "/admin"})
    out = merge_dimensions([crawl, static])

    assert out["total_distinct_paths"] == 4
    assert out["seen_by_all_methods"] == ["/a", "/b"]
    assert out["exclusive_per_method"]["sfg_crawl"] == ["/checkout"]
    assert out["exclusive_per_method"]["static_declared"] == ["/admin"]
    assert out["per_path_sources"]["/a"] == ["sfg_crawl", "static_declared"]
    assert out["per_path_sources"]["/checkout"] == ["sfg_crawl"]


def test_merge_keeps_scalar_dimensions_out_of_path_crosscheck():
    crawl = DimensionResult(method="sfg_crawl", target_id="t", paths={"/a"})
    js_cov = DimensionResult(method="js_css_coverage", target_id="t", paths=set(),
                              meta={"js_pct_executed": 55.5})
    out = merge_dimensions([crawl, js_cov])
    assert out["methods"] == ["sfg_crawl"]
    assert out["scalar_metrics"] == {"js_css_coverage": {"js_pct_executed": 55.5}}


# ─── extract_routes_from_js ──────────────────────────────────────────────

def test_extract_routes_react_router_style():
    js = 'const routes = [{path: "/checkout", component: Checkout}, {path: "/cart"}];'
    assert extract_routes_from_js(js) == {"/checkout", "/cart"}


def test_extract_routes_jsx_style():
    js = '<Route path="/admin/users" element={<Users />} />'
    assert extract_routes_from_js(js) == {"/admin/users"}


def test_extract_routes_ignores_root_and_overlong_matches():
    js = 'path: "/"; path: "' + "/x" * 100 + '";'
    assert extract_routes_from_js(js) == set()


def test_extract_routes_no_matches_on_plain_text():
    assert extract_routes_from_js("function foo() { return 1; }") == set()


# ─── fetch_sitemap_paths / fetch_robots_disallow (mocked transport, no live network) ──

_SITEMAP_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/products</loc></url>
  <url><loc>https://example.com/checkout</loc></url>
</urlset>"""

_ROBOTS_TXT = b"User-agent: *\nDisallow: /admin\nDisallow: /internal-api\nDisallow: /\n"


@pytest.mark.asyncio
async def test_fetch_sitemap_paths_parses_urlset():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=_SITEMAP_XML)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        paths = await fetch_sitemap_paths(client, "https://example.com")
    assert paths == {"/products", "/checkout"}


@pytest.mark.asyncio
async def test_fetch_robots_disallow_skips_bare_slash():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=_ROBOTS_TXT)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        paths = await fetch_robots_disallow(client, "https://example.com")
    assert paths == {"/admin", "/internal-api"}


@pytest.mark.asyncio
async def test_fetch_sitemap_paths_missing_returns_empty():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as client:
        paths = await fetch_sitemap_paths(client, "https://example.com")
    assert paths == set()
