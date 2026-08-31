"""Tests for eval.coverage_dimensions.merge_dimensions, eval.dim_static, and
eval.dim_api_spec — the pure-logic and network-parsing pieces of the coverage
cross-check (no live network, no browser, no Ollama; httpx.MockTransport
stands in for the real site)."""
import httpx
import pytest

from eval.coverage_dimensions import DimensionResult, merge_dimensions
from eval.dim_static import extract_routes_from_js, fetch_robots_disallow, fetch_sitemap_paths
from eval.dim_api_spec import (
    collect as api_spec_collect,
    parse_graphql_introspection,
    parse_openapi_paths,
    probe_graphql,
    probe_openapi,
)


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


# ─── dim_api_spec: parse_openapi_paths / parse_graphql_introspection ─────

_OPENAPI_DOC = {
    "openapi": "3.0.0",
    "paths": {
        "/users": {"get": {}},
        "/users/{id}": {"get": {}, "delete": {}},
        "/orders/{orderId}/items": {"post": {}},
    },
}

_GRAPHQL_INTROSPECTION_RESPONSE = {
    "data": {
        "__schema": {
            "queryType": {"name": "Query"},
            "mutationType": {"name": "Mutation"},
            "subscriptionType": None,
            "types": [
                {"name": "Query", "kind": "OBJECT", "fields": [{"name": "user"}, {"name": "orders"}]},
                {"name": "Mutation", "kind": "OBJECT", "fields": [{"name": "createOrder"}]},
                {"name": "User", "kind": "OBJECT", "fields": [{"name": "id"}, {"name": "email"}]},
                {"name": "String", "kind": "SCALAR", "fields": None},
            ],
        }
    }
}


def test_parse_openapi_paths_extracts_raw_templates():
    assert parse_openapi_paths(_OPENAPI_DOC) == {"/users", "/users/{id}", "/orders/{orderId}/items"}


def test_parse_openapi_paths_missing_paths_key_returns_empty():
    assert parse_openapi_paths({"openapi": "3.0.0"}) == set()


def test_parse_graphql_introspection_extracts_types_and_field_count():
    types, field_count = parse_graphql_introspection(_GRAPHQL_INTROSPECTION_RESPONSE)
    assert types == {"Query", "Mutation", "User", "String"}
    # 2 (Query) + 1 (Mutation) + 2 (User) + 0 (String, fields=None) = 5
    assert field_count == 5


def test_parse_graphql_introspection_malformed_body_returns_empty():
    assert parse_graphql_introspection({"data": {}}) == (set(), 0)
    assert parse_graphql_introspection({}) == (set(), 0)


# ─── dim_api_spec: probe_openapi / probe_graphql (mocked transport) ──────

@pytest.mark.asyncio
async def test_probe_openapi_finds_document_at_first_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/swagger.json":
            return httpx.Response(200, json=_OPENAPI_DOC)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        found_at, paths = await probe_openapi(client, "https://example.com")
    assert found_at == "/swagger.json"
    assert paths == {"/users", "/users/{id}", "/orders/{orderId}/items"}


@pytest.mark.asyncio
async def test_probe_openapi_none_found_returns_none_and_empty():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as client:
        found_at, paths = await probe_openapi(client, "https://example.com")
    assert found_at is None
    assert paths == set()


@pytest.mark.asyncio
async def test_probe_openapi_skips_html_error_pages():
    def handler(request: httpx.Request) -> httpx.Response:
        # A misconfigured server returning a 200 HTML page instead of a 404
        # for every candidate path must not be mistaken for a real spec.
        return httpx.Response(200, content=b"<html>not found</html>",
                               headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        found_at, paths = await probe_openapi(client, "https://example.com")
    assert found_at is None
    assert paths == set()


@pytest.mark.asyncio
async def test_probe_graphql_finds_endpoint_at_first_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_GRAPHQL_INTROSPECTION_RESPONSE)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        found_at, types, field_count = await probe_graphql(client, "https://example.com")
    assert found_at == "/graphql"
    assert types == {"Query", "Mutation", "User", "String"}
    assert field_count == 5


@pytest.mark.asyncio
async def test_probe_graphql_none_found_returns_none_and_empty():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as client:
        found_at, types, field_count = await probe_graphql(client, "https://example.com")
    assert found_at is None
    assert types == set()
    assert field_count == 0


# ─── dim_api_spec: collect() end-to-end ──────────────────────────────────

@pytest.mark.asyncio
async def test_collect_neither_openapi_nor_graphql_present(monkeypatch):
    _orig_init = httpx.AsyncClient.__init__

    def fake_client_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(404))
        return _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", fake_client_init)

    from eval import dim_api_spec

    result = await dim_api_spec.collect("target1", "https://example.com/")
    assert result.method == "api_spec"
    assert result.target_id == "target1"
    assert result.paths == set()
    assert result.meta["openapi_found_at"] is None
    assert result.meta["openapi_endpoint_count"] == 0
    assert result.meta["graphql_found_at"] is None
    assert result.meta["graphql_type_count"] == 0
    assert result.meta["graphql_field_count"] == 0


@pytest.mark.asyncio
async def test_collect_finds_both_openapi_and_graphql(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_DOC)
        if request.url.path == "/graphql":
            return httpx.Response(200, json=_GRAPHQL_INTROSPECTION_RESPONSE)
        return httpx.Response(404)

    _orig_init = httpx.AsyncClient.__init__

    def fake_client_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _orig_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", fake_client_init)

    from eval import dim_api_spec

    result = await dim_api_spec.collect("target1", "https://example.com/")
    assert result.method == "api_spec"
    assert result.paths == {"/users", "/users/{id}", "/orders/{orderId}/items"}
    assert result.meta["openapi_found_at"] == "/openapi.json"
    assert result.meta["openapi_endpoint_count"] == 3
    assert result.meta["graphql_found_at"] == "/graphql"
    assert result.meta["graphql_type_count"] == 4
    assert result.meta["graphql_field_count"] == 5
    assert result.meta["graphql_types"] == ["Mutation", "Query", "String", "User"]
