"""Coverage dimension: declared API surface (OpenAPI/Swagger + GraphQL
introspection).

Reads the site's own machine-readable API contract — an OpenAPI/Swagger
document if one is published at a conventional path, or a GraphQL schema via
the standard introspection query — independent of anything a browser-based
crawler ever clicked into or any route grepped out of a JS bundle. This
catches API endpoints that exist and are documented but are never exercised
by UI navigation (e.g. admin/internal endpoints, webhooks, batch/export
routes) as well as GraphQL types/fields that have no corresponding page at
all. Zero LLM calls, zero DOM interaction; pure HTTP fetch + JSON parsing.
"""
from __future__ import annotations

from urllib.parse import urljoin, urlsplit

import httpx

from src.universal_qa.coverage.dimensions import DimensionResult

_UA = "Mozilla/5.0 (qa-agent coverage probe)"

# Conventional locations an OpenAPI/Swagger document is published at.
_OPENAPI_CANDIDATES = (
    "/openapi.json",
    "/swagger.json",
    "/api-docs",
    "/v2/api-docs",
    "/api/openapi.json",
    "/swagger/v1/swagger.json",
)

# Conventional GraphQL endpoint paths.
_GRAPHQL_CANDIDATES = (
    "/graphql",
    "/api/graphql",
    "/v1/graphql",
)

# The full standard GraphQL introspection query (schema-level: query/mutation/
# subscription root type names + every type's name/kind/fields). Deliberately
# NOT the exhaustive introspection query (no nested field-arg/interface/enum
# walk) — this dimension only needs type + field *names* to report surface
# area, not a full schema mirror.
INTROSPECTION_QUERY = """
{ __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      name
      kind
      fields { name }
    }
  }
}
""".strip()


def parse_openapi_paths(doc: dict) -> set[str]:
    """Extract declared endpoint path templates from an OpenAPI 2.x or 3.x
    document's top-level `paths` object. Raw template as declared, e.g.
    "/users/{id}" — no attempt to resolve path parameters.
    """
    paths_obj = doc.get("paths")
    if not isinstance(paths_obj, dict):
        return set()
    return {p for p in paths_obj if isinstance(p, str)}


def parse_graphql_introspection(body: dict) -> tuple[set[str], int]:
    """Extract type names and a total field count from a standard GraphQL
    introspection response's `data.__schema.types` list. Returns
    (type_names, field_count); field_count sums fields across all types
    (built-in `__Type`/`__Schema` introspection types included, same as the
    server reports them — no filtering, this is a raw surface-area count).
    """
    schema = (body or {}).get("data", {}).get("__schema")
    if not isinstance(schema, dict):
        return set(), 0
    types = schema.get("types")
    if not isinstance(types, list):
        return set(), 0
    type_names: set[str] = set()
    field_count = 0
    for t in types:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if isinstance(name, str):
            type_names.add(name)
        fields = t.get("fields")
        if isinstance(fields, list):
            field_count += len(fields)
    return type_names, field_count


async def probe_openapi(client: httpx.AsyncClient, base: str) -> tuple[str | None, set[str]]:
    for candidate in _OPENAPI_CANDIDATES:
        try:
            r = await client.get(urljoin(base, candidate), timeout=10.0)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        ctype = r.headers.get("content-type", "")
        if "json" not in ctype and not ctype == "":
            # Some servers omit/mislabel content-type for these paths; still
            # attempt a JSON parse below rather than trusting the header alone,
            # but skip obvious non-JSON (e.g. text/html error pages).
            if "html" in ctype:
                continue
        try:
            doc = r.json()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        endpoints = parse_openapi_paths(doc)
        if endpoints:
            return candidate, endpoints
    return None, set()


async def probe_graphql(client: httpx.AsyncClient, base: str) -> tuple[str | None, set[str], int]:
    for candidate in _GRAPHQL_CANDIDATES:
        try:
            r = await client.post(
                urljoin(base, candidate),
                json={"query": INTROSPECTION_QUERY},
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        try:
            body = r.json()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(body, dict):
            continue
        types, field_count = parse_graphql_introspection(body)
        if types:
            return candidate, types, field_count
    return None, set(), 0


async def collect(target_id: str, seed_url: str) -> DimensionResult:
    base = f"{urlsplit(seed_url).scheme}://{urlsplit(seed_url).netloc}"
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": _UA}) as client:
        openapi_found_at, openapi_paths = await probe_openapi(client, base)
        graphql_found_at, graphql_types, graphql_field_count = await probe_graphql(client, base)

    return DimensionResult(
        method="api_spec",
        target_id=target_id,
        paths=openapi_paths,
        meta={
            "openapi_found_at": openapi_found_at,
            "openapi_endpoint_count": len(openapi_paths),
            "graphql_found_at": graphql_found_at,
            "graphql_type_count": len(graphql_types),
            "graphql_types": sorted(graphql_types),
            "graphql_field_count": graphql_field_count,
        },
    )
