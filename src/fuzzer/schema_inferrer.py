"""SchemaInferrer — Sprint 13.

Infers OpenAPI-compatible schema from REAL Playwright-intercepted traffic.

``realworld.habsida.net`` is API-only (no SPA) and exposes no OpenAPI document
(``GET /api`` → 500), so "UI-driven traffic" is realized as a real in-page
``fetch()`` journey (register → login → browse → view article → authed read).
Those fetches fire ``page.on("request")`` interception — real endpoints observed
on the wire, nothing invented. genson builds the JSON Schemas deterministically
from captured samples; Ollama only adds NL constraints (best-effort).

See docs/superpowers/specs/2026-06-03-sprint13-advanced-api-fuzzing-design.md.
"""
from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import Any

from genson import SchemaBuilder
from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_REDACT = "__REDACTED__"
_SENSITIVE_KEYS = {"password", "token", "authorization"}
_COVERAGE_STABLE = 3  # is_candidate flips to False at >= this many distinct shapes

# Path-parameter normalization rules (concrete id → templated param).
_NORMALIZE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(/api/articles)/[^/]+$"), r"\1/{slug}"),
    (re.compile(r"^(/api/profiles)/[^/]+$"), r"\1/{username}"),
]


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    ui_action: str
    endpoint_hint: str  # normalized: /api/articles/{slug}
    method: str
    request_body: dict | None
    response_status: int
    response_body: dict | None
    auth_present: bool
    captured_at: float


class InferredSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    method: str
    request_schema: dict
    response_schema: dict
    constraints: list[str]
    coverage_score: int
    is_candidate: bool = True


class _Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraints: list[str]


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive values (password / token / authorization)."""
    if isinstance(obj, dict):
        return {
            k: (_REDACT if k.lower() in _SENSITIVE_KEYS else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def normalize_endpoint(url: str) -> str:
    """Strip query string and template dynamic path segments."""
    path = urllib.parse.urlsplit(url).path
    for pattern, repl in _NORMALIZE_RULES:
        new = pattern.sub(repl, path)
        if new != path:
            return new
    return path


def _query_dict(url: str) -> dict | None:
    """Parsed query params of a GET as its 'request body' representation."""
    q = urllib.parse.urlsplit(url).query
    if not q:
        return None
    return {k: v[0] if len(v) == 1 else v for k, v in urllib.parse.parse_qs(q).items()}


def _full_schema(schema: dict) -> bool:
    return bool(schema.get("properties"))


# Journey executed in-page: every fetch fires page.on("request") interception.
_JOURNEY_JS = """
async (cfg) => {
  const b = cfg.base, hdr = {'Content-Type': 'application/json'};
  let token = null;
  try {
    const reg = await fetch(b + '/api/users', {method: 'POST', headers: hdr,
      body: JSON.stringify({user: {username: cfg.username, email: cfg.email, password: cfg.password}})});
    const rj = await reg.json(); token = (rj && rj.user) ? rj.user.token : null;
  } catch (e) {}
  try {
    await fetch(b + '/api/users/login', {method: 'POST', headers: hdr,
      body: JSON.stringify({user: {email: cfg.email, password: cfg.password}})});
  } catch (e) {}
  try { await fetch(b + '/api/articles?limit=1'); } catch (e) {}
  let slug = null;
  try {
    const a = await (await fetch(b + '/api/articles?limit=5')).json();
    slug = (a && a.articles && a.articles[0]) ? a.articles[0].slug : null;
  } catch (e) {}
  try { await fetch(b + '/api/tags'); } catch (e) {}
  if (slug) { try { await fetch(b + '/api/articles/' + encodeURIComponent(slug)); } catch (e) {} }
  try { await fetch(b + '/api/articles?tag=Technology'); } catch (e) {}
  try { await fetch(b + '/api/articles?offset=0&limit=2'); } catch (e) {}
  if (token) {
    try { await fetch(b + '/api/user', {headers: {'Authorization': 'Token ' + token}}); } catch (e) {}
    try { await fetch(b + '/api/profiles/' + cfg.username, {headers: {'Authorization': 'Token ' + token}}); } catch (e) {}
  }
  return true;
}
"""


class SchemaInferrer:
    """No required constructor args."""

    async def capture(self, page: Page, url: str) -> list[TraceRecord]:
        import time
        import uuid

        base = url.rstrip("/")
        captured: list[Any] = []

        def _on_request(req: Any) -> None:
            if "/api/" in req.url:
                captured.append(req)

        page.on("request", _on_request)

        # Establish same-origin context (root 500s; /api/tags is a real 200).
        try:
            await page.goto(base + "/api/tags", timeout=15_000)
        except Exception as exc:
            logger.debug(f"SchemaInferrer.capture: goto warning: {exc!r}")

        username = f"race_{uuid.uuid4().hex[:10]}"
        cfg = {
            "base": base,
            "username": username,
            "email": f"{username}@example.com",
            "password": "Passw0rd123",
        }
        try:
            await page.evaluate(_JOURNEY_JS, cfg)
        except Exception as exc:
            logger.warning(f"SchemaInferrer.capture: journey warning: {exc!r}")

        await asyncio.sleep(1.0)  # let in-flight responses settle
        try:
            page.remove_listener("request", _on_request)
        except Exception:
            pass

        traces: list[TraceRecord] = []
        for req in captured:
            try:
                method = req.method
                endpoint = normalize_endpoint(req.url)
                headers = req.headers  # Playwright lowercases header names
                auth_present = "authorization" in {k.lower() for k in headers}

                if method == "GET":
                    request_body = _query_dict(req.url)
                else:
                    try:
                        request_body = req.post_data_json
                    except Exception:
                        request_body = None
                request_body = _redact(request_body) if isinstance(request_body, dict) else None

                resp = await req.response()
                status = resp.status if resp else 0
                response_body: dict | None = None
                if resp is not None:
                    try:
                        parsed = await resp.json()
                        if isinstance(parsed, dict):
                            response_body = _redact(parsed)
                    except Exception:
                        response_body = None

                traces.append(TraceRecord(
                    trace_id=uuid.uuid4().hex[:12],
                    ui_action=f"{method.lower()}_{endpoint.strip('/').replace('/', '_')}",
                    endpoint_hint=endpoint,
                    method=method,
                    request_body=request_body,
                    response_status=status,
                    response_body=response_body,
                    auth_present=auth_present,
                    captured_at=time.time(),
                ))
            except Exception as exc:
                logger.debug(f"SchemaInferrer.capture: trace skipped: {exc!r}")

        logger.info(f"SchemaInferrer.capture: {len(traces)} traces from {url!r}")
        return traces

    async def infer(
        self,
        traces: list[TraceRecord],
        ollama_semaphore: asyncio.Semaphore,
    ) -> list[InferredSchema]:
        groups: dict[tuple[str, str], list[TraceRecord]] = {}
        for t in traces:
            groups.setdefault((t.endpoint_hint, t.method), []).append(t)

        schemas: list[InferredSchema] = []
        client = InstructorClient()
        try:
            for (endpoint, method), recs in groups.items():
                req_builder, resp_builder = SchemaBuilder(), SchemaBuilder()
                req_shapes: set[str] = set()
                for r in recs:
                    if isinstance(r.request_body, dict):
                        req_builder.add_object(r.request_body)
                        req_shapes.add(",".join(sorted(r.request_body.keys())))
                    if r.response_status in (200, 201) and isinstance(r.response_body, dict):
                        resp_builder.add_object(r.response_body)

                request_schema = req_builder.to_schema() if req_shapes else {}
                # Path params ARE the request input for /api/articles/{slug}-style
                # endpoints — synthesize a request schema so they're fuzzable + counted.
                path_params = re.findall(r"\{(\w+)\}", endpoint)
                if not req_shapes and path_params:
                    request_schema = {
                        "type": "object",
                        "properties": {p: {"type": "string"} for p in path_params},
                    }
                response_schema = (
                    resp_builder.to_schema()
                    if any(r.response_status in (200, 201) and r.response_body for r in recs)
                    else {}
                )
                coverage_score = max(len(req_shapes), len(recs))

                constraints: list[str] = []
                if _full_schema(request_schema) or _full_schema(response_schema):
                    constraints = await self._infer_constraints(
                        client, endpoint, method, request_schema, ollama_semaphore
                    )

                schemas.append(InferredSchema(
                    endpoint=endpoint,
                    method=method,
                    request_schema=request_schema,
                    response_schema=response_schema,
                    constraints=constraints,
                    coverage_score=coverage_score,
                    is_candidate=coverage_score < _COVERAGE_STABLE,
                ))
        finally:
            await client.close()

        logger.info(f"SchemaInferrer.infer: {len(schemas)} schemas inferred")
        return schemas

    async def _infer_constraints(
        self,
        client: InstructorClient,
        endpoint: str,
        method: str,
        request_schema: dict,
        ollama_semaphore: asyncio.Semaphore,
    ) -> list[str]:
        prompt = (
            f"API endpoint: {method} {endpoint}\n"
            f"Request JSON Schema: {request_schema}\n"
            "List up to 5 short input validation constraint rules a fuzzer should probe "
            "(e.g. 'username min 3 chars', 'limit must be a positive integer'). "
            "Return JSON with field constraints as a list of short strings."
        )
        try:
            async with ollama_semaphore:
                result = await client.create_structured(
                    prompt=prompt, response_model=_Constraints, temperature=0.1
                )
            return result.constraints[:5]
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"_infer_constraints: Ollama failed for {endpoint}: {exc!r}")
            return []


__all__ = ["InferredSchema", "SchemaInferrer", "TraceRecord", "normalize_endpoint"]
