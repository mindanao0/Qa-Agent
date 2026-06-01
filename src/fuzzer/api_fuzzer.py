"""AutonomousAPIFuzzer — intercepts network via Playwright route()."""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal

import jsonschema
from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.fuzzer.fuzz_vectors import BASE_VECTORS
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.observability.tracer import OTelTracer

_FUZZER_SEMAPHORE = asyncio.Semaphore(1)

BLOCKED_ACTION_PATTERNS = re.compile(
    r"delete|remove|transfer|payment|password", re.IGNORECASE
)

_KNOWN_STATIC_EXTENSIONS = {".js", ".css", ".png", ".ico", ".woff", ".woff2", ".svg", ".gif", ".jpg"}


class FuzzTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    method: str
    schema: dict
    fuzz_vectors: list[str]


class FuzzResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    vector: str
    status_code: int
    anomaly: bool
    anomaly_type: str | None
    duration_ms: float


class _FuzzVectors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fuzz_vectors: list[str]


class AutonomousAPIFuzzer:
    """No required constructor args."""

    def __init__(self, tracer: OTelTracer | None = None) -> None:
        self._tracer = tracer or OTelTracer()

    async def discover_endpoints(self, page: Page, url: str) -> list[FuzzTarget]:
        """
        1. page.goto(url) with request listener active
        2. Capture all fetch/XHR requests via page.on("request")
        3. Trigger known REST paths via page.evaluate
        4. Ollama (Semaphore(1), temp=0.1) → infer fuzz_vectors per endpoint
        5. BLOCKED_ACTION_PATTERNS: skip endpoints matching delete/payment/password
        6. max 10 endpoints, max 5 Ollama vectors each
        """
        captured: list[dict] = []

        def on_request(request) -> None:
            rtype = request.resource_type
            if rtype not in ("xhr", "fetch"):
                return
            req_url: str = request.url
            if BLOCKED_ACTION_PATTERNS.search(req_url):
                return
            if BLOCKED_ACTION_PATTERNS.search(request.method):
                return
            if any(req_url.endswith(ext) for ext in _KNOWN_STATIC_EXTENSIONS):
                return
            captured.append({"url": req_url, "method": request.method})

        page.on("request", on_request)

        try:
            await page.goto(url, wait_until="networkidle", timeout=15_000)
        except Exception as exc:
            logger.warning(f"discover_endpoints: navigation warning: {exc}")

        # Trigger known REST paths to surface discoverable endpoints
        for path in ["/todos", "/posts", "/users", "/albums", "/comments"]:
            try:
                await page.evaluate(f"fetch('{path}').then(r=>r.text())")
            except Exception:
                pass

        await asyncio.sleep(2)

        try:
            page.remove_listener("request", on_request)
        except Exception:
            pass

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[dict] = []
        for req in captured:
            if req["url"] not in seen:
                seen.add(req["url"])
                unique.append(req)

        targets: list[FuzzTarget] = []
        client = InstructorClient()
        try:
            for req in unique[:10]:
                vectors = await self._generate_vectors(client, req["url"])
                targets.append(
                    FuzzTarget(
                        endpoint=req["url"],
                        method=req["method"],
                        schema={"type": "array"},
                        fuzz_vectors=vectors[:5],
                    )
                )
        finally:
            await client.close()

        logger.info(f"discover_endpoints: found {len(targets)} endpoints")
        return targets

    async def fuzz(self, target: FuzzTarget, page: Page) -> list[FuzzResult]:
        """
        For each vector in [*BASE_VECTORS, *target.fuzz_vectors]:
          1. Construct fuzzed URL (path suffix or query param)
          2. page.evaluate fetch to trigger request and capture status + body
          3. jsonschema.validate body against target.schema
          4. anomaly = status not in [200,201,204] OR schema mismatch
        Wrapped in OTelTracer.span("api.fuzz", endpoint=target.endpoint).
        """
        results: list[FuzzResult] = []
        all_vectors = [*BASE_VECTORS, *target.fuzz_vectors]

        async with self._tracer.span("api.fuzz", endpoint=target.endpoint):
            for vector in all_vectors:
                result = await self._fuzz_one(target, page, vector)
                results.append(result)

        return results

    async def _fuzz_one(self, target: FuzzTarget, page: Page, vector: str) -> FuzzResult:
        import urllib.parse as _urlparse

        start_ms = time.time() * 1000
        status_code = 200
        anomaly = False
        anomaly_type: str | None = None

        safe_vector = vector.strip()

        # Path-suffix strategy for numeric/null/undefined vectors → triggers real 404s
        if re.match(r"^-?\d+$|^null$|^undefined$", safe_vector):
            fuzzed_url = target.endpoint.rstrip("/") + "/" + safe_vector
        else:
            sep = "&" if "?" in target.endpoint else "?"
            encoded = _urlparse.quote(safe_vector[:50], safe="")
            fuzzed_url = target.endpoint + sep + "fuzz=" + encoded

        try:
            js_result = await page.evaluate(
                f"""async () => {{
                    const resp = await fetch({json.dumps(fuzzed_url)});
                    const text = await resp.text();
                    return {{ status: resp.status, text: text }};
                }}"""
            )
            status_code = js_result.get("status", 200)

            if status_code not in (200, 201, 204):
                anomaly = True
                anomaly_type = "unexpected_status"
            else:
                raw = js_result.get("text", "")
                try:
                    body = json.loads(raw)
                    jsonschema.validate(body, target.schema)
                except jsonschema.ValidationError:
                    anomaly = True
                    anomaly_type = "schema_drift"
                except json.JSONDecodeError:
                    if raw.strip():
                        anomaly = True
                        anomaly_type = "schema_drift"
        except Exception:
            status_code = 408
            anomaly = True
            anomaly_type = "timeout"

        duration_ms = time.time() * 1000 - start_ms
        return FuzzResult(
            endpoint=target.endpoint,
            vector=vector[:100],
            status_code=status_code,
            anomaly=anomaly,
            anomaly_type=anomaly_type,
            duration_ms=round(duration_ms, 2),
        )

    async def _generate_vectors(self, client: InstructorClient, endpoint_url: str) -> list[str]:
        prompt = (
            f"REST endpoint URL: {endpoint_url}\n"
            "Generate exactly 5 short fuzz test strings for this endpoint. "
            "Include at least: an empty string, a SQL injection probe, an XSS probe, "
            "an integer boundary value, and a unicode string. "
            "Return JSON with field fuzz_vectors as a list of strings."
        )
        try:
            async with _FUZZER_SEMAPHORE:
                result = await client.create_structured(
                    prompt=prompt,
                    response_model=_FuzzVectors,
                    temperature=0.1,
                )
            return result.fuzz_vectors[:5]
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"_generate_vectors: Ollama failed for {endpoint_url}: {exc!r}")
            return []


__all__ = ["AutonomousAPIFuzzer", "FuzzResult", "FuzzTarget"]
