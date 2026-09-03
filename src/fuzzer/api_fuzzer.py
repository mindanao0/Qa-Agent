"""AutonomousAPIFuzzer — intercepts network via Playwright route()."""
from __future__ import annotations

import asyncio
import json
import re
import time

import jsonschema
from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.fuzzer.anomaly_classifier import AnomalyClassifier
from src.fuzzer.anomaly_classifier import FuzzResult as ClassifiedResult
from src.fuzzer.fuzz_vectors import BASE_VECTORS
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.observability.tracer import OTelTracer

_FUZZER_SEMAPHORE = asyncio.Semaphore(1)

# Sprint 13 safety knobs (carry-forward rules).
_MIN_SPACING_S = 0.5          # ≤ 2 req/sec per endpoint
_BREAKER_CONSECUTIVE_5XX = 3  # pause endpoint after this many consecutive 5xx
_REQUEST_TIMEOUT_MS = 12_000

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


class FuzzRequest(BaseModel):
    """A prepared fuzz injection (Sprint 13): a concrete request carrying one vector."""

    model_config = ConfigDict(extra="forbid")

    vector: str
    method: str
    url: str
    body: dict | None = None


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

    async def fuzz_endpoint(
        self,
        endpoint_label: str,
        requests: list[FuzzRequest],
        page: Page,
        classifier: AnomalyClassifier,
        response_schema: dict | None = None,
    ) -> list[ClassifiedResult]:
        """Send prepared fuzz requests against a REAL endpoint and classify each (Sprint 13).

        Honors the carry-forward safety rules: ≥0.5s spacing (≤2 req/sec) and a
        circuit breaker that skips the endpoint's remaining vectors after
        ``_BREAKER_CONSECUTIVE_5XX`` consecutive 5xx (the spec's "pause 10 minutes"
        intent — a literal sleep is impractical in a measurement run, so we skip the
        remainder and let the caller record the pause). Wrapped in an OTel span.
        """
        results: list[ClassifiedResult] = []
        consecutive_5xx = 0
        last_ts = 0.0
        tripped = False

        async with self._tracer.span("fuzz.endpoint", endpoint=endpoint_label):
            for fr in requests:
                if consecutive_5xx >= _BREAKER_CONSECUTIVE_5XX:
                    tripped = True
                    logger.warning(
                        f"fuzz_endpoint: breaker tripped for {endpoint_label!r} after "
                        f"{consecutive_5xx} consecutive 5xx — skipping remaining vectors"
                    )
                    break

                elapsed = time.monotonic() - last_ts
                if last_ts and elapsed < _MIN_SPACING_S:
                    await asyncio.sleep(_MIN_SPACING_S - elapsed)

                status, body, duration_ms, timed_out = await self._send_one(page, fr)
                last_ts = time.monotonic()

                results.append(classifier.classify(
                    endpoint=endpoint_label,
                    vector=fr.vector,
                    status_code=status,
                    duration_ms=duration_ms,
                    body=body,
                    response_schema=response_schema,
                    timed_out=timed_out,
                ))

                consecutive_5xx = consecutive_5xx + 1 if status >= 500 else 0

        if tripped:
            logger.info(f"fuzz_endpoint: {endpoint_label!r} paused (breaker)")
        return results

    async def _send_one(
        self, page: Page, fr: FuzzRequest
    ) -> tuple[int, object, float, bool]:
        """Issue one real request via page.request → (status, body, duration_ms, timed_out)."""
        start = time.monotonic()
        try:
            method = fr.method.upper()
            if method == "POST":
                resp = await page.request.post(fr.url, data=fr.body or {}, timeout=_REQUEST_TIMEOUT_MS)
            elif method == "PUT":
                resp = await page.request.put(fr.url, data=fr.body or {}, timeout=_REQUEST_TIMEOUT_MS)
            else:
                resp = await page.request.get(fr.url, timeout=_REQUEST_TIMEOUT_MS)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.debug(f"_send_one: request did not complete {fr.url!r}: {exc!r}")
            return 0, None, duration_ms, True

        duration_ms = (time.monotonic() - start) * 1000
        body: object = None
        try:
            body = await resp.json()
        except Exception:
            try:
                body = await resp.text()
            except Exception:
                body = None
        return resp.status, body, duration_ms, False

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


__all__ = ["AutonomousAPIFuzzer", "FuzzRequest", "FuzzResult", "FuzzTarget"]
