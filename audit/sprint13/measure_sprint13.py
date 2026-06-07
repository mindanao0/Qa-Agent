"""
Sprint 13 gate measurement — Advanced API Fuzzing on a REAL backend.

Flow (per spec):
  1. BackendProbe resolves a live Conduit mirror (realworld.habsida.net).
  2. SchemaInferrer.capture drives a real in-page fetch journey → TraceRecords
     (the backend is API-only, so "UI-driven traffic" = real fetch() that fires
     Playwright page.on("request") interception — real endpoints, nothing invented).
  3. SchemaInferrer.infer → InferredSchemas (genson structure + Ollama constraints).
  4. AdvancedVectorGenerator.generate_fields → per-field fuzz vectors.
  5. AutonomousAPIFuzzer.fuzz_endpoint sends real fuzzed requests (≤2 req/sec,
     circuit breaker) → AnomalyClassifier.classify each. OTel span per endpoint;
     CryptoAuditTrail per anomaly.

Grounded in live probes 2026-06-03 (see the reconciled design): the real anomaly
class on this backend is 5xx on malformed input (limit=-1/0/1.5, offset=-5, slug
traversal); SQLi/XSS that returns a safe 200 is NOT an anomaly (classifier demotes
it — flagging it would be a false positive). Read-focused scope (user-chosen): GET
query/path fuzzing + /login body fuzzing (no DB writes); register once for discovery.

Gates
-----
  endpoints_discovered    >= 5
  schema_inferred         >= 3
  fuzz_vectors_generated  >= 20
  anomalies_found         >= 3
  false_positive_rate     <= 0.20
  otel_spans_emitted      >= 10
  regression              == False  (True iff pass_rate < 0.75)
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import sys
import urllib.parse
import uuid

from loguru import logger
from playwright.async_api import async_playwright

from src.fuzzer.anomaly_classifier import AnomalyClassifier, AnomalyType
from src.fuzzer.api_fuzzer import AutonomousAPIFuzzer, FuzzRequest
from src.fuzzer.schema_inferrer import InferredSchema, SchemaInferrer
from src.fuzzer.vector_generator import AdvancedVectorGenerator, base_vectors_for
from src.observability.audit_chain import CryptoAuditTrail
from src.observability.tracer import OTelTracer
from src.race.backend_probe import BackendProbe, BackendProbeResult

_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint13_results.json"
_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]

_CANDIDATES = [
    "https://conduit.realworld.how",
    "https://realworld.habsida.net",
    "https://node-express-conduit.appspot.com",
]
_FALLBACK = "https://jsonplaceholder.typicode.com"

# Gates
_GATE_ENDPOINTS = 5
_GATE_SCHEMAS = 3
_GATE_VECTORS = 20
_GATE_ANOMALIES = 3
_GATE_FP_RATE = 0.20
_GATE_OTEL = 10
_REGRESSION_THRESHOLD = 0.75
_MAX_VECTORS = 5

# Conduit-aware fuzz targets: (endpoint, method, field, type, location). Only
# targets whose (endpoint, method) was actually DISCOVERED are fuzzed — we never
# invent an endpoint. location ∈ {"query", "path", "login_email"}.
_TARGETS = [
    ("/api/articles", "GET", "limit", "integer", "query"),
    ("/api/articles", "GET", "offset", "integer", "query"),
    ("/api/articles", "GET", "tag", "string", "query"),
    ("/api/articles/{slug}", "GET", "slug", "slug", "path"),
    ("/api/users/login", "POST", "email", "email", "login_email"),
]


def _q(v: str) -> str:
    return urllib.parse.quote(v, safe="")


def _inject_query(base: str, endpoint: str, field: str, vector: str) -> FuzzRequest:
    return FuzzRequest(vector=vector, method="GET",
                       url=f"{base}{endpoint}?{field}={_q(vector)}", body=None)


def _inject_path(base: str, endpoint: str, field: str, vector: str) -> FuzzRequest:
    m = re.search(r"\{(\w+)\}", endpoint)
    placeholder = m.group(0) if m else "{slug}"
    url = base + endpoint.replace(placeholder, _q(vector))  # str.replace → no regex escapes
    return FuzzRequest(vector=vector, method="GET", url=url, body=None)


def _inject_login_email(base: str, endpoint: str, field: str, vector: str) -> FuzzRequest:
    # Fuzz user.email; password fixed (NOT fuzzed). Garbage email → 401/422, no account.
    return FuzzRequest(vector=vector, method="POST", url=f"{base}{endpoint}",
                       body={"user": {"email": vector, "password": "Passw0rd123"}})


_INJECTORS = {"query": _inject_query, "path": _inject_path, "login_email": _inject_login_email}


def _full(schema: dict) -> bool:
    return bool(schema.get("properties"))


async def _resolve_backend(probe: BackendProbe, browser, tracer) -> tuple[str, BackendProbeResult]:
    async with tracer.span("backend.resolve"):
        for url in _CANDIDATES:
            ctx = await browser.new_context()
            page = await ctx.new_page()
            try:
                async with tracer.span("backend.probe", url=url):
                    res = await probe.probe(url, page)
                if res.has_real_backend:
                    return url, res
            finally:
                await ctx.close()
    return _FALLBACK, BackendProbeResult(
        url=_FALLBACK, has_real_backend=False, api_endpoints=[],
        probe_note="all conduit candidates unreachable; using mock fallback",
    )


def _objective_anomaly(status: int, atype: AnomalyType | None) -> bool:
    """Status-grounded ground truth (independent of injection/effect heuristics)."""
    if status >= 500:
        return True
    if atype == AnomalyType.TIMEOUT:
        return True
    if status in (400, 401, 403, 404, 422):
        return False
    if 200 <= status < 300:
        return atype == AnomalyType.SCHEMA_DRIFT  # a 2xx is only anomalous if drift
    return False


async def main() -> dict:
    endpoints_discovered = 0
    schema_inferred = 0
    fuzz_vectors_generated = 0
    anomalies_found = 0
    false_positive_rate = 0.0
    otel_spans_emitted = 0
    pass_rate = 0.0
    regression = True
    backend_used = _FALLBACK
    real_backend_confirmed = False
    breaker_trips = 0

    tracer = OTelTracer()
    audit = CryptoAuditTrail(
        path=pathlib.Path(__file__).parent / f"sprint13_audit_{uuid.uuid4().hex[:8]}.jsonl"
    )
    probe = BackendProbe()
    inferrer = SchemaInferrer()
    generator = AdvancedVectorGenerator()
    classifier = AnomalyClassifier()
    fuzzer = AutonomousAPIFuzzer(tracer=tracer)
    sem = asyncio.Semaphore(1)

    all_results = []
    anomaly_breakdown: dict[str, int] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        try:
            async with tracer.span("sprint13.run"):
                backend_used, probe_res = await _resolve_backend(probe, browser, tracer)
                real_backend_confirmed = probe_res.has_real_backend
                audit.append("backend_probe", probe_res.model_dump())
                logger.info(f"measure_sprint13: backend={backend_used!r} real={real_backend_confirmed}")

                base = backend_used.rstrip("/")
                ctx = await browser.new_context()
                page = await ctx.new_page()
                try:
                    async with tracer.span("schema.capture"):
                        traces = await inferrer.capture(page, backend_used)
                    audit.append("capture", {"traces": len(traces)})

                    async with tracer.span("schema.infer"):
                        schemas = await inferrer.infer(traces, sem)

                    schemas_by_key: dict[tuple[str, str], InferredSchema] = {
                        (s.endpoint, s.method): s for s in schemas
                    }
                    endpoints_discovered = len(schemas_by_key)
                    schema_inferred = sum(
                        1 for s in schemas if _full(s.response_schema) and _full(s.request_schema)
                    )
                    logger.info(
                        f"measure_sprint13: endpoints={endpoints_discovered} "
                        f"schema_inferred={schema_inferred} "
                        f"endpoints={sorted(f'{m} {e}' for (e, m) in schemas_by_key)}"
                    )
                    audit.append("infer", {
                        "endpoints_discovered": endpoints_discovered,
                        "schema_inferred": schema_inferred,
                        "endpoints": sorted(f"{m} {e}" for (e, m) in schemas_by_key),
                    })

                    # Per-schema field vectors (spec generate path; OTel span each).
                    field_vectors: dict[tuple[str, str], dict[str, list[str]]] = {}
                    for key, s in schemas_by_key.items():
                        async with tracer.span("vector.generate", endpoint=s.endpoint, method=s.method):
                            field_vectors[key] = await generator.generate_fields(s, sem, _MAX_VECTORS)

                    # Build injection requests for DISCOVERED targets only.
                    per_endpoint: dict[tuple[str, str], list[FuzzRequest]] = {}
                    for endpoint, method, field, ftype, location in _TARGETS:
                        key = (endpoint, method)
                        if key not in schemas_by_key:
                            logger.info(f"target {method} {endpoint} not discovered — skipped")
                            continue
                        if location == "query":
                            vectors = field_vectors.get(key, {}).get(field) or base_vectors_for(ftype, _MAX_VECTORS)
                        elif location == "path":
                            vectors = (field_vectors.get(key, {}).get(field)
                                       or base_vectors_for(ftype, _MAX_VECTORS)) + ["../../../etc/passwd"]
                        else:  # login_email
                            vectors = base_vectors_for("email", _MAX_VECTORS) + base_vectors_for("string", 3)
                        injector = _INJECTORS[location]
                        reqs = [injector(base, endpoint, field, v) for v in vectors]
                        per_endpoint.setdefault(key, []).extend(reqs)
                        fuzz_vectors_generated += len(reqs)

                    logger.info(f"measure_sprint13: fuzz_vectors_generated={fuzz_vectors_generated}")

                    # Fuzz each endpoint (its own breaker + OTel span).
                    for (endpoint, method), reqs in per_endpoint.items():
                        resp_schema = schemas_by_key[(endpoint, method)].response_schema or None
                        before = len(all_results)
                        results = await fuzzer.fuzz_endpoint(
                            f"{method} {endpoint}", reqs, page, classifier, resp_schema
                        )
                        all_results.extend(results)
                        if len(results) < len(reqs):
                            breaker_trips += 1
                        for r in results:
                            if r.anomaly and not r.false_positive:
                                anomaly_breakdown[r.anomaly_type.value if r.anomaly_type else "?"] = \
                                    anomaly_breakdown.get(
                                        r.anomaly_type.value if r.anomaly_type else "?", 0) + 1
                                audit.append("fuzz_result", {
                                    "endpoint": r.endpoint, "vector": r.vector,
                                    "status_code": r.status_code,
                                    "anomaly_type": r.anomaly_type.value if r.anomaly_type else None,
                                    "duration_ms": r.duration_ms,
                                })
                        logger.info(
                            f"fuzz {method} {endpoint}: {len(results)}/{len(reqs)} sent, "
                            f"anomalies={sum(1 for r in results if r.anomaly and not r.false_positive)}"
                        )
                finally:
                    await ctx.close()

                flagged = [r for r in all_results if r.anomaly]
                false_positives = [r for r in flagged if r.false_positive]
                anomalies_found = len(flagged) - len(false_positives)
                false_positive_rate = len(false_positives) / max(1, len(flagged))

                if all_results:
                    correct = sum(
                        1 for r in all_results
                        if r.anomaly == _objective_anomaly(r.status_code, r.anomaly_type)
                    )
                    pass_rate = correct / len(all_results)
                regression = pass_rate < _REGRESSION_THRESHOLD
        finally:
            await browser.close()

    otel_spans_emitted = tracer.flush()
    expected_4xx = sum(1 for r in all_results if r.anomaly_type == AnomalyType.EXPECTED_4XX)
    safe_2xx = sum(
        1 for r in all_results
        if 200 <= r.status_code < 300 and not r.anomaly
    )
    audit.append("sprint13.complete", {
        "backend_used": backend_used,
        "anomalies_found": anomalies_found,
        "otel_spans_emitted": otel_spans_emitted,
    })

    sprint13_pass = (
        endpoints_discovered >= _GATE_ENDPOINTS
        and schema_inferred >= _GATE_SCHEMAS
        and fuzz_vectors_generated >= _GATE_VECTORS
        and anomalies_found >= _GATE_ANOMALIES
        and false_positive_rate <= _GATE_FP_RATE
        and otel_spans_emitted >= _GATE_OTEL
        and not regression
    )

    results_json = {
        "endpoints_discovered": endpoints_discovered,
        "schema_inferred": schema_inferred,
        "fuzz_vectors_generated": fuzz_vectors_generated,
        "anomalies_found": anomalies_found,
        "false_positive_rate": round(false_positive_rate, 6),
        "otel_spans_emitted": otel_spans_emitted,
        "regression": regression,
        "sprint13_status": "PASS" if sprint13_pass else "FAIL",
        # ── transparency extras (not gated) ──
        "backend_used": backend_used,
        "real_backend_confirmed": real_backend_confirmed,
        "pass_rate": round(pass_rate, 6),
        "total_fuzz_requests": len(all_results),
        "anomaly_breakdown": anomaly_breakdown,
        "expected_4xx_demoted": expected_4xx,
        "safe_2xx_not_flagged": safe_2xx,
        "breaker_trips": breaker_trips,
        "audit_trail_entries": audit._seq,
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    print("\n=== Sprint 13 Results ===")
    gates = {
        "endpoints_discovered": (_GATE_ENDPOINTS, ">="),
        "schema_inferred": (_GATE_SCHEMAS, ">="),
        "fuzz_vectors_generated": (_GATE_VECTORS, ">="),
        "anomalies_found": (_GATE_ANOMALIES, ">="),
        "false_positive_rate": (_GATE_FP_RATE, "<="),
        "otel_spans_emitted": (_GATE_OTEL, ">="),
    }
    for k, v in results_json.items():
        if k in gates:
            gate, op = gates[k]
            ok = (v >= gate) if op == ">=" else (v <= gate)
            print(f"  {k}: {v} (gate {op} {gate}){' ✓' if ok else ' ✗'}")
        elif k == "regression":
            print(f"  {k}: {v} (gate == False){' ✓' if not v else ' ✗'}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  -> sprint13_status: {results_json['sprint13_status']}")
    print(f"  -> written to {_OUTPUT_PATH}")
    return results_json


if __name__ == "__main__":
    res = asyncio.run(main())
    sys.exit(0 if res.get("sprint13_status") == "PASS" else 1)
