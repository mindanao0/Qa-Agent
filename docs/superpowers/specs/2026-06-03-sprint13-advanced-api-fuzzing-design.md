# Sprint 13 — Advanced API Fuzzing (Reconciled Design)

Date: 2026-06-03
Spec: `docs/specs/Sprint 13 Advanced API Fuzzing.md`
Baseline: Sprint 12 PASS (BackendProbe + real backend `realworld.habsida.net`).

## Honest reconciliations (grounded in LIVE probes 2026-06-03, not assumptions)

Probing `https://realworld.habsida.net` disproved several spec assumptions. Intent
honored; no MUST-NOT broken; no gate fudged.

1. **No OpenAPI spec.** `GET /api` → **500** (`{"errors":{"body":["Internal server error"]}}`),
   not a spec doc. Schema is inferred from captured traffic — the sprint's real objective.
2. **Backend is API-only (no SPA).** `GET /` → 500 JSON; no frontend bundle. "UI-driven
   traffic capture" is realized as real **in-page `fetch()`** of the documented Conduit
   journey, which fires Playwright `page.on("request"/"response")` interception. Real
   endpoints observed on the wire; nothing invented. (Same class of reconciliation as
   Sprint 12's API-only handling.)
3. **Spec's "200 on SQLi → INJECTION_SIGNAL" is a false-positive trap here.** Probed:
   `tag=' OR '1'='1'` and `tag=<script>alert(1)</script>` → **200 + articlesCount:0**
   (safely handled). The classifier treats safe-2xx-on-injection as **NOT** an anomaly;
   a 200-on-injection is only flagged if the response shows actual *effect* (SQL error
   leak or row-count divergence from baseline). On this backend that never fires → honest.
4. **The real anomaly class is 5xx on malformed input.** Confirmed reproducible 500s:
   `limit=-1`, `limit=0`, `limit=1.5`, `offset=-5`, and slug `../../../etc/passwd`
   (path traversal). These are genuine robustness/security defects (server should 4xx,
   not 5xx). Expected behaviors — `422` on `limit=abc`, `404` on nonexistent slug, `401`
   unauth — are classified as **non-anomalies** (correct backend behavior).
5. **Circuit breaker** "pause endpoint 10 min after ≥3 consecutive 5xx" → implemented as
   *skip the endpoint's remaining vectors for this run + record the pause event*. A literal
   600 s sleep in a measurement run is impractical; the protective intent (stop hammering a
   failing endpoint) is preserved and reported transparently.
6. **Read-focused fuzzing (user-chosen 2026-06-03).** Fuzz GET endpoints (query + path
   injection) + POST `/api/users/login` body (fails 401/422 safely → NO accounts created).
   `register` is called **once** for discovery + an auth token. ~1–2 throwaway `race_`
   accounts total — within the Sprint 12 authorization. Genuine anomalies come from the
   GET 5xx defects, so the `anomalies≥3` gate is met without DB pollution.

## Confirmed backend behaviors (oracle for the classifier & pass_rate)

| Request | Status | Truth |
|---|---|---|
| `GET /api/tags` | 200 (real tag list) | ok |
| `GET /api/articles?limit=5` | 200 (real articles) | ok |
| `GET /api/articles/{real-slug}` | 200 | ok |
| `GET /api/articles/{nonexistent}` | 404 | expected_4xx |
| `GET /api/articles?limit=-1 / 0 / 1.5` | **500** | anomaly |
| `GET /api/articles?offset=-5` | **500** | anomaly |
| `GET /api/articles?limit=99999999` | 200 (all 143) | ok (safe) |
| `GET /api/articles?limit=abc` | 422 | expected_4xx |
| `GET /api/articles?tag=<sqli/xss>` | 200, count:0 | ok (safe) |
| `GET /api/articles/../../../etc/passwd` | **500** | anomaly (injection) |
| `POST /api/users` (fresh user, even invalid email) | 200 + JWT | ok (lax validation) |
| `POST /api/users` (duplicate username) | 422 | expected (Sprint 12) |
| `POST /api/articles` (unauth) | 401 | expected_4xx |
| `GET /api/profiles/{x}` | 401 (auth required) | expected_4xx |

## Components

### `src/fuzzer/schema_inferrer.py` (NEW)
- `TraceRecord`, `InferredSchema` (Pydantic V2, `extra="forbid"`, fields per spec).
- `SchemaInferrer.capture(page, url)`: `page.on("request")` collects `/api/` requests; one
  `page.evaluate` async journey (register → login → list articles → tags → single article
  by slug → `?tag=` filter → auth `GET /api/user` → `GET /api/profiles/{u}`). After the
  journey, for each captured request: `await req.response()` for status/body, `req.post_data_json`
  for request body. **Redaction**: Authorization header value, `password`, and `token`
  fields → `"__REDACTED__"`. Path normalization: `/api/articles/<slug>` → `/api/articles/{slug}`,
  `/api/profiles/<u>` → `/api/profiles/{username}`; query string stripped from `endpoint_hint`.
- `SchemaInferrer.infer(traces, semaphore)`: group by `(endpoint, method)`; **genson**
  `SchemaBuilder` builds `request_schema` (POST body, or synthesized query-param schema for
  GET) + `response_schema` from 200 bodies (deterministic). **Ollama** (Semaphore(1), temp 0.1)
  fills `constraints` only — best-effort, degrades to `[]` on failure so gates never depend on
  the 7B model. `coverage_score` = distinct request shapes; `is_candidate = coverage_score < 3`.

### `src/fuzzer/vector_generator.py` (NEW)
- `BASE_VECTORS_BY_TYPE` exactly as spec (string/integer/email/slug). `AdvancedVectorGenerator.generate(schema, semaphore, max_vectors=5)`: identify fuzzable fields, pick base by type,
  Ollama-augment from `constraints` (best-effort), return ≤`max_vectors` unique per field.
  `BLOCKED_ACTION_PATTERNS` (delete|remove|transfer|payment|password) enforced — such vectors
  are never emitted, and the `password` field is never fuzzed.

### `src/fuzzer/anomaly_classifier.py` (NEW)
- `AnomalyType(str, Enum)` + new richer `FuzzResult` (`anomaly`, `anomaly_type`, `false_positive`,
  `duration_ms`, per spec). `AnomalyClassifier.classify(endpoint, vector, status, body,
  duration_ms, response_schema, is_injection)`:
  - `duration_ms > 10000` → `TIMEOUT`, anomaly, fp=False.
  - `status >= 500` → `INJECTION_SIGNAL` if `is_injection` else `UNEXPECTED_STATUS`; anomaly, fp=False.
  - `status in {400,401,403,404,422}` → `EXPECTED_4XX`, anomaly=False, fp=False.
  - `2xx` missing a `response_schema.required` field → `SCHEMA_DRIFT`, anomaly, fp=False.
  - `2xx` on injection with effect signal (SQL-error string / row-count divergence) →
    `INJECTION_SIGNAL`, anomaly; otherwise anomaly=False.
  - `false_positive=True` only if an anomaly matches a documented expected-behavior predicate
    (≈never here; reported transparently).

### `audit/sprint13/measure_sprint13.py`
BackendProbe (reuse) → capture → infer → generate → fuzz+classify → OTel + audit → results JSON.
Fuzz execution = a Conduit-aware injection method on `AutonomousAPIFuzzer` (query/path/body),
≥500 ms per-endpoint spacing (≤2 req/sec), per-endpoint consecutive-5xx breaker. Sprint 10's
`discover_endpoints`/`fuzz`/`FuzzResult` untouched (additive).

## Honest metric → gate mapping
- `endpoints_discovered` = distinct `(method, normalized-path)` in traces (journey → ~7; gate ≥5).
- `schema_inferred` = endpoints with a non-empty genson `response_schema` AND a request
  representation (POST body or GET query-param schema) (gate ≥3).
- `fuzz_vectors_generated` = total across fuzzable fields {limit,offset,tag,slug,email} (≥24; gate ≥20).
- `anomalies_found` = `anomaly=True AND false_positive=False` (≥5 genuine 500s available; gate ≥3).
- `false_positive_rate` = flagged anomalies matching an expected-behavior predicate ÷ flagged
  (grounded classifier → ~0.0; gate ≤0.20; NOT gamed via vector selection).
- `otel_spans_emitted` ≥10 (run / resolve / probe / capture / infer / generate×N / fuzz.endpoint×N).
- `pass_rate` = classifier accuracy vs an objective label (`status≥500 or timeout ⇒ anomaly`;
  `4xx ⇒ expected`; `2xx ⇒ ok unless genson-drift`); `regression = pass_rate < 0.75`.

## Safety rules (carry-forward, enforced)
Semaphore(1) on all Ollama calls; pathlib everywhere; Pydantic V2 `extra="forbid"`; CDP-only
(no `page.accessibility`); BLOCKED_ACTION_PATTERNS in generation; never log Authorization/
password/token; ≤2 req/sec per endpoint; pause endpoint on ≥3 consecutive 5xx; `is_candidate`
until `coverage_score ≥ 3`; OTel + CryptoAuditTrail per anomaly; BFT stays disabled.

## Testing
`tests/fuzzer/test_schema_inferrer.py` (validation, normalization, redaction, genson build,
coverage/is_candidate), `test_vector_generator.py` (type selection, uniqueness, BLOCKED),
`test_anomaly_classifier.py` (grounded rules — most important). Existing `test_api_fuzzer.py`
/ `test_fuzz_vectors.py` stay green. Dep: `genson` (added).
