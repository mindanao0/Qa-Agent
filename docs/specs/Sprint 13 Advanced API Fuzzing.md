## CONTEXT
PROJECT: D:\Code\qa-agent
BASELINE: Sprint 12 PASS — InterleavingRecorder working,
          BackendProbe confirmed real backend detection
TARGET: https://realworld.habsida.net  (same backend, Conduit REST API)
        OpenAPI spec available at: https://realworld.habsida.net/api
STACK: Python 3.13 + uv, Ollama, LangGraph async, Playwright CDP

## OBJECTIVE
GOAL: Agent intercepts real UI-driven API traffic → infers OpenAPI schema →
      generates targeted fuzz vectors per endpoint →
      detects schema drift, injection vulnerabilities, boundary violations

## ACCEPTANCE GATE (sprint13)
  endpoints_discovered        ≥ 5     (from Playwright network interception)
  schema_inferred             ≥ 3     (endpoints with full request+response schema)
  fuzz_vectors_generated      ≥ 20    (total across all endpoints)
  anomalies_found             ≥ 3     (real: 4xx/5xx unexpected, schema drift, timeout)
  false_positive_rate         ≤ 0.20  (anomalies that are expected behavior)
  otel_spans_emitted          ≥ 10
  REGRESSION if pass_rate     < 0.75

## ARCHITECTURE — New files:

### src/fuzzer/schema_inferrer.py  (NEW — replaces naive Sprint 10 approach)
# SchemaInferrer — infers OpenAPI-compatible schema from captured Playwright traffic
#
# class TraceRecord(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     trace_id:         str
#     ui_action:        str        # Playwright step name e.g. "click_login"
#     endpoint_hint:    str        # normalized: /api/articles/{id}
#     method:           str
#     request_body:     dict | None
#     response_status:  int
#     response_body:    dict | None
#     auth_present:     bool       # Authorization header detected
#     captured_at:      float
#
# class InferredSchema(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     endpoint:         str
#     method:           str
#     request_schema:   dict       # JSON Schema of request body
#     response_schema:  dict       # JSON Schema of 200 response
#     constraints:      list[str]  # LLM-extracted rules e.g. "username min 3 chars"
#     coverage_score:   int        # count of distinct request shapes seen
#     is_candidate:     bool = True  # always True until stabilized (≥3 samples)
#
# class SchemaInferrer:
#     """No required constructor args."""
#
#     async def capture(self, page: Page, url: str) -> list[TraceRecord]:
#         """
#         1. Attach page.on("request") + page.on("response") listeners
#         2. page.goto(url) → interact with UI to trigger API calls:
#            - register new user (race_{uuid4().hex[:8]})
#            - browse articles (GET /api/articles, /api/tags)
#            - view single article (GET /api/articles/{slug})
#         3. Collect TraceRecords
#         4. Normalize path params: /api/articles/my-slug → /api/articles/{slug}
#         5. NEVER capture Authorization header value (redact to "__REDACTED__")
#         6. NEVER capture password fields (redact to "__REDACTED__")
#         """
#
#     async def infer(
#         self,
#         traces: list[TraceRecord],
#         ollama_semaphore: asyncio.Semaphore,
#     ) -> list[InferredSchema]:
#         """
#         Group traces by (endpoint, method)
#         For each group with ≥1 trace:
#           → Ollama (Semaphore(1), temp=0.1, format="json") infer:
#             - request_schema (JSON Schema)
#             - response_schema
#             - constraints list
#         Mark is_candidate=False only when coverage_score ≥ 3
#         """

### src/fuzzer/vector_generator.py  (NEW — replaces FuzzVectorLibrary Sprint 10)
# AdvancedVectorGenerator — combines deterministic base + LLM augmentation
#
# BASE_VECTORS_BY_TYPE = {
#     "string": [
#         "", " " * 500, "null", "<script>alert(1)</script>",
#         "' OR '1'='1", "../../../etc/passwd", "𝕳𝖊𝖑𝖑𝖔",
#         "a" * 256,   # max length probe
#         "a" * 1,     # min length probe
#     ],
#     "integer": ["0", "-1", "99999999", "1.5", "null", ""],
#     "email":   ["notanemail", "@nodomain", "a@b", "test+tag@example.com"],
#     "slug":    ["", "-", "a" * 100, "UPPERCASE", "has space", "../traverse"],
# }
#
# class AdvancedVectorGenerator:
#     """No required constructor args."""
#
#     async def generate(
#         self,
#         schema: InferredSchema,
#         ollama_semaphore: asyncio.Semaphore,
#         max_vectors: int = 5,
#     ) -> list[str]:
#         """
#         1. Identify fuzzable fields from schema.request_schema
#         2. Select BASE_VECTORS_BY_TYPE by field type
#         3. Ollama (Semaphore(1)) augment with schema.constraints
#            e.g. if constraint="username min 3 chars" → add 1-char vector
#         4. Return ≤ max_vectors unique vectors per field
#         5. NEVER generate vectors that match BLOCKED_ACTION_PATTERNS
#            (delete/remove/transfer/payment/password)
#         """

### src/fuzzer/anomaly_classifier.py  (NEW)
# AnomalyClassifier — distinguish real anomalies from expected behavior
#
# class AnomalyType(str, Enum):
#     UNEXPECTED_STATUS = "unexpected_status"
#     SCHEMA_DRIFT      = "schema_drift"
#     INJECTION_SIGNAL  = "injection_signal"   # 500 on SQLi/XSS vector
#     TIMEOUT           = "timeout"
#     EXPECTED_4XX      = "expected_4xx"       # NOT an anomaly
#
# class FuzzResult(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     endpoint:       str
#     vector:         str
#     status_code:    int
#     anomaly:        bool
#     anomaly_type:   AnomalyType | None
#     false_positive: bool    # True if anomaly is expected behavior
#     duration_ms:    float
#
# EXPECTED_BEHAVIORS (not anomalies, false_positive=True):
#   - 422 on invalid email format  → EXPECTED_4XX
#   - 401 on unauthenticated write → EXPECTED_4XX
#   - 404 on nonexistent slug      → EXPECTED_4XX
#
# REAL ANOMALIES (false_positive=False):
#   - 500 on any input             → INJECTION_SIGNAL
#   - 200 on SQLi payload          → INJECTION_SIGNAL
#   - response body missing required field → SCHEMA_DRIFT
#   - request timeout (>10s)       → TIMEOUT

### audit/sprint13/measure_sprint13.py
# Flow:
#   1. SchemaInferrer.capture(page, "https://realworld.habsida.net")
#   2. SchemaInferrer.infer(traces) → list[InferredSchema]
#   3. For each schema: AdvancedVectorGenerator.generate()
#   4. AutonomousAPIFuzzer.fuzz() with each vector
#      → page.route() intercept + AnomalyClassifier.classify()
#   5. OTelTracer.span("fuzz.endpoint") per endpoint
#   6. CryptoAuditTrail.append("fuzz_result") per anomaly found
#
# false_positive_rate = false_positives / max(1, total_anomalies)

## OUTPUT audit/sprint13/sprint13_results.json:
{
  "endpoints_discovered":    <int>,
  "schema_inferred":         <int>,
  "fuzz_vectors_generated":  <int>,
  "anomalies_found":         <int>,
  "false_positive_rate":     <float>,
  "otel_spans_emitted":      <int>,
  "regression":              <bool>,
  "sprint13_status":         "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - BLOCKED_ACTION_PATTERNS enforced in vector generation
# - NEVER capture/log Authorization header values or passwords
# - NEVER fuzz at >2 req/sec per endpoint (rate limit)
# - NEVER invent endpoints not seen in Playwright traffic
# - Pause endpoint 10 minutes if ≥3 consecutive 5xx responses
# - Schema marked is_candidate=True until coverage_score ≥ 3
# - OTelTracer + CryptoAuditTrail on every anomaly
# - BFT: disabled (feature flag preserved)
# - uv add genson (for JSON Schema inference from samples)
#   if not already in pyproject.toml