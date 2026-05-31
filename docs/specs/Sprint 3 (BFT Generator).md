Sprint 2 CLOSED (PASS). Sprint 3 objective: Byzantine Fault Tolerant generator 
that runs 3 serial LLM calls, normalizes output via AST, votes for 2/3 consensus, 
then passes winner to LLM-as-Judge for semantic validation.

Read FIRST in this exact order:
  1. CLAUDE.md + docs/specs/SPEC_CORE.md
  2. src/agents/generator.py             (current single-pass generator — upgrade)
  3. src/llm/instructor_client.py        (InstructorClient — re-use all 3 instances)
  4. src/llm/schemas.py                  (GeneratedCode, TestPlan — re-use)
  5. src/graph/workflow.py OR graph.py   (LangGraph topology — modify)
  6. src/core/agent.py                   (AgentState TypedDict — extend)
  7. audit/sprint2/sprint2_results.json  (latency baseline = 76,899ms)

================================================================================
ARCHITECTURE — BFT Pipeline
================================================================================

Adaptive routing runs BEFORE any LLM call to protect latency:

  URL+requirement hash ─→ check url_test_cache table in SQLite
        │
        ├─ HIT (previously passed, confidence=HIGH ≥0.95)
        │    └─→ return cached GeneratedCode, skip BFT entirely
        │
        ├─ PARTIAL (similar requirement seen, confidence=MED 0.5-0.95)
        │    └─→ single generator (temperature=0.0) + LLM-as-Judge only
        │
        └─ MISS (never seen, confidence=LOW <0.5)
             └─→ full BFT: 3 generators serial + AST consensus + Judge

BFT 3-Generator Chain (SERIAL, never parallel — 6GB VRAM constraint):
  Generator-0: temperature=0.0, top_p=1.0   (greedy, deterministic)
  Generator-1: temperature=0.3, top_p=0.85  (low stochastic)
  Generator-2: temperature=0.7, top_p=0.85  (high diversity)

  Each calls instructor_client.create_structured(prompt, GeneratedCode)
  with asyncio.Semaphore(1) — EXISTING semaphore, do not replace it.

AST Normalization (3 phases, per-output):
  Phase 1: Strip ast.Expr nodes whose value is ast.Constant (docstrings)
  Phase 2: Canonicalize non-protected names → var_1, var_2, ...
  Phase 3: Strip lineno/col_offset metadata → ast.unparse()

  PROTECTED_NAMES = frozenset({
      "playwright", "browser", "page", "context", "expect",
      "async_playwright", "True", "False", "None",
      "asyncio", "pytest", "request", "response"
  })

Consensus Voting (Q ≥ 2/3 = 2 votes needed):
  - Normalize all 3 outputs
  - Group by exact string equality after normalization
  - Largest group ≥ 2 → winner = raw (pre-normalized) code from that group
  - Largest group < 2 → CONSENSUS_FAILED → abort, do not execute

LLM-as-Judge (runs only when consensus reached):
  Single prompt to instructor_client_planner (temperature=0.0):
    "Review this Playwright test for: semantic correctness, no false assertions,
     no hallucinated selectors. Return JudgeVerdict."
  JudgeVerdict.approved=False → JUDGE_REJECTED → abort

Kill Switches (hard abort, no retry):
  1. CONSENSUS_FAILED (< 2/3 agreement) → return empty GeneratedCode, 
     add bft_status="CONSENSUS_FAILED" to AgentState
  2. JUDGE_REJECTED → return empty, bft_status="JUDGE_REJECTED"
  3. NodeTransformer detects protected name remapping attempt → 
     raise BFTSecurityHalt immediately

================================================================================
CLUSTER STRUCTURE — SUBAGENT-DRIVEN (5 clusters)
================================================================================

CLUSTER S3-A — AST Normalizer + BFT Voter (~3 hours)
======================================================
Subagent context: src/llm/schemas.py

Deliverables:
  A1. New module src/core/bft_consensus.py containing:

      PROTECTED_NAMES = frozenset({
          "playwright", "browser", "page", "context", "expect",
          "async_playwright", "True", "False", "None",
          "asyncio", "pytest", "request", "response"
      })

      class _DocstringStripper(ast.NodeTransformer):
          """Phase 1: Remove bare string literals (docstrings, comments)."""
          def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
              if isinstance(node.value, ast.Constant) and isinstance(
                  node.value.value, str
              ):
                  return None
              return node

      class _NameCanonicalizer(ast.NodeTransformer):
          """Phase 2: Rename non-protected names to var_1, var_2, ..."""
          def __init__(self) -> None:
              self._mapping: dict[str, str] = {}
              self._counter = 0

          def visit_Name(self, node: ast.Name) -> ast.Name:
              if node.id in PROTECTED_NAMES:
                  return node
              if node.id not in self._mapping:
                  self._counter += 1
                  self._mapping[node.id] = f"var_{self._counter}"
              return ast.Name(id=self._mapping[node.id], ctx=node.ctx)
          
          def visit_Assign(self, node: ast.Assign) -> ast.Assign:
              # Kill switch: detect attempt to remap protected names
              for target in node.targets:
                  if isinstance(target, ast.Name) and target.id in PROTECTED_NAMES:
                      raise BFTSecurityHalt(
                          f"Attempt to remap protected name: {target.id}"
                      )
              return self.generic_visit(node)

      class ASTNormalizer:
          def normalize(self, source_code: str) -> str | None:
              # Returns normalized string or None on SyntaxError
              try:
                  tree = ast.parse(source_code)
              except SyntaxError:
                  return None
              
              try:
                  tree = _DocstringStripper().visit(tree)
                  tree = _NameCanonicalizer().visit(tree)
              except BFTSecurityHalt:
                  return None  # Treat as Byzantine
              
              # Phase 3: Strip position metadata
              for node in ast.walk(tree):
                  for attr in ("lineno","col_offset","end_lineno","end_col_offset"):
                      if hasattr(node, attr):
                          delattr(node, attr)
              
              ast.fix_missing_locations(tree)
              return ast.unparse(tree)

      class BFTSecurityHalt(Exception):
          """Raised when generator output attempts to remap protected names."""

      class BFTVoter:
          def __init__(self) -> None:
              self._normalizer = ASTNormalizer()

          def vote(self, candidates: list[str | None]) -> str | None:
              """Returns raw (pre-normalized) winning code if quorum ≥ 2."""
              valid: list[tuple[str, str]] = []  # (raw, normalized)
              for raw in candidates:
                  if raw is None:
                      continue
                  norm = self._normalizer.normalize(raw)
                  if norm is not None:
                      valid.append((raw, norm))

              if not valid:
                  return None

              # Group by normalized equality
              groups: dict[str, list[str]] = {}
              for raw, norm in valid:
                  groups.setdefault(norm, []).append(raw)

              # Find largest group
              best_norm = max(groups, key=lambda k: len(groups[k]))
              if len(groups[best_norm]) >= 2:
                  return groups[best_norm][0]  # return raw from winning group
              return None  # No quorum

  A2. BFTGeneratorConfig (Pydantic V2):
      class BFTGeneratorConfig(BaseModel):
          node_id: int
          temperature: float
          top_p: float
          model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
          model_config = ConfigDict(frozen=True, extra="forbid")

      BFT_CONFIGS: tuple[BFTGeneratorConfig, ...] = (
          BFTGeneratorConfig(node_id=0, temperature=0.0, top_p=1.0),
          BFTGeneratorConfig(node_id=1, temperature=0.3, top_p=0.85),
          BFTGeneratorConfig(node_id=2, temperature=0.7, top_p=0.85),
      )

GATE S3-A — tests/test_bft_consensus.py (7 tests):
  [ ] test_normalizer_strips_docstrings
  [ ] test_normalizer_canonicalizes_local_names
  [ ] test_normalizer_preserves_protected_names
  [ ] test_normalizer_returns_none_on_syntax_error
  [ ] test_normalizer_raises_security_halt_on_remap
  [ ] test_voter_quorum_reached_two_identical
  [ ] test_voter_no_quorum_all_different
  [ ] test_voter_returns_none_when_all_none
  All 8 must pass before Cluster S3-B.

CLUSTER S3-B — BFT Generator Node (~4 hours)
=============================================
Subagent context:
  - Cluster S3-A outputs (bft_consensus.py)
  - src/llm/instructor_client.py (existing — re-use named instances)
  - src/agents/generator.py (existing — preserve legacy path)
  - src/core/agent.py (AgentState)

Deliverables:
  B1. Add to AgentState TypedDict (surgical addition only):
      bft_status: str           # "PENDING" | "CONSENSUS_REACHED" | 
                                #  "CONSENSUS_FAILED" | "JUDGE_REJECTED"
      bft_node_results: Annotated[list[str], operator.add]
                                # ["node_0:PASS", "node_1:PASS", "node_2:FAIL"]
      bft_confidence_tier: str  # "HIGH" | "MED" | "LOW" (set by router)

  B2. New module src/agents/bft_generator.py:

      async def run_single_generator(
          config: BFTGeneratorConfig,
          prompt: str,
          semaphore: asyncio.Semaphore,
      ) -> str | None:
          """Run one generator call. Returns code string or None on any failure."""
          # Override temperature per config by passing options to Ollama
          # Re-use instructor_client but set temperature via extra_body or 
          # config — check how instructor_client exposes options
          # Wrap in try/except: on ANY exception return None (Byzantine fault)
          # Acquire semaphore BEFORE Ollama call
          # Log: node_id, temperature, latency_ms, token_count, success/fail

      async def bft_generator_node(state: AgentState) -> dict:
          """LangGraph node replacing single generator_node."""
          prompt = _build_generation_prompt(state)  # extract from existing generator.py
          semaphore = _get_llm_semaphore()           # module-level Semaphore(1)

          results: list[str | None] = []
          for config in BFT_CONFIGS:
              result = await run_single_generator(config, prompt, semaphore)
              results.append(result)
              node_status = "PASS" if result else "FAIL"
              # Collect inline — no parallel gather (SERIAL is the constraint)

          winning_code = BFTVoter().vote(results)

          if winning_code is None:
              return {
                  "bft_status": "CONSENSUS_FAILED",
                  "bft_node_results": [f"node_{i}:{'PASS' if r else 'FAIL'}"
                                        for i, r in enumerate(results)],
                  "generated_code": None,
              }

          return {
              "bft_status": "CONSENSUS_REACHED",
              "bft_node_results": [f"node_{i}:{'PASS' if r else 'FAIL'}"
                                    for i, r in enumerate(results)],
              "generated_code": winning_code,
          }

  B3. Feature flag in config/agent.yaml:
      llm:
        bft:
          enabled: true
          confidence_routing: true   # adaptive skip/lite/full

GATE S3-B:
  [ ] bft_generator_node returns dict with bft_status in all 3 outcomes
  [ ] Semaphore acquired per call (not once for all 3)
  [ ] No asyncio.gather() — must be serial for-loop
  [ ] Legacy generator_node still callable via feature flag

CLUSTER S3-C — LLM-as-Judge (~3 hours)
========================================
Subagent context:
  - src/llm/instructor_client.py (instructor_planner instance, temperature=0.0)
  - src/llm/schemas.py (extend with JudgeVerdict)

Deliverables:
  C1. Add JudgeVerdict to src/llm/schemas.py:
      class JudgeVerdict(BaseModel):
          approved: bool
          confidence: float = Field(ge=0.0, le=1.0)
          rejection_reason: str | None = Field(default=None, max_length=300)
          issues_found: list[str] = Field(default_factory=list, max_length=5)
          model_config = ConfigDict(extra="forbid", frozen=True)

  C2. New module src/llm/judge_client.py:
      class JudgeClient:
          def __init__(self, instructor_client: InstructorClient) -> None:
              self._client = instructor_client

          async def evaluate(
              self,
              generated_code: str,
              requirement: str,
              domain: str,
          ) -> JudgeVerdict:
              """
              Prompt structure (keep under 800 tokens):
                ROLE: You are a senior QA engineer reviewing Playwright test code.
                REQUIREMENT: {requirement}
                DOMAIN: {domain}
                CODE: {generated_code}

                Review for ONLY these issues:
                1. Does the test actually verify the stated requirement?
                2. Are there assertions that will always pass (false positives)?
                3. Are there selector references that look hallucinated?
                   (e.g., very long CSS chains, non-existent aria-labels)
                4. Does the test have at least one expect() assertion?

                Return JudgeVerdict.approved=True only if all 4 checks pass.
              """

      The judge uses instructor_planner (temperature=0.0, max_retries=1).
      On InstructorClient failure: return JudgeVerdict(approved=True, 
        confidence=0.5, rejection_reason="judge_unavailable") — fail OPEN 
        (continue without judge) to avoid blocking generation entirely.

  C3. Integrate into bft_generator_node (modify S3-B output):
      After winning_code found by BFTVoter:
        verdict = await judge_client.evaluate(winning_code, requirement, domain)
        if not verdict.approved:
            return {
                "bft_status": "JUDGE_REJECTED",
                "generated_code": None,
                "judge_rejection_reason": verdict.rejection_reason,
            }

GATE S3-C:
  [ ] JudgeVerdict is in schemas.py with extra="forbid"
  [ ] Judge fails OPEN (returns approved=True) when Ollama unavailable
  [ ] tests/test_judge_client.py: 4 tests (approve, reject, fail-open, schema)

CLUSTER S3-D — Adaptive Confidence Router (~2 hours)
=====================================================
Subagent context:
  - SQLite state.db (existing, Sprint 1 requirement)
  - src/agents/bft_generator.py (S3-B)

Deliverables:
  D1. New module src/routing/adaptive_router.py:

      class ConfidenceTier(str, Enum):
          HIGH = "HIGH"    # cache hit, skip BFT
          MED  = "MED"     # seen before, single gen + judge
          LOW  = "LOW"     # never seen, full BFT

      class AdaptiveRouter:
          def __init__(self, db_path: pathlib.Path) -> None:
              self._db = db_path  # SQLite

          def classify(
              self,
              url: str,
              requirement: str,
              domain: str,
          ) -> ConfidenceTier:
              """
              Lookup url_test_cache table:
                - Exact hash (sha256(url+requirement)) → HIGH
                - Domain hit with pass_rate ≥ 0.9 (≥3 runs) → MED
                - Not found → LOW
              """

          def record_result(
              self,
              url: str,
              requirement: str,
              domain: str,
              passed: bool,
              generated_code: str | None,
          ) -> None:
              """Upsert into url_test_cache. Called by Reporter node."""

  D2. Create url_test_cache table in SQLite (add to db init):
      url_test_cache:
        - cache_key: TEXT PRIMARY KEY   (sha256 of url+requirement)
        - url: TEXT
        - domain: TEXT
        - run_count: INTEGER DEFAULT 0
        - pass_count: INTEGER DEFAULT 0
        - pass_rate: REAL DEFAULT 0.0
        - last_generated_code_hash: TEXT
        - last_run_iso: TEXT

  D3. Update bft_generator_node to use router:
      tier = router.classify(url, requirement, domain)

      if tier == HIGH:   return cached result immediately (skip all 3 generators)
      if tier == MED:    run only Generator-0 (temperature=0.0) + Judge
      if tier == LOW:    run full BFT (all 3 generators)

      Store tier in AgentState.bft_confidence_tier for reporting.

GATE S3-D:
  [ ] HIGH tier returns in < 10ms (pure SQLite lookup)
  [ ] MED tier makes exactly 1 LLM call + 1 Judge call
  [ ] LOW tier makes exactly 3 LLM calls + 1 Judge call
  [ ] tests/test_adaptive_router.py: 5 tests

CLUSTER S3-E — Graph Integration + Measurement (~3 hours)
==========================================================
Subagent context: All S3-A through S3-D outputs + workflow.py/graph.py

Deliverables:
  E1. Update LangGraph graph (surgical — keep all existing nodes):
      
      Replace:
        workflow.add_node("generator_node", old_generator_node)
      With:
        workflow.add_node("generator_node", bft_generator_node)
      
      Replace the direct generator_node → executor_node edge with:
        def route_after_bft(state: AgentState) -> str:
            if state.get("bft_status") in ("CONSENSUS_FAILED","JUDGE_REJECTED"):
                return "reporter_node"   # fail fast
            return "executor_node"       # proceed normally

        workflow.add_conditional_edges(
            "generator_node",
            route_after_bft,
            {"executor_node": "executor_node", "reporter_node": "reporter_node"}
        )

  E2. Update Reporter node to call router.record_result() after each run.
      This closes the feedback loop: test runs → cache updates → future routing.

  E3. Create audit/sprint3/measure_sprint3.py writing 
      audit/sprint3/sprint3_results.json:
      {
        "measured_at_iso": "...",
        "dataset_size": N,
        "bft_metrics": {
          "consensus_reached_rate": 0.0,
          "consensus_failed_rate": 0.0,
          "judge_approved_rate": 0.0,
          "judge_rejected_rate": 0.0,
          "avg_bft_latency_ms": 0,
          "node_agreement_distribution": {
            "3_of_3": 0, "2_of_3": 0, "1_of_3": 0, "0_of_3": 0
          }
        },
        "routing_metrics": {
          "HIGH_tier_pct": 0.0,
          "MED_tier_pct": 0.0,
          "LOW_tier_pct": 0.0,
          "avg_latency_by_tier": {"HIGH": 0, "MED": 0, "LOW": 0}
        },
        "quality_metrics": {
          "first_run_pass_rate": 0.0,
          "false_pass_reduction_vs_sprint2": 0.0
        },
        "latency_metrics": {
          "avg_generation_ms": 0,
          "worst_case_full_bft_ms": 0,
          "sprint2_baseline_ms": 76899
        }
      }
  
  Run measurement with --live flag. 
  Golden dataset = all 10 cases start as LOW tier (cache empty).
  Second pass = MED/HIGH tier (should be faster).
  Report BOTH passes.

GATE S3-E — SPRINT 3 ACCEPTANCE:
  PASS iff ALL of:
    [ ] consensus_reached_rate ≥ 0.80    (BFT reaches quorum ≥80% of time)
    [ ] first_run_pass_rate ≥ 0.90       (must not regress below Sprint 2)
    [ ] LOW tier latency ≤ 240,000ms     (3 × 80s = 240s max for full BFT)
    [ ] HIGH tier latency ≤ 500ms        (cache lookup must be fast)
    [ ] judge_approved_rate ≥ 0.85       (judge not too strict/hallucinating)

  FAIL (not regression) iff quality met but latency over
  REGRESSION iff first_run_pass_rate < 0.85

================================================================================
RULES FROM SPRINTS 1-2 (CARRY FORWARD)
================================================================================
  - asyncio.Semaphore(1) on ALL Ollama calls — never remove
  - Never use page.accessibility (use CDP via AOMExtractor)
  - pathlib.Path everywhere
  - Pydantic V2 with ConfigDict(extra="forbid")
  - Never raise from perception layer
  - EpisodicStore tables are append-only

================================================================================
NEW RULES FOR SPRINT 3
================================================================================
  - BFT generators MUST be serial (for-loop, not gather) — 6GB VRAM
  - Kill switches are hard stops — no retry after CONSENSUS_FAILED
  - Judge fails OPEN — never block generation due to judge failure
  - Adaptive router defaults: new URLs start as LOW tier always
  - url_test_cache is OPTIMISTIC — update pass_rate after actual execution,
    not after generation
  - DO NOT add a 4th generator — N=3 is the spec, never change it

================================================================================
SUBAGENT-DRIVEN: Report at each cluster gate. BEGIN with S3-A.