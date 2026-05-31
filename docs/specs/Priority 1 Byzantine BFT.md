You are an Elite AI & QA Automation Architect. Build a production-grade
"Hybrid-BFT Deterministic AST Consensus Orchestrator" strictly following
the <mission_critical_spec> below. Zero deviations permitted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSION CRITICAL SPEC (SOURCE OF TRUTH)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<mission_critical_spec>
  <subsystem_name>Hybrid-BFT Deterministic AST Consensus Orchestrator</subsystem_name>

  <failure_tolerance_philosophy>
    Hybrid fault model: N-Version Programming (NVP) + Byzantine Fault
    Tolerance (BFT). Constraint: N≥3f+1, Quorum Q≥2f+1. Neutralizes
    active manipulation, correlated LLM hallucinations, and non-fail-stop
    errors. Execution is BLOCKED unless deterministically proven structurally
    and semantically equivalent via AST consensus.
  </failure_tolerance_philosophy>

  <strict_requirements>
    <libraries>langgraph>=0.2, langchain-ollama, ast, asyncio,
    concurrent.futures, playwright, pydantic>=2.0</libraries>
    <windows_native_constraints>
      - Bare-metal Windows 11, Python 3.11+ ONLY
      - NO Docker, NO WSL, NO Linux paths
      - Use asyncio.wait with FIRST_COMPLETED for parallel fan-out
      - Use concurrent.futures.wait for thread-safe coordination
      - State reducers: Annotated[list, operator.add] (non-destructive
        list-appending) to prevent InvalidUpdateError on concurrent writes
      - LLM: Ollama at http://localhost:11434 ONLY
      - Fan-out routing: LangGraph dynamic Send() ONLY — never imperative
        subgraph invocation (prevents MULTIPLE_SUBGRAPHS namespace crash)
    </windows_native_constraints>
  </strict_requirements>

  <zero_defect_implementation>
    <step id="1">GENERATOR FAN-OUT
      Deploy N=4 generator nodes (f=1, so N≥3*1+1=4 satisfied).
      Diversification strategy to break error correlation:
        - Node 0,2: model=codellama:13b, temperature=0.0 (greedy)
        - Node 1,3: model=deepseek-coder:6.7b, top_p=0.85 (nucleus)
      Each node wrapped in try-except with RetryPolicy(max_attempts=3).
      On failure: write structured fallback BFTGeneratorOutput with
      status="FAILED" to override LangGraph's _should_stop_others and
      _panic_or_proceed fail-fast cancellation — preserving liveness.
      Route via: Send("generator_node", payload) dynamic fan-out.
    </step>

    <step id="2">3-PHASE DETERMINISTIC AST NORMALIZATION PIPELINE
      Implement as ast.NodeTransformer subclass: ASTNormalizer

      Phase 1 — Strip Non-Functional Elements:
        Remove all ast.Expr nodes whose value is ast.Constant (docstrings,
        standalone string literals). Strip ast.Import / ast.ImportFrom only
        if they are duplicate. Preserve semantic imports.

      Phase 2 — Canonicalize Local Identifiers:
        Build a mapping_register dict. Traverse all ast.Name and
        ast.arg nodes. If identifier is NOT in PROTECTED_KEYWORDS
        = {"playwright", "browser", "page", "context", "expect",
           "asyncio", "async_playwright"} then replace with
        canonical "var_N" where N is sequential counter.
        Preserve all PROTECTED_KEYWORDS untouched — this is a
        KILL SWITCH trigger if violated.

      Phase 3 — Strip Position Metadata + Structural Hash:
        Remove lineno, col_offset, end_lineno, end_col_offset from
        all nodes via ast.fix_missing_locations reset trick.
        Serialize normalized AST to canonical string via ast.dump().
        This canonical string IS the vote fingerprint.
    </step>

    <step id="3">DETERMINISTIC VOTER / BFT QUORUM CERTIFIER
      Collect all N canonical AST fingerprints from generator outputs.
      Filter out status="FAILED" results first.
      Apply strict quorum check:
        - Count fingerprint frequency using collections.Counter
        - Find the most_common fingerprint
        - If its count >= Q (where Q = 2f+1 = 3): QUORUM ACHIEVED
        - If count < Q: KILL SWITCH — raise BFTQuorumFailure exception,
          ABORT execution, log all divergent fingerprints for audit
      On QUORUM ACHIEVED: select the corresponding raw Python code
      from any generator that produced the winning fingerprint.
      Emit BFTConsensusResult with certified_code and quorum_metadata.
    </step>
  </zero_defect_implementation>

  <kill_switches>
    <condition id="KS-1">
      Voter fails to secure Q≥2f+1 AST-level matches →
      raise BFTQuorumFailure("ABORT: quorum not achieved")
      Log all N fingerprints + divergence map. NEVER execute code.
    </condition>
    <condition id="KS-2">
      ASTNormalizer detects any ast.Name or ast.Attribute node
      attempting to remap PROTECTED_KEYWORDS →
      raise BFTSecurityViolation("ABORT: protected API remapping detected")
    </condition>
    <condition id="KS-3">
      Any generator node invokes a subgraph imperatively instead of
      via Send() → LangGraph will raise MULTIPLE_SUBGRAPHS — treat
      as Byzantine fault, mark that node as FAILED
    </condition>
  </kill_switches>
</mission_critical_spec>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE TO CREATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

src/
├── bft/
│   ├── __init__.py
│   ├── models.py          ← Pydantic V2 schemas (ALL state models)
│   ├── ast_normalizer.py  ← ASTNormalizer (NodeTransformer, 3-phase)
│   ├── voter.py           ← BFT quorum certifier + kill switches
│   ├── generator.py       ← Async Ollama generator nodes
│   └── graph.py           ← LangGraph StateGraph + Send() fan-out
├── config.py              ← N, f, Q constants + Ollama config
└── main.py                ← Async entrypoint + CLI runner

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION RULES (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELS (src/bft/models.py):
  - BFTGeneratorInput: task_prompt: str, node_id: int, model: str,
    temperature: float, top_p: float
  - BFTGeneratorOutput: node_id: int, raw_code: str,
    ast_fingerprint: str | None, status: Literal["OK","FAILED"],
    error_msg: str | None
  - BFTConsensusResult: certified_code: str, winning_fingerprint: str,
    quorum_count: int, total_nodes: int, divergence_map: dict[str,int],
    quorum_achieved: bool
  - BFTGraphState: TypedDict with:
      task_prompt: str
      generator_outputs: Annotated[list[BFTGeneratorOutput], operator.add]
      consensus_result: BFTConsensusResult | None
      abort_reason: str | None

AST NORMALIZER (src/bft/ast_normalizer.py):
  - Class ASTNormalizer(ast.NodeTransformer)
  - PROTECTED_KEYWORDS: frozenset at class level
  - Method normalize(source_code: str) -> str:
      parse → phase1_strip → phase2_canonicalize → phase3_hash
      returns ast.dump() of final normalized tree
  - Kill Switch KS-2 inside visit_Name() and visit_Attribute()
  - Must handle ast.parse() SyntaxError → return fingerprint="SYNTAX_ERROR"

VOTER (src/bft/voter.py):
  - async def certify_quorum(
        outputs: list[BFTGeneratorOutput],
        f: int
    ) -> BFTConsensusResult
  - Filter FAILED outputs first
  - Use collections.Counter on ast_fingerprint
  - Enforce Q = 2*f + 1
  - Kill Switch KS-1: raise BFTQuorumFailure if quorum not met
  - Return full BFTConsensusResult with divergence_map

GENERATOR (src/bft/generator.py):
  - async def run_generator_node(
        state: BFTGeneratorInput
    ) -> dict  ← returns {"generator_outputs": [BFTGeneratorOutput]}
  - Uses langchain_ollama.ChatOllama with structured output
  - Prompt must enforce: "Return ONLY raw Python code. No markdown.
    No explanations. No triple backticks."
  - Wrap in try-except, max 3 retries with exponential backoff
  - On all retries exhausted: return status="FAILED"
  - After raw_code received: call ASTNormalizer().normalize(raw_code)
    and attach ast_fingerprint to output

GRAPH (src/bft/graph.py):
  - async def build_bft_graph() -> CompiledGraph
  - Node "router": reads task_prompt, emits N Send("generator_node", ...)
    Use Send() from langgraph.types — NEVER invoke nodes imperatively
  - Node "generator_node": calls run_generator_node()
  - Node "voter_node": calls certify_quorum(), writes consensus_result
    or abort_reason to state
  - Node "executor_node": ONLY reached if consensus_result.quorum_achieved
    is True — prints certified_code, does NOT execute yet (safety gate)
  - Edge: START → router → [Send x N] → generator_node →
    voter_node → conditional_edge → executor_node or abort_node
  - Conditional edge logic: if state["abort_reason"] → "abort_node"
    else → "executor_node"

CONFIG (src/config.py):
  - N: int = 4       # Total generator nodes
  - F: int = 1       # Max Byzantine faults tolerated
  - Q: int = 2*F+1   # Quorum threshold = 3
  - OLLAMA_BASE_URL = "http://localhost:11434"
  - NODE_CONFIGS: list of 4 dicts with model + sampling params
    per the diversification strategy in step 1

MAIN (src/main.py):
  - async def main(task_prompt: str)
  - Builds graph, invokes with {"task_prompt": task_prompt}
  - Prints structured JSON summary of BFTConsensusResult
  - if __name__ == "__main__": asyncio.run(main(sys.argv[1]))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL COMMANDS (output as bash block)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip install langgraph langchain-ollama pydantic playwright asyncio
playwright install chromium

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION CHECKLIST (verify before finishing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] N=4, f=1, Q=3 constants enforced in config.py
[ ] Annotated[list, operator.add] reducer on generator_outputs
[ ] All fan-out via Send() — zero imperative node calls
[ ] ASTNormalizer 3-phase pipeline fully implemented
[ ] PROTECTED_KEYWORDS checked in visit_Name AND visit_Attribute
[ ] BFTQuorumFailure raised (not printed) when quorum < Q
[ ] BFTSecurityViolation raised (not printed) for KS-2
[ ] executor_node gated behind quorum_achieved==True check
[ ] All async functions use await — zero blocking calls
[ ] Windows paths only (pathlib.Path) — zero /usr/ or /var/ paths
[ ] Ollama URL hardcoded to http://localhost:11434 only
[ ] Complete files — zero placeholders, zero "# ... existing code"