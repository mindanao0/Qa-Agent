You are an Elite AI & QA Automation Architect. Execute the following build task autonomously with zero deviation from the specifications below.

═══════════════════════════════════════════════════════════════
MISSION: Build Priority 6 — Property-Based Testing (PBT) Subsystem
SOURCE SPEC: Precision Through Properties — Principles of LLM Code Refinement
ARCHITECTURE: Automated Property-Generated Solver (PGS) + Deterministic AST Validation Core
═══════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD CONSTRAINTS (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- OS: Windows 11 bare-metal ONLY. No Docker, WSL, Linux paths.
- Python: 3.11+ with full async/await throughout.
- LLM: Ollama at http://localhost:11434 ONLY. No OpenAI, no external APIs.
- Libraries: langgraph, hypothesis, playwright, pandas, numpy, scipy, tree-sitter, pydantic>=2.0
- All file paths use pathlib.Path (Windows-compatible).
- LangGraph orchestrates ALL multi-agent state machines.
- FAILURE PHILOSOPHY: Completely decouple code generation from validation. 
  LLMs must NEVER evaluate their own logic (eliminates "cycle of self-deception").
- Compliance target: DO-178C Design Assurance Level (DAL) A.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE: Invariant Latent Space Hypothesis (ILSH) ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Convert latent LLM logic errors into explicit machine-checkable assertion 
failures by anchoring verification in:
  1. Property-Based Testing (PBT) via Hypothesis
  2. Linear Temporal Logic (LTL) necessity operators (□)
  3. Computation Tree Logic (CTL)
  4. AST Node-Index Traversal (AST(NIT)) validation
  5. Dynamic Knowledge Base (KB) introspection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE TO CREATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

src/pbt/
├── __init__.py
├── models.py                    # Pydantic V2 schemas
├── ast_validator.py             # AST(NIT) depth-first traversal + KB introspection
├── ltl_engine.py                # LTL/CTL property definition + assertion runtime
├── hypothesis_runner.py         # Stateful PBT orchestrator with shrinking
├── knowledge_base.py            # Dynamic KB: introspects installed Python 3.11 libs
├── pgs_solver.py                # Automated Property-Generated Solver (main logic)
└── agent/
    ├── __init__.py
    ├── state.py                 # LangGraph TypedDict state definition
    ├── generator_node.py        # Generator Agent: CNL → Python code via Ollama
    ├── tester_node.py           # Tester Agent: PBT + AST validation orchestrator
    ├── kill_switch_node.py      # Kill-switch evaluator (4 conditions)
    └── graph.py                 # LangGraph DAG wiring: Generator ↔ Tester loop

tests/pbt/
├── conftest.py
├── test_ast_validator.py
├── test_ltl_engine.py
├── test_hypothesis_runner.py
└── test_pgs_graph.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 1 — src/pbt/models.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement Pydantic V2 schemas (model_config = ConfigDict(strict=True)):

class CNLSpecification(BaseModel):
  spec_id: str
  natural_language: str
  pre_conditions: list[str]       # LTL □ safety properties
  post_conditions: list[str]      # CTL liveness properties  
  invariants: list[str]           # Multiset(A_out) = Multiset(A_in) style
  domain: Literal["sorting","data_transform","api_contract","playwright_flow"]

class ASTValidationResult(BaseModel):
  is_valid: bool
  node_count: int
  kch_violations: list[str]       # Knowledge Conflicting Hallucinations
  bare_critical_calls: list[str]  # e.g., loads() instead of json.loads()
  unknown_api_calls: list[str]    # APIs not found in KB
  traversal_path: list[str]       # Serialized AST(NIT) node sequence

class PBTResult(BaseModel):
  passed: bool
  total_examples: int
  shrunk_counterexample: str | None   # Minimal failing input after shrinking
  violated_invariant: str | None
  ltl_violation: bool
  hypothesis_database_path: str

class KillSwitchDecision(BaseModel):
  halt: bool
  triggered_conditions: list[str]
  condition_codes: list[Literal["KCH","BARE_CALL","LTL_VIOLATION","MC_DC_FAILURE"]]

class PGSCycleState(BaseModel):
  cycle_id: int
  generated_code: str
  ast_result: ASTValidationResult | None = None
  pbt_result: PBTResult | None = None
  kill_switch: KillSwitchDecision | None = None
  feedback: str = ""
  is_solved: bool = False

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 2 — src/pbt/knowledge_base.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement DynamicKnowledgeBase class:

- async def build_kb(self) -> None:
  * Introspect all locally installed Python 3.11 packages via importlib.metadata
  * For each package, dynamically import and inspect all public callables
  * Store as: {module.callable: inspect.signature()} in self._kb dict
  * Persist to pathlib cache file: Path("src/pbt/.kb_cache.json")

- def validate_call_site(self, module: str, func: str, args: list) -> tuple[bool, str]:
  * Check module.func exists in self._kb
  * Validate argument count/names against stored signature
  * Return (True, "") or (False, "KCH: pd.read_exel does not exist")

- def detect_bare_critical_calls(self, source_code: str) -> list[str]:
  * Use ast.walk() to find ast.Call nodes
  * Flag any unqualified calls matching: loads, dumps, open, read, write, connect
  * Return list of "bare_call:line_N:func_name" strings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 3 — src/pbt/ast_validator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement ASTValidator class with Augmented AST(NIT) approach:

- async def parse_to_nit(self, source_code: str) -> list[str]:
  * Parse with ast.parse()
  * Execute Depth-First Node-Index Traversal
  * Inject lexical identifiers into terminal nodes: "N{index}:{NodeType}:{value}"
  * Return serialized traversal path list

- async def validate(self, source_code: str, spec: CNLSpecification) -> ASTValidationResult:
  * Run parse_to_nit() for structural analysis
  * Run O(n*m) validation pass: for each AST call node, check against KB
  * Detect bare critical calls via KB.detect_bare_critical_calls()
  * Detect unknown API calls via KB.validate_call_site()
  * Count ast nodes, return full ASTValidationResult

- async def compute_mcdc_coverage(self, source_code: str, test_results: list[bool]) -> float:
  * Use ast.walk() to count all ast.BoolOp, ast.Compare, ast.If nodes (decision points)
  * Estimate MC/DC coverage from test_results boolean coverage matrix
  * Return float 0.0–1.0 (must reach 1.0 for DAL-A compliance)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 4 — src/pbt/ltl_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement LTLEngine class:

class LTLProperty(BaseModel):
  name: str
  operator: Literal["GLOBALLY","EVENTUALLY","UNTIL","NEXT"]  # □, ◇, U, ○
  predicate: str          # Python lambda as string: "lambda x: x == sorted(x)"
  is_safety: bool         # True = □(necessity), False = ◇(liveness)

- def register_property(self, prop: LTLProperty) -> None

- def evaluate_globally(self, sequence: list[Any], predicate_str: str) -> tuple[bool, int | None]:
  * Evaluate □P — P must hold at EVERY state in sequence
  * If violated, return (False, violation_index)
  * Compile predicate_str safely via compile() + restricted eval namespace

- def evaluate_eventually(self, sequence: list[Any], predicate_str: str) -> tuple[bool, int | None]:
  * Evaluate ◇P — P must hold at SOME state
  
- async def check_spec_invariants(self, spec: CNLSpecification, execution_trace: list[Any]) -> tuple[bool, str | None]:
  * For each pre_condition in spec: evaluate as GLOBALLY property
  * For each post_condition in spec: evaluate as EVENTUALLY property  
  * For each invariant: evaluate GLOBALLY with Multiset equality check
  * Return (all_passed, first_violated_property_name)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 5 — src/pbt/hypothesis_runner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement HypothesisRunner using Hypothesis library:

- async def run_pbt_suite(
    self,
    source_code: str,
    spec: CNLSpecification,
    ltl_engine: LTLEngine,
    max_examples: int = 500
  ) -> PBTResult:

  Strategy:
  1. Dynamically compile source_code into an executable module using exec() + importlib
  2. Build Hypothesis @given strategies from spec.invariants:
     - For sorting domain: st.lists(st.integers(), min_size=0, max_size=100)
     - For data_transform: st.dictionaries(st.text(), st.floats(allow_nan=False))
     - For api_contract: st.fixed_dictionaries with required keys from spec
  3. For each generated example, call ltl_engine.check_spec_invariants()
  4. On Hypothesis Falsifying Example: capture the shrunk_counterexample string
  5. Use settings(max_examples=max_examples, deriving_new=True, database=None)
  6. Capture all results into PBTResult

CRITICAL: Hypothesis shrinking MUST execute automatically on invariant failure.
Wrap the inner test function so that AssertionError from LTL violations 
triggers Hypothesis's built-in minimize loop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 6 — src/pbt/agent/state.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class PGSGraphState(TypedDict):
  spec: CNLSpecification
  cycles: Annotated[list[PGSCycleState], operator.add]
  current_cycle: int
  generated_code: str
  ast_result: ASTValidationResult | None
  pbt_result: PBTResult | None
  kill_switch: KillSwitchDecision | None
  feedback_history: list[str]
  is_solved: bool
  halt: bool
  max_cycles: int  # Default: 5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 7 — src/pbt/agent/generator_node.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement async generator_node(state: PGSGraphState) -> dict:

1. Build system prompt enforcing:
   - "Translate the CNL specification into Python 3.11+ code ONLY"
   - "Output ONLY raw Python. No markdown, no backticks, no comments"
   - "Use fully qualified imports: import json; json.loads() NOT loads()"
   - "Pre-conditions: {spec.pre_conditions}"
   - "Post-conditions: {spec.post_conditions}"  
   - "Invariants to satisfy: {spec.invariants}"

2. If state["feedback_history"] is non-empty, append feedback as correction context

3. Call Ollama via httpx.AsyncClient:
   POST http://localhost:11434/api/generate
   model: "qwen2.5-coder:7b" (fallback: "codellama:7b")
   stream: False
   options: {temperature: 0.1, seed: 42, num_predict: 2048}

4. Extract raw Python code from response["response"]
   Strip any accidental markdown fences with regex: r'```python?(.*?)```'

5. Return {"generated_code": clean_code, "current_cycle": state["current_cycle"] + 1}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 8 — src/pbt/agent/tester_node.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement async tester_node(state: PGSGraphState) -> dict:

ZERO-DEFECT PIPELINE (execute in strict sequence):

Step 1 — AST(NIT) Validation:
  validator = ASTValidator(kb=DynamicKnowledgeBase())
  ast_result = await validator.validate(state["generated_code"], state["spec"])
  If ast_result.kch_violations OR ast_result.bare_critical_calls OR ast_result.unknown_api_calls:
    feedback = f"AST_FAIL: KCH={ast_result.kch_violations}, BARE={ast_result.bare_critical_calls}"
    return {"ast_result": ast_result, "feedback_history": [feedback], "is_solved": False}

Step 2 — LTL Property Registration:
  ltl = LTLEngine()
  For each invariant in spec.invariants: register as GLOBALLY □ safety property
  For each post_condition: register as EVENTUALLY ◇ liveness property

Step 3 — Hypothesis PBT Execution:
  runner = HypothesisRunner()
  pbt_result = await runner.run_pbt_suite(generated_code, spec, ltl, max_examples=300)
  
  If pbt_result.ltl_violation OR pbt_result.shrunk_counterexample:
    feedback = f"PBT_FAIL: violated={pbt_result.violated_invariant}, minimal_input={pbt_result.shrunk_counterexample}"
    return {"pbt_result": pbt_result, "feedback_history": [feedback], "is_solved": False}

Step 4 — MC/DC Coverage Check:
  mcdc = await validator.compute_mcdc_coverage(generated_code, [pbt_result.passed])
  If mcdc < 1.0:
    feedback = f"MCDC_FAIL: coverage={mcdc:.2%} (required: 100% for DAL-A)"
    return {"feedback_history": [feedback], "is_solved": False}

Step 5 — Mark Solved:
  return {"ast_result": ast_result, "pbt_result": pbt_result, "is_solved": True}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 9 — src/pbt/agent/kill_switch_node.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implement async kill_switch_node(state: PGSGraphState) -> dict:

Evaluate ALL 4 spec-defined kill conditions simultaneously:

CONDITION 1 — KCH (Knowledge Conflicting Hallucination):
  Trigger if: ast_result.unknown_api_calls is non-empty
  Code: "KCH"

CONDITION 2 — BARE_CALL:
  Trigger if: ast_result.bare_critical_calls is non-empty
  Code: "BARE_CALL"

CONDITION 3 — LTL_VIOLATION:
  Trigger if: pbt_result is not None AND pbt_result.ltl_violation is True
  AND this is the FINAL cycle (current_cycle >= max_cycles)
  Code: "LTL_VIOLATION"

CONDITION 4 — MC_DC_FAILURE:
  Trigger if: current_cycle >= max_cycles AND is_solved is False
  Code: "MC_DC_FAILURE"

If ANY condition triggers: halt=True, log all triggered_conditions
Return: {"kill_switch": KillSwitchDecision(...), "halt": bool}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 10 — src/pbt/agent/graph.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build LangGraph StateGraph with this exact DAG topology:

NODES: generator → tester → kill_switch → [END or loop back]

EDGES:
  START → generator_node
  generator_node → tester_node
  tester_node → kill_switch_node
  
  kill_switch_node → conditional routing function:
    def route_after_kill_switch(state) -> str:
      if state["halt"]: return "END"
      if state["is_solved"]: return "END"
      if state["current_cycle"] >= state["max_cycles"]: return "END"
      return "generator_node"   # Loop back for refinement

Compile graph:
  graph = builder.compile(
    checkpointer=MemorySaver(),   # Safe state checkpointing
    interrupt_before=["tester_node"]  # Human-in-loop inspection point
  )

Expose: async def run_pgs(spec: CNLSpecification, max_cycles: int = 5) -> PGSCycleState

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE 11 — tests/pbt/test_pgs_graph.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write pytest async integration tests:

TEST 1 — test_sorting_invariant_passes:
  spec = CNLSpecification(
    spec_id="SORT-001",
    natural_language="Implement a function sort_list(lst) that sorts a list of integers ascending",
    pre_conditions=["lambda x: isinstance(x, list)"],
    post_conditions=["lambda result: result == sorted(result)"],
    invariants=["lambda x, result: sorted(x) == result", "lambda x, result: len(x) == len(result)"],
    domain="sorting"
  )
  result = await run_pgs(spec, max_cycles=3)
  assert result.is_solved == True
  assert result.kill_switch is None or result.kill_switch.halt == False

TEST 2 — test_kch_triggers_kill_switch:
  Inject deliberately hallucinated code with pd.read_exel() (typo)
  Verify: kill_switch.halt == True
  Verify: "KCH" in kill_switch.condition_codes

TEST 3 — test_bare_call_detection:
  Inject code with bare loads() call (no json. prefix)
  Verify: ast_result.bare_critical_calls is non-empty
  Verify: kill_switch triggered with "BARE_CALL"

TEST 4 — test_hypothesis_shrinking_on_invariant_violation:
  spec with invariant that fails on empty list input
  Verify: pbt_result.shrunk_counterexample is not None
  Verify: len of shrunk_counterexample < 50 chars (minimal input)

TEST 5 — test_ltl_globally_operator:
  Define □P: all elements in result must be >= all elements in input minimum
  Verify LTLEngine.evaluate_globally() correctly catches violation at specific index

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL COMMANDS (run first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip install hypothesis langgraph pydantic>=2.0 httpx pytest pytest-asyncio tree-sitter numpy scipy pandas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTION VERIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After all files are written, run:

  pytest tests/pbt/ -v --asyncio-mode=auto

Expected: All 5 tests PASS with zero warnings.
If Ollama is offline: generator_node must raise OllamaConnectionError 
with message "Ollama unreachable at http://localhost:11434" — do NOT silently return empty string.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-PATTERNS — STRICTLY FORBIDDEN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER let Generator agent call Tester agent or evaluate its own output
- NEVER use time.sleep() — use asyncio.sleep() only
- NEVER use Python's ast module to unparse+rewrite code (use libcst if editing)
- NEVER use broad except: pass — all exceptions must surface with full context
- NEVER hardcode model names without fallback logic
- NEVER use Docker paths, Linux paths, or WSL paths
- NEVER skip the KB introspection step before AST validation