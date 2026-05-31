You are an Elite AI & QA Automation Architect. Execute the following build plan 
with ZERO deviation. Read each constraint carefully before writing any code.

═══════════════════════════════════════════════════════════════
SECTION 1: HARD CONSTRAINTS (NON-NEGOTIABLE)
═══════════════════════════════════════════════════════════════

ENVIRONMENT:
- OS: Windows 11 Native (NO Docker, NO WSL, NO Linux paths)
- Python: 3.11+
- LLM: Ollama at http://localhost:11434 ONLY (NO OpenAI, NO external APIs)
- Hardware: 16GB RAM, 6GB VRAM
- Perception: Accessibility Tree FIRST, Screenshot/VLM ONLY as fallback

LIBRARIES (install if missing):
pip install langgraph>=0.2 pydantic>=2.0 playwright>=1.44 lancedb>=0.8 \
            httpx>=0.27 networkx>=3.3 numpy>=1.26 scikit-fuzzy>=0.4 \
            asyncio python-dotenv rich

═══════════════════════════════════════════════════════════════
SECTION 2: ARCHITECTURE TO BUILD
═══════════════════════════════════════════════════════════════

Build exactly 7 files in this structure:

src/
├── core/
│   ├── state_schema.py          # Pydantic V2 State definitions
│   ├── sfg_engine.py            # State Flow Graph (SFG) + SA-KG builder
│   └── contract_skill.py        # ContractSkill artifact CRUD + repair logic
├── agents/
│   ├── planner_agent.py         # SGE Strategy generator
│   ├── generator_agent.py       # Action mapper → Playwright bulk actions
│   └── healer_agent.py          # Minimal patch operator (SelReplace, PreInsert)
└── graph/
    └── langgraph_pipeline.py    # LangGraph async pipeline wiring all agents


═══════════════════════════════════════════════════════════════
SECTION 3: FILE-BY-FILE SPECIFICATIONS
═══════════════════════════════════════════════════════════════

──────────────────────────────────────────
FILE 1: src/core/state_schema.py
──────────────────────────────────────────
Define ALL Pydantic V2 models with strict=True, extra="forbid":

1. GUIState:
   - state_hash: str          # SHA-256 of active ARIA elements snapshot
   - url: str
   - accessibility_snapshot: str   # compressed AxTree ≤3000 tokens
   - timestamp: float

2. SFGEdge:
   - source_hash: str
   - target_hash: str
   - aria_role: str
   - accessible_name: str
   - action_type: Literal["click","fill","select","navigate","press"]
   - locator_strategy: Literal["getByRole","getByLabel","getByTestId"]

3. ContractSkillArtifact:
   - skill_id: str             # UUID4
   - goal: str
   - preconditions: list[str]
   - action_steps: list[ActionStep]
   - postconditions: list[str]
   - recovery_rules: list[RecoveryRule]
   - version: int = 1
   - status: Literal["active","degraded","patched"]

4. ActionStep:
   - step_index: int
   - intent: str
   - locator_strategy: str
   - locator_value: str
   - action_type: str
   - payload: str | None

5. RecoveryRule:
   - trigger_condition: Literal["NOT_FOUND","INPUT_INVALID","TIMEOUT","STALE"]
   - patch_operator: Literal["SelReplace","PreInsert","ArgCorrect","PostInsert"]
   - patch_payload: str

6. AgentPipelineState (TypedDict with Annotated reducers):
   - task_goal: str
   - current_url: str
   - sfg_nodes: Annotated[list[GUIState], operator.add]
   - sfg_edges: Annotated[list[SFGEdge], operator.add]
   - active_contract: ContractSkillArtifact | None
   - execution_log: Annotated[list[str], operator.add]
   - failed_step_index: int | None
   - repair_attempts: int
   - memory_summary: str
   - mode: Literal["explore","execute","heal"]
   - final_result: str | None

──────────────────────────────────────────
FILE 2: src/core/sfg_engine.py
──────────────────────────────────────────
Implement async class SFGEngine:

METHOD: async def capture_gui_state(page: Page) -> GUIState
  - Call page.accessibility.snapshot() to get AxTree (NOT screenshot)
  - Strip all script/style tags from any HTML fragments
  - Truncate/compress snapshot to ≤3000 tokens using char-count heuristic
    (1 token ≈ 4 chars → max 12000 chars)
  - Compute state_hash = SHA-256(url + compressed_snapshot)
  - Return GUIState model

METHOD: async def extract_interactive_edges(page: Page, 
                                             source_hash: str) -> list[SFGEdge]
  - Query page for all interactive elements using JS evaluate:
    document.querySelectorAll('button,a,input,select,textarea,[role]')
  - For each element extract: role, aria-label, name, type
  - Apply Locator Strategy Decision Tree:
      IF has aria role → strategy = "getByRole"
      ELIF has label   → strategy = "getByLabel"  
      ELSE             → strategy = "getByTestId"
  - Return list[SFGEdge] with source=source_hash, target="pending"

METHOD: def build_networkx_graph(nodes: list[GUIState], 
                                   edges: list[SFGEdge]) -> nx.DiGraph
  - Use networkx.DiGraph
  - Add nodes with state_hash as node ID, store GUIState as attribute
  - Add edges with SFGEdge fields as attributes
  - Detect and merge duplicate states using hash equality
  - Return compiled DiGraph

METHOD: def detect_stagnation(graph: nx.DiGraph, 
                               recent_hashes: list[str]) -> bool
  - If last 3 state_hashes are identical → return True (UI tarpit detected)
  - If out-degree of current node == 0 and no new edges → return True
  - Return False otherwise

METHOD: async def compress_memory(execution_log: list[str],
                                    ollama_url: str) -> str
  - Call Ollama /api/generate with model "qwen2.5:7b"
  - Prompt: "Summarize completed sub-goals in ≤3 sentences: {log_tail[-10:]}"
  - Return compressed string for state.memory_summary

──────────────────────────────────────────
FILE 3: src/core/contract_skill.py
──────────────────────────────────────────
Implement async class ContractSkillRepository:

INIT: 
  - self.skills: dict[str, ContractSkillArtifact] = {}
  - self.cache: dict[str, list[ActionStep]] = {}  # URL-hash → steps cache

METHOD: def compile_from_trajectory(goal: str,
                                     steps: list[ActionStep],
                                     pre: list[str],
                                     post: list[str]) -> ContractSkillArtifact
  - Create ContractSkillArtifact with UUID4 skill_id
  - Add default RecoveryRule for each of: NOT_FOUND, TIMEOUT, INPUT_INVALID
  - Store in self.skills[skill_id]
  - Return artifact

METHOD: def apply_patch(artifact: ContractSkillArtifact,
                         failed_step_idx: int,
                         operator: str,
                         patch_payload: str) -> ContractSkillArtifact
  - Implement each operator:
    SelReplace → replace action_steps[failed_step_idx].locator_value with patch_payload
    PreInsert  → insert new ActionStep at index failed_step_idx (shift rest down)
    ArgCorrect → update action_steps[failed_step_idx].payload with patch_payload
    PostInsert → insert new ActionStep at index failed_step_idx + 1
  - Increment artifact.version
  - Set artifact.status = "patched"
  - DO NOT rewrite business logic — only structural locator/arg changes
  - Return patched artifact

METHOD: def get_cache_key(url: str) -> str
  - Return SHA-256(url)[:16]

METHOD: def check_cache(url: str) -> list[ActionStep] | None
  - Return cached steps or None

METHOD: def write_cache(url: str, steps: list[ActionStep]) -> None
  - Store in self.cache

──────────────────────────────────────────
FILE 4: src/agents/planner_agent.py
──────────────────────────────────────────
Implement async def planner_node(state: AgentPipelineState, 
                                  config: dict) -> dict:

LOGIC:
1. If stagnation detected (via SFGEngine.detect_stagnation()):
   - Set strategy prompt to INCLUDE: "ESCAPE CURRENT UI TARPIT. 
     Use browser_back or navigate to alternative path. 
     Do NOT repeat previous actions."
   - Use temperature=0.9 for creative escape

2. Else normal planning:
   - Build prompt using:
     "Goal: {state.task_goal}
      Current URL: {state.current_url}
      Memory: {state.memory_summary}
      Snapshot (trimmed): {state.sfg_nodes[-1].accessibility_snapshot}
      Generate a 3-5 step Strategy in plain English to achieve the goal."
   - Use temperature=0.3

3. Call Ollama via httpx.AsyncClient:
   POST http://localhost:11434/api/generate
   {
     "model": "qwen2.5:7b",
     "prompt": <built above>,
     "stream": false,
     "options": {"temperature": <above>, "num_ctx": 4096}
   }

4. Parse response.json()["response"]

5. Return {"execution_log": [f"[PLANNER] Strategy: {strategy[:100]}..."],
           "mode": "execute"}

──────────────────────────────────────────
FILE 5: src/agents/generator_agent.py
──────────────────────────────────────────
Implement async def generator_node(state: AgentPipelineState,
                                    config: dict) -> dict:

STEP A — Build Constrained JSON Prompt:
  Use this EXACT system instruction to prevent hallucinations:
  """
  You are a Playwright Action Generator. 
  Output ONLY a valid JSON array. NO markdown. NO explanation.
  Schema per element:
  {
    "step_index": int,
    "intent": str,
    "locator_strategy": "getByRole"|"getByLabel"|"getByTestId",
    "locator_value": str,
    "action_type": "click"|"fill"|"select"|"navigate"|"press",
    "payload": str|null
  }
  Rules:
  - NEVER use CSS classes or XPath
  - NEVER use page.locator() with structural selectors
  - Max 8 steps per batch
  - Group independent fill actions into sequential steps (Bulk Action pattern)
  """

STEP B — Call Ollama:
  model: "qwen2.5:7b", temperature: 0.1, stream: false

STEP C — Parse & Validate:
  - Strip any markdown fences (```json ... ```) before parsing
  - json.loads() the response
  - Validate each item against ActionStep Pydantic model
  - If validation fails → return {"mode": "heal", "failed_step_index": 0}

STEP D — Check ContractSkill Cache:
  - If ContractSkillRepository.check_cache(state.current_url) is not None:
    - Use cached steps directly, skip LLM call
    - Log "[GENERATOR] Cache HIT - bypassing inference"

STEP E — Compile ContractSkill:
  - Call ContractSkillRepository.compile_from_trajectory()
  - Return {
      "active_contract": new_contract,
      "execution_log": ["[GENERATOR] Contract compiled: {skill_id}"],
      "mode": "execute"
    }

──────────────────────────────────────────
FILE 6: src/agents/healer_agent.py
──────────────────────────────────────────
Implement async def healer_node(state: AgentPipelineState,
                                 config: dict) -> dict:

CONSTRAINT: Healer MUST NOT change business logic. 
            ONLY fix: locator values, arg payloads, insert missing precondition steps.
            If business logic change is detected → HALT and set mode="human_review"

STEP A — Diagnose failure:
  failed_step = state.active_contract.action_steps[state.failed_step_index]
  Build diagnosis prompt:
  """
  A Playwright step failed. Diagnose and output ONLY JSON repair instruction.
  Schema: {"operator": "SelReplace"|"PreInsert"|"ArgCorrect"|"PostInsert",
           "patch_payload": str,
           "reason": str}
  Failed step: {failed_step.model_dump()}
  Current snapshot: {state.sfg_nodes[-1].accessibility_snapshot[:2000]}
  Failure condition: {state.active_contract.recovery_rules}
  """

STEP B — Call Ollama:
  model: "qwen2.5:7b", temperature: 0.0 (deterministic)

STEP C — Validate & Apply Patch:
  - Parse JSON response (strip markdown fences first)
  - Validate operator is one of 4 allowed values
  - Guard: if "reason" contains words ["logic","requirement","business","flow"] 
    → set mode="human_review", log WARNING, return early
  - Call ContractSkillRepository.apply_patch()

STEP D — Increment repair_attempts:
  - If state.repair_attempts >= 3:
    → set mode="human_review"
    → log "[HEALER] Max retries exceeded. Human intervention required."
  - Else: set mode="execute"

STEP E — Return:
  {
    "active_contract": patched_contract,
    "repair_attempts": state.repair_attempts + 1,
    "execution_log": [f"[HEALER] Patch applied: {operator} on step {idx}"],
    "mode": <determined above>
  }

──────────────────────────────────────────
FILE 7: src/graph/langgraph_pipeline.py
──────────────────────────────────────────
Wire the complete async LangGraph pipeline.

IMPORTS: All 3 agent node functions, SFGEngine, ContractSkillRepository,
         AgentPipelineState, StateGraph, START, END, InMemorySaver

BUILD GRAPH:

1. Define router function route_after_generator(state) -> str:
   - "heal" if state.mode == "heal"
   - "end"  if state.mode == "execute" and state.active_contract is not None
   - "planner" by default (re-plan)

2. Define router function route_after_healer(state) -> str:
   - "human_review" if state.mode == "human_review"  → map to END
   - "executor"     if state.mode == "execute"
   - "healer"       if state.repair_attempts < 3      (retry same healer)

3. Add EXECUTOR NODE as inline async function execute_contract_node:
   - Use async Playwright (async_playwright)
   - Launch chromium with headless=False (stealth: non-headless)
   - For each ActionStep in active_contract.action_steps:
     * Build locator using strategy:
       getByRole → page.get_by_role(role, name=value)
       getByLabel → page.get_by_label(value)
       getByTestId → page.get_by_test_id(value)
     * Execute action (click/fill/select/press/navigate)
     * On Playwright exception:
       - Set state.failed_step_index = step.step_index
       - Set state.mode = "heal"
       - Return partial state immediately
   - After all steps: verify postconditions via page.url and page.get_by_text()
   - Call SFGEngine.capture_gui_state() and append to sfg_nodes
   - Call ContractSkillRepository.write_cache(url, steps)
   - Return {"mode": "execute", "final_result": "SUCCESS"}

4. SAFETY GUARD in executor (Deterministic Safety Boundary):
   BEFORE executing any ActionStep:
   - Check action_type + payload for FORBIDDEN keywords:
     ["delete","drop","transfer","password","admin","sudo","truncate"]
   - If found → raise RuntimeError(f"SAFETY HALT: forbidden action detected: {payload}")
   - Log the halt event

5. Add nodes: "planner", "generator", "healer", "executor"
6. Add edges:
   START → "planner"
   "planner" → "generator"
   "generator" → conditional(route_after_generator)
   "healer" → conditional(route_after_healer)
   "executor" → conditional(route_after_generator)  # re-verify or end

7. Compile with InMemorySaver checkpointer

8. Add async def run_pipeline(goal: str, start_url: str) -> str:
   - Build initial AgentPipelineState
   - Use thread_id = f"skill-{uuid4()}"
   - async for chunk in graph.astream(initial_state, config):
       print(f"[NODE OUTPUT] {chunk}")
   - Return final state["final_result"]

9. Add if __name__ == "__main__" entrypoint:
   import asyncio
   asyncio.run(run_pipeline(
       goal="Navigate to login page and fill in test credentials",
       start_url="http://localhost:3000"
   ))

═══════════════════════════════════════════════════════════════
SECTION 4: QUALITY RULES (ENFORCE ON EVERY FILE)
═══════════════════════════════════════════════════════════════

1. Every async function uses httpx.AsyncClient with timeout=30.0
2. Every Pydantic model has model_config = ConfigDict(strict=True, extra="forbid")
3. Every JSON parse from LLM output MUST strip markdown fences first:
   text.strip().removeprefix("```json").removesuffix("```").strip()
4. Accessibility snapshot MUST be truncated to ≤12000 chars before storage
5. All file paths use pathlib.Path (NO hardcoded Linux paths)
6. Rich console used for all terminal output:
   from rich.console import Console; console = Console()
7. NO placeholder comments like "# ... rest of code". Full code only.

═══════════════════════════════════════════════════════════════
SECTION 5: DELIVERABLES
═══════════════════════════════════════════════════════════════

After all 7 files, output ONE bash block:

pip install langgraph pydantic playwright lancedb httpx networkx \
            scikit-fuzzy numpy rich python-dotenv
playwright install chromium

Then output final validation checklist:
[ ] All 7 files exist with correct paths
[ ] No Docker/WSL references
[ ] No OpenAI/external API calls
[ ] All LLM calls → http://localhost:11434
[ ] All JSON parse has markdown-fence stripper
[ ] Safety guard blocks forbidden keywords
[ ] Stagnation detector returns True on 3 identical hashes
[ ] Healer halts on business-logic change detection
[ ] Cache bypass works on repeated URL visits