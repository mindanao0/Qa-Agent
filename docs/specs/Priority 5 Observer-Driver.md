You are an Elite AI & QA Automation Architect. Build the "Observer-Driver Multi-Agent System" as Priority 5 of the Production-Grade Universal Local AI QA Agent.

## ARCHITECTURE SOURCE (from knowledge base)
Observer-Driver Multi-Agent Pattern:
- Driver Agents: Execute write-actions (clicks, form inputs, navigation) and state transitions
- Observer Agents: Run asynchronously, consuming the trace stream in real-time to perform specialized audits (Accessibility, Security, Performance) WITHOUT disrupting Driver execution
- Pattern enforces zero state-mutation conflicts between parallel agents

## HARD CONSTRAINTS
- OS: Native Windows 11 ONLY — use pathlib.Path, no Linux paths, no Docker, no WSL
- LLM: Ollama at http://localhost:11434 ONLY (model: qwen2.5-coder or codellama)
- Hardware: 16GB RAM, 6GB VRAM — no simultaneous VLM + LLM loading
- Python 3.11+ with full async/await — NEVER use nest_asyncio
- Pydantic V2 with extra="forbid" on ALL state schemas
- LangGraph async graph ONLY — no sync .invoke(), always use .ainvoke() or .astream()
- Playwright objects are NOT thread-safe — never share browser instances across tasks

## FILE STRUCTURE TO CREATE

src/
  agents/
    observer_driver/
      __init__.py
      state.py          ← Pydantic V2 AgentState + ObserverReport schemas
      driver_agent.py   ← Async Driver: AXTree snapshot → Ollama action planning → Playwright execution
      observers/
        __init__.py
        base_observer.py       ← Abstract async BaseObserver with asyncio.Queue trace stream
        accessibility_observer.py  ← AXTree diff audit, WCAG flags
        security_observer.py       ← CSP header scan, mixed-content, form exposure check
        performance_observer.py    ← CDP timing metrics, LCP/CLS/FID capture
      coordinator.py    ← LangGraph graph wiring Driver + all 3 Observers as parallel branches
      trace_bus.py      ← asyncio.Queue-based event bus (TraceEvent dataclass, fan-out broadcast)

tests/
  test_observer_driver.py  ← pytest-asyncio integration test against https://example.com

## IMPLEMENTATION RULES

### state.py
- ObserverReport: fields = observer_name, findings: list[str], severity: Literal["info","warn","critical"], timestamp_ms: int
- DriverAction: fields = action_type, target_role, target_name, value, ax_snapshot_path: Path
- AgentState(BaseModel, extra="forbid"): 
    url: str
    driver_actions: Annotated[list[DriverAction], operator.add]
    observer_reports: Annotated[list[ObserverReport], operator.add]
    trace_events: Annotated[list[dict], operator.add]
    current_step: int = 0
    max_steps: int = 5
    halt: bool = False

### trace_bus.py
- TraceEvent dataclass: event_type (str), payload (dict), timestamp_ms (int)
- TraceBus class:
    - subscribers: list[asyncio.Queue[TraceEvent]]
    - async def publish(event: TraceEvent) → fan-out to ALL subscriber queues (non-blocking put_nowait)
    - def subscribe() → returns new asyncio.Queue[TraceEvent]
    - NEVER block publish() — use put_nowait with queue size cap = 100

### driver_agent.py
- async def capture_axtree(page) → saves YAML to Path("agent_state/ax_snapshot_{step}.yml"), returns path
- async def plan_action(ax_path: Path, step: int, ollama_url: str) → calls Ollama /api/generate with:
    model: "qwen2.5-coder"
    temperature: 0.0
    seed: 42
    prompt: structured prompt asking for JSON with keys: action_type, target_role, target_name, value
    Parse response as DriverAction via Pydantic — raise ValueError if parse fails (no retry hallucination)
- async def execute_action(page, action: DriverAction, bus: TraceBus):
    Execute Playwright action based on action_type (click/fill/navigate/assert)
    After EACH action: bus.publish(TraceEvent(event_type="driver_action", payload=action.model_dump(), timestamp_ms=...))
    Handle Shadow DOM: use page.locator("css=*").filter(has=page.get_by_role(...))

### base_observer.py
- Abstract class BaseObserver:
    def __init__(self, name: str, queue: asyncio.Queue[TraceEvent])
    async def run(self, page, stop_event: asyncio.Event) → loop: while not stop_event.is_set(): process events
    abstract async def _audit(self, event: TraceEvent, page) → ObserverReport | None
    asyncio.wait_for with timeout=2.0 on queue.get() to prevent blocking on stop

### accessibility_observer.py
- On each "driver_action" event: capture new AXTree snapshot, diff against previous snapshot
- Flag: missing aria-label, role="none" on interactive elements, tab-order violations
- Build ObserverReport(observer_name="accessibility", findings=[...], severity=...)

### security_observer.py  
- On "driver_action" events of type navigate/click:
    response = await page.request.get(page.url)
    Check headers: Content-Security-Policy, X-Frame-Options, Strict-Transport-Security
    Check for password inputs without autocomplete="off"
    Check for mixed HTTP content on HTTPS pages
- Build ObserverReport with critical/warn severity

### performance_observer.py
- Use CDP session: client = await page.context.new_cdp_session(page)
- Capture: await client.send("Performance.getMetrics")
- On navigation events: record LCP approximation via JS: page.evaluate("window.performance.timing")
- Build ObserverReport with raw metric values as findings

### coordinator.py
- Build LangGraph StateGraph(AgentState)
- Nodes:
    "driver_step": runs driver agent for 1 action, publishes to bus
    "observe_parallel": asyncio.gather(*[obs.run() for obs in observers]) with asyncio.Event for stop
    "router": conditional edge → "driver_step" if state.current_step < state.max_steps and not state.halt else "finalize"
    "finalize": aggregates all ObserverReports, logs summary
- PARALLEL EXECUTION PATTERN:
    Use asyncio.create_task() to start all Observer coroutines BEFORE driver starts
    Driver runs its step, observers consume from TraceBus queues concurrently
    After driver step completes: set stop_event → await asyncio.gather(all observer tasks)
    Collect results into state.observer_reports via Annotated reducer
- Compile with MemorySaver checkpointer (SQLite ONLY for dev per project knowledge — NOT PostgreSQL)
- thread_id config = f"observer_driver_{url_hash}"

### tests/test_observer_driver.py
- @pytest.mark.asyncio test using async_playwright
- Launch Chromium headful=False
- Navigate to https://example.com
- Run coordinator graph for max_steps=3
- Assert: len(state.driver_actions) == 3
- Assert: at least 1 ObserverReport exists from each observer type
- Assert: no observer_report has severity="critical" for example.com (clean baseline)

## INSTALL REQUIREMENTS
Create requirements_p5.txt with exact versions:
langgraph>=0.2.0
langchain-core>=0.2.0
playwright>=1.44.0
pydantic>=2.7.0
pytest-asyncio>=0.23.0
aiohttp>=3.9.0
pyyaml>=6.0.1

## CLI COMMANDS TO PROVIDE
1. pip install command
2. playwright install chromium command  
3. Command to create agent_state/ directory with pathlib
4. pytest run command with -v --asyncio-mode=auto

## OUTPUT FORMAT
- Provide each file as: # path/to/file.py followed by COMPLETE runnable code
- Zero placeholders — every function fully implemented
- All Windows paths use pathlib.Path(__file__).parent syntax
- All async functions properly typed with return annotations
- Each file starts with module-level docstring describing its role in Observer-Driver pattern