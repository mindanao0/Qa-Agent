# Sprint 9: JS/TS Code Testing — AST-aware test generation for JavaScript/TypeScript
# Project: D:\Code\qa-agent
# Baseline: all 8 sprints green, pass_rate=1.00, audit_chain_valid=true
# Goal: Extend code testing pipeline to JS/TS source files using
#       Babel/TypeScript AST parser → generate Jest/Vitest suites → execute

## ACCEPTANCE GATE (sprint9)
#   js_functions_parsed        ≥ 10
#   tests_generated            ≥ 10
#   test_pass_rate             ≥ 0.70
#   metamorphic_pairs          ≥ 3
#   otel_spans_emitted         ≥ 5    (reuse Sprint 8 OTelTracer)
#   REGRESSION if pass_rate    < 0.75

## ARCHITECTURE — New files:

### src/codetest/js_ast_parser.py
# JSASTParser — calls Node.js via subprocess to parse JS/TS with @babel/parser
# Python side orchestrates; Node.js side does the AST walk
#
# class JSFunctionSpec(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     func_id:       str         # sha256[:10] of file_path+func_name
#     module_path:   str         # relative path e.g. "src/utils/helpers.js"
#     func_name:     str
#     params:        list[str]   # parameter names (no type info for JS)
#     return_type:   str | None  # TypeScript annotation if present
#     is_async:      bool
#     is_exported:   bool
#     jsdoc:         str | None
#     complexity:    int         # McCabe via branch count in AST
#
# class JSASTParser:
#     """No required constructor args."""
#
#     async def parse_file(self, path: pathlib.Path) -> list[JSFunctionSpec]:
#         """
#         1. Write path content to tmp .js/.ts file
#         2. subprocess.run(["node", "scripts/ast_walker.js", tmp_file], timeout=15)
#         3. Parse JSON output → list[JSFunctionSpec]
#         SecurityASTChecker equivalent: validate node script output is JSON only
#         """
#
#     async def parse_dir(
#         self,
#         root: pathlib.Path,
#         glob: str = "**/*.{js,ts}",
#         exclude: tuple[str,...] = ("node_modules", "dist", ".next"),
#     ) -> list[JSFunctionSpec]:
#         """Glob files, call parse_file() concurrently (asyncio.gather, cap=4)"""

### scripts/ast_walker.js
# Node.js script — called by JSASTParser subprocess
# Dependencies: @babel/parser, @babel/traverse (install via npm)
#
# Usage: node scripts/ast_walker.js <filepath>
# Output: JSON array of function objects to stdout
# Stderr: error messages only
#
# Logic:
#   1. fs.readFileSync(filepath)
#   2. @babel/parser.parse(code, {plugins:["typescript","jsx"], sourceType:"module"})
#   3. @babel/traverse → visit FunctionDeclaration, ArrowFunctionExpression,
#      FunctionExpression, MethodDefinition
#   4. For each: extract name, params, returnType, isAsync, isExported,
#      leading JSDoc comment, McCabe complexity
#   5. JSON.stringify(results) → stdout
#   6. process.exit(0)

### src/codetest/js_generator.py
# JSTestGenerator — Ollama → Jest/Vitest test code
#
# async def generate(specs: list[JSFunctionSpec]) -> list[GeneratedJSTest]
#   — asyncio.Semaphore(1) on Ollama
#   — temp=0.2, format="json"
#   — system prompt rules:
#       "Use Vitest syntax: import {describe,it,expect} from 'vitest'"
#       "Never use require() — use ES module import"
#       "All async tests must use async/await, never .then()"
#       "No hardcoded setTimeout or setInterval in tests"
#       "Use absolute imports matching the module_path field"
#
# class GeneratedJSTest(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     test_id:              str
#     func_id:              str
#     test_code:            str    # complete Vitest test as string
#     test_type:            Literal["happy_path","edge_case","metamorphic"]
#     metamorphic_relation: str | None

### src/codetest/js_judge.py
# JSCodeJudge — 4-check judge for JS/TS tests (reuse pattern, new checks)
# Checks (exactly 4):
#   1. has_expect       — test contains at least one expect() call
#   2. no_settimeout    — no setTimeout/setInterval hardcoded in test body
#   3. valid_vitest_sig — test wrapped in it() or test() block
#   4. metamorphic_valid — if type=metamorphic, relation is falsifiable
#                          (same unbounded always/never check from Sprint 6)
# grade: "acceptable" | "needs_revision" | "reject"
# max 2 revision loops per test

### src/codetest/js_executor.py
# JSTestExecutor — runs generated Vitest tests via subprocess
#
# async def run(tests: list[GeneratedJSTest], work_dir: pathlib.Path) -> list[JSTestResult]
#   1. Write each test_code to work_dir/test_{test_id}.test.ts
#   2. subprocess.run(
#        ["npx", "vitest", "run", "--reporter=json", str(work_dir)],
#        timeout=60, cwd=work_dir
#      )
#   3. Parse Vitest JSON reporter output → pass/fail per test
#   4. Wrap OTelTracer.span("js_test.execute") around subprocess call
#
# class JSTestResult(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     test_id:   str
#     func_id:   str
#     passed:    bool
#     error:     str | None
#     duration_ms: float

### scripts/setup_vitest.py
# One-time setup script — initialise package.json + install deps
#
# pathlib.Path("package.json").write_text(json.dumps({
#     "type": "module",
#     "scripts": {"test": "vitest run"},
#     "devDependencies": {
#         "vitest": "^1.6.0",
#         "@babel/parser": "^7.24.0",
#         "@babel/traverse": "^7.24.0",
#         "@types/node": "^20.0.0"
#     }
# }, indent=2))
# subprocess.run(["npm", "install"], timeout=120)

### audit/sprint9/measure_sprint9.py
# Target JS/TS files — use existing project sources or generate stubs:
TARGET_FILES = [
    "src/js_targets/utils.ts",       # create: 5 pure utility functions
    "src/js_targets/validators.ts",  # create: 3 input validators
    "src/js_targets/formatters.ts",  # create: 4 string/date formatters
]
# If targets don't exist, generate them with realistic TS content first.
#
# Flow:
#   1. JSASTParser.parse_dir("src/js_targets/")
#   2. JSTestGenerator.generate(specs)
#   3. JSCodeJudge.judge() — filter rejects
#   4. JSTestExecutor.run(accepted_tests)
#   5. OTelTracer spans wrapping steps 1–4
#   6. Write sprint9_results.json

## OUTPUT audit/sprint9/sprint9_results.json:
{
  "js_functions_parsed":   <int>,
  "tests_generated":       <int>,
  "test_pass_rate":        <float>,
  "metamorphic_pairs":     <int>,
  "otel_spans_emitted":    <int>,
  "regression":            <bool>,
  "sprint9_status":        "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere (no os.path string concat)
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP only (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS in execution layer
# - SecurityASTChecker equivalent for Node.js subprocess output
# - subprocess timeouts: Node.js=15s, Vitest=60s
# - OTelTracer.span() wrapping all 4 pipeline stages
# - BFT: disabled (feature flag preserved)
# - CryptoAuditTrail.append() for each test execution result