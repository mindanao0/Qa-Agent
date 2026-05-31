# Sprint 6: Python Code Testing — AI-generated pytest suites from source AST
# Project: D:\Code\qa-agent
# Baseline: pass_rate=1.00, hypotheses=6, coverage=1.00
# Goal: Agent receives a Python module → AST-parse → generate pytest → execute → report

## ACCEPTANCE GATE (sprint6)
#   ast_functions_parsed      ≥ 10
#   tests_generated           ≥ 10
#   test_pass_rate            ≥ 0.70
#   metamorphic_pairs         ≥ 3     (LLM-as-Judge validates input↔output relationships)
#   REGRESSION if pass_rate   < 0.75

## ARCHITECTURE — New files to create:

### src/codетест/ast_parser.py
# ASTParser — extracts function signatures, docstrings, type hints from .py files
#
# class FunctionSpec(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     func_id:      str        # sha256[:10] of module_path+func_name
#     module_path:  str        # relative path e.g. "src/contractskill/sfg.py"
#     func_name:    str
#     args:         list[ArgSpec]
#     return_type:  str | None
#     docstring:    str | None
#     decorators:   list[str]
#     complexity:   int        # McCabe cyclomatic complexity
#
# class ArgSpec(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     name:         str
#     annotation:   str | None
#     default:      str | None
#
# def parse_module(path: pathlib.Path) -> list[FunctionSpec]
#   — use ast.parse() + ast.walk()
#   — skip private funcs (_name) unless decorated @property
#   — compute McCabe: count (If + For + While + With + ExceptHandler) + 1

### src/codetest/generator.py
# PytestGenerator — Ollama qwen2.5-coder:7b → pytest code
#
# async def generate(specs: list[FunctionSpec]) -> list[GeneratedTest]
#   — asyncio.Semaphore(1) on Ollama
#   — temp=0.2, format="json"
#   — system prompt: enforce pytest style, no mocks unless annotated
#   — output schema (Pydantic V2, extra="forbid"):
#
# class GeneratedTest(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     test_id:      str
#     func_id:      str        # links back to FunctionSpec
#     test_code:    str        # complete pytest function as string
#     test_type:    Literal["happy_path","edge_case","metamorphic"]
#     metamorphic_relation: str | None  # e.g. "f(x+1) > f(x) for positive x"

### src/codetest/judge.py
# CodeJudge — 4-check LLM-as-Judge (reuse pattern from Sprint 3, new checks)
# Checks (exactly 4, no more for 7B):
#   1. has_assert         — test contains at least one assert statement
#   2. no_hardcoded_sleep — no time.sleep() calls
#   3. valid_pytest_sig   — function name starts with test_
#   4. metamorphic_valid  — if type=metamorphic, relation is logically sound
#
# grade: "acceptable" | "needs_revision" | "reject"
# max 2 revision loops per test

### src/codetest/executor.py
# TestExecutor — runs generated pytest code in subprocess (sandboxed)
# Uses SecurityASTChecker pattern from knowledge base:
#   — block imports: os, sys, subprocess, ctypes (except pytest itself)
#   — block eval/exec
#   — run via: subprocess.run(["uv", "run", "pytest", tmp_file, "-v", "--tb=short"])
#   — timeout: 30s per test file
#   — parse stdout → pass/fail/error per test

### audit/sprint6/measure_sprint6.py
# Target modules to parse (use existing Sprint 4+5 source):
TARGET_MODULES = [
    "src/contractskill/sfg.py",
    "src/contractskill/grounder.py",
    "src/contractskill/compiler.py",
    "src/contractskill/repair.py",
    "src/explorer/hypothesis.py",
]
#
# Flow:
#   1. ASTParser.parse_module() on each target → collect FunctionSpecs
#   2. PytestGenerator.generate(specs) → GeneratedTest list
#   3. CodeJudge.judge() on each → filter grade != "reject"
#   4. TestExecutor.run() on accepted tests
#   5. Write audit/sprint6/sprint6_results.json

## OUTPUT FORMAT audit/sprint6/sprint6_results.json:
{
  "ast_functions_parsed":  <int>,
  "tests_generated":       <int>,
  "test_pass_rate":        <float>,
  "metamorphic_pairs":     <int>,
  "regression":            <bool>,
  "sprint6_status":        "PASS" | "FAIL"
}

## RULES (carry-forward — no exceptions):
# - asyncio.Semaphore(1) on ALL Ollama calls
# - pathlib.Path everywhere (no os.path string concat)
# - Pydantic V2 ConfigDict(extra="forbid")
# - CDP/AOMExtractor (no page.accessibility)
# - Judge: 4 checks only
# - BLOCKED_ACTION_PATTERNS: delete/remove/transfer/payment/password
# - SecurityASTChecker before any subprocess execution
# - subprocess timeout=30s hard limit
# - BFT: disabled (feature flag preserved)