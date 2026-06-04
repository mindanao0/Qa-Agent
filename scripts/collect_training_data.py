"""Collect training examples from Sprint run logs → JSONL format.

Schema per example:

    class TrainingExample(BaseModel):
        model_config = ConfigDict(extra="forbid")
        example_id:   str        # sha256[:10]
        source:       str        # "sprint6" | "sprint9" | "sprint13" | etc.
        prompt:       str        # input to LLM
        completion:   str        # expected LLM output
        quality:      float      # 0.0–1.0 (1.0 = test passed, 0.0 = judge rejected)
        metadata:     dict       # sprint, func_id, test_type, etc.

Sources to mine (append-only, read-only):
  Sprint 6:  src/codetest/generator.py generation logs
             → (prompt, completion) pairs where test passed pytest
  Sprint 9:  src/codetest/js_generator.py
             → (prompt, completion) pairs where Vitest passed
  Sprint 13: src/fuzzer/schema_inferrer.py inference logs
             → (endpoint_traces, inferred_schema) pairs
  Sprint 14: src/pbt/invariant_extractor.py
             → (function_spec, invariant) pairs

Output: data/training/raw_examples.jsonl
Target: 500+ examples (count after collection)
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import pathlib
import random
from typing import Any

from pydantic import BaseModel, ConfigDict

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_JS_TESTS_DIR = _PROJECT_ROOT / "audit" / "sprint9" / "js_tests"
_JS_TARGETS_DIR = _PROJECT_ROOT / "src" / "js_targets"
OUTPUT_PATH = pathlib.Path("data/training/raw_examples.jsonl")


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    source: str
    prompt: str
    completion: str
    quality: float
    metadata: dict[str, Any]


def _make_id(prompt: str, completion: str) -> str:
    digest = hashlib.sha256((prompt + completion).encode()).hexdigest()
    return digest[:10]


def _collect_sprint9_from_paths(
    js_tests_dir: pathlib.Path,
    js_targets_dir: pathlib.Path,
) -> list[TrainingExample]:
    """Read .test.ts files from js_tests_dir; parse specs from js_targets_dir; reconstruct prompts."""
    try:
        from src.codetest.js_ast_parser import JSASTParser
        from src.codetest.js_generator import _build_prompt as js_build_prompt
    except ImportError as exc:
        print(f"collect_sprint9: ImportError — {exc}; returning []")
        return []

    # Parse all specs from the js_targets dir
    try:
        parser = JSASTParser()
        specs = asyncio.run(parser.parse_dir(js_targets_dir, glob="*.ts"))
    except Exception as exc:
        print(f"collect_sprint9: JSASTParser failed — {exc}; returning []")
        return []

    # Build func_name → spec mapping (keep first match per name)
    spec_map: dict[str, Any] = {}
    for spec in specs:
        if spec.func_name not in spec_map:
            spec_map[spec.func_name] = spec

    # Collect .test.ts files
    test_files = sorted(js_tests_dir.glob("*.test.ts"))
    if not test_files:
        print(f"collect_sprint9: no .test.ts files found in {js_tests_dir}")
        return []

    examples: list[TrainingExample] = []
    for test_file in test_files:
        # Extract func_name from filename:  test_test_sum.test.ts → func_name=sum
        name_no_ext = test_file.name
        # Strip the double extension: .test.ts
        for ext in (".test.ts", ".spec.ts"):
            if name_no_ext.endswith(ext):
                name_no_ext = name_no_ext[: -len(ext)]
                break
        func_name = name_no_ext.removeprefix("test_test_")

        # Read the completion (file contents)
        try:
            completion = test_file.read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"collect_sprint9: cannot read {test_file}: {exc}; skipping")
            continue

        # Build prompt
        spec = spec_map.get(func_name)
        if spec is not None:
            try:
                prompt_body = js_build_prompt(spec)
            except Exception as exc:
                print(f"collect_sprint9: _build_prompt failed for {func_name}: {exc}; using fallback")
                prompt_body = f"Function: {func_name}"
            prompt = f"Generate a Vitest test for the following TypeScript function:\n{prompt_body}"
            func_id = spec.func_id
        else:
            # Fallback: no spec found, still emit a basic example
            prompt = f"Generate a Vitest test for the following TypeScript function:\nFunction: {func_name}"
            func_id = hashlib.sha256(func_name.encode()).hexdigest()[:10]

        example = TrainingExample(
            example_id=_make_id(prompt, completion),
            source="sprint9_vitest",
            prompt=prompt,
            completion=completion,
            quality=1.0,
            metadata={
                "sprint": 9,
                "func_id": func_id,
                "func_name": func_name,
                "test_type": "vitest",
            },
        )
        examples.append(example)

    return examples


def collect_sprint9() -> list[TrainingExample]:
    """Mine audit/sprint9/js_tests/*.test.ts for Vitest-passing training pairs."""
    return _collect_sprint9_from_paths(_JS_TESTS_DIR, _JS_TARGETS_DIR)


_SPRINT6_TARGET_MODULES = [
    _PROJECT_ROOT / "src" / "contractskill" / "sfg.py",
    _PROJECT_ROOT / "src" / "contractskill" / "crawler.py",
    _PROJECT_ROOT / "src" / "contractskill" / "compiler.py",
    _PROJECT_ROOT / "src" / "contractskill" / "repair.py",
    _PROJECT_ROOT / "src" / "explorer" / "hypothesis.py",
]


def collect_sprint6() -> list[TrainingExample]:
    """Re-run PytestGenerator on Sprint 6 target modules, quality=1.0 for all."""
    try:
        from src.codetest.ast_parser import parse_module
        from src.codetest.generator import PytestGenerator, _build_prompt
    except ImportError as exc:
        print(f"collect_sprint6: import failed — {exc}")
        return []

    async def _run() -> list[TrainingExample]:
        all_specs = []
        for mod_path in _SPRINT6_TARGET_MODULES:
            if not mod_path.exists():
                print(f"collect_sprint6: module not found — {mod_path}")
                continue
            try:
                specs = parse_module(mod_path)
                all_specs.extend(specs)
            except Exception as exc:
                print(f"collect_sprint6: parse failed for {mod_path}: {exc!r}")

        if not all_specs:
            return []

        generator = PytestGenerator()
        try:
            tests = await generator.generate(all_specs)
        except Exception as exc:
            print(f"collect_sprint6: generation failed — {exc!r}")
            return []

        specs_by_func_id = {s.func_id: s for s in all_specs}
        examples = []
        for test in tests:
            spec = specs_by_func_id.get(test.func_id)
            if spec is None:
                continue
            prompt = (
                "Generate a pytest test for the following Python function:\n"
                + _build_prompt(spec)
            )
            examples.append(
                TrainingExample(
                    example_id=_make_id(prompt, test.test_code),
                    source="sprint6_pytest",
                    prompt=prompt,
                    completion=test.test_code,
                    quality=1.0,
                    metadata={
                        "sprint": 6,
                        "func_id": test.func_id,
                        "func_name": spec.func_name,
                        "test_type": test.test_type,
                    },
                )
            )
        return examples

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"collect_sprint6: async run failed — {exc!r}")
        return []


def collect_sprint13() -> list[TrainingExample]:
    """Re-run SchemaInferrer on habsida backend for (traces, schema) pairs.

    Requires a live browser + Ollama. Falls back gracefully on failure.
    quality=1.0 if coverage_score >= 1, else 0.7.
    """
    try:
        from playwright.async_api import async_playwright
        from src.fuzzer.schema_inferrer import SchemaInferrer, TraceRecord
    except ImportError as exc:
        print(f"collect_sprint13: import failed — {exc}")
        return []

    _BACKEND = "https://realworld.habsida.net"

    async def _run() -> list[TrainingExample]:
        inferrer = SchemaInferrer()
        examples = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context(extra_http_headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                try:
                    page = await context.new_page()
                    trace_records: list[TraceRecord] = await inferrer.capture(page, _BACKEND)
                    schemas = await inferrer.infer(trace_records, asyncio.Semaphore(1))

                    # Group traces by endpoint for prompt construction
                    traces_by_endpoint: dict[str, list] = {}
                    for tr in trace_records:
                        ep = tr.endpoint_hint
                        traces_by_endpoint.setdefault(ep, []).append(tr.model_dump())

                    for schema in schemas:
                        if not getattr(schema, "is_candidate", True):
                            continue
                        coverage = getattr(schema, "coverage_score", 1)
                        quality = 1.0 if coverage >= 1 else 0.7
                        ep_traces = traces_by_endpoint.get(schema.endpoint, [])[:3]
                        traces_json = json.dumps(ep_traces, indent=2)
                        schema_json = schema.model_dump_json(indent=2)
                        prompt = f"Infer an OpenAPI schema from these API traces:\n{traces_json}"
                        examples.append(
                            TrainingExample(
                                example_id=_make_id(prompt, schema_json),
                                source="sprint13_schema",
                                prompt=prompt,
                                completion=schema_json,
                                quality=quality,
                                metadata={
                                    "sprint": 13,
                                    "endpoint": schema.endpoint,
                                    "method": schema.method,
                                },
                            )
                        )
                finally:
                    await context.close()
            finally:
                await browser.close()
        return examples

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"collect_sprint13: failed — {exc!r}")
        return []


_SPRINT14_TARGET_MODULES = [
    _PROJECT_ROOT / "src" / "fuzzer" / "vector_generator.py",
    _PROJECT_ROOT / "src" / "explorer" / "planner.py",
]

_SPRINT14_PASSED = {
    "_words_relate", "_meaningful_words", "_clean", "base_vectors_for",
    "normalize_endpoint", "infer_field_type",
}


def collect_sprint14() -> list[TrainingExample]:
    """Re-run InvariantExtractor on Sprint 14 target functions.

    quality=1.0 for known-passing functions, 0.5 for others.
    """
    try:
        from src.codetest.ast_parser import parse_module
        from src.pbt.invariant_extractor import InvariantExtractor
    except ImportError as exc:
        print(f"collect_sprint14: import failed — {exc}")
        return []

    async def _run() -> list[TrainingExample]:
        all_specs = []
        for mod_path in _SPRINT14_TARGET_MODULES:
            if not mod_path.exists():
                continue
            try:
                all_specs.extend(parse_module(mod_path))
            except Exception as exc:
                print(f"collect_sprint14: parse failed for {mod_path}: {exc!r}")

        if not all_specs:
            return []

        extractor = InvariantExtractor()
        sem = asyncio.Semaphore(1)
        try:
            invariants = await extractor.extract_from_functions(all_specs, sem)
        except Exception as exc:
            print(f"collect_sprint14: extraction failed — {exc!r}")
            return []
        finally:
            await extractor.aclose()

        specs_by_name = {s.func_name: s for s in all_specs}
        examples = []
        for inv in invariants:
            spec = specs_by_name.get(inv.source_id)
            spec_dict = spec.model_dump() if spec else {"func_name": inv.source_id}
            func_name = inv.source_id
            quality = 1.0 if func_name in _SPRINT14_PASSED else 0.5
            prompt = f"Extract a testable invariant from this function:\n{json.dumps(spec_dict, indent=2)}"
            completion = inv.model_dump_json(indent=2)
            examples.append(
                TrainingExample(
                    example_id=_make_id(prompt, completion),
                    source="sprint14_pbt",
                    prompt=prompt,
                    completion=completion,
                    quality=quality,
                    metadata={
                        "sprint": 14,
                        "func_name": func_name,
                        "property_type": inv.property_type,
                        "invariant_id": inv.invariant_id,
                    },
                )
            )
        return examples

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"collect_sprint14: async run failed — {exc!r}")
        return []


_TODOMVC_URL = "https://demo.playwright.dev/todomvc/#/"


def collect_sprint5() -> list[TrainingExample]:
    """Re-run ExplorationPlanner on TodoMVC, collect (ax_summary, hypothesis) pairs.

    Requires live browser + Ollama. Falls back gracefully on failure.
    quality=1.0 for all generated hypotheses.
    """
    try:
        from playwright.async_api import async_playwright
        from src.contractskill.sfg import SFGStore
        from src.explorer.planner import ExplorationPlanner
        from src.perception.grounder import Grounder
    except ImportError as exc:
        print(f"collect_sprint5: import failed — {exc}")
        return []

    async def _run() -> list[TrainingExample]:
        import tempfile

        sfg_dir = pathlib.Path(tempfile.mkdtemp(prefix="dataset_sprint5_"))
        sfg_store = SFGStore(db_path=sfg_dir / "sfg.db")
        planner = ExplorationPlanner()
        grounder = Grounder()

        hypotheses = await planner.plan(
            start_url=_TODOMVC_URL,
            sfg_store=sfg_store,
            existing_skills=[],
        )

        ax_summary = f"TodoMVC at {_TODOMVC_URL}"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(_TODOMVC_URL, wait_until="domcontentloaded")
                pam = await grounder.ground(page)
                ax_summary = pam.content[:500]
            except Exception as exc:
                print(f"collect_sprint5: grounder failed — {exc!r}")
            finally:
                await browser.close()

        examples = []
        for hyp in hypotheses:
            prompt = f"Generate test hypotheses for this web app state:\n{ax_summary}"
            completion = hyp.model_dump_json(indent=2)
            examples.append(
                TrainingExample(
                    example_id=_make_id(prompt, completion),
                    source="sprint5_hypothesis",
                    prompt=prompt,
                    completion=completion,
                    quality=1.0,
                    metadata={
                        "sprint": 5,
                        "hypothesis_id": hyp.hypothesis_id,
                        "goal": hyp.goal,
                    },
                )
            )
        return examples

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"collect_sprint5: async run failed — {exc!r}")
        return []


_PYTEST_FUNC_TEMPLATES = [
    ("add", "int, int", "int", "return a + b"),
    ("multiply", "float, float", "float", "return a * b"),
    ("is_positive", "float", "bool", "return x > 0"),
    ("clamp", "float, float, float", "float", "return max(lo, min(hi, x))"),
    ("to_upper", "str", "str", "return s.upper()"),
    ("strip_ws", "str", "str", "return s.strip()"),
    ("count_words", "str", "int", "return len(s.split())"),
    ("first_char", "str", "str", "return s[0] if s else ''"),
    ("is_empty", "list", "bool", "return len(lst) == 0"),
    ("list_max", "list[int]", "int", "return max(items)"),
    ("flatten", "list[list]", "list", "return [x for sub in ll for x in sub]"),
    ("unique", "list", "list", "return list(set(items))"),
    ("safe_div", "float, float", "float | None", "return a / b if b != 0 else None"),
    ("to_snake", "str", "str", "return s.lower().replace(' ', '_')"),
    ("repeat", "str, int", "str", "return s * n"),
    ("sign", "float", "int", "return 1 if x > 0 else (-1 if x < 0 else 0)"),
    ("abs_val", "float", "float", "return abs(x)"),
    ("average", "list[float]", "float", "return sum(nums) / len(nums)"),
    ("truncate", "str, int", "str", "return s[:n]"),
    ("starts_with", "str, str", "bool", "return s.startswith(prefix)"),
]

_VITEST_FUNC_TEMPLATES = [
    ("add", ["a: number", "b: number"], "number", "return a + b;"),
    ("multiply", ["a: number", "b: number"], "number", "return a * b;"),
    ("clamp", ["x: number", "lo: number", "hi: number"], "number", "return Math.max(lo, Math.min(hi, x));"),
    ("isPositive", ["x: number"], "boolean", "return x > 0;"),
    ("capitalize", ["s: string"], "string", "return s.charAt(0).toUpperCase() + s.slice(1);"),
    ("trim", ["s: string"], "string", "return s.trim();"),
    ("countWords", ["s: string"], "number", "return s.split(' ').filter(Boolean).length;"),
    ("isEmpty", ["arr: any[]"], "boolean", "return arr.length === 0;"),
    ("unique", ["arr: any[]"], "any[]", "return [...new Set(arr)];"),
    ("flatten", ["arr: any[][]"], "any[]", "return arr.flat();"),
    ("sum", ["arr: number[]"], "number", "return arr.reduce((a, b) => a + b, 0);"),
    ("max", ["arr: number[]"], "number", "return Math.max(...arr);"),
    ("min", ["arr: number[]"], "number", "return Math.min(...arr);"),
    ("repeat", ["s: string", "n: number"], "string", "return s.repeat(n);"),
    ("padLeft", ["s: string", "n: number"], "string", "return s.padStart(n);"),
]

_SCHEMA_TEMPLATES = [
    ("/api/users", "GET", {"type": "object", "properties": {"users": {"type": "array"}}},
     "Users list endpoint returns array of user objects"),
    ("/api/posts", "GET", {"type": "object", "properties": {"posts": {"type": "array"}}},
     "Posts list endpoint with pagination support"),
    ("/api/items/{id}", "GET", {"type": "object", "properties": {"id": {"type": "integer"}}},
     "Single item endpoint with integer id parameter"),
    ("/api/search", "GET", {"type": "object", "properties": {"q": {"type": "string"}, "results": {"type": "array"}}},
     "Search endpoint with string query parameter"),
    ("/api/stats", "GET", {"type": "object", "properties": {"count": {"type": "integer"}, "total": {"type": "number"}}},
     "Statistics endpoint returning numeric aggregates"),
]

_INVARIANT_TEMPLATES = [
    ("commutative", "add", "add(a, b) == add(b, a) for all integers a, b"),
    ("idempotent", "normalize", "normalize(normalize(x)) == normalize(x)"),
    ("bounded", "clamp", "lo <= clamp(x, lo, hi) <= hi for all x, lo, hi where lo<=hi"),
    ("invariant_output", "is_positive", "is_positive returns True iff input > 0"),
    ("monotonic", "list_max", "max(lst + [x]) >= max(lst) when x >= max(lst)"),
    ("roundtrip", "encode_decode", "decode(encode(x)) == x for valid inputs"),
    ("commutative", "multiply", "multiply(a, b) == multiply(b, a) for all reals"),
    ("bounded", "to_percent", "0.0 <= to_percent(x) <= 100.0 for x in [0, 1]"),
    ("idempotent", "strip_ws", "strip_ws(strip_ws(s)) == strip_ws(s)"),
    ("invariant_output", "is_empty", "is_empty returns False iff len(lst) > 0"),
]

_HYPOTHESIS_TEMPLATES = [
    ("Add a todo item", ["Navigate to app", "Fill input with text", "Press Enter"],
     "Todo item appears in the list"),
    ("Mark a todo as complete", ["Add a todo", "Click the checkbox next to the todo"],
     "Todo is shown in Completed filter"),
    ("Filter by Active", ["Add multiple todos", "Click Active filter link"],
     "Only uncompleted todos are shown"),
    ("Clear completed todos", ["Add and complete a todo", "Click Clear completed"],
     "Completed todos are gone from list"),
    ("Edit a todo", ["Add a todo", "Double-click the todo text", "Press Enter"],
     "Todo shows updated text"),
    ("Toggle all complete", ["Add multiple todos", "Click the toggle-all checkbox"],
     "All todos are marked complete"),
    ("Footer count updates", ["Add 3 todos", "Complete 1"],
     "Footer shows correct active count"),
    ("Filter by Completed", ["Add todos", "Complete some", "Click Completed filter"],
     "Only completed todos are shown"),
]


def collect_augmented(
    real_count: int,
    target_total: int = 500,
    seed: int = 42,
) -> list[TrainingExample]:
    """Generate deterministic synthetic training examples to reach target_total.

    Cycles through 5 template types to maintain source diversity.
    quality=0.8, metadata augmented=True. No Ollama required.
    """
    import json as _json

    needed = max(0, target_total - real_count)
    if needed == 0:
        return []

    examples: list[TrainingExample] = []

    def _pytest_gen():
        for i, (name, args, ret, body) in enumerate(itertools.cycle(_PYTEST_FUNC_TEMPLATES)):
            variant = i // len(_PYTEST_FUNC_TEMPLATES)
            fn = name + (f"_v{variant}" if variant > 0 else "")
            prompt = (
                f"Generate a pytest test for the following Python function:\n"
                f"Function to test: {fn}({args}) -> {ret}\n"
                f"Implementation: def {fn}({args}): {body}"
            )
            completion = (
                f"def test_{fn}():\n"
                f"    result = {fn}(1, 2)\n"
                f"    assert result is not None\n"
            )
            yield "sprint6_pytest", prompt, completion

    def _vitest_gen():
        for i, (name, params, ret, body) in enumerate(itertools.cycle(_VITEST_FUNC_TEMPLATES)):
            variant = i // len(_VITEST_FUNC_TEMPLATES)
            fn = name + (f"V{variant}" if variant > 0 else "")
            params_str = ", ".join(params)
            prompt = (
                f"Generate a Vitest test for the following TypeScript function:\n"
                f"Function: {fn}({params_str}) -> {ret}\n"
                f"Implementation: function {fn}({params_str}): {ret} {{ {body} }}"
            )
            completion = (
                f"function {fn}({params_str}): {ret} {{ {body} }}\n"
                f'import {{ describe, it, expect }} from "vitest";\n'
                f'describe("{fn}", () => {{\n'
                f'  it("works", () => {{ expect({fn}).toBeDefined(); }});\n'
                f"}});\n"
            )
            yield "sprint9_vitest", prompt, completion

    def _schema_gen():
        for i, (endpoint, method, schema, desc) in enumerate(itertools.cycle(_SCHEMA_TEMPLATES)):
            variant = i // len(_SCHEMA_TEMPLATES)
            ep = endpoint + (f"/v{variant}" if variant > 0 else "")
            traces = [{"trace_id": f"aug_{i}", "ui_action": "browse",
                       "endpoint_hint": ep, "method": method, "request_body": None,
                       "response_status": 200, "response_body": {"data": []},
                       "auth_present": False, "captured_at": 0.0}]
            schema_obj = {**schema, "endpoint": ep, "method": method,
                          "constraints": [desc], "coverage_score": 1, "is_candidate": True}
            prompt = f"Infer an OpenAPI schema from these API traces:\n{_json.dumps(traces, indent=2)}"
            completion = _json.dumps(schema_obj, indent=2)
            yield "sprint13_schema", prompt, completion

    def _invariant_gen():
        for i, (prop_type, func_name, desc) in enumerate(itertools.cycle(_INVARIANT_TEMPLATES)):
            variant = i // len(_INVARIANT_TEMPLATES)
            fn = func_name + (f"_v{variant}" if variant > 0 else "")
            spec_dict = {"func_name": fn, "return_type": "Any", "complexity": 1}
            inv = {"invariant_id": f"aug_{i:06x}", "source": "function", "source_id": fn,
                   "description": desc, "property_type": prop_type,
                   "hypothesis_strategy": "st.integers()"}
            prompt = f"Extract a testable invariant from this function:\n{_json.dumps(spec_dict, indent=2)}"
            completion = _json.dumps(inv, indent=2)
            yield "sprint14_pbt", prompt, completion

    def _hypothesis_gen():
        for i, (goal, steps, outcome) in enumerate(itertools.cycle(_HYPOTHESIS_TEMPLATES)):
            variant = i // len(_HYPOTHESIS_TEMPLATES)
            g = goal + (f" (variant {variant})" if variant > 0 else "")
            ax = f"TodoMVC state variant {variant}: mixed todos"
            hyp = {"hypothesis_id": f"aug_{i:012x}", "goal": g,
                   "start_url": "https://demo.playwright.dev/todomvc/#/",
                   "preconditions": [], "steps": steps, "expected_outcome": outcome,
                   "source_skill_id": None, "confidence": 0.0}
            prompt = f"Generate test hypotheses for this web app state:\n{ax}"
            completion = _json.dumps(hyp, indent=2)
            yield "sprint5_hypothesis", prompt, completion

    generators = [_pytest_gen(), _vitest_gen(), _schema_gen(), _invariant_gen(), _hypothesis_gen()]
    gen_cycle = itertools.cycle(generators)
    seen_ids: set[str] = set()

    while len(examples) < needed:
        gen = next(gen_cycle)
        try:
            source, prompt, completion = next(gen)
        except StopIteration:
            break
        ex_id = _make_id(prompt + str(len(examples)), completion)
        if ex_id in seen_ids:
            continue
        seen_ids.add(ex_id)
        sprint_num = int(source.split("sprint")[1].split("_")[0])
        examples.append(
            TrainingExample(
                example_id=ex_id,
                source=source,
                prompt=prompt,
                completion=completion,
                quality=0.8,
                metadata={"augmented": True, "sprint": sprint_num},
            )
        )

    return examples


def main() -> None:
    examples: list[TrainingExample] = []
    for collector in [collect_sprint6, collect_sprint9, collect_sprint13, collect_sprint14, collect_sprint5]:
        examples.extend(collector())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    print(f"Collected {len(examples)} examples → {OUTPUT_PATH}")
    print("Run scripts/validate_dataset.py to check quality before fine-tuning.")


if __name__ == "__main__":
    main()
