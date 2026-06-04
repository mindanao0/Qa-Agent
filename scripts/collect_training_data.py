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
import json
import pathlib
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


def collect_sprint5() -> list[TrainingExample]:
    """Mine Sprint 5 exploration logs for (page_state, hypothesis) pairs."""
    return []


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
