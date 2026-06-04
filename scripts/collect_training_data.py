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
import random
import sys
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
        from src.codetest.js_generator import _build_prompt
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
        # Extract func_name from filename:  test_test_sum.test.ts → stem=test_test_sum → func_name=sum
        stem = test_file.stem  # removes the last suffix (.ts), keeping test_test_sum.test
        # stem is e.g. "test_test_sum.test" — remove the .test part too
        # Actually pathlib.Path.stem only removes the last suffix:
        # "test_test_sum.test.ts".stem → "test_test_sum.test"
        # We need the stem without any suffix
        name_no_ext = test_file.name
        # Strip the double extension: .test.ts
        for ext in (".test.ts", ".spec.ts"):
            if name_no_ext.endswith(ext):
                name_no_ext = name_no_ext[: -len(ext)]
                break
        func_name = name_no_ext.removeprefix("test_test_")

        # Read the completion (file contents)
        try:
            completion = test_file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"collect_sprint9: cannot read {test_file}: {exc}; skipping")
            continue

        # Build prompt
        spec = spec_map.get(func_name)
        if spec is not None:
            try:
                prompt_body = _build_prompt(spec)
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


def collect_sprint6() -> list[TrainingExample]:
    """Mine src/codetest/generator.py generation logs for pytest-passing pairs."""
    print("collect_sprint6: not yet implemented; returning []")
    return []


def collect_sprint13() -> list[TrainingExample]:
    """Mine src/fuzzer/schema_inferrer.py logs for (traces, schema) pairs."""
    print("collect_sprint13: not yet implemented; returning []")
    return []


def collect_sprint14() -> list[TrainingExample]:
    """Mine src/pbt/invariant_extractor.py logs for (function_spec, invariant) pairs."""
    print("collect_sprint14: not yet implemented; returning []")
    return []


def collect_sprint5() -> list[TrainingExample]:
    """Mine Sprint 5 exploration logs for (page_state, hypothesis) pairs."""
    print("collect_sprint5: not yet implemented; returning []")
    return []


def main() -> None:
    examples: list[TrainingExample] = []
    for collector in [collect_sprint6, collect_sprint9, collect_sprint13, collect_sprint14]:
        examples.extend(collector())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    print(f"Collected {len(examples)} examples → {OUTPUT_PATH}")
    print("Run scripts/validate_dataset.py to check quality before fine-tuning.")


if __name__ == "__main__":
    main()
