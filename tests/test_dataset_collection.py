import pathlib
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def test_collect_sprint9_returns_training_examples(tmp_path):
    # Create fake js_tests dir with one .test.ts file
    js_tests_dir = tmp_path / "js_tests"
    js_tests_dir.mkdir()
    (js_tests_dir / "test_test_sum.test.ts").write_text(
        "function sum(arr) { return arr.reduce((a, b) => a + b, 0); }\n"
        'import { describe, it, expect } from "vitest";\n'
        'describe("sum", () => { it("adds", () => { expect(sum([1,2])).toBe(3); }); });'
    )
    # Create fake js_targets dir
    js_targets_dir = tmp_path / "js_targets"
    js_targets_dir.mkdir()
    (js_targets_dir / "utils.ts").write_text(
        "export function sum(arr: number[]): number { return arr.reduce((a, b) => a + b, 0); }"
    )

    from scripts.collect_training_data import _collect_sprint9_from_paths

    examples = _collect_sprint9_from_paths(js_tests_dir, js_targets_dir)
    assert len(examples) >= 1
    ex = examples[0]
    assert ex.source == "sprint9_vitest"
    assert "sum" in ex.prompt
    assert "vitest" in ex.completion.lower() or "describe" in ex.completion.lower()
    assert ex.quality == 1.0
    assert len(ex.example_id) == 10
    assert ex.metadata["sprint"] == 9
    assert ex.metadata["func_name"] == "sum"
    assert ex.metadata["test_type"] == "vitest"


def test_collect_sprint6_filters_quality():
    """Verify quality=1.0 assigned and example_id is set correctly."""
    from scripts.collect_training_data import TrainingExample, _make_id

    prompt = "Generate a pytest test for the following Python function:\nfunc_name=foo"
    completion = "def test_foo():\n    assert foo() is None"
    ex = TrainingExample(
        example_id=_make_id(prompt, completion),
        source="sprint6_pytest",
        prompt=prompt,
        completion=completion,
        quality=1.0,
        metadata={"sprint": 6, "func_id": "abc123", "func_name": "foo", "test_type": "happy_path"},
    )
    assert ex.quality == 1.0
    assert ex.source == "sprint6_pytest"
    assert len(ex.example_id) == 10
    assert ex.metadata["sprint"] == 6
    assert ex.metadata["func_name"] == "foo"


def test_collect_sprint13_prompt_format():
    """Verify prompt/completion format for schema inference examples."""
    import json
    from scripts.collect_training_data import TrainingExample, _make_id

    traces = [{"trace_id": "t1", "ui_action": "browse", "endpoint_hint": "/api/articles",
               "method": "GET", "request_body": None, "response_status": 200,
               "response_body": {"articles": []}, "auth_present": False, "captured_at": 0.0}]
    schema = {"endpoint": "/api/articles", "method": "GET",
              "request_schema": {}, "response_schema": {"type": "object"},
              "constraints": [], "coverage_score": 1, "is_candidate": True}

    traces_json = json.dumps(traces, indent=2)
    schema_json = json.dumps(schema, indent=2)
    prompt = f"Infer an OpenAPI schema from these API traces:\n{traces_json}"

    ex = TrainingExample(
        example_id=_make_id(prompt, schema_json),
        source="sprint13_schema",
        prompt=prompt,
        completion=schema_json,
        quality=1.0,
        metadata={"sprint": 13, "endpoint": "/api/articles", "method": "GET"},
    )
    assert ex.source == "sprint13_schema"
    assert "API traces" in ex.prompt
    assert "OpenAPI" in ex.prompt
    assert ex.quality == 1.0
    assert ex.metadata["sprint"] == 13
    assert ex.metadata["endpoint"] == "/api/articles"


def test_collect_sprint14_quality_values():
    """Invariant examples must have quality 0.5 or 1.0 based on known-passing set."""
    import json
    from scripts.collect_training_data import TrainingExample, _make_id, _SPRINT14_PASSED

    spec_dict = {"func_name": "_words_relate", "func_id": "abc123", "module_path": "src/fuzzer/vector_generator.py",
                 "return_type": "bool", "args": [], "docstring": None, "decorators": [], "complexity": 1,
                 "class_name": None, "is_async": False}
    inv_dict = {"invariant_id": "abc1234567", "source": "function",
                "source_id": "_words_relate", "description": "commutative",
                "property_type": "commutative", "hypothesis_strategy": "st.text()"}

    prompt = f"Extract a testable invariant from this function:\n{json.dumps(spec_dict, indent=2)}"
    completion = json.dumps(inv_dict, indent=2)

    # Known-passing function → quality=1.0
    ex1 = TrainingExample(
        example_id=_make_id(prompt, completion),
        source="sprint14_pbt",
        prompt=prompt,
        completion=completion,
        quality=1.0,
        metadata={"sprint": 14, "func_name": "_words_relate",
                  "property_type": "commutative", "invariant_id": "abc1234567"},
    )
    assert ex1.quality == 1.0
    assert "_words_relate" in _SPRINT14_PASSED

    # Unknown function → quality=0.5
    ex2 = TrainingExample(
        example_id=_make_id(prompt + "x", completion),
        source="sprint14_pbt",
        prompt=prompt,
        completion=completion,
        quality=0.5,
        metadata={"sprint": 14, "func_name": "unknown_func",
                  "property_type": "commutative", "invariant_id": "def9876543"},
    )
    assert ex2.quality == 0.5
    assert "unknown_func" not in _SPRINT14_PASSED
    assert ex2.source == "sprint14_pbt"
    assert ex2.metadata["sprint"] == 14


def test_collect_sprint5_hypothesis_format():
    """Hypothesis completions must be valid JSON with required fields."""
    import json
    from scripts.collect_training_data import TrainingExample, _make_id

    ax_summary = "TodoMVC app: 0 todos, filter=All, no completed"
    hyp = {
        "hypothesis_id": "abc123def456",
        "goal": "Add a todo item",
        "start_url": "https://demo.playwright.dev/todomvc/#/",
        "preconditions": [],
        "steps": ["Fill 'What needs to be done?' with 'Buy milk'", "Press Enter"],
        "expected_outcome": "Todo 'Buy milk' appears in list",
        "source_skill_id": None,
        "confidence": 0.0,
    }
    prompt = f"Generate test hypotheses for this web app state:\n{ax_summary}"
    completion = json.dumps(hyp, indent=2)
    ex = TrainingExample(
        example_id=_make_id(prompt, completion),
        source="sprint5_hypothesis",
        prompt=prompt,
        completion=completion,
        quality=1.0,
        metadata={"sprint": 5, "hypothesis_id": hyp["hypothesis_id"], "goal": hyp["goal"]},
    )
    assert ex.source == "sprint5_hypothesis"
    assert "web app state" in ex.prompt
    assert ex.quality == 1.0
    assert ex.metadata["sprint"] == 5
    assert ex.metadata["goal"] == "Add a todo item"
    parsed = json.loads(ex.completion)
    assert "goal" in parsed
    assert "steps" in parsed
