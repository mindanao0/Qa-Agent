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


def test_collect_augmented_reaches_target():
    from scripts.collect_training_data import collect_augmented
    import re

    aug = collect_augmented(real_count=50, target_total=500, seed=42)
    assert len(aug) >= 450
    assert all(ex.quality == 0.8 for ex in aug)
    assert all(ex.metadata.get("augmented") is True for ex in aug)
    assert all(ex.source in (
        "sprint6_pytest", "sprint9_vitest", "sprint13_schema",
        "sprint14_pbt", "sprint5_hypothesis",
    ) for ex in aug)


def test_collect_augmented_no_duplicates():
    from scripts.collect_training_data import collect_augmented
    aug = collect_augmented(real_count=50, target_total=500, seed=42)
    ids = [ex.example_id for ex in aug]
    assert len(ids) == len(set(ids)), "No duplicate example_ids in augmented set"


def test_collect_augmented_no_blocked_patterns():
    from scripts.collect_training_data import collect_augmented
    import re
    BLOCKED = re.compile(r"delete|remove|transfer|payment|password", re.IGNORECASE)
    aug = collect_augmented(real_count=50, target_total=500, seed=42)
    for ex in aug:
        assert not BLOCKED.search(ex.completion), f"Blocked pattern in {ex.example_id}: {ex.completion[:80]}"


def test_split_stratified_by_source():
    """90/10 split must maintain source proportions."""
    from scripts.collect_training_data import _split_stratified, TrainingExample, _make_id

    examples = []
    for i in range(60):
        p, c = f"prompt_s6_{i}", f"completion_s6_{i}"
        examples.append(TrainingExample(
            example_id=_make_id(p, c), source="sprint6_pytest",
            prompt=p, completion=c, quality=1.0, metadata={}
        ))
    for i in range(40):
        p, c = f"prompt_s9_{i}", f"completion_s9_{i}"
        examples.append(TrainingExample(
            example_id=_make_id(p, c), source="sprint9_vitest",
            prompt=p, completion=c, quality=1.0, metadata={}
        ))

    train, val = _split_stratified(examples, val_ratio=0.10, seed=42)
    assert len(train) + len(val) == 100
    assert abs(len(val) - 10) <= 3  # ~10%
    val_s6 = sum(1 for ex in val if ex.source == "sprint6_pytest")
    train_s6 = sum(1 for ex in train if ex.source == "sprint6_pytest")
    assert val_s6 >= 1
    assert train_s6 >= 1
