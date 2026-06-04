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
