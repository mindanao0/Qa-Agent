from __future__ import annotations

from src.universal_qa.models import TestResult


class TerminalReporter:
    """Prints real-time pass/fail output to stdout."""

    def report_one(self, result: TestResult) -> None:
        status = "PASS" if result.passed else "FAIL"
        duration_s = result.duration_ms / 1000
        tc = result.test_case
        print(f"[{status}] {tc.type:<13} {tc.title:<55} ({duration_s:.1f}s)")
        if not result.passed:
            if result.failure_reason:
                print(f"       → {result.failure_reason}")
            for trace in result.steps_trace:
                if trace.status == "failed":
                    print(f"       → Step failed: {trace.step}")
                    if trace.error:
                        print(f"         Error: {trace.error}")

    def report_summary(self, results: list[TestResult]) -> None:
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total = len(results)
        print("\n" + "=" * 70)
        print(f"  SUMMARY  total={total}  passed={passed}  failed={failed}")
        if failed > 0:
            print(f"\n  Failed tests:")
            for r in results:
                if not r.passed:
                    print(f"    - [{r.test_case.type}] {r.test_case.title}")
                    if r.failure_reason:
                        print(f"      {r.failure_reason}")
        print("=" * 70)


__all__ = ["TerminalReporter"]
