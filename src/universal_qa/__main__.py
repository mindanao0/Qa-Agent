# src/universal_qa/__main__.py
import argparse
import asyncio
import json
import pathlib

from src.universal_qa.agent import UniversalQAAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal QA Agent — test any website")
    parser.add_argument("--url", required=True, help="Website URL to test")
    parser.add_argument("--username", default=None, help="Login username/email (optional)")
    parser.add_argument("--password", default=None, help="Login password (optional)")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages to crawl (default 50)")
    parser.add_argument("--explore-timeout", type=int, default=5,
                        help="Exploration timeout in minutes (default 5)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max navigation depth (default 4)")
    parser.add_argument("--allow-destructive", action="store_true", default=False,
                        help="Allow clicking delete/remove/payment actions")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run browser in headless mode (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser window")
    parser.add_argument("--detail-output", default=None,
                        help="Save detailed results JSON to this path (for training data)")
    args = parser.parse_args()

    agent = UniversalQAAgent(
        url=args.url,
        username=args.username,
        password=args.password,
        max_pages=args.max_pages,
        explore_timeout=args.explore_timeout,
        max_depth=args.max_depth,
        allow_destructive=args.allow_destructive,
        headless=args.headless,
    )

    results = asyncio.run(agent.run())
    passed = sum(1 for r in results if r.passed)
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
    }
    print(json.dumps(summary, indent=2))

    if args.detail_output and results:
        detail = [
            {
                "passed": r.passed,
                "failure_reason": r.failure_reason,
                "duration_ms": r.duration_ms,
                "test_case": {
                    "title": r.test_case.title,
                    "type": r.test_case.type,
                    "priority": r.test_case.priority,
                    "steps": r.test_case.steps,
                    "expected_outcome": r.test_case.expected_outcome,
                    "source_url": r.test_case.source_url,
                    "preconditions": r.test_case.preconditions,
                },
            }
            for r in results
        ]
        out_path = pathlib.Path(args.detail_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
