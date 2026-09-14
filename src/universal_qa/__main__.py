# src/universal_qa/__main__.py
import argparse
import asyncio
import json

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
                        help="Write full per-test results JSON (for training-data collection)")
    parser.add_argument("--coverage-crosscheck", action="store_true", default=False,
                        help="After exploration, cross-check the SFG crawl against 8 "
                             "independent coverage dimensions and write "
                             "<output-dir>/coverage_crosscheck.json (see eval/FINDINGS_explore.md)")
    parser.add_argument("--race-testing", action="store_true", default=None,
                        help="After exploration, run concurrent read-only GET race scenarios "
                             "against discovered pages and write <output-dir>/race_report.json "
                             "(see src/universal_qa/race_check.py). Default: config/agent.yaml "
                             "race.enable_race_testing (False).")
    parser.add_argument("--allow-destructive-race-scenarios", action="store_true", default=None,
                        help="Reserved: no safe generic destructive-scenario synthesis exists "
                             "yet for arbitrary sites, so this currently has no effect beyond a "
                             "log line — see src/universal_qa/race_check.py.")
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
        enable_coverage_crosscheck=args.coverage_crosscheck,
        enable_race_testing=args.race_testing,
        allow_destructive_race_scenarios=args.allow_destructive_race_scenarios,
    )

    results = asyncio.run(agent.run())
    passed = sum(1 for r in results if r.passed)

    if args.detail_output:
        import pathlib
        out = pathlib.Path(args.detail_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([r.model_dump() for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
