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
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run browser in headless mode (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser window")
    args = parser.parse_args()

    agent = UniversalQAAgent(
        url=args.url,
        username=args.username,
        password=args.password,
        max_pages=args.max_pages,
        headless=args.headless,
    )

    results = asyncio.run(agent.run())
    passed = sum(1 for r in results if r.passed)
    print(json.dumps({
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
