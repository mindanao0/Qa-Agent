"""Builder for eval/explore_golden.jsonl — exploration benchmark targets.

10 real, publicly-reachable sites with genuine multi-page and/or in-page (modal /
SPA) state, chosen to expose the "explorer only does one page" bug:

  - classic multi-URL apps (saucedemo, parabank, automationexercise, the-internet)
  - SPA / client-routed apps where URL barely changes (realworld/Conduit, orangehrm,
    todomvc) — these punish URL-only state tracking the hardest
  - modal-heavy pages (demoqa)

Each target:
  seed_url                 - where exploration starts
  username/password        - demo creds (optional; public demo accounts)
  expected_states_min      - floor on distinct states a good explorer should find
  expected_reachable_pages - normalized path fragments that should be reached
                             (matched as path-suffix / substring, scheme-agnostic)

These are a fixed spec (same input for baseline vs +SFG), so deltas are valid.

Run: python eval/_build_explore_golden.py  ->  eval/explore_golden.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path


TARGETS: list[dict] = [
    {
        "target_id": "saucedemo_flow", "seed_url": "https://www.saucedemo.com/",
        "auth_required": True, "cred_key": "saucedemo",
        "expected_post_login_states": ["/inventory.html", "/cart.html", "/checkout-step-one.html"],
        "expected_states_min": 6,
        "expected_reachable_pages": [
            "/inventory.html", "/inventory-item.html", "/cart.html",
            "/checkout-step-one.html", "/checkout-step-two.html",
        ],
        "notes": "post-login multi-page shopping flow; burger menu = modal-ish",
    },
    {
        "target_id": "the_internet", "seed_url": "https://the-internet.herokuapp.com/",
        "expected_states_min": 12,
        "expected_reachable_pages": [
            "/login", "/dropdown", "/checkboxes", "/add_remove_elements/",
            "/dynamic_loading", "/javascript_alerts", "/tables", "/windows",
        ],
        "notes": "40+ independent example pages reachable from the index",
    },
    {
        "target_id": "realworld_conduit", "seed_url": "https://realworld.habsida.net/",
        "expected_states_min": 4,
        "expected_reachable_pages": ["/login", "/register", "/"],
        "notes": "SPA (Angular/React) — client routing, URL changes minimally",
    },
    {
        "target_id": "orangehrm", "seed_url": "https://opensource-demo.orangehrmlive.com/",
        "auth_required": True, "cred_key": "orangehrm",
        "expected_post_login_states": ["/dashboard", "/admin/viewSystemUsers", "/pim/viewEmployeeList"],
        "expected_states_min": 6,
        "expected_reachable_pages": [
            "/dashboard", "/admin/viewSystemUsers", "/pim/viewEmployeeList",
            "/leave/viewLeaveList", "/recruitment/viewCandidates",
        ],
        "notes": "SPA HR portal; left-nav menu states behind auth",
    },
    {
        "target_id": "parabank", "seed_url": "https://parabank.parasoft.com/parabank/index.htm",
        "expected_states_min": 6,
        "expected_reachable_pages": [
            "/register.htm", "/about.htm", "/services.htm", "/contact.htm",
            "/admin.htm",
        ],
        "notes": "multi-page bank; overview/transfer/billpay behind login",
    },
    {
        "target_id": "automationexercise", "seed_url": "https://automationexercise.com/",
        "expected_states_min": 7,
        "expected_reachable_pages": [
            "/products", "/view_cart", "/login", "/contact_us",
            "/test_cases", "/api_list",
        ],
        "notes": "e-commerce; product list, cart, signup",
    },
    {
        "target_id": "demoqa_modals", "seed_url": "https://demoqa.com/",
        "expected_states_min": 6,
        "expected_reachable_pages": [
            "/elements", "/forms", "/alertsWindows", "/widgets",
            "/interaction", "/modal-dialogs",
        ],
        "notes": "modal dialogs, alerts, frames — in-page state",
    },
    {
        "target_id": "todomvc_react", "seed_url": "https://todomvc.com/examples/react/dist/",
        "expected_states_min": 5,
        "expected_reachable_pages": ["/active", "/completed", "/"],
        "notes": "pure SPA in-page states (empty/active/completed/all) — URL hash only",
    },
    {
        "target_id": "saucedemo_problem", "seed_url": "https://www.saucedemo.com/",
        "auth_required": True, "cred_key": "saucedemo_problem",
        "expected_post_login_states": ["/inventory.html", "/cart.html"],
        "expected_states_min": 5,
        "expected_reachable_pages": [
            "/inventory.html", "/cart.html", "/inventory-item.html",
        ],
        "notes": "problem_user variant — different post-login behaviour",
    },
    {
        "target_id": "the_internet_dynamic", "seed_url": "https://the-internet.herokuapp.com/dynamic_controls",
        "expected_states_min": 4,
        "expected_reachable_pages": ["/dynamic_controls"],
        "notes": "single URL, multiple in-page states (enable/disable, add/remove) — pure DOM-state dedup test",
    },
]


def main() -> None:
    out = Path(__file__).parent / "explore_golden.jsonl"
    assert len(TARGETS) == 10, f"expected 10 targets, got {len(TARGETS)}"
    seen = set()
    with out.open("w", encoding="utf-8") as f:
        for t in TARGETS:
            assert t["target_id"] not in seen, f"dup {t['target_id']}"
            seen.add(t["target_id"])
            assert t["expected_reachable_pages"], t["target_id"]
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(TARGETS)} targets -> {out}")
    print(f"sum expected_states_min = {sum(t['expected_states_min'] for t in TARGETS)}")


if __name__ == "__main__":
    main()
