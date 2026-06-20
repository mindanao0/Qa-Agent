# Explorer / Locator / Form-Filling — Findings

Three coupled runtime problems, each measured on the same 10 live targets
(`eval/explore_golden.jsonl`) with real Chromium + Ollama. Branch lineage:
`explorer-locator-quality` (E0–E2 + opt A) → `form-filling-crawler` (F0–F2).
Every number below is a live run; nothing mocked.

## 1. Exploration (states / coverage / dedup)

| config | states_avg | reachable_coverage | dedup_precision |
|---|---|---|---|
| baseline (URL-BFS) | 13.1 | 0.636 | 1.0 |
| +SFG traversal (E1) | 15.9 | 0.591 | 1.0 |
| +Layer-1 locator heal (E2) | 11.9 | 0.685 | 1.0 |

## 2. Locator success (deterministic, no LLM)

| config | clicks | fail | success_rate | LLM-call% |
|---|---|---|---|---|
| E1 (no heal) | 1056 | 270 | 0.744 | 0 |
| E2 (Layer-1) | 564 | 119 | **0.789** | **0** |

## 3. Form-filling / post-login (the F-track)

| config | post_login_states | auth_success_rate | coverage | blocked_submits |
|---|---|---|---|---|
| baseline (no preauth, click-only) | 0 | 0.00 | 0.525 | 0 |
| +fill (F1) | 7 | 0.667 | 0.685 | 0 |
| +submit (F2) | **9** | **1.00** | **0.705** | 0 |

## Verdict (per-phase, honest)

- **E1 SFG traversal**: replaced URL-BFS with real state-flow traversal + multi-signal
  (url + a11y multiset) dedup. v1 regressed dedup (volatile content); **v2 fixed it**
  (multiset of stable controls). States up; coverage was locator-gated.
- **E2 Layer-1 locator healing**: semantic-locator fallback chain (getByRole > Label >
  Placeholder > Text > TestId) + Jaro-Winkler fuzzy — **all deterministic, zero LLM**.
  locator_success 0.744→0.789, coverage 0.636→0.685. (The fuzzy sub-step rarely fired;
  the semantic ordering did the work.)
- **opt A form/SPA-aware scanner**: fixed a real bug (scanner was `input[type=submit]`-only
  → forms invisible). Necessary but **not sufficient** for the click-only crawler — which
  led to the F-track.
- **F1 compound fill→submit**: the crawler logs *itself* in (fill username/password from a
  gitignored config, PII/financial-safe; credentials re-derived at replay, never stored —
  replay scripts mask them as `<test_credential>`). post_login 0→7, auth 0→0.667.
- **F2 safe submit + SPA hydration**: a hydration poll surfaces late Angular forms
  (**orangehrm 0.0 → logs in**), lifting auth_success to **1.0**. Submission is gated three
  ways (`_SAFE_SUBMIT_KW` + `_is_blocked` + form-action `_action_blocked`); **0 blocked
  submits**. Honest caveat: blocked forms (transfer/payment) are post-login/deep, so the
  refusal path wasn't *exercised* at runtime — it's guaranteed by construction and verified
  in a single-site test.

**Net:** both original bugs measurably fixed — the explorer now does real multi-state
traversal (incl. SPA/in-page), locator success rose with no LLM, and the crawler reaches
post-login states on every auth site (0 → 9 states, auth 0 → 1.0). Production explorer
unchanged (this is the eval/experiment branch); changes are opt-in.

## Invariants honored
`asyncio.Semaphore(1)` on Ollama; CDP-only AOM; `pathlib.Path`; **blocked forms never
submitted**; test creds from gitignored config, never hardcoded/committed/logged; separate
BrowserContext per session; depth/action/state/time circuit breakers. realworld stayed at
0 — it served HTTP 500 (dead target this window), not a crawler failure.

## Reproduce
```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  .venv/bin/python eval/run_explore_eval.py --mode sfg --no-preauth --out eval/explore_f2.json
.venv/bin/python eval/summarize_explore.py     # rebuilds all three tables
```
(Requires `eval/test_credentials.json` — copy from `eval/test_credentials.example.json`, public demo accounts only.)
