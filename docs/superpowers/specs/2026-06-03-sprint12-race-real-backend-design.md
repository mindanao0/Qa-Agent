# Sprint 12 — Race Condition on Real Backend (Reconciled Design)

**Date:** 2026-06-03
**Source spec:** `docs/specs/Sprint 12 Race Condition (Real Backend).md`
**Status:** Approved (user authorized live-mirror writes + auth-free scenarios) — ready to implement.

Goal: produce a GENUINE race conflict on a real stateful REST+DB backend (not a
localStorage-only or mock target), using the Sprint 10 `semantic_hash_comparison`
swarm — overcoming Sprint 10's honest-0 limitation.

## Live reality (probed 2026-06-03)

| Target | Status | Verdict |
|---|---|---|
| `conduit.realworld.how` (spec primary) | HTTP 000 (down) | unusable |
| `api.realworld.io` (classic) | HTTP 530 / CF 1016 (down) | unusable |
| **`realworld.habsida.net`** | live; concurrent same-username register → **1×200 + 2×422 (`SQLiteError: UNIQUE constraint`)** | **REAL stateful backend ✅ (primary target)** |
| `api.realworld.show` | live tags, but 3× same-username register all returned `201` + identical fake token | **register-MOCK ✗** |
| `node-express-conduit.appspot.com` | live tags | secondary candidate |
| `jsonplaceholder` | mock | last-resort fallback → honest `race=0` |

User decisions: **(1)** authorize writing throwaway `race_<random>` accounts to a live
mirror; **(2)** auth-free scenario set (register-collision + reads).

## Acceptance gate (sprint12)

| Metric | Gate | How measured (honest) |
|---|---|---|
| `race_scenarios_tested` | ≥ 5 | `len(results)` |
| `real_backend_confirmed` | True | `BackendProbe` got real `/api/tags`+`/api/articles` data |
| `race_conditions_detected` | ≥ 1 | `ConflictDetector.analyze()["conflicts_found"]` (register collision) |
| `interleaving_patterns_found` | ≥ 1 | scenarios whose `InterleavingRecorder` captured a pattern |
| `semantic_hash_method` | True | swarm uses `_semantic_hash`/`_normalize_http_response` (no CDP nodeId) |
| `otel_spans_emitted` | ≥ 8 | `OTelTracer.flush()` (probe + 5 scenarios + detect + run) |
| `regression` | — | True iff `pass_rate < 0.75`; pass_rate = scenarios whose observed safety matched `expected_safe` |

## Architecture — files

```
src/race/backend_probe.py          (NEW)  BackendProbe + BackendProbeResult
src/race/interleaving_recorder.py  (NEW)  AgentEvent + InterleavingRecorder
src/race/swarm.py                  (MODIFY) conduit actions + status + recorder + skip-goto
audit/sprint12/measure_sprint12.py (NEW)  orchestrate, write sprint12_results.json
tests/race/test_backend_probe.py, test_interleaving_recorder.py, test_swarm_conduit.py (TDD)
```

### `BackendProbeResult` (Pydantic V2, extra="forbid")
`url: str`, `has_real_backend: bool`, `api_endpoints: list[str]`, `probe_note: str | None`.

### `BackendProbe.probe(url, page)` — robust, frontend-independent
Attaches `page.on("request")` to capture `/api/` calls, `page.goto(url)` (best-effort; the
mirror may be API-only), then DIRECTLY `page.request.get(url + "/api/tags")` and
`"/api/articles?limit=1"`. `has_real_backend = True` iff a probe returns non-empty real data
(`tags` non-empty, or `articles`/`articlesCount>0`). Distinguishes a real Conduit backend from
jsonplaceholder (which 404s `/api/tags`).

### `AgentEvent` (Pydantic V2, extra="forbid")
`agent_id: str`, `action: str`, `started_at: float` (monotonic ms), `ended_at: float`,
`status_code: int | None`, `response_hash: str` (sha256[:8] of body).

### `InterleavingRecorder`
`record(event)` appends; `pattern()` sorts all start+end points by timestamp →
`"agent0-start, agent1-start, agent0-end(200), agent1-end(422)"`; `to_dict()` → `{events, pattern}`.

### `swarm.py` modifications (additive; Sprint 10 behavior preserved)
- `RaceScenario`: add `payload: dict | None = None` (declared optional — `extra="forbid"` still
  rejects undeclared keys; Sprint 10 scenarios omit it → default None).
- `_perform_action(page, action, target_url, payload) -> tuple[str|None, int|None]` returns
  `(signature, status)`. New actions: `conduit_register` (POST `{base}/api/users` with the SHARED
  `payload["username"]` → 201/422 race), `conduit_read_tags`, `conduit_read_articles` (GET, safe).
  Existing `http_*` now also return their status; UI actions return `(None, None)`.
- `agent_task`: **skip `page.goto` when action starts with `http_`/`conduit_`** (page.request needs
  no navigation; this removes the navigation-variance that manufactured Sprint 10 drift
  false-positives), capture `status` + monotonic start/end, and feed an optional `InterleavingRecorder`.
- `run(scenario, browser, recorder=None)`: conflict logic =
  `any(status not in {200,201,204}) or len(set(hashes))>1 or bool(errors)`.
- `overlap_ms=200` for conduit scenarios (drift tolerance; the conflict signal is the HTTP status,
  not timing). `SynchronizationDriftError` mechanism unchanged.

### `measure_sprint12.py` flow
1. `BackendProbe` over candidates `[conduit.realworld.how, realworld.habsida.net,
   node-express-conduit, api.realworld.show]`; pick first `has_real_backend=True`. If none →
   jsonplaceholder fallback, `real_backend_confirmed=False`, expect `race=0` (documented honest FAIL).
2. Build 5 scenarios (base = resolved url): s1 register N=2, s2 register N=3 (each a FRESH random
   username, `expected_safe=False`), s3 read tags N=3, s4 read articles N=2, s5 read tags N=2
   (`expected_safe=True`).
3. Per scenario: `OTelTracer.span("race.scenario")`, `recorder=InterleavingRecorder()`,
   `swarm.run(scenario, browser, recorder)`, `CryptoAuditTrail.append("race_result", {...})`.
4. `ConflictDetector.analyze(results)`; collect patterns; compute gates; write JSON.

## Rules compliance
- Each agent isolated `BrowserContext` (already); `asyncio.Barrier` sync (not sleep);
  `SynchronizationDriftError` on drift > `overlap_ms*2`; semantic hash excludes CDP
  nodeId/backendDOMNodeId/childIds; per-agent action timeout 5s; CDP only; pathlib; Pydantic V2
  extra="forbid"; OTel + audit per scenario; BFT disabled; record EVERY agent event (start+end).
- `BLOCKED_ACTION_PATTERNS` (delete/remove/transfer/payment/password): no scenario action matches
  ("conduit_register" etc.); the "password" register payload field is not in the action string.
- No Ollama calls in this sprint (Semaphore rule N/A).

## Honest posture (carried from Sprint 11)
Real numbers reported as-is. If every candidate backend is down at measure time → jsonplaceholder
fallback → `race_conditions_detected=0` documented as an honest FAIL (the spec sanctions this).
External writes limited to a handful of throwaway `race_<random>` accounts on the user-authorized
mirror (public RealWorld demo backend intended for testing).

## Implementation outcome (2026-06-03) — PASS (genuine conflict)

Live run: `backend_used=https://realworld.habsida.net`, `real_backend_confirmed=true`,
`race_scenarios_tested=5`, **`race_conditions_detected=2`**, `interleaving_patterns_found=5`,
`semantic_hash_method=true`, `otel_spans_emitted=10`, `pass_rate=1.0`, `regression=false` → **PASS**.
42 race unit tests pass; 94 pass across race+observability+continuous (no regression).

The two register scenarios produced a **real** conflict — the interleaving shows genuine overlap then
a unique-constraint split:
- `s1`: `agent1-start, agent0-start, agent0-end(422), agent1-end(200)`
- `s2`: `agent2-start, agent0-start, agent1-start, agent0-end(422), agent2-end(200), agent1-end(422)`

`conduit.realworld.how` was probed and correctly found down → `realworld.habsida.net` selected. This is
the first time the swarm has detected a real race (Sprint 10 was an honest 0 on a stateless mock). No
deviations from the design above were needed.
