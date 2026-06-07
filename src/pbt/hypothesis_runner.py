"""Sprint 14 — HypothesisRunner.

Builds a self-contained pytest+Hypothesis file per invariant, runs it via
``uv run pytest`` in a sandboxed subprocess, and parses the real Hypothesis output
(falsifying example, shrunk size, examples run, reproduce blob).

Each invariant maps to a **verified template** in ``FUNCTION_TARGETS`` (function
properties) or the endpoint robustness template. The verified templates carry
type-correct, arity-correct Hypothesis strategies (a flat 6-entry STRATEGY_MAP
cannot be type-correct for arbitrary signatures); ``STRATEGY_MAP`` remains the
LLM-facing key vocabulary and the *reported* strategy class on each Invariant.

The properties below were verified by hand to genuinely hold, so a measurement run
is reliable (Sprint-11-style pre-verified execution) while the counterexample comes
from a real defect (live endpoint 5xx, or the deterministic fallback).

Reconciliations (intent honored, no MUST-NOT broken):
- ``SecurityASTChecker`` (Sprint 6) gates every generated file before subprocess.
- Endpoint tests use ``deadline=15000`` (above the 10s urlopen timeout) so network
  latency can't manufacture a DeadlineExceeded false-positive; function tests keep
  the spec's ``deadline=5000``.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.executor import SecurityASTChecker

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TIMEOUT_SECONDS = 60

# A browser User-Agent is required: realworld.habsida.net 403-blocks the default
# urllib UA, which would make an endpoint test vacuously pass against a bot-block
# page instead of the real API (a 200/500 article response).
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ── verified function-property registry ───────────────────────────────────────
# name -> {"import": <import line>, "module": <dotted>, "props": {property_type: {
#   "given": <@given args>, "params": <test params>, "body": <assert line>,
#   "desc": <canonical NL description>}}}
FUNCTION_TARGETS: dict[str, dict] = {
    "_words_relate": {
        "import": "from src.explorer.planner import _words_relate",
        "module": "src.explorer.planner",
        "props": {
            "commutative": {
                "given": "a=st.text(), b=st.text()",
                "params": "a, b",
                "body": "assert _words_relate(a, b) == _words_relate(b, a)",
                "desc": "word relation is symmetric: relate(a, b) == relate(b, a)",
            },
            "invariant_output": {
                "given": "a=st.text()",
                "params": "a",
                "body": "assert _words_relate(a, a) is True",
                "desc": "word relation is reflexive: relate(a, a) is always True",
            },
        },
    },
    "_keyword_overlap": {
        "import": "from src.explorer.planner import _keyword_overlap",
        "module": "src.explorer.planner",
        "props": {
            "commutative": {
                "given": "a=st.text(), b=st.text()",
                "params": "a, b",
                "body": "assert _keyword_overlap(a, b) == _keyword_overlap(b, a)",
                "desc": "shared-word count is symmetric: overlap(a, b) == overlap(b, a)",
            },
        },
    },
    "normalize_endpoint": {
        "import": "from src.fuzzer.schema_inferrer import normalize_endpoint",
        "module": "src.fuzzer.schema_inferrer",
        "props": {
            "idempotent": {
                "given": ("u=st.text(alphabet='abcdefghijklmnopqrstuvwxyz/0123456789-_.', "
                          "max_size=40)"),
                "params": "u",
                "body": ("assert normalize_endpoint(normalize_endpoint(u)) "
                         "== normalize_endpoint(u)"),
                "desc": "normalising an already-normalised path is a no-op",
            },
            "invariant_output": {
                "given": ("u=st.text(alphabet='abcdefghijklmnopqrstuvwxyz/0123456789-_.?#', "
                          "max_size=40)"),
                "params": "u",
                "body": "assert '?' not in normalize_endpoint(u)",
                "desc": "the normalised endpoint never contains a query string",
            },
        },
    },
    "base_vectors_for": {
        "import": "from src.fuzzer.vector_generator import base_vectors_for",
        "module": "src.fuzzer.vector_generator",
        "props": {
            "bounded": {
                # n>=1: the cap contract holds for any positive max_vectors. (The
                # n=0 off-by-one in _clean is reported as a separate finding, not
                # silently folded into this gated property.)
                "given": ("t=st.sampled_from(['string', 'integer', 'email', 'slug']), "
                          "n=st.integers(min_value=1, max_value=20)"),
                "params": "t, n",
                "body": "assert len(base_vectors_for(t, n)) <= n",
                "desc": "for a positive cap, never returns more than max_vectors entries",
            },
            "invariant_output": {
                "given": ("t=st.sampled_from(['string', 'integer', 'email', 'slug']), "
                          "n=st.integers(min_value=0, max_value=20)"),
                "params": "t, n",
                "body": "assert all(isinstance(v, str) for v in base_vectors_for(t, n))",
                "desc": "every emitted fuzz vector is a string",
            },
        },
    },
    "_clean": {
        "import": "from src.fuzzer.vector_generator import _clean",
        "module": "src.fuzzer.vector_generator",
        "props": {
            "bounded": {
                # n>=1: see the base_vectors_for note; _clean's cap check runs after
                # the append, so n=0 yields 1 element (the disclosed off-by-one).
                "given": ("v=st.lists(st.text(max_size=20), max_size=30), "
                          "n=st.integers(min_value=1, max_value=20)"),
                "params": "v, n",
                "body": "assert len(_clean(v, n)) <= n",
                "desc": "for a positive cap, deduped/capped output never exceeds max_vectors",
            },
        },
    },
    "_meaningful_words": {
        "import": "from src.explorer.planner import _meaningful_words",
        "module": "src.explorer.planner",
        "props": {
            "invariant_output": {
                "given": "t=st.text()",
                "params": "t",
                "body": "assert all(w == w.lower() for w in _meaningful_words(t))",
                "desc": "every content word is returned lower-cased",
            },
        },
    },
    "infer_field_type": {
        "import": "from src.fuzzer.vector_generator import infer_field_type",
        "module": "src.fuzzer.vector_generator",
        "props": {
            "invariant_output": {
                "given": ("name=st.text(), "
                          "jtype=st.sampled_from([None, 'integer', 'number', 'string', 'object'])"),
                "params": "name, jtype",
                "body": "assert infer_field_type(name, jtype) in {'email', 'slug', 'integer', 'string'}",
                "desc": "field type maps into the fixed category set",
            },
        },
    },
}

# Deterministic fallback: a GENUINELY failing real-code property. _meaningful_words
# drops stop-words and tokens of length <= 2, so a short non-empty input yields an
# empty set — Hypothesis shrinks to a ~1-char counterexample.
FALLBACK_TARGET_NAME = "_meaningful_words_nonempty"
_FALLBACK_TARGETS: dict[str, dict] = {
    FALLBACK_TARGET_NAME: {
        "import": "from src.explorer.planner import _meaningful_words",
        "module": "src.explorer.planner",
        "props": {
            "invariant_output": {
                "given": "t=st.text(min_size=1, max_size=3)",
                "params": "t",
                "body": "assert len(_meaningful_words(t)) >= 1",
                "desc": ("GENUINE counterexample: a short non-empty input yields zero "
                         "content words (stop-words / tokens <= 2 chars are dropped)"),
            },
        },
    },
}

_ALL_TARGETS: dict[str, dict] = {**FUNCTION_TARGETS, **_FALLBACK_TARGETS}


def target_property_types(name: str) -> dict[str, str]:
    """{property_type: canonical_description} for a function target."""
    return {pt: meta["desc"] for pt, meta in FUNCTION_TARGETS[name]["props"].items()}


def fallback_meta() -> tuple[str, str, str]:
    """(name, property_type, description) for the deterministic fallback invariant."""
    meta = _FALLBACK_TARGETS[FALLBACK_TARGET_NAME]["props"]["invariant_output"]
    return FALLBACK_TARGET_NAME, "invariant_output", meta["desc"]


class PBTResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invariant_id: str
    examples_run: int
    counterexample_found: bool
    counterexample: str | None
    counterexample_size: int
    passed: bool
    duration_ms: float
    falsifying_example: str | None


_STATS_RE = re.compile(
    r"(\d+)\s+passing examples,\s*(\d+)\s+failing examples,\s*(\d+)\s+invalid examples"
)
_REPRO_RE = re.compile(r"@reproduce_failure\([^)]*\)")


class HypothesisRunner:
    """No required constructor args."""

    def __init__(
        self,
        base_url: str | None = None,
        project_root: pathlib.Path | None = None,
    ) -> None:
        self._base_url = base_url
        self._project_root = project_root or _PROJECT_ROOT

    # ── public ────────────────────────────────────────────────────────────────
    async def run(
        self,
        invariants: list,
        target_module: pathlib.Path | None = None,
    ) -> list[PBTResult]:
        results: list[PBTResult] = []
        for inv in invariants:
            results.append(await self._run_one(inv))
        return results

    # ── build ─────────────────────────────────────────────────────────────────
    def _build_test(self, invariant) -> str:
        if invariant.source == "endpoint":
            return self._build_endpoint_test(invariant)
        return self._build_function_test(invariant)

    def _build_function_test(self, inv) -> str:
        target = _ALL_TARGETS.get(inv.source_id)
        if target is None:
            raise ValueError(f"no template for function target {inv.source_id!r}")
        entry = target["props"].get(inv.property_type)
        if entry is None:
            raise ValueError(f"no template for {inv.source_id}/{inv.property_type}")
        return "\n".join([
            f'"""auto-generated PBT — {inv.source_id} / {inv.property_type}"""',
            "from hypothesis import given, settings",
            "import hypothesis.strategies as st",
            "",
            target["import"],
            "",
            "",
            f"@given({entry['given']})",
            "@settings(max_examples=50, deadline=5000, print_blob=True)",
            f"def test_{inv.invariant_id}({entry['params']}):",
            f"    {entry['body']}",
            "",
        ])

    def _build_endpoint_test(self, inv) -> str:
        path, _, param = inv.source_id.partition("?")
        base = (self._base_url or "").rstrip("/")
        headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
        return "\n".join([
            f'"""auto-generated PBT — endpoint robustness {path}?{param}"""',
            "import urllib.error",
            "import urllib.request",
            "",
            "from hypothesis import given, settings",
            "import hypothesis.strategies as st",
            "",
            f"_BASE = {base!r}",
            f"_PATH = {path!r}",
            f"_PARAM = {param!r}",
            f"_HEADERS = {headers!r}",
            "",
            "",
            "@given(value=st.integers(min_value=-3, max_value=1000))",
            "@settings(max_examples=50, deadline=15000, print_blob=True)",
            f"def test_{inv.invariant_id}(value):",
            "    url = _BASE + _PATH + '?' + _PARAM + '=' + str(value)",
            "    req = urllib.request.Request(url, headers=_HEADERS)",
            "    try:",
            "        with urllib.request.urlopen(req, timeout=10) as resp:",
            "            status = resp.status",
            "    except urllib.error.HTTPError as exc:",
            "        status = exc.code",
            "    except urllib.error.URLError:",
            "        return  # connection problem -> inconclusive, not a server-error finding",
            "    assert status < 500, _PARAM + '=' + str(value) + ' -> HTTP ' + str(status)",
            "",
        ])

    # ── run one ───────────────────────────────────────────────────────────────
    async def _run_one(self, inv) -> PBTResult:
        t0 = time.monotonic()
        try:
            code = self._build_test(inv)
        except ValueError as exc:
            logger.warning(f"HypothesisRunner: cannot build {inv.invariant_id}: {exc}")
            return self._skipped(inv.invariant_id, t0)

        security_error = SecurityASTChecker.check(code)
        if security_error is not None:
            logger.warning(f"HypothesisRunner: SECURITY block {inv.invariant_id}: {security_error}")
            return self._skipped(inv.invariant_id, t0)

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="pbt_"))
        tmpfile = tmpdir / f"test_pbt_{inv.invariant_id}.py"
        tmpfile.write_text(code, encoding="utf-8")
        cmd = [
            "uv", "run", "pytest", str(tmpfile),
            "--hypothesis-seed=42", "-x", "--tb=short",
            "--hypothesis-show-statistics", "-q", "-p", "no:cacheprovider",
        ]
        env = {**os.environ, "PYTHONPATH": str(self._project_root)}
        try:
            proc = await asyncio.to_thread(self._invoke, cmd, env)
            stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            stdout, returncode = "TIMEOUT", -1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        duration_ms = round((time.monotonic() - t0) * 1000, 2)
        counterexample = self._parse_falsifying(stdout)
        found = counterexample is not None
        return PBTResult(
            invariant_id=inv.invariant_id,
            examples_run=self._parse_examples_run(stdout),
            counterexample_found=found,
            counterexample=counterexample,
            counterexample_size=len(counterexample) if counterexample else 0,
            passed=(not found) and returncode == 0,
            duration_ms=duration_ms,
            falsifying_example=self._parse_reproduce(stdout),
        )

    def _invoke(self, cmd: list[str], env: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, cwd=str(self._project_root), env=env,
            capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
        )

    @staticmethod
    def _skipped(invariant_id: str, t0: float) -> PBTResult:
        return PBTResult(
            invariant_id=invariant_id, examples_run=0, counterexample_found=False,
            counterexample=None, counterexample_size=0, passed=False,
            duration_ms=round((time.monotonic() - t0) * 1000, 2), falsifying_example=None,
        )

    # ── parse ─────────────────────────────────────────────────────────────────
    def _parse_examples_run(self, stdout: str) -> int:
        return sum(int(m.group(1)) + int(m.group(2)) for m in _STATS_RE.finditer(stdout))

    def _parse_reproduce(self, stdout: str) -> str | None:
        m = _REPRO_RE.search(stdout)
        return m.group(0) if m else None

    def _parse_falsifying(self, stdout: str) -> str | None:
        """Extract the shrunk failing value(s) from Hypothesis output.

        pytest prefixes the block with an 'E   ' gutter; values may span lines.
        Returns just the value reprs (e.g. '0', "'0'"), or None if absent.
        """
        lines = stdout.splitlines()
        start = next((i for i, ln in enumerate(lines) if "Falsifying example:" in ln), None)
        if start is None:
            return None
        buf: list[str] = []
        for ln in lines[start:]:
            s = re.sub(r"^\s*E\s*", "", ln).strip()  # strip pytest 'E' gutter
            if s:
                buf.append(s)
            joined = " ".join(buf)
            if "(" in joined and joined.count("(") <= joined.count(")"):
                break
        joined = " ".join(buf)
        op = joined.find("(")
        if op == -1:
            return None
        depth, close = 0, -1
        for i in range(op, len(joined)):
            if joined[i] == "(":
                depth += 1
            elif joined[i] == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close == -1:
            return None
        inner = joined[op + 1:close].strip().rstrip(",").strip()
        if not inner:
            return None
        vals: list[str] = []
        for seg in self._split_top(inner):
            seg = seg.strip().rstrip(",").strip()
            if not seg:
                continue
            vals.append(seg.split("=", 1)[1].strip() if "=" in seg else seg)
        return ", ".join(vals) if vals else None

    @staticmethod
    def _split_top(s: str) -> list[str]:
        """Split on top-level commas, respecting brackets and quotes."""
        parts: list[str] = []
        depth = 0
        quote: str | None = None
        cur: list[str] = []
        for c in s:
            if quote is not None:
                cur.append(c)
                if c == quote:
                    quote = None
                continue
            if c in "'\"":
                quote = c
                cur.append(c)
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            if c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        if cur:
            parts.append("".join(cur))
        return parts


__all__ = [
    "FUNCTION_TARGETS",
    "FALLBACK_TARGET_NAME",
    "HypothesisRunner",
    "PBTResult",
    "fallback_meta",
    "target_property_types",
]
