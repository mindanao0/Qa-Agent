# audit/phase0/build_golden_dataset.py
"""
Phase 0 — Golden Dataset builder.

Reads ``audit/phase0/sites_manifest.yaml``, drives the existing agent via
``python main.py --mode generate``, runs each generated Playwright test
``flakiness_runs`` times against the real public site, and bucketises the
results into three JSONL files:

  audit/phase0/golden_dataset/passing.jsonl  — passed every run
  audit/phase0/golden_dataset/failing.jsonl  — failed every run (with signature)
  audit/phase0/golden_dataset/flaky.jsonl    — passed sometimes (with pass_rate)

Standalone usage:

    uv run python audit/phase0/build_golden_dataset.py
    uv run python audit/phase0/build_golden_dataset.py --pilot   # tiny smoke run
    uv run python audit/phase0/build_golden_dataset.py --resume  # reuse checkpoint

Hard constraints honoured per Phase 0 spec:
  * Windows 11 bare-metal — pathlib.Path everywhere; no os.path concat.
  * Read-only against existing src/ — invokes main.py via subprocess.
  * Logging via loguru (matches src/ convention).
  * Resume capability via audit/phase0/golden_dataset/_progress.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]    # D:\Code\qa-agent
AUDIT_DIR = PROJECT_ROOT / "audit" / "phase0"
DATASET_DIR = AUDIT_DIR / "golden_dataset"
WORK_DIR = AUDIT_DIR / "work_tests"
PROGRESS_FILE = DATASET_DIR / "_progress.json"
MANIFEST_FILE = AUDIT_DIR / "sites_manifest.yaml"

PASSING_PATH = DATASET_DIR / "passing.jsonl"
FAILING_PATH = DATASET_DIR / "failing.jsonl"
FLAKY_PATH = DATASET_DIR / "flaky.jsonl"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Per spec: known failure signatures.
_FAILURE_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    ("SYNTAX_ERROR",
     re.compile(r"SyntaxError|IndentationError", re.IGNORECASE)),
    ("JSON_PARSE_FAIL",
     re.compile(r"JSONDecodeError|json\.decoder\.JSONDecodeError", re.IGNORECASE)),
    ("AUTH_REQUIRED",
     re.compile(r"401\s*Unauthorized|403\s*Forbidden|authentication required",
                re.IGNORECASE)),
    ("LOCATOR_NOT_FOUND",
     re.compile(r"locator.*(?:not found|resolved to 0 elements|"
                r"strict mode violation|element\(s\) not found)",
                re.IGNORECASE)),
    ("TIMEOUT",
     re.compile(r"TimeoutError|Timeout \d+ms exceeded|"
                r"waiting for|timed out", re.IGNORECASE)),
    ("WRONG_ASSERTION",
     re.compile(r"AssertionError|expect\(.*\)\..*Expected|"
                r"Expected substring|Expected pattern", re.IGNORECASE)),
]


@dataclass
class CaseSpec:
    """A single (site, requirement) generation target."""

    case_id: str
    requirement_text: str
    target_url: str
    role: str
    bucket_hint: str   # "stable" | "vague_failing" | "broken"


@dataclass
class RunOutcome:
    """Result of running ONE pytest invocation."""

    passed: bool
    returncode: int
    duration_ms: int
    output: str          # captured stdout+stderr (truncated)
    signature: str       # one of _FAILURE_SIGNATURES keys, or "" if passed


@dataclass
class CaseRecord:
    """A finished case — what gets serialised to JSONL."""

    case_id: str
    requirement_text: str
    target_url: str
    role: str
    expected_outcome: str
    notes: str
    created_at_iso: str
    # Provenance — useful for debugging the dataset later
    generation_seconds: float = 0.0
    pytest_runs: list[dict[str, Any]] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    # Bucket-specific fields (populated only for the right bucket)
    failure_signature: str = ""
    raw_error_text: str = ""
    pass_rate_observed: float = 0.0
    suspected_cause: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Manifest expansion
# ──────────────────────────────────────────────────────────────────────────────


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _expand_cases(manifest: dict[str, Any]) -> list[CaseSpec]:
    """Flatten the manifest into a list of CaseSpec, with stable case_ids."""
    cases: list[CaseSpec] = []
    index = 0

    def _emit(bucket: str, group: list[dict[str, Any]]) -> None:
        nonlocal index
        for site in group or []:
            url = site["url"]
            role = site.get("role", "admin")
            for req in site.get("requirements", []) or []:
                case_id = f"{bucket}-{index:04d}"
                cases.append(CaseSpec(
                    case_id=case_id,
                    requirement_text=req,
                    target_url=url,
                    role=role,
                    bucket_hint=bucket,
                ))
                index += 1

    _emit("stable", manifest.get("stable_sites", []))
    _emit("vague_failing", manifest.get("vague_failing", []))
    _emit("broken", manifest.get("broken_sites", []))
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess: agent generate
# ──────────────────────────────────────────────────────────────────────────────


def _run_generate(
    case: CaseSpec,
    output_dir: Path,
    timeout_sec: int,
    model: str,
    max_retries: int,
) -> tuple[Path | None, str, float]:
    """
    Invoke ``python main.py --mode generate ...`` and return:
      (generated_script_path | None, captured_output, elapsed_seconds)
    """
    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "main.py"),
        "--mode", "generate",
        "--requirement", case.requirement_text,
        "--url", case.target_url,
        "--role", case.role,
        "--output-dir", str(output_dir),
        "--model", model,
        "--max-retries", str(max_retries),
        "--log-level", "INFO",
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        partial = (exc.stdout or "") + (exc.stderr or "")
        logger.warning(f"generate timed out for {case.case_id} after {elapsed:.1f}s")
        return None, f"TIMEOUT after {timeout_sec}s\n{partial}", elapsed
    except Exception as exc:
        elapsed = time.monotonic() - started
        logger.warning(f"generate exception for {case.case_id}: {exc}")
        return None, f"EXCEPTION: {exc}", elapsed

    elapsed = time.monotonic() - started
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        logger.warning(
            f"generate non-zero exit {proc.returncode} for {case.case_id} "
            f"(elapsed={elapsed:.1f}s)"
        )
        return None, output, elapsed

    # Find the newest .py file in output_dir
    candidates = sorted(output_dir.glob("*.py"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.warning(f"generate produced no .py for {case.case_id}")
        return None, output, elapsed
    return candidates[-1], output, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess: pytest
# ──────────────────────────────────────────────────────────────────────────────


def _run_pytest(
    script: Path,
    timeout_sec: int,
) -> RunOutcome:
    """Invoke pytest on *script* and classify the outcome."""
    cmd = [
        sys.executable, "-m", "pytest", str(script),
        "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider",
        "-p", "no:warnings",
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        partial = (exc.stdout or "") + (exc.stderr or "")
        return RunOutcome(
            passed=False,
            returncode=-1,
            duration_ms=elapsed_ms,
            output=f"PYTEST_TIMEOUT after {timeout_sec}s\n{partial[-2000:]}",
            signature="TIMEOUT",
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    passed = proc.returncode == 0
    signature = "" if passed else _classify_failure(output)
    return RunOutcome(
        passed=passed,
        returncode=proc.returncode,
        duration_ms=elapsed_ms,
        output=output[-4000:],          # keep tail for debugging
        signature=signature,
    )


def _classify_failure(output: str) -> str:
    """Return the first matching failure signature, or 'WRONG_ASSERTION' as fallback."""
    for name, pattern in _FAILURE_SIGNATURES:
        if pattern.search(output):
            return name
    return "WRONG_ASSERTION"


# ──────────────────────────────────────────────────────────────────────────────
# Per-case orchestration
# ──────────────────────────────────────────────────────────────────────────────


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_case(
    case: CaseSpec,
    work_dir: Path,
    flakiness_runs: int,
    pytest_timeout: int,
    generate_timeout: int,
    model: str,
    max_retries: int,
) -> CaseRecord | None:
    """
    Run the full pipeline for one case:
      generate → run pytest N times → return a CaseRecord (or None if generation failed
      AND the case is not a vague_failing target — vague_failing cases that fail to
      generate are still counted as failing.jsonl entries).
    """
    case_work_dir = work_dir / case.case_id
    case_work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{case.case_id}] generate | {case.target_url} "
        f"| {case.requirement_text[:80]!r}"
    )
    script_path, gen_output, gen_elapsed = _run_generate(
        case=case,
        output_dir=case_work_dir,
        timeout_sec=generate_timeout,
        model=model,
        max_retries=max_retries,
    )

    if script_path is None:
        # Generation failed. Classify the failure signature; if this was a
        # vague_failing target, that's expected and goes straight to failing.jsonl.
        signature = _classify_failure(gen_output) or "JSON_PARSE_FAIL"
        return CaseRecord(
            case_id=case.case_id,
            requirement_text=case.requirement_text,
            target_url=case.target_url,
            role=case.role,
            expected_outcome="generation_failed",
            notes=f"agent generate returned no script (bucket_hint={case.bucket_hint})",
            created_at_iso=_iso_now(),
            generation_seconds=round(gen_elapsed, 2),
            pytest_runs=[],
            pass_count=0,
            fail_count=flakiness_runs,
            failure_signature=signature,
            raw_error_text=gen_output[-3000:],
        )

    runs: list[RunOutcome] = []
    pass_count = 0
    for run_i in range(flakiness_runs):
        outcome = _run_pytest(script_path, timeout_sec=pytest_timeout)
        runs.append(outcome)
        if outcome.passed:
            pass_count += 1
        logger.info(
            f"[{case.case_id}] run {run_i + 1}/{flakiness_runs} "
            f"-> {'PASS' if outcome.passed else 'FAIL'} "
            f"({outcome.duration_ms} ms, sig={outcome.signature!r})"
        )

    fail_count = flakiness_runs - pass_count
    first_failure = next((r for r in runs if not r.passed), None)
    failure_signature = first_failure.signature if first_failure else ""
    raw_error_text = first_failure.output if first_failure else ""

    record = CaseRecord(
        case_id=case.case_id,
        requirement_text=case.requirement_text,
        target_url=case.target_url,
        role=case.role,
        expected_outcome=(
            "pass" if pass_count == flakiness_runs
            else "fail" if pass_count == 0
            else "flaky"
        ),
        notes=(
            f"bucket_hint={case.bucket_hint}; "
            f"script={script_path.name}"
        ),
        created_at_iso=_iso_now(),
        generation_seconds=round(gen_elapsed, 2),
        pytest_runs=[
            {
                "run_index": i,
                "passed": r.passed,
                "returncode": r.returncode,
                "duration_ms": r.duration_ms,
                "signature": r.signature,
            }
            for i, r in enumerate(runs)
        ],
        pass_count=pass_count,
        fail_count=fail_count,
        failure_signature=failure_signature,
        raw_error_text=raw_error_text,
        pass_rate_observed=round(pass_count / flakiness_runs, 2),
        suspected_cause=_suspect_cause(runs),
    )
    return record


def _suspect_cause(runs: list[RunOutcome]) -> str:
    """Heuristic — only meaningful for the flaky bucket."""
    if all(r.passed for r in runs) or all(not r.passed for r in runs):
        return ""   # not flaky
    signatures = [r.signature for r in runs if not r.passed]
    if not signatures:
        return ""
    most_common = max(set(signatures), key=signatures.count)
    return {
        "TIMEOUT": "race condition on load",
        "LOCATOR_NOT_FOUND": "DOM not stable between runs",
        "WRONG_ASSERTION": "non-deterministic content",
        "JSON_PARSE_FAIL": "intermittent LLM JSON failure",
        "SYNTAX_ERROR": "intermittent LLM syntax error",
        "AUTH_REQUIRED": "intermittent auth/session issue",
    }.get(most_common, f"intermittent {most_common}")


# ──────────────────────────────────────────────────────────────────────────────
# Bucket routing
# ──────────────────────────────────────────────────────────────────────────────


def _bucket_for(record: CaseRecord) -> str:
    if record.pass_count > 0 and record.fail_count == 0:
        return "passing"
    if record.pass_count == 0 and record.fail_count > 0:
        return "failing"
    return "flaky"


def _to_jsonl_record(record: CaseRecord, bucket: str) -> dict[str, Any]:
    """Render the record in the exact schema the spec requires per-bucket."""
    base = {
        "case_id": record.case_id,
        "requirement_text": record.requirement_text,
        "target_url": record.target_url,
        "role": record.role,
        "expected_outcome": record.expected_outcome,
        "notes": record.notes,
        "created_at_iso": record.created_at_iso,
        # Provenance fields (extension over the spec — useful for §6 baseline)
        "generation_seconds": record.generation_seconds,
        "pytest_runs": record.pytest_runs,
        "pass_count": record.pass_count,
        "fail_count": record.fail_count,
    }
    if bucket == "failing":
        base["failure_signature"] = record.failure_signature
        base["raw_error_text"] = record.raw_error_text
    elif bucket == "flaky":
        base["pass_rate_observed"] = record.pass_rate_observed
        base["suspected_cause"] = record.suspected_cause
    return base


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────


def _load_progress() -> dict[str, Any]:
    if not PROGRESS_FILE.exists():
        return {"completed_case_ids": []}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"completed_case_ids": []}


def _save_progress(progress: dict[str, Any]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    os.replace(tmp, PROGRESS_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0 — Golden Dataset builder"
    )
    parser.add_argument("--manifest", default=str(MANIFEST_FILE),
                        help="YAML manifest of target sites/requirements")
    parser.add_argument("--pilot", action="store_true",
                        help="Run a tiny pilot (5/5/2 instead of 50/50/20)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from progress checkpoint (default if file exists)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing JSONLs + checkpoint and start fresh")
    parser.add_argument("--model",
                        default=os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Hard cap on cases to process this session")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    # Manifest + targets
    manifest = _load_manifest(Path(args.manifest))
    targets = manifest.get("targets", {})
    cfg = manifest.get("run_config", {})

    if args.pilot:
        targets = {"passing": 5, "failing": 5, "flaky": 2}
        logger.info("PILOT mode: targets = 5 / 5 / 2")
    flakiness_runs = int(cfg.get("flakiness_runs", 5))
    pytest_timeout = int(cfg.get("pytest_timeout_sec", 90))
    generate_timeout = int(cfg.get("generate_timeout_sec", 300))
    logger.info(
        f"Targets: passing={targets['passing']} failing={targets['failing']} "
        f"flaky={targets['flaky']} | flakiness_runs={flakiness_runs} "
        f"| model={args.model}"
    )

    # Reset?
    if args.reset:
        for p in (PASSING_PATH, FAILING_PATH, FLAKY_PATH, PROGRESS_FILE):
            if p.exists():
                p.unlink()
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR, ignore_errors=True)
        logger.info("Reset: cleared previous dataset + checkpoint")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    progress = _load_progress()
    completed_ids: set[str] = set(progress.get("completed_case_ids", []))
    logger.info(f"Resume: {len(completed_ids)} cases already completed")

    # Counts so far
    counts = {
        "passing": _load_jsonl_count(PASSING_PATH),
        "failing": _load_jsonl_count(FAILING_PATH),
        "flaky": _load_jsonl_count(FLAKY_PATH),
    }
    logger.info(
        f"Current counts: passing={counts['passing']} "
        f"failing={counts['failing']} flaky={counts['flaky']}"
    )

    cases = _expand_cases(manifest)
    logger.info(f"Manifest expanded → {len(cases)} candidate cases")

    processed_this_session = 0
    for case in cases:
        if case.case_id in completed_ids:
            continue
        if all(counts[b] >= targets[b] for b in ("passing", "failing", "flaky")):
            logger.info("All bucket targets reached — stopping early")
            break
        if args.max_cases is not None and processed_this_session >= args.max_cases:
            logger.info(f"max-cases={args.max_cases} reached — stopping")
            break

        record = _process_case(
            case=case,
            work_dir=WORK_DIR,
            flakiness_runs=flakiness_runs,
            pytest_timeout=pytest_timeout,
            generate_timeout=generate_timeout,
            model=args.model,
            max_retries=args.max_retries,
        )
        if record is None:
            # Should not happen — _process_case always returns a record
            completed_ids.add(case.case_id)
            _save_progress({"completed_case_ids": sorted(completed_ids)})
            continue

        bucket = _bucket_for(record)
        if counts[bucket] >= targets[bucket]:
            logger.info(
                f"[{case.case_id}] bucket={bucket} already full "
                f"({counts[bucket]}/{targets[bucket]}) — skipping save"
            )
        else:
            jsonl_path = {
                "passing": PASSING_PATH,
                "failing": FAILING_PATH,
                "flaky":   FLAKY_PATH,
            }[bucket]
            _append_jsonl(jsonl_path, _to_jsonl_record(record, bucket))
            counts[bucket] += 1
            logger.info(
                f"[{case.case_id}] saved -> {bucket}.jsonl "
                f"({counts[bucket]}/{targets[bucket]})"
            )

        completed_ids.add(case.case_id)
        _save_progress({"completed_case_ids": sorted(completed_ids)})
        processed_this_session += 1

    logger.info(
        f"Build complete | passing={counts['passing']} "
        f"failing={counts['failing']} flaky={counts['flaky']} "
        f"| processed_this_session={processed_this_session}"
    )

    # Summary banner
    if all(counts[b] >= targets[b] for b in ("passing", "failing", "flaky")):
        logger.success("GOLDEN_DATASET_COMPLETE — all targets satisfied")
        return 0
    logger.warning(
        "GOLDEN_DATASET_PARTIAL — re-run to continue: "
        f"passing={counts['passing']}/{targets['passing']}, "
        f"failing={counts['failing']}/{targets['failing']}, "
        f"flaky={counts['flaky']}/{targets['flaky']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
