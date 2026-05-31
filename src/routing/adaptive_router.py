"""
Adaptive confidence router — Sprint 3 / Cluster S3-D.

Classifies an incoming (url, requirement, domain) into one of three
confidence tiers and gates how many BFT generators run:
  HIGH → cached result (skip BFT entirely)
  MED  → single greedy generator + judge
  LOW  → full 3-generator BFT
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MED = "MED"
    LOW = "LOW"


_DEFAULT_DB_PATH = Path.home() / ".qa-agent" / "state.db"
_MED_TIER_MIN_RUNS = 3
_MED_TIER_PASS_RATE = 0.9


def _cache_key(url: str, requirement: str) -> str:
    h = hashlib.sha256(f"{url}|||{requirement}".encode("utf-8")).hexdigest()
    return h


def _domain_from_url(url: str) -> str:
    """Best-effort: extract netloc.lower() as the domain."""
    from urllib.parse import urlparse
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc or url
    except Exception:
        return url


class AdaptiveRouter:
    """SQLite-backed (url, requirement) → confidence tier classifier."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path: Path = (db_path or _DEFAULT_DB_PATH).expanduser()
        self._lock = threading.Lock()
        self._ensure_table()

    # ── Schema ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_test_cache (
                    cache_key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    pass_count INTEGER NOT NULL DEFAULT 0,
                    pass_rate REAL NOT NULL DEFAULT 0.0,
                    last_generated_code_hash TEXT,
                    last_run_iso TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_url_test_cache_domain "
                "ON url_test_cache(domain)"
            )
            conn.commit()

    # ── Classification ─────────────────────────────────────────────────

    def classify(self, url: str, requirement: str, domain: str) -> ConfidenceTier:
        """Lookup tier from url_test_cache:
            - Exact (url+requirement) hash with pass_rate >= 0.95 → HIGH
            - Domain hit with pass_rate >= 0.9 and run_count >= 3 → MED
            - Otherwise → LOW
        """
        key = _cache_key(url, requirement)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT pass_rate, run_count FROM url_test_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is not None and row["run_count"] >= 1 and row["pass_rate"] >= 0.95:
                return ConfidenceTier.HIGH

            dom = (domain or _domain_from_url(url)).lower()
            dom_row = conn.execute(
                """
                SELECT
                    SUM(pass_count) AS s_pass,
                    SUM(run_count) AS s_run
                FROM url_test_cache
                WHERE domain = ?
                """,
                (dom,),
            ).fetchone()
            if dom_row is not None and dom_row["s_run"] is not None:
                total_runs = int(dom_row["s_run"] or 0)
                total_passes = int(dom_row["s_pass"] or 0)
                if total_runs >= _MED_TIER_MIN_RUNS:
                    rate = total_passes / total_runs if total_runs else 0.0
                    if rate >= _MED_TIER_PASS_RATE:
                        return ConfidenceTier.MED

        return ConfidenceTier.LOW

    # ── Cached result lookup ───────────────────────────────────────────

    def lookup_cached(self, url: str, requirement: str) -> dict[str, Any] | None:
        """Return last cached row (raw dict) for HIGH-tier reuse, or None."""
        key = _cache_key(url, requirement)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM url_test_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            return dict(row) if row else None

    # ── Record outcome ─────────────────────────────────────────────────

    def record_result(
        self,
        url: str,
        requirement: str,
        domain: str,
        passed: bool,
        generated_code: str | None,
    ) -> None:
        """UPSERT into url_test_cache. Called by Reporter after each run."""
        key = _cache_key(url, requirement)
        dom = (domain or _domain_from_url(url)).lower()
        now_iso = datetime.now(timezone.utc).isoformat()
        code_hash = (
            hashlib.sha256(generated_code.encode("utf-8")).hexdigest()
            if generated_code
            else None
        )

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT run_count, pass_count FROM url_test_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                run_count = 1
                pass_count = 1 if passed else 0
                conn.execute(
                    """
                    INSERT INTO url_test_cache
                        (cache_key, url, domain, run_count, pass_count,
                         pass_rate, last_generated_code_hash, last_run_iso)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key, url, dom, run_count, pass_count,
                        pass_count / run_count, code_hash, now_iso,
                    ),
                )
            else:
                run_count = int(row["run_count"]) + 1
                pass_count = int(row["pass_count"]) + (1 if passed else 0)
                conn.execute(
                    """
                    UPDATE url_test_cache
                       SET run_count = ?,
                           pass_count = ?,
                           pass_rate = ?,
                           last_generated_code_hash = COALESCE(?, last_generated_code_hash),
                           last_run_iso = ?
                     WHERE cache_key = ?
                    """,
                    (
                        run_count, pass_count, pass_count / run_count,
                        code_hash, now_iso, key,
                    ),
                )
            conn.commit()
        logger.debug(
            f"AdaptiveRouter.record_result | key={key[:10]} domain={dom} "
            f"passed={passed} run_count={run_count} pass_rate={pass_count / max(run_count, 1):.2f}"
        )


__all__ = ["ConfidenceTier", "AdaptiveRouter"]
