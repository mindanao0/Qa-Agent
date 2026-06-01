"""CryptoAuditTrail — append-only sha256 hash chain for tamper-evidence."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    ts: float
    event: str
    payload_hash: str
    prev_hash: str
    chain_hash: str


def _link_hash(event: str, payload_hash: str, prev_hash: str) -> str:
    """Hash used as the *next* entry's prev_hash.

    Covers the event name so any mutation to event is detected by the
    subsequent entry's prev_hash check.
    """
    return hashlib.sha256(
        (event + payload_hash + prev_hash).encode()
    ).hexdigest()


class CryptoAuditTrail:
    """Append-only audit trail. Never mutate existing entries."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = self._count_existing()
        self._last_link_hash = self._read_last_link_hash()

    def _count_existing(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for line in self._path.read_text().splitlines() if line.strip())

    def _read_last_link_hash(self) -> str:
        if not self._path.exists():
            return "GENESIS"
        lines = [l for l in self._path.read_text().splitlines() if l.strip()]
        if not lines:
            return "GENESIS"
        last = json.loads(lines[-1])
        # Recompute the link hash from the last stored entry.
        return _link_hash(last["event"], last["payload_hash"], last["prev_hash"])

    def append(self, event: str, payload: dict) -> AuditEntry:
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        # prev_hash is the link hash of the previous entry (or GENESIS for the first).
        prev_hash = "GENESIS" if self._seq == 0 else self._last_link_hash
        # chain_hash covers only payload_hash + prev_hash (as per test spec).
        chain_hash = hashlib.sha256(
            (payload_hash + prev_hash).encode()
        ).hexdigest()
        entry = AuditEntry(
            seq=self._seq,
            ts=time.time(),
            event=event,
            payload_hash=payload_hash,
            prev_hash=prev_hash,
            chain_hash=chain_hash,
        )
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        self._seq += 1
        # Store the link hash (includes event) for the next entry's prev_hash.
        self._last_link_hash = _link_hash(event, payload_hash, prev_hash)
        return entry

    def verify(self) -> bool:
        if not self._path.exists():
            return True
        lines = [l for l in self._path.read_text().splitlines() if l.strip()]
        for i, line in enumerate(lines):
            e = json.loads(line)
            # Recompute expected prev_hash:
            # - entry 0: must be GENESIS
            # - entry i>0: must equal _link_hash of previous entry
            if i == 0:
                expected_prev = "GENESIS"
            else:
                prev_e = json.loads(lines[i - 1])
                expected_prev = _link_hash(
                    prev_e["event"], prev_e["payload_hash"], prev_e["prev_hash"]
                )
            if e["prev_hash"] != expected_prev:
                return False
            # Verify chain_hash = sha256(payload_hash + prev_hash)
            expected_chain = hashlib.sha256(
                (e["payload_hash"] + e["prev_hash"]).encode()
            ).hexdigest()
            if e["chain_hash"] != expected_chain:
                return False
        return True
