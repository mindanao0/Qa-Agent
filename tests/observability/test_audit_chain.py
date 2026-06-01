"""Tests for CryptoAuditTrail."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.observability.audit_chain import AuditEntry, CryptoAuditTrail


@pytest.fixture
def trail(tmp_path: Path) -> CryptoAuditTrail:
    return CryptoAuditTrail(path=tmp_path / "audit.jsonl")


def test_first_entry_has_genesis_prev_hash(trail: CryptoAuditTrail):
    entry = trail.append("test_event", {"key": "value"})
    assert entry.seq == 0
    assert entry.prev_hash == "GENESIS"


def test_chain_grows_sequentially(trail: CryptoAuditTrail):
    for i in range(5):
        e = trail.append(f"event_{i}", {"i": i})
        assert e.seq == i


def test_verify_returns_true_for_valid_chain(trail: CryptoAuditTrail):
    for i in range(10):
        trail.append(f"event_{i}", {"i": i})
    assert trail.verify() is True


def test_verify_returns_false_when_tampered(trail: CryptoAuditTrail, tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    for i in range(3):
        trail.append(f"event_{i}", {"i": i})
    lines = path.read_text().splitlines()
    # Corrupt the second entry
    entry_dict = json.loads(lines[1])
    entry_dict["event"] = "TAMPERED"
    lines[1] = json.dumps(entry_dict)
    path.write_text("\n".join(lines) + "\n")
    assert trail.verify() is False


def test_append_writes_to_disk_immediately(trail: CryptoAuditTrail, tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    trail.append("disk_write", {"x": 1})
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_chain_hash_covers_payload_and_prev(trail: CryptoAuditTrail):
    import hashlib
    e0 = trail.append("first", {"a": 1})
    payload_hash = hashlib.sha256(
        json.dumps({"a": 1}, sort_keys=True).encode()
    ).hexdigest()
    assert e0.payload_hash == payload_hash
    expected_chain = hashlib.sha256(
        (payload_hash + "GENESIS").encode()
    ).hexdigest()
    assert e0.chain_hash == expected_chain
