"""Build the planner RAG index (Phase-3 of the eval track).

Indexes the quality==1.0 universal_qa examples from data/training/train.jsonl
into a LanceDB table keyed by (url + element) embeddings, so the planner can
retrieve the top-k most similar PAST test cases and inject them as references.

Per the plan:
  * embedding model: all-MiniLM-L6-v2 (384-dim). Served via Ollama's `all-minilm`
    (same model) rather than the Python sentence-transformers package, which is
    currently broken in this venv (torchao<>torch 2.6 incompatibility poisons the
    transformers import chain). Using Ollama keeps us in-stack ("Ollama only") and
    needs no dependency surgery on the fragile torch/unsloth env.
  * quality filter: index ONLY quality==1.0; drop quality 0.5 (and 0.8) entirely.
  * scope: universal_qa_* sources only (real UI test-planning examples). sprint_*
    rows are irrelevant to UI planning and would pollute retrieval.

Each row:  key  = "<url> <element_role> <element_name>"  (embedded + queried by)
           value= the completion (test-case JSON)          (injected as reference)

Stored in a SEPARATE LanceDB db/table from the app's 768-dim nomic store so the
two never collide.

Run (after any GPU eval finishes, to avoid Ollama model-swap thrash):
  .venv/bin/python eval/build_rag_index.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402
from loguru import logger  # noqa: E402

EMBED_MODEL = "all-minilm"          # Ollama tag for all-MiniLM-L6-v2
EMBED_DIM = 384
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DB_PATH = str(Path.home() / ".qa-agent" / "planner_rag")
TABLE = "planner_examples"

_URL_RE = re.compile(r"หน้า:\s*(\S+)")
_EL_RE = re.compile(r'Element:\s*(\S+)\s+"([^"]+)"')
_NOISE = ("XSS", "SQL", "injection", "Accessibility")


def _records():
    src = ROOT / "data" / "training" / "train.jsonl"
    seen: set[str] = set()
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("quality") != 1.0:                       # DROP 0.5 and 0.8
            continue
        if not d.get("source", "").startswith("universal_qa"):
            continue
        prompt = d.get("prompt", "")
        mu, me = _URL_RE.search(prompt), _EL_RE.search(prompt)
        if not mu or not me:
            continue
        url, role, name = mu.group(1), me.group(1), me.group(2)
        if any(k in name for k in _NOISE):
            continue
        key = f"{url} {role} {name}"
        content = d.get("completion", "")
        if not content:
            continue
        rid = str(abs(hash(f"{key}::{content}")) % (10**16))
        if rid in seen:
            continue
        seen.add(rid)
        yield key, content, {"url": url, "element": name, "role": role,
                             "site": d["source"].replace("universal_qa_", "")}


def _embed_batch(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    r = client.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]


def main() -> None:
    rows = list(_records())
    logger.info(f"indexable quality=1.0 universal_qa examples: {len(rows)}")
    if not rows:
        raise SystemExit("no rows to index")

    keys = [r[0] for r in rows]
    embs: list[list[float]] = []
    with httpx.Client(timeout=120.0) as client:
        for i in range(0, len(keys), 64):
            embs.extend(_embed_batch(client, keys[i:i + 64]))
            if i % 640 == 0:
                logger.info(f"embedded {min(i + 64, len(keys))}/{len(keys)}")
    assert len(embs) == len(rows), f"embed count {len(embs)} != rows {len(rows)}"
    assert len(embs[0]) == EMBED_DIM, f"unexpected dim {len(embs[0])} (want {EMBED_DIM})"

    records = [
        {"id": str(i), "key": keys[i], "content": rows[i][1],
         "embedding": embs[i], "metadata_json": json.dumps(rows[i][2], ensure_ascii=False)}
        for i in range(len(rows))
    ]

    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(DB_PATH)
    if TABLE in db.table_names():
        db.drop_table(TABLE)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("key", pa.string()),
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("metadata_json", pa.string()),
    ])
    tbl = db.create_table(TABLE, schema=schema)
    tbl.add(records)
    logger.info(f"wrote {tbl.count_rows()} rows -> {DB_PATH}/{TABLE} (dim={EMBED_DIM})")


if __name__ == "__main__":
    main()
