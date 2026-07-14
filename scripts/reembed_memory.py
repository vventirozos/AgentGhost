#!/usr/bin/env python3
"""Re-embed the whole vector store with the currently-configured model.

Needed whenever ``GHOST_EMBED_MODEL`` changes (2026-07-13: all-MiniLM-L6-v2
→ BAAI/bge-small-en-v1.5). Both are 384-d, so NOTHING errors on a mismatch —
the vectors simply stop meaning anything, and retrieval returns
plausible-looking garbage. ``VectorMemory.__init__`` therefore refuses to
boot on a fingerprint mismatch and points here.

What it does:
  1. Reads every (id, document, metadata) out of the Chroma collection.
  2. Deletes the collection and recreates it with the NEW embedding function.
  3. Re-adds everything in batches — Chroma re-embeds on write.
  4. Stamps the embedder sidecar so the agent boots clean.

Safe to re-run. Documents, skills, episodes, identity — all preserved; only
the vectors change. The agent MUST be stopped first (Chroma is single-writer;
a concurrent process risks HNSW corruption).

Usage:
    GHOST_HOME=/Users/vasilis/Data/AI/Data/ \
    PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python \
        scripts/reembed_memory.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BATCH = 128


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be re-embedded, change nothing")
    args = ap.parse_args()

    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    from ghost_agent.memory.vector import (
        EMBED_MODEL_NAME, EMBEDDER_SIDECAR, EXPECTED_EMBED_DIM,
    )

    ghost_home = os.environ.get("GHOST_HOME")
    if not ghost_home:
        print("ERROR: set GHOST_HOME (e.g. /Users/vasilis/Data/AI/Data/)")
        return 2
    chroma_dir = Path(ghost_home) / "system" / "memory"
    if not chroma_dir.exists():
        print(f"ERROR: no vector store at {chroma_dir}")
        return 2

    # Refuse to run against a live agent — Chroma is single-writer.
    try:
        import subprocess
        live = subprocess.run(
            ["pgrep", "-f", "ghost_agent.main"],
            capture_output=True, text=True).stdout.strip()
        if live and not args.dry_run:
            print("ERROR: the agent is RUNNING (pid " + live.replace("\n", ",")
                  + "). Stop it first — Chroma is single-writer and a "
                    "concurrent writer risks HNSW corruption.")
            return 2
    except Exception:
        pass

    print(f"store        : {chroma_dir}")
    print(f"target model : {EMBED_MODEL_NAME} ({EXPECTED_EMBED_DIM}-d)")

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    # Read the old collection WITHOUT an embedding function (we only need the
    # stored text + metadata; nothing is embedded on a plain get()).
    col = client.get_or_create_collection(name="agent_memory")
    got = col.get(include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    print(f"fragments    : {len(ids)}")
    if not ids:
        print("nothing to re-embed (empty store)")
    types = {}
    for m in metas:
        t = (m or {}).get("type", "?")
        types[t] = types.get(t, 0) + 1
    print(f"by type      : {types}")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        return 0

    # Snapshot to disk before destroying anything — a crash mid-migration
    # must not lose the store.
    backup = chroma_dir.parent / f"memory_backup_{int(time.time())}.jsonl"
    with backup.open("w", encoding="utf-8") as f:
        for i, d, m in zip(ids, docs, metas):
            f.write(json.dumps({"id": i, "doc": d, "meta": m},
                               ensure_ascii=False) + "\n")
    print(f"backup       : {backup}")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME)

    client.delete_collection(name="agent_memory")
    new_col = client.get_or_create_collection(
        name="agent_memory", embedding_function=embed_fn)

    t0 = time.time()
    done = 0
    for i in range(0, len(ids), BATCH):
        new_col.upsert(
            ids=ids[i:i + BATCH],
            documents=docs[i:i + BATCH],
            metadatas=[m or {} for m in metas[i:i + BATCH]],
        )
        done += len(ids[i:i + BATCH])
        print(f"  re-embedded {done}/{len(ids)}", end="\r", flush=True)
    el = time.time() - t0
    print(f"\nre-embedded  : {done} fragments in {el:.1f}s")

    (chroma_dir / EMBEDDER_SIDECAR).write_text(json.dumps({
        "model": EMBED_MODEL_NAME,
        "dim": EXPECTED_EMBED_DIM,
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    print(f"stamped      : {chroma_dir / EMBEDDER_SIDECAR}")
    print("\nDone. Start the agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
