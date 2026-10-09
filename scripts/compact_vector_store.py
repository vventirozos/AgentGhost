#!/usr/bin/env python3
"""§4MN: compact the vector store — agent STOPPED.

The HNSW index keeps one slot per vector EVER added (deleted ones too):
35,087 slots for 423 live vectors (98.6% dead, 58 MB, loaded into RAM whole,
+~0.7 MB/day), and two live vectors no longer found themselves at k=1.
`chroma.sqlite3` is ~97% full-text index nothing queries (39 MB for 89 KB of
text). This rewrites the collection from its OWN stored vectors (no model
call — the same vectors to float precision), checks every id survived and that sampled
vectors find themselves, then optimises the full-text index and VACUUMs.

A verified snapshot (`memory/snapshot.py`) is taken FIRST; nothing runs
without one. Restore = copy the snapshot's `memory/` back (see
`scripts/snapshot_stores.py`).

  PYTHONPATH=src GHOST_HOME=/Users/vasilis/Data/AI/Data/ \\
    /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/compact_vector_store.py [--dry-run] [--no-snapshot]
(`--no-snapshot` only for a scratch COPY in tests.)
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
BATCH = 256


def _segment_stats(mem: Path) -> dict:
    out = {}
    for seg in mem.iterdir():
        if seg.is_dir() and seg.name.count("-") == 4 and (seg / "data_level0.bin").exists():
            out[seg.name] = (seg / "data_level0.bin").stat().st_size
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-snapshot", action="store_true")
    a = ap.parse_args(argv)
    home = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data/"))
    mem = home / "system" / "memory"
    from ghost_agent.memory.store_lock import assert_no_other_writer
    assert_no_other_writer(mem, "compact_vector_store.py")
    import chromadb
    from chromadb.config import Settings
    report = {"before_bytes": (mem / "chroma.sqlite3").stat().st_size, "before_segments": _segment_stats(mem)}
    client = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection("agent_memory")
    n = col.count()
    ids, embs, docs, metas = [], [], [], []
    for off in range(0, n, BATCH):
        page = col.get(include=["embeddings", "documents", "metadatas"], limit=BATCH, offset=off)
        ids += page["ids"]
        embs += [list(e) for e in page["embeddings"]]
        docs += page["documents"]
        metas += page["metadatas"]
    report["rows"] = len(ids)
    if len(ids) != n or len(set(ids)) != n:
        print(json.dumps({**report, "error": f"read {len(ids)} of {n} rows — nothing changed"}))
        return 2
    if a.dry_run:
        print(json.dumps({**report, "dry_run": True}, indent=1))
        return 0
    if not a.no_snapshot:
        from ghost_agent.memory.snapshot import take_snapshot
        snap = take_snapshot(home, None, "pre-compact")
        if not snap.get("ok"):
            print(json.dumps({**report, "error": "snapshot failed — nothing changed", "snapshot": snap}))
            return 2
        report["snapshot"] = snap["path"]
    # the same embedding function the agent opens the collection with
    # (a different one is a config conflict on the next boot)
    from chromadb.utils import embedding_functions
    from ghost_agent.memory.vector import EMBED_MODEL_NAME
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    client.delete_collection("agent_memory")
    new = client.get_or_create_collection(name="agent_memory", embedding_function=ef)
    for off in range(0, len(ids), BATCH):
        new.add(ids=ids[off:off + BATCH], embeddings=embs[off:off + BATCH],
                documents=docs[off:off + BATCH], metadatas=[m or {} for m in metas[off:off + BATCH]])
    got = set(new.get(include=[])["ids"])
    missing = [i for i in ids if i not in got]
    if missing:                       # r2: retry from what we hold in memory before giving up
        idx = {i: k for k, i in enumerate(ids)}
        for off in range(0, len(missing), BATCH):
            chunk = missing[off:off + BATCH]
            new.add(ids=chunk, embeddings=[embs[idx[i]] for i in chunk], documents=[docs[idx[i]] for i in chunk],
                    metadatas=[metas[idx[i]] or {} for i in chunk])
        got = set(new.get(include=[])["ids"])
        missing = [i for i in ids if i not in got]
    misses = 0
    for i, e in list(zip(ids, embs))[:: max(1, len(ids) // 40)]:
        if i not in new.query(query_embeddings=[e], n_results=min(10, len(ids)), include=[])["ids"][0]:
            misses += 1
    report.update({"after_rows": new.count(), "missing": len(missing), "self_query_misses": misses})
    del new, col, client
    # the full-text index nothing queries: optimise it, then reclaim the space
    with sqlite3.connect(mem / "chroma.sqlite3") as c:
        try:
            c.execute("INSERT INTO embedding_fulltext_search(embedding_fulltext_search) VALUES('optimize')")
        except sqlite3.Error as e:
            report["fts_optimize"] = str(e)
    c = sqlite3.connect(mem / "chroma.sqlite3")
    c.execute("VACUUM")
    report["integrity"] = c.execute("PRAGMA integrity_check").fetchone()[0]
    c.close()
    # the old segment folder: chroma leaves it behind; remove only folders
    # the segments table no longer names (the snapshot above holds a copy)
    with sqlite3.connect(f"file:{mem / 'chroma.sqlite3'}?mode=ro", uri=True) as c:
        live = {r[0] for r in c.execute("SELECT id FROM segments")}
    removed = []
    if not missing and report["integrity"] == "ok":
        import shutil
        for seg in mem.iterdir():
            if seg.is_dir() and seg.name.count("-") == 4 and len(seg.name) == 36 and seg.name not in live:
                shutil.rmtree(seg)
                removed.append(seg.name)
    report["old_segments_removed"] = removed
    report["after_bytes"] = (mem / "chroma.sqlite3").stat().st_size
    report["after_segments"] = _segment_stats(mem)
    print(json.dumps(report, indent=1))
    return 0 if not missing and report["integrity"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
