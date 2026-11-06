#!/usr/bin/env python3
"""§4TD deploy step (2026-11-06): purge what is left of the removed lesson store, so that `forget`'s "Nothing
stored matches" is true. The vector rows of type `skill` (the lessons' twins, which still quote their text) are
exported and deleted; `skills_playbook.json`, its lock and `skills_pruned_archive.jsonl` are moved out of the
memory dir. Nothing else is touched, and the export is read back before the first delete.

    PYTHONPATH=src python3 scripts/lesson_purge_4td.py <GHOST_HOME> <BACKUP_DIR> [--dry-run]

Refuses to run, dry-run included (a second client on a live store is the risk, not only the delete), when the
agent answers on 127.0.0.1:8000, when another writer holds the store, or when BACKUP_DIR is not writable. Run it
again at will: an export already in BACKUP_DIR is merged into, never overwritten, and a moved file never replaces
one already there. The BACKUP_DIR keeps the lesson text: delete it once the deploy is settled.
"""
import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AGENT_PORT = 8000
COLLECTION = "agent_memory"
ROW_TYPE = "skill"
EXPORT_NAME = "lesson_twins_4td.json"
FILES = ("skills_playbook.json", "skills_playbook.lock", "skills_pruned_archive.jsonl")
BATCH = 500


class Refused(Exception):
    """Nothing was changed; the message says why."""


class ExportBad(Exception):
    """The export was written and is wrong: nothing was deleted or moved."""


def agent_answers(port: int = 0) -> bool:
    """Only a refused connection proves nobody listens; any other failure counts as an answer."""
    try:
        with socket.create_connection(("127.0.0.1", port or AGENT_PORT), timeout=2):
            return True
    except ConnectionRefusedError:
        return False
    except OSError:
        return True


def check_backup_dir(backup: Path, memory_dir: Path) -> None:
    """Writable, proven by writing when it exists; otherwise its nearest existing parent must be, and nothing is
    created here (a dry run changes nothing)."""
    if backup.resolve() == memory_dir.resolve() or memory_dir.resolve() in backup.resolve().parents:
        raise Refused(f"the backup dir {backup} is inside the memory dir")
    probe = backup
    while not probe.exists():
        if probe.parent == probe:
            raise Refused(f"no existing parent for the backup dir {backup}")
        probe = probe.parent
    if not probe.is_dir() or not os.access(probe, os.W_OK | os.X_OK):
        raise Refused(f"the backup dir {backup} is not writable")
    if probe == backup:
        try:
            with tempfile.TemporaryFile(dir=backup):
                pass
        except OSError as exc:
            raise Refused(f"the backup dir {backup} is not writable ({exc})") from exc


def open_collection(memory_dir: Path):
    """The store's collection, or None when there is none. Never creates one. No embedding function: this script
    reads and deletes by id and metadata, it never embeds."""
    if not (memory_dir / "chroma.sqlite3").exists():
        return None
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(memory_dir),
                                       settings=Settings(allow_reset=True, anonymized_telemetry=False))
    if COLLECTION not in [getattr(c, "name", c) for c in client.list_collections()]:
        return None
    return client.get_collection(name=COLLECTION, embedding_function=None)


def read_rows(collection) -> dict:
    """Every row whose type is exactly `skill`: {id: (document, metadata)}. The store's filter is checked again on
    each row, so a store that returned more loses nothing."""
    rows, offset = {}, 0
    while True:
        got = collection.get(where={"type": ROW_TYPE}, include=["documents", "metadatas"], limit=BATCH, offset=offset)
        ids = got.get("ids") or []
        if not ids:
            return rows
        for i, doc, meta in zip(ids, got["documents"], got["metadatas"]):
            if isinstance(meta, dict) and meta.get("type") == ROW_TYPE:
                rows[i] = (doc, meta)
        offset += len(ids)


def _load_export(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    if not (len(ids) == len(docs) == len(metas) == data["count"] == len(set(ids))):
        raise ValueError("its counts do not agree")
    return {i: (d, m) for i, d, m in zip(ids, docs, metas)}


def write_export(path: Path, rows: dict) -> int:
    """Merge `rows` into the export, write it atomically and read it back. Returns the rows the file holds.
    Raises before anything is deleted when the file does not hold them all."""
    merged = {}
    if path.exists():
        try:
            merged = _load_export(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise Refused(f"the export already in the backup dir cannot be read ({exc})") from exc
    for i, row in rows.items():
        merged.setdefault(i, row)          # the first export of a row stands: "never overwritten"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"count": len(merged), "ids": list(merged),
                   "documents": [v[0] for v in merged.values()],
                   "metadatas": [v[1] for v in merged.values()]}, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    back = _load_export(path)
    if back != {i: (v[0], v[1]) for i, v in merged.items()}:
        raise ExportBad("the export did not read back with the same rows; no row was deleted, no file moved "
                        f"(the export file {path} was already written)")
    return len(back)


def delete_rows(collection, ids: list) -> None:
    """Delete exactly `ids`; the store must end exactly that many rows smaller with none of them left."""
    before = collection.count()
    for i in range(0, len(ids), BATCH):
        collection.delete(ids=ids[i:i + BATCH])
    left = collection.get(ids=ids, include=[]).get("ids") if ids else []
    after = collection.count()
    if left or before - after != len(ids):
        raise RuntimeError(f"the store went {before} -> {after} deleting {len(ids)} rows, {len(left or [])} left")


def move_files(memory_dir: Path, backup: Path) -> list:
    """Move the store's files to the backup dir. A file already there is never replaced: the newcomer takes a
    numbered name. Returns the names moved."""
    moved = []
    for name in FILES:
        src = memory_dir / name
        if not src.exists():
            continue
        dst, n = backup / name, 0
        while dst.exists():
            n += 1
            dst = backup / f"{name}.{n}"
        shutil.move(str(src), str(dst))
        moved.append(name)
    return moved


def purge(home, backup, dry_run: bool = False) -> dict:
    """The whole step, in its order; raises Refused with nothing changed. Returns the counts."""
    home, backup = Path(home), Path(backup)
    memory_dir = home / "system" / "memory"
    if not memory_dir.is_dir():
        raise Refused(f"no memory dir at {memory_dir}")
    if agent_answers():
        raise Refused(f"something answers on 127.0.0.1:{AGENT_PORT}: stop the agent first")
    check_backup_dir(backup, memory_dir)
    files = [n for n in FILES if (memory_dir / n).exists()]
    if dry_run:
        return {"dry_run": True, "rows": count_rows_read_only(memory_dir), "rows_deleted": 0, "exported": 0,
                "files": files, "files_moved": []}
    from ghost_agent.memory.store_lock import StoreLockedError, acquire_writer_lock
    try:
        acquire_writer_lock(memory_dir, "lesson_purge_4td")
    except StoreLockedError as exc:
        raise Refused(f"another writer holds the store ({exc})") from exc
    collection = open_collection(memory_dir)
    rows = read_rows(collection) if collection is not None else {}
    backup.mkdir(parents=True, exist_ok=True)
    exported = write_export(backup / EXPORT_NAME, rows) if rows else 0
    if rows:
        delete_rows(collection, list(rows))
    moved = move_files(memory_dir, backup)
    return {"dry_run": False, "rows": len(rows), "rows_deleted": len(rows), "exported": exported,
            "files": files, "files_moved": moved}


def count_rows_read_only(memory_dir: Path) -> int:
    """The dry run's count, from a read-only connection: opening a chroma client can write to the store."""
    db = memory_dir / "chroma.sqlite3"
    if not db.exists():
        return 0
    import sqlite3
    conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        return conn.execute("SELECT COUNT(DISTINCT id) FROM embedding_metadata "
                            "WHERE key = 'type' AND string_value = ?", (ROW_TYPE,)).fetchone()[0]
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Purge the removed lesson store's data (§4TD).")
    ap.add_argument("ghost_home", help="the GHOST_HOME whose system/memory is purged")
    ap.add_argument("backup_dir", help="where the export and the moved files go")
    ap.add_argument("--dry-run", action="store_true", help="count and report; change nothing")
    a = ap.parse_args(argv)
    try:
        r = purge(a.ghost_home, a.backup_dir, dry_run=a.dry_run)
    except Refused as exc:
        print(f"REFUSED: {exc}. Nothing was changed.")
        return 2
    except ExportBad as exc:
        print(f"STOPPED: {exc}.")
        return 3
    except (OSError, RuntimeError) as exc:
        print(f"FAILED part-way: {exc}. The export is complete; run it again to finish.")
        return 4
    if r["dry_run"]:
        print(f"DRY RUN: {r['rows']} vector rows of type '{ROW_TYPE}' would be exported and deleted; "
              f"{len(r['files'])} files would be moved ({', '.join(r['files']) or 'none'}). Nothing was changed.")
    else:
        print(f"Exported: the export holds {r['exported']} rows (0: none this run).")
        print(f"Deleted: {r['rows_deleted']} vector rows of type '{ROW_TYPE}'.")
        print(f"Moved: {len(r['files_moved'])} files ({', '.join(r['files_moved']) or 'none'}).")
        print("The backup dir still quotes the lessons, and so do freed pages of chroma.sqlite3 and any older store "
              "copy: forget's reply is true for what the agent reads, not for every byte on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
