"""§4MN: verified, paired snapshots of the agent's stores.

There was no backup of any kind: no Time Machine destination, no scheduled
copy — only the copies repair scripts make before they run, and those copied
`chroma.sqlite3` WITHOUT its HNSW segment folder, which cannot be restored
(reproduced: "Error finding id"). A snapshot here is:

* **complete for the stores that must agree** — every SQLite database through
  the online-backup API (consistent even while the agent writes), the chroma
  segment folders WITH `chroma.sqlite3`, and the JSON/JSONL stores;
* **verified before it counts** — written to `<name>.partial/`, then every
  database passes `integrity_check`, chroma opens from the copy and finds its
  own vectors; only then is it renamed to `<name>/` (an unverified snapshot
  is never left looking complete);
* **bounded** — the newest ``KEEP`` complete snapshots are kept.

Operator decision (2026-10-08): local only for now (`system/backups/`); an
off-disk destination is a later choice — `take_snapshot(dest_root=...)`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

KEEP = 7
#: regenerable or not the agent's state — left out of every snapshot
_SKIP_DIRS = {"eval", "bench", "tokenizer", "llm_recordings", "backups", "__pycache__"}
_SKIP_FILES = {"ghost-agent.log", "ghost-agent.log.1"}
_SQLITE_SUFFIXES = (".db", ".sqlite3")


def _is_repair_copy(name: str) -> bool:
    """A repair script's copy or a crash product — never a live store. r2:
    a bare "removed" also matched the live `profile_removed.jsonl`; the
    repair artifacts are `*.removed-4mi.jsonl` / `*_removed`."""
    n = name.lower()
    return (n.endswith((".bak", ".tmp", ".lock")) or ".pre-" in n or ".bak-" in n or ".removed-" in n
            or n.endswith("_removed") or ".corrupt" in n or n.endswith(("-wal", "-shm", "-journal")))


#: the sandbox (project files, generated images) is snapshotted too — a
#: restored `projects.db` without its folders is inconsistent (r2) — minus
#: any single file bigger than this (a downloaded model, a dataset)
SANDBOX_FILE_MAX = 50 << 20


def _sqlite_backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as s, sqlite3.connect(dst) as d:
        s.backup(d)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_sandbox(sandbox: Path, out: Path) -> list:
    """The sandbox is MODEL-WRITABLE: copied with `copytree_nofollow`
    (every entry opened O_NOFOLLOW relative to its parent fd — a planted
    link can never pull a host file into the snapshot), skipping caches and
    any file over `SANDBOX_FILE_MAX`."""
    if not sandbox.is_dir():
        return []
    from ..tools.file_system import copytree_nofollow

    def _ignore(dirpath, names):
        skip = set()
        for n in names:
            if n.startswith(".") or n in ("node_modules", "__pycache__", ".venv"):
                skip.add(n)
                continue
            try:
                st = os.lstat(os.path.join(dirpath, n))
                if st.st_size > SANDBOX_FILE_MAX and not os.path.isdir(os.path.join(dirpath, n)):
                    skip.add(n)
            except OSError:
                skip.add(n)
        return skip
    copytree_nofollow(sandbox, out, sandbox, ignore=_ignore, dirs_exist_ok=True)
    manifest = []
    for root, _dirs, names in os.walk(out):          # our own copy, not the sandbox
        for n in names:
            f = Path(root) / n
            if f.is_file() and not f.is_symlink():
                manifest.append({"path": "sandbox/" + str(f.relative_to(out)), "bytes": f.stat().st_size})
    return manifest


def _copy_tree(system: Path, out: Path) -> list:
    """Copy `system/` into `out/` — SQLite through the backup API, chroma
    segment folders after their `chroma.sqlite3` — and return the manifest."""
    manifest = []
    files = []
    for root, dirs, names in os.walk(system):
        rel_root = Path(root).relative_to(system)
        dirs[:] = sorted(d for d in dirs if not (rel_root == Path(".") and d in _SKIP_DIRS)
                         and not _is_repair_copy(d) and not d.startswith("."))
        for n in sorted(names):
            if (rel_root == Path(".") and n in _SKIP_FILES) or _is_repair_copy(n) or n.startswith("."):
                continue
            files.append(rel_root / n)
    # databases FIRST, then everything else: a chroma segment file copied
    # after its sqlite is at least as new as it (an older segment than its
    # sqlite is what made the repair backups unrestorable)
    files.sort(key=lambda r: (0 if r.suffix in _SQLITE_SUFFIXES else 1, str(r)))
    for rel in files:
        src, dst = system / rel, out / rel
        try:
            if src.is_symlink() or not src.is_file():
                continue
            if src.suffix in _SQLITE_SUFFIXES:
                _sqlite_backup(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            manifest.append({"path": str(rel), "bytes": dst.stat().st_size, "sha256": _sha256(dst)})
        except FileNotFoundError:
            continue                        # a store rotated away while we walked
    return manifest


def verify_snapshot(snap: Path) -> list:
    """Problems found restoring ``snap`` ([] = sound): every database passes
    `integrity_check`; chroma opens from the copy and each sampled vector
    finds itself."""
    problems = []
    for db in sorted(snap.rglob("*")):
        if db.suffix in _SQLITE_SUFFIXES and db.is_file():
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                    r = c.execute("PRAGMA integrity_check").fetchone()[0]
                if r != "ok":
                    problems.append(f"{db.relative_to(snap)}: integrity {r}")
            except sqlite3.Error as e:
                problems.append(f"{db.relative_to(snap)}: {e}")
    chroma = snap / "memory" / "chroma.sqlite3"
    if chroma.exists():
        problems += _verify_chroma(chroma.parent)
    return problems


def _verify_chroma(memory_copy: Path) -> list:
    """Open the copied chroma store (a scratch copy of it — opening may
    write) and check that sampled vectors find themselves."""
    import tempfile
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:  # noqa: BLE001
        return [f"chroma: cannot import ({e})"]
    with tempfile.TemporaryDirectory(prefix="ghost-snapverify-") as td:
        tmp = Path(td)
        shutil.copy2(memory_copy / "chroma.sqlite3", tmp / "chroma.sqlite3")
        for seg in memory_copy.iterdir():
            if seg.is_dir() and len(seg.name) == 36 and seg.name.count("-") == 4:
                shutil.copytree(seg, tmp / seg.name)
        try:
            client = chromadb.PersistentClient(path=str(tmp), settings=Settings(anonymized_telemetry=False))
            col = client.get_collection("agent_memory")
            n = col.count()
            if n == 0:
                return []
            # a RANDOM sample across the store (r2: the first 20 rows are the
            # oldest — a stale segment found "1 of 20" there)
            import random
            offs = sorted(random.sample(range(n), min(n, 20)))
            got = {"ids": [], "embeddings": []}
            for o in offs:
                g = col.get(limit=1, offset=o, include=["embeddings"])
                got["ids"] += g.get("ids") or []
                got["embeddings"] += list(g.get("embeddings") if g.get("embeddings") is not None else [])
            ids, embs = got.get("ids") or [], got.get("embeddings")
            embs = [] if embs is None else list(embs)
            missed = 0
            for i, e in zip(ids, embs):
                hit = col.query(query_embeddings=[list(e)], n_results=min(n, 10))
                if i not in ((hit.get("ids") or [[]])[0]):
                    missed += 1
            return [f"chroma: {missed} of {len(ids)} sampled vectors do not find themselves"] if missed else []
        except Exception as e:  # noqa: BLE001
            return [f"chroma: {type(e).__name__}: {e}"]


def prune_snapshots(dest_root: Path, keep: int = KEEP) -> list:
    """Delete all but the newest ``keep`` COMPLETE snapshots, and any stale
    `.partial` older than a day. Returns what was removed."""
    removed = []
    if not dest_root.is_dir():
        return removed
    done = sorted((p for p in dest_root.iterdir() if p.is_dir() and not p.name.endswith(".partial")
                   and (p / "MANIFEST.json").exists()), key=lambda p: p.name)
    for p in done[:-keep] if keep > 0 else done:
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p.name)
    for p in dest_root.glob("*.partial"):
        if time.time() - p.stat().st_mtime > 86400:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
    return removed


def latest_snapshot(dest_root: Path) -> Optional[Path]:
    if not dest_root.is_dir():
        return None
    done = sorted(p for p in dest_root.iterdir() if p.is_dir() and (p / "MANIFEST.json").exists()
                  and not p.name.endswith(".partial"))
    return done[-1] if done else None


def take_snapshot_isolated(home: Path, dest_root: Optional[Path] = None, tag: str = "auto",
                           hold=None, timeout_s: float = 1800.0) -> dict:
    """`take_snapshot` from a CHILD process, while holding ``hold`` (the
    vector store's write lock) when given.

    §4MR: taken inside the agent process, the copy of `chroma.sqlite3` tore
    while the agent wrote vectors — chroma writes through its own bundled
    SQLite, and POSIX file locks do not exclude within one process (6 of 6
    torn in one measurement; verification refused every one, so the cost was
    a missed daily backup). Across processes the file locks hold (5 of 5
    clean); holding the in-process write lock as well pauses the agent's own
    writers for the copy."""
    import subprocess
    import sys
    pkg_root = str(Path(__file__).resolve().parents[2])        # …/src
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = ("import json, sys; from pathlib import Path; "
            "from ghost_agent.memory.snapshot import take_snapshot; "
            "d = sys.argv[2] or None; "
            "print(json.dumps(take_snapshot(Path(sys.argv[1]), Path(d) if d else None, sys.argv[3])))")
    argv = [sys.executable, "-c", code, str(home), str(dest_root or ""), str(tag)]

    def _run():
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout_s)
        lines = [l for l in (p.stdout or "").splitlines() if l.strip().startswith("{")]
        if p.returncode != 0 or not lines:
            return {"ok": False, "path": None, "files": 0, "bytes": 0, "seconds": 0,
                    "problems": [f"snapshot child exited {p.returncode}: {(p.stderr or '')[-300:]}"]}
        return json.loads(lines[-1])
    if hold is None:
        return _run()
    with hold:
        return _run()


def take_snapshot(home: Path, dest_root: Optional[Path] = None, tag: str = "auto", keep: int = KEEP) -> dict:
    """Copy, verify, publish, prune. Returns
    ``{"ok", "path", "bytes", "files", "problems", "seconds", "pruned"}``."""
    t0 = time.time()
    system = Path(home) / "system"
    dest_root = Path(dest_root) if dest_root else system / "backups"
    base = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{tag}"
    dest_root.mkdir(parents=True, exist_ok=True)
    # §4MS: unique within the second — two snapshots in one second (a
    # cleanup re-run) collided and the second publish failed (ENOTEMPTY)
    name, _n = base, 1
    while (dest_root / name).exists() or (dest_root / f"{name}.partial").exists():
        _n += 1
        name = f"{base}-{_n}"
    part = dest_root / f"{name}.partial"
    try:
        manifest = _copy_tree(system, part)
        manifest += _copy_sandbox(Path(home) / "sandbox", part / "sandbox")
        problems = verify_snapshot(part)
    except Exception as e:  # noqa: BLE001
        # r2: a raise (ENOSPC, a locked db, a permission) left ~185 MB of
        # .partial behind on EVERY attempt — clean up and say why
        shutil.rmtree(part, ignore_errors=True)
        return {"ok": False, "path": None, "files": 0, "bytes": 0,
                "problems": [f"{type(e).__name__}: {e}"], "seconds": round(time.time() - t0, 1)}
    out = {"ok": not problems, "path": None, "files": len(manifest),
           "bytes": sum(m["bytes"] for m in manifest), "problems": problems}
    if problems:
        logger.warning("snapshot %s NOT published — %s", name, "; ".join(problems[:3]))
        shutil.rmtree(part, ignore_errors=True)
    else:
        (part / "MANIFEST.json").write_text(json.dumps(
            {"created": name, "home": str(home), "files": manifest}, indent=1), encoding="utf-8")
        final = dest_root / name
        os.replace(part, final)
        out["path"] = str(final)
        out["pruned"] = prune_snapshots(dest_root, keep)
    out["seconds"] = round(time.time() - t0, 1)
    return out
