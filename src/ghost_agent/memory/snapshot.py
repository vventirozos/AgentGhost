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
import re
import shutil
import stat
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

KEEP = 7
#: regenerable or not the agent's state — left out of every snapshot.
#: `head_saves` holds llama-server KV-cache slot files (~6 GB each): a cache,
#: not a store — 18 of a 23 GB backups folder on 2026-10-27. `scratch_trash`
#: holds MODEL-written sandbox files kept 7 days with their modes (§4NY): one
#: chmod 000 file there failed every snapshot for a week (review r6)
_SKIP_DIRS = {"eval", "bench", "tokenizer", "llm_recordings", "backups", "head_saves",
              "scratch_trash", "__pycache__"}
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
        # review r4: the sandbox is MODEL-WRITABLE — a text file named
        # `notes.db` (or a project db half-written while copied) refused
        # every snapshot of the stores; sandbox files are copied, not judged
        if db.relative_to(snap).parts[:1] == ("sandbox",):
            continue
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
            # 10-28: k=10 failed a sound store ("1 of 20"): 16 of 8421 live
            # vectors missed their own top-10 — and top-100 — yet were rank 0
            # at distance 0 from k=200 (HNSW nodes the greedy search reaches
            # only with a wide ef). k=n searches the whole index (1.4 s for 20
            # at 8.4k), so only a vector ABSENT from the index fails.
            missed = 0
            for i, e in zip(ids, embs):
                hit = col.query(query_embeddings=[list(e)], n_results=n, include=["distances"])
                if i not in ((hit.get("ids") or [[]])[0]):
                    missed += 1
            return [f"chroma: {missed} of {len(ids)} sampled vectors do not find themselves"] if missed else []
        except Exception as e:  # noqa: BLE001
            return [f"chroma: {type(e).__name__}: {e}"]


#: a snapshot folder's name (`take_snapshot`): `<UTC stamp>-<tag>[-n]`
_SNAP_NAME = re.compile(r"^\d{8}T\d{6}Z-")
#: review r2: a tag is one plain name part — `x.partial` was pruned as
#: unfinished, `a/b` nested out of reach, `../` wrote outside backups/
_TAG = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _rmtree(p: Path) -> bool:
    """Delete ``p`` whole; True when it is gone.

    10-27: the sandbox copy keeps the sandbox's modes, and a read-only project
    folder (`48e0373aaab3`) made `rmtree(ignore_errors=True)` fail on every old
    snapshot without a word — it took the writable top level (MANIFEST.json
    with it) and left the rest, which no later prune counted again: 29
    snapshots against KEEP=7. Our own copies, so a failed entry is made
    writable (its folder too, never ``p``'s own parent — the backups folder)
    and retried."""
    top = os.path.abspath(str(p))

    def _retry(func, path, _exc):
        if not os.path.lexists(path):   # review r3: another prune took it — re-deleting recursed forever
            return
        for q in ((path,) if os.path.abspath(path) == top else (os.path.dirname(path), path)):
            try:
                st = os.lstat(q)
                if stat.S_ISLNK(st.st_mode):      # never through a link to outside the copy
                    continue
                os.chmod(q, st.st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            except OSError:
                pass
        try:
            if func in (os.rmdir, os.unlink, os.remove):
                func(path)
            else:
                # review 10-27: an unreadable folder fails in os.open/os.scandir,
                # which cannot be re-called bare — delete it, now readable, whole
                shutil.rmtree(path, onerror=_retry)
        except FileNotFoundError:       # review r2: another prune removed it meanwhile
            pass
    try:
        shutil.rmtree(p, onerror=_retry)
    except Exception as e:  # noqa: BLE001 — a prune must never fail a published snapshot
        logger.warning("snapshot %s could not be removed: %s", p.name, e)
    return not os.path.lexists(p)


def _retire(p: Path) -> bool:
    """Delete a published snapshot: renamed to ``<name>.partial`` FIRST, so a
    delete that fails half-way (MANIFEST.json gone, the rest left) is retried
    by the daily `.partial` sweep instead of stranded (review r2). True when
    it is gone; False when another prune got there first or it is left."""
    gone = p.with_name(p.name + ".partial")
    try:
        os.rename(p, gone)
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("snapshot %s could not be removed: %s", p.name, e)
        return False
    return _rmtree(gone)


def _ours(p: Path, partial: bool) -> bool:
    """A complete snapshot folder (``partial=False``) or a `.partial` older
    than a day (``partial=True``) of `take_snapshot`'s naming — never a link.
    Review r2: one odd entry (no x bit, renamed away meanwhile) raised out of
    the scan and failed a snapshot that was already published."""
    try:
        if (not _SNAP_NAME.match(p.name) or p.name.endswith(".partial") != partial
                or p.is_symlink() or not p.is_dir()):
            return False
        if partial:
            return time.time() - p.stat().st_mtime > 86400
        return (p / "MANIFEST.json").exists()
    except OSError:
        return False


def prune_snapshots(dest_root: Path, keep: int = KEEP) -> list:
    """Delete all but the newest ``keep`` COMPLETE snapshots, and any stale
    `.partial` older than a day. Returns what was removed.

    Only snapshot-named real folders with a MANIFEST.json count (review
    10-27: a link or a `zz…` folder sorted "newest" and pushed real ones
    out). A stamp-named folder WITHOUT one is never touched: repair scripts
    keep their own backups here under the same names
    (`<stamp>-4qn/memory/fact_ledger.db`)."""
    removed = []
    if not dest_root.is_dir():
        return removed
    done = sorted((p for p in dest_root.iterdir() if _ours(p, False)), key=lambda p: p.name)
    for p in done[:-keep] if keep > 0 else done:
        if _retire(p):
            removed.append(p.name)
    for p in [q for q in dest_root.iterdir() if _ours(q, True)]:
        if _rmtree(p):
            removed.append(p.name)
    return removed


def latest_snapshot(dest_root: Path) -> Optional[Path]:
    if not dest_root.is_dir():
        return None
    # review r4: the same test as prune — a `zz…` folder with a manifest read
    # as the newest (no daily snapshot due), a folder without x raised
    done = sorted(p for p in dest_root.iterdir() if _ours(p, False))
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
    if not _TAG.fullmatch(str(tag)):     # review r5: `$` let "auto\n" through
        return {"ok": False, "path": None, "files": 0, "bytes": 0, "seconds": 0.0,
                "problems": [f"bad tag {tag!r}: letters, digits, '-' and '_' only"]}
    system = Path(home) / "system"
    dest_root = Path(dest_root) if dest_root else system / "backups"
    base = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{tag}"
    # §4MS: unique within the second — two snapshots in one second (a
    # cleanup re-run) collided and the second publish failed (ENOTEMPTY)
    # review r3: the name is RESERVED (an exclusive mkdir of its .partial) —
    # an exists() check let two processes in one second share one .partial;
    # r5: checked again AFTER the reservation — another process can publish
    # the name between the first check and the mkdir
    name, _n = base, 1
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        while True:
            part = dest_root / f"{name}.partial"
            if not (dest_root / name).exists():
                try:
                    part.mkdir()
                    if not (dest_root / name).exists():
                        break
                    part.rmdir()
                except FileExistsError:
                    pass
            _n += 1
            name = f"{base}-{_n}"
    except OSError as e:                 # r5: a read-only or full backups folder
        return {"ok": False, "path": None, "files": 0, "bytes": 0, "seconds": round(time.time() - t0, 1),
                "problems": [f"{type(e).__name__}: {e}"]}
    try:
        manifest = _copy_tree(system, part)
        manifest += _copy_sandbox(Path(home) / "sandbox", part / "sandbox")
        problems = verify_snapshot(part)
    except Exception as e:  # noqa: BLE001
        # r2: a raise (ENOSPC, a locked db, a permission) left ~185 MB of
        # .partial behind on EVERY attempt — clean up and say why
        _rmtree(part)
        return {"ok": False, "path": None, "files": 0, "bytes": 0,
                "problems": [f"{type(e).__name__}: {e}"], "seconds": round(time.time() - t0, 1)}
    out = {"ok": not problems, "path": None, "files": len(manifest),
           "bytes": sum(m["bytes"] for m in manifest), "problems": problems}
    if problems:
        logger.warning("snapshot %s NOT published — %s", name, "; ".join(problems[:3]))
        _rmtree(part)
    else:
        final = dest_root / name
        try:
            # review r4: a full disk here raised out of take_snapshot (and the
            # scripts that call it directly) and left the .partial behind
            (part / "MANIFEST.json").write_text(json.dumps(
                {"created": name, "home": str(home), "files": manifest}, indent=1), encoding="utf-8")
            os.replace(part, final)
        except OSError as e:            # a full disk, or its name published meanwhile
            _rmtree(part)
            out.update(ok=False, problems=[f"publish failed: {type(e).__name__}: {e}"])
            out["seconds"] = round(time.time() - t0, 1)
            return out
        out["path"] = str(final)
        try:
            out["pruned"] = prune_snapshots(dest_root, keep)
        except Exception as e:  # noqa: BLE001 — published is published
            logger.warning("snapshot prune failed: %s", e)
            out["pruned"] = []
    out["seconds"] = round(time.time() - t0, 1)
    return out
