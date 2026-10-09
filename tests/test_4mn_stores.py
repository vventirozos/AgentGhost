"""§4MN (2026-10-08): health of the data stores. Each test names the defect
it FAILS on."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ── N1: one writer process per store ───────────────────────────────────────
def test_a_second_process_cannot_open_the_store_for_writing(tmp_path):
    """Fails where a repair script and the agent both wrote chroma — the
    store then segfaulted on every open (4 of 7 runs) and the agent
    crash-looped at boot."""
    from ghost_agent.memory.store_lock import acquire_writer_lock, release_writer_lock, LOCK_NAME
    acquire_writer_lock(tmp_path, "test-holder")
    try:
        child = subprocess.run(
            [sys.executable, "-c",
             "import sys; from ghost_agent.memory.store_lock import acquire_writer_lock, StoreLockedError\n"
             f"try:\n    acquire_writer_lock({str(tmp_path)!r}, 'second')\nexcept StoreLockedError as e:\n"
             "    print('LOCKED', e); sys.exit(3)\nsys.exit(0)"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=60)
        assert child.returncode == 3 and "test-holder" in child.stdout, child.stdout + child.stderr
        acquire_writer_lock(tmp_path)                        # re-entrant in the holder
    finally:
        release_writer_lock(tmp_path)
    free = subprocess.run(
        [sys.executable, "-c", "from ghost_agent.memory.store_lock import acquire_writer_lock\n"
         f"acquire_writer_lock({str(tmp_path)!r}, 'after')"],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=60)
    assert free.returncode == 0, free.stderr                 # released → the next writer may open it
    assert (tmp_path / LOCK_NAME).exists()


def test_the_vector_store_takes_the_lock_before_chroma_opens():
    import ast
    import inspect
    from ghost_agent.memory import vector as V
    init = next(n for n in ast.walk(ast.parse(inspect.getsource(V)))
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                and any(isinstance(c, ast.Attribute) and c.attr == "PersistentClient" for c in ast.walk(n)))
    lock = min(n.lineno for n in ast.walk(init) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "acquire_writer_lock")
    client = min(n.lineno for n in ast.walk(init) if isinstance(n, ast.Attribute) and n.attr == "PersistentClient")
    assert lock < client
    # …outside the try that turns a chroma failure into "collection=None"
    tries = [t for t in ast.walk(init) if isinstance(t, ast.Try)]
    assert not any(t.lineno <= lock <= max(getattr(x, "lineno", 0) for x in ast.walk(t)) for t in tries)


def test_a_locked_store_stops_the_boot_instead_of_running_memoryless():
    import ast
    import inspect
    from ghost_agent import main as Mn
    tree = ast.parse(inspect.getsource(Mn))
    hit = [n for n in ast.walk(tree) if isinstance(n, ast.If) and "StoreLockedError" in ast.dump(n.test)]
    assert hit and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "exit" for c in ast.walk(hit[0]))


def test_repair_scripts_refuse_beside_a_writer(tmp_path):
    from ghost_agent.memory.store_lock import acquire_writer_lock, release_writer_lock
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    acquire_writer_lock(mem, "the agent")
    try:
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "memory_repair_4mm.py"), "--apply"],
                           capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "GHOST_HOME": str(tmp_path)})
        assert p.returncode != 0 and "refusing to run" in (p.stdout + p.stderr), p.stdout + p.stderr
    finally:
        release_writer_lock(mem)


# ── N3: the write paths that lose or tear data ─────────────────────────────
def test_a_torn_embedder_sidecar_does_not_refuse_the_boot(tmp_path):
    """Fails where a sidecar torn mid-write read as "legacy MiniLM store"
    and the agent refused to boot forever, advising a full re-embed."""
    from ghost_agent.memory.vector import _embedder_sidecar_mismatch
    p = tmp_path / "embedder.json"
    p.write_text('{"model": "BAAI/bge-sm')
    assert _embedder_sidecar_mismatch(p, "BAAI/bge-small-en-v1.5", 423) is None
    assert list(tmp_path.glob("embedder.json.torn-*"))                       # the evidence is kept
    p.unlink(missing_ok=True)
    assert _embedder_sidecar_mismatch(p, "BAAI/bge-small-en-v1.5", 423)       # truly absent: still refused
    p.write_text('{"model": "other-model"}')
    assert _embedder_sidecar_mismatch(p, "BAAI/bge-small-en-v1.5", 423)       # a real mismatch: still refused


def test_the_sidecar_is_written_atomically_and_only_when_it_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ghost_agent.memory import vector as V
    from ghost_agent.utils import json_store as J
    calls = []
    real = J.write_json_atomic
    monkeypatch.setattr(J, "write_json_atomic", lambda path, data, **k: (calls.append(path), real(path, data, **k)))
    vm = SimpleNamespace(_embedder_sidecar=tmp_path / "embedder.json")
    V.VectorMemory._stamp_embedder_sidecar(vm)
    V.VectorMemory._stamp_embedder_sidecar(vm)              # unchanged → no second write
    assert len(calls) == 1 and json.loads((tmp_path / "embedder.json").read_text())["model"] == V.EMBED_MODEL_NAME


def test_a_forgotten_episode_is_archived_whole_and_durably(tmp_path):
    """Fails where the archive dropped the label (outcome_success), req_id
    and the actions' success — a forget could not be undone faithfully."""
    import sqlite3
    from ghost_agent.memory.episodes import EpisodicMemory
    em = EpisodicMemory(tmp_path)
    with sqlite3.connect(em.db_path) as c:
        c.execute("INSERT INTO episodes (trigger, outcome, outcome_success, lesson, timestamp, req_id, cluster_id) "
                  "VALUES ('t', 'o', 1, 'l', 1.0, 'req-1', 'c1')")
        c.execute("INSERT INTO episode_actions (episode_id, action_order, tool_name, success) VALUES (1, 0, 'x', 0)")
    assert em.delete_episodes([1], None, reason="test") == 1
    rec = json.loads((tmp_path / "episodes_forgotten.jsonl").read_text().splitlines()[-1])
    assert rec["outcome_success"] == 1 and rec["req_id"] == "req-1" and rec["cluster_id"] == "c1"
    assert rec["actions"][0]["success"] == 0 and rec["forgot"] == "test"


def test_the_archive_purge_runs_under_the_store_lock():
    """Fails where the purge ran after the lock was released: a concurrent
    forget's archive line was lost between the purge's read and its swap."""
    import ast
    import inspect
    from ghost_agent.memory import episodes as E
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(E)))
              if isinstance(n, ast.FunctionDef) and n.name == "delete_episodes")
    locked = next(w for w in ast.walk(fn) if isinstance(w, ast.With) and "_lock" in ast.dump(w.items[0]))
    purges = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_purge_forgotten_archive"]
    inside = {id(c) for c in ast.walk(locked)}
    assert purges and all(id(c) in inside for c in purges)


def test_an_archived_graph_edge_keeps_its_validity_window(tmp_path):
    """Fails where a restored EXPIRED edge came back as current."""
    import sqlite3
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    with sqlite3.connect(g.db_path) as c:
        c.execute("INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from, valid_until) "
                  "VALUES ('bob', 'WORKS_AT', 'google', 1, datetime('now'), 100.0, 200.0)")
    assert g._archive_rows("test", [("bob", "WORKS_AT", "google", 1, "t")])
    rec = json.loads((tmp_path / g._ARCHIVE_FILENAME).read_text().splitlines()[-1])
    assert rec["valid_from"] == 100.0 and rec["valid_until"] == 200.0


def test_a_profile_removal_is_archived_first_and_refused_when_it_cannot_be(tmp_path, monkeypatch):
    """Fails where the owner's own facts were the one store deleted with no
    copy — a confirmed forget could not be undone."""
    from ghost_agent.memory.profile import ProfileMemory
    pm = ProfileMemory(tmp_path)
    pm.update("family", "children", "Leonidas")
    out = pm.delete("family", "children")
    assert "Removed" in out
    rec = json.loads((tmp_path / "profile_removed.jsonl").read_text().splitlines()[-1])
    assert rec["how"] == "delete" and "Leonidas" in json.dumps(rec["value"])
    pm.update("family", "children", "Leonidas")
    monkeypatch.setattr(pm, "_archive_removed", lambda *a, **k: False)
    assert "NOTHING was removed" in pm.delete("family", "children")
    assert "Leonidas" in json.dumps(pm.load_raw())


def test_reset_all_keeps_every_row_before_wiping(tmp_path, monkeypatch):
    """Fails where the full vector wipe kept no copy — identity, document
    and auto rows do not re-embed themselves."""
    import asyncio
    from types import SimpleNamespace
    import ghost_agent.tools.memory as M
    from tests._wipe_confirm import confirming
    rows = {f"id{i}": (f"doc {i}", {"type": "identity" if i % 2 else "document"}) for i in range(1203)}

    class _Coll:
        def get(self, ids=None, include=None):
            ids = list(rows) if ids is None else [i for i in ids if i in rows]
            out = {"ids": ids, "metadatas": [rows[i][1] for i in ids]}
            if include and "documents" in include:
                out["documents"] = [rows[i][0] for i in ids]
            return out

        def delete(self, ids=None, where=None):
            for i in ids or []:
                rows.pop(i, None)
    mem = SimpleNamespace(collection=_Coll(), chroma_dir=tmp_path)
    before = dict(rows)
    out = asyncio.run(confirming(M.tool_knowledge_base)(action="reset_all", memory_system=mem))
    assert "Wiped" in out and not rows, out
    dumps = list(tmp_path.glob("vector_reset_*.jsonl"))
    assert len(dumps) == 1
    kept = {json.loads(l)["id"]: json.loads(l) for l in dumps[0].read_text().splitlines()}
    assert set(kept) == set(before) and kept["id7"]["document"] == "doc 7"
    assert kept["id7"]["metadata"]["type"] == "identity"


def test_reset_all_wipes_nothing_when_the_copy_cannot_be_kept(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import ghost_agent.tools.memory as M
    from tests._wipe_confirm import confirming
    rows = {"a": ("doc a", {"type": "identity"})}

    class _Coll:
        def get(self, ids=None, include=None):
            ids = list(rows) if ids is None else ids
            return {"ids": ids, "metadatas": [rows[i][1] for i in ids], "documents": [rows[i][0] for i in ids]}

        def delete(self, ids=None, where=None):
            for i in ids or []:
                rows.pop(i, None)
    gone = tmp_path / "not-a-dir"
    mem = SimpleNamespace(collection=_Coll(), chroma_dir=gone)          # the dump cannot be written
    out = asyncio.run(confirming(M.tool_knowledge_base)(action="reset_all", memory_system=mem))
    assert "NOTHING was wiped" in out and rows == {"a": ("doc a", {"type": "identity"})}, out


# ── N4: the reapers that never ran ─────────────────────────────────────────
def test_the_dream_reconcile_reaps_orphan_skill_vectors():
    """Fails where the purge ran only when the owner called manage_skills —
    6 of 8 acquired-skill vectors stayed orphaned for days."""
    import ast
    import inspect
    from ghost_agent.core import dream as D
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(D)))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_reconcile_memory_stores")
    assert any(isinstance(c, ast.Attribute) and c.attr == "purge_orphaned_skill_embeddings" for c in ast.walk(fn))


def test_a_hard_delete_removes_only_that_projects_changelog(tmp_path):
    from types import SimpleNamespace
    from ghost_agent.tools.projects import _remove_project_changelog
    (tmp_path / "CHANGELOG.abc123.md").write_text("x")
    (tmp_path / "CHANGELOG.other.md").write_text("y")
    ctx = SimpleNamespace(workspace_model=SimpleNamespace(root=tmp_path))
    assert _remove_project_changelog(ctx, "abc123") is True
    assert not (tmp_path / "CHANGELOG.abc123.md").exists() and (tmp_path / "CHANGELOG.other.md").exists()
    outside = tmp_path / "CHANGELOG..."                          # f"CHANGELOG.{pid}.md" with pid "../victim"
    outside.mkdir()
    (outside / "victim.md").write_text("z")                                # reachable by "../victim"
    assert _remove_project_changelog(ctx, "../victim") is False
    assert (outside / "victim.md").exists()


def test_the_hard_delete_calls_the_changelog_removal():
    import ast
    import inspect
    from ghost_agent.tools import projects as P
    names = {n.func.id for n in ast.walk(ast.parse(inspect.getsource(P)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_remove_project_changelog" in names


# ── N2: verified, paired snapshots ─────────────────────────────────────────
def _home_with_stores(tmp_path):
    import sqlite3
    sysd = tmp_path / "home" / "system"
    mem = sysd / "memory"
    mem.mkdir(parents=True)
    with sqlite3.connect(mem / "episodic_memory.db") as c:
        c.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, trigger TEXT)")
        c.execute("INSERT INTO episodes (trigger) VALUES ('hello')")
    (mem / "user_profile.json").write_text('{"root": {"name": "x"}}')
    (mem / "skills_playbook.json.pre-4mk.bak").write_text("[]")          # a repair copy: left out
    (mem / "composed_skills.json.bak").write_text("[]")                  # a plain .bak too
    (sysd / "eval").mkdir()
    (sysd / "eval" / "big.bin").write_bytes(b"0" * 1000)                 # regenerable: left out
    (sysd / "autonomous_activity.jsonl").write_text('{"a": 1}\n')
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("agent_memory")
    col.add(ids=["e1", "e2", "e3"], embeddings=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            documents=["a", "b", "c"], metadatas=[{"type": "episode"}] * 3)
    # enough rows that chroma FLUSHES them to the HNSW segment folder (its
    # sync threshold is 1000) — below it the vectors live in the SQLite queue
    # and a copy WITHOUT the segment would wrongly verify
    import random
    rnd = random.Random(3)
    for off in range(0, 1200, 300):
        col.add(ids=[f"x{i}" for i in range(off, off + 300)],
                embeddings=[[rnd.random(), rnd.random(), rnd.random()] for _ in range(300)],
                documents=[f"x{i}" for i in range(off, off + 300)], metadatas=[{"type": "auto"}] * 300)
    del client
    return tmp_path / "home"


def test_a_snapshot_pairs_chroma_with_its_segment_and_is_verified(tmp_path):
    """Fails where the only copies were `chroma.sqlite3` alone — not
    restorable ("Error finding id") — and nothing was verified."""
    from ghost_agent.memory.snapshot import take_snapshot, verify_snapshot
    home = _home_with_stores(tmp_path)
    res = take_snapshot(home, tmp_path / "snaps", "t")
    assert res["ok"], res
    snap = Path(res["path"])
    names = {p.name for p in snap.rglob("*")}
    assert "chroma.sqlite3" in names and "episodic_memory.db" in names and "user_profile.json" in names
    assert any(p.is_dir() and p.name.count("-") == 4 for p in (snap / "memory").iterdir())   # the segment
    assert "skills_playbook.json.pre-4mk.bak" not in names and "big.bin" not in names
    assert "composed_skills.json.bak" not in names
    assert verify_snapshot(snap) == []
    assert json.loads((snap / "MANIFEST.json").read_text())["files"]


def test_a_snapshot_that_fails_verification_is_never_published(tmp_path, monkeypatch):
    from ghost_agent.memory import snapshot as S
    home = _home_with_stores(tmp_path)
    monkeypatch.setattr(S, "verify_snapshot", lambda p: ["chroma: broken"])
    res = S.take_snapshot(home, tmp_path / "snaps", "t")
    assert res["ok"] is False and res["path"] is None
    assert list((tmp_path / "snaps").iterdir()) == []                     # no .partial left, nothing "complete"
    assert S.latest_snapshot(tmp_path / "snaps") is None


def test_a_restored_snapshot_opens_and_finds_its_vectors(tmp_path):
    """The restore an operator would do: copy the snapshot's memory/ back."""
    import shutil
    import chromadb
    from chromadb.config import Settings
    from ghost_agent.memory.snapshot import take_snapshot
    home = _home_with_stores(tmp_path)
    snap = Path(take_snapshot(home, tmp_path / "snaps", "t")["path"])
    restored = tmp_path / "restored"
    shutil.copytree(snap / "memory", restored)
    col = chromadb.PersistentClient(path=str(restored), settings=Settings(anonymized_telemetry=False)) \
        .get_collection("agent_memory")
    assert col.count() == 1203
    assert col.query(query_embeddings=[[0, 1.0, 0]], n_results=1)["ids"][0] == ["e2"]


def test_snapshots_are_bounded(tmp_path):
    from ghost_agent.memory.snapshot import prune_snapshots
    root = tmp_path / "snaps"
    for i in range(10):
        d = root / f"2026100{i}T000000Z-auto"
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text("{}")
    removed = prune_snapshots(root, keep=7)
    assert len(removed) == 3 and sorted(p.name for p in root.iterdir())[0] == "20261003T000000Z-auto"


def test_the_dream_takes_a_snapshot_at_most_daily(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from ghost_agent.core.dream import Dreamer
    from ghost_agent.memory import snapshot as S
    calls = []
    # §4MR: the dream calls the ISOLATED snapshot (a child process, the
    # vector writers held); stub that one
    monkeypatch.setattr(S, "take_snapshot_isolated", lambda home, dest, tag, hold=None: (
        calls.append(tag), (dest / "20261008T000000Z-auto").mkdir(parents=True, exist_ok=True),
        (dest / "20261008T000000Z-auto" / "MANIFEST.json").write_text("{}"),
        {"ok": True, "path": str(dest / "20261008T000000Z-auto"), "files": 1, "bytes": 1, "seconds": 0.1})[-1])
    d = Dreamer.__new__(Dreamer)
    (tmp_path / "system" / "memory").mkdir(parents=True)
    d.context = SimpleNamespace(memory_dir=tmp_path / "system" / "memory")
    assert asyncio.run(d._maybe_snapshot()).startswith("snapshot")
    assert asyncio.run(d._maybe_snapshot()) == ""                       # the newest is fresh: none taken
    assert calls == ["auto"]
    monkeypatch.setenv("GHOST_SNAPSHOTS", "0")
    monkeypatch.setattr(Dreamer, "SNAPSHOT_EVERY_S", 0)
    assert asyncio.run(d._maybe_snapshot()) == ""


def test_compaction_keeps_every_vector_and_drops_the_dead_slots(tmp_path):
    """Fails where the index kept one slot per vector EVER added — 35,087
    slots for 423 live rows, two of which no longer found themselves."""
    import chromadb
    from chromadb.config import Settings
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("agent_memory")
    import random
    rnd = random.Random(7)
    vecs = {f"v{i}": [rnd.random() for _ in range(384)] for i in range(300)}
    col.add(ids=list(vecs), embeddings=list(vecs.values()), documents=[f"d{i}" for i in range(300)],
            metadatas=[{"type": "episode"}] * 300)
    col.delete(ids=[f"v{i}" for i in range(20, 300)])             # 280 dead slots
    keep = {k: vecs[k] for k in (f"v{i}" for i in range(20))}
    del col, client
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "compact_vector_store.py"), "--no-snapshot"],
                       capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "GHOST_HOME": str(tmp_path),
                            "HF_HUB_OFFLINE": "1"})
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    rep = json.loads(p.stdout[p.stdout.index("{"):])
    assert rep["rows"] == rep["after_rows"] == 20 and rep["missing"] == 0 and rep["self_query_misses"] == 0
    assert rep["old_segments_removed"]
    col = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False)) \
        .get_collection("agent_memory")
    assert sorted(col.get(include=[])["ids"]) == sorted(keep)


# ── §4MN r2: the fresh reader's findings inside the fixes ─────────────────
def test_a_refused_profile_removal_is_never_reported_as_removed():
    """r2 M1: "✅ Profile: Removed assets.car" printed over a refusal."""
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.tools.memory import _profile_line
    line = _profile_line(ProfileMemory._ARCHIVE_FAILED, "Removed assets.car")
    assert line.startswith("⚠️") and "NOTHING was removed" in line


def test_a_snapshot_that_raises_leaves_nothing_behind(tmp_path, monkeypatch):
    """r2 M2: a PermissionError / ENOSPC / locked db left ~185 MB of
    .partial on every attempt."""
    from ghost_agent.memory import snapshot as S
    home = tmp_path / "home"
    (home / "system").mkdir(parents=True)
    monkeypatch.setattr(S, "_copy_tree", lambda system, out: (out.mkdir(parents=True),
                                                               (_ for _ in ()).throw(OSError(28, "No space left"))))
    res = S.take_snapshot(home, tmp_path / "snaps", "t")
    assert res["ok"] is False and "No space left" in res["problems"][0]
    assert not any((tmp_path / "snaps").iterdir())


def test_the_snapshot_runs_before_the_rem_gate():
    """r2 M4: wired after the REM freshness gate, a quiet day (145 of 185
    cycles skip REM) meant no backup at all."""
    import ast
    import inspect
    from ghost_agent.core import dream as D
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(D)))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "dream")
    care = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "_daily_store_care")
    skip = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Constant)
               and isinstance(n.value, str) and n.value.startswith("Skipping REM"))
    assert care < skip


def test_every_forget_copy_expires_without_a_next_forget(tmp_path):
    """r2 M3: one rule — 30 days — enforced by the dream, for the profile
    archive, the reset dumps and the graph archive."""
    import time
    from types import SimpleNamespace
    from ghost_agent.core.dream import Dreamer
    old, new = time.time() - 40 * 86400, time.time()
    (tmp_path / "profile_removed.jsonl").write_text(
        json.dumps({"removed_at": old, "value": "x"}) + "\n" + "not json\n" + json.dumps({"removed_at": new}) + "\n")
    dump = tmp_path / "vector_reset_1.jsonl"
    dump.write_text("{}\n")
    os.utime(dump, (old, old))
    (tmp_path / "graph_pruned_archive.jsonl").write_text(
        json.dumps({"archived_at": old}) + "\n" + json.dumps({"archived_at": new}) + "\n")
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(memory_dir=tmp_path, episodic_memory=None)
    out = d._purge_forget_archives()
    assert out["profile"] == 1 and out["vector_reset"] == 1 and out["graph"] == 1
    kept = (tmp_path / "profile_removed.jsonl").read_text().splitlines()
    assert len(kept) == 2 and "not json" in kept                       # an unreadable line no longer stops the purge
    assert not dump.exists()


def test_the_snapshot_keeps_the_live_archives_and_the_sandbox(tmp_path):
    from ghost_agent.memory.snapshot import _is_repair_copy, take_snapshot
    assert not _is_repair_copy("profile_removed.jsonl")
    assert _is_repair_copy("activity.removed-4mi.jsonl") and _is_repair_copy("x.json.consolidated_0117_removed")
    home = tmp_path / "home"
    (home / "system").mkdir(parents=True)
    (home / "system" / "user_profile.json").write_text("{}")
    (home / "sandbox" / "proj").mkdir(parents=True)
    (home / "sandbox" / "proj" / "main.py").write_text("print(1)")
    res = take_snapshot(home, tmp_path / "snaps", "t")
    assert res["ok"] and (Path(res["path"]) / "sandbox" / "proj" / "main.py").exists()


def test_the_boot_takes_the_writer_lock_before_any_store_opens():
    """r2: the lock was taken inside VectorMemory, after the profile,
    playbook and journal constructors had already written."""
    import ast
    import inspect
    from ghost_agent import main as Mn
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(Mn)))
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    lock = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "acquire_writer_lock"]
    assert lock
    exits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "exit"
             and n.args and getattr(n.args[0], "value", None) == 75]
    assert exits
    # §4MR: the ORDER the name promises — the lock precedes every store
    # constructor in main() (the pin only checked that both existed)
    import re as _re
    ctors = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and _re.search(r"(?:Memory|Store|Journal|Playbook|Ledger)$",
                            getattr(n.func, "id", "") or getattr(n.func, "attr", ""))]
    assert ctors and min(l.lineno for l in lock) < min(c.lineno for c in ctors), \
        [(getattr(c.func, "id", "") or c.func.attr, c.lineno) for c in sorted(ctors, key=lambda c: c.lineno)[:3]]



def test_a_mocked_context_never_snapshots_into_the_working_directory():
    """A dream test's MagicMock `memory_dir` stringified to "MagicMock/…"
    and the daily store care created that folder in the repo."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.core.dream import Dreamer
    d = Dreamer.__new__(Dreamer)
    d.context = SimpleNamespace(memory_dir=MagicMock(), episodic_memory=None)
    assert asyncio.run(d._maybe_snapshot()) == "" and d._purge_forget_archives() == {}
    assert not Path("MagicMock").exists()
