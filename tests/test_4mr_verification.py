"""§4MR (2026-10-09): fresh-eye verification of §4MK–§4MQ. Each test names
the defect a reviewer showed on the real path."""
from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_the_daily_snapshot_is_consistent_while_the_agent_process_writes(tmp_path):
    """Lens 5 MAJOR: taken inside the agent process, the chroma copy tore
    while the agent wrote (POSIX locks do not exclude within one process).
    The isolated path copies from a child process — measured with the writer
    NOT holding any lock, so the child's file locking alone must suffice."""
    import chromadb
    from chromadb.config import Settings
    from ghost_agent.memory.snapshot import take_snapshot_isolated
    home = tmp_path / "home"
    mem = home / "system" / "memory"
    mem.mkdir(parents=True)
    (home / "sandbox").mkdir()
    col = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False)) \
        .get_or_create_collection("agent_memory")
    rnd = lambda: [random.random() for _ in range(32)]  # noqa: E731
    col.add(ids=[f"s{i}" for i in range(300)], embeddings=[rnd() for _ in range(300)], documents=["x"] * 300)
    stop = []

    def writer():
        i = 0
        while not stop:
            col.add(ids=[f"w{i}-{j}" for j in range(5)], embeddings=[rnd() for _ in range(5)], documents=["y"] * 5)
            i += 1
    t = threading.Thread(target=writer)
    t.start()
    try:
        results = [take_snapshot_isolated(home, home / "bk", tag=f"h{k}") for k in range(3)]
    finally:
        stop.append(1)
        t.join()
    assert all(r["ok"] for r in results), [r.get("problems") for r in results]


def test_the_dream_takes_the_isolated_snapshot_holding_the_vector_lock():
    import inspect
    from ghost_agent.core import dream as D
    tree = ast.parse(inspect.getsource(D))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and any(isinstance(a, ast.Name) and a.id in ("take_snapshot", "take_snapshot_isolated")
                     for a in n.args)]
    assert calls and all(any(isinstance(a, ast.Name) and a.id == "take_snapshot_isolated" for a in c.args)
                         for c in calls)


def _store_writing_apply_scripts():
    """Scripts with an --apply mode that open a live store for writing —
    found from the AST (a string constant '--apply' and a call to a store
    opener)."""
    openers = {"PersistentClient", "VectorMemory", "connect", "ProjectStore", "SkillMemory",
               "EpisodicMemory", "GraphMemory", "write_text"}
    out = []
    for f in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        called = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if "--apply" in consts and called & openers:
            names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            out.append((f.name, "assert_no_other_writer" in names))
    return out


def test_every_store_writing_repair_script_refuses_beside_a_writer():
    """Lens 5 MAJOR: the §4MN claim held only for §4MJ+ scripts; three older
    ones opened chroma on the live store directly — the two-writer segfault."""
    found = _store_writing_apply_scripts()
    assert len(found) >= 26
    assert [n for n, ok in found if not ok] == []


def test_an_old_repair_script_refuses_to_run_beside_the_agent(tmp_path):
    from ghost_agent.memory.store_lock import acquire_writer_lock, release_writer_lock
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    acquire_writer_lock(mem, "the agent")
    try:
        env = {**os.environ, "GHOST_HOME": str(tmp_path) + "/", "PYTHONPATH": str(ROOT / "src")}
        p = subprocess.run([sys.executable, str(SCRIPTS / "memory_repair_4kw.py"), "--apply"],
                           env=env, capture_output=True, text=True, timeout=120)
    finally:
        release_writer_lock(mem)
    assert p.returncode != 0 and "refusing to run" in (p.stderr + p.stdout)


@pytest.mark.parametrize("pred,obj,moving", [
    ("HAS_RELEASE", "18.4", True), ("HAS_STABLE_RELEASE", "18.4", True),
    ("TRADES_AT", "$61,000", True), ("PRICED_AT", "1199 eur", True), ("IS_WORTH", "61000 USD", True),
    ("RELEASED_IN", "2008", False), ("HAS_FEATURE", "logical replication", False),
    ("HAS_VERSION", "anything", True),
])
def test_a_moving_world_fact_is_known_by_its_object_too(pred, obj, moving):
    """Lens 5 MINOR: only the predicate's WORDS counted."""
    from ghost_agent.memory.graph import is_moving_target_world_fact
    assert is_moving_target_world_fact(pred, obj) is moving


def test_the_graph_never_stores_a_moving_world_fact(tmp_path):
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "postgresql", "predicate": "HAS_RELEASE", "object": "18.4"},
                    {"subject": "bitcoin", "predicate": "TRADES_AT", "object": "$61,000"},
                    {"subject": "postgresql", "predicate": "RELEASED_IN", "object": "1996"}])
    import sqlite3
    db = next(Path(tmp_path).rglob("*.db"))
    with sqlite3.connect(db) as c:
        objs = {r[0] for r in c.execute("SELECT object FROM triplets")}
    assert "18.4" not in objs and "$61,000" not in objs and "1996" in objs


@pytest.mark.parametrize("q", ["What version of PostgreSQL is current?",
                               "which postgres release is newest?",
                               "what is the price of gold today?"])
def test_a_reversed_moving_target_question_is_recognised(q):
    from ghost_agent.memory.episodes import is_moving_target_question
    assert is_moving_target_question(q)


# ── the web client (executed under node) ─────────────────────────────
STATIC = ROOT / "interface" / "static"
APP = (STATIC / "app.js").read_text(encoding="utf-8")


def test_an_svg_image_blob_never_keeps_a_scriptable_type():
    """Lens 5: a blob kept the server's type; "open image in new tab" on an
    agent-linked SVG ran script in the page's origin, next to the key."""
    from tests.helpers import eval_js, extract_js_function
    src = ("const _RASTER_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp',"
           " 'image/avif', 'image/bmp']);\n" + extract_js_function(APP, "_safeImageBlobType"))
    got = eval_js(src, "['image/png', 'IMAGE/JPEG; charset=x', 'image/svg+xml', 'text/html', ''].map(_safeImageBlobType)")
    assert got == ["image/png", "image/jpeg", "application/octet-stream", "application/octet-stream",
                   "application/octet-stream"]
