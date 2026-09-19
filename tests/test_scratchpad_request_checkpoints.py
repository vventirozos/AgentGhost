"""§4IC — a turn checkpoint belongs to one request.

THE LIVE FAILURE (2026-09-17). At turns 15 and 30 the loop writes
`_checkpoint_t15` / `_checkpoint_t30` (the last five tool/assistant
snippets) into the scratchpad — with no namespace, SQLite-persisted, and
injected into EVERY request's DYNAMIC SYSTEM STATE. Request 21b295ef's
30-turn eckit checkpoints were still in the prompt when the same ask was
re-sent an hour later: the model opened with "let me pick up where I left
off … the checkpoint mentions eckit.grid" (3cb143fc, 3a2afac2) and ignored
the working script sitting in /workspace.

Now: the write is scoped `req:<id>`, the request's `finally` clears that
scope, and every load purges any `_checkpoint_t*` key regardless of scope
(no request survives a boot).

World where each pin fails: the write loses its namespace, the finally
stops clearing, or a legacy key survives a load.
"""
import ast
import inspect
import json
import sqlite3
import time

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import request_checkpoint_namespace
from ghost_agent.memory.scratchpad import TURN_CHECKPOINT_RE, Scratchpad


def _seed_db(path, rows):
    sp = Scratchpad(persist_path=path)      # creates the schema
    with sqlite3.connect(path) as conn:
        for key, value, ns in rows:
            conn.execute(
                "INSERT OR REPLACE INTO scratchpad (key, value, accessed_at, namespace) "
                "VALUES (?, ?, ?, ?)", (key, json.dumps(value), time.time(), ns))
        conn.commit()
    return sp


def test_legacy_checkpoints_are_purged_at_load(tmp_path):
    db = tmp_path / "s.db"
    _seed_db(db, [("_checkpoint_t15", "[Turn 15 checkpoint] eckit.grid …", None),
                  ("_checkpoint_t30", "[Turn 30 checkpoint] reduced_gg …", "req:dead"),
                  ("Self-Play Report", "kept", None),
                  ("notes", {"a": 1}, "projX")])
    sp = Scratchpad(persist_path=db)
    assert "_checkpoint_t15" not in sp.keys() and "_checkpoint_t30" not in sp.keys()
    assert "Self-Play Report" in sp.keys() and "notes" in sp.keys()
    with sqlite3.connect(db) as conn:
        left = {k for (k,) in conn.execute("SELECT key FROM scratchpad")}
    assert left == {"Self-Play Report", "notes"}          # gone from disk too


@pytest.mark.parametrize("key,stale", [
    ("_checkpoint_t15", True), ("_checkpoint_t30", True), ("_checkpoint_t7", True),
    ("_checkpoint_t15x", False), ("checkpoint_t15", False), ("_checkpoint", False),
    ("Self-Play Report", False),
])
def test_only_turn_checkpoint_keys_match(key, stale):
    assert bool(TURN_CHECKPOINT_RE.match(key)) is stale


def test_request_scope_is_visible_during_and_gone_after(tmp_path):
    sp = Scratchpad(persist_path=tmp_path / "s.db")
    ns = request_checkpoint_namespace("3a2afac2")
    sp.set("_checkpoint_t15", "[Turn 15 checkpoint] …", namespace=ns)
    sp.set("Self-Play Report", "kept")
    assert "_checkpoint_t15" in sp.list_all()              # the request itself sees it
    victims = sp.clear_namespace(ns)
    assert victims == ["_checkpoint_t15"]
    assert "_checkpoint_t15" not in sp.list_all() and "Self-Play Report" in sp.list_all()


def test_namespace_is_per_request():
    assert request_checkpoint_namespace("abc") == "req:abc"
    assert request_checkpoint_namespace("abc") != request_checkpoint_namespace("abd")
    assert request_checkpoint_namespace(None) == "req:anon"


# --- the sites ---------------------------------------------------------------

def _handle_chat():
    for n in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat":
            return n
    raise AssertionError("handle_chat not found")


def test_checkpoint_write_is_request_scoped():
    fn = _handle_chat()
    sets = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "set"
            and c.args and isinstance(c.args[0], ast.JoinedStr)
            and "_checkpoint_t" in ast.unparse(c.args[0])]
    assert len(sets) == 1
    kw = {k.arg: k.value for k in sets[0].keywords}
    assert "namespace" in kw
    assert getattr(kw["namespace"].func, "id", "") == "request_checkpoint_namespace"


def test_clear_request_checkpoints_drives_the_scratchpad(tmp_path):
    """Behavioural: the finally's helper removes exactly this request's
    scope and nothing else, and reports what it removed."""
    import types
    from ghost_agent.core.agent import GhostAgent
    sp = Scratchpad(persist_path=tmp_path / "s.db")
    sp.set("_checkpoint_t15", "mine", namespace=request_checkpoint_namespace("r1"))
    sp.set("_checkpoint_t30", "theirs", namespace=request_checkpoint_namespace("r2"))
    sp.set("Self-Play Report", "kept")
    a = GhostAgent.__new__(GhostAgent)
    a.context = types.SimpleNamespace(scratchpad=sp)
    assert a._clear_request_checkpoints("r1") == ["_checkpoint_t15"]
    assert set(sp.keys()) == {"_checkpoint_t30", "Self-Play Report"}
    # no scratchpad / a broken one → no exception, nothing removed
    a.context = types.SimpleNamespace(scratchpad=None)
    assert a._clear_request_checkpoints("r1") == []
    a.context = types.SimpleNamespace(scratchpad=types.SimpleNamespace(
        clear_namespace=lambda ns: (_ for _ in ()).throw(RuntimeError("db"))))
    assert a._clear_request_checkpoints("r1") == []


def test_request_finally_calls_the_helper_unconditionally():
    fn = _handle_chat()
    finals = [t for t in ast.walk(fn) if isinstance(t, ast.Try) and t.finalbody]
    hits = []
    for t in finals:
        for stmt in t.finalbody:                       # top level of the finally, not nested
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "attr", "") == "_clear_request_checkpoints"):
                hits.append(stmt.value)
    assert len(hits) == 1
    assert getattr(hits[0].args[0], "id", "") == "req_id"
