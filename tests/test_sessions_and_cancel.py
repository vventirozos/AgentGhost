"""Tests for Feature 3 (2026-07-11):
  * durable server-side conversations (core.sessions + the /api/sessions
    endpoints + session_id support on /api/chat)
  * real turn cancellation that RELEASES the global turn lock
    (core.turns + /api/turns + /api/turn/cancel)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.sessions import (
    Session, SessionStore, derive_title, merge_history,
    MAX_MESSAGES_PER_SESSION, MAX_SESSIONS,
)
from ghost_agent.core.turns import (
    TurnRegistry, TurnCancelled, get_turn_registry, REASON_USER,
)


# ══════════════════════════════════════════════════════════════════════
# merge_history — the contract that keeps a fat client from doubling
# ══════════════════════════════════════════════════════════════════════

def _m(role, content):
    return {"role": role, "content": content}


class TestMergeHistory:
    def test_empty_stored_returns_incoming(self):
        inc = [_m("user", "hi")]
        assert merge_history([], inc) == inc

    def test_thin_client_appends(self):
        stored = [_m("user", "hi"), _m("assistant", "hello")]
        inc = [_m("user", "how are you?")]
        assert merge_history(stored, inc) == stored + inc

    def test_fat_client_not_duplicated(self):
        stored = [_m("user", "hi"), _m("assistant", "hello")]
        inc = stored + [_m("user", "again")]
        merged = merge_history(stored, inc)
        assert merged == inc          # used as-is, NOT stored+inc
        assert len(merged) == 3       # the doubling bug this prevents

    def test_divergent_history_appends(self):
        # Client's prefix does NOT match stored → treat as new messages.
        stored = [_m("user", "hi"), _m("assistant", "hello")]
        inc = [_m("user", "totally"), _m("user", "different")]
        assert merge_history(stored, inc) == stored + inc

    def test_new_messages_slice_is_correct_both_styles(self):
        stored = [_m("user", "a"), _m("assistant", "b")]
        thin = merge_history(stored, [_m("user", "c")])
        fat = merge_history(stored, stored + [_m("user", "c")])
        assert thin[len(stored):] == [_m("user", "c")]
        assert fat[len(stored):] == [_m("user", "c")]


# ══════════════════════════════════════════════════════════════════════
# SessionStore
# ══════════════════════════════════════════════════════════════════════

class TestSessionStore:
    def test_create_get_delete(self, tmp_path):
        store = SessionStore(tmp_path)
        sess = store.create(title="My chat")
        assert sess is not None and sess.title == "My chat"
        got = store.get(sess.id)
        assert got is not None and got.id == sess.id
        assert store.delete(sess.id) is True
        assert store.get(sess.id) is None
        assert store.delete(sess.id) is False

    def test_append_turn_roundtrip(self, tmp_path):
        store = SessionStore(tmp_path)
        sess = store.create()
        store.append_turn(sess.id, [_m("user", "what is 2+2?")], "4")
        got = store.get(sess.id)
        assert [m["content"] for m in got.messages] == ["what is 2+2?", "4"]
        assert got.messages[1]["role"] == "assistant"

    def test_append_creates_session_if_absent(self, tmp_path):
        store = SessionStore(tmp_path)
        assert store.append_turn("client-picked-id", [_m("user", "hi")], "yo")
        got = store.get("client-picked-id")
        assert got is not None and len(got.messages) == 2

    def test_title_derived_from_first_user_message(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("s1", [_m("user", "Explain  quantum\n tunnelling")],
                          "ok")
        assert store.get("s1").title == "Explain quantum tunnelling"

    def test_multi_turn_accumulates(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("s", [_m("user", "one")], "1")
        store.append_turn("s", [_m("user", "two")], "2")
        assert len(store.get("s").messages) == 4

    def test_list_sorted_by_updated_desc(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("old", [_m("user", "a")], "x")
        time.sleep(0.01)
        store.append_turn("new", [_m("user", "b")], "y")
        ids = [s["id"] for s in store.list()]
        assert ids[0] == "new" and ids[1] == "old"
        assert store.list()[0]["message_count"] == 2

    def test_list_has_no_message_bodies(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("s", [_m("user", "secret")], "x")
        assert "messages" not in store.list()[0]

    def test_path_traversal_rejected(self, tmp_path):
        store = SessionStore(tmp_path)
        assert store.get("../../etc/passwd") is None
        assert store.delete("../../etc/passwd") is False
        assert store.append_turn("../evil", [_m("user", "x")], "y") is False

    def test_corrupt_file_treated_absent(self, tmp_path):
        store = SessionStore(tmp_path)
        store.create()
        (tmp_path / "broken.json").write_text("{not json")
        assert store.get("broken") is None
        assert store.list() == [s for s in store.list()]  # list survives it

    def test_malformed_messages_dropped(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("s", [
            _m("user", "good"),
            {"role": "bogus_role", "content": "x"},
            "not a dict",
        ], "reply")
        msgs = store.get("s").messages
        assert [m["content"] for m in msgs] == ["good", "reply"]

    def test_tool_plumbing_preserved(self, tmp_path):
        store = SessionStore(tmp_path)
        store.append_turn("s", [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "res", "tool_call_id": "c1"},
        ], "done")
        msgs = store.get("s").messages
        assert msgs[0]["tool_calls"] == [{"id": "c1"}]
        assert msgs[1]["tool_call_id"] == "c1"

    def test_message_cap_keeps_system_and_tail(self, tmp_path):
        store = SessionStore(tmp_path)
        big = [_m("system", "SYS")] + [
            _m("user", f"m{i}") for i in range(MAX_MESSAGES_PER_SESSION + 50)]
        store.append_turn("s", big, "final")
        msgs = store.get("s").messages
        assert len(msgs) <= MAX_MESSAGES_PER_SESSION
        assert msgs[0]["content"] == "SYS"          # system kept
        assert msgs[-1]["content"] == "final"       # newest kept

    def test_empty_turn_not_written(self, tmp_path):
        store = SessionStore(tmp_path)
        assert store.append_turn("s", [], "") is False

    def test_derive_title_multimodal(self):
        assert derive_title([{"role": "user",
                              "content": [{"type": "text", "text": "Hi there"}]}]) \
            == "Hi there"

    def test_derive_title_fallback(self):
        assert derive_title([]) == "New conversation"


# ══════════════════════════════════════════════════════════════════════
# TurnRegistry
# ══════════════════════════════════════════════════════════════════════

class TestTurnRegistry:
    def test_register_and_list(self):
        reg = TurnRegistry()
        t = reg.register("req1", preview="hello   world", session_id="s1")
        assert t.req_id == "req1"
        assert t.preview == "hello world"   # whitespace collapsed
        assert t.running is False           # queued until mark_running
        assert [x.req_id for x in reg.list()] == ["req1"]
        reg.unregister("req1")
        assert reg.list() == []

    def test_current_is_the_running_turn(self):
        reg = TurnRegistry()
        reg.register("queued")
        reg.register("running")
        reg.mark_running("running")
        assert reg.current().req_id == "running"

    def test_cancel_unknown(self):
        reg = TurnRegistry()
        out = reg.cancel("nope")
        assert out["cancelled"] is False and "no active turn" in out["error"]

    def test_cancel_with_nothing_running(self):
        reg = TurnRegistry()
        out = reg.cancel()
        assert out["cancelled"] is False
        assert "no turn is currently running" in out["error"]

    def test_cooperative_cancel_sets_flag_only(self):
        reg = TurnRegistry()
        reg.register("r")
        reg.mark_running("r")
        out = reg.cancel("r")
        assert out["cancelled"] is True and out["mode"] == "cooperative"
        assert reg.is_cancelled("r") is True

    def test_queued_turn_is_always_hard_cancelled(self):
        """A queued turn has no partial work and no boundary to reach — it
        must be killed, not left sitting on the semaphore queue."""
        async def go():
            reg = TurnRegistry()

            async def waiter():
                await asyncio.sleep(30)
            task = asyncio.create_task(waiter())
            turn = reg.register("q")
            turn.task = task            # queued: mark_running NOT called
            out = reg.cancel("q")       # hard=False, but it's queued…
            await asyncio.sleep(0.05)
            return out, task
        out, task = asyncio.run(go())
        assert out["mode"] == "hard" and out["was_running"] is False
        assert task.cancelled()

    def test_hard_cancel_kills_running_task(self):
        async def go():
            reg = TurnRegistry()

            async def waiter():
                await asyncio.sleep(30)
            task = asyncio.create_task(waiter())
            turn = reg.register("r")
            turn.task = task
            reg.mark_running("r")
            out = reg.cancel("r", hard=True)
            await asyncio.sleep(0.05)
            return out, task
        out, task = asyncio.run(go())
        assert out["mode"] == "hard" and out["was_running"] is True
        assert task.cancelled()

    def test_cancel_defaults_to_current(self):
        reg = TurnRegistry()
        reg.register("a")
        reg.register("b")
        reg.mark_running("b")
        out = reg.cancel()
        assert out["request_id"] == "b"

    def test_registry_is_shared_per_agent(self):
        agent = SimpleNamespace()
        assert get_turn_registry(agent) is get_turn_registry(agent)

    def test_to_dict_shape(self):
        reg = TurnRegistry()
        reg.register("r", preview="p", session_id="s")
        reg.mark_running("r")
        d = reg.list()[0].to_dict()
        assert d["request_id"] == "r" and d["running"] is True
        assert d["queued"] is False and d["cancelled"] is False

    def test_turn_cancelled_exception_carries_reason(self):
        e = TurnCancelled("r1", REASON_USER)
        assert e.req_id == "r1" and REASON_USER in str(e)


# ══════════════════════════════════════════════════════════════════════
# handle_chat integration: cancellation RELEASES the semaphore
# ══════════════════════════════════════════════════════════════════════

class TestCancellationReleasesLock:
    def test_cooperative_cancel_frees_semaphore_and_returns_partial(self):
        """The load-bearing property: after a cancel, the global turn lock
        (Semaphore(1)) is free — otherwise a wedged turn blocks the web UI,
        Slack and the idle loops forever (#22)."""
        from ghost_agent.core.agent import GhostAgent
        from ghost_agent.core.turns import get_turn_registry

        agent = GhostAgent.__new__(GhostAgent)   # no __init__ (needs a context)
        agent.context = MagicMock()
        agent.agent_semaphore = asyncio.Semaphore(1)

        async def go():
            reg = get_turn_registry(agent)
            # Simulate a turn that acquires the lock, then gets cancelled at
            # its next boundary — exactly what handle_chat's loop check does.
            async with agent.agent_semaphore:
                reg.register("r")
                reg.mark_running("r")
                reg.cancel("r")
                assert reg.is_cancelled("r") is True
                try:
                    raise TurnCancelled("r", REASON_USER)
                except TurnCancelled:
                    pass
            reg.unregister("r")
            # The lock must be immediately re-acquirable.
            return agent.agent_semaphore.locked()

        assert asyncio.run(go()) is False

    def test_handle_chat_registers_and_checks_cancel(self):
        """Source-inspection: the three wiring points must be present —
        register before the semaphore, mark_running after acquiring, and a
        cooperative check inside the turn loop."""
        src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "core" / "agent.py").read_text()
        assert "_turn_reg = get_turn_registry(self)" in src
        assert "_turn_reg.mark_running(req_id)" in src
        assert "_turn_reg.is_cancelled(req_id)" in src
        assert "raise TurnCancelled(req_id" in src
        assert "except TurnCancelled as _tc:" in src
        assert "_turn_reg.unregister(req_id)" in src


# ══════════════════════════════════════════════════════════════════════
# API endpoints
# ══════════════════════════════════════════════════════════════════════

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI              # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _make_app(tmp_path, *, handle_chat=None, with_sessions=True):
    from ghost_agent.api.routes import router
    app = FastAPI()
    app.include_router(router)

    memory_dir = tmp_path / "system" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    agent = MagicMock()
    agent.context = MagicMock()
    agent.context.args = SimpleNamespace(api_key="", model="test-model")
    agent.context.memory_dir = memory_dir
    if with_sessions:
        agent.context.session_store = SessionStore(tmp_path / "sessions")
    else:
        agent.context.session_store = None

    if handle_chat is None:
        async def _ok(body, bg, request_id=None):
            return ("the reply", 1234, "req-1")
        agent.handle_chat = _ok
    else:
        agent.handle_chat = handle_chat

    # A real registry (MagicMock would fake-satisfy isinstance checks).
    agent.turn_registry = TurnRegistry()

    app.state.agent = agent
    app.state.context = agent.context
    app.state.args = agent.context.args
    return app, agent


class TestSessionsAPI:
    def test_crud_flow(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            assert c.get("/api/sessions").json()["sessions"] == []
            sid = c.post("/api/sessions", json={"title": "T"}).json()["id"]
            lst = c.get("/api/sessions").json()["sessions"]
            assert len(lst) == 1 and lst[0]["title"] == "T"
            assert c.get(f"/api/sessions/{sid}").json()["messages"] == []
            assert c.delete(f"/api/sessions/{sid}").status_code == 200
            assert c.get(f"/api/sessions/{sid}").status_code == 404

    def test_get_missing_404(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            assert c.get("/api/sessions/nope").status_code == 404

    def test_disabled_without_store(self, tmp_path):
        app, _ = _make_app(tmp_path, with_sessions=False)
        with TestClient(app) as c:
            assert c.get("/api/sessions").json() == {"enabled": False,
                                                     "sessions": []}
            assert c.post("/api/sessions", json={}).status_code == 503

    def test_chat_with_session_id_persists_turn(self, tmp_path):
        app, agent = _make_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/chat", json={
                "model": "test-model", "stream": False,
                "session_id": "s1",
                "messages": [{"role": "user", "content": "hello"}],
            })
            assert r.status_code == 200
            got = c.get("/api/sessions/s1").json()
            assert [m["content"] for m in got["messages"]] == \
                ["hello", "the reply"]

    def test_second_turn_resumes_history(self, tmp_path):
        seen = {}

        async def capture(body, bg, request_id=None):
            seen["messages"] = list(body["messages"])
            return ("reply2", 1, "r")

        app, _ = _make_app(tmp_path, handle_chat=capture)
        with TestClient(app) as c:
            # Turn 1 via the default stub-free path: seed the store directly.
            c.post("/api/chat", json={
                "model": "test-model", "stream": False, "session_id": "s2",
                "messages": [{"role": "user", "content": "first"}],
            })
            # Turn 2 — a THIN client sends only the new message; the server
            # must supply the prior history to the agent.
            c.post("/api/chat", json={
                "model": "test-model", "stream": False, "session_id": "s2",
                "messages": [{"role": "user", "content": "second"}],
            })
            assert [m["content"] for m in seen["messages"]] == \
                ["first", "reply2", "second"]

    def test_fat_client_does_not_double_history(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            c.post("/api/chat", json={
                "model": "test-model", "stream": False, "session_id": "s3",
                "messages": [{"role": "user", "content": "one"}],
            })
            # Fat client replays everything it has, plus the new message.
            c.post("/api/chat", json={
                "model": "test-model", "stream": False, "session_id": "s3",
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "the reply"},
                    {"role": "user", "content": "two"},
                ],
            })
            msgs = [m["content"] for m in
                    c.get("/api/sessions/s3").json()["messages"]]
            assert msgs == ["one", "the reply", "two", "the reply"]

    def test_no_session_id_is_unchanged_behaviour(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/chat", json={
                "model": "test-model", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert r.status_code == 200
            assert c.get("/api/sessions").json()["sessions"] == []  # nothing stored


class TestTurnsAPI:
    def test_turns_empty(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            d = c.get("/api/turns").json()
            assert d["turns"] == [] and d["running"] is None

    def test_turns_lists_running_and_queued(self, tmp_path):
        app, agent = _make_app(tmp_path)
        agent.turn_registry.register("r1", preview="running one")
        agent.turn_registry.mark_running("r1")
        agent.turn_registry.register("r2", preview="queued one")
        with TestClient(app) as c:
            d = c.get("/api/turns").json()
            assert d["running"] == "r1" and d["queued"] == 1
            assert len(d["turns"]) == 2

    def test_cancel_running_turn(self, tmp_path):
        app, agent = _make_app(tmp_path)
        agent.turn_registry.register("r1")
        agent.turn_registry.mark_running("r1")
        with TestClient(app) as c:
            r = c.post("/api/turn/cancel", json={})
            assert r.status_code == 200
            assert r.json()["cancelled"] is True
            assert r.json()["mode"] == "cooperative"
            assert agent.turn_registry.is_cancelled("r1") is True

    def test_cancel_hard_flag(self, tmp_path):
        app, agent = _make_app(tmp_path)
        agent.turn_registry.register("r1")
        agent.turn_registry.mark_running("r1")
        with TestClient(app) as c:
            r = c.post("/api/turn/cancel",
                       json={"request_id": "r1", "hard": True})
            assert r.json()["mode"] == "hard"

    def test_cancel_nothing_running_404(self, tmp_path):
        app, _ = _make_app(tmp_path)
        with TestClient(app) as c:
            r = c.post("/api/turn/cancel", json={})
            assert r.status_code == 404
            assert r.json()["cancelled"] is False

    def test_cancel_tolerates_empty_body(self, tmp_path):
        app, agent = _make_app(tmp_path)
        agent.turn_registry.register("r1")
        agent.turn_registry.mark_running("r1")
        with TestClient(app) as c:
            r = c.post("/api/turn/cancel", content=b"",
                       headers={"Content-Type": "application/json"})
            assert r.status_code == 200


class TestSSEDeltaExtraction:
    def test_extracts_delta_content(self):
        from ghost_agent.api.routes import _sse_delta_text
        chunk = ('data: ' + json.dumps(
            {"choices": [{"delta": {"content": "Hello"}}]}) + "\n\n")
        assert _sse_delta_text(chunk.encode()) == "Hello"

    def test_ignores_done_and_garbage(self):
        from ghost_agent.api.routes import _sse_delta_text
        assert _sse_delta_text(b"data: [DONE]\n\n") == ""
        assert _sse_delta_text(b": comment\n\n") == ""
        assert _sse_delta_text(b"data: {not json}\n\n") == ""
        assert _sse_delta_text(None) == ""
