"""§4SX B0: project focus per conversation (UNIFIED_DESIGN §10, r2 #7/#17/#19).

The old design kept ONE global binding (four scratchpad sentinels) and grew
guests / held / park / a word-match escape hatch around it. Now:

  - a conversation's identity is its session id (``session:<id>``,
    ``slack:<…>``);
  - its focus is a durable row in the project store (``conversation_focus``);
    many conversations may focus one project, a switch changes only the
    switching conversation's row;
  - the request's project is a value in a ContextVar scope;
    ``GhostContext.current_project_id`` is a property over it;
  - members, replays, probes, unattended requests, leaves and clients with
    no session id never read or write a row.
"""

import asyncio
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import request_scope as rs
from ghost_agent.core.request_scope import RequestScope, conversation_key_for_session
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    _LEGACY_BINDING_KEYS, begin_request_focus, conversation_fingerprint,
    drop_legacy_binding, tool_manage_projects,
)
from tests.focus_helpers import scope_reset, start_request  # noqa: F401


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(project_store=store,
                           scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, current_project_id=None)


A, B, C = "session:aaa", "session:bbb", "slack:C1:T1"


async def _create(context, title):
    res = await tool_manage_projects(context, action="create", title=title, kind="CODING", goal="x")
    return json.loads(res)["created"]


async def _switch(context, pid):
    return json.loads(await tool_manage_projects(context, action="switch", project_id=pid))


# ------------------------------------------------------------ fingerprint (kept)

def test_fingerprint_stable_as_conversation_grows():
    first = {"role": "user", "content": "hello world"}
    assert conversation_fingerprint([first]) == conversation_fingerprint(
        [first, {"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}])


def test_fingerprint_empty_inputs():
    assert conversation_fingerprint([]) == ""
    assert conversation_fingerprint(None) == ""
    assert conversation_fingerprint([{"role": "assistant", "content": "x"}]) == ""


def test_conversation_key_for_session():
    assert conversation_key_for_session("abc") == "session:abc"
    assert conversation_key_for_session(" abc ") == "session:abc"
    assert conversation_key_for_session("slack:C1:T9") == "slack:C1:T9"
    assert conversation_key_for_session("") == ""
    assert conversation_key_for_session(None) == ""


# ------------------------------------------------- acceptance: two sessions

async def test_two_sessions_on_one_project_switch_independently(context, store):
    start_request(context, A)
    p = await _create(context, "Memory Garden")
    start_request(context, B)
    assert context.current_project_id is None          # B never inherits A's
    await _switch(context, p)                          # B joins P: no owner, no guest
    assert store.get_focus(A) == p and store.get_focus(B) == p
    q = await _create(context, "Other")                # B moves on
    assert store.get_focus(B) == q
    assert store.get_focus(A) == p                     # A's row untouched
    start_request(context, A)
    assert context.current_project_id == p
    await tool_manage_projects(context, action="exit")
    assert store.get_focus(A) is None
    assert store.get_focus(B) == q                     # exit is A's only
    start_request(context, B)
    assert context.current_project_id == q


async def test_exit_with_nothing_focused_touches_no_other_row(context, store):
    start_request(context, A)
    p = await _create(context, "P")
    start_request(context, B)
    res = json.loads(await tool_manage_projects(context, action="exit"))
    assert res["exited"] is None
    assert store.get_focus(A) == p


async def test_focus_survives_a_restart(tmp_path, context, store):
    start_request(context, A)
    p = await _create(context, "Durable")
    again = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    assert again.get_focus(A) == p


async def test_archive_and_delete_clear_every_row_on_the_project(context, store):
    start_request(context, A)
    p = await _create(context, "Shared")
    start_request(context, B)
    await _switch(context, p)
    start_request(context, C)
    await _switch(context, p)
    assert len(store.focus_rows(p)) == 3
    await tool_manage_projects(context, action="archive", project_id=p)
    assert store.focus_rows(p) == []
    for k in (A, B, C):
        assert store.get_focus(k) is None
    start_request(context, A)
    assert context.current_project_id is None


async def test_a_status_archive_by_any_route_clears_rows(context, store):
    start_request(context, A)
    p = await _create(context, "Patched")
    store.update_project(p, status="ARCHIVED")         # the PATCH route's call
    assert store.focus_rows(p) == []


async def test_a_hard_delete_clears_rows(context, store):
    start_request(context, A)
    p = await _create(context, "Gone")
    assert store.delete_project(p, hard=True)
    assert store.get_focus(A) is None
    assert store.focus_rows() == []


def test_a_row_naming_a_missing_project_is_dropped(store):
    p = store.create_project(title="X", kind="CODING", goal="g")
    assert store.set_focus(A, p)
    with store._connect() as conn:          # a row whose project vanished behind the store's back
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM projects WHERE id=?", (p,))
        conn.commit()
    assert store.get_focus(A) is None
    assert store.focus_rows() == []


async def test_an_explicit_switch_to_an_archived_project_holds(context, store):
    """Review r1 MAJOR: the console switch to an ARCHIVED project returned 200
    and the next turn's read dropped the row — the switch silently vanished.
    The archive clears rows when it HAPPENS; a later explicit choice stays."""
    p = store.create_project(title="Old", kind="CODING", goal="g")
    store.update_project(p, status="ARCHIVED")
    assert store.set_focus(A, p)
    assert store.get_focus(A) == p
    assert store.get_focus(A) == p           # and still on the turn after


def test_set_focus_refuses_unknown_projects_and_empty_keys(store):
    assert store.set_focus(A, "nope00000000") is False
    p = store.create_project(title="X", kind="CODING", goal="g")
    assert store.set_focus("", p) is False
    assert store.focus_rows() == []


def test_idle_rows_are_pruned_after_30_days(store):
    p = store.create_project(title="X", kind="CODING", goal="g")
    store.set_focus(A, p)
    with store._connect() as conn:
        conn.execute("UPDATE conversation_focus SET last_used_at = ?",
                     (time.time() - store.FOCUS_IDLE_PRUNE_S - 10,))
        conn.commit()
    store.set_focus(B, p)                    # any write prunes
    assert store.get_focus(A) is None and store.get_focus(B) == p


def test_reading_a_row_refreshes_it(store):
    p = store.create_project(title="X", kind="CODING", goal="g")
    store.set_focus(A, p)
    with store._connect() as conn:
        conn.execute("UPDATE conversation_focus SET last_used_at = 5")
        conn.commit()
    store.get_focus(A)
    assert store.focus_rows()[0]["last_used_at"] > 5


# -------------------------------------- request-only scopes never touch rows

@pytest.mark.parametrize("origin", ["probe", "member", "replay", "unattended", "leaf"])
async def test_request_only_scopes_never_read_or_write_rows(context, store, origin):
    start_request(context, A)
    p = await _create(context, "Owner's")
    q = store.create_project(title="Q", kind="CODING", goal="g")
    start_request(context, A, persist=False, origin=origin)
    assert context.current_project_id is None          # the row is not read
    await _switch(context, q)
    assert context.current_project_id == q             # the switch lasts this request
    await tool_manage_projects(context, action="exit")
    assert store.get_focus(A) == p                     # …and wrote nothing
    assert store.focus_rows(q) == []


async def test_a_leaf_scope_keeps_its_pin(context, store):
    p = store.create_project(title="Leaf", kind="CODING", goal="g")
    start_request(context, "", persist=False, origin="leaf", project_id=p)
    assert context.current_project_id == p
    assert store.focus_rows() == []


# --------------------------------------------------- scratchpad follows focus

async def test_scratchpad_follows_the_conversation(context):
    start_request(context, A)
    p = await _create(context, "P")
    context.scratchpad.set("p_key", "P state")
    start_request(context, B)
    q = await _create(context, "Q")
    assert context.scratchpad.get("p_key") is None
    context.scratchpad.set("q_key", "Q state")
    start_request(context, A)
    assert context.scratchpad.get("p_key") == "P state"
    assert context.scratchpad.get("q_key") is None
    start_request(context, B)
    assert context.scratchpad.get("q_key") == "Q state"


async def test_consecutive_turns_on_one_project_hydrate_nothing(context, monkeypatch):
    start_request(context, A)
    p = await _create(context, "P")
    start_request(context, B)
    await _switch(context, p)
    import ghost_agent.tools.projects as pm
    calls = []
    monkeypatch.setattr(pm, "_hydrate_scratchpad", lambda c, pid: calls.append(pid))
    start_request(context, A)
    start_request(context, B)
    assert calls == []


async def test_a_deleted_key_does_not_come_back_on_the_next_visit(context):
    """Review r1 MAJOR: an emptied project wrote no snapshot, so the next
    swap back hydrated the OLDER snapshot — the key the owner deleted came
    back. The emptiness is recorded."""
    start_request(context, A)
    p = await _create(context, "P")
    context.scratchpad.set("p_key", "v1")
    start_request(context, B)                       # snapshot {p_key}
    start_request(context, A)
    assert context.scratchpad.get("p_key") == "v1"
    assert context.scratchpad.delete("p_key")
    start_request(context, B)                       # snapshot must record {}
    start_request(context, A)
    assert context.scratchpad.get("p_key") is None


async def test_an_empty_project_alternating_writes_one_empty_snapshot(context, store):
    """The emptiness is written once (when the latest snapshot held keys) —
    not an event per idle alternation."""
    start_request(context, A)
    p = await _create(context, "P")
    for _ in range(3):
        start_request(context, B)
        start_request(context, A)
    assert store.list_events(p, event_type="scratchpad_snapshot") == []


async def test_a_member_turn_hides_the_owners_project_keys(context):
    """The keyspace is flat: moving only the write scope (the pre-B0 member
    path) left the owner's project keys readable. The swap snapshots them
    away for the member turn and the owner's next turn brings them back.
    (Review r1 proposed "members don't swap"; this pins why they must.)"""
    start_request(context, A)
    await _create(context, "P")
    context.scratchpad.set("p_key", "owner")
    start_request(context, "", persist=False, origin="member")
    assert context.scratchpad.get("p_key") is None
    start_request(context, A)
    assert context.scratchpad.get("p_key") == "owner"


# ----------------------------------------------------------- legacy (B4)

def test_the_old_global_binding_is_dropped(tmp_path):
    sp = Scratchpad(persist_path=tmp_path / "sp.db")
    for k in _LEGACY_BINDING_KEYS:
        sp.set(k, "41f745517fe4|x", namespace=None)
    sp.set("other", "kept")
    assert drop_legacy_binding(sp) == 4
    again = Scratchpad(persist_path=tmp_path / "sp.db")
    for k in _LEGACY_BINDING_KEYS:
        assert again.get(k) is None
    assert again.get("other") == "kept"
    assert drop_legacy_binding(again) == 0


async def test_after_the_drop_the_next_owner_message_starts_unfocused(context):
    for k in _LEGACY_BINDING_KEYS:
        context.scratchpad.set(k, "41f745517fe4", namespace=None)
    drop_legacy_binding(context.scratchpad)
    start_request(context, A)
    assert context.current_project_id is None


# ---------------------------------------------- the GhostContext property

def _ghost_context(tmp_path):
    from ghost_agent.core.agent import GhostContext
    return GhostContext(SimpleNamespace(), tmp_path, tmp_path, None)


def test_property_reads_the_scope_of_its_own_context(tmp_path):
    ctx = _ghost_context(tmp_path)
    assert ctx.current_project_id is None
    ctx.current_project_id = "outside"                 # no request: the object's own
    sc = RequestScope(conversation_key=A, project_id="p1", context=ctx)
    rs._scope_var.set(sc)
    assert ctx.current_project_id == "p1" and ctx.conversation_key == A
    ctx.current_project_id = "p2"
    assert sc.project_id == "p2"
    other = _ghost_context(tmp_path)                   # a scope never reaches another context
    assert other.current_project_id is None
    rs._scope_var.set(None)
    assert ctx.current_project_id == "outside"


def test_a_copy_keeps_the_value_and_leaves_the_scope(tmp_path):
    ctx = _ghost_context(tmp_path)
    sc = RequestScope(conversation_key=A, project_id="p1", context=ctx)
    rs._scope_var.set(sc)
    iso = copy.copy(ctx)
    assert iso.current_project_id == "p1" and iso.conversation_key == A
    iso.current_project_id = "leafpin"
    assert sc.project_id == "p1"                        # the parent's scope is untouched
    assert ctx.current_project_id == "p1"


async def test_a_switch_in_a_child_task_reaches_the_request(tmp_path):
    ctx = _ghost_context(tmp_path)
    rs._scope_var.set(RequestScope(conversation_key=A, project_id=None, context=ctx))

    async def tool_call():
        ctx.current_project_id = "switched"

    await asyncio.gather(asyncio.create_task(tool_call()))
    assert ctx.current_project_id == "switched"


async def test_background_work_keeps_its_requests_project(tmp_path):
    ctx = _ghost_context(tmp_path)
    seen = []
    gate = asyncio.Event()

    async def late():
        await gate.wait()
        seen.append(ctx.current_project_id)

    async def request(pid):
        tok = rs.install(RequestScope(conversation_key=A, project_id=pid, context=ctx))
        try:
            return asyncio.create_task(late()) if pid == "first" else None
        finally:
            rs.reset(tok)

    t = await asyncio.create_task(request("first"))
    await asyncio.create_task(request("second"))
    gate.set()
    await t
    assert seen == ["first"]


# ------------------------------------------------ handle_chat's scope builder

def _scope_for(ctx, body, rid="abcd1234", origin="user", role=""):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.utils.logging import requester_role_context
    tok = requester_role_context.set(role)
    try:
        return GhostAgent._build_request_scope(SimpleNamespace(context=ctx), body, rid, origin)
    finally:
        requester_role_context.reset(tok)


def test_scope_builder_owner_with_session_persists(tmp_path):
    ctx = _ghost_context(tmp_path)
    sc = _scope_for(ctx, {"session_id": "s1", "messages": []})
    assert (sc.conversation_key, sc.persist, sc.origin) == ("session:s1", True, "owner")
    assert sc.context is ctx


def test_scope_builder_no_session_is_request_only(tmp_path):
    ctx = _ghost_context(tmp_path)
    sc = _scope_for(ctx, {"messages": [{"role": "user", "content": "hi"}]})
    assert sc.persist is False
    assert sc.conversation_key == conversation_fingerprint([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("rid,origin,role,want", [
    ("abcd", "user", "member", "member"),
    ("probe-fr-1", "probe", "", "replay"),
    ("probe-x1", "probe", "", "probe"),
    ("sched-job1", "user", "", "unattended"),
    ("abcd", "sim", "", "unattended"),
])
def test_scope_builder_non_owner_requests_never_persist(tmp_path, rid, origin, role, want):
    ctx = _ghost_context(tmp_path)
    sc = _scope_for(ctx, {"session_id": "s1", "messages": []}, rid=rid, origin=origin, role=role)
    assert sc.origin == want and sc.persist is False


def test_scope_builder_uses_a_passed_scope(tmp_path):
    ctx = _ghost_context(tmp_path)
    pin = RequestScope(project_id="pinned", origin="leaf")
    from ghost_agent.core.agent import GhostAgent
    sc = GhostAgent._build_request_scope(SimpleNamespace(context=ctx), {"session_id": "s1"},
                                         "sub-leaf-1", "leaf", pin)
    assert sc is pin and sc.context is ctx and sc.persist is False


def test_begin_request_focus_without_a_scope_changes_nothing(context, store):
    context.current_project_id = "keep"
    assert begin_request_focus(context) == "keep"


async def test_a_request_only_projects_keys_are_kept_for_its_next_visit(context, store):
    """A probe/no-session switch writes keys under that project's scope; the
    next request (any conversation) snapshots them, and a later switch to the
    project brings them back."""
    q = store.create_project(title="Q", kind="CODING", goal="g")
    start_request(context, A, persist=False, origin="probe")
    await _switch(context, q)
    context.scratchpad.set("q_note", "kept")
    start_request(context, B)
    assert context.scratchpad.get("q_note") is None
    await _switch(context, q)
    assert context.scratchpad.get("q_note") == "kept"


async def test_an_exited_projects_keys_are_not_left_for_free_chat(context):
    start_request(context, A)
    await _create(context, "P")
    context.scratchpad.set("p_note", "x")
    await tool_manage_projects(context, action="exit")
    assert context.scratchpad.get("p_note") is None
    start_request(context, B)
    assert context.scratchpad.get("p_note") is None


def test_a_non_string_focus_value_is_no_project(tmp_path):
    """A mock or a corrupt store value must not become the request's project
    (a MagicMock pid scoped every path to projects/<MagicMock …>/)."""
    from unittest.mock import MagicMock
    store = MagicMock()
    store.get_focus.return_value = MagicMock()
    ctx = SimpleNamespace(project_store=store, scratchpad=None, current_project_id=None)
    sc = start_request(ctx, A)
    assert sc.project_id is None and ctx.current_project_id is None
