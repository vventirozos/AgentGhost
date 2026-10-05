"""§4LX — low-traffic tools: behaviour pins for the review's fixes."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.memory import tool_scratchpad
from ghost_agent.utils.logging import request_id_context


@pytest.fixture
def sp(tmp_path):
    s = Scratchpad(persist_path=tmp_path / "sp.db")
    s.set("__current_project__", "abc123", namespace=None)
    s.set("note_a", "global note", namespace=None)
    s.set("proj_note", "for project p1", namespace="proj::p1")
    s.set("other_note", "for project p2", namespace="proj::p2")
    return s


def _as(rid):
    return request_id_context.set(rid)


async def test_clear_empties_only_the_active_scope_and_keeps_system_keys(sp, tmp_path):
    sp.active_namespace = "proj::p1"
    tok = _as("req-owner")
    try:
        out = await tool_scratchpad(action="clear", scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    keys = set(Scratchpad(persist_path=tmp_path / "sp.db").keys())   # what survives a restart
    assert "proj_note" not in keys
    assert {"__current_project__", "note_a", "other_note"} <= keys and "Cleared 1" in out


async def test_a_general_scope_clear_never_drops_the_project_binding(sp):
    sp.active_namespace = None
    tok = _as("req-owner")
    try:
        await tool_scratchpad(action="clear", scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    assert sp.get("__current_project__") == "abc123" and sp.get("note_a") is None


@pytest.mark.parametrize("key", ["__current_project__", "_swarm_task_id::x", "_checkpoint_t1", "proj::p1"])
async def test_system_keys_cannot_be_set_or_deleted(sp, key):
    tok = _as("req-owner")
    try:
        s_out = await tool_scratchpad(action="set", key=key, value="x", scratchpad=sp)
        d_out = await tool_scratchpad(action="delete", key=key, scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    assert "system key" in s_out and "system key" in d_out
    assert sp.get("__current_project__") == "abc123"


async def test_delete_removes_one_key(sp):
    tok = _as("req-owner")
    try:
        out = await tool_scratchpad(action="delete", key="note_a", scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    assert "Deleted" in out and sp.get("note_a") is None and sp.get("other_note")


async def test_set_without_a_value_is_refused(sp):
    out = await tool_scratchpad(action="set", key="todo", scratchpad=sp)
    assert "'value' is required" in out and sp.get("todo") is None


async def test_a_probe_cannot_write_the_owners_scratchpad(sp):
    tok = _as("probe-1")
    try:
        out = await tool_scratchpad(action="set", key="pong_pending", value="PONG2", scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    assert "probe" in out and sp.get("pong_pending") is None


async def test_background_notes_stay_out_of_the_owners_prompt(sp):
    tok = _as("sched-chess")
    try:
        await tool_scratchpad(action="set", key="chess_move", value="e4", scratchpad=sp)
    finally:
        request_id_context.reset(tok)
    assert sp.namespace_of("chess_move") == "bg"
    assert "chess_move" not in sp.list_all(hide_namespaces=("bg",))
    assert "chess_move" in sp.list_all()                  # the background job still sees it


# ── self_play_loop counts attempts, and stops on failures in a row ──

async def test_a_failing_self_play_loop_is_bounded(monkeypatch):
    import ghost_agent.core.dream as D
    import ghost_agent.tools.memory as M
    tries = []

    class Dreamer:
        def __init__(self, ctx):
            pass

        async def synthetic_self_play(self, **kw):
            tries.append(1)
            raise RuntimeError("bad model")
    monkeypatch.setattr(D, "Dreamer", Dreamer)
    monkeypatch.setattr(M, "_count_playbook", lambda ctx: 0)

    async def nosleep(*a, **k):
        return None
    monkeypatch.setattr(M, "_consolidate_between_cycles", nosleep, raising=False)
    monkeypatch.setattr(M, "_derive_loop_cooloff", lambda ctx: 0.01)
    ctx = SimpleNamespace(llm_client=SimpleNamespace(foreground_tasks=0))
    stop = asyncio.Event()
    await asyncio.wait_for(M._run_self_play_loop(ctx, model_name="x", max_cycles=10, stop_event=stop), 30)
    assert len(tries) == 3


async def test_max_cycles_counts_attempts(monkeypatch):
    import ghost_agent.core.dream as D
    import ghost_agent.tools.memory as M
    tries = []

    class Dreamer:
        def __init__(self, ctx):
            pass

        async def synthetic_self_play(self, **kw):
            tries.append(1)
            if len(tries) % 2:
                raise RuntimeError("flaky")
    monkeypatch.setattr(D, "Dreamer", Dreamer)
    monkeypatch.setattr(M, "_count_playbook", lambda ctx: 0)

    async def nosleep(*a, **k):
        return None
    monkeypatch.setattr(M, "_consolidate_between_cycles", nosleep, raising=False)
    monkeypatch.setattr(M, "_derive_loop_cooloff", lambda ctx: 0.01)
    ctx = SimpleNamespace(llm_client=SimpleNamespace(foreground_tasks=0))
    await asyncio.wait_for(M._run_self_play_loop(ctx, model_name="x", max_cycles=2,
                                                 stop_event=asyncio.Event()), 30)
    assert len(tries) == 2


# ── postmortem show: exact first, an ambiguous prefix lists the candidates ──

async def test_postmortem_show_never_picks_silently():
    from ghost_agent.tools.postmortem_review import tool_postmortem
    rows = [SimpleNamespace(id="abc111", category="c", status="s"),
            SimpleNamespace(id="abc222", category="c", status="s")]
    q = MagicMock()
    q.all.return_value = rows
    out = await tool_postmortem(action="show", defect_id=" abc ", defect_queue=q)
    assert "matches 2 defects" in out
