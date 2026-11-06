"""§4FH C1 pins: a coding leaf's project pin survives the turn, and a leaf
that changed nothing inside the project workspace cannot be DONE.

The defect: `handle_chat` ran the conversation reconciler first; the leaf's
prompt had no conversation binding, so it parked the pinned project and
every leaf wrote at the sandbox ROOT — six theatrical DONEs. §4SX B0: the
leaf passes its own request scope (the pin) instead.
"""
import asyncio
from types import SimpleNamespace

import pytest

from ghost_agent.core import coding_loop as cl
from ghost_agent.tools import projects as tp


def _base(pid=None):
    return SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=pid,
                           memory_system=object(), graph_memory=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object(),
                           scratchpad=None)


def test_the_leaf_turn_runs_under_its_pinned_scope(monkeypatch):
    """§4SX B0 (r2 #7): run_leaf_turn hands handle_chat a pre-built scope
    holding the pin — no conversation, never a focus row — and the request
    start keeps it. World where it fails: the leaf turn starts with no
    project and writes at the sandbox root."""
    from ghost_agent.core import agent as agent_mod
    seen = {}

    class FakeAgent:
        def __init__(self, iso):
            self.iso = iso; self.available_tools = {"file_system": 1}
            self.disabled_tools = set(); self.max_turns_override = None

        async def handle_chat(self, body, background_tasks=None, request_id=None, scope=None):
            seen["scope"] = scope
            from ghost_agent.core.request_scope import install, reset
            scope.context = self.iso
            tok = install(scope)
            try:
                tp.begin_request_focus(self.iso)
                seen["during"] = self.iso.current_project_id
            finally:
                reset(tok)
            return "ok", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    asyncio.run(cl.run_leaf_turn(_base(), leaf_id="L", prompt="p", is_background=False,
                                 project_id="p42"))
    sc = seen["scope"]
    assert sc.project_id == "p42" and sc.persist is False and sc.conversation_key == ""
    assert seen["during"] == "p42"


def test_run_leaf_turn_fails_closed_when_the_pin_is_lost(monkeypatch):
    """World where it fails: the pin is parked mid-turn and the runner still
    returns the reply — the executor then diffs an empty project and, before
    §4FH, could pass on the verify alone."""
    from ghost_agent.core import agent as agent_mod

    class FakeAgent:
        def __init__(self, iso):
            self.iso = iso; self.available_tools = {"file_system": 1, "execute": 1, "browser": 1}
            self.disabled_tools = set(); self.max_turns_override = None

        async def handle_chat(self, body, background_tasks=None, request_id=None, scope=None):
            scope.project_id = None                     # anything moving the pin
            return "VERIFY: none\nSUMMARY: done", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    with pytest.raises(RuntimeError, match="lost its project pin"):
        asyncio.run(cl.run_leaf_turn(_base(), leaf_id="L", prompt="p", is_background=False,
                                     project_id="p42"))


def test_run_leaf_turn_routes_calls_through_the_background_lane(monkeypatch):
    """§4FH M1. World where it fails: `is_background` is accepted and ignored,
    so an idle-tick leaf competes with a live user turn for the single slot."""
    from ghost_agent.core import agent as agent_mod
    seen = {}

    class Inner:
        async def chat_completion(self, payload, *a, **kw):
            seen.update(kw); return {}

    class FakeAgent:
        def __init__(self, iso):
            self.iso = iso; self.available_tools = {"file_system": 1, "execute": 1}
            self.disabled_tools = set(); self.max_turns_override = None

        async def handle_chat(self, body, background_tasks=None, request_id=None, scope=None):
            await self.iso.llm_client.chat_completion({}, x=1)
            return "ok", None, None
    monkeypatch.setattr(agent_mod, "GhostAgent", FakeAgent)
    base = _base(); base.llm_client = Inner()
    asyncio.run(cl.run_leaf_turn(base, leaf_id="L", prompt="p", is_background=True, project_id="p42"))
    assert seen.get("is_background") is True
    seen.clear()
    asyncio.run(cl.run_leaf_turn(base, leaf_id="L", prompt="p", is_background=False, project_id="p42"))
    assert "is_background" not in seen


def test_no_files_inside_the_workspace_is_a_failed_attempt_even_with_a_verify(monkeypatch, tmp_path):
    """World where it fails: a leaf whose files landed outside the project
    (empty diff) passes on the verify command alone — the theatrical DONE."""
    class Store:
        def get_project(self, pid):
            return {"id": pid, "workspace_dir": str(tmp_path), "metadata": {}}
    ctx = SimpleNamespace(current_project_id="p1", project_store=Store(),
                          llm_client=object(), args=SimpleNamespace(model="m"))

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        return "VERIFY: python -m pytest -q\nSUMMARY: wrote files (elsewhere)"
    monkeypatch.setattr(cl, "run_leaf_turn", fake_turn)

    async def runner(name, args):
        return "3 passed\nEXIT CODE: 0"
    res = asyncio.run(cl.build_coding_task_agentic(ctx, "x", tool_runner=runner, max_attempts=1))
    assert res.ok is False and "no files inside the project workspace" in res.detail
