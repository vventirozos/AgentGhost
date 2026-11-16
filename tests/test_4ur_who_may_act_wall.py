"""§4UR-w: a who-may-act wall is not an argument puzzle (2026-11-16).

probe-3f34a566 ("delete project f28c7a9669de"): manage_projects refused with
"only the user's own conversation can delete a project — a background,
scheduled or test turn cannot". The ×3 loop breaker's refusal advice said the
message "names exactly what is wrong with the arguments … re-issue the SAME
tool with that one thing corrected … do NOT abandon the task". The model
read the wall as an argument puzzle: 4 delete calls and a file_system delete
of the folder in 7 minutes, cancelled by the operator.

Walls carry a reason code in ``outcome.WHO_MAY_ACT_REASONS``. The loop gives
them their own per-failure diagnostic, and warns at the 2nd identical refusal
instead of the 3rd, saying "do NOT call it again; tell the user".
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.agent import GhostAgent, _wall_diagnostic, _wall_steer
from ghost_agent.tools.outcome import (ToolOutcome, WHO_MAY_ACT_REASONS,
                                       is_who_may_act_wall)

_WALL_TEXT = ("ERROR: NOT done: only the user's own conversation can delete a project — a background, "
              "scheduled or test turn cannot. Nothing changed.")


# ------------------------------------------------------------- classifier

def test_every_wall_code_is_a_wall_and_nothing_else_is():
    for code in WHO_MAY_ACT_REASONS:
        assert is_who_may_act_wall(ToolOutcome.rejected("x", reason_code=code)), code
    # an argument refusal, a failure with a wall code, plain text: not walls
    assert not is_who_may_act_wall(ToolOutcome.rejected("x", reason_code="manage_projects_refused"))
    assert not is_who_may_act_wall(ToolOutcome.failed("x", reason_code="who_may_act"))
    assert not is_who_may_act_wall("ERROR: only the owner can …")


def _emitted_reason_codes():
    """Every reason code the source EMITS: `reason_code="x"` keywords, and a
    refuse helper's trailing string argument (`_refuse(msg, "x")`,
    `_mood_refusal(MSG, "x")`, `_err(msg, X)`). Parsed, not grepped: a code
    left only in a comment or another set does not count (r2 m8)."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "ghost_agent"
    codes, names = set(), {}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names[t.id] = node.value.value
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg == "reason_code" and isinstance(kw.value, ast.Constant):
                    codes.add(kw.value.value)
            fn = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fn in ("_refuse", "_mood_refusal", "_err") and len(node.args) >= 2:
                a = node.args[-1]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    codes.add(a.value)
                elif isinstance(a, ast.Name):
                    codes.add(("NAME", a.id))
    return {names.get(c[1], c) if isinstance(c, tuple) else c for c in codes}


def test_the_wall_codes_are_the_ones_the_tools_emit():
    """Every code in the set is emitted by some tool (a renamed code would
    silently drop out of the class)."""
    emitted = _emitted_reason_codes()
    for code in WHO_MAY_ACT_REASONS:
        assert code in emitted, code


# ------------------------------------------------- manage_projects walls

@pytest.fixture
def store(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(tmp_path, store):
    from ghost_agent.memory.scratchpad import Scratchpad
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None)


@pytest.fixture
def probe_rid():
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("probe-3f34a566")
    yield
    request_id_context.reset(tok)


@pytest.mark.asyncio
async def test_a_probe_delete_is_a_wall(tmp_path, store, probe_rid):
    from ghost_agent.tools.projects import tool_manage_projects_for_model as entry
    pid = store.create_project("Owner Project", goal="g")
    out = await entry(_ctx(tmp_path, store), action="delete", project_id=pid)
    assert is_who_may_act_wall(out), (out, getattr(out, "reason_code", None))
    assert store.get_project(pid) is not None


@pytest.mark.asyncio
async def test_probe_autopilot_budget_and_close_are_walls(tmp_path, store, probe_rid):
    from ghost_agent.tools.projects import tool_manage_projects
    ctx = _ctx(tmp_path, store)
    pid = store.create_project("Owner Project 2", goal="g")
    for kw in ({"action": "autopilot", "project_id": pid, "enabled": True},
               {"action": "autopilot", "project_id": "all", "enabled": False},
               {"action": "budget", "project_id": pid, "metadata": {"unattended_per_day": 3}},
               {"action": "update", "project_id": pid, "status": "DONE"}):
        out = await tool_manage_projects(ctx, **kw)
        assert is_who_may_act_wall(out), (kw, str(out)[:200], getattr(out, "reason_code", None))


@pytest.mark.asyncio
async def test_an_argument_refusal_is_not_a_wall(tmp_path, store, probe_rid):
    from ghost_agent.tools.projects import tool_manage_projects
    out = await tool_manage_projects(_ctx(tmp_path, store), action="task_add", project_id="nope00000000")
    assert getattr(out, "is_rejection", False) and not is_who_may_act_wall(out)


# ------------------------------------------------------------- the texts

def test_the_wall_texts_say_stop_and_tell_the_user():
    d, st = _wall_diagnostic("manage_projects", _WALL_TEXT), _wall_steer("manage_projects", 2, _WALL_TEXT)
    for t in (d, st):
        assert "tell the user in one line what the refusal says has to come from them" in t
        assert "If the refusal itself names a different, narrower call this turn MAY make" in t
        assert "Re-issue the SAME tool" not in t and "do NOT abandon" not in t
    assert "NOT ALLOWED HERE — a who-may-act wall, not an argument error" in d
    assert "do NOT repeat it" in st and "WHO may do it" in st


def test_file_system_project_delete_no_longer_promises_a_preview(tmp_path):
    """§4UK removed the preview: the refusal and the sandbox rm guard said
    delete "shows the user a preview and asks"."""
    from ghost_agent.tools.file_system import _protected_dir_refusal
    proj = tmp_path / "projects" / "abcdef012345"
    proj.mkdir(parents=True)
    out = _protected_dir_refusal(tmp_path, proj, "projects/abcdef012345")
    assert out is not None and "a whole project" in str(out)
    assert "preview" not in str(out) and "user's own conversation" in str(out)
    from ghost_agent.sandbox.docker import SAFE_RM_SH
    assert "preview" not in SAFE_RM_SH and "own conversation" in SAFE_RM_SH


# ------------------------------------------------- end to end, handle_chat

@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _drive(agent, result, n_calls):
    seen, state = [], {"n": 0}

    async def fake(payload, *a, **kw):
        seen.append(payload)
        state["n"] += 1
        if state["n"] <= n_calls:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['n']}", "function": {
                    "name": "manage_projects",
                    "arguments": json.dumps({"action": "delete", "project_id": "f28c7a9669de"})}}]}}]}
        return {"choices": [{"message": {"content": "It has to come from your own conversation.",
                                         "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["manage_projects"] = AsyncMock(return_value=result)
    return seen


def _new_user_msgs(seen, token):
    """The user-role messages of the loop's last call that carry THIS test's
    token (the mock context keeps history across tests, so an earlier test's
    identical alert is already in the conversation). A diagnostic is cut at
    its sandbox listing: its re-anchor tail quotes the conversation."""
    last = max(seen, key=lambda p: len(p.get("messages", [])))   # the loop's last call, not a side call
    out = []
    for m in last.get("messages", []):
        c = str(m.get("content", ""))
        if m.get("role") != "user":
            continue
        if c.startswith("AUTO-DIAGNOSTIC"):
            c = c.split("\n\nCURRENT SANDBOX")[0]
            if token in c:
                out.append(c)
            continue
        # an alert may ride inside another message (the pivot joins the state block)
        for seg in c.split("SYSTEM ALERT")[1:]:
            seg = "SYSTEM ALERT" + seg[:700]
            if token in seg:
                out.append(seg)
    return out


def _wall(token):
    return ToolOutcome.rejected(f"{_WALL_TEXT} [{token}]", world_changed=False, reason_code="who_may_act")


_REQ = {"messages": [{"role": "user", "content": "delete project f28c7a9669de"}], "model": "Qwen-Test"}


@pytest.mark.asyncio
async def test_a_wall_warns_at_its_second_refusal_with_the_wall_advice(agent):
    """FAILS IF: the wall gets the argument advice, or waits for a 3rd try."""
    seen = _drive(agent, _wall("tok-w2"), 2)
    await agent.handle_chat(_REQ, background_tasks=None)
    msgs = "\n".join(_new_user_msgs(seen, "tok-w2"))
    assert "WHO may do it" in msgs and "do NOT repeat it" in msgs
    assert "NOT ALLOWED HERE" in msgs
    assert "Re-issue the SAME tool with that one thing corrected" not in msgs


@pytest.mark.asyncio
async def test_an_argument_refusal_keeps_its_advice_and_its_threshold(agent):
    """Control: a refusal that is NOT a wall still gets the corrected-re-issue
    advice, and not at the 2nd repeat."""
    txt = "ERROR: unknown task: zzz [tok-a2]"
    seen = _drive(agent, ToolOutcome.rejected(txt, world_changed=False, reason_code="manage_projects_refused"), 2)
    await agent.handle_chat(_REQ, background_tasks=None)
    msgs = "\n".join(_new_user_msgs(seen, "tok-a2"))
    assert "AUTO-DIAGNOSTIC" in msgs                      # the run did produce its diagnostics
    assert "WHO may do it" not in msgs and "NOT ALLOWED HERE" not in msgs
    assert "REFUSED the same call" not in msgs           # ×2: no breaker yet


@pytest.mark.asyncio
async def test_an_argument_refusal_gets_the_reissue_advice_at_three(agent):
    seen = _drive(agent, ToolOutcome.rejected("ERROR: unknown task: zzz [tok-a3]", world_changed=False,
                                              reason_code="manage_projects_refused"), 3)
    await agent.handle_chat(_REQ, background_tasks=None)
    msgs = "\n".join(_new_user_msgs(seen, "tok-a3"))
    assert "Re-issue the SAME tool with that one thing corrected" in msgs
    assert "WHO may do it" not in msgs


# ------------------------------------------------- review r1 (§4UR-w)

@pytest.mark.asyncio
async def test_a_probe_archive_of_an_owner_project_is_a_wall(tmp_path, store):
    """r1 M1: archive goes through `_teardown_refusal` only — the obvious next
    try after a refused delete got the argument advice."""
    from ghost_agent.core import request_scope as RS
    from ghost_agent.tools.projects import tool_manage_projects
    ctx = _ctx(tmp_path, store)
    pid = store.create_project("Owner Project 3", goal="g")
    tok = RS.install(RS.RequestScope(origin=RS.ORIGIN_PROBE, context=ctx))
    try:
        for kw in ({"action": "archive", "project_id": pid},
                   {"action": "update", "project_id": pid, "status": "ARCHIVED"}):
            out = await tool_manage_projects(ctx, **kw)
            assert is_who_may_act_wall(out), (kw, str(out)[:200], getattr(out, "reason_code", None))
    finally:
        RS.reset(tok)
    assert str(store.get_project(pid)["status"]).upper() != "ARCHIVED"


def test_scheduled_task_and_foreign_job_walls_carry_the_code(probe_rid):
    from ghost_agent.tools.tasks import _schedule_refusal
    out = _schedule_refusal("create", None, "t", "", 60)
    assert is_who_may_act_wall(out), (out, getattr(out, "reason_code", None))
    # delegate.py's two foreign-job walls (read, cancel): parsed — each
    # `ToolOutcome.rejected(<text>, …)` carrying the text has the wall code
    import ast
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "src" / "ghost_agent" / "tools" / "delegate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "rejected" and node.args:
            text = "".join(v.value for v in ast.walk(node.args[0])
                           if isinstance(v, ast.Constant) and isinstance(v.value, str))
            code = next((k.value.value for k in node.keywords
                         if k.arg == "reason_code" and isinstance(k.value, ast.Constant)), None)
            for key in ("yours to read", "may cancel it"):
                if key in text:
                    found[key] = code
    assert found == {"yours to read": "who_may_act", "may cancel it": "who_may_act"}, found


@pytest.mark.asyncio
async def test_the_first_wall_refusal_already_gets_the_wall_diagnostic(agent):
    """r1 m4: ×1 — no breaker yet, the diagnostic alone carries it."""
    # two DIFFERENT wall texts: no repeat (no breaker), and the loop's next
    # call is one the mock records (a 1-call drive ends on another path)
    seen = _drive(agent, None, 2)
    agent.available_tools["manage_projects"] = AsyncMock(side_effect=[_wall("tok-w1a"), _wall("tok-w1b")])
    await agent.handle_chat(_REQ, background_tasks=None)
    msgs = "\n".join(_new_user_msgs(seen, "tok-w1a"))
    assert "NOT ALLOWED HERE — a who-may-act wall" in msgs
    assert "WHO may do it" not in msgs                    # the ×2 alert has not fired


@pytest.mark.asyncio
async def test_an_execute_wall_is_a_wall_too(agent):
    """r1 m4: the `execute` capture site (a refusal envelope with EXIT CODE: 1)."""
    seen, state = [], {"n": 0}

    async def fake(payload, *a, **kw):
        seen.append(payload)
        state["n"] += 1
        if state["n"] <= 2:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['n']}", "function": {"name": "execute",
                                         "arguments": json.dumps({"command": "curl https://x.example"})}}]}}]}
        return {"choices": [{"message": {"content": "Not from this request.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["execute"] = AsyncMock(side_effect=[ToolOutcome.rejected(
        f"EXIT CODE: 1\nSTDOUT/STDERR:\nNot sent: this request read outside content (page x), so nothing was "
        f"sent. [tok-ex{i}]", world_changed=False, reason_code="untrusted_egress") for i in (1, 2)])
    await agent.handle_chat({"messages": [{"role": "user", "content": "fetch x"}], "model": "Qwen-Test"},
                            background_tasks=None)
    assert "NOT ALLOWED HERE — a who-may-act wall" in "\n".join(_new_user_msgs(seen, "tok-ex1"))


@pytest.mark.asyncio
async def test_the_recovery_pivot_does_not_send_a_wall_round_three_ways(agent):
    """r1 M2: at structural failure 4 the System-3 pivot said "lay out three
    DIFFERENT ways … then ACT" — round the wall the steer just closed."""
    seen = _drive(agent, _wall("tok-w4"), 4)
    await agent.handle_chat(_REQ, background_tasks=None)
    msgs = "\n".join(_new_user_msgs(seen, "tok-w4"))
    assert "SYSTEM ALERT (recovery): the repeated failures end on a who-may-act wall" in msgs
    assert "three DIFFERENT" not in msgs and "DIFFERENT ways" not in msgs


# ------------------------------------------------- review r2 (§4UR-w)

@pytest.mark.asyncio
async def test_a_wall_loop_ends_at_six_with_the_wall_reply(agent):
    """r2 M1 + m8: no decay on a wall pivot, so the 6-cap ends a pure wall
    loop at the 6th call — and its reply says who must do it, not "rephrase"."""
    _drive(agent, _wall("tok-cap"), 12)
    out = await agent.handle_chat(_REQ, background_tasks=None)
    calls = agent.available_tools["manage_projects"].await_count
    assert calls == 6, calls
    text = json.dumps(out) if not isinstance(out, str) else out
    assert "refused on who may do it" in text
    assert "Please rephrase or break it into smaller steps" not in text


@pytest.mark.asyncio
async def test_a_stray_wall_after_ordinary_failures_keeps_the_ordinary_pivot(agent):
    """r2 m2: three argument refusals, then ONE wall: the pivot is about the
    real problem (three ways), not the wall."""
    seen = _drive(agent, None, 4)
    agent.available_tools["manage_projects"] = AsyncMock(side_effect=[
        ToolOutcome.rejected(f"ERROR: unknown task: t{i} [tok-st]", world_changed=False,
                             reason_code="manage_projects_refused") for i in range(3)] + [_wall("tok-st")])
    await agent.handle_chat(_REQ, background_tasks=None)
    last = max(seen, key=lambda p: len(p.get("messages", [])))
    body = "\n".join(str(m.get("content", "")) for m in last.get("messages", []) if m.get("role") == "user")
    # THIS run's pivot: the message that names three ways AND this test's token
    # (the mock context's history can hold another test's pivot — r3 m)
    assert any("three DIFFERENT ways" in str(m.get("content", "")) and "tok-st" in str(m.get("content", ""))
               for m in last.get("messages", []) if m.get("role") == "user"), "no ordinary pivot of this run"
    walls = "\n".join(_new_user_msgs(seen, "tok-st"))
    assert "SYSTEM ALERT (recovery): the repeated failures end on a who-may-act wall" not in walls


# ------------------------------------------------- review r3 (§4UR-w)

@pytest.mark.asyncio
async def test_a_wall_then_timeouts_does_not_blame_the_wall(agent):
    """r3 M1: one wall, then network timeouts: the cap fires on the transient
    arm and must not say "refused on who may do it"."""
    _drive(agent, None, 12)
    agent.available_tools["manage_projects"] = AsyncMock(side_effect=[_wall("tok-stale")] + [
        ToolOutcome.failed(f"Error: connection timed out (attempt {i})", world_changed=False) for i in range(11)])
    out = await agent.handle_chat(_REQ, background_tasks=None)
    text = json.dumps(out) if not isinstance(out, str) else out
    assert "refused on who may do it" not in text


@pytest.mark.asyncio
async def test_one_stray_wall_does_not_take_over_the_cap(agent):
    """r3 m: seven argument refusals then ONE wall reach the structural cap —
    the wall did not end the turn alone."""
    _drive(agent, None, 12)
    agent.available_tools["manage_projects"] = AsyncMock(side_effect=[
        ToolOutcome.rejected(f"ERROR: unknown task: t{i} [tok-sw]", world_changed=False,
                             reason_code="manage_projects_refused") for i in range(7)] + [_wall("tok-sw")] * 5)
    out = await agent.handle_chat(_REQ, background_tasks=None)
    text = json.dumps(out) if not isinstance(out, str) else out
    calls = agent.available_tools["manage_projects"].await_count
    # 7 argument refusals (pivot #1 at 4 decays to 2; pivot #2 at 5) + 1 wall = 6:
    # capped right after the first wall
    assert calls == 8, calls
    assert "refused on who may do it" not in text


# ------------------------------------------------- review r4 (§4UR-w)

@pytest.mark.asyncio
async def test_a_repeated_wall_then_unknown_tools_does_not_blame_the_wall(agent):
    """r4: strikes outside the failure chain (here: an unknown tool, a
    synthetic strike) end the "latest failure was a wall" story."""
    state = {"n": 0}

    async def fake(payload, *a, **kw):
        state["n"] += 1
        name, args = (("manage_projects", {"action": "delete", "project_id": "f28c7a9669de"})
                      if state["n"] <= 2 else ("no_such_tool_4ur", {"x": state["n"]}))
        if state["n"] <= 14:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['n']}", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}
        return {"choices": [{"message": {"content": "done", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    mp = agent.available_tools["manage_projects"] = AsyncMock(return_value=_wall("tok-r4"))
    out = await agent.handle_chat(_REQ, background_tasks=None)
    text = json.dumps(out) if not isinstance(out, str) else out
    assert mp.await_count == 2, mp.await_count
    assert "refused on who may do it" not in text
