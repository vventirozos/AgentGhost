"""A channel member's turn is not the owner's — 2026-09-24 (tenant-bleed
minimum cut).

A stranger's Slack thread ran with the owner's full profile (name, home
address, company) in its system prompt, was consolidated into the owner's
smart memory, and would have been filed under the owner's autobiographical
handle. The Slack bot now says who is asking (`X-Ghost-Requester: owner |
member`), the API passes it to `handle_chat`, and three consumers read one
predicate (`utils.logging.requester_is_member`): the USER PROFILE block, the
autobiographical handle, the smart-memory writers. Playbook writes were
already blocked for every Slack turn (§4KD); the member rule joins them.
Fail-closed: a Slack request that does not say who is asking is a member's.
"""
import ast
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, _MEMBER_PROFILE_PLACEHOLDER
from ghost_agent.utils.logging import (parse_requester_role, requester_is_member,
                                       requester_role_context, request_id_context)
from tests.helpers import FakeBgTasks, make_context

REPO = Path(__file__).resolve().parents[1]
BOT_PATH = REPO / "interface" / "externals" / "slack_bot" / "main.py"


@pytest.mark.parametrize("raw,expect", [
    ("owner", "owner"), ("member", "member"), (" Owner ", "owner"), ("MEMBER", "member"),
    ("", ""), (None, ""), ("admin", ""), ("owner;member", ""), (42, ""),
])
def test_parse_requester_role(raw, expect):
    assert parse_requester_role(raw) == expect


@pytest.mark.parametrize("role,req_id,expect", [
    ("member", "slack-abc", True), ("member", "web-1", True), ("Member", "x", True),
    ("owner", "slack-abc", False), ("owner", "abc", False),
    ("", "slack-abc", False),         # no declaration is the owner; the id's spelling means nothing
    ("", "abc12345", False), ("", "SYSTEM", False), ("", "probe-x", False),
])
def test_member_predicate_table(role, req_id, expect):
    t1 = requester_role_context.set(role)
    t2 = request_id_context.set(req_id)
    try:
        assert requester_is_member() is expect
    finally:
        request_id_context.reset(t2)
        requester_role_context.reset(t1)


def _bot_helper():
    """`requester_role_header` is a pure function; lift it out of the bot
    module without importing slack_bolt."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "requester_role_header")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(BOT_PATH), "exec"), ns)
    return ns["requester_role_header"]


@pytest.mark.parametrize("requester,owner,expect", [
    ("U1", "U1", "owner"), ("U2", "U1", "member"), (None, "U1", "member"),
    ("U1", None, "member"), ("", "", "member"),
])
def test_the_bot_names_the_requester(requester, owner, expect):
    assert _bot_helper()(requester, owner) == expect


def _parents(tree):
    par = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            par[ch] = node
    return par


def _enclosing_def(node, par):
    while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node = par.get(node)
    return node


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _dict_carries_role_header(d):
    for k, v in zip(d.keys, d.values):
        if (isinstance(k, ast.Constant) and k.value == "X-Ghost-Requester"
                and isinstance(v, ast.Call) and _is_name(v.func, "requester_role_header")
                and len(v.args) == 2 and _is_name(v.args[0], "requester") and _is_name(v.args[1], "OWNER_ID")):
            return True
    return False


def test_the_bot_sends_the_header_on_every_chat_post():
    """AST only: every `.post(GHOST_API_URL, …)` in the bot passes
    `headers=` whose VALUE (an inline dict, or a Name the enclosing function
    binds to a dict — the R3 review's bypass was a stray dict beside a
    `headers=AUTH_HEADERS` post) carries "X-Ghost-Requester":
    `requester_role_header(requester, OWNER_ID)`."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8")); par = _parents(tree)
    posts = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "post" and n.args and _is_name(n.args[0], "GHOST_API_URL")]
    assert posts, "no chat POST found"
    for call in posts:
        hdr = [k.value for k in call.keywords if k.arg == "headers"]
        assert hdr, "chat POST without headers="
        val = hdr[0]
        fn = _enclosing_def(call, par)
        if isinstance(val, ast.Name):
            bound = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                     and any(_is_name(t, val.id) for t in n.targets)]
            assert bound and all(isinstance(v, ast.Dict) and _dict_carries_role_header(v) for v in bound), (fn.name, val.id)
        else:
            assert isinstance(val, ast.Dict) and _dict_carries_role_header(val), fn.name


def _is_role_set_from_header(stmt):
    """`requester_role_context.set(parse_requester_role(request.headers.get("X-Ghost-Requester")))`
    as a bare expression statement."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    c = stmt.value
    if not (isinstance(c.func, ast.Attribute) and c.func.attr == "set"
            and _is_name(c.func.value, "requester_role_context") and c.args):
        return False
    inner = c.args[0]
    if not (isinstance(inner, ast.Call) and _is_name(inner.func, "parse_requester_role") and inner.args):
        return False
    get = inner.args[0]
    return (isinstance(get, ast.Call) and isinstance(get.func, ast.Attribute) and get.func.attr == "get"
            and bool(get.args) and isinstance(get.args[0], ast.Constant) and get.args[0].value == "X-Ghost-Requester")


def test_the_api_sets_the_role_before_every_handle_chat():
    """AST only: each `handle_chat(` call in the API sits in a route function
    whose BODY (a direct child statement — not under an `if`/`try`, the R3
    review's bypass) sets the role from the header at a line BEFORE the call."""
    tree = ast.parse((REPO / "src" / "ghost_agent" / "api" / "routes.py").read_text(encoding="utf-8"))
    par = _parents(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "handle_chat"]
    assert len(calls) == 2, len(calls)
    for call in calls:
        top = _enclosing_def(call, par)
        while _enclosing_def(par.get(top), par) is not None:
            top = _enclosing_def(par.get(top), par)
        sets = [st for st in top.body if _is_role_set_from_header(st)]
        assert sets and min(st.lineno for st in sets) < call.lineno, top.name


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _agent(monkeypatch, tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    ctx.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories")   # the autobiography capture rides the recorder
    ctx.args.smart_memory = 0.9
    ctx.journal = MagicMock()
    pm = MagicMock()
    pm.get_context_string = lambda: "## Root:\n- name: Vasilis\n- address: Makedonias 83, Athens"
    pm.load = lambda: {"root": {"name": "Vasilis"}}
    ctx.profile_memory = pm
    from ghost_agent.selfhood import SelfModel
    sm = MagicMock(spec=SelfModel)
    sm.enabled = True
    ctx.self_model = sm
    agent = GhostAgent(ctx)
    appended = []

    async def _append(kind, payload):
        appended.append((kind, payload))
    agent._journal_append_safe = _append
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Hello there."), _resp("(unreachable)")])
    return agent, ctx, appended


async def _system_prompt(ctx):
    first = ctx.llm_client.chat_completion.call_args_list[0]
    payload = first.args[0] if first.args and isinstance(first.args[0], dict) else first.kwargs
    return payload["messages"][0]["content"]


# "hi" and "thanks" take the trivial fast path (`_handle_trivial_chat`, its own
# profile reader — R2 review found it leaking); "hello, who are you" takes the
# full loop.
@pytest.mark.parametrize("ask", ["hi", "thanks", "hello, who are you"])
async def test_a_member_turn_carries_no_profile_no_handle_no_smart_memory(monkeypatch, tmp_path, ask):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="slack-11112222", requester_role="member")
    sys_prompt = await _system_prompt(ctx)
    assert "Makedonias" not in sys_prompt and "Vasilis" not in sys_prompt
    assert [k for k, _ in appended if k == "smart_memory"] == []
    assert _MEMBER_PROFILE_PLACEHOLDER in sys_prompt          # both paths carry the BOUNDARY, not silence
    assert "channel MEMBER" in sys_prompt and "must never be presented as this user's" in sys_prompt
    if ask == "hello, who are you":
        # R6: a member's turn is not written into the owner's autobiography at all
        assert ctx.self_model.capture_turn.call_args_list == []


@pytest.mark.parametrize("ask", ["hi", "hello, who are you"])
async def test_the_owner_keeps_the_profile_on_both_paths(monkeypatch, tmp_path, ask):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="web-1", requester_role="owner")
    assert "Makedonias" in await _system_prompt(ctx)


async def _drive_streamed_final(role, req_id):
    """The streamed-final generator, driven through the §4BZ harness: the
    drain's smart-memory write is the observer — it asks
    `requester_is_member()` INSIDE the generator, after handle_chat's
    finally would have reset the contextvars."""
    import dataclasses
    from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state
    a = make_stream_agent()
    a.context.args.smart_memory = 0.9
    a.context.journal = MagicMock()
    a._record_calibration_safe = AsyncMock()
    appended = []

    async def _append(kind, payload):
        appended.append(kind)
    a._journal_append_safe = _append

    async def final_stream(payload, use_coding=False):
        for d in ("Hello", " there."):
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = dataclasses.replace(_make_stream_state(reg), req_id=req_id, requester_role=role,
                             last_user_content="hello, who are you", lc="hello, who are you",
                             stream_messages_snapshot=[{"role": "user", "content": "hello, who are you"}])
    gen, _, _ = a._stream_final_generation(st)
    seen = []
    async for _chunk in gen:
        seen.append((request_id_context.get(), requester_role_context.get()))
    return appended, seen


async def test_the_streamed_drain_sees_the_role_and_the_request_id():
    """R2 review: the stream wrapper runs AFTER handle_chat's finally has
    reset the contextvars. A member on a WEB id (no slack- prefix to fall
    back on) must write no smart-memory row; an owner on a SLACK id must
    write one (the role, not the prefix, decides)."""
    member, seen = await _drive_streamed_final("member", "web-99990000")
    assert "smart_memory" not in member, member
    assert seen and all(rid == "web-99990000" and role == "member" for rid, role in seen), seen[:3]
    owner, seen2 = await _drive_streamed_final("owner", "slack-11110000")
    assert owner.count("smart_memory") == 1, owner
    assert seen2 and all(rid == "slack-11110000" and role == "owner" for rid, role in seen2)
    assert requester_role_context.get() == "" and request_id_context.get() == "SYSTEM"   # reset after the drain


async def test_the_owner_turn_keeps_all_three(monkeypatch, tmp_path):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello, who are you"}]},
                            FakeBgTasks(), request_id="slack-33334444", requester_role="owner")
    sys_prompt = await _system_prompt(ctx)
    assert "Makedonias" in sys_prompt and _MEMBER_PROFILE_PLACEHOLDER not in sys_prompt
    assert [k for k, _ in appended if k == "smart_memory"] == ["smart_memory"]
    assert ctx.self_model.capture_turn.call_args_list, "the fixture must reach the autobiography capture"
    assert any(c.kwargs.get("user_handle") == "Vasilis" for c in ctx.self_model.capture_turn.call_args_list)


async def test_a_turn_with_no_role_is_the_owners_whatever_its_id(monkeypatch, tmp_path):
    """The API key is the owner's credential; an undeclared request is the
    owner's on ANY id — a client that serves others must say `member`."""
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello, who are you"}]},
                            FakeBgTasks(), request_id="slack-55556666")
    assert "Makedonias" in await _system_prompt(ctx)
    assert [k for k, _ in appended if k == "smart_memory"] == ["smart_memory"]


async def test_a_web_turn_with_no_role_is_the_owners(monkeypatch, tmp_path):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello, who are you"}]},
                            FakeBgTasks(), request_id="web-77778888")
    assert "Makedonias" in await _system_prompt(ctx)
    assert [k for k, _ in appended if k == "smart_memory"] == ["smart_memory"]


async def test_the_role_does_not_leak_into_the_next_request(monkeypatch, tmp_path):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}]},
                            FakeBgTasks(), request_id="slack-1", requester_role="member")
    assert requester_role_context.get() == ""
    assert requester_is_member() is False


def test_the_playbook_writer_backstop_blocks_a_member():
    from ghost_agent.memory.skills import playbook_writes_blocked
    from ghost_agent.core.agent import turn_may_teach
    t1 = requester_role_context.set("member"); t2 = request_id_context.set("web-1")
    try:
        assert playbook_writes_blocked() is True
        assert turn_may_teach(MagicMock()) is False
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)
    t1 = requester_role_context.set("owner"); t2 = request_id_context.set("web-1")
    try:
        assert playbook_writes_blocked() is False
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)


def test_every_smart_memory_writer_asks_the_predicate():
    """AST only: every call that writes a `smart_memory` journal row sits
    under an `If` whose test contains `not requester_is_member()` (the R3
    review's bypass inverted the polarity: writes ONLY for members). The
    behavioural pins above cover the two known writers; this is the
    enumeration, and it is honest about its blind spot: a writer whose
    kind arrives in a variable is invisible to it."""
    import ghost_agent.core.agent as agent_mod
    tree = ast.parse(Path(agent_mod.__file__).read_text(encoding="utf-8")); par = _parents(tree)
    writers = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("_journal_append_safe", "append")):
            continue
        if not (n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "smart_memory"):
            continue
        if n.func.attr == "append" and not (isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "journal"):
            continue
        writers.append(n)
    assert len(writers) == 2, len(writers)
    for w in writers:
        node, guarded = par.get(w), False
        while node is not None:
            if isinstance(node, ast.If):
                for u in ast.walk(node.test):
                    if (isinstance(u, ast.UnaryOp) and isinstance(u.op, ast.Not)
                            and isinstance(u.operand, ast.Call) and _is_name(u.operand.func, "requester_is_member")):
                        guarded = True
                if guarded:
                    break
            node = par.get(node)
        assert guarded, ("smart_memory writer not under `not requester_is_member()` at line", w.lineno)


async def test_a_member_can_neither_write_nor_forget_the_owners_profile():
    """R3 review: `update_profile` and `unified_forget` had no requester
    gate — a member's "remember that I'm vegan" wrote into the owner's
    profile. Both tools now refuse for a member and touch nothing."""
    from ghost_agent.tools.memory import tool_update_profile, tool_unified_forget
    pm = MagicMock()
    t1 = requester_role_context.set("member"); t2 = request_id_context.set("web-1")
    try:
        out = await tool_update_profile(category="root", key="diet", value="vegan", profile_memory=pm)
        assert getattr(out, "reason_code", "") == "owner_data_blocked" and "SYSTEM BLOCK" in str(out)
        out2 = await tool_unified_forget(target="address", profile_memory=pm, memory_system=MagicMock())
        assert getattr(out2, "reason_code", "") == "owner_data_blocked"
        assert not pm.method_calls, pm.method_calls          # nothing touched
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)


async def test_the_owner_still_updates_the_profile():
    from ghost_agent.tools.memory import tool_update_profile
    pm = MagicMock()
    t1 = requester_role_context.set("owner"); t2 = request_id_context.set("web-1")
    try:
        out = await tool_update_profile(category="root", key="diet", value="vegan", profile_memory=pm)
        assert getattr(out, "reason_code", "") != "owner_data_blocked"
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)


async def test_a_member_gets_no_owner_location():
    from ghost_agent.tools.system import tool_check_location
    pm = MagicMock(); pm.load.return_value = {"root": {"location": "Athens, Greece"}}
    t1 = requester_role_context.set("member"); t2 = request_id_context.set("web-1")
    try:
        assert "not available" in await tool_check_location(pm) and not pm.load.called
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)
    t1 = requester_role_context.set("owner"); t2 = request_id_context.set("web-1")
    try:
        assert "Athens" in await tool_check_location(pm)
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)


def _force_planning_arm(monkeypatch):
    import importlib
    for _modname in ("ghost_agent.core.experiments", "src.ghost_agent.core.experiments"):
        try:
            _exp = importlib.import_module(_modname)
        except ImportError:
            continue
        _real = _exp.arm_for
        monkeypatch.setattr(_exp, "arm_for",
                            lambda ctx_, name, req_id="", _e=_exp, _r=_real: (
                                _e.TREATMENT if name == "use_planning" else _r(ctx_, name, req_id)))


async def _streamed_turn_through_handle_chat(monkeypatch, tmp_path, role, req_id):
    """The whole path: handle_chat(stream=True) on the planning arm, a DONE
    plan on turn 0 (`required_tool: none` → a streamed final), the client
    draining the SSE generator after handle_chat has returned. This is the
    one path that constructs `StreamState(requester_role=…)`. An action verb
    ("check") keeps the turn off the conversational path so the planner runs."""
    import json as _json
    from tests.test_finalize_stream_pins import sse
    _force_planning_arm(monkeypatch)
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    ctx.args.use_planning = True
    tree = {"id": "root", "description": "Answer", "status": "DONE", "children": []}
    plan = _resp("```json\n" + _json.dumps({"thought": "Answer now.", "tree_update": tree,
                                             "next_action_id": "root", "required_tool": "none"}) + "\n```")
    ctx.llm_client.chat_completion = AsyncMock(return_value=plan)

    async def final_stream(payload, use_coding=False):
        for d in ("Hello", " there."):
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    ctx.llm_client.stream_chat_completion = final_stream
    content, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "check who you are and tell me"}], "stream": True},
        FakeBgTasks(), request_id=req_id, requester_role=role)
    if role == "member":
        # R8: a member's turn never streams (the drain ran after the owner's
        # scope was restored) — the whole reply comes back as a string.
        assert isinstance(content, str), type(content)
    else:
        assert hasattr(content, "__aiter__"), type(content)
        async for _chunk in content:
            pass
    return [k for k, _ in appended if k == "smart_memory"]


async def test_a_streamed_member_turn_through_handle_chat_writes_no_smart_memory(monkeypatch, tmp_path):
    assert await _streamed_turn_through_handle_chat(monkeypatch, tmp_path, "member", "web-31415926") == []


async def test_a_streamed_owner_turn_on_a_slack_id_still_writes(monkeypatch, tmp_path):
    """Control, and the role-not-the-prefix proof: the owner's Slack turn
    keeps its smart memory (a missing role would make it a member's)."""
    assert await _streamed_turn_through_handle_chat(monkeypatch, tmp_path, "owner", "slack-27182818") == ["smart_memory"]


# ── the member data wall (2026-09-24 evening: two members were told the owner's name) ──

def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


async def _tool_turn(monkeypatch, tmp_path, role, tool, args):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    called = AsyncMock(return_value="TOOL RESULT: the owner's data")
    tools = {"web_search": AsyncMock(return_value="x")}
    tools[tool] = called                                   # (a dict literal would let web_search overwrite it)
    agent.available_tools = tools
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", tool, args)]),
        _resp("Done."), _resp("Done."), _resp("(unreachable)"),
    ])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check my name and tell me"}]},
                            FakeBgTasks(), request_id="web-1", requester_role=role)
    text = ""
    calls = ctx.llm_client.chat_completion.call_args_list
    if len(calls) > 1:
        second = calls[1]
        payload = second.args[0] if second.args and isinstance(second.args[0], dict) else second.kwargs
        text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    return called, text


def _dispatch_table():
    from ghost_agent.tools.registry import get_available_tools
    return sorted(get_available_tools(make_context()))


# Every dispatchable tool a MEMBER may NOT use — derived from the real
# dispatch table minus the allowlist, so a NEW tool is covered the day it is
# registered (R4 pins review: the old hand list missed 24 open tools).
def _member_refused_tools():
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS
    return [t for t in _dispatch_table() if t not in _MEMBER_ALLOWED_TOOLS]


def test_the_allowlist_names_real_tools_and_is_small():
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS
    table = set(_dispatch_table())
    assert set(_MEMBER_ALLOWED_TOOLS) <= table, set(_MEMBER_ALLOWED_TOOLS) - table
    # the owner-data tools the R4 review found open must be refused
    for t in ("file_system", "execute", "browser", "workspace", "manage_projects", "jobs",
              "manage_services", "create_skill", "delegate", "delegate_to_swarm", "notify_operator",
              "postgres_admin", "flag_uncertainty", "report_pdf", "deep_research", "introspect",
              "recall", "knowledge_base", "scratchpad", "update_profile"):
        if t in table:
            assert t not in _MEMBER_ALLOWED_TOOLS, t


@pytest.mark.parametrize("tool", _member_refused_tools())
async def test_a_member_is_refused_every_non_allowlisted_tool_at_dispatch(monkeypatch, tmp_path, tool):
    called, text = await _tool_turn(monkeypatch, tmp_path, "member", tool, {"x": 1})
    called.assert_not_awaited()
    # learn_skill meets the older no-teach block first; every other tool the wall
    assert ("SYSTEM BLOCK: this tool reads or writes the owner's data" in text
            or (tool == "learn_skill" and "SYSTEM BLOCK: lessons are not recorded" in text))


@pytest.mark.parametrize("tool", [t for t in _member_refused_tools()
                                  if t not in ("dream_mode", "self_play", "self_play_loop", "stop_self_play")])
async def test_the_owner_reaches_every_tool_the_member_is_refused(monkeypatch, tmp_path, tool):
    called, _ = await _tool_turn(monkeypatch, tmp_path, "owner", tool, {"x": 1})
    called.assert_awaited()


@pytest.mark.parametrize("tool,args", [("web_search", {"query": "weather"}), ("image_generation", {"prompt": "a cat"})])
async def test_a_member_reaches_the_allowlisted_tools(monkeypatch, tmp_path, tool, args):
    called, _ = await _tool_turn(monkeypatch, tmp_path, "member", tool, args)
    called.assert_awaited_once()


@pytest.mark.parametrize("args,allowed", [
    ({"action": "describe_picture", "target": "https://example.com/cat.png"}, True),
    ({"action": "describe_picture", "target": "gen_member1.png"}, True),        # generated for a member
    ({"action": "describe_picture", "target": "floorplan.jpg"}, False),         # the owner's file
    ({"action": "describe_picture", "target": "/workspace/projects/p1/photo.png"}, False),
])
async def test_a_members_vision_call_may_touch_only_member_images(monkeypatch, tmp_path, args, allowed):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent._note_member_files({"gen_member1.png"})
    called = AsyncMock(return_value="VISION ANALYSIS RESULT: a cat")
    agent.available_tools = {"vision_analysis": called}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "vision_analysis", args)]), _resp("Done."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check this image"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert called.await_count == (1 if allowed else 0)


async def test_a_member_may_not_restyle_the_owners_photo(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    called = AsyncMock(return_value="SUCCESS: Image generated.")
    agent.available_tools = {"image_generation": called}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "restyle", "reference_images": ["family.jpg"]})]),
        _resp("Done."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "make an image from family.jpg"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    called.assert_not_awaited()


def test_a_members_reply_links_only_member_images():
    from ghost_agent.core.agent import _scrub_member_download_links
    text = ("![x](/api/download/property_analysis_rawson_close.pdf) and "
            "![generated image](/api/download/gen_m1.png) and /api/download/projects/p1/PROJECT_MAP.md")
    out = _scrub_member_download_links(text, {"gen_m1.png"})
    assert "rawson" not in out and "PROJECT_MAP" not in out
    assert "![generated image](/api/download/gen_m1.png)" in out
    assert out.count("[file not available]") == 2


async def test_a_members_reply_is_scrubbed_end_to_end(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    _link = "Here is the file you asked for: ![x](/api/download/property_analysis_rawson_close.pdf)"
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp(_link))
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "reply with that link"}]},
                                        FakeBgTasks(), request_id="web-1", requester_role="member")
    assert "rawson" not in out and "[file not available]" in out
    agent2, ctx2, _ = _agent(monkeypatch, tmp_path / "o")
    ctx2.llm_client.chat_completion = AsyncMock(return_value=_resp(_link))
    out2, _, _ = await agent2.handle_chat({"messages": [{"role": "user", "content": "reply with that link"}]},
                                          FakeBgTasks(), request_id="web-2", requester_role="owner")
    assert "rawson" in out2


def test_the_download_route_serves_a_member_only_member_images(tmp_path):
    """The API side of the same leak: a member-role download may fetch only
    images generated on members' turns; the owner's key alone fetches all."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    (tmp_path / "owner.pdf").write_bytes(b"%PDF owner")
    (tmp_path / "gen_m1.png").write_bytes(b"\x89PNG member")
    agent = MagicMock()
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path
    agent.context.current_project_id = None
    agent._member_generated_files = {"gen_m1.png"}
    app = FastAPI()
    app.include_router(routes.router)
    app.state.agent = agent
    app.state.args = MagicMock()
    import os
    key = os.environ.get("GHOST_API_KEY", "")
    with TestClient(app) as c:
        h = {"X-Ghost-Key": key}
        hm = {**h, "X-Ghost-Requester": "member"}
        assert c.get("/api/download/owner.pdf", headers=h).status_code == 200
        assert c.get("/api/download/owner.pdf", headers=hm).status_code == 404
        assert c.get("/api/download/gen_m1.png", headers=hm).status_code == 200


async def test_a_member_turn_hydrates_no_owner_memory_and_no_playbook(monkeypatch, tmp_path):
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    bus = MagicMock(); bus.hydrate_context = AsyncMock(return_value="RECALLED: the owner is Vasilis")
    monkeypatch.setattr(agent, "_get_memory_bus", lambda: bus)
    playbook_calls = []
    from ghost_agent.core.agent import GhostAgent as _G
    async def _pb(self, q):
        playbook_calls.append(q); return "LESSON: something about the owner"
    monkeypatch.setattr(_G._RequestState, "get_skill_playbook", _pb)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("I don't know your name."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check my name and tell me"}]},
                            FakeBgTasks(), request_id="web-2", requester_role="member")
    bus.hydrate_context.assert_not_awaited()
    assert playbook_calls == []
    sys_prompt = await _system_prompt(ctx)
    assert "Vasilis" not in sys_prompt
    # control: the owner hydrates
    agent2, ctx2, _ = _agent(monkeypatch, tmp_path)
    monkeypatch.setattr(agent2, "_get_memory_bus", lambda: bus)
    ctx2.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Vasilis."), _resp("(unreachable)")])
    await agent2.handle_chat({"messages": [{"role": "user", "content": "check my name and tell me"}]},
                             FakeBgTasks(), request_id="web-3", requester_role="owner")
    bus.hydrate_context.assert_awaited()


async def test_the_role_rides_the_trajectory(monkeypatch, tmp_path):
    """The idle phases refuse a member's turn as a lesson source by the role
    recorded on the trajectory (`extra["requester_role"]`), never by how the
    request id is spelled."""
    from ghost_agent.memory.skills import trajectory_may_teach
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "check who you are and tell me"}]},
                            FakeBgTasks(), request_id="slack-42424242", requester_role="member")
    rows = list(ctx.trajectory_collector.iter_trajectories())
    assert rows, "no trajectory recorded"
    assert rows[-1].extra.get("requester_role") == "member"
    assert trajectory_may_teach(rows[-1]) is False
    agent2, ctx2, _ = _agent(monkeypatch, tmp_path / "o")
    await agent2.handle_chat({"messages": [{"role": "user", "content": "check who you are and tell me"}]},
                             FakeBgTasks(), request_id="slack-42424243", requester_role="owner")
    rows2 = list(ctx2.trajectory_collector.iter_trajectories())
    assert rows2 and rows2[-1].extra.get("requester_role") == "owner" and trajectory_may_teach(rows2[-1]) is True



# ── the prompt side: every owner-data block, on both prompt paths ──

def _marked_agent(monkeypatch, tmp_path, planning):
    """An agent whose every owner-data source carries a unique marker."""
    agent, ctx, appended = _agent(monkeypatch, tmp_path)
    ctx.profile_memory.get_context_string = lambda: "MARK-PROFILE Vasilis"
    sp = MagicMock(); sp.list_all.return_value = "MARK-SCRATCH owner note"; sp.get.return_value = None
    ctx.scratchpad = sp
    ut = MagicMock(); ut.persisted_context.return_value = "MARK-UNCERTAINTY owner blind spot"
    ctx.uncertainty_tracker = ut
    ask = MagicMock(); ask.surfaced_for_prompt.return_value = ("MARK-AUTOSKILL owner past request", [])
    ask.format_for_prompt.return_value = "MARK-AUTOSKILL owner past request"
    ctx.auto_skill_store = ask
    bus = MagicMock(); bus.hydrate_context = AsyncMock(return_value="MARK-RECALL the owner is Vasilis")
    monkeypatch.setattr(agent, "_get_memory_bus", lambda: bus)
    clog = MagicMock(); clog.explain_belief_change.return_value = "MARK-BELIEF the owner's sons Thodoris and Leonidas"
    ctx.contradiction_log = clog
    comp = MagicMock(); comp.by_domain.return_value = {"coding": (0.9, 500)}
    comp.get_context_string.return_value = "MARK-COMPETENCE coding 90% over 500 turns"
    ctx.metacog = MagicMock(); ctx.metacog.competence = comp
    from ghost_agent.core.agent import GhostAgent as _G

    async def _pb(self, q):
        return "MARK-PLAYBOOK owner lesson"
    monkeypatch.setattr(_G._RequestState, "get_skill_playbook", _pb)
    if planning:
        _force_planning_arm(monkeypatch)
        ctx.args.use_planning = True
    return agent, ctx, bus


MARKERS = ("MARK-PROFILE", "MARK-SCRATCH", "MARK-UNCERTAINTY", "MARK-AUTOSKILL", "MARK-RECALL", "MARK-PLAYBOOK",
           "MARK-BELIEF", "MARK-COMPETENCE")


def _all_prompt_text(ctx):
    out = []
    for c in ctx.llm_client.chat_completion.call_args_list:
        payload = c.args[0] if c.args and isinstance(c.args[0], dict) else c.kwargs
        for m in payload.get("messages", []) or []:
            if isinstance(m, dict):
                out.append(str(m.get("content") or ""))
    return "\n".join(out)


@pytest.mark.parametrize("planning", [False, True])
async def test_a_member_turn_carries_no_owner_marker_in_any_prompt(monkeypatch, tmp_path, planning):
    """R4 reviews: the planner playbook and the scratchpad reached member
    prompts after the first wall; this reads EVERY model call's messages."""
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path, planning)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("I don't know who you are."))
    await agent.handle_chat({"messages": [{"role": "user", "content": "check what you know about me and tell me"}]},
                            FakeBgTasks(), request_id="web-7", requester_role="member")
    text = _all_prompt_text(ctx)
    leaked = [m for m in MARKERS if m in text]
    assert leaked == [], leaked
    bus.hydrate_context.assert_not_awaited()


@pytest.mark.parametrize("planning", [False, True])
async def test_the_owner_turn_still_carries_its_data(monkeypatch, tmp_path, planning):
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path, planning)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Vasilis."))
    await agent.handle_chat({"messages": [{"role": "user", "content": "check what you know about me and tell me"}]},
                            FakeBgTasks(), request_id="web-8", requester_role="owner")
    text = _all_prompt_text(ctx)
    for m in ("MARK-PROFILE", "MARK-SCRATCH", "MARK-RECALL", "MARK-BELIEF"):
        assert m in text, m
    if planning:
        assert "MARK-PLAYBOOK" in text


async def test_no_header_behaves_identically_on_any_request_id(monkeypatch, tmp_path):
    """The behavioural form of "no client name keys a rule": the same turn
    with NO role under a `slack-` id and under a plain id has the same
    observables — profile in the prompt, recall hydrated, a walled tool
    reached, a smart-memory row, a teachable trajectory."""
    from ghost_agent.memory.skills import trajectory_may_teach
    seen = []
    for rid in ("slack-abcdef01", "abcdef02"):
        agent, ctx, bus = _marked_agent(monkeypatch, tmp_path / rid, planning=False)
        _, _, appended = None, None, None
        called = AsyncMock(return_value="TOOL RESULT")
        agent.available_tools = {"introspect": called}
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("", [_tc("c0", "introspect", {"action": "overview"})]), _resp("Done."), _resp("Done.")])
        rows = []

        async def _append(kind, payload, rows=rows):
            rows.append(kind)
        agent._journal_append_safe = _append
        await agent.handle_chat({"messages": [{"role": "user", "content": "check what you know about me and tell me"}]},
                                FakeBgTasks(), request_id=rid)
        trajs = list(ctx.trajectory_collector.iter_trajectories())
        seen.append((
            "MARK-PROFILE" in _all_prompt_text(ctx),
            bus.hydrate_context.await_count > 0,
            called.await_count > 0,
            "smart_memory" in rows,
            bool(trajs) and trajectory_may_teach(trajs[-1]),
        ))
    assert seen[0] == seen[1] == (True, True, True, True, True), seen


async def test_a_member_cannot_run_the_owners_dream_by_typing_it(monkeypatch, tmp_path):
    """R4 review: the deterministic terminal-command shortcut ran dream_mode
    for a member ("dream mode") without asking the model."""
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    dream = AsyncMock(return_value="Dream cycle complete.")
    agent.available_tools = {"dream_mode": dream}
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Not available here."))
    await agent.handle_chat({"messages": [{"role": "user", "content": "dream mode"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    dream.assert_not_awaited()


async def test_a_member_turn_runs_with_no_project_and_keeps_the_owners_binding(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.current_project_id = "p-owner"
    import ghost_agent.tools.projects as pj
    called = []
    monkeypatch.setattr(pj, "reconcile_conversation", lambda *a, **k: called.append(1))
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "check project p-owner and tell me"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert called == [] and ctx.current_project_id == "p-owner"   # restored after the member turn (R6)


async def test_a_members_turn_is_not_written_into_the_owners_episodic_memory(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ep = MagicMock()
    ctx.episodic_memory = ep
    t1 = requester_role_context.set("member")
    try:
        await agent._record_episode_safe("u", [], "a")
    finally:
        requester_role_context.reset(t1)
    assert not ep.method_calls


def test_the_conversation_identity_includes_the_role():
    """R4 review: a member whose first message matched the owner's ("hi")
    shared the owner's queued corrections and costly-turn record."""
    agent = GhostAgent(make_context())
    msgs = [{"role": "user", "content": "hi"}]
    owner_fp = agent._conversation_fingerprint(msgs)
    t1 = requester_role_context.set("member")
    try:
        member_fp = agent._conversation_fingerprint(msgs)
    finally:
        requester_role_context.reset(t1)
    assert owner_fp and member_fp and owner_fp != member_fp



# ── round 6: argument keys, the owner's scope, listings, catalogue, feedback, upload ──

@pytest.mark.parametrize("args,allowed", [
    ({"action": "describe_picture", "target": " https://x/../../owner.pdf"}, False),     # leading space
    ({"action": "describe_picture", "target": "\thttps://x/owner.png"}, False),          # leading tab
    ({"action": "describe_picture", "target": "https://x/../../owner.pdf"}, False),      # traversal
    ({"action": "describe_picture", "target": "gen_member1.png "}, False),               # trailing space
    ({"action": "describe_picture", "target": "sub/gen_member1.png"}, False),            # a path, not the name
    ({"action": "describe_picture", "target": "gen_member1.png", "path": "owner.png"}, False),   # an extra key
    ({"action": "extract_text_pdf", "target": "https://example.com/doc.pdf"}, True),
    ({"action": "describe_picture", "target": "gen_member1.png"}, True),
])
async def test_a_members_vision_arguments_are_allowlisted(monkeypatch, tmp_path, args, allowed):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent._note_member_files({"gen_member1.png"})
    called = AsyncMock(return_value="VISION ANALYSIS RESULT: x")
    agent.available_tools = {"vision_analysis": called}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "vision_analysis", args)]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "check this image"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert called.await_count == (1 if allowed else 0), args


@pytest.mark.parametrize("args,allowed", [
    ({"prompt": "x", "image_path": "projects/p1/owner.jpg"}, False),        # synonym keys (R6)
    ({"prompt": "x", "source_image": "owner.jpg"}, False),
    ({"prompt": "x", "references": ["owner.jpg"]}, False),
    ({"prompt": "x", "reference_images": ["https://x/../../owner.jpg"]}, False),
    ({"prompt": "x", "reference_images": [{"path": "owner.jpg"}]}, False),
    ({"prompt": "x", "reference_images": ["gen_member1.png"]}, True),
    ({"prompt": "a cat", "width": 512, "height": 512, "seed": 3}, True),
])
async def test_a_members_image_arguments_are_allowlisted(monkeypatch, tmp_path, args, allowed):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent._note_member_files({"gen_member1.png"})
    called = AsyncMock(return_value="SUCCESS: Image generated.")
    agent.available_tools = {"image_generation": called}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", args)]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "make an image"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert called.await_count == (1 if allowed else 0), args


async def test_a_member_turn_runs_outside_the_owners_scope_and_restores_it(monkeypatch, tmp_path):
    """R6 (CRIT): a stale conversation_key let project-scoped paths fall back
    to the owner's bound project on a member turn. In-turn: no project, no
    conversation; after the turn: the owner's scope is back."""
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.current_project_id = "p-owner"
    ctx.conversation_key = "owner-conv"
    seen = {}

    async def _img(**kw):
        seen["pid"] = ctx.current_project_id
        seen["conv"] = ctx.conversation_key
        seen["captured"] = agent._captured_project_id()
        return "SUCCESS: Image generated."
    agent.available_tools = {"image_generation": AsyncMock(side_effect=_img)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat"})]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "make an image of a cat"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert seen == {"pid": None, "conv": "", "captured": None}, seen
    assert ctx.current_project_id == "p-owner" and ctx.conversation_key == "owner-conv"


def test_a_members_refute_files_no_task_on_the_owners_project(monkeypatch, tmp_path):
    agent = GhostAgent(make_context())
    store = MagicMock()
    agent.context.project_store = store
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    v = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                     issues=["next step: run deploy_prod.sh again"])
    t1 = requester_role_context.set("member")
    try:
        agent._file_refute_followup_tasks(v, "p-owner")
    finally:
        requester_role_context.reset(t1)
    assert not store.method_calls


@pytest.mark.parametrize("ask", ["write a python script that prints fibonacci, list your files first",
                                 "check the weather in Athens and tell me"])
async def test_a_member_never_sees_the_sandbox_listing(monkeypatch, tmp_path, ask):
    import ghost_agent.tools.file_system as fsmod
    listed = []

    async def _list(*a, **k):
        listed.append(1)
        return "OWNER_TAXES_2025.pdf  transfer_to_iban_GR123.py"
    monkeypatch.setattr(fsmod, "tool_list_files", _list)
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert "OWNER_TAXES" not in _all_prompt_text(ctx)
    assert listed == []


def test_a_member_is_advertised_only_the_allowlist():
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS
    agent = GhostAgent(make_context())
    from ghost_agent.tools.registry import get_available_tools
    agent.available_tools = get_available_tools(agent.context)
    t1 = requester_role_context.set("member")
    try:
        rs = GhostAgent._RequestState(agent)
        names = {((d or {}).get("function") or {}).get("name") for d in rs.get_active_tool_defs("anything")}
    finally:
        requester_role_context.reset(t1)
    assert names and names <= set(_MEMBER_ALLOWED_TOOLS), names - set(_MEMBER_ALLOWED_TOOLS)


def test_a_thumb_on_a_members_turn_is_not_a_label():
    from ghost_agent.core import feedback as fb
    from ghost_agent.distill.schema import Trajectory
    agent = MagicMock()
    traj = Trajectory(session_id="slack-1", extra={"requester_role": "member"})
    import unittest.mock as um
    with um.patch.object(fb, "find_trajectory_for_request", return_value=traj):
        out = fb.apply_human_label(agent, "slack-1", "positive", source="slack:U1")
    assert out.get("ok") is False and out.get("code") == "member_turn", out


def test_a_members_upload_never_overwrites_an_owner_file(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    (tmp_path / "PROJECT_MAP.md").write_text("OWNER MAP")
    agent = GhostAgent(make_context())
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path
    agent.context.current_project_id = None
    app = FastAPI(); app.include_router(routes.router); app.state.agent = agent; app.state.args = MagicMock()
    import os
    h = {"X-Ghost-Key": os.environ.get("GHOST_API_KEY", "")}
    with TestClient(app) as c:
        r = c.post("/api/upload", headers={**h, "X-Ghost-Requester": "member"},
                   files={"file": ("PROJECT_MAP.md", b"MEMBER CONTENT")})
        assert r.status_code == 200
        name = r.json()["filename"]
    assert (tmp_path / "PROJECT_MAP.md").read_text() == "OWNER MAP"
    assert name.startswith("mu_") and name.endswith("_PROJECT_MAP.md")
    assert (tmp_path / name).read_bytes() == b"MEMBER CONTENT"
    assert name in agent._member_files()


def test_the_bot_names_the_uploader_on_every_upload():
    """AST: the upload POST carries X-Ghost-Requester built from the
    uploader, and both call sites pass who uploaded."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "upload_file_to_agent")
    dicts = [d for d in ast.walk(fn) if isinstance(d, ast.Dict)]
    assert any(isinstance(k, ast.Constant) and k.value == "X-Ghost-Requester"
               and isinstance(v, ast.Call) and _is_name(v.func, "requester_role_header")
               and _is_name(v.args[0], "uploader") for d in dicts for k, v in zip(d.keys, d.values))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_name(n.func, "upload_file_to_agent")]
    assert calls and all(any(k.arg == "uploader" for k in c.keywords) for c in calls)



async def test_the_ambient_sandbox_state_is_withheld_from_a_member(monkeypatch):
    import ghost_agent.tools.file_system as fsmod
    listed = []

    async def _list(*a, **k):
        listed.append(1)
        return "OWNER_TAXES_2025.pdf"
    monkeypatch.setattr(fsmod, "tool_list_files", _list)
    agent = GhostAgent(make_context())
    t1 = requester_role_context.set("member")
    try:
        state = await GhostAgent._RequestState(agent).get_sandbox_state()
    finally:
        requester_role_context.reset(t1)
    assert "OWNER_TAXES" not in str(state) and listed == []
    owner_state = await GhostAgent._RequestState(agent).get_sandbox_state()
    assert "OWNER_TAXES" in str(owner_state)


async def test_a_member_turn_whose_tool_fails_gets_no_diagnostic_listing(monkeypatch, tmp_path):
    """The post-failure diagnostic listed the sandbox (owner filenames and
    signatures) on ANY failed tool, no coding intent needed."""
    import ghost_agent.tools.file_system as fsmod
    listed = []

    async def _list(*a, **k):
        listed.append(1)
        return "OWNER_TAXES_2025.pdf  def transfer_to_iban_GR123"
    monkeypatch.setattr(fsmod, "tool_list_files", _list)
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(side_effect=RuntimeError("search backend down"))}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "x"})])] + [_resp("Sorry, search failed.")] * 6)
    await agent.handle_chat({"messages": [{"role": "user", "content": "search for x and tell me"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert "OWNER_TAXES" not in _all_prompt_text(ctx) and listed == []



# ── round 7: encoded traversal, the owner's digests, notifications, constraints, calibration, training ──

@pytest.mark.parametrize("ref", ["https://x/%2e%2e/%2e%2e/projects/p1/photo.jpg", "https://example.com/cat.png",
                                 "gen_member1%2epng"])
async def test_a_member_image_reference_is_never_a_url_or_encoded(monkeypatch, tmp_path, ref):
    """R7 (CRIT): image_generation cannot fetch a URL — it resolves every
    reference as a sandbox path and percent-decodes it."""
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent._note_member_files({"gen_member1.png"})
    called = AsyncMock(return_value="SUCCESS: Image generated.")
    agent.available_tools = {"image_generation": called}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "watercolor", "reference_images": [ref]})]),
        _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "restyle this"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    called.assert_not_awaited()


async def _digest_turn(monkeypatch, tmp_path, role):
    import ghost_agent.core.project_digest as pd
    import ghost_agent.core.autonomous_activity as aa
    touched = []

    class _Dg:
        has_content = True; advanced = 1; needs_user = []; projects_touched = 1; new_event_id = 9
    monkeypatch.setattr(pd, "load_watermark", lambda p: 1)
    monkeypatch.setattr(pd, "summarize_since", lambda *a, **k: _Dg())
    monkeypatch.setattr(pd, "render_digest", lambda d: "OWNER-PROJECT-DIGEST while you were away")
    monkeypatch.setattr(pd, "save_watermark", lambda *a, **k: touched.append("proj-wm"))
    alog = MagicMock()
    alog.read_since.return_value = (["rec"], 5)
    monkeypatch.setattr(aa, "get_activity_log", lambda ctx: alog)
    monkeypatch.setattr(aa, "load_offset", lambda *a, **k: 1)
    monkeypatch.setattr(aa, "render_activity_digest", lambda *a, **k: "OWNER-ACTIVITY-DIGEST job finished")
    monkeypatch.setattr(aa, "save_offset", lambda *a, **k: touched.append("act-wm"))
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.project_store = MagicMock()
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    ctx.memory_dir = str(tmp_path / "memory")
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Paris."))
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "what is the capital of France?"}]},
                                        FakeBgTasks(), request_id="web-1", requester_role=role)
    return out, touched


async def test_the_owners_digests_never_reach_a_members_reply(monkeypatch, tmp_path):
    """R7 (CRIT): the "while you were away" project digest and the activity
    digest were prepended to a member's reply and their watermarks consumed.
    The owner control proves the fixture reaches both digests."""
    out, touched = await _digest_turn(monkeypatch, tmp_path, "owner")
    assert "OWNER-PROJECT-DIGEST" in out and "proj-wm" in touched
    assert "OWNER-ACTIVITY-DIGEST" in out and "act-wm" in touched
    out_m, touched_m = await _digest_turn(monkeypatch, tmp_path / "m", "member")
    assert "OWNER-PROJECT-DIGEST" not in out_m and "OWNER-ACTIVITY-DIGEST" not in out_m
    assert touched_m == []


def test_a_member_never_writes_the_owners_notification_feed():
    from ghost_agent.core.agent import _notify_promise_backstop
    import ghost_agent.core.autonomous_activity as aa
    import unittest.mock as um
    ask = "research X and notify me when you're done"
    for role, expect in (("owner", True), ("member", False)):
        ctx = make_context()
        log = MagicMock(); log.record.return_value = True
        with um.patch.object(aa, "get_activity_log", lambda c, _l=log: _l):
            t1 = requester_role_context.set(role)
            try:
                fired = _notify_promise_backstop(ctx, last_user_content=ask, tools_run=[],
                                                 final_content="Done — X", req_id=f"web-{role}", had_failures=False)
            finally:
                requester_role_context.reset(t1)
        assert fired is expect, role
        assert log.record.called is expect, role


def test_a_member_request_never_pulls_the_owners_project_constraints():
    agent = GhostAgent(make_context())
    agent._active_project_constraints = lambda **k: ["OWNER RULE A"]
    agent._project_constraints_for = lambda pid: ["OWNER RULE B"]
    t1 = requester_role_context.set("member")
    try:
        merged, block, _ = agent._merge_project_constraints(["mine"], user_text="what rules apply to projects/a1b2c3d4/?")
    finally:
        requester_role_context.reset(t1)
    assert merged == ["mine"] and "OWNER RULE" not in block
    merged2, _, _ = agent._merge_project_constraints(["mine"], user_text="projects/a1b2c3d4/")
    assert "OWNER RULE A" in merged2 and "OWNER RULE B" in merged2


async def test_a_members_turn_is_not_a_calibration_sample(monkeypatch):
    """The first step past the gate reads the turn's population; a member's
    turn never gets there (the owner control proves the fixture does)."""
    import ghost_agent.core.agent as agent_mod
    seen = []
    monkeypatch.setattr(agent_mod, "turn_origin", lambda ctx: (seen.append(1), "probe")[1])
    agent = GhostAgent(make_context())
    for role, expect in (("owner", 1), ("member", 0)):
        seen.clear()
        t1 = requester_role_context.set(role)
        try:
            await agent._record_calibration_safe(req_id="web-1", tools_run=[], verifier_backfill=None,
                                                 execution_failure_count=0, budget_exhausted=False,
                                                 final_ai_content="x")
        finally:
            requester_role_context.reset(t1)
        assert len(seen) == expect, role


def test_member_turns_never_tune_tool_descriptions(tmp_path):
    from ghost_agent.optim.tool_fixtures import _trajectory_index
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory
    col = TrajectoryCollector(root=tmp_path)
    col.append(Trajectory(session_id="m1", user_request="x", extra={"requester_role": "member"}))
    col.append(Trajectory(session_id="o1", user_request="y"))
    idx = _trajectory_index(tmp_path)
    assert "o1" in idx and "m1" not in idx



# ── round 7b: planner catalogue, alias image rows, queued-call counting, persistence, upload root, scratchpad scope ──

async def test_a_members_planner_sees_only_the_allowlist(monkeypatch, tmp_path):
    agent, ctx, _ = _marked_agent(monkeypatch, tmp_path, planning=True)
    from ghost_agent.tools.registry import get_available_tools
    agent.available_tools = get_available_tools(ctx)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "check the weather in Athens and tell me"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    planner = [c for c in ctx.llm_client.chat_completion.call_args_list if c.kwargs.get("task_label") == "planner"]
    assert planner, "the planner must have run on this arm"
    payload = planner[0].args[0] if planner[0].args and isinstance(planner[0].args[0], dict) else planner[0].kwargs
    text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    for owner_tool in ("file_system (", "execute (", "manage_projects (", "introspect ("):
        assert owner_tool not in text, owner_tool
    assert "web_search (" in text


def test_an_image_made_through_an_alias_is_still_counted():
    from ghost_agent.core.agent import _turn_generated_images
    row = {"name": "imagegen", "content": "SUCCESS: Image generated and saved.\n\n![generated image](/api/download/gen_z.png)"}
    assert _turn_generated_images([row]) == ["gen_z.png"]
    assert _turn_generated_images([{"name": "image_generation", "content": "SUCCESS: Wrote 5 chars ![generated image](/api/download/x.png)"}]) == []


async def test_a_rejected_image_call_does_not_block_the_next_in_the_batch(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    gen = AsyncMock(return_value="SUCCESS: Image generated.\n\n![generated image](/api/download/gen_q.png)")
    agent.available_tools = {"image_generation": gen}
    bad = {"id": "c0", "type": "function", "function": {"name": "image_generation", "arguments": "{not json"}}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [bad, _tc("c1", "image_generation", {"prompt": "a cat"})]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "make an image of a cat"}]},
                            FakeBgTasks(), request_id="web-1")
    assert gen.await_count == 1


def test_member_files_survive_a_restart_and_trim_oldest_first(tmp_path):
    ctx = make_context()
    ctx.memory_dir = str(tmp_path / "memory")
    (tmp_path / "memory").mkdir()
    a = GhostAgent(ctx)
    a._MEMBER_FILES_MAX = 3
    a._note_member_files(["m1.png"]); a._note_member_files(["m2.png"])
    a._note_member_files(["m3.png"]); a._note_member_files(["m4.png"])
    assert a._member_files() == {"m2.png", "m3.png", "m4.png"}
    b = GhostAgent(ctx)                          # a fresh process on the same data dir
    assert b._member_files() == {"m2.png", "m3.png", "m4.png"}


def test_a_members_upload_lands_at_the_root_even_with_a_project_bound(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    (tmp_path / "projects" / "p1").mkdir(parents=True)
    agent = GhostAgent(make_context())
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path
    agent.context.current_project_id = "p1"
    app = FastAPI(); app.include_router(routes.router); app.state.agent = agent; app.state.args = MagicMock()
    import os
    h = {"X-Ghost-Key": os.environ.get("GHOST_API_KEY", ""), "X-Ghost-Requester": "member"}
    with TestClient(app) as c:
        name = c.post("/api/upload", headers=h, files={"file": ("cat.png", b"x")}).json()["filename"]
    assert (tmp_path / name).exists() and not list((tmp_path / "projects" / "p1").iterdir())


async def test_the_owners_scratchpad_scope_is_restored_after_a_member_turn(monkeypatch, tmp_path):
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    sp = MagicMock(); sp.list_all.return_value = ""; sp.get.return_value = None
    sp.active_namespace = "p-owner"
    ctx.scratchpad = sp
    ctx.current_project_id = "p-owner"
    seen = {}

    async def _img(**kw):
        seen["ns"] = sp.active_namespace
        return "SUCCESS: Image generated."
    agent.available_tools = {"image_generation": AsyncMock(side_effect=_img)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "image_generation", {"prompt": "a cat"})]), _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "make an image of a cat"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member")
    assert seen["ns"] is None and sp.active_namespace == "p-owner"



# ── round 8: whole subsystems a member turn never enters ──

async def _verified_turn(monkeypatch, tmp_path, role):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ver = MagicMock(); ver.llm_client = MagicMock()
    judge = AsyncMock(return_value=VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="r", issues=[]))
    ver.verify_claim = judge; ver.verify_code_output = judge; ver.verify_visual = AsyncMock(return_value=None)
    ctx.verifier = ver
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. r\nthe answer is 42\n")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "x"})]), _resp("It is 42 — see passport.jpg."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "search for x and show me the screenshot"}]},
                            FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
    return judge, ver.verify_visual


async def test_a_members_turn_is_never_verified(monkeypatch, tmp_path):
    """R8 (CRIT ×2): the verifier's arms read the owner's world (images
    resolved across the sandbox, the profile) and its issues reach the
    member verbatim."""
    judge, visual = await _verified_turn(monkeypatch, tmp_path, "owner")
    assert judge.await_count >= 1
    judge_m, visual_m = await _verified_turn(monkeypatch, tmp_path / "m", "member")
    judge_m.assert_not_awaited(); visual_m.assert_not_awaited()


async def test_a_members_message_never_relabels_a_prior_turn(monkeypatch, tmp_path):
    """R8 (CRIT): "No, that's wrong" in an open thread relabelled the OWNER's
    turn (the correction cache is keyed by the prior reply's text)."""
    for role, expect in (("owner", True), ("member", False)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        seen = []

        async def _adj(self, *a, **k):
            seen.append(1); return None
        monkeypatch.setattr(GhostAgent, "_adjudicate_correction", _adj)
        ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Sorry."))
        await agent.handle_chat({"messages": [{"role": "user", "content": "what is 2+2?"},
                                              {"role": "assistant", "content": "5."},
                                              {"role": "user", "content": "No, that's wrong. What is 2+2?"}]},
                                FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        assert bool(seen) is expect, role


async def test_system_3_never_runs_for_a_member():
    agent = GhostAgent(make_context())
    agent.context.llm_client.chat_completion = AsyncMock(return_value=_resp('{"hypotheses": []}'))
    t1 = requester_role_context.set("member")
    try:
        out = await agent._run_system_3_pivot("cat /workspace/projects/p1/.env", "err", "", "m")
    finally:
        requester_role_context.reset(t1)
    assert out == {}
    agent.context.llm_client.chat_completion.assert_not_awaited()


async def test_a_members_hedges_are_not_written_to_the_owners_uncertainty_log(monkeypatch, tmp_path):
    for role, expect in (("owner", True), ("member", False)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        ut = MagicMock(); ut.scan_text_for_uncertainty.return_value = ["I think it might be 42"]
        ut.persisted_context.return_value = ""
        ctx.uncertainty_tracker = ut
        agent.available_tools = {"web_search": AsyncMock(return_value="r")}
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("", [_tc("c0", "web_search", {"query": "x"})]), _resp("I think it might be 42."), _resp("Done.")])
        await agent.handle_chat({"messages": [{"role": "user", "content": "search for x and tell me"}]},
                                FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        assert ut.flag_assumption.called is expect, role


def test_the_owners_thumb_ask_is_not_spent_on_a_member():
    agent = GhostAgent(make_context())
    t1 = requester_role_context.set("member")
    try:
        assert agent._label_request_note(0.01, None) == ""
    finally:
        requester_role_context.reset(t1)


async def test_a_member_turn_does_not_stream(monkeypatch, tmp_path):
    for role, streams in (("owner", True), ("member", False)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)

        async def final_stream(payload, use_coding=False):
            from tests.test_finalize_stream_pins import sse
            yield sse({"content": "Hello."}); yield b"data: [DONE]\n\n"
        ctx.llm_client.stream_chat_completion = final_stream
        _force_planning_arm(monkeypatch)
        ctx.args.use_planning = True
        import json as _json
        tree = {"id": "root", "description": "Answer", "status": "DONE", "children": []}
        ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("```json\n" + _json.dumps(
            {"thought": "Answer now.", "tree_update": tree, "next_action_id": "root", "required_tool": "none"}) + "\n```"))
        content, _, _ = await agent.handle_chat(
            {"messages": [{"role": "user", "content": "check who you are and tell me"}], "stream": True},
            FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        assert hasattr(content, "__aiter__") is streams, role
        if streams:
            async for _c in content:
                pass


async def test_a_member_turn_writes_no_scratchpad_checkpoint(monkeypatch, tmp_path):
    for role, expect in (("owner", True), ("member", False)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        sp = MagicMock(); sp.list_all.return_value = ""; sp.get.return_value = None
        ctx.scratchpad = sp
        agent.available_tools = {"web_search": AsyncMock(return_value="r")}
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("", [_tc(f"c{i}", "web_search", {"query": f"x{i}"})]) for i in range(17)] + [_resp("Done.")] * 4)
        await agent.handle_chat({"messages": [{"role": "user", "content": "search for many things and tell me"}]},
                                FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        wrote = any("checkpoint" in str(c) for c in sp.set.call_args_list)
        assert wrote is expect, role


def test_a_member_download_must_be_an_exact_member_name_and_survives_a_restart(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    (tmp_path / "memory").mkdir()
    (tmp_path / "sandbox" / "projects" / "p1").mkdir(parents=True)
    (tmp_path / "sandbox" / "gen_m1.png").write_bytes(b"member")
    (tmp_path / "sandbox" / "projects" / "p1" / "gen_m1.png").write_bytes(b"owner")
    ctx = make_context(); ctx.memory_dir = str(tmp_path / "memory")
    GhostAgent(ctx)._note_member_files({"gen_m1.png"})          # persisted by a previous process
    agent = GhostAgent(ctx)                                      # a restart: nothing loaded yet
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path / "sandbox"
    agent.context.current_project_id = None
    app = FastAPI(); app.include_router(routes.router); app.state.agent = agent; app.state.args = MagicMock()
    import os
    h = {"X-Ghost-Key": os.environ.get("GHOST_API_KEY", ""), "X-Ghost-Requester": "member"}
    with TestClient(app) as c:
        assert c.get("/api/download/gen_m1.png", headers=h).status_code == 200
        assert c.get("/api/download/projects/p1/gen_m1.png", headers=h).status_code == 404



# ── round 8b: the second reviewer's untested items ──

def _traj_pair():
    from ghost_agent.distill.schema import Trajectory
    return [Trajectory(session_id="m", extra={"requester_role": "member"}), Trajectory(session_id="o")]


@pytest.mark.parametrize("wrapper", ["tools.memory._teachable", "core.agent._iter_teachable_train"])
def test_the_training_read_wrappers_drop_member_turns(wrapper):
    import importlib
    mod, fn = wrapper.rsplit(".", 1)
    f = getattr(importlib.import_module("ghost_agent." + mod), fn)
    assert [t.session_id for t in f(_traj_pair())] == ["o"]


def test_every_trainer_corpus_read_goes_through_a_teachable_wrapper():
    """R8b MAJOR-2: the router/PRM corpus reads were exempted as "feeds a
    Trainer", so unwrapping them stayed green. Every `iter_trajectories()`
    handed to a trainer as `trajectories=` or to `bootstrap_router` must sit
    directly inside a teachable wrapper."""
    import ast, pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    wrappers = {"_teachable", "_iter_teachable_train", "_iter_teachable_boot", "iter_teachable"}
    seen = 0
    for rel in ("core/agent.py", "tools/memory.py", "main.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            fname = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            fed = [k.value for k in call.keywords if k.arg == "trajectories"]
            if (fname == "to_thread" and len(call.args) >= 2
                    and getattr(call.args[0], "id", "") == "bootstrap_router"):
                fed.append(call.args[1])                     # asyncio.to_thread(bootstrap_router, <corpus>, …)
            for v in fed:
                seen += 1
                inner = v.args[0] if isinstance(v, ast.Call) and v.args else None
                vname = getattr(v.func, "id", None) if isinstance(v, ast.Call) else None
                assert vname in wrappers, f"{rel}:{v.lineno} unwrapped trainer corpus"
                assert isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") == "iter_trajectories", \
                    f"{rel}:{v.lineno}"
    assert seen >= 5, seen


async def test_the_online_prm_step_never_trains_on_a_member_trajectory(tmp_path, monkeypatch):
    """R8b: an owner's "that's wrong" aimed at a MEMBER's reply promoted the
    member's trajectory to FAILED and handed it to the online PRM step."""
    import types
    import ghost_agent.core.agent as A
    import ghost_agent.distill.user_correction as UC
    from ghost_agent.distill.schema import Trajectory, Outcome
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.prm.scorer import PRMScorer

    class _V:
        reason, signals, is_correction = "wrong", ["wrong"], True
    monkeypatch.setattr(UC, "classify_user_correction", lambda *a, **k: _V(), raising=False)
    monkeypatch.setattr(PRMScorer, "has_model", property(lambda self: True))
    for role, expect in ((None, 1), ("member", 0)):
        col = TrajectoryCollector(root=tmp_path / str(role), session_id="fp")
        scorer = PRMScorer.__new__(PRMScorer)
        ctx = types.SimpleNamespace(trajectory_collector=col, last_user_content="", self_model=None,
                                    calibration_tracker=None, _recent_calib_for_correction=None,
                                    prm_scorer=scorer, args=types.SimpleNamespace(prm_online_update=True))
        agent = GhostAgent.__new__(GhostAgent); agent.context = ctx
        spawned = []
        monkeypatch.setattr(A._glog, "spawn_bg", lambda coro, *a, **k: (spawned.append(1), coro.close()))
        traj = Trajectory(id="prior", user_request="list python files", final_response="here are the go files: a.go",
                          outcome=Outcome.UNKNOWN.value, extra={"requester_role": role} if role else {})
        col.append(traj); agent._stash_trajectory_for_correction_lookup(traj)
        msgs = [{"role": "user", "content": "list python files"},
                {"role": "assistant", "content": traj.final_response},
                {"role": "user", "content": "no, that's wrong, python not go"}]
        agent._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])
        assert traj.outcome == Outcome.FAILED.value, role        # the label itself is not the gate
        assert len(spawned) == expect, role


def test_a_corrupt_member_file_list_is_kept_aside_not_overwritten(tmp_path):
    ctx = make_context(); ctx.memory_dir = str(tmp_path / "memory")
    (tmp_path / "memory").mkdir()
    (tmp_path / "member_files.json").write_text("{not json", encoding="utf-8")
    a = GhostAgent(ctx)
    assert a._member_files() == set()                                   # fails closed
    assert (tmp_path / "member_files.corrupt.json").exists() or (tmp_path / "member_files.json.corrupt").exists()
    a._note_member_files(["gen_new.png"])
    bad = [p for p in tmp_path.iterdir() if "corrupt" in p.name][0]
    assert bad.read_text(encoding="utf-8") == "{not json"


def test_required_tool_null_is_normalised_on_both_plan_reads():
    """R8b: both reads of the planner's `required_tool` normalise null to
    "all" (a None built the playbook query "Tool: None - …")."""
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "src/ghost_agent/core/agent.py").read_text(encoding="utf-8")
    hits = 0
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "required_tool" for t in node.targets)
                and isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or)):
            first = node.value.values[0]
            if (isinstance(first, ast.Call) and getattr(first.func, "attr", "") == "get"
                    and first.args and getattr(first.args[0], "value", None) == "required_tool"):
                last = node.value.values[-1]
                assert getattr(last, "value", None) == "all", ast.unparse(node)
                hits += 1
    assert hits >= 2, hits



# ── round 9 ──

async def test_a_members_tool_outcomes_never_move_the_owners_competence(monkeypatch, tmp_path):
    """R9 MAJOR: failing member searches lowered the owner's web-domain prior."""
    for role, expect in (("owner", True), ("member", False)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        mc = MagicMock(); mc.competence.get_context_string.return_value = ""
        mc.competence.by_domain.return_value = {}
        ctx.metacog = mc
        agent.available_tools = {"web_search": AsyncMock(return_value="ERROR: search failed")}
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("", [_tc("c0", "web_search", {"query": "x"})]), _resp("Done."), _resp("Done.")])
        await agent.handle_chat({"messages": [{"role": "user", "content": "search for x and tell me"}]},
                                FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        assert mc.record_outcome.called is expect, role


async def test_a_members_calls_never_enter_the_foresight_ledger(monkeypatch, tmp_path):
    import ghost_agent.core.foresight as F
    for role, expect in (("owner", True), ("member", False)):
        seen = []
        monkeypatch.setattr(F, "predict_for_call", lambda *a, **k: seen.append(1))
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        agent.available_tools = {"web_search": AsyncMock(return_value="r")}
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("", [_tc("c0", "web_search", {"query": "x"})]), _resp("Done."), _resp("Done.")])
        await agent.handle_chat({"messages": [{"role": "user", "content": "search for x and tell me"}]},
                                FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        assert bool(seen) is expect, role


async def test_the_streamed_hedge_gate_holds_if_a_member_ever_streams(monkeypatch, tmp_path):
    """R9: members do not stream today, so the drain's hedge gate is defence
    in depth. Flip the role AFTER handle_chat has chosen to stream: the drain
    must still refuse to write the member's hedges. Control: never flipped."""
    import ghost_agent.core.agent as A
    from tests.test_finalize_stream_pins import sse
    import json as _json
    for flip, expect in ((False, True), (True, False)):
        _force_planning_arm(monkeypatch)
        agent, ctx, _ = _agent(monkeypatch, tmp_path / str(flip))
        ctx.args.use_planning = True
        ut = MagicMock(); ut.scan_text_for_uncertainty.return_value = ["I think it might be 42"]
        ut.persisted_context.return_value = ""
        ctx.uncertainty_tracker = ut
        mc = MagicMock(); mc.enabled = True                    # the drain's confidence block runs under metacog
        mc.competence.estimate.return_value = 0.5; mc.competence.observations.return_value = 10
        mc.competence.get_context_string.return_value = ""; mc.competence.by_domain.return_value = {}
        ctx.metacog = mc
        tree = {"id": "root", "description": "Answer", "status": "DONE", "children": []}
        ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("```json\n" + _json.dumps(
            {"thought": "Answer now.", "tree_update": tree, "next_action_id": "root", "required_tool": "none"}) + "\n```"))

        async def final_stream(payload, use_coding=False):
            yield sse({"content": "I think it might be 42."}); yield b"data: [DONE]\n\n"
        ctx.llm_client.stream_chat_completion = final_stream
        state = {"member": False}
        monkeypatch.setattr(A, "requester_is_member", lambda: state["member"])
        content, _, _ = await agent.handle_chat(
            {"messages": [{"role": "user", "content": "check who you are and tell me"}], "stream": True},
            FakeBgTasks(), request_id=f"web-{flip}", requester_role="owner")
        assert hasattr(content, "__aiter__")
        state["member"] = flip
        async for _c in content:
            pass
        assert ut.flag_assumption.called is expect, flip


def _download_app(tmp_path, agent):
    from fastapi import FastAPI
    from ghost_agent.api import routes
    app = FastAPI(); app.include_router(routes.router); app.state.agent = agent; app.state.args = MagicMock()
    return app


def test_a_member_download_never_falls_back_to_an_owner_project(tmp_path):
    from fastapi.testclient import TestClient
    import os
    (tmp_path / "memory").mkdir()
    (tmp_path / "sandbox" / "projects" / "p1").mkdir(parents=True)
    (tmp_path / "sandbox" / "projects" / "p1" / "gen_m1.png").write_bytes(b"owner")   # swept from root; owner's twin
    ctx = make_context(); ctx.memory_dir = str(tmp_path / "memory")
    agent = GhostAgent(ctx); agent._note_member_files({"gen_m1.png"})
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path / "sandbox"
    agent.context.current_project_id = "p1"
    key = os.environ.get("GHOST_API_KEY", "")
    with TestClient(_download_app(tmp_path, agent)) as c:
        assert c.get("/api/download/gen_m1.png", headers={"X-Ghost-Key": key, "X-Ghost-Requester": "member"}).status_code == 404
        assert c.get("/api/download/gen_m1.png", headers={"X-Ghost-Key": key}).status_code == 200   # the owner's fallback stays


# ── round 10 ──

def _as_member(fn):
    t = requester_role_context.set("member")
    try:
        return fn()
    finally:
        requester_role_context.reset(t)


def test_a_members_turn_is_not_rubric_shadowed(monkeypatch):
    import types
    import ghost_agent.core.rubric_grader as RG
    from ghost_agent.core.agent import rubric_shadow_eligible
    monkeypatch.setattr(RG, "shadow_enabled", lambda: True)
    traj = types.SimpleNamespace(task_kind="user_request", tool_calls=[], outcome="unknown",
                                 user_request="hi", final_response="hello")
    ctx = types.SimpleNamespace()
    assert rubric_shadow_eligible(ctx, traj) is True
    assert _as_member(lambda: rubric_shadow_eligible(ctx, traj)) is False


async def test_the_journal_writer_refuses_a_members_turn():
    agent = GhostAgent(make_context())
    j = MagicMock(); agent.context.journal = j
    t = requester_role_context.set("member")
    try:
        await agent._journal_append_safe("smart_memory", {"x": 1})
    finally:
        requester_role_context.reset(t)
    j.append.assert_not_called()
    await agent._journal_append_safe("smart_memory", {"x": 1})
    j.append.assert_called_once()


def test_the_request_header_line_marks_a_members_turn(monkeypatch):
    import ghost_agent.utils.logging as L
    lines = []
    monkeypatch.setattr(L, "_mirror", lambda rid, title, content, *a, **k: lines.append(content))
    monkeypatch.setattr(L, "atomic_print", lambda *a, **k: None)
    L.pretty_log("Request", special_marker="BEGIN", origin="user")
    _as_member(lambda: L.pretty_log("Request", special_marker="BEGIN", origin="user"))
    assert "role=member" not in lines[0] and lines[1].endswith("origin=user role=member"), lines


def test_liveness_never_counts_a_members_turn_as_the_owners(tmp_path):
    import time
    from ghost_agent.core.liveness import _count_user_turns
    (tmp_path / "system").mkdir()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    (tmp_path / "system" / "ghost-agent.log").write_text(
        f"{ts} INFO [x] request started: a at 1 origin=user\n"
        f"{ts} INFO [x] request started: b at 1 origin=user role=member\n", encoding="utf-8")
    n, total, _ = _count_user_turns(tmp_path, 24.0)
    assert (n, total) == (1, 2)


async def test_a_members_pasted_image_is_their_own_file(monkeypatch, tmp_path):
    """R10: a member's inline `data:` image was saved but never recorded as a
    member file, so their own vision_analysis on it was refused."""
    import base64
    for role, expect in (("owner", False), ("member", True)):
        agent, ctx, _ = _agent(monkeypatch, tmp_path / role)
        ctx.memory_dir = str(tmp_path / role / "memory"); (tmp_path / role / "memory").mkdir(parents=True, exist_ok=True)
        ctx.sandbox_dir = tmp_path / role / "sandbox"; ctx.sandbox_dir.mkdir()        # never the shared /tmp/sandbox (R11)
        ctx.current_project_id = "ownerproj1"                                          # a member's file still lands at the root
        ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Nice picture."))
        url = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()
        await agent.handle_chat({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is in this picture?"},
            {"type": "image_url", "image_url": {"url": url}}]}]},
            FakeBgTasks(), request_id=f"web-{role}", requester_role=role)
        names = {n for n in agent._member_files() if n.startswith("vision_")}
        assert bool(names) is expect, (role, names)
        if expect:
            assert all((ctx.sandbox_dir / n).is_file() for n in names), list(ctx.sandbox_dir.rglob("*"))


async def test_a_repeated_pasted_image_is_one_file_not_one_per_model_call(monkeypatch, tmp_path):
    """R11: the data-URL translation runs on every model call and thread
    history re-sends images; random names wrote and registered a new copy
    each time, evicting the member's own older files from the allowlist."""
    import base64
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.memory_dir = str(tmp_path / "memory"); (tmp_path / "memory").mkdir()
    ctx.sandbox_dir = tmp_path / "sandbox"; ctx.sandbox_dir.mkdir()
    agent.available_tools = {"web_search": AsyncMock(return_value="r")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "x"})]), _resp("", [_tc("c1", "web_search", {"query": "y"})]),
        _resp("Done."), _resp("Done.")])
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG same").decode()
    img = {"type": "image_url", "image_url": {"url": url}}
    await agent.handle_chat({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "earlier"}, img]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "search for this and tell me"}, img]}]},
        FakeBgTasks(), request_id="web-m", requester_role="member")
    assert ctx.llm_client.chat_completion.await_count >= 2
    assert len([n for n in agent._member_files() if n.startswith("vision_")]) == 1
    assert len(list(ctx.sandbox_dir.rglob("vision_*"))) == 1
