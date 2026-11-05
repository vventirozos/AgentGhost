"""§4QG (2026-10-21): member files are keyed by WHICH member owns them.

The member file allowlist (`member_files.json`) was one flat set shared by
every Slack member, so any member could download — or caption — another
member's generated image, a DM's included, by its name. The bot now sends
`X-Ghost-Member: <Slack user id>` beside `X-Ghost-Requester: member` on chat,
upload and download; the agent records each member file with its owner(s);
a member reaches only their own. No member id, or an entry recorded before
§4QG (no owner), → denied to every member. The owner is unchanged."""
import ast
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.utils.logging import (parse_requester_member, requester_member_context, requester_member_id,
                                       requester_role_context)
from tests.helpers import FakeBgTasks, make_context

REPO = Path(__file__).resolve().parents[1]
BOT_PATH = REPO / "interface" / "externals" / "slack_bot" / "main.py"
KEY = os.environ.get("GHOST_API_KEY", "")


def _as(role, member):
    t1 = requester_role_context.set(role)
    t2 = requester_member_context.set(member)
    return lambda: (requester_member_context.reset(t2), requester_role_context.reset(t1))


# ── the header and the predicate ────────────────────────────────────────────
@pytest.mark.parametrize("raw,expect", [
    ("U012AB3CD", "U012AB3CD"), (" W0X ", "W0X"), ("", ""), (None, ""), ("U1 U2", ""),
    ("../U1", ""), ("U1\n", "U1"), ("x" * 65, ""), ("-U1", ""), (42, "42"),
])
def test_parse_requester_member(raw, expect):
    assert parse_requester_member(raw) == expect


@pytest.mark.parametrize("role,member,expect", [
    ("member", "U1", "U1"), ("member", "", ""), ("owner", "U1", ""), ("", "U1", ""),
])
def test_a_member_id_counts_only_on_a_member_request(role, member, expect):
    undo = _as(role, member)
    try:
        assert requester_member_id() == expect
    finally:
        undo()


# ── the store ───────────────────────────────────────────────────────────────
def _store_agent(tmp_path):
    (tmp_path / "memory").mkdir(exist_ok=True)
    ctx = make_context(); ctx.memory_dir = str(tmp_path / "memory")
    return GhostAgent(ctx), ctx


def test_a_member_file_belongs_to_its_member_only(tmp_path):
    a, _ = _store_agent(tmp_path)
    a._note_member_files({"gen_a.png"}, member="U1")
    a._note_member_files({"gen_b.png"}, member="U2")
    assert a._member_files(owner="U1") == {"gen_a.png"}
    assert a._member_files(owner="U2") == {"gen_b.png"}
    assert a._member_files(owner="") == set()                  # an unknown member owns nothing
    assert a._member_files() == {"gen_a.png", "gen_b.png"}      # the registry itself


def test_the_owner_comes_from_the_request_by_default(tmp_path):
    a, _ = _store_agent(tmp_path)
    undo = _as("member", "U1")
    try:
        a._note_member_files({"gen_a.png"})
    finally:
        undo()
    undo = _as("member", "")                                    # a member that did not say who
    try:
        a._note_member_files({"gen_x.png"})
    finally:
        undo()
    assert a._member_files(owner="U1") == {"gen_a.png"}
    assert "gen_x.png" in a._member_files() and not any(
        "gen_x.png" in a._member_files(owner=m) for m in ("U1", "", "U2"))


def test_owners_survive_a_restart_and_a_pasted_image_is_co_owned(tmp_path):
    a, ctx = _store_agent(tmp_path)
    a._note_member_files({"vision_abc.png"}, member="U1")
    a._note_member_files({"vision_abc.png"}, member="U2")       # the same bytes pasted by U2 (content name, R11)
    b = GhostAgent(ctx)
    assert b._member_files(owner="U1") == {"vision_abc.png"} == b._member_files(owner="U2")
    assert b._member_files(owner="U3") == set()


def test_a_legacy_entry_has_no_owner_and_no_member_reaches_it(tmp_path):
    a, ctx = _store_agent(tmp_path)
    (tmp_path / "member_files.json").write_text(json.dumps({"gen_old.png": 1700000000.0}), encoding="utf-8")
    assert a._member_files() == {"gen_old.png"}
    assert a._member_files(owner="U1") == set()
    a._note_member_files({"gen_new.png"}, member="U1")            # writing keeps the legacy row unowned
    assert GhostAgent(ctx)._member_files(owner="U1") == {"gen_new.png"}
    a._note_member_files({"gen_old.png"}, member="U2")            # U2's own turn produced those bytes again
    assert a._member_files(owner="U2") == {"gen_old.png"}
    assert a._member_files(owner="U1") == {"gen_new.png"}


def test_a_set_seeded_registry_is_owner_unknown(tmp_path):
    a, _ = _store_agent(tmp_path)
    a._member_generated_files = {"gen_s.png"}
    assert a._member_files(owner="U1") == set() and a._member_files() == {"gen_s.png"}


# ── the routes ──────────────────────────────────────────────────────────────
def _app(tmp_path, agent):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    agent.context.args.api_key = None
    agent.context.sandbox_dir = tmp_path / "sandbox"
    agent.context.current_project_id = None
    app = FastAPI(); app.include_router(routes.router); app.state.agent = agent; app.state.args = MagicMock()
    return TestClient(app)


def _h(member=None, role="member"):
    h = {"X-Ghost-Key": KEY}
    if role:
        h["X-Ghost-Requester"] = role
    if member is not None:
        h["X-Ghost-Member"] = member
    return h


def test_a_member_downloads_any_members_file_never_the_owners(tmp_path):
    """§4QH (operator 2026-10-21): the per-member rule is removed — members
    in a shared thread use each other's renders. The owner's files stay out."""
    a, _ = _store_agent(tmp_path)
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox" / "gen_a.png").write_bytes(b"U1's")
    (tmp_path / "sandbox" / "gen_old.png").write_bytes(b"legacy")
    (tmp_path / "sandbox" / "owner.pdf").write_bytes(b"owner's")
    a._note_member_files({"gen_a.png"}, member="U1")
    a._member_file_store()["gen_old.png"] = 1700000000.0          # a pre-§4QG row
    with _app(tmp_path, a) as c:
        assert c.get("/api/download/gen_a.png", headers=_h("U1")).status_code == 200
        assert c.get("/api/download/gen_a.png", headers=_h("U2")).status_code == 200   # another member (§4QH)
        assert c.get("/api/download/gen_a.png", headers=_h()).status_code == 200       # no member id
        assert c.get("/api/download/gen_old.png", headers=_h("U1")).status_code == 200  # legacy member row
        assert c.get("/api/download/owner.pdf", headers=_h("U1")).status_code == 404    # never the owner's
        assert c.get("/api/download/owner.pdf", headers=_h()).status_code == 404
        assert c.get("/api/download/gen_a.png", headers=_h(role=None)).status_code == 200        # the owner
        assert c.get("/api/download/gen_old.png", headers=_h(role=None)).status_code == 200
        assert c.get("/api/download/gen_a.png", headers=_h("U2", role="owner")).status_code == 200  # owner ignores it


def test_a_members_upload_is_theirs(tmp_path):
    a, _ = _store_agent(tmp_path)
    (tmp_path / "sandbox").mkdir()
    with _app(tmp_path, a) as c:
        r1 = c.post("/api/upload", headers=_h("U1"), files={"file": ("cat.png", b"\x89PNG one")})
        r0 = c.post("/api/upload", headers=_h(), files={"file": ("dog.png", b"\x89PNG two")})
        n1, n0 = r1.json()["filename"], r0.json()["filename"]
        assert a._member_files(owner="U1") == {n1}
        assert c.get(f"/api/download/{n1}", headers=_h("U1")).status_code == 200
        assert c.get(f"/api/download/{n1}", headers=_h("U2")).status_code == 200   # §4QH: any member
        assert n0 in a._member_files()                               # a member file (never the owner's)
        assert c.get(f"/api/download/{n0}", headers=_h("U1")).status_code == 200


@pytest.mark.parametrize("headers,expect", [
    ({"X-Ghost-Requester": "member", "X-Ghost-Member": "U1"}, "U1"),
    ({"X-Ghost-Requester": "member"}, ""),
    ({"X-Ghost-Requester": "owner", "X-Ghost-Member": "U1"}, ""),
    ({"X-Ghost-Member": "U1"}, ""),
])
def test_the_chat_route_names_the_member_for_the_turn(headers, expect):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes
    seen = []

    async def _hc(*a, **k):
        assert requester_member_id() == requester_member_context.get()   # an owner request never carries one
        seen.append(requester_member_context.get())
        raise RuntimeError("stop")
    fake = MagicMock(); fake.context.args.api_key = None; fake.context.args.model = "test-model"
    fake.handle_chat = AsyncMock(side_effect=_hc)
    app = FastAPI(); app.state.agent = fake; app.state.args = MagicMock(); app.state.args.model = "test-model"
    app.include_router(routes.router)
    with TestClient(app) as c:
        c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
               headers={"X-Ghost-Key": KEY, **headers})
    assert seen == [expect]


# ── the turn: another member's file is neither inspected nor linked ─────────
def _member_turn(monkeypatch, tmp_path, member, model_turns, tools):
    from tests.test_requester_role import _agent
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    agent._note_member_files({"gen_a.png"}, member="U1")
    agent.available_tools = tools
    ctx.llm_client.chat_completion = AsyncMock(side_effect=model_turns)
    return agent


@pytest.mark.parametrize("member,allowed", [("U1", True), ("U2", True), ("", True)])   # §4QH: any member's
async def test_a_members_vision_call_reaches_any_members_image(monkeypatch, tmp_path, member, allowed):
    from tests.test_requester_role import _resp, _tc
    called = AsyncMock(return_value="VISION ANALYSIS RESULT: a cat")
    agent = _member_turn(monkeypatch, tmp_path, member, [
        _resp("", [_tc("c0", "vision_analysis", {"action": "describe_picture", "target": "gen_a.png"})]),
        _resp("Done."), _resp("Done.")], {"vision_analysis": called})
    await agent.handle_chat({"messages": [{"role": "user", "content": "describe gen_a.png"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member", requester_member=member)
    assert called.await_count == (1 if allowed else 0)


@pytest.mark.parametrize("member,kept", [("U1", True), ("U2", True)])   # §4QH: any member's
async def test_a_members_reply_links_any_members_image(monkeypatch, tmp_path, member, kept):
    from tests.test_requester_role import _resp
    agent = _member_turn(monkeypatch, tmp_path, member,
                         [_resp("Here: ![x](/api/download/gen_a.png)"), _resp("Done.")], {})
    content, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "show gen_a.png"}]},
                                            FakeBgTasks(), request_id="web-1", requester_role="member",
                                            requester_member=member)
    assert ("/api/download/gen_a.png" in content) is kept, content


async def test_handle_chat_restores_the_callers_member(monkeypatch, tmp_path):
    from tests.test_requester_role import _resp
    agent = _member_turn(monkeypatch, tmp_path, "U1", [_resp("ok"), _resp("ok")], {})
    t = requester_member_context.set("U9")
    try:
        await agent.handle_chat({"messages": [{"role": "user", "content": "hi"}]}, FakeBgTasks(),
                                request_id="web-1", requester_role="member", requester_member="U1")
        assert requester_member_context.get() == "U9"
    finally:
        requester_member_context.reset(t)


# ── the bot ─────────────────────────────────────────────────────────────────
def _bot_fns():
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in ("requester_role_header", "member_id_header")]
    ns = {}
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(BOT_PATH), "exec"), ns)
    return ns["member_id_header"]


@pytest.mark.parametrize("requester,owner,expect", [
    ("U2", "U1", "U2"), ("U1", "U1", ""), (None, "U1", ""), ("", "U1", ""), ("U2", None, "U2"),
])
def test_the_bot_names_the_member(requester, owner, expect):
    assert _bot_fns()(requester, owner) == expect


def test_the_bot_sends_the_member_beside_every_requester_header():
    """AST: chat, upload and download — every dict that carries
    `X-Ghost-Requester: requester_role_header(<who>, OWNER_ID)` also carries
    `X-Ghost-Member: member_id_header(<who>, OWNER_ID)` for the same person."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    par = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}
    where = []
    for d in (n for n in ast.walk(tree) if isinstance(n, ast.Dict)):
        kv = {k.value: v for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)}
        role = kv.get("X-Ghost-Requester")
        if not (isinstance(role, ast.Call) and getattr(role.func, "id", "") == "requester_role_header"):
            continue
        mem = kv.get("X-Ghost-Member")
        assert isinstance(mem, ast.Call) and getattr(mem.func, "id", "") == "member_id_header", d.lineno
        assert ast.dump(mem.args[0]) == ast.dump(role.args[0]) and getattr(mem.args[1], "id", "") == "OWNER_ID"
        fn = d
        while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = par.get(fn)
        where.append(fn.name)
    assert sorted(where) == ["_process_message", "_process_message", "upload_file_to_agent"], where


def test_an_autonomous_turn_names_no_member():
    """A scheduled fire copies the creating turn's contextvars (§4LD): a task
    made on a member's turn must not run as that member."""
    import asyncio
    from types import SimpleNamespace
    import ghost_agent.main as M
    seen = []

    async def _hc(body, bg, request_id=None):
        seen.append(requester_member_context.get())
        return "ok", None, None
    ctx = SimpleNamespace(agent=SimpleNamespace(handle_chat=_hc), llm_client=SimpleNamespace(foreground_requests=0))
    undo = _as("member", "U1")
    try:
        asyncio.run(M._handle_chat_foreground(ctx, {"messages": []}, "sched-x"))
        assert seen == [""] and requester_member_context.get() == "U1"
    finally:
        undo()


# ── §4QH: the owner's files stay out of a member's turn ─────────────────────
async def test_a_members_vision_call_never_reaches_an_owner_file(monkeypatch, tmp_path):
    from tests.test_requester_role import _resp, _tc
    called = AsyncMock(return_value="VISION ANALYSIS RESULT: a page")
    agent = _member_turn(monkeypatch, tmp_path, "U2", [
        _resp("", [_tc("c0", "vision_analysis", {"action": "describe_picture", "target": "owner.png"})]),
        _resp("Done."), _resp("Done.")], {"vision_analysis": called})
    await agent.handle_chat({"messages": [{"role": "user", "content": "describe owner.png"}]},
                            FakeBgTasks(), request_id="web-1", requester_role="member", requester_member="U2")
    assert called.await_count == 0


async def test_a_members_reply_never_links_an_owner_file(monkeypatch, tmp_path):
    from tests.test_requester_role import _resp
    agent = _member_turn(monkeypatch, tmp_path, "U2",
                         [_resp("Here: ![x](/api/download/owner.png)"), _resp("Done.")], {})
    content, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "show owner.png"}]},
                                            FakeBgTasks(), request_id="web-1", requester_role="member",
                                            requester_member="U2")
    assert "/api/download/owner.png" not in content, content


def test_the_reply_image_download_sends_the_threads_session():
    """§4SX B0 review r4 (AST, like the pin above): the reply-image download
    in `_process_message` carries `X-Ghost-Session:
    _slack_session_id(thread_ts, requester)` — the chat payload's string —
    or a bare image name in the thread's focused project 404s."""
    tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_process_message")
    hits = []
    for d in (n for n in ast.walk(fn) if isinstance(n, ast.Dict)):
        kv = {k.value: v for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)}
        s = kv.get("X-Ghost-Session")
        if "X-Ghost-Requester" in kv and isinstance(s, ast.Call) and getattr(s.func, "id", "") == "_slack_session_id":
            hits.append([getattr(a, "id", "") for a in s.args])
    assert hits == [["thread_ts", "requester"]], hits
