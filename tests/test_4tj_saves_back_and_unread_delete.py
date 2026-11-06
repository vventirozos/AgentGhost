"""§4TJ (2026-11-06): req c6751cb9.

1. One llama without --slot-save-path at the agent's boot left head saves off
   for good: two later llama restarts WITH the flag changed nothing. At a
   picture look the only save plan left named the model before the swap, so
   `_conv_save` refused it silently and 80k tokens were re-read.
2. Refused a write over an unread 28 KB file, the model deleted the file
   unread to save the read, then wrote a new one.
"""
from __future__ import annotations

import io
import os
import time
import urllib.error
from types import SimpleNamespace

import pytest

from ghost_agent.core import head_saves as H
from ghost_agent.core import agent as A
from ghost_agent.core import llm as L
from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools import file_system as FS
from ghost_agent.utils import logging as GL
from ghost_agent.utils.logging import request_id_context, slot_session_context
from tests.test_4ro_head_saves import warm  # noqa: F401
from tests.test_4rv_conv_keep import conv, cwarm  # noqa: F401

BASE = "http://llama:8088"


def _http_error(code, body):
    return urllib.error.HTTPError(BASE, code, "x", {}, io.BytesIO(body.encode()))


NO_FLAG = '{"error":{"code":501,"message":"This server does not support slots action. Start it with `--slot-save-path`"}}'
FLAG = '{"error":{"code":400,"message":"Invalid filename","type":"invalid_request_error"}}'


@pytest.fixture(autouse=True)
def _clean():
    H._unsupported.clear()
    FS._DELETE_ASKED.clear()
    yield
    H._unsupported.clear()
    FS._DELETE_ASKED.clear()


def _serve(monkeypatch, result):
    calls = []

    def fake(url, body=None, timeout=5.0):
        calls.append((url, body))
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(H, "_http", fake)
    return calls


# ── the mark comes off exactly when llama takes slot actions ──
def test_a_server_that_now_has_the_flag_lifts_the_mark(monkeypatch):
    H._unsupported.add(BASE)
    calls = _serve(monkeypatch, _http_error(400, FLAG))
    assert H.recheck_supported(BASE) is True and H.supported(BASE)
    # the probe can write nothing: the name is empty
    assert calls == [(f"{BASE}/slots/{H.SLOT_ID}?action=save", {"filename": ""})]


@pytest.mark.parametrize("answer", [_http_error(501, NO_FLAG), _http_error(503, "loading"),
                                    OSError("connection refused"), TimeoutError()])
def test_anything_else_keeps_the_mark(monkeypatch, answer):
    H._unsupported.add(BASE)
    _serve(monkeypatch, answer)
    assert H.recheck_supported(BASE) is False and not H.supported(BASE)


def test_a_server_not_marked_is_never_asked(monkeypatch):
    calls = _serve(monkeypatch, _http_error(400, FLAG))
    assert H.recheck_supported(BASE) is False and calls == []


# ── the idle tick does it, and its warm-up then runs (that installs the hooks) ──
def _agent(monkeypatch):
    monkeypatch.setenv("GHOST_HEAD_SAVES", "1")
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(llm_client=SimpleNamespace(upstream_url=BASE + "/v1", foreground_requests=0))
    return a


async def test_head_saves_back_reports_only_the_change(monkeypatch):
    a = _agent(monkeypatch)
    calls = _serve(monkeypatch, _http_error(400, FLAG))
    assert await a._head_saves_back() is False and calls == []        # not marked: llama not asked
    H._unsupported.add(BASE)
    assert await a._head_saves_back() is True and H.supported(BASE)
    monkeypatch.setenv("GHOST_HEAD_SAVES", "0")
    H._unsupported.add(BASE)
    assert await a._head_saves_back() is False and not H.supported(BASE)


async def test_the_tick_warms_up_when_saves_came_back_though_nothing_ran(monkeypatch):
    """Without the warm-up the plan stays absent and no hook is installed."""
    a = _agent(monkeypatch)
    _serve(monkeypatch, _http_error(400, FLAG))
    warms, ticks = [], []

    async def wait(period, sleep, early=True):
        ticks.append(1)
        if len(ticks) == 2:
            H._unsupported.add(BASE)
        if len(ticks) > 3:
            raise RuntimeError("stop")
        return ""

    async def warm(**kw):
        warms.append(kw)

    async def metrics():
        return 0, 100
    async def no_member():
        return False
    a._rewarm_wait, a.warm_up_main_prefix, a._member_head_refresh = wait, warm, no_member
    a._note_llama_restart = lambda total: None
    with pytest.raises(RuntimeError):
        await a.rewarm_main_prefix_loop(60, metrics=metrics)
    # tick 1 warms (first), tick 2 finds the mark gone: warms though the total never moved, tick 3 skips
    assert len(warms) == 2


# ── a side call with no hook says so ──
async def _side(monkeypatch, *, keep="1"):
    for k in ("GHOST_CONV_KEEP", "GHOST_HEAD_SAVES", "GHOST_STATE_ANCHOR"):
        monkeypatch.setenv(k, keep)
    lines = []
    monkeypatch.setattr(L, "pretty_log", lambda title, content, **kw: lines.append((title, content, kw)))
    c = L.LLMClient.__new__(L.LLMClient)
    c.before_side_main_dispatch = None
    await c._before_side_main_dispatch({})
    return lines


async def test_a_side_call_without_a_hook_is_logged(monkeypatch):
    lines = await _side(monkeypatch)
    assert len(lines) == 1
    title, content, kw = lines[0]
    assert title == "Conversation" and "not saved before this request's picture look" in content
    assert kw["icon"] == L.Icons.HEAD_FAIL


async def test_the_line_names_the_call(monkeypatch):
    tok = L._side_call_what_ctx.set("this request's cut-write continuation")
    try:
        lines = await _side(monkeypatch)
    finally:
        L._side_call_what_ctx.reset(tok)
    assert "before this request's cut-write continuation" in lines[0][1]


async def test_with_the_feature_off_no_hook_is_normal(monkeypatch):
    assert await _side(monkeypatch, keep="0") == []


async def test_an_installed_hook_runs_and_logs_nothing(monkeypatch):
    for k in ("GHOST_CONV_KEEP", "GHOST_HEAD_SAVES", "GHOST_STATE_ANCHOR"):
        monkeypatch.setenv(k, "1")
    lines, ran = [], []
    monkeypatch.setattr(L, "pretty_log", lambda *a, **k: lines.append(a))

    async def hook(p):
        ran.append(p)
    c = L.LLMClient.__new__(L.LLMClient)
    c.before_side_main_dispatch = hook
    await c._before_side_main_dispatch({"x": 1})
    assert ran == [{"x": 1}] and lines == []


# ── delete: an unread file stays ──
OLD = time.time() - 86400


def _old(p, text="prior work\n"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    os.utime(p, (OLD, OLD))                  # (on APFS an earlier mtime moves the birth time back too)
    return p


@pytest.fixture
def req(monkeypatch):
    """A request that began a minute ago, in conversation sess-a."""
    t1 = request_id_context.set("req-1")
    t2 = slot_session_context.set("sess-a")
    monkeypatch.setattr(FS, "request_elapsed_s", lambda rid: 60.0)
    yield
    slot_session_context.reset(t2)
    request_id_context.reset(t1)


def _refused(out):
    return out.startswith("REJECTED") and getattr(out, "reason_code", "") == "unread_delete"


async def test_the_unread_file_is_refused_and_stays(req, tmp_path):
    p = _old(tmp_path / "webos" / "index.html")
    out = await FS.tool_delete_file("webos/index.html", tmp_path)
    assert _refused(out) and p.exists() and "Nothing was deleted" in out and "ask the user" in out


async def test_the_tool_route_is_refused_too(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    out = await FS.tool_file_system(operation="delete", path="a.txt", sandbox_dir=tmp_path)
    assert str(out).startswith("REJECTED") and p.exists()


async def test_a_file_read_whole_may_go(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    FS._mark_read(p, whole=True)
    assert (await FS.tool_delete_file("a.txt", tmp_path)).startswith("SUCCESS") and not p.exists()


async def test_a_file_changed_since_the_read_stays(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    FS._mark_read(p, whole=True)
    p.write_text("someone added much more after the read\n")
    os.utime(p, (OLD + 5, OLD + 5))
    assert _refused(await FS.tool_delete_file("a.txt", tmp_path)) and p.exists()


async def test_a_partly_read_file_stays(req, tmp_path):
    p = _old(tmp_path / "a.txt", "line\n" * 100)
    FS._mark_read(p, start=1, end=10)
    assert _refused(await FS.tool_delete_file("a.txt", tmp_path)) and p.exists()


async def test_a_file_this_request_wrote_may_go(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    FS._mark_written(p)
    assert (await FS.tool_delete_file("a.txt", tmp_path)).startswith("SUCCESS") and not p.exists()


async def test_an_empty_file_may_go(req, tmp_path):
    p = _old(tmp_path / "e.txt", "")
    assert (await FS.tool_delete_file("e.txt", tmp_path)).startswith("SUCCESS") and not p.exists()


async def test_a_file_born_in_this_request_may_go(req, tmp_path):
    """`execute` output (a tar, a screenshot) cannot be read first."""
    p = tmp_path / "out.tar.gz"
    p.write_bytes(b"\x1f\x8b binary")
    assert (await FS.tool_delete_file("out.tar.gz", tmp_path)).startswith("SUCCESS") and not p.exists()


async def test_an_edit_does_not_make_an_old_file_this_requests(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    p.write_text("one line changed by a script, the rest unseen\n")      # mtime now, born earlier
    assert getattr(p.stat(), "st_birthtime", 0) < time.time() - 60
    assert _refused(await FS.tool_delete_file("a.txt", tmp_path)) and p.exists()


async def test_outside_a_tracked_request_nothing_counts_as_new(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "request_elapsed_s", lambda rid: None)
    p = tmp_path / "fresh.txt"
    p.write_text("x")
    assert _refused(await FS.tool_delete_file("fresh.txt", tmp_path)) and p.exists()


# ── a directory is judged by its files ──
async def test_a_directory_with_an_unread_file_stays_whole(req, tmp_path):
    a, b = _old(tmp_path / "d" / "a.txt"), _old(tmp_path / "d" / "sub" / "b.txt")
    new = tmp_path / "d" / "new.txt"
    new.write_text("mine")
    out = await FS.tool_delete_file("d", tmp_path)
    assert _refused(out) and a.exists() and b.exists() and new.exists()
    assert "2 file(s)" in out and "sub/b.txt" in out and "new.txt" not in out


async def test_a_directory_of_this_requests_files_may_go(req, tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("mine")
    (d / "empty").mkdir()
    assert (await FS.tool_delete_file("d", tmp_path)).startswith("SUCCESS") and not d.exists()


async def test_a_link_in_a_directory_is_not_judged_by_its_target(req, tmp_path):
    target = _old(tmp_path / "keep.txt")
    d = tmp_path / "d"
    d.mkdir()
    (d / "l").symlink_to(target)
    assert (await FS.tool_delete_file("d", tmp_path)).startswith("SUCCESS") and target.exists()


async def test_a_tree_too_big_to_walk_stays_even_after_asking(req, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DELETE_WALK_MAX", 2)
    for n in range(4):
        _old(tmp_path / "d" / f"{n}.txt")
    out = await FS.tool_delete_file("d", tmp_path)
    assert _refused(out) and "at least 2" in out and "Nothing was deleted" in out
    t = request_id_context.set("req-2")
    try:
        assert _refused(await FS.tool_delete_file("d", tmp_path))
    finally:
        request_id_context.reset(t)
    assert (tmp_path / "d" / "0.txt").exists()


# ── the way out: the user is asked, a later request deletes ──
async def _later(call, rid="req-2", conv="sess-a"):
    t1, t2 = request_id_context.set(rid), slot_session_context.set(conv)
    try:
        return await call()
    finally:
        slot_session_context.reset(t2)
        request_id_context.reset(t1)


async def test_a_later_request_of_the_conversation_may_delete(req, tmp_path):
    p = _old(tmp_path / "a.bin")
    assert _refused(await FS.tool_delete_file("a.bin", tmp_path))
    assert _refused(await FS.tool_delete_file("a.bin", tmp_path)) and p.exists()     # same request: no
    out = await _later(lambda: FS.tool_delete_file("a.bin", tmp_path))
    assert out.startswith("SUCCESS") and not p.exists() and FS._DELETE_ASKED == {}


async def test_another_conversation_does_not_inherit_the_answer(req, tmp_path):
    p = _old(tmp_path / "a.bin")
    await FS.tool_delete_file("a.bin", tmp_path)
    assert _refused(await _later(lambda: FS.tool_delete_file("a.bin", tmp_path), conv="sess-b")) and p.exists()


async def test_a_file_changed_after_the_refusal_is_a_new_question(req, tmp_path):
    p = _old(tmp_path / "a.bin")
    await FS.tool_delete_file("a.bin", tmp_path)
    p.write_text("different content now\n")
    os.utime(p, (OLD + 9, OLD + 9))
    assert _refused(await _later(lambda: FS.tool_delete_file("a.bin", tmp_path))) and p.exists()


async def test_unattended_work_has_nobody_to_ask(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "request_elapsed_s", lambda rid: 60.0)
    p = _old(tmp_path / "a.bin")
    out = await _later(lambda: FS.tool_delete_file("a.bin", tmp_path), rid="tick-1", conv="")
    assert _refused(out) and "ask the user" not in out and "leave it" in out and FS._DELETE_ASKED == {}
    assert _refused(await _later(lambda: FS.tool_delete_file("a.bin", tmp_path), rid="tick-2", conv="")) and p.exists()


async def test_the_memory_of_refusals_is_bounded(req, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_IN_PROGRESS_MAX", 8)
    for n in range(20):
        _old(tmp_path / f"{n}.bin")
        await FS.tool_delete_file(f"{n}.bin", tmp_path)
    assert len(FS._DELETE_ASKED) <= 8
    # the newest still counts
    assert (await _later(lambda: FS.tool_delete_file("19.bin", tmp_path))).startswith("SUCCESS")


async def test_a_link_itself_is_removed_as_before(req, tmp_path):
    target = _old(tmp_path / "keep.txt")
    (tmp_path / "l").symlink_to(target)
    assert (await FS.tool_delete_file("l", tmp_path)).startswith("SUCCESS") and target.exists()


# ── review r2 ──
async def test_an_append_in_progress_does_not_unlock_the_delete(req, tmp_path):
    p = _old(tmp_path / "owner.txt", "x" * 55000)
    assert _refused(await FS.tool_delete_file("owner.txt", tmp_path))
    out = await FS.tool_file_system(operation="write", path="owner.txt", content="hello", append=True,
                                    more_parts=True, sandbox_dir=tmp_path)
    assert str(out).startswith("SUCCESS") and FS._progress_key(p) in FS._IN_PROGRESS
    assert _refused(await FS.tool_delete_file("owner.txt", tmp_path)) and p.exists()


async def test_a_finished_append_replaces_the_inode_and_still_does_not_unlock(req, tmp_path):
    p = _old(tmp_path / "owner.txt", "x" * 500)
    out = await FS.tool_file_system(operation="write", path="owner.txt", content="hello", append=True,
                                    sandbox_dir=tmp_path)
    assert str(out).startswith("SUCCESS") and FS._progress_key(p) not in FS._IN_PROGRESS
    assert p.stat().st_birthtime > time.time() - 60                      # reads as just born
    assert _refused(await FS.tool_delete_file("owner.txt", tmp_path)) and p.exists()


async def test_a_tool_edit_does_not_unlock_either(req, tmp_path):
    p = _old(tmp_path / "owner.txt", "keep this\nand this\n")
    FS._note_mutation(p)
    os.utime(p, None)
    assert _refused(await FS.tool_delete_file("owner.txt", tmp_path)) and p.exists()


async def test_a_file_this_request_wrote_stays_its_own_after_an_append(req, tmp_path):
    await FS.tool_file_system(operation="write", path="n.txt", content="a\n", sandbox_dir=tmp_path)
    await FS.tool_file_system(operation="write", path="n.txt", content="b\n", append=True, sandbox_dir=tmp_path)
    assert (await FS.tool_delete_file("n.txt", tmp_path)).startswith("SUCCESS")


async def test_its_own_file_changed_by_a_script_after_an_append_may_go(req, tmp_path):
    """Written, appended, then changed by `execute`: the written stamp is stale,
    so only the birth exit lets it go, and an append must not have voided that."""
    p = tmp_path / "n.txt"
    await FS.tool_file_system(operation="write", path="n.txt", content="a\n", sandbox_dir=tmp_path)
    await FS.tool_file_system(operation="write", path="n.txt", content="b\n", append=True, sandbox_dir=tmp_path)
    p.write_text("rewritten by a script\n")
    assert (await FS.tool_delete_file("n.txt", tmp_path)).startswith("SUCCESS") and not p.exists()


async def test_reading_after_an_append_still_opens_the_way(req, tmp_path):
    p = _old(tmp_path / "owner.txt", "x\n")
    await FS.tool_file_system(operation="write", path="owner.txt", content="y\n", append=True, sandbox_dir=tmp_path)
    FS._mark_read(p, whole=True)
    assert (await FS.tool_delete_file("owner.txt", tmp_path)).startswith("SUCCESS")


def test_the_mutation_record_ends_with_the_request(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    FS._note_mutation(p)
    assert FS._progress_key(p) in FS._MUTATED
    FS.forget_request_receipts("req-1")
    assert FS._progress_key(p) not in FS._MUTATED


def test_the_mutation_record_is_bounded(req, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_MUTATED", {})
    monkeypatch.setattr(FS, "_IN_PROGRESS_MAX", 8)
    for n in range(20):
        FS._note_mutation(tmp_path / f"{n}.txt")
    assert len(FS._MUTATED) <= 8


async def test_the_guard_for_writes_still_counts_a_part_in_progress(req, tmp_path):
    p = _old(tmp_path / "a.txt")
    FS._mark_in_progress(p, True)
    assert FS.unread_rewrite_refusal(p, "a.txt") == ""
    assert FS.unread_rewrite_refusal(p, "a.txt", count_in_progress=False) != ""


async def test_a_new_file_in_parts_may_go(req, tmp_path):
    out = await FS.tool_file_system(operation="write", path="n.txt", content="first\n", more_parts=True,
                                    sandbox_dir=tmp_path)
    assert str(out).startswith("SUCCESS")
    assert (await FS.tool_delete_file("n.txt", tmp_path)).startswith("SUCCESS")


async def test_a_directory_refusal_does_not_say_to_read_the_folder(req, tmp_path):
    _old(tmp_path / "d" / "a.txt")
    out = await FS.tool_delete_file("d", tmp_path)
    assert "operation='read' on each" in out and "Read it whole" not in out


async def test_read_after_a_refusal_drops_the_record(req, tmp_path):
    """A later replacement must be a new question, whatever it fingerprints as."""
    p = _old(tmp_path / "a.txt")
    await FS.tool_delete_file("a.txt", tmp_path)
    assert FS._DELETE_ASKED
    FS._mark_read(p, whole=True)
    assert (await FS.tool_delete_file("a.txt", tmp_path)).startswith("SUCCESS") and FS._DELETE_ASKED == {}


async def test_nothing_to_save_is_not_reported_as_lost(cwarm, monkeypatch, rid):
    agent, llm, st = await _stale(cwarm, monkeypatch)
    agent._conv_plan_last = agent._head_plan_last = None
    llm._slot_session = None
    assert await agent._conv_save(llm, "this test") is False and st["lines"] == []


# ── review r1 ──
async def test_a_marked_server_still_without_the_flag_is_not_reported_back(monkeypatch):
    a = _agent(monkeypatch)
    lines = []
    monkeypatch.setattr(A, "pretty_log", lambda *x, **k: lines.append(x))
    _serve(monkeypatch, _http_error(501, NO_FLAG))
    H._unsupported.add(BASE)
    assert await a._head_saves_back() is False and not H.supported(BASE) and lines == []


async def test_the_smoke_gate_cleans_up_where_no_request_is_tracked(tmp_path, monkeypatch):
    from ghost_agent.core.build_gates import smoke_gate
    monkeypatch.setattr(FS, "request_elapsed_s", lambda rid: None)
    script = tmp_path / ".smoke_gate.py"

    async def runner(name, args):
        if name == "execute":
            _old(script, args["content"])
            return "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nSMOKE OK"
        return await FS.tool_file_system(sandbox_dir=tmp_path, **args)
    await smoke_gate(runner, ["app.py"])
    assert not script.exists()


async def test_a_file_named_like_scratch_is_the_only_exemption(req, tmp_path):
    p = _old(tmp_path / "smoke_gate.py")
    assert _refused(await FS.tool_delete_file("smoke_gate.py", tmp_path)) and p.exists()


async def test_a_directory_read_in_the_refused_request_passes_the_next(req, tmp_path):
    """Receipts are per request: the file read in req-1 is unread in req-2."""
    a, b = _old(tmp_path / "d" / "a.txt"), _old(tmp_path / "d" / "b.txt")
    FS._mark_read(a, whole=True)
    out = await FS.tool_delete_file("d", tmp_path)
    assert _refused(out) and "1 file(s)" in out
    assert (await _later(lambda: FS.tool_delete_file("d", tmp_path))).startswith("SUCCESS")
    assert not a.exists() and not b.exists()


async def test_a_changed_file_in_a_directory_is_a_new_question(req, tmp_path):
    a, b = _old(tmp_path / "d" / "a.txt"), _old(tmp_path / "d" / "b.txt")
    FS._mark_read(a, whole=True)
    await FS.tool_delete_file("d", tmp_path)
    a.write_text("changed after the refusal, though read before it\n")
    os.utime(a, (OLD + 9, OLD + 9))
    assert _refused(await _later(lambda: FS.tool_delete_file("d", tmp_path))) and b.exists()


async def test_the_answer_expires(req, tmp_path, monkeypatch):
    p = _old(tmp_path / "a.bin")
    await FS.tool_delete_file("a.bin", tmp_path)
    k = next(iter(FS._DELETE_ASKED))
    rid, sig, at = FS._DELETE_ASKED[k]
    FS._DELETE_ASKED[k] = (rid, sig, at - FS._DELETE_ASK_TTL_S - 1)
    assert _refused(await _later(lambda: FS.tool_delete_file("a.bin", tmp_path))) and p.exists()
    FS._DELETE_ASKED[k] = (rid, FS._DELETE_ASKED[k][1], time.monotonic() - FS._DELETE_ASK_TTL_S + 60)
    assert (await _later(lambda: FS.tool_delete_file("a.bin", tmp_path))).startswith("SUCCESS")


async def test_a_tree_too_big_does_not_promise_the_way_out(req, tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DELETE_WALK_MAX", 2)
    for n in range(4):
        _old(tmp_path / "d" / f"{n}.txt")
    out = await FS.tool_delete_file("d", tmp_path)
    assert "ask the user" not in out and "tell the user" in out


# ── the incident: the only plan left names the model before the swap ──
@pytest.fixture
def rid():
    tok = request_id_context.set("req-1")
    yield "req-1"
    request_id_context.reset(tok)


async def _stale(cwarm, monkeypatch):
    from tests.test_4rwg_side_call import _held
    agent, llm, st = cwarm
    await _held(agent, llm, st, monkeypatch)
    st["lines"].clear()
    return agent, llm, st


async def test_the_incident_path_is_logged_and_saves_nothing(cwarm, monkeypatch, rid):
    from tests.test_4rwg_side_call import LOOK
    agent, llm, st = await _stale(cwarm, monkeypatch)
    agent._conv_plan_last, agent._head_plan_last = agent._head_plan_last, None
    monkeypatch.setattr(H, "server_identity", lambda base: "the model after the swap")
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == []
    assert any("not saved before this request's picture look" in m and "another model" in m for m in st["lines"])


async def test_same_model_with_an_old_plan_still_saves(cwarm, monkeypatch, rid):
    from tests.test_4rwg_side_call import LOOK
    agent, llm, st = await _stale(cwarm, monkeypatch)
    agent._conv_plan_last, agent._head_plan_last = agent._head_plan_last, None
    monkeypatch.setattr(H, "server_identity", lambda base: agent._conv_plan_last["identity"])
    await agent._side_call_before(LOOK)
    assert len(st["conv_saves"]) == 1 and not any("another model" in m for m in st["lines"])


async def test_no_plan_at_all_is_logged(cwarm, monkeypatch, rid):
    from tests.test_4rwg_side_call import LOOK
    agent, llm, st = await _stale(cwarm, monkeypatch)
    agent._conv_plan_last = agent._head_plan_last = None
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == [] and any("no save plan yet" in m for m in st["lines"])


async def test_a_slot_llama_did_not_report_is_logged(cwarm, monkeypatch, rid):
    from tests.test_4rwg_side_call import LOOK
    agent, llm, st = await _stale(cwarm, monkeypatch)
    monkeypatch.setattr(H, "slot_tokens", lambda base: None)
    await agent._side_call_before(LOOK)
    assert st["conv_saves"] == [] and any("did not report the slot" in m for m in st["lines"])


async def test_after_the_mark_lifts_a_real_warm_up_installs_the_hooks(cwarm, monkeypatch):
    agent, llm, st = cwarm
    base = H.server_base(llm.upstream_url)
    H._unsupported.add(base)
    llm.before_side_main_dispatch = None
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_plan_last is None and llm.before_side_main_dispatch is None
    _serve(monkeypatch, _http_error(400, FLAG))
    assert await agent._head_saves_back() is True
    await agent.warm_up_main_prefix(quiet=True)
    assert isinstance(agent._head_plan_last, dict)
    assert llm.before_side_main_dispatch == agent._side_call_before
