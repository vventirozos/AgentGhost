"""§4UM-s (operator 2026-11-13, req 56dff518's change #2): the change-log entry's summary is the reply the owner
was shown (smoothed — not the narration written beside tool calls), and its time is when the files' content last
changed — a metadata-only touch (an xattr written on open, a chmod) neither dates the entry nor un-deploys it."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.request_scope import RequestScope, install, reset
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools import projects as TP
from ghost_agent.utils.logging import request_id_context


@pytest.fixture(autouse=True)
def _clean():
    MP._PENDING_CHANGES.clear()
    MP._REQUEST_ASK.clear()
    yield
    MP._PENDING_CHANGES.clear()
    MP._REQUEST_ASK.clear()


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def ctx(tmp_path, store):
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None,
                           llm_client=None, workspace_model=None, sandbox_dir=str(tmp_path / "sb"),
                           last_user_content="")


def _built(store):
    pid = store.create_project("Site")
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("<h1>hi</h1>")
    t = store.add_task(pid, "Build index.html")
    tok = request_id_context.set("req-build")
    try:
        store.update_task(t, status="DONE", result_summary="built")
        store.register_file_artifact(t, "index.html")
    finally:
        request_id_context.reset(tok)
    MP.flush_deferred_sweeps("req-build")
    return pid, ws


class _Owner:
    def __init__(self, ctx, rid, ask):
        self.ctx, self.rid, self.ask = ctx, rid, ask

    def __enter__(self):
        self.ctx.last_user_content = self.ask
        self.t1 = request_id_context.set(self.rid)
        self.t2 = install(RequestScope(context=self.ctx))
        TP.begin_request_watch(self.ctx, self.rid, self.ask)
        return self

    def __exit__(self, *exc):
        reset(self.t2)
        request_id_context.reset(self.t1)
        return False


def _sh(ws, script):
    subprocess.run(["bash", "-c", script], cwd=ws, check=True)


# ───────────────────────────── the entry's time ─────────────────────────────

def test_a_metadata_touch_after_the_write_does_not_date_the_entry(ctx, store):
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-s-1", "make the title bigger"):
        TP.prepare_project_write(ctx, project_id=pid)                   # what execute does before it runs
        _sh(ws, "sed -i.bak 's/h1/h2/g' index.html && rm index.html.bak")
        written = os.stat(ws / "index.html").st_mtime
        time.sleep(1.1)
        os.chmod(ws / "index.html", 0o600)                              # the file opened / chmod-ed: ctime moves
    assert os.stat(ws / "index.html").st_ctime > written + 1            # the test's premise
    MP.flush_change_log("req-s-1", reply="Bigger.")
    (e,) = store.list_changes(pid, kind="change")
    assert e["changed"] == ["index.html"]
    assert abs(float(e["ts"]) - written) < 0.01                         # when the content changed, not the touch


def test_content_renamed_into_place_after_a_push_is_not_live(ctx, store):
    """Review MAJOR: written 11:52, pushed 12:00, `mv draft index` 12:19 — the lab serves the old page."""
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-s-5", "swap in the new page"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "mkdir -p ../drafts-s5 && echo '<h1>new page</h1>' > ../drafts-s5/draft.html")
        time.sleep(1.1)
        MP.note_change_push(store, pid, None, url="https://lab.example/", lab_id="lab_00000001",
                            how="deploy", rid="req-s-5")
        time.sleep(1.1)
        _sh(ws, "mv ../drafts-s5/draft.html index.html")
    MP.flush_change_log("req-s-5", reply="Swapped.")
    (e,) = store.list_changes(pid, kind="change")
    assert e["changed"] == ["index.html"] and not e["deployed"]


@pytest.mark.parametrize("switch_in", ["the same request", "a later request"])
def test_content_renamed_into_place_after_a_staged_deploy_is_not_in_the_switch(ctx, store, switch_in):
    """Review r2 MAJOR: a staged deploy cuts the version a later `lab switch` serves — not a push, still a cut."""
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-s-7", "try a new page"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "echo '<h1>new design</h1>' > draft.html")
        time.sleep(1.1)
        MP.note_change_stage(store, pid, "lab_00000001")               # the staged version: the old index.html
        time.sleep(1.1)
        _sh(ws, "mv draft.html index.html")
        if switch_in == "the same request":
            time.sleep(1.1)
            MP.note_change_push(store, pid, None, url="https://lab.example/", lab_id="lab_00000001",
                                how="switch", rid="req-s-7", before_t=MP.staged_at(store, pid, "lab_00000001"))
    MP.flush_change_log("req-s-7", reply="Swapped.")
    if switch_in == "a later request":
        store.record_deploy(pid, url="https://lab.example/", files=None, how="switch", lab_id="lab_00000001",
                            before_ts=MP.staged_at(store, pid, "lab_00000001"))
    (e,) = store.list_changes(pid, kind="change")
    assert e["changed"] == ["index.html"] and not e["deployed"]


def test_a_file_moved_in_with_an_older_content_time_is_dated_by_its_move(ctx, store, tmp_path):
    pid, ws = _built(store)
    ctx.current_project_id = pid
    outside = tmp_path / "keep.js"
    outside.write_text("kept")
    old = time.time() - 3600
    os.utime(outside, (old, old))
    with _Owner(ctx, "req-s-2", "bring in keep.js"):
        TP.prepare_project_write(ctx, project_id=pid)
        start = time.time()
        os.rename(outside, ws / "keep.js")                              # mtime stays an hour old
    MP.flush_change_log("req-s-2", reply="Brought in.")
    (e,) = store.list_changes(pid, kind="change")
    assert e["added"] == ["keep.js"]
    assert float(e["ts"]) >= start - 1.0                                # the move, not the hour-old content


# ───────────────────────────── the entry's summary ─────────────────────────────

async def test_the_request_end_logs_the_shown_reply_not_the_narration_beside_tool_calls(monkeypatch):
    import ghost_agent.memory.projects as P
    from ghost_agent.core import agent as A
    from tests.test_forced_final_no_answer import _agent as _ff_agent, _resp, _tc
    from tests.helpers import FakeBgTasks
    got = []
    monkeypatch.setattr(P, "flush_change_log", lambda rid, reply="": got.append((rid, reply)) or 0)
    agent, _ = _ff_agent(monkeypatch, [
        _resp("Now let me look that up before I change anything.",
              [_tc("c0", "web_search", {"query": "arkanoid power ups"})]),
        _resp("Arkanoid is in and live: one new app file, registered in index.html. Controls: arrows or the "
              "mouse to move, space to launch. Eight power-ups, thirty procedurally generated levels, and the "
              "best score is kept in the browser between visits."),          # ≥ 200 chars: smoothing applies
        _resp("(unreachable)"),
    ])
    out = await agent.handle_chat({"messages": [{"role": "user", "content": "add arkanoid"}]}, FakeBgTasks(),
                                  request_id="req-s-3")
    replies = [r for rid, r in got if rid == "req-s-3"]
    assert replies and "Arkanoid is in and live" in replies[0]
    assert "Now let me look that up" not in replies[0]                  # the narration the owner never kept
    assert "req-s-3" not in A._CHANGE_LOG_REPLIES                       # taken at the request's end
    assert "Now let me look that up" not in out[0]                      # the shown reply, the same text
    assert replies[0].strip() == out[0].strip()


async def test_the_turn_budget_banner_does_not_fill_the_summary(monkeypatch):
    import ghost_agent.memory.projects as P
    from ghost_agent.core import agent as A
    from tests.test_forced_final_no_answer import _agent as _ff_agent, _resp
    from tests.helpers import FakeBgTasks
    got = []
    monkeypatch.setattr(P, "flush_change_log", lambda rid, reply="": got.append((rid, reply)) or 0)
    agent, _ = _ff_agent(monkeypatch, [_resp("unused")])
    real = agent._finalize_and_return

    async def _budget(fs):                                              # the reply handle_chat builds on exhaustion
        fs.final_ai_content = A.budget_exhausted_note(30, report_forced=True, project_active=True) + \
            "The board renders; the power-up timer still drifts."
        fs.turn_budget_exhausted = True
        return await real(fs)
    monkeypatch.setattr(agent, "_finalize_and_return", _budget)
    await agent.handle_chat({"messages": [{"role": "user", "content": "add x"}]}, FakeBgTasks(),
                            request_id="req-s-6")
    (reply,) = [r for rid, r in got if rid == "req-s-6"]
    assert reply.startswith("Not finished (turn budget reached). The board renders")
    assert "TURN BUDGET" not in reply


async def test_a_streamed_reply_s_summary_is_the_treated_text(monkeypatch):
    """The stream path: handle_chat's finally has no shown reply (""), the drain fills the summary with the
    treated text — the narration beside lookups left out."""
    import ghost_agent.memory.projects as P
    from tests.test_4lk_stream_path import _ANSWER, _drive2
    seen = []
    monkeypatch.setattr(P, "set_change_summary", lambda rid, text: seen.append(text) or 0)
    narration = "Let me pull the release notes first."
    await _drive2([_ANSWER], prefix=narration + "\n\n", pre_tool=[narration],
                  tools=[{"name": "web_search", "content": "results"}])
    assert seen and narration not in seen[-1] and "I rewrote app.py" in seen[-1]


async def test_a_request_that_ends_without_a_shown_reply_logs_no_summary(monkeypatch):
    """A request that never finished its reply (an error or a cancel before the end, a stream whose drain fills
    the summary later) logs no summary here — never the text so far, narration included."""
    import ghost_agent.memory.projects as P
    from tests.test_forced_final_no_answer import _agent as _ff_agent, _resp, _tc
    from tests.helpers import FakeBgTasks
    got = []
    monkeypatch.setattr(P, "flush_change_log", lambda rid, reply="": got.append((rid, reply)) or 0)
    agent, _ = _ff_agent(monkeypatch, [
        _resp("Now let me look that up.", [_tc("c0", "web_search", {"query": "x"})]),
        _resp("Arkanoid is in."),
        _resp("(unreachable)"),
    ])

    async def _dies(fs):
        raise RuntimeError("finalize died")
    monkeypatch.setattr(agent, "_finalize_and_return", _dies)
    with pytest.raises(RuntimeError):
        await agent.handle_chat({"messages": [{"role": "user", "content": "add x"}]}, FakeBgTasks(),
                                request_id="req-s-4")
    assert [r for rid, r in got if rid == "req-s-4"] == [""]


async def test_a_streamed_request_s_finally_passes_no_summary(monkeypatch):
    """Review r2 m: the drain's `set_change_summary` fills only an EMPTY summary — so the finally of a streamed
    request must pass "" (it used to pass the narration, and the drain then changed nothing)."""
    import ghost_agent.memory.projects as P
    from ghost_agent.core import agent as A
    from tests.test_forced_final_no_answer import _agent as _ff_agent, _resp
    from tests.helpers import FakeBgTasks
    got = []
    monkeypatch.setattr(P, "flush_change_log", lambda rid, reply="": got.append((rid, reply)) or 0)
    agent, _ = _ff_agent(monkeypatch, [_resp("unused")])

    async def _streamed(fs):        # a streamed request hands back the SSE generator; finalize never runs
        return "stream", 0, fs.req_id
    monkeypatch.setattr(agent, "_finalize_and_return", _streamed)
    await agent.handle_chat({"messages": [{"role": "user", "content": "add x"}]}, FakeBgTasks(),
                            request_id="req-s-8")
    assert [r for rid, r in got if rid == "req-s-8"] == [""]
    assert "req-s-8" not in A._CHANGE_LOG_REPLIES
