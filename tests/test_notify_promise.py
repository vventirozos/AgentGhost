"""Project-scoped notify-promise persistence (2026-08-01).

Request 5299219b ("proceed with all tasks, notify me in slack") died 3.9s
in when a restart landed mid-turn; the follow-up requests completed the
project and NO ping fired — the per-request detector only reads the
current message. The promise now persists on the project (like
constraints) and fires when the project settles, from whichever request
or idle tick gets it there. core/notify_promise.py; wired at request
start (capture), both finalize paths (fire), and main.py's
on_project_done hook (idle completions).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.autonomous_activity import ActivityLog, SEVERITY_NOTIFY
from ghost_agent.core.notify_promise import (
    PROMISE_KEY, capture_promise, consume_promise, fire_promise_if_settled,
    peek_promise,
)
from ghost_agent.memory.projects import ProjectStore

_SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


@pytest.fixture(autouse=True)
def _fresh_notify_budget(monkeypatch):
    import ghost_agent.tools.notify_tool as nt
    monkeypatch.setattr(nt, "_sent_timestamps", [])


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def alog(tmp_path):
    return ActivityLog(tmp_path / "activity.jsonl")


def _records(tmp_path):
    p = tmp_path / "activity.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------- store layer

def test_capture_peek_consume_roundtrip(store):
    pid = store.create_project("Mini AI v3")
    assert capture_promise(store, pid,
                           "proceed with all tasks, notify me in slack "
                           "when you're done.", req_id="5299219b")
    entry = peek_promise(store, pid)
    assert entry and "notify me in slack" in entry["ask"]
    assert entry["req_id"] == "5299219b"
    # atomic pop: first consume wins, second gets nothing
    assert consume_promise(store, pid) is not None
    assert consume_promise(store, pid) is None
    assert peek_promise(store, pid) is None


def test_capture_refreshes_existing_ask(store):
    pid = store.create_project("P")
    capture_promise(store, pid, "notify me when done", req_id="a")
    capture_promise(store, pid, "ping me on slack after", req_id="b")
    entry = peek_promise(store, pid)
    assert entry["req_id"] == "b" and "ping me" in entry["ask"]


def test_capture_skips_settled_project(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "only task")
    store.update_task(tid, status="DONE")           # rolls project DONE
    assert store.get_project(pid)["status"] == "DONE"
    assert capture_promise(store, pid, "notify me when done") is False
    assert peek_promise(store, pid) is None


# ------------------------------------------------------------- fire logic

def test_fire_noop_while_project_open(store, alog, tmp_path):
    pid = store.create_project("P")
    store.add_task(pid, "open task")
    capture_promise(store, pid, "notify me when done")
    assert fire_promise_if_settled(store, alog, pid) is False
    assert peek_promise(store, pid) is not None      # flag survives
    assert _records(tmp_path) == []


def test_fire_delivers_on_done_and_consumes(store, alog, tmp_path):
    pid = store.create_project("Mini AI v3")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me in slack when done",
                    req_id="5299219b")
    store.update_task(tid, status="DONE")
    assert store.get_project(pid)["status"] == "DONE"
    fired = fire_promise_if_settled(store, alog, pid, req_id="17606976",
                                    headline="All 7 tasks completed; demo "
                                             "runs.")
    assert fired is True
    recs = _records(tmp_path)
    assert len(recs) == 1
    assert recs[0]["severity"] == SEVERITY_NOTIFY
    assert recs[0]["phase"] == "project"
    assert "Mini AI v3" in recs[0]["summary"]
    assert "is DONE" in recs[0]["summary"]
    assert "All 7 tasks completed" in recs[0]["summary"]
    assert recs[0]["meta"]["project_id"] == pid
    assert recs[0]["meta"]["req_id"] == "17606976"
    # consumed: no double delivery
    assert peek_promise(store, pid) is None
    assert fire_promise_if_settled(store, alog, pid) is False
    assert len(_records(tmp_path)) == 1


def test_fire_failed_wording(store, alog, tmp_path):
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me when done")
    store.update_task(tid, status="FAILED", failure_reason="boom")
    assert store.get_project(pid)["status"] == "FAILED"
    assert fire_promise_if_settled(store, alog, pid) is True
    assert "FAILED" in _records(tmp_path)[0]["summary"]


def test_fire_model_notified_consumes_silently(store, alog, tmp_path):
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me when done")
    store.update_task(tid, status="DONE")
    fired = fire_promise_if_settled(store, alog, pid, model_notified=True)
    assert fired is False
    assert peek_promise(store, pid) is None          # consumed
    assert _records(tmp_path) == []                  # no second ping


def test_fire_rate_limited_leaves_flag_for_retry(store, alog, tmp_path,
                                                 monkeypatch):
    import ghost_agent.tools.notify_tool as nt
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me when done")
    store.update_task(tid, status="DONE")
    monkeypatch.setattr(nt, "_rate_limited", lambda: True)
    assert fire_promise_if_settled(store, alog, pid) is False
    assert peek_promise(store, pid) is not None      # promise NOT lost
    monkeypatch.setattr(nt, "_rate_limited", lambda: False)
    assert fire_promise_if_settled(store, alog, pid) is True
    assert len(_records(tmp_path)) == 1


def test_fire_without_activity_log_leaves_flag(store, tmp_path):
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me when done")
    store.update_task(tid, status="DONE")
    assert fire_promise_if_settled(store, None, pid) is False
    assert peek_promise(store, pid) is not None


def test_fire_restores_flag_when_record_write_fails(store, alog, tmp_path,
                                                    monkeypatch):
    """The atomic pop must not EAT the promise when the ledger write fails
    — the entry is put back so a later settle check retries (review
    finding)."""
    pid = store.create_project("P")
    tid = store.add_task(pid, "task")
    capture_promise(store, pid, "notify me when done")
    store.update_task(tid, status="DONE")
    # ⚠ SCOPED, because `monkeypatch` is ONE instance per test — the
    # autouse `_fresh_notify_budget` above shares it. A bare
    # `monkeypatch.undo()` here therefore also undid the budget reset,
    # handing the second fire back the PROCESS-GLOBAL
    # `notify_tool._sent_timestamps` with whatever earlier tests in this
    # worker had left in it. At the 12/hour cap the notify is suppressed,
    # `fire_promise_if_settled` returns False, and this test fails on a
    # neighbour's spending. Reproduced 2026-08-24 by priming the global
    # with 12 stamps: `assert False is True` at the line below — the exact
    # failure `-n 8 --dist loadfile` reported intermittently, because which
    # files share a worker changes run to run. It passed 30/30 alone.
    with monkeypatch.context() as m:
        m.setattr(type(alog), "record", lambda self, *a, **kw: False)
        assert fire_promise_if_settled(store, alog, pid) is False
        assert peek_promise(store, pid) is not None      # restored
    assert fire_promise_if_settled(store, alog, pid) is True
    assert len(_records(tmp_path)) == 1


# ----------------------------------------------------- incident replay

def test_request_52_shape_end_to_end(store, alog, tmp_path):
    """The exact incident: the ask arrives in a request that dies before
    doing anything; LATER requests complete the project; the promise still
    fires exactly once."""
    pid = store.create_project("Mini AI v3")
    t1 = store.add_task(pid, "Research: tiny AI architectures")
    t2 = store.add_task(pid, "Core: network.py")
    # request 52: capture happens at request start… then the process dies.
    assert capture_promise(store, pid, "proceed with all tasks, notify me "
                                       "in slack when you're done.",
                           req_id="5299219b")
    # later requests (no notify ask in their text) complete the work
    store.update_task(t1, status="DONE")
    assert fire_promise_if_settled(store, alog, pid) is False  # still open
    store.update_task(t2, status="DONE")
    assert store.get_project(pid)["status"] == "DONE"
    assert fire_promise_if_settled(store, alog, pid,
                                   headline="all tasks complete") is True
    recs = _records(tmp_path)
    assert len(recs) == 1 and "is DONE" in recs[0]["summary"]


# ------------------------------------------------------------- wiring pins

def test_wiring_capture_fire_and_hook():
    agent_src = (_SRC / "core" / "agent.py").read_text()
    main_src = (_SRC / "main.py").read_text()
    # capture at request start: shortly after the constraint replay, and
    # gated on internal requests (a sub-agent prompt echoing "notify me…"
    # must not store a promise the user never made)
    cap_idx = agent_src.find("capture_promise(")
    assert cap_idx > 0
    assert "_merge_project_constraints" in agent_src[cap_idx - 8000:cap_idx]
    assert "_np_is_internal" in agent_src[cap_idx - 3000:cap_idx]
    # fire in BOTH finalize paths
    fin_idx = agent_src.find("def _finalize_and_return")
    stream_idx = agent_src.find("def _stream_final_generation")
    assert 0 < fin_idx < stream_idx
    fin_region = agent_src[fin_idx:stream_idx]
    assert "fire_promise_if_settled(" in fin_region
    assert "fire_promise_if_settled(" in agent_src[stream_idx:]
    # the fire runs for INTERNAL turns too (cron/sub-agent completions must
    # keep the user's promise) — only simulation-gated
    assert "if _np_pid and not is_simulation:" in fin_region
    # an active promise suppresses the per-request backstop (no premature
    # "Done —" ping mid-project → no double delivery at settle time)
    assert "not _promise_active" in fin_region
    assert "not _promise_active" in agent_src[stream_idx:]
    # the streamed twin fires against the DRAIN SNAPSHOT project id, never
    # the live pointer a concurrent turn can re-point mid-drain
    s_region = agent_src[stream_idx:]
    assert "_np_pid = _drain_pid" in s_region
    # idle-completion hook in main.py, standing down under a foreground req
    hook_idx = main_src.find("def _on_project_done")
    assert hook_idx > 0
    hook = main_src[hook_idx:hook_idx + 2000]
    assert "fire_promise_if_settled(" in hook
    assert "request_id_context.get()" in hook
    assert "sweep_project_workspace(_store, pid)" in hook
