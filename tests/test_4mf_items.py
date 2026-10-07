"""§4MF — operator "proceed with all items": behaviour pins."""
from __future__ import annotations

import datetime as dt
import json
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_old_trajectory_days_are_archived_not_deleted(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector, archive_old_partitions
    today = dt.date(2026, 10, 7)
    for day in ("2026-07-01", "2026-07-08", "2026-07-09", "2026-10-06"):
        d = tmp_path / day
        d.mkdir()
        (d / "session-s.jsonl").write_text(json.dumps({"id": day, "task_kind": "chat"}) + "\n")
    (tmp_path / "corrections.jsonl").write_text("")
    done = archive_old_partitions(tmp_path, 90, now=today)
    assert done == ["2026-07-01", "2026-07-08"]
    assert not (tmp_path / "2026-07-01").exists() and (tmp_path / "2026-07-09").exists()
    with tarfile.open(tmp_path / "archive" / "2026-07-01.tar.gz") as tf:
        assert "2026-07-01/session-s.jsonl" in tf.getnames()
    ids = {t.id for t in TrajectoryCollector(tmp_path).iter_trajectories()}
    assert ids == {"2026-07-09", "2026-10-06"}          # readers skip the archive dir
    assert archive_old_partitions(tmp_path, 90, now=today) == []


def _docker_ok():
    import shutil
    import subprocess
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", "ghost-agent-base:latest"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _docker_ok(), reason="needs docker and the sandbox image")
def test_the_sandbox_rm_refuses_whole_projects_and_allows_the_rest():
    import subprocess
    from ghost_agent.sandbox.docker import SAFE_RM_INSTALL_CMD
    pid = "48e0373aaab3"
    script = (
        SAFE_RM_INSTALL_CMD + " >/dev/null && "
        f"mkdir -p /workspace/projects/{pid}/src /workspace/uploads /workspace/projects/notes && "
        f"touch /workspace/projects/{pid}/src/a.py /workspace/uploads/x.pdf /workspace/tmp.txt && "
        f"cd /workspace && (rm -rf projects/{pid} 2>/dev/null; echo P$?) && "
        f"(rm -rf /workspace/projects/{pid}/ 2>/dev/null; echo Q$?) && "
        "(rm -rf -- uploads 2>/dev/null; echo U$?) && (rm -rf /workspace 2>/dev/null; echo W$?) && "
        f"(rm -rf projects/{pid}/src; echo S$?) && (rm tmp.txt; echo T$?) && "
        "(rm -r projects/notes; echo N$?) && "
        "(rm -rf /workspace/Projects/48E0373AAAB3 2>/dev/null; echo C$?) && "
        f"ln -s /workspace/projects/{pid} /workspace/lnk && (rm /workspace/lnk; echo L$?) && "
        "(cd /tmp && touch t1 && rm t1; echo O$?) && "
        f"ls -d /workspace/projects/{pid} /workspace/uploads/x.pdf"
    )
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "sh",
                        "ghost-agent-base:latest", "-c", script], capture_output=True, text=True, timeout=120)
    out = r.stdout
    assert all(t in out for t in ("P1", "Q1", "U1", "W1")), out + r.stderr
    assert all(t in out for t in ("S0", "T0", "N0", "L0", "O0")), out + r.stderr
    assert "C1" in out, out + r.stderr                 # a case variant is the same project
    assert f"/workspace/projects/{pid}" in out and "/workspace/uploads/x.pdf" in out


async def test_the_knowledge_shadow_records_a_verdict_and_nothing_else(tmp_path):
    from ghost_agent.core import knowledge_shadow as KS
    from ghost_agent.core.knowledge_shadow import check_and_record
    KS._RUNNING["last"] = 0.0
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content":
        '{"verdict": "errors", "issues": [{"quote": "pg_last_subscription_replay_lsn()", '
        '"why": "no such function"}]}'}}]})
    req = "review my postgres logical replication cutover plan from DB5 to DB6 please"
    ans = "Your plan is sound. " * 25 + "Use pg_last_subscription_replay_lsn() to check lag."
    row = await check_and_record(req, ans, llm, trajectory_id="t1", req_id="r1", home=tmp_path)
    assert row["verdict"] == "errors"
    kw = llm.chat_completion.call_args.kwargs
    assert kw.get("off_main_only") is True and kw.get("is_background") is True
    sent = llm.chat_completion.call_args.args[0]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}      # a thinking critic returned nothing
    assert await check_and_record(req, ans, llm, home=tmp_path) is None   # rate-limited
    lines = (tmp_path / "system" / "verifier" / "knowledge_shadow.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["issues"][0]["why"] == "no such function"


async def test_a_short_or_chit_chat_answer_is_not_checked(tmp_path):
    from ghost_agent.core import knowledge_shadow as KS
    from ghost_agent.core.knowledge_shadow import check_and_record
    KS._RUNNING["last"] = 0.0
    llm = MagicMock()
    llm.chat_completion = AsyncMock()
    assert await check_and_record("hi", "Hello! " * 40, llm, home=tmp_path) is None
    assert await check_and_record("what is the capital of peru", "Lima.", llm, home=tmp_path) is None
    llm.chat_completion.assert_not_called()


def test_only_declined_owner_turns_are_eligible(monkeypatch):
    from ghost_agent.core.agent import knowledge_shadow_eligible
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "user")
    t = SimpleNamespace(task_kind="user_request", tool_calls=[], outcome="unknown",
                        user_request="q", final_response="a")
    assert knowledge_shadow_eligible(object(), t)
    assert not knowledge_shadow_eligible(object(), SimpleNamespace(**{**t.__dict__, "tool_calls": [1]}))
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "sim")
    assert not knowledge_shadow_eligible(object(), t)
    monkeypatch.setattr(A, "turn_origin", lambda ctx: "user")
    monkeypatch.setenv("GHOST_KNOWLEDGE_SHADOW", "0")
    assert not knowledge_shadow_eligible(object(), t)


@pytest.mark.parametrize("rid,cap", [("39c394ca", 1800.0), ("probe-1", 1800.0), ("sched-x", 0.0),
                                     ("sub-y", 0.0), ("sim-z", 0.0), ("bench-1", 0.0)])
def test_a_conversation_request_gets_a_server_deadline(monkeypatch, rid, cap):
    from ghost_agent.utils import logging as L
    monkeypatch.delenv("GHOST_MAX_REQUEST_S", raising=False)
    assert L.server_request_cap_s(rid) == cap


def test_the_server_cap_drives_the_report_when_the_client_sent_none(monkeypatch):
    from ghost_agent.utils import logging as L
    from ghost_agent.core.agent import deadline_needs_report, effective_report_floor
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "1800")
    tok = L.client_deadline_context.set(0.0)
    try:
        monkeypatch.setattr(L, "request_elapsed_s", lambda rid: 1600.0)
        rem = L.request_remaining_s("39c394ca")
        assert rem == 200.0
        assert deadline_needs_report(rem, effective_report_floor(L.request_deadline_s("39c394ca")), False, False)
        monkeypatch.setenv("GHOST_MAX_REQUEST_S", "0")
        assert L.request_remaining_s("39c394ca") is None
    finally:
        L.client_deadline_context.reset(tok)


@pytest.mark.parametrize("msg,status", [
    ("progress report?", True), ("where are we with it?", True), ("πώς πάει;", True),
    ("is it done?", True), ("how's it going?", True),
    ("continue the build", False), ("build me a todo app", False),
    ("τι γίνεται με τον καιρό αύριο στην Αθήνα;", False), ("φτιάξε μια μπαρα προοδου στο UI", False),
    ("add a progress update endpoint to the API", False), ("Is it ready? if so deploy it to prod", False),
    ("are you done? then run the tests and push", False), ("write a status report generator script", False),
    ("how is it going to handle 10k users? benchmark it", False),
    ("where are we storing the logs? check the container", False)])
def test_status_questions_are_recognised(msg, status):
    from ghost_agent.core.agent import is_status_question
    assert is_status_question(msg) is status


async def test_a_status_question_with_live_work_starts_no_new_work():
    from ghost_agent.core.strikes import StrikeLedger
    from ghost_agent.utils.logging import request_id_context
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    calls = []

    async def ex(**kw):
        calls.append(kw)
        return "ran"
    agent.available_tools = {"execute": ex}
    tok = request_id_context.set("own-st")
    agent.context._status_only_req = "own-st"
    try:
        ts = H._ts([("execute", {"command": "make"})], StrikeLedger(), set())
        await agent._dispatch_and_process_tool_batch(ts)
    finally:
        request_id_context.reset(tok)
    assert not calls
    assert any("STATUS" in str(m.get("content")) for m in ts.messages if m.get("role") == "tool")


def test_a_torn_ledger_line_costs_only_itself(tmp_path):
    from ghost_agent.utils.json_store import open_append
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"a": 1}\n{"b": 2')                    # killed mid-append
    with open_append(p) as fh:
        fh.write(json.dumps({"c": 3}) + "\n")
    good = []
    for line in p.read_text().splitlines():
        try:
            good.append(json.loads(line))
        except ValueError:
            pass
    assert {"a": 1} in good and {"c": 3} in good


def test_the_corrections_ledger_survives_a_torn_line(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    col = TrajectoryCollector(tmp_path)
    p = col._corrections_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"trajectory_id": "x", "outcome": "succ')        # a killed append
    assert col.update_outcome("t-2", "failure", source="human:test") is True
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    assert any(r.get("trajectory_id") == "t-2" for r in rows)


def test_only_a_dead_writers_old_temps_are_swept(tmp_path):
    import os
    from ghost_agent.utils.json_store import sweep_orphan_temps
    old = time.time() - 7200
    dead = tmp_path / "a.json.999999.deadbeef.tmp"
    live = tmp_path / f"b.json.{os.getpid()}.cafebabe.tmp"
    fresh = tmp_path / "c.json.999998.feedface.tmp"
    other = tmp_path / "notes.tmp"
    for f in (dead, live, fresh, other):
        f.write_text("x")
    for f in (dead, live, other):
        os.utime(f, (old, old))
    assert sweep_orphan_temps(tmp_path) == 1
    assert not dead.exists() and live.exists() and fresh.exists() and other.exists()


@pytest.mark.parametrize("final,fits", [
    ("Lima is the capital of Peru, as listed in the CIA factbook.", True),
    ("[TURN BUDGET EXHAUSTED] I used all 40 reasoning turns; here is where things stand…", False),
    ("partial notes [ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit", False),
    ("", False),
])
def test_the_thumb_ask_skips_state_reports(final, fits):
    from ghost_agent.core.agent import thumb_ask_fits
    assert thumb_ask_fits(final, "\n\n---\n*This was one of the shakier answers*") is fits


def test_an_archive_that_does_not_verify_keeps_the_day(tmp_path, monkeypatch):
    import tarfile as T
    from ghost_agent.distill.collector import archive_old_partitions
    d = tmp_path / "2026-01-01"
    d.mkdir()
    (d / "session-s.jsonl").write_text("{}\n")
    real_open = T.open

    class Short:
        def __init__(self, tf): self.tf = tf
        def __enter__(self): return self
        def __exit__(self, *a): self.tf.close()
        def getnames(self): return ["2026-01-01"]          # the session file is missing
        def add(self, *a, **k): return self.tf.add(*a, **k)

    def fake_open(name, mode="r", *a, **k):
        tf = real_open(name, mode, *a, **k)
        return Short(tf) if mode.startswith("r") else tf
    monkeypatch.setattr(T, "open", fake_open)
    assert archive_old_partitions(tmp_path, 90, now=dt.date(2026, 10, 7)) == []
    assert d.exists() and not (tmp_path / "archive" / "2026-01-01.tar.gz").exists()


def test_the_rm_guard_is_installed_once_per_container(tmp_path):
    from ghost_agent.sandbox import docker as D
    sb = D.DockerSandbox.__new__(D.DockerSandbox)
    sb._privilege_checked = False
    sb.container = MagicMock()
    sb.container.attrs = {}
    calls = []
    sb._exec_run = lambda cmd, **kw: (calls.append((cmd, kw)), (0, b"installed"))[1]
    sb._settle_privileges_once()
    sb._settle_privileges_once()
    assert [c for c, kw in calls if c == D.SAFE_RM_INSTALL_CMD and kw.get("user") == "root"] == [D.SAFE_RM_INSTALL_CMD]


def _day(root, day, tid):
    d = root / day
    d.mkdir(parents=True, exist_ok=True)
    (d / f"session-{tid}.jsonl").write_text(json.dumps({"id": tid, "task_kind": "chat"}) + "\n")


def test_cumulative_readers_still_see_archived_days(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector, archive_old_partitions
    _day(tmp_path, "2026-06-01", "old")
    _day(tmp_path, "2026-10-06", "new")
    archive_old_partitions(tmp_path, 90, now=dt.date(2026, 10, 7))
    col = TrajectoryCollector(tmp_path)
    assert {t.id for t in col.iter_trajectories()} == {"new"}
    assert {t.id for t in col.iter_trajectories(include_archive=True)} == {"old", "new"}


def test_a_re_archived_day_does_not_overwrite_its_archive(tmp_path):
    from ghost_agent.distill.collector import archive_old_partitions
    _day(tmp_path, "2026-06-01", "first")
    archive_old_partitions(tmp_path, 90, now=dt.date(2026, 10, 7))
    _day(tmp_path, "2026-06-01", "restored")
    archive_old_partitions(tmp_path, 90, now=dt.date(2026, 10, 7))
    assert sorted(p.name for p in (tmp_path / "archive").glob("*.tar.gz")) == \
        ["2026-06-01.1.tar.gz", "2026-06-01.tar.gz"]


def test_a_future_dated_partition_does_not_hide_recent_history(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    today = dt.date.today()
    _day(tmp_path, (today - dt.timedelta(days=2)).isoformat(), "recent")
    _day(tmp_path, (today + dt.timedelta(days=400)).isoformat(), "skewed")
    ids = {t.id for t in TrajectoryCollector(tmp_path).iter_trajectories(since_days=30)}
    assert "recent" in ids


def test_the_router_probe_reads_date_partitions_only(tmp_path, monkeypatch):
    from ghost_agent.core import liveness as L
    root = tmp_path / "system" / "trajectories"
    for i, day in enumerate(("2026-10-05", "2026-10-06", "2026-10-07")):
        _day(root, day, f"t{i}")
    (root / "archive").mkdir()
    (root / "archive" / "2026-06-01.tar.gz").write_bytes(b"\x1f\x8b not text")
    opened = []
    real_open = Path.open

    def spy(self, *a, **k):
        opened.append(self)
        return real_open(self, *a, **k)
    monkeypatch.setattr(Path, "open", spy)
    L._trajectory_router_signal_probe(tmp_path)
    assert opened and all("archive" not in str(p) for p in opened)
    assert {p.parent.name for p in opened} == {"2026-10-05", "2026-10-06", "2026-10-07"}
