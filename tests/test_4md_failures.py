"""§4MD — what happens when something breaks: behaviour pins."""
from __future__ import annotations

import errno
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.sandbox.docker import (_container_died_since as _D_died,
                                        _settle_readiness_after_exec as _D_settle)


# ── C1: a container restarted in place gets Tor-only egress again ──

def _sandbox(tmp_path, started="T1"):
    from ghost_agent.sandbox.docker import DockerSandbox
    cur = {"started": started}

    class C:
        id = "abc"
        status = "running"

        def __init__(self):
            self.attrs = {"State": {"StartedAt": started}, "HostConfig": {"NetworkMode": "bridge"},
                          "NetworkSettings": {"Networks": {"b": {}}}}

        def reload(self):
            self.attrs["State"]["StartedAt"] = cur["started"]

        def exec_run(self, cmd, **k):
            return (0, b"OK")
    d = DockerSandbox.__new__(DockerSandbox)
    d.container = C()
    d._lock = threading.RLock()
    d.host_workspace = tmp_path
    d.tor_proxy = "socks5://x"
    d._egress_state = ""
    d._env_verified = True
    d._privilege_checked = True
    d._cut_off = False
    d._cut_off_at = 0.0
    d._last_ready_ok = 0.0
    d.NotFound = KeyError
    d.container_name = "n"
    d._published_service_ports = set()
    d._exec_run = lambda cmd, **k: (0, b"OK")
    calls = []
    d._enforce_tor_egress = lambda: calls.append("ENFORCE")
    return d, cur, calls


def test_egress_is_reapplied_when_the_container_restarts_in_place(tmp_path):
    d, cur, calls = _sandbox(tmp_path)
    d._ensure_running_impl()
    assert calls == ["ENFORCE"]                  # first generation enforced
    d._last_ready_ok = 0.0
    d._ensure_running_impl()
    assert calls == ["ENFORCE"]                  # same generation: idempotent no-op
    cur["started"] = "T2"                        # `docker restart`: same id, new StartedAt
    d._last_ready_ok = 0.0
    d._ensure_running_impl()
    assert calls == ["ENFORCE", "ENFORCE"]


def test_an_unknown_generation_with_egress_marked_done_is_reapplied(tmp_path):
    d, cur, calls = _sandbox(tmp_path)
    d._tor_attempted = True                      # state from before generations were recorded
    d._ensure_running_impl()
    assert calls == ["ENFORCE"]


# ── M10: a container dying under a command is not "out of memory" ──

def test_a_container_that_died_during_the_command_is_reported_as_infra(tmp_path):
    d, cur, _ = _sandbox(tmp_path)
    t0 = time.time()
    d.container.status = "exited"
    assert _D_died(d, t0)
    d.container.status = "running"
    cur["started"] = "2020-01-01T00:00:00.000000000Z"
    assert not _D_died(d, t0)
    from datetime import datetime, timezone
    cur["started"] = datetime.fromtimestamp(t0 + 5, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    assert _D_died(d, t0)


async def test_the_oom_advice_is_not_given_for_an_infra_death(tmp_path):
    from ghost_agent.tools.execute import tool_execute
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=(
        "[SANDBOX INFRA ERROR] the sandbox container stopped or restarted while this command ran\n", 137))
    res = await tool_execute(command="python3 big.py", sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "SANDBOX INFRA ERROR" in res and "PEAK MEMORY" not in res and "out of memory" not in res
    mgr.execute = MagicMock(return_value=("Killed\n", 137))
    res = await tool_execute(command="python3 big.py", sandbox_dir=tmp_path, sandbox_manager=mgr)
    assert "PEAK MEMORY" in res                    # a real early 137 still gets the OOM advice


# ── M5: the log writer never raises ──

def test_a_full_disk_does_not_crash_the_logger(monkeypatch):
    import sys
    from ghost_agent.utils.logging import atomic_print

    class Full:
        def write(self, s):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(sys, "stdout", Full())
    atomic_print("a line")                      # no exception


# ── M6/M7: the scheduled-task store ──

@pytest.fixture
def task_store(tmp_path, monkeypatch):
    import ghost_agent.tools.tasks as T
    f = tmp_path / "tasks.json"
    monkeypatch.setattr(T, "task_store_path", str(f))
    T._STORE_STATE["unreadable"] = False
    yield T, f
    T._STORE_STATE["unreadable"] = False


def test_an_unreadable_task_store_is_not_overwritten(task_store, monkeypatch):
    T, f = task_store
    T._persist_task("task_a", "A", "p", "0 3 * * *")
    T._persist_task("task_b", "B", "p", "0 4 * * *")
    real = Path.read_text

    def eacces(self, *a, **k):
        if self == f:
            raise PermissionError(errno.EACCES, "denied")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", eacces)
    assert T._persist_task("task_c", "C", "p", "0 5 * * *") is False
    monkeypatch.setattr(Path, "read_text", real)
    assert set(json.loads(f.read_text())["tasks"]) == {"task_a", "task_b"}


async def test_a_task_that_could_not_be_saved_is_not_reported_as_success(task_store, monkeypatch):
    T, f = task_store
    import ghost_agent.utils.json_store as JS

    def full(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(JS, "write_json_atomic", full)
    monkeypatch.setattr(T, "run_proactive_task_fn", AsyncMock())
    sched = MagicMock()
    sched.get_jobs.return_value = []
    out = await T.tool_schedule_task("daily", "say hi", "0 9 * * *", sched, None)
    assert getattr(out, "status", None) is OutcomeStatus.PARTIAL and "lost at the next restart" in out


# ── M8: a forget whose archive failed says nothing was deleted ──

def test_a_forget_that_could_not_archive_says_so(tmp_path, monkeypatch):
    from ghost_agent.memory.episodes import EpisodicMemory
    em = EpisodicMemory(tmp_path)
    eid = em.record_episode("trigger", outcome="o", success=True)
    monkeypatch.setattr(em, "_archive_path", lambda: tmp_path / "no" / "such" / "dir" / "a.jsonl")
    assert em.delete_episodes([eid], None) == 0
    assert em.last_archive_failed is True
    from ghost_agent.tools import memory as M
    out = M._execute_item({"kind": "episode", "label": "episode 1", "ref": {"id": eid}, "target": "x"},
                          None, None, None, None, em)
    assert "Could NOT forget" in out and "already gone" not in out


# ── M9: a damaged job registry is set aside, never overwritten ──

def test_a_torn_job_registry_is_set_aside(tmp_path):
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor

    class SM:
        host_workspace = tmp_path
        container = None

        def execute(self, cmd, timeout=30, quiet=False):
            return ("", 1)
    s = SandboxJobSupervisor(SM())
    s._save({"job-0000aaaa": {"id": "job-0000aaaa", "pid": 4242, "deadline_at": time.time() + 999,
                              "state": "running", "command": "long build"}})
    p = s._registry_path
    p.write_text(p.read_text()[:30])                       # torn
    assert s._load() == {}
    kept = [x for x in p.parent.iterdir() if ".corrupt-" in x.name]
    assert kept and "job-0000aaaa" in kept[0].read_text()


def test_an_unreadable_job_registry_holds_every_save(tmp_path):
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    from ghost_agent.sandbox.registry_guard import registry_load_failed

    class SM:
        host_workspace = tmp_path
        container = None
    s = SandboxJobSupervisor(SM())
    err = ValueError("refusing")
    err.__cause__ = PermissionError(errno.EACCES, "denied")
    registry_load_failed(s._registry_path, err, s)
    with pytest.raises(RuntimeError):
        s._save({"job-0000bbbb": {"id": "job-0000bbbb", "pid": 1, "deadline_at": 1, "state": "running"}})


# ── M11: a graceful shutdown leaves live work running ──

def test_the_sandbox_is_kept_when_jobs_or_services_live_in_it(monkeypatch):
    import ghost_agent.main as MAIN
    import ghost_agent.sandbox.jobs as J
    import ghost_agent.sandbox.services as S
    monkeypatch.setattr(J, "get_job_supervisor",
                        lambda m: SimpleNamespace(list_entries=lambda: [{"state": "running"}]))
    monkeypatch.setattr(S, "get_service_supervisor", lambda m: SimpleNamespace(list_entries=lambda: []))
    assert MAIN._sandbox_has_live_work(object())[0] is True
    monkeypatch.setattr(J, "get_job_supervisor",
                        lambda m: SimpleNamespace(list_entries=lambda: [{"state": "done"}]))
    assert MAIN._sandbox_has_live_work(object())[0] is False


# ── M12: every claim at boot belongs to the previous process ──

def test_a_fresh_claim_is_reset_at_boot(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("P")
    tid = store.create_task(pid, "step") if hasattr(store, "create_task") else store.add_task(pid, "step")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE tasks SET status='IN_PROGRESS', updated_at=? WHERE id=?", (time.time() - 120, tid))
    assert store.reset_orphaned_in_progress(older_than_seconds=0.0) == 1


# ── M1/M3/M4: the model server failing ──

def _req():
    return httpx.Request("POST", "http://up/v1/chat/completions")


async def _run(ctx_value):
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import FakeBgTasks, make_context

    class Coll:
        def __init__(self):
            self.rows = []

        def append(self, t):
            self.rows.append(t)
    ctx = make_context()
    coll = Coll()
    ctx.trajectory_collector = coll
    agent = GhostAgent(ctx)
    if isinstance(ctx_value, Exception):
        ctx.llm_client.chat_completion = AsyncMock(side_effect=ctx_value)
    else:
        ctx.llm_client.chat_completion = AsyncMock(return_value=ctx_value)
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "what is the capital of peru?"}]}, FakeBgTasks())
    return str(out), [r.outcome for r in coll.rows], ctx.llm_client.chat_completion.await_count


@pytest.mark.parametrize("case,value", [
    ("refused", httpx.ConnectError("refused", request=_req())),
    ("503", httpx.HTTPStatusError("503", request=_req(),
                                  response=httpx.Response(503, text="Loading model /Users/x/secret.gguf",
                                                          request=_req()))),
])
async def test_an_upstream_failure_is_a_failed_turn_and_hides_internals(monkeypatch, case, value):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    out, outcomes, _ = await _run(value)
    from ghost_agent.core.agent import reply_carries_abort_marker, strip_abort_markers
    assert "secret.gguf" not in out
    assert reply_carries_abort_marker(out)               # booked as an abort …
    assert "[ATTEMPT_ABORTED" not in strip_abort_markers(out)   # … a member's copy never shows it
    assert outcomes and all(o != "success" for o in outcomes)


async def test_empty_replies_are_the_server_not_the_users_wording(monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    out, outcomes, calls = await _run({"choices": [{"message": {"content": ""}}]})
    assert "rephrase" not in out.lower()
    assert "model server" in out.lower()
    assert calls <= 3


async def test_an_aborted_turn_credits_no_lesson():
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import make_context
    ctx = make_context()
    sm = MagicMock()
    ctx.skill_memory = sm
    agent = GhostAgent(ctx)
    await agent._credit_turn_lessons(0, "what is the capital of peru",
                                     final_text="CRITICAL: Upstream error 503 [ATTEMPT_ABORTED_UPSTREAM]")
    sm.credit_recent_retrievals.assert_not_called()


def test_a_cut_answer_is_announced_and_marked():
    from ghost_agent.core.agent import TRUNCATED_ANSWER_NOTE, UPSTREAM_ABORT_MARKER, _ABORT_MARKER_RE
    assert _ABORT_MARKER_RE.search(UPSTREAM_ABORT_MARKER)
    assert "incomplete" in TRUNCATED_ANSWER_NOTE


# ── M13/M14: knowledge-base failures ──

async def test_a_failed_ingest_leaves_nothing_behind(tmp_path):
    from ghost_agent.tools import memory as M
    ms = MagicMock()
    ms.ingest_document.return_value = (False, "embedding endpoint down")
    ms.library_names = lambda: []
    calls = []
    ms.rollback_partial_document = lambda name: calls.append(name)
    M._rollback_partial(ms, "notes.md")
    assert calls == ["notes.md"]


def test_the_youtube_and_audio_paths_roll_back(tmp_path):
    import ghost_agent.memory.youtube_ingest as Y
    ms = MagicMock()
    ms.ingest_document.side_effect = [(True, "ok"), (False, "down")]
    rolled = []
    ms.rollback_partial_document = lambda n: rolled.append(n)
    with pytest.raises(RuntimeError):
        Y._store_passages(ms, "yt.md", [(float(i), float(i + 1), f"text {i}") for i in range(Y.BATCH_CHUNKS * 2)])
    assert rolled == ["yt.md"]


def test_a_failed_document_search_raises_instead_of_returning_nothing():
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)

    class Coll:
        def query(self, *a, **k):
            raise RuntimeError("chroma down")

    class L:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    vm.collection = Coll()
    vm._get_lock = lambda: L()
    with pytest.raises(RuntimeError):
        vm.search_document("doc.md", "question")


# ── MINORs ──

def test_tor_down_search_says_so(monkeypatch):
    from ghost_agent.tools import search as S
    assert S._proxy_reachable("socks5h://127.0.0.1:1") is False
    assert S._proxy_reachable(None) is True


def test_a_scratchpad_note_that_could_not_be_saved_says_so(tmp_path, monkeypatch):
    from ghost_agent.memory.scratchpad import Scratchpad
    pad = Scratchpad(persist_path=tmp_path / "s.db") if "persist_path" in Scratchpad.__init__.__code__.co_varnames else Scratchpad(tmp_path / "s.db")
    monkeypatch.setattr(pad, "_persist_entry", lambda *a, **k: False)
    assert "THIS session only" in pad.set("k", "v")


def test_a_missed_cron_fire_runs_once_after_boot(monkeypatch):
    import ghost_agent.tools.tasks as T
    if T.CronTrigger is None:
        pytest.skip("apscheduler missing")
    sched = MagicMock()
    monkeypatch.setattr(T, "run_proactive_task_fn", AsyncMock())
    now = time.time()
    # daily 09:00 UTC; last fired 25 h ago → today's 09:00 was missed if now is past it
    import datetime as dt
    today9 = dt.datetime.now(dt.timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
    if now < today9 + 60:
        today9 -= 86400
    rec = {"cron_expression": "0 9 * * *", "last_fire_ts": today9 - 86400 + 5, "prompt": "p", "task_name": "daily"}
    fired = T._schedule_missed_fire(sched, "task_x", rec, today9 + 600)
    assert fired and sched.add_job.call_args.kwargs["id"] == "task_x__catchup"
    sched.reset_mock()
    rec["last_fire_ts"] = today9 + 5                       # it ran
    assert T._schedule_missed_fire(sched, "task_x", rec, today9 + 600) is False


def test_a_damaged_offsets_file_keeps_the_other_consumers(tmp_path):
    from ghost_agent.core.autonomous_activity import save_consumer_offset
    f = tmp_path / "notify_consumers.json"
    f.write_text('{"slack": 10, "web"')
    save_consumer_offset(f, "web", 20)
    assert list(tmp_path.glob("notify_consumers.json.corrupt-*"))


def test_an_exec_137_from_a_dead_container_is_infra_and_unready(tmp_path):
    d, cur, _ = _sandbox(tmp_path)
    marks = []
    d.mark_ready = lambda: marks.append("ready")
    d.invalidate_ready = lambda: marks.append("invalid")
    t0 = time.time()
    d.container.status = "exited"
    out = _D_settle(d, 137, "partial\n", t0)
    assert out.startswith("[SANDBOX INFRA ERROR]") and marks == ["invalid"]
    d.container.status = "running"
    cur["started"] = "2020-01-01T00:00:00Z"
    marks.clear()
    assert _D_settle(d, 137, "Killed\n", t0) == "Killed\n" and marks == ["ready"]


async def test_a_thinking_only_reply_is_a_model_stall_not_an_outage(monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    out, _, _ = await _run({"choices": [{"message": {"content": "<think>The capital is Lima.</think>"}}]})
    assert "model server returned empty" not in out


def test_a_restart_inside_the_readiness_ttl_is_still_caught(tmp_path):
    d, cur, calls = _sandbox(tmp_path)
    d._ensure_running_impl()
    assert calls == ["ENFORCE"]
    d.mark_ready()                                # a command just succeeded: TTL fresh
    cur["started"] = "T2"
    d._ensure_running_impl()
    assert calls == ["ENFORCE", "ENFORCE"]


@pytest.mark.parametrize("v,expect", [
    ("2026-10-07T11:02:03Z", 1791370923.0), ("2026-10-07T11:02:03.5Z", 1791370923.5),
    ("2026-10-07T11:02:03.12345Z", 1791370923.12345), ("2026-10-07T11:02:03.123456789Z", 1791370923.123456789),
    ("2026-10-07T14:02:03.1+03:00", 1791370923.1), ("", 0.0),
])
def test_docker_times_parse_in_every_form(v, expect):
    from ghost_agent.sandbox.docker import _docker_ts
    assert abs(_docker_ts(v) - expect) < 1e-3


async def test_only_consecutive_empty_replies_end_the_turn(monkeypatch):
    """empty → a real reply → empty must not read as "the server is down"."""
    from ghost_agent.core.agent import GhostAgent
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent._upstream_empty_by_req = {"r-x": 1}
    replies = iter([{"choices": [{"message": {"content": "Lima is the capital of Peru."}}]}] * 5)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=lambda *a, **k: next(replies))
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "capital of peru?"}]},
                                        FakeBgTasks(), request_id="r-x")
    assert "Lima" in str(out)
    assert "r-x" not in agent._upstream_empty_by_req       # a real reply clears the count


def test_stopping_a_task_cancels_its_pending_catch_up(monkeypatch, tmp_path):
    import asyncio
    import ghost_agent.tools.tasks as T
    monkeypatch.setattr(T, "task_store_path", str(tmp_path / "t.json"))
    T._STORE_STATE["unreadable"] = False
    sched = MagicMock()
    sched.get_jobs.return_value = [SimpleNamespace(id="task_x", name="daily")]
    out = asyncio.run(T.tool_stop_task("daily", sched))
    removed = [c.args[0] for c in sched.remove_job.call_args_list]
    assert "task_x" in removed and "task_x__catchup" in removed and str(out).startswith("SUCCESS")


def test_a_non_dict_task_store_is_set_aside_not_held(tmp_path, monkeypatch):
    import ghost_agent.tools.tasks as T
    f = tmp_path / "tasks.json"
    f.write_text("null")
    monkeypatch.setattr(T, "task_store_path", str(f))
    T._STORE_STATE["unreadable"] = False
    assert T._load_task_store() == {} and T._STORE_STATE["unreadable"] is False
    assert list(tmp_path.glob("tasks.json.corrupt-*"))
    assert T._persist_task("task_a", "A", "p", "0 3 * * *") is True
