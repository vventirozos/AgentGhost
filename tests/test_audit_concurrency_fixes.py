"""Regression tests for the concurrency-race fixes:

* reflection.loop.Reflector.run: added traj.id to the shared dedup set
  only AFTER awaiting _reflect_one — a concurrent reflect_one() in that
  window double-reflected the same trajectory. Now marked before the await.
* core.project_advancer.advance_once: marked IN_PROGRESS only AFTER an
  `await llm_classifier(...)`, so two ticks claimed the same leaf. Now
  the leaf is claimed (read + IN_PROGRESS) atomically, before any await,
  under a per-project lock.
* sandbox.docker.DockerSandbox: ensure_running had no lock (concurrent
  to_thread calls raced container creation/provisioning); now a thin
  locking wrapper around _ensure_running_impl, plus 409 adoption.
"""

import asyncio
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.loop import Reflector, ReflectionOutcome
from ghost_agent.core.project_advancer import advance_once, _get_project_lock
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.sandbox.docker import DockerSandbox


# =================================================================
# reflection.loop — run() must claim the id BEFORE awaiting
# =================================================================

def _failed_traj(tid):
    return Trajectory(
        id=tid, user_request="do the thing", failure_reason="nope",
        outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name="execute", arguments={"code": "x"},
                             result="", error="SyntaxError")],
    )


async def test_run_marks_reflected_before_awaiting():
    async def _critique(_p):
        return "DIAGNOSIS: x\nREVISED PLAN:\n1. y"
    refl = Reflector(critique_fn=_critique)

    shared = set()
    seen_in_set = {}

    async def spy(traj):
        # At the instant _reflect_one runs, run() must have ALREADY added
        # the id to the shared set (claim-before-await), so a concurrent
        # reflect_one() would see it and skip.
        seen_in_set[traj.id] = traj.id in shared
        return ReflectionOutcome(source_trajectory_id=traj.id, error="spy")

    refl._reflect_one = spy
    await refl.run(failed_source=[_failed_traj("tX")], already_reflected=shared)

    # Claim-before-await: during _reflect_one the id was already in the set
    # (so a concurrent reflect_one would skip it).
    assert seen_in_set.get("tX") is True
    # ...but the spy returned an ERROR outcome, and a transient failure must
    # NOT permanently claim the trajectory — it's un-claimed after the await
    # so a later tick can retry it.
    assert "tX" not in shared


# =================================================================
# project_advancer — atomic leaf claim (no double-processing)
# =================================================================

def test_get_project_lock_is_per_project_and_reused():
    a1 = _get_project_lock("proj-A")
    a2 = _get_project_lock("proj-A")
    b = _get_project_lock("proj-B")
    assert a1 is a2
    assert a1 is not b
    assert isinstance(a1, type(threading.Lock()))


async def test_concurrent_advance_does_not_double_claim_leaf(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    context = SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        current_project_id=None,
    )
    pid = store.create_project("P")
    tid = store.add_task(pid, "Research the single leaf")

    runner_calls = []

    async def runner(name, args):
        runner_calls.append(name)
        return "results"

    async def slow_classifier(desc):
        # The await here used to be the preemption point where a second
        # tick grabbed the same not-yet-claimed leaf.
        await asyncio.sleep(0.05)
        return "research"

    r1, r2 = await asyncio.gather(
        advance_once(context, pid, tool_runner=runner, llm_classifier=slow_classifier),
        advance_once(context, pid, tool_runner=runner, llm_classifier=slow_classifier),
    )

    classifications = sorted([r1.classification, r2.classification])
    # Exactly one tick processed the leaf; the other found no READY leaf.
    assert "idle" in classifications, f"both ticks claimed the leaf: {classifications}"
    assert runner_calls.count("web_search") == 1, "leaf's tool ran more than once"
    assert store.get_task(tid)["status"] == "DONE"


# =================================================================
# docker — ensure_running locks, and adopts on a 409 name conflict
# =================================================================

def _docker_stub(workspace=None):
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = None
    sb.client = MagicMock()
    sb.image = "python:3.11-slim-bookworm"
    sb.container_name = "ghost-agent-sandbox-deadbeef"
    # A real path: ensure_running mkdir()s the host workspace (root-owned
    # bind-mount defense), and Path(MagicMock()) materialises a junk
    # ./MagicMock directory in the repo.
    sb.host_workspace = workspace if workspace is not None else Path(tempfile.mkdtemp(prefix="ghost-sb-test-"))
    sb.tor_proxy = None
    sb.docker_lib = MagicMock()
    sb._lock = threading.Lock()
    return sb


def test_ensure_running_holds_lock_during_impl():
    sb = _docker_stub()
    observed = {}

    def fake_impl():
        observed["locked"] = sb._lock.locked()

    sb._ensure_running_impl = fake_impl
    sb.ensure_running()
    assert observed.get("locked") is True, "impl must run while the lock is held"
    assert sb._lock.locked() is False, "lock must be released afterwards"


def test_ensure_running_adopts_container_on_409(monkeypatch):
    monkeypatch.setattr("ghost_agent.sandbox.docker.time.sleep", lambda *a, **k: None)
    sb = _docker_stub()

    class APIError(Exception):
        def __init__(self, msg, status_code=None):
            super().__init__(msg)
            self.status_code = status_code

    class NotFound(Exception):
        pass

    sb.APIError = APIError
    sb.NotFound = NotFound

    adopted = MagicMock(name="adopted_container")
    adopted.exec_run.return_value = (0, b"")  # marker test -f → present

    # containers.get: 1st (initial), 2nd (old-removal) → NotFound; 3rd (adopt) → adopted
    gets = [NotFound(), NotFound(), adopted]

    def _get(name):
        v = gets.pop(0) if gets else adopted
        if isinstance(v, Exception):
            raise v
        return v

    sb.client.containers.get.side_effect = _get
    sb.client.images.get.return_value = MagicMock()  # image present → no pull
    sb.client.containers.run.side_effect = APIError(
        "Conflict. The container name is already in use", status_code=409)

    # Skip the (now adopted) container's readiness/provisioning work.
    monkeypatch.setattr(sb, "_is_container_ready", lambda: True)
    monkeypatch.setattr(sb, "_chromium_binary_present", lambda: True)

    sb._ensure_running_impl()
    assert sb.container is adopted, "should adopt the existing container on a 409"
