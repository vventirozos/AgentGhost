"""Queue #9 — a job landing must reach the ledger AND wake the model, once.

`JobSupervisor.reap()` is the only writer of terminal states and returns each
transition EXACTLY ONCE, to whoever happened to call it. It has four callers
and three of them dropped that value:

  * `tools/delegate._sync_sandbox_jobs` recorded the landing but never woke;
  * `sandbox/jobs.register` and `.promote` discard it entirely, so a job that
    landed during an `execute` call was reported to nobody at all.

Measured on the live box 2026-08-21: **7 jobs, 6 landed, none collected, 6
ledger records written from the tool path, and ZERO wake turns in the whole
corpus** — for the loop that exists to stop a promoted command stranding
until the operator speaks again ("the half that makes promoting at 90s
instead of 600s safe").

The state change was always durable; only the NOTIFICATION was ephemeral.
These pins cover making the notification durable too.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
import time
from pathlib import Path

import pytest

from ghost_agent.sandbox import jobs as sbx_jobs


# ──────────────────────────────────────────────────────────────────────
# take_unreported
# ──────────────────────────────────────────────────────────────────────

class _FakeSandbox:
    """Minimal sandbox stand-in: the supervisor only needs a root for its
    registry in these pins (no command ever runs)."""

    def __init__(self, tmp_path):
        self.host_workspace = Path(tmp_path)
        self.host_workspace.mkdir(parents=True, exist_ok=True)
        self.root = tmp_path
        self.sandbox_dir = tmp_path
        self.calls = []

    def execute(self, cmd, timeout=600, workdir=None, **kwargs):
        self.calls.append(cmd)
        return "", 0

    def _exec_run(self, cmd, deadline_s=None, workdir=None, **kwargs):
        self.calls.append(cmd)
        return type("R", (), {"output": b"", "exit_code": 0})()


def _row(jid, state, **extra):
    """A registry row that SURVIVES `_load`'s validation — the key must be
    `job-<8 hex>`, `pid` an int > 1 and `deadline_at` numeric. The first
    draft of this file used `job-a` with neither field; the loader dropped
    every row and the drain looked broken when the fixture was."""
    row = {"id": jid, "state": state, "pid": 4242,
           "deadline_at": time.time() + 600.0,
           "command": "bash -c 'pytest'"}
    row.update(extra)
    return {jid: row}


def _sup(tmp_path, rows):
    """A REAL JobSupervisor whose registry is seeded through its own writer
    — no hand-rolled internals, so these pins cannot pass against a class
    that has moved on (the first version of this file constructed the object
    by hand, failed, and SKIPPED itself: four green-looking vacuous tests)."""
    sup = sbx_jobs.SandboxJobSupervisor(_FakeSandbox(tmp_path))
    sup._save(rows)
    return sup


class TestTakeUnreported:
    def test_a_fresh_landing_is_drained_exactly_once(self, tmp_path):
        sup = _sup(tmp_path, _row("job-1a2b3c4d", "done",
                                  exit_code=0, reported_at=None))

        first = sup.take_unreported()
        second = sup.take_unreported()

        assert [e["id"] for e in first] == ["job-1a2b3c4d"]
        assert second == []

    def test_a_LEGACY_row_without_the_key_is_treated_as_reported(
            self, tmp_path):
        """⚠ THE DEPLOY-SAFETY PROPERTY. Rows that landed before this shipped
        carry no `reported_at` key. If absence meant "unreported", the first
        drain after deploy would wake the model once per historical landing —
        six of them on the live box, against a cap of 12/hour."""
        sup = _sup(tmp_path, _row("job-0badcafe", "done", exit_code=0))

        assert sup.take_unreported() == []

    def test_a_still_running_job_is_never_drained(self, tmp_path):
        sup = _sup(tmp_path, _row("job-deadbeef", "running",
                                  reported_at=None))

        assert sup.take_unreported() == []

    def test_the_stamp_is_persisted_so_a_restart_cannot_re_report(
            self, tmp_path):
        """The whole point of moving the marker onto disk."""
        sup = _sup(tmp_path, _row("job-1a2b3c4d", "done",
                                  exit_code=0, reported_at=None))
        assert sup.take_unreported(), "precondition: the drain sees it once"

        # Re-read through a FRESH supervisor over the same root: the stamp
        # has to be on disk, not in the object that made it.
        reborn = sbx_jobs.SandboxJobSupervisor(_FakeSandbox(tmp_path))

        assert reborn.take_unreported() == []


class TestReapStampsNewTransitions:
    def test_every_terminal_transition_in_reap_marks_itself_unreported(self):
        """Source-level completeness: `reap()` has several terminal branches
        (done / lost / expired-by-log-cap / expired-by-TTL / late-exit), and
        one that forgets the stamp is a landing that can never be reported.
        Counted against `changed.append`, which every branch must also do."""
        src = Path(sbx_jobs.__file__).read_text()
        body = src.split("    def reap(self)", 1)[1].split("\n    def ", 1)[0]

        assert body.count("changed.append(entry)") >= 4
        assert (body.count('entry["reported_at"] = None')
                == body.count("changed.append(entry)"))


# ──────────────────────────────────────────────────────────────────────
# The sweeper: report AND wake, from the durable marker
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_sweeper_reports_and_wakes_from_the_DRAIN_not_from_reap():
    """The defect in one test: `reap()` returns nothing (another caller
    already observed the landing), and the model must still be woken."""
    import ghost_agent.main as main_mod
    import ghost_agent.sandbox.jobs as sbx

    recorded, woke = [], []

    class _Log:
        def record(self, phase, summary, **meta):
            recorded.append((phase, summary, meta))
            return True

    entry = {"id": "job-1a2b3c4d", "state": "done", "exit_code": 0,
             "command": "bash -c 'pytest'"}
    stamped = []

    class _Sup:
        def reap(self):
            return []                      # someone else got there first

        def take_unreported(self):
            return [entry]

        def pending_wakes(self):
            return [] if stamped else [entry]

        def mark_woken(self, jid):
            stamped.append(jid)
            return True

    class _Ctx:
        sandbox_manager = object()
        activity_log = _Log()

    ctx = _Ctx()
    orig_get = sbx.get_job_supervisor
    orig_every = main_mod._SANDBOX_JOB_REAP_EVERY_S
    orig_resume = main_mod._resume_after_job
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01

    async def _fake_resume(context, entry):
        woke.append(entry.get("id"))
        return True

    main_mod._resume_after_job = _fake_resume
    try:
        task = asyncio.ensure_future(main_mod._reap_sandbox_jobs(ctx))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if recorded and woke:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every
        main_mod._resume_after_job = orig_resume

    assert recorded, "the landing must reach the activity ledger"
    assert woke == ["job-1a2b3c4d"], "and must WAKE the model"


@pytest.mark.asyncio
async def test_nothing_unreported_means_no_ledger_row_and_no_wake():
    """The quiet path must stay quiet — an empty drain is the common case,
    every 60s, forever."""
    import ghost_agent.main as main_mod
    import ghost_agent.sandbox.jobs as sbx

    recorded, woke = [], []

    class _Sup:
        def reap(self):
            return [{"id": "job-x", "state": "done"}]   # landed, but...

        def take_unreported(self):
            return []                                   # ...already reported

        def pending_wakes(self):
            return []

        def mark_woken(self, jid):
            return True

    class _Ctx:
        sandbox_manager = object()
        activity_log = type("L", (), {
            "record": lambda self, *a, **k: recorded.append(a) or True})()

    orig_get = sbx.get_job_supervisor
    orig_every = main_mod._SANDBOX_JOB_REAP_EVERY_S
    orig_resume = main_mod._resume_after_job
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01

    async def _fake_resume(context, entry):
        woke.append(entry)
        return True

    main_mod._resume_after_job = _fake_resume
    try:
        task = asyncio.ensure_future(main_mod._reap_sandbox_jobs(_Ctx()))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every
        main_mod._resume_after_job = orig_resume

    assert not recorded and not woke


@pytest.mark.asyncio
async def test_the_sweeper_still_lands_states_every_tick():
    """Draining is not a substitute for reaping: without the `reap()` call
    nothing ever transitions, and the drain would stay empty for ever."""
    import ghost_agent.main as main_mod
    import ghost_agent.sandbox.jobs as sbx

    reaped = []

    class _Sup:
        def reap(self):
            reaped.append(True)
            return []

        def take_unreported(self):
            return []

        def pending_wakes(self):
            return []

        def mark_woken(self, jid):
            return True

    class _Ctx:
        sandbox_manager = object()
        activity_log = None

    orig_get = sbx.get_job_supervisor
    orig_every = main_mod._SANDBOX_JOB_REAP_EVERY_S
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01
    try:
        task = asyncio.ensure_future(main_mod._reap_sandbox_jobs(_Ctx()))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every

    assert reaped, "states must still be landed on every tick"


# ──────────────────────────────────────────────────────────────────────
# The internal-request contract
# ──────────────────────────────────────────────────────────────────────

class TestInternalPrefixContract:
    def test_every_dispatch_prefix_is_accounted_for(self):
        """The producers are `sched-`, `job-`, `sub-` and `bench-`. The first
        three are internal; `bench-` is deliberately NOT — bench turns must
        stay enrollable in bench-scoped experiments, and adding it here would
        silently kill `tts_bon`. Pinned so nobody "completes" the tuple."""
        from ghost_agent.core.autonomous_activity import (
            INTERNAL_REQUEST_PREFIXES, is_internal_request)

        assert set(INTERNAL_REQUEST_PREFIXES) == {"sched-", "job-", "sub-"}
        assert is_internal_request("sched-nightly")
        assert is_internal_request("job-1a2b3c4d")
        assert is_internal_request("sub-9f")
        assert not is_internal_request("bench-deadbeef")
        assert not is_internal_request("a1b2c3d4")      # an ordinary API turn

    def test_bench_turns_still_enroll(self):
        """The consequence that makes the exclusion load-bearing."""
        from ghost_agent.core.autonomous_activity import is_internal_request

        assert not is_internal_request("bench-0123456789")


# ──────────────────────────────────────────────────────────────────────
# R2: recording and WAKING fail differently, so they drain differently
# ──────────────────────────────────────────────────────────────────────

class TestWakeMarkerIsSeparate:
    def test_pending_wakes_does_NOT_stamp(self, tmp_path):
        """A wake DEFERS whenever a turn is in flight. Stamping on read
        consumed the deferred wake and lost it for ever — the very defect
        this pass fixes, re-created inside the fix (found in R2)."""
        sup = _sup(tmp_path, _row("job-1a2b3c4d", "done", exit_code=0,
                                  reported_at=None, woken_at=None))

        first = [e["id"] for e in sup.pending_wakes()]
        second = [e["id"] for e in sup.pending_wakes()]

        assert first == second == ["job-1a2b3c4d"]

    def test_mark_woken_removes_it_from_the_pending_set(self, tmp_path):
        sup = _sup(tmp_path, _row("job-1a2b3c4d", "done", exit_code=0,
                                  reported_at=None, woken_at=None))

        assert sup.mark_woken("job-1a2b3c4d") is True
        assert sup.pending_wakes() == []
        assert sup.mark_woken("job-1a2b3c4d") is False   # idempotent

    def test_recording_does_not_consume_the_WAKE(self, tmp_path):
        """The two markers are independent: the sweeper records on the same
        tick it wakes, and recording first must not swallow the wake."""
        sup = _sup(tmp_path, _row("job-1a2b3c4d", "done", exit_code=0,
                                  reported_at=None, woken_at=None))

        sup.take_unreported()

        assert [e["id"] for e in sup.pending_wakes()] == ["job-1a2b3c4d"]

    def test_a_legacy_row_owes_no_wake(self, tmp_path):
        """Deploy safety for the second marker too."""
        sup = _sup(tmp_path, _row("job-0badcafe", "done", exit_code=0))

        assert sup.pending_wakes() == []


@pytest.mark.asyncio
async def test_a_DEFERRED_wake_is_retried_on_the_next_tick():
    """The property R2 exists for. `_resume_after_job` returns False and
    leaves `_RESUMED_JOBS` untouched when it defers (a turn is in flight);
    the row must stay pending so the next tick tries again."""
    import ghost_agent.main as main_mod
    import ghost_agent.sandbox.jobs as sbx

    entry = {"id": "job-1a2b3c4d", "state": "done", "exit_code": 0,
             "command": "bash -c 'pytest'"}
    attempts, stamped = [], []

    class _Sup:
        def reap(self):
            return []

        def take_unreported(self):
            return []

        def pending_wakes(self):
            return [] if stamped else [entry]

        def mark_woken(self, jid):
            stamped.append(jid)
            return True

    class _Ctx:
        sandbox_manager = object()
        activity_log = None

    orig_get = sbx.get_job_supervisor
    orig_every = main_mod._SANDBOX_JOB_REAP_EVERY_S
    orig_resume = main_mod._resume_after_job
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01

    async def _deferring_resume(context, entry):
        attempts.append(entry.get("id"))
        return False                      # deferred: a turn is in flight

    main_mod._resume_after_job = _deferring_resume
    try:
        task = asyncio.ensure_future(main_mod._reap_sandbox_jobs(_Ctx()))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if len(attempts) >= 2:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every
        main_mod._resume_after_job = orig_resume

    assert len(attempts) >= 2, "a deferred wake must be retried"
    assert not stamped, "and must NOT be stamped while it is still owed"


@pytest.mark.asyncio
async def test_a_PERMANENTLY_declined_wake_is_stamped_and_not_retried():
    """The other half: the capped and already-woken paths record themselves
    in `_RESUMED_JOBS` before returning False. Without distinguishing them
    from a deferral, a capped job would be retried every 60s for ever."""
    import ghost_agent.main as main_mod
    import ghost_agent.sandbox.jobs as sbx

    entry = {"id": "job-1a2b3c4d", "state": "done", "exit_code": 0,
             "command": "bash -c 'pytest'"}
    stamped = []

    class _Sup:
        def reap(self):
            return []

        def take_unreported(self):
            return []

        def pending_wakes(self):
            return [] if stamped else [entry]

        def mark_woken(self, jid):
            stamped.append(jid)
            return True

    class _Ctx:
        sandbox_manager = object()
        activity_log = None

    orig_get = sbx.get_job_supervisor
    orig_every = main_mod._SANDBOX_JOB_REAP_EVERY_S
    orig_resume = main_mod._resume_after_job
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01

    async def _capped_resume(context, entry):
        main_mod._RESUMED_JOBS.add(str(entry.get("id")))   # what capping does
        return False

    main_mod._resume_after_job = _capped_resume
    try:
        task = asyncio.ensure_future(main_mod._reap_sandbox_jobs(_Ctx()))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if stamped:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every
        main_mod._resume_after_job = orig_resume
        main_mod._RESUMED_JOBS.discard("job-1a2b3c4d")

    assert stamped == ["job-1a2b3c4d"]
