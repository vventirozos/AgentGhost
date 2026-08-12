"""The `jobs` tool over PROMOTED SANDBOX COMMANDS.

A promoted command runs in the container, not in this process, so its
registry row is a mirror with no asyncio task behind it. Everything that
makes the mirror trustworthy is pinned here: the reconcile that lands a
finished job's exit code and output, adoption of a job promoted before this
process started, a cancel that kills the real process group BEFORE the row is
marked (a row-only cancel would leave exactly the invisible orphan the
subsystem exists to prevent), and the failure direction when a record is gone.
"""

import pytest

from ghost_agent.core.jobs import (
    JobRegistry, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED,
)
from ghost_agent.sandbox import jobs as sbx_jobs
from ghost_agent.tools import delegate as delegate_mod
from ghost_agent.tools.delegate import tool_jobs


class FakeSupervisor:
    def __init__(self, entries=None, tail="the output"):
        self.entries = list(entries or [])
        self.tail = tail
        self.reaped = 0
        self.cancelled = []
        self.forgotten = []
        self.cancel_result = (True, "Job cancelled.")

    def reap(self):
        self.reaped += 1
        return []

    def list_entries(self):
        return [dict(e) for e in self.entries]

    def log_tail(self, jid, lines=40):
        return self.tail

    def cancel(self, jid):
        self.cancelled.append(jid)
        return self.cancel_result

    def forget(self, jid):
        self.forgotten.append(jid)
        self.entries = [e for e in self.entries if e.get("id") != jid]


class Ctx:
    def __init__(self, reg):
        self.job_registry = reg
        self.sandbox_manager = object()


def _entry(jid="job-1a2b3c4d", state=sbx_jobs.STATE_RUNNING, code=None):
    return {"id": jid, "state": state, "exit_code": code,
            "command": "bash -c 'yt-dlp x'", "label": "yt-dlp x",
            "log": f".jobs/{jid}.log"}


@pytest.fixture
def wired(monkeypatch):
    """A registry + context + supervisor, with the supervisor lookup patched
    at its definition site (delegate imports it lazily inside the call)."""
    reg = JobRegistry()
    sup = FakeSupervisor()
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: sup)
    return reg, sup, Ctx(reg)


def _mirror(reg, jid="job-1a2b3c4d"):
    return reg.register("sandbox", "yt-dlp x", sandbox_job_id=jid,
                        log=f".jobs/{jid}.log", command="bash -c 'yt-dlp x'")


# ----------------------------------------------------------------- status

@pytest.mark.asyncio
async def test_status_lists_a_running_sandbox_job_and_warns_off_the_retry(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    job = _mirror(reg)
    out = await tool_jobs(action="status", context=ctx)
    assert job.id in out
    assert "sandbox" in out
    assert "do NOT re-run" in out
    assert sup.reaped == 1, "status must reconcile before answering"


@pytest.mark.asyncio
async def test_status_adopts_a_job_promoted_before_this_process(wired):
    """A promoted job outlives an agent restart; without adoption it would
    run (and be reaped) with nothing able to report it."""
    reg, sup, ctx = wired
    sup.entries = [_entry("job-000000aa")]
    out = await tool_jobs(action="status", context=ctx)
    adopted = reg.list(kind="sandbox")
    assert len(adopted) == 1
    assert adopted[0].meta["sandbox_job_id"] == "job-000000aa"
    assert adopted[0].meta["adopted"] is True
    assert adopted[0].id in out


@pytest.mark.asyncio
async def test_adoption_does_not_duplicate_an_existing_mirror(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    _mirror(reg)
    await tool_jobs(action="status", context=ctx)
    await tool_jobs(action="status", context=ctx)
    assert len(reg.list(kind="sandbox")) == 1


# ----------------------------------------------------------------- landing

@pytest.mark.asyncio
async def test_finished_job_lands_with_its_exit_code_and_output(wired):
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=0)]
    sup.tail = "[download] 100% — saved video.mp4"
    out = await tool_jobs(action="collect", context=ctx)
    landed = reg.get(job.id)
    assert landed.status == STATUS_DONE
    assert "EXIT CODE: 0" in landed.result
    assert "saved video.mp4" in landed.result
    assert "saved video.mp4" in out
    # The FULL log is NOT destroyed at landing. `forget` deletes it, and this
    # sync runs on every jobs call — so forgetting here left only the tail and
    # turned the promotion banner's ".jobs/<id>.log" into a dangling path.
    assert sup.forgotten == []
    assert ".jobs/job-1a2b3c4d.log" in landed.result, (
        "the landed result must point at the full log it did not inline")


@pytest.mark.asyncio
async def test_a_nonzero_exit_lands_as_failed_but_keeps_the_output(wired):
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=2)]
    sup.tail = "ERROR: unable to download"
    await tool_jobs(action="status", context=ctx)
    landed = reg.get(job.id)
    assert landed.status == STATUS_FAILED
    assert "exited 2" in landed.error
    assert "unable to download" in landed.result


@pytest.mark.asyncio
async def test_expired_job_says_it_hit_the_lifetime_cap(wired):
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = [_entry(state=sbx_jobs.STATE_EXPIRED)]
    await tool_jobs(action="status", context=ctx)
    landed = reg.get(job.id)
    assert landed.status == STATUS_FAILED
    assert "lifetime cap" in landed.error


@pytest.mark.asyncio
async def test_a_vanished_record_fails_loudly_rather_than_hanging_running(wired):
    """A sid absent from a registry that HAS other rows really is gone."""
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = [_entry("job-000000bb")]   # someone else's row, not ours
    await tool_jobs(action="status", context=ctx)
    landed = reg.get(job.id)
    assert landed.status == STATUS_FAILED
    assert "no longer retrievable" in landed.error


@pytest.mark.asyncio
async def test_an_empty_registry_read_does_not_mass_fail_live_jobs(wired):
    """`_load` returns {} for ANY unparseable payload, so one transient bad
    read would declare every live promoted job dead in the model's view while
    its container process kept running."""
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = []
    await tool_jobs(action="status", context=ctx)
    assert reg.get(job.id).status == STATUS_RUNNING


@pytest.mark.asyncio
async def test_but_an_impossibly_old_row_is_still_failed(wired):
    """…bounded: past the longest lifetime a job can have, "the registry is
    empty" stops being an excuse and the row must not hang RUNNING for ever."""
    import time
    reg, sup, ctx = wired
    job = _mirror(reg)
    job.created_at = time.time() - (10 * 3600)
    sup.entries = []
    await tool_jobs(action="status", context=ctx)
    landed = reg.get(job.id)
    assert landed.status == STATUS_FAILED
    assert "no longer retrievable" in landed.error


@pytest.mark.asyncio
async def test_a_landed_job_is_never_re_adopted(wired):
    """The in-process registry FIFO-evicts completed rows across ALL job
    kinds, so a burst of sub-agents evicts landed sandbox rows whose
    supervisor entries still exist. Re-adopting those resurrects finished
    jobs as RUNNING, re-lands them, un-marks them collected, and re-dumps
    already-seen output — and each eviction feeds the next resurrection."""
    reg, sup, ctx = wired
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=0)]
    await tool_jobs(action="status", context=ctx)      # nothing to adopt
    assert reg.list(kind="sandbox") == []
    # …and with a mirror present it lands once and stays landed.
    job = _mirror(reg)
    await tool_jobs(action="status", context=ctx)
    assert reg.get(job.id).status == STATUS_DONE
    await tool_jobs(action="status", context=ctx)
    assert len(reg.list(kind="sandbox")) == 1


@pytest.mark.asyncio
async def test_collect_by_id_on_a_running_sandbox_job_forecloses_the_retry(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    job = _mirror(reg)
    out = await tool_jobs(action="collect", job_id=job.id, context=ctx)
    assert "still RUNNING" in out
    assert "do NOT re-run" in out
    assert ".jobs/job-1a2b3c4d.log" in out


# ----------------------------------------------------------------- cancel

@pytest.mark.asyncio
async def test_cancel_kills_the_container_process(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    job = _mirror(reg)
    out = await tool_jobs(action="cancel", job_id=job.id, context=ctx)
    assert sup.cancelled == ["job-1a2b3c4d"], (
        "cancelling the row without killing the process leaves an orphan")
    assert reg.get(job.id).status == STATUS_CANCELLED
    assert "killed" in out


@pytest.mark.asyncio
async def test_a_failed_kill_leaves_the_row_running(wired):
    """Never report a cancel that did not happen — the process would still be
    holding CPU with the model believing it stopped."""
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    # The supervisor's BOOLEAN is what decides. Note the message carries no
    # "Error:" prefix — the shape that a message-sniffing caller read as a
    # successful kill, overwriting a completed job's result with CANCELLED.
    sup.cancel_result = (False, "the job had already finished.")
    job = _mirror(reg)
    out = await tool_jobs(action="cancel", job_id=job.id, context=ctx)
    assert out.startswith("Error:")
    assert "already finished" in out, "the caller must relay WHY"
    assert reg.get(job.id).status == STATUS_RUNNING


@pytest.mark.asyncio
async def test_cancelling_a_non_sandbox_job_is_unchanged(wired, monkeypatch):
    reg, sup, ctx = wired
    job = reg.register("subagent", "research something")
    out = await tool_jobs(action="cancel", job_id=job.id, context=ctx)
    assert sup.cancelled == []
    assert reg.get(job.id).status == STATUS_CANCELLED
    assert "cancelled" in out


# ------------------------------------------------------------- degradation

@pytest.mark.asyncio
async def test_no_sandbox_is_a_clean_no_op(monkeypatch):
    reg = JobRegistry()
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = Ctx(reg)
    ctx.sandbox_manager = None
    out = await tool_jobs(action="status", context=ctx)
    assert "No background jobs" in out


@pytest.mark.asyncio
async def test_a_raising_supervisor_does_not_break_status(wired, monkeypatch):
    reg, sup, ctx = wired
    _mirror(reg)

    def _boom():
        raise RuntimeError("docker daemon wedged")

    sup.reap = _boom
    out = await tool_jobs(action="status", context=ctx)
    assert "RUNNING" in out, "a docker hiccup must not fail the status surface"


@pytest.mark.asyncio
async def test_delegated_and_sandbox_jobs_share_one_surface(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    _mirror(reg)
    reg.register("subagent", "research the thing")
    out = await tool_jobs(action="status", context=ctx)
    assert "sandbox" in out and "subagent" in out


def test_execute_registers_the_kind_delegate_reads():
    """execute writes the kind string, delegate filters on it. A rename on
    one side alone would make every promoted job invisible here, and no other
    test would notice."""
    from ghost_agent.tools.execute import _register_promoted_job
    reg = JobRegistry()
    _register_promoted_job(reg, {"id": "job-11111111", "log": ".jobs/x.log",
                                 "command": "sleep 1"}, "label")
    assert [j.kind for j in reg.list()] == [delegate_mod.SANDBOX_KIND]
    assert reg.list(kind=delegate_mod.SANDBOX_KIND)


def test_registering_a_promotion_never_raises_on_a_broken_registry():
    class Hostile:
        def register(self, *a, **k):
            raise RuntimeError("nope")

    from ghost_agent.tools.execute import _register_promoted_job
    _register_promoted_job(Hostile(), {"id": "job-11111111"}, "label")
    _register_promoted_job(None, {"id": "job-11111111"}, "label")
    _register_promoted_job(JobRegistry(), "not-a-dict", "label")


@pytest.mark.asyncio
async def test_collect_returns_the_output_of_a_FAILED_sandbox_job(wired):
    """The case where diagnostics matter most. A build that failed at minute
    12 used to collect as bare "exited 1" — leaving the model nothing to act
    on but the 12-minute command it had just been told not to re-run."""
    reg, sup, ctx = wired
    job = _mirror(reg)
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=1)]
    sup.tail = "ERROR: ./src/app.ts(42,7): TS2322: Type 'string'…"
    await tool_jobs(action="status", context=ctx)
    out = await tool_jobs(action="collect", job_id=job.id, context=ctx)
    assert "TS2322" in out, "a failed job must still yield its output"
    assert "exited 1" in out


@pytest.mark.asyncio
async def test_two_mirror_rows_for_one_job_are_both_reconciled(wired):
    """`execute` registers a row and the adoption pass can create another for
    the same sandbox job (a `jobs` call in the same parallel tool batch).
    Keyed by a single row, the loser printed "still running" for ever."""
    reg, sup, ctx = wired
    first = _mirror(reg)
    second = _mirror(reg)          # the duplicate
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=0)]
    await tool_jobs(action="status", context=ctx)
    assert reg.get(first.id).status == STATUS_DONE
    assert reg.get(second.id).status == STATUS_DONE


@pytest.mark.asyncio
async def test_the_reaper_records_a_landed_job_in_the_activity_ledger():
    """The operator's live stream is WATCHED, not queried. Without a ledger
    entry, a job that landed while nobody was looking is visible nowhere but
    a `jobs` call the model has no reason to make — and `introspect
    action='activity'` is where "what happened while I was away" is
    answered."""
    import asyncio
    from ghost_agent.main import _reap_sandbox_jobs

    recorded = []

    class _Log:
        def record(self, phase, summary, **meta):
            recorded.append((phase, summary, meta))
            return True

    class _Sup:
        def reap(self):
            return [{"id": "job-1a2b3c4d", "state": "done", "exit_code": 0,
                     "command": "bash -c 'yt-dlp x'"}]

    class _Ctx:
        sandbox_manager = object()
        activity_log = _Log()

    ctx = _Ctx()
    import ghost_agent.sandbox.jobs as sbx
    import ghost_agent.main as main_mod
    orig_get, orig_every = sbx.get_job_supervisor, main_mod._SANDBOX_JOB_REAP_EVERY_S
    sbx.get_job_supervisor = lambda mgr: _Sup()
    main_mod._SANDBOX_JOB_REAP_EVERY_S = 0.01
    try:
        task = asyncio.ensure_future(_reap_sandbox_jobs(ctx))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if recorded:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        sbx.get_job_supervisor = orig_get
        main_mod._SANDBOX_JOB_REAP_EVERY_S = orig_every

    assert recorded, "a landed job must reach the activity ledger"
    phase, summary, meta = recorded[0]
    assert phase == "job"
    assert "job-1a2b3c4d" in summary and "done" in summary
    assert meta["exit_code"] == "0"


@pytest.mark.asyncio
async def test_a_job_landing_during_a_jobs_call_also_reaches_the_ledger(wired):
    """reap() returns the transitions IT observed and is the only writer of
    terminal states, so whoever observes a landing owns recording it.
    Discarding the return here made "every landed job reaches the ledger"
    false for any job that landed during a `jobs` call."""
    reg, sup, ctx = wired
    recorded = []

    class _Log:
        def record(self, phase, summary, **meta):
            recorded.append((phase, summary, meta))
            return True

    ctx.activity_log = _Log()
    sup.reap = lambda: [{"id": "job-1a2b3c4d", "state": "done",
                         "exit_code": 0, "command": "bash -c 'yt-dlp x'"}]
    sup.entries = [_entry(state=sbx_jobs.STATE_DONE, code=0)]
    _mirror(reg)
    await tool_jobs(action="status", context=ctx)
    assert recorded, "a landing observed here must reach the activity ledger"
    assert recorded[0][0] == "job"
    assert "job-1a2b3c4d" in recorded[0][1]


@pytest.mark.asyncio
async def test_recording_a_landing_never_breaks_the_tool(wired):
    reg, sup, ctx = wired

    class _Hostile:
        def record(self, *a, **k):
            raise RuntimeError("ledger is on fire")

    ctx.activity_log = _Hostile()
    sup.reap = lambda: [{"id": "job-1a2b3c4d", "state": "done",
                         "exit_code": 0, "command": "x"}]
    sup.entries = [_entry()]
    _mirror(reg)
    out = await tool_jobs(action="status", context=ctx)
    assert "job-" in out


# ------------------------------------------ the id the MODEL is given

@pytest.mark.asyncio
async def test_collect_and_cancel_accept_the_id_from_the_banner(wired):
    """⚠ TWO `job-<8hex>` namespaces exist and look identical: the in-process
    registry mints one, the sandbox supervisor mints another. The promotion
    banner — the feature's entire interface — quotes the SANDBOX id. A plain
    reg.get() therefore failed on exactly the id the model was told to use:
    collect answered "no job … may have been evicted" and cancel reported a
    kill that never happened. Every earlier test passed the REGISTRY id,
    which is why none of them caught it."""
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    row = _mirror(reg)                       # meta['sandbox_job_id'] = job-1a2b3c4d
    assert row.id != "job-1a2b3c4d", "the two ids must genuinely differ"

    out = await tool_jobs(action="collect", job_id="job-1a2b3c4d", context=ctx)
    assert "no job" not in out
    assert "still RUNNING" in out

    out = await tool_jobs(action="cancel", job_id="job-1a2b3c4d", context=ctx)
    assert sup.cancelled == ["job-1a2b3c4d"], (
        "cancel by the banner's id must reach the real process")
    assert reg.get(row.id).status == STATUS_CANCELLED


@pytest.mark.asyncio
async def test_the_registry_id_still_works_too(wired):
    reg, sup, ctx = wired
    sup.entries = [_entry()]
    row = _mirror(reg)
    out = await tool_jobs(action="collect", job_id=row.id, context=ctx)
    assert "still RUNNING" in out


@pytest.mark.asyncio
async def test_an_unknown_id_still_reports_not_found(wired):
    reg, sup, ctx = wired
    _mirror(reg)
    out = await tool_jobs(action="collect", job_id="job-99999999", context=ctx)
    assert out.startswith("Error: no job")


# ------------------------------------------------ auto-resume (2026-08-12)

class _ResumeCtx:
    """Context for the wake path: records the internal turn it triggers."""

    def __init__(self, foreground=0):
        self.turns = []
        self.sandbox_manager = object()
        self.args = type("A", (), {"model": "m"})()
        self.llm_client = type("L", (), {"foreground_requests": foreground})()
        ctx = self

        class _Agent:
            async def handle_chat(self, body, bg, request_id=None):
                ctx.turns.append((request_id, body["messages"][0]["content"]))
                return "ok", None, None

        self.agent = _Agent()


@pytest.fixture(autouse=True)
def _clear_resume_state():
    import ghost_agent.main as main_mod
    main_mod._RESUMED_JOBS.clear()
    main_mod._resume_times.clear()
    yield
    main_mod._RESUMED_JOBS.clear()
    main_mod._resume_times.clear()


def _landed(jid="job-1a2b3c4d", state="done", code=0):
    return {"id": jid, "state": state, "exit_code": code,
            "command": "bash -c 'pytest -q'"}


@pytest.mark.asyncio
async def test_a_landed_job_wakes_the_model_with_its_result(monkeypatch):
    """The half that makes promoting at 90s safe. Without it, an early
    promotion on a `pytest` ends the turn and STRANDS the work until the
    operator speaks again."""
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx()
    assert await main_mod._resume_after_job(ctx, _landed()) is True
    assert len(ctx.turns) == 1
    req_id, prompt = ctx.turns[0]
    assert req_id == "job-job-1a2b3c4d", (
        "the wake must be an INTERNAL turn — the `job-` prefix is what keeps "
        "it out of the operator digest and the smart-memory corpus")
    assert "has finished" in prompt and "EXIT CODE: 0" in prompt
    assert "pytest -q" in prompt
    assert "not a new request from the user" in prompt
    assert "rather than re-running the same long command" in prompt


@pytest.mark.asyncio
async def test_the_same_job_never_wakes_the_model_twice(monkeypatch):
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx()
    assert await main_mod._resume_after_job(ctx, _landed()) is True
    assert await main_mod._resume_after_job(ctx, _landed()) is False
    assert len(ctx.turns) == 1


@pytest.mark.asyncio
async def test_it_stands_down_while_a_user_request_is_in_flight(monkeypatch):
    """Turns are serialized on one semaphore, so waking here would make a
    live user queue behind autonomous work — the same rule scheduled tasks
    follow."""
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx(foreground=1)
    assert await main_mod._resume_after_job(ctx, _landed()) is False
    assert ctx.turns == []
    # …and it is NOT consumed: the next sweep may still wake it.
    ctx2 = _ResumeCtx(foreground=0)
    assert await main_mod._resume_after_job(ctx2, _landed()) is True


@pytest.mark.asyncio
async def test_autonomous_resumes_are_rate_capped(monkeypatch):
    """A woken turn can promote another job, which lands and wakes again.
    The counter is the backstop on that recursion."""
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx()
    for i in range(main_mod._MAX_RESUMES_PER_HOUR):
        assert await main_mod._resume_after_job(
            ctx, _landed(jid=f"job-0000{i:04x}")) is True
    assert await main_mod._resume_after_job(
        ctx, _landed(jid="job-ffffffff")) is False, "the cap must hold"
    assert len(ctx.turns) == main_mod._MAX_RESUMES_PER_HOUR


@pytest.mark.asyncio
async def test_a_failed_wake_never_kills_the_reaper(monkeypatch):
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx()

    class _Boom:
        async def handle_chat(self, *a, **k):
            raise RuntimeError("upstream is down")

    ctx.agent = _Boom()
    assert await main_mod._resume_after_job(ctx, _landed()) is False


@pytest.mark.asyncio
async def test_a_failed_job_wakes_with_its_diagnostics(monkeypatch):
    import ghost_agent.main as main_mod
    monkeypatch.setattr(sbx_jobs, "get_job_supervisor", lambda mgr: None)
    ctx = _ResumeCtx()
    await main_mod._resume_after_job(
        ctx, _landed(state="expired", code=None))
    _req, prompt = ctx.turns[0]
    assert "STATE: expired" in prompt
    assert "EXIT CODE" not in prompt, "no exit code exists for an expired job"
    assert "diagnose from the output" in prompt
