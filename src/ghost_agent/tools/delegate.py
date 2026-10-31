"""The ``jobs`` tool — the status surface for background work: list / check /
collect / cancel by id (:mod:`core.jobs`). (The module kept its name when the
``delegate`` sub-agent tool and the swarm were removed, 2026-10-30 §4SI.)

Since 2026-08-11 it also covers PROMOTED SANDBOX COMMANDS — an ``execute``
run that outlived its budget while still working is detached rather than
killed (:mod:`sandbox.jobs`) and appears here as a ``sandbox`` job. Those
live in a container, not in this process, so their rows carry no asyncio
task: :func:`_sync_sandbox_jobs` reconciles them from the supervisor
(landing finished ones, reaping expired ones) before any answer is composed,
and cancel routes to a real process-group kill.
"""

from __future__ import annotations

import asyncio
import logging

from ..core.jobs import (
    get_job_registry, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
    STATUS_CANCELLED,
)

logger = logging.getLogger("GhostAgent")

SANDBOX_KIND = "sandbox"
# How much of a promoted command's output is handed back when it lands. The
# registry caps results at MAX_RESULT_CHARS anyway; this bounds the LINES so
# a chatty downloader's progress bars can't crowd out its ending.
_SANDBOX_RESULT_LINES = 80


def _fmt_job(job, *, with_result: bool = False) -> str:
    line = (f"- {job.id} [{job.status}] {job.kind}: {job.label[:70]} "
            f"({job.duration_s:.0f}s)")
    if job.kind == SANDBOX_KIND and job.status == STATUS_RUNNING:
        line += ("\n    (a sandbox command that outran its execute budget "
                 "and was detached — still running; do NOT re-run it)")
    if job.error:
        line += f"\n    error: {job.error[:200]}"
    if with_result and job.result:
        line += f"\n    result: {job.result}"
    return line


def _sandbox_supervisor(context):
    """The sandbox job supervisor for this context, or None when there is no
    sandbox (docker absent, stub manager in a test)."""
    try:
        from ..sandbox.jobs import get_job_supervisor
        return get_job_supervisor(getattr(context, "sandbox_manager", None))
    except Exception:  # noqa: BLE001 — status must never fail over this
        logger.debug("sandbox job supervisor unavailable", exc_info=True)
        return None


def record_landings(context, changed) -> None:
    """Write each observed sandbox-job landing to the activity ledger.

    ONE caller since queue #9 (2026-08-21): ``main``'s periodic sweeper,
    driving off the durable ``reported_at`` marker. It used to be shared with
    this module's reconcile on the theory that "whoever observes a transition
    owns recording it" — but `reap()` hands each transition over exactly
    once, and this module's copy recorded WITHOUT waking, so a job landing
    during a tool call reached the ledger and then stranded (measured: 6
    landed jobs, 6 ledger rows from here, zero wakes ever). The ledger — not
    the operator's live log stream — is what answers "what happened while I
    was away" (``introspect action='activity'``); nothing re-engages the
    model after a turn ends, so a landing that is only logged is effectively
    invisible. Never raises."""
    for entry in changed or ():
        try:
            from ..utils.logging import is_probe_request_id
            if is_probe_request_id(str(entry.get("started_by") or "")):
                continue      # §4NS: a probe's job is not the owner's activity
        except Exception:  # noqa: BLE001
            pass
        try:
            from ..core.autonomous_activity import get_activity_log
            alog = get_activity_log(context)
            if alog is None:
                return
            alog.record(
                "job",
                f"sandbox job {entry.get('id')} {entry.get('state')}: "
                f"{str(entry.get('command'))[:120]}",
                job_id=str(entry.get("id") or ""),
                state=str(entry.get("state") or ""),
                exit_code=str(entry.get("exit_code")),
            )
        except Exception:  # noqa: BLE001 — bookkeeping never breaks a tool
            logger.debug("sandbox job activity record skipped", exc_info=True)



def _sync_sandbox_jobs(reg, context) -> None:
    """Reconcile ``sandbox``-kind rows against the container before answering.

    The registry row is a MIRROR: the process lives in the container and
    outlives this Python process, so nothing in-process can land it. This is
    where a finished job's exit code + output arrive, where the TTL is
    enforced (``reap``), and where a job promoted before an agent restart is
    adopted back into the in-memory view.
    """
    sup = _sandbox_supervisor(context)
    if sup is None:
        return
    try:
        from ..sandbox import jobs as sbx_jobs
        # Land the states; do NOT report here (queue #9, 2026-08-21).
        # Reporting used to be owned by whoever observed the transition,
        # because `reap()` hands it over exactly once — but this path only
        # ever recorded it, never WOKE the model, so a job landing during a
        # `jobs`/`execute` call reached the ledger and then stranded. The
        # transition is now marked durably at landing and drained by the
        # single reporter (main's sweeper, within one tick), which both
        # records and wakes. One reporter, so nothing is double-counted and
        # nothing depends on who raced to observe it.
        sup.reap()
        entries = {e.get("id"): e for e in sup.list_entries()}
    except Exception:  # noqa: BLE001
        logger.debug("sandbox job reap failed", exc_info=True)
        return

    # sid -> [rows]. Deliberately a LIST: `execute` registers a row and the
    # adoption below can create another for the same sid (a `jobs` call in
    # the same parallel tool batch). Keyed by a single row, the loser would
    # never be reconciled and would print "still running" for ever.
    mirrored = {}
    for job in reg.list(kind=SANDBOX_KIND):
        sid = (job.meta or {}).get("sandbox_job_id")
        if sid:
            mirrored.setdefault(sid, []).append(job)

    # Adopt jobs this process has never seen — promoted before a restart, or
    # by another entry point. Without this they would run (and be reaped)
    # invisibly.
    #
    # ⚠ ONLY RUNNING ONES, and only once. The in-process registry FIFO-evicts
    # completed rows at MAX_RETAINED across ALL job kinds, so a burst of
    # delegated sub-agents evicts landed sandbox rows whose supervisor entries
    # are still present — and adopting those would resurrect finished jobs as
    # fresh RUNNING rows, re-land them, un-mark them collected, and re-dump up
    # to 8 × 8000 chars of already-seen output on the next collect. Every
    # eviction would feed the next resurrection: a treadmill.
    for sid, entry in entries.items():
        if sid in mirrored or entry.get("state") != sbx_jobs.STATE_RUNNING:
            continue
        job = reg.register(
            SANDBOX_KIND,
            str(entry.get("label") or entry.get("command") or "")[:200],
            sandbox_job_id=sid, log=entry.get("log"),
            command=entry.get("command"), adopted=True,
            started_by=str(entry.get("started_by") or ""))
        # Age from when the COMMAND started, not from this adoption: a job
        # promoted half an hour before a restart otherwise reports "(0s)".
        try:
            job.created_at = float(entry.get("started_at") or job.created_at)
        except (TypeError, ValueError):
            pass
        mirrored.setdefault(sid, []).append(job)

    # A registry that reads as EMPTY is not evidence that every job died.
    # `_load` returns {} for any unparseable or non-dict payload, so one
    # transient bad read would declare every live promoted job permanently
    # dead in the model's view while its container process kept running.
    # "No rows at all" is missing information; "this sid is absent from a
    # registry that has other rows" is a real disappearance.
    # …but only for as long as the job COULD still be alive. A row that has
    # outlived the maximum possible lifetime is gone no matter what the
    # registry says, so this cannot hang RUNNING for ever.
    _blind = not entries
    _max_life = _max_job_lifetime_s(sbx_jobs)
    for sid, rows in mirrored.items():
        for job in rows:
            if job.status != STATUS_RUNNING:
                continue
            entry = entries.get(sid)
            if entry is None and _blind and job.duration_s < _max_life:
                logger.warning(
                    "sandbox job registry read empty — leaving %s RUNNING "
                    "(%.0fs old, max %.0fs) rather than declaring it gone",
                    sid, job.duration_s, _max_life)
                continue
            _land_sandbox_row(reg, sup, sbx_jobs, job, sid, entry)
    # NOTE: the supervisor's copy is deliberately NOT forgotten here.
    # `forget` deletes the log, and this sync runs on EVERY jobs call — so
    # forgetting at landing destroyed the full output before the model could
    # read it, leaving only the tail and turning the promotion banner's "your
    # live output is in .jobs/<id>.log" into a dangling path. The supervisor
    # registry bounds itself instead (`_trim_terminal`, 20 rows).


def _resolve_job(reg, job_id):
    """Find a job by EITHER id.

    ⚠ There are two `job-<8hex>` namespaces and they look identical: the
    in-process registry mints one, and the sandbox supervisor mints its own
    (carried in ``meta['sandbox_job_id']``). The promotion banner — the
    feature's entire user interface — quotes the SANDBOX id, so a plain
    ``reg.get(job_id)`` failed on exactly the id the model was told to use:
    `collect` answered "no job … it may have been evicted" and **`cancel`
    reported a kill that never happened**, leaving the container process
    running. Resolve both, preferring an exact registry-id hit.
    """
    jid = str(job_id or "")
    if not jid:
        return None
    job = reg.get(jid)
    if job is not None:
        return job
    for candidate in reg.list(kind=SANDBOX_KIND):
        if (candidate.meta or {}).get("sandbox_job_id") == jid:
            return candidate
    return None


def _max_job_lifetime_s(sbx_jobs) -> float:
    """The longest a promoted job can possibly still be running: the exec
    budget it survived, plus its TTL, plus the runner's ceiling margin. Past
    this, "the registry is empty" stops being an excuse."""
    try:
        from .execute import _EXEC_TIMEOUT_S
        return float(_EXEC_TIMEOUT_S) + sbx_jobs.job_ttl_s() + 600.0
    except Exception:  # noqa: BLE001
        return 4 * 3600.0


def _land_sandbox_row(reg, sup, sbx_jobs, job, sid, entry) -> None:
    """Move ONE mirrored row to its terminal state, carrying the job's exit
    code and output across. No-op while the job is still running."""
    if entry is None:
        # The supervisor dropped the row (registry trimmed, or a different
        # container generation). Nothing can be collected.
        reg.finish(job.id, status=STATUS_FAILED,
                   error="sandbox job record is gone (agent restart or "
                         "container recreate) — its output is no longer "
                         "retrievable")
        return
    state = entry.get("state")
    if state == sbx_jobs.STATE_RUNNING:
        return
    try:
        tail = sup.log_tail(sid, lines=_SANDBOX_RESULT_LINES)
    except Exception:  # noqa: BLE001
        tail = "(output unavailable)"
    code = entry.get("exit_code")
    try:
        # the stage-status marker line is ours, never the job's output —
        # stripped from a successful tail too (§4LP follow-up)
        from .execute import _take_pipestatus
        tail, _ = _take_pipestatus(tail)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(code, int) and code != 0:
        # §4LO: the same exit-code reading as a foreground run — a detached
        # `find / … 2>/dev/null` that printed results, or a `| head` that
        # closed the pipe early, is not "exited 1, failed"
        try:
            from .execute import _normalise_exit
            code = _normalise_exit(str(entry.get("command") or ""), code, tail)
        except Exception:  # noqa: BLE001
            pass
    log_hint = ""
    if entry.get("log"):
        log_hint = (f"\n[full output: {entry['log']} — file_system "
                    f"operation='read'/'search'; only the last "
                    f"{_SANDBOX_RESULT_LINES} lines are shown here]")
    if state == sbx_jobs.STATE_DONE:
        body = (f"[sandbox job {sid} finished — EXIT CODE: {code}]\n"
                f"$ {str(entry.get('command'))[:300]}\n"
                f"{tail}{log_hint}")
        if code == 0:
            reg.finish(job.id, status=STATUS_DONE, result=body)
        else:
            reg.finish(job.id, status=STATUS_FAILED,
                       error=f"exited {code}", result=body)
        return
    reason = {
        sbx_jobs.STATE_EXPIRED: "killed at its lifetime cap (it ran too long)",
        sbx_jobs.STATE_CANCELLED: "cancelled",
        sbx_jobs.STATE_LOST:
            "vanished (container recreated, or killed from outside)",
    }.get(state, str(state))
    reg.finish(job.id,
               status=(STATUS_CANCELLED if state == sbx_jobs.STATE_CANCELLED
                       else STATUS_FAILED),
               error=f"sandbox job {reason}",
               result=f"[sandbox job {sid} {state}]\n"
                      f"$ {str(entry.get('command'))[:300]}\n"
                      f"{tail}{log_hint}")


def _cancel_sandbox_job(job, context):
    """Kill the container-side process behind a mirrored row.

    Returns ``(killed, message)``. Cancelling only the registry row would
    leave the real process running — exactly the invisible orphan this whole
    subsystem exists to prevent — so the row is marked only on a True."""
    sid = (job.meta or {}).get("sandbox_job_id")
    sup = _sandbox_supervisor(context)
    if not sid or sup is None:
        return False, "no sandbox supervisor for this job"
    try:
        killed, message = sup.cancel(sid)
    except Exception:  # noqa: BLE001
        logger.warning("sandbox job cancel failed for %s", sid, exc_info=True)
        return False, "the supervisor raised while cancelling"
    # The BOOLEAN decides, never the message. Sniffing for an "Error:" prefix
    # read "already done — nothing to cancel" as a successful kill, and the
    # caller then overwrote a COMPLETED job's result with CANCELLED.
    return killed, message


async def tool_jobs(action: str = None, job_id=None, context=None, **kwargs):
    """Status surface for background work (promoted sandbox commands).

    Declares its own outcome. Its reports QUOTE the jobs they describe —
    including a finished job's `EXIT CODE: 1` and `[FAILED]` lines — and the
    dispatch loop's banner rule read those as the `jobs` call itself failing,
    drawing a strike every time the model re-read the same failed job. A
    successful READ of a failed job is a success.
    """
    from .outcome import ToolOutcome
    _res = await _tool_jobs_impl(action=action, job_id=job_id,
                                 context=context, **kwargs)
    if isinstance(_res, ToolOutcome):
        return _res
    _t = str(_res)
    if _t.lstrip().startswith(("Error:", "ERROR:")):
        return ToolOutcome.failed(_t, world_changed=False,
                                  reason_code="jobs_error")
    return ToolOutcome.ok(_t)


async def _tool_jobs_impl(action: str = None, job_id=None, context=None,
                          **kwargs):
    action = str(action or kwargs.get("operation") or "status").strip().lower()
    job_id = job_id or kwargs.get("id") or kwargs.get("job")
    if action in ("list", ""):
        action = "status"
    if action in ("result", "get", "fetch"):
        action = "collect"

    if context is None:
        return "Error: job registry unavailable (no agent context)."
    reg = get_job_registry(context)
    # Container-side jobs can only change state out-of-process, so every
    # action reads through a reconcile first — including cancel, which must
    # not kill a job that already finished.
    await asyncio.to_thread(_sync_sandbox_jobs, reg, context)

    # §4NS: a turn that is NOT the owner's (a probe, a scheduled run, a job's
    # wake, a sub-agent) sees, collects and cancels only the jobs IT started
    # — a probe or a wake turn could cancel the owner's running job (never
    # reported) and read the owner's output
    try:
        from ..utils.logging import request_id_context, request_kind
        _caller = str(request_id_context.get() or "")
        # no request context (a direct call, a test) keeps the old behaviour
        _owner_turn = (not _caller) or request_kind(_caller) in ("owner", "system")
    except Exception:  # noqa: BLE001
        _caller, _owner_turn = "", True

    def _mine(j) -> bool:
        if _owner_turn or str((j.meta or {}).get("started_by") or "") == _caller:
            return True
        # r1: a job's own WAKE turn (`job-<sid>`, `job-job-<sid>`,
        # `sub-job-<sid>` …) owns that job — it was refused its full result
        _sid = str((j.meta or {}).get("sandbox_job_id") or "")
        return bool(_sid) and _caller.endswith(_sid) and request_kind(_caller) in ("job", "background")

    def _mark_sandbox_delivered(jobs_) -> None:
        # §4NS: a sandbox job the model COLLECTED is delivered — its wake turn
        # reported it a second time (probe-d28bd3a4, 2026-09-20)
        sup = _sandbox_supervisor(context)
        if sup is None:
            return
        for j in jobs_:
            sid = (j.meta or {}).get("sandbox_job_id")
            if j.kind == SANDBOX_KIND and sid:
                try:
                    sup.mark_woken(sid)
                except Exception:  # noqa: BLE001
                    pass

    if action == "status":
        jobs = [j for j in reg.list() if _mine(j)]
        if not jobs:
            return "No background jobs."
        running = [j for j in jobs if j.status == STATUS_RUNNING]
        finished = [j for j in jobs if j.status != STATUS_RUNNING]
        out = []
        if running:
            out.append(f"RUNNING ({len(running)}):")
            out += [_fmt_job(j) for j in running]
        if finished:
            out.append(f"FINISHED ({len(finished)}):")
            out += [_fmt_job(j) for j in finished[-10:]]
            out.append("Use jobs(action='collect', job_id='...') to read a "
                       "finished job's full result.")
        return "\n".join(out)

    if action == "collect":
        if not job_id:
            # Collect ALL unread finished jobs — the common case. Returned
            # jobs are read-marked so a repeated collect-all does NOT re-dump
            # the same (up to 50 × 8000-char) results into context; an
            # explicit by-id collect can always re-fetch one.
            done = [j for j in reg.list(status=STATUS_DONE)
                    if j.result and not j.collected and _mine(j)]
            # §4O D-1: per-JOB result is capped (MAX_RESULT_CHARS) but the
            # BATCH count was not — a burst of finished jobs could dump up
            # to 50×8000=400KB into a pressure-sensitive context in one
            # tool result. Cap the batch; the rest stay unread for the next
            # collect (read-marking already prevents re-dumping).
            _COLLECT_BATCH_MAX = 8
            _overflow = 0
            if len(done) > _COLLECT_BATCH_MAX:
                _overflow = len(done) - _COLLECT_BATCH_MAX
                done = done[:_COLLECT_BATCH_MAX]
            if not done:
                seen = [j for j in reg.list(status=STATUS_DONE)
                        if j.result and j.collected and _mine(j)]
                if seen:
                    return (f"No NEW results — all {len(seen)} finished "
                            f"job(s) were already collected. Re-read one "
                            f"with jobs(action='collect', job_id='...') "
                            f"(e.g. {seen[-1].id}); jobs(action='status') "
                            f"shows what's still running.")
                return ("No finished jobs with results to collect "
                        "(jobs(action='status') shows what's running).")
            out = [f"Collected {len(done)} finished job(s):"]
            for j in done:
                out.append(f"\n--- {j.id}: {j.label[:70]} ---\n{j.result}")
            reg.mark_collected([j.id for j in done])
            await asyncio.to_thread(_mark_sandbox_delivered, done)
            if _overflow:
                out.append(f"\n\n[{_overflow} more finished job(s) not shown "
                           f"— call jobs(action='collect') again to read the "
                           f"next batch]")
            return "\n".join(out)
        job = _resolve_job(reg, job_id)
        if job is not None and not _mine(job):
            return f"Error: job {job_id!r} was started by another request — it is not yours to read."
        if job is None:
            return (f"Error: no job {job_id!r} (it may have been evicted — "
                    f"only the last {reg.max_retained} finished jobs are "
                    f"retained, and the registry does not survive an agent "
                    f"restart).")
        if job.status == STATUS_RUNNING:
            extra = ""
            if job.kind == SANDBOX_KIND:
                log = (job.meta or {}).get("log")
                extra = (" It is a detached sandbox command — do NOT re-run "
                         "it; finish your turn and report the result when it "
                         "lands."
                         + (f" Live output so far: {log} (file_system "
                            f"operation='read')." if log else ""))
            return (f"Job {job.id} is still RUNNING ({job.duration_s:.0f}s so "
                    f"far): {job.label[:80]}. Check back later.{extra}")
        if job.status == STATUS_DONE:
            # Explicit by-id fetch ALWAYS returns the result (intentional
            # re-fetch, even if already collected) — it only read-marks.
            reg.mark_collected([job.id])
            await asyncio.to_thread(_mark_sandbox_delivered, [job])
            return f"--- {job.id}: {job.label[:70]} ---\n{job.result or '(no output)'}"
        if job.result:
            # A FAILED/EXPIRED/CANCELLED job can still carry output, and for a
            # sandbox job that output is the whole diagnosis — a `npm run
            # build` that failed at minute 12 used to return only "exited 1",
            # leaving the model nothing to act on but the 12-minute command
            # it had just been told not to re-run.
            reg.mark_collected([job.id])
            await asyncio.to_thread(_mark_sandbox_delivered, [job])
            return (f"--- {job.id} [{job.status.upper()}]: {job.label[:70]} "
                    f"---\n{job.error or ''}\n{job.result}")
        return (f"Job {job.id} {job.status.upper()}: "
                f"{job.error or 'no error detail'}")

    if action == "cancel":
        if not job_id:
            return "Error: 'job_id' is required for cancel."
        job = _resolve_job(reg, job_id)
        if job is not None and not _mine(job):
            return (f"Error: job {job_id!r} was started by another request — only the owner's "
                    f"own turn may cancel it. Nothing was cancelled.")
        if job is not None and job.kind == SANDBOX_KIND \
                and job.status == STATUS_RUNNING:
            # Kill the container process FIRST: marking the row cancelled
            # while the process keeps running is the invisible-orphan bug.
            killed, why = await asyncio.to_thread(
                _cancel_sandbox_job, job, context)
            if not killed:
                return (f"Error: sandbox job {job_id} was NOT cancelled — "
                        f"{why} Nothing was killed and the job's own state "
                        f"is unchanged; jobs(action='status') re-checks it.")
            reg.finish(job.id, status=STATUS_CANCELLED,
                       error="cancelled (sandbox process killed)")
            return (f"Job {job_id} cancelled — the detached sandbox command "
                    f"was killed.")
        ok = reg.cancel(job.id) if job is not None else False
        return (f"Job {job_id} cancelled." if ok else
                f"Job {job_id} is not running (nothing to cancel).")

    return (f"Error: unknown action {action!r}. "
            f"Valid: status, collect, cancel.")


JOBS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "jobs",
        "description": (
            "Check on BACKGROUND WORK: LONG SANDBOX COMMANDS that outran the "
            "execute budget and were "
            "detached instead of killed (kind 'sandbox'). Shows what's still "
            "running, what finished, and what it produced. Call this when an "
            "execute call told you it was promoted to a job, or before re-doing work "
            "that may already be running. NEVER re-run a command that was "
            "promoted to a job — it was not killed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "collect", "cancel"],
                    "description": (
                        "status (default): list running + recently finished "
                        "jobs. collect: read a finished job's full result "
                        "(omit job_id to collect every unread result). "
                        "cancel: stop a running job."
                    ),
                },
                "job_id": {
                    "type": "string",
                    "description": "The job id (e.g. 'job-1a2b3c4d').",
                },
            },
            "required": ["action"],
        },
    },
}
