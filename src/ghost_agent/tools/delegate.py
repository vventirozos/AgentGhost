"""``delegate`` + ``jobs`` tools — tool-using sub-agents and job status.

``delegate`` spawns bounded, tool-using sub-agents (:mod:`core.subagent`)
that actually DO work — unlike ``delegate_to_swarm``, whose workers are
stateless single completions with no tools. ``jobs`` is the status surface
every fire-and-forget mechanism was missing: list / check / collect /
cancel background work by id (:mod:`core.jobs`).

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
from ..core.subagent import (
    resolve_allowed_tools, run_subagent, DEFAULT_ALLOWED_TOOLS,
    MAX_SUBAGENT_TURNS, DEFAULT_TIMEOUT_S,
)
from ..utils.logging import pretty_log, Icons, spawn_bg

logger = logging.getLogger("GhostAgent")

MAX_PARALLEL_DELEGATES = 4


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
            command=entry.get("command"), adopted=True)
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


async def tool_delegate(task=None, tasks=None, tools=None, wait: bool = False,
                        max_turns: int = MAX_SUBAGENT_TURNS,
                        timeout_s: float = DEFAULT_TIMEOUT_S,
                        context=None, **kwargs):
    """Spawn one or more tool-using sub-agents. Returns job ids immediately
    (``wait=false``, the default) or the finished results (``wait=true``)."""
    # --- PARAMETER HALLUCINATION HEALING ---
    task = task or kwargs.get("instruction") or kwargs.get("prompt")
    tasks = tasks or kwargs.get("subtasks")
    if tools is None:
        tools = kwargs.get("allowed_tools") or kwargs.get("tool_names")
    if isinstance(wait, str):
        wait = wait.strip().lower() in ("1", "true", "yes")

    if context is None:
        return "Error: delegation is unavailable (no agent context)."

    # Deadlock guard: a sub-agent's LLM calls are forced is_background=True
    # (core/subagent.py), and a background call targeting the main node parks
    # on _wait_for_foreground_clear() for as long as a user request is active
    # (core/llm.py). But wait=True blocks THIS request inside reg.wait() while
    # it still holds foreground_requests up — so the sub-agent parks on the
    # very request that is awaiting it. Result: 600s of dead air, then a
    # timed-out FAILED job. foreground_requests>0 means exactly "an interactive
    # turn is in flight" (it is 0 in idle/autoadvance contexts, where wait is
    # safe), so downgrade to fire-and-forget there: the jobs run once this turn
    # ends and the model collects them next turn.
    _fg = getattr(getattr(context, "llm_client", None), "foreground_requests", 0)
    _wait_downgraded = bool(wait) and _fg > 0
    if _wait_downgraded:
        wait = False

    task_list = []
    if isinstance(tasks, list) and tasks:
        for t in tasks:
            if isinstance(t, dict):
                t = t.get("task") or t.get("instruction") or ""
            if str(t).strip():
                task_list.append(str(t).strip())
    elif task and str(task).strip():
        task_list.append(str(task).strip())
    if not task_list:
        return ("Error: 'task' (or a non-empty 'tasks' list) is required — "
                "describe what the sub-agent should accomplish.")
    if len(task_list) > MAX_PARALLEL_DELEGATES:
        return (f"Error: at most {MAX_PARALLEL_DELEGATES} delegated tasks at "
                f"once (got {len(task_list)}) — they share one upstream LLM "
                f"slot, so more is slower, not faster.")

    allowed = resolve_allowed_tools(tools)
    if not allowed:
        return ("Error: no usable tools after filtering. Allowed: "
                + ", ".join(sorted(DEFAULT_ALLOWED_TOOLS)))

    try:
        max_turns = max(1, min(int(max_turns), MAX_SUBAGENT_TURNS))
    except (TypeError, ValueError):
        max_turns = MAX_SUBAGENT_TURNS
    try:
        timeout_s = max(10.0, min(float(timeout_s), 1800.0))
    except (TypeError, ValueError):
        timeout_s = DEFAULT_TIMEOUT_S

    reg = get_job_registry(context)
    jobs = []
    for t in task_list:
        job = reg.register("subagent", t, tools=len(allowed))
        coro = run_subagent(context, job_id=job.id, task=t,
                            allowed_tools=allowed, max_turns=max_turns,
                            timeout_s=timeout_s)
        # Register FIRST (the sub-agent names its sandbox dir after the job
        # id), then attach the task — the registry's done-callback lands the
        # result on it. spawn_bg (not a bare create_task) gives the strong
        # ref + contextvar propagation + exception logging the codebase
        # requires of every fire-and-forget site.
        reg.attach(job.id, spawn_bg(coro, name=f"delegate:{job.id}"))
        jobs.append(job)

    pretty_log("Delegate",
               f"spawned {len(jobs)} sub-agent(s) · tools={len(allowed)} · "
               f"wait={bool(wait)}",
               icon=Icons.NODE_WORKER)

    if not wait:
        if _wait_downgraded:
            lines = [f"Delegated {len(jobs)} task(s) to tool-using sub-agents. "
                     f"NOTE: wait=true is unavailable while serving an "
                     f"interactive turn — a sub-agent shares your single "
                     f"inference slot, so blocking on it here would deadlock "
                     f"(it can't run until this turn ends). Running them in the "
                     f"background instead:"]
        else:
            lines = [f"Delegated {len(jobs)} task(s) to tool-using sub-agents "
                     f"(running in the background):"]
        lines += [f"- {j.id}: {j.label[:70]}" for j in jobs]
        lines.append("")
        lines.append("They keep working while you do other things. Check with "
                     "jobs(action='status'), then jobs(action='collect', "
                     "job_id='...') for the answer. Do NOT re-do their work "
                     "yourself in the meantime.")
        return "\n".join(lines)

    await reg.wait([j.id for j in jobs], timeout=timeout_s + 30.0)
    out = [f"Delegated {len(jobs)} task(s); results:"]
    for j in jobs:
        fresh = reg.get(j.id) or j
        out.append(f"\n--- {fresh.id} [{fresh.status}] {fresh.label[:70]} ---")
        if fresh.status == STATUS_DONE:
            out.append(fresh.result or "(no output)")
        elif fresh.status == STATUS_RUNNING:
            out.append("(still running — collect it later with "
                       f"jobs(action='collect', job_id='{fresh.id}'))")
        else:
            out.append(f"FAILED: {fresh.error or 'unknown error'}")
    return "\n".join(out)


async def tool_jobs(action: str = None, job_id=None, context=None, **kwargs):
    """Status surface for background work (delegated sub-agents, swarm)."""
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

    if action == "status":
        jobs = reg.list()
        if not jobs:
            return ("No background jobs. Start one with "
                    "delegate(task='...').")
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
                    if j.result and not j.collected]
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
                        if j.result and j.collected]
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
            if _overflow:
                out.append(f"\n\n[{_overflow} more finished job(s) not shown "
                           f"— call jobs(action='collect') again to read the "
                           f"next batch]")
            return "\n".join(out)
        job = _resolve_job(reg, job_id)
        if job is None:
            return (f"Error: no job {job_id!r} (it may have been evicted — "
                    f"only the last {reg.max_retained} finished jobs are "
                    f"retained).")
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
            return f"--- {job.id}: {job.label[:70]} ---\n{job.result or '(no output)'}"
        if job.result:
            # A FAILED/EXPIRED/CANCELLED job can still carry output, and for a
            # sandbox job that output is the whole diagnosis — a `npm run
            # build` that failed at minute 12 used to return only "exited 1",
            # leaving the model nothing to act on but the 12-minute command
            # it had just been told not to re-run.
            reg.mark_collected([job.id])
            return (f"--- {job.id} [{job.status.upper()}]: {job.label[:70]} "
                    f"---\n{job.error or ''}\n{job.result}")
        return (f"Job {job.id} {job.status.upper()}: "
                f"{job.error or 'no error detail'}")

    if action == "cancel":
        if not job_id:
            return "Error: 'job_id' is required for cancel."
        job = _resolve_job(reg, job_id)
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


DELEGATE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Hand a self-contained SUB-TASK to a background sub-agent that has "
            "its own tools (web/browser/files/execute) and works autonomously "
            "while you continue. Use for parallel research legs or a "
            "long-running investigation whose details you don't need in your "
            "own context — you get back a summary, not the whole transcript. "
            "Different from delegate_to_swarm (whose workers just process text "
            "you already have, with NO tools). Returns job ids immediately; "
            "read results with the jobs tool. NOT for: trivial work (just do "
            "it), anything needing the user, or scheduling (use manage_tasks)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The sub-task, stated self-containedly — the sub-agent "
                        "cannot see your conversation or ask the user anything. "
                        "Say what to produce (e.g. 'Research X, write findings "
                        "to research_x.md, and report the 3 key points')."
                    ),
                },
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        f"Optional: up to {MAX_PARALLEL_DELEGATES} independent "
                        "sub-tasks to run in parallel instead of one."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional: restrict the sub-agent to these tools "
                        "(default: research + files + execute + browser). "
                        "It can never delegate, schedule, or write your memory."
                    ),
                },
                "wait": {
                    "type": "boolean",
                    "description": (
                        "If true, block until the sub-agent(s) finish and "
                        "return their results directly. Default false (returns "
                        "job ids; collect with the jobs tool)."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": (
                        f"Tool-turn budget per sub-agent (default/cap "
                        f"{MAX_SUBAGENT_TURNS})."
                    ),
                },
            },
            "required": ["task"],
        },
    },
}

JOBS_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "jobs",
        "description": (
            "Check on BACKGROUND WORK: delegated sub-agents, swarm tasks, and "
            "LONG SANDBOX COMMANDS that outran the execute budget and were "
            "detached instead of killed (kind 'sandbox'). Shows what's still "
            "running, what finished, and what it produced. Call this when you "
            "delegated something earlier and need the answer, when an execute "
            "call told you it was promoted to a job, or before re-doing work "
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
