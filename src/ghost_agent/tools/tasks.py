import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

# APScheduler is optional. Production removed it in favour of the native
# asyncio biological_watchdog. Wrapping the import lets `pip uninstall
# apscheduler` not crash the whole tool registry — the user-cron tools
# below simply degrade to "scheduling disabled" when CronTrigger is None.
try:
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    CronTrigger = None

from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

# This will need to be bound to run_proactive_task from agent.py
run_proactive_task_fn = None

# Reactive-WATCH runner (2026-07-16), bound from main.py like the proactive
# runner. A watch task polls a shell CONDITION on an interval and fires its
# reaction prompt only when the condition first becomes true (edge-triggered),
# so the scheduler can react to external state (a log pattern, an endpoint
# going down, a threshold crossed) instead of only firing on a clock.
run_watch_condition_fn = None

# JSON file holding every user-scheduled task, bound by main.py at lifespan
# start (like run_proactive_task_fn above). The AsyncIOScheduler jobstore is
# IN-MEMORY, and the operator deploys by killing the agent — so before this
# store existed, every deploy silently WIPED all user cron tasks while the
# "task X is running" note written to vector memory kept asserting they were
# alive. Scheduled tasks run with nobody watching; a silent wipe is exactly
# the invisible-for-weeks failure class. None ⇒ persistence disabled
# (test/degraded contexts keep the old semantics).
task_store_path = None


def _load_task_store() -> dict:
    """Read the persisted task map ({job_id: {...}}). Best-effort: a missing
    or corrupt store reads as empty rather than raising."""
    if not task_store_path:
        return {}
    try:
        p = Path(task_store_path)
        if not p.is_file():
            _STORE_STATE["unreadable"] = False
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"not a task store: a JSON {type(data).__name__}")
        except (ValueError, UnicodeDecodeError) as je:
            # set it aside, or the next save keeps only the new task (§4LZ C1);
            # a valid-JSON non-dict (null, []) is damaged too (§4MD review: it
            # read as "unreadable" forever)
            from ..utils.json_store import preserve_corrupt
            preserve_corrupt(p, je, "scheduled-task store", is_valid=lambda d: isinstance(d, dict))
            _STORE_STATE["unreadable"] = False
            return {}
        tasks = data.get("tasks")
        _STORE_STATE["unreadable"] = False
        return tasks if isinstance(tasks, dict) else {}
    except Exception as e:  # noqa: BLE001
        # NOT empty: the file is there and could not be READ (EACCES, EIO).
        # Saving "the store plus one task" now would overwrite every other
        # task — the next save is refused until a read succeeds (§4MD M6)
        _STORE_STATE["unreadable"] = True
        logger.warning("scheduled-task store unreadable (%s) — saves are held until it reads again", e)
        return {}


#: the last load could not READ the store (not a decode error, which sets the
#: file aside): a save would overwrite tasks it never saw (§4MD M6)
_STORE_STATE = {"unreadable": False}


def _save_task_store(tasks: dict) -> bool:
    """Atomic write (tmp + os.replace) so a crash mid-save can't truncate
    the store. Never raises; returns False when the store was NOT written —
    the caller must say so (§4MD M7: a failed write replied SUCCESS and the
    task vanished at the next restart). The live scheduler state stays
    authoritative for this session."""
    if not task_store_path:
        return True
    if _STORE_STATE["unreadable"]:
        logger.warning("scheduled-task store not saved: it could not be read, and a save would drop its tasks")
        return False
    try:
        p = Path(task_store_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        from ..utils.json_store import write_json_atomic
        write_json_atomic(p, {"tasks": tasks})               # fsync (§4LZ C1)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("scheduled-task store write failed: %s", e)
        return False


def _persist_task(job_id: str, task_name: str, prompt: str,
                  cron_expression: str, kind: str = "task",
                  check_command: str = None) -> bool:
    tasks = _load_task_store()
    rec = {
        "task_name": task_name,
        "prompt": prompt,
        "cron_expression": cron_expression,
        "created_at": time.time(),
    }
    if kind == "watch":
        # A watch carries its condition command + the edge-trigger state so
        # a restart doesn't re-fire a condition that was already true.
        rec["kind"] = "watch"
        rec["check_command"] = check_command
        rec["last_fired"] = False
    tasks[job_id] = rec
    return _save_task_store(tasks)


def get_watch_record(job_id: str) -> dict:
    """The persisted record for a watch job (check_command, prompt,
    last_fired), or {} if absent. Read by the watch runner each tick."""
    return _load_task_store().get(job_id) or {}


def set_watch_state(job_id: str, last_fired: bool) -> None:
    """Persist a watch's edge-trigger state so it fires only on the
    transition to true and survives a restart."""
    tasks = _load_task_store()
    if job_id in tasks:
        tasks[job_id]["last_fired"] = bool(last_fired)
        _save_task_store(tasks)


def _unpersist_task(job_id: str) -> bool:
    """False when the removal could NOT be saved — the task comes back at the
    next restart, and the caller must say so (§4MD review)."""
    tasks = _load_task_store()
    if job_id in tasks:
        del tasks[job_id]
        return _save_task_store(tasks)
    return not _STORE_STATE["unreadable"]


def _unpersist_all() -> bool:
    if task_store_path:
        return _save_task_store({})
    return True


def _add_job(scheduler, job_id: str, task_name: str, prompt: str,
             cron_expression: str, kind: str = "task",
             check_command: str = None, anchor_ts: float = None):
    """Register one job on the scheduler. Returns an error STRING on a
    rejected/malformed schedule, None on success. Shared by the create tool
    and the boot-time restore so both interpret expressions identically."""
    if kind == "watch":
        # A watch polls its condition on an interval; the watch runner reads
        # check_command + edge state from the store by job_id each tick.
        if run_watch_condition_fn is None:
            return "Error: the watch runner is not initialized in this context."
        if not str(cron_expression).startswith("interval:"):
            return ("Error: a watch must use 'interval:SECONDS' (it POLLS the "
                    "condition) — cron schedules are for time-based tasks.")
        try:
            secs = int(cron_expression.split(":", 1)[1].strip())
        except (IndexError, ValueError):
            return (f"Error: malformed watch interval '{cron_expression}'. "
                    "Use 'interval:SECONDS', e.g. 'interval:60'.")
        if secs < 10:
            return "Error: watch interval must be >= 10 seconds (don't hammer the check)."
        scheduler.add_job(
            run_watch_condition_fn, 'interval', seconds=secs,
            args=[job_id], id=job_id, name=task_name, replace_existing=True,
            misfire_grace_time=300, coalesce=True,
        )
        return None
    if cron_expression.startswith("interval:"):
        parts = cron_expression.split(":")
        raw = parts[1].strip() if len(parts) > 1 else ""
        try:
            secs = int(raw)
        except ValueError:
            # Reject rather than silently run every 60s while reporting
            # SUCCESS with the original (wrong) expression — the agent
            # would believe "interval:5m" fires every 5 minutes when it
            # actually fired every minute.
            return (
                f"Error: malformed interval schedule '{cron_expression}'. "
                "Use 'interval:SECONDS' with an integer, e.g. 'interval:300' "
                "for every 5 minutes."
            )
        if secs <= 0:
            return f"Error: interval must be a positive number of seconds, got {secs}."
        # Fresh review (§4KW): restored with no start date, each restart put
        # the next run a FULL interval after boot — a daily task on a box that
        # restarts more often than daily never fired, while `list` showed a
        # future "Next Run". Anchored at creation, the cadence survives a
        # restart (next run = created_at + k·interval after now).
        _anchor = {}
        if anchor_ts:
            try:
                import datetime as _dt
                _anchor = {"start_date": _dt.datetime.fromtimestamp(float(anchor_ts), tz=_dt.timezone.utc)}
            except (TypeError, ValueError, OverflowError, OSError):
                _anchor = {}
        scheduler.add_job(
            run_proactive_task_fn,
            'interval',
            seconds=secs,
            **_anchor,
            args=[job_id, prompt],
            id=job_id,
            name=task_name,
            replace_existing=True,
            # APScheduler's default misfire grace is 1s — a fire landing in
            # any >1s event-loop stall (prompt assembly, sync memory work)
            # was silently SKIPPED with nothing in the activity ledger.
            misfire_grace_time=300,
            coalesce=True,
        )
        return None
    if CronTrigger is None:
        return "Error: cron-style schedules require apscheduler. Use 'interval:SECONDS' instead."
    scheduler.add_job(
        run_proactive_task_fn,
        # timezone MUST be explicit: a pre-built trigger instance never
        # inherits the scheduler's UTC — from_crontab() alone defaults to
        # the LOCAL zone, so every cron task fired hours off the UTC time
        # the tool contract tells the model to convert to (and drifted
        # with DST).
        CronTrigger.from_crontab(cron_expression, timezone="UTC"),
        args=[job_id, prompt],
        id=job_id,
        name=task_name,
        replace_existing=True,
        misfire_grace_time=300,
        coalesce=True,
    )
    return None


def note_task_fired(job_id: str, ts: float) -> None:
    """Record when a scheduled task last fired (§4MD MINOR 7)."""
    tasks = _load_task_store()
    rec = tasks.get(job_id)
    if isinstance(rec, dict):
        rec["last_fire_ts"] = float(ts)
        _save_task_store(tasks)


#: how far back a missed cron fire is still caught up at boot
CATCHUP_WINDOW_S = 6 * 3600


def _schedule_missed_fire(scheduler, job_id: str, rec: dict, now: float) -> bool:
    """A cron task whose fire fell inside a restart was skipped until its next
    slot — a daily task due during a one-minute deploy waited a day (§4MD
    MINOR 7). Run it ONCE, a minute after boot, when the missed fire is
    recent; older misses are left to the schedule."""
    expr = str(rec.get("cron_expression") or "")
    if CronTrigger is None or not expr or expr.startswith("interval:") or rec.get("kind") == "watch":
        return False
    last = rec.get("last_fire_ts") or rec.get("created_at")
    if not last:
        return False
    import datetime as _dt
    trig = CronTrigger.from_crontab(expr, timezone="UTC")
    prev = _dt.datetime.fromtimestamp(float(last), tz=_dt.timezone.utc)
    nxt = trig.get_next_fire_time(None, prev + _dt.timedelta(seconds=1))
    if nxt is None:
        return False
    missed = nxt.timestamp()
    if not (missed < now and now - missed <= CATCHUP_WINDOW_S):
        return False
    scheduler.add_job(
        run_proactive_task_fn, 'date',
        run_date=_dt.datetime.fromtimestamp(now + 60, tz=_dt.timezone.utc),
        args=[f"{job_id}__catchup", str(rec.get("prompt") or "")],
        id=f"{job_id}__catchup", name=f"{rec.get('task_name') or job_id} (missed run)",
        replace_existing=True, misfire_grace_time=300)
    logger.info("scheduled task %s missed its %s fire during downtime — running it once now",
                job_id, nxt.isoformat())
    return True


def restore_persisted_tasks(scheduler) -> int:
    """Re-register every persisted task on a fresh scheduler at boot.
    Returns the number restored. A malformed record is skipped with a
    warning (and dropped from the store) rather than aborting the rest —
    one rotten task must not take down every other schedule."""
    if not scheduler or run_proactive_task_fn is None:
        return 0
    tasks = _load_task_store()
    if not tasks:
        return 0
    restored = 0
    dropped = []
    for job_id, rec in tasks.items():
        try:
            err = _add_job(
                scheduler, job_id,
                str(rec.get("task_name") or job_id),
                str(rec.get("prompt") or ""),
                str(rec.get("cron_expression") or ""),
                kind=str(rec.get("kind") or "task"),
                check_command=rec.get("check_command"),
                anchor_ts=rec.get("created_at"),
            )
            if err:
                raise ValueError(err)
            restored += 1
            try:
                _schedule_missed_fire(scheduler, job_id, rec, time.time())
            except Exception as _ce:  # noqa: BLE001 — a catch-up never blocks a restore
                logger.debug("catch-up check for %s skipped: %s", job_id, _ce)
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping persisted task %s (%s): %s",
                           job_id, rec.get("task_name"), e)
            dropped.append(job_id)
    if dropped:
        for j in dropped:
            tasks.pop(j, None)
        _save_task_store(tasks)
    if restored:
        pretty_log(
            "Scheduled Tasks Restored",
            f"{restored} user task(s) re-registered from the persistent store"
            + (f"; {len(dropped)} malformed record(s) dropped" if dropped else ""),
            icon=Icons.BRAIN_PLAN,
        )
    return restored


def should_defer_scheduled_task(llm_client) -> bool:
    """True when a scheduled (idle-time autonomous) task should skip THIS
    firing because a live user request is in flight.

    Turns are serialized (agent_semaphore == 1, see core.agent #22), so a
    scheduled job dispatched now would queue against the user's turn. These
    jobs are idle-time work and the scheduler re-fires them on the next tick,
    so skipping is strictly better than making a user wait. Only a REAL
    positive int defers — a missing attr or a mocked client (MagicMock) reads
    as "no user active" so tests and partial contexts proceed."""
    cur = getattr(llm_client, "foreground_requests", 0)
    return isinstance(cur, int) and cur > 0

async def tool_schedule_task(task_name: str, prompt: str, cron_expression: str, scheduler, memory_system):
    pretty_log("Task Schedule", f"Name: {task_name} | Expr: {cron_expression}", icon=Icons.BRAIN_PLAN)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    if run_proactive_task_fn is None:
        return "Error: Proactive task runner not initialized."
        
    try:
        job_id = f"task_{hashlib.md5(str(task_name).encode()).hexdigest()[:10]}"

        err = _add_job(scheduler, job_id, task_name, prompt, cron_expression)
        if err:
            return err

        # Persist AFTER the live registration succeeds, so the store only
        # ever holds tasks that were actually schedulable. Survives agent
        # restarts (the jobstore itself is in-memory); best-effort like the
        # memory note below.
        _saved = _persist_task(job_id, task_name, prompt, cron_expression)

        # The job is already scheduled at this point. A failure to write the
        # bookkeeping memory entry must NOT be reported as a scheduling
        # failure (the task WOULD still fire) — isolate it so the outcome we
        # return matches the real scheduler state.
        memory_entry = f"Scheduled task '{task_name}' is running with ID {job_id} on schedule {cron_expression}."
        if memory_system:
            try:
                # §4M (Lens C MINOR): "manual" is a prunable type and the
                # eviction tie-break sorts a MISSING timestamp as oldest —
                # a timestamp-less note is the first eviction victim.
                from ..utils.helpers import get_utc_timestamp as _uts
                await asyncio.to_thread(
                    memory_system.add, memory_entry,
                    {"type": "manual", "task_id": job_id,
                     "timestamp": _uts()})
            except Exception as mem_err:
                pretty_log("Schedule Memory", f"note write failed (task still scheduled): {mem_err}",
                           level="WARNING", icon=Icons.WARN)

        if not _saved:
            from .outcome import ToolOutcome
            return ToolOutcome.partial(
                f"PARTIAL: Task '{task_name}' is scheduled for THIS session (ID: {job_id}) but could NOT "
                f"be saved — it will be lost at the next restart. Tell the user.",
                reason_code="task_not_persisted")
        return f"SUCCESS: Task '{task_name}' scheduled (ID: {job_id})."
    except Exception as e:
        pretty_log("Schedule Error", str(e), level="ERROR", icon=Icons.FAIL)
        return f"ERROR: {e}"

async def tool_watch_condition(task_name: str, check_command: str,
                               reaction_prompt: str, interval_secs,
                               scheduler, memory_system):
    """Register a reactive WATCH: poll ``check_command`` every
    ``interval_secs``; when it first SUCCEEDS (exit 0 — shell ``if``
    semantics), fire ``reaction_prompt`` as a background agent turn with the
    check's output attached. Edge-triggered (fires on the transition to true,
    not every tick it stays true)."""
    pretty_log("Watch Register",
               f"Name: {task_name} | every {interval_secs}s | check: {str(check_command)[:60]}",
               icon=Icons.BRAIN_PLAN)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    if run_watch_condition_fn is None:
        return "Error: the watch runner is not initialized."
    if not (task_name and check_command and reaction_prompt):
        return "Error: watch requires task_name, check_command, and prompt (the reaction)."
    try:
        secs = int(interval_secs)
    except (TypeError, ValueError):
        return "Error: interval_secs must be an integer number of seconds (e.g. 60)."
    try:
        job_id = f"watch_{hashlib.md5(str(task_name).encode()).hexdigest()[:10]}"
        cron = f"interval:{secs}"
        err = _add_job(scheduler, job_id, task_name, reaction_prompt, cron,
                       kind="watch", check_command=check_command)
        if err:
            return err
        _saved = _persist_task(job_id, task_name, reaction_prompt, cron,
                               kind="watch", check_command=check_command)
        if memory_system:
            try:
                from ..utils.helpers import get_utc_timestamp as _uts
                await asyncio.to_thread(
                    memory_system.add,
                    f"Watch '{task_name}' (ID {job_id}) polls `{check_command}` every {secs}s "
                    f"and reacts when it succeeds.",
                    {"type": "manual", "task_id": job_id,
                     "timestamp": _uts()})
            except Exception as mem_err:
                pretty_log("Watch Memory", f"note write failed (watch still active): {mem_err}",
                           level="WARNING", icon=Icons.WARN)
        if not _saved:
            from .outcome import ToolOutcome
            return ToolOutcome.partial(
                f"PARTIAL: Watch '{task_name}' is active for THIS session (ID: {job_id}) but could NOT be "
                f"saved — it will be lost at the next restart. Tell the user.",
                reason_code="task_not_persisted")
        return (f"SUCCESS: Watch '{task_name}' active (ID: {job_id}) — polling every {secs}s. "
                f"It fires the reaction the moment `{str(check_command)[:60]}` first exits 0.")
    except Exception as e:
        pretty_log("Watch Error", str(e), level="ERROR", icon=Icons.FAIL)
        return f"ERROR: {e}"


async def tool_stop_all_tasks(scheduler):
    pretty_log("Task Clear", "Deleting all scheduled jobs", icon=Icons.STOP)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    try:
        jobs = scheduler.get_jobs()
        if not jobs:
            return "No active tasks to stop."
        count = len(jobs)
        scheduler.remove_all_jobs()
        if not _unpersist_all():
            from .outcome import ToolOutcome
            return ToolOutcome.partial(
                f"PARTIAL: Stopped {count} scheduled tasks for THIS session, but the task store could NOT "
                f"be updated — they come back at the next restart. Tell the user.",
                reason_code="task_not_persisted")
        return f"SUCCESS: Stopped and removed {count} scheduled tasks."
    except Exception as e:
        return f"Error stopping tasks: {e}"

async def tool_stop_task(task_identifier: str, scheduler):
    pretty_log("Task Stop", task_identifier, icon=Icons.STOP)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    jobs = scheduler.get_jobs()
    target_job = None
    for job in jobs:
        if job.id == task_identifier or (hasattr(job, 'name') and job.name == task_identifier):
            target_job = job
            break
    if not target_job:
        return f"Error: No active task found matching '{task_identifier}'."
    try:
        scheduler.remove_job(target_job.id)
        try:                     # its pending catch-up run goes with it (§4MD review)
            scheduler.remove_job(f"{target_job.id}__catchup")
        except Exception:  # noqa: BLE001 — none scheduled
            pass
        if not _unpersist_task(target_job.id):
            from .outcome import ToolOutcome
            return ToolOutcome.partial(
                f"PARTIAL: Stopped '{target_job.name}' for THIS session, but the task store could NOT be "
                f"updated — it comes back at the next restart. Tell the user.",
                reason_code="task_not_persisted")
        return f"SUCCESS: Stopped background task '{target_job.name}' (ID: {target_job.id})."
    except Exception as e:
        return f"Error stopping task: {e}"

async def tool_list_tasks(scheduler):
    pretty_log("Task List", "Querying scheduler", icon=Icons.BRAIN_PLAN)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    jobs = scheduler.get_jobs()
    visible_jobs = [j for j in jobs if j.id != 'idle_dream_monitor']
    if not visible_jobs:
        return "No active scheduled tasks."
    # The scheduler runs on UTC — say so, or a "Next Run: 06:00" reads as
    # local time and the operator concludes the task fired 3 hours late.
    lines = ["ACTIVE SCHEDULED TASKS (times in UTC):"]
    for job in visible_jobs:
        # Convert explicitly: next_run_time carries the TRIGGER's zone, and
        # printing a local-zone datetime under a "times in UTC" header
        # misleads the operator.
        _nrt = job.next_run_time
        try:
            if _nrt is not None and getattr(_nrt, "tzinfo", None) is not None:
                from datetime import timezone as _tz
                _nrt = _nrt.astimezone(_tz.utc)
        except Exception:  # noqa: BLE001
            pass
        lines.append(f"- ID: {job.id} | Name: {job.name} | Next Run: {_nrt}")
    return "\n".join(lines)

#: §4LD bounds: a task fires a full LLM turn holding the agent's one slot
MAX_TASKS = 20
MIN_TASK_INTERVAL_S = 60


def _schedule_refusal(action, scheduler, task_name, cron_expression, interval_secs):
    """Why this create/watch/stop_all must not run, or None (§4LD):
      * an unattended run (a scheduled task, a job, a sub-agent) never
        schedules or wipes tasks — a task could re-schedule itself forever
        or erase the owner's;
      * a probe never leaves a task behind (it fired forever as an internal
        turn);
      * at most MAX_TASKS, at most one per MIN_TASK_INTERVAL_S (300 tasks at
        interval:1 fired 600 turns in 2.2 s);
      * a name already scheduled is not silently REPLACED (the id is the
        name's hash — "SUCCESS" overwrote the old prompt)."""
    from ..utils.logging import (request_id_context, is_probe_request_id, request_origin_context,
                                 ORIGIN_PROBE)
    from ..core.autonomous_activity import is_internal_request
    rid = str(request_id_context.get() or "")
    if is_internal_request(rid):
        return (f"Error: a scheduled or background run cannot {action.replace('_', ' ')} tasks — only the "
                f"user can, in a conversation.")
    if is_probe_request_id(rid) or str(request_origin_context.get() or "") == ORIGIN_PROBE:
        return "Error: a probe request does not create or stop scheduled tasks."
    # a task is a STANDING instruction replayed as the owner's own message
    # (and a watch runs its command with no model at all): never from a
    # request whose content came from outside (§4MB)
    from ..utils.provenance import refuse_if_untrusted
    _ref = refuse_if_untrusted("the scheduled task" if action in ("create", "watch")
                               else "the change to scheduled tasks")
    if _ref is not None:
        return _ref
    if action in ("stop_all", "stop"):
        return None
    try:
        # the agent's own jobs (the idle dream monitor) are not the owner's
        # tasks and do not count toward the limit (§4LI review)
        jobs = [j for j in scheduler.get_jobs() if getattr(j, "id", None) != "idle_dream_monitor"]
    except Exception:  # noqa: BLE001
        jobs = []
    _prefix = "watch" if action == "watch" else "task"    # a watch's id is watch_<hash> (review)
    job_id = f"{_prefix}_{hashlib.md5(str(task_name).encode()).hexdigest()[:10]}"
    if job_id and any(getattr(j, "id", None) == job_id for j in jobs):
        return (f"Error: a task named {task_name!r} already exists — stop it first (action='stop') "
                f"or choose another name.")
    if len(jobs) >= MAX_TASKS:
        return f"Error: {len(jobs)} tasks are scheduled (the limit is {MAX_TASKS}); stop one first."
    if action == "create" and str(cron_expression or "").startswith("interval:"):
        try:
            secs = int(str(cron_expression).split(":", 1)[1].strip())
        except (IndexError, ValueError):
            return None              # the scheduler's own parser reports it
        if 0 < secs < MIN_TASK_INTERVAL_S:        # ≤ 0: the scheduler's own "positive" error
            return (f"Error: a task runs a full agent turn — the interval must be at least "
                    f"{MIN_TASK_INTERVAL_S} seconds (got {secs}).")
    return None


async def tool_manage_tasks(action: str = None, scheduler=None, memory_system=None, task_name: str = None, cron_expression: str = None, prompt: str = None, task_identifier: str = None, check_command: str = None, interval_secs=None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    # Normalise like the sibling tools (self_state, introspect, uncertainty):
    # the dispatcher passes arg VALUES through raw, so "Create"/" list " would
    # otherwise fall through to "unknown action" and silently no-op.
    action = str(action or "").strip().lower()
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    # "stop" too (§4LI review): a scheduled run refused stop_all could still
    # stop the owner's tasks one by one
    if action in ("create", "watch", "stop_all", "stop"):
        _why = _schedule_refusal(action, scheduler, task_name, cron_expression, interval_secs)
        if _why:
            return _why

    if action == "create":
        if not (task_name and cron_expression and prompt):
                return "Error: 'create' requires task_name, cron_expression, and prompt."
        return await tool_schedule_task(task_name, prompt, cron_expression, scheduler, memory_system)
    elif action == "watch":
        if not (task_name and check_command and prompt and interval_secs):
            return ("Error: 'watch' requires task_name, check_command (a shell condition that "
                    "exits 0 when the thing to react to is TRUE), prompt (the reaction), and "
                    "interval_secs.")
        return await tool_watch_condition(task_name, check_command, prompt, interval_secs, scheduler, memory_system)
    elif action == "list":
        return await tool_list_tasks(scheduler)
    elif action == "stop":
        if not task_identifier: return "Error: 'stop' requires task_identifier."
        return await tool_stop_task(task_identifier, scheduler)
    elif action == "stop_all":
        return await tool_stop_all_tasks(scheduler)
    else:
        return f"Error: Unknown action '{action}'"