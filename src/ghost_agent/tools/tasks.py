import asyncio
import hashlib
import json
import logging
import os
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
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        tasks = data.get("tasks")
        return tasks if isinstance(tasks, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.warning("scheduled-task store unreadable (%s) — treating as empty", e)
        return {}


def _save_task_store(tasks: dict) -> None:
    """Atomic write (tmp + os.replace) so a crash mid-save can't truncate
    the store. Best-effort: a persistence failure is logged, never raised —
    the live scheduler state stays authoritative for this session."""
    if not task_store_path:
        return
    try:
        p = Path(task_store_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"tasks": tasks}, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        logger.warning("scheduled-task store write failed: %s", e)


def _persist_task(job_id: str, task_name: str, prompt: str,
                  cron_expression: str) -> None:
    tasks = _load_task_store()
    tasks[job_id] = {
        "task_name": task_name,
        "prompt": prompt,
        "cron_expression": cron_expression,
        "created_at": time.time(),
    }
    _save_task_store(tasks)


def _unpersist_task(job_id: str) -> None:
    tasks = _load_task_store()
    if job_id in tasks:
        del tasks[job_id]
        _save_task_store(tasks)


def _unpersist_all() -> None:
    if task_store_path:
        _save_task_store({})


def _add_job(scheduler, job_id: str, task_name: str, prompt: str,
             cron_expression: str):
    """Register one job on the scheduler. Returns an error STRING on a
    rejected/malformed schedule, None on success. Shared by the create tool
    and the boot-time restore so both interpret expressions identically."""
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
        scheduler.add_job(
            run_proactive_task_fn,
            'interval',
            seconds=secs,
            args=[job_id, prompt],
            id=job_id,
            name=task_name,
            replace_existing=True
        )
        return None
    if CronTrigger is None:
        return "Error: cron-style schedules require apscheduler. Use 'interval:SECONDS' instead."
    scheduler.add_job(
        run_proactive_task_fn,
        CronTrigger.from_crontab(cron_expression),
        args=[job_id, prompt],
        id=job_id,
        name=task_name,
        replace_existing=True
    )
    return None


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
            )
            if err:
                raise ValueError(err)
            restored += 1
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
        job_id = f"task_{hashlib.md5(task_name.encode()).hexdigest()[:10]}"

        err = _add_job(scheduler, job_id, task_name, prompt, cron_expression)
        if err:
            return err

        # Persist AFTER the live registration succeeds, so the store only
        # ever holds tasks that were actually schedulable. Survives agent
        # restarts (the jobstore itself is in-memory); best-effort like the
        # memory note below.
        _persist_task(job_id, task_name, prompt, cron_expression)

        # The job is already scheduled at this point. A failure to write the
        # bookkeeping memory entry must NOT be reported as a scheduling
        # failure (the task WOULD still fire) — isolate it so the outcome we
        # return matches the real scheduler state.
        memory_entry = f"Scheduled task '{task_name}' is running with ID {job_id} on schedule {cron_expression}."
        if memory_system:
            try:
                await asyncio.to_thread(memory_system.add, memory_entry, {"type": "manual", "task_id": job_id})
            except Exception as mem_err:
                pretty_log("Schedule Memory", f"note write failed (task still scheduled): {mem_err}",
                           level="WARNING", icon=Icons.WARN)

        return f"SUCCESS: Task '{task_name}' scheduled (ID: {job_id})."
    except Exception as e:
        pretty_log("Schedule Error", str(e), level="ERROR")
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
        _unpersist_all()
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
        _unpersist_task(target_job.id)
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
        lines.append(f"- ID: {job.id} | Name: {job.name} | Next Run: {job.next_run_time}")
    return "\n".join(lines)

async def tool_manage_tasks(action: str = None, scheduler=None, memory_system=None, task_name: str = None, cron_expression: str = None, prompt: str = None, task_identifier: str = None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    # Normalise like the sibling tools (self_state, introspect, uncertainty):
    # the dispatcher passes arg VALUES through raw, so "Create"/" list " would
    # otherwise fall through to "unknown action" and silently no-op.
    action = action.strip().lower()
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."

    if action == "create":
        if not (task_name and cron_expression and prompt):
                return "Error: 'create' requires task_name, cron_expression, and prompt."
        return await tool_schedule_task(task_name, prompt, cron_expression, scheduler, memory_system)
    elif action == "list":
        return await tool_list_tasks(scheduler)
    elif action == "stop":
        if not task_identifier: return "Error: 'stop' requires task_identifier."
        return await tool_stop_task(task_identifier, scheduler)
    elif action == "stop_all":
        return await tool_stop_all_tasks(scheduler)
    else:
        return f"Error: Unknown action '{action}'"