"""``delegate`` + ``jobs`` tools — tool-using sub-agents and job status.

``delegate`` spawns bounded, tool-using sub-agents (:mod:`core.subagent`)
that actually DO work — unlike ``delegate_to_swarm``, whose workers are
stateless single completions with no tools. ``jobs`` is the status surface
every fire-and-forget mechanism was missing: list / check / collect /
cancel background work by id (:mod:`core.jobs`).
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


def _fmt_job(job, *, with_result: bool = False) -> str:
    line = (f"- {job.id} [{job.status}] {job.kind}: {job.label[:70]} "
            f"({job.duration_s:.0f}s)")
    if job.error:
        line += f"\n    error: {job.error[:200]}"
    if with_result and job.result:
        line += f"\n    result: {job.result}"
    return line


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
            # Collect ALL unread finished jobs — the common case.
            done = [j for j in reg.list(status=STATUS_DONE) if j.result]
            if not done:
                return ("No finished jobs with results to collect "
                        "(jobs(action='status') shows what's running).")
            out = [f"Collected {len(done)} finished job(s):"]
            for j in done:
                out.append(f"\n--- {j.id}: {j.label[:70]} ---\n{j.result}")
            return "\n".join(out)
        job = reg.get(job_id)
        if job is None:
            return (f"Error: no job {job_id!r} (it may have been evicted — "
                    f"only the last {reg.max_retained} finished jobs are "
                    f"retained).")
        if job.status == STATUS_RUNNING:
            return (f"Job {job.id} is still RUNNING ({job.duration_s:.0f}s so "
                    f"far): {job.label[:80]}. Check back later.")
        if job.status == STATUS_DONE:
            return f"--- {job.id}: {job.label[:70]} ---\n{job.result or '(no output)'}"
        return (f"Job {job.id} {job.status.upper()}: "
                f"{job.error or 'no error detail'}")

    if action == "cancel":
        if not job_id:
            return "Error: 'job_id' is required for cancel."
        ok = reg.cancel(job_id)
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
            "Check on BACKGROUND WORK you started (delegated sub-agents, swarm "
            "tasks): what's still running, what finished, and what it produced. "
            "Call this when you delegated something earlier and need the answer, "
            "or before re-doing work a sub-agent may already have done."
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
