import asyncio
import hashlib
import logging
import uuid
from apscheduler.triggers.cron import CronTrigger
from ..utils.logging import pretty_log, Icons

logger = logging.getLogger("GhostAgent")

# This will need to be bound to run_proactive_task from agent.py
run_proactive_task_fn = None

async def tool_schedule_task(task_name: str, prompt: str, cron_expression: str, scheduler, memory_system):
    pretty_log("Task Schedule", f"Name: {task_name} | Expr: {cron_expression}", icon=Icons.BRAIN_PLAN)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    if run_proactive_task_fn is None:
        return "Error: Proactive task runner not initialized."
        
    try:
        job_id = f"task_{hashlib.md5(task_name.encode()).hexdigest()[:6]}_{uuid.uuid4().hex[:4]}"
        
        if cron_expression.startswith("interval:"):
            parts = cron_expression.split(":")
            try:
                secs = int(parts[1].strip()) if len(parts) > 1 else 60
            except ValueError:
                secs = 60
            
            scheduler.add_job(
                run_proactive_task_fn, 
                'interval', 
                seconds=secs, 
                args=[job_id, prompt], 
                id=job_id,
                name=task_name,
                replace_existing=True
            )
        else:
            scheduler.add_job(
                run_proactive_task_fn, 
                CronTrigger.from_crontab(cron_expression), 
                args=[job_id, prompt], 
                id=job_id,
                name=task_name,
                replace_existing=True
            )
            
        memory_entry = f"Scheduled task '{task_name}' is running with ID {job_id} on schedule {cron_expression}."
        if memory_system:
            await asyncio.to_thread(memory_system.add, memory_entry, {"type": "manual", "task_id": job_id})
            
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
        return f"SUCCESS: Stopped background task '{target_job.name}' (ID: {target_job.id})."
    except Exception as e:
        return f"Error stopping task: {e}"

async def tool_list_tasks(scheduler):
    pretty_log("Task List", "Querying scheduler", icon=Icons.BRAIN_PLAN)
    if not scheduler:
        return "Error: Background task scheduling is disabled or not available in this context."
    jobs = scheduler.get_jobs()
    if not jobs:
        return "No active scheduled tasks."
    lines = ["ACTIVE SCHEDULED TASKS:"]
    for job in jobs:
        lines.append(f"- ID: {job.id} | Name: {job.name} | Next Run: {job.next_run_time}")
    return "\n".join(lines)

async def tool_manage_tasks(action: str, scheduler, memory_system, task_name: str = None, cron_expression: str = None, prompt: str = None, task_identifier: str = None, **kwargs):
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