"""Background-job registry — one place to look up async work (2026-07-11).

Before this, the agent had three fire-and-forget mechanisms (swarm workers,
scheduled tasks, self-play) and NO way to ask "what is running / did it
finish / what did it say". Swarm results only surfaced if the model
remembered to read a specific scratchpad key; task ids were returned but
never queryable. This registry is the missing status surface, shared by
``delegate_to_swarm`` and the tool-using sub-agent delegation
(:mod:`core.subagent`).

In-memory by design: jobs are per-process (their asyncio tasks die with the
process), so persisting them would only ever resurrect zombies. Completed
jobs are retained (bounded, FIFO-evicted) so a later turn can still collect
a result the model never read.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

MAX_RETAINED = 50          # completed jobs kept for later collection
MAX_RESULT_CHARS = 8000    # per-job result cap (context safety)


@dataclass
class Job:
    id: str
    kind: str                       # "swarm" | "subagent" | …
    label: str
    status: str = STATUS_RUNNING
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: str = ""
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None   # not serialised
    # Optional callable(raw_task_result) -> str|None. When set, its return
    # (if not None) becomes the recorded result instead of str(task.result()).
    # Used by swarm jobs whose worker returns a bool while the real content
    # lives in the scratchpad — without this `collect` returned "True".
    result_resolver: Any = None           # not serialised

    @property
    def duration_s(self) -> float:
        return (self.finished_at or time.time()) - self.created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "status": self.status, "duration_s": round(self.duration_s, 1),
            "result": self.result, "error": self.error, "meta": dict(self.meta),
        }


class JobRegistry:
    """Thread-safe registry of in-flight and recently-finished jobs."""

    def __init__(self, max_retained: int = MAX_RETAINED):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self.max_retained = max_retained

    def register(self, kind: str, label: str, *,
                 task: Optional[asyncio.Task] = None, **meta) -> Job:
        job = Job(id=f"job-{uuid.uuid4().hex[:8]}", kind=str(kind),
                  label=str(label)[:200], task=task, meta=dict(meta))
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        if task is not None:
            self.attach(job.id, task)
        return job

    def attach(self, job_id: str, task: asyncio.Task) -> None:
        """Bind an asyncio task to an already-registered job. Needed when the
        coroutine must know its own job id (the sub-agent names its sandbox
        dir after it), so the job is created BEFORE the task exists."""
        job = self.get(job_id)
        if job is None:
            return
        job.task = task
        task.add_done_callback(lambda t, jid=job_id: self._on_done(jid, t))

    def _on_done(self, job_id: str, task: asyncio.Task) -> None:
        """asyncio done-callback: land the outcome on the job. Never raises
        (a callback exception would be swallowed by the loop anyway, but a
        half-updated job would confuse the status surface)."""
        try:
            if task.cancelled():
                self.finish(job_id, status=STATUS_CANCELLED,
                            error="cancelled")
                return
            exc = task.exception()
            if exc is not None:
                self.finish(job_id, status=STATUS_FAILED,
                            error=f"{type(exc).__name__}: {exc}")
                return
            res = task.result()
            # A job may carry a resolver that maps the raw task result to the
            # real payload (e.g. a swarm worker returns a bool; the content is
            # in the scratchpad under output_key). Resolver returning None →
            # fall back to the raw result.
            resolved = None
            job = self.get(job_id)
            resolver = getattr(job, "result_resolver", None) if job else None
            if resolver is not None:
                try:
                    resolved = resolver(res)
                except Exception as re:  # noqa: BLE001
                    logger.debug("job result_resolver failed for %s: %s", job_id, re)
            final = resolved if resolved is not None else ("" if res is None else str(res))
            self.finish(job_id, status=STATUS_DONE, result=final)
        except Exception as e:  # noqa: BLE001
            logger.debug("job done-callback failed for %s: %s", job_id, e)

    def finish(self, job_id: str, *, status: str = STATUS_DONE,
               result: str = "", error: str = "") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != STATUS_RUNNING:
                return  # unknown, or already landed (idempotent)
            job.status = status
            job.finished_at = time.time()
            job.result = str(result or "")[:MAX_RESULT_CHARS]
            job.error = str(error or "")[:1000]
            job.task = None  # drop the task ref once it's landed
            self._evict_locked()

    def _evict_locked(self) -> None:
        """FIFO-evict the oldest COMPLETED jobs past the retention bound.
        Running jobs are never evicted (their callback still needs them)."""
        completed = [jid for jid in self._order
                     if (j := self._jobs.get(jid)) is not None
                     and j.status != STATUS_RUNNING]
        excess = len(completed) - self.max_retained
        for jid in completed[:max(0, excess)]:
            self._jobs.pop(jid, None)
            try:
                self._order.remove(jid)
            except ValueError:
                pass

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(str(job_id))

    def list(self, *, status: Optional[str] = None,
             kind: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = [self._jobs[j] for j in self._order if j in self._jobs]
        if status:
            jobs = [j for j in jobs if j.status == status]
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        return jobs

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status != STATUS_RUNNING:
            return False
        task = job.task
        if task is not None and not task.done():
            task.cancel()          # the done-callback lands CANCELLED
            return True
        self.finish(job_id, status=STATUS_CANCELLED, error="cancelled")
        return True

    async def wait(self, job_ids: List[str], timeout: float = 60.0) -> None:
        """Await the given jobs (best-effort, bounded). Unknown/finished ids
        are skipped; a timeout leaves them running (the caller can poll)."""
        tasks = []
        for jid in job_ids or []:
            job = self.get(jid)
            if job is not None and job.task is not None and not job.task.done():
                tasks.append(job.task)
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=max(0.1, float(timeout)))


def get_job_registry(context) -> JobRegistry:
    """Get-or-create the registry on the context. Callers never construct
    one directly, so every subsystem shares the same view."""
    reg = getattr(context, "job_registry", None)
    if not isinstance(reg, JobRegistry):
        reg = JobRegistry()
        try:
            context.job_registry = reg
        except Exception:  # noqa: BLE001 — a mock may refuse attributes
            pass
    return reg


__all__ = [
    "STATUS_RUNNING", "STATUS_DONE", "STATUS_FAILED", "STATUS_CANCELLED",
    "MAX_RETAINED", "MAX_RESULT_CHARS",
    "Job", "JobRegistry", "get_job_registry",
]
