"""What is left of the per-task build executor (§4UY C18, 2026-11-18).

The spec executor that lived here (one spec call returning whole files as
JSON, N writes, one verify — 3,272 lines) and the agentic leaf it routed to
(`coding_loop.build_coding_task_agentic`) were REMOVED. Since §4UL (11-12) a
project task runs as one full agent turn (`core/project_turns.py`); the spec
path was only the fallback when that turn could not be built, and its last
run in the live log was 2026-11-07. A fallback to a weaker builder is the
class §4AT-C already refused for a crash, so a task whose turn cannot run now
goes back to the queue with a reason (`project_advancer`, §4UY C18).

Kept: the leaf/turn clocks other modules read, the task-named-file pattern
the advancer's workspace gather uses, the `CodingResult` shape the advancer
still consumes from a caller-supplied executor, and `build_coding_task` as
the production MARKER the callers pass (`project_turns.mark_real_executor`):
the advancer never calls it — a real executor's task runs as a turn or not
at all.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass, field
from typing import List, Optional

#: monotonic deadline set by the caller that bounds this leaf / task turn (0 = none)
leaf_deadline_context: contextvars.ContextVar = contextvars.ContextVar(
    "coding_leaf_deadline", default=0.0)
#: §4SD open items r5: does the caller wait for this build when it is stopped?
#: (set by the advancer's unattended step; no build reads it since §4UY C18)
caller_waits_context: contextvars.ContextVar = contextvars.ContextVar(
    "coding_caller_waits", default=True)


def _request_left_s() -> Optional[float]:
    """The most time the request this work runs inside can still have: its
    ceiling WITH every extension it could earn (`extension_ceiling_s`), less
    what it has spent; None = no bound (background work; the web and the CLI
    with no ceiling)."""
    try:
        from ..utils import logging as _glog
        rid = str(_glog.request_id_context.get() or "")
        ceil = float(_glog.extension_ceiling_s(rid) or 0.0)
        if not (0 < ceil < float("inf")):
            return None
        spent = _glog.request_elapsed_s(rid)
        return None if spent is None else ceil - float(spent)
    except Exception:  # noqa: BLE001 — no request clock = the caller's own
        return None


@dataclass
class CodingResult:
    ok: bool
    summary: str
    files: List[str] = field(default_factory=list)
    ledger_note: str = ""
    detail: str = ""
    #: §4PG (R2-1): a verify command the leaf gave RAN and exited 0 — a
    #: SUMMARY claim is not evidence; unattended, an unverified build is held
    verified: bool = False
    #: §4SD: the leaf's time ran out (not the build) — the advancer puts the
    #: task back to READY as for a timed-out step, not FAILED
    out_of_time: bool = False
    #: §4SX D4: the build could not get a lab (``core.lab_waits``: labs_down / lab_capacity) — the advancer
    #: puts the task back to PENDING with a hold that lifts when the labs host has room, not FAILED
    lab_wait: str = ""
    #: the MB the refused lab create needed (a capacity hold lifts only when that much is free)
    lab_needed_mb: int = 0


# A file path/name mentioned in a task description ("Implement: core.py - the
# model core"). Used to notice that the task's deliverable ALREADY EXISTS.
_TASK_NAMED_FILE_RE = re.compile(
    r"(?<![\w./-])(?:\./)*(\.?[\w][\w./-]{0,60}\.(?:py|js|mjs|cjs|json|html?|htm|css|md|"
    r"sh|yml|yaml|toml|txt))(?!\w)(?!\.\w)", re.IGNORECASE)   # §4QA-d: "…into h.py." ends a sentence


#: the refusal a direct call gets — nothing builds a task outside its turn any more
SPEC_EXECUTOR_REMOVED = ("the build executor was removed (§4UY C18): a project task runs as an agent turn "
                         "(core/project_turns.py) or waits in the queue")


async def build_coding_task(context, description: str, **_ignored) -> CodingResult:
    """The production executor MARKER (§4UY C18). The advancer runs a task
    wired with it as an agent turn and never calls it; a direct call builds
    nothing and says so."""
    return CodingResult(False, SPEC_EXECUTOR_REMOVED)


# §4UL (decision 1A): the production executor — the advancer runs a task wired with it as a full agent turn
from .project_turns import mark_real_executor as _mark_real_executor  # noqa: E402
_mark_real_executor(build_coding_task)
