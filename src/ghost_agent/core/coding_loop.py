"""Leaf / task turns: one bounded agent turn on an isolated, project-pinned context (§4FG, 2026-09-07).

`run_leaf_turn` runs ONE agent turn on a context built by `build_leaf_context`
(project pinned, memory read-only, nothing written to user-population
ledgers, `task_kind="leaf"` unless the caller labels it) with a contained
tool surface. Since §4UL every project task runs through it as a full agent
turn (`core/project_turns.py`).

§4UY C18 (2026-11-18): the agentic coding LEAF executor that lived here
(`build_coding_task_agentic` — a VERIFY:/SUMMARY: edit-test loop with
workspace snapshots and diffs, selected by `executor_kind` / the leaf's
shape) was REMOVED with the spec executor it was routed from
(`core/coding_executor.py`): both ran only when a task turn could not be
built, and a task now waits in the queue instead.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

from ..utils.helpers import env_positive

# §4KB (round 5): `git` was added here and then REMOVED. Four review rounds
# produced nine sandbox escapes and every one of them came from the git tool
# — `.git` is a configuration surface that names programs to run, and the
# model can write files. The leaf reads its own diff through `execute` as it
# always did. See PROJECT_JOURNAL.md §4KB round 5.
# §4SX D4: `lab` (a disposable machine on the labs host, origin autopilot — kept machines and apps are refused by
# the tool and by the host; design review D6 M6: `run` joins at D5)
LEAF_ALLOWED_TOOLS = ("file_system", "execute", "lab", "run")      # §4SX D5 (D6 M6): `run` joins at the switch
LEAF_MAX_TURNS = int(env_positive("GHOST_LEAF_MAX_TURNS", 14.0))
LEAF_TIMEOUT_S = env_positive("GHOST_LEAF_TIMEOUT_S", 600.0)
LEAF_TASK_KIND = "leaf"


#: folders a workspace walk skips (the advancer's workspace gather reads this)
_SNAPSHOT_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".venv"}



def build_leaf_context(context, *, leaf_id: str, project_id: Optional[str] = None):
    """An isolated context for one leaf: project PINNED (explicitly — the
    advancer's context carries no binding), workspace shared, memory
    read-only, nothing recorded into user-population ledgers."""
    from ..memory.readonly import ReadOnlyGraphMemory, ReadOnlyVectorMemory
    from .isolation import mark_simulated
    iso = copy.copy(context)
    # §4TD: FIRST and unconditionally — the simulation guards (no selfhood
    # capture, no operator notification, no objection pass …) read this flag.
    # It was the read-only lesson-store wrapper, installed inside the `try`
    # below, where a failed wrap left the leaf looking like the live context.
    mark_simulated(iso)
    # Keep sandbox_dir; PIN the project id: file_system must resolve the
    # PROJECT workspace, which is the whole point of the leaf. Copying the
    # context's binding is not enough — idle ticks and the HTTP route run
    # with `current_project_id=None` (the 2026-07-08 root-vs-project class).
    pid = project_id or getattr(context, "current_project_id", None)
    if pid:
        # outside the leaf's turn (the diff, the verify); inside it the
        # leaf's own scope carries the pin (`run_leaf_turn`, §4SX B0)
        iso.current_project_id = pid
    iso.workspace_model = None
    iso.episodic_memory = None
    iso.journal = None
    iso.turn_origin_label = LEAF_TASK_KIND
    iso.confine_writes_to_project = True     # §4PG (R2-4): file writes stay inside the pinned workspace
    iso.trajectory_task_kind = LEAF_TASK_KIND
    iso.trajectory_extra_static = {"leaf_id": leaf_id}
    iso.trajectory_user_request_override = None
    iso.scheduler = None
    iso.profile_memory = None
    iso.memory_bus = None
    # §4LB: NO owner memory at all — `_get_memory_bus` built a fresh bus from
    # the read-only stores when this was None, and the scratchpad / past
    # requests were shared, so the owner's recall (health, family) reached a
    # delegate that writes web queries
    iso.owner_memory_isolated = True
    # §4LB r2: no owner memory, but its outbound queries are still scrubbed
    # of the owner's street address (its prompt carries no profile; the
    # scrubber needs one) — and `recall` refuses (it read the owner's store)
    from ..memory.egress import egress_profile as _egress_profile
    iso.egress_profile = _egress_profile(context)
    try:
        from ..memory.scratchpad import Scratchpad
        iso.scratchpad = Scratchpad()
    except Exception:  # noqa: BLE001
        iso.scratchpad = None
    try:
        iso.memory_system = ReadOnlyVectorMemory(context.memory_system)
        iso.graph_memory = ReadOnlyGraphMemory(getattr(context, "graph_memory", None))
    except Exception as e:  # noqa: BLE001
        logger.debug("leaf read-only memory wrap skipped: %s", e)
    iso.args = copy.copy(context.args)
    iso.args.perfect_it = False
    iso.args.smart_memory = 0.0
    if not getattr(iso.args, "native_tools", False):
        iso.args.native_tools = True
    for attr in ("verifier", "frontier_tracker", "metacog"):
        try:
            setattr(iso, attr, None)
        except Exception:  # noqa: BLE001
            pass
    iso._subagent_allowed_tools = frozenset(LEAF_ALLOWED_TOOLS)
    return iso


class _BackgroundOnlyLLM:
    """Route every completion through the client's background lane (waits
    for foreground to clear, capped concurrency) — the same wrapper
    `build_subagent_context` uses. §4FH M1: the first leaf runner took
    `is_background` and never used it, so an idle-tick leaf competed with a
    live user turn for the single slot."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        return await self._inner.chat_completion(payload, *a, **kw)

    async def stream_chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        async for chunk in self._inner.stream_chat_completion(payload, *a, **kw):
            yield chunk



#: leaf end (review 11-05 m8): bound on stopping the leaf's runs, then on destroying its labs
#: leaf ends still running after their turn was cancelled (held until done)
_LEAF_END_TASKS: set = set()
LEAF_RUN_STOP_BOUND_S = 90.0
LEAF_LAB_DESTROY_BOUND_S = 150.0

async def run_leaf_turn(context, *, leaf_id: str, prompt: str, is_background: bool,
                        max_turns: int = LEAF_MAX_TURNS,
                        timeout_s: float = LEAF_TIMEOUT_S,
                        project_id: Optional[str] = None,
                        allowed_tools=LEAF_ALLOWED_TOOLS,
                        origin: Optional[str] = None,
                        request_id: Optional[str] = None,
                        label: Optional[str] = None,
                        leaf_end: bool = True,
                        prepare_agent: Optional[Callable[[Any], None]] = None) -> str:
    """One bounded agentic run; returns the final reply text. Tool surface
    is contained the same way `run_subagent` does it (fail CLOSED).

    §4UL (project task turns, `core/project_turns.py`): ``allowed_tools=None``
    keeps the FULL tool surface; ``origin`` / ``request_id`` / ``label``
    replace the leaf's; ``prepare_agent(agent)`` wraps tools before the turn
    (it raising stops the run); ``leaf_end=False`` neither tags the turn's
    labs/runs as a leaf's nor ends them after it (an owner's run keeps what
    it started, like any owner turn)."""
    from .agent import GhostAgent
    iso = build_leaf_context(context, leaf_id=leaf_id, project_id=project_id)
    if label:
        iso.turn_origin_label = label
        iso.trajectory_task_kind = label
    if allowed_tools is None:
        iso._subagent_allowed_tools = None
    if is_background and getattr(iso, "llm_client", None) is not None:
        iso.llm_client = _BackgroundOnlyLLM(iso.llm_client)
    agent = GhostAgent(iso)
    if allowed_tools is not None:
        allow = set(allowed_tools)
        try:
            from ..tools.registry import TOOL_DEFINITIONS
            advertised = {t["function"]["name"] for t in TOOL_DEFINITIONS}
            agent.disabled_tools = (advertised | set(agent.available_tools)) - allow
            agent.available_tools = {k: v for k, v in agent.available_tools.items() if k in allow}
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"leaf tool containment failed ({type(e).__name__}: {e})") from e
    agent.max_turns_override = max(2, int(max_turns))
    if prepare_agent is not None:
        prepare_agent(agent)
    body = {"model": getattr(iso.args, "model", "default"),
            "messages": [{"role": "user", "content": prompt}], "stream": False}
    # §4SX B0 (r2 #7): the leaf's turn runs under a pre-built scope holding
    # its pin (no conversation, never a focus row); the after-turn check
    # reads THAT object, not a ContextVar from inside the awaited task
    from .request_scope import RequestScope, ORIGIN_LEAF
    _want = project_id or getattr(context, "current_project_id", None)
    leaf_scope = RequestScope(project_id=_want, origin=origin or ORIGIN_LEAF)
    # §4SX D4 review r2 #5/#7: the leaf's labs and its "no lab" answers are tagged with THIS leaf / build
    from .lab_waits import BUILD_KEY, LEAF_ID, RUNNING_LEAVES
    _tk = ((BUILD_KEY.set(str(leaf_id).rsplit("-a", 1)[0]), LEAF_ID.set(str(leaf_id))) if leaf_end else None)
    # §4UD F1: THIS turn's deadline, where the lab tool reads what is left (`tools/labs._time_left_s`) — never past
    # a caller's own (the unattended step's cap). It was not set, so a leaf with 4 minutes left sent `lab exec
    # timeout_s=900`: the command outlived the turn, held its lab, and the leaf-end destroy was refused.
    from .coding_executor import leaf_deadline_context as _ldc
    _turn_s = max(30.0, float(timeout_s))
    _outer = float(_ldc.get() or 0.0)
    _mine = time.monotonic() + _turn_s
    _tk_dl = _ldc.set(min(_mine, _outer) if _outer > 0 else _mine)
    # the leaf counts as running until its end clean-up below has finished (the periodic leaf sweep skips it)
    if leaf_end:
        RUNNING_LEAVES.add(str(leaf_id))
    _end_pending = False
    try:
        content, _, _ = await asyncio.wait_for(
            agent.handle_chat(body, background_tasks=None, request_id=request_id or f"sub-leaf-{leaf_id}",
                              scope=leaf_scope),
            timeout=_turn_s)
    finally:
        _ldc.reset(_tk_dl)
        if _tk is not None:
            BUILD_KEY.reset(_tk[0])
            LEAF_ID.reset(_tk[1])
        # §4SX D4: a leaf's labs are disposable and end with the leaf (a timeout or a crash too); bounded, and a
        # cancel that lands during it is re-raised (review r2 #11)
        if leaf_end and not _want:
            RUNNING_LEAVES.discard(str(leaf_id))
        if leaf_end and _want:
            _t_end = None
            try:
                from ..tools.labs import destroy_leaf_labs
                from ..tools.runs import stop_leaf_runs

                async def _end_leaf():
                    # §4SX D5: its runs first (a run on a lab stops before the lab goes), then its labs. Each step
                    # has its OWN bound (review 11-05 m8: one shared 120 s let slow run stops starve the lab destroy)
                    try:
                        await asyncio.wait_for(stop_leaf_runs(context, f"project:{_want}", str(leaf_id)),
                                               LEAF_RUN_STOP_BOUND_S)
                    except asyncio.TimeoutError:
                        logger.warning("leaf %s: stopping its runs timed out — its labs are destroyed anyway",
                                       leaf_id)
                    await asyncio.wait_for(destroy_leaf_labs(context, f"project:{_want}", str(leaf_id)),
                                           LEAF_LAB_DESTROY_BOUND_S)
                _t_end = asyncio.ensure_future(_end_leaf())
                await asyncio.shield(_t_end)
            except asyncio.CancelledError:
                if _t_end is not None and not _t_end.done():
                    # r12/r13: the turn was cancelled; the shielded end runs on — held until done (a bare task can be
                    # collected mid-run) and its error logged once, here (nobody awaits it any more)
                    _LEAF_END_TASKS.add(_t_end)

                    _end_pending = True

                    def _seen(f, _lid=leaf_id):
                        _LEAF_END_TASKS.discard(f)
                        RUNNING_LEAVES.discard(str(_lid))      # the end has run: the sweep may look at it now
                        if not f.cancelled() and f.exception() is not None:
                            logger.warning("leaf %s end: %s: %s", _lid, type(f.exception()).__name__, f.exception())
                    _t_end.add_done_callback(_seen)
                raise
            except Exception as e:  # noqa: BLE001 — a cleanup never masks the turn's own outcome
                logger.warning("leaf labs not cleaned up: %s: %s", type(e).__name__, e)
            finally:
                if not _end_pending:
                    RUNNING_LEAVES.discard(str(leaf_id))
    # §4FH C1: the pin must have survived the turn — if anything moved it,
    # the leaf's files went somewhere other than the project and the
    # attempt is a failure, not a success with an empty diff.
    if _want and leaf_scope.project_id != _want:
        raise RuntimeError(
            f"leaf lost its project pin during the turn "
            f"(wanted {_want}, got {leaf_scope.project_id!r})")
    return str(content or "").strip()
