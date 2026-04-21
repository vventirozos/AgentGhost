"""Self-advancing research loop for long-term projects.

Each tick pulls the next READY/PENDING leaf from a project's plan,
classifies it as research-flavored or coding-flavored, runs a single
constrained step, and updates the task in the store. The tick is a
pure async function so it can be driven by APScheduler in production
and by pytest in tests — the caller injects a ``tool_runner`` and an
``llm_classifier`` (both optional; sensible defaults are used when
omitted).

Budget model:
  project metadata carries two budget knobs:
    - ``steps_cap``  — the total number of advancer steps allowed for
      this project. Defaults to :data:`DEFAULT_STEPS_CAP` when unset.
    - ``steps_used`` — incremented after every completed tick.
  When ``steps_used >= steps_cap`` the advancer refuses to proceed and
  logs a ``budget_exhausted`` event. The user has to raise the cap
  explicitly (via ``manage_projects`` update) before autoadvance
  resumes — budgets are a hard stop, not a soft warning.

The classifier is intentionally a keyword heuristic. An LLM classifier
can be plugged in later by passing ``llm_classifier=...``; the tests
verify both paths.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("GhostAgent")


DEFAULT_STEPS_CAP = 50


# Keyword buckets used by the lightweight classifier. The lists are
# deliberately short and high-precision; anything not hitting one
# bucket defaults to RESEARCH, which is the safer autonomy mode (no
# sandbox side-effects).
_CODING_KEYWORDS = {
    "implement", "build", "write code", "refactor", "fix", "debug",
    "patch", "unit test", "add tests", "deploy", "migrate", "compile",
    "install", "scaffold", "ship", "benchmark",
}

_RESEARCH_KEYWORDS = {
    "research", "find", "investigate", "compare", "review", "summarize",
    "survey", "analyze", "explain", "how does", "what is", "why",
    "read about", "look up", "gather", "source",
}

_NEEDS_USER_KEYWORDS = {
    "decide", "approve", "choose", "pick one", "sign off", "confirm",
    "authorize", "review and approve", "publish", "announce", "send",
    "delete production", "drop database",
}


@dataclass
class AdvanceResult:
    """What happened during a single tick.

    Attributes:
      ok: True when the advancer ran cleanly, even if the result was
          "no work" or "budget exhausted" — reserved for genuine crashes.
      task_id: id of the task the tick targeted, if any.
      classification: "research" / "coding" / "needs_user" / "idle" / "blocked".
      summary: short human-readable sentence for the event log.
      artifact_id: id of the artifact written, if the step produced one.
    """

    ok: bool
    task_id: Optional[str]
    classification: str
    summary: str
    artifact_id: Optional[str] = None


ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]
LLMClassifier = Callable[[str], Awaitable[str]]


def classify_task(description: str) -> str:
    """Return one of "needs_user", "coding", "research".

    Precedence: ``needs_user`` → ``research`` → ``coding`` → default
    research. ``research`` beats ``coding`` because a task like
    "Research benchmarks" contains the word "benchmark" (coding
    keyword) but clearly intends reading/summarizing, not writing
    code. ``needs_user`` always wins so "Implement and approve X" is
    routed to the human.
    """
    if not description:
        return "research"
    lower = description.lower()
    for kw in _NEEDS_USER_KEYWORDS:
        if kw in lower:
            return "needs_user"
    for kw in _RESEARCH_KEYWORDS:
        if kw in lower:
            return "research"
    for kw in _CODING_KEYWORDS:
        if kw in lower:
            return "coding"
    return "research"


def _get_budget(store, project_id: str) -> Dict[str, int]:
    proj = store.get_project(project_id) or {}
    meta = proj.get("metadata") or {}
    cap = int(meta.get("steps_cap", DEFAULT_STEPS_CAP))
    used = int(meta.get("steps_used", 0))
    return {"cap": cap, "used": used, "meta": meta}


def _increment_budget(store, project_id: str) -> None:
    budget = _get_budget(store, project_id)
    new_meta = dict(budget["meta"])
    new_meta["steps_used"] = budget["used"] + 1
    new_meta.setdefault("steps_cap", budget["cap"])
    store.update_project(project_id, metadata=new_meta)


async def advance_once(
    context,
    project_id: str,
    tool_runner: Optional[ToolRunner] = None,
    llm_classifier: Optional[LLMClassifier] = None,
) -> AdvanceResult:
    """Run a single autoadvance tick for ``project_id``.

    Args:
      context: the GhostContext-like object (needs ``project_store``).
      project_id: which project to advance.
      tool_runner: async callable ``(tool_name, args) -> str``. Defaults
        to ``get_available_tools(context)[tool_name](**args)``.
      llm_classifier: async callable returning one of
        "needs_user"/"coding"/"research" for a free-form description.
        Defaults to :func:`classify_task` (synchronous, wrapped).
    """
    store = getattr(context, "project_store", None)
    if store is None:
        return AdvanceResult(False, None, "idle",
                             "project_store missing on context")

    proj = store.get_project(project_id)
    if not proj:
        return AdvanceResult(False, None, "idle",
                             f"project not found: {project_id}")
    if proj["status"] != "ACTIVE":
        return AdvanceResult(True, None, "blocked",
                             f"project is {proj['status']}, not ACTIVE")

    budget = _get_budget(store, project_id)
    if budget["used"] >= budget["cap"]:
        store.log_event(project_id, None, "budget_exhausted",
                        {"used": budget["used"], "cap": budget["cap"]})
        return AdvanceResult(True, None, "blocked",
                             f"budget exhausted: {budget['used']}/{budget['cap']}")

    # Secondary rails: runtime + tool-call caps (optional per project).
    from .project_safety import check_budget
    secondary = check_budget(proj.get("metadata") or {})
    if not secondary.allowed:
        store.log_event(project_id, None, "budget_exhausted",
                        dict(secondary.remaining))
        return AdvanceResult(True, None, "blocked", secondary.reason)

    _tick_started_at = time.time()

    # Lazy import: ProjectPlan lives in planning.py which also owns
    # TaskTree logic; importing at module top would pull the planning
    # graph into every scheduler tick which is wasteful.
    from .planning import ProjectPlan, TaskStatus

    plan = ProjectPlan(store, project_id)
    nxt = plan.next_ready_leaf()
    if not nxt:
        return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")

    if llm_classifier is None:
        classification = classify_task(nxt.description)
    else:
        try:
            classification = await llm_classifier(nxt.description)
        except Exception:
            classification = classify_task(nxt.description)
    classification = (classification or "research").lower()

    # Mark IN_PROGRESS up front so concurrent ticks don't pick the same leaf.
    plan.update_status(nxt.id, TaskStatus.IN_PROGRESS)

    if classification == "needs_user":
        plan.update_status(nxt.id, TaskStatus.NEEDS_USER,
                           result="flagged for human review")
        store.log_event(project_id, nxt.id, "autoadvance_needs_user",
                        {"description": nxt.description})
        _increment_budget(store, project_id)
        return AdvanceResult(True, nxt.id, "needs_user",
                             "task requires human input")

    # Pick a tool by classification. Research → web_search,
    # coding → execute (sandbox runs it). A missing tool runner means
    # we can only classify + mark, not actually execute — still a
    # useful signal, so we don't treat it as a hard failure.
    if classification == "coding":
        tool_name = "execute"
        tool_args = {"command": f"# Autoadvance stub for: {nxt.description[:120]}"}
    else:
        tool_name = "web_search"
        tool_args = {"query": nxt.description[:200]}

    output = ""
    artifact_id: Optional[str] = None
    if tool_runner is not None:
        try:
            output = await tool_runner(tool_name, tool_args)
        except Exception as e:
            logger.warning("autoadvance tool_runner failed: %s", e)
            plan.update_status(
                nxt.id, TaskStatus.FAILED,
                failure_reason=f"tool {tool_name} raised: {e}",
            )
            _increment_budget(store, project_id)
            return AdvanceResult(True, nxt.id, classification,
                                 f"tool error: {e}")
        try:
            artifact_id = store.add_artifact(
                nxt.id, "tool_call",
                _truncate_payload(output),
            )
        except Exception:
            logger.debug("artifact write skipped", exc_info=True)

    result_summary = _short_summary(output) if output else "(no tool runner)"

    # Human-gate postconditions force NEEDS_USER regardless of output.
    # The store row is the authoritative source for the postcondition
    # list because ProjectPlan may not refresh between ticks.
    from .project_safety import enforce_human_gate, detect_contradiction, route_contradiction

    task_row = store.get_task(nxt.id) or {}
    gate_reason = enforce_human_gate(task_row)
    if gate_reason:
        plan.update_status(
            nxt.id, TaskStatus.NEEDS_USER,
            result=f"human gate: {gate_reason}",
        )
        store.log_event(project_id, nxt.id, "human_gate_triggered",
                        {"reason": gate_reason, "tool": tool_name})
        _increment_budget(store, project_id)
        return AdvanceResult(True, nxt.id, "needs_user",
                             f"human gate: {gate_reason}", artifact_id)

    # Contradiction detection: compare against any DONE sibling's
    # result_summary. Siblings share the same parent; when parent is
    # None (root-level task) we treat the whole project as the peer set.
    parent_id = task_row.get("parent_id")
    peers = [
        t for t in store.list_tasks(project_id)
        if t["id"] != nxt.id and t["status"] == "DONE"
        and (t.get("parent_id") == parent_id)
    ]
    contradiction_log = getattr(context, "contradiction_log", None)
    for peer in peers:
        conflict = detect_contradiction(result_summary,
                                        peer.get("result_summary", ""))
        if conflict:
            store.log_event(project_id, nxt.id, "contradiction_detected",
                            {"peer_task_id": peer["id"], "conflict": conflict})
            route_contradiction(
                contradiction_log,
                new_fact=f"task:{nxt.id}: {result_summary}",
                prior_facts=[f"task:{peer['id']}: {peer.get('result_summary','')}"],
                reason=conflict,
            )
            break

    plan.update_status(nxt.id, TaskStatus.DONE,
                       result=result_summary, actual_tool=tool_name)
    _increment_budget(store, project_id)
    from .project_safety import record_runtime as _record
    _record(store, project_id,
            seconds=max(0.0, time.time() - _tick_started_at),
            tool_calls=1 if tool_runner is not None else 0)
    store.log_event(project_id, nxt.id, "autoadvance_step",
                    {"tool": tool_name, "classification": classification})
    return AdvanceResult(True, nxt.id, classification,
                         f"advanced via {tool_name}", artifact_id)


def _short_summary(output: str, max_len: int = 200) -> str:
    if not output:
        return ""
    out = str(output).strip().replace("\n", " ")
    if len(out) > max_len:
        return out[:max_len] + "…"
    return out


def _truncate_payload(output: str, max_len: int = 8000) -> str:
    if not output:
        return ""
    s = str(output)
    if len(s) > max_len:
        return s[:max_len] + f"\n… [truncated {len(s) - max_len} chars]"
    return s


# ------------------------------------------------------------------ dream pass

def project_dream_pass(store, llm_summarize=None) -> int:
    """Per-project consolidation step run from ``core/dream.py``.

    Walks every ACTIVE project, collects any ``autoadvance_step`` /
    ``task_updated`` events since the last dream pass, and writes a
    single consolidated ``dream_digest`` event with the takeaways.
    Returns the number of digests written.

    The ``llm_summarize`` arg is optional — when absent, we log a
    raw event count so the dream pass is still useful without an LLM.
    Reading this back in future sessions gives the agent a quick "what
    did I do last night" handle without replaying raw events.
    """
    if store is None:
        return 0
    count = 0
    for proj in store.list_projects(status_filter="ACTIVE"):
        pid = proj["id"]
        events = store.list_events(pid, limit=200)
        relevant = [
            e for e in events
            if e["type"] in {"autoadvance_step", "task_updated", "artifact_added"}
        ]
        if not relevant:
            continue
        payload: Dict[str, Any] = {"event_count": len(relevant)}
        if llm_summarize is not None:
            try:
                payload["summary"] = llm_summarize(relevant)
            except Exception:
                logger.debug("dream summarize failed", exc_info=True)
        store.log_event(pid, None, "dream_digest", payload)
        count += 1
    return count
