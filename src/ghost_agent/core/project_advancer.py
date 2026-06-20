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
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger("GhostAgent")


DEFAULT_STEPS_CAP = 50


# Per-project lock serializing the leaf-claim step of advance_once.
# advance_once is reachable concurrently from the autoadvance tool, the
# HTTP /advance route, and the scheduler; without serializing the
# read-leaf → mark-IN_PROGRESS step, two ticks would claim the SAME leaf
# and then double-run its tool, double-write artifacts, and double-charge
# budget. A threading.Lock (not asyncio) so it is correct whether ticks
# share one event loop or run on different threads; the claim span is
# fully synchronous (no await), so the lock is never held across a
# suspension point.
_project_locks: Dict[str, "threading.Lock"] = {}
_project_locks_guard = threading.Lock()


def _get_project_lock(project_id: str) -> "threading.Lock":
    with _project_locks_guard:
        lk = _project_locks.get(project_id)
        if lk is None:
            lk = threading.Lock()
            _project_locks[project_id] = lk
        return lk


# Keyword buckets used by the lightweight classifier. The lists are
# deliberately short and high-precision; anything not hitting one
# bucket defaults to RESEARCH, which is the safer autonomy mode (no
# sandbox side-effects).
_CODING_KEYWORDS = {
    "implement", "build", "write code", "refactor", "fix", "debug",
    "patch", "unit test", "add tests", "deploy", "migrate", "compile",
    "install", "scaffold", "ship", "benchmark",
    # File-creation / build signals — high precision so a project leaf like
    # "create a file hello.txt" or "build the parser module" routes to the
    # real coding executor instead of being web-searched (theatrical
    # completion observed live). File-extension tokens are unambiguous.
    "create a file", "write a file", "make a file", "create file",
    "create the file", "a script", "html page", "web page", "webpage",
    "javascript", "css", "function ", "endpoint", "component", "module",
    ".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", ".sh",
    ".sql", ".jsx", ".tsx", ".vue",
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


def classify_task(description: str, default: str = "research") -> str:
    """Return one of "needs_user", "coding", "research".

    Precedence: ``needs_user`` → ``research`` → ``coding`` → ``default``.
    ``research`` beats ``coding`` because a task like "Research benchmarks"
    contains the word "benchmark" (coding keyword) but clearly intends
    reading/summarizing, not writing code. ``needs_user`` always wins so
    "Implement and approve X" is routed to the human.

    ``default`` is the bucket for a task that hits no keyword. It defaults to
    "research" (the side-effect-free autonomy mode), but a CODING-kind
    project passes ``default="coding"`` so an unlabelled build leaf reaches
    the real coding executor instead of being web-searched.
    """
    if not description:
        return default
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
    return default


def _get_budget(store, project_id: str) -> Dict[str, int]:
    proj = store.get_project(project_id) or {}
    meta = proj.get("metadata") or {}
    cap = int(meta.get("steps_cap", DEFAULT_STEPS_CAP))
    used = int(meta.get("steps_used", 0))
    return {"cap": cap, "used": used, "meta": meta}


def _increment_budget(store, project_id: str) -> None:
    # The read-modify-write below replaces the WHOLE metadata dict. Two
    # concurrent ticks on the same project (claiming different leaves) would
    # otherwise lose an increment — undercounting the budget so the cap can
    # be exceeded — and could clobber keys other writers (research index,
    # safety runtime) merged in between the read and the write. Callers
    # never hold this lock here (the leaf-claim span releases it before any
    # increment), and the span is fully synchronous, so this cannot deadlock.
    with _get_project_lock(project_id):
        budget = _get_budget(store, project_id)
        new_meta = dict(budget["meta"])
        new_meta["steps_used"] = budget["used"] + 1
        new_meta.setdefault("steps_cap", budget["cap"])
        # Stamp when this project was last autonomously advanced so the idle
        # scheduler can round-robin across ACTIVE projects (least-recently-
        # advanced first) instead of repeatedly picking whichever project sorts
        # to the top of `updated_at DESC` — which was always the one just
        # advanced, so a single project monopolised every tick and the rest
        # starved.
        new_meta["last_autoadvance_ts"] = time.time()
        store.update_project(project_id, metadata=new_meta)


def _metacog_set_task(context, task_id) -> None:
    """Best-effort: stash the executing task id on the metacog bundle so the
    ReplanBridge can attribute a triggered replan. No-op when metacog is off
    or absent. Never raises."""
    try:
        mc = getattr(context, "metacog", None)
        if mc is not None and getattr(mc, "enabled", False):
            mc.set_active_task(task_id)
    except Exception:
        pass


def _finalize_coding(context, store, plan, project_id, nxt, cres,
                     tick_started_at) -> AdvanceResult:
    """Persist the outcome of a real coding build (CodingResult) for one leaf.

    On success: register the produced files as deliverable artifacts (so the
    end-of-project cleanup keeps them), append the ledger note, mark DONE.
    On failure: mark FAILED with the build's reason — this stops the batch
    loop so the user can take the hard task themselves, instead of a shallow
    DONE.
    """
    from .planning import TaskStatus
    from .project_safety import record_runtime

    if cres.ok:
        for rel in (cres.files or []):
            try:
                store.register_file_artifact(nxt.id, rel)
            except Exception:
                logger.debug("artifact register skipped: %s", rel, exc_info=True)
        if cres.ledger_note:
            try:
                store.append_ledger(project_id, cres.ledger_note)
            except Exception:
                logger.debug("ledger append skipped", exc_info=True)
        plan.update_status(nxt.id, TaskStatus.DONE,
                           result=cres.summary, actual_tool="code_executor")
        _metacog_set_task(context, None)
        _increment_budget(store, project_id)
        record_runtime(store, project_id,
                       seconds=max(0.0, time.time() - tick_started_at),
                       tool_calls=1)
        store.log_event(project_id, nxt.id, "autoadvance_step",
                        {"tool": "code_executor", "classification": "coding",
                         "files": list(cres.files or [])[:8]})
        return AdvanceResult(True, nxt.id, "coding",
                             f"built: {cres.summary}", None)

    plan.update_status(nxt.id, TaskStatus.FAILED,
                       failure_reason=f"code_executor: {cres.summary}")
    _metacog_set_task(context, None)
    _increment_budget(store, project_id)
    store.log_event(project_id, nxt.id, "autoadvance_failed",
                    {"tool": "code_executor", "reason": (cres.summary or "")[:200]})
    return AdvanceResult(True, nxt.id, "coding",
                         f"code build failed: {cres.summary}", None)


async def advance_once(
    context,
    project_id: str,
    tool_runner: Optional[ToolRunner] = None,
    llm_classifier: Optional[LLMClassifier] = None,
    code_generator: Optional[Callable[[str], Awaitable[str]]] = None,
    coding_executor: Optional[Callable[..., Awaitable[Any]]] = None,
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
      coding_executor: async callable ``(context, description, *,
        tool_runner, ledger) -> CodingResult`` that BUILDS a coding leaf for
        real (writes files, verifies). When supplied it handles every coding
        task — the strong path. See :mod:`core.coding_executor`.
      code_generator: async callable ``(description) -> str`` — the weaker
        fallback used only when ``coding_executor`` is absent or crashes: it
        returns a single executable command run via ``execute``. Without
        either, a coding task degrades to *researching* the task.
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

    # Atomically claim the next leaf. The original code marked IN_PROGRESS
    # only AFTER `await llm_classifier(...)`, so on a single event loop
    # that await was a preemption point where a concurrent tick grabbed
    # the SAME leaf (and across threads it raced outright). Claim the leaf
    # — read it and mark IN_PROGRESS — while holding the per-project lock
    # and BEFORE any await, so the claim is atomic. Classification and the
    # tool run then happen OUTSIDE the lock, so different leaves still
    # advance in parallel.
    with _get_project_lock(project_id):
        plan = ProjectPlan(store, project_id)
        nxt = plan.next_ready_leaf()
        if not nxt:
            _metacog_set_task(context, None)  # nothing executing → clear
            return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")
        plan.update_status(nxt.id, TaskStatus.IN_PROGRESS)

    # Tell the metacog ReplanBridge which task is now executing, so a
    # trigger (host-resource pressure, etc.) can attribute a replan to THIS
    # node instead of being dropped as `noop:no_plan`. Cleared at the start
    # of the next advance and after this node completes. No-op unless
    # --enable-metacog is set. (Previously set_active_task had no caller, so
    # the entire trigger→replan pipeline was inert.)
    _metacog_set_task(context, nxt.id)

    # A CODING-kind project biases an unlabelled leaf toward coding (so a
    # build task that hits no keyword reaches the executor, not web_search).
    _default_bucket = "coding" if (proj.get("kind") == "CODING") else "research"
    if llm_classifier is None:
        classification = classify_task(nxt.description, default=_default_bucket)
    else:
        try:
            classification = await llm_classifier(nxt.description)
        except Exception:
            classification = classify_task(nxt.description, default=_default_bucket)
    classification = (classification or _default_bucket).lower()

    if classification == "needs_user":
        plan.update_status(nxt.id, TaskStatus.NEEDS_USER,
                           result="flagged for human review")
        store.log_event(project_id, nxt.id, "autoadvance_needs_user",
                        {"description": nxt.description})
        _increment_budget(store, project_id)
        return AdvanceResult(True, nxt.id, "needs_user",
                             "task requires human input")

    # STRONG coding path: when a coding_executor is wired, BUILD the leaf for
    # real (write files + verify) and finalize from its structured result —
    # the antidote to single-command theatrical completion. Falls through to
    # the lighter command path only if the executor is unavailable/crashes.
    if classification == "coding" and coding_executor is not None and tool_runner is not None:
        try:
            ledger = store.get_ledger(project_id)
        except Exception:
            ledger = ""
        cres = None
        try:
            cres = await coding_executor(
                context, nxt.description, tool_runner=tool_runner, ledger=ledger)
        except Exception as e:
            logger.warning("coding_executor crashed: %s", e)
        if cres is not None:
            return _finalize_coding(context, store, plan, project_id, nxt,
                                    cres, _tick_started_at)

    # Pick a tool by classification. Research → web_search,
    # coding → execute (sandbox runs it). A missing tool runner means
    # we can only classify + mark, not actually execute — still a
    # useful signal, so we don't treat it as a hard failure.
    if classification == "coding":
        tool_name = "execute"
        generated = ""
        if code_generator is not None:
            try:
                generated = (await code_generator(nxt.description) or "").strip()
            except Exception as e:
                logger.warning("autoadvance code_generator failed: %s", e)
                generated = ""
        if generated:
            tool_args = {"command": generated}
        else:
            # No generator wired (or it produced nothing usable): research
            # the task to gather context rather than executing an inert
            # comment. The previous behaviour ran a shell COMMENT
            # ("# Autoadvance stub for: …") which marked the task DONE
            # having executed nothing — that is now fixed.
            tool_name = "web_search"
            tool_args = {"query": nxt.description[:200]}
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

        # Stricter completion: a tool that ran but returned an error or
        # produced nothing usable must NOT be recorded as DONE. The old
        # path marked the task DONE on *any* output, so a failed
        # web_search ("ERROR: …") or an empty result still counted as
        # progress — theatrical completion. Detect the failure signal and
        # fail the task instead, so the next tick can retry/alternative it
        # rather than leaving a dead task masquerading as finished. (Only
        # applies when a runner actually executed — the no-runner classify-
        # only path below still marks DONE, as before.)
        if _looks_like_failure(output):
            reason = (_short_summary(output) or "empty tool output") if output \
                else "tool produced no output"
            plan.update_status(
                nxt.id, TaskStatus.FAILED,
                failure_reason=f"{tool_name} failed: {reason}",
            )
            store.log_event(project_id, nxt.id, "autoadvance_failed",
                            {"tool": tool_name, "reason": reason[:200]})
            _increment_budget(store, project_id)
            return AdvanceResult(True, nxt.id, classification,
                                 f"{tool_name} produced no usable result",
                                 artifact_id)

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

    # Auto-research persistence: when a research-classified task ran a real
    # web_search and got usable output, turn that output into a durable,
    # summarised brief in the project's workspace (research/<slug>.md) +
    # index it, so background auto-advance leaves persistent findings the
    # agent stays aware of (surfaced in the project briefing) rather than
    # just a transient tool_call artifact. Reuses the output already in
    # hand — no second search. Best-effort: never breaks the tick.
    research_path: Optional[str] = None
    if classification == "research" and output and tool_name == "web_search":
        try:
            from .project_research import persist_research_from_output
            rr = await persist_research_from_output(
                context, project_id, nxt.description, output, task_id=nxt.id)
            if rr.ok:
                research_path = rr.path
                # Prefer the synthesised summary as the task's result.
                if rr.summary:
                    result_summary = _short_summary(rr.summary)
        except Exception:
            logger.debug("auto-research persist skipped", exc_info=True)

    plan.update_status(nxt.id, TaskStatus.DONE,
                       result=result_summary, actual_tool=tool_name)
    _metacog_set_task(context, None)  # node finished → don't replan a done task
    _increment_budget(store, project_id)
    from .project_safety import record_runtime as _record
    _record(store, project_id,
            seconds=max(0.0, time.time() - _tick_started_at),
            tool_calls=1 if tool_runner is not None else 0)
    step_payload: Dict[str, Any] = {"tool": tool_name,
                                     "classification": classification}
    if research_path:
        step_payload["research_path"] = research_path
    store.log_event(project_id, nxt.id, "autoadvance_step", step_payload)
    summary = (f"advanced via {tool_name}"
               + (f"; saved research to {research_path}" if research_path else ""))
    return AdvanceResult(True, nxt.id, classification, summary, artifact_id)


# ──────────────────────────────────────────────────────────────────────
# Multi-task pacing
#
# A single "proceed"/"next" stays a full agent turn (higher quality — the
# agent writes files, runs, verifies). The loop below exists for the BATCH
# case ("do the next 3", "proceed with all remaining tasks"), where the
# alternative is one chat turn grinding the whole tree and flooding the
# context window. Each iteration is a bounded advance_once tick that
# checkpoints status+result to the store, so context stays bounded no
# matter how many tasks run. NOTE: advance_once's coding path generates a
# SINGLE command per task — adequate for scriptable/iterative work, lighter
# than a full agent turn for complex multi-file builds (a task that needs
# more will FAIL its tick and stop the loop for the user).
# ──────────────────────────────────────────────────────────────────────

# Backstop on an "all" run, independent of the project step budget — guards
# against an uncapped project. The per-project budget and the stop
# conditions normally end the loop well before this.
ADVANCE_ALL_HARD_CAP = 40

_INTENT_ALL = re.compile(
    r"\b(all|everything|every\s+(remaining\s+)?task|the\s+rest|"
    r"remaining\s+tasks?|finish\s+(the\s+)?project|complete\s+(the\s+)?project|"
    r"whole\s+(project|thing)|to\s+the\s+end|until\s+(it'?s\s+|you'?re\s+)?done|"
    r"keep\s+going\s+until)\b",
    re.I,
)
_INTENT_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_INTENT_N = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:more\s+|next\s+)?tasks?\b",
    re.I,
)


@dataclass
class AdvanceManyResult:
    """Outcome of a bounded advance_many loop."""
    advanced: list            # [{task_id, classification, status, summary}]
    stop_reason: str          # project_done · count_reached · needs_user ·
                              # budget_or_inactive · failed · hard_cap · no_store
    requested: Optional[int]  # count asked for; None == "all"

    @property
    def count(self) -> int:
        return len(self.advanced)


def classify_advance_intent(text: str) -> Dict[str, Any]:
    """Map a user pacing directive to ``{mode, count}``.

      * "all"  → count=None   ("proceed with all remaining tasks", "finish the project")
      * "n"    → count=N>1     ("do the next 3 tasks", "two more tasks")
      * "one"  → count=1       ("proceed", "next task", or anything ambiguous —
                               the safe default so a vague nudge never runs away)
    """
    t = (text or "").strip().lower()
    if not t:
        return {"mode": "one", "count": 1}
    if _INTENT_ALL.search(t):
        return {"mode": "all", "count": None}
    m = _INTENT_N.search(t)
    if m:
        tok = m.group(1)
        n = int(tok) if tok.isdigit() else _INTENT_NUM_WORDS.get(tok, 1)
        if n > 1:
            return {"mode": "n", "count": n}
    return {"mode": "one", "count": 1}


def default_llm_classifier(context):
    """An LLM-backed task classifier from ``context.llm_client`` (or None →
    callers fall back to the keyword heuristic). Mirrors the idle
    autoadvancer so the tool path and idle path classify identically."""
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return None
    model = getattr(getattr(context, "args", None), "model", "default")

    async def _classify(description: str) -> str:
        try:
            r = await llm.chat_completion({
                "model": model,
                "messages": [{"role": "user", "content": (
                    "Classify this task into EXACTLY one word: 'coding' "
                    "(writes/runs code), 'research' (reads/searches/"
                    "summarizes), or 'needs_user' (requires a human "
                    "decision/approval/publish). Output ONLY the one word."
                    "\n\nTASK: " + str(description)[:500])}],
                "temperature": 0.0, "max_tokens": 8, "stream": False,
            })
            out = ((r or {}).get("choices", [{}])[0]
                   .get("message", {}).get("content", "") or "").strip().lower()
            for label in ("needs_user", "coding", "research"):
                if label in out:
                    return label
        except Exception as e:  # pragma: no cover - network/LLM variance
            logger.debug("advance classify failed: %s", e)
        return classify_task(description)

    return _classify


def default_code_generator(context):
    """A single-command code generator from ``context.llm_client`` (or None).
    Produces ONE shell command per task — lighter than a full agent turn."""
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return None
    model = getattr(getattr(context, "args", None), "model", "default")

    async def _gen(description: str) -> str:
        r = await llm.chat_completion({
            "model": model,
            "messages": [{"role": "user", "content": (
                "Write a SINGLE shell command (you may invoke python3 -c). "
                "Output ONLY the command — no explanation, no markdown "
                "fences.\n\nTASK: " + str(description)[:500])}],
            "temperature": 0.2, "max_tokens": 1024, "stream": False,
        })
        out = ((r or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "").strip()
        if out.startswith("```"):
            out = out.strip("`")
            if "\n" in out:
                out = out.split("\n", 1)[1]
        return out.strip()

    return _gen


async def advance_many(
    context,
    project_id: str,
    *,
    max_tasks: Optional[int],
    tool_runner: Optional[ToolRunner] = None,
    llm_classifier: Optional[LLMClassifier] = None,
    code_generator: Optional[Callable[[str], Awaitable[str]]] = None,
    coding_executor: Optional[Callable[..., Awaitable[Any]]] = None,
    stop_on_fail: bool = True,
    hard_cap: int = ADVANCE_ALL_HARD_CAP,
) -> AdvanceManyResult:
    """Advance up to ``max_tasks`` tasks (``None`` == "all") as a BOUNDED
    loop of advance_once ticks, checkpointing to the store between each.

    Stops at the FIRST of: count reached · project done (no ready leaf) ·
    a human-gate / NEEDS_USER task · budget exhausted · a task FAILED (when
    ``stop_on_fail``) · the hard iteration cap. Each tick claims one leaf,
    runs one constrained step, and persists status+result, so per-tick
    context stays bounded however many tasks run. Returns what advanced and
    why it stopped — the caller reports that and waits for the next
    direction.
    """
    store = getattr(context, "project_store", None)
    advanced: list = []
    if store is None:
        return AdvanceManyResult(advanced, "no_store", max_tasks)

    # Iteration ceiling: the smaller of the requested count and the hard
    # cap; "all" (None) runs to the hard cap (budget/stop-conditions end it
    # first in practice).
    limit = hard_cap if max_tasks is None else max(1, min(int(max_tasks), hard_cap))
    stop_reason = "count_reached"

    for _ in range(limit):
        res = await advance_once(
            context, project_id,
            tool_runner=tool_runner,
            llm_classifier=llm_classifier,
            code_generator=code_generator,
            coding_executor=coding_executor,
        )
        cls = (res.classification or "").lower()
        if cls == "idle":
            stop_reason = "project_done"
            break
        if cls == "blocked":
            # "blocked" covers both budget exhaustion and a non-ACTIVE
            # project — and a project that JUST rolled to DONE because its
            # last task completed. Distinguish so we report completion as
            # completion, not as a budget stall.
            pstatus = (store.get_project(project_id) or {}).get("status")
            stop_reason = "project_done" if pstatus == "DONE" else "budget_or_inactive"
            break
        # The tick targeted a task — record it with its persisted status.
        status = None
        if res.task_id:
            status = (store.get_task(res.task_id) or {}).get("status")
        advanced.append({
            "task_id": res.task_id,
            "classification": res.classification,
            "status": status,
            "summary": res.summary,
        })
        if cls == "needs_user" or status == "NEEDS_USER":
            stop_reason = "needs_user"
            break
        if status == "FAILED" and stop_on_fail:
            stop_reason = "failed"
            break
    else:
        stop_reason = "hard_cap" if max_tasks is None else "count_reached"

    return AdvanceManyResult(advanced, stop_reason, max_tasks)


def _looks_like_failure(output: str) -> bool:
    """True when a tool ran but its output indicates failure or emptiness.

    Used to stop the advancer from recording a task DONE on a result that
    accomplished nothing. Deliberately conservative — it fires on an empty
    result or on an output whose FIRST non-empty line is an explicit error
    marker (the convention every tool here uses: ``ERROR: …``). It does NOT
    scan the whole body for the word "error", which would misfire on a
    legitimate search result that merely *mentions* errors.
    """
    if output is None:
        return True
    s = str(output).strip()
    if not s:
        return True
    first = s.splitlines()[0].strip().lower()
    return (
        first.startswith("error:")
        or first.startswith("error ")
        or first.startswith("traceback")
        or first == "error"
    )


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
