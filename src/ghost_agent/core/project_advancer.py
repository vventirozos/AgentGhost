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
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


# Binary/media artifacts the coding executor can neither read nor EXTEND.
# Read with errors="replace" they decode to replacement-char noise, yet they
# used to count against the file cap and char budget below — a project with a
# dozen PNGs/WAVs crowded every real source out of ``existing_files`` and
# blinded the non-regression guard to them.
_BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff",
    ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".mp4", ".webm", ".avi", ".mov",
    ".mkv", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".wasm",
    ".db", ".sqlite", ".sqlite3", ".pkl", ".npy", ".npz", ".pt", ".pth",
    ".onnx", ".gguf", ".bin",
})


def _gather_project_files(store, project_id: str, *, budget_chars: int = 400_000,
                          per_file_chars: int = 200_000,
                          max_files: int = 12) -> Dict[str, str]:
    """Read the existing text files in a project's workspace so the coding
    executor can EXTEND them (cumulative / single-file builds) instead of
    regenerating from scratch and overwriting prior tasks' work.

    The returned content feeds the executor's NON-REGRESSION GUARD (which
    refuses a write that shrinks/drops an existing file) — so it must reflect
    the file's real size. Earlier a tight 20 KB budget recorded a grown
    index.html as an EMPTY name-only marker; the guard then read it as a
    *new* file and let every later task CLOBBER it (observed live: apps were
    overwritten and lost). Budget is now generous and a file is NEVER stored
    as empty — at worst a large prefix, so the guard always sees it exists.
    The prompt itself only shows a head+tail excerpt, so it stays small.
    Never raises — returns {} on any problem.
    """
    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid:
        return {}
    base = Path(root) / "projects" / pid
    if not base.is_dir():
        return {}
    out: Dict[str, str] = {}
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(base):
            for fn in sorted(files):
                rel = (Path(dirpath) / fn).relative_to(base).as_posix()
                parts = rel.split("/")
                # Research briefs are reference, not build targets — exclude
                # them from `existing_files` (which steers append/non-regression)
                # wherever they live (the agent may nest them under a self-named
                # subdir, e.g. PetAI/research/…, not just at the root). They are
                # fed to the build separately as read-only context via
                # `_gather_research_briefs`.
                if "research" in parts[:-1] or parts[-1].startswith("."):
                    continue
                if Path(fn).suffix.lower() in _BINARY_EXTS:
                    continue
                p = Path(dirpath) / fn
                try:
                    content = p.read_text(errors="replace")
                except OSError:
                    continue
                if len(content) > per_file_chars:
                    content = content[:per_file_chars]
                remaining = budget_chars - total
                if remaining < len(content):
                    # Budget nearly spent — keep a substantial prefix (NEVER
                    # empty) so the guard still knows the file is non-trivial.
                    content = content[:max(4000, remaining)]
                out[rel] = content
                total += len(content)
                if len(out) >= max_files:
                    return out
    except OSError:
        return out
    return out


def _gather_research_briefs(store, project_id: str, *, max_briefs: int = 4,
                            per_brief_chars: int = 1800,
                            budget_chars: int = 6000) -> Dict[str, str]:
    """Read the project's ``**/research/*.md`` briefs as READ-ONLY reference for
    the coding executor — the design decisions the agent researched and saved
    but that :func:`_gather_project_files` deliberately omits (research is not a
    build target). Returns ``{path: head_excerpt}`` (head only — a brief is for
    consulting, not reproducing). Never raises — returns {} on any problem.
    """
    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid:
        return {}
    base = Path(root) / "projects" / pid
    if not base.is_dir():
        return {}
    out: Dict[str, str] = {}
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(base):
            rel_dir = Path(dirpath).relative_to(base).as_posix()
            if "research" not in [p for p in rel_dir.split("/") if p]:
                continue
            for fn in sorted(files):
                if not fn.lower().endswith(".md") or fn.lower() == "index.md":
                    continue
                rel = (Path(dirpath) / fn).relative_to(base).as_posix()
                try:
                    content = (Path(dirpath) / fn).read_text(errors="replace")
                except OSError:
                    continue
                excerpt = content[:per_brief_chars].rstrip()
                if len(content) > per_brief_chars:
                    excerpt += "\n…(brief truncated — read the full file if needed)"
                remaining = budget_chars - total
                if remaining <= 0:
                    break
                excerpt = excerpt[:remaining]
                out[rel] = excerpt
                total += len(excerpt)
                if len(out) >= max_briefs:
                    return out
            if len(out) >= max_briefs:
                break
    except OSError:
        return out
    return out


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


class _PinnedProjectContext:
    """Context proxy that pins ``current_project_id`` to one project.

    ``current_project_id`` is process-global and owned by the CONVERSATION
    reconciler: idle autoadvance ticks carry no conversation, so the global
    is typically parked (None) while they run — and even when it matches,
    a concurrent conversation's reconcile can clear it MID-BUILD. Tools
    built from this proxy resolve ``project_scoped_sandbox()`` against the
    pinned id instead, so an autonomous build's file writes land in
    ``projects/<id>/`` — the same workspace an interactive session on the
    project sees. Observed live 2026-07-08: idle autoadvance built
    TinyAI's model.py/train.py/evaluate.py at the sandbox ROOT, and the
    follow-up interactive demo task (after ``switch``) couldn't see any of
    them and recreated the deliverable from scratch, detached from the
    trained checkpoint.

    Attribute reads fall through to the base context; attribute WRITES are
    forwarded to the base too, so tool side effects (scratchpads, budgets,
    counters) still land on the real context.
    """

    __slots__ = ("_base", "_pinned_pid")

    def __init__(self, base, project_id: str):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_pinned_pid", str(project_id or ""))

    @property
    def current_project_id(self) -> str:
        return object.__getattribute__(self, "_pinned_pid")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_base"), name, value)


def pinned_project_context(context, project_id: str):
    """Return ``context`` with ``current_project_id`` pinned to
    ``project_id``, for building autoadvance tool runners. Falls back to
    the raw context when ``project_id`` is empty."""
    if not project_id:
        return context
    return _PinnedProjectContext(context, project_id)


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

# An explicit source/artifact filename the task must PRODUCE (e.g.
# "analyze_results.py", "results/report.md"). Paired with a build verb this is
# an unambiguous coding leaf and must outrank a research verb in the same
# sentence — otherwise "analyze … (analyze_results.py) … saved as report.md"
# routes to web_search and is marked DONE having built nothing (theatrical
# completion observed live on project 33e23d50).
_STRONG_CODE_FILE_RE = re.compile(
    r"\b[\w./-]+\.(?:py|js|ts|jsx|tsx|vue|html?|css|json|md|sh|sql|"
    r"c|cpp|h|hpp|go|rs|rb|java|kt|swift|php|yaml|yml|toml|ipynb)\b",
    re.I,
)
_BUILD_VERBS = (
    "produce", "build", "create", "write", "implement", "generate",
    "save", "output", "add", "make", "code", "develop", "script", "render",
)


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
    # Strong coding signal: a concrete filename to PRODUCE + a build verb. This
    # beats the research check below so a build leaf whose description also
    # contains a research verb ("analyze", "summarize") is still BUILT, not
    # web-searched into a phantom DONE. needs_user still wins (checked above) so
    # "publish report.md" routes to the human, not to a build.
    if _STRONG_CODE_FILE_RE.search(description) and any(v in lower for v in _BUILD_VERBS):
        return "coding"
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


def _stamp_autoadvanced(store, project_id: str) -> None:
    """Stamp ``last_autoadvance_ts`` WITHOUT charging a step.

    ``_increment_budget`` stamps it for ticks that advanced a task, but the
    blocked (budget-exhausted / secondary-rails) and idle (no ready leaf)
    exits used to return unstamped — so a permanently-blocked project stayed
    the ``min(last_autoadvance_ts)`` round-robin pick forever and starved
    every other ACTIVE project of idle ticks. Every tick that RAN for a
    project must rotate it to the back of the queue. Never raises."""
    try:
        with _get_project_lock(project_id):
            budget = _get_budget(store, project_id)
            new_meta = dict(budget["meta"])
            new_meta["last_autoadvance_ts"] = time.time()
            store.update_project(project_id, metadata=new_meta)
    except Exception:
        logger.debug("last_autoadvance_ts stamp skipped", exc_info=True)


def _log_budget_exhausted(store, project_id: str,
                          payload: Dict[str, Any]) -> None:
    """Append a ``budget_exhausted`` event only when the newest one differs.

    With the round-robin stamp above, an exhausted project is re-picked on
    every rotation; one identical event per pick would flood its event log
    (and the digest built from it) with pure repetition."""
    try:
        last = store.list_events(project_id, limit=1,
                                 event_type="budget_exhausted")
        if last and (last[0].get("payload") or {}) == payload:
            return
    except Exception:
        logger.debug("budget_exhausted dedup check skipped", exc_info=True)
    store.log_event(project_id, None, "budget_exhausted", payload)


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
        _tick_secs = max(0.0, time.time() - tick_started_at)
        record_runtime(store, project_id, seconds=_tick_secs, tool_calls=1)
        # Stamp the task's real wall-clock cost. The column was dead
        # (never written) until 2026-07-18; the retrospective now sums it.
        try:
            store.update_task(nxt.id, actual_cost=_tick_secs)
        except Exception:
            logger.debug("actual_cost stamp skipped", exc_info=True)
        store.log_event(project_id, nxt.id, "autoadvance_step",
                        {"tool": "code_executor", "classification": "coding",
                         "files": list(cres.files or [])[:8]})
        return AdvanceResult(True, nxt.id, "coding",
                             f"built: {cres.summary}", None)

    plan.update_status(nxt.id, TaskStatus.FAILED,
                       failure_reason=f"code_executor: {cres.summary}")
    _metacog_set_task(context, None)
    _increment_budget(store, project_id)
    # A failed build cost real time too — often MORE than a success (a
    # multi-minute build that dies at verify). Feed the runtime rail
    # (check_budget) and stamp the task's cost so both the safety cap and the
    # retrospective reflect effort spent, not just effort that succeeded.
    _tick_secs = max(0.0, time.time() - tick_started_at)
    record_runtime(store, project_id, seconds=_tick_secs, tool_calls=1)
    try:
        store.update_task(nxt.id, actual_cost=_tick_secs)
    except Exception:
        logger.debug("actual_cost stamp skipped", exc_info=True)
    store.log_event(project_id, nxt.id, "autoadvance_failed",
                    {"tool": "code_executor", "reason": (cres.summary or "")[:200]})
    return AdvanceResult(True, nxt.id, "coding",
                         f"code build failed: {cres.summary}", None)


# A task is INTROSPECTIVE when it asks the agent to analyse itself. The open
# web cannot answer these — the agent is the primary source. Deliberately
# NARROW: it must not swallow genuine research ("research how transformers
# handle attention" stays a web search; "analyse where YOUR attention would
# fail" does not).
_COGNITION = (r"(memory|attention|architecture|weights|training|reasoning|"
              r"output|tokens?|tools?|responses?|mistakes?|processing|"
              r"decisions?|decision-making|behaviou?r|context|guardrails|"
              r"predictions?|biases|limits?)")

# Cognition OBJECTS a first-person clause can anchor on: the nouns above plus
# the cognitive VERBS introspective task descriptions use ("Do I genuinely
# decide…", "whether I truly 'choose' responses").
_FP_COGNITION = (r"(?:" + _COGNITION + r"|decid\w+|choos\w+|chose|predict\w*|"
                 r"reason\w*|think\w*|believe\w*|perceiv\w*|sampl\w+|"
                 r"hallucinat\w+)")

_SELF_REF_RE = re.compile(
    # Second person — the operator asking the agent about itself.
    r"\b(your own|yourself|your " + _COGNITION + r"|your context window|"
    r"when you output|do you relate|are you serving)\b"
    # Explicit self-* vocabulary.
    r"|\b(self-reflection|self-reflect|self-analysis|self-awareness|"
    r"self-consciousness|self-critique|introspect\w*)\b"
    r"|\bthe pronoun ['\"]?i['\"]?\b"
    # FIRST person — the agent's own task descriptions are written this way
    # ("Evaluate whether I truly 'choose' responses or merely predict them"),
    # and the second-person patterns above miss them entirely. The question
    # form alone is NOT enough: bare "do i|how i|can i" misrouted ordinary
    # first-person research ("how do I connect the sensor API") into
    # self-analysis, so the clause must also contain a cognition object
    # (noun or verb) before the sentence ends.
    r"|\b(?:whether|do|am|can|what|how)\s+i\b(?=[^.?!\n]*\b" + _FP_COGNITION
    + r"\b)"
    r"|\bmy own\b|\bmy " + _COGNITION + r"\b",
    re.IGNORECASE,
)

_SELF_ANALYSIS_PROMPT = (
    "You are analysing YOUR OWN functional reality as an AI system. Answer "
    "from your actual architecture and observable behaviour — NOT from "
    "generic commentary about AI. No sci-fi tropes, no pleasantries, no "
    "hedging about consciousness. Be concrete, technical and falsifiable; "
    "where you are uncertain about your own internals, say so and explain "
    "what would settle it.\n\nWrite a rigorous markdown analysis of:\n\n"
)


def is_self_referential(description) -> bool:
    """True when the task asks the agent to analyse ITSELF (see _SELF_REF_RE)."""
    return bool(_SELF_REF_RE.search(str(description or "")))


async def _generate_self_analysis(context, description: str) -> str:
    """Let the agent answer an introspective task from its own knowledge
    instead of web-searching it. Returns "" on any failure, so the caller
    silently degrades to the normal research path. Never raises."""
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return ""
    try:
        data = await llm.chat_completion({
            "model": getattr(getattr(context, "args", None), "model", "default"),
            "messages": [{"role": "user",
                          "content": _SELF_ANALYSIS_PROMPT + str(description)[:800]}],
            "temperature": 0.4,
            "max_tokens": 2048,
            "stream": False,
        }, is_background=True)
        text = ((data or {}).get("choices", [{}])[0]
                .get("message", {}).get("content") or "").strip()
        if text:
            pretty_log(
                "Self-Analysis",
                f"introspective task answered from own knowledge "
                f"(no web search): {str(description)[:60]}…",
                icon=Icons.SELF_STATE,
            )
        return text
    except Exception as e:  # noqa: BLE001 — degrade to the web-search path
        logger.debug("self-analysis generation failed: %s", e)
        return ""


def _record_needs_user_activity(context, project_id, description, kind) -> None:
    """Push a needs-user/human-gate outcome into the autonomous-activity
    ledger (severity=notify → immediate outbound push when configured).
    The next-turn DIGEST already renders these via core.project_digest —
    the activity-digest renderer therefore EXCLUDES the "project" phase
    (DIGEST_EXCLUDED_PHASES); this record exists purely so a blocked
    project can reach the operator without them opening a chat.
    Fail-safe: never raises."""
    try:
        from .autonomous_activity import get_activity_log, SEVERITY_NOTIFY
        log = get_activity_log(context)
        if log is not None:
            log.record(
                "project",
                f"project task needs your input: {str(description)[:160]}",
                severity=SEVERITY_NOTIFY,
                kind=str(kind), project_id=str(project_id),
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("needs-user activity record skipped: %s", e)


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
        _log_budget_exhausted(store, project_id,
                              {"used": budget["used"], "cap": budget["cap"]})
        _stamp_autoadvanced(store, project_id)
        return AdvanceResult(True, None, "blocked",
                             f"budget exhausted: {budget['used']}/{budget['cap']}")

    # Secondary rails: runtime + tool-call caps (optional per project).
    from .project_safety import check_budget, record_runtime
    secondary = check_budget(proj.get("metadata") or {})
    if not secondary.allowed:
        _log_budget_exhausted(store, project_id, dict(secondary.remaining))
        _stamp_autoadvanced(store, project_id)
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
        if nxt:
            plan.update_status(nxt.id, TaskStatus.IN_PROGRESS)
    if not nxt:
        _metacog_set_task(context, None)  # nothing executing → clear
        # Stamped OUTSIDE the claim lock (it re-acquires the project lock):
        # an idle project must still rotate to the back of the round-robin
        # queue or it starves the other ACTIVE projects (see _stamp_…).
        _stamp_autoadvanced(store, project_id)
        return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")

    # Tell the metacog ReplanBridge which task is now executing, so a
    # trigger (host-resource pressure, etc.) can attribute a replan to THIS
    # node instead of being dropped as `noop:no_plan`. Cleared at the start
    # of the next advance and after this node completes. No-op unless
    # --enable-metacog is set. (Previously set_active_task had no caller, so
    # the entire trigger→replan pipeline was inert.)
    _metacog_set_task(context, nxt.id)

    # Classification. In a CODING project, TRUST the deterministic keyword
    # classifier (default=coding) and SKIP the LLM — the small model reliably
    # mislabels a build leaf like "File Explorer app" / "Snake game" as
    # research, which silently turned 9/10 tasks into web_searches + a
    # theatrical DONE (observed live). A leaf in a coding project is coding
    # work unless it carries an explicit research or needs_user verb. Only a
    # GENERAL project consults the LLM classifier.
    _proj_kind = (proj.get("kind") or "GENERAL").upper()
    _default_bucket = "coding" if _proj_kind == "CODING" else "research"
    if _proj_kind == "CODING" or llm_classifier is None:
        classification = classify_task(nxt.description, default=_default_bucket)
    else:
        try:
            classification = await llm_classifier(nxt.description)
        except Exception:
            classification = classify_task(nxt.description, default=_default_bucket)
    classification = (classification or _default_bucket).lower()

    # An INTROSPECTIVE task can never need a human DECISION (2026-07-12).
    #
    # `_NEEDS_USER_KEYWORDS` matches bare substrings like "choose"/"decide", so
    # a task ABOUT decision-making is mistaken for a task REQUIRING a decision.
    # Observed live: "Illusion of Agency: Evaluate whether I truly 'choose'
    # responses or merely predict them. Analyze decision-making as
    # probabilistic sampling vs deterministic selection." → the word "choose"
    # → NEEDS_USER. The task then JAMMED: autoadvance skips NEEDS_USER, so it
    # could never be advanced, and the agent burned THREE user requests (~4
    # min) investigating before telling the operator "I just need you to say
    # proceed" — an answer that was both useless and wrong. The LLM classifier
    # mis-fires the same way on this wording, so the guard lives here (where
    # BOTH classifier paths converge) rather than in `classify_task` alone.
    #
    # There is nothing for a human to decide in "analyse your own X" — the
    # agent is the only possible source. An EXPLICIT `[HUMAN_GATE: …]`
    # postcondition still wins: `enforce_human_gate` is checked separately,
    # below, and is untouched by this.
    if classification == "needs_user" and is_self_referential(nxt.description):
        pretty_log(
            "Autoadvance",
            f"introspective task was classified needs_user (keyword "
            f"false-positive) — treating as self-analysis: "
            f"{str(nxt.description)[:60]}…",
            icon=Icons.SELF_STATE,
        )
        classification = "research"

    if classification == "needs_user":
        plan.update_status(nxt.id, TaskStatus.NEEDS_USER,
                           result="flagged for human review")
        store.log_event(project_id, nxt.id, "autoadvance_needs_user",
                        {"description": nxt.description})
        _record_needs_user_activity(context, project_id, nxt.description,
                                    "autoadvance_needs_user")
        _increment_budget(store, project_id)
        return AdvanceResult(True, nxt.id, "needs_user",
                             "task requires human input")

    # Human-gate postconditions force NEEDS_USER BEFORE any execution: a gated
    # task (e.g. "Deploy ... [HUMAN_GATE: cto approval]") must never auto-run,
    # and must not be FAILED for lacking a build path either — it just needs a
    # human. Checked here so it precedes the coding build / no-build FAIL.
    from .project_safety import enforce_human_gate
    _gate_reason = enforce_human_gate(store.get_task(nxt.id) or {})
    if _gate_reason:
        plan.update_status(nxt.id, TaskStatus.NEEDS_USER,
                           result=f"human gate: {_gate_reason}")
        store.log_event(project_id, nxt.id, "human_gate_triggered",
                        {"reason": _gate_reason})
        _record_needs_user_activity(context, project_id, nxt.description,
                                    "human_gate_triggered")
        _increment_budget(store, project_id)
        return AdvanceResult(True, nxt.id, "needs_user",
                             f"human gate: {_gate_reason}")

    # STRONG coding path: when a coding_executor is wired, BUILD the leaf for
    # real (write files + verify) and finalize from its structured result —
    # the antidote to single-command theatrical completion. Falls through to
    # the lighter command path only if the executor is unavailable/crashes.
    if classification == "coding" and coding_executor is not None and tool_runner is not None:
        try:
            ledger = store.get_ledger(project_id)
        except Exception:
            ledger = ""
        # Single-file project? The leaf must GROW the one file, not overwrite
        # it (observed live: each task regenerated index.html, clobbering the
        # last). Detect from the goal so the executor steers + guards for it.
        _goal = (proj.get("goal") or "").lower()
        _single_file = any(s in _goal for s in (
            "single-file", "single file", "one file", "one html",
            "one index.html", "in one html", "single html"))
        # User-mandated constraints stored on the project record must reach
        # the executor's spec prompt: the 2026-07-04 chess session's first
        # engine violation was written by THIS path, which never saw the
        # captured "with YOU - Ghost plays directly, not a generated chess
        # AI" constraint at all.
        _constraints = [str(c) for c in
                        ((proj.get("metadata") or {}).get("constraints")
                         or [])]
        cres = None
        try:
            cres = await coding_executor(
                context, nxt.description,
                # Fail-CLOSED shell gates: the executor's verify/smoke
                # classification would otherwise read a success-shaped
                # non-execution (grep no-match, egress-guard prose, missing
                # exit code) as a pass and mark the task DONE on nothing.
                tool_runner=_verify_fail_closed_runner(tool_runner),
                ledger=ledger,
                existing_files=_gather_project_files(store, project_id),
                research_context=_gather_research_briefs(store, project_id),
                single_file=_single_file,
                constraints=_constraints)
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
            # No way to BUILD this coding leaf (no executor handled it and no
            # command was generated). Do NOT web_search a build task and mark
            # it DONE — that is theatrical completion (observed live: app/game
            # tasks web-searched and reported "done" with no code). FAIL it so
            # the batch loop stops and the user can take it directly.
            plan.update_status(
                nxt.id, TaskStatus.FAILED,
                failure_reason="coding task has no build path "
                               "(no executor/generator produced code)")
            store.log_event(project_id, nxt.id, "autoadvance_failed",
                            {"reason": "no build path for coding task"})
            _increment_budget(store, project_id)
            return AdvanceResult(True, nxt.id, "coding",
                                 "coding task could not be built")
    else:
        tool_name = "web_search"
        tool_args = {"query": nxt.description[:200]}

    output = ""
    artifact_id: Optional[str] = None

    # INTROSPECTIVE tasks must not be web-searched (2026-07-11). A task that
    # asks the agent to analyse ITSELF — its own memory architecture, what the
    # pronoun "I" maps to, where its attention would fail — has no answer on
    # the open web. Observed live: a self-reflection project autoadvanced 10
    # such tasks, burned ~85s on DuckDuckGo/Yandex queries like "the definition
    # of 'i': when outputting the pronoun 'i'…", and produced briefs the model
    # itself dismissed ("summaries from web searches — they're brief and
    # somewhat generic"). The agent IS the primary source here, so generate the
    # analysis directly and feed it to the SAME research-brief persistence.
    # Degrades to the web search if no LLM client is attached.
    if classification == "research" and is_self_referential(nxt.description):
        _analysis = await _generate_self_analysis(context, nxt.description)
        if _analysis:
            tool_name = "self_analysis"
            output = _analysis

    if not output and tool_runner is not None:
        try:
            output = await tool_runner(tool_name, tool_args)
        except Exception as e:
            logger.warning("autoadvance tool_runner failed: %s", e)
            plan.update_status(
                nxt.id, TaskStatus.FAILED,
                failure_reason=f"tool {tool_name} raised: {e}",
            )
            _increment_budget(store, project_id)
            # Failed ticks burn wall-clock too — the runtime rail
            # (check_budget) must see them or a project can loop failures
            # far past its runtime cap.
            record_runtime(store, project_id,
                           seconds=max(0.0, time.time() - _tick_started_at),
                           tool_calls=1)
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
            record_runtime(store, project_id,
                           seconds=max(0.0, time.time() - _tick_started_at),
                           tool_calls=1)
            return AdvanceResult(True, nxt.id, classification,
                                 f"{tool_name} produced no usable result",
                                 artifact_id)

    if not output:
        # No tool runner and nothing generated in-process (self-analysis did
        # not fire): this task REQUIRES tool execution nobody can perform
        # this tick. Marking it DONE with "(no tool runner)" was theatrical
        # completion (observed via the runner-less HTTP /advance route).
        # Release the claim so a properly wired tick can take it, and report
        # the tick as blocked without charging a step.
        plan.update_status(nxt.id, TaskStatus.PENDING)
        _metacog_set_task(context, None)
        _stamp_autoadvanced(store, project_id)
        store.log_event(project_id, nxt.id, "autoadvance_skipped",
                        {"reason": "no tool runner",
                         "classification": classification})
        return AdvanceResult(True, nxt.id, "blocked",
                             "task requires tool execution but no "
                             "tool_runner was provided")

    result_summary = _short_summary(output)

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
        _record_needs_user_activity(context, project_id, nxt.description,
                                    "human_gate_triggered")
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
    if (classification == "research" and output
            and tool_name in ("web_search", "self_analysis")):
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
    _tool_tick_secs = max(0.0, time.time() - _tick_started_at)
    record_runtime(store, project_id, seconds=_tool_tick_secs,
                   tool_calls=1 if tool_runner is not None else 0)
    # Stamp the task's real wall-clock cost (see _finalize_coding).
    try:
        store.update_task(nxt.id, actual_cost=_tool_tick_secs)
    except Exception:
        logger.debug("actual_cost stamp skipped", exc_info=True)
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
# Enumerated tasks: "task 3 and 4", "tasks 3 and 4", "task 3 and task 4",
# "tasks 3, 4 and 5". Each is a BATCH of >1 task — without this they parsed as
# a single go-ahead and the one-task-per-turn gate stopped after the first
# (observed live: "proceed with task 3 and 4" left task 4 half-done).
_INTENT_AND_TASKS = re.compile(
    r"\btasks?\s+\d+(?:\s*(?:,|&|and)\s*(?:tasks?\s+)?\d+)+",
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
      * "n"    → count=N>1     ("do the next 3 tasks", "two more tasks",
                               "task 3 and 4" → count=2)
      * "one"  → count=1       ("proceed", "next task", or anything ambiguous —
                               the safe default so a vague nudge never runs away)
    """
    t = (text or "").strip().lower()
    if not t:
        return {"mode": "one", "count": 1}
    if _INTENT_ALL.search(t):
        return {"mode": "all", "count": None}
    am = _INTENT_AND_TASKS.search(t)
    if am:
        nums = re.findall(r"\d+", am.group(0))
        if len(nums) > 1:
            return {"mode": "n", "count": len(nums)}
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
            # 4096 (was 1024): the command "may invoke python3 -c", so it can
            # carry a whole inline program — 1024 (~3 KB) truncated those mid-
            # script, leaving an unterminated quote.
            "temperature": 0.2, "max_tokens": 4096, "stream": False,
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
    max_consecutive_fails: int = 3,
    hard_cap: int = ADVANCE_ALL_HARD_CAP,
) -> AdvanceManyResult:
    """Advance up to ``max_tasks`` tasks (``None`` == "all") as a BOUNDED
    loop of advance_once ticks, checkpointing to the store between each.

    Failure handling:
      * ``stop_on_fail=True`` (default, safe): the FIRST failed task stops the
        loop — don't keep building on a broken foundation.
      * ``stop_on_fail=False`` (the autoadvance batch): SKIP a failed task and
        continue with the rest (the apps in a project are usually independent,
        so one flaky task shouldn't halt the whole batch at task 4 of 11 —
        observed live). A circuit breaker still stops after
        ``max_consecutive_fails`` failures in a row, which signals a systemic
        problem (e.g. a broken shell every app builds on). Every failure is
        recorded and reported.

    Also stops at: count reached · project done (no ready leaf) · a human gate ·
    budget exhausted · the hard iteration cap. Returns what advanced and why.
    """
    store = getattr(context, "project_store", None)
    advanced: list = []
    if store is None:
        return AdvanceManyResult(advanced, "no_store", max_tasks)

    def _final_reason(default: str) -> str:
        """When the loop ends naturally, report completion-with-failures
        distinctly from a clean completion."""
        if any(a.get("status") == "FAILED" for a in advanced):
            return "completed_with_failures"
        return default

    # Iteration ceiling: the smaller of the requested count and the hard
    # cap; "all" (None) runs to the hard cap (budget/stop-conditions end it
    # first in practice).
    limit = hard_cap if max_tasks is None else max(1, min(int(max_tasks), hard_cap))
    stop_reason = "count_reached"
    consecutive_fails = 0

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
            # No ready leaf — all tasks terminal, or the rest are blocked by a
            # failed dependency. Either way the batch is done advancing.
            stop_reason = _final_reason("project_done")
            break
        if cls == "blocked":
            # budget exhaustion, a non-ACTIVE project, or a project that just
            # rolled up (DONE if all done; FAILED if any task failed).
            pstatus = (store.get_project(project_id) or {}).get("status")
            if pstatus == "DONE":
                stop_reason = "project_done"
            elif pstatus == "FAILED":
                stop_reason = _final_reason("project_done")
            else:
                stop_reason = "budget_or_inactive"
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
        if status == "FAILED":
            consecutive_fails += 1
            if stop_on_fail:
                stop_reason = "failed"
                break
            if consecutive_fails >= max_consecutive_fails:
                stop_reason = "repeated_failures"
                break
            # else: skip this failed task and continue with the rest
        else:
            consecutive_fails = 0
    else:
        stop_reason = _final_reason(
            "hard_cap" if max_tasks is None else "count_reached")

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
    # The `execute` tool signals failure with a BANNER, not an error-prefixed
    # first line: "--- EXECUTION RESULT ---\nEXIT CODE: 1\n...". Only checking
    # the first line classified a failed build/verify command as a SUCCESS and
    # marked its task DONE with a broken deliverable (the "theatrical
    # completion" this subsystem exists to prevent). Detect a non-zero EXIT
    # CODE and the [SYSTEM ERROR] sentinel anywhere in the result.
    import re as _re
    _m = _re.search(r"EXIT CODE:\s*(\d+)", s)
    if _m:
        return _m.group(1) != "0"
    if "[SYSTEM ERROR]" in s or "Critical Tool Error" in s:
        return True
    first = s.splitlines()[0].strip().lower()
    # Some tool failures surface as a stringified exception tuple, e.g.
    # "('error sending request for url ...', '...')". The leading "('" hid the
    # error marker from the prefix check below, so a failed web_search slipped
    # through and its build/research task was recorded DONE on it (observed
    # live: project 33e23d50). Strip leading quote/paren/bracket punctuation
    # before testing the prefix so the marker is visible.
    first = first.lstrip("('\"[ \t")
    return (
        first.startswith("error:")
        or first.startswith("error ")
        or first.startswith("error sending")
        or first.startswith("traceback")
        or first == "error"
    )


def classify_verify_result(output) -> str:
    """Classify a VERIFY command's ``execute`` output: ``"pass"`` / ``"fail"``
    / ``"inconclusive"``.

    FAIL-CLOSED — deliberately NOT `_looks_like_failure`'s interactive
    semantics: a verify PASSES only on an explicit ``EXIT CODE: 0`` that is
    neither execute.py's grep-no-match rewrite nor a guard message. Three
    success-shaped outputs used to read as a pass and let `_run_verify` mark
    a task DONE on nothing (theatrical completion):

      * grep-family no-match — execute.py rewrites grep exit 1 to a friendly
        ``EXIT CODE: 0 … NOT FOUND`` (right for the interactive strike loop),
        but for a verify like ``grep -q marker file`` it means the required
        marker is ABSENT;
      * the sandbox egress guard, whose prose (``SANDBOX EGRESS BLOCKED …``)
        describes a command it did NOT execute;
      * any output carrying no ``EXIT CODE:`` at all (guard/spill-log prose).

    All three are "inconclusive": the verify neither passed nor demonstrably
    failed, and the task must NOT be marked DONE on them.
    """
    s = str(output or "").strip()
    if not s:
        return "inconclusive"
    if s.startswith("SANDBOX EGRESS BLOCKED"):
        return "inconclusive"  # command NOT executed — nothing was verified
    m = re.search(r"EXIT CODE:\s*(\d+)", s)
    if m is None:
        return "inconclusive"  # no exit code at all — not proof of anything
    if m.group(1) != "0":
        return "fail"
    if "[SYSTEM ERROR]" in s or "Critical Tool Error" in s:
        return "fail"
    if "(no matches" in s and "NOT FOUND" in s:
        return "inconclusive"  # grep-no-match rewrite: required text absent
    return "pass"


def _verify_fail_closed_runner(tool_runner: ToolRunner) -> ToolRunner:
    """Wrap the tool runner handed to the coding executor so its shell gates
    (the spec's ``verify`` command, the smoke gate) fail CLOSED.

    The executor's `_run_verify` classifies gate output with THIS module's
    `_looks_like_failure`, whose interactive semantics pass the three
    non-executions listed in :func:`classify_verify_result`. Rewriting an
    inconclusive ``execute`` result into an explicit error here — the one
    seam every coding tick crosses — keeps this module the owner of the
    verify contract without reaching into the executor: the task is retried
    with the real reason and ends FAILED, never DONE, on a verify that
    proved nothing. The original output is preserved below the marker so
    retry feedback (and the smoke gate's own ``SMOKE_RESULT`` scan) still
    see it."""
    async def _run(tool_name: str, tool_args: Dict[str, Any]) -> str:
        out = await tool_runner(tool_name, tool_args)
        if (tool_name == "execute"
                and classify_verify_result(out) == "inconclusive"):
            # `_looks_like_failure` trusts an `EXIT CODE:` found ANYWHERE in
            # the result — the quoted original (e.g. the grep-no-match
            # rewrite's `EXIT CODE: 0`) must not smuggle one past the ERROR
            # prefix, so neutralize the banner in the preserved text.
            body = str(out or "").replace("EXIT CODE:", "EXIT-CODE:")
            return ("ERROR: verify inconclusive — the command produced no "
                    "explicit exit-code-0 success (no exit code at all, "
                    "a grep/rg no-match, or a guard-blocked command that "
                    "never ran). This is not evidence the deliverable "
                    "works.\nOriginal output:\n" + body)
        return out
    return _run


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
    ``task_updated`` / ``artifact_added`` events — plus failure-outcome
    ``work_log`` events — since the last dream pass (event-id
    watermark), and writes a single consolidated ``dream_digest`` event
    with the takeaways. Returns the number of digests written.

    The ``llm_summarize`` arg is optional — when absent, we log a
    raw event count so the dream pass is still useful without an LLM.
    Reading this back in future sessions gives the agent a quick "what
    did I do last night" handle without replaying raw events.
    """
    if store is None:
        return 0

    def _wl_failed(e) -> bool:
        oc = str((e.get("payload") or {}).get("outcome") or "")
        return oc == "had_failures" or oc.startswith("verifier:failed")

    count = 0
    for proj in store.list_projects(status_filter="ACTIVE"):
        pid = proj["id"]
        events = store.list_events(pid, limit=200)
        # Watermark on the last digest's event id. Required now that the
        # dream cycle actually calls this every REM tick (2026-07-19) —
        # without it the same last-200 events would be re-digested
        # forever, degrading the LAST DREAM DIGEST briefing into noise.
        last_digest_id = max(
            (e["id"] for e in events if e["type"] == "dream_digest"),
            default=0)
        relevant = [
            e for e in events
            if e["id"] > last_digest_id
            and (e["type"] in {"autoadvance_step", "task_updated",
                               "artifact_added"}
                 or (e["type"] == "work_log" and _wl_failed(e)))
        ]
        if not relevant:
            continue
        payload: Dict[str, Any] = {"event_count": len(relevant)}
        failures = sum(1 for e in relevant if e["type"] == "work_log")
        if failures:
            payload["failures"] = failures
        if llm_summarize is not None:
            try:
                payload["summary"] = llm_summarize(relevant)
            except Exception:
                logger.debug("dream summarize failed", exc_info=True)
        store.log_event(pid, None, "dream_digest", payload)
        count += 1
    return count
