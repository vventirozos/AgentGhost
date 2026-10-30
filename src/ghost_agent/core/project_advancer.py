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
  logs a ``budget_exhausted`` event (unattended: autopilot pauses and the
  owner is told, §4MR). Only the owner raises the cap (``manage_projects``
  action=budget, §4MQ — the model's ``metadata=`` cannot) — budgets are a
  hard stop, not a soft warning. The cap counts EVERY advancer step, the
  owner's batches too.

The classifier is intentionally a keyword heuristic. An LLM classifier
can be plugged in later by passing ``llm_classifier=...``; the tests
verify both paths.
"""

import asyncio
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from ..utils.logging import Icons, pretty_log
from ..tools.file_system import read_text_nofollow, walk_nofollow  # §4GJ: symlink-safe walk/read over a model-writable tree
from .project_research import _idle_invocation
from .task_deliverable import declared_deliverable, task_topic

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


# Appended to a snapshot entry that holds only a PREFIX of the real file.
# Consumers must not treat a marked entry as the file's full prior content.
SNAPSHOT_TRUNCATED_MARK = "\n…[SNAPSHOT TRUNCATED — prefix only]"


def _gather_project_files(store, project_id: str, *, budget_chars: int = 400_000,
                          per_file_chars: int = 200_000,
                          max_files: int = 12,
                          description: str = "",
                          max_named_extra: int = 6) -> Dict[str, str]:
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

    §4QA-d: files the task ``description`` names (path or basename) are kept
    outside the ``max_files`` cap (up to ``max_named_extra`` of them) and
    come first — the alphabetical cap left a named target out of a 30-file project.
    Never raises — returns {} on any problem.
    """
    root = getattr(store, "sandbox_root", None)
    pid = str(project_id or "").strip().lower()
    if not root or not pid:
        return {}
    base = Path(root) / "projects" / pid
    if not base.is_dir():
        return {}
    from .coding_executor import _TASK_NAMED_FILE_RE
    from .coding_loop import _SNAPSHOT_SKIP_DIRS
    named = {m.group(1).lstrip("./").lower()
             for m in _TASK_NAMED_FILE_RE.finditer(description or "")}
    out: Dict[str, str] = {}
    first: list = []            # the named files read, in walk order

    def _eligible(rel: str) -> bool:
        parts = rel.split("/")
        # Research briefs are reference, not build targets — exclude
        # them from `existing_files` (which steers append/non-regression)
        # wherever they live (the agent may nest them under a self-named
        # subdir, e.g. PetAI/research/…, not just at the root). They are
        # fed to the build separately as read-only context via
        # `_gather_research_briefs`.
        if "research" in parts[:-1] or parts[-1].startswith("."):
            return False
        # §4QA-d review: with a named file the walk no longer stops at 12,
        # so vendored trees (node_modules/*/package.json) would fill the
        # named slots — skip them (r20: only these; .github/… stay readable)
        if any(p in _SNAPSHOT_SKIP_DIRS for p in parts[:-1]):
            return False
        return Path(parts[-1]).suffix.lower() not in _BINARY_EXTS

    # §4QA-d r21: choose the named files BEFORE reading, exact paths first —
    # counting slots during the walk let a root-level "package.json" or six
    # basename hits use the slots the exact path "zapp/package.json" needed
    chosen: set = set()
    if named:
        try:
            _cand = [(Path(d) / fn).relative_to(base).as_posix()
                     for d, files, _dfd in walk_nofollow(base) for fn in sorted(files)]
        except (OSError, ValueError):
            _cand = []
        _cand = [r for r in _cand if _eligible(r)]
        _exact = [r for r in _cand if r.lower() in named]
        _base = [r for r in _cand if r not in _exact and r.rsplit("/", 1)[-1].lower() in named]
        chosen = set((_exact + _base)[:max_named_extra])
    total = 0
    try:
        # §4GJ round 3: `os.walk` only declines to DESCEND a linked
        # DIRECTORY — a linked FILE is still listed and `read_text` follows
        # it, with no race needed. This walk lists regular files only and
        # the read is dir_fd-relative (atomic wrt every component).
        for dirpath, files, _dfd in walk_nofollow(base):
            for fn in sorted(files):
                rel = (Path(dirpath) / fn).relative_to(base).as_posix()
                if not _eligible(rel):
                    continue
                # a named file past its own cap competes for a plain slot
                is_named = rel in chosen
                if not is_named and len(out) - len(first) >= max_files:
                    continue
                try:
                    content = read_text_nofollow(fn, dir_fd=_dfd, errors="replace")
                except (OSError, ValueError):
                    continue
                if len(content) > per_file_chars:
                    # ⚠ MARK THIS ONE TOO. My first pass marked only the
                    # budget-exhaustion branch below — but THIS is the
                    # truncation that fires first and far more often (a
                    # 300 KB file is cut to 200 KB here before the budget is
                    # anywhere near spent). The producer-side pin caught it:
                    # the consumer's guard would have stayed dark for the
                    # common case.
                    content = content[:per_file_chars] + SNAPSHOT_TRUNCATED_MARK
                remaining = budget_chars - total
                if remaining < len(content):
                    # Budget nearly spent — keep a substantial prefix (NEVER
                    # empty) so the guard still knows the file is non-trivial.
                    #
                    # ⚠ AND SAY THAT IT IS A PREFIX (2026-08-11, §4AT-F). The
                    # truncation was silent, so `coding_executor._apply_file`
                    # took `snap[path]` as the file's FULL prior content and
                    # ran the non-regression guard against it: a 300 KB
                    # index.html stored as 4 KB made a 20 KB full rewrite look
                    # like GROWTH (20000 > 4000×0.85), so the guard waved
                    # through an overwrite that destroyed 280 KB. For .py the
                    # prefix usually fails `ast.parse`, which DISABLES the
                    # overwrite guard outright.
                    content = content[:max(4000, remaining)] + SNAPSHOT_TRUNCATED_MARK
                out[rel] = content
                total += len(content)
                if is_named:
                    first.append(rel)
                if len(out) - len(first) >= max_files and len(first) >= len(chosen):
                    return {**{k: out[k] for k in sorted(first, key=lambda r: r.lower() not in named)}, **out}
    # `_require_dir_fd` fails CLOSED with a ValueError on a platform
    # without dir_fd support, and the nofollow readers raise one for a
    # non-regular file — neither is an OSError, so this function's
    # "never raises" contract needed both (§4GK round 4).
    except (OSError, ValueError):
        pass
    return {**{k: out[k] for k in sorted(first, key=lambda r: r.lower() not in named)}, **out}


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
        for dirpath, files, _dfd in walk_nofollow(base):
            rel_dir = Path(dirpath).relative_to(base).as_posix()
            if "research" not in [p for p in rel_dir.split("/") if p]:
                continue
            for fn in sorted(files):
                if not fn.lower().endswith(".md") or fn.lower() == "index.md":
                    continue
                rel = (Path(dirpath) / fn).relative_to(base).as_posix()
                try:
                    content = read_text_nofollow(fn, dir_fd=_dfd, errors="replace")
                except (OSError, ValueError):
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
    # `_require_dir_fd` fails CLOSED with a ValueError on a platform
    # without dir_fd support, and the nofollow readers raise one for a
    # non-regular file — neither is an OSError, so this function's
    # "never raises" contract needed both (§4GK round 4).
    except (OSError, ValueError):
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
    # Class attribute (normal lookup wins over __getattr__): lets tools tell
    # an autonomous leaf's calls (research briefs already saved) from a live
    # conversation's — the main-loop research write-back keys on it (§4EK).
    is_pinned_project_context = True
    #: §4PG (R2-4): an autonomous tick's file writes stay inside its project
    confine_writes_to_project = True

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


# §4QA: a leaf that CHANGES or PRODUCES project work is not research, whatever
# the classifier said. Live (448d5afa86f0): "Replace lexicon metrics with
# semantic embeddings" and "Update final report with actual quantitative data"
# were web-searched; each closed DONE on a summary of search results while
# tools/recursive_harness.py and report/final-report.md were never touched.
# The verb must OPEN a clause (start, or after ':' / '—' / ';') — "Research
# how teams update their reports" stays research.
_PROJECT_WORK_VERBS = (r"replace|update|rewrite|revise|refactor|modify|extend|"
                       r"integrate|rework|convert|design|draft|implement")
# …and is a VERB: "Update: what changed in 3.13", "Research: update on the EU
# AI act" and "Design patterns in …" are topics, not edits (fresh-eye m7).
_PROJECT_WORK_RE = re.compile(
    r"(?:^|[:;—–]\s*|\.\s+)(?:" + _PROJECT_WORK_VERBS + r")\b"
    r"(?!\s*:|\s+(?:on|of|about|patterns?)\b)", re.IGNORECASE)
def _route_project_work(description, classification: str, curated=None) -> str:
    """The classification after the §4QA override: ``research`` becomes
    ``coding`` (the build path — it writes and verifies files) when the task
    declares a deliverable outside ``research/`` or opens a clause with an
    edit/produce verb. Introspective tasks keep their own path.

    A declared ``research/<name>.md`` stays research — the brief is written
    there — unless the brief writer cannot produce that path (nested, another
    suffix, a name it would slugify, INDEX.md) or ``curated(path)`` says the
    file holds someone's work (``is_generated_brief``: not byte-for-byte a
    brief the writer made). A fresh web summary over a curated literature
    review is the r2 MAJOR-4 harm. Decided on the file, not on verbs — two
    verb lists in a row missed "Integrate … into the review" (r3/r4), and a
    heading check missed notes appended to our own brief (r5)."""
    if classification != "research":
        return classification
    from .project_research import research_brief_slug
    declared = declared_deliverable(description)
    topic = task_topic(description)
    if declared and research_brief_slug(declared) and not (curated and curated(declared)):
        return classification
    if is_self_referential(description):
        # the self-analysis is saved as a brief: a promised file it cannot
        # write (theory/x.md, someone's research/x.md) is built instead —
        # else the close gate FAILED it on every advance (r9 MAJOR-1)
        if declared:
            pretty_log("Autoadvance", f"self-analysis → build: the task declares {declared}"
                       f" — {topic[:60]}", icon=Icons.BRAIN_PLAN)
            return "coding"
        return classification
    if declared or _PROJECT_WORK_RE.search(topic):
        pretty_log("Autoadvance",
                   f"research → build: the task {'declares ' + declared if declared else 'changes project work'}"
                   f" — {topic[:60]}", icon=Icons.BRAIN_PLAN)
        return "coding"
    return classification


# §4QA: a leaf that RUNS something on a language model. Autoadvance has no way
# to call a model as a subject (its build path writes files), so it wrote the
# "responses" itself — 448d5afa86f0's layer results, control condition and
# "cross-model validation" were all prose the agent authored. Only in a
# project whose GOAL names a model ("Refine the model of housing prices" names
# none); a task that runs a code file is code.
#
# INVERTED (r5, 2026-10-18): the goal used to have to say the model is run on
# — three rounds of phrase lists each missed common goals ("Evaluate Qwen on
# GSM8K", "Have Qwen answer 50 questions"), and a miss is the original harm.
# Now any experiment-shaped task in a model project is held; a false positive
# ("Test the chatbot replies" in "Build an LLM chatbot") only waits for the
# owner's turn, where probe_model or a normal build both remain available.
_MODEL_NAMES = (r"llms?|language\s+models?|ai\s+(?:model|system|agent)s?|"
                r"chatbots?|chatgpt|gpt(?:-\w+)?|qwen\w*|llama\w*|claude")
_MODEL_GOAL_RE = re.compile(
    r"\b(?:" + _MODEL_NAMES + r"|introspection\s+loops?)\b", re.IGNORECASE)
_EXPERIMENT_VERB_RE = re.compile(
    r"(?:^|[:;—–,]\s*|\band\s+)(?:re-?)?(?:run|test|probe|measure|replicate|repeat)\b(?![-\w])",
    re.IGNORECASE)
_EXPERIMENT_NOUN_RE = re.compile(
    r"\b(?:control\s+conditions?|falsification\s+tests?|cross-model|ablations?)\b", re.IGNORECASE)
_CODE_FILE_RE = re.compile(r"\b[\w./-]+\.(?:py|js|mjs|ts|sh|ipynb|r|jl)\b", re.IGNORECASE)


def is_model_experiment(description, goal) -> bool:
    """True when the task's data would be a model's answers (see above)."""
    topic = task_topic(description)
    goal = str(goal or "")
    if not _MODEL_GOAL_RE.search(goal) or _CODE_FILE_RE.search(topic):
        return False
    return bool(_EXPERIMENT_VERB_RE.search(topic) or _EXPERIMENT_NOUN_RE.search(topic))


#: AdvanceResult.summary prefix for a leaf left to the owner's turn (§4QA)
MODEL_EXPERIMENT_PREFIX = "model experiment"


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
        # ONE atomic read-modify-write (§4LZ C2): a whole-dict write-back
        # put stale values back over keys merged in between
        def _bump(meta):
            meta["steps_used"] = int(meta.get("steps_used", 0) or 0) + 1
            meta.setdefault("steps_cap", DEFAULT_STEPS_CAP)
            meta["last_autoadvance_ts"] = time.time()
            return meta
        if callable(getattr(store, "_atomic_metadata_update", None)):
            store._atomic_metadata_update(project_id, _bump)
            return
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
            # just the one key — metadata merges (§4LZ C2)
            store.update_project(project_id, metadata={"last_autoadvance_ts": time.time()})
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
        _seen = {k: v for k, v in (last[0].get("payload") or {}).items() if k != "origin"} if last else None
        if last and _seen == {k: v for k, v in (payload or {}).items() if k != "origin"}:
            return                              # §4PF: the store stamps `origin`; compare the substance
    except Exception:
        logger.debug("budget_exhausted dedup check skipped", exc_info=True)
    store.log_event(project_id, None, "budget_exhausted", payload)


def _work_log_step(store, project_id, nxt, *, outcome: str,
                   files=None, tools=None, note: str = "") -> None:
    """Mirror an autoadvance step into the project's work_log journal.

    Until 2026-07-24 only INTERACTIVE turns wrote work_log (the finalize
    chain), so a project mostly built by autoadvance had a near-empty
    journal (live: 2 work_log rows vs 8 autoadvance_step events on
    6a471d630e81) and `file_history` / the RECENT WORK LOG briefing
    couldn't see what the idle loop did. Best-effort — never breaks a step.
    """
    try:
        store.add_work_log(
            project_id,
            request=f"[autoadvance] {(nxt.description or '').strip()}",
            files=list(files or []),
            tools=dict(tools or {}),
            outcome=outcome,
            note=note or "",
        )
    except Exception:
        logger.debug("autoadvance work_log skipped", exc_info=True)


#: The task asks for a check — ANYWHERE in its text (§4LQ, after four review
#: rounds): narrowing the word list flip-flopped — each round closed real
#: check tasks unattended ("Ensure all tests pass", "Verify data
#: integrity", "Final verification") or held build tasks. Since the owner's
#: own run takes a held task back (no jam), holding a build task by mistake
#: costs one look; closing a check unattended is the incident this exists
#: for. So the rule errs toward holding. Whole words only: "checkout",
#: "testimonials", "latest", "contest" are not checks.
_VERIFY_TASK_RE = re.compile(
    r"\b(?:verif(?:y|ies|ied|ying|ication)|(?:re-?)?test(?:s|ing|ed)?|check(?:s|ing|ed)?"
    r"|validat(?:e|es|ed|ing|ion)|confirm(?:s|ed|ing|ation)?|ensure|make\s+sure"
    r"|qa|sanity|smoke[- ]?test\w*|e2e)\b(?!-in\b)", re.I)
#: the result tail of a held task — an owner-run advance re-opens these
_HELD_MARK = "not closed unattended"


#: §4PG (R3-4): search outputs that are not evidence — the tool's own notes
_UNUSABLE_RESEARCH_RE = re.compile(
    r"\A\s*(?:NOTE: low relevance|NOTE: every web search engine failed|ERROR: No search results|"
    r"Error:|ERROR:)", re.IGNORECASE)      # r3: the tool's own head only — a snippet is not a verdict


def _research_output_usable(output) -> bool:
    """A brief written from off-topic or engine-failure results is not
    research that landed (the persister writes one for any non-error
    output); unattended, that task waits for the owner."""
    text = str(output or "")
    if not text.strip():
        return False
    return not _UNUSABLE_RESEARCH_RE.search(text[:600])


def _unattended_close(owner_requested: bool, description: str,
                      evidence_ok: bool = True, unverified_build: bool = False):
    """(status, note) for closing a leaf (§4LP, operator: "go with your
    recommendations"). Unattended (the idle loop) a task that asks to
    verify / test / check is not done because a file was written ("Verify
    full pipeline" closed with "wrote index.html"), and a research task
    whose research did not land is not done on a summary of off-topic
    results. Those wait for the owner (NEEDS_USER). An owner-run advance
    closes DONE as before."""
    from .planning import TaskStatus
    if owner_requested:
        return TaskStatus.DONE, ""
    if _VERIFY_TASK_RE.search(str(description or "")[:400]):
        return TaskStatus.NEEDS_USER, f"needs a real check — {_HELD_MARK}"
    if not evidence_ok:
        return (TaskStatus.NEEDS_USER,
                f"the research did not land — {_HELD_MARK}")
    if unverified_build:
        # §4PG (R2-1): nothing ran on the files — the model's SUMMARY is the
        # only witness; unattended that is not DONE
        return (TaskStatus.NEEDS_USER,
                f"built but no verification command ran — {_HELD_MARK}")
    return TaskStatus.DONE, ""


async def _note_misquotes_if_done(store, plan, project_id, nxt) -> None:
    """§4QA-e: advisory — once the task really closed DONE (r32: not a build
    held NEEDS_USER), a write-up that seems to reword a probe transcript gets
    the owner a ledger line. Never raises."""
    try:
        from .planning import TaskStatus
        _node = plan.tree.nodes.get(nxt.id)
        if _node is None or _node.status != TaskStatus.DONE:
            return
        from .task_deliverable import note_misquotes
        await note_misquotes(store, project_id, nxt.id, nxt.description)
    except Exception:  # noqa: BLE001
        logger.warning("quote check note skipped for %s", nxt.id, exc_info=True)


#: §4SD: a coding step whose leaf ran out of time (task back to READY)
OUT_OF_TIME_PREFIX = "code build ran out of time (task back to READY)"


def _finalize_coding(context, store, plan, project_id, nxt, cres,
                     tick_started_at, owner_requested: bool = False) -> AdvanceResult:
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
        # §4QA: the build passed, but the task promised a file (→ path) — a
        # build that left it missing or hollow did not do the task
        from .task_deliverable import deliverable_problem
        try:
            _undelivered = deliverable_problem(store.ensure_workspace(project_id), nxt.description)
        except Exception:  # noqa: BLE001 — the build's own verdict stands
            _undelivered = None
        if _undelivered:
            # the files it DID write stay the project's (the end-of-project
            # cleanup deletes unregistered ones)
            for rel in (cres.files or []):
                try:
                    store.register_file_artifact(nxt.id, rel, description=_short_summary(cres.summary or "")[:180])
                except Exception:  # noqa: BLE001
                    logger.debug("artifact register skipped", exc_info=True)
            import dataclasses as _dc
            cres = _dc.replace(cres, ok=False,
                               summary=f"{_undelivered} (the build reported: {_short_summary(cres.summary or '')})")

    if cres.ok:
        # Per-file manifest seed (2026-07-24): the build's summary (or the
        # task description) is the best available "what this file does" at
        # creation time — with file-per-task granularity it describes the
        # file. The model/dream can refine it later via describe_file.
        _fdesc = " ".join(str(cres.summary or nxt.description or "").split())[:180]
        for rel in (cres.files or []):
            try:
                store.register_file_artifact(nxt.id, rel, description=_fdesc)
            except Exception:
                logger.debug("artifact register skipped: %s", rel, exc_info=True)
        if cres.ledger_note:
            try:
                store.append_ledger(project_id, cres.ledger_note)
            except Exception:
                logger.debug("ledger append skipped", exc_info=True)
        _st, _why = _unattended_close(owner_requested, nxt.description,
                                      unverified_build=not bool(getattr(cres, "verified", False)))
        plan.update_status(nxt.id, _st,
                           result=(f"{cres.summary} — {_why}" if _why else cres.summary),
                           actual_tool="code_executor")
        _increment_budget(store, project_id)
        _tick_secs = max(0.0, time.time() - tick_started_at)
        record_runtime(store, project_id, seconds=_tick_secs, tool_calls=1)
        # Stamp the task's real wall-clock cost. The column was dead
        # (never written) until 2026-07-18; the retrospective now sums it.
        try:
            store.update_task(nxt.id, actual_cost=_tick_secs)
        except Exception:
            logger.debug("actual_cost stamp skipped", exc_info=True)
        # §4PG (R1-3): update_status(DONE) demotes the leaf to FAILED on an
        # unsatisfied postcondition — the record said "completed" regardless
        _final = _st
        try:
            _node = plan.tree.nodes.get(nxt.id)
            if _node is not None:
                _final = _node.status
        except Exception:  # noqa: BLE001
            pass
        if _final == TaskStatus.FAILED and _st != TaskStatus.FAILED:
            store.log_event(project_id, nxt.id, "autoadvance_step_failed",
                            {"tool": "code_executor", "classification": "coding", "postcondition_failed": True,
                             "files": list(cres.files or [])[:8], "owner_requested": bool(owner_requested)})
            _work_log_step(store, project_id, nxt, outcome="had_failures",
                           files=list(cres.files or []), tools={"code_executor": 1},
                           note=f"build landed but a postcondition FAILED: {str(cres.summary or '')[:200]}")
            if not owner_requested:
                _record_failed_activity(context, project_id, nxt.description, "postcondition failed")
            return AdvanceResult(True, nxt.id, "coding",
                                 f"code build failed a postcondition: {cres.summary}", None)
        if _why:
            store.log_event(project_id, nxt.id, "autoadvance_needs_user",
                            {"description": nxt.description, "reason": _why,
                             "files": list(cres.files or [])[:8]})
            _record_needs_user_activity(context, project_id, nxt.description,
                                        "autoadvance_needs_user")
        else:
            store.log_event(project_id, nxt.id, "autoadvance_step",
                            {"tool": "code_executor", "classification": "coding",
                             "files": list(cres.files or [])[:8],
                             "owner_requested": bool(owner_requested)})
        _work_log_step(store, project_id, nxt,
                       outcome="needs_user" if _why else "completed",
                       files=list(cres.files or []),
                       tools={"code_executor": 1},
                       note=str(cres.summary or "")[:280])
        return AdvanceResult(True, nxt.id, "coding",
                             (f"built: {cres.summary}; held for the owner: {_why}"
                              if _why else f"built: {cres.summary}"), None)

    if getattr(cres, "out_of_time", False):
        # §4SD review r11: the leaf's TIME ran out (an unattended step's cap,
        # a 30-min Slack request), not the build — as a timed-out step
        # (`_after_abort`) the task goes back to READY, with the files that
        # completed on disk for the next step to build on; FAILED is never
        # retried unattended
        try:
            store.update_task(nxt.id, status="READY", failure_reason=f"code_executor: {cres.summary}"[:300])
        except Exception:  # noqa: BLE001
            logger.debug("out-of-time leaf not reset", exc_info=True)
        _increment_budget(store, project_id)
        _tick_secs = max(0.0, time.time() - tick_started_at)
        record_runtime(store, project_id, seconds=_tick_secs, tool_calls=1)
        try:
            store.update_task(nxt.id, actual_cost=_tick_secs)
        except Exception:  # noqa: BLE001
            logger.debug("actual_cost stamp skipped", exc_info=True)
        store.log_event(project_id, nxt.id, "autoadvance_out_of_time",
                        {"tool": "code_executor", "reason": (cres.summary or "")[:200],
                         "files": list(cres.files or [])[:8]})
        _work_log_step(store, project_id, nxt, outcome="had_failures",
                       files=list(cres.files or []), tools={"code_executor": 1},
                       note=f"out of time, back to READY: {(cres.summary or '')[:240]}")
        # `advance_many` stops on this prefix (review r12: the READY task was
        # picked again at once — 40 times in one owner batch)
        return AdvanceResult(True, nxt.id, "coding", f"{OUT_OF_TIME_PREFIX}: {cres.summary}", None)
    plan.update_status(nxt.id, TaskStatus.FAILED,
                       failure_reason=f"code_executor: {cres.summary}")
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
    _work_log_step(store, project_id, nxt, outcome="had_failures",
                   files=list(cres.files or []),
                   tools={"code_executor": 1},
                   note=f"build FAILED: {(cres.summary or '')[:240]}")
    if not owner_requested:
        _record_failed_activity(context, project_id, nxt.description, (cres.summary or "")[:160])   # §4PG (R1-8)
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
    r"self-consciousness|self-critique|self-introspect\w*)\b"
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


# Bare "introspection" names the agent only when no HUMAN subject is in the
# task (§4QA): "Human-subject replication protocol: design introspection
# journal format for 3-5 humans" is a design task about people, and the
# self-analysis path answered it from the agent's own architecture.
_INTROSPECT_RE = re.compile(r"\bintrospect\w*", re.IGNORECASE)
_HUMAN_SUBJECT_RE = re.compile(
    r"\b(?:humans?|human-subjects?|people|persons?|participants?|volunteers?|"
    r"subjects|respondents?|interviewees?)\b", re.IGNORECASE)


def is_self_referential(description) -> bool:
    """True when the task asks the agent to analyse ITSELF (see _SELF_REF_RE)."""
    d = str(description or "")
    if _SELF_REF_RE.search(d):
        return True
    return bool(_INTROSPECT_RE.search(d)) and not _HUMAN_SUBJECT_RE.search(d)


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
        # §4QA: background ONLY when no user request is live — inside the
        # owner's own autoadvance a background call parks behind its OWN
        # request for 600 s, then raises (live 13:04→13:14, req 571a5020)
        }, is_background=_idle_invocation(context), task_label="project-self-analysis",
            side_main=True)     # §4SD: a foreground call saves the owner's conversation first
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


def _record_failed_activity(context, project_id, description, why) -> None:
    """§4PG (R1-8/R3-3): an unattended build that FAILED reaches the owner
    through the activity ledger too — the digest's window was its only
    channel and a busy night pushed it out."""
    try:
        from .autonomous_activity import get_activity_log, SEVERITY_NOTIFY
        log = get_activity_log(context)
        if log is not None:
            log.record("project", f"autopilot build FAILED: {str(description)[:120]} — {str(why)[:160]}",
                       severity=SEVERITY_NOTIFY, kind="autoadvance_failed", project_id=str(project_id))
    except Exception as e:  # noqa: BLE001
        logger.debug("failed-build activity record skipped: %s", e)


def _record_needs_user_activity(context, project_id, description, kind) -> None:
    """Push a needs-user/human-gate outcome into the autonomous-activity
    ledger (severity=notify → immediate outbound push when configured).
    The next-turn DIGEST already renders these via core.project_digest —
    the activity-digest renderer therefore EXCLUDES the "project" phase
    (DIGEST_EXCLUDED_PHASES); this record exists purely so a blocked
    project can reach the operator without them opening a chat.
    Fail-safe: never raises."""
    try:
        _key = (str(project_id), str(description)[:160])
        if _key in _NOTIFIED_HOLDS:
            return                             # §4PG (R1-4): one push per held task, not one per tick
        from .autonomous_activity import get_activity_log, SEVERITY_NOTIFY
        log = get_activity_log(context)
        if log is not None:
            log.record(
                "project",
                f"project task needs your input: {str(description)[:160]}",
                severity=SEVERITY_NOTIFY,
                kind=str(kind), project_id=str(project_id),
            )
            _NOTIFIED_HOLDS.add(_key)          # r3: only once it is on the record
            if len(_NOTIFIED_HOLDS) > 512:
                _NOTIFIED_HOLDS.clear()
    except Exception as e:  # noqa: BLE001
        logger.debug("needs-user activity record skipped: %s", e)


#: (project, task description) holds already pushed this process
_NOTIFIED_HOLDS: set = set()


async def advance_once(context, project_id: str, *args, owner_requested: bool = False, **kwargs) -> "AdvanceResult":
    """§4PF (W5): every ledger row an advance tick writes says WHO did it —
    "autopilot" for an unattended tick, "owner" for the owner's own
    `autoadvance` — so autopilot in production never reads as the owner's work."""
    from ..utils.logging import event_origin_context as _eoc
    _tok = _eoc.set(_tick_origin(owner_requested))
    # §4SD open items r2/r3 (review MAJORs): a stop after the claim (Slack's
    # hard limit, the owner's Stop) cancels the tool — every await of the body
    # caught `Exception` only, and the leaf stayed IN_PROGRESS, unclaimable
    # until the next boot. A caller that tracks the claim itself (`claim_sink`:
    # the unattended step, whose `_after_abort` charges and reopens) keeps it.
    _own = kwargs.get("claim_sink") is None
    if _own:
        kwargs["claim_sink"] = []
    try:
        return await _advance_once_body(context, project_id, *args, owner_requested=owner_requested, **kwargs)
    except asyncio.CancelledError:
        if _own:
            _reopen_stopped(getattr(context, "project_store", None), project_id, kwargs["claim_sink"])
        raise
    finally:
        _eoc.reset(_tok)


def _reopen_stopped(store, project_id, claimed) -> None:
    """The leaf a stopped tick claimed goes back to READY — only while it is
    still IN_PROGRESS (a leaf the tick had closed, or the owner changed, stays)."""
    for tid in claimed or []:
        if store is not None and _task_status(store, project_id, tid) == "IN_PROGRESS":
            try:
                store.update_task(tid, status="READY", failure_reason="stopped mid-step")
            except Exception:  # noqa: BLE001
                logger.debug("stopped leaf not reset", exc_info=True)


def _tick_origin(owner_requested: bool) -> str:
    """§4PF r4: a probe's advance stays "probe" (the pin had overridden it and
    the digest showed a probe's step as the agent's own work)."""
    try:
        from ..utils.logging import request_kind
        if request_kind() == "probe":
            return "probe"
    except Exception:  # noqa: BLE001
        pass
    return "owner" if owner_requested else "autopilot"


async def _advance_once_body(
    context,
    project_id: str,
    tool_runner: Optional[ToolRunner] = None,
    llm_classifier: Optional[LLMClassifier] = None,
    code_generator: Optional[Callable[[str], Awaitable[str]]] = None,
    coding_executor: Optional[Callable[..., Awaitable[Any]]] = None,
    owner_requested: bool = False,
    claim_sink: Optional[list] = None,
) -> AdvanceResult:
    """Run a single autoadvance tick for ``project_id``.

    ``claim_sink`` (§4MQ): when given, the id of the leaf this tick claims is
    appended to it — the unattended gate resets only ITS leaf on a timeout.

    ``owner_requested=False`` (the idle loop): a task the VERIFIER filed on its
    own ("Verifier follow-up: …") is never built — the owner did not ask for
    it. §4LP: the background advancer edited two shipped apps to "resolve"
    such complaints (a made-up retry button; the real bug untouched) and
    marked them DONE. They stay on the books for the owner's own turns.

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
    # The owner's run may take back what an unattended run held for them
    # (§4LQ review M4: it skipped them, and the hold had rolled the project
    # to NEEDS_USER, so the owner's advance was refused). Nothing is reopened
    # here: the held tasks join the normal ready-leaf pick in the claim below
    # (dependencies, leaf-only, no paused parent — review round 3 F1), and
    # only the one claimed leaves NEEDS_USER (round 2 N7, round 3 F3).
    _owner_may_take_back = (
        owner_requested and proj["status"] == "NEEDS_USER"
        and any(str(_t.get("status") or "").upper() == "NEEDS_USER"
                and str(_t.get("result_summary") or "").endswith(_HELD_MARK)
                for _t in (store.list_tasks(project_id) or [])))
    if proj["status"] != "ACTIVE" and not _owner_may_take_back:
        return AdvanceResult(True, None, "blocked",
                             f"project is {proj['status']}, not ACTIVE")

    # Inter-project dependency gate (2026-07-25): a project that depends on
    # others (metadata.depends_on_projects, set via action=set_dependency)
    # doesn't autoadvance until every dependency is DONE or RELEASED — the
    # advancer's round-robin can now sequence project chains.
    _deps = ((proj.get("metadata") or {}).get("depends_on_projects") or [])
    for _dep in _deps:
        _dp = store.get_project(_dep)
        _dep_status = str((_dp or {}).get("status", "")).upper()
        if _dep_status in ("DONE", "RELEASED"):
            continue
        # A dependency that can NEVER become DONE must not block forever
        # (the inter-project mirror of the intra-project rule that treats an
        # unknown dep id as satisfied "so a bad reference can't deadlock the
        # whole plan", planning.deps_satisfied). Missing (deleted), ARCHIVED
        # and FAILED are terminal-unsatisfiable — clear the block and warn
        # loudly rather than freezing the dependent's autoadvance silently.
        if _dp is None or _dep_status in ("ARCHIVED", "FAILED"):
            logger.warning(
                "project %s depends on %s which is %s — treating the stale "
                "dependency as cleared so autoadvance is not deadlocked; "
                "remove it via set_dependency if intentional.",
                project_id, _dep, _dep_status or "missing")
            continue
        # Recoverable states (PAUSED / NEEDS_USER / BLOCKED / ACTIVE /
        # PENDING) legitimately still block — the dependency may complete.
        # §4PG (R1-2): stamp the pick so the round-robin rotates, and say so
        # ONCE — the blocked project was re-picked every tick, silently,
        # starving the other autopilot projects
        _stamp_autoadvanced(store, project_id)
        try:
            _last = store.list_events(project_id, limit=1, event_type="autopilot_blocked")
            if not _last or (_last[0].get("payload") or {}).get("dependency") != _dep:
                store.log_event(project_id, None, "autopilot_blocked",
                                {"dependency": _dep, "status": _dep_status or "missing"})
        except Exception:  # noqa: BLE001
            pass
        return AdvanceResult(
            True, None, "blocked",
            f"waiting on dependency project "
            f"'{(_dp or {}).get('title') or _dep}' ({_dep_status or 'missing'})")

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
        # in memory only: a held task counts as PENDING for this one pick
        _held = ([n for n in plan.tree.nodes.values()
                  if n.status == TaskStatus.NEEDS_USER
                  and str(n.result_summary or "").endswith(_HELD_MARK)]
                 if owner_requested else [])
        for _n in _held:
            _n.status = TaskStatus.PENDING
        nxt = plan.next_ready_leaf(
            skip=None if owner_requested else _is_unrequested_task)
        for _n in _held:
            if _n is not nxt:
                _n.status = TaskStatus.NEEDS_USER
        _nxt_held_summary = str(nxt.result_summary or "") if nxt is not None and nxt in _held else ""
        if nxt:
            plan.update_status(nxt.id, TaskStatus.IN_PROGRESS)
            if claim_sink is not None:
                claim_sink.append(nxt.id)
    if not nxt:
        # Stamped OUTSIDE the claim lock (it re-acquires the project lock):
        # an idle project must still rotate to the back of the round-robin
        # queue or it starves the other ACTIVE projects (see _stamp_…).
        _stamp_autoadvanced(store, project_id)
        return AdvanceResult(True, None, "idle", "no READY/PENDING leaf")

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
    from .project_research import is_generated_brief
    classification = _route_project_work(
        nxt.description, classification,
        curated=lambda rel: not is_generated_brief(store, project_id, rel))
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
        classification = _route_project_work(          # r9: a promised file still routes
            nxt.description, "research",
            curated=lambda rel: not is_generated_brief(store, project_id, rel))

    # (after the override above — r10: a needs_user introspective experiment
    # skipped the hold and the self-analysis wrote its "responses")
    # (no self-referential exemption — r9: "Run Layer 3 … self-awareness"
    # took the self-analysis path, which wrote the "responses" itself)
    if (_proj_kind != "CODING" and classification in ("research", "coding")
            and is_model_experiment(nxt.description, proj.get("goal"))):
        # §4QA: run in the turn with probe_model, never written up unrun
        _stamp_autoadvanced(store, project_id)
        if not owner_requested:
            # unattended nobody can run it: HOLD it for the owner (skipped by
            # later ticks, reported once) — released to PENDING it was picked
            # again every tick and the autopilot paused on "no progress"
            _why = (f"{MODEL_EXPERIMENT_PREFIX}? — its data may be a model's answers: in your turn, "
                    f"probe_model if so, the normal way if not — {_HELD_MARK}")
            plan.update_status(nxt.id, TaskStatus.NEEDS_USER, result=_why)
            store.log_event(project_id, nxt.id, "autoadvance_needs_user",
                            {"description": nxt.description, "reason": _why})
            _record_needs_user_activity(context, project_id, nxt.description,
                                        "autoadvance_needs_user")
            return AdvanceResult(True, nxt.id, "needs_user", _why)
        if nxt in _held:
            # the owner's turn runs it — but until it does, the hold stands
            plan.update_status(nxt.id, TaskStatus.NEEDS_USER, result=_nxt_held_summary)
        else:
            plan.update_status(nxt.id, TaskStatus.PENDING)
        store.log_event(project_id, nxt.id, "autoadvance_skipped",
                        {"reason": "model experiment", "classification": classification})
        return AdvanceResult(True, nxt.id, "blocked",
                             f"{MODEL_EXPERIMENT_PREFIX}: {task_topic(nxt.description)[:160]}")

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
        # ⚠ USE THE SHARED NORMALISER (2026-08-11, §4AT-C). Metadata is
        # model-written JSON, so `{"constraints": "no external APIs"}` is a
        # legal shape — and iterating it raw yields 17 SINGLE-CHARACTER
        # "constraints" fed straight to the constraint gate. `_constraint_list`
        # exists precisely because this shredding already destroyed a
        # constraint record once (2026-08-01); this call site kept its own copy
        # and so kept the bug. Three other sites in tools/projects.py use the
        # helper correctly.
        from ..memory.projects import _constraint_list
        _constraints = _constraint_list(
            (proj.get("metadata") or {}).get("constraints"))
        cres = None
        try:
            cres = await coding_executor(
                context, nxt.description,
                # Fail-CLOSED shell gates: the executor's verify/smoke
                # classification would otherwise read a success-shaped
                # non-execution (grep no-match, egress-guard prose, missing
                # exit code) as a pass and mark the task DONE on nothing.
                tool_runner=_verify_fail_closed_runner(tool_runner),
                # §4FG: the executor seam and the agentic leaf need the
                # project EXPLICITLY — this context is not pinned (the HTTP
                # route and idle ticks carry no conversation binding), so
                # reading `context.current_project_id` selected the spec
                # executor for an "agentic" project. Measured: the first leaf
                # pilot ran spec vs spec.
                project_id=project_id,
                ledger=ledger,
                existing_files=_gather_project_files(store, project_id,
                                                     description=nxt.description),
                research_context=_gather_research_briefs(store, project_id),
                single_file=_single_file,
                constraints=_constraints)
        except Exception as e:
            # ⚠ A CRASH MUST NOT DOWNGRADE TO A WEAKER CLOSER (2026-08-11,
            # §4AT-C). This logged and fell through to the generic
            # single-shell-command path below, which marks the leaf DONE on
            # ANY output that is not `ERROR:`-prefixed and not a non-zero
            # exit — with the verify gate, the smoke gate, the files-written
            # check and the constraint gate ALL bypassed. `echo`-shaped
            # output closed a build task.
            #
            # Same family as the two coding-executor CRITICALs fixed the same
            # day (dropped edits, truncated-snapshot overwrite): a task
            # reaching DONE on evidence that no work happened. A transient
            # exception is a reason to STOP, not a reason to accept a lower
            # standard of proof. The leaf stays open and the next tick retries
            # it with the full gate set.
            logger.warning("coding_executor crashed: %s", e)
            # §4MR: "left open" must mean claimable — the leaf stayed
            # IN_PROGRESS, so no tick could retry it and the progress gate
            # saw nothing until the next boot's reaper
            from .llm import BackgroundDeferred as _BgDeferred
            if isinstance(e, _BgDeferred):
                # §4PG r3 (R2-3): the owner is back — the leaf is reopened
                # and the tick is neither a failure nor "no progress"
                try:
                    store.update_task(nxt.id, status="READY", failure_reason="deferred for the owner's request")
                except Exception:  # noqa: BLE001
                    logger.debug("deferred leaf not reset", exc_info=True)
                return AdvanceResult(True, nxt.id, "deferred",
                                     f"deferred: the owner's request took the slot — '{nxt.id}' reopened")
            try:
                store.update_task(nxt.id, status="READY",
                                  failure_reason=f"coding executor crashed: {type(e).__name__}")
            except Exception:  # noqa: BLE001
                logger.debug("crashed leaf not reset", exc_info=True)
            return AdvanceResult(
                True, nxt.id, "blocked",
                f"coding executor crashed on '{nxt.id}' "
                f"({type(e).__name__}: {e}) — leaf left open rather than "
                f"closed by the weaker fallback path")
        if cres is not None:
            _res = _finalize_coding(context, store, plan, project_id, nxt,
                                    cres, _tick_started_at,
                                    owner_requested=owner_requested)
            await _note_misquotes_if_done(store, plan, project_id, nxt)
            return _res

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
        # §4QA: the topic, not "Task 1.3: … → research/x.md" — the numbering
        # and the path went into the query and every result was off-topic
        tool_args = {"query": task_topic(nxt.description)[:200]}

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
    from .project_safety import enforce_human_gate, detect_contradiction

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
    # §4NN: the event is the record; the belief-versioning ledger copy
    # (route_contradiction → ContradictionLog) was removed with that store.
    for peer in peers:
        conflict = detect_contradiction(result_summary,
                                        peer.get("result_summary", ""))
        if conflict:
            store.log_event(project_id, nxt.id, "contradiction_detected",
                            {"peer_task_id": peer["id"], "conflict": conflict})
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
            from .project_research import persist_research_from_output, research_brief_slug
            rr = await persist_research_from_output(
                context, project_id, task_topic(nxt.description), output, task_id=nxt.id,
                # §4QA: a declared research/<name>.md is the brief's file name
                slug=research_brief_slug(declared_deliverable(nxt.description)))
            if rr.ok:
                research_path = rr.path
                # Prefer the synthesised summary as the task's result.
                if rr.summary:
                    result_summary = _short_summary(rr.summary)
        except Exception:
            logger.debug("auto-research persist skipped", exc_info=True)

    _fail_reason = ""
    _st, _why = _unattended_close(
        owner_requested, nxt.description,
        evidence_ok=not (classification == "research" and tool_name == "web_search"
                         and (not research_path or not _research_output_usable(output))))   # §4PG (R3-4)
    if _st == TaskStatus.DONE:
        # §4QA (fresh-eye M3): this path closed DONE with the task's declared
        # file missing — a brief under another name, or a theory/x.md never
        # written. Unattended it is held for the owner; owner-run it FAILS
        # with the reason (r2 MAJOR-3: a hold is re-picked first by every
        # owner advance — the same search, the same miss, forever)
        try:
            from .task_deliverable import deliverable_problem
            _undelivered = deliverable_problem(store.ensure_workspace(project_id), nxt.description)
        except Exception:  # noqa: BLE001 — the close stands
            _undelivered = None
        if _undelivered and owner_requested:
            _st, _fail_reason = TaskStatus.FAILED, _undelivered
        elif _undelivered:
            _st, _why = TaskStatus.NEEDS_USER, f"{_undelivered} — {_HELD_MARK}"
    plan.update_status(nxt.id, _st,
                       result=(f"{result_summary} — {_why}" if _why else result_summary),
                       failure_reason=_fail_reason, actual_tool=tool_name)
    # update_status(DONE) silently DEMOTES the leaf to FAILED when a
    # postcondition isn't satisfied (planning.py). Read the resulting
    # status so the event log / work_log / dream digest record what
    # actually happened — logging "completed" on a postcondition-failed
    # task corrupted the operator's "what did I do last night" ledger.
    _final_status = TaskStatus.DONE
    try:
        _node_after = plan.tree.nodes.get(nxt.id)
        if _node_after is not None:
            _final_status = _node_after.status
    except Exception:
        pass
    _step_ok = _final_status == TaskStatus.DONE
    # held for the owner by _unattended_close — not a postcondition failure
    _held = bool(_why) and _final_status == TaskStatus.NEEDS_USER
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
    if _held:
        step_payload["description"] = nxt.description
        step_payload["reason"] = _why
    elif not _step_ok:
        step_payload["postcondition_failed"] = True
    # §4LP: the "While you were away … on my own" digest counts only steps
    # nobody asked for — an owner-run advance is the owner's own work
    step_payload["owner_requested"] = bool(owner_requested)
    store.log_event(project_id, nxt.id,
                    "autoadvance_needs_user" if _held
                    else "autoadvance_step" if _step_ok
                    else "autoadvance_step_failed",
                    step_payload)
    if _held:
        _record_needs_user_activity(context, project_id, nxt.description,
                                    "autoadvance_needs_user")
    _work_log_step(store, project_id, nxt,
                   outcome=("needs_user" if _held
                            else "completed" if _step_ok else "had_failures"),
                   files=([research_path] if research_path else []),
                   tools=({tool_name: 1} if tool_name else {}),
                   note=str(result_summary or "")[:280])
    summary = (
        (f"advanced via {tool_name}" if _step_ok
         else f"task ran via {tool_name}; held for the owner: {_why}" if _held
         else f"task ran via {tool_name} but FAILED: {_fail_reason}" if _fail_reason
         else f"task ran via {tool_name} but a postcondition FAILED")
        + (f"; saved research to {research_path}" if research_path else ""))
    # after every write of the step (r33: a cancel here skipped the budget,
    # runtime, event and work log of a task that really closed)
    await _note_misquotes_if_done(store, plan, project_id, nxt)
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
    r"remaining\s+(?:tasks?|work)|finish\s+(the\s+)?project|complete\s+(the\s+)?project|"
    r"whole\s+(project|thing)|to\s+the\s+end|until\s+(it'?s\s+|you'?re\s+)?done|"
    r"keep\s+going\s+until)\b",
    re.I,
)
# §4QA: "proceed the project on your own" is a go-ahead for the REST (live
# 25a54a0a: read as one task; the owner had to say "proceed all tasks"). A
# continue-verb must lead, so "can you write task 3 on your own?" is no batch.
# Fresh-eye M5: the object may only be the project / the rest / nothing —
# "run the tests by yourself", "work on this on your own" and "proceed with
# task 3 on your own" are one piece of work.
_INTENT_AUTONOMOUS = re.compile(
    r"\b(?:proceed|continue|carry\s+on|go\s+on|keep\s+going|advance)\b"
    r"(?:\s+(?:with|on))?(?:\s+(?:the|this|my|our))?"
    r"(?:\s+(?:project|plan|tasks|work|rest|remaining\s+(?:tasks|work)|it))?\s*,?\s*"
    r"(?:on\s+your\s+own|by\s+yourself|autonomously|"
    r"without\s+(?:asking|stopping|checking\s+(?:in|with\s+me)))\b",
    re.I,
)
_INTENT_ONE_TASK = re.compile(r"\btasks?\s*#?\s*\d", re.I)
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
    if _INTENT_AUTONOMOUS.search(t) and not _INTENT_ONE_TASK.search(t):
        return {"mode": "all", "count": None}
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
            }, side_main=True, task_label="project-classify")     # §4SD: the owner's conversation saved first
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
        }, side_main=True, task_label="project-command")         # §4SD: the owner's conversation saved first
        out = ((r or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "").strip()
        if out.startswith("```"):
            out = out.strip("`")
            if "\n" in out:
                out = out.split("\n", 1)[1]
        return out.strip()

    return _gen


def idle_candidates(store) -> list:
    """The projects the IDLE loop may advance (§4LP, operator: "go with your
    recommendations"): ACTIVE ones the OWNER put on autopilot — by running
    `manage_projects action=autoadvance` in their own turn, or
    `action=autopilot enabled=true`. A single web "advance" step does not
    opt in. Every ACTIVE project used to be, and that is how a released
    app's unasked fork got built."""
    return [p for p in (store.list_projects("ACTIVE") or [])
            if (p.get("metadata") or {}).get("autopilot")]


# ------------------------------------------------------------------ §4MQ
# Multi-day work = many short, bounded UNATTENDED steps on an autopilot
# project. Each step must move the plan forward, or autopilot pauses and the
# owner is told once; it never runs past its own wall cap; a step lost to a
# crash is charged; and every `checkpoint_every` steps that moved forward the
# owner is asked to look before more run. Every pause turns autopilot OFF —
# only the owner's own turn turns it back on (§4LQ).

#: consecutive unattended steps that moved nothing forward → autopilot pauses
NO_PROGRESS_PAUSE_AT = 3
#: unattended steps that moved forward between owner check-ins (owner may change)
DEFAULT_CHECKPOINT_EVERY = 10
#: one unattended step's wall cap (a build that verifies can run long; a
#: sandbox job is promoted and survives its turn on its own)
UNATTENDED_STEP_TIMEOUT_S = 45 * 60
#: unattended runtime a project may spend when the owner set no cap
DEFAULT_UNATTENDED_RUNTIME_CAP_S = 6 * 3600
#: a batch's FIRST step has no measured duration yet — this is its estimate
#: (§4MR: the first step was never checked; a build takes minutes)
FIRST_STEP_ESTIMATE_S = 120.0

#: metadata keys only the system or the owner's `action=budget` may write
BUDGET_METADATA_KEYS = (
    "steps_cap", "steps_used", "runtime_cap_seconds", "runtime_used_seconds",
    "tool_call_cap", "tool_call_used", "checkpoint_every", "no_progress_streak",
    "steps_since_checkpoint", "step_in_flight", "autopilot_paused", "last_autoadvance_ts",
    "unattended_runtime_seconds", "unattended_runtime_cap_seconds")

#: this process — a `step_in_flight` stamped by another boot is a step that
#: died with its process; one stamped by this boot is a step still running
_BOOT_ID = f"{os.getpid()}-{int(time.time())}"


def _task_status(store, project_id, task_id) -> str:
    try:
        for t in store.list_tasks(project_id) or []:
            if str(t.get("id")) == str(task_id):
                return str(t.get("status") or "").upper()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _step_score(store, project_id, task_id) -> str:
    """"moved" when the task the step worked on is now DONE; "held" when it
    waits on the owner (NEEDS_USER — the hold already told them and stops
    the project; not progress, not a failure); else "none". A written file
    or a summary is not progress ("wrote index.html" was booked as a step
    for months, §4MQ lens)."""
    if not task_id:
        return "none"
    st = _task_status(store, project_id, task_id)
    return "moved" if st == "DONE" else ("held" if st == "NEEDS_USER" else "none")


def pause_autopilot(context, project_id, reason: str, detail: str = "") -> None:
    """Turn autopilot OFF, record why, and tell the owner — ONCE: only the
    on → off transition notifies (r2 C1: every later run re-notified)."""
    store = getattr(context, "project_store", None)
    if store is None:
        return
    rec = {"reason": str(reason), "detail": str(detail)[:300], "ts": time.time()}
    was_on = {"v": False}

    def _off(m):
        was_on["v"] = bool(m.get("autopilot"))
        if was_on["v"]:
            m["autopilot"] = False
            m["autopilot_paused"] = rec
        return m
    try:
        _bump_meta(store, project_id, _off)
    except Exception:  # noqa: BLE001
        logger.warning("autopilot pause not recorded for %s", project_id, exc_info=True)
        return
    if not was_on["v"]:
        return
    try:
        store.log_event(project_id, None, "autopilot_paused", rec)
    except Exception:  # noqa: BLE001
        pass
    try:
        title = ((store.get_project(project_id) or {}).get("title")) or project_id
        from .autonomous_activity import get_activity_log, SEVERITY_NOTIFY
        log = get_activity_log(context)
        if log is not None:
            log.record("project",
                       f"autopilot paused on '{str(title)[:60]}' — {reason}"
                       + (f": {str(detail)[:160]}" if detail else "")
                       + ". Say 'resume autopilot on it' to continue.",
                       severity=SEVERITY_NOTIFY, kind="autopilot_paused",
                       project_id=str(project_id))
    except Exception as e:  # noqa: BLE001
        logger.debug("autopilot pause notify skipped: %s", e)
    pretty_log("Autopilot", f"paused {str(project_id)[:8]} — {reason}", icon=Icons.STOP, level="WARNING")


def resume_autopilot_metadata() -> Dict[str, Any]:
    """The metadata an OWNER's resume writes: autopilot on, gates reset — the
    unattended runtime too: each look grants a fresh allowance (r2 M3)."""
    # NOT `step_in_flight`: a step of this process may be running; clearing
    # its marker let a second step start beside it (§4MR M2). A marker left
    # by a dead process is charged as a lost step on the next tick, as it
    # should be.
    return {"autopilot": True, "autopilot_paused": None, "no_progress_streak": 0,
            "steps_since_checkpoint": 0, "unattended_runtime_seconds": 0}


def _bump_meta(store, project_id, mutate) -> Dict[str, Any]:
    if callable(getattr(store, "_atomic_metadata_update", None)):
        return store._atomic_metadata_update(project_id, mutate) or {}
    meta = dict(((store.get_project(project_id) or {}).get("metadata")) or {})
    meta = mutate(meta) or meta
    store.update_project(project_id, metadata=meta)
    return meta


def _unattended_cap_s(meta) -> float:
    cap = meta.get("unattended_runtime_cap_seconds")
    return DEFAULT_UNATTENDED_RUNTIME_CAP_S if cap is None else float(cap)


async def advance_unattended(context, project_id: str, **kwargs):
    """§4PF r4: the whole unattended run — its pauses and no-progress events
    included, not only the tick inside — is the autopilot's."""
    from ..utils.logging import event_origin_context as _eoc
    _tok = _eoc.set(_tick_origin(False))
    try:
        return await _advance_unattended_body(context, project_id, **kwargs)
    finally:
        _eoc.reset(_tok)


async def _with_leaf_deadline(coro, deadline: float):
    """§4SD: run ``coro`` with `coding_executor.leaf_deadline_context` set, so a
    coding leaf inside it sizes each file call by the step's time left (and
    stops as out of time rather than starting one the step's cap would cut)."""
    from .coding_executor import caller_waits_context, leaf_deadline_context
    tok = leaf_deadline_context.set(deadline)
    tok_w = caller_waits_context.set(False)      # §4SD r5: a stopped step does not wait for its build
    try:
        return await coro
    finally:
        caller_waits_context.reset(tok_w)
        leaf_deadline_context.reset(tok)


async def _advance_unattended_body(context, project_id: str, *, step_timeout_s: Optional[float] = None,
                             **kw) -> AdvanceResult:
    """ONE unattended step under the §4MQ gates — the only entry for the idle
    loop and for every non-owner batch: autopilot must be ON; one step per
    project at a time; the lost-step charge; the unattended runtime cap; the
    wall cap; the progress gate; the check-in cadence. ``kw`` goes to
    `advance_once`."""
    from ..utils.aio import wait_for as _wait_for
    store = getattr(context, "project_store", None)
    if store is None:
        return AdvanceResult(False, None, "idle", "project_store missing on context")
    meta = dict(((store.get_project(project_id) or {}).get("metadata")) or {})
    if not meta.get("autopilot"):
        # r2 C1/C2: a paused (or never opted-in) project is not advanced by a
        # background run — only the owner's own turn turns autopilot on
        return AdvanceResult(True, None, "blocked",
                             "autopilot is off — only the owner's own turn turns it on")
    _mine = {"boot": _BOOT_ID, "ts": time.time(), "id": os.urandom(4).hex()}
    _claim = {"lost": False, "busy": False}

    def _enter(m):
        prev = m.get("step_in_flight")
        if isinstance(prev, dict) and prev.get("boot") == _BOOT_ID:
            # §4PG (R1-6): a same-boot marker older than the step cap plus
            # grace cannot be a live step (its `_leave` failed) — take it
            if time.time() - float(prev.get("ts") or 0) <= UNATTENDED_STEP_TIMEOUT_S + 600:
                _claim["busy"] = True           # a step of THIS process is running (r2 M1)
                return m
            _claim["lost"] = True
        if prev:                                # stamped by a process that died mid-step
            _claim["lost"] = True
            m["no_progress_streak"] = int(m.get("no_progress_streak") or 0) + 1
        m["step_in_flight"] = _mine
        return m
    meta = _bump_meta(store, project_id, _enter)
    if _claim["busy"]:
        return AdvanceResult(True, None, "blocked", "another unattended step on this project is running")
    try:
        if _claim["lost"]:
            _increment_budget(store, project_id)
            try:
                store.log_event(project_id, None, "autopilot_step_lost",
                                {"streak": meta.get("no_progress_streak")})
            except Exception:  # noqa: BLE001
                pass
            if int(meta.get("no_progress_streak") or 0) >= NO_PROGRESS_PAUSE_AT:
                pause_autopilot(context, project_id, "no progress",
                                f"{NO_PROGRESS_PAUSE_AT} steps in a row moved nothing forward "
                                "(one was lost to a restart)")
                return AdvanceResult(True, None, "blocked", "autopilot paused: no progress")
        # §4MR: an exhausted step budget (or the lifetime runtime / tool-call
        # caps) blocked every tick silently with autopilot left ON — it is a
        # pause like the others: autopilot off, the owner told once
        _bud = _get_budget(store, project_id)
        if _bud["used"] >= _bud["cap"]:
            pause_autopilot(context, project_id, "step budget used",
                            f"{_bud['used']}/{_bud['cap']} steps — the owner raises it with action=budget")
            return AdvanceResult(True, None, "blocked", "autopilot paused: step budget")
        from .project_safety import check_budget as _check_budget
        _sec = _check_budget(meta)
        if not _sec.allowed:
            pause_autopilot(context, project_id, "budget used", _sec.reason)
            return AdvanceResult(True, None, "blocked", "autopilot paused: " + _sec.reason)
        if float(meta.get("unattended_runtime_seconds") or 0) >= _unattended_cap_s(meta):
            pause_autopilot(context, project_id, "runtime budget used",
                            f"{_unattended_cap_s(meta) / 3600:.1f} h of unattended work since your last look")
            return AdvanceResult(True, None, "blocked", "autopilot paused: runtime budget")
        t0 = time.time()
        timeout = UNATTENDED_STEP_TIMEOUT_S if step_timeout_s is None else float(step_timeout_s)
        claimed: list = []
        try:
            # §4SD: the leaf sizes its calls by the time it has — a 64k file
            # call (~73 min) must not start in a 45-min step it cannot finish in
            res = await _wait_for(_with_leaf_deadline(
                advance_once(context, project_id, owner_requested=False, claim_sink=claimed, **kw),
                time.monotonic() + max(0.0, timeout - 60.0)), timeout)
        except asyncio.TimeoutError:
            res = _after_abort(store, project_id, claimed, t0,
                               "timeout", f"step passed its {int(timeout)} s cap")
        except asyncio.CancelledError:
            # §4SD open items r4 (review MAJOR): the step stopped from outside
            # (a scheduled/probe batch at its limit, a Stop) — `advance_once`
            # leaves a tracked claim to this caller: reopened and charged
            try:                                   # r6 m3: nothing here replaces the cancel
                _after_abort(store, project_id, claimed, t0, "stopped", "step was stopped")
                _bump_meta(store, project_id, lambda m: {**m, "unattended_runtime_seconds": float(   # r5 m3
                    m.get("unattended_runtime_seconds") or 0) + (time.time() - t0)})
            except Exception:  # noqa: BLE001
                logger.warning("stopped unattended step not booked", exc_info=True)
            raise
        except Exception as e:  # noqa: BLE001 — r2 m3: a raising step is charged and scored too
            logger.warning("unattended step on %s raised: %s", project_id, e, exc_info=True)
            res = _after_abort(store, project_id, claimed, t0, "error", f"step raised {type(e).__name__}: {e}")
        else:
            # §4MR: a step that RETURNED with its leaf still claimed (a path
            # that stopped without closing it) would wedge the leaf until the
            # next boot — and score nothing, so the gate never paused
            for _tid in claimed:
                if _task_status(store, project_id, _tid) == "IN_PROGRESS":
                    try:
                        store.update_task(_tid, status="READY",
                                          failure_reason="unattended step ended without closing it")
                    except Exception:  # noqa: BLE001
                        logger.debug("left-claimed leaf not reset", exc_info=True)
        _elapsed = time.time() - t0
        _bump_meta(store, project_id, lambda m: {**m, "unattended_runtime_seconds": float(
            m.get("unattended_runtime_seconds") or 0) + _elapsed})
    finally:
        def _leave(m):
            cur = m.get("step_in_flight")
            if isinstance(cur, dict) and cur.get("id") == _mine["id"]:
                m["step_in_flight"] = None
            return m
        _bump_meta(store, project_id, _leave)
    # only a step that WORKED on a task (or was aborted) is scored; a tick
    # that claimed nothing is not a step
    if not (res.task_id or res.classification in ("timeout", "error")) or res.classification == "deferred":
        return res                     # §4PG r3: a deferral is not a step
    score = _step_score(store, project_id, res.task_id)
    if score == "held" and "did not land" not in str(res.summary or ""):
        return res                     # waits on the owner — the hold already said so (r2 m1)
    # §4PG (R1-4): a hold because the research could not be WRITTEN is not
    # the owner's business — it is a systemic failure and counts toward the
    # no-progress pause like any other step that moved nothing

    def _score(m):
        if score == "moved":
            m["no_progress_streak"] = 0
            m["steps_since_checkpoint"] = int(m.get("steps_since_checkpoint") or 0) + 1
        else:
            m["no_progress_streak"] = int(m.get("no_progress_streak") or 0) + 1
        return m
    meta = _bump_meta(store, project_id, _score)
    if int(meta.get("no_progress_streak") or 0) >= NO_PROGRESS_PAUSE_AT:
        pause_autopilot(context, project_id, "no progress",
                        f"{NO_PROGRESS_PAUSE_AT} steps in a row moved nothing forward (last: {res.summary[:120]})")
    else:
        every = meta.get("checkpoint_every")
        every = DEFAULT_CHECKPOINT_EVERY if every is None else int(every)
        if score == "moved" and every > 0 and int(meta.get("steps_since_checkpoint") or 0) >= every:
            pause_autopilot(context, project_id, "check-in",
                            f"{every} steps done since your last look — review the project before more run")
    return res


def unattended_step_live(store, project_id) -> str:
    """§4PG (R1-1): "N min" when a step of THIS process is running on the
    project (a fresh same-boot `step_in_flight`), else "". The owner's own
    batch refuses rather than build beside it."""
    try:
        m = ((store.get_project(project_id) or {}).get("metadata") or {})
        cur = m.get("step_in_flight")
        if isinstance(cur, dict) and cur.get("boot") == _BOOT_ID:
            age = time.time() - float(cur.get("ts") or 0)
            if age <= UNATTENDED_STEP_TIMEOUT_S + 600:
                return f"{int(age // 60)} min" if age >= 60 else f"{int(age)} s"
    except Exception:  # noqa: BLE001
        pass
    return ""


def _after_abort(store, project_id, claimed, t0, kind, why) -> AdvanceResult:
    """A step that timed out or raised: ITS leaf (only — r2 M2) goes back to
    READY, and the step is charged — unless it had in fact finished in the
    cancel grace (r2 m2: then it was charged already and is scored as is)."""
    tid = claimed[0] if claimed else None
    if tid and _task_status(store, project_id, tid) != "IN_PROGRESS":     # §4PG (R1-5): both kinds
        return AdvanceResult(True, tid, "finished", f"finished before the abort ({why})")
    if tid:
        try:
            store.update_task(tid, status="READY", failure_reason=f"unattended {why}")
        except Exception:  # noqa: BLE001
            logger.debug("aborted leaf not reset", exc_info=True)
    _increment_budget(store, project_id)
    try:
        from .project_safety import record_runtime
        record_runtime(store, project_id, seconds=time.time() - t0)
    except Exception:  # noqa: BLE001
        pass
    return AdvanceResult(True, tid, kind, why)


def _paused_since(store, project_id, t0: float) -> bool:
    try:
        rec = ((store.get_project(project_id) or {}).get("metadata") or {}).get("autopilot_paused")
        return bool(rec) and float(rec.get("ts") or 0) >= t0
    except Exception:  # noqa: BLE001
        return False


def _batch_would_cross_deadline(next_step_s: float) -> bool:
    """True when this request's client deadline would arrive inside its
    report reserve before another step of ``next_step_s`` could finish.
    No deadline known (a background run) → False."""
    try:
        from ..utils.logging import request_id_context, request_remaining_s, request_deadline_s
        rid = request_id_context.get() or ""
        remaining = request_remaining_s(rid)
        if remaining is None:
            return False
        from .agent import effective_report_floor
        return float(next_step_s) > float(remaining) - effective_report_floor(request_deadline_s(rid), rid)   # §4RA r2: sized
    except Exception:  # noqa: BLE001
        return False


def _is_unrequested_task(node) -> bool:
    """A task nobody asked for: filed by the verifier on its own."""
    return str(getattr(node, "description", "") or "").lstrip().lower().startswith("verifier follow-up:")


async def advance_many(
    context,
    project_id: str,
    *,
    max_tasks: Optional[int],
    owner_requested: bool = False,
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

    _batch_t0 = time.time()
    _longest_step = 0.0
    for _ in range(limit):
        # §4MQ: a batch inside a request stops BEFORE a step that would end
        # inside the report reserve — "finish the project" ran up to 40
        # builds in one tool call and nothing checked the clock
        if _batch_would_cross_deadline(_longest_step or FIRST_STEP_ESTIMATE_S):
            stop_reason = "deadline"
            break
        _step_t0 = time.time()
        _kw = dict(tool_runner=tool_runner, llm_classifier=llm_classifier,
                   code_generator=code_generator, coding_executor=coding_executor)
        if owner_requested:
            res = await advance_once(context, project_id, owner_requested=True, **_kw)
        else:
            # a scheduled task's, sub-agent's or probe's batch runs under
            # the unattended gates (§4MQ) — and stops when they pause
            res = await advance_unattended(context, project_id, **_kw)
        _longest_step = max(_longest_step, time.time() - _step_t0)
        _off = not owner_requested and res.summary.startswith("autopilot is off")
        if not owner_requested and (_off or _paused_since(store, project_id, _batch_t0)):
            stop_reason = "autopilot_off" if _off else "autopilot_paused"
            if res.task_id:
                advanced.append({"task_id": res.task_id, "classification": res.classification,
                                 "status": (store.get_task(res.task_id) or {}).get("status"),
                                 "summary": res.summary})
            break
        cls = (res.classification or "").lower()
        if cls == "idle":
            # No ready leaf — all tasks terminal, or the rest are blocked by a
            # failed dependency. Either way the batch is done advancing. But
            # "done advancing" is not "done": when the LEDGER holds FAILED
            # tasks (from an earlier batch/request — _final_reason only sees
            # THIS batch), reporting project_done relays a false completion.
            stop_reason = _final_reason("project_done")
            if stop_reason == "project_done":
                try:
                    _sts = [str(t.get("status", "")).upper()
                            for t in store.list_tasks(project_id)]
                    if "FAILED" in _sts:
                        stop_reason = "project_failed"
                    elif "NEEDS_USER" in _sts:
                        # tasks wait on the owner — "done" would be relayed as
                        # "all tasks are complete" (§4LQ review R1)
                        stop_reason = "needs_user"
                    elif "IN_PROGRESS" in _sts:
                        # §4MR: claimed and never closed (another run, or an
                        # interrupted one) — not "all tasks are complete"
                        stop_reason = "in_progress"
                    elif any(x in _sts for x in ("PENDING", "READY", "BLOCKED", "PAUSED")):
                        # §4QA-d review: no ready leaf, yet open tasks remain —
                        # held by a dependency that is BLOCKED/PAUSED (or one
                        # the picker skips). That is a stall, not a done project.
                        stop_reason = "open_but_waiting"
                except Exception:
                    logger.debug("idle-stop ledger scan skipped",
                                 exc_info=True)
            break
        if cls == "blocked" and res.summary.startswith("another unattended step"):
            stop_reason = "busy"                  # §4MR: not a budget
            break
        if cls == "blocked" and res.task_id and res.summary.startswith(MODEL_EXPERIMENT_PREFIX):
            advanced.append({"task_id": res.task_id, "classification": res.classification,
                             "status": (store.get_task(res.task_id) or {}).get("status"),
                             "summary": res.summary})
            stop_reason = "model_experiment"      # §4QA: the turn runs it with probe_model
            break
        if res.task_id and res.summary.startswith(OUT_OF_TIME_PREFIX):
            advanced.append({"task_id": res.task_id, "classification": res.classification,
                             "status": (store.get_task(res.task_id) or {}).get("status"),
                             "summary": res.summary})
            stop_reason = "out_of_time"           # §4SD r12: not re-picked in the same batch
            break
        if cls == "blocked" and res.task_id and res.summary.startswith("coding executor crashed"):
            advanced.append({"task_id": res.task_id, "classification": res.classification,
                             "status": (store.get_task(res.task_id) or {}).get("status"),
                             "summary": res.summary})
            stop_reason = "step_crashed"          # §4MR: was reported as a budget stop
            break
        if cls == "blocked":
            # budget exhaustion, a non-ACTIVE project, or a project that just
            # rolled up (DONE if all done; FAILED if any task failed).
            pstatus = (store.get_project(project_id) or {}).get("status")
            if pstatus == "DONE":
                stop_reason = "project_done"
            elif pstatus == "FAILED":
                # NEVER "project_done" here. A batch invoked on an
                # already-FAILED project used to fall through
                # _final_reason("project_done") — with nothing advanced in
                # THIS batch it reported "All tasks are complete — the
                # project is done" over a ledger that said FAILED, and the
                # agent relayed the completion to the user verbatim
                # (2026-08-01 Mini AI incident, final autoadvance call).
                stop_reason = ("completed_with_failures"
                               if any(a.get("status") == "FAILED"
                                      for a in advanced)
                               else "project_failed")
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
    # A migrated tool ANSWERS this question — read it before sniffing its
    # prose. Measured on the 4,391-call corpus, 82 of 82 refusals reached
    # this function as clean successes, and this is the UNATTENDED path: a
    # refused edit let the idle autoadvancer mark its task DONE with nothing
    # done. UNRESOLVED counts as a failure here for the same reason the
    # promoted-job check below does — see that comment.
    #
    # ADD-only: an `ok` status falls through to the prose rules, so the
    # non-zero `EXIT CODE:` banner and the `[SYSTEM ERROR]` sentinel keep
    # all the authority they had.
    _st = getattr(output, "status", None)
    if _st is not None and str(getattr(_st, "value", _st)) != "ok":
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
    # A command DETACHED at its budget (sandbox/jobs.py) reports exit 0 while
    # STILL RUNNING. It is not a failure — but it is emphatically not evidence
    # of success either, and this function's answer is what lets a tick call
    # `update_status(..., DONE)`. Treated as a failure HERE on purpose: the
    # caller's only two options are DONE and retry, and retrying an unfinished
    # command is far cheaper than marking a task complete on work that has not
    # happened. (The re-run guard makes that retry a no-op that returns the
    # same job.) The sibling classify_verify_result calls it "inconclusive",
    # which is the same verdict in the vocabulary that has three answers.
    from ..sandbox.jobs import is_promoted_result as _promoted
    if _promoted(s):
        return True
    from ..tools.tool_failure import exec_exit_code as _exec_exit_code
    _code = _exec_exit_code(s)  # R4-1: execute-shaped only, line-anchored
    if _code is not None:
        return _code != 0
    if "[SYSTEM ERROR]" in s or "Critical Tool Error" in s:
        return True
    # …and the SHARED classifier (§4LZ B7): this private copy missed
    # "CRITICAL ERROR", "SYSTEM ERROR", "Security Error", "SYSTEM
    # INSTRUCTION" and "REJECTED" heads, so the unattended advancer could
    # close a task DONE on a refusal
    try:
        from ..tools.tool_failure import result_is_failure, result_is_rejection
        if result_is_failure(s) or result_is_rejection(s) or s.startswith("Security Error"):
            return True
    except Exception:  # noqa: BLE001
        pass
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
    # A migrated tool ANSWERS this, and this gate is what marks a task DONE.
    # A non-OK status can never be a PASS: a refusal verified nothing, and
    # an unfinished job has no verdict — both are "inconclusive", which is
    # this function's word for "do NOT mark it done".
    _st = getattr(output, "status", None)
    if _st is not None and str(getattr(_st, "value", _st)) != "ok":
        return "fail" if str(getattr(_st, "value", _st)) == "failed" \
            else "inconclusive"
    s = str(output or "").strip()
    if not s:
        return "inconclusive"
    if s.startswith("SANDBOX EGRESS BLOCKED"):
        return "inconclusive"  # command NOT executed — nothing was verified
    # A verify command that outran its budget was DETACHED, not killed
    # (sandbox/jobs.py). Its result is success-SHAPED (`EXIT CODE: 0`, no
    # error sentinel) so the turn loop does not score a strike — but the
    # command has NOT finished, so it is the fourth "success-shaped output
    # that verified nothing" this function exists to reject. Without this a
    # 600 s+ `pytest`/`npm run build` marks its task DONE, unattended, in the
    # idle autoadvancer, and the job's eventual exit 1 is never reconciled.
    from ..sandbox.jobs import is_promoted_result
    if is_promoted_result(s):
        return "inconclusive"
    from ..tools.tool_failure import exec_exit_code as _exec_exit_code
    _code = _exec_exit_code(s)  # R4-1: the execute banner, not a quoted one
    if _code is None:
        return "inconclusive"  # no exit code at all — not proof of anything
    if _code != 0:
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
