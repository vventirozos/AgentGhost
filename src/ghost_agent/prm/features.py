"""Per-step feature extraction for the Process Reward Model.

The PRM scores `(state, candidate_action)` tuples. To keep the model
inspectable, deterministic, and dependency-free at feature time, the
feature set is hand-crafted: plain regex / string ops over the request,
the plan progress so far, and the candidate action. No embeddings.

Order of features is FROZEN — the trained model's weights index into
``PRM_FEATURE_NAMES`` so any reorder breaks loaded checkpoints. New
features must be APPENDED, never inserted, and existing models must be
retrained when ``PRM_FEATURE_NAMES`` grows (the saved JSON records the
names so a stale checkpoint will be detected at load time).

The state inputs (``PlanState``) are the minimum the planner already
knows: the user request, what tools have run this turn, how many of
them failed. The action inputs (``ActionFeatures``) come straight from
``core.mcts.ActionCandidate`` (description, tool name, args). Both
shapes are dataclasses so callers can build them once per turn and
reuse across candidate scoring.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple


# ──────────────────────────────────────────────────────────────────────
# Tool taxonomy
#
# Buckets, not one-hots, so adding a new tool doesn't invalidate every
# trained checkpoint. Each tool falls into exactly one bucket; unknown
# tools land in ``tool_is_unknown`` (the bucket trained on garbage-in
# requests so the model has somewhere to express "I have no signal").
# ──────────────────────────────────────────────────────────────────────

_HEAVYWEIGHT_TOOLS: FrozenSet[str] = frozenset({
    "browser",
    "execute",
    "image_gen",
    "vision",
    "deep_research",
    "delegate_to_swarm",
    "swarm_consensus",
})

_LIGHTWEIGHT_TOOLS: FrozenSet[str] = frozenset({
    "file_system",
    "scratchpad",
    "list_files",
    "read_file",
    "write_file",
    "search_filesystem",
})

_EXTERNAL_TOOLS: FrozenSet[str] = frozenset({
    "web_search",
    "fast_url",
    "knowledge_base",
    "database",
    "url_fetch",
    "tor_browse",
})

_MEMORY_TOOLS: FrozenSet[str] = frozenset({
    "skill_memory",
    "episodic_memory",
    "graph_memory",
    "vector_memory",
    "manage_projects",
    "forget",
    "smart_update",
    "self_play",
    "self_play_loop",
})


# ──────────────────────────────────────────────────────────────────────
# Compiled regex used by feature extraction
# ──────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_FILE_PATH_RE = re.compile(
    r"(?:\./|[a-zA-Z_][\w.\-/]*?\.(?:py|js|ts|go|rs|c|cpp|h|sh|sql|md|"
    r"json|yml|yaml|toml|ini|cfg|html|css))"
)
_CODE_FENCE_RE = re.compile(r"```")
_QUESTION_WORDS_RE = re.compile(
    r"\b(?:what|why|how|when|where|which|who|can|could|should|would|"
    r"is|are|do|does|did)\b",
    re.IGNORECASE,
)
_IMPERATIVE_VERBS: FrozenSet[str] = frozenset([
    "write", "build", "create", "generate", "implement", "refactor",
    "debug", "fix", "run", "execute", "compile", "deploy", "analyze",
    "analyse", "scrape", "extract", "parse", "crawl", "download",
    "search", "find", "show", "list", "explain", "summarize", "design",
    "optimize", "benchmark", "profile", "test", "install", "configure",
    "investigate", "research",
])
_TECHNICAL_JARGON: FrozenSet[str] = frozenset([
    "api", "endpoint", "database", "sql", "regex", "parser", "json",
    "yaml", "http", "tcp", "socket", "thread", "async", "await",
    "docker", "container", "kernel", "linux", "ssh", "vpn", "proxy",
    "tor", "onion", "cve", "exploit", "vulnerability", "subnet", "ip",
    "dns", "ssl", "tls", "oauth", "token", "header", "cookie",
    "selector", "xpath", "dom", "playwright", "selenium", "browser",
    "function", "class", "variable", "loop", "iteration", "recursion",
    "concurrency", "threadpool", "error", "exception", "stacktrace",
    "algorithm", "graph", "tree", "hashmap", "index", "optimization",
])


# ──────────────────────────────────────────────────────────────────────
# Feature names — FROZEN ORDER
# ──────────────────────────────────────────────────────────────────────

PRM_FEATURE_NAMES: Tuple[str, ...] = (
    # ── State: request shape ─────────────────────────────────────────
    "request_length_log1p",
    "request_word_count_log1p",
    "request_has_code_fence",
    "request_has_url",
    "request_imperative_count_log1p",
    "request_jargon_count_log1p",
    "request_question_words_ratio",
    "request_has_question_mark",

    # ── State: plan progress ─────────────────────────────────────────
    "plan_steps_so_far_log1p",
    "plan_failures_so_far_log1p",
    "plan_pending_count_log1p",
    "plan_depth",
    "plan_has_any_failure",

    # ── Action: candidate shape ──────────────────────────────────────
    "action_desc_length_log1p",
    "action_args_count_log1p",
    "action_args_total_length_log1p",
    "action_has_url_in_args",
    "action_has_filepath_in_args",

    # ── Action: tool category buckets ────────────────────────────────
    "tool_is_heavyweight",
    "tool_is_lightweight",
    "tool_is_external",
    "tool_is_memory",
    "tool_is_unknown",

    # ── Cross features ───────────────────────────────────────────────
    "tool_already_used_this_turn",
    "tool_failed_this_turn",
)


# ──────────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanState:
    """Snapshot of where the agent is in its current turn.

    The PRM only needs the bits that affect plan-quality scoring —
    full plan tree state is intentionally NOT passed in (would couple
    the PRM to ``planning.TaskTree`` schema and break feature stability
    when that schema evolves).
    """

    user_request: str = ""
    steps_so_far: int = 0
    failures_so_far: int = 0
    pending_count: int = 0
    plan_depth: int = 0
    # Names of tools already invoked this turn — used to flag repeats.
    tools_used_this_turn: Tuple[str, ...] = ()
    # Names of tools that errored this turn — used to flag refire on
    # already-failing tools as risky.
    tools_failed_this_turn: Tuple[str, ...] = ()


@dataclass
class ActionFeatures:
    """The candidate action shape the PRM scores.

    Fields mirror ``core.mcts.ActionCandidate`` so callers can adapt
    one to the other without translation logic in the hot path.
    """

    description: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Frozen-order float tuple — feed straight to the model. ``by_name``
    is a parallel dict for human inspection / debugging."""

    values: Tuple[float, ...]
    by_name: Dict[str, float]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.by_name)


# ──────────────────────────────────────────────────────────────────────
# Feature computation
# ──────────────────────────────────────────────────────────────────────

def _log1p(n: int) -> float:
    return math.log1p(max(0, int(n)))


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _request_features(request: str) -> Dict[str, float]:
    t = request or ""
    t_low = t.lower()
    words = re.findall(r"\w+", t_low)
    n_words = len(words)
    n_questions = len(_QUESTION_WORDS_RE.findall(t))
    n_imperative = sum(1 for w in words if w in _IMPERATIVE_VERBS)
    n_jargon = sum(1 for w in words if w in _TECHNICAL_JARGON)
    return {
        "request_length_log1p": _log1p(len(t)),
        "request_word_count_log1p": _log1p(n_words),
        "request_has_code_fence": 1.0 if _CODE_FENCE_RE.search(t) else 0.0,
        "request_has_url": 1.0 if _URL_RE.search(t) else 0.0,
        "request_imperative_count_log1p": _log1p(n_imperative),
        "request_jargon_count_log1p": _log1p(n_jargon),
        "request_question_words_ratio": (n_questions / n_words) if n_words else 0.0,
        "request_has_question_mark": 1.0 if "?" in t else 0.0,
    }


def _plan_features(state: PlanState) -> Dict[str, float]:
    return {
        "plan_steps_so_far_log1p": _log1p(state.steps_so_far),
        "plan_failures_so_far_log1p": _log1p(state.failures_so_far),
        "plan_pending_count_log1p": _log1p(state.pending_count),
        "plan_depth": float(max(0, int(state.plan_depth))),
        "plan_has_any_failure": 1.0 if state.failures_so_far > 0 else 0.0,
    }


def _action_args_total_length(args: Dict[str, Any]) -> int:
    if not isinstance(args, dict):
        return 0
    total = 0
    for v in args.values():
        if isinstance(v, str):
            total += len(v)
        elif isinstance(v, (list, tuple, dict)):
            try:
                total += len(str(v))
            except Exception:
                continue
        elif v is None:
            continue
        else:
            total += len(str(v))
    return total


def _action_args_has_pattern(args: Dict[str, Any], pattern: re.Pattern) -> bool:
    if not isinstance(args, dict):
        return False
    for v in args.values():
        if isinstance(v, str) and pattern.search(v):
            return True
    return False


def _action_features(action: ActionFeatures) -> Dict[str, float]:
    args = action.tool_args if isinstance(action.tool_args, dict) else {}
    return {
        "action_desc_length_log1p": _log1p(len(action.description or "")),
        "action_args_count_log1p": _log1p(len(args)),
        "action_args_total_length_log1p": _log1p(_action_args_total_length(args)),
        "action_has_url_in_args": (
            1.0 if _action_args_has_pattern(args, _URL_RE) else 0.0
        ),
        "action_has_filepath_in_args": (
            1.0 if _action_args_has_pattern(args, _FILE_PATH_RE) else 0.0
        ),
    }


def _tool_bucket_features(tool_name: str) -> Dict[str, float]:
    name = (tool_name or "").strip()
    is_heavy = name in _HEAVYWEIGHT_TOOLS
    is_light = name in _LIGHTWEIGHT_TOOLS
    is_ext = name in _EXTERNAL_TOOLS
    is_mem = name in _MEMORY_TOOLS
    is_unknown = not (is_heavy or is_light or is_ext or is_mem)
    return {
        "tool_is_heavyweight": 1.0 if is_heavy else 0.0,
        "tool_is_lightweight": 1.0 if is_light else 0.0,
        "tool_is_external": 1.0 if is_ext else 0.0,
        "tool_is_memory": 1.0 if is_mem else 0.0,
        "tool_is_unknown": 1.0 if is_unknown else 0.0,
    }


def _cross_features(state: PlanState, action: ActionFeatures) -> Dict[str, float]:
    name = (action.tool_name or "").strip()
    used = name and name in state.tools_used_this_turn
    failed = name and name in state.tools_failed_this_turn
    return {
        "tool_already_used_this_turn": 1.0 if used else 0.0,
        "tool_failed_this_turn": 1.0 if failed else 0.0,
    }


def extract_step_features(
    state: PlanState,
    action: ActionFeatures,
) -> FeatureVector:
    """Return a frozen-order feature vector for ``(state, action)``.

    The returned vector is deterministic: same inputs → same vector,
    bit-for-bit. That's what lets the model JSON checkpoint persist
    across runs.
    """
    d: Dict[str, float] = {}
    d.update(_request_features(state.user_request))
    d.update(_plan_features(state))
    d.update(_action_features(action))
    d.update(_tool_bucket_features(action.tool_name))
    d.update(_cross_features(state, action))

    # Sanity: feature dict must contain every name in the frozen list.
    # If it doesn't, the model would silently mis-align — fail loud.
    missing = set(PRM_FEATURE_NAMES) - set(d.keys())
    if missing:
        raise RuntimeError(
            f"PRM feature extraction missing keys: {sorted(missing)}"
        )

    values = tuple(float(d[name]) for name in PRM_FEATURE_NAMES)
    return FeatureVector(values=values, by_name=d)


def feature_vector_to_list(fv: FeatureVector) -> List[float]:
    """Convenience: list form of the frozen tuple."""
    return list(fv.values)
