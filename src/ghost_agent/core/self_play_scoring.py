"""Scoring helpers for synthetic self-play runs.

The old pipeline drove curriculum targeting purely off a `compression
_delta` value (tool-invocation count vs. prior best). That metric
rewards "fewer tool calls" without caring about correctness — a
confidently-wrong one-shot solve would outrank a careful five-step
correct solve. This module exposes a correctness-weighted score that
the frontier tracker and the adaptive cooldown can use instead.

The score is deliberately cheap to compute and has sane behaviour on
boundary cases (no prior best, failed run, etc).
"""

from typing import Iterable, Optional


# Tool names whose result content is USER DATA (fixture file contents,
# retrieved documents, search results). The self-play challenge often
# involves counting "ERROR" / "WARN" lines in a log fixture, so a
# legitimate `file_read` result will carry those tokens inside the data
# — it is NOT a tool failure. The original heuristic blindly
# substring-matched and counted every such read as an error; production
# trace 20:17 cycle 2 scored +0.400 (6 "errors") on a correct solve
# where the agent read 5 log files that happened to contain ERROR/WARN
# tokens. Excluding these tools fixes that false positive.
_DATA_TOOL_NAMES = frozenset({
    "file_system",       # file_read / list / search return raw file content
    "recall",            # vector search over ingested docs
    "knowledge_base",    # doc retrieval / listing
    "web_search",        # search-engine result pages
    "deep_research",     # synthesised research summaries
    "fact_check",        # third-party retrieval
    "postgres_admin",    # SELECT results are data
})

# Patterns that are UNAMBIGUOUS tool failures — safe to scan anywhere
# in the content because they essentially only appear in actual stack
# traces / system alerts, not in fixture data.
_UNAMBIGUOUS_ERROR_PATTERNS = (
    "Traceback (most recent call last)",
    "SYSTEM ERROR: Tool",
    "SYSTEM ALERT: You have failed",
    "CalledProcessError",
    "TimeoutExpired",
)

# Markers that are failure signals ONLY when they appear at the START
# of the tool result (after optional whitespace). A file-read result
# whose third line contains "ERROR: disk full" is fixture data; a tool
# result that STARTS with "Error: Invalid JSON" is a genuine tool
# failure emitted by the dispatch code.
_LINE_START_ERROR_PREFIXES = (
    "Error:", "ERROR:", "error:",
    "AssertionError", "SyntaxError", "NameError",
    "IndentationError", "ImportError", "ModuleNotFoundError",
)


def count_tool_errors(messages: Iterable[dict]) -> int:
    """Count tool-result messages whose content looks like an error.

    Three-tier discrimination, tightened from the old "any marker
    anywhere in content" heuristic which produced false positives on
    fixture reads (any log file that contained the literal text
    "ERROR:" made every `file_read` look like a tool failure):

      1. Data-retrieval tools (file_system, recall, web_search, …)
         never count — their content is fixture / retrieved data, not
         failure output.
      2. Unambiguous patterns ("Traceback (most recent call last)",
         "SYSTEM ALERT: You have failed", etc.) count anywhere in the
         content.
      3. Error-prefix markers ("Error:", "AssertionError", …) only
         count when they appear at the START of the content — that's
         the shape dispatch code uses when it synthesizes a failure
         message, and it doesn't collide with fixture data.
    """
    if not messages:
        return 0
    n = 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "tool":
            continue

        name = str(m.get("name") or "").lower()
        if name in _DATA_TOOL_NAMES:
            # Retrieved data: content is the payload, not a failure
            # signal. Skip unconditionally.
            continue

        content = str(m.get("content") or "")
        if not content:
            continue

        # Tier 2: unambiguous patterns (stack traces, system alerts).
        if any(p in content for p in _UNAMBIGUOUS_ERROR_PATTERNS):
            n += 1
            continue

        # Tier 3: prefix markers on the first non-whitespace line only.
        head = content.lstrip()[:120]
        if any(head.startswith(p) for p in _LINE_START_ERROR_PREFIXES):
            n += 1
            continue

    return n


def correctness_weighted_score(
    *,
    passed: bool,
    compression_delta: float,
    tool_errors: int,
    alpha: float = 0.4,
    beta: float = 0.1,
    novelty: Optional[float] = None,
    attempts_used: Optional[int] = None,
    gamma_novelty: float = 0.6,
    delta_attempts: float = 0.3,
) -> float:
    """Multi-signal correctness-weighted score.

    Pre-2026-05 the formula was just::

        score = passed*(1 + α*compression_delta) − β*tool_errors

    On deterministic templates `compression_delta` (the old tool-count
    proxy) was pinned at 0 because tool counts barely move between
    consecutive wins. The score collapsed to pass/fail and the frontier
    tracker saw a flat signal — see the post-mortem dated 2026-05-17.

    The new combined score is::

        if passed:
            base = 1.0 + α·compression_delta + γ·novelty + δ·attempts_efficiency
        else:
            base = 0.0
        score = base − β·tool_errors

    Where:
      * `compression_delta` ∈ [-1, +1] — kept for back-compat, but now
        only one of three positive signals.
      * `novelty` ∈ [0, 1] — structural diversity of the solution AST
        relative to prior winning solutions for this cluster. None
        means "caller didn't supply it" → treated as 0 contribution
        (preserves pre-existing test expectations).
      * `attempts_used` — when supplied, contributes via
        ``attempts_efficiency`` (1-shot=1.0, 2-shot=0.5, 3-shot=0.2);
        None → no contribution.

    Defaults: with both new signals at None the formula reduces to the
    historical one, so every existing test continues to assert the
    same numbers. New call sites should supply both new signals to
    actually break the score plateau.
    """
    try:
        delta = float(compression_delta)
    except Exception:
        delta = 0.0
    if passed:
        base = 1.0 + alpha * delta
        if novelty is not None:
            try:
                nv = float(novelty)
                if nv < 0.0:
                    nv = 0.0
                elif nv > 1.0:
                    nv = 1.0
                base += gamma_novelty * nv
            except Exception:
                pass
        if attempts_used is not None:
            try:
                from .solution_novelty import attempts_efficiency
                base += delta_attempts * attempts_efficiency(int(attempts_used))
            except Exception:
                pass
    else:
        base = 0.0
    score = base - beta * max(0, int(tool_errors))
    return round(score, 4)
