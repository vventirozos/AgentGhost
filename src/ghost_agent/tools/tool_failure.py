"""Tool failure classification and routing.

Categorises tool execution errors into three buckets so the agent loop
can decide whether to retry, replan, or self-correct:

* **RETRYABLE** — transient infrastructure errors (timeout, rate-limit,
  connection reset, sandbox busy). The loop should retry with exponential
  back-off (up to a cap).
* **FATAL** — permanent errors that no retry will fix (permission denied,
  invalid arguments, tool not found). The loop should mark the task FAILED
  and trigger a replan.
* **DIAGNOSTIC** — errors that contain useful debugging information the LLM
  can reason about (assertion failures, runtime errors, syntax errors).
  The error message is injected into context so the LLM can self-correct.
"""

import logging
import re
from enum import Enum
from typing import Tuple

logger = logging.getLogger("GhostAgent")


class FailureClass(str, Enum):
    RETRYABLE = "retryable"
    FATAL = "fatal"
    DIAGNOSTIC = "diagnostic"
    UNKNOWN = "unknown"


# Pattern → classification mapping. Order matters: first match wins.
_RETRYABLE_PATTERNS = [
    re.compile(r"timed?\s*out", re.IGNORECASE),
    re.compile(r"timeout", re.IGNORECASE),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"connection.?(reset|refused|error|closed)", re.IGNORECASE),
    re.compile(r"ECONNREFUSED|ECONNRESET|ETIMEDOUT", re.IGNORECASE),
    re.compile(r"sandbox.?(busy|unavailable|starting)", re.IGNORECASE),
    re.compile(r"container.?(not running|starting)", re.IGNORECASE),
    re.compile(r"\b(?:502|503|504)\b", re.IGNORECASE),
    re.compile(r"service.?unavailable", re.IGNORECASE),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
]

_FATAL_PATTERNS = [
    re.compile(r"permission.?denied", re.IGNORECASE),
    re.compile(r"access.?denied", re.IGNORECASE),
    re.compile(r"not found.*tool", re.IGNORECASE),
    re.compile(r"tool.*not found", re.IGNORECASE),
    re.compile(r"MANDATORY", re.IGNORECASE),
    re.compile(r"invalid.?(arg|param|schema)", re.IGNORECASE),
    re.compile(r"authentication.?(failed|required|error)", re.IGNORECASE),
    re.compile(r"401|403", re.IGNORECASE),
]

_DIAGNOSTIC_PATTERNS = [
    re.compile(r"AssertionError|AssertError", re.IGNORECASE),
    re.compile(r"RuntimeError", re.IGNORECASE),
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"IndentationError", re.IGNORECASE),
    re.compile(r"TypeError|ValueError|KeyError|IndexError|AttributeError", re.IGNORECASE),
    re.compile(r"NameError", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"EXIT CODE: [1-9]", re.IGNORECASE),
    re.compile(r"FileNotFoundError|IOError|OSError", re.IGNORECASE),
    re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ZeroDivisionError", re.IGNORECASE),
]


def classify_tool_failure(error_text: str) -> Tuple[FailureClass, str]:
    """Classify a tool error string into a failure category.

    Returns ``(FailureClass, matched_pattern_description)``.
    """
    if not error_text or not isinstance(error_text, str):
        return FailureClass.UNKNOWN, "empty error"

    for pat in _RETRYABLE_PATTERNS:
        m = pat.search(error_text)
        if m:
            return FailureClass.RETRYABLE, m.group(0)

    for pat in _FATAL_PATTERNS:
        m = pat.search(error_text)
        if m:
            return FailureClass.FATAL, m.group(0)

    for pat in _DIAGNOSTIC_PATTERNS:
        m = pat.search(error_text)
        if m:
            return FailureClass.DIAGNOSTIC, m.group(0)

    return FailureClass.UNKNOWN, "unclassified"


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Exponential back-off with jitter for retryable failures."""
    import random
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add up to 25% jitter to prevent thundering herd
    jitter = delay * 0.25 * random.random()
    return delay + jitter


# Maximum retry attempts for retryable failures
MAX_RETRIES = 3


def should_retry(failure_class: FailureClass, attempt: int) -> bool:
    """Whether the agent should retry a tool call given its failure class and attempt number."""
    return failure_class == FailureClass.RETRYABLE and attempt < MAX_RETRIES


def format_failure_context(error_text: str, failure_class: FailureClass, tool_name: str = "") -> str:
    """Format the failure for injection into the LLM context.

    * RETRYABLE: short notice that the system will retry
    * FATAL: clear stop signal with reason
    * DIAGNOSTIC: full error for self-correction
    """
    prefix = f"[Tool: {tool_name}] " if tool_name else ""
    if failure_class == FailureClass.RETRYABLE:
        return f"{prefix}TRANSIENT ERROR (will retry): {error_text[:200]}"
    elif failure_class == FailureClass.FATAL:
        return f"{prefix}PERMANENT ERROR — do NOT retry this tool call: {error_text[:500]}"
    elif failure_class == FailureClass.DIAGNOSTIC:
        # Give the LLM the full error for self-correction, capped at reasonable size
        return f"{prefix}DIAGNOSTIC ERROR — analyze and fix:\n{error_text[:2000]}"
    else:
        return f"{prefix}ERROR: {error_text[:500]}"


def summarize_multi_op_outcomes(op_outcomes) -> str:
    """Aggregate a turn's per-call results into one explicit summary.

    The agent emits one tool call per id, so "delete A and B" becomes two
    calls. When one succeeds and one fails, the loop used to book the whole
    turn as a single undifferentiated failure and inject a generic
    diagnostic that named only the *last* error — the model never saw a
    clean "A deleted, B not found" picture and would drift onto stale
    context. This produces that picture.

    ``op_outcomes`` is a list of dicts ``{"tool": str, "ok": bool,
    "preview": Optional[str]}``. Returns "" when there is nothing worth
    aggregating (0–1 ops, or every op the same outcome with a single op),
    so single-call failures keep their existing terse diagnostic.
    """
    if not op_outcomes or len(op_outcomes) < 2:
        return ""
    ok_ops = [o for o in op_outcomes if o.get("ok")]
    failed_ops = [o for o in op_outcomes if not o.get("ok")]
    # Only worth a summary when the turn was MIXED — a uniform all-fail turn
    # is served fine by the normal diagnostic.
    if not ok_ops or not failed_ops:
        return ""
    succeeded = "; ".join(o.get("tool", "?") for o in ok_ops)
    failed = "; ".join(
        f"{o.get('tool', '?')}: {(o.get('preview') or 'failed').strip()[:140]}"
        for o in failed_ops
    )
    return (
        f"MULTI-STEP OUTCOME — {len(ok_ops)} of {len(op_outcomes)} call(s) "
        f"SUCCEEDED, {len(failed_ops)} FAILED.\n"
        f"  SUCCEEDED: {succeeded}\n"
        f"  FAILED: {failed}\n"
        "The successful operations DID take effect — do NOT retry them or "
        "report them as failed. This live outcome is AUTHORITATIVE over any "
        "prior context, memory, or system-state hint. Report exactly what "
        "succeeded and what failed, then stop.\n\n"
    )


# Per-tool fallback hints. Maps a (tool_name, error_pattern_substring) →
# concrete remediation hint that the agent loop can inject into context as
# a follow-up nudge after a failure. The mapping is intentionally tiny and
# specific — broad hints for everything are noise.
_FALLBACK_HINTS = {
    "execute": [
        ("ModuleNotFoundError", "Install the missing Python module via execute(command='pip install <pkg>') first, or write the script in a way that doesn't depend on it."),
        ("ImportError", "Check the module name spelling. If it's a third-party package, install it via execute(command='pip install <pkg>')."),
        ("FORBIDDEN IMPORT", "You tried to import a Native JSON Tool as a Python module. Stop writing Python and call the JSON tool directly."),
        ("PermissionError", "The sandbox blocked this operation. Use file_system instead of raw OS calls, and avoid touching paths outside /workspace."),
        ("Syntax Error", "Re-read the script you submitted; the parser rejected it. Most common cause is unbalanced quotes/brackets or a stray markdown fence."),
        ("Kernel Timeout", "The Jupyter kernel exceeded 5 minutes. Split the work into smaller chunks or drop stateful=True for a fresh process."),
    ],
    "file_system": [
        ("not found", "The path doesn't exist. Run file_system(operation='list_files') to see what IS in the sandbox before re-trying."),
        ("MANDATORY", "You omitted a required parameter. Re-read the tool schema and re-issue the call with the missing field."),
        ("binary file", "This file is binary. Use vision_analysis (for images) or download/inspect via execute() instead of read."),
        ("too large", "The file exceeds the read limit. Use operation='read_chunked' with page=1, or operation='search' to find the specific line you need."),
    ],
    "web_search": [
        ("CAPTCHA", "DuckDuckGo is rate-limiting you. Wait, then retry with a SHORTER, keyword-focused query (no full sentences)."),
        ("ZERO results", "Your query was too narrow. Strip dates/version numbers, or convert to a question form (e.g. 'how to ...')."),
    ],
    "deep_research": [
        ("search phase failed", "Try web_search with a keyword-focused version of the query first; deep_research is for synthesising across many pages, not as a first attempt."),
    ],
    "postgres_admin": [
        ("connection_string is required", "No DB URI is configured. Either ask the user for one or skip this tool — do not retry without configuration."),
        ("statement_timeout", "The query exceeded the timeout. Add a `LIMIT` clause, push filters into the WHERE, or run EXPLAIN ANALYZE first to identify the slow part."),
    ],
    "delegate_to_swarm": [
        ("not configured", "The swarm cluster isn't set up. Process the task synchronously in your main context — do not retry delegate_to_swarm."),
        ("0 of", "No swarm node could route the task. Process synchronously."),
    ],
    "vision_analysis": [
        ("not found", "The image path doesn't exist in the sandbox. Use file_system(operation='list_files') to verify the filename first."),
    ],
    # `system` is the synthetic tool name used when the XML/JSON tool-call
    # parser rejects the model's output. The most common root cause is a
    # `<parameter name="content">` body that itself contains literal
    # `</parameter>`, unescaped angle brackets, or embedded JSON — the
    # regex-based parser truncates early. CDATA wrapping is the cleanest
    # fix; heredoc-via-execute is the bulletproof fallback.
    "system": [
        ("invalid or contained broken JSON", "Wrap the offending parameter body in `<![CDATA[ ... ]]>` so the parser tolerates literal `</parameter>`, `<`, `>`, JSON, and quotes. If the issue persists, write the file via `execute(command=\"cat > path <<'EOF'\\n...\\nEOF\")` instead of file_system.write."),
        ("ESCAPE HATCH", "You are in a parse-error loop. Switch tool-call shape — use CDATA wrapping, a heredoc via `execute`, or split the write into multiple smaller `replace` operations."),
    ],
}


def get_fallback_hint(tool_name: str, error_text: str) -> str | None:
    """Return a concrete remediation hint for a known (tool, error) pair, or None.

    The agent loop calls this after a tool failure to enrich the context
    injection with actionable advice. Returns the FIRST matching hint
    string, or None if neither the tool nor the error pattern matches.

    Pattern match is case-insensitive substring against `error_text`.
    """
    if not tool_name or not error_text or not isinstance(error_text, str):
        return None
    hints = _FALLBACK_HINTS.get(tool_name)
    if not hints:
        return None
    et_lower = error_text.lower()
    for needle, hint in hints:
        if needle.lower() in et_lower:
            return hint
    return None
