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
    # HTTP auth failures. Match the reason phrase, or a status code with an
    # explicit http/status context — NOT a bare "401"/"403", which as an
    # unanchored substring matched byte counts ("40301"), line numbers
    # ("line 403"), and ids, misclassifying self-correctable diagnostics as
    # PERMANENT "do not retry" errors. (The 502/503/504 pattern above is
    # already word-boundaried; this brings 401/403 in line.)
    re.compile(r"\b(?:401\s+unauthorized|403\s+forbidden)\b", re.IGNORECASE),
    re.compile(r"\b(?:https?|status)\b.{0,8}?\b(?:401|403)\b", re.IGNORECASE),
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


#: A failure whose cause is the CALL's arguments, not the tool. Recoverable by
#: re-issuing with a corrected argument, so it gets an instruction permitting
#: exactly that.
#:
#: Consulted for FATAL *and* UNKNOWN. Only `MANDATORY` is a FATAL pattern —
#: the other two classify UNKNOWN, so restricting the check to FATAL left two
#: thirds of this expression dead while its comment claimed they were "kept in
#: sync with the FATAL patterns". The live message it was missing is
#: `update_profile`'s "Error: 'key' is a required argument", which is exactly
#: where the knowledge_base redirect sends a model.
_ARGUMENT_ERROR = re.compile(
    r"MANDATORY|is a required argument|required parameter"
    r"|does not accept one of the arguments", re.IGNORECASE)


#: The prefixes a tool result uses to say "this failed". ONE HOME — the turn
#: loop, the foresight seeder and the tests all read it from here.
#:
#: `"Error"` is matched as a WORD, not as `"Error:"`. Nine live tool returns
#: say "Error " with a space — `tool_remember`'s "Error storing memory: …",
#: `file_system`'s "Error 404 - Failed to download …", and seven more — and a
#: colon-only gate booked every one of them as a clean SUCCESS. Measured
#: consequence: a failed `insert_fact` scored zero strikes AND had its hash
#: written to `executed_idempotent`, so the model's identical retry was
#: refused with "the intended state is already applied" and the fact was
#: never stored. A failing `file_system` download went further — booked ok,
#: it took the world-changed branch and CLEARED every recorded failure,
#: disarming the pre-flight repeat guard it should have been feeding.
#: `Error` as a WORD — `\b` so "Errors were avoided" is prose, not a failure.
_FAILURE_PREFIX_RE = re.compile(
    # `CRITICAL ERROR:` added 2026-08-29: `web_search` and `deep_research`
    # emit it (search.py 737/842/863) and NOTHING recognised it. Measured, a
    # search that never ran was a clean success to the loop and the
    # competence profile and a failure to foresight and the corpus — the two
    # halves disagreeing inside one iteration of one method. The §4DO audit
    # parsed 1,026 result heads and still missed it, which is why the pin is
    # at the producer now.
    r"\s*(?:SYSTEM ERROR|Critical Tool Error|CRITICAL ERROR|ERROR\b|Error\b)")


#: Heads a tool uses to REFUSE a call it never carried out. Distinct from a
#: failure: the tool ran, decided not to act, and CHANGED NOTHING.
#:
#: Measured, 63 live occurrences across 36 requests: `file_system`'s
#: "SYSTEM INSTRUCTION: You used operation='replace' but forgot
#: 'replace_with'" and "REPLACE REJECTED (byte-identical)" matched no
#: failure predicate, so a mutating op that did nothing was CREDITED with
#: changing the world — clearing every recorded pre-flight failure, wiping
#: the loop-breaker's memory, DECREMENTING the strike count (so a run of
#: rejected replaces erases earlier strikes and the cap can never fire),
#: marking the file modified in the work log, and teaching the foresight
#: model that the call succeeds.
#:
#: Deliberately NOT folded into `result_is_failure`: the composed-skill and
#: acquired-skill success_rate readers exempt steering SYSTEM INSTRUCTIONs
#: on purpose. This is consumed only where the question is "did this call
#: change anything?".
#: ANCHORED. The second arm used to be `\s*[^\n]*\bREJECTED\b` — any first
#: line CONTAINING the word — and it was already firing live on a successful
#: `manage_projects` call whose 1,950-char task tree quoted an earlier
#: refusal verbatim, and on a `file_system` read of a log line reading
#: "conn REJECTED by peer". A tool's own refusal always LEADS with its
#: marker; a result that merely mentions one is data.
_REJECTION_RE = re.compile(
    r"\s*(?:SYSTEM INSTRUCTION|REJECTED|SYSTEM BLOCK|PARTIAL:"
    # ...and the one refusal head that does not lead with a marker word.
    # ANCHORED like the rest: a log line that merely mentions it is data.
    r"|REPLACE REJECTED"

    # ⚠ `Security Error` — the SANDBOX CONTAINMENT REFUSAL (§4DX round 2).
    #
    # `_get_safe_path` raises `ValueError("Security Error: …")` and TWELVE
    # `file_system` handlers catch it and return the bare message as a plain
    # str (read/write/replace/delete/copy/rename/list/search/inspect/
    # download×2/read_chunked). It matched neither regex: `_FAILURE_PREFIX_RE`
    # is anchored on `ERROR\b|Error\b` and this head starts with an "S".
    #
    # Measured consequence for `file_system(operation='write',
    # path='../evil.txt')`: the operation is mutating, so the loop took the
    # world-changed branch — `_failure_guard.note_world_changed()` cleared
    # every recorded pre-flight failure, `strikes.note_world_changed()`
    # wiped the loop-breaker's memory and decayed a strike, and the project
    # work_log recorded a file that was never written. A path-traversal
    # refusal — including the guard that stops an `rmtree` of the entire
    # workspace — was booked as a successful, world-changing write.
    #
    # `core/coding_executor.py` already hand-rolled
    # `head.startswith("security error")` for its own use: two
    # implementations of one decision, and the canonical one (this module,
    # whose docstring says "ONE HOME") did not have it.
    r"|Security Error\b"

    # ⚠ `Skipped:` / `NOOP:` / `Nothing recorded` / `Nothing to diff` WERE
    # ADDED HERE AND REVERTED THE SAME DAY. Recording the mistake because
    # the reasoning that put them in is seductive and will recur.
    #
    # The argument was: the tool wrote nothing, so it must not be credited
    # with a world change. True — and the wrong lever. Classifying them as
    # REJECTIONS also sets `may_record_as_applied = False`
    # (`outcome.py`: `not is_failure`), and `agent.py`'s idempotency ledger
    # is gated on exactly that:
    #
    #     if _is_idem_setter and _outcome.may_record_as_applied:
    #         executed_idempotent.add(a_hash)
    #
    # The three idempotent setters — `update_profile`, `learn_skill`,
    # `knowledge_base insert_fact` — are PRECISELY the three producers of
    # `NOOP:`. Their no-op reply means "the state you asked for is already
    # there", which is the successful end state. Marked as a rejection, the
    # DURABLE ledger was never written, the guard never armed, and the model
    # re-issued the identical call every turn: measured 3 dispatches, three
    # strikes, and the loop-breaker firing with a steer that tells the model
    # the message "names exactly what is wrong with the arguments" and to
    # re-issue the same tool — the exact 9×-in-a-row `update_profile` loop
    # the ledger was built to stop.
    #
    # `Nothing to diff` was worse still: `tool_workspace` is documented
    # read-only, and that string is its CORRECT answer for an empty
    # watchlist — which is the live state of this box. It became a failure
    # banner, a strike, and a `turn outcome: failed` label feeding the bench
    # flywheel and the competence prior. Label noise manufactured from a
    # correct read.
    #
    # A no-op is a SUCCESS that changed nothing. If `world_changed` is ever
    # wrong for these, fix it where world-change is decided — not by
    # relabelling the result a refusal.

    # Argument refusals that never reach an action.
    r"|Unknown action\b)")


def result_is_rejection(text) -> bool:
    """Did the tool refuse to act, leaving the world untouched?

    A rejection is not a success and must never be credited as a world
    change, recorded as an applied idempotent write, or allowed to decay a
    strike. It is also not necessarily a `result_is_failure` — several are
    deliberately phrased as instructions.
    """
    return bool(_REJECTION_RE.match(str(text or "")))


def result_is_failure(text) -> bool:
    """Does this tool result read as a failure to the turn loop?

    Every caller must use THIS, not a copy of the prefix tuple. The tuple
    was inlined at four sites and copied again into a test, so the test
    agreed with the loop about a contract neither of them shared with the
    nine tools that actually produce these strings.
    """
    return bool(_FAILURE_PREFIX_RE.match(str(text or "")))


def is_argument_error(text) -> bool:
    """True when a failure is about the CALL's arguments rather than the
    tool — recoverable by re-issuing corrected. One home: `format_failure_context`
    picks its wording from this, and the turn loop sizes the mixed-turn
    preview from it, so the two cannot disagree about what an argument
    error is."""
    return bool(_ARGUMENT_ERROR.search(str(text or "")))


def format_failure_context(error_text: str, failure_class: FailureClass, tool_name: str = "") -> str:
    """Format the failure for injection into the LLM context.

    * RETRYABLE: short notice that the system will retry
    * FATAL: clear stop signal with reason
    * DIAGNOSTIC: full error for self-correction
    """
    prefix = f"[Tool: {tool_name}] " if tool_name else ""
    # Argument errors first, for every class that is not a transient retry:
    # the recovery is a corrected re-issue whether the classifier called it
    # FATAL or could not place it at all.
    # RETRYABLE keeps its "will retry" notice (the system retries it, and
    # telling the model to re-issue would duplicate the call); DIAGNOSTIC
    # keeps its 2000-char "analyze and fix" budget, which a traceback
    # mentioning a required parameter would otherwise trade for a 500-char
    # nudge.
    if failure_class not in (FailureClass.RETRYABLE, FailureClass.DIAGNOSTIC) \
            and is_argument_error(error_text):
        return (f"{prefix}ARGUMENT ERROR — this call cannot succeed as "
                f"issued. Do NOT repeat it unchanged; re-issue the SAME "
                f"tool with the argument the error names: {error_text[:500]}")
    if failure_class == FailureClass.RETRYABLE:
        return f"{prefix}TRANSIENT ERROR (will retry): {error_text[:200]}"
    elif failure_class == FailureClass.FATAL:
        # An argument error is fatal for the call AS ISSUED, but the recovery
        # is to re-issue it corrected — the opposite of "do NOT retry this
        # tool call". That advice used to be accidentally right, because
        # errors of this shape named parameters the tool would not accept, so
        # no re-issue could work (the 2026-08-28 knowledge_base forget loop).
        # Now that such an error names an accepted parameter and carries a
        # worked call, a blanket stop signal is the only thing standing
        # between the model and the fix. Handled above; the class stays FATAL
        # so it never spends the transient retry budget.
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
    # A STILL-RUNNING call is neither. Listing it under SUCCEEDED told the
    # model "this DID take effect — do NOT retry it" about work that had not
    # happened yet, in the one message that calls itself AUTHORITATIVE.
    running_ops = [o for o in op_outcomes if o.get("unresolved")]
    op_outcomes = [o for o in op_outcomes if not o.get("unresolved")]
    ok_ops = [o for o in op_outcomes if o.get("ok")]
    failed_ops = [o for o in op_outcomes if not o.get("ok")]
    # A turn carrying in-flight work is worth summarising even when the
    # RESOLVED calls are uniform — "do NOT re-dispatch these" is the whole
    # point, and filtering them out before the mixed-outcome gate below made
    # it unreachable in 4 of 5 shapes.
    if running_ops and not (ok_ops and failed_ops):
        _r = "; ".join(str(o.get("tool", "?")) for o in running_ops)
        _done = "; ".join(str(o.get("tool", "?")) for o in op_outcomes) or "none"
        return (f"MULTI-STEP OUTCOME — {len(running_ops)} call(s) STILL "
                f"RUNNING, {len(op_outcomes)} finished.\n"
                f"  STILL RUNNING (no verdict yet — do NOT re-dispatch, and "
                f"do NOT report these as done): {_r}\n"
                f"  FINISHED: {_done}\n\n")
    # Only worth a summary when the turn was MIXED — a uniform all-fail turn
    # is served fine by the normal diagnostic.
    if not ok_ops or not failed_ops:
        return ""
    succeeded = "; ".join(o.get("tool", "?") for o in ok_ops)

    def _one(o):
        text = (o.get("preview") or "failed").strip()
        # ONE budget decision, here. The turn loop sizes `preview` with the
        # same predicate, and re-cutting at a flat 140 made that widening
        # dead code: an argument error's worked call sits at the END of the
        # message, so the model still read "…filename to erase. Worked " and
        # nothing more — the exact symptom the widening was for.
        return f"{o.get('tool', '?')}: {text[:300 if is_argument_error(text) else 140]}"

    failed = "; ".join(_one(o) for o in failed_ops)
    running = ""
    if running_ops:
        running = ("  STILL RUNNING (no verdict yet — do NOT re-dispatch, "
                   "and do NOT report these as done): "
                   + "; ".join(str(o.get("tool", "?")) for o in running_ops)
                   + "\n")
    return (
        f"MULTI-STEP OUTCOME — {len(ok_ops)} of {len(op_outcomes)} call(s) "
        f"SUCCEEDED, {len(failed_ops)} FAILED.\n"
        f"  SUCCEEDED: {succeeded}\n"
        f"  FAILED: {failed}\n"
        f"{running}"
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
        ("command not found", "The sandbox container is minimal — that utility isn't installed. Use built-in equivalents: `file <f>` → `head -c 16 <f> | od -An -tx1` (or python3 with open(...,'rb')); `xxd` → `od -A x -t x1z`; check availability first with `command -v <tool>`."),
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
