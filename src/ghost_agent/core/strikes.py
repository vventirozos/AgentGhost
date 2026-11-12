"""Per-request strike / loop-detection accounting for the turn loop.

The reasoning loop in ``core.agent`` historically juggled the loop-safety
state as a handful of bare locals — two signature dicts, a "persistent
failure seen" flag, a warned-signatures set, and a consecutive-clean-
success counter — mutated across several hundred lines. This module pulls
that cohesive bundle into one ``StrikeLedger`` object plus the pure
signature helpers, so the loop holds a single ledger instead of five
interacting locals.

Scope is deliberately narrow: this owns the *signature* tracking and the
decay-freeze decision. The raw strike COUNTERS
(``execution_failure_count`` / ``transient_failure_count``) stay in the
loop because they are cross-cutting (strike caps, mid-loop caps, System-3
pivot triggers) — folding them in too would touch dozens of sites for no
clarity gain. The ledger only answers: "has this exact failure looped?",
"is this exact action making no progress?", and "should success-decay be
frozen right now?".

The three module-level functions are kept as standalone pure functions
(not just methods) because existing tests import them directly; the ledger
delegates to them.
"""

from __future__ import annotations

import hashlib
import re


def note_repeated_failure(sigs: dict, fname, error, threshold: int = 3):
    """Record one structural tool failure and report whether the SAME
    failure has now recurred enough times to be a persistent loop.

    The signature is ``tool | whitespace-normalised error head`` — stable
    across byte-identical repeats (e.g. the same "'x' not found." every
    turn) but distinct for different tools/errors. Pure aside from mutating
    the caller-owned ``sigs`` dict, so it is unit-testable. Returns
    ``(signature, count, is_persistent)``. Used to freeze the strike-decay
    once a loop is detected — otherwise an interleaved success cancels the
    strike and the cap never fires."""
    sig = (
        f"{fname or '?'}|" + re.sub(r"\s+", " ", str(error or "")[:160].lower()).strip()
    )
    count = sigs.get(sig, 0) + 1
    sigs[sig] = count
    return sig, count, count >= threshold


#: §4IB — the execute-loop class the breaker could not see (req 21b295ef,
#: 2026-09-17). Twenty turns of `Grid('reduced_gg, npoints=127')` variants,
#: every run printing the same `cannot build grid without 'type'` — and
#: every run EXIT 0, because the model's own try/except printed the error.
#: `note_failure` never saw a failure; `note_action` skips `execute` (it is
#: a mutating tool); and even keyed on the result, the output carried a
#: catalog dump of MEMORY ADDRESSES that differed on every run, so no two
#: results would ever have shared a fingerprint. The signal is the ERROR
#: LINE with the volatile tokens normalised out, keyed on the command HEAD
#: (the program that ran), not the heredoc that changed a little each turn.
_VOLATILE_RES = (
    (re.compile(r"0x[0-9a-fA-F]{4,}"), "0xADDR"),
    (re.compile(r"\b[0-9a-f]{16,}\b"), "HEX"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "TIME"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "DATE"),
    (re.compile(r"\b(?:pid|PID)[ =:]+\d+\b"), "pid=N"),
    (re.compile(r"/tmp/[\w.\-]+"), "/tmp/X"),
    # §4KT R3: a URL is the thing that was tried, not the error — one
    # `net::ERR_HTTP2_PROTOCOL_ERROR` keyed as eleven, one per page.
    # (No quotes in the class: the quote rules below still collapse a
    # quoted literal that follows a quoted URL. `file://` too — one
    # `net::ERR_FILE_NOT_FOUND` was a key per file.)
    (re.compile(r"(?:https?|file)://[^\s'\"]+"), "URL"),
    # The model's own `print(f"ERR: {spec!r} -> {e}")` puts the THING IT
    # TRIED inside quotes — the one part that changes per attempt while the
    # error stays the same. Quoted literals collapse.
    (re.compile(r"'[^'\n]{0,200}'"), "'…'"),
    (re.compile(r'"[^"\n]{0,200}"'), '"…"'),
)
_ERROR_LINE_RE = re.compile(
    # An exception NAME is CamelCase (`ValueError`, `TimeoutError`) — matched
    # case-sensitively (§4KT: under IGNORECASE `[A-Za-z]*error` matched
    # "terror" in a news snippet); a bare `error` / `exception` word in any
    # case (`error:`, `[error]`, `ERROR`) is still one.
    r"(?:Traceback \(most recent call last\)|(?-i:\b[A-Za-z]*(?:Error|Exception)\b)|"
    r"\b(?:error|exception)\b|"
    r"\bcannot\b|\bfailed\b|\bfailure\b|No module named|No such file|not found|"
    r"\bKilled\b|\bERR[:\s]|\bexit(?:\s*code)?\s*[=:]\s*[1-9]|EXIT CODE:\s*[1-9])",
    re.IGNORECASE,
)
#: Tools whose result is a PROGRAM'S OUTPUT — the one kind of text whose
#: failure words mean the tool call failed (a script that caught its own
#: error and printed it, §4IB). Every other tool returns CONTENT: page text,
#: a file's bytes, JSON records, search snippets, prose — where "not found",
#: "cannot" and "terror" are the subject matter. §4KT (2026-09-30), measured
#: on 9,119 corpus results: 96/1,750 successful searches, 324/761 loaded
#: pages and 450/1,399 file reads yielded an "error line".
PROGRAM_OUTPUT_TOOLS = frozenset({"execute"})
#: §4SX review 11-05 (R-m5): `jobs` / `manage_services` are now aliases whose results are `run` results (a
#: listing's "failed"/"lost" rows are its content) — they are read like `run`; a job's OWN result (the jobs
#: supervisor's collect, reached by `run collect name=<job id>`) is program output under any of the three names
_RUN_LIKE_TOOLS = frozenset({"run", "jobs", "manage_services"})
_JOB_COLLECT_HEAD_RE = re.compile(r"^(?:--- job-[0-9a-f]{6,16}(?: \[[A-Z_]+\])?: |Collected \d+ finished job)")
#: §4SX B: only a `lab` EXEC result is program output (its header names the
#: lab/member and the exit); `logs`, `status`, `http`, `spec` return CONTENT —
#: a Postgres log's "ERROR: relation x does not exist" is its subject (review
#: MAJOR 2, the §4KT class)
_LAB_EXEC_HEAD_RE = re.compile(r"^\[lab_[0-9a-f]{8}/[a-z][a-z0-9-]{0,30} (?:exit=|TIMED OUT|no exit code|CANCELLED)")
#: §4SX D5: only a `run collect` result is program output (its head names the run and the exit, then the
#: EXIT CODE line); `logs` / `status` / `url` return CONTENT — a server log's "ERROR" is its subject
_RUN_COLLECT_HEAD_RE = re.compile(r"^\[run [a-z][a-z0-9-]{0,30} (?!logs\b|app\b)[^\]\n]*\]\s*\nEXIT CODE: ")


def is_program_output(tool, output="") -> bool:
    """Is this result a program's output (scanned for failure words)?"""
    if tool is None:
        return True
    t = str(tool)
    if t in PROGRAM_OUTPUT_TOOLS:
        return True
    if t in _RUN_LIKE_TOOLS:
        _o = "\n".join(ln for ln in str(output or "").splitlines() if not ln.strip().startswith(_FAILURE_BANNER))
        _o = re.sub(r"^RUN (?:ERROR|UNRESOLVED) — ", "", _o.strip())   # the inline exit head over a job collect
        return bool(_RUN_COLLECT_HEAD_RE.match(_o) or _JOB_COLLECT_HEAD_RE.match(_o))
    if t != "lab":
        return False
    for ln in str(output or "").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(_FAILURE_BANNER):
            continue
        return bool(_LAB_EXEC_HEAD_RE.match(ln))
    return False
#: The turn loop's own mark on a result its failure gate rejected
#: (`[FAILURE BANNER] <label>\n<result>`): the LABEL is the loop's reading
#: of the head, the failure itself is in the result under it.
_FAILURE_BANNER = "[FAILURE BANNER]"
_BROWSER_FAILURE_STATUS_RE = re.compile(r"^STATUS:\s*(?:ERROR|BLOCKED)\b")


#: The formatter's per-action failure line (`tools/browser.py`: `[{idx}]
#: {status} {act}: {error}`); the runner's implicit-goto abort is index -1.
_BROWSER_ACTION_ERR_RE = re.compile(r"^\[-?\d+\]\s+ERR\b")


def _content_failure_head(lines: list, declared: bool = False) -> str:
    """The line a CONTENT tool's failure names, read from the tool's own
    structure — the first choice for a declared failure and the only one
    for an undeclared result: the failure head the tools write
    (`tool_failure.result_is_failure` — one vocabulary, shared with the
    turn loop's failure gate); a browser result whose header says
    ERROR/BLOCKED (for ERROR, the message under it); or a browser interact
    result's first failed action when the result is a DECLARED failure
    (`[n] ERR …` — its header says OK, and the `ACTIONS: 2 OK, 1 error`
    summary and a `NOTE:` nudge come before the per-action lines; §4KT R3
    review). Only for a declared failure: a failed action is reported by
    the tool's status, never inferred from an OK result's text."""
    if not lines:
        return ""
    head = lines[0]
    if head.startswith("--- BROWSER RESULT ---"):
        if len(lines) > 1 and re.match(r"STATUS:\s*(?:ERROR \(every action failed\)|PARTIAL\b)", lines[1]):
            # §4LN: an interact's header names no error — its first failed
            # action does (else every such failure keyed as "OP: interact")
            for s in lines:
                if _BROWSER_ACTION_ERR_RE.match(s):
                    return s[:240]
        if len(lines) > 1 and _BROWSER_FAILURE_STATUS_RE.match(lines[1]):
            # `STATUS: ERROR` is followed by the message itself
            # (`_browser_error`); `STATUS: BLOCKED (HTTP 403 …)` names it.
            if lines[1].startswith("STATUS: ERROR") and len(lines) > 2:
                return lines[2][:240]
            return lines[1][:240]
        if declared:
            for s in lines:
                if _BROWSER_ACTION_ERR_RE.match(s):
                    return s[:240]
        return ""
    from ..tools.tool_failure import result_is_failure
    return head[:240] if result_is_failure(head) else ""


def _first_failure_word_line(lines: list) -> str:
    """The FIRST line naming a failure — for a content tool's declared
    failure, the tool's own statement comes first and its hint prose,
    console bullets and file snippets after it (§4KT R2 review: the LAST
    such line named the hint)."""
    for s in lines:
        if s.startswith("Traceback (most recent call last)"):
            continue        # a header: the exception line follows it
        if _ERROR_LINE_RE.search(s):
            return s[:240]
    return ""


def _last_failure_word_line(lines: list) -> str:
    """The LAST line naming a failure (a traceback opens with its header and
    ends with the exception that matters; a probe script prints its verdict
    after its attempts); "" when none does."""
    found = ""
    for s in lines:
        if _ERROR_LINE_RE.search(s):
            found = s[:240]
    return found
#: The target suffix the dispatch pipeline keys the execute same-error
#: class under (`"<head> (same error)"`); `note_world_changed` keeps those.
SAME_ERROR_TARGET_SUFFIX = "(same error)"


def is_same_error_signature(sig: str) -> bool:
    """True for a `note_action` signature (`tool|target|fp`) of the execute
    same-error class."""
    parts = str(sig or "").split("|", 2)
    return len(parts) == 3 and parts[1].endswith(SAME_ERROR_TARGET_SUFFIX)


#: Soft steer after this many same-error runs of one command head …
EXECUTE_SAME_ERROR_STEER = 3
#: … and from this many the class is past its steer: §4UI made it advice only
#: (it forced a report; it also keeps the generic abort from firing on it).
EXECUTE_SAME_ERROR_HARD_STOP = 5


def normalise_volatile(text: str, *, keep_quoted: bool = False) -> str:
    """Collapse the tokens that change on every run of the same failure —
    memory addresses, long hex ids, clock times, dates, pids, temp names —
    so two prints of one error compare equal. Digits that carry meaning
    (a count, a line number) are kept. ``keep_quoted`` leaves quoted
    literals alone (§4PH: `No module named 'foo'` and `'bar'` are two
    errors, not one)."""
    out = str(text or "")
    for rx, rep in _VOLATILE_RES:
        if keep_quoted and rep in ("'…'", '"…"'):
            continue
        out = rx.sub(rep, out)
    return out


def error_line(output: str, *, tool: "str | None" = None) -> str:
    """The line of a tool result that names its failure, or "" when the
    result reports none.

    * A DECLARED failure (a `ToolOutcome` whose status is not ok/unresolved
      — a refusal, a failed or partial run) is an error whatever its prose
      says. Read the status, never only the text (the outcome-consumers R3
      rule). The turn loop's `[FAILURE BANNER] <label>` mark is skipped:
      it is the loop's reading, not the tool's line (§4KT R1 review: the
      digest printed the marker itself). The line named: for program
      output the LAST failure-word line; for a content tool the declared
      head (a browser status/message, a failure prefix) or else the FIRST
      failure-word line — the tool's own statement, which precedes its
      hint prose, console bullets and file snippets (R2 review) — and,
      failing both, the first line of the body.
    * A PROGRAM-OUTPUT tool (`PROGRAM_OUTPUT_TOOLS`, or an unnamed caller)
      is scanned for failure words, and the LAST such line wins. This is
      how a script that caught its own error and printed it under exit 0
      is still counted (§4IB).
    * Any other tool returns CONTENT, and its failure is only what the
      tool itself declares — a status, the loop's banner, or a failure
      head (`tool_failure.result_is_failure`, a browser `STATUS:
      ERROR/BLOCKED`). The words inside a page, a file, a record or a
      snippet are its subject, not a failure (§4KT: "the user's browser
      cannot open it" in a successful start report, "terror" in a news
      snippet and a page's console `[error]` lines were all "errors hit").
    """
    lines = [ln.strip() for ln in str(output or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    bannered = lines[0].startswith(_FAILURE_BANNER)
    body = lines[1:] if bannered else lines
    _st = getattr(output, "status", None)
    _sv = getattr(_st, "value", _st)
    declared = _sv is not None and str(_sv) not in ("ok", "unresolved")
    content = not is_program_output(tool, output)
    if declared or bannered:
        if content:
            named = (_content_failure_head(body, declared=True)
                     or _first_failure_word_line(body))
        else:
            named = _last_failure_word_line(body)
        return (named or (body[0] if body else lines[0]))[:240]
    if content:
        return _content_failure_head(lines)
    return _last_failure_word_line(lines)


#: The first exception NAME on an error line (`RuntimeError`, `SpecError`,
#: `eckit.SpecError`, `ModuleNotFoundError`). §4IE: the model's probe
#: harnesses label each attempt UNQUOTED — `dict npts=31: ERR RuntimeError:
#: SpecError: [pl]` / `ERR  dict nxacc=16: RuntimeError: SpecError: [pl]` —
#: so 20 runs of one dead end fingerprinted as 20 different errors and the
#: same-error breaker never fired (probe ifs18371…, 36 executes, no steer).
#: The error IS the exception and what follows it; the label is the thing
#: that was tried.
_EXC_TOKEN_RE = re.compile(r"\b[A-Za-z_][\w.]*(?:Error|Exception)\b")
#: …and, only when no CLASS is named, a bare `Error` / `Exception` label
#: (Playwright's own class is `Error`: `[0] ERR goto: Error: Page.goto: …`;
#: §4KT R4). A bare label BEFORE a class (`Error item_16: ValueError: …`,
#: the §4IE harness shape with another word) must not win over it — R5.
_BARE_EXC_TOKEN_RE = re.compile(r"\b(?:Error|Exception)\b")


def exception_signature(line: str) -> str:
    """`line` from its first exception name onward; the whole line when it
    names none (a `No module named x` / `not found` line has no label to
    strip and stays as it is)."""
    text = str(line or "")
    m = _EXC_TOKEN_RE.search(text) or _BARE_EXC_TOKEN_RE.search(text)
    if not m:
        return text
    sig = text[m.start():]
    # `initial navigation failed (Error): Page.goto: …` — the runner's
    # parenthesised class is the same class as `Error: Page.goto: …`.
    if sig[m.end() - m.start():].startswith(")"):
        sig = sig[:m.end() - m.start()] + sig[m.end() - m.start() + 1:]
    return sig


#: §4PH (R1-2): an error whose quoted literal IS the error — a missing module,
#: a missing name, an unknown option — keeps it; two different modules are two
#: dead ends, not one (the quote rule collapsed them and steered falsely).
_DISTINCT_LITERAL_RE = re.compile(
    r"No module named|cannot import name|has no attribute|is not defined"
    r"|unknown option|unrecognized arguments|command not found", re.I)
#: …and the NUMBERS in an error line are the run's values, not its identity:
#: `assert 369.5 == 372.0` that reads 371.2 after the next rewrite is the same
#: failing assertion; a traceback line number moves with every edit. (Line
#: numbers are kept by `normalise_volatile` for the strike LEDGER's display;
#: the fingerprint drops them.)
_COUNT_LINE_RE = re.compile(r"\b\d+ (?:failed|passed|errors?|skipped|xfailed|xpassed|warnings?|tests? (?:ran|passed|failed))\b", re.I)
_NUMERIC_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[a-zA-Z%]{1,2})?(?![\w])")   # 12, 369.5, 0.3s, 2GB


def error_line_fingerprint(output: str, *, tool: "str | None" = None) -> str:
    """Fingerprint of the result's error line under `normalise_volatile`
    and `exception_signature`, with numeric literals collapsed; "" when the
    result has no error line. This — not the whole output — is what an
    `execute` run is counted under."""
    line = error_line(output, tool=tool)
    if not line:
        return ""
    _keep = bool(_DISTINCT_LITERAL_RE.search(line))
    norm = re.sub(r"\s+", " ", exception_signature(
        normalise_volatile(line, keep_quoted=_keep))).strip().lower()
    # …except on a test-runner summary (`9 failed, 3 passed`): there the
    # numbers ARE the progress — collapsing them made a converging fix loop
    # read as the same error five times and forced a blocker report
    if not _COUNT_LINE_RE.search(norm):
        norm = _NUMERIC_RE.sub("n", norm)
    return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()[:12]


def action_result_fingerprint(result: str) -> str:
    """Whitespace-normalised fingerprint of a tool result.

    Used by ``note_repeated_action`` to decide whether two SUCCESSFUL
    calls produced "the same observation". Whitespace-only normalisation
    (digits intentionally kept) so a result whose content genuinely
    changed — a re-read that now returns edited bytes, a counter that
    advanced — looks different and does NOT count as a no-progress
    repeat. The FULL normalised string is hashed: a head-only slice made
    long outputs with a stable header (a polling loop whose progress
    only shows past the slice) collide, so genuine progress got steered
    and then hard-aborted as "no new info". Hashing is cheap; slicing
    was the expensive part."""
    norm = re.sub(r"\s+", " ", str(result or "")).strip().lower()
    return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()[:12]


#: Fingerprint substituted for a result whose OWN text says it found
#: nothing. Every re-worded search of the same document then collapses onto
#: ONE signature, so the second fruitless probe trips the no-progress
#: breaker at its usual threshold.
NO_ANSWER_FP = "no-answer"


def result_says_nothing_found(result: str) -> bool:
    """True when a tool's own output DECLARES that it found nothing.

    WHY (request e0f4a8bd, 2026-09-08). `note_repeated_action` keys on
    ``tool | target | result-fingerprint``, so it fires only when the same
    call returns the same bytes. The agent asked the same document ten
    questions, each re-worded, each returning DIFFERENT irrelevant passages
    — same tool, same target, same futility, ten different fingerprints,
    and the breaker never moved. Progress is not "the bytes changed"; ten
    different ways of finding nothing is ten times no progress.

    Read from the emitting tool's own constant (imported lazily — the tools
    package imports core), never from a phrase written out again here: two
    copies of a banner is how a check goes dark when one of them is
    reworded.
    """
    text = str(result or "")
    if not text:
        return False
    try:
        from ..tools.memory import KB_NO_ANSWER_MARKER
    except Exception:  # noqa: BLE001 — the breaker must survive an import
        return False
    # §4PE (K7): the tool writes the marker at the start of its own "⚠ " line;
    # a document containing the sentence is quoted ("> ") by the tool and
    # must not collapse every query onto the no-answer fingerprint
    return any(ln.lstrip().startswith(("⚠ " + KB_NO_ANSWER_MARKER, KB_NO_ANSWER_MARKER))
               for ln in str(text or "").splitlines())


def breaker_fingerprint(result: str) -> str:
    """The fingerprint the no-progress breaker should count this result
    under — the ONE place that decision is made.

    Two results are "the same observation" when their bytes match, EXCEPT
    when the tool itself reports that it found nothing: those all collapse
    onto :data:`NO_ANSWER_FP`, because ten re-wordings of a hopeless search
    are ten repeats of one failure, not ten observations (e0f4a8bd)."""
    if result_says_nothing_found(result):
        return NO_ANSWER_FP
    return action_result_fingerprint(result)


def note_repeated_action(sigs: dict, fname, target, result_fp, threshold: int = 3):
    """Companion to ``note_repeated_failure`` for the INVERSE pathology:
    a turn loop where every tool call SUCCEEDS but the agent keeps taking
    the same action against the same target and getting the same result —
    the ungrounded-verification loop (double-click the icon → screenshot →
    "no change" → repeat). ``note_repeated_failure`` can't see this
    because nothing errors and the strike counter never moves; the
    reasoning-similarity breaker misses it because the prose phrasing
    varies turn to turn.

    Keyed by ``tool | target | result-fingerprint`` — so it fires only on
    a genuine no-progress repeat (same action, same target, same
    observation), not on an action whose target or result is changing.
    Pure aside from mutating the caller-owned ``sigs`` dict. Returns
    ``(signature, count, tripped)`` where ``tripped`` is count >=
    threshold."""
    sig = f"{fname or '?'}|{target or ''}|{result_fp or ''}"
    count = sigs.get(sig, 0) + 1
    sigs[sig] = count
    return sig, count, count >= threshold


#: Tools that both READ and MUTATE through a single dispatch name, with the
#: action chosen by an argument (e.g. ``manage_tasks(action="list"`` vs
#: ``"create")``, ``file_system(operation="read"`` vs ``"write")``). A
#: no-progress loop on one of these is almost always the agent re-READING to
#: orient itself before performing the WRITE it was actually asked to do.
#:
#: The no-progress breaker's first-trip remedy is to set
#: ``force_final_response`` — which drops the toolset and routes the next turn
#: as text-only. For an ordinary re-observation loop (re-screenshot, re-read
#: the same file with nothing left to do) that is correct. For a read/write
#: tool it is destructive: it bars the pending mutation forever, so the agent
#: "finishes" having silently done nothing. Observed failure (on the macro
#: manager, removed with the macro engine in §4TA, 2026-11-05): a request to
#: reconfigure a composed skill looped on ``action="list"``, got
#: force-finalised at 3x, and the model's subsequent ``action="define"`` was
#: scrubbed by the final-generation stream guard — the change never landed.
#:
#: For these tools the breaker still STEERS the model off the wasteful re-read
#: but leaves tools available so the write can land. The
#: :data:`READWRITE_HARD_STOP` backstop fires if it genuinely keeps thrashing.
READWRITE_LOOP_TOOLS = frozenset({
    "manage_tasks",
    # `manage_projects` is the SAME read-then-write shape (2026-07-11): the
    # agent orients with action=status/list/task_next and mutates with
    # action=task_update/task_decompose/autoadvance — through the one tool.
    # Omitting it meant a no-progress READ loop force-stopped the turn into a
    # text-only final response, which BARRED the pending write forever. Seen
    # live twice in one session: (a) two identical action=status calls →
    # force-stop → the model emitted a tool call instead of prose → the stream
    # scrub consumed the entire response → the user got a fallback message
    # instead of their project status; (b) a task_update blocked twice by the
    # constraint gate → force-stop → the task could never be closed. Exactly
    # the read-then-write bug this set exists to prevent.
    "manage_projects",
    "file_system",
    # `knowledge_base` was already here for the read/write reason; e0f4a8bd
    # added a second one. The remedy for a FRUITLESS search is another call
    # to this same tool — `action='outline'` — so force-finalising the turn
    # would bar the one call that answers the question.
    "knowledge_base",
    "update_profile",
    # §4SX B: `lab status` polls a starting lab, then `exec` uses it
    "lab",
    # §4SX D5: `run status` / `logs` poll a starting server, then the work uses it
    "run",
})

#: Hard-stop threshold for a no-progress loop on a READWRITE_LOOP_TOOLS
#: tool. Two-tier contract, stated ONLY here (agent.py imports this
#: constant rather than restating the number): every tool gets the
#: corrective STEER at the general no-progress threshold (2), but the
#: exempt read/write tools skip the first-trip force-final so the pending
#: WRITE can still land — backstopped by a hard stop once the identical
#: (action, target, result) has repeated this many times. When the general
#: threshold moved 3→2, the enforcement site silently drifted down to >=3
#: with it; pinning the value here keeps the two tiers independent.
READWRITE_HARD_STOP = 5


# ⚠ RETRACTED 2026-09-07 (§4FH). A "repeated-mutation" breaker keyed on call
# IDENTITY (tool, action, target — result ignored) shipped here for a few
# hours and was pulled by the verification pass. Its premise was the
# ten-click request fb705dcf; the fact check showed every one of those
# clicks had file edits between it and the next — an edit→verify cycle, not
# an ungrounded repeat — and a corpus replay measured 28/1067 real requests
# false-steered and 11 false-stopped (reads, green test re-runs, id-keyed
# task updates collapsing to one key) against 3 rows of the target class.
# A guard with no measured true positive does not ship; the deletion is
# pinned in tests/test_4fh_mutation_breaker_retracted.py.


# §4JJ (2026-09-21): the search-yield STEER. Req e69cab30 ran 36 web
# searches without opening a single result and shipped nothing after 677 s;
# the no-progress breaker keys on an IDENTICAL query, and 36 different
# useless queries trip nothing until the turn cap. Measured on Aug+Sep (75
# requests with >=4 searches): every request that ended with nothing had a
# run of >=10 consecutive un-opened searches — but so did 26 that answered
# from snippets. So this is a steer with the tools KEPT, never a stop. It
# shipped behind the `search_yield_steer` randomized arm; the arm was
# concluded (§4LF) and since §4MO (operator) every requester is steered.
# One steer per request; the existing caps remain the stops.
SEARCH_YIELD_STEER = 10
#: Tools whose call OPENS a search result (reset the un-opened run).
SEARCH_OPEN_TOOLS = frozenset({"browser", "deep_research"})
#: §4OE-4: searches per request — a steer, then a hard cap (7 segments ran ≥15, max 39)
SEARCH_TOTAL_STEER = 20
SEARCH_TOTAL_CAP = 30


def is_readwrite_loop_exempt(fname) -> bool:
    """True if a no-progress READ loop on ``fname`` must NOT force a text-only
    final response, because the same tool is how the agent performs the
    pending WRITE. See :data:`READWRITE_LOOP_TOOLS`."""
    return fname in READWRITE_LOOP_TOOLS


class StrikeLedger:
    """Request-scoped loop-detection state for one ``handle_chat`` call.

    Bundles the signature dicts, the warned-signature set, the
    decay-freeze flag, and the consecutive-clean-success counter that the
    turn loop previously tracked as separate locals. Behaviour is
    identical to the inlined version — this is an encapsulation seam, not a
    policy change.
    """

    #: consecutive clean successes that unfreeze a detected failure loop.
    UNFREEZE_AFTER_CLEAN_SUCCESSES = 3

    def __init__(self) -> None:
        self.failure_sigs: dict = {}
        self.action_sigs: dict = {}
        self._batch_seen = None  # §4HP: set once begin_batch() is called
        self.persistent_failure_seen: bool = False
        self.persistent_warned_sigs: set = set()
        self.consecutive_clean_successes: int = 0
        # §4JJ: consecutive web searches with no result opened since; the
        # steer fires once per request when the run reaches SEARCH_YIELD_STEER.
        self.search_run: int = 0
        #: §4OE-4: every web_search this request (the run above resets on an open)
        self.search_total: int = 0
        self.search_total_steered: bool = False
        self.SEARCH_TOTAL_CAP = SEARCH_TOTAL_CAP
        self.search_yield_steered: bool = False
        # §4KL: consecutive tool batches that refused a member call and ran
        # nothing (no non-synthetic row) — a model that keeps trying after it
        # has read the refusal. A batch in which any call executed resets it.
        self.member_refused_batches: int = 0
        # §4MO: consecutive batches where every call was blocked before it
        # ran (any requester) — the report is forced at MEMBER_REFUSAL_REPORT_AT
        self.blocked_batches: int = 0
        self.project_refusal_batches: int = 0  # §4MO: 2nd batch with a refused project closes tools

    # -- failure path ------------------------------------------------------

    def note_search_yield(self, fname) -> int:
        """§4JJ: advance the un-opened search run for one dispatched call —
        a `web_search` extends it, a SEARCH_OPEN_TOOLS call resets it, any
        other tool leaves it. Returns the run after the call."""
        if fname == "web_search":
            self.search_run += 1
            self.search_total += 1
        elif fname in SEARCH_OPEN_TOOLS:
            self.search_run = 0
        return self.search_run

    def reset_clean_streak(self) -> None:
        """Break the consecutive-clean-success streak. Called on ANY failure
        (transient or structural) — the decay only unfreezes after a run of
        UNINTERRUPTED clean successes, so any failure in between resets it."""
        self.consecutive_clean_successes = 0

    def note_failure(self, fname, error, threshold: int = 3):
        """Record a structural failure and freeze decay once the same failure
        has recurred ``threshold`` times. (The clean-success streak is reset
        separately via ``reset_clean_streak`` so transient failures break it
        too.) Returns ``(signature, count, is_persistent, is_first_warning)``
        — ``is_first_warning`` is True exactly once per signature so the loop
        can emit the "stop retrying" steer a single time."""
        sig, count, is_persistent = note_repeated_failure(
            self.failure_sigs, fname, error, threshold
        )
        is_first_warning = False
        if is_persistent:
            self.persistent_failure_seen = True
            if sig not in self.persistent_warned_sigs:
                self.persistent_warned_sigs.add(sig)
                is_first_warning = True
        return sig, count, is_persistent, is_first_warning

    # -- success path ------------------------------------------------------

    def note_clean_success(self) -> bool:
        """Record one successful, non-mutating tool result. Returns True if
        this success unfroze a previously-detected failure loop (a genuine
        pivot produced ``UNFREEZE_AFTER_CLEAN_SUCCESSES`` clean results in a
        row; the fail→auto-list→fail oscillation never can). Signature
        counts are intentionally kept so the same failure re-freezes on its
        next occurrence."""
        self.consecutive_clean_successes += 1
        if (
            self.persistent_failure_seen
            and self.consecutive_clean_successes >= self.UNFREEZE_AFTER_CLEAN_SUCCESSES
        ):
            self.persistent_failure_seen = False
            # Reset the streak on unfreeze — otherwise the counter keeps
            # climbing, and a LATER re-freeze would be unfrozen by a SINGLE
            # clean success (counter already ≥ threshold) rather than requiring
            # a fresh run of clean successes.
            self.consecutive_clean_successes = 0
            return True
        return False

    # -- no-progress path --------------------------------------------------

    def begin_batch(self) -> None:
        """§4HP (2026-09-16, req 1234e131): one tool batch is ONE observation
        per (tool, target, result). The model listed the same URL twice in a
        four-call batch; both calls succeeded with identical text and
        ``note_action`` counted them as "repeated 2x with no new info" — a
        loop the model never took, since it had not SEEN the first result
        when it issued the second. The forced text-only conclusion then
        barred the plan's last step (write the report). Called at the top
        of each batch; a duplicate signature inside the batch is not noted
        again. Historically 5 of 63 2x trips were this shape."""
        self._batch_seen = set()

    def note_action(self, fname, target, result_fp, threshold: int = 3):
        """Record a successful action's (tool, target, result) fingerprint.
        Returns ``(signature, count, tripped)``. Inside a batch (see
        :meth:`begin_batch`) a repeated signature is reported with its
        current count and never trips."""
        key = (fname, str(target or ""), result_fp)
        seen = getattr(self, "_batch_seen", None)
        if seen is not None:
            if key in seen:
                sig = f"{fname or '?'}|{target or ''}|{result_fp or ''}"
                return sig, self.action_sigs.get(sig, 0), False
            seen.add(key)
        return note_repeated_action(
            self.action_sigs, fname, target, result_fp, threshold
        )

    def note_world_changed(self) -> None:
        """Forget every accumulated no-progress observation. Called when a
        file mutation SUCCEEDS: the workspace just changed, so a repeat of
        an earlier observation (re-navigate the served page, re-read the
        file) is now VERIFICATION of the change, not an ungrounded loop —
        even when the observation comes back byte-identical (the fix may
        target a different page state than the one being re-observed).

        Without this reset the breaker and the verifier fight each other:
        every fix-verify turn of the 2026-07-17 overnight session
        (requests 26/3B/72/1E/91) ended with the post-fix browser
        navigate killed by "repeated 2x with no new info", and in request
        3B the verifier gate then REFUTED the turn precisely because the
        evidence only showed the pre-fix page load — the auto-repair
        round's evidence-gathering navigate was itself cut by the
        breaker. Counts restart from zero; the abort backstop still
        protects against endless edit→observe cycles because each fresh
        observation run needs threshold repeats WITHOUT an intervening
        write to trip again, and the turn cap bounds the whole loop.

        §4IG: the execute SAME-ERROR class is exempt. Its count grows only
        when the SAME error line recurs, and the live pattern is precisely
        write-probe → run → same error → write-probe → run … (probe
        ifs19450…: 6 identical `SpecError: [pl]` runs, each preceded by a
        new probeN.py, never counted past 2). A rewrite that produced the
        same error is not progress; a rewrite that fixed it produces a
        different line and never increments the old signature."""
        self.action_sigs = {k: v for k, v in self.action_sigs.items()
                            if is_same_error_signature(k)}

    @property
    def decay_frozen(self) -> bool:
        """True while success-decay should be suppressed (a failure loop is
        active). The loop gates its ``execution_failure_count`` decrement on
        ``not decay_frozen``."""
        return self.persistent_failure_seen


# ---------------------------------------------------------------------------
# §4HZ / §4IB — the program a shell command RUNS. The same-error strike
# path keys `execute` repeats on the normalised ERROR LINE and annotates
# it with this head; it was the FIRST token, so `cd /workspace && python3
# x.py` filed under `cd` beside every other command that happened to start
# with a directory change. The head is the program after `cd … &&`,
# `timeout [opts] N`, `sudo`, `env`, `nohup`, …
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEAD_WRAPPERS = frozenset({"sudo", "env", "nohup", "nice", "time", "exec",
                            "command", "builtin", "stdbuf", "unbuffer"})
_HEAD_SEPARATORS = ("&&", "||", ";", "|")
_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
#: Wrapper options that consume the NEXT token as their value.
_WRAPPER_VALUE_OPTS = {
    "sudo": frozenset({"-u", "-g", "--user", "--group", "-C", "-h", "--host"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
}


def command_head(command: str) -> str:
    """The token that names what a shell command RUNS — skipping env
    assignments, `cd <dir> &&`/`;` prefixes, `set …;`, `timeout [-k N] N`,
    and bare wrappers (`sudo -n -u user`, `env`, `nohup`, `nice`, …).
    Returns "" when nothing runs (only assignments / a bare `cd`)."""
    toks = str(command or "").split()
    i, n = 0, len(toks)
    guard = 0
    while i < n and guard < 64:
        guard += 1
        tok = toks[i]
        if _ENV_ASSIGN_RE.match(tok):
            i += 1
            continue
        low = tok.lower().rstrip(";")
        if low in ("cd", "set", "export", "source", "."):
            # Skip to the next separator, then past it.
            i += 1
            while i < n and toks[i] not in _HEAD_SEPARATORS and not toks[i].endswith(";"):
                i += 1
            i += 1
            continue
        if tok in _HEAD_SEPARATORS:
            i += 1
            continue
        if low == "timeout":
            i += 1
            # options (`-k 5s`, `--signal=KILL`, `-s TERM`) then the duration
            while i < n and toks[i].startswith("-"):
                needs_val = toks[i] in ("-k", "-s", "--kill-after", "--signal")
                i += 2 if needs_val else 1
            if i < n and _DURATION_RE.match(toks[i]):
                i += 1
            continue
        if low in _HEAD_WRAPPERS:
            i += 1
            while i < n and toks[i].startswith("-"):
                # `sudo -u user`, `nice -n 10`, `env -u VAR` take a value;
                # `sudo -n`, `-E`… do not.
                needs_val = toks[i] in _WRAPPER_VALUE_OPTS.get(low, ())
                i += 2 if needs_val else 1
            continue
        return tok.rstrip(";")
    return ""
