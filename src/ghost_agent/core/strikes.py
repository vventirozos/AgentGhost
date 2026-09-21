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
    # The model's own `print(f"ERR: {spec!r} -> {e}")` puts the THING IT
    # TRIED inside quotes — the one part that changes per attempt while the
    # error stays the same. Quoted literals collapse.
    (re.compile(r"'[^'\n]{0,200}'"), "'…'"),
    (re.compile(r'"[^"\n]{0,200}"'), '"…"'),
)
_ERROR_LINE_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|\b[A-Za-z]*(?:Error|Exception)\b|"
    r"\bcannot\b|\bfailed\b|\bfailure\b|No module named|No such file|not found|"
    r"\bKilled\b|\bERR[:\s]|\bexit(?:\s*code)?\s*[=:]\s*[1-9]|EXIT CODE:\s*[1-9])",
    re.IGNORECASE,
)
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
#: … and a forced report (not an abort marker) after this many.
EXECUTE_SAME_ERROR_HARD_STOP = 5


def normalise_volatile(text: str) -> str:
    """Collapse the tokens that change on every run of the same failure —
    memory addresses, long hex ids, clock times, dates, pids, temp names —
    so two prints of one error compare equal. Digits that carry meaning
    (a count, a line number) are kept."""
    out = str(text or "")
    for rx, rep in _VOLATILE_RES:
        out = rx.sub(rep, out)
    return out


def error_line(output: str) -> str:
    """The LAST line of a tool result that names a failure, or "" when no
    line does (a clean result is not an error, however long). Last, not
    first: a traceback opens with its header and ends with the exception
    that matters; a probe script prints its verdict after its attempts."""
    found = ""
    lines = [ln.strip() for ln in str(output or "").splitlines() if ln.strip()]
    for s in lines:
        if _ERROR_LINE_RE.search(s):
            found = s[:240]
    if found:
        return found
    # A DECLARED failure (a `ToolOutcome` whose status is not ok/unresolved
    # — a refusal, a failed run) is an error whatever its prose says: its
    # first line is the failure it names. Read the status, never only the
    # text (the outcome-consumers R3 rule).
    _st = getattr(output, "status", None)
    _sv = getattr(_st, "value", _st)
    if _sv is not None and str(_sv) not in ("ok", "unresolved") and lines:
        return lines[0][:240]
    return ""


#: The first exception NAME on an error line (`RuntimeError`, `SpecError`,
#: `eckit.SpecError`, `ModuleNotFoundError`). §4IE: the model's probe
#: harnesses label each attempt UNQUOTED — `dict npts=31: ERR RuntimeError:
#: SpecError: [pl]` / `ERR  dict nxacc=16: RuntimeError: SpecError: [pl]` —
#: so 20 runs of one dead end fingerprinted as 20 different errors and the
#: same-error breaker never fired (probe ifs18371…, 36 executes, no steer).
#: The error IS the exception and what follows it; the label is the thing
#: that was tried.
_EXC_TOKEN_RE = re.compile(r"\b[A-Za-z_][\w.]*(?:Error|Exception)\b")


def exception_signature(line: str) -> str:
    """`line` from its first exception name onward; the whole line when it
    names none (a `No module named x` / `not found` line has no label to
    strip and stays as it is)."""
    m = _EXC_TOKEN_RE.search(str(line or ""))
    return str(line or "")[m.start():] if m else str(line or "")


def error_line_fingerprint(output: str) -> str:
    """Fingerprint of the result's error line under `normalise_volatile`
    and `exception_signature`; "" when the result has no error line. This
    — not the whole output — is what an `execute` run is counted under."""
    line = error_line(output)
    if not line:
        return ""
    norm = re.sub(r"\s+", " ", exception_signature(normalise_volatile(line))).strip().lower()
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
    return KB_NO_ANSWER_MARKER in text


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
#: action chosen by an argument (e.g. ``manage_composed_skills(action="list"``
#: vs ``"define")``, ``file_system(operation="read"`` vs ``"write")``). A
#: no-progress loop on one of these is almost always the agent re-READING to
#: orient itself before performing the WRITE it was actually asked to do.
#:
#: The no-progress breaker's first-trip remedy is to set
#: ``force_final_response`` — which drops the toolset and routes the next turn
#: as text-only. For an ordinary re-observation loop (re-screenshot, re-read
#: the same file with nothing left to do) that is correct. For a read/write
#: tool it is destructive: it bars the pending mutation forever, so the agent
#: "finishes" having silently done nothing. Observed failure: a request to
#: reconfigure a composed skill looped on ``action="list"``, got
#: force-finalised at 3x, and the model's subsequent ``action="define"`` was
#: scrubbed by the final-generation stream guard — the change never landed.
#:
#: For these tools the breaker still STEERS the model off the wasteful re-read
#: but leaves tools available so the write can land. The
#: :data:`READWRITE_HARD_STOP` backstop fires if it genuinely keeps thrashing.
READWRITE_LOOP_TOOLS = frozenset({
    "manage_composed_skills",
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
    # the "reconfigure-a-composed-skill" bug this set exists to prevent.
    "manage_projects",
    "file_system",
    # `knowledge_base` was already here for the read/write reason; e0f4a8bd
    # added a second one. The remedy for a FRUITLESS search is another call
    # to this same tool — `action='outline'` — so force-finalising the turn
    # would bar the one call that answers the question.
    "knowledge_base",
    "update_profile",
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
# from snippets. So this is a steer with the tools KEPT, never a stop, and
# it ships behind the `search_yield_steer` randomized arm: the corpus cannot
# say whether an earlier nudge helps or hurts the answerers, only live
# traffic can. One steer per request; the existing caps remain the stops.
SEARCH_YIELD_STEER = 10
#: Tools whose call OPENS a search result (reset the un-opened run).
SEARCH_OPEN_TOOLS = frozenset({"browser", "deep_research"})


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
        self.search_yield_steered: bool = False

    # -- failure path ------------------------------------------------------

    def note_search_yield(self, fname) -> int:
        """§4JJ: advance the un-opened search run for one dispatched call —
        a `web_search` extends it, a SEARCH_OPEN_TOOLS call resets it, any
        other tool leaves it. Returns the run after the call."""
        if fname == "web_search":
            self.search_run += 1
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
