"""What a tool call actually did — as data, not as a string to sniff.

WHY THIS EXISTS
===============
An audit (§4DO, 2026-08-29) AST-parsed every ``return``/``raise`` in this
package — **1,026 result heads** — ran every "did this fail?" predicate in the
tree over each, and cross-checked against **4,391 recorded live tool calls**.
It found **ten** distinct vocabularies where the code claims one, five of them
inlined in the dispatch loop, and the inline ones owning the load-bearing
decisions. Measured consequences, all live:

* 63 occurrences across 36 requests where a `file_system` REFUSAL
  ("SYSTEM INSTRUCTION: …forgot 'replace_with'", "REPLACE REJECTED
  (byte-identical)") was credited with CHANGING THE WORLD — clearing every
  recorded pre-flight failure, wiping the loop-breaker's memory, and
  *decrementing* the strike count, so a run of rejected replaces erased the
  strikes earlier failures had earned and the cap could never fire.
* 15 turns where a shell command that exited non-zero was reported to the
  model as SUCCEEDED, under the one banner that claims to be AUTHORITATIVE.
* 219 calls where the turn loop booked a success and the trajectory corpus
  booked a failure — so the corpus, skill graduation, the verifier escalation
  and foresight seeding all disagreed with the loop that produced the row.

Five rounds of review had responded by adding prefixes, and each addition
created the next round's split: widening `Error:` to `Error\\b` fixed nine
producers and desynchronised two other readers. A prefix cannot encode
`PARTIAL:`, nor a write that landed while its syntax check did not — those
are genuinely three-valued — and it cannot distinguish a failure that touched the
world from one that refused to.

THE CONTRACT
============
A tool may return a plain ``str`` (the legacy path — classified by exactly ONE
predicate, so behaviour is unchanged) or a ``ToolOutcome``. ``ToolOutcome``
stringifies to its text, so every consumer that only wants the message keeps
working untouched.

``status`` is the whole point:

``OK``          the call did what was asked.
``FAILED``      it tried and could not. May have changed the world partially.
``REJECTED``    it REFUSED — malformed arguments, a no-op edit, a guard trip.
                **Nothing was touched.** This is the state the tree had no way
                to say, and the one the world-changed credit needs.
``PARTIAL``     some of it landed. Not a success, not a clean failure; an
                idempotent setter must NOT record it as applied.
``UNRESOLVED``  no verdict yet — a detached job still running. TWO producers:
                `execute._promoted_result` (a command detached at its budget)
                and `swarm`'s "N still running, they were NOT cancelled"
                branch. Every reader must SKIP it rather than label it; a
                bool cannot carry a third state, which is why
                `is_unresolved_tool_result` exists alongside the failure
                sniffer. `sandbox.jobs.is_promoted_result` is still consulted
                directly in the text-based paths that predate the status.
"""

from __future__ import annotations

import re

from enum import Enum
from typing import Any, Optional


_EXIT_CODE_RE = re.compile(r"EXIT CODE:\s*(\d+)")


class OutcomeStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    REJECTED = "rejected"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


#: Statuses that imply the call changed NOTHING unless it says otherwise.
#:
#: REJECTED never touched anything by definition. FAILED and UNRESOLVED
#: usually did not either — a write that could not open its file, a command
#: that never started — which is exactly why the pre-flight guard records
#: them: "re-running this unchanged will fail the same way". The exception is
#: a call that got partway (a truncating write that then hit ENOSPC), and
#: that is what `world_changed=True` is for. PARTIAL says it in the status.
_NO_WORLD_CHANGE = frozenset({
    OutcomeStatus.REJECTED, OutcomeStatus.FAILED, OutcomeStatus.UNRESOLVED,
})


class ToolOutcome(str):
    """A tool result that IS its text, and also carries what it did.

    ⚠ A `str` SUBCLASS, deliberately. The first version was a dataclass with
    a hand-written proxy — `__str__`, `__eq__`, `__len__`, `__getattr__`
    delegation — and reviewers found the holes one at a time: `isinstance(x,
    str)` was False (an existing test asserts it on a tool result),
    `json.dumps` raised, `"".join([...])` raised, `re.search` raised, `+`
    raised, `os.path.join` raised, and the hash changed when the text was
    edited in place. Roughly a thousand call sites in this tree treat a tool
    result as a string. Being one is the only version of "day 1 is
    behaviour-identical" that actually holds.

    The cost is that equality and hashing are the string's, so
    `ok("x") == rejected("x")`. Nothing in the tree containerises results by
    value, and consistency with `str` is worth more here than a distinction
    no consumer asks for.
    """

    __slots__ = ("status", "world_changed", "reason_code", "declared")

    def __new__(cls, text: Any = "", status: "OutcomeStatus" = OutcomeStatus.OK,
                world_changed: Optional[bool] = None,
                reason_code: Optional[str] = None,
                declared: bool = True):
        self = super().__new__(cls, "" if text is None else str(text))
        self.status = status
        self.world_changed = world_changed
        self.reason_code = reason_code
        # Did a PRODUCER set this status, or did `coerce` guess it from the
        # text? The distinction is what lets the shell predicates stop
        # re-sniffing a result whose author already answered the question —
        # see `exit_code_failed`. `retryable` and `meta` used to live here
        # too: zero producers ever set either (they were forwarded and
        # never read), and `meta` allocated a dict on EVERY construction
        # for a field nothing wrote.
        self.declared = declared
        return self

    # `text` stays available: it is what the dispatch loop reads, and the
    # name says "the message" where the object itself says "the result".
    @property
    def text(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return (f"ToolOutcome({str(self)[:60]!r}, "
                f"status={self.status.value}, "
                f"reason_code={self.reason_code!r})")

    def __reduce__(self):
        return (_rebuild_outcome,
                (str(self), self.status, self.world_changed,
                 self.reason_code, self.declared))

    # -- what the loop asks ------------------------------------------------
    @property
    def is_failure(self) -> bool:
        """Did this call not do what was asked? REJECTED counts."""
        return self.status is not OutcomeStatus.OK

    @property
    def is_rejection(self) -> bool:
        return self.status is OutcomeStatus.REJECTED

    @property
    def changed_the_world(self) -> bool:
        """May this call have mutated anything?

        The question the pre-flight guard and the loop-breaker's
        world-changed reset should have been asking all along.
        """
        if self.world_changed is not None:
            return self.world_changed
        return self.status not in _NO_WORLD_CHANGE

    @property
    def may_record_as_applied(self) -> bool:
        """May an idempotent setter record this call's args as done?

        Only a clean OK. A crashed or partial write that gets recorded means
        the model's legitimate retry is refused with "the intended state is
        already applied" — measured live, with the tool having run zero
        times.

        Deliberately `not is_failure` rather than its own status set: an
        earlier version kept a separate frozenset listing every non-OK
        status, which is the same question asked twice — the duplication
        this module exists to remove. It stays named because the CALLER's
        question is worth naming even when the answer is one other property.
        """
        return not self.is_failure

    @property
    def exit_code_failed(self) -> bool:
        """Does this result carry a NON-ZERO exit-code envelope?

        Just the banner — no prose sniffing. Separate from `shell_failed`
        because the two questions had been conflated: a reviewer measured
        226 live successes booked as failures when the marker fallback was
        applied to every tool, including a `file_system` read of a Python
        file containing `except ValueError:`.

        A DECLARED status short-circuits the banner: the producer already
        answered this, and re-reading its prose can only overrule it with a
        guess. Measured on the 4,391-call corpus, the banner rule fires on
        12 non-`execute` results — 8 are genuine failures whose banner IS
        their own envelope, and 4 are `manage_projects` SUCCESSES whose JSON
        body quotes an `EXIT CODE: 1` out of a stored `autoadvance_failed`
        event. Position cannot separate those two (a JSON payload is one
        line, so "the banner heads the result" is true for both) — only the
        producer can, which is why `_ok` in projects.py now says so.
        """
        if self.declared:
            return self.status is OutcomeStatus.FAILED
        m = _EXIT_CODE_RE.search(self)
        return m is not None and m.group(1).lstrip("0") != ""

    @property
    def shell_failed(self) -> bool:
        """For a SHELL-shaped result: did the command fail?

        ⚠ Only for results that are shell output. The prose fallback below
        is unanchored by necessity (a crashing script's traceback can be
        anywhere in its stdout) and is therefore WRONG for any tool whose
        output merely quotes code — which is most of them. Ask
        `exit_code_failed` instead when the tool is not a shell.

        Status first: a REJECTED command did not run, whatever its text
        says. The exit-code envelope is authoritative next; the crash
        markers are the last resort.

        UNRESOLVED is NOT failure and must be tested BEFORE the general
        non-OK rule: a detached job that has not finished has no verdict
        yet. Answering True here was how the exemption 600 lines up in the
        dispatch loop got undone — `exit_code_val = 1 if shell_failed` gave
        an in-flight run a strike anyway.
        """
        if self.status is OutcomeStatus.UNRESOLVED:
            return False
        if self.status is not OutcomeStatus.OK:
            return True
        # NO `declared` short-circuit here, deliberately — unlike
        # `exit_code_failed`. This question is asked only of the SHELL, and
        # for a shell result the exit code outranks any status: a producer
        # that declares OK while its own envelope says 127 is wrong, and the
        # banner is the evidence. `exit_code_failed` can defer to a declared
        # status because it is asked of tools that merely QUOTE a banner.
        m = _EXIT_CODE_RE.search(self)
        if m is not None:
            return m.group(1).lstrip("0") != ""
        return any(k in self for k in ("Error", "Exception", "Traceback"))

    # -- constructors ------------------------------------------------------
    @classmethod
    def ok(cls, text, **kw) -> "ToolOutcome":
        return cls(text, status=OutcomeStatus.OK, **kw)

    @classmethod
    def failed(cls, text, **kw) -> "ToolOutcome":
        return cls(text, status=OutcomeStatus.FAILED, **kw)

    @classmethod
    def rejected(cls, text, **kw) -> "ToolOutcome":
        """The tool refused. NOTHING was touched."""
        kw.setdefault("world_changed", False)
        return cls(text, status=OutcomeStatus.REJECTED, **kw)

    @classmethod
    def partial(cls, text, **kw) -> "ToolOutcome":
        return cls(text, status=OutcomeStatus.PARTIAL, **kw)

    @classmethod
    def unresolved(cls, text, **kw) -> "ToolOutcome":
        return cls(text, status=OutcomeStatus.UNRESOLVED, **kw)

    @classmethod
    def coerce(cls, result: Any) -> "ToolOutcome":
        """Normalise ANY tool result into an outcome.

        A `ToolOutcome` passes through. Anything else is stringified and
        classified here — two paths through this decision instead of ten.

        ⚠ NOT behaviour-identical for a REFUSAL, and the first version of
        this docstring wrongly said it was. Before, `_res_is_error` was
        `result_is_failure` alone, and a `"SYSTEM INSTRUCTION: …"` refusal
        from an unmigrated tool scored **zero** strikes. It now scores one,
        because a refusal IS a failure — that is the whole point of §4DO,
        which measured 63 live refusals being booked as successes and
        crediting world changes. The deliberate consequence: four identical
        malformed calls now reach the strike cap and the loop breaker
        instead of looping silently. Everything that is not a refusal
        classifies exactly as it did before, and
        `test_tool_outcome_contract` pins both halves separately.
        """
        if isinstance(result, ToolOutcome):
            return result
        # Imported here: `tool_failure` imports nothing from this module, and
        # keeping the dependency one-way lets tools import ToolOutcome
        # without pulling in the classifier.
        from .tool_failure import result_is_failure, result_is_rejection

        text = "" if result is None else str(result)
        # PARTIAL before REJECTED: `_REJECTION_RE` covers both heads, and
        # taking the rejection branch made a legacy "PARTIAL: 1/2 task(s)
        # dispatched" assert that NOTHING was touched — for a call that
        # dispatched half its work and left the rest running.
        if text.lstrip().startswith("PARTIAL:"):
            return cls.partial(text, reason_code="legacy_text", declared=False)
        if result_is_rejection(text):
            return cls.rejected(text, reason_code="legacy_text", declared=False)
        if result_is_failure(text):
            return cls.failed(text, reason_code="legacy_text", declared=False)
        return cls.ok(text, declared=False)


def with_text(res, new_text: str):
    """Rewrite a result's TEXT while keeping everything it said.

    THE helper for this whole class of defect. `ToolOutcome` is a `str`
    subclass, so every ordinary way of touching the text —
    ``res + note``, ``f"{res}…"``, ``str(res).strip()``, a slice, `.replace`
    — goes through `str` and returns a plain `str`, silently discarding the
    status. Found at nine sites in six modules across four review rounds,
    three of them AFTER the same defect had been fixed one boundary away.
    If you are changing the text of something that might be an outcome, come
    through here.
    """
    if isinstance(res, ToolOutcome):
        return ToolOutcome(new_text, status=res.status,
                           world_changed=res.world_changed,
                           reason_code=res.reason_code,
                           declared=res.declared)
    return new_text


def append_note(res, note: str):
    """Append trailing text WITHOUT dropping a `ToolOutcome`'s status.

    `ToolOutcome` is a `str` SUBCLASS, so `res + note` and
    ``f"{res}\nnote"`` both go through `str` and return a plain `str`,
    silently discarding the status. Found at five sites in three modules
    across two review rounds — twice AFTER the same defect had been fixed
    one boundary away. Anything that decorates a tool result on its way out
    must come through here.
    """
    if not note:
        return res
    return with_text(res, str(res) + note)


def _rebuild_outcome(text, status, world_changed, reason_code, declared=True):
    """Module-level so pickle/copy can find it."""
    return ToolOutcome(text, status=status, world_changed=world_changed,
                       reason_code=reason_code, declared=declared)
