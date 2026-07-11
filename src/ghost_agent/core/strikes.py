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


def action_result_fingerprint(result: str) -> str:
    """Whitespace-normalised, bounded fingerprint of a tool result.

    Used by ``note_repeated_action`` to decide whether two SUCCESSFUL
    calls produced "the same observation". Whitespace-only normalisation
    (digits intentionally kept) so a result whose content genuinely
    changed — a re-read that now returns edited bytes, a counter that
    advanced — looks different and does NOT count as a no-progress
    repeat. Bounded slice keeps it cheap and stable."""
    norm = re.sub(r"\s+", " ", str(result or "")).strip().lower()[:600]
    return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()[:12]


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
#: but leaves tools available so the write can land. The >=5 hard stop is the
#: backstop if it genuinely keeps thrashing.
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
    "knowledge_base",
    "update_profile",
})


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
        self.persistent_failure_seen: bool = False
        self.persistent_warned_sigs: set = set()
        self.consecutive_clean_successes: int = 0

    # -- failure path ------------------------------------------------------

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

    def note_action(self, fname, target, result_fp, threshold: int = 3):
        """Record a successful action's (tool, target, result) fingerprint.
        Returns ``(signature, count, tripped)``."""
        return note_repeated_action(
            self.action_sigs, fname, target, result_fp, threshold
        )

    @property
    def decay_frozen(self) -> bool:
        """True while success-decay should be suppressed (a failure loop is
        active). The loop gates its ``execution_failure_count`` decrement on
        ``not decay_frozen``."""
        return self.persistent_failure_seen
