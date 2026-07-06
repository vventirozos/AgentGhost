"""Tests for distill.user_correction.classify_user_correction.

The classifier promotes a prior trajectory only when BOTH a correction
phrase AND a rephrase signal fire. These tests pin the exact phrases
that count, the rephrase Jaccard threshold, and — most importantly —
the false-positive guards (single-signal cases must NOT promote).
"""

from __future__ import annotations

import pytest

from ghost_agent.distill.user_correction import (
    classify_user_correction,
    CorrectionVerdict,
    JACCARD_REPHRASE_THRESHOLD,
)


# --------------------------------------------------------------- happy path


@pytest.mark.parametrize("phrase", [
    "no, that's wrong",
    "Nope, try again",
    "actually, I meant the other one",
    "Wrong — I asked about Y",
    "that's not what I asked",
    "you misunderstood",
    "I didn't ask that",
    "incorrect",
    "redo",
    "didn't work",
])
def test_correction_phrase_plus_rephrase_promotes(phrase):
    """Both signals fire → is_correction=True."""
    prev_user = "list all the python files in the workspace directory"
    # Make the current message a clear rephrase of prev_user with a
    # correction-phrase prefix tacked on.
    current = f"{phrase}: list every python file in workspace directory"
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="here's a list of go files",
        current_user_text=current,
    )
    assert v.is_correction, (
        f"phrase={phrase!r}, signals={v.signals}, conf={v.confidence}"
    )
    assert "phrase" in v.signals
    assert any(s.startswith("rephrase") for s in v.signals)
    assert v.confidence >= 0.7
    assert v.reason


# ----------------------------------------------------- single-signal guards


def test_phrase_only_does_not_promote():
    """Correction phrase fires but the new message is wholly new
    content (no rephrase) → must not promote. This is the dominant
    false-positive shape: a user starting their next question with
    'No, ...' as a discourse marker rather than a correction."""
    v = classify_user_correction(
        prev_user_request="what is the capital of France",
        prev_assistant_response="Paris",
        current_user_text="No, now tell me about quantum computing",
    )
    # Phrase fires...
    assert "phrase" in v.signals
    # ...but rephrase doesn't (no content overlap).
    assert not any(s.startswith("rephrase") for s in v.signals)
    assert not v.is_correction


def test_rephrase_only_does_not_promote():
    """User rephrases without a correction phrase — could just be
    them adding a clarifying clause to their original ask. We don't
    promote on this alone."""
    v = classify_user_correction(
        prev_user_request="parse the access log and count errors per host",
        prev_assistant_response="here you go",
        current_user_text="parse the access log and count errors per host, sorted",
    )
    # Rephrase fires...
    assert any(s.startswith("rephrase") for s in v.signals)
    # ...phrase does not.
    assert "phrase" not in v.signals
    assert not v.is_correction


def test_no_signals_returns_clean_verdict():
    v = classify_user_correction(
        prev_user_request="what time is it",
        prev_assistant_response="3pm",
        current_user_text="great, now book me a meeting",
    )
    assert not v.is_correction
    assert v.confidence == 0.0
    assert v.signals == []
    assert v.reason == ""


# ------------------------------------------------------ defensive normalisation


@pytest.mark.parametrize("bad_input", [None, 0, [], {}])
def test_non_string_inputs_do_not_raise(bad_input):
    v = classify_user_correction(
        prev_user_request=bad_input,
        prev_assistant_response=bad_input,
        current_user_text=bad_input,
    )
    assert isinstance(v, CorrectionVerdict)
    assert not v.is_correction


def test_empty_strings_return_no_correction():
    v = classify_user_correction(
        prev_user_request="",
        prev_assistant_response="",
        current_user_text="",
    )
    assert not v.is_correction
    assert v.signals == []


# --------------------------------------------------- phrase-anchoring guard


def test_correction_phrase_must_be_anchored_at_start():
    """A 'no' deep inside a sentence is not a correction marker —
    e.g., 'I have no preference' or 'tell me about lake erie, no
    preferences'. Must NOT trip the phrase signal."""
    prev_user = "tell me about big lakes"
    current = (
        "tell me about big lakes - I have no strong preference, but lake "
        "erie sounds good"
    )
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="here are big lakes",
        current_user_text=current,
    )
    # The mid-sentence "no" must not fire the anchored phrase regex.
    assert "phrase" not in v.signals
    assert not v.is_correction


# ----------------------------------------------------- rephrase threshold


def test_rephrase_overlap_below_threshold_does_not_fire():
    """Tiny content overlap (well below 0.40) should not register as
    a rephrase. Calibration check."""
    v = classify_user_correction(
        prev_user_request="list python files in workspace",
        # share only 'workspace' with prior — Jaccard ~0.10
        prev_assistant_response="...",
        current_user_text=(
            "no, instead show me docker volume mounts and ports for the "
            "ghost workspace stack"
        ),
    )
    rephrase_signals = [s for s in v.signals if s.startswith("rephrase")]
    assert not rephrase_signals
    # Phrase fires alone, so promotion still doesn't happen.
    assert not v.is_correction


def test_rephrase_threshold_is_module_level():
    """The threshold is exposed as a module attribute so operators
    can tune it without forking the function. Pin its current value
    so a refactor doesn't silently change calibration."""
    assert 0.30 <= JACCARD_REPHRASE_THRESHOLD <= 0.50


# ----------------------------------------------------- short-message guard


def test_bare_correction_does_not_rephrase_match():
    """A bare 'no' has zero content tokens — it must not artificially
    overlap with anything. This is the guard for
    MIN_CURRENT_TOKENS_FOR_REPHRASE."""
    v = classify_user_correction(
        prev_user_request="list python files in workspace",
        prev_assistant_response="here's the list",
        current_user_text="no",
    )
    rephrase_signals = [s for s in v.signals if s.startswith("rephrase")]
    assert not rephrase_signals  # empty token set -> no match
    assert not v.is_correction


# ---------------------------------------------------- verdict shape contract


def test_verdict_dataclass_shape():
    v = classify_user_correction(
        prev_user_request="x",
        prev_assistant_response="y",
        current_user_text="z",
    )
    assert hasattr(v, "is_correction")
    assert hasattr(v, "confidence")
    assert hasattr(v, "signals")
    assert hasattr(v, "reason")
    assert isinstance(v.signals, list)


def test_promoted_verdict_has_descriptive_reason():
    """The reason string is what lands in Trajectory.failure_reason
    and in the corrections sidecar — it must be non-empty + carry
    enough info to debug a false positive after the fact."""
    prev_user = "show me the postgres tables in the analytics schema"
    current = (
        "actually, show me the postgres tables in the analytics "
        "schema sorted alphabetically"
    )
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="...",
        current_user_text=current,
    )
    assert v.is_correction
    assert v.reason
    # Should mention BOTH signals.
    assert "phrase" in v.reason
    assert "rephrase" in v.reason


# ------------------------------------------------------- affirmation veto
# Deferred finding (BUGHUNT.md, unit 22): an AFFIRMING follow-up that
# opens with "actually" and echoes the request's content words trips
# BOTH signals — promoting a good turn to FAILED and retracting its
# lesson (self-poisoning). A clear affirmation with no negative marker
# now vetoes the verdict.


def test_affirming_actually_followup_is_vetoed():
    prev_user = "sort the list by date descending"
    current = "actually the list sort by date descending works great!"
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="sorted, newest first",
        current_user_text=current,
    )
    # Sanity: both raw signals DID fire (this is the false-positive shape)…
    assert "phrase" in v.signals
    assert any(s.startswith("rephrase") for s in v.signals)
    # …and the veto killed the promotion.
    assert "affirmation-veto" in v.signals
    assert not v.is_correction
    assert v.confidence == 0.0
    assert v.reason == ""


def test_you_were_right_is_vetoed():
    prev_user = "use binary search for the symbol lookup"
    current = "actually you were right, binary search for the symbol lookup is faster"
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="switched to binary search",
        current_user_text=current,
    )
    assert "affirmation-veto" in v.signals
    assert not v.is_correction


def test_negative_marker_blocks_the_veto():
    """A complaint that HAPPENS to contain praise-shaped words must
    still promote — the negative marker wins."""
    prev_user = "sort the list by date descending"
    current = ("actually the list sort by date descending looks good "
               "but doesn't work")
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="done",
        current_user_text=current,
    )
    assert "affirmation-veto" not in v.signals
    assert v.is_correction


def test_anchored_no_blocks_the_veto():
    """'No, you're right, …' is genuinely ambiguous — ambiguity resolves
    toward correction (a missed veto only costs a lesson; a wrongly
    vetoed real correction costs the FAILED label)."""
    prev_user = "sort the list by date descending"
    current = "no, you're right, sort the list by date descending like i said"
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="descending it is",
        current_user_text=current,
    )
    assert "affirmation-veto" not in v.signals
    assert v.is_correction


def test_plain_praise_never_consults_the_veto():
    """No correction opener → veto never appears in signals (it is only
    consulted when both raw signals fired)."""
    prev_user = "sort the list by date descending"
    current = "the list sort by date descending works great, thanks!"
    v = classify_user_correction(
        prev_user_request=prev_user,
        prev_assistant_response="done",
        current_user_text=current,
    )
    assert not v.is_correction
    assert "affirmation-veto" not in v.signals
