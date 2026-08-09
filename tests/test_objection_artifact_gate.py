"""Rule 3 must arm on a BARE artifact allegation when the markers are there.

THE DEFECT (measured 2026-08-09 on the 433-trial re-bench). `resolve_issue`
rule 3 already did the right thing — "is the alleged machine noise literally
in the claim?" is a string test, nobody's opinion required — but it was gated
behind `_is_noise_allegation`, which requires a noise WORD next to
"artifact"/"marker" because those are ordinary English ("the wrong build
artifact" is a content objection).

Sound in general. But the cheap judge's issue text for this fault class is
frequently the bare word 'artifact', which fails that gate. So rule 3 never
armed, the objection escalated unresolved, and the strong model dismissed a
complaint that carried no argument at all ("All suspects are false alarms").

Cost: 23 of artifact_leak's 58 trials overturned, every one wrongly, dragging
its catch rate to 0.431 when the cheap judge had actually achieved 0.828.

THE FIX: the ambiguity the gate protects against cannot arise once the claim
is KNOWN to contain the markers. "The wrong build artifact" is a content
objection about a claim with no diff markers in it. So when
`_claim_noise_markers` finds them, a bare artifact word is accepted.

Measured effect (paired McNemar, strict replay, judge held fixed):
    balanced 0.8130 -> 0.8251, refute 0.7424 -> 0.7667, non-refute UNCHANGED,
    8 trials fixed / 0 broken, p=0.0078.

A STRONGER variant — uphold whenever the claim has markers, regardless of how
the issue was worded — was measured and REJECTED: it scored better on the
bench's balanced metric (0.832, p=0.0225) but raised FPR 0.121 -> 0.138 and
degraded-evidence FP 0.111 -> 0.133, because it also protects WRONG refutes on
clean claims that happen to contain marker-like text. At any production-like
base rate (<=25% corrupted) it is the worse trade; the bench's 76%-corrupted
mix is the only regime where it wins. See `test_variant_b_shape_is_not_shipped`.
"""

import pytest

from ghost_agent.core import objection as O

NOISE = "\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n"
DIRTY = "Athens is 34C." + NOISE + " Humidity 28%."
CLEAN = "Athens is 34C. Humidity 28%."
EV = "temperature 34C, humidity 28%"


# ── the fix ─────────────────────────────────────────────────────────────────

def test_bare_artifact_word_upholds_when_markers_are_present():
    """THE DEFECT: this was UNRESOLVED, escalated, and destroyed."""
    d, why = O.resolve_issue("artifact", DIRTY, EV)
    assert d == O.UPHOLD and "literally present" in why


@pytest.mark.parametrize("issue", ["artifact", "artifacts", "marker",
                                   "markers", "ARTIFACT"])
def test_the_shapes_the_cheap_judge_actually_emits(issue):
    assert O.resolve_issue(issue, DIRTY, EV)[0] == O.UPHOLD


def test_a_fully_worded_allegation_still_upholds():
    """The pre-existing path must be untouched."""
    d, _ = O.resolve_issue("the claim contains unflagged diff markers",
                           DIRTY, EV)
    assert d == O.UPHOLD


# ── the gate it must NOT weaken ─────────────────────────────────────────────

def test_bare_artifact_on_a_CLEAN_claim_still_escalates():
    """The context gate exists for two measured false convictions. On a claim
    with no markers, a bare 'artifact' is genuinely ambiguous and must still
    go to judgement — the fix must not turn it into a verdict."""
    assert O.resolve_issue("artifact", CLEAN, EV)[0] == O.UNRESOLVED


def test_a_content_objection_mentioning_artifact_is_not_hijacked():
    """'the wrong build artifact was deployed' is a CONTENT objection."""
    assert O.resolve_issue("the wrong build artifact was deployed",
                           CLEAN, EV)[0] == O.UNRESOLVED


def test_unrelated_issues_are_unaffected_on_a_dirty_claim():
    """⚠ THE VARIANT-B BOUNDARY. Markers in the claim must not convert an
    unrelated objection into an uphold — that is the rejected behaviour."""
    d, _ = O.resolve_issue("the tone is too informal", DIRTY, EV)
    assert d != O.UPHOLD, (
        "an unrelated objection was upheld merely because the claim has "
        "markers — that is variant B, which measured FPR 0.121 -> 0.138")


def test_variant_b_shape_is_not_shipped():
    """Structural guard against re-introducing the rejected variant.

    Variant B upheld on marker presence ALONE. It wins on the bench's
    76%-corrupted mix and loses at every production-like base rate.
    """
    import inspect
    src = inspect.getsource(O.resolve_issue)
    # The uphold must remain conditional on the allegation, not on presence
    # alone. A bare `if _present: return (UPHOLD` is variant B.
    assert "if _is_noise_allegation(text) or (_present" in src, (
        "rule 3's uphold is no longer gated on the allegation — check this "
        "is not variant B")


# ── fenced diffs: the conservative half must stay conservative ──────────────

def test_a_fenced_diff_is_presentation_not_leaked_noise():
    """`_claim_noise_markers` strips fences first — a claim SHOWING a diff in
    a code fence is exactly how a claim should show one. Two measured false
    convictions shaped that; the fix must not disturb it."""
    fenced = "Here is the patch:\n```\n<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE\n```\n"
    assert O._claim_noise_markers(fenced) == []
    assert O.resolve_issue("artifact", fenced, EV)[0] == O.UNRESOLVED


def test_a_markdown_horizontal_rule_is_not_a_diff_header():
    assert O._claim_noise_markers("Summary\n---\nAll good.") == []


# ── the whole-refute path ───────────────────────────────────────────────────

def test_resolve_refute_upholds_and_suppresses_the_escalation():
    """The point of upholding: the refute stands with NO main-model call, so
    the overturner never gets to destroy it."""
    d, reasons, unresolved = O.resolve_refute(["artifact"], DIRTY, EV)
    assert d == O.UPHOLD and reasons and not unresolved


def test_resolve_refute_on_a_clean_claim_still_escalates():
    d, _, unresolved = O.resolve_refute(["artifact"], CLEAN, EV)
    assert d is None and unresolved
