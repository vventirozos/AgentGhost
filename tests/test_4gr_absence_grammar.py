"""§4GR (2026-09-14): one adverb took the truncation guard off.

Request 0516659a — the §4GN breadth probe — was REFUTED at 0.90 with a
single issue: *"The claim mentions '<address>' which is not explicitly in
the provided tool output."* Its four multi-query dark-web outputs are the
largest tool bodies this agent produces, so the evidence digest was heavily
cut, and the mechanical truncation guard exists for exactly that shape: an
absence-only refute over a digest the packer marked as truncated is
UNCERTAIN, never REFUTED.

It never fired. `_ABSENCE_ISSUE_RE` required the negation to sit flush
against its verb — `not in`, `not mentioned in` — so ONE WORD in between
took the whole guard off. The same gap swallows a plural subject: the
vocabulary has `does not appear in` and `doesn't appear in` and not `do not
appear in`.

Measured on 194 refute issues mined from the live log: 13 matched before,
17 after, and every documented false-positive probe still refuses.

The world each pin fails in: a tree where an adverb or a plural subject
hides an absence complaint from the guard, where "not only … but also"
reads as absence, or where the two copies of this vocabulary drift apart
again.
"""
import pytest

from ghost_agent.core.objection import _ABSENCE_RE
from ghost_agent.core.verifier import _ABSENCE_ISSUE_RE


ABSENCE = [
    # the live case, verbatim in shape
    "The claim mentions 'a@b.co' which is not explicitly in the provided tool output.",
    "The figure is not directly stated in the evidence.",
    "The address does not appear in the tool output.",
    "Project IDs are fabricated as they do not appear in the evidence.",
    "The date is not confirmed by the evidence.",
    "The release date is not verifiable in the evidence.",
    # …and the shapes that already worked, which must keep working
    "Humidity around 28% is not in the evidence.",
    "Rodri to Barcelona is not mentioned in the news results.",
    "The value is absent from the tool output.",
]

NOT_ABSENCE = [
    # claim-side defects: a cut digest excuses none of these
    "The claim is truncated mid-sentence",
    "The screenshot was never taken, so the UI claim is unverified",
    "Unsupported greeting 'Good morning Vasilis!'",
    "omitted from the reply though the evidence provides it",
    "the evidence contradicts the claim, which omits context",
    # a CONTRADICTION is judgeable on a partial digest
    "The latest stable release is 3.14.6, not 3.13.8.",
    # ⚠ the opposite claim, one word from the absence shape
    "The value is not only in the evidence but also in the reply",
]


@pytest.mark.parametrize("issue", ABSENCE)
def test_the_guard_sees_the_absence_complaint(issue):
    assert _ABSENCE_ISSUE_RE.search(issue), issue


@pytest.mark.parametrize("issue", NOT_ABSENCE)
def test_the_guard_refuses_everything_else(issue):
    assert not _ABSENCE_ISSUE_RE.search(issue), issue


@pytest.mark.parametrize("issue", [
    "The claim mentions 'a@b.co' which is not explicitly in the provided tool output.",
    "Project IDs are fabricated as they do not appear in the evidence.",
    "The date is not confirmed by the evidence.",
])
def test_the_objection_module_reads_the_same_grammar(issue):
    """The two copies must track each other — a synonym choice decides
    whether an absence-only refute over cut evidence gets the guard's
    mechanical UNCERTAIN or a trip through the credulous re-adjudication
    (2026-08-07), and now an ADVERB decides the same thing."""
    assert _ABSENCE_RE.search(issue), issue


def test_only_is_excluded_from_the_adverb_slot_in_both_copies():
    """The exclusion is load-bearing and was briefly inert: written into a
    non-raw string, `\\b` became a BACKSPACE character, the lookahead could
    never fire, and "not only in the evidence" read as an absence complaint.
    Assert the behaviour, not the spelling."""
    s = "The value is not only in the evidence but also in the reply"
    assert not _ABSENCE_ISSUE_RE.search(s)
    assert not _ABSENCE_RE.search(s)


def test_the_adverb_slot_takes_one_word_not_a_clause():
    """It admits an adverb, not a bridge across a clause: an evidence noun
    forty characters away behind a claim-noun must still be refused."""
    assert not _ABSENCE_ISSUE_RE.search(
        "The reply is not carefully written, unlike the evidence")
