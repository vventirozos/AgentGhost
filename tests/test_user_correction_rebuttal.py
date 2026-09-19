"""§4IA — the correction that denies what the reply said.

THE LIVE MISS (2026-09-17). Reply: "eckit is ECMWF's internal C++ library
…". Next message: "eckit is not internal and you can request gribs from
mars as a public user". No anchored phrase (Signal A), Jaccard with the
prior request 0.1 (Signal B) — the turn stayed `passed` until the human
clicked 👎. Signal C is the shape: a negated clause whose subject AND object
both come from the reply.

Measured over 1490 consecutive live pairs: 24 candidates (1.6%), ~a third
genuine. So C is a CANDIDATE for the judge, never a promoter: it promotes
only on `contradicts=True`.

World where each pin fails: C promotes without a judge (birthdays and
philosophy become FAILED labels), the gate stops admitting C-candidates to
the judge, the subject/object-in-reply requirement is dropped, or the
verdict is asked twice.
"""
import ast
import inspect

import pytest

from ghost_agent.distill import user_correction as uc
from ghost_agent.distill.user_correction import (classify_user_correction,
                                                 is_correction_candidate, rebuttal_clause)

REPLY = ("**Did I use eckit?** No — pure NumPy (no ECMWF eckit library). "
         "eckit is ECMWF's internal C++ library for the reduced grid/spherical harmonics.")
LIVE = "eckit is not internal and you can request gribs from mars as a public user"
PREV_REQ = "can you show me the PNG. Also did you use eckit?"


# --- the clause -------------------------------------------------------------

def test_live_message_is_a_rebuttal_of_the_reply():
    assert rebuttal_clause(LIVE, REPLY).startswith("eckit is not internal")


@pytest.mark.parametrize("cur", [
    "the game is not right, the ball is lost the moment i press space.",
    "save session doesn't work. we need permanent persistency",
    "the controls now work. but the position of the ball, relative to the launcher "
    "doesn't allow it to launch so the ball never enters the table",
])
def test_corpus_complaints_are_candidates(cur):
    """Three live complaints (2026-07/08 sessions) against a reply that
    asserted the opposite in the same words — the corpus rows the
    measurement counted."""
    reply = ("The game is right: press space to launch the ball from the launcher; "
             "save session will work and store the state; the controls allow the "
             "ball to launch and enter the table.")
    assert rebuttal_clause(cur, reply)


@pytest.mark.parametrize("cur", [
    "leonidas was born march 12 2026, thodoris was born november 25 2016.",
    "the ball is going straight up",
    "at what date is mars gonna be the closest to earth?",
])
def test_plain_copula_is_not_a_negation(cur):
    """The first draft listed bare `is|are|was|were` as negations; these
    corpus messages were all 'rebuttals' of replies that mentioned the
    same words."""
    reply = ("Leonidas was born in March 2026 and Thodoris in November 2016; the ball "
             "is going up; Mars is closest to Earth at opposition, date TBD.")
    assert rebuttal_clause(cur, reply) is None


def test_subject_and_object_must_both_come_from_the_reply():
    assert rebuttal_clause(LIVE, "Here's your PNG: /api/download/plot.png") is None     # neither
    assert rebuttal_clause(LIVE, "eckit was used for nothing here.") is None            # subject only
    assert rebuttal_clause(LIVE, "This is an internal detail of the plotting code.") is None  # object only


def test_negation_without_a_clause_is_not_a_rebuttal():
    assert rebuttal_clause("not sure, can you check?", REPLY) is None
    assert rebuttal_clause("thanks — no further questions", REPLY) is None


def test_clause_deep_in_a_long_message_is_commentary():
    filler = "word " * 200
    assert rebuttal_clause(filler + LIVE, REPLY) is None
    assert uc.REBUTTAL_SCAN_CHARS <= 1000


def test_garbage_inputs():
    assert rebuttal_clause(None, REPLY) is None
    assert rebuttal_clause(LIVE, None) is None
    assert rebuttal_clause("", "") is None


# --- the candidate gate ------------------------------------------------------

def test_gate_admits_phrase_or_rebuttal_and_nothing_else():
    assert is_correction_candidate(LIVE, REPLY)                       # C
    assert is_correction_candidate("no, that's wrong", "anything")    # A
    assert not is_correction_candidate("thanks, looks good!", REPLY)
    assert not is_correction_candidate(LIVE, "unrelated reply text")


# --- promotion ---------------------------------------------------------------

def test_rebuttal_promotes_only_when_a_judge_agrees():
    kw = dict(prev_user_request=PREV_REQ, prev_assistant_response=REPLY,
              current_user_text=LIVE)
    none = classify_user_correction(**kw, contradicts=None)
    yes = classify_user_correction(**kw, contradicts=True)
    no = classify_user_correction(**kw, contradicts=False)
    assert not none.is_correction and any(s.startswith("rebuttal") for s in none.signals)
    assert yes.is_correction and "adjudicated(yes)" in yes.signals
    assert not no.is_correction and "adjudicated(no)" in no.signals


def test_phrase_plus_rephrase_still_promotes_without_a_judge():
    """The pre-§4IA path is untouched: A + B, no judge → correction."""
    v = classify_user_correction(
        prev_user_request="what is the capital of australia",
        prev_assistant_response="The capital of Australia is Sydney.",
        current_user_text="no, the capital of australia is not sydney, it's canberra",
        contradicts=None)
    assert v.is_correction and "phrase" in v.signals


def test_rebuttal_alone_never_promotes_even_with_rephrase():
    """A rebuttal that also rephrases the request is still C without A: no
    judge, no promotion — B corroborates only an opener."""
    v = classify_user_correction(
        prev_user_request="is eckit internal to ecmwf?",
        prev_assistant_response="Yes, eckit is internal to ECMWF.",
        current_user_text="eckit is not internal to ecmwf, is it open source?",
        contradicts=None)
    assert not v.is_correction
    assert any(s.startswith("rebuttal") for s in v.signals)
    assert any(s.startswith("rephrase") for s in v.signals)


def test_affirmation_veto_still_applies():
    v = classify_user_correction(
        prev_user_request="fix the sort", prev_assistant_response="The sort is fixed and stable.",
        current_user_text="actually the sort is not slow any more, it works great now",
        contradicts=True)
    assert not v.is_correction and "affirmation-veto" in v.signals


# --- the caller uses the module's gate ----------------------------------------

def test_adjudicator_gate_is_the_modules_candidate_test():
    from ghost_agent.core import agent as ag
    src = inspect.getsource(ag.GhostAgent._adjudicate_correction)
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    names = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "is_correction_candidate" in names
    assert "has_correction_phrase" not in names         # no private copy of half the gate
