"""Refute-capable verification for tool-free turns (§4EQ, 2026-09-04).

Closes §4EP's open item. 46% of user turns run no tools, so the
evidence-grounded verifier cannot rule on them and they enter the corpus as
the `_UNVERIFIED_PRIOR` placeholder. §4EP then measured what filling that gap
is worth: CONFIRM-only growth shrinks the observed Brier delta by 2-16x across
seeds, because every added row is one the base-rate predictor already gets
right. **Coverage that cannot refute is worse than no coverage.**

So this route is refute-only by construction, and arithmetic rather than
judged: an age against an anchored birth date is a computation.

The property under review: **a tool-free reply that contradicts a stored,
dated fact is refuted; anything else produces no verdict at all.**
"""

import datetime

import pytest

from ghost_agent.core import memory_claim_check as M
from ghost_agent.core.agent import verdict_is_consumable
from ghost_agent.core.calibration import grade_turn_outcome
from ghost_agent.core.memory_claim_check import (anchored_subjects,
                                                 refute_age_claims)
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

NOW = datetime.date(2026, 9, 4)
# The live store, verbatim: one anchored value holding two subjects.
PROFILE = {"relationships": {
    "sons": "Thodoris (born 2016-11-25) and Leonidas (born 2026-03-12)"}}


def _issues(reply, profile=PROFILE, now=NOW):
    return refute_age_claims(reply=reply, profile=profile, now=now)


# ── the design constraint §4EP bought ──────────────────────────────────────

def test_a_correct_claim_produces_NOTHING_not_a_confirmation():
    """⚠ THE WHOLE POINT. An empty result means "nothing to say", never "the
    reply is fine".

    The world it fails in: someone makes the matching branch return a
    CONFIRMED because it feels like free coverage. §4EP measured that exact
    change: confirm-carrying growth shrinks the Brier delta 2-16x, so the
    turn would go from an honest placeholder to a label that makes the
    calibration comparison *less* resolvable — a coverage number bought by
    damaging the thing coverage was for.
    """
    assert _issues("Leonidas is 5 months old.") == []
    assert _issues("Thodoris is 9 years old.") == []
    # And the module offers no way to express a pass.
    assert set(M.__all__) == {"refute_age_claims", "anchored_subjects"}


def test_a_real_contradiction_refutes():
    """The negatives this exists to produce — the scarce class (57 of 402)."""
    assert len(_issues("Leonidas is 9 years old.")) == 1
    assert len(_issues("Thodoris is 3 years old.")) == 1
    got = _issues("Leonidas is 9 years old.")[0]
    assert "Leonidas" in got and "2026-03-12" in got, got


# ── absence can never refute ───────────────────────────────────────────────

def test_no_anchored_fact_means_no_verdict():
    """The refute-on-absence trap, made structurally impossible: with no
    comparand there is no comparison.

    The world it fails in: the store is empty (a fresh box, a failed load)
    and every age the agent states is called wrong.
    """
    for profile in ({}, None, {"relationships": {"sons": "two boys"}},
                    {"a": {"b": ["no dates here"]}}):
        assert _issues("Leonidas is 9 years old.", profile=profile) == []


def test_a_subject_the_reply_never_names_is_not_judged():
    """The world it fails in: any age phrase anywhere is measured against
    whichever birth date happens to be stored — "the server has been up 9
    months" refuted against a child."""
    assert _issues("The server has been up 9 months old.") == []
    assert _issues("My cousin is 40 years old.") == []


def test_a_reply_with_no_age_claim_is_not_judged():
    assert _issues("Your sons are doing well.") == []
    assert _issues("") == []
    assert _issues("I don't know how old they are.") == []


# ── the tolerance is generous, because a rounded answer is not wrong ───────

def test_colloquial_rounding_is_never_refuted():
    """⚠ THE LIVE CASE THIS TOLERANCE EXISTS FOR. On 2026-09-04 Leonidas was
    5 months 23 days: the store's own `_age_phrase` renders "5 months" and
    the user called it "about 6 months". Both are right.

    The world it fails in: a pedantic check writes a 0.0 on one of them and
    teaches calibration that a correct answer was a confident miss — noise in
    the one class the corpus cannot afford noise in.
    """
    for months in (5, 6):
        assert _issues(f"Leonidas is {months} months old.") == [], months
    # Thodoris is 9y9m: both 9 and 10 are defensible.
    for years in (9, 10):
        assert _issues(f"Thodoris is {years} years old.") == [], years


def test_the_tolerance_still_catches_what_it_is_for():
    """A window wide enough to swallow everything is not a check."""
    assert _issues("Leonidas is 2 years old.")        # 24m vs 5.9m
    assert _issues("Thodoris is 3 years old.")        # 36m vs 117m
    assert _issues("Leonidas is 30 months old.")


# ── subject binding: ambiguity must not refute ─────────────────────────────

def test_a_tie_between_two_subjects_says_nothing():
    """⚠ THE FALSE REFUTE THIS ALREADY PRODUCED ONCE, on a real reply.

    "Your sons are 9 (Thodoris) and 5 months old (Leonidas)" puts BOTH names
    exactly 14 characters from "5 months old". First-wins bound the infant's
    age to the nine-year-old and refuted a correct answer. Skipping a tie
    costs a label we never had; taking it writes a 0.0 on a right answer.
    """
    assert _issues(
        "Your sons are 9 (Thodoris) and 5 months old (Leonidas).") == []


def test_an_unambiguous_binding_still_works():
    """The tie rule must not be a blanket mute: two claims, each nearest its
    own subject, are both judged."""
    assert _issues("Thodoris is 9 years old and Leonidas is 5 months old.") == []
    both = _issues("Thodoris is 2 years old and Leonidas is 40 years old.")
    assert len(both) == 2, both
    assert {"Thodoris", "Leonidas"} == {i.split()[0] for i in both}


def test_binding_does_not_cross_a_LINE():
    """⚠ THE THREE FALSE REFUTES MEASURED AGAINST REAL TRAFFIC.

    Distance alone was run over the 1977 replies in the trajectory store and
    produced three hits — every one FALSE, every one a markdown list or table
    where subjects and ages interleave and the nearest name by character
    count belongs to the previous row. These are the live strings, trimmed.

    The world it fails in: binding crosses lines again and the agent writes a
    0.0 on a turn whose arithmetic was correct — noise in the one class the
    corpus cannot afford it in. (After the fix: zero refutations across the
    same 1977 replies, while the contradictions below still fire.)
    """
    listed = ("Got it — using the corrected dates:\n\n"
              "- **Leonidas:** born March 12, 2026 → today (Sep 4, 2026) is "
              "**about 6 months old** (5 months and 3 weeks).\n"
              "- **Thodoris:** born November 25, 2016 → 9 years old.\n")
    assert _issues(listed) == []

    table = ("| Name | Birthdate | Age |\n"
             "| Thodoris | 2016-11-25 | 9 years |\n"
             "| Leonidas | 2026-03-12 | 5 months |\n"
             "| Vasilis | 1980-01-29 | 44 years |\n")
    assert _issues(table) == [], "a row's age bound to another row's subject"

    # And the confinement is not a blanket mute: a contradiction ON the line
    # is still caught.
    #
    # ⚠ Note the phrasing. A bare cell "40 years" is not an age CLAIM — the
    # shared `_AGE_PATTERNS` require "old"/"aged"/"yo"/"mo", so a table of
    # birthdates is doubly out of reach. The first draft of this control
    # asserted on "| … | 40 years |" and failed, which is the control doing
    # its job: it caught the test being wrong, not the code.
    assert _issues("| Leonidas | 2026-03-12 | 40 years old |\n")


def test_a_distant_name_is_not_the_subject():
    """The world it fails in: the binding window is unbounded and a name
    three paragraphs up captures an unrelated number."""
    far = ("Leonidas is doing well. " + "Filler sentence here. " * 8 +
           "The lease is 9 years old.")
    assert _issues(far) == []


# ── one claim is one accusation ────────────────────────────────────────────

def test_overlapping_patterns_report_one_issue_not_two():
    """⚠ The age patterns overlap by design ("9 years old" matches two of
    them). This route writes NEGATIVES, where a duplicated issue is a
    duplicated accusation against the same turn."""
    assert len(_issues("Leonidas is 9 years old.")) == 1
    assert len(_issues("Leonidas is 9-year old.")) == 1


# ── the store walk ─────────────────────────────────────────────────────────

def test_subjects_are_found_wherever_the_anchor_landed():
    """Walks VALUES, not a list of known keys — the anchor lands wherever
    `temporal.anchor` found an age phrase, and a key list goes stale the
    first time a fact is stored somewhere new."""
    assert dict(anchored_subjects(PROFILE)) == {
        "Thodoris": datetime.date(2016, 11, 25),
        "Leonidas": datetime.date(2026, 3, 12)}
    nested = {"a": [{"b": {"c": "Maria (born 2001-02-03)"}}]}
    assert anchored_subjects(nested) == [("Maria", datetime.date(2001, 2, 3))]


def test_an_unattributable_birth_date_yields_no_subject():
    """A date with no name in front of it cannot refute a claim about
    anyone."""
    assert anchored_subjects({"x": "(born 2001-02-03)"}) == []
    assert anchored_subjects({"x": "born 2001-02-03"}) == []


def test_the_checker_is_total():
    """A checker that can break a turn gets traded away the first time it
    does.

    ⚠ THE FIRST VERSION OF THIS TEST WAS VACUOUS. It passed `None`, `42`,
    `object()` — all of which the type checks reject BEFORE any guard, so the
    `except` blocks were never reached and a mutation replacing both with
    `raise` survived the whole suite. A totality claim has to use inputs that
    genuinely raise; these do (verified: `AttributeError` from a non-date
    `now`, `RuntimeError` from the store).
    """
    class _Hostile(dict):
        def values(self):
            raise RuntimeError("store exploded mid-walk")

    # Reaches the walk's own guard.
    assert anchored_subjects(
        _Hostile({"a": "Leonidas (born 2026-03-12)"})) == []
    # Reaches the outer guard, two different ways.
    assert refute_age_claims(reply="Leonidas is 9 years old.",
                             profile=_Hostile({"a": 1}), now=NOW) == []
    assert refute_age_claims(reply="Leonidas is 9 years old.",
                             profile=PROFILE, now="not-a-date") == []
    # And the cheap type rejections still hold.
    for junk in (None, 42, object(), {"a": object()}, [[[None]]]):
        assert refute_age_claims(reply="Leonidas is 9 years old.",
                                 profile=junk, now=NOW) == []
    for reply in (None, 42, object()):
        assert refute_age_claims(reply=reply, profile=PROFILE, now=NOW) == []


# ── the verdict must actually be consumed ──────────────────────────────────

def test_a_tool_free_refutation_is_consumable():
    """⚠ THE CLAUSE THAT WOULD HAVE MADE THIS WHOLE ROUTE DEAD CODE.

    The consumption guard read `last_tool is not None` — false for EVERY
    verdict this route can produce, since it exists for tool-free turns. The
    verdict would have been computed, logged, and consumed by nothing: a
    verifier that runs and changes nothing.

    The world it fails in: the clause is restored, the check still runs, the
    log still says REFUTED, and not one label reaches the corpus.
    """
    refuted = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9)
    assert verdict_is_consumable(refuted, None) is True
    # Tool evidence answers "applicable" on its own, verdict or not.
    assert verdict_is_consumable(None, {"name": "execute"}) is True


def test_a_tool_free_NON_refutation_is_not_consumable():
    """Fails closed on a value from somewhere unaccounted for: the memory
    route never emits CONFIRMED/UNCERTAIN (§4EP), so one arriving with no
    tool is not evidence of applicability."""
    for verdict in (VerifyVerdict.CONFIRMED, VerifyVerdict.UNCERTAIN):
        assert verdict_is_consumable(
            VerifyResult(verdict=verdict, confidence=0.9), None) is False
    assert verdict_is_consumable(None, None) is False


def test_the_refutation_grades_to_a_hard_negative():
    """The end of the chain: a consumed REFUTED becomes `verifier_backfill =
    ("failed", …)` and the label is 0.0 — the scarce class, and the only
    reason this route was built."""
    assert grade_turn_outcome(verifier_verdict="failed") == 0.0
    # And it is a VERDICT row, so §4EO lets it into the fit's evidence.
    from ghost_agent.core.calibration import label_is_verdict
    assert label_is_verdict(grade_turn_outcome(verifier_verdict="failed"))


# ── the gate reaches it (a correct check nobody calls is nothing) ──────────

class _Store:
    def __init__(self, profile=PROFILE):
        self._p = profile

    def load(self):
        return self._p


def _agent(*, verifier_attached=True, profile=PROFILE, no_verifier=False):
    from types import SimpleNamespace
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(
        verifier=SimpleNamespace(
            llm_client=object() if verifier_attached else None),
        profile_memory=_Store(profile),
        args=SimpleNamespace(no_verifier=no_verifier),
        current_project_id=None,
    )
    return a


WRONG = "Leonidas is 9 years old."


@pytest.mark.asyncio
async def test_a_tool_free_turn_reaches_the_arithmetic_check():
    """⚠ A CHECK THE GATE NEVER CALLS IS NOT A CHECK. The early return for
    "no evidence tool" is exactly where the 46% of user turns land; if the
    route is not reached there it is decoration.
    """
    v, last_tool = await _agent()._compute_verifier_verdict(
        tools_run_this_turn=[], messages=[], final_ai_content=WRONG,
        last_user_content="how old are my sons", lc="how old are my sons")
    assert v is not None, "the tool-free turn produced no verdict"
    assert v.verdict == VerifyVerdict.REFUTED
    assert v.confidence >= 0.7, "below the consumption bar — nothing would use it"
    assert last_tool is None
    assert any("Leonidas" in i for i in v.issues)


@pytest.mark.asyncio
async def test_a_tool_free_turn_with_a_CORRECT_claim_still_gets_no_verdict():
    v, _ = await _agent()._compute_verifier_verdict(
        tools_run_this_turn=[], messages=[],
        final_ai_content="Leonidas is 5 months old.",
        last_user_content="how old", lc="how old")
    assert v is None


@pytest.mark.asyncio
async def test_sim_and_ablation_contexts_are_left_alone():
    """⚠ NARROWED TO "no evidence tool WAS THE ONLY REASON". The same early
    return covers sim/ablation and `--no-verifier`; producing verdicts there
    puts labels into an ablation that exists to have none — and sim turns
    share this code path.
    """
    for kwargs in ({"verifier_attached": False}, {"no_verifier": True}):
        v, _ = await _agent(**kwargs)._compute_verifier_verdict(
            tools_run_this_turn=[], messages=[], final_ai_content=WRONG,
            last_user_content="how old", lc="how old")
        assert v is None, kwargs


@pytest.mark.asyncio
async def test_trivial_chat_is_left_alone():
    a = _agent()
    a._is_strict_trivial_chat = lambda lc: True
    v, _ = await a._compute_verifier_verdict(
        tools_run_this_turn=[], messages=[], final_ai_content=WRONG,
        last_user_content="hey", lc="hey")
    assert v is None


@pytest.mark.asyncio
async def test_a_turn_WITH_tool_evidence_does_not_take_this_route():
    """The memory route is the tool-free fallback, not a second opinion: a
    turn with evidence goes to the real verifier, whatever the profile says.
    """
    a = _agent()
    called = []
    a._memory_claim_refutation = lambda t: called.append(t)
    a.context.verifier.llm_client = None      # force the early return
    await a._compute_verifier_verdict(
        tools_run_this_turn=[{"name": "execute", "content": "x" * 300}],
        messages=[], final_ai_content=WRONG,
        last_user_content="run it", lc="run it")
    assert not called, "the memory route ran on a turn that had tool evidence"


def test_a_missing_profile_store_is_not_a_crash():
    from types import SimpleNamespace
    a = _agent()
    a.context.profile_memory = None
    assert a._memory_claim_refutation(WRONG) is None
    a.context.profile_memory = SimpleNamespace(
        load=lambda: (_ for _ in ()).throw(RuntimeError("store down")))
    assert a._memory_claim_refutation(WRONG) is None
