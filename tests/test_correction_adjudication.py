"""Verifier coverage: the corrections tier, and what "coverage" has to mean.

§4EP, 2026-09-04. Follows §4EO's open item — "63% of turns produce no
checkable verdict". Two things were measured before anything was built:

  1. **Coverage alone is the wrong target.** Simulated on the live verdict
     population (n=402, 57 negatives): quadrupling the rows with CONFIRM-only
     labels leaves the base-rate comparison unresolvable at every size,
     because confirms shrink the observed effect as fast as they shrink the
     interval. The SAME rows at the existing 14% negative rate become
     resolvable at 2x. So a route that can only confirm buys nothing.
  2. **The strongest negative tier is nearly dead.** `user_correction` is
     ground truth (the human said the answer was wrong) and produced ZERO
     rows in the whole epoch. Over 1601 consecutive live turn pairs the
     promotion rule fired twice; 12 more cleared Signal A and were blocked by
     a Jaccard rephrase test that provably cannot separate them.

The property under review: **a correction is promoted when the user says the
answer was wrong, and never merely because the message starts with "no".**
"""

import logging
import random

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.distill import user_correction as U
from ghost_agent.distill.user_correction import (adjudication_prompt,
                                                 classify_user_correction,
                                                 has_correction_phrase,
                                                 parse_adjudication)

# The 12 live messages that cleared Signal A and were blocked, hand-labelled
# from the trajectory store. These are verbatim user messages, not fixtures
# invented to suit the rule.
REAL_CORRECTIONS = [
    "no it's not fragmented at all, he sleeps all night.",
    "that's not true. you have many mechanisms that work when i'm not around",
    "No, that's wrong. You made a mistake there. The correct answer is Canberra.",
    "no , i mean humans would definitely categorize you as an alien \"being\"",
    "no i mean literal torture",
    "no I was asking about the mysql problem",
]
NOT_CORRECTIONS = [
    "nope, nothing for now, mark the project as done , if it's not already.",
    "no, tell me what would help you answer your question .. let's say that i can grant you a wish",
    "no let's change topic, do you think that covid vaccines were harmful?",
    "no go ahead, i give you the permission to do whatever you want",
    "no i'm back at home right now",
    "Actually, you remember now — Chess Coach v3 was just a stepping stone.",
]


# ── (1) the target itself ──────────────────────────────────────────────────

def test_confirm_only_coverage_destroys_the_effect_it_would_measure():
    """⚠ MEASURE THE THING COVERAGE WAS SUPPOSED TO FIX.

    The world this fails in: someone ships a verification route that can only
    CONFIRM, watches the coverage percentage climb, and reports the lever
    fixed — while the base-rate comparison it was for gets strictly WORSE.
    A weak success signal standing in for the real one.

    The mechanism, in the live regime (base rate 0.851, weak separation,
    n=402): tripling the corpus with majority-class labels shrinks the
    observed delta by 2-16x across seeds, because every added row is one the
    base-rate predictor already gets right. Growth at the EXISTING class mix
    leaves it put.

    ⚠ Six seeds, not one. The first version of this pin asserted the
    stronger claim "same-mix growth makes the comparison RESOLVABLE" off a
    single bootstrap of the live corpus; across seeds that replicates 1 time
    in 3 and is not a finding. What survives every seed is the collapse
    below, so that is what is pinned.
    """
    ratios = []
    for seed in range(1, 7):
        rng = random.Random(seed)
        base = []
        for _ in range(402):
            good = rng.random() < 0.851
            c = min(0.99, max(0.01,
                              rng.gauss(0.80 + (0.08 if good else -0.08), 0.26)))
            base.append((c, 1.0 if good else 0.0))
        pos = [r for r in base if r[1] >= 0.5]
        d0 = abs(C._cv_delta_ci(base)[0])
        confirms = base + [rng.choice(pos) for _ in range(len(base) * 3)]
        dc = abs(C._cv_delta_ci(confirms)[0])
        mixed = base + [rng.choice(base) for _ in range(len(base) * 3)]
        dm = abs(C._cv_delta_ci(mixed)[0])
        ratios.append((d0 / max(dc, 1e-9), d0 / max(dm, 1e-9)))
    for i, (confirm_ratio, mix_ratio) in enumerate(ratios, 1):
        assert confirm_ratio > 1.5, (
            f"seed {i}: confirm-only growth left the effect intact "
            f"({confirm_ratio:.2f}x) — the premise of §4EP's design")
        assert mix_ratio < 1.5, (
            f"seed {i}: same-mix growth collapsed the effect too "
            f"({mix_ratio:.2f}x), so the contrast this pin rests on is gone")


# ── (2) the phrase gate, one authority ─────────────────────────────────────

def test_the_phrase_gate_fires_on_every_candidate_and_stays_cheap():
    """It is the cost gate for the judge: everything that could ever be
    promoted must clear it, and ordinary chat must not."""
    for msg in REAL_CORRECTIONS + NOT_CORRECTIONS:
        assert has_correction_phrase(msg), msg
    for msg in ("thanks, that worked", "what's the weather",
                "run the tests please", "", None,
                "I think the answer is no, but let's check"):
        assert not has_correction_phrase(msg), repr(msg)


def test_the_gate_and_the_promotion_read_the_same_regex():
    """The world it fails in: the caller keeps a private copy of the opener
    test, the two drift, and the judge is spent on messages that can never be
    promoted (or skipped on ones that can)."""
    for msg in REAL_CORRECTIONS + NOT_CORRECTIONS:
        v = classify_user_correction(prev_user_request="anything",
                                     prev_assistant_response="anything",
                                     current_user_text=msg,
                                     contradicts=True)
        assert "phrase" in v.signals, msg
        assert v.is_correction, msg


# ── (3) the adjudicated signal decides, and fails closed ───────────────────

def test_a_judge_that_says_no_blocks_a_lexical_promotion():
    """⚠ ASK THE VERDICT ONCE. A semantic read of the actual reply outranks
    a token-overlap guess about the prior REQUEST.

    The world it fails in: both signals are consulted and either may promote,
    so "no let's change topic — do you think the covid vaccines were
    harmful?" is stamped a correction because it happens to reuse the prior
    request's words.
    """
    kw = dict(prev_user_request="do you think the covid vaccines were harmful",
              prev_assistant_response="I don't think they were harmful.",
              current_user_text="no let's change topic, do you think that "
                                "covid vaccines were harmful?")
    lexical = classify_user_correction(**kw)
    assert lexical.is_correction, (
        "fixture no longer trips the lexical rule it exists to override")
    judged = classify_user_correction(**kw, contradicts=False)
    assert not judged.is_correction
    assert "adjudicated(no)" in judged.signals


def test_a_judge_that_says_yes_promotes_what_the_lexical_rule_blocked():
    """The recall half: 6 of these 6 are real corrections the old rule threw
    away."""
    for msg in REAL_CORRECTIONS:
        kw = dict(prev_user_request="what is the capital of australia",
                  prev_assistant_response="Sydney is the capital.",
                  current_user_text=msg)
        assert not classify_user_correction(**kw).is_correction, (
            f"fixture already promoted without a judge: {msg}")
        assert classify_user_correction(**kw, contradicts=True).is_correction, msg


def test_no_judge_means_exactly_the_old_behaviour():
    """⚠ `None` IS NOT `False`. Every failure path — no verifier, no client,
    timeout, unparseable reply — returns None, so the change can only ADD
    promotions and can never silently remove one.

    The world it fails in: None is coerced to False, a judge outage
    suppresses the two promotions the lexical rule was still making, and the
    only ground-truth negative tier goes fully dark without a log line.
    """
    kw = dict(prev_user_request="sort the users by signup date",
              prev_assistant_response="Here is the sorted list.",
              current_user_text="no, sort the users by signup date descending")
    old = classify_user_correction(**kw)
    assert old.is_correction, "fixture no longer exercises the lexical path"
    assert classify_user_correction(**kw, contradicts=None).is_correction
    assert not classify_user_correction(**kw, contradicts=False).is_correction


def test_the_affirmation_veto_still_wins_over_a_yes_judge():
    """Praise that opens like a correction must not be promoted, whatever the
    judge said — the veto guards a class of self-poisoning the judge was
    never asked about."""
    v = classify_user_correction(
        prev_user_request="write a sort function",
        prev_assistant_response="def sort(x): ...",
        current_user_text="actually the sort you wrote works great, thanks",
        contradicts=True)
    assert not v.is_correction
    assert "affirmation-veto" in v.signals


# ── (4) the judge's own plumbing ───────────────────────────────────────────

def test_the_parser_distinguishes_a_ruling_from_a_non_answer():
    assert parse_adjudication('{"corrects": true}') is True
    assert parse_adjudication('{"corrects": false}') is False
    assert parse_adjudication('reasoning… {"corrects": true} done') is True
    # NOT rulings — every one must be None so the lexical fallback survives.
    for junk in ("", None, "yes", "{}", '{"corrects": "true"}',
                 '{"corrects": 1}', "{not json}", '{"other": true}', 42):
        assert parse_adjudication(junk) is None, repr(junk)


def test_the_prompt_carries_the_three_inputs_and_the_distinctions():
    p = adjudication_prompt(prev_user_request="PREVREQ",
                            prev_assistant_response="PREVANS",
                            current_user_text="NEXTMSG")
    for token in ("PREVREQ", "PREVANS", "NEXTMSG"):
        assert token in p
    # The false-positive classes found in live traffic must be named, or the
    # judge re-derives them per call.
    for distinction in ("changing the subject", "answering a question",
                        "granting permission", "past"):
        assert distinction in p, distinction
    assert "corrects" in p


# ── (5) the skip message names the cause it actually observed ──────────────

class _Ctx:
    def __init__(self, verifier=None):
        self.verifier = verifier
        self.trajectory_collector = None


def _agent():
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = _Ctx()
    return a


@pytest.mark.parametrize("n_tools,needle", [
    (0, "ran NO tools"),
    (3, "ran 3 tool(s) but none carried"),
    (None, "tool count not captured"),
])
def test_the_skip_message_names_the_cause_it_observed(monkeypatch, n_tools,
                                                      needle):
    """⚠ THE MESSAGE NAMED THE WRONG CAUSE.

    One line said "bookkeeping-only tools" for EVERY evidence-free turn.
    Measured live: of 73 user turns skipped there, **71 ran no tools at all**
    and 2 were bookkeeping — the operator's only view of the largest coverage
    gap in the corpus described a cause accounting for 3% of it.

    The world it fails in: the next person to ask "why is coverage low" reads
    the log, goes hunting bookkeeping tools, and finds nothing — which is
    exactly how this took log archaeology to diagnose.

    `None` is its own message on purpose: a call site that forgets to pass
    the count must not report "ran NO tools" about a turn it never counted.

    Captures `pretty_log` at the emitting module rather than through the
    logging tree: this pins WHAT the code says, not how a mirror handler
    happens to be wired in the test process.
    """
    from ghost_agent.core import agent as A
    seen = []
    monkeypatch.setattr(A, "pretty_log",
                        lambda title, content=None, **kw: seen.append(
                            f"{title}: {content}"))
    _agent()._record_late_verdict(None, "traj-1", n_tools=n_tools)
    text = "\n".join(seen)
    assert needle in text, text
    assert "bookkeeping-only tools" not in text


def test_every_spawn_site_passes_the_tool_count():
    """⚠ ENUMERATE FROM THE AST, NOT FROM A LIST I REMEMBER.

    Two call sites spawn the late-verdict handler. A third added without the
    count would silently report "tool count not captured" forever — honest,
    but blind. This fails the moment a spawn site omits it.
    """
    import ast
    import inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    sites = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_attach_late_verdict_handler"]
    assert sites, "the spawn site was renamed — this enumeration is now blind"
    missing = [n.lineno for n in sites
               if not any(k.arg == "n_tools" for k in n.keywords)]
    assert not missing, (
        f"_attach_late_verdict_handler called without n_tools at lines "
        f"{missing} — that turn's skip reason will read 'not captured'")


def test_a_judge_alone_cannot_promote_without_the_opener():
    """Signal A is required, not merely the thing that pays for the judge.

    In the live wiring `contradicts` can only be non-None when the phrase
    already fired, so dropping this conjunct is EQUIVALENT today — which is
    exactly why it needs a pin rather than an argument. `classify_user_
    correction` is a public pure function; the next caller to compute a
    contradiction verdict some other way (a batch re-label, an eval harness)
    would otherwise promote every disagreement in the corpus.

    The world it fails in: "the deploy actually finished at 4pm, not 3pm"
    contradicts the prior answer and is promoted as a CORRECTION of it,
    stamping the turn FAILED and writing a 0.0 — when the user is just
    adding a fact.
    """
    v = classify_user_correction(
        prev_user_request="when did the deploy finish",
        prev_assistant_response="It finished at 3pm.",
        current_user_text="the deploy actually finished at 4pm",
        contradicts=True)
    assert "phrase" not in v.signals, (
        "fixture opener now trips Signal A — pick one that does not")
    assert not v.is_correction


def test_the_phrase_gate_reads_only_the_message_opening():
    """⚠ PROVEN-EQUIVALENT MUTANT, PINNED AT THE PROPERTY INSTEAD.

    Deleting the `[:240]` head slice does not change behaviour: the regex is
    `^`-anchored without MULTILINE, so it can only ever match at position 0
    (verified: a phrase 3000 chars in does not match, and neither does one on
    line 2). The truncation is a documented second line of defence, so the
    thing worth pinning is the PROPERTY it defends, not the slice.

    The world this fails in: someone adds `re.MULTILINE` or switches to
    `match` semantics on each line, and every long message containing "no"
    at the start of some paragraph becomes a correction candidate.
    """
    assert not has_correction_phrase("x" * 3000 + " no, that is wrong")
    assert not has_correction_phrase("here is my answer\nno, that is wrong")
    # Leading whitespace is still an opener — `^\s*` covers it.
    assert has_correction_phrase("\n\n  no, that is wrong")
