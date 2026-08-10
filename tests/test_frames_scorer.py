"""Grading FRAMES — and the instrument failure that made this file necessary.

MEASURED 2026-08-10, in the readiness smoke, before any number was published.
The plan was to reuse the vendored GAIA scorer. On two tasks it reported
accuracy **0.0**, and BOTH answers were semantically correct:

    truth "Mulona barnesi and mulona schausi" / answer "Mulona barnesi, Mulona schausi"
    truth "5 minutes and 31 seconds"          / answer "5:31"

Not a bug in the GAIA scorer. It assumes GAIA's MANDATED answer format on both
sides (comma lists, no units, digits plain); FRAMES ground truth is ordinary
natural language, and FRAMES' own paper grades with an LLM judge. Exact match
was never the protocol. Had the full run gone ahead, it would have produced a
near-zero score that measured the ruler and nothing else.

⚠ THE DISCIPLINE THAT SHAPES THIS FILE. The obvious fix is normalisation rules
— treat " and " as a comma, canonicalise durations, strip units. Every such
rule would have been written by staring at MY agent's failures, which makes it
a scorer tuned to raise my agent's score. Exactly ONE normalisation is kept,
because it is defensible without reference to any failure: " and " is the
natural-language form of the separator GAIA already splits on. Everything else
goes to a judge that is a DIFFERENT model family from the answerer, is told to
say NO when unsure, and is itself validated against hand labels before any
headline number is quoted.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from frames_scorer import (  # noqa: E402
    judge_match,
    rescore_details,
    score_one,
    strict_match,
    validate_judge,
)


def _ask(reply):
    return lambda prompt: reply


# ── the strict pre-pass ─────────────────────────────────────────────────────

def test_the_measured_smoke_failure_is_fixed():
    """THE CASE THAT CAUGHT IT."""
    assert strict_match("Mulona barnesi, Mulona schausi",
                        "Mulona barnesi and mulona schausi") is True


def test_list_order_does_not_matter():
    assert strict_match("B and A", "A, B") is True


def test_a_genuinely_wrong_answer_is_still_wrong():
    """⚠ THE ANTI-FLATTERY GUARD. Loosening a scorer until everything passes
    is the failure mode this whole file is written against."""
    assert strict_match("Someone else", "Jane Ballou") is False
    assert strict_match("Mulona barnesi", "Mulona barnesi and mulona schausi") is False
    assert strict_match("", "Jane Ballou") is False
    assert strict_match(None, "Jane Ballou") is False


def test_a_partial_list_is_not_a_match():
    """Dropping half the reference is a miss, not a formatting difference."""
    assert strict_match("A", "A and B") is False
    assert strict_match("A and B", "A, B, C") is False


def test_strict_does_not_reach_semantic_equivalence():
    """Honest about its own limit — this is what the judge is FOR. A string
    rule that cracked this would be a rule invented to pass this case."""
    assert strict_match("5:31", "5 minutes and 31 seconds") is False


# ── the judge ───────────────────────────────────────────────────────────────

def test_a_yes_is_a_match():
    ok, _ = judge_match("5:31", "5 minutes and 31 seconds", "q?", _ask("YES"))
    assert ok is True


def test_a_no_is_not_a_match():
    ok, _ = judge_match("5:31", "9 hours", "q?", _ask("NO"))
    assert ok is False


@pytest.mark.parametrize("reply", ["", "   ", "maybe", "YES and NO",
                                   "I cannot determine this", "42"])
def test_an_UNCLEAR_judge_reply_counts_as_NO(reply):
    """⚠ CONSERVATIVE BY CONSTRUCTION. A scoreboard that errs must err
    DOWNWARD: an understated number invites a re-measure, an overstated one
    gets quoted. Treating an unparseable reply as a match would inflate the
    score by exactly the judge's own failure rate — the instrument's defects
    silently becoming the agent's credit."""
    ok, why = judge_match("a", "b", "q?", _ask(reply))
    assert ok is False and why


def test_a_judge_that_RAISES_counts_as_NO():
    """A judge that errored graded nothing."""
    def boom(_):
        raise RuntimeError("node down")
    ok, why = judge_match("a", "b", "q?", boom)
    assert ok is False and "judge error" in why


def test_the_verdict_is_read_from_the_END_of_a_chatty_reply():
    """A thinking model restates the question before answering; matching the
    first YES/NO anywhere would grade the restatement."""
    ok, _ = judge_match("a", "b", "q?", _ask(
        "The candidate says no such thing... Let me compare.\nYES"))
    assert ok is True


def test_every_judgement_carries_its_reason():
    """A run has to be auditable after the fact without re-running it."""
    _, why = judge_match("a", "b", "q?", _ask("NO — different entity"))
    assert "different entity" in why


# ── strict first, judge only on the residual ────────────────────────────────

def test_the_judge_is_NOT_consulted_when_strict_already_matched():
    """Determinism and cost: a string equality must not become a model's
    opinion. If the judge ran here, a model mood could flip an exact match."""
    def must_not_run(_):
        raise AssertionError("judge called on a strict match")
    assert score_one("A and B", "A, B", "q?", must_not_run)["correct"] is True


def test_without_a_judge_the_residual_is_simply_wrong():
    """Degrades to a reproducible LOWER BOUND rather than crashing — and says
    so in `how`, so the number is never mistaken for a judged one."""
    r = score_one("5:31", "5 minutes and 31 seconds", "q?", ask=None)
    assert r["correct"] is False and r["how"] == "strict"


def test_the_grading_METHOD_is_recorded_per_answer():
    """How much of a score came from string equality vs a model's opinion is
    the first thing a sceptical reader should be able to check."""
    assert score_one("A, B", "A and B", "q?")["how"] == "strict"
    assert score_one("x", "y", "q?", _ask("YES"))["how"] == "judge"


# ── an absent answer is never a match ───────────────────────────────────────

@pytest.mark.parametrize("empty", ["", "   ", "\n", None])
def test_an_EMPTY_answer_never_reaches_the_judge(empty):
    """⚠⚠ MEASURED ON THE FIRST REAL RUN, 2026-08-10, and it inflated the
    headline by five points.

    Asked to compare ground truth "10.81" against an EMPTY candidate, the
    judge replied **YES**. All three no-answer tasks (2 empty replies + 1
    ReadTimeout) scored CORRECT: 0.767 became 0.817 — pure credit for
    answering nothing, in precisely the direction this scorer exists to never
    err in.

    So it short-circuits. "Did the agent answer at all" is a definition, not
    something to delegate to a model's opinion.
    """
    def must_not_run(_):
        raise AssertionError("the judge was consulted about an empty answer")
    r = score_one(empty, "10.81", "How many?", must_not_run)
    assert r["correct"] is False and r["how"] == "empty"


def test_a_whitespace_answer_is_not_an_answer():
    assert score_one("  \t \n ", "Five", "q?", _ask("YES"))["correct"] is False


def test_the_empty_case_is_reported_as_its_own_method():
    """Distinguishable in the breakdown from a genuine wrong answer: 'the
    agent produced nothing' and 'the agent was wrong' are different failures
    and a scoreboard that merges them hides an infrastructure problem."""
    assert score_one("", "x", "q?")["how"] == "empty"
    assert score_one("wrong", "x", "q?")["how"] == "strict"


def test_a_nonempty_answer_still_reaches_the_judge():
    """⚠ OVER-SUPPRESSION GUARD: the empty short-circuit must not swallow
    real answers on their way to semantic grading."""
    assert score_one("5:31", "5 minutes and 31 seconds", "q?",
                     _ask("YES")) == {"correct": True, "how": "judge",
                                      "reason": "YES"}


# ── validating the judge before quoting it ──────────────────────────────────

def test_validate_judge_reports_the_INFLATING_direction():
    """False positives are the direction that raises the headline, so they
    get their own rate rather than being averaged into 'agreement'."""
    labelled = [
        {"question": "q", "ground_truth": "a", "model_answer": "zzz", "correct": False},
        {"question": "q", "ground_truth": "b", "model_answer": "yyy", "correct": False},
    ]
    out = validate_judge(labelled, _ask("YES"))      # judge says match to all
    assert out["false_pos"] == 2 and out["false_positive_rate"] == 1.0
    assert out["agreement"] == 0.0


def test_validate_judge_scores_a_perfect_judge():
    labelled = [{"question": "q", "ground_truth": "a", "model_answer": "a",
                 "correct": True}]
    out = validate_judge(labelled, _ask("NO"))       # strict catches it first
    assert out["agreement"] == 1.0 and out["false_pos"] == 0


# ── re-grading without re-running the agent ─────────────────────────────────

def test_details_can_be_rescored_without_rerunning(tmp_path):
    """⚠ The agent's answers are the expensive artifact; grading is cheap and
    has already been wrong once. Decoupling them means the NEXT scorer fix
    costs zero agent time."""
    p = tmp_path / "details.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"question": "q1", "ground_truth": "A and B",
         "model_answer": "B, A", "correct": False},
        {"question": "q2", "ground_truth": "Jane", "model_answer": "Bob",
         "correct": False},
    ]))
    out = rescore_details(p)
    assert out["n"] == 2 and out["n_correct"] == 1
    assert out["accuracy"] == 0.5
    assert out["rows"][0]["correct"] is True
    assert out["rows"][1]["correct"] is False


def test_rescoring_an_empty_file_does_not_claim_an_accuracy(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("")
    assert rescore_details(p)["accuracy"] is None


# ── the two switches that make the judge exist at all ───────────────────────

def test_BOTH_no_think_switches_are_sent(monkeypatch):
    """⚠⚠ MEASURED 2026-08-10 AND LOAD-BEARING.

    Without these the critic node returns EMPTY content with
    finish_reason='length' at every max_tokens tried — the budget goes to
    thinking tokens that never surface. Every judgement then parsed as
    unreadable and, per the conservative rule, became NO.

    Why this needs a test rather than a comment: the broken state looked
    PLAUSIBLE. Validation reported 0 false positives and agreement 0.69,
    which reads as "the small judge is a bit weak". It was not weak, it was
    absent — `strict_match` was doing all of the work. With both switches,
    agreement on the same 16 hand-labelled pairs is 1.0 (0 FP / 0 FN).

    Remove them and the scoreboard does not fail loudly; it silently reports
    a much lower number that reads as the AGENT regressing. That is the most
    expensive kind of instrument bug this project keeps finding.
    """
    import json as _json

    sent = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps(
                {"choices": [{"message": {"content": "YES"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent["body"] = _json.loads(req.data.decode())
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    from frames_scorer import nova_judge_ask
    assert nova_judge_ask()("grade this") == "YES"

    body = sent["body"]
    assert body["messages"][0]["content"].endswith("/no_think"), (
        "soft switch missing — the judge will return empty content and every "
        "answer will grade NO")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, (
        "hard switch missing — either switch alone has been observed "
        "insufficient on this node")
    assert body["temperature"] == 0, "a graded score must be reproducible"
