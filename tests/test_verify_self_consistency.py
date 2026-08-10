"""Majority-of-N adjudication — the miss-side lever that does NOT read confidence.

WHY THIS MECHANISM. The verifier's miss side is 0.218 false-CONFIRM, and A1
measured that the judge's own confidence carries NO information about it:
AUC 0.5087, chance. So the usual "act when the model is unsure" lever does not
exist here. Cross-sample DISAGREEMENT is a signal the judge does express and
the confidence field does not.

MEASURED BEFORE BUILDING (2026-08-10) — three identical repeat runs of the real
path, cache off:
  * 2 of 12 trials disagreed across repeats (17% run-to-run variance);
  * BOTH were `artifact_leak` (36% of the miss mass), and in both the
    majority verdict is the CORRECT one;
  * `clean` never varied → the FPR <= 0.121 gate should survive;
  * `fact_swap` never varied and was wrong every time → a SYSTEMATIC error
    this cannot touch. Self-consistency will NOT reach the 0.10 target alone.

⚠ A SHORT SYNTHETIC PROBE SAID THE OPPOSITE — 5/5 identical verdicts at
temperature 0.1, 0.7 AND 1.0 — and taken alone would have killed this as inert.
It was the wrong instrument: a one-line hand-written prompt is not the 5.4KB
adjudication path. Recorded because the cheap probe was the tempting one.

SHIPS DEFAULT-OFF (`GHOST_VERIFY_SELF_CONSISTENCY` unset ⇒ 1 ⇒ byte-identical
to the single-sample path) until a live paired re-bench clears the gates.
"""

import asyncio
import os

import pytest

from ghost_agent.core.verifier import (
    VerifyResult,
    VerifyVerdict,
    Verifier,
    _self_consistency_n,
)


def _r(verdict, conf=0.9, reasoning="r"):
    return VerifyResult(verdict=verdict, confidence=conf,
                        reasoning=reasoning, issues=[])


C, R, U = (VerifyVerdict.CONFIRMED, VerifyVerdict.REFUTED,
           VerifyVerdict.UNCERTAIN)


def _verifier(samples):
    """A Verifier whose adjudication returns `samples` in order."""
    v = Verifier(llm_client=None)
    seq = list(samples)

    async def _call(*a, **k):
        return {"_i": len(seq)}

    def _build(_raw):
        return seq.pop(0) if seq else None

    object.__setattr__(v, "_call_llm", _call)
    object.__setattr__(v, "_build_verify_result", _build)
    return v


def _run(v, n=3):
    return asyncio.run(v._adjudicate_self_consistent("p", n=n,
                                                     force_main=False))


# ── the flag ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,want", [
    (None, 1), ("1", 1), ("3", 3), ("5", 5),
    ("7", 5),            # capped: a typo must not multiply every verify by 50
    ("4", 3),            # even n has no majority
    ("0", 1), ("-2", 1), ("abc", 1), ("", 1),
])
def test_the_flag_parses_conservatively(monkeypatch, val, want):
    if val is None:
        monkeypatch.delenv("GHOST_VERIFY_SELF_CONSISTENCY", raising=False)
    else:
        monkeypatch.setenv("GHOST_VERIFY_SELF_CONSISTENCY", val)
    assert _self_consistency_n() == want


def test_it_is_OFF_by_default(monkeypatch):
    """⚠ Nothing changes in production until a live re-bench clears the gates.
    A mechanism that switches itself on before it is measured is how an
    unmeasured change becomes the baseline."""
    monkeypatch.delenv("GHOST_VERIFY_SELF_CONSISTENCY", raising=False)
    assert _self_consistency_n() == 1


def test_the_single_sample_path_is_untouched_when_off():
    """Structural: n<=1 must take the ORIGINAL call, not a 1-sample vote."""
    import inspect
    src = inspect.getsource(Verifier._verify_claim_two_stage)
    assert "if _n <= 1:" in src
    assert "_adjudicate_self_consistent" in src


# ── the vote ────────────────────────────────────────────────────────────────

def test_the_majority_verdict_wins():
    """THE MEASURED CASE: [CONFIRMED, REFUTED, REFUTED] on an artifact_leak
    trial whose correct answer is REFUTED."""
    out = _run(_verifier([_r(C), _r(R), _r(R)]))
    assert out.verdict == R


def test_a_unanimous_vote_is_unchanged():
    assert _run(_verifier([_r(C), _r(C), _r(C)])).verdict == C


def test_the_winning_sample_is_a_REAL_judgement_not_a_merge():
    """`reasoning`/`issues` must stay coherent — a synthesized blend of three
    judgements is text no judge actually produced."""
    out = _run(_verifier([_r(C, 0.9, "wrong"), _r(R, 0.7, "lo"),
                          _r(R, 0.95, "hi")]))
    assert out.verdict == R and out.reasoning == "hi", (
        "should carry the highest-confidence sample from the winning side")


def test_the_vote_is_recorded_for_attribution():
    """So a bench delta can be attributed to THIS mechanism rather than to
    the judge having a different day."""
    out = _run(_verifier([_r(C), _r(R), _r(R)]))
    assert out.self_consistency_n == 3 and out.self_consistency_agree == 2


# ── degradation must never be worse than today ──────────────────────────────

def test_unparseable_samples_are_DROPPED_not_counted_as_votes():
    """A reply the parser could not read is not a vote for anything. Counting
    it would make the parser's failure rate a hidden thumb on the scale."""
    out = _run(_verifier([None, _r(R), _r(R)]))
    assert out.verdict == R and out.self_consistency_n == 2


def test_a_single_surviving_sample_behaves_like_today():
    """⚠ This must never FAIL a verification that would otherwise succeed."""
    out = _run(_verifier([None, None, _r(C)]))
    assert out is not None and out.verdict == C


def test_all_samples_unparseable_returns_None_as_before():
    assert _run(_verifier([None, None, None])) is None


def test_a_raising_sample_does_not_kill_the_vote():
    """⚠ The fixture counts INVOCATIONS, not remaining items. The first
    version keyed on `len(seq)`, and because the samples run concurrently all
    three saw the same length and all three raised — the test failed on
    correct code. Concurrency makes queue-depth a racy proxy for 'which call
    am I'."""
    v = Verifier(llm_client=None)
    calls = {"n": 0}
    seq = [_r(R), _r(R)]

    async def _call(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("node blipped")
        return {}
    object.__setattr__(v, "_call_llm", _call)
    object.__setattr__(v, "_build_verify_result",
                       lambda _raw: seq.pop(0) if seq else None)
    out = _run(v)
    assert out is not None and out.verdict == R


def test_a_TIE_keeps_todays_verdict_rather_than_letting_ORDER_decide():
    """Ties become possible once samples are dropped. Resolving one by list
    position would make a scheduling detail decide a verification."""
    out = _run(_verifier([_r(C), _r(R)]))
    assert out.verdict == C, "the first sample's verdict must hold on a tie"


# ── cost ────────────────────────────────────────────────────────────────────

def test_samples_are_CONCURRENT_not_sequential():
    """n sequential adjudications would multiply latency by n against the
    120s critic ceiling — measured single-adjudication latency leaves room
    for parallel samples, not for serial ones."""
    import inspect
    src = inspect.getsource(Verifier._adjudicate_self_consistent)
    assert "gather" in src, "samples run serially; the ceiling will be hit"
