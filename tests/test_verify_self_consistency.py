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

THE GLOBAL FLAG SHIPS OFF (`GHOST_VERIFY_SELF_CONSISTENCY` unset ⇒ 1 ⇒
byte-identical to the single-sample path) — and that is still true, but it no
longer means "nothing changes in production".

Since §4BR (2026-08-16) the mechanism IS live on part of real traffic: the
`verify_depth` arm raises n to 3 on turns the complexity router calls
confidently hard, for the treatment half. The env flag remains the global
default and the floor; the arm is a separate, randomised, killable
(`GHOST_VERIFY_DEPTH_ROUTING=0`) route to the same mechanism, pre-registered
at system/eval/verify_depth/DECISION_RULE.md. Anyone reading this file to
answer "can this affect users yet" must read that as YES for ~29% of turns
(58% trigger rate x 50% arm).
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


def test_the_GLOBAL_default_is_OFF(monkeypatch):
    """The unrouted path must stay byte-identical to the single-sample one.

    ⚠ This is NOT the same claim as "nothing changes in production" — the
    §4BR verify_depth arm reaches this mechanism through `deep`, on ~29% of
    real turns. See `test_a_confident_hard_turn_gets_majority_of_three` in
    tests/test_verify_depth_routing.py for the routed half."""
    monkeypatch.delenv("GHOST_VERIFY_SELF_CONSISTENCY", raising=False)
    assert _self_consistency_n() == 1
    assert _self_consistency_n(deep=False) == 1


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
    assert out.verdict == R
    assert out.reasoning in ("lo", "hi"), (
        "must be one of the actual samples on the winning side, not a merge")


def test_confidence_is_NOT_an_order_statistic_in_either_direction():
    """TWO wrong rules preceded this one, and the second shipped AS the fix
    for the first (§4BR R17 MAJOR-5 → R18 MAJOR-1).

    v1 took the MOST confident agreeing sample: max-of-k, which lifts
    verdicts over the 0.7 action threshold a single sample would have left
    below it. v2 took the "lower median", which sounds conservative but is
    the MINIMUM whenever k=2 — and k=2 is the modal case, because the vote
    early-stops as soon as the first two agree. Min-of-2 is stochastically
    smaller than one sample (Beta(8,2): P(conf >= 0.7) falls 0.804 → 0.646),
    so v2 SUPPRESSED corrections exactly as reliably as v1 manufactured
    them. Same defect, opposite sign.

    0.7 is not cosmetic: it gates the correction note appended to the reply,
    the in-loop repair round, and the passed/failed outcome label the whole
    learning stack trains on. Any rule that picks by RANK is a change to
    that gate wearing a vote's clothing.

    Draw order has no rank in it: sample 1 is drawn exactly as control's
    single sample is.
    """
    # Unanimous — the common case. The result must be sample 1 itself, i.e.
    # byte-identical to what control would have returned.
    first = _r(R, 0.95, "first")
    out = _run(_verifier([first, _r(R, 0.6, "second"), _r(C, 0.9, "third")]))
    assert out is first, "a unanimous vote must return control's own sample"
    assert out.confidence == 0.95

    # And the mirror: the same shape with the confidences swapped must give
    # the same POSITION, not the same value. A max rule passes the case
    # above and fails this one; a min rule fails the case above.
    first_lo = _r(R, 0.6, "first")
    out2 = _run(_verifier([first_lo, _r(R, 0.95, "second"), _r(C, 0.9, "x")]))
    assert out2 is first_lo
    assert out2.confidence == 0.6


def test_a_LOSING_first_sample_yields_to_the_first_AGREEING_one():
    """When sample 1 loses the vote its confidence goes with it — otherwise
    the returned object would carry one sample's verdict and another's
    reasoning, which is the synthetic merge this design rejects."""
    out = _run(_verifier([_r(C, 0.9, "loser"), _r(R, 0.8, "winner-a"),
                          _r(R, 0.55, "winner-b")]))
    assert out.verdict == R
    assert out.reasoning == "winner-a" and out.confidence == 0.8


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

def test_samples_are_SEQUENTIAL_not_concurrent():
    """⚠ THIS TEST USED TO ASSERT THE OPPOSITE, on reasoning alone.

    It pinned `asyncio.gather` because "n sequential adjudications would
    multiply latency by n". Timed against the live critic on the real
    2,285-token adjudication prompt (§4BR R17), that is backwards:

        1 sample  ........ 10.8s
        3 sequential ..... 17.8s   (10.3 / 2.7 / 4.7)
        3 concurrent ..... 24.9s

    llama.cpp keeps a per-slot prompt cache, so repeating the SAME prompt
    on the same slot skips the prefill; concurrent samples are dealt to
    different slots and each pays the full 2.3k-token prefill again. The
    parallelism was being bought with cache misses — sequential is 1.4x
    faster, and the number that mattered was never measured until a review
    asked for it.
    """
    import ast
    import inspect
    import textwrap
    src = textwrap.dedent(
        inspect.getsource(Verifier._adjudicate_self_consistent))
    # CODE only. The docstring above explains what was removed and why, and
    # asserting over raw text made that explanation read as the removed
    # mechanism still being there.
    tree = ast.parse(src)
    body = tree.body[0].body
    if (isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)):
        body = body[1:]
    code = "\n".join(ast.unparse(n) for n in body)
    assert "gather" not in code, (
        "concurrent samples defeat the per-slot prompt cache — measured "
        "24.9s vs 17.8s sequential")
    assert "for " in code, "the samples must be drawn in a loop"


def _counting_verifier(samples, route="critic"):
    """A Verifier that records how many adjudication samples were DRAWN."""
    v = Verifier(llm_client=None)
    seq = list(samples)
    drawn = []

    async def _call(*a, **k):
        drawn.append(1)
        if k.get("route_out") is not None:
            k["route_out"]["route"] = route
        return {"x": 1}

    object.__setattr__(v, "_call_llm", _call)
    object.__setattr__(v, "_build_verify_result",
                       lambda _raw: seq.pop(0) if seq else None)
    return v, drawn


def test_the_vote_EARLY_STOPS_once_it_is_decided():
    """The common case is unanimity, and at n=3 a third sample cannot
    change a 2-0 lead. Drawing it anyway is a full extra adjudication of
    pure latency on every hard turn — the difference between ~13.0s and
    17.8s measured, i.e. between '+2s over a single sample' and '+7s'."""
    v, drawn = _counting_verifier([_r(R), _r(R), _r(C)])
    out = _run(v)
    assert out.verdict == R
    assert len(drawn) == 2, f"drew {len(drawn)} samples; the 3rd cannot flip 2-0"


def test_a_DISAGREEMENT_still_draws_the_decider():
    """The early stop must not truncate a vote that is genuinely open —
    that would silently turn majority-of-3 into first-past-the-post."""
    v, drawn = _counting_verifier([_r(C), _r(R), _r(R)])
    out = _run(v)
    assert len(drawn) == 3
    assert out.verdict == R


def test_sampling_STOPS_when_the_adjudication_degrades_onto_the_MAIN_model():
    """§4BR R17 MAJOR-6. `_call_llm` falls critic → worker → MAIN, and live
    the critic and worker pools are THE SAME BOX — so one node outage sends
    every sample to the 35B, where they serialize on the single foreground
    inference slot at up to 90s each: 270s of the scarcest resource on the
    machine per verdict, to compute a majority nobody asked for.

    Sampling k times is only affordable because it runs off-host. When it
    stops being off-host, it stops."""
    v, drawn = _counting_verifier([_r(R), _r(R), _r(R)], route="main")
    out = _run(v)
    assert len(drawn) == 1, f"drew {len(drawn)} samples on the MAIN model"
    assert out is not None and out.verdict == R, (
        "degrading must fall back to single-sample semantics, not to None")


class TestTheVoteSurvivesTheEscalationLadder:
    """§4BR R19 CRITICAL-1 — and the third round running in which a FIX
    carried the same defect class as the thing it fixed.

    R18 promoted the vote counters from dynamic attributes to dataclass
    fields "so they reach disk", and stopped one layer short of where they
    are read. Between the cheap-leg vote and the sidecar, `verify_claim`
    runs the mechanical guard and both escalations, and FIVE sites there
    build a REPLACEMENT VerifyResult. Each already carries `suspects` and
    `probe_score` forward by hand; none carried the new fields.

    The bias is not random. Every REFUTED verdict enters `_escalate_refute`
    (17% of decided verdicts live, 47/59 escalation-ledger rows end at a
    newly built object), and the 2026-08-10 measurement this arm rests on
    found BOTH contested trials were `artifact_leak` with a REFUTED
    majority. So the counters were being deleted on exactly the turns that
    demonstrate the mechanism working — and because the sidecar omits the
    keys when they are None, such a turn is indistinguishable from a control
    turn that never voted. Gate 0 would have read a sample biased toward
    unanimity and retired the arm for being inert.

    `probe_score` had this identical bug, fixed the same way in §4BF.
    """

    @staticmethod
    def _verifier(cheap_verdicts, main_verdict):
        """Real `_build_verify_result` and real ladder; only the wire is faked.

        ⚠ The client MUST advertise `critic_clients`. `_escalate_refute`
        returns immediately when no cheap route exists ("the main model
        already judged it"), so a `llm_client=None` fixture skips the entire
        ladder — which is how the first version of this test passed
        identically with and without the fix it exists to pin.
        """
        class _Client:
            critic_clients = [object()]

        v = Verifier(llm_client=_Client())
        state = {"cheap": list(cheap_verdicts)}

        async def _call(prompt, *a, **k):
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            suspects = [{"quote": "q", "check": "support", "reason": "r"}]
            if k.get("force_main"):
                return {"suspects": suspects, "verdict": main_verdict,
                        "confidence": 0.9, "reasoning": "main", "issues": []}
            if "suspect" in prompt.lower() and state.get("stage1_done") is None:
                state["stage1_done"] = True
                return {"suspects": suspects}
            nxt = (state["cheap"].pop(0) if state["cheap"] else "UNCERTAIN")
            return {"suspects": suspects, "verdict": nxt, "confidence": 0.8,
                    "reasoning": "cheap", "issues": ["i"]}

        object.__setattr__(v, "_call_llm", _call)
        return v

    def _final(self, cheap, main, **kw):
        import asyncio
        v = self._verifier(cheap, main)
        return asyncio.run(v.verify_claim("c", "e", "ctx", deep=True, **kw))

    def test_an_OVERTURNED_refute_still_reports_its_vote(self):
        """The headline case: a contested vote lands REFUTED, the main model
        overturns it, and a brand-new result object ships."""
        out = self._final(["CONFIRMED", "REFUTED", "REFUTED"], "CONFIRMED")
        assert out is not None
        assert out.self_consistency_n == 3, (
            "the escalation replaced the object and dropped the vote")
        assert out.self_consistency_agree == 2
        assert out.self_consistency_drawn == 3

    def test_an_UPHELD_refute_still_reports_its_vote(self):
        out = self._final(["CONFIRMED", "REFUTED", "REFUTED"], "REFUTED")
        assert out is not None and out.self_consistency_n == 3

    def test_the_counters_reach_to_dict_after_escalation(self):
        """to_dict is the shape a future consumer will read."""
        out = self._final(["CONFIRMED", "REFUTED", "REFUTED"], "CONFIRMED")
        d = out.to_dict()
        assert d.get("self_consistency_n") == 3
        assert d.get("self_consistency_agree") == 2
        assert d.get("self_consistency_drawn") == 3, (
            "sc_drawn must survive to_dict too — it is what separates "
            "'the vote stopped early' from 'the judge returned garbage'")

    def test_an_ALL_UNPARSEABLE_vote_is_still_recorded_as_a_vote(self):
        """§4BR R20 MAJOR-2 — the loss that happened UPSTREAM of R19's fix.

        When every adjudication sample fails to parse, `_verify_claim_two_stage`
        returns None and `verify_claim` falls back to the classic single
        prompt. The result that ships was never touched by the vote, so a
        turn that paid for three samples filed as a control turn that never
        voted — and this is precisely the "judge returned garbage" case that
        `self_consistency_drawn` exists to separate from "the vote stopped
        early". R19's snapshot read the result OBJECT, which covered the
        five replacement sites downstream and nothing upstream.
        """
        import asyncio

        class _Client:
            critic_clients = [object()]

        v = Verifier(llm_client=_Client())
        state = {"n": 0}

        async def _call(prompt, *a, **k):
            state["n"] += 1
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            if state["n"] == 1:                      # stage 1 enumerates
                return {"suspects": [{"quote": "q", "check": "support",
                                      "reason": "r"}]}
            if state["n"] <= 4:                      # 3 unparseable samples
                return {}
            return {"verdict": "CONFIRMED", "confidence": 0.8,   # classic
                    "reasoning": "fallback", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        out = asyncio.run(v.verify_claim("c", "e", "ctx", deep=True))
        assert out is not None and out.reasoning == "fallback"
        assert state["n"] == 5, f"expected 1+3+1 calls, got {state['n']}"
        assert out.self_consistency_drawn == 3, (
            "three samples were paid for; the turn must not file as control")
        assert out.self_consistency_n == 0, "none of them parsed"

    def test_a_DEGRADED_vote_survives_the_ladder_too(self):
        """§4BR R21 MAJOR-4. The single-sample/degraded branch publishes
        out-of-band on its own line, and deleting that line left the whole
        suite green while reproducing R19's CRITICAL on this path: the
        escalation replaces the object, the counters vanish, and a turn that
        DID degrade files as a control turn that never voted.

        This is the branch the decision rule promises records (`sc_n ==
        sc_drawn == 1`) and the branch abort condition #3 watches. Every
        other test for it calls `_adjudicate_self_consistent` directly and
        reads the object, so none of them crosses the ladder.
        """
        import asyncio

        class _Client:
            critic_clients = [object()]

        v = Verifier(llm_client=_Client())
        state = {"n": 0}

        async def _call(prompt, *a, **k):
            state["n"] += 1
            force_main = k.get("force_main")
            if k.get("route_out") is not None:
                # Sample 1 comes back on the MAIN route -> the vote stops.
                k["route_out"]["route"] = "main"
            suspects = [{"quote": "q", "check": "support", "reason": "r"}]
            if state["n"] == 1:
                return {"suspects": suspects}
            return {"suspects": suspects,
                    "verdict": "CONFIRMED" if force_main else "REFUTED",
                    "confidence": 0.9, "reasoning": "x", "issues": []}

        object.__setattr__(v, "_call_llm", _call)
        out = asyncio.run(v.verify_claim("c", "e", "ctx", deep=True))
        assert out is not None
        assert out.verdict.value == "CONFIRMED", "the escalation should overturn"
        assert out.self_consistency_n == 1, (
            "a degraded vote must still be recorded as a vote")
        assert out.self_consistency_agree == 1
        assert out.self_consistency_drawn == 1

    def test_the_snapshot_is_taken_BEFORE_the_mechanical_guard(self):
        """§4BR R20 MAJOR-5. Moving the capture below
        `_guard_truncated_absence` survived the whole suite, and would
        delete the counters on every truncation-guard turn — ~1/3 of live
        digests carry a cut mark. Order asserted structurally because the
        guard rewrites the object without an LLM call, so no call-count
        test can see it."""
        import ast
        import inspect
        import textwrap
        src = textwrap.dedent(inspect.getsource(Verifier.verify_claim))
        tree = ast.parse(src)
        lines = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("_vote", "_cheap"):
                        lines[t.id] = node.lineno
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", "")
                if fn in ("_guard_truncated_absence", "_escalate_refute",
                          "_escalate_confirm"):
                    lines.setdefault(fn, node.lineno)
        assert lines["_vote"] < lines["_guard_truncated_absence"], lines
        assert lines["_vote"] < lines["_escalate_refute"], lines
        assert lines["_vote"] < lines["_escalate_confirm"], lines

    def test_a_CONTROL_verdict_still_carries_no_vote(self):
        """The mirror. Carrying the fields through must not invent them for
        turns that never voted, or every control row looks like a vote."""
        import asyncio
        v = self._verifier(["CONFIRMED"], "CONFIRMED")
        out = asyncio.run(v.verify_claim("c", "e", "ctx", deep=False))
        assert out is not None
        assert out.self_consistency_n is None
        assert "self_consistency_n" not in out.to_dict()


class TestStageOneFailureIsVisibleOnARoutedTurn:
    """§4BR R21 MINOR-10 — the THIRD silent treatment-to-control path.

    When stage 1 produces no suspects there is nothing to adjudicate, so
    `verify_claim` falls back to the classic single prompt and no vote is
    ever taken. On an ordinary turn that is routine plumbing (the cheap leg
    fails <0.5%). On a depth-routed turn it means the treatment ran control
    behaviour, and the operator's stream is WARNING+ — the other two
    degradation paths in this feature were promoted for exactly this
    reason, and this one was left behind both times.
    """

    @staticmethod
    def _run(deep):
        import asyncio
        v = Verifier(llm_client=None)

        async def _call(*a, **k):
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            return {"suspects": []}          # stage 1 yields nothing usable

        object.__setattr__(v, "_call_llm", _call)
        return asyncio.run(v._verify_claim_two_stage("c", "e", "x", deep=deep))

    def test_a_routed_turn_says_so_at_WARNING(self, caplog):
        import logging
        with caplog.at_level(logging.DEBUG):
            assert self._run(deep=True) is None
        hits = [r for r in caplog.records if "stage 1" in r.getMessage()]
        assert hits, "no record that the routed turn took no vote"
        assert any(r.levelno >= logging.WARNING for r in hits), (
            f"logged only at {[r.levelname for r in hits]} — invisible in "
            "the shipped configuration, which runs at INFO")

    def test_an_ALL_UNPARSEABLE_adjudication_also_says_so(self, caplog):
        """§4BR R22 MINOR-1 — the FOURTH silent path, the sibling branch 15
        lines below the one that was just promoted, and strictly more
        expensive: stage 1 succeeded, every sample was drawn and PAID FOR,
        and none parsed. Left at DEBUG when its twin was raised."""
        import asyncio
        import logging
        v = Verifier(llm_client=None)
        state = {"n": 0}

        async def _call(*a, **k):
            state["n"] += 1
            if k.get("route_out") is not None:
                k["route_out"]["route"] = "critic"
            if state["n"] == 1:
                return {"suspects": [{"quote": "q", "check": "support",
                                      "reason": "r"}]}
            return {}                      # every sample unparseable

        object.__setattr__(v, "_call_llm", _call)
        with caplog.at_level(logging.DEBUG):
            out = asyncio.run(
                v._verify_claim_two_stage("c", "e", "x", deep=True))
        assert out is None
        hits = [r for r in caplog.records
                if "unparseable" in r.getMessage().lower()]
        assert hits and any(r.levelno >= logging.WARNING for r in hits), (
            f"logged only at {[r.levelname for r in hits]}")

    def test_an_ORDINARY_turn_stays_quiet(self, caplog):
        """The mirror: promoting this unconditionally would put a line in
        the operator's stream on every ordinary stage-1 miss."""
        import logging
        with caplog.at_level(logging.DEBUG):
            assert self._run(deep=False) is None
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING], (
            "an unrouted stage-1 miss is plumbing, not news")


class TestTheRouteStampIsPRODUCED_notJustConsumed:
    """§4BR R18 MAJOR-3. Every sampler test above stubs `_call_llm` and sets
    `route_out` itself, so the guard's CONSUMER was pinned and its PRODUCER
    was not. Two mutations proved it: making `_call_llm` never stamp on the
    critic path, and making it stamp "critic" on the MAIN fallback, both
    left 62/62 green — the second silently disarms the degraded-route guard
    in production while the guard's own test stays green.

    These drive the REAL `_call_llm` against a fake llm_client.
    """

    @staticmethod
    def _client(*, critic=None, route=None, main=None, has_critic=True):
        class _C:
            critic_clients = [object()] if has_critic else None

            async def chat_completion(self, payload, **kw):
                if kw.get("use_critic"):
                    if critic is None:
                        raise RuntimeError("critic down")
                    return {"choices": [{"message": {"content": critic}}]}
                if main is None:
                    raise RuntimeError("main down")
                return {"choices": [{"message": {"content": main}}]}

            async def route(self, *a, **k):
                if route is None:
                    raise RuntimeError("worker down")
                return route
        return _C()

    def _call(self, client):
        import asyncio
        v = Verifier(llm_client=client)
        out = {}
        got = asyncio.run(v._call_llm("p", route_out=out))
        return got, out.get("route")

    def test_the_critic_path_stamps_critic(self):
        got, route = self._call(self._client(critic='{"verdict":"CONFIRMED"}'))
        assert got and route == "critic"

    def test_the_worker_path_stamps_worker(self):
        got, route = self._call(self._client(route='{"verdict":"CONFIRMED"}'))
        assert got and route == "worker"

    def test_the_MAIN_fallback_stamps_main_and_does_not_LIE(self):
        """THE one that matters. If this said "critic", the vote would keep
        sampling on the single foreground inference slot."""
        got, route = self._call(self._client(main='{"verdict":"CONFIRMED"}'))
        assert got and route == "main", (
            "a call that fell through to the main model must say so")

    def test_an_unparseable_critic_reply_reports_where_it_ACTUALLY_landed(self):
        """Critic answers but the parser cannot read it → falls through. The
        stamp must describe the route that produced the returned value."""
        got, route = self._call(self._client(
            critic="not json at all", main='{"verdict":"CONFIRMED"}'))
        assert got and route == "main"

    def test_a_total_failure_stamps_failed(self):
        _got, route = self._call(self._client())
        assert route == "failed"

    def test_a_missing_client_stamps_failed_rather_than_nothing(self):
        """An unstamped return reads as "route unknown, keep sampling"."""
        import asyncio
        v = Verifier(llm_client=None)
        out = {}
        asyncio.run(v._call_llm("p", route_out=out))
        assert out.get("route") == "failed"

    def test_the_stamp_is_OPTIONAL_for_every_other_caller(self):
        """`route_out` defaults to None; no other call site passes it."""
        import asyncio
        v = Verifier(llm_client=self._client(critic='{"verdict":"CONFIRMED"}'))
        assert asyncio.run(v._call_llm("p"))


def test_the_early_stop_bound_is_STRICT_and_that_matters_at_n_5():
    """Mutating `lead > (n-i-1)` to `>=` left the suite green while
    producing 60 verdict mismatches at n=5 — invisible at n=3, but n=5 is a
    supported and tested setting (`GHOST_VERIFY_SELF_CONSISTENCY=5`).

    `>=` stops when the runner-up could still draw LEVEL, and a tie is
    resolved to the FIRST sample's verdict — so stopping early can hand the
    vote to the current leader when a full draw would have handed it to the
    other side. This is the case that separates them:

        C, R, R, <unparseable>, C     budget 5

    After 4 draws R leads 2-1 with 1 draw left. `>=` stops and returns R.
    The strict bound draws the 5th (C), making it 2-2 — a tie, resolved to
    sample 1, which is C. Different verdict, same inputs.

    ⚠ The first version of this test used C,R,R,C,C and asserted 5 draws:
    both bounds draw all 5 there, so it discriminated nothing and the
    mutation survived it.
    """
    v, drawn = _counting_verifier([_r(C), _r(R), _r(R), None, _r(C)])
    out = _run(v, n=5)
    assert len(drawn) == 5, "a 2-1 lead with a draw left is not decided"
    assert out.verdict == C, (
        "the 5th sample ties it 2-2 and the tie goes to sample 1 (C); "
        "a >= bound would have stopped at R")


def test_the_whole_VOTE_has_a_wall_clock_ceiling(monkeypatch):
    """§4BR R18 MAJOR-4. The per-call timeouts (critic 120s, worker 45s)
    bound ONE adjudication; nothing bounded a loop of them. The realistic
    bad case is not a clean fall-through to the main model — the route
    guard catches that — but a PARTIAL outage: critic times out, worker
    answers, route stays "worker", and three samples cost 3x165s for a
    verdict the turn stops waiting for after 25s.
    """
    import asyncio
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "5")
    # route="worker", NOT main: the route guard is deliberately blind here.
    v, drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(R)], 10.0)
    out = asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    assert len(drawn) == 1, f"budget ignored; drew {len(drawn)} samples"
    assert out is not None, "the budget must degrade the vote, not the verdict"


def test_the_budget_DEFAULT_is_above_the_live_verify_distribution(monkeypatch):
    """§4BR R19. The default was a bare magic number with no test: mutating
    20.0 → 0.0001 collapsed every live vote to a single sample and the suite
    stayed green.

    The number itself was also wrong. It was picked to sit under the 25s
    in-loop repair window using a 17.8s "contested vote" measured on an IDLE
    critic with a synthetic prompt. Live USER-FACING claim verifies (INFO
    level, bench-drain excluded, n=60) run p50 27.6s / p90 54.9s END TO END,
    65% already over 25s in CONTROL — so that window is routinely missed
    anyway, and a 20s budget would have truncated ordinary votes into
    control while still recording them as votes. (An earlier draft quoted
    "n=109 / p50 25.4 / p90 54.3": that count silently mixed in 28
    background verdicts and 21 bench rows, and the 54.3 figure belongs to a
    different subset again. The honest population is slower, so the 60s
    choice only gets safer — but the label was wrong.)

    The budget exists to cut the 495s partial-outage pathology, not to trim
    normal votes, so it must sit above the live whole-verify p90.
    """
    monkeypatch.delenv("GHOST_VERIFY_VOTE_BUDGET", raising=False)
    from ghost_agent.core.verifier import _vote_budget_s
    assert _vote_budget_s() >= 54.3, (
        "a budget below the live p90 whole-verify latency truncates real "
        "votes into control")
    # ⚠ The upper bound was 3*(120+45) — the TOTAL pathology — which is a
    # proxy, not the thing. The loop breaks after sample i once i*165s has
    # elapsed, so the budget only cuts to one sample when it is at most ONE
    # per-call bound. Measured: 60s draws 1/3 (167s-equivalent), while 400s
    # passed the old assertion and drew 3/3 (502s-equivalent) — i.e. the
    # assertion certified a budget that does not bound the pathology at all.
    assert _vote_budget_s() < (120 + 45), (
        "a budget above one per-call bound cannot cut the multiplication "
        "it exists to cut")


def test_the_budget_is_PER_VOTE_not_per_sample(monkeypatch):
    """§4BR R19. Moving the deadline inside the loop (making it a per-sample
    timeout) survived the suite — the fix's own test could not distinguish
    the fix from the defect it replaced, which is the whole point of the
    guard: a per-sample bound is exactly what already existed and exactly
    what failed to bound the loop.

    Here each sample fits comfortably inside the budget, but their SUM does
    not. A per-sample reading draws all 3; a per-vote reading stops.
    """
    import asyncio
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "50")
    v, drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(R)], 30.0)
    asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    assert len(drawn) == 2, (
        f"drew {len(drawn)}: 30s x2 exceeds the 50s VOTE budget, but no "
        "single sample does — a per-sample deadline would draw all 3")


def test_the_budget_ALSO_binds_force_main(monkeypatch):
    """The route guard exempts force_main (its route IS main by design); the
    budget must not, because force_main is where the pathology is worst —
    n=5 on the 35B at 90s each is 450s on the single foreground slot."""
    import asyncio
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "5")
    v, drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(R)], 10.0,
                                 route="main")
    asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=True))
    assert len(drawn) == 1, f"force_main drew {len(drawn)} past the budget"


def test_a_FAILED_route_also_stops_the_vote():
    """`_call_llm` stamps "failed" when it could not produce anything. The
    producer side is pinned above; this pins the CONSUMER — without it,
    dropping "failed" from the guard leaves n futile round trips against a
    broken client on every hard turn."""
    v, drawn = _counting_verifier([_r(R), _r(R), _r(R)], route="failed")
    _run(v)
    assert len(drawn) == 1


def test_the_degradation_is_logged_at_WARNING(caplog):
    """This line is the ONLY instrument for abort condition #3 of the arm's
    decision rule. The operator's live stream is WARNING+, so demoting it to
    DEBUG makes the abort unobservable while every test stays green."""
    import logging
    v, _drawn = _counting_verifier([_r(R), _r(R), _r(R)], route="main")
    with caplog.at_level(logging.DEBUG):
        _run(v)
    hits = [r for r in caplog.records if "degraded" in r.getMessage()]
    assert hits, "no degradation record at all"
    assert any(r.levelno >= logging.WARNING for r in hits), (
        f"degradation logged only at {[r.levelname for r in hits]}")


def test_a_TIE_uses_SAMPLE_ORDER_not_Counter_insertion_order():
    """§4BR R19. `results[0].verdict` and `top[0][0]` are identical at n<=3
    because Counter breaks ties by first occurrence — so the mutation
    survived. They diverge at n=5, which is a supported setting.

    U,C,C,R,R: C and R tie at 2 each and U has 1, so `most_common()` puts
    CONFIRMED first — while sample 1, which is control's own draw, said
    UNCERTAIN. Counter breaks ties by FIRST-OCCURRENCE order, which is not
    sample order once a third verdict is in play. Taking `top[0][0]` would
    answer CONFIRMED for a vote that decided nothing.
    """
    v, _ = _counting_verifier([_r(U), _r(C), _r(C), _r(R), _r(R)])
    out = _run(v, n=5)
    assert out.verdict == U, (
        "a tie must fall back to control's own sample, not to whichever "
        "verdict Counter happens to rank first")


def _clocked_verifier(monkeypatch, samples, per_sample_s, route="worker"):
    """A verifier whose samples advance a FAKE monotonic clock.

    §4BR R21 MAJOR-1/-2. The budget tests were written with real
    `asyncio.sleep`, and both ways of getting that wrong showed up:

      * a 0.0001s budget against a vote that runs in 3-10 MICROseconds
        means the deadline never expires, so the assertion "no TRUNCATED
        line was logged" was trivially true — the test passed identically
        with and without the fix it existed to pin (the THIRD vacuous test
        in this feature);
      * the one test that did discriminate used 2x sleep(0.02) against a
        0.05s budget, measured to exceed its margin on 4.3% of runs under
        12-way CPU load.

    A controlled clock removes both failure modes: elapsed time becomes an
    input, not a race. `_adjudicate_self_consistent` does `import time as
    _time` inside the function, so patching the module attribute reaches it.
    """
    import time as _time_mod
    v = Verifier(llm_client=None)
    seq = list(samples)
    drawn = []
    now = {"t": 1000.0}
    monkeypatch.setattr(_time_mod, "monotonic", lambda: now["t"])

    async def _call(*a, **k):
        drawn.append(1)
        now["t"] += per_sample_s          # the sample "takes" this long
        if k.get("route_out") is not None:
            k["route_out"]["route"] = route
        return {"x": 1}

    object.__setattr__(v, "_call_llm", _call)
    object.__setattr__(v, "_build_verify_result",
                       lambda _raw: seq.pop(0) if seq else None)
    return v, drawn


def test_a_DECIDED_vote_is_not_reported_as_budget_exhausted(monkeypatch,
                                                            caplog):
    """§4BR R20 CRITICAL-1(b). The deadline was checked BEFORE the
    early-stop check, so a vote that ended because the majority was settled
    logged "budget exhausted" — a line DECISION_RULE.md treats as an ABORT
    signal ("the sizing is wrong"). It must not fire on healthy votes.

    Sized so the budget expires DURING the deciding sample, not before it:
    10s per sample against a 15s budget means sample 1 is inside the budget,
    the vote is decided 2-0 at sample 2, and the clock is past the deadline
    by the time that sample returns. Under the correct order the early stop
    wins and nothing is logged; under the defect the deadline is read first
    and a healthy vote is filed as truncated. Both orders draw 2 samples —
    only the log distinguishes them, which is the whole point.
    """
    import asyncio
    import logging
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "15")
    v, drawn = _clocked_verifier(monkeypatch, [_r(R), _r(R), _r(C)], 10.0)
    with caplog.at_level(logging.DEBUG):
        out = asyncio.run(v._adjudicate_self_consistent("p", n=3,
                                                        force_main=False))
    assert len(drawn) == 2 and out.verdict == R      # decided, not truncated
    assert not [r for r in caplog.records if "TRUNCATED" in r.getMessage()], (
        "a decided 2-0 vote drew every sample it needed, and was reported "
        "as budget-truncated")


def test_a_COMPLETE_vote_is_not_reported_as_truncated(monkeypatch, caplog):
    """The other half: on the LAST iteration there is nothing left to draw,
    so an expired budget has truncated nothing. Three mutually disagreeing
    samples never trigger the early stop, so the loop runs to completion
    with the budget long gone."""
    import asyncio
    import logging
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "50")
    v, drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(U)], 20.0)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    assert len(drawn) == 3, f"all three samples must be drawn, drew {len(drawn)}"
    assert not [r for r in caplog.records if "TRUNCATED" in r.getMessage()], (
        "a vote that drew every sample was reported as truncated")


def test_a_TRUNCATED_vote_is_visible_to_the_OPERATOR(monkeypatch, caplog):
    """§4BR R20 CRITICAL-1(a). The line shipped at logger.debug, and the
    live agent runs at INFO — the launcher passes no --debug and the 20MB
    live log contains ZERO GhostAgent DEBUG lines. The abort signal was
    unobservable in the only configuration that runs. This is the second
    instrument for this feature to ship below the visible threshold."""
    import asyncio
    import logging
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "5")
    v, drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(R)], 10.0)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    hits = [r for r in caplog.records if "TRUNCATED" in r.getMessage()]
    assert hits, "a truncated vote produced no record at all"
    assert any(r.levelno >= logging.WARNING for r in hits), (
        f"truncation logged only at {[r.levelname for r in hits]} — "
        "invisible in the shipped configuration")
    # §4BR R22 MINOR-2: the message must state the budget it hit. "%.0f"
    # renders every sub-second budget as "0s", which reads as a
    # misconfiguration rather than the tight test/debug setting it is.
    assert "5s budget" in hits[0].getMessage(), hits[0].getMessage()


def test_a_SUB_SECOND_budget_is_printed_legibly(monkeypatch, caplog):
    """The `%g` fix had no test and could be reverted silently."""
    import asyncio
    import logging
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "0.25")
    v, _drawn = _clocked_verifier(monkeypatch, [_r(C), _r(R), _r(R)], 1.0)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    msg = next(r.getMessage() for r in caplog.records
               if "TRUNCATED" in r.getMessage())
    assert "0.25s budget" in msg, msg
    assert "0s budget" not in msg


def test_the_budget_can_be_disabled_by_the_operator(monkeypatch):
    import asyncio
    monkeypatch.setenv("GHOST_VERIFY_VOTE_BUDGET", "0")
    v, drawn = _counting_verifier([_r(C), _r(R), _r(R)])
    out = asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=False))
    assert len(drawn) == 3 and out.verdict == R


def test_the_DEGRADED_vote_still_records_what_happened():
    """§4BR R18 CRITICAL-2. The `len(results) < 2` early return set neither
    counter — so the single case the mechanism gate and the latency abort
    exist to detect was the one case with no record that it occurred."""
    v, drawn = _counting_verifier([_r(R), _r(R), _r(R)], route="main")
    out = _run(v)
    assert len(drawn) == 1
    assert out.self_consistency_n == 1
    assert out.self_consistency_agree == 1
    assert out.self_consistency_drawn == 1


def test_DRAWN_and_PARSED_are_counted_separately():
    """Otherwise "the vote stopped early" and "the judge returned garbage"
    read identically in the corpus."""
    v, drawn = _counting_verifier([None, _r(R), _r(R)])
    out = _run(v)
    assert out.self_consistency_drawn == 3 and out.self_consistency_n == 2


def test_a_force_main_vote_is_NOT_cut_short_by_its_own_route():
    """`force_main=True` asks for the main model deliberately (the refute
    escalation). Stopping there would make the degradation guard fire on
    the one caller whose route is intentional."""
    import asyncio
    v, drawn = _counting_verifier([_r(C), _r(R), _r(R)], route="main")
    out = asyncio.run(v._adjudicate_self_consistent("p", n=3, force_main=True))
    assert len(drawn) == 3
    assert out.verdict == R
