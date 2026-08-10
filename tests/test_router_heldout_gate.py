"""§4AA — the router's deploy gate is EMPIRICAL, not a prior about weights.

THE DEFECT (measured 2026-08-09). `looks_sane()` rejected any model whose
technical/coding weights were negative, encoding the prior "more jargon ⇒
harder". On 1354 labelled trajectories (92% real `user_request` turns —
self-play contamination ruled out) this agent's traffic says the opposite:
jargon is 4.1× and coding mentions 6.8× MORE common in EASY turns, and even
LENGTH inverts (longer ⇒ easier). Vagueness, not technicality, predicts work
here — and no labelling bug makes longer requests easier.

Consequences, both measured on a 70/30 split (seed 7, n=407 held out):

    escalate-all baseline   accuracy 0.560
    FITTED model            accuracy 0.695 · skips planner on 13.2% of hard
                            → looks_sane() False → REJECTED
    SIGN-FLIPPED model      accuracy 0.305 · skips planner on 86.8% of hard
                            → looks_sane() True  → ACCEPTED

The gate rejected the good model and would have ACCEPTED the catastrophic
one — precisely the §4O failure it was built to prevent. A gate that is
anti-correlated with quality is worse than no gate, because it is trusted.
The router sat escalate-all for ~29h and would have forever, since every fit
on this corpus reproduces the same signs.

THE REPLACEMENT asks the only question that matters: on data it did NOT
train on, does the model beat doing nothing, without skipping the planner too
often? Fail-closed — no evidence means no deploy, and rejection leaves the
router escalate-all, which is today's behaviour and the safe direction.
"""

import copy

import numpy as np
import pytest

from ghost_agent.router.model import ComplexityClassifier


def _toy(n=400, seed=3):
    """A separable corpus: feature 0 predicts 'hard'."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        hard = i % 2 == 0
        v = np.zeros(len(ComplexityClassifier().feature_names_))
        v[0] = rng.normal(2.0 if hard else -2.0, 0.4)
        X.append(v)
        y.append("hard" if hard else "easy")
    return X, y


@pytest.fixture()
def fitted():
    X, y = _toy()
    cut = int(len(y) * 0.7)
    clf = ComplexityClassifier()
    clf.fit(X[:cut], y[:cut])
    clf.gate_report_ = clf.evaluate(X[cut:], y[cut:])
    return clf, X[cut:], y[cut:]


# ── the gate's arithmetic ───────────────────────────────────────────────────

def test_a_model_that_beats_escalate_all_passes(fitted):
    clf, _, _ = fitted
    assert clf.looks_sane() is True
    ok, why = ComplexityClassifier.gate_verdict(clf.gate_report_)
    assert ok and "escalate-all" in why


def test_evaluate_reports_the_escalate_all_baseline(fitted):
    """`baseline` = always predict 'hard' = exactly what the router does
    when it is escalate-all. "Beats baseline" must mean "better than doing
    nothing"."""
    clf, Xte, yte = fitted
    ev = clf.evaluate(Xte, yte)
    assert ev["baseline"] == pytest.approx(
        sum(1 for v in yte if v == "hard") / len(yte))


@pytest.mark.parametrize("ev,why", [
    ({}, "no held-out evidence"),
    ({"n": 5, "accuracy": 1.0, "baseline": 0.5, "false_easy_on_hard": 0.0,
      "classes": 2}, "held-out n"),
    ({"n": 100, "accuracy": 1.0, "baseline": 0.5, "false_easy_on_hard": 0.0,
      "classes": 1}, "one class"),
    ({"n": 100, "accuracy": 0.50, "baseline": 0.56, "false_easy_on_hard": 0.0,
      "classes": 2}, "does not beat"),
    ({"n": 100, "accuracy": 0.90, "baseline": 0.56, "false_easy_on_hard": 0.4,
      "classes": 2}, "skips the planner"),
])
def test_every_rejection_reason_is_stated(ev, why):
    ok, reason = ComplexityClassifier.gate_verdict(ev)
    assert not ok and why in reason


def test_the_gate_is_asymmetric_by_design():
    """Predicting 'easy' for a HARD request skips the planner (harmful);
    predicting 'hard' for an easy one only wastes compute. A model can be
    accurate overall and still be rejected for the dangerous error."""
    ev = {"n": 200, "accuracy": 0.90, "baseline": 0.56, "classes": 2,
          "false_easy_on_hard": ComplexityClassifier._GATE_MAX_FALSE_EASY + 0.01}
    assert not ComplexityClassifier.gate_verdict(ev)[0]
    ev["false_easy_on_hard"] = ComplexityClassifier._GATE_MAX_FALSE_EASY - 0.01
    assert ComplexityClassifier.gate_verdict(ev)[0]


# ── THE MUTATION the old gate failed ────────────────────────────────────────

def test_a_sign_flipped_model_is_REJECTED(fitted):
    """THE §4O CATASTROPHE. The old prior-based gate ACCEPTED this — it has
    the 'right' weight signs and is 30% accurate, skipping the planner on
    87% of hard requests."""
    clf, Xte, yte = fitted
    bad = copy.deepcopy(clf)
    bad.weights_ = -np.asarray(bad.weights_)
    bad.bias_ = -float(bad.bias_)
    bad.gate_report_ = bad.evaluate(Xte, yte)
    assert bad.gate_report_["accuracy"] < bad.gate_report_["baseline"]
    assert bad.looks_sane() is False


def test_stolen_evidence_does_not_transfer(fitted):
    """⚠ FOUND BY MUTATION-TESTING THE GATE ITSELF. `looks_sane()` reads a
    STORED report, so a corrupted checkpoint carrying real weights beside a
    passing report would be waved through — a forgery the OLD gate was immune
    to, because it inspected the weights directly. The evidence is bound to a
    fingerprint of the weights it was measured on."""
    clf, _, _ = fitted
    forged = copy.deepcopy(clf)
    forged.weights_ = -np.asarray(forged.weights_)
    forged.gate_report_ = dict(clf.gate_report_)      # steal the passing report
    assert forged.looks_sane() is False


def test_the_fingerprint_changes_with_any_weight(fitted):
    clf, _, _ = fitted
    before = clf.weights_fingerprint()
    clf.weights_ = np.asarray(clf.weights_).copy()
    clf.weights_[0] += 1e-6
    assert clf.weights_fingerprint() != before


# ── fail-closed ─────────────────────────────────────────────────────────────

def test_a_model_with_no_evidence_is_rejected(fitted):
    """A legacy checkpoint or a hand-built model has no held-out evidence.
    No evidence ⇒ no deploy; the router stays escalate-all, which is the
    safe default and unchanged behaviour."""
    clf, _, _ = fitted
    clf.gate_report_ = None
    assert clf.looks_sane() is False


def test_a_non_finite_model_is_still_rejected(fitted):
    """The pre-existing NaN guard must survive the rewrite.

    ⚠ The evidence is re-fingerprinted AFTER the corruption, so it MATCHES
    the NaN weights. Without that, the model is rejected by the fingerprint
    check and this test says nothing about `is_finite` — which is exactly
    how it first passed with the is_finite guard deleted (weak pin, caught
    by revert-testing)."""
    clf, _, _ = fitted
    clf.weights_ = np.asarray(clf.weights_).copy()
    clf.weights_[0] = float("nan")
    ev = dict(clf.gate_report_)
    ev["weights_sha"] = clf.weights_fingerprint()   # evidence now MATCHES
    clf.gate_report_ = ev
    assert ComplexityClassifier.gate_verdict(ev)[0] is True, (
        "the evidence itself must pass, so only is_finite can reject")
    assert clf.looks_sane() is False


def test_evidence_survives_a_save_load_round_trip(fitted, tmp_path):
    clf, _, _ = fitted
    p = clf.save(tmp_path / "ckpt.json")
    back = ComplexityClassifier.load(p)
    assert back.gate_report_ is not None
    assert back.looks_sane() is True, "a deployed model lost its own evidence"


def test_a_checkpoint_without_gate_evidence_loads_but_does_not_deploy(fitted, tmp_path):
    """The CURRENT live checkpoint is exactly this shape."""
    import json
    clf, _, _ = fitted
    p = clf.save(tmp_path / "ckpt.json")
    raw = json.loads(p.read_text())
    raw.pop("gate_report", None)
    p.write_text(json.dumps(raw))
    back = ComplexityClassifier.load(p)
    assert back.weights_ is not None, "it must still LOAD"
    assert back.looks_sane() is False, "…but must not deploy"


# ── the prior is gone ───────────────────────────────────────────────────────

def test_negative_coding_weights_no_longer_block_deployment(fitted):
    """The whole point: a model may contradict the old prior and still ship,
    provided it earns it on held-out data."""
    clf, Xte, yte = fitted
    idx = {n: i for i, n in enumerate(clf.feature_names_)}
    clf.weights_ = np.asarray(clf.weights_).copy()
    for name in ComplexityClassifier._CORE_HARD_FEATURES:
        if name in idx:
            clf.weights_[idx[name]] = -0.9      # would have failed the old gate
    clf.gate_report_ = clf.evaluate(Xte, yte)   # re-earn it on the data
    ok, _ = ComplexityClassifier.gate_verdict(clf.gate_report_)
    assert ok is clf.looks_sane()
    assert clf.looks_sane() is True


# ── fresh-eye review findings (2026-08-09), neither previously covered ──────

def test_the_gate_scores_the_DEPLOYED_decision_not_the_bare_label(fitted):
    """⚠ REVIEW FINDING. `ComplexityDispatcher` escalates whenever confidence
    < its threshold, WHATEVER the label — so a low-confidence "easy" never
    skips the planner in production. A first version of `evaluate()` read the
    bare label and therefore gated on an operating point the router does not
    ship, overstating false-easy 0.044 -> 0.131 on the real corpus.
    """
    clf, Xte, yte = fitted
    # A threshold above ANY achievable confidence must escalate EVERY sample,
    # which forces the deployed decision to "hard" everywhere — so accuracy
    # must collapse to exactly the escalate-all baseline. That is only true
    # if evaluate() honours the confidence gate. (Asserting false-easy == 0
    # was too weak: this fixture is separable, so it is 0 either way — a weak
    # pin caught by revert-testing.)
    strict = clf.evaluate(Xte, yte, confidence_threshold=1.01)
    assert strict["accuracy"] == pytest.approx(strict["baseline"]), (
        "evaluate() is reading the bare label and ignoring the confidence "
        "gate the dispatcher actually applies")
    assert strict["false_easy_on_hard"] == 0.0
    loose = clf.evaluate(Xte, yte, confidence_threshold=0.0)
    assert loose["accuracy"] > strict["accuracy"]


def test_the_threshold_is_READ_from_the_dispatcher_not_copied():
    """⚠ REVIEW FINDING. The live value is also a CLI flag
    (--router-confidence-threshold), so a hardcoded copy would silently score
    an operating point the router does not run — the same two-copies-drift
    defect fixed in the bench pool earlier today."""
    import inspect

    from ghost_agent.router.dispatch import ComplexityDispatcher
    live = inspect.signature(ComplexityDispatcher.__init__) \
        .parameters["confidence_threshold"].default
    assert ComplexityClassifier._deploy_confidence_threshold() == float(live)
    # ⚠ Check the CODE, not the prose: getsource() includes the docstring,
    # which names ComplexityDispatcher — so a hardcoded `return 0.3` body
    # still matched a bare name check (weak pin, caught by revert-testing).
    # ⚠ The READ moved into `_deploy_threshold_probe` (audit 2026-08-10) so the
    # SOURCE of the value can be reported; the public accessor delegates.
    # Follow it — the intent is unchanged: read from the dispatcher, never copy.
    src = inspect.getsource(ComplexityClassifier._deploy_threshold_probe)
    body = src.split('"""')[-1]
    assert "inspect.signature" in body and "ComplexityDispatcher" in body, (
        "the threshold is hardcoded rather than read from the dispatcher")


def test_the_threshold_SOURCE_is_reported_so_a_broken_read_is_visible():
    """⚠ AUDIT 2026-08-10. The fallback returns 0.3 — which is ALSO the
    dispatcher default and ALSO the live value. A broken read therefore
    produced the RIGHT answer and was undetectable, right up until the
    operator moved `--router-confidence-threshold`, at which point the gate
    would silently score an operating point the router does not run.

    "Could a disconnected instrument produce this same output?" — yes. So the
    PATH taken is now reported alongside the value."""
    val, source = ComplexityClassifier._deploy_threshold_probe()
    assert source == "dispatcher", (
        "the dispatcher read is failing and the fallback is masking it")
    assert val == ComplexityClassifier._deploy_confidence_threshold()


def test_a_broken_dispatcher_read_reports_FALLBACK_not_silence(monkeypatch):
    """The failure must be NAMED: same value, different provenance."""
    import ghost_agent.router.dispatch as D
    monkeypatch.delattr(D, "ComplexityDispatcher", raising=False)
    val, source = ComplexityClassifier._deploy_threshold_probe()
    assert source == "fallback"
    assert val == 0.3, "the fallback must remain a plausible threshold"


def test_the_gate_evidence_records_which_threshold_source_was_used(fitted):
    """Recorded in the evidence dict so a PAST gate report can be re-read and
    its operating point VERIFIED rather than assumed.

    ⚠ My first version built `X` as raw dicts, hit `cannot vectorize dict`,
    and swallowed it in `except: pytest.skip(...)` — a silent skip dressed as
    an environment problem, which is the "guard that never runs" pattern this
    repo keeps finding. It reuses the module's own `fitted` fixture instead.
    """
    clf, X_held, y_held = fitted
    ev = clf.evaluate(X_held, y_held)
    assert ev.get("confidence_threshold_source") in ("dispatcher", "fallback")

    ev2 = clf.evaluate(X_held, y_held, confidence_threshold=0.42)
    assert ev2["confidence_threshold"] == 0.42
    assert ev2["confidence_threshold_source"] == "caller", (
        "an explicitly-passed threshold must not be attributed to the "
        "dispatcher — that would launder a caller's choice as a live read")


def test_the_threshold_used_is_recorded_in_the_evidence(fitted):
    """A report that does not say which operating point it measured cannot
    be checked later."""
    clf, Xte, yte = fitted
    ev = clf.evaluate(Xte, yte, confidence_threshold=0.42)
    assert ev["confidence_threshold"] == 0.42


def test_the_trainer_threads_the_live_threshold():
    """Structural: both deploy paths (boot bootstrap, idle retrain) must pass
    the dispatcher's actual value, or the gate silently scores the default."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    main_src = (repo / "src" / "ghost_agent" / "main.py").read_text()
    agent_src = (repo / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert "confidence_threshold=float(\n" in main_src or \
           "args.router_confidence_threshold)" in main_src, \
        "boot bootstrap no longer passes the operator's threshold"
    # ⚠ Assert the COMPOSED call. Grepping only for the getattr line passed
    # while the value was computed and then thrown away (weak pin).
    assert "RouterTrainer(confidence_threshold=_thr)" in agent_src, \
        "idle retrain computes the live threshold but no longer passes it"
    assert 'getattr(dispatcher, "confidence_threshold", None)' in agent_src


# ── convergence: the defect that made the gate unreachable ──────────────────
#
# MEASURED 2026-08-10. At epochs=300 the fit reported `converged: False`. An
# under-converged logistic regression has small-magnitude weights, which
# COMPRESSES its sigmoid outputs — max confidence over 800 live requests was
# **0.710**, so the planner-skip gate at 0.75 was UNREACHABLE and the router's
# only live consumer could never fire. The router trained, gated and deployed
# correctly every idle cycle while changing NOTHING in production.
#
# The fix is convergence, not a lower bar: scaling weights leaves 800/800
# LABELS identical (sigmoid is monotonic), so the compression was a pure
# calibration defect and correcting it cannot cost accuracy.

def test_the_default_fit_CONVERGES(fitted):
    """⚠ THE PIN. If this goes red, the router's consumer silently stops
    being able to fire — and nothing else would report it."""
    clf, _X, _y = fitted
    rep = clf.report_
    assert rep is not None
    assert getattr(rep, "converged", False) is True, (
        "the router fit no longer converges — its confidence will compress "
        "and the planner-skip gate becomes unreachable again")


def test_the_defaults_are_the_CONVERGING_ones():
    """Structural: the constant is the fix. 300/1e-5 did not converge."""
    import inspect
    p = inspect.signature(ComplexityClassifier.__init__).parameters
    assert p["epochs"].default >= 3000, (
        "epochs fell back to a value that does not converge")
    assert p["tol"].default <= 1e-5


def test_a_converged_model_can_actually_REACH_the_consumer_bar(fitted):
    """The invariant that matters: confidence must span the 0.75 the
    planner-skip reads. A model that is accurate but permanently below its
    own consumer's threshold is a loop with no effect.

    ⚠ THIS IS AN INVARIANT, NOT THE REGRESSION DETECTOR. Revert-testing showed
    it still PASSES at epochs=300, because `_toy()` is linearly separable and
    converges fast even under-trained. On the REAL corpus 300 epochs capped at
    0.710. `test_the_default_fit_CONVERGES` and
    `test_the_defaults_are_the_CONVERGING_ones` are the pins that actually go
    red — recorded so nobody mistakes this one for cover it does not give.
    """
    clf, X_held, _y = fitted
    confs = [clf.predict(x)[1] for x in X_held]
    assert max(confs) >= 0.75, (
        f"max confidence {max(confs):.3f} is still below the 0.75 "
        f"planner-skip bar — the consumer remains unreachable")


# ── the retired planner-skip consumer ───────────────────────────────────────

def test_the_confidence_thresholded_planner_skip_stays_RETIRED():
    """⛔ RETIRED 2026-08-10 (§4AN). Measured: it could only ever fire where
    the router carries NO information.

        family    n     majority-class  model acc   LIFT
        chess    189        0.868         0.868    +0.000
        other   1172        0.626         0.677    +0.051

    113 of 114 confident-easy requests were one chess template; 0 of 600
    non-chess requests ever cleared 0.75. The bar selected for "saturates
    every easy-indicator feature at once" — which only a synthetic template
    does — not for "easy". And it was NOT free: it was the sole source of the
    false-easy-on-hard risk (0.0558 held-out), i.e. skipping the planner on
    genuinely hard requests.

    ⚠ Re-adding ANY confidence-thresholded consumer requires first showing
    the model is informative IN THE REGION THAT CLEARS THE BAR. Being
    accurate overall is not sufficient — the router is accurate overall and
    was still worthless here.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "core" / "agent.py").read_text()
    # Anchor on the CODE, not the prose: the retirement comment necessarily
    # quotes the condition it removed.
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]

    # ⚠ NAME-BASED PINNING IS NOT ENOUGH. Asserting `"_rd_plan" not in code`
    # passes the moment someone re-adds the identical logic under any other
    # variable name — found by mutation-testing this very test. Pin the SHAPE
    # instead: nothing may disable the planner off a router CONFIDENCE score.
    for i, line in enumerate(code):
        if "use_plan = False" not in line:
            continue
        window = "\n".join(code[max(0, i - 15):i + 1])
        assert not ("confidence" in window and "router" in window.lower()), (
            "a router-confidence-thresholded planner-skip is back near line "
            f"{i}: it can only fire where the router adds ZERO information, "
            "and it is the sole source of the false-easy-on-hard risk.\n"
            f"{window}")


def test_the_router_decision_is_still_RECORDED_and_the_MCTS_gate_intact():
    """⚠ OVER-REMOVAL GUARD. Retiring the planner-skip must NOT take the
    router's recording or its label-level consumer with it — the model's real
    +5.1pp lives at the LABEL, which the MCTS gate reads with no confidence
    bar. That consumer is where this signal actually fits."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "core" / "agent.py").read_text()
    assert 'body["_router_decision"] = {' in src, "the decision stopped being recorded"
    assert 'router_label=str(decision.label or "")' in src, "turn_facts lost the label"
    assert '_rd.get("label") == "hard"' in src, "the MCTS label gate was removed too"
