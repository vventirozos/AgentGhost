"""§4BQ flip (vi) — embedding features for the complexity router.

WHAT EARNED THE CHANGE. §4AN found the router "confidently right exactly
where it adds nothing": its 18 lexical features saturate on one synthetic
chess template, and on natural traffic easy/hard barely separated. §4BQ
measured a 384-d sentence-embedding block (the vector store's
already-loaded bge-small-en-v1.5) on the real 1,482-turn corpus. Through
the PRODUCTION trainer and its held-out gate: accuracy 0.674 → 0.728
(+0.054, 95% CI [+0.030, +0.078], exact McNemar p=8.4e-6). On the metric
the MCTS depth gate reads (`label=="hard" and not escalated`), firing
precision goes 0.663 → 0.782 against a 0.624 base rate while firing LESS
often, and it is a better ranking rather than a luckier threshold (AUC
0.599 → 0.757).

⚠ That gate is currently DISABLED (`core/agent.py`
`_MCTS_TURNSTART_ENABLED = False`), and the confident-easy planner skip
was retired in §4AN — so the router's verdict is recorded and acted on by
nothing today. This flip improves a decision quality, not a behaviour.

Claims that did NOT survive review, recorded so they are not
resurrected: the false-easy delta (8.3% → 7.4%) is noise (McNemar
p=0.688), and at the 70% TEMPORAL cut neither representation beats
escalate-all — though that cut quarantines the chess-template burst into
train, and with the template excluded the combined arm wins at all four
cuts tested. The flip is a real improvement over the lexical arm; it is
not proof the router earns its keep.

WHAT THESE TESTS PIN. Not the accuracy — that lives in the §4AA gate,
which re-measures every fit. They pin the SAFETY CONTRACT that makes the
flip shippable:

  1. A router that cannot embed ESCALATES. It never scores a 402-dim
     model with an 18-dim vector, nor the reverse.
  2. A checkpoint knows WHICH embedder trained it. Width is not identity:
     MiniLM is also 384-d and also in the local HF cache, so a
     GHOST_EMBED_MODEL migration would otherwise score one model's
     weights against another model's vectors — measured at accuracy
     0.913 → 0.497 with 14 confident-wrong routes, silently.
  3. Routing depends on THIS request, and training pairs each text with
     ITS OWN embedding.

(3) is why the corpus below is built the way it is. An earlier version of
this file embedded random noise, so a mutation making the dispatcher
embed a CONSTANT STRING — i.e. the whole feature inert — passed every
test. `_signal_corpus` fixes that: its two classes are LEXICALLY
IDENTICAL (asserted, not assumed), so the 18 lexical features cannot
separate them and only the embedding can. Any test using it fails if the
embedding is constant, scrambled, or ignored.

Determinism note: vectors are seeded with `zlib.crc32`, never `hash()`.
`hash()` on str is PYTHONHASHSEED-randomised per process, which made an
earlier mutation-testing run irreproducible — two mutations reported
"killed" in one process and survived in the next.
"""

import json
import math
import zlib

import numpy as np
import pytest

from ghost_agent.router import (
    EMBED_DIM,
    FEATURE_NAMES,
    ComplexityClassifier,
    ComplexityDispatcher,
    embed_text,
    embed_texts,
    embeddings_enabled,
    extract_features,
    model_feature_names,
    reset_router_embedder,
    set_router_embedder,
)
from ghost_agent.router import embedding as emb_mod
from ghost_agent.router.embedding import (
    EmbeddingStatus,
    probe_router_embedder,
)
from ghost_agent.router.model import FeatureSchemaMismatch
from ghost_agent.router.trainer import _jaccard


# ------------------------------------------------------- signal corpus

_HARD_MARK, _EASY_MARK = "alpha", "bravo"


def _vec_for(text, dim=EMBED_DIM):
    """Deterministic vector that CARRIES the label signal in one column."""
    rng = np.random.default_rng(zlib.crc32(str(text).encode()) % (2**32))
    v = rng.normal(size=dim) * 0.01
    if _HARD_MARK in str(text):
        v[0] = 1.0
    elif _EASY_MARK in str(text):
        v[0] = -1.0
    return v


def _signal_corpus(n=60):
    """Two classes that are LEXICALLY indistinguishable.

    `task alpha 7` vs `task bravo 7`: same length, word count, punctuation,
    jargon count, digit density — every lexical feature is equal. Only the
    embedding can tell them apart, so any test built on this corpus is
    sensitive to the embedding actually being used, correctly aligned, and
    taken from the request under test.
    """
    texts = [f"task {_HARD_MARK} {i}" for i in range(n)]
    ys = ["hard"] * n
    texts += [f"task {_EASY_MARK} {i}" for i in range(n)]
    ys += ["easy"] * n
    return texts, ys


def _fake_embedder(dim=EMBED_DIM, count=None, value=None, record=None):
    """Stand-in for the vector store's embedding_fn.

    `dim`/`count`/`value` forge the malformed outputs a real embedder can
    produce (wrong model width, batch mismatch, NaN). `record` captures
    the texts it was handed, so a test can assert WHAT was embedded.
    """
    def fn(texts):
        texts = list(texts)
        if record is not None:
            record.extend(texts)
        n = len(texts) if count is None else count
        out = []
        for i in range(n):
            t = texts[i] if i < len(texts) else ""
            if value is not None:
                out.append([value] * dim)
            elif dim != EMBED_DIM:
                out.append(list(np.zeros(dim)))
            else:
                out.append(_vec_for(t, dim))
        return out
    return fn


def _fit(use_embeddings, epochs=400, texts=None, ys=None):
    if texts is None:
        texts, ys = _signal_corpus()
    embs = [_vec_for(t) for t in texts] if use_embeddings else [None] * len(texts)
    X = [extract_features(t, embedding=e) for t, e in zip(texts, embs)]
    clf = ComplexityClassifier(epochs=epochs)
    clf.fit(X, ys)
    return clf


def _passing_gate_report(clf):
    """Held-out evidence that genuinely clears `gate_verdict`."""
    return {
        "weights_sha": clf.weights_fingerprint(),
        "n": max(200, ComplexityClassifier._GATE_MIN_HELDOUT),
        "classes": 2,
        # accuracy DERIVED from the counts — the gate enforces
        # accuracy - baseline == (win - lose)/n.
        "accuracy": 0.55 + (60 - 5) / max(200, ComplexityClassifier._GATE_MIN_HELDOUT),
        "baseline": 0.55,
        "false_easy_on_hard": 0.05,
        "discordant_win": 60,
        "discordant_lose": 5,
    }


def _accuracy(clf, texts, ys, embedder=_vec_for):
    ok = 0
    for t, y in zip(texts, ys):
        e = embedder(t) if clf.uses_embeddings_ else None
        ok += clf.predict(extract_features(t, embedding=e))[0] == y
    return ok / len(ys)


# ---------------------------------------------------------- the premise

def test_the_signal_corpus_is_lexically_indistinguishable():
    """The premise every discrimination test below rests on.

    If this ever fails, those tests could pass on lexical features alone
    and would stop proving anything about embeddings."""
    texts, ys = _signal_corpus()
    hard = [t for t, y in zip(texts, ys) if y == "hard"]
    easy = [t for t, y in zip(texts, ys) if y == "easy"]
    for h, e in zip(hard, easy):
        assert extract_features(h).values == extract_features(e).values


# ---------------------------------------------------------------- features

class TestFeatureSchema:
    def test_lexical_only_is_unchanged(self):
        fv = extract_features("hello world")
        assert len(fv.values) == len(FEATURE_NAMES)
        assert fv.embedding is None

    def test_embedding_appends_after_every_lexical_feature(self):
        emb = [0.5] * EMBED_DIM
        base = extract_features("hello world")
        fv = extract_features("hello world", embedding=emb)
        assert fv.values[:len(FEATURE_NAMES)] == base.values
        assert list(fv.values[len(FEATURE_NAMES):]) == emb
        assert fv.embedding == tuple(emb)

    def test_by_name_maps_each_name_to_its_own_value(self):
        """Pins name→VALUE correspondence, not merely name ORDER.

        The previous assertion compared `tuple(fv.by_name)[-1]` against
        `model_feature_names(True)[-1]` — a tautology, since both derive
        from EMBED_FEATURE_NAMES. It passed happily when the embedding
        was zipped in REVERSED."""
        emb = [float(i) for i in range(EMBED_DIM)]
        fv = extract_features("hello world", embedding=emb)
        rebuilt = [fv.by_name[n] for n in model_feature_names(True)]
        assert rebuilt == list(fv.values)

    def test_wrong_width_embedding_raises_not_pads(self):
        for bad in (EMBED_DIM - 1, EMBED_DIM + 1, 0, 1):
            with pytest.raises(ValueError):
                extract_features("x", embedding=[0.1] * bad)

    def test_model_feature_names_matches_both_schemas(self):
        assert model_feature_names(False) == FEATURE_NAMES
        combined = model_feature_names(True)
        assert combined[:len(FEATURE_NAMES)] == FEATURE_NAMES
        assert len(combined) == len(FEATURE_NAMES) + EMBED_DIM
        assert len(set(combined)) == len(combined)


# ------------------------------------------------------------------- model

class TestModelSchema:
    def test_fit_adopts_the_schema_the_data_arrived_in(self):
        for use_emb in (False, True):
            clf = _fit(use_emb)
            assert clf.uses_embeddings_ is use_emb
            assert clf.feature_names_ == model_feature_names(use_emb)
            assert clf.weights_.shape[0] == len(model_feature_names(use_emb))
            assert len(clf.report_.weights) == clf.weights_.shape[0]

    def test_refitting_the_same_instance_resets_the_representation(self):
        """The reset branch is only reachable by REUSING a classifier —
        a fresh instance per case never exercises it."""
        clf = ComplexityClassifier(epochs=50)
        texts, ys = _signal_corpus()
        clf.fit([extract_features(t, embedding=_vec_for(t)) for t in texts], ys)
        assert clf.uses_embeddings_ is True
        clf.fit([extract_features(t) for t in texts], ys)
        assert clf.uses_embeddings_ is False
        assert clf.feature_names_ == FEATURE_NAMES

    def test_fit_rejects_an_unrecognised_width(self):
        X = [np.zeros(99) for _ in range(4)]
        with pytest.raises(ValueError):
            ComplexityClassifier(epochs=5).fit(X, ["easy", "hard"] * 2)

    def test_embedding_model_refuses_a_lexical_vector(self):
        clf = _fit(True)
        with pytest.raises(FeatureSchemaMismatch):
            clf.predict(extract_features("what is 2+2"))
        with pytest.raises(FeatureSchemaMismatch):
            clf.predict_from_text("what is 2+2")

    def test_lexical_model_refuses_an_embedding_vector(self):
        """The MIRROR direction. Testing only short-vectors let the guard
        be weakened from `!=` to `<` with every test still green — and a
        402-vector against 18 weights then raised a raw ValueError from
        np.dot instead of the named error the dispatcher contracts on."""
        clf = _fit(False)
        with pytest.raises(FeatureSchemaMismatch):
            clf.predict(extract_features("x", embedding=_vec_for("x")))

    def test_non_1d_input_is_refused(self):
        """A (402, 1) column matched on shape[0], produced a shape-(1,)
        dot product, and float() accepted it — scoring cleanly through
        the only guard that exists to stop that."""
        clf = _fit(True)
        with pytest.raises(FeatureSchemaMismatch):
            clf.predict(np.zeros((len(model_feature_names(True)), 1)))

    def test_online_updates_also_enforce_the_width(self):
        """`fit` may adopt a new representation; `partial_fit` REFINES
        existing weights and must not accept a differently-shaped row —
        it would otherwise apply a gradient step from a vector that means
        something else."""
        clf = _fit(True, epochs=50)
        with pytest.raises(FeatureSchemaMismatch):
            clf.partial_fit([extract_features("x")], ["easy"])

    def test_embedding_model_scores_with_an_embedding(self):
        clf = _fit(True)
        label, conf = clf.predict_from_text(
            f"task {_HARD_MARK} 999",
            embedding=_vec_for(f"task {_HARD_MARK} 999"))
        assert label == "hard"
        assert 0.0 <= conf <= 1.0

    def test_clone_preserves_representation_and_evidence(self):
        clf = _fit(True)
        # A gate report that actually PASSES. With a stub report both the
        # original and the clone are False, so the assertion below held
        # under either hypothesis and could not detect a dropped
        # gate_report_ — the evidence has to be real to be evidence.
        clf.gate_report_ = _passing_gate_report(clf)
        assert clf.looks_sane() is True

        m = clf.clone()
        assert m.uses_embeddings_ is True
        assert m.feature_names_ == clf.feature_names_
        # Without gate_report_ the gate is fail-closed, so a clone could
        # never be installed by any caller that checks looks_sane().
        assert m.looks_sane() is True


class TestTrainingUsesTheRightEmbedding:
    def test_each_row_is_paired_with_its_own_embedding(self):
        """Scrambling text↔embedding must destroy the model.

        Nothing previously asserted alignment: rows trained against
        ANOTHER row's embedding still fit, still reported
        uses_embeddings=True, and still cleared the held-out gate."""
        texts, ys = _signal_corpus()
        aligned = ComplexityClassifier(epochs=400)
        aligned.fit([extract_features(t, embedding=_vec_for(t))
                     for t in texts], ys)

        scrambled = ComplexityClassifier(epochs=400)
        rev = [_vec_for(t) for t in texts][::-1]
        scrambled.fit([extract_features(t, embedding=e)
                       for t, e in zip(texts, rev)], ys)

        assert _accuracy(aligned, texts, ys) > 0.95
        assert _accuracy(scrambled, texts, ys) < 0.55


class TestPersistence:
    @pytest.mark.parametrize("use_emb", [False, True])
    def test_round_trip_restores_the_representation(self, tmp_path, use_emb):
        clf = _fit(use_emb)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        back = ComplexityClassifier.load(p)
        assert back.uses_embeddings_ is use_emb
        assert back.feature_names_ == model_feature_names(use_emb)
        assert np.allclose(back.weights_, clf.weights_)

    def test_reordered_names_are_rejected(self, tmp_path):
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        raw["feature_names"][0], raw["feature_names"][1] = (
            raw["feature_names"][1], raw["feature_names"][0])
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ComplexityClassifier.load(p)

    def test_names_and_weights_must_agree_in_length(self, tmp_path):
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        raw["weights"] = raw["weights"][:-1]
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ComplexityClassifier.load(p)

    def test_unknown_schema_width_is_rejected(self, tmp_path):
        clf = _fit(False)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        raw["feature_names"] = raw["feature_names"] + ["bogus"]
        raw["weights"] = raw["weights"] + [0.1]
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ComplexityClassifier.load(p)

    def test_feature_names_outrank_the_diagnostic_flag(self, tmp_path):
        """`uses_embeddings` in the JSON is documented as diagnostic and
        `feature_names` as authoritative — pinned here, because a loader
        that trusted the flag would disagree with the weights it loaded."""
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        raw["uses_embeddings"] = False          # lie
        p.write_text(json.dumps(raw))
        assert ComplexityClassifier.load(p).uses_embeddings_ is True


class TestGateRejectsNoiseAtThisWidth:
    """§4BQ took the model from 18 to 402 parameters and the deploy gate
    silently stopped doing its job.

    Its only quantitative test was `accuracy > baseline`, with no margin.
    An 18-feature fit on signal-free labels degenerates to
    constant-predict, so accuracy == baseline EXACTLY and the tie rejected
    it — the protection was incidental, not designed. A 402-parameter fit
    on ~140 training rows has enough variance to land a hair above
    baseline: measured through the production trainer on shuffled labels,
    lexical-18 deployed noise in 1/14 runs while combined-402 deployed it
    in 6/14 at the corpus floor and 7/14 at n=600, winning by 0.003-0.033
    — one to three held-out turns."""

    @staticmethod
    def _ev(**kw):
        """Evidence whose `accuracy` is DERIVED from its own counts.

        The gate enforces `accuracy - baseline == (win - lose)/n`, because
        that identity is the only reason the paired test is valid. Hand-
        written literals violated it constantly (an "accuracy 0.95" beside
        counts implying 0.66), so they are computed here instead — a test
        fixture that contradicts itself pins nothing."""
        ev = {"n": 445, "classes": 2, "baseline": 0.544,
              "majority_baseline": 0.544, "false_easy_on_hard": 0.05,
              "discordant_win": 100, "discordant_lose": 18}
        ev.update(kw)
        if "accuracy" not in kw:
            try:
                ev["accuracy"] = (ev["baseline"]
                                  + (ev["discordant_win"] - ev["discordant_lose"])
                                  / max(1, ev["n"]))
            except (TypeError, ValueError, OverflowError):
                # The FIXTURE must survive the malformed inputs the tests
                # exist to feed the gate — otherwise it crashes first and
                # the gate is never exercised at all.
                ev["accuracy"] = 0.70
        return ev

    def test_the_gate_uses_the_PAIRED_test_when_counts_are_present(self):
        """`accuracy - baseline == (win - lose)/n` exactly, so McNemar on
        the discordant pairs is the correct one-sided test. A first
        version substituted 1.645*sqrt(p(1-p)/n) — a CONSTANT for
        (win+lose) — which ignores how aggressive the model is: measured
        up to 3.6x too strict on a cautious model and 1.5-1.9x too lax on
        an aggressive one."""
        # 30 vs 25: a +5 edge on 55 discordant pairs is noise.
        passed, why = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=30, discordant_lose=25,
                     accuracy=0.544 + 5 / 445))
        assert passed is False
        assert "turns fixed" in why and "one-sided p=" in why

    def test_the_paired_test_counts_what_was_BROKEN_not_only_what_was_fixed(self):
        """Same wins, different losses, must flip the verdict — otherwise
        the test is measuring aggressiveness, not skill."""
        ok, _ = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=40, discordant_lose=2))
        bad, _ = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=40, discordant_lose=38))
        assert ok is True and bad is False

    def test_noise_is_rejected_at_EVERY_hard_rate(self):
        """The null is the POPULATION easy-rate, not a coin flip.

        A label-independent model's easy-calls are genuinely easy at rate
        (1 - hard_rate), so on an easy-dominated corpus it beats
        always-hard by q(1-2p) in expectation and a 0.5-null McNemar
        eventually passes it. Testing win ~ Binomial(win+lose,
        1-hard_rate) is calibrated at every rate — and, unlike the
        majority-class guard it replaced, does not reject a genuinely
        significant model on the same corpus."""
        for hard_rate in (0.30, 0.35, 0.45, 0.544, 0.70):
            n_disc, p_easy = 200, 1.0 - hard_rate
            noise_w = int(round(n_disc * p_easy))
            assert ComplexityClassifier.gate_verdict(self._ev(
                baseline=hard_rate, discordant_win=noise_w,
                discordant_lose=n_disc - noise_w))[0] is False, hard_rate
            # ...while a real edge on the SAME corpus still deploys.
            real_w = int(round(n_disc * min(0.95, p_easy + 0.25)))
            assert ComplexityClassifier.gate_verdict(self._ev(
                baseline=hard_rate, discordant_win=real_w,
                discordant_lose=n_disc - real_w))[0] is True, hard_rate

    def test_evaluate_emits_what_the_gate_needs(self):
        """The counts and the majority baseline must TRAVEL with the
        evidence, or the gate silently falls back to the proxy."""
        texts, ys = _signal_corpus()
        clf = _fit(True, texts=texts, ys=ys)
        X = [extract_features(t, embedding=_vec_for(t)) for t in texts]
        ev = clf.evaluate(X, ys)
        assert ev["discordant_win"] >= 0 and ev["discordant_lose"] >= 0
        # The identity the paired test rests on, asserted numerically.
        assert abs((ev["accuracy"] - ev["baseline"])
                   - (ev["discordant_win"] - ev["discordant_lose"]) / ev["n"]) < 1e-9

    def test_majority_baseline_differs_from_escalate_all_when_easy_dominates(self):
        texts = [f"task {_EASY_MARK} {i}" for i in range(80)] + \
                [f"task {_HARD_MARK} {i}" for i in range(20)]
        ys = ["easy"] * 80 + ["hard"] * 20
        clf = _fit(True)
        X = [extract_features(t, embedding=_vec_for(t)) for t in texts]
        ev = clf.evaluate(X, ys)
        assert ev["baseline"] == pytest.approx(0.20)
        assert ev["majority_baseline"] == pytest.approx(0.80)

    def test_the_corpus_floor_stayed_at_60(self):
        """I raised this to 150 twice and was wrong both times, by the
        same self-confirming artifact: measuring "does a real fit deploy
        at corpus 200?" with the raise already installed, so the runs were
        rejected by the SIZE check and never reached the gate. Measured
        correctly, a real fit deploys at every corpus size 200-499, and 6
        wins / 0 losses at baseline 0.546 gives p=0.0087 — 60 held-out
        turns can demonstrate significance perfectly well. A size floor
        cannot lower a significance test's type-I rate; only the test
        can."""
        assert ComplexityClassifier._GATE_MIN_HELDOUT == 60
        # Just under the floor: rejected for SIZE.
        passed, why = ComplexityClassifier.gate_verdict(
            self._ev(n=59, discordant_win=20, discordant_lose=1))
        assert passed is False and "held-out n" in why
        # Exactly at it: allowed to be judged on its evidence, and a
        # decisive 40-vs-1 on 60 held-out turns DOES deploy — the claim
        # that "60 held-out cannot demonstrate significance" was false.
        assert ComplexityClassifier.gate_verdict(
            self._ev(n=60, discordant_win=20,
                     discordant_lose=1))[0] is True

    def test_the_minimum_effect_size_SCALES_with_the_held_out_set(self):
        """A fixed count is an absolute bar against a variable-sized
        held-out set. At 30 it demanded a 60-turn held-out set be
        rerouted by half, and measured, it cost ALL real deployment at
        corpus 200 and 96% at 300 to avoid one noise deploy in 300 runs —
        while its motivating case (win=4, lose=0) was already rejected by
        the alpha. Scale-free instead."""
        import math as _m
        need = lambda n: max(ComplexityClassifier._GATE_MIN_DISCORDANT_FLOOR,
                             _m.ceil(ComplexityClassifier._GATE_MIN_DISCORDANT_FRACTION * n))
        assert need(60) < need(445) < need(1482)
        # A small held-out set is judged on its evidence, not blocked.
        assert ComplexityClassifier.gate_verdict(
            self._ev(n=60, discordant_win=20,
                     discordant_lose=2))[0] is True

    # ── MALFORMED / HOSTILE EVIDENCE ────────────────────────────────
    # Evidence is `raw.get("gate_report")` straight from checkpoint JSON,
    # and `load()` never validates it, so a hand-edited or partially
    # written file reaches this function verbatim. Each guard below was
    # added after a review found the gate either CRASHING (contained only
    # by every caller's try/except) or FAILING OPEN. Each is pinned
    # separately, because "the gate returned False" cannot distinguish
    # which guard caught it.

    @pytest.mark.parametrize("field", ["n", "classes"])
    @pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf")])
    def test_a_malformed_size_field_does_not_crash(self, field, bad):
        """`n` and `classes` were coerced with a bare int() ABOVE the try
        that a previous round said covered "every numeric field"."""
        assert ComplexityClassifier.gate_verdict(self._ev(**{field: bad}))[0] is False

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
    def test_infinite_counts_do_not_crash(self, bad):
        """int(inf) raises OverflowError, which is NOT a ValueError — so
        it slipped past the inner guard that listed the other three."""
        assert ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=bad))[0] is False

    @pytest.mark.parametrize("field", ["baseline", "accuracy", "false_easy_on_hard"])
    def test_a_non_finite_rate_is_REJECTED_not_ignored(self, field):
        """THE fail-open: `nan > 0.25` is False, so a NaN false-easy rate
        sailed through the one check that protects capability. Caught by
        the RANGE guard (every comparison against NaN is False), asserted
        by reason so this pins which guard rather than merely that
        something rejected it."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            passed, why = ComplexityClassifier.gate_verdict(
                self._ev(**{field: bad}))
            assert passed is False and "out-of-range" in why

    @pytest.mark.parametrize("field,val", [
        ("baseline", 5.0), ("baseline", -1.0), ("accuracy", 1.5),
        ("false_easy_on_hard", -0.5)])
    def test_an_out_of_range_rate_is_rejected(self, field, val):
        """`baseline=5.0` clamped the null easy-rate to 1e-9 and made
        everything look significant — fail-open by arithmetic."""
        passed, why = ComplexityClassifier.gate_verdict(self._ev(**{field: val}))
        assert passed is False and "out-of-range" in why

    def test_evidence_that_contradicts_itself_is_rejected(self):
        """The paired test is valid ONLY because
        accuracy - baseline == (win - lose)/n. Without enforcing it here,
        evidence could PASS while its own reason string was arithmetically
        false — and six test fixtures in this repo turned out to violate
        it, one implying 100.4% accuracy."""
        passed, why = ComplexityClassifier.gate_verdict(
            self._ev(accuracy=0.10))          # counts imply ~0.728
        assert passed is False and "inconsistent" in why

    def test_more_differing_turns_than_held_out_turns_is_rejected(self):
        passed, why = ComplexityClassifier.gate_verdict(
            self._ev(n=60, discordant_win=40, discordant_lose=35))
        assert passed is False and "inconsistent" in why

    def test_rounding_in_stored_evidence_is_tolerated(self):
        """The live checkpoint stores rates rounded, so its own report is
        off by ~3e-5 from the identity. An exact comparison would reject
        the shipped model."""
        assert ComplexityClassifier.gate_verdict(self._ev(
            n=445, baseline=0.5438, accuracy=0.7281,
            discordant_win=100, discordant_lose=18,
            false_easy_on_hard=0.0744))[0] is True

    def test_the_effect_size_FLOOR_blocks_the_degenerate_high_hard_rate_case(self):
        """The floor is live only where the fraction rounds below it: at a
        0.75 hard-rate, 4 clean wins on 60 held-out turns is significant
        (p=0.0039) yet changes 4 decisions. Without the floor it deploys."""
        ev = self._ev(n=60, baseline=0.75, discordant_win=4, discordant_lose=0,
                      false_easy_on_hard=0.0)
        passed, why = ComplexityClassifier.gate_verdict(ev)
        assert passed is False and "too small a change" in why

    def test_the_p_value_is_the_EXACT_binomial_tail(self):
        """Pinned against a recomputed value, not a property.

        4-vs-0 sits just inside the 5% bar (0.456^4 = 0.0432) and 3-vs-0
        just outside (0.0948) — a 1-in-20 window the normal approximation
        gets wrong in both directions. Asserting only "thin evidence is
        rejected" cannot see that."""
        from ghost_agent.router.model import _binom_tail_ge
        assert _binom_tail_ge(3, 3, 0.456) == pytest.approx(0.456 ** 3)
        assert _binom_tail_ge(4, 4, 0.456) == pytest.approx(0.456 ** 4)
        # ...and the gate's verdict follows the tail across the boundary,
        # once the evidence is large enough to be allowed to matter at all
        # (4-vs-0 is now blocked earlier, by the minimum effect size —
        # significance on 4 of 445 turns is not a reason to deploy).
        assert _binom_tail_ge(40, 40, 0.456) < ComplexityClassifier._GATE_ALPHA
        assert ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=40, discordant_lose=0, ))[0] is True
        assert ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=17, discordant_lose=15, ))[0] is False

    def test_a_large_discordant_count_does_not_overflow(self):
        """The direct product overflows the binomial coefficient to `inf`
        and would return a garbage p-value rather than failing."""
        from ghost_agent.router.model import _binom_tail_ge
        v = _binom_tail_ge(2600, 5000, 0.5)
        assert 0.0 < v < 1.0 and math.isfinite(v)

    @pytest.mark.parametrize("w,l,guard", [
        (1, 0, "too small a change"), (3, 0, "too small a change"),
        (7, 2, "not significant"), (40, 35, "not significant"),
        (60, 50, "not significant"),
    ])
    def test_thin_or_balanced_evidence_is_not_enough(self, w, l, guard):
        """TWO distinct guards, asserted separately.

        A handful of differing turns is rejected for EFFECT SIZE (the
        router exists to change decisions); a large but balanced split is
        rejected for SIGNIFICANCE. Matching a substring both reasons share
        would not distinguish them, and it was 3-vs-0 — which the normal
        approximation passed at z=1.73 against an exact p of 0.125 — that
        measurably deployed label-independent models."""
        passed, why = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=w, discordant_lose=l,
                     accuracy=0.544 + (w - l) / 445))
        assert passed is False and guard in why

    def test_the_same_RATIO_needs_enough_evidence_to_count(self):
        """Power comes from the number of discordant pairs, not from the
        accuracy delta. The same win:lose ratio must be rejected when it
        rests on a handful of turns and accepted when it rests on many —
        the property the old proxy margin could not express, because it
        substituted a constant for those counts."""
        thin = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=3, discordant_lose=1, ))[0]
        thick = ComplexityClassifier.gate_verdict(
            self._ev(discordant_win=30, discordant_lose=10, ))[0]
        assert thin is False and thick is True

    def test_a_report_with_no_counts_is_rejected_not_approximated(self):
        """Fail-closed, and ONE standard. A proxy fallback would let a
        pre-§4BQ checkpoint be judged by a different (weaker) test than
        the one the shipped model faces — and that branch is reachable,
        since every checkpoint written before this change lacks counts."""
        ev = self._ev()
        del ev["discordant_win"], ev["discordant_lose"]
        passed, why = ComplexityClassifier.gate_verdict(ev)
        assert passed is False and "predates" in why

    def test_the_gate_never_raises_on_malformed_evidence(self):
        """FAIL-CLOSED was documented and the code fail-CRASHED: a missing
        `baseline` raised KeyError, and only every caller's try/except
        made it look like the guarantee held."""
        passed, why = ComplexityClassifier.gate_verdict({"n": 200, "classes": 2})
        assert passed is False and why

    def test_the_real_measured_model_still_passes(self):
        """The live margin is 0.184 against a required ~0.039 — the gate
        must reject noise WITHOUT rejecting the thing it exists to ship."""
        passed, _why = ComplexityClassifier.gate_verdict(self._ev(
            baseline=0.5438, accuracy=0.7281, false_easy_on_hard=0.0744,
            discordant_win=100, discordant_lose=18))
        assert passed is True


class TestEmbedderIdentity:
    """Width is not identity — the CRITICAL finding of the R1 review."""

    def test_checkpoint_records_which_embedder_trained_it(self, tmp_path):
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        assert raw["embed_model"] == emb_mod.current_embed_model_name()

    def test_a_different_embedder_is_rejected(self, tmp_path, monkeypatch):
        """The GHOST_EMBED_MODEL migration path. MiniLM is also 384-d and
        also cached locally, so width cannot catch this."""
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        monkeypatch.setattr(emb_mod, "current_embed_model_name",
                            lambda: "sentence-transformers/all-MiniLM-L6-v2")
        with pytest.raises(ValueError, match="embedder"):
            ComplexityClassifier.load(p)

    def test_a_checkpoint_that_cannot_name_its_embedder_is_rejected(self, tmp_path):
        """Fail-closed: 'unknown' is not evidence of 'the same'. This is
        also what retires every pre-guard checkpoint."""
        clf = _fit(True)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        raw = json.loads(p.read_text())
        del raw["embed_model"]
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            ComplexityClassifier.load(p)

    def test_lexical_checkpoints_are_unaffected(self, tmp_path, monkeypatch):
        """A lexical model uses no embedder, so an embedder change must
        not retire it."""
        clf = _fit(False)
        clf.gate_report_ = {"weights_sha": clf.weights_fingerprint()}
        p = clf.save(tmp_path / "ck.json")
        monkeypatch.setattr(emb_mod, "current_embed_model_name",
                            lambda: "something/else")
        assert ComplexityClassifier.load(p).uses_embeddings_ is False


# --------------------------------------------------------------- embedder

class TestEmbedder:
    def test_kill_switch(self, monkeypatch):
        monkeypatch.delenv("GHOST_ROUTER_EMBED", raising=False)
        assert embeddings_enabled() is True
        for on in ("", "  ", "1", "yes"):
            monkeypatch.setenv("GHOST_ROUTER_EMBED", on)
            assert embeddings_enabled() is True
        for off in ("0", "false", "no", "off", "OFF", " off "):
            monkeypatch.setenv("GHOST_ROUTER_EMBED", off)
            assert embeddings_enabled() is False

    def test_kill_switch_gates_serving_not_only_training(self, monkeypatch):
        """The revert lever must actually revert.

        Only the trainer consulted the switch, so a 402-dim model restored
        from disk under GHOST_ROUTER_EMBED=0 still computed and consumed
        embeddings — contradicting the contract on the one path an
        operator uses to back the flip out."""
        set_router_embedder(_fake_embedder())
        assert embed_text("hi") is not None
        monkeypatch.setenv("GHOST_ROUTER_EMBED", "0")
        assert embed_text("hi") is None
        assert embed_texts(["hi"]) is None

    def test_no_embedder_returns_none(self):
        reset_router_embedder()
        assert embed_text("hi") is None
        assert embed_texts(["hi"]) is None

    def test_non_callable_registration_is_ignored(self):
        """Pinned on the OBSERVABLE difference. `embed_text` alone returns
        None either way (a non-callable raises TypeError inside `_call`
        and is swallowed), so asserting only that could not tell the
        filter from its absence."""
        from ghost_agent.router import get_router_embedder
        set_router_embedder("not a function")
        assert get_router_embedder() is None
        assert embed_text("hi") is None

    def test_empty_batch_returns_none(self):
        set_router_embedder(_fake_embedder())
        assert embed_texts([]) is None

    @pytest.mark.parametrize("dim", [EMBED_DIM - 1, EMBED_DIM + 1, 1])
    def test_rejects_wrong_width_in_both_directions(self, dim):
        set_router_embedder(_fake_embedder(dim=dim))
        assert embed_text("hi") is None
        assert embed_texts(["hi", "there"]) is None

    @pytest.mark.parametrize("count", [1, 5])
    def test_rejects_batch_count_mismatch_in_both_directions(self, count):
        set_router_embedder(_fake_embedder(count=count))
        assert embed_texts(["a", "b", "c"]) is None

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            set_router_embedder(_fake_embedder(value=bad))
            assert embed_text("hi") is None

    def test_raising_embedder_is_not_fatal(self):
        def boom(texts):
            raise RuntimeError("model unloaded")
        set_router_embedder(boom)
        assert embed_text("hi") is None
        assert embed_texts(["hi"]) is None

    def test_supports_the_chroma_keyword_convention(self):
        """Live code drives `memory_system.embedding_fn` with `input=`
        (see tests/test_metacog_wiring.py), so the keyword fallback is
        load-bearing, not defensive decoration."""
        calls = {}

        def keyword_only(*, input):  # noqa: A002 — chroma's own name
            calls["seen"] = list(input)
            return [_vec_for(t) for t in input]

        set_router_embedder(keyword_only)
        assert embed_text("hi") is not None
        assert calls["seen"] == ["hi"]

    def test_distinct_texts_get_distinct_vectors(self):
        """Trivially true of a correct implementation, and the only thing
        that catches a lookup which ignores its key."""
        set_router_embedder(_fake_embedder())
        a = embed_text(f"task {_HARD_MARK} 1")
        b = embed_text(f"task {_EASY_MARK} 1")
        assert a is not None and b is not None
        assert a != b

    def test_text_is_capped_identically_for_train_and_serve(self):
        """The cap lives in ONE place so a long request cannot be
        embedded differently at fit time and at serve time."""
        seen = []
        set_router_embedder(_fake_embedder(record=seen))
        long_text = "x" * (emb_mod._MAX_EMBED_CHARS + 5000)
        embed_text(long_text)
        embed_texts([long_text])
        assert seen and all(len(s) == emb_mod._MAX_EMBED_CHARS for s in seen)


class TestProbeStatus:
    """The boot decision, extracted from main.py so it can be tested at
    all — it previously had zero coverage in any direction."""

    def test_no_embedder_is_degraded_and_does_not_persist(self):
        reset_router_embedder()
        s = probe_router_embedder()
        assert (s.enabled, s.available) == (True, False)
        assert s.degraded is True

    def test_working_embedder_is_available(self):
        set_router_embedder(_fake_embedder())
        s = probe_router_embedder()
        assert s.available is True
        assert s.degraded is False

    def test_registered_but_broken_embedder_is_not_available(self):
        """Registration is not capability. Trusting the object's mere
        presence made the fit degrade to lexical while boot still wanted
        embeddings — a retrain loop that never converges."""
        set_router_embedder(_fake_embedder(dim=7))
        s = probe_router_embedder()
        assert s.available is False
        assert s.degraded is True

    def test_kill_switch_is_operator_intent_and_persists(self, monkeypatch):
        monkeypatch.setenv("GHOST_ROUTER_EMBED", "0")
        set_router_embedder(_fake_embedder())
        s = probe_router_embedder()
        assert (s.enabled, s.available) == (False, False)
        # NOT degraded: the operator asked for lexical, so the resulting
        # lexical checkpoint is legitimate and is written to disk (the
        # policy itself lives in RouterTrainer.run — see
        # TestTrainerRepresentation).
        assert s.degraded is False

    def test_status_reports_the_live_embedder_name(self):
        assert probe_router_embedder().model == emb_mod.current_embed_model_name()


# -------------------------------------------------------------- dispatcher

class TestDispatcherFailSafe:
    def test_escalates_when_the_embedder_is_missing(self):
        clf = _fit(True)
        reset_router_embedder()
        d = ComplexityDispatcher(clf, confidence_threshold=0.0)
        r = d.route("what is 2+2")
        assert r.escalated is True
        assert r.label == "hard"
        # Pin WHICH guard fired. Asserting only `escalated` could not
        # distinguish "the dispatcher checked for an embedding" from "the
        # model's width guard saved us" — the early return was removable
        # with every test still green.
        assert r.escalation_kind == "no_embedding"
        assert set(d.HARD_POOLS).issubset(set(r.allowed_pools))

    def test_escalates_when_the_embedder_is_broken(self):
        clf = _fit(True)
        set_router_embedder(_fake_embedder(dim=7))
        r = ComplexityDispatcher(clf, confidence_threshold=0.0).route("hi")
        assert r.escalated is True
        assert r.escalation_kind == "no_embedding"

    def test_scoring_failure_degrades_instead_of_raising(self):
        clf = _fit(True)
        set_router_embedder(_fake_embedder())

        class Boom:
            uses_embeddings_ = True
            weights_ = np.zeros(len(model_feature_names(True)))

            def predict(self, x):
                raise RuntimeError("scoring exploded")

        r = ComplexityDispatcher(Boom(), confidence_threshold=0.0).route("hi")
        assert r.escalated is True
        assert r.escalation_kind == "scoring_error"

    def test_untrained_router_is_distinguishable_from_a_failed_embed(self):
        r = ComplexityDispatcher(None).route("hi")
        assert r.escalation_kind == "no_model"

    def test_low_confidence_is_distinguishable_from_a_failed_embed(self):
        clf = _fit(True)
        set_router_embedder(_fake_embedder())
        r = ComplexityDispatcher(clf, confidence_threshold=1.01).route("hi")
        assert r.escalated is True
        assert r.escalation_kind == "low_confidence"

    def test_routing_depends_on_the_request_text(self):
        """THE central behavioural pin.

        The two requests are lexically identical, so only the embedding
        can separate them. A dispatcher that embedded a constant string —
        making the entire feature inert — passed every earlier test."""
        texts, ys = _signal_corpus()
        clf = _fit(True, texts=texts, ys=ys)
        set_router_embedder(_fake_embedder())
        d = ComplexityDispatcher(clf, confidence_threshold=0.0)
        assert d.route(f"task {_HARD_MARK} 999").label == "hard"
        assert d.route(f"task {_EASY_MARK} 999").label == "easy"

    def test_the_request_itself_is_what_gets_embedded(self):
        seen = []
        clf = _fit(True)
        set_router_embedder(_fake_embedder(record=seen))
        ComplexityDispatcher(clf, confidence_threshold=0.0).route("route me")
        assert seen == ["route me"]

    def test_constructor_embed_fn_is_used(self):
        """The `embed_fn=` override was entirely dead as far as the suite
        was concerned."""
        seen = []
        clf = _fit(True)
        reset_router_embedder()          # nothing registered globally
        d = ComplexityDispatcher(clf, confidence_threshold=0.0,
                                 embed_fn=_fake_embedder(record=seen))
        assert d.route(f"task {_HARD_MARK} 5").escalated is False
        assert seen == [f"task {_HARD_MARK} 5"]

    def test_lexical_model_needs_no_embedder(self):
        clf = _fit(False, texts=[f"write and deploy service {i}" for i in range(60)]
                   + [f"what is {i}?" for i in range(60)],
                   ys=["hard"] * 60 + ["easy"] * 60)
        reset_router_embedder()
        assert ComplexityDispatcher(clf, confidence_threshold=0.0
                                    ).route("what is 2+2").escalated is False

    def test_a_route_never_raises(self):
        """The dispatcher's contract, across every broken embedder shape."""
        clf = _fit(True)
        for bad in (_fake_embedder(dim=3), _fake_embedder(count=9),
                    _fake_embedder(value=float("nan")),
                    lambda t: (_ for _ in ()).throw(RuntimeError("x"))):
            set_router_embedder(bad)
            assert ComplexityDispatcher(clf).route("hi").escalated is True


# ----------------------------------------------------------------- trainer

def _traj(req, steps, calls, outcome="passed", heavy=False):
    from ghost_agent.distill.schema import Trajectory, ToolCall
    tcs = [ToolCall(name=("execute" if heavy else "web_search"))
           for _ in range(calls)]
    t = Trajectory(user_request=req, n_steps=steps, tool_calls=tcs,
                   outcome=outcome)
    # STABLE identity, derived from the request. `Trajectory.id` defaults
    # to a fresh uuid4, so rebuilding "the same corpus" produced a
    # completely different id set — and the looks control compares corpus
    # IDENTITY. On disk a trajectory keeps its id, so a fixture that
    # re-randomises it is not modelling production.
    t.id = f"t-{req}"
    return t


def _corpus_floor_each():
    """Per-class size that clears the trainer's DERIVED corpus floor.

    §4BQ left `_GATE_MIN_HELDOUT` at 60 — two attempts to raise it to
    150 were retracted, both justified by self-confirming artifacts
    (see the constant). Sizes are DERIVED so a future recalibration
    cannot silently turn these into vacuous bails; that had already
    happened once, to the bench-cap test."""
    from ghost_agent.router.trainer import _gate_min_trajectories
    return int(_gate_min_trajectories() * 0.6) + 1


def _balanced(n_each=None, tag=""):
    """`tag` keeps a bench corpus TEXTUALLY DISJOINT from the real one.

    Without it both corpora emit the same request strings, so a stub that
    fails "only the bench batch" also failed the real batch — and the
    all-or-nothing test below passed via the wrong branch entirely."""
    n_each = _corpus_floor_each() if n_each is None else n_each
    easy = [_traj(f"what is {tag}{i}?", 1, 1) for i in range(n_each)]
    hard = [_traj(f"build deploy {tag}{i}", 6, 5, outcome="failed", heavy=True)
            for i in range(n_each)]
    return easy + hard


def _signal_trajectories(n_each=None):
    n_each = _corpus_floor_each() if n_each is None else n_each
    easy = [_traj(f"task {_EASY_MARK} {i}", 1, 1) for i in range(n_each)]
    hard = [_traj(f"task {_HARD_MARK} {i}", 6, 5, outcome="failed", heavy=True)
            for i in range(n_each)]
    return easy + hard


class TestTrainerRepresentation:
    def test_trains_with_embeddings_when_available(self):
        from ghost_agent.router import RouterTrainer
        set_router_embedder(_fake_embedder())
        t = RouterTrainer()
        r = t.run(trajectories=_balanced())
        assert r.fit_succeeded is True
        assert r.uses_embeddings is True
        assert t.classifier.uses_embeddings_ is True
        assert t.classifier.weights_.shape[0] == len(model_feature_names(True))

    def test_the_trained_model_separates_on_embedding_signal_alone(self):
        """End-to-end through the REAL trainer, on a corpus whose classes
        are lexically identical: if the trainer paired any row with
        another row's embedding, this cannot pass."""
        from ghost_agent.router import RouterTrainer
        set_router_embedder(_fake_embedder())
        t = RouterTrainer()
        r = t.run(trajectories=_signal_trajectories())
        assert r.fit_succeeded is True and r.uses_embeddings is True
        d = ComplexityDispatcher(t.classifier, confidence_threshold=0.0)
        assert d.route(f"task {_HARD_MARK} 9001").label == "hard"
        assert d.route(f"task {_EASY_MARK} 9001").label == "easy"

    def test_degrades_to_lexical_when_the_embedder_is_gone(self):
        from ghost_agent.router import RouterTrainer
        reset_router_embedder()
        t = RouterTrainer()
        r = t.run(trajectories=_balanced())
        assert r.fit_succeeded is True
        assert r.uses_embeddings is False
        assert t.classifier.weights_.shape[0] == len(FEATURE_NAMES)

    def test_kill_switch_forces_lexical(self, monkeypatch):
        from ghost_agent.router import RouterTrainer
        monkeypatch.setenv("GHOST_ROUTER_EMBED", "0")
        set_router_embedder(_fake_embedder())
        t = RouterTrainer()
        r = t.run(trajectories=_balanced())
        assert r.fit_succeeded is True
        assert r.uses_embeddings is False

    @pytest.mark.parametrize("failing", ["bench", "real"])
    def test_partial_embedding_failure_never_mixes_widths(self, failing):
        """BOTH directions. Covering only the bench-fails case left the
        mirror (real fails, bench succeeds) able to stack 18-wide rows on
        402-wide ones — np.stack raises and the whole fit BAILS instead of
        degrading to lexical."""
        from ghost_agent.router import RouterTrainer

        real = _balanced()
        bench = _balanced(30, tag="bench-")
        bench_texts = {str(t.user_request) for t in bench}
        real_texts = {str(t.user_request) for t in real}
        assert not (bench_texts & real_texts)      # the premise
        target = bench_texts if failing == "bench" else real_texts
        good = _fake_embedder()

        def flaky(texts):
            texts = list(texts)
            if any(str(t) in target for t in texts):
                raise RuntimeError(f"{failing} embed failed")
            return good(texts)

        set_router_embedder(flaky)
        t = RouterTrainer()
        r = t.run(trajectories=real, bench_trajectories=bench)
        assert r.fit_succeeded is True
        assert r.uses_embeddings is False
        assert t.classifier.weights_.shape[0] == len(FEATURE_NAMES)

    def test_a_degraded_fit_never_overwrites_the_checkpoint(self, tmp_path):
        """THE persistence policy, enforced in the TRAINER so all three
        retrain sites inherit it.

        The first version of this guard lived in main.py only, so the idle
        retrain and the self-play refit still overwrote the production
        checkpoint from a `--no-memory` run — the wrapper-split class,
        reproduced by the very fix meant to close it."""
        from ghost_agent.router import RouterTrainer
        ckpt = tmp_path / "checkpoint.json"

        # A genuine 402-dim checkpoint, written by a healthy run.
        set_router_embedder(_fake_embedder())
        assert RouterTrainer().run(trajectories=_balanced(),
                                   save_path=ckpt).uses_embeddings is True
        before = ckpt.read_text()

        # Now a degraded run: embeddings still enabled, embedder gone.
        reset_router_embedder()
        r = RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        assert r.fit_succeeded is True and r.uses_embeddings is False
        assert ckpt.read_text() == before, (
            "a degraded lexical fit overwrote an embedding checkpoint")

    def test_an_UNLOADABLE_checkpoint_is_not_protected(self, tmp_path, monkeypatch):
        """Protection is for checkpoints that still WORK.

        A 402-name checkpoint the loader refuses — e.g. one trained under
        a different GHOST_EMBED_MODEL, a supported migration — was being
        protected by name-count alone. A degraded process could then never
        heal it: every boot load-failed, retrained lexical in memory, and
        refused to write. That is the never-converges failure the rule was
        narrowed to avoid, resurrected for the unloadable case."""
        from ghost_agent.router import RouterTrainer
        ckpt = tmp_path / "checkpoint.json"
        set_router_embedder(_fake_embedder())
        assert RouterTrainer().run(trajectories=_balanced(),
                                   save_path=ckpt).uses_embeddings is True

        # The embedder changes underneath it — the checkpoint no longer loads.
        monkeypatch.setattr(emb_mod, "current_embed_model_name",
                            lambda: "sentence-transformers/all-MiniLM-L6-v2")
        with pytest.raises(ValueError):
            ComplexityClassifier.load(ckpt)

        reset_router_embedder()          # ...and now the process is degraded
        r = RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        assert r.uses_embeddings is False
        assert ComplexityClassifier.load(ckpt).uses_embeddings_ is False, (
            "an unloadable checkpoint blocked its own replacement")

    def test_a_lexical_checkpoint_is_not_protected(self, tmp_path):
        """Only a RICHER checkpoint is protected. Guarding lexical ones too
        would freeze the router at whatever it first trained."""
        from ghost_agent.router import RouterTrainer
        ckpt = tmp_path / "checkpoint.json"
        reset_router_embedder()
        RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        first = ckpt.read_text()
        # Materially larger corpus — the looks control below deliberately
        # refuses to re-gate on nearly the same data.
        RouterTrainer().run(trajectories=_balanced(_corpus_floor_each() * 2),
                            save_path=ckpt)
        assert ckpt.read_text() != first, "a lexical checkpoint blocked its own refresh"

    def test_a_degraded_fit_may_still_create_a_missing_checkpoint(self, tmp_path):
        """The narrow rule matters: blanket "degraded never persists" also
        stopped a box that legitimately has no vector store from ever
        writing a checkpoint, so it retrained from scratch every boot. Only
        overwriting a RICHER model is forbidden."""
        from ghost_agent.router import RouterTrainer
        ckpt = tmp_path / "checkpoint.json"
        reset_router_embedder()
        r = RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        assert r.uses_embeddings is False
        assert ckpt.exists()

    def test_the_kill_switch_is_intent_and_does_persist(self, tmp_path, monkeypatch):
        """The mirror: an operator who asked for lexical gets a lexical
        checkpoint written. Without this the guard above would freeze the
        checkpoint forever once the switch was set."""
        from ghost_agent.router import RouterTrainer
        monkeypatch.setenv("GHOST_ROUTER_EMBED", "0")
        ckpt = tmp_path / "checkpoint.json"
        r = RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        assert r.fit_succeeded is True and r.uses_embeddings is False
        assert ckpt.exists()
        assert ComplexityClassifier.load(ckpt).uses_embeddings_ is False

    def test_a_healthy_embedding_fit_does_persist(self, tmp_path):
        from ghost_agent.router import RouterTrainer
        set_router_embedder(_fake_embedder())
        ckpt = tmp_path / "checkpoint.json"
        r = RouterTrainer().run(trajectories=_balanced(), save_path=ckpt)
        assert r.uses_embeddings is True
        assert ComplexityClassifier.load(ckpt).uses_embeddings_ is True

    def test_report_summary_names_the_representation(self):
        from ghost_agent.router import RouterTrainer
        set_router_embedder(_fake_embedder())
        assert "lexical+embedding" in RouterTrainer().run(
            trajectories=_balanced()).summary()
        reset_router_embedder()
        assert "lexical only" in RouterTrainer().run(
            trajectories=_balanced()).summary()

    def test_bootstrap_persists_and_reloads_an_embedding_model(self, tmp_path):
        """The boot path end-to-end: train → save → load → route."""
        from ghost_agent.router import bootstrap_router
        set_router_embedder(_fake_embedder())
        save = tmp_path / "router" / "checkpoint.json"
        clf, report = bootstrap_router(_signal_trajectories(), save_path=save)
        assert clf is not None and clf.uses_embeddings_ is True
        assert save.exists()
        back = ComplexityClassifier.load(save)
        assert back.uses_embeddings_ is True
        d = ComplexityDispatcher(back, confidence_threshold=0.0)
        assert d.route(f"task {_HARD_MARK} 4242").label == "hard"


class TestProductionImportShape:
    """The whole flip was DEAD in production and every test still passed.

    `main.py` imports from the PACKAGE (`from .router import
    probe_router_embedder`); the helper was exported from the SUBMODULE's
    `__all__` only. The import raised, a broad `except` logged "Embedder
    wiring skipped", no embedder was ever registered, and the router
    trained and served lexical-18 forever — the measured flip simply did
    not exist. Every test here imported from the submodule, and the boot
    tests are source greps that never execute the block.

    See [[production-import-shape-guard]] and
    [[silent-inoperative-subsystems]]."""

    def test_boot_helpers_are_importable_from_the_package(self):
        import importlib
        pkg = importlib.import_module("ghost_agent.router")
        for name in ("probe_router_embedder", "set_router_embedder",
                     "EmbeddingStatus", "current_embed_model_name",
                     "embed_text", "embed_texts", "embeddings_enabled",
                     "get_router_embedder", "reset_router_embedder"):
            assert hasattr(pkg, name), (
                f"main.py-style `from .router import {name}` would raise")

    def test_every_name_main_imports_from_router_actually_exists(self):
        """Generic: parse main.py's OWN import statements rather than
        listing names by hand, so a future helper cannot be added to the
        boot block and silently fail to import."""
        import ast
        import importlib
        from pathlib import Path
        import ghost_agent.main as m

        tree = ast.parse(Path(m.__file__).read_text())
        # Cover the SUBMODULE forms too (`from .router.embedding import X`),
        # not just the exact form that broke. Pinning only `.router` would
        # catch the instance and miss the class.
        checked = 0
        missing = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            parts = node.module.split(".")
            if "router" not in parts:
                continue
            mod = "ghost_agent." + ".".join(parts[parts.index("router"):])
            target = importlib.import_module(mod)
            for a in node.names:
                checked += 1
                if not hasattr(target, a.name):
                    missing.append(f"{mod}.{a.name}")
        assert checked, "expected main.py to import router names"
        assert not missing, f"main.py imports names that do not exist: {missing}"


class TestBootWiringSourcePins:
    """The boot block has no unit-testable seam, and a review found every
    mutation of it survived — it is the only thing that makes the flip
    take effect on an existing checkpoint. Source pins, in the style of
    `tests/test_router_boot_resilience.py::TestMainSourcePins`."""

    @staticmethod
    def _src():
        from pathlib import Path
        import ghost_agent.main as m
        return Path(m.__file__).read_text()

    def test_registers_the_raw_passage_encoder_not_embed_query(self):
        """`embed_query` prepends a BGE instruction prefix that was never
        part of the measurement — registering it would train and serve a
        representation nobody measured, silently."""
        src = self._src()
        assert 'getattr(_mem_sys, "embedding_fn", None)' in src
        # As an ATTRIBUTE FETCH, not as prose — the comment right above
        # the registration explains why embed_query is wrong, and a bare
        # substring check would fail on that explanation.
        assert '"embed_query"' not in src
        assert ".embed_query" not in src

    def test_representation_staleness_compares_for_INEQUALITY(self):
        """`==` here would retrain exactly when it should not, and never
        when it should — a perfect inversion that nothing else detects."""
        assert ('if bool(getattr(clf, "uses_embeddings_", False)) != _want_emb:'
                in self._src())

    def test_a_stale_model_is_kept_as_a_fallback(self):
        src = self._src()
        assert "_stale_clf = clf" in src
        assert "clf = _stale_clf" in src

    def test_boot_actually_registers_the_embedder(self):
        """Pinning only the `getattr` left the REGISTRATION removable —
        `_set_router_embedder(None)` passed every check, and the flip
        would simply never engage."""
        assert ("_set_router_embedder(_embed_fn if callable(_embed_fn) else None)"
                in self._src())

    def test_boot_probes_the_embedder_rather_than_trusting_registration(self):
        src = self._src()
        assert "_emb_status = _probe_embedder()" in src
        assert "_want_emb = bool(_emb_status and _emb_status.available)" in src

    def test_the_router_is_not_fed_a_feature_it_was_never_trained_on(self):
        """`context_turn_coupling` was a train/serve skew: the trainer
        builds rows from `user_request` alone, so it is identically 0.0
        across the corpus, while serving passed a real prior turn. Only
        harmless by accident — L2 decayed the unlearned weight to ~-0.003.
        Re-enabling requires the TRAINER to supply it first."""
        from pathlib import Path
        import ghost_agent.core.agent as _agent
        src = Path(_agent.__file__).read_text()
        assert "decision = dispatcher.route(last_user_content)" in src
        assert "prior_turn_text=str(prev_ai_for_router" not in src

    def test_the_restore_log_is_only_pessimistic_when_it_should_be(self):
        """An equality test told the operator "every turn will escalate"
        in the configuration that is live TODAY (lexical checkpoint,
        embedder available) — where the router in fact routes normally,
        because `route()` consults the embedder only when the MODEL asks
        for one. Only an embedding model without an embedder is unusable."""
        src = self._src()
        assert "_restored_usable = not (" in src
        assert "and not bool(_emb_status and _emb_status.available))" in src

    def test_escalation_kind_is_made_DURABLE(self):
        """Setting the field is not the point — recording it is.

        It was stamped on every escalation path and then dropped at the
        trajectory boundary, so a failed embed still landed in the 0.0-0.3
        confidence bucket of `router_confidence_backtest.py` as though it
        were something the model believed. The instrument this flip is
        justified by must not pool 'could not embed' with 'was not sure'."""
        from pathlib import Path
        import ghost_agent.core.agent as _agent
        src = Path(_agent.__file__).read_text()
        assert "router_escalation_kind=" in src

    def test_persistence_policy_is_not_reimplemented_at_this_call_site(self):
        """It belongs in RouterTrainer.run so the idle retrain and the
        self-play refit inherit it. Implemented here instead, it covered
        one of three sites — see
        TestTrainerRepresentation::test_a_degraded_fit_never_overwrites_the_checkpoint
        for the behavioural pin."""
        assert "save_path=router_ckpt_path," in self._src()

    def test_bootstrap_runs_off_the_event_loop(self):
        assert "await asyncio.to_thread(\n                    bootstrap_router," in self._src()


class TestConfidenceBacktestExcludesNonBeliefs:
    """The instrument §4BQ is justified by must not pool "could not
    embed" with "was not sure".

    `escalation_kind` was set on every escalation path and then read
    NOWHERE — `router_confidence_backtest.py` and the CUPED covariate both
    bucket on `router_confidence` alone. Measured on the live corpus: 14 of
    193 stamped turns carry confidence exactly 0.0, and those are 28% of
    the two sub-0.30 buckets the script's verdict turns on."""

    @staticmethod
    def _traj(conf, kind, outcome="failed"):
        from ghost_agent.distill.schema import Trajectory
        t = Trajectory(user_request="x", outcome=outcome)
        t.task_kind = "user_request"
        t.extra = {"router_confidence": conf}
        if kind is not None:
            t.extra["router_escalation_kind"] = kind
        return t

    def _collect(self, trajs):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "_rcb", Path("scripts/router_confidence_backtest.py").resolve())
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.collect(trajs)

    def test_non_belief_escalations_are_excluded_from_every_bucket(self):
        trajs = ([self._traj(0.0, k) for k in
                  ("no_model", "no_embedding", "scoring_error")]
                 + [self._traj(0.0, "low_confidence")])
        stats, cov = self._collect(trajs)
        assert cov["non_belief"] == 3
        assert sum(s["n"] for s in stats.values()) == 1

    def test_legacy_unstamped_zeros_are_counted_as_ambiguous(self):
        """Pre-§4BQ rows cannot be resolved after the fact — reported,
        not silently bucketed as beliefs."""
        stats, cov = self._collect([self._traj(0.0, None)])
        assert cov.get("legacy_ambiguous") == 1
        assert sum(s["n"] for s in stats.values()) == 1


class TestCupedCovariateExcludesStructuralZeros:
    """THIRD instance of the same wrapper split on this feature.

    Round 3 taught `router_confidence_backtest.py` to drop non-belief
    escalations and left the OTHER consumer of the same signal — the CUPED
    covariate — absorbing them. Measured live: 4 such zeros among 104
    pairs inflated theta +0.114 -> +0.193 (69%)."""

    @staticmethod
    def _traj(conf, kind=None):
        from ghost_agent.distill.schema import Trajectory
        t = Trajectory(user_request="x", outcome="passed")
        t.extra = {"router_confidence": conf}
        if kind is not None:
            t.extra["router_escalation_kind"] = kind
        return t

    def test_a_structural_zero_is_not_a_covariate(self):
        from ghost_agent.core.experiments import _covariate_of
        for kind in ("no_model", "no_embedding", "scoring_error"):
            assert _covariate_of(self._traj(0.0, kind)) is None

    def test_a_genuine_low_confidence_call_still_is(self):
        from ghost_agent.core.experiments import _covariate_of
        assert _covariate_of(self._traj(0.0, "low_confidence")) == 0.0
        assert _covariate_of(self._traj(0.42)) == pytest.approx(0.42)

    def test_both_consumers_share_one_definition(self):
        """They are separate modules; if the sets drift, one consumer
        silently starts counting what the other drops."""
        import importlib.util
        from pathlib import Path
        from ghost_agent.core.experiments import _NON_BELIEF_ESCALATIONS as A
        spec = importlib.util.spec_from_file_location(
            "_rcb2", Path("scripts/router_confidence_backtest.py").resolve())
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert A == m._NON_BELIEF_ESCALATIONS


class TestMultipleLooksControl:
    """ONE question: has the LABELLED CORPUS materially changed?

    Five earlier designs each keyed off a PROXY for "new evidence" —
    deployed corpus size, a representation flag, a config fingerprint,
    bench mass, deployability — and every proxy had a channel that moved
    without the evidence moving. Measured in the live configuration they
    were indistinguishable from the kill switch: growing bench gave
    101/201 looks, a flapping embedder 10/10, an orphaned ledger 201/201.

    The gate evaluates a split of the labelled corpus, so the corpus is
    what is compared. Bench rows are train-only and cannot reach it; an
    embedder flap cannot reach it; a deleted checkpoint cannot reach it.
    Compared on the CORPUS rather than the held-out subset because the
    split is a positional shuffle — n and n+1 barely overlap, which made
    every added turn read as new evidence (8/10 looks, benefit gone)."""

    @staticmethod
    def _ck():
        import tempfile, pathlib as _pl
        return _pl.Path(tempfile.mkdtemp()) / "checkpoint.json"

    @staticmethod
    def _looked(r):
        return "labelled corpus" not in (r.bail_reason or "")

    def test_a_near_identical_corpus_is_not_re_tested(self):
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        assert self._looked(RouterTrainer().run(trajectories=_balanced(each),
                                                save_path=ck))
        r = RouterTrainer().run(trajectories=_balanced(each + 2), save_path=ck)
        assert not self._looked(r), r.bail_reason

    def test_a_materially_larger_corpus_IS_re_tested(self):
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        RouterTrainer().run(trajectories=_balanced(each), save_path=ck)
        assert self._looked(RouterTrainer().run(
            trajectories=_balanced(each * 2), save_path=ck))

    def test_BENCH_MASS_cannot_re_open_a_look(self):
        """Bench joins the TRAIN side only, so it changes nothing the gate
        scores — yet it was in the fingerprint, and at the live bench
        growth rate that alone restored 101 of 201 looks."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        real = _balanced(each)
        RouterTrainer().run(trajectories=real, save_path=ck)
        looks = sum(1 for i in range(1, 6) if self._looked(
            RouterTrainer().run(trajectories=real, save_path=ck,
                                bench_trajectories=_balanced(i * 10, tag="b"))))
        assert looks == 0, f"{looks} looks bought with bench rows alone"

    def test_an_ALTERNATING_embedder_cannot_buy_unlimited_looks(self):
        """Two distinct configs exist (lexical, embedding), so at most two
        looks — not one per flip. Previously 10/10 on a growing corpus."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        looks = 0
        for i in range(8):
            if i % 2 == 0:
                set_router_embedder(_fake_embedder())
            else:
                reset_router_embedder()
            if self._looked(RouterTrainer().run(
                    trajectories=_balanced(each + i), save_path=ck)):
                looks += 1
        assert looks <= 2, f"{looks} looks on a corpus that barely moved"

    def test_RELABELLING_the_corpus_re_opens_a_look(self):
        """The gate scores (X_test, y_test). Keyed on trajectory id alone
        the control was blind to y: flipping 877 of 1,707 outcomes left
        the id set identical and the look blocked. Live mechanism —
        `corrections.jsonl` overlays 350 rows and `derive_label` branches
        on `traj.outcome`."""
        from ghost_agent.router import RouterTrainer
        import copy
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        corpus = _balanced(each)
        RouterTrainer().run(trajectories=corpus, save_path=ck)
        assert not self._looked(RouterTrainer().run(trajectories=corpus,
                                                    save_path=ck))
        flipped = [copy.copy(t) for t in corpus]
        for t in flipped[:len(flipped) // 2]:
            t.outcome = "failed" if t.outcome != "failed" else "passed"
        assert self._looked(RouterTrainer().run(trajectories=flipped,
                                                save_path=ck))

    def test_a_broken_checkpoint_re_opens_exactly_one_look(self):
        """Recovery was promised by a docstring and delivered by nothing —
        the rebuild dropped it and left the function dead. Restored via
        the fingerprint so it is BOUNDED: one look, not one per flip."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        RouterTrainer().run(trajectories=_balanced(each), save_path=ck)
        assert not self._looked(RouterTrainer().run(
            trajectories=_balanced(each), save_path=ck))
        ck.write_text("{ corrupt")
        assert self._looked(RouterTrainer().run(
            trajectories=_balanced(each), save_path=ck))
        assert ck.exists() and "corrupt" not in ck.read_text()

    def test_the_wait_is_bounded_in_ABSOLUTE_rows(self):
        """The SLACK bar must be the one that decides here.

        The previous version compared 242 rows against 742 — overlap
        0.326, which the RATIO bar alone already re-opens — so deleting
        the absolute-slack term left all 25 tests in this class green
        while the constant it guards is the bar that actually decides at
        the live corpus size. This corpus is chosen so the ratio says
        "same" and only the slack says "moved": a pure ratio compounds
        (+263, +309, +364, +428 ...) and the bound is what keeps the wait
        linear."""
        from ghost_agent.router.trainer import _evidence_unchanged
        slack = ComplexityClassifier._GATE_LOOK_ABSOLUTE_SLACK
        ratio = ComplexityClassifier._GATE_MAX_HELDOUT_OVERLAP
        # Big enough that adding `slack` rows keeps overlap ABOVE the
        # ratio bar — so only the slack term can re-open the look.
        base = {f"row-{i}" for i in range(int(slack / (1 - ratio)) + 500)}
        grown = base | {f"new-{i}" for i in range(slack + 5)}
        assert _jaccard(grown, base) >= ratio, "premise: ratio says 'same'"
        assert _evidence_unchanged(grown, base) is False, (
            "the absolute-slack bound is not being applied")
        # One row under the bound is still "same".
        nearly = base | {f"new-{i}" for i in range(slack - 5)}
        assert _evidence_unchanged(nearly, base) is True

    def test_a_SUBSET_of_the_corpus_is_never_new_evidence(self):
        """`bootstrap_router` caps its read at 20,000 oldest-first while
        the other two sites read uncapped, so above that the windows
        alternate — measured 12 looks in 12 at 70-80% overlap."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        RouterTrainer().run(trajectories=_balanced(each * 2), save_path=ck)
        looks = sum(1 for _ in range(4) if self._looked(
            RouterTrainer().run(trajectories=_balanced(each), save_path=ck)))
        assert looks == 0, f"{looks} looks bought by reading a smaller window"

    def test_the_decision_and_the_record_share_ONE_definition(self):
        """`run` and `_record_look` had two different expressions for
        "same evidence" and disagreed in both directions — 8 looks in 8
        runs one way, a healthy retrain blocked on 257 new rows the
        other. Pinned structurally: both must call the same function."""
        from pathlib import Path
        import ghost_agent.router.trainer as tr
        src = Path(tr.__file__).read_text()
        assert src.count("_evidence_unchanged(") >= 3      # def + 2 callers
        # ...and neither side may re-derive it inline.
        body = src[src.index("def _record_look("):src.index("def _last_look(")]
        assert "_GATE_MAX_HELDOUT_OVERLAP" not in body
        run_body = src[src.index("_same_evidence ="):src.index("_blocked = (")]
        assert "_GATE_MAX_HELDOUT_OVERLAP" not in run_body

    def test_the_bail_tells_the_truth_when_nothing_is_deployed(self):
        """With no checkpoint the bail claimed "the deployed model stays"
        at INFO — below the live stream's WARNING floor — and withheld the
        recovery hint in the one state that needs it. Reachable: the first
        look's model can be rejected by the gate, spending a look without
        writing a checkpoint."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        # A corpus that cannot deploy: the label is assigned by a coin
        # flip over the SAME text distribution, so no representation can
        # separate it. (An earlier version used different text per class
        # and duly deployed — the premise assert caught it.)
        import random as _r
        rng = _r.Random(11)
        flat = []
        for i in range(2 * each):
            hard = rng.random() < 0.5
            flat.append(_traj(f"task {i % 40} variant {i}",
                              6 if hard else 1, 5 if hard else 1,
                              outcome="failed" if hard else "passed",
                              heavy=hard))
        RouterTrainer().run(trajectories=flat, save_path=ck)
        assert not ck.exists(), "premise: nothing deployed"
        r = RouterTrainer().run(trajectories=flat, save_path=ck)
        assert "NOTHING IS DEPLOYED" in (r.bail_reason or ""), r.bail_reason
        assert "GHOST_ROUTER_GATE_LOOKS" in (r.bail_reason or "")

    def test_the_kill_switch_releases_it(self, monkeypatch):
        """A guard that can delay a deploy needs an off switch — and it is
        the documented affordance when a stale ledger blocks recovery."""
        from ghost_agent.router import RouterTrainer
        ck, each = self._ck(), _corpus_floor_each()
        reset_router_embedder()
        RouterTrainer().run(trajectories=_balanced(each), save_path=ck)
        assert not self._looked(RouterTrainer().run(
            trajectories=_balanced(each + 2), save_path=ck))
        monkeypatch.setenv("GHOST_ROUTER_GATE_LOOKS", "0")
        assert self._looked(RouterTrainer().run(
            trajectories=_balanced(each + 2), save_path=ck))

    def test_an_unwritable_ledger_is_LOUD(self, caplog):
        """A silent failure disables the control entirely. This warning
        caught a real NameError in the ledger writer within minutes of it
        being introduced."""
        import logging, tempfile, pathlib as _pl
        from ghost_agent.router.trainer import _record_look
        d = _pl.Path(tempfile.mkdtemp())
        (d / "x.gate_looks.json").mkdir()
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            _record_look(d / "x.gate_looks.json", frozenset({"a"}), "fp")
        assert any("UNWRITABLE" in r.getMessage() for r in caplog.records)

    def test_the_ledger_is_per_checkpoint(self):
        from ghost_agent.router.trainer import _looks_path
        import pathlib as _pl
        assert (_looks_path(_pl.Path("/tmp/r/checkpoint.json"))
                != _looks_path(_pl.Path("/tmp/r/experiment.json")))


class TestProcessStateHermeticity:
    """A tripwire for the autouse `clear_router_embedder` fixture.

    The fixture is genuinely load-bearing — with its body no-op'd, a stub
    left registered here makes later router suites fail their held-out
    gate. But nothing pinned that, because the file's last test happened
    to reset the registry itself."""

    # ORDER MATTERS: pytest runs these in definition order, so the second
    # only passes because the fixture cleaned up after the first. Written
    # the other way round it would pin nothing.
    def test_a_leaves_an_embedder_registered(self):
        from ghost_agent.router import get_router_embedder
        set_router_embedder(_fake_embedder())
        assert get_router_embedder() is not None

    def test_b_registry_was_reset_between_tests(self):
        from ghost_agent.router import get_router_embedder
        assert get_router_embedder() is None
