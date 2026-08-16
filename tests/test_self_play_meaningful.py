"""Tests for the 2026-05-17 self-play redesign (proposals A-H).

Covers the core invariants of:
  A. multi-signal score (novelty + attempts efficiency on top of the
     legacy compression_delta + tool_errors)
  B. qualitative tier twists (orthogonal axes, not just N× rows)
  C. write gate opens on first-try wins with high structural novelty
  D. journal mining materialises shape-appropriate fixtures
  E. PRM scheduler helper is safe to call with missing collector / scorer
  F. reflector reflects FAILED trajectories only (the proposal-F
     low-novelty-pass opt-in was removed 2026-07-20 as dead code)
  G. adversarial generator tracker pass-rate + bias suggestion
  H. per-template saturation independent of cluster-level saturation

These tests are pure-Python — no sandbox / LLM / docker — and run in
well under a second each. They're the contract layer that should keep
the new code honest as the surrounding system evolves.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
import types
from unittest.mock import MagicMock

from ghost_agent.core.self_play_scoring import correctness_weighted_score
from ghost_agent.core.solution_novelty import (
    canonical_hash,
    jaccard_novelty,
    attempts_efficiency,
)
from ghost_agent.core.challenge_templates import (
    _TWIST_AXES,
    _TIER_TWIST_COUNT,
    _resolve_twists_for_tier,
    try_template,
)
from ghost_agent.core.journal_challenges import (
    _detect_data_shape,
    _shape_specific_setup,
    _shape_specific_validator,
    mine_challenges,
)
from ghost_agent.core.adversarial_generator import (
    AdversarialGeneratorTracker,
    fingerprint_prompt,
)
from ghost_agent.memory.frontier import FrontierTracker


# ---------------------------------------------------------------------------
# A. Multi-signal score
# ---------------------------------------------------------------------------


class TestMultiSignalScore:
    def test_backcompat_no_novelty_no_attempts_matches_legacy(self):
        """When neither novelty nor attempts_used is supplied, the score
        must be identical to the pre-2026-05 formula
        ``passed*(1 + α*Δ) − β*errors``. This is what every legacy
        test asserts."""
        score = correctness_weighted_score(
            passed=True,
            compression_delta=0.0,
            tool_errors=0,
        )
        assert score == 1.0

    def test_novelty_adds_to_a_passing_score(self):
        """A first-try win with high novelty (different AST from
        prior wins) must outrank an identical run with low novelty."""
        novel = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=1.0, attempts_used=1,
        )
        boring = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=0.0, attempts_used=1,
        )
        assert novel > boring

    def test_attempts_efficiency_punishes_retries(self):
        """A first-try win must outrank a third-try win with the same
        compression and novelty."""
        first = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=0.5, attempts_used=1,
        )
        third = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=0.5, attempts_used=3,
        )
        assert first > third

    def test_failure_zeros_base_signals(self):
        """A failed run must produce a non-positive score regardless
        of novelty / attempts — those signals only count when the
        solver actually solved the task."""
        score = correctness_weighted_score(
            passed=False, compression_delta=1.0, tool_errors=0,
            novelty=1.0, attempts_used=1,
        )
        assert score <= 0.0

    def test_tool_errors_penalty_still_applies(self):
        clean = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=0.5, attempts_used=1,
        )
        dirty = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=3,
            novelty=0.5, attempts_used=1,
        )
        assert dirty < clean


class TestSolutionNovelty:
    def test_identical_source_is_zero_novelty(self):
        src = "for i in range(10):\n    print(i)\n"
        assert jaccard_novelty(src, [src]) == 0.0

    def test_no_prior_wins_is_full_novelty(self):
        src = "print('hi')\n"
        assert jaccard_novelty(src, []) == 1.0

    def test_variable_rename_is_still_zero(self):
        """Two solutions that differ ONLY in variable names canonicalise
        to the same AST shape — no novelty awarded for cosmetic
        differences."""
        a = "x = 1\ny = x + 2\nprint(y)\n"
        b = "foo = 1\nbar = foo + 2\nprint(bar)\n"
        assert jaccard_novelty(a, [b]) == 0.0

    def test_structurally_different_solution_is_novel(self):
        loopy = "total = 0\nfor i in range(10):\n    total += i\nprint(total)\n"
        comprehensive = "print(sum(i for i in range(10)))\n"
        # Different shapes → novelty should be clearly positive.
        nov = jaccard_novelty(comprehensive, [loopy])
        assert nov > 0.3

    def test_unparseable_source_gets_zero_credit(self):
        """A `solution.py` that doesn't even parse must NOT be awarded
        novelty — even if it's textually unique it's not a learning
        signal."""
        assert jaccard_novelty("def broken(:::", ["print('hi')\n"]) == 0.0

    def test_canonical_hash_stable_across_renames(self):
        a = canonical_hash("def f(x): return x + 1\n")
        b = canonical_hash("def g(y): return y + 1\n")
        assert a == b and a != ""

    def test_attempts_efficiency_table(self):
        assert attempts_efficiency(1) == 1.0
        assert attempts_efficiency(2) == 0.5
        assert attempts_efficiency(3) == 0.2
        # Clamped on both ends.
        assert attempts_efficiency(0) == 1.0
        assert attempts_efficiency(99) == 0.2


# ---------------------------------------------------------------------------
# B. Qualitative tier twists
# ---------------------------------------------------------------------------


class TestTierTwists:
    def test_basic_tier_picks_zero_twists(self):
        twists = _resolve_twists_for_tier("data_analysis", "basic", seed=42)
        assert twists == set()

    def test_intermediate_picks_one_twist(self):
        twists = _resolve_twists_for_tier("data_analysis", "intermediate", seed=42)
        assert len(twists) == 1

    def test_expert_picks_three_twists(self):
        twists = _resolve_twists_for_tier("data_analysis", "expert", seed=42)
        assert len(twists) == 3

    def test_deterministic_for_seed(self):
        """Same (cluster, tier, seed) → same twist set, every time.
        The setup script and validator share a seed; if the twist set
        weren't deterministic they could disagree."""
        a = _resolve_twists_for_tier("data_analysis", "advanced", seed=7)
        b = _resolve_twists_for_tier("data_analysis", "advanced", seed=7)
        assert a == b

    def test_unknown_cluster_has_no_twists(self):
        twists = _resolve_twists_for_tier("not_a_cluster", "expert", seed=1)
        assert twists == set()

    def test_data_analysis_advanced_template_renders(self):
        """End-to-end: the actual template function must produce a
        valid triple when asked for an advanced-tier challenge with
        twists baked in."""
        triple = try_template("data_analysis", tier="advanced")
        assert triple is not None
        prompt, setup, validator = triple
        assert "data.csv" in prompt
        assert "csv" in setup
        assert "subprocess" in validator


# ---------------------------------------------------------------------------
# D. Journal mining materialises shape-appropriate fixtures
# ---------------------------------------------------------------------------


class TestJournalMining:
    def test_detects_csv_shape(self):
        assert _detect_data_shape("parse the spreadsheet at data.csv")["kind"] == "csv"

    def test_detects_json_shape(self):
        assert _detect_data_shape("walk this JSON payload")["kind"] == "json"

    def test_detects_log_shape(self):
        assert _detect_data_shape("parse the nginx access log")["kind"] == "log"

    def test_detects_sql_shape(self):
        assert _detect_data_shape("write a SELECT against postgres")["kind"] == "sql"

    def test_falls_back_to_text(self):
        assert _detect_data_shape("do something interesting")["kind"] == "text"

    def test_setup_script_is_stdlib_only(self):
        """The self-play sandbox enforces stdlib-only setup scripts.
        Every shape we materialise must obey that."""
        for kind in ("csv", "json", "log", "sql", "text"):
            setup = _shape_specific_setup(kind)
            assert "faker" not in setup
            assert "pandas" not in setup

    def test_validator_runs_solution_py(self):
        for kind in ("csv", "json", "log", "sql", "text"):
            v = _shape_specific_validator(kind, f"input.{kind}")
            assert "python3" in v
            assert "solution.py" in v

    def test_mine_challenges_uses_shape_detection(self):
        entries = [{
            "type": "post_mortem",
            "data": {
                "user": "I need to parse our JSON metrics file and emit a histogram of latencies",
                "summary": "agent gave up after timeout",
            },
        }]
        mined = mine_challenges(entries, max_out=1)
        assert len(mined) == 1
        # The JSON shape detector should have steered the fixture.
        assert "input.json" in mined[0].challenge


# ---------------------------------------------------------------------------
# F. Reflector reflects FAILED trajectories only (proposal-F opt-in removed)
# ---------------------------------------------------------------------------


class TestReflectorFailedOnly:
    """The proposal-F ``accept_low_novelty_passes`` opt-in was removed
    2026-07-20 as dead-by-construction: nothing in the live tree writes
    ``extra["solution_novelty"]`` (it only flows to
    frontier_tracker.record_run) and the live dream loop's isolated
    context sets trajectory_collector=None, so no self_play trajectories
    reach the collector. A pass is never reflectable."""

    def _make_traj(self, *, outcome: str, novelty=None, kind: str = "self_play"):
        # Build a minimal Trajectory-like duck for _is_reflectable.
        t = MagicMock()
        t.outcome = outcome
        t.task_kind = kind
        t.extra = {"solution_novelty": novelty} if novelty is not None else {}
        return t

    def test_passed_never_reflected_even_low_novelty_selfplay(self):
        from ghost_agent.reflection.loop import Reflector
        from ghost_agent.distill.schema import Outcome
        r = Reflector(critique_fn=lambda x: "diag")
        boring = self._make_traj(outcome=Outcome.PASSED.value, novelty=0.05)
        assert r._is_reflectable(boring) is False

    def test_failed_always_reflected(self):
        from ghost_agent.reflection.loop import Reflector
        from ghost_agent.distill.schema import Outcome
        r = Reflector(critique_fn=lambda x: "diag")
        failed = self._make_traj(outcome=Outcome.FAILED.value)
        assert r._is_reflectable(failed) is True

    def test_removed_kwarg_rejected(self):
        # Locks the clean removal in place — the main.py wiring that
        # passed this kwarg is being removed alongside it.
        from ghost_agent.reflection.loop import Reflector
        with pytest.raises(TypeError):
            Reflector(critique_fn=lambda x: "diag", accept_low_novelty_passes=True)


# ---------------------------------------------------------------------------
# G. Adversarial generator tracker
# ---------------------------------------------------------------------------


class TestAdversarialTracker:
    def test_fingerprint_stable(self):
        a = fingerprint_prompt("Generate a JSON challenge")
        b = fingerprint_prompt("Generate a JSON challenge")
        assert a == b
        assert len(a) == 12

    def test_pass_rate_returns_none_below_two_samples(self, tmp_path):
        t = AdversarialGeneratorTracker(tmp_path)
        t.record("fp_a", passed=True, cluster="sql")
        assert t.pass_rate("fp_a") is None  # below minimum

    def test_pass_rate_computed(self, tmp_path):
        t = AdversarialGeneratorTracker(tmp_path)
        for _ in range(3):
            t.record("fp_b", passed=True, cluster="sql")
        for _ in range(1):
            t.record("fp_b", passed=False, cluster="sql")
        assert t.pass_rate("fp_b") == 0.75

    def test_worst_fingerprints_orders_by_low_pass_rate(self, tmp_path):
        t = AdversarialGeneratorTracker(tmp_path)
        # Easy: 4 wins, 0 fails
        for _ in range(4):
            t.record("easy", passed=True, cluster="csv")
        # Hard: 1 win, 3 fails
        t.record("hard", passed=True, cluster="concurrency")
        for _ in range(3):
            t.record("hard", passed=False, cluster="concurrency")
        worst = t.worst_fingerprints(limit=2)
        assert worst[0][0] == "hard"
        assert worst[0][1] == 0.25

    def test_suggest_bias_quotes_struggling_cluster(self, tmp_path):
        t = AdversarialGeneratorTracker(tmp_path)
        for _ in range(2):
            t.record("hard", passed=False, cluster="concurrency")
        t.record("hard", passed=True, cluster="concurrency")
        bias = t.suggest_bias()
        assert "concurrency" in bias
        assert "%" in bias  # the rendered pass rate

    def test_suggest_bias_empty_when_no_signal(self, tmp_path):
        t = AdversarialGeneratorTracker(tmp_path)
        assert t.suggest_bias() == ""


# ---------------------------------------------------------------------------
# C. Write gate semantics (unit-test the gate decisions, not the LLM)
# ---------------------------------------------------------------------------


class TestWriteGateOpensOnNovelty:
    """The gate is implemented inline in dream.synthetic_self_play; we
    test the *boolean inputs* that the gate consumes so any regression
    in those inputs surfaces here, even before reaching dream.py."""

    def test_high_novelty_first_try_pass_is_a_learning_signal(self):
        """The new gate path (proposal C): a first-try pass with
        novelty ≥ 0.5 must produce a positive score even when the
        compression delta is flat."""
        score = correctness_weighted_score(
            passed=True, compression_delta=0.0, tool_errors=0,
            novelty=0.8, attempts_used=1,
        )
        # The boring baseline (novelty=0) sits at:
        # 1.0 + 0.0 + 0.6*0 + 0.3*1.0 = 1.3
        # High novelty bumps it to:
        # 1.0 + 0.0 + 0.6*0.8 + 0.3*1.0 = 1.78
        assert score > 1.5


# ---------------------------------------------------------------------------
# H. Per-template saturation
# ---------------------------------------------------------------------------


class TestPerTemplateSaturation:
    def test_template_saturates_after_repeated_zero_novelty_wins(self, tmp_path):
        tracker = FrontierTracker(tmp_path)
        # Two consecutive first-try wins with novelty=0 against the
        # same template_key — that's exactly the "boring" loop the
        # saturation guard is supposed to catch.
        for i in range(2):
            tracker.record_run(
                cluster_key="sql",
                challenge=f"unique challenge prompt {i}",
                attempts_used=1,
                passed=True,
                description_length=4,
                template_key="sql_group_by",
                novelty=0.0,
                solution_source="print('ok')\n",
            )
        saturated = tracker.list_saturated_templates()
        assert ("sql", "sql_group_by") in saturated

    def test_template_not_saturated_after_one_novel_win(self, tmp_path):
        tracker = FrontierTracker(tmp_path)
        # First win: novel
        tracker.record_run(
            cluster_key="sql",
            challenge="unique prompt A",
            attempts_used=1,
            passed=True,
            description_length=4,
            template_key="sql_group_by",
            novelty=0.9,
            solution_source="print('a')\n",
        )
        # Second win: boring
        tracker.record_run(
            cluster_key="sql",
            challenge="unique prompt B",
            attempts_used=1,
            passed=True,
            description_length=4,
            template_key="sql_group_by",
            novelty=0.0,
            solution_source="print('b')\n",
        )
        assert ("sql", "sql_group_by") not in tracker.list_saturated_templates()

    def test_recent_winning_solutions_returns_sources(self, tmp_path):
        tracker = FrontierTracker(tmp_path)
        tracker.record_run(
            cluster_key="bash",
            challenge="unique bash prompt",
            attempts_used=1,
            passed=True,
            description_length=4,
            solution_source="print('once')\n",
        )
        prior = tracker.recent_winning_solutions("bash")
        assert any("once" in s for s in prior)

    def test_winning_solutions_deduped_by_canonical_hash(self, tmp_path):
        """Two solutions with the same AST shape (just variable
        renames) must share one slot in the ring buffer."""
        tracker = FrontierTracker(tmp_path)
        a = "x = 1\nprint(x)\n"
        b = "y = 1\nprint(y)\n"
        tracker.record_run(
            cluster_key="python_general", challenge="p1",
            attempts_used=1, passed=True, description_length=3,
            solution_source=a,
        )
        tracker.record_run(
            cluster_key="python_general", challenge="p2",
            attempts_used=1, passed=True, description_length=3,
            solution_source=b,
        )
        prior = tracker.recent_winning_solutions("python_general")
        # Only one canonical shape recorded.
        assert len(prior) == 1


# ---------------------------------------------------------------------------
# E. PRM scheduler helper is safe with missing wiring
# ---------------------------------------------------------------------------


def _twin_ctx_that_would_train(tmp_path, monkeypatch, **argkw):
    """A ctx for `_maybe_retrain_prm` that passes EVERY guard downstream
    of the consumer gate, so the gate is the only thing that can stop it.

    R7 CRIT-1: the previous probe asserted `PRMTrainer.__init__` never
    ran on a bare `MagicMock` ctx — but `memory.py`'s own
    `isinstance(traj_collector, TrajectoryCollector)` check already
    guarantees that outcome, whichever way the gate is written. The probe
    measured a property a DIFFERENT guard supplied, so widening the gate
    with the producer flag left 77 tests green: the §4BN retraction was
    unpinned on the twin. (The phase-2.7 pin uses real objects and does
    catch it — this is the twin-path miss again.)

    Real collector, real scorer: now only the consumer gate stands
    between this ctx and a fitted model.
    """
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.prm.scorer import PRMScorer
    from ghost_agent.prm import trainer as _tr

    work = []
    _orig = _tr.PRMTrainer.__init__

    def _spy(self, *a, **k):
        work.append("trainer")
        return _orig(self, *a, **k)

    monkeypatch.setattr(_tr.PRMTrainer, "__init__", _spy)

    ctx = types.SimpleNamespace(
        trajectory_collector=TrajectoryCollector(str(tmp_path)),
        prm_scorer=PRMScorer(),
        mcts_reasoner=None,
        memory_dir=tmp_path,
        args=types.SimpleNamespace(prm_train_cooldown=None, model="m",
                                   **argkw),
    )
    return ctx, work


class TestPRMSchedulerHelper:
    """R1 MIN-6: both cases below used a bare MagicMock ctx, so
    `getattr(ctx.args, 'frontier_selfplay', False) is True` was False on
    the mock and the helper returned at the CONSUMER GATE — neither test
    ever reached the collector/scorer branch it is named for. Silent test
    decay of the twin predicate. They now pass the gate first, and a
    third case pins the gate itself."""

    @staticmethod
    def _ctx_past_the_consumer_gate():
        ctx = MagicMock()
        ctx.args.frontier_selfplay = True   # a real value-reading consumer
        return ctx

    def test_no_op_when_collector_missing(self):
        from ghost_agent.tools.memory import _maybe_retrain_prm
        ctx = self._ctx_past_the_consumer_gate()
        ctx.trajectory_collector = None
        # Must not raise.
        _maybe_retrain_prm(ctx)

    def test_no_op_when_scorer_missing(self, tmp_path):
        """R2 MIN-1: this used an auto-MagicMock collector, so
        `isinstance(tc, TrajectoryCollector)` was False and the function
        returned at the COLLECTOR guard — byte-for-byte the same coverage
        as the test above, never reaching the scorer guard it names.
        Give it a real collector so the scorer guard is what stops it."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.tools.memory import _maybe_retrain_prm
        ctx = self._ctx_past_the_consumer_gate()
        ctx.trajectory_collector = TrajectoryCollector(str(tmp_path))
        ctx.prm_scorer = None
        _maybe_retrain_prm(ctx)

    def test_twin_bridges_a_fresh_model_into_mcts(self, tmp_path,
                                                  monkeypatch):
        """R7 MIN-6 (twin divergence): phase 2.7 bridges a freshly-fitted
        model into `mcts.prm_scorer` on the first-ever fit; this twin did
        not. On a `.score()`-live box that boots without a checkpoint,
        `mcts.prm_scorer` stays None and MCTS keeps failing its own
        `prm_scorer is not None and .has_model` guard after an in-loop
        refit — this path trained a model nothing reads, the §4BN class,
        on the twin."""
        from ghost_agent.tools import memory as _mem
        from ghost_agent.core import agent as _core_agent
        from ghost_agent.prm import trainer as _tr

        ctx, _ = _twin_ctx_that_would_train(
            tmp_path, monkeypatch, frontier_selfplay=True,
            prm_online_update=False)
        mcts = types.SimpleNamespace(prm_scorer=None)
        ctx.mcts_reasoner = mcts
        monkeypatch.setattr(_core_agent, "_MCTS_TURNSTART_ENABLED", False)

        class _Rep:
            fit_succeeded = True
            def summary(self): return "ok"

        def _fake_run(self, *a, **k):
            self.model = object()
            return _Rep()
        monkeypatch.setattr(_tr.PRMTrainer, "run", _fake_run)
        monkeypatch.setattr(type(ctx.prm_scorer), "set_model",
                            lambda self, m: None)
        _mem._maybe_retrain_prm(ctx)
        assert mcts.prm_scorer is not None, (
            "the in-loop retrain fitted a model and never bridged it into "
            "MCTS — after this refit MCTS still fails its own prm_scorer "
            "guard, i.e. trained a model nothing reads")

    def test_twin_skip_logs_and_names_the_missing_conjunct(self, monkeypatch):
        """R5 MAJOR-1, twin site: replacing the derived cause with the old
        hardcoded literal left 116 tests green. Capture the real log.

        Captures `logger.debug` directly rather than via `caplog`: the
        first version used caplog and PASSED in isolation but FAILED in
        the full suite with zero records — some earlier test mutates
        global logging state (handler/propagate/disable), so the capture
        never fired. An order-dependent pin is not a pin; this one does
        not depend on logging configuration at all."""
        from ghost_agent.tools import memory as _mem
        from ghost_agent.core import agent as _core_agent
        ctx = MagicMock()
        ctx.args.frontier_selfplay = False
        ctx.args.deep_reason = False   # argparse always supplies a bool
        ctx.mcts_reasoner = None
        monkeypatch.setattr(_core_agent, "_MCTS_TURNSTART_ENABLED", True)
        captured = []
        monkeypatch.setattr(_mem.logger, "debug",
                            lambda msg, *a, **k: captured.append(str(msg)))
        _mem._maybe_retrain_prm(ctx)
        body = "\n".join(captured)
        assert "PRM retrain skipped" in body, "twin skipped silently"
        assert "--deep-reason is not set" in body, \
            f"twin skip log does not name the missing conjunct: {body!r}"
        assert "module-gated off" not in body, \
            "twin blames a module gate that is ON in this run"
        # R18 MAJOR-2 (sixteenth twin-path miss): phase 2.7's skip log has
        # `assert "PRODUCER" in body` and this twin did not, so deleting
        # the producer explanation here left 1,767 tests green. That
        # sentence IS the §4BN retraction, stated where a future reader
        # of this gate will see it.
        assert "PRODUCER" in body, (
            "the twin skip log no longer records WHY --prm-online-update "
            "is excluded — that sentence is the retraction")

    def test_module_gate_without_a_reasoner_still_short_circuits(
            self, monkeypatch, tmp_path):
        """R4 MAJOR-3: the twin gate's use of the shared predicate was
        unpinned — the only test driving it set the module constant False,
        so it could not see the conjunct at all. Re-inlining R3's
        one-conjunct bug here left 83 tests green while this path went
        back to training a model nothing can read."""
        from ghost_agent.tools import memory as _mem
        from ghost_agent.core import agent as _core_agent
        ctx = MagicMock()
        ctx.args.frontier_selfplay = False
        ctx.args.deep_reason = False   # argparse always supplies a bool
        ctx.mcts_reasoner = None            # no --deep-reason
        monkeypatch.setattr(_core_agent, "_MCTS_TURNSTART_ENABLED", True)
        _real_ctx, trained = _twin_ctx_that_would_train(
            tmp_path, monkeypatch, frontier_selfplay=False,
            prm_online_update=False)
        _real_ctx.mcts_reasoner = None       # module gate ON, no reasoner
        _mem._maybe_retrain_prm(_real_ctx)
        assert not trained, (
            "the twin retrain gate walked past a module constant that is "
            "NECESSARY but not SUFFICIENT — nothing can call .score() "
            "without a live mcts_reasoner (--deep-reason)")

    def test_consumer_gate_short_circuits_before_any_work(self, monkeypatch,
                                                          tmp_path):
        """With no consumer live the helper must return BEFORE touching
        the collector — that skip is the 41-wasted-retrains fix, and
        --prm-online-update must not satisfy it (§4BN)."""
        from ghost_agent.tools import memory as _mem
        from ghost_agent.core import agent as _core_agent
        ctx = MagicMock()
        ctx.args.frontier_selfplay = False
        ctx.args.deep_reason = False   # argparse always supplies a bool
        ctx.args.prm_online_update = True      # the PRODUCER — must not count
        # `_maybe_retrain_prm` does a call-time `from ..core.agent import
        # _MCTS_TURNSTART_ENABLED`, so the SOURCE module is the patch
        # target — patching the importer would be a no-op that passes.
        monkeypatch.setattr(_core_agent, "_MCTS_TURNSTART_ENABLED", False)
        # R6 MAJOR-5: this used to install a property probe on
        # `trajectory_collector` and assert it was never READ — but R5
        # made the collector a legitimate INPUT to the gate, so the probe
        # stayed quiet only by `and`-short-circuit order. Hoisting the
        # read into a local (an honest readability refactor) false-failed
        # it. Assert the real property instead: no training work happened.
        _real_ctx, trained = _twin_ctx_that_would_train(
            tmp_path, monkeypatch, frontier_selfplay=False,
            prm_online_update=True)      # the PRODUCER is ON
        _mem._maybe_retrain_prm(_real_ctx)
        assert not trained, (
            "the retrain helper walked past the consumer gate with no "
            "consumer live — --prm-online-update is a producer and must "
            "not re-arm training (§4BN)")
