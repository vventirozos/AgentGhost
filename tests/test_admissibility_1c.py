"""§4BF Track 1c — per-consumer admissibility of origin=bench outcomes.

Pins the operator-decided matrix (core/admissibility.py) and every seam
that enforces it: experiment scopes, the calibration origin + oracle
re-label + equal-mass cap, the PRM/router origin feature with real-only
floors/gates, the GEPA trainset origin field, the extracted lesson gate,
and the learning-health denominator surfaces.
"""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core import admissibility as adm
from ghost_agent.core import experiments as ex
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory


# ──────────────────────────────────────────────────────────────────────
# The matrix itself
# ──────────────────────────────────────────────────────────────────────

class TestMatrix:
    def test_operator_decided_rows(self):
        # These four take bench WITH origin as a feature (2026-08-13).
        for consumer in ("gepa_trainset", "gepa_tool_fixtures", "prm",
                         "router", "calibration"):
            assert adm.policy_for(consumer) == adm.POLICY_BENCH_FEATURE
        assert adm.policy_for("lessons") == adm.POLICY_BENCH_TAGGED
        assert adm.policy_for("experiments_live") == adm.POLICY_REAL_ONLY
        assert adm.policy_for("experiments_bench") == adm.POLICY_BENCH_SCOPED

    def test_default_deny_rows_are_explicit(self):
        for consumer in ("reflection", "postmortem", "skills_auto",
                         "macro_mining", "foresight", "frontier",
                         "rem_fragments", "framing_leak", "human_feedback",
                         "prm_online_holdout"):
            assert adm.policy_for(consumer) == adm.POLICY_REAL_ONLY, consumer

    def test_unknown_consumer_is_denied(self):
        assert adm.policy_for("some_future_reader") == adm.POLICY_REAL_ONLY
        assert not adm.admits_bench("some_future_reader")
        assert adm.admitted_task_kinds("nope") == ("user_request",)

    def test_admitted_kinds_shapes(self):
        assert adm.admitted_task_kinds("prm") == ("user_request", "bench")
        assert adm.admitted_task_kinds("experiments_bench") == ("bench",)
        assert adm.admitted_task_kinds("reflection") == ("user_request",)


class TestIterBenchTrajectories:
    def _seed_bench_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.eval.banks import trajectories_root
        root = trajectories_root()
        col = TrajectoryCollector(root=root, session_id="bench-mbpp")
        col.append(Trajectory(user_request="bench one",
                              outcome=Outcome.PASSED.value,
                              task_kind="bench"))
        # A foreign record in the bench root must be filtered on read.
        col.append(Trajectory(user_request="stray real",
                              outcome=Outcome.PASSED.value,
                              task_kind="user_request"))
        return root

    def test_admitted_consumer_reads_bench_kind_only(self, tmp_path, monkeypatch):
        self._seed_bench_root(tmp_path, monkeypatch)
        got = list(adm.iter_bench_trajectories("prm"))
        assert [t.user_request for t in got] == ["bench one"]
        assert all(t.task_kind == "bench" for t in got)

    def test_real_only_consumer_gets_nothing_even_with_data(self, tmp_path, monkeypatch):
        self._seed_bench_root(tmp_path, monkeypatch)
        assert list(adm.iter_bench_trajectories("reflection")) == []
        assert list(adm.iter_bench_trajectories("unregistered")) == []

    def test_missing_root_is_a_quiet_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        assert adm.bench_trajectory_collector() is None
        assert list(adm.iter_bench_trajectories("prm")) == []
        assert adm.bench_corpus_fingerprint() is None

    def test_no_bench_boot_stops_consumption_too(self, tmp_path, monkeypatch):
        # R4 review MAJOR: --no-bench stopped only GENERATION — the
        # full_no_bench ablation arm's idle retrains kept training on
        # accumulated bench rows, so the arm measured ~nothing while
        # ABLATION.md claimed it isolated bench.
        self._seed_bench_root(tmp_path, monkeypatch)
        no_bench_args = types.SimpleNamespace(no_bench=True)
        assert list(adm.iter_bench_trajectories("prm", no_bench_args)) == []
        assert adm.bench_corpus_fingerprint(no_bench_args) is None
        # Enabled / MagicMock-args / None-args all read as enabled.
        on_args = types.SimpleNamespace(no_bench=False)
        assert list(adm.iter_bench_trajectories("prm", on_args))
        assert adm.bench_corpus_fingerprint(on_args)
        assert adm.bench_consumption_enabled(None) is True
        assert adm.bench_consumption_enabled(MagicMock()) is True

    def test_fingerprint_changes_with_new_bench_data(self, tmp_path, monkeypatch):
        root = self._seed_bench_root(tmp_path, monkeypatch)
        fp1 = adm.bench_corpus_fingerprint()
        assert fp1
        col = TrajectoryCollector(root=root, session_id="bench-gsm8k")
        col.append(Trajectory(user_request="more", outcome="failed",
                              task_kind="bench"))
        assert adm.bench_corpus_fingerprint() != fp1


# ──────────────────────────────────────────────────────────────────────
# Experiment scopes
# ──────────────────────────────────────────────────────────────────────

class TestExperimentScope:
    def test_default_specs_are_all_live(self):
        for s in ex.DEFAULT_SPECS:
            assert s.scope == ex.SCOPE_LIVE

    def test_scope_parses_and_invalid_scope_fails_closed(self):
        assert ex._spec_from_dict({"name": "tts_bon",
                                   "scope": "bench"}).scope == ex.SCOPE_BENCH
        assert ex._spec_from_dict({"name": "plain"}).scope == ex.SCOPE_LIVE
        # A typo'd scope must SKIP the spec, not run it on live traffic.
        assert ex._spec_from_dict({"name": "oops", "scope": "benhc"}) is None

    def test_assign_all_filters_by_scope(self):
        reg = ex.ExperimentRegistry([
            ex._spec_from_dict({"name": "live_one"}),
            ex._spec_from_dict({"name": "bench_one", "scope": "bench"}),
        ])
        live = reg.assign_all("unit-1")
        bench = reg.assign_all("unit-1", scope=ex.SCOPE_BENCH)
        assert set(live) == {"live_one"}
        assert set(bench) == {"bench_one"}
        assert reg.names_for_scope(ex.SCOPE_BENCH) == ("bench_one",)

    def _ctx_with_registry(self, tmp_path, specs):
        (tmp_path / "system").mkdir(parents=True, exist_ok=True)
        (tmp_path / "system" / "experiments.json").write_text(
            json.dumps({"salt": "t", "experiments": specs}), encoding="utf-8")
        ex.reset_registry_cache()
        return types.SimpleNamespace(memory_dir=tmp_path / "system" / "memory")

    def test_enroll_bench_origin_gets_bench_specs_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        ctx = self._ctx_with_registry(tmp_path, [
            {"name": "live_one"},
            {"name": "bench_one", "scope": "bench"},
        ])
        arms = ex.enroll_request(ctx, "req-b1", eligible=True, origin="bench")
        assert set(arms) == {"bench_one"}
        # And the stash serves behaviour reads for the bench turn.
        assert ex.arm_for(ctx, "bench_one", "req-b1") in ("control", "treatment")
        assert ex.arm_for(ctx, "live_one", "req-b1") == ""

    def test_enroll_user_origin_never_gets_bench_specs(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        ctx = self._ctx_with_registry(tmp_path, [
            {"name": "live_one"},
            {"name": "bench_one", "scope": "bench"},
        ])
        arms = ex.enroll_request(ctx, "req-u1", eligible=True, origin="user")
        assert set(arms) == {"live_one"}

    def test_enroll_sim_origin_gets_nothing_even_if_marked_eligible(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        ctx = self._ctx_with_registry(tmp_path, [
            {"name": "live_one"},
            {"name": "bench_one", "scope": "bench"},
        ])
        assert ex.enroll_request(ctx, "req-s1", eligible=True,
                                 origin="sim") == {}


def _traj(kind, stamp=None, outcome="passed"):
    extra = {"experiments": stamp} if stamp else {}
    return types.SimpleNamespace(task_kind=kind, extra=extra,
                                 outcome=outcome, n_steps=2, duration_s=1.0)


class TestSummarizeKinds:
    def test_bench_stamps_never_fold_into_the_default_report(self):
        trajs = [
            _traj("user_request", {"live_one": "treatment"}),
            _traj("bench", {"bench_one": "treatment"}),
        ]
        all_stats, _trig, cov = ex.summarize_streaming(trajs)
        assert "bench_one" not in all_stats
        assert "live_one" in all_stats
        assert cov["skipped_kind"] == 1
        assert cov["user_turns"] == 1

    def test_bench_report_admits_bench_only(self):
        trajs = [
            _traj("user_request", {"live_one": "control"}),
            _traj("bench", {"bench_one": "control"}),
        ]
        all_stats, _trig, cov = ex.summarize_streaming(
            trajs, admit_task_kinds=("bench",))
        assert set(all_stats) == {"bench_one"}
        assert cov["skipped_kind"] == 1

    def test_legacy_blank_kind_still_counts_as_a_real_turn(self):
        trajs = [_traj("", {"live_one": "treatment"})]
        all_stats, _trig, cov = ex.summarize_streaming(trajs)
        assert "live_one" in all_stats
        assert cov["user_turns"] == 1
        assert cov["skipped_kind"] == 0


class TestAnnouncementPopulation:
    def _fake_cmp(self):
        c = MagicMock()
        c.verdict = "TREATMENT BETTER"
        c.confound = ""
        c.metric = "failure_rate"
        c.control_mean, c.treatment_mean = 0.5, 0.2
        c.control_n, c.treatment_n = 40, 40
        return c

    def test_bench_verdict_line_says_gate_not_autoapply(self):
        with patch.object(ex, "compare_arms", return_value=[self._fake_cmp()]):
            out = ex.pending_announcements(
                {"tts_bon": {}}, {}, population=ex.SCOPE_BENCH)
        assert len(out) == 1
        key, line = out[0]
        assert key.startswith("bench|tts_bon|")
        assert "BENCH experiment 'tts_bon'" in line
        assert "gates a LIVE watch-window" in line
        assert "not an auto-apply" in line

    def test_live_keys_keep_their_pre_1c_shape(self):
        # An already-announced live verdict must never re-announce after
        # the upgrade — the key format is load-bearing.
        with patch.object(ex, "compare_arms", return_value=[self._fake_cmp()]):
            out = ex.pending_announcements({"risk_steer": {}}, {})
        key, line = out[0]
        assert key == "risk_steer|all|failure_rate|TREATMENT BETTER"
        assert not line.startswith("BENCH")

    def test_populations_dedupe_independently(self):
        with patch.object(ex, "compare_arms", return_value=[self._fake_cmp()]):
            live = ex.pending_announcements({"x_one": {}}, {})
            bench = ex.pending_announcements({"x_one": {}}, {},
                                             already=[live[0][0]],
                                             population=ex.SCOPE_BENCH)
        assert bench, "a live key must not silence the bench verdict"


# ──────────────────────────────────────────────────────────────────────
# Calibration: origin, oracle re-label, equal-mass cap
# ──────────────────────────────────────────────────────────────────────

class TestCalibrationOrigin:
    def _tracker(self, tmp_path):
        from ghost_agent.core.calibration import CalibrationTracker
        return CalibrationTracker(tmp_path / "calibration")

    def _record_turn(self, ct, req_id, origin, outcome=1.0):
        assert ct.record(composite=0.7, entropy_component=0.5,
                         competence_component=0.6, outcome=outcome,
                         source="turn", req_id=req_id, origin=origin)

    def test_origin_round_trips_and_legacy_defaults_to_user(self, tmp_path):
        ct = self._tracker(tmp_path)
        self._record_turn(ct, "rb", "bench")
        # Hand-write a legacy row with no origin field at all.
        with ct.history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"composite": 0.5, "entropy_component": 0.5,
                                 "competence_component": 0.5, "outcome": 1.0,
                                 "ts": "2026-08-13T00:00:00Z",
                                 "source": "turn", "req_id": "legacy"}) + "\n")
        samples = ct._load_samples()
        by_req = {s.req_id: s for s in samples}
        assert by_req["rb"].origin == "bench"
        assert by_req["legacy"].origin == "user"

    def test_bench_validator_relabels_disagreeing_bench_turn(self, tmp_path):
        ct = self._tracker(tmp_path)
        self._record_turn(ct, "rb", "bench", outcome=1.0)  # proxy said pass
        assert ct.record_bench_validator_verdict("rb", passed=False) is True
        rows = ct._load_samples()
        assert rows[-1].source == "bench_validator"
        assert rows[-1].outcome == 0.0
        assert rows[-1].origin == "bench"
        # Idempotent per request.
        assert ct.record_bench_validator_verdict("rb", passed=False) is False

    def test_bench_validator_skips_agreement_and_real_turns(self, tmp_path):
        ct = self._tracker(tmp_path)
        self._record_turn(ct, "agree", "bench", outcome=1.0)
        assert ct.record_bench_validator_verdict("agree", passed=True) is False
        self._record_turn(ct, "real", "user", outcome=1.0)
        # A real turn's sample must never be re-labeled by a bench oracle.
        assert ct.record_bench_validator_verdict("real", passed=False) is False

    def test_rank_order_oracle_beats_verifier_human_beats_oracle(self):
        from ghost_agent.core.calibration import (
            CalibrationSample, _SOURCE_RANK, _resolve_superseded)
        assert _SOURCE_RANK["bench_validator"] > _SOURCE_RANK["verifier_late"]
        assert _SOURCE_RANK["user_correction"] > _SOURCE_RANK["bench_validator"]

        def s(source, outcome):
            return CalibrationSample(
                composite=0.5, entropy_component=0.5,
                competence_component=0.5, uncertainty_pressure=0.0,
                outcome=outcome, req_id="r1", source=source)
        resolved = _resolve_superseded(
            [s("turn", 1.0), s("bench_validator", 0.0),
             s("verifier_late", 1.0)])
        assert len(resolved) == 1
        assert resolved[0].source == "bench_validator"

    def test_equal_mass_cap_keeps_newest_bench_and_needs_real_rows(self):
        from ghost_agent.core.calibration import (
            CalibrationSample, _apply_bench_mass_cap)

        def s(req, origin):
            return CalibrationSample(
                composite=0.5, entropy_component=0.5,
                competence_component=0.5, uncertainty_pressure=0.0,
                outcome=1.0, req_id=req, origin=origin)
        real = [s("r1", "user")]
        bench = [s(f"b{i}", "bench") for i in range(4)]
        capped = _apply_bench_mass_cap(real + bench)
        origins = [x.origin for x in capped]
        assert origins.count("bench") == 1
        assert capped[-1].req_id == "b3"          # newest bench row kept
        assert _apply_bench_mass_cap(bench) == [] # no real ⇒ no bench
        # Under the cap: everything passes through untouched.
        assert len(_apply_bench_mass_cap(real + bench[:1])) == 2

    def test_metrics_describe_the_capped_population(self, tmp_path):
        ct = self._tracker(tmp_path)
        self._record_turn(ct, "r1", "user", outcome=1.0)
        for i in range(5):
            self._record_turn(ct, f"b{i}", "bench", outcome=0.0)
        rows = ct._load_epoch()
        assert sum(1 for r in rows if r.origin == "bench") == 1
        assert sum(1 for r in rows if r.origin == "user") == 1


# ──────────────────────────────────────────────────────────────────────
# PRM origin feature + real-only floors
# ──────────────────────────────────────────────────────────────────────

def _corpus(kind, n_pass, n_fail, tool="scratchpad"):
    out = []
    for i in range(n_pass):
        out.append(Trajectory(user_request=f"{kind} pass {i}",
                              outcome=Outcome.PASSED.value,
                              tool_calls=[ToolCall(name=tool)],
                              n_steps=1, task_kind=kind))
    for i in range(n_fail):
        out.append(Trajectory(user_request=f"{kind} fail {i}",
                              outcome=Outcome.FAILED.value,
                              tool_calls=[ToolCall(name="execute", error="x")],
                              n_steps=1, task_kind=kind))
    return out


class TestPRMOrigin:
    def test_feature_appended_and_wired(self):
        from ghost_agent.prm.features import (
            PRM_FEATURE_NAMES, ActionFeatures, PlanState,
            extract_step_features)
        assert PRM_FEATURE_NAMES[-1] == "origin_bench"
        bench_fv = extract_step_features(
            PlanState(user_request="x", origin_bench=True), ActionFeatures())
        real_fv = extract_step_features(
            PlanState(user_request="x"), ActionFeatures())
        assert bench_fv.by_name["origin_bench"] == 1.0
        assert real_fv.by_name["origin_bench"] == 0.0

    def test_label_builder_derives_origin_from_task_kind(self):
        from ghost_agent.prm.labels import _build_state_for_step
        bench = Trajectory(user_request="b", task_kind="bench")
        real = Trajectory(user_request="r")
        assert _build_state_for_step(bench, 0).origin_bench is True
        assert _build_state_for_step(real, 0).origin_bench is False

    def test_bench_augments_after_real_floors_and_is_mass_capped(self):
        from ghost_agent.prm import PRMTrainer
        real = _corpus("user_request", 4, 4)
        bench = _corpus("bench", 20, 20)
        trainer = PRMTrainer(min_samples=5, min_trajectories=2)
        report = trainer.run(real, bench_trajectories=bench)
        assert report.fit_succeeded
        assert 0 < report.n_bench_samples <= len(real)
        # Real-only floor: a bench flood must not unlock a fit the real
        # corpus is too thin to earn.
        starved = PRMTrainer(min_samples=5, min_trajectories=2).run(
            [], bench_trajectories=bench)
        assert not starved.fit_succeeded
        assert "trajectories" in starved.bail_reason


class TestRouterOrigin:
    def test_feature_appended_and_default_real(self):
        from ghost_agent.router.features import FEATURE_NAMES, extract_features
        assert FEATURE_NAMES[-1] == "origin_bench"
        assert extract_features("hi").by_name["origin_bench"] == 0.0
        assert extract_features("hi", origin_bench=True).by_name["origin_bench"] == 1.0

    def test_floors_are_real_only(self):
        from ghost_agent.router.trainer import RouterTrainer
        bench = _corpus("bench", 30, 30, tool="execute")
        report = RouterTrainer().run([], bench_trajectories=bench)
        assert not report.fit_succeeded
        assert "too few" in (report.bail_reason or "")


# ──────────────────────────────────────────────────────────────────────
# GEPA trainset origin
# ──────────────────────────────────────────────────────────────────────

class TestTrainsetOrigin:
    def _bench_traj(self):
        return Trajectory(user_request="write fib", task_kind="bench",
                          outcome=Outcome.PASSED.value,
                          final_response="def fib(n): ...", n_steps=2)

    def test_bench_pass_is_gold_and_carries_origin(self):
        from ghost_agent.optim.trainset import build_trainset
        exs = build_trainset([self._bench_traj()], "planning.decompose")
        assert len(exs) == 1
        assert exs[0].origin == "bench"
        assert exs[0].expected_output["final_response"] == "def fib(n): ..."

    def test_reflection_final_response_stays_blanked(self):
        from ghost_agent.optim.trainset import build_trainset
        refl = Trajectory(user_request="x", task_kind="reflection",
                          outcome=Outcome.PASSED.value,
                          final_response="DIAGNOSIS: ...",
                          planning_output="1. do it", n_steps=2)
        exs = build_trainset([refl], "planning.decompose")
        assert exs[0].expected_output["final_response"] == ""
        assert exs[0].origin == "reflection"

    def test_holdout_identity_ignores_origin(self):
        from ghost_agent.optim.trainset import TrainExample, _example_identity
        a = TrainExample(signature_name="s", source_trajectory_id="t1",
                         origin="user_request")
        b = TrainExample(signature_name="s", source_trajectory_id="t1",
                         origin="bench")
        assert _example_identity(a) == _example_identity(b)


class TestToolFixtureOrigin:
    def test_fixture_dataclass_carries_origin(self):
        from ghost_agent.optim.tool_fixtures import ToolChoiceFixture
        fx = ToolChoiceFixture(fixture_id="f", request_id="r", ts="",
                               user_request="u")
        assert fx.origin == "user_request"
        assert "origin" in fx.to_dict()

    def test_bench_index_keys_by_req_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.eval.banks import trajectories_root
        col = TrajectoryCollector(root=trajectories_root(),
                                  session_id="bench-mbpp")
        col.append(Trajectory(user_request="solve", task_kind="bench",
                              outcome=Outcome.PASSED.value,
                              extra={"req_id": "abc123"}))
        from ghost_agent.optim.tool_fixtures import _bench_trajectory_index
        idx = _bench_trajectory_index()
        assert set(idx) == {"abc123"}
        assert idx["abc123"].task_kind == "bench"


# ──────────────────────────────────────────────────────────────────────
# The lesson gate (extracted pure — behavioral pins, not mirrors)
# ──────────────────────────────────────────────────────────────────────

class TestLessonGate:
    def _gate(self, **kw):
        from ghost_agent.core.dream import lesson_gate_decision
        base = dict(aborted_by_solver=False, validator_infra_crash=False,
                    mastered=False, journal_source=None, passed=True,
                    attempt=0, is_new_cluster=False, compression_delta=0.0,
                    solution_novelty=None)
        base.update(kw)
        return lesson_gate_decision(**base)

    def test_bench_failure_shape_suppresses(self):
        # Bench pins the neutral frontier shape (is_new_cluster=False) —
        # a failed bench item must land in repeat-failure suppression,
        # NOT the "first failure on new cluster" mint (which the
        # self-play-shaped default is_new_cluster=True would trigger).
        write, reason = self._gate(passed=False, is_new_cluster=False)
        assert write is False
        assert "repeat failure" in reason

    def test_bench_struggled_then_won_mints(self):
        write, reason = self._gate(passed=True, attempt=2)
        assert write is True
        assert "struggled-then-won" in reason

    def test_novelty_branch_is_generic_only(self):
        # The novelty branch exists for SELF-PLAY. Bench pins
        # solution_novelty=None at the call site (R1: bench never records
        # frontier winners, so novelty read ~1.0 on every first-try pass
        # and would have minted unverified lessons at volume).
        write, _ = self._gate(passed=True, attempt=0, solution_novelty=0.7)
        assert write is True
        write, reason = self._gate(passed=True, attempt=0,
                                   solution_novelty=0.2)
        assert write is False
        assert "low novelty" in reason

    def test_bench_first_try_pass_shape_defers(self):
        # The bench-shaped inputs (novelty=None, neutral frontier): a
        # first-try pass must DEFER, never mint — the end-to-end twin
        # lives in tests/test_bench_solve_loop_1c.py.
        write, reason = self._gate(passed=True, attempt=0,
                                   solution_novelty=None)
        assert write is False
        assert "no new signal" in reason

    def test_poison_guards_hold(self):
        assert self._gate(aborted_by_solver=True)[0] is False
        assert self._gate(validator_infra_crash=True)[0] is False
        assert self._gate(mastered=True)[0] is False
        assert self._gate(journal_source="j", passed=True)[0] is False
        # Journal FAILURES still fall through to the failure branches.
        write, _ = self._gate(journal_source="j", passed=False,
                              is_new_cluster=True)
        assert write is True

    def test_first_try_new_cluster_mints(self):
        assert self._gate(passed=True, attempt=0, is_new_cluster=True)[0]
        assert self._gate(passed=True, attempt=0,
                          compression_delta=0.1)[0]


# ──────────────────────────────────────────────────────────────────────
# Learning-health denominators
# ──────────────────────────────────────────────────────────────────────

class TestLearningHealthSurfaces:
    def test_label_health_counts_origins(self):
        from ghost_agent.core.learning_health import _label_health
        out = _label_health([
            {"outcome": 1.0, "source": "turn"},
            {"outcome": 0.0, "source": "turn", "origin": "bench"},
            {"outcome": 1.0, "source": "bench_validator", "origin": "bench"},
        ])
        assert out["label_origins"] == {"user": 1, "bench": 2}

    def test_bench_health_lines_render_the_scoreboard(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.eval import banks
        item = {"bank": "mbpp", "item_id": "mbpp-1", "cluster": "algo"}
        banks.record_result(item, passed=True, status="SUCCESS (in 1 attempts)",
                            attempts=1)
        banks.record_result(item, passed=False,
                            status="NO_RESULT (run did not conclude)",
                            attempts=0)
        from ghost_agent.core.learning_health import _bench_health_lines
        lines = _bench_health_lines()
        joined = "\n".join(lines)
        assert "BENCH FLYWHEEL" in joined
        assert "mbpp: 2 runs" in joined
        assert "1 unresolved" in joined

    def test_no_banks_renders_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.core.learning_health import _bench_health_lines
        assert _bench_health_lines() == []


# ──────────────────────────────────────────────────────────────────────
# R1 review fixes — behavioral pins
# ──────────────────────────────────────────────────────────────────────

class TestBenchEnrollmentUnit:
    def _ctx(self, tmp_path, bank="mbpp", item="mbpp-7"):
        (tmp_path / "system").mkdir(parents=True, exist_ok=True)
        (tmp_path / "system" / "experiments.json").write_text(json.dumps({
            "salt": "u",
            "experiments": [{"name": "tts_bon", "scope": "bench"}],
        }), encoding="utf-8")
        ex.reset_registry_cache()
        return types.SimpleNamespace(
            memory_dir=tmp_path / "system" / "memory",
            trajectory_extra_static={"bench_bank": bank, "bench_item": item})

    def test_all_attempts_of_one_item_share_an_arm(self, tmp_path, monkeypatch):
        # R1 MAJOR: per-req_id assignment re-randomized every attempt —
        # attempt 2's input (the rejection message) was a function of
        # attempt 1's ARM. The unit is the bench ITEM.
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        ctx = self._ctx(tmp_path)
        arms = [ex.enroll_request(ctx, f"bench-rq{i}", eligible=True,
                                  origin="bench")["tts_bon"]
                for i in range(3)]
        assert len(set(arms)) == 1
        # The stash stays keyed by req_id (late reads find their own arm).
        assert ex.arm_for(ctx, "tts_bon", "bench-rq2") == arms[0]

    def test_different_items_can_land_on_different_arms(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        seen = set()
        for i in range(12):
            ctx = self._ctx(tmp_path, item=f"mbpp-{i}")
            seen.add(ex.enroll_request(ctx, f"r{i}", eligible=True,
                                       origin="bench")["tts_bon"])
            if len(seen) == 2:
                break
        assert seen == {"control", "treatment"}


class TestGepaGateHygiene:
    def _ex(self, i, origin):
        from ghost_agent.optim.trainset import TrainExample
        return TrainExample(signature_name="s",
                            source_trajectory_id=f"t{origin}{i}",
                            origin=origin)

    def test_real_only_gate_moves_bench_out_of_private(self):
        from ghost_agent.optim.trainset import real_only_gate
        pub = [self._ex(i, "user_request") for i in range(3)]
        priv = [self._ex(0, "bench"), self._ex(9, "user_request")]
        new_pub, new_priv, moved = real_only_gate(pub, priv)
        assert moved == 1
        assert all(e.origin != "bench" for e in new_priv)
        assert len(new_pub) == 4 and len(new_priv) == 1

    def test_real_only_gate_recaps_the_public_tier(self):
        # R2 review MAJOR: moving 100% of bench into a tier holding only
        # ~70% of real made public bench-MAJORITY right after the run
        # printed "equal-mass cap" — the move must re-cap.
        from ghost_agent.optim.trainset import real_only_gate
        pub = ([self._ex(i, "user_request") for i in range(7)]
               + [self._ex(i, "bench") for i in range(5)])
        priv = [self._ex(i + 100, "bench") for i in range(6)]
        new_pub, _new_priv, moved = real_only_gate(pub, priv)
        assert moved == 6
        real_pub = [e for e in new_pub if e.origin != "bench"]
        bench_pub = [e for e in new_pub if e.origin == "bench"]
        assert len(bench_pub) <= len(real_pub)
        # Newest bench (the just-moved rows are the tail) survive.
        assert bench_pub[-1].source_trajectory_id == "tbench105"

    def test_per_origin_selection_keeps_bench_under_a_healthy_cap(self):
        # R1 MAJOR: head truncation dropped every bench example whenever
        # the real corpus filled the cap — a printed no-op.
        from ghost_agent.optim.trainset import per_origin_selection
        real = [self._ex(i, "user_request") for i in range(300)]
        bench = [self._ex(i, "bench") for i in range(100)]
        r, b = per_origin_selection(real + bench, max_examples=200)
        assert len(r) + len(b) == 200
        assert len(b) == 100          # the bench reserve held
        assert b[-1].source_trajectory_id == "tbench99"   # newest kept

    def test_per_origin_selection_equal_mass_and_no_real_no_bench(self):
        from ghost_agent.optim.trainset import per_origin_selection
        real = [self._ex(i, "user_request") for i in range(3)]
        bench = [self._ex(i, "bench") for i in range(50)]
        r, b = per_origin_selection(real + bench, max_examples=None)
        assert len(r) == 3 and len(b) == 3          # equal-mass cap
        assert per_origin_selection(bench, None) == ([], [])


class TestRouterBenchCap:
    def test_bench_rows_are_capped_and_counted(self):
        # R1 MAJOR: uncapped near-single-class bench mass outweighed a
        # balanced real corpus ~10:1 and drove the gate to permanent
        # escalate-all with no trace in the report.
        from ghost_agent.router.trainer import (
            RouterTrainer, _GATE_TRAIN_FRACTION, _gate_min_trajectories)
        # Sized from the trainer's DERIVED floor, not a literal. This test
        # had silently stopped exercising the cap it is named for: its
        # corpus bailed at the floor before ever reading a bench row.
        _each = int(_gate_min_trajectories() * 0.6) + 1
        real = _corpus("user_request", _each, _each)
        bench = _corpus("bench", 400, 400, tool="execute")
        report = RouterTrainer().run(real, bench_trajectories=bench)
        assert report.n_bench == int(len(real) * _GATE_TRAIN_FRACTION)
        assert f"+{report.n_bench} bench" in report.summary() \
            or not report.fit_succeeded


class TestCalibrationCapOrdering:
    def test_resolution_runs_before_the_cap(self, tmp_path):
        # A bench turn row + its oracle re-label are ONE observation.
        # Cap-before-resolve would count them as two bench rows against
        # real mass and drop one real-mass slot.
        from ghost_agent.core.calibration import CalibrationTracker
        ct = CalibrationTracker(tmp_path / "calibration")
        assert ct.record(composite=0.7, entropy_component=0.5,
                         competence_component=0.6, outcome=1.0,
                         source="turn", req_id="b1", origin="bench")
        assert ct.record_bench_validator_verdict("b1", passed=False)
        assert ct.record(composite=0.6, entropy_component=0.5,
                         competence_component=0.6, outcome=1.0,
                         source="turn", req_id="r1", origin="user")
        rows = ct._load_epoch()
        assert len(rows) == 2
        assert {r.req_id for r in rows} == {"b1", "r1"}
        assert next(r for r in rows if r.req_id == "b1").source == \
            "bench_validator"

    def test_user_correction_row_supersedes_via_req_id(self, tmp_path):
        # R1 MAJOR (pre-existing): the correction stash dropped req_id, so
        # the ground-truth negative sat NEXT TO the proxy positive in the
        # fit instead of replacing it.
        from ghost_agent.core.calibration import CalibrationTracker
        ct = CalibrationTracker(tmp_path / "calibration")
        ct.record(composite=0.8, entropy_component=0.5,
                  competence_component=0.7, outcome=1.0,
                  source="turn", req_id="R9")
        ct.record(composite=0.8, entropy_component=0.5,
                  competence_component=0.7, outcome=0.0,
                  source="user_correction", req_id="R9")
        rows = ct._load_epoch()
        assert len(rows) == 1
        assert rows[0].source == "user_correction"
        assert rows[0].outcome == 0.0

    def test_dict_cap_twin_matches_the_fit_cap(self):
        from ghost_agent.core.calibration import apply_bench_mass_cap_rows
        rows = ([{"origin": "user", "req_id": "r"}]
                + [{"origin": "bench", "req_id": f"b{i}"} for i in range(4)])
        capped = apply_bench_mass_cap_rows(rows)
        assert sum(1 for r in capped if r["origin"] == "bench") == 1
        assert capped[-1]["req_id"] == "b3"
        assert apply_bench_mass_cap_rows(rows[1:]) == []


class TestBenchClusterTrust:
    def test_bank_cluster_is_trusted_past_the_journal_whitelist(self):
        # R2/R3: gsm8k ships cluster="python_general", which has no
        # CLUSTER_KEYWORDS row — the journal-guess whitelist rejected it
        # and every gsm8k solve was filed under the abandoned frontier
        # seed. Bank clusters are operator-curated: trusted.
        from ghost_agent.core.dream import resolve_cluster_key
        got = resolve_cluster_key({"cluster_key": "sql"}, "", "avg a list",
                                  challenge_domains=["python_general"],
                                  trusted_domains=True)
        assert got == "python_general"

    def test_journal_guesses_still_face_the_whitelist(self):
        # The journal fallback's "no idea" default must NOT become
        # authoritative (R3: pre-1c semantics restored for untrusted
        # domains — python_general from a guess falls through).
        from ghost_agent.core.dream import resolve_cluster_key
        got = resolve_cluster_key({"cluster_key": "sql"}, "", "whatever",
                                  challenge_domains=["python_general"],
                                  trusted_domains=False)
        assert got == "sql"


class TestFloodDetectorPopulation:
    def test_raw_origin_mix_is_current_epoch_precap(self):
        # R3: counting ALL epochs let ~retired real rows mask a current
        # epoch flooded with bench. The detector must see the epoch the
        # fit reads, pre-cap.
        from ghost_agent.core.calibration import CURRENT_EPOCH
        from ghost_agent.core.learning_health import _live_epoch_filter
        rows = (
            # A retired epoch full of real rows.
            [{"origin": "user", "req_id": f"old{i}", "epoch": "old",
              "ts": "2026-01-01T00:00:00Z"} for i in range(50)]
            # A current epoch flooded 4:1 bench.
            + [{"origin": "bench", "req_id": f"b{i}",
                "epoch": CURRENT_EPOCH} for i in range(8)]
            + [{"origin": "user", "req_id": f"r{i}",
                "epoch": CURRENT_EPOCH} for i in range(2)]
        )
        capped, _n_other, _n_sup, precap = _live_epoch_filter(rows)
        pre_mix = {}
        for s in precap:
            pre_mix[s.get("origin") or "user"] = \
                pre_mix.get(s.get("origin") or "user", 0) + 1
        # Pre-cap = the flood is visible (8 bench vs 2 real), and the
        # retired-epoch real mass is NOT diluting it.
        assert pre_mix == {"bench": 8, "user": 2}
        # Capped = what the fit consumes (≤1:1).
        cap_bench = sum(1 for s in capped
                        if (s.get("origin") or "user") == "bench")
        cap_real = sum(1 for s in capped
                       if (s.get("origin") or "user") != "bench")
        assert cap_bench <= cap_real == 2


class TestReportPopulations:
    def test_bench_coverage_divides_its_own_population(self):
        # R5 review MAJOR: the bench report divided bench stamps by USER
        # turns → "128/1 recorded user turns (12800%)"; with a clean
        # bench corpus the line vanished entirely.
        trajs = [_traj("bench", {"tts_bon": "treatment"}) for _ in range(4)]
        all_stats, trig, cov = ex.summarize_streaming(
            trajs, admit_task_kinds=("bench",))
        assert cov["admitted_turns"] == 4 and cov["user_turns"] == 0
        out = ex.render_report(all_stats, triggered=trig, coverage=cov,
                               population=ex.SCOPE_BENCH)
        assert "BENCH-scoped arms" in out
        assert "4/4 bench solve turns" in out
        assert "live randomized arms" not in out
        assert "n counts ATTEMPTS" in out   # items-vs-attempts label

    def test_admit_names_keeps_rescoped_stamps_out(self):
        # A spec re-scoped live↔bench must not render its STALE
        # other-population stamps under the wrong banner.
        trajs = [_traj("bench", {"tts_bon": "treatment",
                                 "risk_steer": "control"})]
        all_stats, _t, _c = ex.summarize_streaming(
            trajs, admit_task_kinds=("bench",), admit_names={"tts_bon"})
        assert set(all_stats) == {"tts_bon"}

    def test_malformed_stamp_is_not_counted_as_covered(self):
        # R5 review: {"name": null} counted as stamped while folding into
        # no arm — the coverage instrument overstated.
        trajs = [_traj("user_request", {"use_planning": None})]
        _a, _t, cov = ex.summarize_streaming(trajs)
        assert cov["stamped"] == 0

    def test_live_announce_pass_denies_rescoped_specs_only(
            self, tmp_path, monkeypatch):
        # R5 found the live pass unfiltered; R6 found the allow-list fix
        # WORSE (a disabled spec's accumulated history vanished behind a
        # false "stamp regressed" alarm, and an experiments.json listing
        # only a bench spec silenced every live verdict). The settled
        # form is a DENY-list of the other scope: a spec re-scoped to
        # bench must not announce from its stale live stamps, while a
        # disabled spec's own live-collected verdict still announces.
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        sysd = tmp_path / "system"
        (sysd / "memory").mkdir(parents=True)
        (sysd / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [
                {"name": "dead_spec", "enabled": False},
                {"name": "moved_spec", "scope": "bench"},
            ]}), encoding="utf-8")
        ex.reset_registry_cache()
        from ghost_agent.distill.collector import TrajectoryCollector
        col = TrajectoryCollector(root=sysd / "trajectories",
                                  session_id="s1")
        for spec in ("dead_spec", "moved_spec"):
            col.append(Trajectory(user_request="q", outcome="passed",
                                  extra={"experiments": {spec: "treatment"}}))

        class _Log:
            def __init__(self):
                self.lines = []

            def record(self, phase, line, **kw):
                self.lines.append(line)

        log = _Log()
        context = types.SimpleNamespace(
            memory_dir=sysd / "memory", args=None, activity_log=log)
        fake = MagicMock()
        fake.verdict = "TREATMENT BETTER"
        fake.confound = ""
        fake.metric = "failure_rate"
        fake.control_mean, fake.treatment_mean = 0.5, 0.2
        fake.control_n, fake.treatment_n = 40, 40
        # No get_activity_log patch: announce imports it INSIDE the fn
        # and resolves the ledger via context.activity_log (R7 nit — the
        # old create=True patch fabricated a module attr nothing read).
        with patch.object(ex, "compare_arms", return_value=[fake]):
            lines = ex.announce_new_verdicts(context)
        assert not any("moved_spec" in l for l in lines), (
            "a bench-re-scoped spec announced from stale LIVE stamps: "
            f"{lines}")
        assert any("dead_spec" in l for l in lines), (
            "a disabled spec's own live-collected verdict was erased — "
            "the documented enabled:false dial must stay cost-free")

    def test_no_bench_boot_blocks_bench_verdict_computation(
            self, tmp_path, monkeypatch):
        # R5 review MAJOR: announce/introspect read the bench corpus
        # directly, bypassing the --no-bench consumption gate — the
        # full_no_bench arm still computed and ANNOUNCED bench verdicts
        # (whose notify records then reach real replies via the digest).
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        sysd = tmp_path / "system"
        (sysd / "memory").mkdir(parents=True)
        (sysd / "experiments.json").write_text(json.dumps({
            "salt": "t",
            "experiments": [{"name": "tts_bon", "scope": "bench"}],
        }), encoding="utf-8")
        ex.reset_registry_cache()
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.eval.banks import trajectories_root
        bcol = TrajectoryCollector(root=trajectories_root(),
                                   session_id="bench-mbpp")
        bcol.append(Trajectory(user_request="q", outcome="passed",
                               task_kind="bench",
                               extra={"experiments":
                                      {"tts_bon": "treatment"}}))

        fake = MagicMock()
        fake.verdict = "TREATMENT BETTER"
        fake.confound = ""
        fake.metric = "failure_rate"
        fake.control_mean, fake.treatment_mean = 0.5, 0.2
        fake.control_n, fake.treatment_n = 40, 40

        class _Log:
            def record(self, *a, **k):
                pass

        for no_bench, expect_bench_line in ((True, False), (False, True)):
            (sysd / "experiments_announced.json").unlink(missing_ok=True)
            ex._ANNOUNCED_THIS_PROCESS.clear()
            context = types.SimpleNamespace(
                memory_dir=sysd / "memory",
                args=types.SimpleNamespace(no_bench=no_bench),
                activity_log=_Log())
            with patch.object(ex, "compare_arms", return_value=[fake]):
                lines = ex.announce_new_verdicts(context)
            has_bench = any("BENCH experiment" in l for l in lines)
            assert has_bench == expect_bench_line, (no_bench, lines)


class TestHydratedLessonsStamp:
    def _record(self, ctx_extra):
        from ghost_agent.core.agent import GhostAgent

        class _Col:
            def __init__(self):
                self.appended = []

            def append(self, traj):
                self.appended.append(traj)
                return Path("/dev/null")

        col = _Col()
        ctx = types.SimpleNamespace(
            trajectory_collector=col, self_model=None, profile_memory=None,
            _recent_calib_for_correction=None,
            _recent_trajectories_for_correction=None, **ctx_extra)
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        agent._record_turn_trajectory(
            messages=[{"role": "user", "content": "solve it"},
                      {"role": "assistant", "content": "done"}],
            final_content="done", req_id="rH", model="m",
            user_request="solve it")
        return col.appended[0]

    def test_bench_records_never_carry_the_real_turns_lessons(self):
        # R5 review MAJOR: the read-only wrapper passed
        # last_playbook_triggers through, so bench trajectories were
        # stamped with the LAST REAL TURN's lesson set as their own
        # hydration — counterfactual-attribution poison.
        sm = types.SimpleNamespace(
            is_read_only=True,
            last_playbook_triggers=["real lesson trigger"])
        traj = self._record({
            "trajectory_task_kind": "bench",
            "turn_origin_label": "bench",
            "skill_memory": sm,
        })
        assert traj.task_kind == "bench"
        assert "hydrated_lessons" not in (traj.extra or {})

    def test_real_turns_still_stamp_their_hydration(self):
        sm = types.SimpleNamespace(
            is_read_only=False,
            last_playbook_triggers=["real lesson trigger"])
        traj = self._record({"skill_memory": sm})
        assert (traj.extra or {}).get("hydrated_lessons") == \
            ["real lesson trigger"]


class TestBenchStatsHome:
    def test_learning_health_reads_the_memory_dirs_home(
            self, tmp_path, monkeypatch):
        # R5 review MAJOR: the bench section read $GHOST_HOME while every
        # sibling section derives from memory_dir — pointing the report
        # at an archive rendered the LIVE box's banks.
        home_a = tmp_path / "homeA"
        home_b = tmp_path / "homeB"
        (home_a / "system" / "memory").mkdir(parents=True)
        (home_b / "system" / "memory").mkdir(parents=True)
        from ghost_agent.eval import banks
        banks.record_result({"bank": "mbpp", "item_id": "m1",
                             "cluster": "algo"},
                            passed=True, status="SUCCESS (in 1 attempts)",
                            attempts=1, home=str(home_a))
        monkeypatch.setenv("GHOST_HOME", str(home_b))
        from ghost_agent.core.learning_health import (
            _bench_health_lines, collect_learning_health)
        report = collect_learning_health(home_a / "system" / "memory")
        assert "bench" in report and "mbpp" in report["bench"]
        lines = _bench_health_lines(home_a / "system" / "memory")
        assert any("mbpp" in l for l in lines)
        # And home B (the env home) genuinely has none.
        assert _bench_health_lines(home_b / "system" / "memory") == []


class TestBenchOnlyEpochRender:
    def test_render_survives_a_bench_only_epoch(self, tmp_path, monkeypatch):
        # R4 review, REPRODUCED: a bench-only current epoch empties the
        # capped population, _label_health returns {}, and the first cut
        # of the flood-detector render line took the WHOLE learning-health
        # report down with a KeyError — the operator instrument dying at
        # exactly maximal flood.
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        from ghost_agent.core.calibration import CalibrationTracker
        md = tmp_path / "system" / "memory"
        md.mkdir(parents=True, exist_ok=True)
        ct = CalibrationTracker(tmp_path / "system" / "calibration")
        for i in range(5):
            ct.record(composite=0.6, entropy_component=0.5,
                      competence_component=0.5, outcome=0.0,
                      source="turn", req_id=f"b{i}", origin="bench")
        from ghost_agent.core.learning_health import render_learning_health
        out = render_learning_health(md)
        assert "origin mix (current epoch, pre-cap)" in out
        assert "BENCH-ONLY epoch" in out

    def test_recent_samples_filters_before_the_window(self, tmp_path):
        # R4 review: filtering AFTER the tail was a no-op against
        # dilution — the real rows bench pushed out stayed out.
        from ghost_agent.core.calibration import CalibrationTracker
        ct = CalibrationTracker(tmp_path / "calibration")
        ct.record(composite=0.5, entropy_component=0.5,
                  competence_component=0.5, outcome=1.0,
                  source="turn", req_id="real-1", origin="user")
        for i in range(10):
            ct.record(composite=0.5, entropy_component=0.5,
                      competence_component=0.5, outcome=0.0,
                      source="turn", req_id=f"b{i}", origin="bench")
        # A window of 5 taken tail-first holds zero real rows…
        assert all(s.origin == "bench" for s in ct.recent_samples(5))
        # …but the origin-aware window still finds the real one.
        got = ct.recent_samples(5, exclude_origin="bench")
        assert [s.req_id for s in got] == ["real-1"]


class TestSourceWindowPins:
    # Honest window pins for one-line guards whose behavioral harnesses
    # would cost more than they return (R3 MAJOR-2 c/d). Their observable
    # halves are pinned behaviorally elsewhere (test_bench_handle_chat_1c
    # / test_bench_solve_loop_1c).
    def test_digest_block_is_user_origin_gated(self):
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("Autoadvance Digest")
        window = src[max(0, i - 4000):i]
        assert 'turn_origin(self.context) == "user"' in window

    def test_activity_digest_block_is_user_origin_gated_too(self):
        # R4 review CRIT: the SIBLING activity-digest block 110 lines
        # below the project digest was left with only the internal-request
        # gate — bench finalizes consumed the watermark and ate the very
        # notify events (incl. bench experiment verdicts) 1c produces.
        import inspect
        import ghost_agent.core.agent as agent_mod
        src = inspect.getsource(agent_mod)
        i = src.index("render_activity_digest as _render_adg")
        window = src[i:i + 1200]
        assert 'turn_origin(self.context) == "user"' in window

    def test_run_gepa_ungated_promotion_is_stamped(self):
        p = Path(__file__).resolve().parent.parent / "scripts" / "run_gepa.py"
        src = p.read_text(encoding="utf-8")
        i = src.index("def _promote_staging")
        window = src[i:i + 4000]
        assert "except NameError" in window
        assert "UNGATED (--no-ab-gate)" in window

    def test_every_in_process_bench_read_passes_args(self):
        # R5 review MAJOR: stripping args from all the call sites left
        # 830 tests green while full_no_bench went back to consuming
        # bench. This wiring scan fails on any in-process call that
        # doesn't thread the runtime args into the chokepoint.
        import re
        src_root = Path(__file__).resolve().parent.parent / "src"
        bad = []
        for p in src_root.rglob("*.py"):
            if p.name == "admissibility.py":
                continue          # the definitions themselves
            text = p.read_text(encoding="utf-8")
            # A consumer-only call: iter_bench_trajectories("x") with the
            # paren closing right after the string ⇒ args not threaded.
            for m in re.finditer(
                    r'iter_bench_trajectories\(\s*"[a-z_]+"\s*\)', text):
                # optim/tool_fixtures is the documented OFFLINE reader
                # (no runtime args exist there).
                if p.name != "tool_fixtures.py":
                    bad.append(f"{p.name}: {m.group(0)}")
            # A bare fingerprint call (or a to_thread passing only the
            # fn) ⇒ the skip gate ignores --no-bench.
            for m in re.finditer(
                    r'bench_corpus_fingerprint\(\s*\)', text):
                bad.append(f"{p.name}: {m.group(0)}")
            for m in re.finditer(
                    r'to_thread\(\s*_[a-z_]*bench_fp_fn\s*\)', text):
                bad.append(f"{p.name}: {m.group(0)}")
        assert not bad, f"chokepoint calls missing runtime args: {bad}"

    def test_playbook_triggers_stay_off_the_readonly_passthrough(self):
        # R6: this R5 fix was revertible with 297 tests green (the belt
        # caught the observable effect, but the fix itself was unpinned).
        # The passthrough serving the LAST REAL TURN's lesson set to sim
        # reads is the root cause; it must stay denied.
        import inspect
        import ghost_agent.core.dream as dream_mod
        src = inspect.getsource(dream_mod)
        i = src.index("_SAFE_PASSTHROUGH = frozenset({")
        block = src[i:src.index("})", i)]
        assert "last_playbook_triggers" not in block

    def test_introspect_bench_section_is_scope_filtered(self):
        # R6: reverting the introspect deny wiring survived the suite.
        import inspect
        import ghost_agent.tools.introspect as intro_mod
        src = inspect.getsource(intro_mod)
        i = src.index("BENCH-SCOPED EXPERIMENTS")
        window = src[max(0, i - 3000):i]
        assert "deny_names=_deny_bench" in window
        # …and the LIVE call carries its deny too.
        j = src.index("report_from_trajectories(")
        window2 = src[j:j + 600]
        assert "deny_names=_deny_live" in window2

    def test_every_deny_site_is_wired(self):
        # R7: four of the R6 deny wirings were revertible with the whole
        # suite green. Scan each surface for its deny call (and for the
        # UNFILTERED names_in_scope form — an enabled-only deny let
        # re-scope-then-disable resurrect stale stamps).
        import inspect
        import ghost_agent.core.experiments as ex_mod
        import ghost_agent.core.learning_health as lh_mod
        ex_src = inspect.getsource(ex_mod)
        i = ex_src.index("def announce_new_verdicts")
        window = ex_src[i:i + 6000]
        assert "deny_names=bench_scoped" in window          # live pass
        assert "deny_names=set(reg.names_in_scope(SCOPE_LIVE))" in window
        assert "names_in_scope(SCOPE_BENCH)" in window
        lh_src = inspect.getsource(lh_mod)
        j = lh_src.index("def _experiment_health_lines")
        assert "deny_names=_deny" in lh_src[j:j + 3000]
        assert "names_in_scope" in lh_src[j:j + 3000]
        script = (Path(__file__).resolve().parent.parent / "scripts"
                  / "experiment_report.py").read_text(encoding="utf-8")
        assert "deny_names=_deny" in script
        assert "names_in_scope" in script

    def test_undelivered_verdicts_are_never_marked_announced(
            self, tmp_path, monkeypatch):
        # R7 (behavioral, proven observable): with the marker adds above
        # the log-None check, a tick before the ledger exists poisoned
        # the process-local set and the NEXT tick (with a ledger)
        # announced nothing — verdict lost for good.
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
        sysd = tmp_path / "system"
        (sysd / "memory").mkdir(parents=True)
        (sysd / "experiments.json").write_text(json.dumps({
            "salt": "t", "experiments": [{"name": "x_live"}]}),
            encoding="utf-8")
        ex.reset_registry_cache()
        col = TrajectoryCollector(root=sysd / "trajectories",
                                  session_id="s1")
        col.append(Trajectory(user_request="q", outcome="passed",
                              extra={"experiments": {"x_live": "treatment"}}))
        fake = MagicMock()
        fake.verdict = "TREATMENT BETTER"
        fake.confound = ""
        fake.metric = "failure_rate"
        fake.control_mean, fake.treatment_mean = 0.5, 0.2
        fake.control_n, fake.treatment_n = 40, 40

        ex._ANNOUNCED_THIS_PROCESS.clear()
        no_log_ctx = types.SimpleNamespace(
            memory_dir=sysd / "memory", args=None, activity_log=None)
        with patch.object(ex, "compare_arms", return_value=[fake]):
            assert ex.announce_new_verdicts(no_log_ctx) == []
        assert all(not v for v in ex._ANNOUNCED_THIS_PROCESS.values()), \
            "undelivered verdicts were marked announced"

        class _Log:
            def __init__(self):
                self.lines = []

            def record(self, phase, line, **kw):
                self.lines.append(line)

        log = _Log()
        with_log_ctx = types.SimpleNamespace(
            memory_dir=sysd / "memory", args=None, activity_log=log)
        with patch.object(ex, "compare_arms", return_value=[fake]):
            lines = ex.announce_new_verdicts(with_log_ctx)
        assert lines, "the verdict was lost after the log-less tick"

    def test_dream_bench_block_resets_the_full_ring_inventory(self):
        # Source twin of the behavioral ring test: the reset list in
        # dream.py's bench block must name every known shared ring, so a
        # deletion is caught even if the behavioral harness drifts.
        import inspect
        import ghost_agent.core.dream as dream_mod
        src = inspect.getsource(dream_mod)
        i = src.index("PROTECTIVE resets FIRST")
        window = src[i:i + 3000]
        for name in ("_recent_trajectories_for_correction",
                     "_recent_calib_for_correction",
                     "_experiment_arms_recent",
                     "_turn_facts_recent",
                     "_recent_turn_outcome",
                     "_surfaced_triggers_by_traj",
                     "self_model = None"):
            assert name in window, f"bench isolate reset missing: {name}"
