"""Tests for the GEPA prompt loader (M7) and the router trainer (H9) — the
two offline-optimization stages that were previously write-only / untrained."""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------- GEPA loader

class TestGepaLoader:
    def _setup(self, monkeypatch, tmp):
        monkeypatch.setenv("GHOST_HOME", str(tmp))
        import ghost_agent.optim.loader as L
        L.clear_cache()
        return L

    def test_absent_returns_baseline(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        assert L.tuned_instruction("planning.decompose", "BASELINE") == "BASELINE"

    def test_present_returns_optimized(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "planning.decompose.json").write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "TUNED: plan tersely.",
        }))
        L.clear_cache()
        assert L.tuned_instruction("planning.decompose", "BASELINE") == "TUNED: plan tersely."

    def test_corrupt_falls_back(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "reflection.critique.json").write_text("{ not json ]")
        L.clear_cache()
        assert L.tuned_instruction("reflection.critique", "BASE") == "BASE"

    def test_empty_optimized_falls_back(self, monkeypatch, tmp_path):
        L = self._setup(monkeypatch, tmp_path)
        d = tmp_path / "system" / "optim"
        d.mkdir(parents=True)
        (d / "x.json").write_text(json.dumps({"optimized_instruction": "   "}))
        L.clear_cache()
        assert L.tuned_instruction("x", "BASE") == "BASE"


# --------------------------------------------------------------- router trainer

def _traj(req, steps, calls, outcome="passed", heavy=False):
    from ghost_agent.distill.schema import Trajectory, ToolCall
    tcs = [ToolCall(name=("execute" if heavy else "web_search")) for _ in range(calls)]
    return Trajectory(user_request=req, n_steps=steps, tool_calls=tcs, outcome=outcome)


class TestRouterTrainer:
    def test_fits_and_persists_and_routes(self, tmp_path):
        from ghost_agent.router import RouterTrainer, ComplexityClassifier, ComplexityDispatcher
        trajs = [_traj(f"what is {i}", 1, 1) for i in range(13)]
        trajs += [_traj(f"build deploy {i}", 6, 5, outcome="failed", heavy=True) for i in range(13)]
        save = tmp_path / "router" / "checkpoint.json"
        rep = RouterTrainer(min_trajectories=15).run(trajs, save_path=save)
        assert rep.fit_succeeded
        assert rep.easy >= 1 and rep.hard >= 1
        assert save.exists()
        # The persisted model loads and produces real (non-escalate) routing.
        clf = ComplexityClassifier.load(save)
        disp = ComplexityDispatcher(classifier=clf, disabled=False)
        assert disp.route("what is 7?").escalated is False

    def test_bails_below_min_trajectories(self, tmp_path):
        from ghost_agent.router import RouterTrainer
        rep = RouterTrainer(min_trajectories=50).run([_traj("q", 1, 1) for _ in range(5)])
        assert rep.fit_succeeded is False
        assert "too few" in rep.bail_reason

    def test_bails_on_single_class(self):
        from ghost_agent.router import RouterTrainer
        # All easy → fit would raise; the trainer must bail gracefully instead.
        rep = RouterTrainer(min_trajectories=3).run([_traj(f"q{i}", 1, 1) for i in range(10)])
        assert rep.fit_succeeded is False
        assert "single-class" in rep.bail_reason

    def test_maybe_retrain_router_hot_swaps_live_dispatcher(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.router import ComplexityDispatcher
        from ghost_agent.tools.memory import _maybe_retrain_router

        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="s1")
        for i in range(13):
            coll.append(_traj(f"what is {i}", 1, 1))
        for i in range(13):
            coll.append(_traj(f"build deploy {i}", 6, 5, outcome="failed", heavy=True))

        class _Ctx:
            trajectory_collector = coll
            complexity_dispatcher = ComplexityDispatcher(classifier=None, disabled=True)
            memory_dir = tmp_path / "mem"
            _router_checkpoint_path = tmp_path / "router" / "checkpoint.json"

        ctx = _Ctx()
        assert ctx.complexity_dispatcher.route("what is 5?").escalated is True  # untrained
        _maybe_retrain_router(ctx)
        assert ctx.complexity_dispatcher.disabled is False
        assert ctx.complexity_dispatcher.route("what is 5?").escalated is False  # now routing

    def test_maybe_retrain_router_noop_without_collector(self):
        from ghost_agent.tools.memory import _maybe_retrain_router

        class _Ctx:
            trajectory_collector = None
            complexity_dispatcher = None
        # Must not raise.
        _maybe_retrain_router(_Ctx())
