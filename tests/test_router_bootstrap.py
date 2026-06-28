"""Tests for the startup bootstrap-train (bootstrap_router).

The router ships untrained and a model is otherwise only produced by an IDLE
retrain (needs a long-lived idle process); a busy server or benchmark never
idles, so the dispatcher would stay escalate-all forever. bootstrap_router()
trains ONCE at boot from the existing trajectory log when enough labeled,
multi-class data exists, with a safe fallback to pass-through.

Pure-python: fake Trajectory records + temp files. No docker / live model.
"""

import pytest


def _traj(req, steps, calls, outcome="passed", heavy=False):
    from ghost_agent.distill.schema import Trajectory, ToolCall
    tcs = [ToolCall(name=("execute" if heavy else "web_search")) for _ in range(calls)]
    return Trajectory(user_request=req, n_steps=steps, tool_calls=tcs, outcome=outcome)


def _balanced_corpus(n_each):
    """n_each easy + n_each hard trajectories (both classes present)."""
    easy = [_traj(f"what is {i}?", 1, 1) for i in range(n_each)]
    hard = [_traj(f"build deploy {i}", 6, 5, outcome="failed", heavy=True) for i in range(n_each)]
    return easy + hard


class TestBootstrapTrains:
    def test_trains_when_enough_multiclass_samples(self, tmp_path):
        from ghost_agent.router import bootstrap_router, ComplexityDispatcher
        # 30 easy + 30 hard = 60 labeled, well over the default min of 50.
        trajs = _balanced_corpus(30)
        save = tmp_path / "router" / "checkpoint.json"
        clf, report = bootstrap_router(trajs, save_path=save)
        assert clf is not None
        assert report.fit_succeeded is True
        assert report.easy >= 1 and report.hard >= 1
        # Persisted to disk so a later boot loads instead of re-training.
        assert save.exists()
        # The trained model actually routes (does not escalate everything).
        disp = ComplexityDispatcher(classifier=clf, disabled=False)
        assert disp.route("what is 7?").escalated is False

    def test_respects_custom_min_samples(self, tmp_path):
        from ghost_agent.router import bootstrap_router
        trajs = _balanced_corpus(15)  # 30 labeled
        clf, report = bootstrap_router(trajs, min_samples=20)
        assert clf is not None
        assert report.fit_succeeded is True


class TestBootstrapStaysPassThrough:
    def test_too_few_samples(self, tmp_path):
        from ghost_agent.router import bootstrap_router
        # Only 8 labeled — far below the default min of 50.
        clf, report = bootstrap_router(_balanced_corpus(4))
        assert clf is None
        assert report.fit_succeeded is False
        assert "too few" in report.bail_reason

    def test_single_class(self, tmp_path):
        from ghost_agent.router import bootstrap_router
        # All easy: fit() would raise; bootstrap must bail gracefully.
        trajs = [_traj(f"q{i}", 1, 1) for i in range(80)]
        clf, report = bootstrap_router(trajs, min_samples=20)
        assert clf is None
        assert report.fit_succeeded is False
        assert "single-class" in report.bail_reason

    def test_empty_log(self, tmp_path):
        from ghost_agent.router import bootstrap_router
        clf, report = bootstrap_router([])
        assert clf is None
        assert report.fit_succeeded is False


class TestBootstrapNeverRaises:
    def test_malformed_records_do_not_crash(self, tmp_path):
        from ghost_agent.router import bootstrap_router

        # An iterator that yields junk objects with no Trajectory shape.
        def _bad():
            yield object()
            yield None
            yield 42

        clf, report = bootstrap_router(_bad())
        assert clf is None
        assert report.fit_succeeded is False

    def test_exploding_iterator_does_not_crash(self, tmp_path):
        from ghost_agent.router import bootstrap_router

        def _explode():
            yield _traj("ok", 1, 1)
            raise RuntimeError("disk read blew up mid-stream")

        # Must swallow the exception and fall back to pass-through.
        clf, report = bootstrap_router(_explode())
        assert clf is None
        assert report.fit_succeeded is False
        assert report.bail_reason  # some reason recorded

    def test_missing_log_via_collector(self, tmp_path):
        # End-to-end through the real collector pointed at a non-existent dir:
        # iter_trajectories() yields nothing, bootstrap stays pass-through.
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.router import bootstrap_router
        coll = TrajectoryCollector(root=tmp_path / "does-not-exist", session_id="s1")
        clf, report = bootstrap_router(coll.iter_trajectories())
        assert clf is None
        assert report.fit_succeeded is False


class TestBootstrapFromCollector:
    def test_trains_from_real_collector_log(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.router import bootstrap_router, ComplexityDispatcher
        coll = TrajectoryCollector(root=tmp_path / "traj", session_id="s1")
        for t in _balanced_corpus(30):
            coll.append(t)
        save = tmp_path / "router" / "checkpoint.json"
        clf, report = bootstrap_router(coll.iter_trajectories(), save_path=save)
        assert clf is not None
        assert report.fit_succeeded is True
        assert save.exists()
        disp = ComplexityDispatcher(classifier=clf, disabled=False)
        assert disp.route("what is 7?").escalated is False
