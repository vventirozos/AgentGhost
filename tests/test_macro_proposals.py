"""Tests for auto-discovered macro proposals.

Covers the dream-cycle pipeline that mines recurring tool-call sequences
from the trajectory log and registers the strongest as PROPOSED composed
skills (via compile_from_pattern), awaiting user approval.
"""

import pytest

from ghost_agent.core.dream import (
    mine_recurring_tool_sequences, _safe_macro_name, Dreamer,
)
from ghost_agent.distill.schema import Trajectory, ToolCall
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.tools.composed_skills import _registry_from_context


def _traj(tid, seq, outcome="unknown"):
    """seq is a list of (tool_name, args_dict)."""
    return Trajectory(
        id=tid, outcome=outcome,
        tool_calls=[ToolCall(name=n, arguments=a) for n, a in seq],
    )


class TestMining:
    def test_safe_macro_name(self):
        assert _safe_macro_name(["web_search", "deep_research"]) == "auto_web_search_deep_research"
        n = _safe_macro_name(["a-b", "c.d"])  # non-identifier chars sanitized
        assert all(ch.isalnum() or ch == "_" for ch in n)

    def test_finds_recurring_pair_with_args(self):
        seq = [("web_search", {"query": "x"}), ("deep_research", {"query": "y"})]
        trajs = [_traj("t1", seq), _traj("t2", seq), _traj("t3", seq)]
        props = mine_recurring_tool_sequences(trajs, min_support=3)
        assert len(props) == 1
        p = props[0]
        assert p["signature"] == ("web_search", "deep_research")
        assert p["support"] == 3
        assert p["steps"][0]["params"] == {"query": "x"}  # mined args, not empty
        assert p["name"] == "auto_web_search_deep_research"

    def test_respects_min_support(self):
        seq = [("a", {}), ("b", {})]
        trajs = [_traj("t1", seq), _traj("t2", seq)]  # support 2
        assert mine_recurring_tool_sequences(trajs, min_support=3) == []

    def test_skips_failed_trajectories(self):
        seq = [("web_search", {}), ("deep_research", {})]
        trajs = [_traj("t1", seq), _traj("t2", seq), _traj("t3", seq, outcome="failed")]
        # Only 2 non-failed occurrences → below support 3.
        assert mine_recurring_tool_sequences(trajs, min_support=3) == []

    def test_support_is_distinct_trajectories_not_occurrences(self):
        # Same sequence appears twice in ONE trajectory: support must be 1.
        trajs = [_traj("t1", [("a", {}), ("b", {}), ("a", {}), ("b", {})])]
        assert mine_recurring_tool_sequences(trajs, min_support=2) == []

    def test_ignores_single_tool_repeats(self):
        seq = [("file_system", {}), ("file_system", {})]
        trajs = [_traj(f"t{i}", seq) for i in range(5)]
        assert mine_recurring_tool_sequences(trajs, min_support=3) == []

    def test_ignores_all_meta_windows(self):
        seq = [("replan", {}), ("flag_uncertainty", {})]
        trajs = [_traj(f"t{i}", seq) for i in range(5)]
        assert mine_recurring_tool_sequences(trajs, min_support=3) == []

    def test_subwindow_dedup(self):
        # A recurring triple subsumes its same-support sub-pairs.
        seq = [("a", {}), ("b", {}), ("c", {})]
        trajs = [_traj(f"t{i}", seq) for i in range(4)]
        sigs = [p["signature"] for p in mine_recurring_tool_sequences(trajs, min_support=3, max_proposals=5)]
        assert ("a", "b", "c") in sigs
        assert ("a", "b") not in sigs
        assert ("b", "c") not in sigs

    def test_most_common_args_wins(self):
        trajs = [
            _traj("t1", [("web_search", {"q": "common"}), ("deep_research", {})]),
            _traj("t2", [("web_search", {"q": "common"}), ("deep_research", {})]),
            _traj("t3", [("web_search", {"q": "common"}), ("deep_research", {})]),
            _traj("t4", [("web_search", {"q": "rare"}), ("deep_research", {})]),
        ]
        props = mine_recurring_tool_sequences(trajs, min_support=3)
        assert props[0]["steps"][0]["params"] == {"q": "common"}

    def test_caps_at_max_proposals(self):
        trajs = []
        for pair in (("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")):
            for i in range(3):
                trajs.append(_traj(f"{pair}-{i}", [(pair[0], {}), (pair[1], {})]))
        props = mine_recurring_tool_sequences(trajs, min_support=3, max_proposals=2)
        assert len(props) == 2


class _FakeCtx:
    def __init__(self, base, collector, args=None):
        self.memory_dir = base
        self.sandbox_dir = base
        self.memory_system = None
        self.trajectory_collector = collector
        self.args = args


class _Args:
    def __init__(self, no_trajectories=False):
        self.no_trajectories = no_trajectories


class TestProposeMacrosIntegration:
    def test_proposes_then_requires_approval(self, tmp_path):
        collector = TrajectoryCollector(root=tmp_path / "traj", session_id="s1")
        seq = [("web_search", {"query": "x"}), ("deep_research", {"query": "y"})]
        for i in range(3):
            collector.append(_traj(f"t{i}", seq))
        collector.append(_traj("noise", [("execute", {"command": "ls"})]))

        ctx = _FakeCtx(tmp_path / "mem", collector)
        dreamer = Dreamer(ctx)

        res = dreamer._propose_macros_sync()
        assert res["proposed"] >= 1
        name = res["names"][0]

        reg = _registry_from_context(ctx)
        assert reg.skills[name].status == "proposed"
        # A proposed macro is NOT advertised to the LLM until approved.
        assert name not in {d["function"]["name"] for d in reg.to_tool_definitions()}

        # Idempotent: a second cycle does not re-propose the same signature.
        res2 = dreamer._propose_macros_sync()
        assert name not in res2["names"]

    def test_no_collector_is_noop(self, tmp_path):
        dreamer = Dreamer(_FakeCtx(tmp_path, collector=None))
        assert dreamer._propose_macros_sync() == {"proposed": 0, "names": []}

    def test_too_few_trajectories_is_noop(self, tmp_path):
        collector = TrajectoryCollector(root=tmp_path / "traj", session_id="s1")
        collector.append(_traj("t1", [("a", {}), ("b", {})]))
        dreamer = Dreamer(_FakeCtx(tmp_path / "mem", collector))
        assert dreamer._propose_macros_sync()["proposed"] == 0


class TestFallbackCollector:
    """When the context has no live trajectory_collector, mining falls back to
    a read-only collector at the canonical on-disk root (<memory_dir>/../trajectories)."""

    def test_mines_canonical_root_without_live_collector(self, tmp_path):
        mem = tmp_path / "mem"
        mem.mkdir()
        # Recording writes to <memory_dir>/../trajectories == tmp_path/trajectories.
        coll = TrajectoryCollector(root=tmp_path / "trajectories", session_id="s1")
        seq = [("web_search", {"query": "x"}), ("deep_research", {"query": "y"})]
        for i in range(3):
            coll.append(_traj(f"t{i}", seq))

        ctx = _FakeCtx(mem, collector=None, args=_Args(no_trajectories=False))
        res = Dreamer(ctx)._propose_macros_sync()
        assert res["proposed"] >= 1  # mined from disk despite no live collector

    def test_fallback_respects_no_trajectories_kill_switch(self, tmp_path):
        mem = tmp_path / "mem"
        mem.mkdir()
        coll = TrajectoryCollector(root=tmp_path / "trajectories", session_id="s1")
        seq = [("web_search", {}), ("deep_research", {})]
        for i in range(3):
            coll.append(_traj(f"t{i}", seq))

        ctx = _FakeCtx(mem, collector=None, args=_Args(no_trajectories=True))
        assert Dreamer(ctx)._propose_macros_sync() == {"proposed": 0, "names": []}

    def test_fallback_missing_root_is_noop(self, tmp_path):
        # Nothing on disk → clean no-op even with the fallback active.
        ctx = _FakeCtx(tmp_path / "mem", collector=None, args=_Args())
        assert Dreamer(ctx)._propose_macros_sync()["proposed"] == 0
