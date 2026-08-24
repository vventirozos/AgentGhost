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


def _traj(tid, seq, outcome="passed"):
    """seq is a list of (tool_name, args_dict). Default outcome is
    "passed": since 2026-07-26 the miner counts only VALIDATED successes
    as support — unknown/never-verified turns no longer underwrite a
    "proven" macro."""
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
        # §4CS: the signature is the macro IDENTITY — tools AND the modes
        # each step fixes. Neither tool declares an enum param, so both
        # modes are empty here. Recomputed from the shared definition
        # rather than restated, so the two cannot drift apart.
        from ghost_agent.tools.composed_skills import macro_identity
        assert p["signature"] == macro_identity(
            (st["tool"], st["params"]) for st in p["steps"])
        assert tuple(t for t, _m in p["signature"]) == (
            "web_search", "deep_research")
        assert p["support"] == 3
        # §4CS: a mined arg becomes a runtime SLOT, never a frozen literal.
        # Assert against the OBSERVED value, recomputed here, rather than
        # restating the rule: whatever `web_search` was actually called
        # with must not survive into the template.
        observed = seq[0][1]["query"]
        assert p["steps"][0]["params"] == {"query": "$query"}
        assert observed not in p["steps"][0]["params"].values()
        # The two steps' `query` values differed, so they are two inputs.
        assert p["steps"][1]["params"] == {"query": "$query_2"}
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
        # ⚠ REAL tools: since §4CS a step whose tool is absent from the
        # registry is refused outright (its required params and its modes
        # cannot be checked), so placeholder names never reach the
        # sub-window logic this test is about.
        seq = [("web_search", {"query": "q"}),
               ("browser", {"operation": "navigate", "url": "u"}),
               ("file_system", {"operation": "write", "path": "/p",
                                "content": "c"})]
        trajs = [_traj(f"t{i}", seq) for i in range(4)]
        sigs = [tuple(t for t, _m in p["signature"])
                for p in mine_recurring_tool_sequences(
                    trajs, min_support=3, max_proposals=5)]
        assert ("web_search", "browser", "file_system") in sigs
        assert ("web_search", "browser") not in sigs
        assert ("browser", "file_system") not in sigs

    def test_majority_arg_no_longer_wins_it_becomes_a_slot(self):
        """§4CS retires "most common args wins".

        The old miner froze the MAJORITY argument dict into the template,
        so a macro replayed one past call's payload on every run. A value
        that varies across observations is now a runtime slot, and the
        majority value must not appear anywhere in the minted template.
        """
        values = ["common", "common", "common", "rare"]
        trajs = [
            _traj(f"t{i}", [("web_search", {"query": v}),
                            ("deep_research", {"query": v})])
            for i, v in enumerate(values)
        ]
        props = mine_recurring_tool_sequences(trajs, min_support=3)
        assert props, "a 4-support pair must still be mined"
        templates = [st["params"] for st in props[0]["steps"]]
        assert templates[0] == {"query": "$query"}
        # Identity check against the recomputed observation set: no value
        # the tool was ever called with survives as a literal.
        frozen = {v for t in templates for v in t.values()}
        assert frozen & set(values) == set(), frozen
        # Both steps always carried the SAME value, so they share ONE slot.
        assert templates[1] == {"query": "$query"}

    def test_caps_at_max_proposals(self):
        # Four distinct REAL pairs, each of which mints (see the note in
        # test_subwindow_dedup for why placeholder tool names cannot).
        pairs = (
            [("web_search", {"query": "q"}),
             ("browser", {"operation": "navigate", "url": "u"})],
            [("file_system", {"operation": "read", "path": "/p"}),
             ("manage_services", {"action": "start", "name": "s",
                                  "command": "c"})],
            # ⚠ NOT introspect/workspace: §4CS review round 2 refuses a
            # step that takes NO runtime input (a call fully determined at
            # mint time is a replay), and refuses any sequence containing a
            # meta tool. A read-only summary bundle is the measured cost of
            # not keeping a list of dangerous verbs to leak.
            [("file_system", {"operation": "replace", "path": "/p",
                              "content": "c"}),
             ("manage_services", {"action": "restart", "name": "s"})],
            [("browser", {"operation": "screenshot", "out_path": "/o.png"}),
             ("vision_analysis", {"action": "describe_picture",
                                  "target": "/o.png"})],
        )
        trajs = []
        for n, seq in enumerate(pairs):
            for i in range(3):
                trajs.append(_traj(f"p{n}-{i}", seq))
        assert len(mine_recurring_tool_sequences(
            trajs, min_support=3, max_proposals=10)) == 4
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
