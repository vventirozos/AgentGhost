"""Tests for the reflection → self-play curriculum loop (proposal item #7).

The FrontierTracker now records clusters the reflection phase flagged
as failing in real interactive turns, and pick_seed can drill them.
"""

from pathlib import Path

from ghost_agent.memory.frontier import FrontierTracker


def test_note_reflection_failure_and_hot_clusters(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    ft.note_reflection_failure("sql", diagnosis="missed a join condition")
    ft.note_reflection_failure("sql")
    ft.note_reflection_failure("bash")
    hot = ft.get_reflection_hot_clusters()
    assert hot[0] == ("sql", 2)
    assert ("bash", 1) in hot


def test_note_reflection_failure_ignores_empty_cluster(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    ft.note_reflection_failure("")
    ft.note_reflection_failure("   ")
    assert ft.get_reflection_hot_clusters() == []


def test_note_reflection_failure_bounded(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    for i in range(30):
        ft.note_reflection_failure("python_general", diagnosis=f"d{i}")
    # Recorded count is bounded at 20 per cluster.
    hot = ft.get_reflection_hot_clusters()
    assert hot[0][0] == "python_general"
    assert hot[0][1] == 20


def test_pick_seed_targets_reflection_hot_cluster(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    ft.note_reflection_failure("sql")
    ft.note_reflection_failure("sql")
    # Force the reflection branch: no exploration, always reflection.
    pick = ft.pick_seed(random_explore_prob=0.0, reflection_priority_prob=1.0)
    assert pick["mode"] == "frontier"
    assert pick["cluster_key"] == "sql"
    assert pick.get("source") == "reflection"
    assert "reflection-driven" in pick["hint"]


def test_pick_seed_skips_reflection_when_probability_zero(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    ft.note_reflection_failure("sql")
    # reflection_priority_prob=0 → never take the reflection branch.
    pick = ft.pick_seed(random_explore_prob=0.0, reflection_priority_prob=0.0)
    assert pick.get("source") != "reflection"


def test_pick_seed_no_reflection_clusters_falls_through(tmp_path: Path):
    ft = FrontierTracker(tmp_path)
    # No reflection failures recorded → reflection branch finds nothing.
    pick = ft.pick_seed(random_explore_prob=0.0, reflection_priority_prob=1.0)
    assert pick.get("source") != "reflection"
    assert pick["mode"] in ("frontier", "exploration", "cold_start")
