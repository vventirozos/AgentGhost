"""Unit tests for the FrontierTracker curiosity system."""

import json
from pathlib import Path

import pytest

from ghost_agent.memory.frontier import FrontierTracker, classify_cluster


class TestClassifyCluster:
    def test_sql_keywords(self):
        assert classify_cluster("Write a SQL query with JOIN and window function") == "sql"
        assert classify_cluster("Use a CTE and GROUP BY to aggregate") == "sql"

    def test_bash_keywords(self):
        assert classify_cluster("Write a bash script using awk and grep") == "bash"
        assert classify_cluster("Use sed to replace lines") == "bash"

    def test_data_analysis_keywords(self):
        assert classify_cluster("Use pandas dataframe to analyze the CSV") == "data_analysis"
        assert classify_cluster("Compute numpy statistics on the dataset") == "data_analysis"

    def test_algo_keywords(self):
        assert classify_cluster("Solve this with dynamic programming on a graph") == "algo"
        assert classify_cluster("Write a recursive tree traversal") == "algo"

    def test_concurrency_keywords(self):
        assert classify_cluster("Use async / thread pool to fetch URLs") == "concurrency"

    def test_regex_parse_keywords(self):
        assert classify_cluster("Build a regex parser for the tokens") == "regex_parse"

    def test_default_python_general(self):
        assert classify_cluster("Print hello world") == "python_general"

    def test_empty_string(self):
        assert classify_cluster("") == "python_general"
        assert classify_cluster(None) == "python_general"


class TestFrontierTrackerBasics:
    def test_initial_state_creates_file(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        assert (tmp_path / "self_play_frontier.json").exists()
        state = json.loads((tmp_path / "self_play_frontier.json").read_text())
        assert state == {"runs": [], "clusters": {}}

    def test_record_run_first_success_new_cluster(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        r = ft.record_run("sql", "challenge A", 1, True, 1500)
        assert r["is_new_cluster"] is True
        assert r["compression_delta"] == 0.0
        assert r["best_length"] == 1500
        assert r["runs"] == 1
        assert r["mastered"] is False

    def test_record_run_compression_improvement(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 1000)
        r2 = ft.record_run("sql", "c2", 1, True, 750)
        # 250 / 1000 = 0.25 improvement
        assert abs(r2["compression_delta"] - 0.25) < 1e-6
        assert r2["best_length"] == 750

    def test_record_run_worse_length_keeps_best(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 1000)
        r2 = ft.record_run("sql", "c2", 1, True, 1500)
        # Delta is negative (regression), best_length stays at 1000
        assert r2["compression_delta"] < 0
        assert r2["best_length"] == 1000

    def test_record_run_failure_marks_negative_delta(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        r = ft.record_run("bash", "c1", 3, False, 0, mistake="used wrong flag")
        assert r["compression_delta"] == -1.0
        assert r["is_new_cluster"] is True

    def test_mastered_flag_not_trivially_set_by_first_try_streak(self, tmp_path):
        # C7: mastery is no longer claimed on a 3-run streak with
        # delta=0 on every run. A brand-new cluster with identical
        # lengths gives no compression signal; declaring mastery after
        # 3 such runs is wildly over-confident. We require a longer
        # streak AND at least one positive compression delta.
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c1", 1, True, 500)
        ft.record_run("algo", "c2", 1, True, 500)
        r3 = ft.record_run("algo", "c3", 1, True, 500)
        assert r3["mastered"] is False

    def test_mastered_flag_requires_streak_and_compression(self, tmp_path):
        # 5 first-try wins with at least one compression improvement.
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c1", 1, True, 1000)
        ft.record_run("algo", "c2", 1, True, 800)  # delta > 0.05 here
        ft.record_run("algo", "c3", 1, True, 800)
        ft.record_run("algo", "c4", 1, True, 800)
        r5 = ft.record_run("algo", "c5", 1, True, 800)
        assert r5["mastered"] is True

    def test_mastered_flag_not_set_with_struggle(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c1", 1, True, 1000)
        ft.record_run("algo", "c2", 2, True, 500)  # struggled
        ft.record_run("algo", "c3", 1, True, 500)
        ft.record_run("algo", "c4", 1, True, 500)
        r5 = ft.record_run("algo", "c5", 1, True, 500)
        assert r5["mastered"] is False

    def test_runs_list_capped(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        for i in range(FrontierTracker.MAX_RUNS + 20):
            ft.record_run("sql", f"c{i}", 1, True, 1000)
        state = json.loads((tmp_path / "self_play_frontier.json").read_text())
        assert len(state["runs"]) == FrontierTracker.MAX_RUNS


class TestBrittleAndSeed:
    def test_brittle_cluster_detected_on_failures(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 3, False, 0, mistake="broken join")
        ft.record_run("sql", "c2", 3, False, 0, mistake="broken join")
        brittle = ft.get_brittle_clusters()
        assert len(brittle) == 1
        assert brittle[0][0] == "sql"

    def test_mastered_cluster_excluded_from_brittle(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        for _ in range(3):
            ft.record_run("algo", "cx", 1, True, 500)
        brittle = ft.get_brittle_clusters()
        assert not any(k == "algo" for k, _ in brittle)

    def test_pick_seed_targets_brittle_cluster(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("bash", "c1", 3, False, 0, mistake="bad awk")
        seed = ft.pick_seed(random_explore_prob=0.0)
        assert seed["mode"] == "frontier"
        assert seed["cluster_key"] == "bash"
        assert "FRONTIER TARGET" in seed["hint"]
        assert "bash" in seed["hint"]

    def test_pick_seed_cold_start_when_empty(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        seed = ft.pick_seed(random_explore_prob=0.0)
        assert seed["mode"] == "cold_start"
        assert seed["cluster_key"] is None
        assert seed["hint"] == ""

    def test_pick_seed_exploration_when_forced(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c", 3, False, 0)
        seed = ft.pick_seed(random_explore_prob=1.0)
        assert seed["mode"] == "exploration"
        assert seed["cluster_key"] is None


class TestAdaptiveCooldown:
    def test_no_runs_returns_base(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        assert ft.adaptive_cooldown(base=3600) == 3600

    def test_compression_progress_halves_cooldown(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 1000)
        ft.record_run("sql", "c2", 1, True, 500)  # 0.5 delta
        assert ft.adaptive_cooldown(base=3600) == 1800

    def test_failure_doubles_cooldown(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c", 3, False, 0)
        assert ft.adaptive_cooldown(base=3600) == 7200

    def test_cooldown_respects_ceiling(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c", 3, False, 0)
        assert ft.adaptive_cooldown(base=5000, ceiling=6000) == 6000

    def test_cooldown_respects_floor(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 1000)
        ft.record_run("sql", "c2", 1, True, 100)  # huge delta
        assert ft.adaptive_cooldown(base=400, floor=600) == 600

    def test_neutral_first_try_pass_slight_reduction(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c", 1, True, 1000)  # new cluster, delta=0
        cooldown = ft.adaptive_cooldown(base=3600)
        # new cluster with delta=0 and attempts_used=1 → 0.75 * base
        assert cooldown == 2700


class TestPersistence:
    def test_state_survives_reopen(self, tmp_path):
        ft1 = FrontierTracker(tmp_path)
        ft1.record_run("sql", "c1", 1, True, 1000)
        ft2 = FrontierTracker(tmp_path)
        stats = ft2.get_cluster_stats("sql")
        assert stats["runs"] == 1
        assert stats["best_length"] == 1000


class TestDedupPreservesSaturationSignal:
    """Deterministic templates (shop.db GROUP BY, data.csv aggregation)
    hash to the same challenge text on every re-roll. The dedup path in
    `record_run` must protect mastery counters from inflation BUT must
    still append to `recent_outcomes` — otherwise saturation detection
    and the brittle-pool decay guard can never observe that the agent is
    now acing a template it previously struggled on, and the cluster
    stays fossilised as brittle forever.

    Root-cause incident: 5-cycle post-patch log showed sql picked as
    frontier every cycle because a single 2026-04-21T09:06 struggled-
    then-won run pinned sql in the brittle pool. The subsequent 5
    first-try-clean sql runs hit dedup and never updated state, so
    `recent_outcomes[-1]` stayed the DD entry — decay guard could not
    fire."""

    CHALLENGE = "Deterministic template challenge text (always identical)."

    def test_mastery_counters_stay_frozen_on_duplicate(self, tmp_path):
        """Protected counters (runs, total_first_try_wins, best_length)
        MUST NOT advance on a duplicate challenge — that was the original
        and still-valid reason for dedup."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)
        stats_before = ft.get_cluster_stats("sql")
        runs_before = stats_before["runs"]
        ftw_before = stats_before.get("total_first_try_wins", 0)
        best_before = stats_before["best_length"]

        result = ft.record_run("sql", self.CHALLENGE, 1, True, 500)

        stats_after = ft.get_cluster_stats("sql")
        assert result.get("duplicate") is True
        assert stats_after["runs"] == runs_before, "runs counter must not advance on duplicate"
        assert stats_after.get("total_first_try_wins", 0) == ftw_before, \
            "total_first_try_wins must not advance on duplicate (mastery inflation guard)"
        assert stats_after["best_length"] == best_before, \
            "best_length must not update on duplicate (compression signal is noise)"

    def test_recent_outcomes_appends_on_duplicate(self, tmp_path):
        """The saturation signal lives in recent_outcomes. Dedup MUST
        still append to this list — that's what unblocks the decay guard
        for deterministic templates."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)
        outcomes_before = ft.get_cluster_stats("sql")["recent_outcomes"]
        assert len(outcomes_before) == 1

        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)

        outcomes_after = ft.get_cluster_stats("sql")["recent_outcomes"]
        assert len(outcomes_after) == 2, \
            "duplicate run must still append to recent_outcomes for saturation detection"
        assert outcomes_after[-1].get("duplicate") is True
        assert outcomes_after[-1]["passed"] is True
        assert outcomes_after[-1]["attempts_used"] == 1
        assert outcomes_after[-1]["delta"] == 0.0, \
            "duplicate outcomes carry zero delta (no real compression signal on re-roll)"

    def test_saturation_triggers_from_dedup_entries(self, tmp_path):
        """End-to-end: a struggled-then-won run followed by
        SATURATION_WINDOW (2) first-try duplicate wins must saturate the
        cluster. Pre-fix, the duplicates never appeared in
        recent_outcomes, so saturation could not see the pattern."""
        ft = FrontierTracker(tmp_path)
        # First: a struggled-then-won run (attempts=2) — pins cluster as
        # brittle and inhabits the saturation window.
        ft.record_run("sql", self.CHALLENGE, 2, True, 1000)
        assert not ft._cluster_is_saturated(ft.get_cluster_stats("sql"))

        # Two duplicate first-try wins. With the fix these land in
        # recent_outcomes and slide the struggled entry out of the
        # saturation window.
        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)
        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)

        stats = ft.get_cluster_stats("sql")
        assert ft._cluster_is_saturated(stats), \
            "after 2 clean first-try duplicates, cluster must saturate"

    def test_decay_guard_skips_brittle_after_duplicate_clean_win(self, tmp_path):
        """The duplicate update must let the brittle-pool decay guard
        see the cluster as recovered. A single struggled-then-won run
        followed by one duplicate clean win must NOT keep the cluster
        in the brittle pool."""
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", self.CHALLENGE, 2, True, 1000)  # struggle
        brittle = ft.get_brittle_clusters()
        assert any(k == "sql" for k, _ in brittle), \
            "struggled-then-won run should place sql in brittle pool"

        ft.record_run("sql", self.CHALLENGE, 1, True, 1000)  # duplicate clean win

        brittle = ft.get_brittle_clusters()
        assert not any(k == "sql" for k, _ in brittle), \
            "decay guard must skip sql once its most recent (duplicate) outcome is a clean first-try pass"

    def test_state_persists_duplicate_append(self, tmp_path):
        """The dedup path must call _save(state) — otherwise the change
        survives only in-memory and vanishes on process restart (the
        exact failure mode the 11:56 log-eval exposed)."""
        ft1 = FrontierTracker(tmp_path)
        ft1.record_run("sql", self.CHALLENGE, 1, True, 1000)
        ft1.record_run("sql", self.CHALLENGE, 1, True, 1000)

        ft2 = FrontierTracker(tmp_path)
        outcomes = ft2.get_cluster_stats("sql")["recent_outcomes"]
        assert len(outcomes) == 2, "duplicate append must be persisted to disk"
        assert outcomes[-1].get("duplicate") is True


class TestPartialClusterSchemaBackfill:
    """Regression tests for the live 2026-07-08 failure: a cluster entry
    created with a PARTIAL shape (note_reflection_failure writes into a
    bare dict) made record_run's full-defaults setdefault no-op, so
    cluster["runs"] raised KeyError('runs') and NO self-play run was ever
    recorded for that cluster ("Frontier record_run failed: 'runs'")."""

    def test_record_run_survives_reflection_created_cluster(self, tmp_path):
        tracker = FrontierTracker(tmp_path)
        # Reflection creates the cluster first — exactly the live ordering.
        tracker.note_reflection_failure("python_general", diagnosis="failed turn")
        result = tracker.record_run(
            "python_general", "challenge text", 1, True, 100,
            solution_source="print('x')",
        )
        assert result["is_new_cluster"] is True  # no *runs* recorded before
        state = json.loads((tmp_path / "self_play_frontier.json").read_text())
        cluster = state["clusters"]["python_general"]
        assert cluster["runs"] == 1
        # The reflection data survived the back-fill untouched.
        assert len(cluster["reflection_failures"]) == 1

    def test_record_run_backfills_legacy_partial_state_file(self, tmp_path):
        # Simulate the exact on-disk shape observed in prod: a cluster with
        # ONLY reflection_failures, predating the run-stats schema.
        path = tmp_path / "self_play_frontier.json"
        path.write_text(json.dumps({
            "runs": [],
            "clusters": {
                "python_general": {
                    "reflection_failures": [{"ts": "2026-07-08T10:00:00", "diagnosis": "x"}],
                },
            },
        }))
        tracker = FrontierTracker(tmp_path)
        result = tracker.record_run("python_general", "c", 1, False, 80)
        assert result["compression_delta"] == -1.0
        state = json.loads(path.read_text())
        cluster = state["clusters"]["python_general"]
        assert cluster["runs"] == 1
        assert cluster["recent_outcomes"][-1]["passed"] is False

    def test_reflection_failure_now_creates_full_schema(self, tmp_path):
        tracker = FrontierTracker(tmp_path)
        tracker.note_reflection_failure("sql", diagnosis="bad join")
        state = json.loads((tmp_path / "self_play_frontier.json").read_text())
        cluster = state["clusters"]["sql"]
        for key in ("runs", "recent_outcomes", "recent_hashes", "mastered",
                    "total_first_try_wins", "unlocked_tier_index"):
            assert key in cluster, f"missing {key}"
