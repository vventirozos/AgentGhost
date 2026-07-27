"""Self-play curriculum fixes from the 2026-07-27 overnight log eval.

Four defects, one cluster:
  1. Mastery was unreachable at the length floor — `description_length`
     is a tool-invocation count with a structural floor of ~4, so the C7
     `any(delta > 0.05)` requirement could never be met once best_length
     hit the floor (concurrency 6/6 first-try, sql 6/6, regex_parse 5/5
     all pinned unmastered → endlessly re-picked).
  2. Template draws were uniform — 82% of 154 recorded runs targeted the
     two most-practiced clusters while web_automation had 1 lifetime run.
  3. The forward-fed diversity window was 5 heads while the tracker
     retains 12 — the overnight run generated 0.91/0.93-overlap
     duplicates of a theme just outside the slice.
  4. A 3/3 generation rejection forfeited the whole idle slot (~2h).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import challenge_templates as ct
from ghost_agent.memory.frontier import FrontierTracker

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


class TestMasteryFloorWaiver:
    def test_floor_cluster_masters_on_streak_alone(self, tmp_path):
        # best_length at the structural floor: delta > 0.05 is
        # arithmetically unreachable, so the streak decides.
        ft = FrontierTracker(tmp_path)
        for i in range(4):
            ft.record_run("concurrency", f"c{i}", 1, True, 4)
        r5 = ft.record_run("concurrency", "c4", 1, True, 4)
        assert r5["mastered"] is True

    def test_floor_waiver_still_requires_full_first_try_streak(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("concurrency", "c0", 1, True, 4)
        ft.record_run("concurrency", "c1", 2, True, 4)  # struggled
        for i in range(2, 5):
            ft.record_run("concurrency", f"c{i}", 1, True, 4)
        stats = ft.get_cluster_stats("concurrency")
        assert stats["mastered"] is False

    def test_above_floor_cluster_still_requires_progress(self, tmp_path):
        # Above the floor the C7 rule is unchanged: a flat delta=0
        # streak must NOT self-master.
        ft = FrontierTracker(tmp_path)
        for i in range(5):
            r = ft.record_run("algo", f"c{i}", 1, True, 500)
        assert r["mastered"] is False

    def test_floor_boundary(self, tmp_path):
        # best_length == MASTERY_LENGTH_FLOOR qualifies; floor+1 does not.
        ft = FrontierTracker(tmp_path)
        floor = FrontierTracker.MASTERY_LENGTH_FLOOR
        for i in range(5):
            r_at = ft.record_run("sql", f"a{i}", 1, True, floor)
        assert r_at["mastered"] is True
        for i in range(5):
            r_above = ft.record_run("bash", f"b{i}", 1, True, floor + 1)
        assert r_above["mastered"] is False


class TestClusterRunWeights:
    def test_inverse_frequency_and_unseen_top_weight(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        for i in range(9):
            ft.record_run("data_analysis", f"c{i}", 1, True, 400 - i)
        w = ft.cluster_run_weights(["data_analysis", "web_automation"])
        assert w["web_automation"] == 1.0        # unseen → 0 runs
        assert abs(w["data_analysis"] - 0.1) < 1e-9  # 1/(1+9)
        assert w["web_automation"] > w["data_analysis"]

    def test_empty_candidates(self, tmp_path):
        assert FrontierTracker(tmp_path).cluster_run_weights([]) == {}


class TestLeastPracticedClusters:
    def test_orders_ascending_and_includes_unseen(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        for i in range(3):
            ft.record_run("data_analysis", f"c{i}", 1, True, 400)
        ft.record_run("algo", "a0", 1, True, 400)
        out = ft.least_practiced_clusters(
            limit=3, extra_candidates=["web_automation"])
        assert out[0] == ("web_automation", 0)
        assert out[1] == ("algo", 1)
        assert out[2] == ("data_analysis", 3)

    def test_mastered_clusters_excluded(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        for i in range(5):
            ft.record_run("concurrency", f"c{i}", 1, True, 4)  # masters
        ft.record_run("algo", "a0", 1, True, 400)
        keys = [k for k, _ in ft.least_practiced_clusters(limit=10)]
        assert "concurrency" not in keys
        assert "algo" in keys


class TestWeightedTemplatePick:
    def setup_method(self):
        ct.reset_template_history()

    def teardown_method(self):
        ct.reset_template_history()

    def test_single_positive_weight_always_wins(self):
        # All other clusters at 0 weight → the positive one must be
        # drawn every time (random.choices with a single live weight).
        target = "algo"
        weights = {k: (1.0 if k == target else 0.0) for k in ct.TEMPLATES}
        for _ in range(10):
            ct.reset_template_history()
            result = ct.pick_random_template(cluster_weights=weights)
            assert result is not None
            assert ct._LAST_TEMPLATE_KEY == target

    def test_all_nonpositive_weights_fall_back_to_uniform(self):
        weights = {k: 0.0 for k in ct.TEMPLATES}
        assert ct.pick_random_template(cluster_weights=weights) is not None

    def test_weights_compose_with_exclusion(self):
        # The excluded cluster must not be drawn even at top weight.
        target = "sql"
        weights = {k: (1.0 if k in (target, "bash") else 0.0)
                   for k in ct.TEMPLATES}
        for _ in range(10):
            ct.reset_template_history()
            result = ct.pick_random_template(
                exclude_clusters=[target], cluster_weights=weights)
            assert result is not None
            assert ct._LAST_TEMPLATE_KEY != target

    def test_no_weights_preserves_legacy_uniform(self):
        assert ct.pick_random_template() is not None


class TestDreamWiringPins:
    """Source pins for the dream.py side of the fixes (the synthetic
    self-play orchestrator is not unit-instantiable)."""

    def _src(self):
        return (SRC_DIR / "core" / "dream.py").read_text()

    def test_diversity_window_feeds_full_retained_window(self):
        src = self._src()
        assert ("recent_generated_challenges(\n"
                "                    limit=frontier_tracker.RECENT_CHALLENGE_KEEP)"
                in src)
        assert "recent_generated_challenges(limit=5)" not in src

    def test_coverage_target_block_present_and_seed_guarded(self):
        src = self._src()
        assert "### COVERAGE TARGET (curriculum balance)" in src
        assert 'not seed.get("cluster_key")' in src
        assert "least_practiced_clusters" in src

    def test_template_picks_pass_cluster_weights(self):
        src = self._src()
        assert src.count("cluster_weights=_template_weights()") >= 3

    def test_rejection_fallback_replaces_forfeit(self):
        src = self._src()
        assert '_tpl_source = "rejection_fallback"' in src
        # The forfeit return survives only as the templates-unavailable
        # branch inside the fallback.
        assert "falling back to a deterministic template" in src
