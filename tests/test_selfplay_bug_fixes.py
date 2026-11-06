"""Coverage for the self-play bug fixes (C1-C7, S1-S7, M1-M7).

One test per fix so a regression points directly at the failure mode.
Grouped by module: frontier.py fixes first, then isolation.py, then the
biological-hook wiring in agent.py. §4NM: the dream.py fixes (S1, S4-S7,
C1, C2, C5, C6, M2, M3, M5) went with synthetic self-play.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.memory.frontier import (
    DIFFICULTY_TIERS,
    TIER_UNLOCK_THRESHOLD,
    FrontierTracker,
)


# ------------------------------------------------------------------ #
# FrontierTracker fixes (C7, S2, S3, M4, M6, M7)                     #
# ------------------------------------------------------------------ #


class TestMasteryThreshold_C7:
    def test_three_run_streak_with_delta_zero_does_not_master(self, tmp_path):
        # A brand-new cluster always starts with delta=0 (no prior best).
        # Three first-try wins at identical length give zero compression
        # signal — mastery must NOT be claimed.
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c1", 1, True, 500)
        ft.record_run("algo", "c2", 1, True, 500)
        r3 = ft.record_run("algo", "c3", 1, True, 500)
        assert r3["mastered"] is False

    def test_five_runs_with_real_compression_masters(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        # Run 2 gives a big compression delta (0.5) — enough to satisfy
        # the "any real progress" clause in the streak.
        ft.record_run("algo", "c1", 1, True, 1000)
        ft.record_run("algo", "c2", 1, True, 500)
        ft.record_run("algo", "c3", 1, True, 500)
        ft.record_run("algo", "c4", 1, True, 500)
        r5 = ft.record_run("algo", "c5", 1, True, 500)
        assert r5["mastered"] is True

    def test_struggle_breaks_streak(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        ft.record_run("algo", "c1", 1, True, 1000)
        ft.record_run("algo", "c2", 2, True, 500)  # struggled
        ft.record_run("algo", "c3", 1, True, 500)
        ft.record_run("algo", "c4", 1, True, 500)
        r5 = ft.record_run("algo", "c5", 1, True, 500)
        assert r5["mastered"] is False


class TestMonotonicTier_S2:
    def test_tier_does_not_regress_when_recent_outcomes_rotate(self, tmp_path):
        # Old get_difficulty_tier derived tier from recent_outcomes,
        # which is capped at 10. If tier was unlocked by 3 first-try
        # wins and then 10 non-first-try runs scrolled them out, tier
        # would silently downgrade. The cumulative counter + stored
        # unlocked_tier_index keeps tier monotonic.
        ft = FrontierTracker(tmp_path)
        # Achieve enough wins to unlock the next tier.
        for i in range(TIER_UNLOCK_THRESHOLD):
            ft.record_run("sql", f"unique-early-{i}", 1, True, 500)
        assert ft.get_difficulty_tier("sql") == DIFFICULTY_TIERS[1]

        # Fill recent_outcomes with 10 struggled-wins so the first-try
        # entries scroll out of the rolling buffer.
        for i in range(15):
            ft.record_run("sql", f"struggle-{i}", 2, True, 500)

        # Without monotonicity this would flip back to basic.
        assert ft.get_difficulty_tier("sql") == DIFFICULTY_TIERS[1]


class TestPickSeedWeighted_S3:
    def test_pick_seed_samples_non_top_cluster_sometimes(self, tmp_path):
        import random as _random
        ft = FrontierTracker(tmp_path)
        # Three brittle clusters with different failure counts so the
        # scores differ meaningfully.
        for cluster, fails in [("sql", 2), ("bash", 2), ("algo", 1)]:
            for i in range(fails):
                ft.record_run(cluster, f"{cluster}-fail-{i}", 3, False, 0)

        # Seed the RNG deterministically and exercise the weighted
        # branch many times; at least two clusters must show up.
        _random.seed(1234)
        seen = set()
        for _ in range(200):
            seed = ft.pick_seed(random_explore_prob=0.0)
            if seed["cluster_key"]:
                seen.add(seed["cluster_key"])
        assert len(seen) >= 2, seen


class TestAdaptiveCooldownPerCluster_M4:
    def test_cooldown_respects_cluster_key_over_global_tail(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        # Cluster A had a nice learning streak; cluster B had a
        # failure that is the most-recent run globally.
        ft.record_run("sql", "a1", 1, True, 1000)
        ft.record_run("sql", "a2", 1, True, 500)  # delta=0.5 → "learning"
        ft.record_run("bash", "b1", 3, False, 0)

        # Use a small floor so the halve/double arithmetic is visible.
        # Without cluster_key the last run is the bash failure → long
        # cooldown (base*2 capped by ceiling).
        cd_global = ft.adaptive_cooldown(base=2000, floor=50, ceiling=8000)
        assert cd_global == 4000  # doubled

        # With cluster_key="sql", the last SQL run's positive delta
        # halves the cooldown (clamped by floor).
        cd_sql = ft.adaptive_cooldown(base=2000, floor=50, ceiling=8000, cluster_key="sql")
        assert cd_sql == 1000  # halved
        assert cd_sql < cd_global


class TestCrossProcessLock_M6:
    def test_lock_file_created_and_write_still_atomic(self, tmp_path):
        # The advisory lock uses a sibling `.lock` file. We can't
        # meaningfully test a real cross-process race in-process, but
        # we can verify that a normal write still succeeds with the
        # lock wrapper in place.
        ft = FrontierTracker(tmp_path)
        ft.record_run("sql", "c1", 1, True, 500)
        assert ft.file_path.exists()
        state = json.loads(ft.file_path.read_text())
        assert state["runs"][0]["cluster_key"] == "sql"

    def test_lock_handles_missing_fcntl(self, tmp_path, monkeypatch):
        # On Windows (no fcntl) the lock degrades to a no-op. Simulate
        # by pretending fcntl is unavailable mid-test.
        import ghost_agent.memory.frontier as fmod
        monkeypatch.setattr(fmod, "_HAS_FCNTL", False)
        ft = fmod.FrontierTracker(tmp_path)
        r = ft.record_run("sql", "c1", 1, True, 500)
        assert r["duplicate"] is False


class TestChallengeDedup_M7:
    def test_duplicate_challenge_does_not_inflate_runs(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        r1 = ft.record_run("sql", "same prompt", 1, True, 500)
        r2 = ft.record_run("sql", "same prompt", 1, True, 500)
        assert r1["duplicate"] is False
        assert r2["duplicate"] is True
        # The cluster should still show only 1 real run.
        state = json.loads(ft.file_path.read_text())
        assert state["clusters"]["sql"]["runs"] == 1

    def test_different_challenges_both_record(self, tmp_path):
        ft = FrontierTracker(tmp_path)
        r1 = ft.record_run("sql", "prompt A", 1, True, 500)
        r2 = ft.record_run("sql", "prompt B", 1, True, 500)
        assert r1["duplicate"] is False
        assert r2["duplicate"] is False


class TestReadOnlyVectorMemoryWhitelist_M1:
    """The façade is no longer a method-local class (§4CL S1), so the
    whitelist is EXERCISED instead of grepped — the old version asserted
    two strings were present in ``synthetic_self_play``'s source, which
    passed just as happily when the class had been gutted."""

    def test_whitelisted_read_passes_through(self):
        from unittest.mock import MagicMock
        from ghost_agent.core.isolation import ReadOnlyVectorMemory

        real = MagicMock()
        real.search.return_value = ["hit"]
        ro = ReadOnlyVectorMemory(real)
        assert ro.search("q") == ["hit"]
        # …and the read is de-fanged: it must not bump retrieval stats.
        assert real.search.call_args.kwargs["record_retrievals"] is False

    def test_mutator_is_a_no_op_not_a_passthrough(self):
        from unittest.mock import MagicMock
        from ghost_agent.core.isolation import ReadOnlyVectorMemory

        real = MagicMock()
        ro = ReadOnlyVectorMemory(real)
        ro.add({"text": "x"})
        ro.delete("x")
        real.add.assert_not_called()
        real.delete.assert_not_called()

    def test_unknown_attribute_raises_instead_of_forwarding(self):
        import pytest as _pytest
        from unittest.mock import MagicMock
        from ghost_agent.core.isolation import ReadOnlyVectorMemory

        real = MagicMock()
        ro = ReadOnlyVectorMemory(real)
        with _pytest.raises(AttributeError, match="passthrough whitelist"):
            ro.some_future_mutator


class TestIsolatedContextNulling_C4:
    """The C4 secondary modules moved into the shared detach inventory
    (`core/isolation.ISOLATION_NULLED_ATTRS`, §4CL S1). The pin moved with
    them and stopped being a source-text assertion on the way: the
    EXECUTED end-to-end version — run the real solve loop, capture the
    isolate the mocked GhostAgent was constructed with, and assert the
    whole inventory is None on it — lives in
    ``tests/test_isolation_replay.py``. What stays here is the membership
    contract those attributes must keep."""

    def test_secondary_modules_are_in_the_shared_inventory(self):
        from ghost_agent.core.isolation import ISOLATION_NULLED_ATTRS

        # §4NN: uncertainty_tracker / hypothesis_tester were removed
        for attr in ("verifier", "frontier_tracker"):
            assert attr in ISOLATION_NULLED_ATTRS, (
                f"secondary module {attr} dropped from the detach inventory")

    def test_nulling_is_applied_by_the_shared_helper(self):
        """`null_production_state` is what self-play calls; it must
        actually clear a live handle, not merely enumerate names."""
        from ghost_agent.core.isolation import (
            ISOLATION_NULLED_ATTRS, null_production_state,
        )

        class _Ctx:
            pass

        ctx = _Ctx()
        for attr in ISOLATION_NULLED_ATTRS:
            setattr(ctx, attr, object())
        assert set(null_production_state(ctx)) == set(ISOLATION_NULLED_ATTRS)
        assert [a for a in ISOLATION_NULLED_ATTRS
                if getattr(ctx, a) is not None] == []
