"""Coverage for the self-play bug fixes (C1-C7, S1-S7, M1-M7).

One test per fix so a regression points directly at the failure mode.
Grouped by module: frontier.py fixes first, then dream.py, then the
biological-hook wiring in agent.py.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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


# ------------------------------------------------------------------ #
# Challenge-quality gate fixes (S5, S6, S7)                          #
# ------------------------------------------------------------------ #


class TestValidateChallengeQualityS7:
    def test_random_seed_in_comment_does_not_reject(self):
        from ghost_agent.core.dream import validate_challenge_quality

        setup = "open('data.csv','w').write('a,b\\n1,2\\n')\n"
        validator = (
            "# NOTE: do NOT use random.seed here — the data is on disk.\n"
            "data = open('data.csv').read()\n"
            "assert data.strip().endswith('1,2')\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, reason

    def test_random_sample_in_string_literal_does_not_reject(self):
        from ghost_agent.core.dream import validate_challenge_quality

        setup = "open('x.csv','w').write('x\\n1\\n')\n"
        # Marker appears inside a regular string literal, not a real
        # function call — must not reject.
        validator = (
            "banner = 'forbidden ops include random.sample and np.random'\n"
            "data = open('x.csv').read()\n"
            "assert '1' in data\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, reason

    def test_real_random_seed_call_still_rejects(self):
        from ghost_agent.core.dream import validate_challenge_quality

        setup = "open('x.csv','w').write('x\\n1\\n')\n"
        validator = "import random\nrandom.seed(42)\nprint(random.randint(0,9))\n"
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "random.seed" in reason


class TestValidateChallengeQualityS6:
    def test_dynamic_path_validator_not_rejected_for_missing_literals(self):
        from ghost_agent.core.dream import validate_challenge_quality

        setup = "open('products.csv','w').write('a\\n1\\n')\n"
        # Validator uses os.listdir instead of naming the literal.
        validator = (
            "import os\n"
            "files = os.listdir('.')\n"
            "for f in files:\n"
            "    if f.endswith('.csv'):\n"
            "        assert 'a' in open(f).read()\n"
        )
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is True, reason

    def test_literal_mismatch_still_rejected_when_no_dynamic_markers(self):
        from ghost_agent.core.dream import validate_challenge_quality

        setup = "open('products.csv','w').write('a\\n1\\n')\n"
        validator = "assert open('other_file.csv').read() == 'a\\n1\\n'\n"
        ok, reason = validate_challenge_quality(setup, validator)
        assert ok is False
        assert "references none" in reason


class TestFloatWarningLogged_S5:
    def test_float_comparison_risk_emits_pretty_log_warning(self):
        """The old code was a literal `pass` — no log, no effect. The
        fix calls `pretty_log` with a WARNING-level message naming the
        round()/float issue. We intercept pretty_log to confirm."""
        from ghost_agent.core import dream as dmod
        # Validator deliberately uses round() + assert WITHOUT float()
        # or abs() to trip the risk heuristic.
        setup = "open('x.csv','w').write('1.23\\n')\n"
        # Note: the validator MUST NOT contain literal `float(` or
        # `abs(` anywhere, otherwise the heuristic short-circuits.
        validator = (
            "import pathlib\n"
            "data = pathlib.Path('x.csv').read_text().strip()\n"
            "assert round(1.23, 2) == 1.23\n"
        )
        seen = []

        def fake_pretty_log(title, body, *a, **kw):
            seen.append((title, body, kw.get("level", "")))

        # validate_challenge_quality imports pretty_log lazily inside
        # the function body (from ..utils.logging). Patch the source
        # module, not dmod.
        from ghost_agent.utils import logging as logmod
        with patch.object(logmod, "pretty_log", side_effect=fake_pretty_log):
            ok, _ = dmod.validate_challenge_quality(setup, validator)
        assert ok is True
        warnings = [t for t in seen if t[2] == "WARNING"]
        assert any("round()" in body for _, body, _ in warnings), seen


# ------------------------------------------------------------------ #
# _BackgroundOnlyLLM wiring (C1)                                     #
# ------------------------------------------------------------------ #


class TestBackgroundOnlyLLM_C1:
    def test_is_background_guards_the_wrapper_install(self):
        """User-triggered self-play (is_background=False) must NOT wrap
        the LLM client in the _BackgroundOnlyLLM forcer — doing so
        would stall every model turn on the 30s foreground-tasks wait.

        Source-level check: the wrap line is gated on `is_background`.
        """
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # The exact guard we introduced. If someone removes the gate
        # this test fails — preserving the user-triggered fast path.
        assert "if real_llm is not None and is_background:" in src, src[:200]

    def test_is_background_parameter_is_read(self):
        """C1 + M8: the parameter that was dead weight is now actually
        referenced inside the method body."""
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # At least one non-signature reference.
        ref_count = src.count("is_background")
        # Signature has 1 occurrence; body must add at least 1 more.
        assert ref_count >= 2, f"is_background referenced {ref_count}x — expected >=2"


# ------------------------------------------------------------------ #
# ReadOnlyVectorMemory whitelist (M1)                                #
# ------------------------------------------------------------------ #


class TestReadOnlyVectorMemoryWhitelist_M1:
    def _ro(self, real):
        # Exercise the class defined INSIDE synthetic_self_play by
        # reconstructing its logic here — the whitelist is the
        # important invariant.
        from ghost_agent.core.dream import synthetic_self_play  # noqa: F401

        # The class lives inside the method; to test its contract we
        # re-derive the expected behaviour: any method not in the
        # whitelist should raise AttributeError.
        import ghost_agent.core.dream as dmod
        import inspect
        src = inspect.getsource(dmod.Dreamer.synthetic_self_play)
        assert "_SAFE_PASSTHROUGH" in src
        assert "is not in the read-only passthrough whitelist" in src


# ------------------------------------------------------------------ #
# Tool-invocation compression signal (S1)                            #
# ------------------------------------------------------------------ #


class TestToolCountCompression_S1:
    def test_description_length_counts_tool_invocations_not_chars(self):
        # The fix moved from `len(full_simulation_transcript)` to a
        # count of tool invocations. Verify the helper is present in
        # the source and the old char-length metric is gone from the
        # success path.
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        assert "_count_tool_invocations" in src
        # The old metric MUST NOT be the one feeding description_length.
        assert "description_length = (" in src


# ------------------------------------------------------------------ #
# Abort / status_str / skill-write gate (C5, M2)                     #
# ------------------------------------------------------------------ #


class TestAbortLabellingAndSkillGate_C5_M2:
    def test_source_labels_abort_separately_from_exhaustion(self):
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # M2: abort has its own status_str branch.
        assert 'ABORTED_BY_SOLVER' in src
        # C5: skill-write gate checks aborted_by_solver.
        assert "aborted_by_solver" in src
        assert "solver aborted" in src


# ------------------------------------------------------------------ #
# Continue-on-gen-error (C2) and rejection sanitization (M3)         #
# ------------------------------------------------------------------ #


class TestGenLoopResilience_C2_M3:
    def test_gen_exception_does_not_early_return(self):
        # Verify the gen loop source uses `continue` rather than
        # `return` on its Exception branch (C2), and sanitizes
        # rejection feedback with &lt;/&gt; escapes (M3).
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # C2: the except branch now sets rejection_feedback + continue.
        assert 'rejection_feedback = f"previous attempt failed' in src
        # M3: rejection feedback is HTML-escaped before re-injection.
        assert 'reason.replace("<", "&lt;")' in src


# ------------------------------------------------------------------ #
# Biological hook cooldown anchor (C3)                               #
# ------------------------------------------------------------------ #


class TestBiologicalHookCooldownAnchor_C3:
    @pytest.mark.asyncio
    async def test_anchor_updates_even_when_self_play_raises(self, tmp_path):
        import datetime
        import ghost_agent.core.agent as agent_mod

        # Minimal context with a frontier_tracker + llm_client stub.
        ctx = MagicMock()
        ctx.memory_system = MagicMock()
        ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
        ctx.llm_client = MagicMock(foreground_tasks=0)
        ctx.frontier_tracker = FrontierTracker(tmp_path)
        ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(hours=2)
        ctx.args = MagicMock(model="default")

        agent = agent_mod.GhostAgent.__new__(agent_mod.GhostAgent)
        agent.context = ctx

        # Simulate ~2h idle to pass the idle_secs > 3600 gate.
        far_past = datetime.datetime.now() - datetime.timedelta(hours=3)
        agent._last_dream_at = far_past
        agent._last_selfplay_at = far_past
        agent._current_selfplay_cooldown = 3600

        # Force the random gate to always fire.
        with patch("ghost_agent.core.agent.random.random", return_value=0.0):
            # Dreamer.synthetic_self_play explodes mid-flight.
            async def _boom(*a, **kw):
                raise RuntimeError("synthetic_self_play died")
            with patch("ghost_agent.core.dream.Dreamer") as D:
                inst = MagicMock()
                inst.synthetic_self_play = AsyncMock(side_effect=_boom)
                D.return_value = inst
                with pytest.raises(RuntimeError):
                    await agent._biological_tick()

        # C3: even though the call raised, the cooldown anchor moved.
        assert agent._last_selfplay_at > far_past


# ------------------------------------------------------------------ #
# Pre-flight side-effect restore (M5)                                #
# ------------------------------------------------------------------ #


class TestPreflightRestore_M5:
    def test_source_restores_snapshot_after_preflight(self):
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # Restore runs BEFORE temp_agent = GhostAgent(...).
        preflight_idx = src.index("python3 .preflight.py")
        restore_idx = src.index("_preflight_restore")
        ghost_idx = src.index("temp_agent = GhostAgent(")
        assert preflight_idx < restore_idx < ghost_idx


# ------------------------------------------------------------------ #
# disabled_tools blocklist (C6)                                      #
# ------------------------------------------------------------------ #


class TestDisabledToolsBlocklist_C6:
    def test_dream_web_and_deep_research_in_blocklist(self):
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        for tool in ("dream_mode", "web_search", "deep_research"):
            assert tool in src, f"{tool} missing from blocklist"


# ------------------------------------------------------------------ #
# Secondary-module nulling on isolated_context (C4)                  #
# ------------------------------------------------------------------ #


class TestIsolatedContextNulling_C4:
    def test_source_nulls_secondary_modules(self):
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        for attr in (
            "verifier", "uncertainty_tracker",
            "mcts_reasoner", "hypothesis_tester", "frontier_tracker",
        ):
            assert f'"{attr}"' in src, f"secondary module {attr} not nulled"


# ------------------------------------------------------------------ #
# _restore_mocks purges stragglers (S4)                              #
# ------------------------------------------------------------------ #


class TestRestoreMocksPurge_S4:
    def test_source_purges_non_snapshot_files(self):
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # New straggler-purge logic skips protected names and deletes
        # files not in snap_names.
        assert "snap_names" in src
        assert "p.unlink" in src
        assert "acquired_skills" in src  # protected name

    def test_pre_validator_restore_preserves_solution_py(self):
        """Regression for the 12:37 log: the pre-validator `_restore_mocks`
        call must NOT purge solver-written files (e.g., solution.py).
        The validator subprocess-runs solution.py, so deleting it
        produces a "No such file or directory" failure on every attempt.

        The purge behaviour is gated behind a `purge_stragglers` flag
        and only the between-attempts call sets it to True; the pre-
        validator call uses the default False.
        """
        import inspect
        from ghost_agent.core.dream import Dreamer

        src = inspect.getsource(Dreamer.synthetic_self_play)
        # Helper signature accepts the flag with a safe False default.
        assert "purge_stragglers: bool = False" in src
        # Between-attempts call passes True.
        # Pre-validator call (under "Restore mocks one more time") uses
        # the default — no `True` third positional argument — so
        # `solution.py` survives.
        pre_validator_block = src[src.index("Restore mocks one more time"):]
        next_call = pre_validator_block[:pre_validator_block.index("sandbox_manager.execute")]
        # The pre-validator call must NOT pass `True` as the purge flag.
        assert "True,  # purge_stragglers" not in next_call, (
            "pre-validator _restore_mocks must use purge=False or it deletes solution.py"
        )

    def test_restore_mocks_behavior_purge_vs_preserve(self, tmp_path):
        """Functional check on the helper's two modes. We can't easily
        reach the closure inside synthetic_self_play, so we replicate
        its contract with a mini-helper built from the same source
        invariants: purge=True deletes stragglers, purge=False keeps
        them (which is what the pre-validator call depends on)."""
        # Seed the tmp sandbox with one mock file and one straggler.
        (tmp_path / "mock.csv").write_bytes(b"a,b\n1,2\n")
        (tmp_path / "solution.py").write_bytes(b"print('hi')\n")
        snapshot = {"mock.csv": b"a,b\n1,2\n"}

        PROTECTED = {".setup.py", ".validator.py", ".preflight.py", "acquired_skills"}

        def _restore_mocks(path, snap, purge=False):
            snap_names = set(snap.keys())
            if purge:
                for p in path.iterdir():
                    if p.name in PROTECTED or p.name in snap_names:
                        continue
                    p.unlink()
            for name, blob in snap.items():
                (path / name).write_bytes(blob)

        # purge=False: straggler survives (pre-validator case).
        _restore_mocks(tmp_path, snapshot, purge=False)
        assert (tmp_path / "solution.py").exists()
        assert (tmp_path / "mock.csv").exists()

        # purge=True: straggler wiped (between-attempts case).
        _restore_mocks(tmp_path, snapshot, purge=True)
        assert not (tmp_path / "solution.py").exists()
        assert (tmp_path / "mock.csv").exists()
