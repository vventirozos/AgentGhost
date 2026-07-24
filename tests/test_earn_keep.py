"""Earn-your-keep harness — pure logic (no live boot).

Covers the two safety-critical, testable cores:
  * core.prune_overrides — the catalog + prod-apply (arg/env flips) + the
    protected-set refusal + the defensive load (absent/malformed → {}).
  * scripts/earn_keep — ledger I/O, matched-pair attribution, the pre-registered
    sustained-verdict rule (every branch), and auto-prune (dry-run + protected).
"""
import json
import os
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from ghost_agent.core import prune_overrides as PO
import earn_keep as EK


# ── prune_overrides ─────────────────────────────────────────────────────────

class TestPruneOverridesCatalog:
    def test_protected_set_covers_memory_and_verifier(self):
        assert "memory" in PO.PROTECTED
        assert "verifier" in PO.PROTECTED
        assert "deep_reason" not in PO.PROTECTED

    def test_every_subsystem_has_a_complete_spec(self):
        for name, spec in PO.SUBSYSTEMS.items():
            for key in ("arm", "kind", "target", "disabled_value", "protected",
                        "costs", "track"):
                assert key in spec, f"{name} missing {key}"
            assert spec["kind"] in ("arg", "env")

    def test_trackb_idle_loops_are_catalogued(self):
        """dream / self_play / reflection are the Track-B idle-loop arms."""
        for name in ("dream", "self_play", "reflection"):
            spec = PO.SUBSYSTEMS[name]
            assert spec["track"] == "B", f"{name} must be track B"
            assert spec["kind"] == "arg"
            assert spec["protected"] is False
        # dream/self-play run real idle LLM work → costed (prunable if useless)
        assert PO.SUBSYSTEMS["dream"]["target"] == "no_dream"
        assert PO.SUBSYSTEMS["dream"]["costs"] is True
        assert PO.SUBSYSTEMS["self_play"]["target"] == "no_self_play"
        assert PO.SUBSYSTEMS["self_play"]["costs"] is True

    def test_idle_loop_arms_are_distinct(self):
        arms = [PO.SUBSYSTEMS[n]["arm"] for n in ("dream", "self_play", "reflection")]
        assert len(set(arms)) == 3, f"idle-loop arms must be distinct: {arms}"

    def test_apply_arg_prune_flips_idle_loops(self):
        args = types.SimpleNamespace(no_dream=False, no_self_play=False,
                                     no_reflection=False)
        PO.apply_arg_prunes(args, {"dream": {}, "self_play": {}})
        assert args.no_dream is True
        assert args.no_self_play is True
        assert args.no_reflection is False   # not pruned → untouched


class TestPruneOverridesIO:
    def test_load_absent_is_empty(self, tmp_path):
        assert PO.load_pruned(tmp_path) == {}

    def test_load_malformed_is_empty_never_raises(self, tmp_path):
        p = tmp_path / "system" / "earn_keep" / "pruned.json"
        p.parent.mkdir(parents=True)
        p.write_text("{ not json")
        assert PO.load_pruned(tmp_path) == {}

    def test_load_none_home_is_empty(self):
        assert PO.load_pruned(None) == {}

    def test_record_and_reload_roundtrip(self, tmp_path):
        PO.record_prune(tmp_path, "deep_reason", {"delta_help": -0.01}, "2026-07-22T00:00:00")
        got = PO.load_pruned(tmp_path)
        assert "deep_reason" in got
        assert got["deep_reason"]["evidence"]["delta_help"] == -0.01

    def test_record_refuses_protected(self, tmp_path):
        with pytest.raises(ValueError):
            PO.record_prune(tmp_path, "verifier", {}, "t")
        with pytest.raises(ValueError):
            PO.record_prune(tmp_path, "memory", {}, "t")

    def test_record_refuses_unknown(self, tmp_path):
        with pytest.raises(ValueError):
            PO.record_prune(tmp_path, "not_a_subsystem", {}, "t")


class TestPruneOverridesApply:
    def test_apply_arg_prune_flips_attribute(self):
        args = types.SimpleNamespace(enable_metacog=True, deep_reason=True,
                                     no_self_model=False)
        applied = PO.apply_arg_prunes(args, {"metacog": {}, "self_model": {}})
        assert args.enable_metacog is False   # metacog disabled_value
        assert args.no_self_model is True     # self_model disabled_value
        assert args.deep_reason is True       # untouched
        assert set(applied) == {"metacog", "self_model"}

    def test_apply_arg_never_touches_protected_even_if_in_file(self):
        args = types.SimpleNamespace(no_memory=False, no_verifier=False)
        applied = PO.apply_arg_prunes(args, {"memory": {}, "verifier": {}})
        assert args.no_memory is False        # protected → never applied
        assert args.no_verifier is False
        assert applied == []

    def test_apply_env_prune_sets_env(self):
        env = {}
        applied = PO.apply_env_prunes({"hypothesis": {}}, environ=env)
        assert env.get("GHOST_HYPOTHESIS_GROUNDING") == "0"
        assert applied == ["hypothesis"]

    def test_apply_ignores_unknown_names_in_file(self):
        args = types.SimpleNamespace(deep_reason=True)
        applied = PO.apply_arg_prunes(args, {"garbage": {}, "deep_reason": {}})
        assert args.deep_reason is False
        assert applied == ["deep_reason"]


# ── earn_keep ledger + attribution ──────────────────────────────────────────

def _run(run_ts, full_passes, arm_cfg, arm_passes, n=20):
    """One run: `full` + one arm, over n tasks in repeat 0. full_passes /
    arm_passes are the count of leading True (rest False)."""
    recs = []
    for i in range(n):
        recs.append({"run_ts": run_ts, "config": "full", "repeat": 0,
                     "task_id": f"t{i}", "cluster": "c",
                     "passed": i < full_passes, "duration_s": 2.0})
        recs.append({"run_ts": run_ts, "config": arm_cfg, "repeat": 0,
                     "task_id": f"t{i}", "cluster": "c",
                     "passed": i < arm_passes, "duration_s": 1.0})
    return recs


class TestLedger:
    def test_append_and_load_roundtrip(self, tmp_path):
        led = tmp_path / "results.jsonl"
        EK.append_records(led, [{"run_ts": "a", "config": "full", "passed": True}])
        EK.append_records(led, [{"run_ts": "b", "config": "thin", "passed": False}])
        got = EK.load_ledger(led)
        assert len(got) == 2 and got[0]["run_ts"] == "a"

    def test_load_tolerates_torn_tail(self, tmp_path):
        led = tmp_path / "results.jsonl"
        led.write_text('{"run_ts":"a","config":"full","passed":true}\n{ torn')
        assert len(EK.load_ledger(led)) == 1


class TestAttribution:
    def test_matched_pairs_only_shared_cells(self):
        recs = _run("r1", full_passes=10, arm_cfg="full_no_deepreason", arm_passes=5)
        pairs = EK._matched_pairs(recs, "full", "full_no_deepreason")
        assert len(pairs) == 20

    def test_delta_positive_when_full_beats_arm(self):
        recs = _run("r1", 18, "full_no_deepreason", 10)   # full much better
        d, lo, hi = EK._bootstrap_diff_ci(
            EK._matched_pairs(recs, "full", "full_no_deepreason"))
        assert d == pytest.approx((18 - 10) / 20)
        assert lo <= d <= hi

    def test_bootstrap_is_deterministic(self):
        recs = _run("r1", 12, "full_no_deepreason", 12)
        a = EK._bootstrap_diff_ci(EK._matched_pairs(recs, "full", "full_no_deepreason"))
        b = EK._bootstrap_diff_ci(EK._matched_pairs(recs, "full", "full_no_deepreason"))
        assert a == b


class TestVerdict:
    def _stats(self, **kw):
        base = {"delta_help": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "n_pairs": 100, "n_runs": 5, "protected": False,
                "costs": True, "track": "A"}
        base.update(kw)
        return base

    def test_protected_never_prunes(self):
        assert EK.verdict("verifier", self._stats(protected=True, delta_help=-0.5)) == "protected"

    def test_free_subsystem_is_kept(self):
        assert EK.verdict("self_model", self._stats(costs=False, delta_help=-0.5)) == "keep_free"

    def test_insufficient_evidence(self):
        assert EK.verdict("deep_reason", self._stats(n_runs=2)) == "insufficient"
        assert EK.verdict("deep_reason", self._stats(n_pairs=30)) == "insufficient"

    def test_prune_when_no_help_tight_ci_and_costs(self):
        assert EK.verdict("deep_reason",
                          self._stats(delta_help=-0.01, ci_hi=0.005)) == "prune"

    def test_keep_when_helps(self):
        assert EK.verdict("deep_reason",
                          self._stats(delta_help=0.08, ci_hi=0.14)) == "keep"

    def test_keep_when_uncertain_ci_admits_benefit(self):
        # Δ≤0 but the CI still admits a >2pp benefit → not enough to prune.
        assert EK.verdict("deep_reason",
                          self._stats(delta_help=-0.01, ci_hi=0.05)) == "keep"


class TestAutoPrune:
    def _ledger_for_prune(self):
        # 3 runs, deep_reason arm identical to full (Δ=0, CI≈0) → prune;
        # 20 tasks × 3 = 60 pairs (meets MIN_PAIRS).
        recs = []
        for rt in ("r1", "r2", "r3"):
            recs += _run(rt, 12, "full_no_deepreason", 12)
        return recs

    def test_sustained_no_help_triggers_prune(self, tmp_path):
        attribution = EK.attribute(self._ledger_for_prune())
        assert attribution["deep_reason"]["verdict"] == "prune"
        newly = EK.auto_prune(attribution, tmp_path, "2026-07-22T00:00:00")
        assert "deep_reason" in newly
        assert "deep_reason" in PO.load_pruned(tmp_path)

    def test_dry_run_writes_nothing(self, tmp_path):
        attribution = EK.attribute(self._ledger_for_prune())
        newly = EK.auto_prune(attribution, tmp_path, "t", dry_run=True)
        assert "deep_reason" in newly            # reported
        assert PO.load_pruned(tmp_path) == {}    # but not written

    def test_already_pruned_is_not_repeated(self, tmp_path):
        PO.record_prune(tmp_path, "deep_reason", {}, "earlier")
        attribution = EK.attribute(self._ledger_for_prune())
        newly = EK.auto_prune(attribution, tmp_path, "now")
        assert newly == []                       # nothing new

    def test_helping_subsystem_is_not_pruned(self, tmp_path):
        recs = []
        for rt in ("r1", "r2", "r3"):
            recs += _run(rt, 18, "full_no_deepreason", 8)   # full clearly better
        attribution = EK.attribute(recs)
        assert attribution["deep_reason"]["verdict"] == "keep"
        assert EK.auto_prune(attribution, tmp_path, "t") == []


# ── resumability: checkpoint / done-groups / fold / manifest ────────────────

class TestResume:
    ARMS = ["full", "full_no_deepreason", "thin"]

    def _cells(self, rep, tid, configs):
        return [{"config": c, "repeat": rep, "task_id": tid, "cluster": "c",
                 "passed": True, "duration_s": 1.0} for c in configs]

    def test_group_done_only_when_all_arms_present(self, tmp_path):
        cp = tmp_path / "checkpoint.jsonl"
        EK.append_group(cp, self._cells(0, "t0", self.ARMS))          # complete
        EK.append_group(cp, self._cells(0, "t1", ["full", "thin"]))   # missing one arm
        done = EK.load_done_groups(cp, self.ARMS)
        assert (0, "t0") in done
        assert (0, "t1") not in done          # partial group is NOT done → redone

    def test_partial_group_redone_and_deduped_on_fold(self, tmp_path):
        cp = tmp_path / "checkpoint.jsonl"
        # a killed mid-group left 2 arms; resume re-fires the whole group.
        EK.append_group(cp, self._cells(0, "t0", ["full", "full_no_deepreason"]))
        EK.append_group(cp, self._cells(0, "t0", self.ARMS))          # re-fired
        recs = EK.fold_checkpoint(cp, "run1", "A")
        # dedup keep-last per (config, repeat, task_id): no double-count.
        keys = [(r["config"], r["repeat"], r["task_id"]) for r in recs]
        assert len(keys) == len(set(keys)) == 3

    def test_fold_stamps_run_ts_and_track(self, tmp_path):
        cp = tmp_path / "checkpoint.jsonl"
        EK.append_group(cp, self._cells(1, "t5", self.ARMS))
        recs = EK.fold_checkpoint(cp, "RUN42", "A")
        assert all(r["run_ts"] == "RUN42" and r["track"] == "A" for r in recs)

    def test_append_group_is_line_durable(self, tmp_path):
        cp = tmp_path / "checkpoint.jsonl"
        EK.append_group(cp, self._cells(0, "t0", self.ARMS))
        EK.append_group(cp, self._cells(0, "t1", self.ARMS))
        assert len([l for l in cp.read_text().splitlines() if l.strip()]) == 6

    def test_manifest_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(EK, "EK_ROOT", tmp_path)
        EK.write_manifest("R1", {"run_id": "R1", "status": "running",
                                 "track": "A", "created_at": "2026-07-22T01:00:00"})
        assert EK.read_manifest("R1")["status"] == "running"

    def test_find_resumable_picks_latest_incomplete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(EK, "EK_ROOT", tmp_path)
        EK.write_manifest("R1", {"status": "complete", "track": "A", "created_at": "1"})
        EK.write_manifest("R2", {"status": "running", "track": "A", "created_at": "2"})
        EK.write_manifest("R3", {"status": "running", "track": "A", "created_at": "3"})
        assert EK.find_resumable(track="A") == "R3"      # latest incomplete
        # complete-only → nothing to resume
        EK.write_manifest("R2", {"status": "complete", "track": "A", "created_at": "2"})
        EK.write_manifest("R3", {"status": "complete", "track": "A", "created_at": "3"})
        assert EK.find_resumable(track="A") is None


class TestTrackAIdleQuiesce:
    def test_quiesce_appends_bio_time_scale(self):
        ap = types.SimpleNamespace(COMMON=["--upstream-url", "x", "--api-key", ""])
        out = EK.apply_track_a_quiesce(ap)
        assert "--bio-time-scale" in out and "0.001" in out
        # original flags preserved
        assert "--upstream-url" in out

    def test_quiesce_is_idempotent(self):
        ap = types.SimpleNamespace(COMMON=["--bio-time-scale", "60"])
        out = EK.apply_track_a_quiesce(ap)
        # already has a bio-time-scale → not double-added
        assert out.count("--bio-time-scale") == 1

    def test_quiesce_uniform_not_in_per_arm_config(self):
        # The suppression rides COMMON (every arm), NOT a per-subsystem arm, so
        # it can't bias the paired comparison.
        assert "--bio-time-scale" not in " ".join(EK.IDLE_QUIESCE_FLAGS[:0] + [])
        assert EK.IDLE_QUIESCE_FLAGS == ["--bio-time-scale", "0.001"]


class TestMeasureResumableLoop:
    """The real _measure_resumable loop with a FAKE runner (no agent boot):
    verifies group checkpointing + that a resume skips already-done groups."""

    def _fake_ap(self, calls):
        async def _run_one(url, model, timeout, task):
            calls.append((url, getattr(task, "task_id", "?")))
            return True, 1.0, ""
        return types.SimpleNamespace(_run_one=_run_one)

    def _suite(self, n):
        return [types.SimpleNamespace(task_id=f"t{i}", cluster="c") for i in range(n)]

    @pytest.mark.asyncio
    async def test_checkpoints_all_groups_then_resume_skips_them(self, tmp_path, monkeypatch):
        monkeypatch.setattr(EK, "EK_ROOT", tmp_path)   # never write to the repo
        arms = ["full", "full_no_deepreason", "thin"]
        live = {a: {"url": f"http://x/{a}"} for a in arms}
        suite = self._suite(4)               # 4 tasks × 2 repeats = 8 groups
        cp = tmp_path / "checkpoint.jsonl"
        prog = EK.Progress(tmp_path / "progress.log")
        calls1 = []
        monkeypatch.setattr(EK, "AP", self._fake_ap(calls1))
        await EK._measure_resumable(live, arms, suite, 2, "m", 1.0, 7,
                                    cp, prog, {}, "R1")
        # every group×arm ran once: 8 groups × 3 arms.
        assert len(calls1) == 8 * len(arms)
        assert len(EK.load_done_groups(cp, arms)) == 8
        # progress.log captured live per-cell + per-group lines.
        text = (tmp_path / "progress.log").read_text()
        assert "group 1/8" in text and "PASS" in text

        # --- simulate resume: same checkpoint, fresh call → ZERO new work ---
        calls2 = []
        monkeypatch.setattr(EK, "AP", self._fake_ap(calls2))
        await EK._measure_resumable(live, arms, suite, 2, "m", 1.0, 7,
                                    cp, prog, {}, "R1")
        assert calls2 == []                  # all groups already done → skipped
        prog.close()

    @pytest.mark.asyncio
    async def test_resume_after_partial_completes_the_rest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(EK, "EK_ROOT", tmp_path)   # never write to the repo
        arms = ["full", "thin"]
        live = {a: {"url": f"http://x/{a}"} for a in arms}
        suite = self._suite(3)               # 3 tasks × 1 repeat = 3 groups
        cp = tmp_path / "checkpoint.jsonl"
        # pretend 1 group already completed before a "kill".
        EK.append_group(cp, [{"config": a, "repeat": 0, "task_id": "t0",
                              "cluster": "c", "passed": True, "duration_s": 1.0}
                             for a in arms])
        prog = EK.Progress(tmp_path / "progress.log")
        calls = []
        monkeypatch.setattr(EK, "AP", self._fake_ap(calls))
        await EK._measure_resumable(live, arms, suite, 1, "m", 1.0, 7, cp, prog, {}, "R1")
        # only the 2 remaining groups ran (t0 skipped): 2 groups × 2 arms.
        assert len(calls) == 2 * len(arms)
        assert len(EK.load_done_groups(cp, arms)) == 3   # now all 3 done
        prog.close()
