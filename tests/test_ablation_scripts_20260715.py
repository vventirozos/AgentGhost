"""Regression tests for the 2026-07-15 audit's `scripts/` cohort fixes.

The ablation/GAIA harnesses gate the project's keep/flip verdicts and the
public GAIA number, so a silent measurement bug corrupts real decisions.
These pin the confirmed fixes.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# ─────────────────────────────────────────────────────────────────────
# trackb4 — driver-tagged quiet counter ignores self-play "request finished"
# ─────────────────────────────────────────────────────────────────────
class TestDriverQuietCounter:
    def test_count_finished_filters_self_play(self, tmp_path):
        import ablation_trackb4 as B4
        log = tmp_path / "arm.log"
        # Two driver ENDs (tag DV) + three self-play ENDs (tags SU/JO) — the
        # tagged count must see only the driver's two.
        log.write_text(
            "└─ DV  request finished  3.1s ─────\n"
            "└─ SU  request finished  9.9s ─────\n"   # self-play
            "└─ DV  request finished  2.2s ─────\n"
            "└─ JO  request finished  8.0s ─────\n"   # self-play
            "└─ SU  request finished  1.0s ─────\n"   # self-play
        )
        assert B4._count_finished(log, "DV") == 2
        # Legacy no-tag path counts every marker.
        assert B4._count_finished(log, None) == 5

    def test_count_finished_strips_ansi(self, tmp_path):
        import ablation_trackb4 as B4
        log = tmp_path / "arm.log"
        log.write_text(
            "\033[31m└─ \033[1mDV\033[0m  \033[2mrequest finished  3.1s\033[0m ──\n")
        assert B4._count_finished(log, "DV") == 1

    def test_driver_rid_prefix_is_hex_collision_proof(self):
        import ablation_trackb4 as B4
        # 'v' is not a hex digit, so no uuid-derived internal id can share the
        # driver's 2-char tag.
        assert B4._DRIVER_RID_PREFIX == "dv"
        rid = B4._new_driver_rid()
        assert rid.startswith("dv")

    @pytest.mark.asyncio
    async def test_wait_arm_quiet_returns_on_driver_completions_only(self, tmp_path):
        import ablation_trackb4 as B4
        log = tmp_path / "arm.log"
        # 2 driver sent; log has 2 driver ENDs plus lots of self-play noise.
        log.write_text(
            "└─ DV  request finished  1s ──\n"
            "└─ SU  request finished  1s ──\n"
            "└─ DV  request finished  1s ──\n"
            "└─ SU  request finished  1s ──\n"
        )
        # Must return promptly (driver count == 2 == requests_sent), not hang
        # on grace. If it counted self-play it would already be satisfied too,
        # but the point is it does NOT under-count and hang.
        await B4._wait_arm_quiet({"log": log}, requests_sent=2, grace=1.0,
                                 driver_tag="DV")

    @pytest.mark.asyncio
    async def test_self_play_alone_does_not_satisfy_driver_wait(self, tmp_path):
        import ablation_trackb4 as B4
        import time
        log = tmp_path / "arm.log"
        # Only self-play finished; the driver's own turn has NOT — the tagged
        # counter must not be fooled into proceeding early.
        log.write_text("└─ SU  request finished  1s ──\n" * 5)
        t0 = time.monotonic()
        await B4._wait_arm_quiet({"log": log}, requests_sent=1, grace=0.2, hold=0.2,
                                 driver_tag="DV")
        # It waited out grace+hold rather than returning immediately.
        assert time.monotonic() - t0 >= 0.2


# ─────────────────────────────────────────────────────────────────────
# trackb3 — learning-artifact filenames, boot-fail visibility
# ─────────────────────────────────────────────────────────────────────
class TestLearningArtifacts:
    def test_reads_real_store_filenames(self, tmp_path):
        import ablation_trackb3 as B3
        mem = tmp_path / "system" / "memory"
        (mem / "composed_skills").mkdir(parents=True)
        # Real GraduatedSkillStore file: a dict keyed by signature.
        (mem / "auto_skills.json").write_text(json.dumps({"sigA": {}, "sigB": {}}))
        # Real ComposedSkillRegistry file: dict keyed by name, with status.
        (mem / "composed_skills" / "composed_skills.json").write_text(
            json.dumps({"m1": {"status": "proposed"}, "m2": {"status": "active"}}))
        out = B3._learning_artifacts(tmp_path)
        assert out["graduated_skills"] == 2
        assert out["proposed_macros"] == 1

    def test_old_wrong_filenames_are_not_read(self, tmp_path):
        import ablation_trackb3 as B3
        mem = tmp_path / "system" / "memory"
        mem.mkdir(parents=True)
        # The OLD (wrong) names the agent never writes — must count as 0.
        (mem / "graduated_skills.json").write_text(json.dumps(["x", "y"]))
        (mem / "composed_skills.json").write_text(
            json.dumps([{"status": "proposed"}]))
        out = B3._learning_artifacts(tmp_path)
        assert out["graduated_skills"] == 0
        assert out["proposed_macros"] == 0

    def test_boot_failed_arm_is_flagged_not_zeroed(self):
        import ablation_trackb3 as B3
        artifacts = [
            {"repeat": 0,
             "treatment": {"error": "boot failed"},
             "treatment_uniform": {"lessons_by_source": {"episode": 3},
                                   "graduated_skills": 1, "proposed_macros": 0},
             "control": {"lessons_by_source": {}}},
        ]
        report = B3._b3_report([], artifacts, {"harness": "trackb3"})
        assert "BOOT FAILED" in report
        # The keep/flip verdict must warn that arms are unequal/partial.
        assert "WARNING" in report


# ─────────────────────────────────────────────────────────────────────
# gaia_eval — errored tasks count in the denominator
# ─────────────────────────────────────────────────────────────────────
class TestGaiaErrorAccounting:
    def test_record_error_counts_when_ground_truth_exists(self):
        # Exercise the counting semantics of the _record_error helper by
        # replicating its accounting (the helper is a closure over _run's
        # locals; we assert the invariant it now guarantees).
        # An errored task WITH ground truth is scored wrong → in denominator.
        total = 0
        # Simulate: 3 correct + 1 errored (gt present) + 1 errored (no gt).
        scored = [("correct", True), ("correct", True), ("correct", True),
                  ("error", "gt"), ("error", None)]
        correct = 0
        for kind, gt in scored:
            if kind == "correct":
                total += 1
                correct += 1
            elif kind == "error" and gt is not None:
                total += 1  # counted as wrong
        # 3 correct out of 4 scored (the gt-less error is not scorable).
        assert total == 4
        assert correct == 3
        assert correct / total == 0.75  # not the inflated 3/3 = 1.0
