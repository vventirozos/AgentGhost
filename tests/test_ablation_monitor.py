"""Tests for scripts/ablation_monitor.py — the Track-B4 progress monitor.

The load-bearing parts are pure: mode/total detection from the driver log,
DRIVER-tagged turn counting (must ignore self-play frames), and progress math.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ablation_monitor as M  # noqa: E402


PILOT_LOG = (
    "[15:59:57]   ready: pilot on http://127.0.0.1:8046 (pid 97394)\n"
    "[15:59:57] === pilot pass 1/3 ===\n"
    "[16:20:00] === pilot pass 2/3 ===\n"
)
FULL_LOG = (
    "[10:00:00] === B4 repeat 1/3 ===\n"
    "[10:00:05]   ready: treatment on http://127.0.0.1:8046 (pid 1)\n"
    "[10:30:00]   ready: treatment_uniform on http://127.0.0.1:8046 (pid 2)\n"
    "[11:00:00]   ready: control on http://127.0.0.1:8046 (pid 3)\n"
    "[11:30:00] === B4 repeat 2/3 ===\n"
    "[11:30:05]   ready: treatment on http://127.0.0.1:8046 (pid 4)\n"
)

# An agent log with 3 DRIVER (DV) END frames + 2 self-play (SU/JO) frames and
# some ANSI colour — only the DV frames must be counted.
AGENT_LOG = (
    "\033[36m└─ DV  request finished  3.1s ─────\033[0m\n"
    "└─ SU  request finished  9.9s ─────\n"          # self-play
    "└─ DV  request finished  2.2s ─────\n"
    "└─ JO  request finished  8.0s ─────\n"          # self-play
    "\033[1m└─ DV\033[0m  request finished  1.0s ──\n"
)


class TestDetectConfig:
    def test_pilot(self):
        c = M.detect_config(PILOT_LOG, None)
        assert c.mode == "pilot"
        assert c.repeats == 3 and c.arms_per_repeat == 1
        assert c.battery == M.DEFAULT_BATTERY
        assert c.total_turns == 3 * M.DEFAULT_BATTERY

    def test_full(self):
        c = M.detect_config(FULL_LOG, None)
        assert c.mode == "full"
        assert c.repeats == 3 and c.arms_per_repeat == 3
        assert c.turns_per_arm == c.seeding + c.battery
        assert c.total_turns == 3 * 3 * (c.seeding + c.battery)

    def test_battery_override(self):
        c = M.detect_config(FULL_LOG, 12)
        assert c.battery == 12

    def test_no_frontier_arm_detected_as_two(self):
        log = FULL_LOG.replace(
            "[10:30:00]   ready: treatment_uniform on http://127.0.0.1:8046 (pid 2)\n", "")
        c = M.detect_config(log, None)
        assert c.arms_per_repeat == 2


class TestCountDriverTurns:
    def test_counts_only_dv_frames(self, tmp_path):
        log = tmp_path / "treatment.log"
        log.write_text(AGENT_LOG)
        assert M.count_driver_turns(log) == 3   # not 5

    def test_missing_log_is_zero(self, tmp_path):
        assert M.count_driver_turns(tmp_path / "nope.log") == 0


class TestComputeProgress:
    def test_pilot_position(self, tmp_path):
        (tmp_path / "pilot.log").write_text(AGENT_LOG)  # 3 DV turns
        cfg = M.detect_config(PILOT_LOG, None)
        p = M.compute_progress(cfg, PILOT_LOG, tmp_path, elapsed_s=600, idle_stall=False)
        assert p.done == 3
        assert p.repeat_now == 2                # pass 2 is the latest header
        # 3 done, pass 2 started → 3 - 1*35 clamps to 0 tasks into pass 2
        assert p.turns_in_arm == 0

    def test_full_position_and_phase(self, tmp_path):
        # In repeat 2, current arm 'treatment', with >seeding turns → probe phase.
        (tmp_path / "treatment.log").write_text(AGENT_LOG * 4)  # 12 DV turns
        cfg = M.detect_config(FULL_LOG, None)
        p = M.compute_progress(cfg, FULL_LOG, tmp_path, elapsed_s=3600, idle_stall=False)
        assert p.arm_now == "treatment"
        assert p.repeat_now == 2
        # 4 arms fully done before the current one (3 in rep1 + 0... actually
        # readies=4 → arms_done=3), each turns_per_arm, + 12 in current.
        assert p.done == 3 * cfg.turns_per_arm + 12
        assert p.phase == "probe"              # 12 > seeding(8)

    def test_full_idle_phase_flag(self, tmp_path):
        (tmp_path / "treatment.log").write_text(
            "\n".join(["└─ DV  request finished  1s ──"] * 8))  # exactly seeding
        cfg = M.detect_config(FULL_LOG, None)
        p = M.compute_progress(cfg, FULL_LOG, tmp_path, elapsed_s=100, idle_stall=True)
        assert p.phase == "idle"


class TestFormatStatus:
    def test_pilot_render(self, tmp_path):
        (tmp_path / "pilot.log").write_text(AGENT_LOG)
        cfg = M.detect_config(PILOT_LOG, None)
        p = M.compute_progress(cfg, PILOT_LOG, tmp_path, elapsed_s=600, idle_stall=False)
        out = M.format_status(p)
        assert "B4 PILOT" in out and "pass 2/3" in out
        assert "%" in out and "ETA" in out

    def test_eta_is_none_safe_at_zero(self):
        cfg = M.RunConfig("full", 3, 3, 8, 35)
        p = M.Progress(cfg, 0, cfg.total_turns, 1, "treatment", 1, 0, "seed", 0.0)
        out = M.format_status(p)          # must not divide by zero
        assert "ETA ~—" in out
