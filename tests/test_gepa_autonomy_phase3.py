"""§4DF Phase 3 — the optimizer launcher, driven.

The launcher runs ONE gate script per day and consumes `GateExit` plus
the banner/marker channel. The pins here are the consumer side: each
declared code maps to exactly one action, an exit code without its
marker is an instrument failure (crash-as-1 would otherwise be filed as
the LOG-ONLY "rejected" forever), cadence is one target/day and 7d per
target with staleness round-robin, the launcher never escalates (no
override flags, no optimize_verifier.py), and the RAM preflight fails
CLOSED.
"""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.optim import autonomy as A
from ghost_agent.optim.gate_contract import (
    GATE_NO_CANDIDATE_MARKER,
    GATE_PROMOTED_MARKER_GEPA,
    GATE_PROMOTED_MARKER_OTD,
    GATE_REJECTED_MARKER,
    GATE_RUN_BANNER_GEPA,
    GATE_RUN_BANNER_OTD,
    GateExit,
)


class _Recorder:
    def __init__(self):
        self.notes, self.logs = [], []

    def notify(self, msg):
        self.notes.append(msg)

    def log(self, msg):
        self.logs.append(msg)


def _gate(rc, out="", raw=False, err=""):
    """A stub gate whose stdout matches what the REAL scripts print —
    the run banner first, then the payload the test chooses. `raw=True`
    suppresses the banner for the tests ABOUT missing banners."""
    calls = []

    def run(script, args, *, home, timeout_s):
        calls.append((script, list(args), home, timeout_s))
        _rc = rc(script, args) if callable(rc) else rc
        _out = out
        if not raw:
            banner = (GATE_RUN_BANNER_OTD
                      if "tool_descriptions" in script
                      else GATE_RUN_BANNER_GEPA)
            _out = banner + "\n" + _out
        return SimpleNamespace(returncode=_rc, stdout=_out, stderr=err)
    run.calls = calls
    return run


def _home(tmp_path):
    home = tmp_path / "home"
    (home / "system" / "optim").mkdir(parents=True)
    return str(home)


def _one_target(monkeypatch, target="tool_descriptions"):
    """Pin the round-robin to a single target so a test can drive the
    SAME target through several runs without the never-attempted
    preference redirecting the pick."""
    monkeypatch.setattr(A, "OPTIMIZER_TARGETS", (target,))


# Payloads the real scripts produce, built FROM the constants — a stub
# whose output the real pipeline cannot produce is
# `harness-grades-own-homework`.
_PROMOTED_OTD = f"{GATE_PROMOTED_MARKER_OTD}/x/planning.json"
_PROMOTED_GEPA = f"{GATE_PROMOTED_MARKER_GEPA} to /x/planning.json"
_REJECTED = f"{GATE_REJECTED_MARKER} — live descriptions stand."
_NO_CANDIDATE = f"{GATE_NO_CANDIDATE_MARKER} — live descriptions stand."


# ─────────────────────────────────────────────────────────────────────
# The exit table
# ─────────────────────────────────────────────────────────────────────

class TestTheExitTable:
    def test_promoted_notifies_and_names_the_deploy_path(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", GateExit.PROMOTED)
        assert len(r.notes) == 1 and "PROMOTED" in r.notes[0]
        # The §4DE deploy path, not a restart instruction.
        assert "epoch swap" in r.notes[0]
        assert "restart" not in r.notes[0].replace("no restart", "")
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        tslot = st["optimizer"]["per_target"]["tool_descriptions"]
        assert tslot["last_outcome"] == "promoted"
        assert tslot["last_exit"] == 0

    def test_rejected_is_LOG_ONLY_the_system_working(self, tmp_path,
                                                     monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED, out=_REJECTED)
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", GateExit.REJECTED)
        assert not r.notes, r.notes
        assert any("REJECTED" in l for l in r.logs), r.logs

    def test_could_not_measure_is_LOG_ONLY(self, tmp_path, monkeypatch):
        """The gate's own pre-flights refusing (supply, re-draw age,
        upstream) is routine — exit 2 needs no marker, only the
        banner."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE,
                    out="supply gate: 122 REAL positive fixtures < 200")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", GateExit.COULD_NOT_MEASURE)
        assert not r.notes, r.notes
        assert any("could not measure" in l for l in r.logs), r.logs

    def test_no_candidate_notifies_ONCE_and_rearms(self, tmp_path,
                                                   monkeypatch):
        """A broken reflection LM is news once; weekly repeats of the
        same condition are noise. A different outcome in between
        re-arms the edge (`fire-once-notification`)."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        week = A.OPTIMIZER_TARGET_INTERVAL_S + 1
        run3 = _gate(GateExit.NO_CANDIDATE, out=_NO_CANDIDATE)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run3, now=1_000_000)
        assert len(r.notes) == 1 and "NO candidate" in r.notes[0]
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run3, now=1_000_000 + week)
        assert len(r.notes) == 1, "the repeat re-notified"
        run1 = _gate(GateExit.REJECTED, out=_REJECTED)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run1, now=1_000_000 + 2 * week)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run3, now=1_000_000 + 3 * week)
        assert len(r.notes) == 2, (
            "a rejected run in between did not re-arm the no-candidate "
            "edge: " + str(r.notes))

    def test_an_undeclared_code_is_an_instrument_failure(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(7, out="something odd")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", 7)
        assert len(r.notes) == 1 and "did not run cleanly" in r.notes[0]

    def test_the_inner_timeout_is_the_optimizer_deadline(
            self, tmp_path, monkeypatch):
        """Six hours, not the file-reader deadlines — an optimizer run
        killed at the miner's 900s would never complete honestly."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED, out=_REJECTED)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run, now=1_000_000)
        assert run.calls[0][3] == A.OPTIMIZER_TIMEOUT_S == 6 * 3600

    def test_a_spawn_failure_notifies_by_exception_type(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()

        def _boom(script, args, *, home, timeout_s):
            raise OSError("forced")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=_boom, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1 and "cause=OSError" in r.notes[0]


# ─────────────────────────────────────────────────────────────────────
# An exit code is believed only with its marker
# ─────────────────────────────────────────────────────────────────────

class TestAnExitCodeNeedsItsMarker:
    """⚠ THE CRASH-AS-1 SHAPE, ONE INSTRUMENT OVER. Python exits 1 on
    any uncaught exception, and here 1 (REJECTED) is LOG-ONLY — so a
    permanently crashing optimizer without the marker check would read
    as "rejected weekly" forever, a silently dead loop wearing the
    system-working-as-designed's clothes (the judge-A1 impersonation)."""

    def test_exit_1_without_the_REJECTED_marker_notifies(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED,
                    out="Traceback (most recent call last): ...")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1, (
            "a crash impersonating a rejection was filed LOG-ONLY")
        assert "no-rejected-marker" in r.notes[0]

    def test_exit_0_without_the_PROMOTED_marker_is_not_a_promotion(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out="smoke summary: 35 replays")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1
        assert "no-promoted-marker" in r.notes[0]
        assert "epoch swap" not in r.notes[0], (
            "an unproven exit 0 was announced as a deploy")

    def test_exit_3_without_the_NO_CANDIDATE_marker_is_a_crash(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.NO_CANDIDATE, out="Traceback ...")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert "no-no-candidate-marker" in r.notes[0]

    def test_no_banner_means_the_script_did_not_start(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, raw=True,
                    out="usage: optimize_tool_descriptions.py ...")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1 and "no-banner" in r.notes[0]

    def test_each_gate_requires_ITS_OWN_promoted_marker(
            self, tmp_path, monkeypatch):
        """The gepa gate printing the OTD marker (or vice versa) is a
        cross-wired consumer — the per-target banner AND marker must
        both match the script that ran."""
        _one_target(monkeypatch, "gepa:planning.decompose")
        home = _home(tmp_path)
        r = _Recorder()
        # run_gepa exits 0 but prints the OTD marker: not believed.
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("gepa:planning.decompose", None)
        assert "no-promoted-marker" in r.notes[0]
        # And with its own marker it IS a promotion.
        r2 = _Recorder()
        run2 = _gate(GateExit.PROMOTED, out=_PROMOTED_GEPA)
        out2 = A.run_optimizer(home, notify=r2.notify, log=r2.log,
                               run_script=run2,
                               now=1_000_000 + A.OPTIMIZER_TARGET_INTERVAL_S + 1)
        assert out2 == ("gepa:planning.decompose", GateExit.PROMOTED)
        assert len(r2.notes) == 1 and "epoch swap" in r2.notes[0]


# ─────────────────────────────────────────────────────────────────────
# Cadence and the round-robin
# ─────────────────────────────────────────────────────────────────────

class TestTheCadence:
    def test_one_target_per_day(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is not None
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000 + 3600) \
            is None, "a second launch inside the same day"
        assert len(run.calls) == 1

    def test_never_attempted_targets_go_in_declared_order(self,
                                                          tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        picked = []
        day = A.OPTIMIZER_INTERVAL_S + 1
        for i in range(len(A.OPTIMIZER_TARGETS)):
            out = A.run_optimizer(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000 + i * day)
            picked.append(out[0])
        assert tuple(picked) == A.OPTIMIZER_TARGETS, picked

    def test_a_never_attempted_target_outranks_a_stale_one(self):
        """⚠ BATTERY SURVIVOR (M9): demoting never-attempted from
        "return immediately" to "fallback" survived every other test —
        the distinguishing world is MIXED state: a stale-but-eligible
        target EARLIER in the declared order than a never-attempted one.
        The world that hits it: a NEW target added to
        `OPTIMIZER_TARGETS` mid-life must get its first run next, not
        wait behind the stale rotation."""
        now = 1_000_000
        per = {"tool_descriptions":
               {"last_attempt_epoch": now - 30 * 86400}}
        assert A._pick_target(per, now) == "gepa:planning.decompose", (
            "a stale-eligible target outranked a never-attempted one")

    def test_all_targets_fresh_is_health_not_a_stall(self, tmp_path):
        """Every target attempted within 7d: the day's decision is
        "nothing to do" — returns None, but the persisted clock still
        advances so the liveness probe reads a decision, not a
        stopped schedule."""
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        day = A.OPTIMIZER_INTERVAL_S + 1
        for i in range(len(A.OPTIMIZER_TARGETS)):
            A.run_optimizer(home, notify=r.notify, log=r.log,
                            run_script=run, now=1_000_000 + i * day)
        n_calls = len(run.calls)
        now5 = 1_000_000 + len(A.OPTIMIZER_TARGETS) * day
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=now5) is None
        assert len(run.calls) == n_calls, "a fresh target was re-run"
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert st["optimizer"]["last_outcome"] == "nothing_due"
        assert st["optimizer"]["last_run_epoch"] == now5
        assert any("all targets fresh" in l for l in r.logs), r.logs

    def test_after_7d_the_STALEST_target_goes_first(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        day = A.OPTIMIZER_INTERVAL_S + 1
        for i in range(len(A.OPTIMIZER_TARGETS)):
            A.run_optimizer(home, notify=r.notify, log=r.log,
                            run_script=run, now=1_000_000 + i * day)
        # 8 days after the FIRST attempt: only the first target has
        # cleared its 7d window — it is both eligible and stalest.
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run,
                              now=1_000_000 + 8 * 86400)
        assert out is not None and out[0] == A.OPTIMIZER_TARGETS[0], out

    def test_a_future_attempt_stamp_is_a_clock_jump_not_recent(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        st_path = Path(home) / "system" / "gepa_autonomy_state.json"
        st_path.write_text(json.dumps({"optimizer": {"per_target": {
            "tool_descriptions": {
                "last_attempt_epoch": 9_000_000_000}}}}))
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out is not None, (
            "a future stamp parked the target until the far future")

    def test_hand_edited_state_cannot_stall_the_job(self, tmp_path,
                                                    monkeypatch):
        """The B3 coercion rule, at this job's level: a string where
        per_target belongs means "never ran", not a crash."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        st_path = Path(home) / "system" / "gepa_autonomy_state.json"
        st_path.write_text(json.dumps(
            {"optimizer": {"per_target": "oops",
                           "last_run_epoch": 1}}))
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", GateExit.COULD_NOT_MEASURE)


# ─────────────────────────────────────────────────────────────────────
# The launcher never escalates
# ─────────────────────────────────────────────────────────────────────

class TestTheLauncherNeverEscalates:
    def test_no_override_flags_ever(self, tmp_path):
        """§4DA's lesson: the one flag that bypasses a gate carries its
        own bypass. The launcher passes NO --allow-*, --force-*, or
        --no-ab-gate — for ANY target."""
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        day = A.OPTIMIZER_INTERVAL_S + 1
        for i in range(len(A.OPTIMIZER_TARGETS)):
            A.run_optimizer(home, notify=r.notify, log=r.log,
                            run_script=run, now=1_000_000 + i * day)
        assert len(run.calls) == len(A.OPTIMIZER_TARGETS)
        for script, args, _h, _t in run.calls:
            for a in args:
                assert not str(a).startswith("--allow"), (script, args)
                assert not str(a).startswith("--force"), (script, args)
                assert str(a) != "--no-ab-gate", (script, args)
                assert str(a) != "--smoke", (script, args)

    def test_optimize_verifier_is_outside_the_perimeter(self):
        """Operator decision, pinned: the verifier optimizer is not a
        target and no target command reaches its script."""
        for target in A.OPTIMIZER_TARGETS:
            script = A._target_command(target, "/tmp/x")[0]
            assert "optimize_verifier" not in script, target
        assert not any("verifier" in t for t in A.OPTIMIZER_TARGETS)

    def test_the_fixtures_argv_is_the_miners_output_path(self):
        """Driven before the fix: the launcher spawned the real otd gate
        with NO argv and argparse exited 2 BEFORE the banner — every
        launch an instrument failure. The path is built from the ONE
        shared basename, which the miner's default output also uses."""
        from ghost_agent.optim.gate_contract import TOOL_FIXTURES_BASENAME
        _s, args, _b, _m = A._target_command("tool_descriptions", "/h")
        assert args[:1] == ["--fixtures"]
        assert args[1] == str(Path("/h") / "system" / "optim"
                              / TOOL_FIXTURES_BASENAME)
        miner = Path("scripts/mine_tool_fixtures.py").read_text()
        assert "gate_contract.TOOL_FIXTURES_BASENAME" in miner, (
            "the miner's default output no longer shares the launcher's "
            "basename — mine to one file, gate on another")

    def test_the_signatures_are_the_allow_list(self):
        """Exactly the three §4DF signatures plus the tool-description
        gate — a grown list means a design decision, not drift."""
        assert A.OPTIMIZER_TARGETS == (
            "tool_descriptions",
            "gepa:planning.decompose",
            "gepa:tool_selection.pick",
            "gepa:reflection.critique",
        )


# ─────────────────────────────────────────────────────────────────────
# Kill switches and preflight
# ─────────────────────────────────────────────────────────────────────

class TestKillSwitchesAndPreflight:
    def test_the_master_switch_kills_the_launcher(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.setenv("GHOST_GEPA_AUTONOMY", "0")
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls

    def test_the_job_switch_kills_ONLY_the_launcher(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setenv("GHOST_GEPA_AUTO_OPTIMIZE", "0")
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls
        # The file-reader jobs stay armed.
        assert A.autonomy_enabled() and A.auto_revert_enabled()

    def test_low_RAM_stands_the_launcher_down(self, tmp_path,
                                              monkeypatch):
        """§4U: hours of unattended main-slot replays get the RAM floor
        the file-reader jobs deliberately skip."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        import psutil as _ps
        monkeypatch.setattr(
            _ps, "virtual_memory",
            lambda: SimpleNamespace(available=100 * 1e6))
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls
        assert any("stood down" in l and "RAM" in l for l in r.logs), \
            r.logs
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert st["optimizer"]["last_outcome"] == "stood_down"

    def test_an_unreadable_RAM_reading_REPORTS_not_clears(
            self, tmp_path, monkeypatch):
        """Fail CLOSED (`replay_engine`'s psutil rule): a preflight that
        cannot read a precondition must report that, never clear an
        hours-long launch."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        import psutil as _ps

        def _boom():
            raise OSError("forced")
        monkeypatch.setattr(_ps, "virtual_memory", _boom)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls
        assert any("could not read available RAM" in l
                   for l in r.logs), r.logs

    def test_a_missing_psutil_stands_down_not_launches(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        monkeypatch.setitem(sys.modules, "psutil", None)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls

    def test_the_disk_floor_is_shared_with_the_other_jobs(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        monkeypatch.setattr(A, "MIN_DISK_FREE_MB", 10 ** 9)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        assert not run.calls


# ─────────────────────────────────────────────────────────────────────
# The markers have one home
# ─────────────────────────────────────────────────────────────────────

class TestTheMarkersHaveOneHome:
    """The §4DA shape-1 defect: a marker string restated in a script
    breaks the consumer the day either side is edited. The scripts must
    PRINT through the contract constants."""

    def test_run_gepa_prints_through_the_constants(self):
        src = Path("scripts/run_gepa.py").read_text()
        for const in ("GATE_RUN_BANNER_GEPA", "GATE_PROMOTED_MARKER_GEPA",
                      "GATE_REJECTED_MARKER", "GATE_NO_CANDIDATE_MARKER"):
            assert f"gate_contract.{const}" in src, (
                f"run_gepa.py no longer prints through {const}")

    def test_the_otd_gate_prints_through_the_constants(self):
        src = Path("scripts/optimize_tool_descriptions.py").read_text()
        for const in ("GATE_RUN_BANNER_OTD", "GATE_PROMOTED_MARKER_OTD",
                      "GATE_REJECTED_MARKER", "GATE_NO_CANDIDATE_MARKER"):
            assert f"gate_contract.{const}" in src, (
                f"optimize_tool_descriptions.py no longer prints "
                f"through {const}")

    def test_the_rejected_marker_matches_what_both_gates_said(self):
        """The constants were LIFTED from the scripts' existing output,
        not invented — the operator-facing lines must not have
        changed."""
        assert GATE_REJECTED_MARKER == "A/B gate REJECTED"
        assert GATE_PROMOTED_MARKER_GEPA.startswith("A/B gate PASSED")
        assert GATE_PROMOTED_MARKER_OTD == "PROMOTED "
        assert GATE_NO_CANDIDATE_MARKER == "NO CANDIDATE"


# ─────────────────────────────────────────────────────────────────────
# The real subprocesses
# ─────────────────────────────────────────────────────────────────────

class TestTheRealWiring:
    """⚠ `the-fix-severs-what-it-feeds`: a stubbed runner proves the
    mapping, not the argv/env the real scripts parse. Both gates refuse
    an empty home with exit 2 in seconds (no model, no network) — and
    the refusal must arrive WITH the banner, or the launcher files a
    healthy refusal as an instrument failure."""

    def test_the_real_otd_gate_refuses_an_empty_home_quietly(
            self, tmp_path):
        home = _home(tmp_path)
        (Path(home) / "system" / "llm_recordings").mkdir(parents=True)
        r = _Recorder()
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              now=1_000_000)
        assert out == ("tool_descriptions", GateExit.COULD_NOT_MEASURE), \
            (out, r.logs, r.notes)
        assert not r.notes, (
            "a genuine could-not-measure was filed as an instrument "
            "failure — the banner did not print before the refusal: "
            + str(r.notes))

    def test_the_real_run_gepa_refuses_an_empty_home_quietly(
            self, tmp_path):
        home = _home(tmp_path)
        # Park the otd target as fresh so the round-robin picks the
        # first run_gepa signature.
        st_path = Path(home) / "system" / "gepa_autonomy_state.json"
        st_path.write_text(json.dumps({"optimizer": {"per_target": {
            "tool_descriptions": {"last_attempt_epoch": 999_000}}}}))
        r = _Recorder()
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              now=1_000_000)
        assert out == ("gepa:planning.decompose",
                       GateExit.COULD_NOT_MEASURE), (out, r.logs, r.notes)
        assert not r.notes, r.notes


# ─────────────────────────────────────────────────────────────────────
# The tick wiring
# ─────────────────────────────────────────────────────────────────────

class TestTheTickWiring:
    """The phase is an `if` block in `_biological_tick`, DEEP idle only
    (> 1h) — unlike the file-reader jobs' (15m, 1h] band."""

    def _agent(self, idle=4000):
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=idle)
        # Disarm the neighbouring deep-idle phases (lens B, C1).
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        return agent

    @pytest.mark.asyncio
    async def test_deep_idle_fires_and_the_cooldown_holds(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []

        def _spy(home, **kw):
            calls.append((home, threading.get_ident()))
            return ("tool_descriptions", 2)
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer", _spy)
        agent = self._agent(idle=4000)
        agent._last_gepa_optimizer_at = datetime.datetime.min
        await agent._biological_tick()
        assert len(calls) == 1 and calls[0][0] == str(tmp_path), calls
        # to_thread: the hours-long subprocess wait is OFF the loop.
        assert calls[0][1] != threading.get_ident(), (
            "run_optimizer ran ON the event-loop thread")
        assert agent._last_gepa_optimizer_at != datetime.datetime.min
        await agent._biological_tick()
        assert len(calls) == 1, "the tick-level cooldown did not hold"

    @pytest.mark.asyncio
    async def test_the_shallow_idle_band_does_NOT_launch(
            self, monkeypatch, tmp_path):
        """idle=2000 is the file-reader band — an hours-long main-slot
        run must not start there."""
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer",
            lambda home, **kw: calls.append(home) or None)
        agent = self._agent(idle=2000)
        agent._last_gepa_optimizer_at = datetime.datetime.min
        await agent._biological_tick()
        assert calls == [], calls

    @pytest.mark.asyncio
    async def test_a_FRESH_agent_does_not_fire_in_the_first_hour(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer",
            lambda home, **kw: calls.append(home) or None)
        agent = self._agent(idle=4000)
        await agent._biological_tick()
        assert calls == [], calls

    @pytest.mark.asyncio
    async def test_both_kill_switches_stop_the_phase_at_the_tick(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer",
            lambda home, **kw: calls.append(home) or None)
        for flag in ("GHOST_GEPA_AUTONOMY", "GHOST_GEPA_AUTO_OPTIMIZE"):
            monkeypatch.setenv(flag, "0")
            agent = self._agent(idle=4000)
            agent._last_gepa_optimizer_at = datetime.datetime.min
            await agent._biological_tick()
            monkeypatch.delenv(flag)
        assert calls == [], calls

    def test_the_launcher_is_the_LAST_phase_in_the_tick(self):
        """⚠ STRUCTURAL, deliberately. The optimizer await can hold one
        tick for HOURS; any phase after it runs on hours-stale
        idle_secs/foreground state — self-play launching into live
        traffic. So the block sits after every other `_idle_ran` phase
        and before only the idle-cycle summary. (Serialized ticks also
        keep the other idle phases off the inference slot while the
        optimizer owns it — that half is a feature.)"""
        src = Path("src/ghost_agent/core/agent.py").read_text()
        opt = src.index("§4DF Phase 3: the autonomous optimizer")
        # ⚠ EVERY `_idle_ran.append` except the optimizer's own (round
        # 2, MIN-5): the first version enumerated 6 phases and a NEW
        # phase relocated after the block passed — `mark-it-where-you-
        # catch-it`: scan the marker itself, not a hand-kept list.
        idx, found = 0, 0
        while True:
            idx = src.find("_idle_ran.append(", idx)
            if idx < 0:
                break
            line_end = src.index(")", idx)
            snippet = src[idx:line_end + 1]
            if "gepa-optimizer" not in snippet:
                found += 1
                assert idx < opt, (
                    f"a phase ({snippet}) now runs AFTER the hours-long "
                    f"optimizer await, on stale idle state")
            idx = line_end
        assert found >= 6, (
            f"the scan found only {found} phase markers — the marker "
            f"idiom changed and this pin went blind")
        assert opt < src.index('logger.info("idle cycle: ran %s'), (
            "the optimizer phase moved past the idle-cycle summary")

    @pytest.mark.asyncio
    async def test_a_raising_launcher_does_not_kill_the_tick(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))

        def _boom(home, **kw):
            raise RuntimeError("forced")
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer", _boom)
        agent = self._agent(idle=4000)
        agent._last_gepa_optimizer_at = datetime.datetime.min
        await agent._biological_tick()          # must not raise
        assert agent._last_gepa_optimizer_at != datetime.datetime.min


# ─────────────────────────────────────────────────────────────────────
# The liveness probe
# ─────────────────────────────────────────────────────────────────────

class TestTheLivenessProbe:
    def test_the_probe_watches_the_third_job(self, tmp_path):
        import time as _t

        from ghost_agent.core.liveness import (
            FIRED, _gepa_autonomy_probe)
        d = tmp_path / "system"
        d.mkdir(parents=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 86400},
              "optimizer": {"last_run_epoch": _t.time() - 10 * 86400,
                            "last_outcome": "nothing_due"}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        # 10 days is FRESH against the 21d bound — the job legitimately
        # decides "nothing due" most days and deep idle can be scarce.
        assert res.status == FIRED and res.count == 3, (res.status,
                                                        res.note)

    def test_a_stopped_optimizer_schedule_alarms(self, tmp_path):
        import time as _t

        from ghost_agent.core.liveness import _gepa_autonomy_probe
        d = tmp_path / "system"
        d.mkdir(parents=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 86400},
              "optimizer": {"last_run_epoch": _t.time() - 25 * 86400}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert "SCHEDULE STOPPED" in res.note and "optimizer" in res.note


# ─────────────────────────────────────────────────────────────────────
# §4DF round 1 — the reviewer's findings, each pinned
# ─────────────────────────────────────────────────────────────────────

class TestRoundOneCadencePins:
    def test_of_TWO_eligible_stale_targets_the_OLDER_goes_first(self):
        """⚠ BATTERY SURVIVOR (round 1, MUT-A): flipping oldest-first to
        freshest-first survived — the existing staleness test had exactly
        ONE eligible target, a verification that cannot distinguish. The
        distinguishing world: every target stale after an outage."""
        now = 10_000_000
        per = {
            "tool_descriptions": {"last_attempt_epoch": now - 8 * 86400},
            "gepa:planning.decompose":
                {"last_attempt_epoch": now - 20 * 86400},
            "gepa:tool_selection.pick":
                {"last_attempt_epoch": now - 86400},
            "gepa:reflection.critique":
                {"last_attempt_epoch": now - 86400},
        }
        assert A._pick_target(per, now) == "gepa:planning.decompose", (
            "with two eligible stale targets the rotation did not pick "
            "the OLDEST — after an outage the order inverts")

    def test_a_stand_down_does_NOT_consume_the_targets_7d_window(
            self, tmp_path, monkeypatch):
        """⚠ BATTERY SURVIVOR (round 1, MUT-C): stamping
        `last_attempt_epoch` before the preflight survived — a
        RAM-blocked week would then 'use up' every target's window
        without ever running a gate. A stand-down is not an attempt."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        import psutil as _ps
        monkeypatch.setattr(
            _ps, "virtual_memory",
            lambda: SimpleNamespace(available=100 * 1e6))
        r = _Recorder()
        run = _gate(GateExit.COULD_NOT_MEASURE, out="refused")
        assert A.run_optimizer(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000) is None
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert not st["optimizer"].get("per_target"), (
            "the stand-down stamped a per-target attempt")
        # RAM recovers: the very next due day the target actually runs.
        monkeypatch.setattr(
            _ps, "virtual_memory",
            lambda: SimpleNamespace(available=8_000 * 1e6))
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run,
                              now=1_000_000 + A.OPTIMIZER_INTERVAL_S + 1)
        assert out is not None and run.calls, (
            "the blocked day consumed the window")

    def test_two_consecutive_promotions_BOTH_notify(self, tmp_path,
                                                    monkeypatch):
        """⚠ BATTERY SURVIVOR (round 1, MUT-G): routing the promotion
        notify through `_notify_once` survived — the second weekly
        promotion of the same target would then be silently swallowed
        by the standing 'promoted' condition."""
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.PROMOTED, out=_PROMOTED_OTD)
        week = A.OPTIMIZER_TARGET_INTERVAL_S + 1
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run, now=1_000_000)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run, now=1_000_000 + week)
        assert len(r.notes) == 2, (
            f"two promotions produced {len(r.notes)} notification(s) — "
            f"a real deploy went unannounced")

    def test_a_NaN_stamp_is_never_at_BOTH_levels(self, tmp_path,
                                                 monkeypatch):
        """MIN-6: NaN fails every comparison — a hand-edited NaN parked
        the whole job (job level) or silently starved ONE target forever
        (target level), unalarmed either way. Non-finite means never."""
        nan = float("nan")
        assert A._due({"optimizer": {"last_run_epoch": nan}},
                      "optimizer", A.OPTIMIZER_INTERVAL_S, 1_000_000), (
            "a NaN job clock parked the job forever")
        per = {"tool_descriptions": {"last_attempt_epoch": nan}}
        monkeypatch.setattr(A, "OPTIMIZER_TARGETS",
                            ("tool_descriptions",))
        assert A._pick_target(per, 1_000_000) == "tool_descriptions", (
            "a NaN target stamp starved the target forever")

    def test_every_gepa_target_is_a_registered_signature(self):
        """MIN-8: a SIGNATURES rename makes run_gepa's argparse exit 2
        BEFORE the banner — one 'no-banner' notification, then a
        permanently dead target. The allow-list must be a subset."""
        from ghost_agent.optim.signatures import SIGNATURES
        for target in A.OPTIMIZER_TARGETS:
            if target.startswith("gepa:"):
                sig = target.split(":", 1)[1]
                assert sig in SIGNATURES, (
                    f"{target} names a signature the registry does not "
                    f"have — argparse will refuse it before the banner")


class TestPartialPromotionIsNamedNotDenied:
    """§4DF round 1, MAJOR-4 (consumer half): a PROMOTED marker beside a
    non-zero exit is a PARTIAL promotion — components that reached disk
    ARE live via the epoch swap, and the generic 'nothing was believed
    or acted on' text was a false claim about exactly this world."""

    def test_promoted_marker_with_exit_1_notifies_PARTIAL(
            self, tmp_path, monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED,
                    out=_PROMOTED_OTD + "\nTraceback ...")
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1
        assert "PARTIAL" in r.notes[0], r.notes
        assert "LIVE" in r.notes[0], r.notes
        assert "nothing was believed or acted on" not in r.notes[0], (
            "the notification denies a promotion that is deploying")

    def test_a_clean_rejection_is_still_not_partial(self, tmp_path,
                                                    monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED, out=_REJECTED)
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=run, now=1_000_000)
        assert out == ("tool_descriptions", GateExit.REJECTED)
        assert not r.notes


class TestTheTickOuterWatchdog:
    @pytest.mark.asyncio
    async def test_the_outer_bound_covers_the_six_hour_child(
            self, monkeypatch, tmp_path):
        """⚠ BATTERY SURVIVOR (round 1, MUT-B): shrinking the outer
        watchdog to 600s survived — the awaiter would be killed at
        10min while the 6h child keeps running: daily 'HUNG' notifies
        beside orphan state writes (the lens-B/B4 shape)."""
        import asyncio
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_optimizer",
            lambda home, **kw: ("tool_descriptions", 2))
        seen = []
        _real = asyncio.wait_for

        async def _spy(aw, timeout=None):
            seen.append(timeout)
            return await _real(aw, timeout=timeout)
        monkeypatch.setattr(asyncio, "wait_for", _spy)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=4000)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        agent._last_gepa_optimizer_at = datetime.datetime.min
        await agent._biological_tick()
        assert A.OPTIMIZER_TIMEOUT_S + 600 in seen, (
            f"the launcher's outer watchdog does not cover the 6h "
            f"child: {seen}")


class TestRunGepaRefusesADeadUpstream:
    """§4DF round 1, CRIT-1 (the preflight half): run_gepa had NO
    reachability check before paying for the optimizer — a dead or
    TLS-only port burned the full run and terminated wearing a benign
    code. The refusal is a 1-token ping down the SAME client the run
    uses, before the optimizer."""

    def test_a_dead_upstream_refuses_BEFORE_the_optimizer(
            self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (_corpus, _drive,
                                                   _result)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)

        def _boom(payload):
            raise ConnectionRefusedError("8080 is the TLS console")
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=None, chat=_boom)
        assert rc == GateExit.COULD_NOT_MEASURE, rc
        captured = capsys.readouterr()
        assert "1-token ping" in captured.err, captured.err[-500:]
        assert seen["run_gepa"] == 0, (
            "the optimizer was paid for despite a dead upstream")
        # And the refusal is quiet at the launcher: exit 2 with banner.
        assert GATE_RUN_BANNER_GEPA in captured.out

    def test_a_live_upstream_passes_the_ping_transparently(
            self, tmp_path):
        """The negative control: the same drive with an answering chat
        reaches the optimizer (seen elsewhere too, but this pins the
        ping itself as non-blocking)."""
        from tests.test_gepa_optim_reaudit import (_corpus, _drive,
                                                   _result, _ties)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=_ties)
        assert seen["run_gepa"] == 1, (rc, seen)


class TestTheSeedVetoPrintsItsMarker:
    """⚠ BATTERY SURVIVOR (round 1, MUT-D): deleting the §4DF-added
    stdout REJECTED line on the seed-veto path survived every suite —
    the one-home test was satisfied by the OTHER print sites
    (`token-pins-vs-executed-pins`). Executed: a real seed-veto rc==1
    must carry the marker, or the launcher files a healthy verdict as
    a broken instrument."""

    def test_a_seed_vetoed_rejection_carries_the_marker_on_stdout(
            self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import _ship_run, _two_arm
        rc, _out, _seen = _ship_run(tmp_path, _two_arm(
            main_delta=0.20, main_ships=True, main_p=0.0078, main_cw=8,
            seed_delta=-0.30, seed_wins=12, cand_wins=0))
        assert rc == GateExit.REJECTED, rc
        captured = capsys.readouterr()
        assert GATE_REJECTED_MARKER in captured.out, (
            "a real seed-veto verdict has no stdout marker — the "
            "launcher will notify it as an instrument failure: "
            + captured.out[-400:])


class TestTheOtdVerdictMatchesItsExit:
    """§4DF round 1, MAJOR-2: the old print order emitted a verdict
    marker and THEN decided the exit, so an underpowered run carried
    'A/B gate REJECTED' beside exit 2 — and §4DF had just made that
    string load-bearing. Executed through the real main()."""

    def test_an_undecidable_run_prints_ABORTED_not_a_verdict(
            self, tmp_path, monkeypatch, capsys):
        """The round-17 seed-outage world: a re-promotion whose seed arm
        is gutted by transport refuses with rc==2 — and under the old
        print order that refusal carried the REJECTED verdict marker."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        rc0, _l0, _r0, _n0 = _H()._run(tmp_path, monkeypatch,
                                       cand_wins=6, n_fixtures=70)
        capsys.readouterr()
        assert rc0 == 0, "the setup promotion did not ship"
        rc, _live, _rej, _n = _H()._run(tmp_path, monkeypatch,
                                        cand_wins=6, n_fixtures=70,
                                        transport=55,
                                        transport_arm="seed")
        captured = capsys.readouterr()
        assert rc == GateExit.COULD_NOT_MEASURE, (rc, captured.out[-400:])
        assert "A/B gate ABORTED" in captured.out, captured.out[-400:]
        assert GATE_REJECTED_MARKER not in captured.out, (
            "exit 2 (nothing measured) carries a REJECTED verdict "
            "marker")
        assert GATE_NO_CANDIDATE_MARKER not in captured.out

    def test_a_real_rejection_still_prints_the_marker(
            self, tmp_path, monkeypatch, capsys):
        """The other direction: a measured loss (rc==1) keeps its
        marker — the restructure must not have silenced the verdict."""
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        rc, _live, _rej, _n = _H()._run(tmp_path, monkeypatch,
                                        cand_wins=0, inc_wins=8)
        captured = capsys.readouterr()
        assert rc == GateExit.REJECTED, (rc, captured.out[-400:])
        assert GATE_REJECTED_MARKER in captured.out


class TestPromotionIsAllOrNothing:
    """§4DF round 1, MAJOR-4 (script half): the old loop backed up and
    REPLACED per component, so an OSError on component N left components
    1..N-1 live — a partial promotion the exit-1 'nothing was acted on'
    notification then denied. Now every failure-prone write happens
    before the first rename."""

    def test_an_abort_mid_set_leaves_NOTHING_promoted(
            self, tmp_path, monkeypatch, capsys):
        import shutil as _sh
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        pre = {}

        def _om(mod):
            base = mod._baseline_descriptions()
            tools = sorted(base)[:2]
            d = tmp_path / "home" / "system" / "optim"
            d.mkdir(parents=True, exist_ok=True)
            for t in tools:
                _f = d / f"tool_description.{t}.json"
                _f.write_text(json.dumps({
                    "signature_name": f"tool_description.{t}",
                    "optimized_instruction": base[t] + " OLD"}))
                pre[f"tool_description.{t}"] = _f.read_text()
            calls = {"n": 0}
            _real = _sh.copy2

            def _copy(src, dst, **kw):
                calls["n"] += 1
                if calls["n"] >= 2:
                    raise OSError("ENOSPC (forced)")
                return _real(src, dst, **kw)
            monkeypatch.setattr(_sh, "copy2", _copy)
        with pytest.raises(OSError):
            _H()._run(tmp_path, monkeypatch, cand_wins=6, n_tools=2,
                      mutate=2, on_module=_om)
        capsys.readouterr()
        d = tmp_path / "home" / "system" / "optim"
        assert pre, "the world never installed its incumbents"
        for t, text in pre.items():
            assert (d / f"{t}.json").read_text() == text, (
                f"{t} was PROMOTED before the sibling's abort — the "
                f"partial-promotion world is back")
        assert not list(d.glob("*.staging")), (
            "the abort left staged files behind")


class TestTheProbeNamesTheRightRemedy:
    """§4DF round 1, MIN-5: the stood-down note suggested 'kill switch?
    idle window never sampled?' — two causes the state it just read
    excludes (the tick DID reach the phase; the preflight refused)."""

    def test_a_stood_down_job_points_at_the_preflight(self, tmp_path):
        import time as _t

        from ghost_agent.core.liveness import _gepa_autonomy_probe
        d = tmp_path / "system"
        d.mkdir(parents=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 86400},
              "optimizer": {"last_run_epoch": _t.time() - 60,
                            "last_outcome": "stood_down"}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert "STANDING DOWN" in res.note and "optimizer" in res.note
        assert "preflight refuses" in res.note, res.note
        # The wrong remedies must not be suggested FOR this job: the
        # stall wording appears only when a stall exists.
        assert "kill switch" not in res.note, res.note

    def test_a_NaN_job_clock_reads_never_not_nand(self, tmp_path):
        import time as _t

        from ghost_agent.core.liveness import _gepa_autonomy_probe
        d = tmp_path / "system"
        d.mkdir(parents=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 86400},
              "optimizer": {"last_run_epoch": float("nan")}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert "optimizer (never)" in res.note, res.note
        assert "nand" not in res.note


# ─────────────────────────────────────────────────────────────────────
# §4DF round 2 — the round-1 fixes' own gaps, each pinned
# ─────────────────────────────────────────────────────────────────────

class TestThePromoteMarkerIsExecutedNotMentioned:
    """⚠ Round 2, MAJOR-1 (battery MB7): deleting the OTD PROMOTED
    print survived 405 tests — the one-home checks are token scans and
    every consumer test drives a STUB. In that world every genuine
    autonomous promotion is notified as 'not a promotion' while the
    epoch swap deploys it. This drives the REAL main() to rc==0 and
    reads the marker off captured stdout."""

    def test_a_real_promotion_prints_the_marker_per_component(
            self, tmp_path, monkeypatch, capsys):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        rc, live, _rej, _n = _H()._run(tmp_path, monkeypatch,
                                       cand_wins=6, n_tools=2, mutate=2)
        captured = capsys.readouterr()
        assert rc == 0 and len(live) == 2, (rc, live)
        lines = [l for l in captured.out.splitlines()
                 if l.startswith(GATE_PROMOTED_MARKER_OTD)]
        assert len(lines) == 2, (
            "a real 2-component promotion did not print the PROMOTED "
            "marker per component — the launcher would file the deploy "
            "as an instrument failure: " + captured.out[-400:])
        for path in live:
            assert any(str(path) in l for l in lines), (path, lines)


class TestNoCandidateOutranksUnderpowered:
    """⚠ Round 2, MAJOR-2 (battery MB5+MB6): the verdict order is
    stated twice (print chain + literal return), the comment claims the
    conformance scan forces agreement, and BOTH divergence directions
    survived — every tested no-candidate world had underpowered=False,
    a verification that cannot distinguish. This world has both flags
    true: seed returned verbatim AND transport gutting the usable pairs
    below the evidence bar."""

    def test_the_combined_world_is_exit_3_with_the_NO_CANDIDATE_marker(
            self, tmp_path, monkeypatch, capsys):
        from tests.test_4da_tool_desc_ship_gate import (
            TestTheDecisionIsActuallyUSED as _H)
        rc, _live, _rej, _n = _H()._run(tmp_path, monkeypatch,
                                        mutate=False, transport=15)
        captured = capsys.readouterr()
        assert rc == GateExit.NO_CANDIDATE, (
            f"no-candidate + underpowered returned {rc} — the return "
            f"conditional's precedence flipped: " + captured.out[-400:])
        assert GATE_NO_CANDIDATE_MARKER in captured.out, (
            "the print chain's precedence flipped — exit 3 without its "
            "marker is filed as a crash: " + captured.out[-400:])
        assert "A/B gate ABORTED" not in captured.out, captured.out[-400:]
        assert GATE_REJECTED_MARKER not in captured.out


class TestThePingGuardsAndBound:
    """Round 2, MIN-4: the ping's two unpinned properties (battery
    MB1/MB2). The real client answers a 200-with-error-JSON body as a
    dict WITHOUT 'choices' — the guard is load-bearing; and the 30s
    outer bound is the new sibling of MUT-B's unpinned watchdog."""

    def test_a_choices_less_reply_refuses_not_proceeds(
            self, tmp_path, capsys):
        from tests.test_gepa_optim_reaudit import (_corpus, _drive,
                                                   _result)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc, seen = _drive(
            ["--signature", "planning.decompose",
             "--trajectories", str(tmp_path / "traj"),
             "--output", str(out), "--ab-min-delta", "0.05"],
            gepa_result=_result(), comparison=None,
            chat=lambda payload: {"error": "upstream broke",
                                  "_ghost_leg": {}})
        assert rc == GateExit.COULD_NOT_MEASURE, rc
        assert seen["run_gepa"] == 0, (
            "an error-shaped 200 reply cleared the ping — the guard on "
            "'choices' is decorative")
        assert "unexpected reply" in capsys.readouterr().err

    def test_the_ping_bound_is_30s_not_the_inner_watchdog(
            self, tmp_path, monkeypatch):
        import asyncio as _aio
        from tests.test_gepa_optim_reaudit import (_corpus, _drive,
                                                   _result, _ties)
        _corpus(tmp_path / "traj")
        out = tmp_path / "optim" / "planning.decompose.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        seen_t = []
        _real = _aio.wait_for

        async def _spy(aw, timeout=None):
            seen_t.append(timeout)
            return await _real(aw, timeout=timeout)
        monkeypatch.setattr(_aio, "wait_for", _spy)
        _drive(["--signature", "planning.decompose",
                "--trajectories", str(tmp_path / "traj"),
                "--output", str(out), "--ab-min-delta", "0.05"],
               gepa_result=_result(), comparison=_ties)
        assert 30.0 in seen_t, (
            f"the ping's outer bound is not 30s — a black-holed "
            f"upstream rides to the 6h inner watchdog: {seen_t}")


class TestPartialDeploysAlwaysNotify:
    """Round 2, MIN-2: the deploy-bearing partial arm rode
    `_notify_once`, so the second of two consecutive partial deploys
    was swallowed — MUT-G's law applied to the clean arm only."""

    def test_two_partial_promotions_BOTH_notify(self, tmp_path,
                                                monkeypatch):
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()
        run = _gate(GateExit.REJECTED,
                    out=_PROMOTED_OTD + "\nTraceback ...")
        week = A.OPTIMIZER_TARGET_INTERVAL_S + 1
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run, now=1_000_000)
        A.run_optimizer(home, notify=r.notify, log=r.log,
                        run_script=run, now=1_000_000 + week)
        assert len(r.notes) == 2, (
            f"two partial deploys produced {len(r.notes)} "
            f"notification(s) — a live deploy went unannounced")

    def test_a_timeout_kill_after_PROMOTED_lines_is_partial(
            self, tmp_path, monkeypatch):
        """The hypothesis arm, executed: `TimeoutExpired.stdout` holds
        everything printed before the kill; discarding it filed a
        post-promotion timeout as 'nothing was acted on'."""
        import subprocess as _sp
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()

        def _boom(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(
                cmd=[script], timeout=timeout_s,
                output=(GATE_RUN_BANNER_OTD + "\n"
                        + _PROMOTED_OTD + "\n").encode())
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=_boom, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert len(r.notes) == 1 and "PARTIAL" in r.notes[0], r.notes
        assert "nothing was believed or acted on" not in r.notes[0]


class TestTheProbeCannotBeMaskedByAStandDown:
    """⚠ Round 2, MAJOR-3 (live): the round-1 wording fix checked
    `stood_down` before staleness, so a job whose LAST act was a
    stand-down could never raise SCHEDULE STOPPED — the 100d-stopped
    and 5m-fresh worlds produced byte-identical notes, each asserting
    in the present tense a fact the state contradicts by 100 days."""

    def _state(self, tmp_path, age_s):
        import time as _t
        d = tmp_path / "system"
        d.mkdir(parents=True, exist_ok=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 86400},
              "optimizer": {"last_run_epoch": _t.time() - age_s,
                            "last_outcome": "stood_down"}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))

    def test_a_FRESH_stand_down_reads_standing_not_stopped(
            self, tmp_path):
        from ghost_agent.core.liveness import _gepa_autonomy_probe
        self._state(tmp_path, 300)
        res = _gepa_autonomy_probe(tmp_path)
        assert "STANDING DOWN" in res.note and "optimizer" in res.note
        assert "SCHEDULE STOPPED" not in res.note, res.note

    def test_a_STALE_stand_down_is_a_stopped_schedule(self, tmp_path):
        from ghost_agent.core.liveness import _gepa_autonomy_probe
        self._state(tmp_path, 100 * 86400)
        res = _gepa_autonomy_probe(tmp_path)
        assert "SCHEDULE STOPPED" in res.note, res.note
        assert "optimizer (100.0d ago" in res.note, res.note
        assert "stand-down" in res.note, (
            "the stopped-schedule row hides that the last act was a "
            "stand-down: " + res.note)
        # And the two worlds are DISTINGUISHABLE now.
        self._state(tmp_path, 300)
        fresh_note = _gepa_autonomy_probe(tmp_path).note
        self._state(tmp_path, 100 * 86400)
        stale_note = _gepa_autonomy_probe(tmp_path).note
        assert fresh_note != stale_note


class TestTheSiblingHandlersHarvestToo:
    """⚠ Round 3, MAJOR-1: the round-2 harvest fix was applied to one
    of THREE identical exception handlers in autonomy.py — and the
    unharvested one belonged to the only child that ACTS. A timeout
    kill landing after `gepa_live_check --revert`'s rename left the
    retirement ON DISK (the §4DE swap deploys it in ~a minute) while
    the notification said 'no action taken'."""

    def test_a_timeout_kill_after_the_RETIREMENT_notifies_it(
            self, tmp_path):
        import subprocess as _sp

        from ghost_agent.optim.gate_contract import (
            JUDGE_RETIRED_MARKER, JUDGE_RUN_BANNER, JudgeExit)
        from tests.test_gepa_autonomy_phase01 import _home as _p01_home
        home = _p01_home(tmp_path, live=("planning.decompose",))
        r = _Recorder()

        def _kill(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(
                cmd=[script], timeout=timeout_s,
                output=(JUDGE_RUN_BANNER + " planning.decompose\n"
                        + JUDGE_RETIRED_MARKER + " x.json\n").encode())
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=_kill, now=1_000_000)
        assert out == {"planning.decompose": JudgeExit.NO_LONGER_WINS}
        assert len(r.notes) == 1 and "RETIRED" in r.notes[0], r.notes
        assert "no action taken" not in r.notes[0], (
            "the notification denies a retirement the epoch swap is "
            "deploying")

    def test_a_timeout_with_NO_output_is_still_an_instrument_failure(
            self, tmp_path):
        import subprocess as _sp

        from tests.test_gepa_autonomy_phase01 import _home as _p01_home
        home = _p01_home(tmp_path, live=("planning.decompose",))
        r = _Recorder()

        def _hang(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(cmd=[script], timeout=timeout_s)
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=_hang, now=1_000_000)
        assert out == {"planning.decompose": None}
        assert len(r.notes) == 1
        assert "instrument failure" in r.notes[0]
        # And the two worlds are DISTINGUISHABLE.
        assert "RETIRED on disk" not in r.notes[0]

    def test_a_timeout_after_a_COMPLETED_mine_names_its_cause(
            self, tmp_path):
        """MIN-3: the supply watch's third sibling — a kill after the
        pool write + DONE marker is a completed mine whose exit code
        died with the kill; it must be named distinctly, not filed as
        the same 'TimeoutExpired' a hung mine produces (a later
        different failure must re-notify)."""
        import subprocess as _sp

        from ghost_agent.optim.gate_contract import (
            MINER_DONE_MARKER, MINER_RUN_BANNER)
        from tests.test_gepa_autonomy_phase01 import _home as _p01_home
        home = _p01_home(tmp_path)
        r = _Recorder()

        def _kill(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(
                cmd=[script], timeout=timeout_s,
                output=(MINER_RUN_BANNER + " x\nLabels: {}\n"
                        + MINER_DONE_MARKER + " parked\n").encode())
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=_kill, now=1_000_000)
        assert len(r.notes) == 1
        assert "cause=timeout-after-complete" in r.notes[0], r.notes
        assert "COMPLETED before the kill" in r.notes[0]

    def test_a_harvested_marker_without_its_banner_is_not_believed(
            self, tmp_path, monkeypatch):
        """The harvest holds the live path's discipline: a marker with
        no proof the script started is believed nowhere — for the
        optimizer's PROMOTED and the judge's RETIRED both."""
        import subprocess as _sp

        from ghost_agent.optim.gate_contract import (
            JUDGE_RETIRED_MARKER, )
        from tests.test_gepa_autonomy_phase01 import _home as _p01_home
        # Optimizer: PROMOTED line, no banner -> plain TimeoutExpired.
        _one_target(monkeypatch)
        home = _home(tmp_path)
        r = _Recorder()

        def _kill_o(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(cmd=[script], timeout=timeout_s,
                                     output=_PROMOTED_OTD.encode())
        out = A.run_optimizer(home, notify=r.notify, log=r.log,
                              run_script=_kill_o, now=1_000_000)
        assert out == ("tool_descriptions", None)
        assert "PARTIAL" not in r.notes[0], r.notes
        # Judge: RETIRED marker, no banner -> instrument failure.
        home2 = _p01_home(tmp_path / "j", live=("planning.decompose",))
        r2 = _Recorder()

        def _kill_j(script, args, *, home, timeout_s):
            raise _sp.TimeoutExpired(
                cmd=[script], timeout=timeout_s,
                output=(JUDGE_RETIRED_MARKER + " x.json\n").encode())
        out2 = A.run_live_judge(home2, notify=r2.notify, log=r2.log,
                                run_script=_kill_j, now=1_000_000)
        assert out2 == {"planning.decompose": None}
        assert "instrument failure" in r2.notes[0]
