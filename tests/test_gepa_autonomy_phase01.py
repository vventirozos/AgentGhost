"""§4DC Phase 0+1 — the supply watch and the live judge, driven.

The jobs consume the instruments' EXIT CONTRACTS — the surface eighteen
§4DA rounds hardened — so the pins here are about the CONSUMER side:
each declared code maps to exactly one action, undeclared codes are
instrument failures that act on nothing, notifications fire on
TRANSITIONS not ticks, cadence is wall-clock and survives restarts, and
the kill switches actually kill.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.optim import autonomy as A
from ghost_agent.optim.gate_contract import JudgeExit


class _Recorder:
    def __init__(self):
        self.notes, self.logs = [], []

    def notify(self, msg):
        self.notes.append(msg)

    def log(self, msg):
        self.logs.append(msg)


def _script(rc, out="Labels: {'positive': 40}\nmine complete: parked",
            raw=False, err=""):
    """A stub subprocess whose stdout matches what the REAL scripts
    print — run BANNER first (proof the script started; argparse and the
    interpreter share exit codes with the verdicts), then the payload. A
    stub whose output the real pipeline cannot produce is
    `harness-grades-own-homework`. `raw=True` suppresses the banners for
    the tests ABOUT missing banners/markers."""
    from ghost_agent.optim.gate_contract import (
        JUDGE_RUN_BANNER, MINER_RUN_BANNER)
    calls = []

    def run(script, args, *, home, timeout_s):
        calls.append((script, list(args), home, timeout_s))
        _rc = rc(script, args) if callable(rc) else rc
        _out = out
        if not raw:
            banner = (JUDGE_RUN_BANNER if "live_check" in script
                      else MINER_RUN_BANNER)
            _out = banner + " stub\n" + _out
        return SimpleNamespace(returncode=_rc, stdout=_out, stderr=err)
    run.calls = calls
    return run


def _home(tmp_path, live=()):
    home = tmp_path / "home"
    d = home / "system" / "optim"
    d.mkdir(parents=True)
    for sig in live:
        (d / f"{sig}.json").write_text(json.dumps({
            "signature_name": sig, "optimized_instruction": "T"}))
    return str(home)


# ─────────────────────────────────────────────────────────────────────
# Supply watch
# ─────────────────────────────────────────────────────────────────────

class TestSupplyWatch:
    def test_parked_is_log_only_and_ready_notifies_ONCE(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        # Two parked weeks: no notification.
        for wk in (0, 1):
            rc = A.run_supply_watch(
                home, notify=r.notify, log=r.log,
                run_script=_script(A.MINER_PARKED),
                now=1_000_000 + wk * A.SUPPLY_WATCH_INTERVAL_S)
            assert rc == A.MINER_PARKED
        assert r.notes == [], r.notes
        # The gate opens: exactly one notification…
        rc = A.run_supply_watch(
            home, notify=r.notify, log=r.log,
            run_script=_script(A.MINER_READY,
                   out="Labels: {'positive': 240}\nmine complete: ready"),
            now=1_000_000 + 2 * A.SUPPLY_WATCH_INTERVAL_S)
        assert rc == A.MINER_READY
        assert len(r.notes) == 1 and "SUPPLY GATE OPEN" in r.notes[0]
        # …and a second READY week does not re-notify.
        A.run_supply_watch(
            home, notify=r.notify, log=r.log,
            run_script=_script(A.MINER_READY,
                   out="Labels: {'positive': 240}\nmine complete: ready"),
            now=1_000_000 + 3 * A.SUPPLY_WATCH_INTERVAL_S)
        assert len(r.notes) == 1, r.notes

    def test_the_ready_edge_REARMS_after_a_fallback(self, tmp_path):
        """A pool that falls back below the gate (era cutoff, corpus
        loss) must notify again when it recovers — a fire-once that
        never re-arms goes silent forever."""
        home = _home(tmp_path)
        r = _Recorder()
        t = [1_000_000]

        def tick(rc, out="Labels: x\nmine complete: parked"):
            t[0] += A.SUPPLY_WATCH_INTERVAL_S
            return A.run_supply_watch(home, notify=r.notify, log=r.log,
                                      run_script=_script(rc, out=out),
                                      now=t[0])
        tick(A.MINER_READY, out="Labels: x\nmine complete: ready")
        tick(A.MINER_PARKED)
        tick(A.MINER_READY, out="Labels: x\nmine complete: ready")
        assert len(r.notes) == 2, r.notes

    def test_cadence_is_wall_clock_and_persisted(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(A.MINER_PARKED)
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000) is not None
        # A day later (agent restarted — fresh call, same state file):
        # not due.
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run,
                                  now=1_000_000 + 86_400) is None
        assert len(run.calls) == 1
        # A week later: due.
        assert A.run_supply_watch(
            home, notify=r.notify, log=r.log, run_script=run,
            now=1_000_000 + A.SUPPLY_WATCH_INTERVAL_S) is not None
        assert len(run.calls) == 2

    def test_a_FUTURE_last_run_is_a_clock_jump_not_a_recent_run(
            self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(A.MINER_PARKED)
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=run, now=2_000_000)
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000) is not None

    def test_an_instrument_failure_notifies_once_and_acts_on_nothing(
            self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        t = [1_000_000]
        for _ in range(2):
            t[0] += A.SUPPLY_WATCH_INTERVAL_S
            A.run_supply_watch(home, notify=r.notify, log=r.log,
                               run_script=_script(A.MINER_NO_CORPUS),
                               now=t[0])
        assert len(r.notes) == 1 and "instrument failure" in r.notes[0]

    def test_a_timeout_is_an_instrument_failure_not_a_crash(self,
                                                            tmp_path):
        home = _home(tmp_path)
        r = _Recorder()

        def boom(script, args, *, home, timeout_s):
            raise subprocess.TimeoutExpired(cmd=script, timeout=timeout_s)
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                run_script=boom, now=1_000_000)
        assert rc is None
        assert len(r.notes) == 1 and "TimeoutExpired" in r.notes[0]

    def test_a_BANNERLESS_miner_exit_is_an_instrument_failure(
            self, tmp_path):
        """Lens A, A-1's miner half: argparse's exit 2 is also the
        no-corpus code."""
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(A.MINER_PARKED, out="usage: ...", raw=True)
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                run_script=run, now=1_000_000)
        assert rc is None
        assert len(r.notes) == 1 and "did not start" in r.notes[0]

    def test_the_kill_switch_kills(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_GEPA_AUTONOMY", "0")
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(A.MINER_READY, out="Labels: x\nmine complete: ready")
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000,
                                  force=True) is None
        assert not run.calls and not r.notes


# ─────────────────────────────────────────────────────────────────────
# Live judge
# ─────────────────────────────────────────────────────────────────────

class TestLiveJudge:
    def test_each_declared_code_maps_to_its_action(self, tmp_path):
        """KEEP/could-not-measure: log-only. REVERT: notify (retired).
        Race: notify. One tick, four signatures, four codes."""
        home = _home(tmp_path, live=("sig_keep", "sig_revert",
                                     "sig_thin", "sig_race"))
        codes = {"sig_keep": JudgeExit.STILL_WINS,
                 "sig_revert": JudgeExit.NO_LONGER_WINS,
                 "sig_thin": JudgeExit.COULD_NOT_MEASURE,
                 "sig_race": JudgeExit.REPORTED_NOT_ACTED}

        def rc(script, args):
            return codes[args[args.index("--signature") + 1]]
        r = _Recorder()
        # The stub's stdout carries the judge's markers — exit 1 without
        # them is (correctly) an instrument failure now, since a crash
        # also exits 1.
        run = _script(rc, out="REVERT: t 2/20 vs c 15/20\n"
                              "RETIRED ON DISK: a -> b")
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000)
        assert out == codes
        assert len(r.notes) == 2, r.notes
        assert any("RETIRED on disk" in n and "sig_revert" in n
                   for n in r.notes)
        assert any("vanished mid-run" in n and "sig_race" in n
                   for n in r.notes)
        # Every invocation carried --revert (auto-revert default ON) and
        # the right home.
        for _s, args, h, _t in run.calls:
            assert "--revert" in args and h == home

    def test_a_retirement_notifies_ONCE_not_daily(self, tmp_path):
        """The notify-once key is the artifact FILE's sha, not the
        signature — so the same artifact never re-notifies, and a NEW
        artifact under the same signature (re-promotion) that loses
        again is a new retirement and notifies again (driven in the
        test below) — lens B, C2: the first version keyed per signature
        and this docstring claimed a contract the code did not have."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS,
                      out="RETIRED ON DISK: a -> b")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        # Same artifact still on disk (retire pending restart-less world),
        # judged again a day later: no second notification.
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run,
                         now=1_000_000 + A.LIVE_JUDGE_INTERVAL_S)
        assert len(r.notes) == 1, r.notes
        # A KEEP re-arms, and a later REVERT notifies again.
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(JudgeExit.STILL_WINS),
                         now=1_000_000 + 2 * A.LIVE_JUDGE_INTERVAL_S)
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run,
                         now=1_000_000 + 3 * A.LIVE_JUDGE_INTERVAL_S)
        assert len(r.notes) == 2, r.notes

    def test_a_REPROMOTED_artifact_that_loses_notifies_AGAIN(
            self, tmp_path):
        """Retire -> re-promote (new bytes) -> retire: two retirements,
        two notifications, no intervening KEEP required."""
        home = _home(tmp_path, live=("planning.decompose",))
        art = Path(home) / "system" / "optim" / "planning.decompose.json"
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS,
                      out="RETIRED ON DISK: a -> b")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        # The re-promotion: same signature, different artifact bytes.
        art.write_text(json.dumps({
            "signature_name": "planning.decompose",
            "optimized_instruction": "A DIFFERENT CANDIDATE"}))
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run,
                         now=1_000_000 + A.LIVE_JUDGE_INTERVAL_S)
        assert len(r.notes) == 2, (
            "a second, distinct retirement was silently swallowed")

    def test_report_only_mode_drops_the_flag_and_says_so(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setenv("GHOST_GEPA_AUTO_REVERT", "0")
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS,
                      out="REVERT: t 2/20 vs c 15/20")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        _s, args, _h, _t = run.calls[0]
        assert "--revert" not in args
        assert len(r.notes) == 1 and "report-only" in r.notes[0]
        assert "Nothing was retired" in r.notes[0]

    def test_an_undeclared_code_acts_on_nothing_and_notifies_once(
            self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        for day in (1, 2):
            A.run_live_judge(home, notify=r.notify, log=r.log,
                             run_script=_script(7),
                             now=1_000_000 + day * A.LIVE_JUDGE_INTERVAL_S)
        # (message reworded in round 3's cause-keyed rewrite)
        assert len(r.notes) == 1 and "cause=rc-7" in r.notes[0], r.notes

    def test_no_live_artifacts_is_quiet(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(0)
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=run, now=1_000_000)
        assert out == {} and not run.calls and not r.notes

    def test_the_state_file_is_not_judged_as_a_signature(self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        # Force the state file into existence first.
        r = _Recorder()
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(0), now=1_000_000)
        assert A.live_signatures(home) == ["planning.decompose"]

    def test_suffix_conventions_are_not_live(self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        d = Path(home) / "system" / "optim"
        for junk in ("planning.decompose.json.prev",
                     "x.json.candidate.rejected",
                     "y.json.retired-live-20260101T000000Z",
                     "z.json.notready", "w.json.staging"):
            (d / junk).write_text("{}")
        assert A.live_signatures(home) == ["planning.decompose"]

    def test_the_master_kill_switch_kills_the_judge_too(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv("GHOST_GEPA_AUTONOMY", "0")
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS)
        assert A.run_live_judge(home, notify=r.notify, log=r.log,
                                run_script=run, now=1_000_000,
                                force=True) is None
        assert not run.calls


# ─────────────────────────────────────────────────────────────────────
# The real subprocess boundary — no stubs.
# ─────────────────────────────────────────────────────────────────────

class TestTheRealWiring:
    """⚠ `the-fix-severs-what-it-feeds`: a stubbed runner proves the
    mapping, not the argv/env the real scripts parse. These run the REAL
    scripts against a tmp GHOST_HOME (both are file readers — no model,
    no network)."""

    def test_the_real_judge_runs_and_returns_a_declared_code(
            self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        (Path(home) / "system" / "trajectories").mkdir(parents=True)
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               now=1_000_000)
        # An empty trajectory corpus is COULD_NOT_MEASURE — and the
        # artifact must still be there (nothing retires on no data).
        assert out == {"planning.decompose": JudgeExit.COULD_NOT_MEASURE}
        assert (Path(home) / "system" / "optim"
                / "planning.decompose.json").exists()
        assert not r.notes

    def test_the_real_miner_reports_no_corpus_honestly(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc == A.MINER_NO_CORPUS
        assert len(r.notes) == 1 and "instrument failure" in r.notes[0]


# ─────────────────────────────────────────────────────────────────────
# The tick wiring — the phase inside _biological_tick.
# ─────────────────────────────────────────────────────────────────────

class TestTheTickWiring:
    """The phase is an `if` block in `_biological_tick`; these drive the
    real tick with the jobs stubbed at the module the tick imports."""

    def _agent(self, idle=2000):
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=idle)
        # ⚠ DISARM THE NEIGHBOURING PHASES. `_make_agent` leaves
        # self-play and dream armed (production shape), and at idle=4000
        # the real self-play phase fires on a 20% dice, crashes on the
        # MagicMock and re-raises BY DESIGN — a ~1-in-5-red pin about a
        # phase these tests are not testing (lens B, C1, executed: 4
        # failures in 12 runs). These tests are about the GEPA phase
        # gate arithmetic only.
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        return agent

    @pytest.mark.asyncio
    async def test_an_armed_phase_calls_both_jobs_with_the_home(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, **kw: calls.append(("judge", home)) or {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: calls.append(("supply", home)) or 1)
        agent = self._agent()
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()
        assert ("judge", str(tmp_path)) in calls, calls
        assert ("supply", str(tmp_path)) in calls, calls
        # Anchor advanced — the next tick within the hour must not re-run.
        assert agent._last_gepa_autonomy_at != datetime.datetime.min
        calls.clear()
        await agent._biological_tick()
        assert calls == [], "the tick-level cooldown did not hold"

    @pytest.mark.asyncio
    async def test_a_FRESH_agent_does_not_fire_in_the_first_hour(
            self, monkeypatch, tmp_path):
        """⚠ The lazy anchor is `now`, NOT datetime.min — a min anchor
        fired subprocess spawns in the first idle window after every
        boot, and in every tick-level test whose mocked home would eat a
        real spawn. Nothing is lost: due-ness is persisted."""
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, **kw: calls.append("judge") or {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: calls.append("supply") or 1)
        agent = self._agent()
        await agent._biological_tick()
        assert calls == [], calls

    @pytest.mark.asyncio
    async def test_the_kill_switch_stops_the_phase_at_the_tick(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        monkeypatch.setenv("GHOST_GEPA_AUTONOMY", "0")
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, **kw: calls.append("judge") or {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: calls.append("supply") or 1)
        agent = self._agent()
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()
        assert calls == [], calls

    @pytest.mark.asyncio
    async def test_a_raising_job_does_not_kill_the_tick(self, monkeypatch,
                                                        tmp_path):
        """FAIL LOUD, survive — the negctrl rule. The tick must complete
        and the anchor must advance so a broken job cannot re-fire every
        60 seconds."""
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))

        def _boom(home, **kw):
            raise RuntimeError("forced")
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge", _boom)
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch", _boom)
        agent = self._agent()
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()          # must not raise
        assert agent._last_gepa_autonomy_at != datetime.datetime.min

    @pytest.mark.asyncio
    async def test_outside_the_idle_window_the_phase_stands_down(
            self, monkeypatch, tmp_path):
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, **kw: calls.append("judge") or {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: calls.append("supply") or 1)
        for idle in (100, 4000):        # below the window, above it
            agent = self._agent(idle=idle)
            agent._last_gepa_autonomy_at = datetime.datetime.min
            await agent._biological_tick()
        assert calls == [], calls


class TestThePreflight:
    """§4U trimmed to a file-reader job: disk only. `replay_engine`'s
    full preflight gates on Docker, which would stand these jobs down
    whenever OrbStack is off, for no reason."""

    def test_low_disk_stands_both_jobs_down(self, tmp_path, monkeypatch):
        home = _home(tmp_path)
        monkeypatch.setattr(A, "MIN_DISK_FREE_MB", 10 ** 9)
        r = _Recorder()
        run = _script(A.MINER_READY)
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000) is None
        assert not run.calls
        assert any("stood down" in l for l in r.logs), r.logs
        home2 = _home(tmp_path / "b", live=("planning.decompose",))
        run2 = _script(0)
        assert A.run_live_judge(home2, notify=r.notify, log=r.log,
                                run_script=run2, now=1_000_000) is None
        assert not run2.calls

    def test_an_unreadable_preflight_REPORTS_not_clears(self, tmp_path,
                                                        monkeypatch):
        """A preflight that cannot read a precondition must report that,
        not clear the launch (`replay_engine`'s psutil rule)."""
        home = _home(tmp_path)
        import shutil as _sh

        def _boom(path):
            raise OSError("forced")
        monkeypatch.setattr(_sh, "disk_usage", _boom)
        r = _Recorder()
        run = _script(A.MINER_READY)
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run, now=1_000_000) is None
        assert not run.calls
        assert any("could not read disk" in l for l in r.logs), r.logs

    def test_a_clear_preflight_runs(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        run = _script(A.MINER_PARKED)
        assert A.run_supply_watch(home, notify=r.notify, log=r.log,
                                  run_script=run,
                                  now=1_000_000) == A.MINER_PARKED
        assert run.calls


class TestTheLivenessProbe:
    def test_a_fresh_box_notes_never_ticked(self, tmp_path):
        from ghost_agent.core.liveness import NO_SOURCE, _gepa_autonomy_probe
        res = _gepa_autonomy_probe(tmp_path)
        assert res.status == NO_SOURCE and "never ticked" in res.note

    def test_a_recent_tick_counts_and_a_stale_one_alarms(self, tmp_path):
        import time as _t

        from ghost_agent.core.liveness import (
            FIRED, ZERO, _gepa_autonomy_probe)
        d = tmp_path / "system"
        d.mkdir(parents=True)
        # Judge ran an hour ago (fresh, bound 3d); supply 10 days ago
        # (fresh, bound 21d). The state file lives at system/ — NOT
        # system/optim/, whose *.json glob means "live artifact" (B1).
        st = {"live_judge": {"last_run_epoch": _t.time() - 3600},
              "supply_watch": {"last_run_epoch": _t.time() - 10 * 86400}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert res.status == FIRED and res.count == 2, (res.status,
                                                        res.note)
        # Judge stalls past 3x its daily cadence: the schedule stopped.
        st["live_judge"]["last_run_epoch"] = _t.time() - 5 * 86400
        st["supply_watch"]["last_run_epoch"] = _t.time() - 30 * 86400
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert res.status == ZERO and res.count == 0, res.note
        assert "SCHEDULE STOPPED" in res.note
        assert "not reaching the phase" in res.note

    def test_a_standing_down_job_is_NOT_advancing(self, tmp_path):
        """⚠ Lens B, B2: the jobs stamp last_run_epoch BEFORE the
        preflight, so a box below the disk floor read FIRED for months
        while nothing ran."""
        import time as _t

        from ghost_agent.core.liveness import ZERO, _gepa_autonomy_probe
        d = tmp_path / "system"
        d.mkdir(parents=True)
        st = {"live_judge": {"last_run_epoch": _t.time() - 60,
                             "last_outcome": "stood_down"},
              "supply_watch": {"last_run_epoch": _t.time() - 60,
                               "last_outcome": "stood_down"}}
        (d / "gepa_autonomy_state.json").write_text(json.dumps(st))
        res = _gepa_autonomy_probe(tmp_path)
        assert res.status == ZERO, (res.status, res.note)
        assert "STANDING DOWN" in res.note


class TestACrashIsNotARetirement:
    """⚠ Lens B, A1 — EXECUTED on the pre-fix tree: a judge child that
    crashed on NotADirectoryError exits 1 (Python's uncaught-exception
    code, which is also the judge's most actionable declared code), and
    the caller notified "RETIRED on disk" about an untouched artifact —
    then the false `retired` condition suppressed the REAL retirement's
    notification forever, while the traceback sat on the stderr nobody
    read. The action-bearing codes are marker-gated now."""

    def test_a_crashing_REAL_judge_is_an_instrument_failure(
            self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        # The crash: the trajectory ROOT is a file, so the collector
        # raises NotADirectoryError deep inside the script.
        (Path(home) / "system" / "trajectories").write_text("not a dir")
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               now=1_000_000)
        assert out == {"planning.decompose": None}, out
        assert (Path(home) / "system" / "optim"
                / "planning.decompose.json").exists(), (
            "a crash was allowed to look like a retirement")
        assert len(r.notes) == 1, r.notes
        assert "not a declared code" in r.notes[0] \
            or "marker" in r.notes[0], r.notes[0]
        assert "RETIRED on disk" not in r.notes[0], r.notes[0]
        # The stderr traceback — the only clue — must ride the note.
        assert "Error" in r.notes[0] or "Traceback" in r.notes[0], (
            r.notes[0])

    def test_a_marker_less_exit_1_never_claims_retirement(self, tmp_path):
        """The stub arm of the same pin: exit 1 with empty stdout."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS, out="")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        assert "RETIRED on disk" not in r.notes[0], r.notes[0]
        assert "crash, not a verdict" in r.notes[0], r.notes[0]

    def test_a_BANNERLESS_exit_is_never_a_verdict_at_all(self, tmp_path):
        """⚠ Lens A, A-1: exit 2 is argparse's code, the interpreter's
        "can't open file" code, AND the judge's benign "could not
        measure yet" — so a moved script or renamed argument read as
        thin data, log-only, FOREVER, while REVERTs that should happen
        never did. Any code without the run banner is an instrument
        failure."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        for rc in (JudgeExit.COULD_NOT_MEASURE, JudgeExit.STILL_WINS):
            run = _script(rc, out="usage: gepa_live_check ...", raw=True)
            out = A.run_live_judge(home, notify=r.notify, log=r.log,
                                   run_script=run, now=1_000_000,
                                   force=True)
            assert out == {"planning.decompose": None}, (rc, out)
        assert len(r.notes) == 1, r.notes      # once per condition
        assert "did not start" in r.notes[0], r.notes[0]

    def test_a_MISSING_judge_script_is_loud(self, tmp_path, monkeypatch):
        """The real impersonation, driven: point the runner at a script
        path that does not exist — CPython exits 2 with the diagnosis on
        stderr and nothing on stdout."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()

        def _run_missing(script, args, *, home, timeout_s):
            return A._run_script("scripts/__no_such_script__.py", args,
                                 home=home, timeout_s=timeout_s)
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=_run_missing, now=1_000_000)
        assert out == {"planning.decompose": None}, out
        assert len(r.notes) == 1 and "did not start" in r.notes[0], (
            r.notes)
        assert "No such file" in r.notes[0] or "Errno" in r.notes[0] \
            or "can't open" in r.notes[0], (
            "the stderr diagnosis did not ride the note: " + r.notes[0])

    def test_a_crashing_REAL_miner_is_not_parked(self, tmp_path):
        """The miner's sibling: a crash also exits 1 = 'parked', which
        would file a persistent instrument failure as the supply steady
        state, logged quietly forever."""
        home = _home(tmp_path)
        # recordings root exists (so the no-corpus exit-2 path is not
        # taken) but trajectories is a FILE -> the join crashes.
        rec = Path(home) / "system" / "llm_recordings"
        rec.mkdir(parents=True)
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        (Path(home) / "system" / "trajectories").write_text("not a dir")
        r = _Recorder()
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc is None, rc
        assert len(r.notes) == 1 and "instrument failure" in r.notes[0]


class TestAMidRunPromotionIsNotRetired:
    """⚠ Lens B, A2: the judge derives the live sha at run start, walks
    the corpus, then renamed WHATEVER file is at the path — so a
    promotion completing mid-run had its fresh, gate-passed artifact
    retired on the OLD artifact's evidence. The script re-derives the
    sha immediately before the rename now and exits 3 on mismatch."""

    def test_the_script_refuses_to_retire_a_swapped_artifact(
            self, tmp_path, capsys):
        import importlib.util
        import sys as _sys

        from ghost_agent.optim import live_check as _LC
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        art_dir = home / "system" / "optim"
        art_dir.mkdir(parents=True)
        art = art_dir / "planning.decompose.json"
        art.write_text(json.dumps({"optimized_instruction": "T"}))
        import hashlib as _hl
        sha = _hl.sha256(b"T").hexdigest()[:8]
        rows = []
        for arm, (p_, f_) in (("treatment", (2, 18)),
                              ("control", (15, 5))):
            from types import SimpleNamespace as _NS
            rows += [_NS(outcome="passed",
                         extra={"optim_artifacts": {
                             "planning.decompose": {"sha": sha,
                                                    "arm": arm}}})
                     for _ in range(p_)]
            rows += [_NS(outcome="failed",
                         extra={"optim_artifacts": {
                             "planning.decompose": {"sha": sha,
                                                    "arm": arm}}})
                     for _ in range(f_)]
        spec = importlib.util.spec_from_file_location(
            "glc_swap", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_swap"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        _real_verdict = _LC.verdict

        def _swapping_verdict(*a, **kw):
            # The mid-run promotion: a NEW artifact lands at the path.
            art.write_text(json.dumps(
                {"optimized_instruction": "A FRESH GATE-PASSED ONE"}))
            return _real_verdict(*a, **kw)
        _LC.verdict = _swapping_verdict
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         "planning.decompose", "--revert"]
            rc = mod.main()
        finally:
            _sys.argv = old_argv
            _LC.verdict = _real_verdict
        out = capsys.readouterr()
        assert "REVERT" in out.out
        assert rc == 3, (rc, out.err)
        assert art.exists(), (
            "the fresh artifact was retired on the old one's evidence")
        assert "no longer the one this verdict measured" in out.err, (
            out.err)


# ─────────────────────────────────────────────────────────────────────
# Round-1 lens A survivors — the writer halves, pinned.
# ─────────────────────────────────────────────────────────────────────

class TestTheWriterHalves:
    """⚠ Lens A's battery: several round-1 fixes shipped with their
    READER half pinned and their WRITER half free — deleting the write
    left the suite green while the defect the fix closed returned
    (`pin-both-halves-of-every-identity`)."""

    def test_a_stand_down_WRITES_stood_down(self, tmp_path, monkeypatch):
        """N09: the probe pin hand-wrote its fixture, so deleting
        `job["last_outcome"] = "stood_down"` survived — and the
        disk-starved-but-green probe defect returned undetected."""
        home = _home(tmp_path, live=("planning.decompose",))
        monkeypatch.setattr(A, "MIN_DISK_FREE_MB", 10 ** 9)
        r = _Recorder()
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=_script(A.MINER_PARKED),
                           now=1_000_000)
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(0), now=1_000_000)
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert st["supply_watch"]["last_outcome"] == "stood_down", st
        assert st["live_judge"]["last_outcome"] == "stood_down", st

    def test_a_run_WRITES_ran(self, tmp_path):
        """The pair — a job that ran must not read as standing down."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=_script(A.MINER_PARKED),
                           now=1_000_000)
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(0), now=1_000_000)
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert st["supply_watch"]["last_outcome"] == "ran", st
        assert st["live_judge"]["last_outcome"] == "ran", st

    def test_save_state_survives_a_failing_replace_and_stages_uniquely(
            self, tmp_path, monkeypatch):
        """N07/N08 + lens A B-2: `_save_state` raising crashed the jobs
        (the 'never raises' contract), and a shared staging name raced
        `os.replace` — per-PID did not separate same-PID pairs."""
        import os as _os
        seen = []
        _real = _os.replace

        def _capture(src, dst):
            seen.append(str(src))
            return _real(src, dst)
        monkeypatch.setattr(A.os, "replace", _capture)
        A._save_state(str(tmp_path), {"a": 1})
        A._save_state(str(tmp_path), {"a": 2})
        assert len(seen) == 2 and seen[0] != seen[1], (
            "two saves shared a staging name — same-PID writers race")
        assert f".{A.os.getpid()}." in seen[0], seen[0]

        def _boom(src, dst):
            raise OSError("forced")
        monkeypatch.setattr(A.os, "replace", _boom)
        A._save_state(str(tmp_path), {"a": 3})   # must not raise

    def test_the_inner_timeout_reaches_subprocess_run(self, monkeypatch,
                                                      tmp_path):
        """A18: the timeout test injected the exception rather than
        arming the mechanism — `timeout=None` at the subprocess survived.
        Pin the passthrough at the real boundary."""
        import subprocess as _sp
        seen = {}

        def _fake_run(argv, **kw):
            seen["timeout"] = kw.get("timeout")
            seen["env_home"] = kw.get("env", {}).get("GHOST_HOME")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        monkeypatch.setattr(A.subprocess, "run", _fake_run)
        A._run_script("scripts/gepa_live_check.py", [],
                      home=str(tmp_path), timeout_s=123.5)
        assert seen["timeout"] == 123.5, seen
        assert seen["env_home"] == str(tmp_path), seen

    def test_the_jobs_pass_their_OWN_deadlines(self, tmp_path):
        """The other half of A18: the constants must reach the calls."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(0)
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=run, now=1_000_000)
        deadlines = {s: t for s, _a, _h, t in run.calls}
        assert deadlines["scripts/gepa_live_check.py"] \
            == A.LIVE_JUDGE_TIMEOUT_S
        assert deadlines["scripts/mine_tool_fixtures.py"] \
            == A.SUPPLY_WATCH_TIMEOUT_S


class TestTheTickKeepsTheLoopFree:
    """⚠ Lens A T02/T03/T13/T14/T08: the tick stubs were sync lambdas,
    so `to_thread` removed (a 15-minute mine ON the event loop) looked
    identical — 'a verification that can't distinguish'. These pins
    distinguish."""

    @pytest.mark.asyncio
    async def test_the_job_runs_OFF_the_event_loop_thread(
            self, monkeypatch, tmp_path):
        """⚠ THREAD IDENTITY, NOT WALL-CLOCK GAPS. Two prior versions of
        this pin were wrong two different ways: a single early sleep
        sampled before the phase ran (the sync-call mutant survived —
        round-1 battery), and a whole-tick heartbeat with a 0.9s gap
        threshold flaked under a loaded full suite, where OTHER phases'
        legitimate sync work plus CPU starvation exceed the bar (the
        lens-B C1 class, reproduced by this test's own second version).
        The distinguishing observable is exact and load-free: under
        `asyncio.to_thread` the job executes on a WORKER thread; under
        the sync-call mutant it executes on the event-loop thread — and
        a 15-minute mine on that thread stalls every live request."""
        import datetime
        import threading
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        seen = {"judge": [], "supply": []}

        # ⚠ BOTH LEGS, EVERY CALL, EXACTLY ONE EACH. Round 3 (F4):
        # probing only the judge let the SUPPLY leg run sync on the
        # event loop with every test green; and recording one ident per
        # leg would miss a sync PRE-call added beside the to_thread call
        # (peer round-3 watch-item b — the surviving M16 shape).
        def _probe_judge(home, **kw):
            seen["judge"].append(threading.current_thread())
            return {}

        def _probe_supply(home, **kw):
            seen["supply"].append(threading.current_thread())
            return 1
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge", _probe_judge)
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            _probe_supply)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=2000)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        agent._last_gepa_autonomy_at = datetime.datetime.min
        _loop_thread = threading.current_thread()
        await agent._biological_tick()
        for leg in ("judge", "supply"):
            assert len(seen[leg]) == 1, (
                f"the {leg} leg ran {len(seen[leg])} times in one tick "
                f"— a sync pre-call beside the to_thread call is the "
                f"survivor shape this pin exists for")
            assert all(t is not _loop_thread for t in seen[leg]), (
                f"the {leg} leg ran ON the event-loop thread — the "
                f"to_thread wrapper is gone and a real mine would stall "
                f"every live request for 15 minutes")

    @pytest.mark.asyncio
    async def test_the_outer_watchdog_scales_with_the_live_count(
            self, monkeypatch, tmp_path):
        """T14: a fixed 4x multiplier survived — dozens of
        tool_description artifacts would trip daily false 'HUNG'
        notifies while the child chain kept running."""
        import asyncio
        import datetime
        home = tmp_path
        d = home / "system" / "optim"
        d.mkdir(parents=True)
        for i in range(7):
            (d / f"tool_description.t{i}.json").write_text("{}")
        monkeypatch.setenv("GHOST_HOME", str(home))
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, **kw: {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: 1)
        seen = []
        _real = asyncio.wait_for

        async def _spy(aw, timeout=None):
            seen.append(timeout)
            return await _real(aw, timeout=timeout)
        monkeypatch.setattr(asyncio, "wait_for", _spy)
        from ghost_agent.optim.autonomy import (
            LIVE_JUDGE_TIMEOUT_S, SUPPLY_WATCH_TIMEOUT_S)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=2000)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()
        assert LIVE_JUDGE_TIMEOUT_S * 8 + 60 in seen, (
            f"the judge's outer bound did not scale with 7 live "
            f"artifacts: {seen}")
        assert SUPPLY_WATCH_TIMEOUT_S + 60 in seen, seen

    @pytest.mark.asyncio
    async def test_a_notify_from_a_job_carries_severity_notify(
            self, monkeypatch, tmp_path):
        """T13: downgrading the tick's severity to "info" survived —
        one word, and every retirement notification silently demotes to
        digest-only, never reaching the Slack feed."""
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge",
            lambda home, notify, log, **kw: notify("A RETIREMENT") or {})
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch",
            lambda home, **kw: 1)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=2000)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        recorded = []
        monkeypatch.setattr(
            agent, "_record_autonomous_activity",
            lambda phase, msg, severity="info", **m: recorded.append(
                (phase, msg, severity)))
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()
        assert ("gepa_autonomy", "A RETIREMENT", "notify") in recorded, (
            f"the retirement did not ride severity='notify' — it will "
            f"never reach the Slack feed: {recorded}")

    @pytest.mark.asyncio
    async def test_a_raising_job_is_LOUD(self, monkeypatch, tmp_path):
        """T08: silencing the phase's except handler survived — the
        negctrl rule ('could not run' and 'ran, nothing to do' must
        never look alike) was restated in a comment, not checked."""
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))

        def _boom(home, **kw):
            raise RuntimeError("forced")
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_live_judge", _boom)
        monkeypatch.setattr(
            "ghost_agent.optim.autonomy.run_supply_watch", _boom)
        from tests.test_biological_watchdog import _make_agent
        agent = _make_agent(idle_seconds=2000)
        agent.context.args.no_self_play = True
        agent.context.args.no_dream = True
        lines = []
        monkeypatch.setattr(
            agent, "_safe_pretty_log",
            lambda title, msg, **kw: lines.append((title, msg,
                                                   kw.get("level"))))
        agent._last_gepa_autonomy_at = datetime.datetime.min
        await agent._biological_tick()
        assert any(t == "GEPA Autonomy" and lv == "ERROR"
                   and "FAILED" in m for t, m, lv in lines), (
            f"a raising job produced no ERROR line: {lines}")


# ─────────────────────────────────────────────────────────────────────
# Round 2 — the fix region's own defects, pinned.
# ─────────────────────────────────────────────────────────────────────

class TestAMarkerCertifiesTheOutcomeItFollows:
    """⚠ Round 2, F1 — executed on the pre-fix tree: `Labels:` prints
    BEFORE the supply gates and the pool write, so the real miner
    crashing AT the write (system/optim as a file) exited 1 with
    banner+marker present and was filed as the parked steady state —
    zero notifications, Phase 0's only action silently dead. The gate
    key is `mine complete:`, the miner's LAST line."""

    def test_a_crash_AT_the_write_is_an_instrument_failure(self,
                                                           tmp_path):
        home = tmp_path / "home"
        rec = home / "system" / "llm_recordings"
        rec.mkdir(parents=True)
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        (home / "system" / "trajectories").mkdir()
        # The crash site: the pool's parent dir is a FILE.
        (home / "system" / "optim").write_text("not a dir")
        r = _Recorder()
        rc = A.run_supply_watch(str(home), notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc is None, rc
        assert len(r.notes) == 1, r.notes
        assert "did not COMPLETE the mine" in r.notes[0] \
            or "instrument failure" in r.notes[0], r.notes[0]
        assert not any("still parked" in l for l in r.logs), (
            "a crash at the write was filed as the parked steady state: "
            + str(r.logs))

    def test_a_COMPLETED_real_park_is_still_parked(self, tmp_path):
        """The pair: a genuine parked mine (real miner, real corpus too
        thin for the gates) must NOT become an instrument failure."""
        home = tmp_path / "home"
        rec = home / "system" / "llm_recordings"
        rec.mkdir(parents=True)
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        (home / "system" / "trajectories").mkdir()
        (home / "system" / "optim").mkdir()
        r = _Recorder()
        rc = A.run_supply_watch(str(home), notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc == A.MINER_PARKED, (rc, r.logs)
        assert r.notes == [], r.notes
        assert any("still parked" in l for l in r.logs), r.logs


# ⚠ TWO SESSIONS FIXED ROUND 2 CONCURRENTLY and both added a class
# with this name — the second SHADOWED the first, silently dropping
# its driven report-only pin from collection while the total count
# (62) matched the recorded one, so nothing noticed (round 3, F2).
# Renamed; a duplicate class name in a test module is a dropped
# test suite wearing a green run's clothes.
class TestTheMarkersAreDrivenEndToEnd:
    """⚠ Round 2, MUT21: the report-only `REVERT:` marker existed only
    because the verdict f-string happened to use a colon — changing the
    separator left the suite green while every real report-only REVERT
    became an instrument failure. The scripts must PRINT through the
    contract constants; a restated string is the §4DA shape-1 defect."""

    @pytest.mark.parametrize("script,constants", [
        ("scripts/gepa_live_check.py",
         ("JUDGE_RETIRED_MARKER", "JUDGE_REVERT_MARKER",
          "JUDGE_RUN_BANNER")),
        ("scripts/mine_tool_fixtures.py",
         ("MINER_RAN_MARKER", "MINER_DONE_MARKER", "MINER_RUN_BANNER")),
    ])
    def test_the_scripts_print_via_the_constants(self, script, constants):
        src = Path(script).read_text()
        for c in constants:
            assert f"gate_contract.{c}" in src, (
                f"{script} does not print through gate_contract.{c} — "
                f"the marker is a restated string that can drift from "
                f"the consumer's check")

    def test_the_report_only_marker_is_DRIVEN_through_the_real_script(
            self, tmp_path, capsys, monkeypatch):
        """The executed half: a real report-only REVERT must carry the
        marker the consumer gates on."""
        monkeypatch.setenv("GHOST_GEPA_AUTO_REVERT", "0")
        import hashlib as _hl
        import importlib.util
        import sys as _sys
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        d = home / "system" / "optim"
        d.mkdir(parents=True)
        (d / "planning.decompose.json").write_text(
            json.dumps({"optimized_instruction": "T"}))
        sha = _hl.sha256(b"T").hexdigest()[:8]
        from types import SimpleNamespace as _NS
        rows = []
        for arm, (p_, f_) in (("treatment", (2, 18)),
                              ("control", (15, 5))):
            rows += [_NS(outcome="passed",
                         extra={"optim_artifacts": {
                             "planning.decompose": {"sha": sha,
                                                    "arm": arm}}})
                     for _ in range(p_)]
            rows += [_NS(outcome="failed",
                         extra={"optim_artifacts": {
                             "planning.decompose": {"sha": sha,
                                                    "arm": arm}}})
                     for _ in range(f_)]
        spec = importlib.util.spec_from_file_location(
            "glc_marker", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_marker"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        old = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         "planning.decompose"]
            rc = mod.main()
        finally:
            _sys.argv = old
        out = capsys.readouterr().out
        from ghost_agent.optim.gate_contract import (
            JUDGE_REVERT_MARKER, JUDGE_RUN_BANNER)
        assert rc == 1
        assert JUDGE_REVERT_MARKER in out, out
        assert JUDGE_RUN_BANNER in out, out


class TestTheNarrowResurrectionOfA1:
    """⚠ Round 2, MUT25: accepting EITHER judge marker on exit 1 passed
    the whole battery — i.e. 'auto-revert mode, exit 1, REVERT: present,
    RETIRED absent' (a crash between the verdict print and the rename)
    had no pin, and that is the narrowest resurrection of the false
    'RETIRED on disk' whose condition swallows the real retirement."""

    def test_a_verdict_without_a_rename_is_not_a_retirement(self,
                                                            tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS,
                      out="REVERT: t 2/20 vs c 15/20")   # no RETIRED
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        assert "RETIRED on disk" not in r.notes[0], r.notes[0]
        assert "crash, not a verdict" in r.notes[0], r.notes[0]


class TestTheBannerPrintsBeforeAnyIO:
    """⚠ Round 2, MUT22: banner-before-I/O was comment-enforced only —
    moved after the trajectory-root check, a no-root exit 2 misfiled as
    'did not start'. The real script with a MISSING root must still
    carry the banner (could-not-measure, not instrument failure)."""

    def test_a_missing_root_still_carries_the_banner(self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        # no trajectories dir at all
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               now=1_000_000)
        assert out == {"planning.decompose":
                       JudgeExit.COULD_NOT_MEASURE}, out
        assert not any("did not start" in n for n in r.notes), r.notes


class TestHandEditedStateCannotStallTheJobs:
    """⚠ Round 2, F2: `per_signature: "oops"` raised AttributeError,
    aborting the phase BEFORE the supply watch with no state save —
    both jobs stalled with hourly ERROR spam until hand repair. The B3
    coercion rule, one level below where round 1 applied it."""

    @pytest.mark.parametrize("poison", [
        {"live_judge": {"per_signature": "oops"}},
        {"live_judge": "oops"},
        {"supply_watch": 7, "live_judge": {"per_signature": None}},
    ])
    def test_every_level_coerces(self, tmp_path, poison):
        home = _home(tmp_path, live=("planning.decompose",))
        (Path(home) / "system" / "gepa_autonomy_state.json").write_text(
            json.dumps(poison))
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=_script(0), now=1_000_000)
        assert out is not None, "the judge refused a coercible state"
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                run_script=_script(A.MINER_PARKED),
                                now=1_000_000)
        assert rc == A.MINER_PARKED


class TestExitThreeNamesItsCause:
    """⚠ Round 2, F3: all three exit-3 causes said 'vanished mid-run' —
    including the sha mismatch, where nothing vanished: a promotion
    completed. And one shared condition meant a later different cause
    never re-notified."""

    def test_a_swap_says_promotion_and_a_vanish_says_vanished(
            self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run_swap = _script(
            JudgeExit.REPORTED_NOT_ACTED,
            out="REVERT: x\nno longer the one this verdict measured")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run_swap, now=1_000_000)
        assert len(r.notes) == 1 and "PROMOTION completed" in r.notes[0]
        run_vanish = _script(JudgeExit.REPORTED_NOT_ACTED,
                             out="REVERT: x\nnothing to retire at ...")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run_vanish,
                         now=1_000_000 + A.LIVE_JUDGE_INTERVAL_S)
        assert len(r.notes) == 2, r.notes
        assert "vanished" in r.notes[1], r.notes[1]


class TestTheJudgeStandDownWritesItsSummary:
    """Round 2, F5: the probe's note points at last_summary, which only
    the supply watch wrote."""

    def test_both_jobs_record_the_reason(self, tmp_path, monkeypatch):
        home = _home(tmp_path, live=("planning.decompose",))
        monkeypatch.setattr(A, "MIN_DISK_FREE_MB", 10 ** 9)
        r = _Recorder()
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(0), now=1_000_000)
        st = json.loads(
            (Path(home) / "system" / "gepa_autonomy_state.json")
            .read_text())
        assert "stood down" in st["live_judge"].get("last_summary", ""), (
            st)


class TestRoundTwoFindings:
    """§4DC round 2 — two code defects inside round 1's own fix region,
    plus the marker-chain battery survivors, each pinned by execution."""

    def test_a_crash_AT_THE_WRITE_is_an_instrument_failure(self,
                                                           tmp_path):
        """⚠ F1, the round-2 executed repro verbatim: `Labels:` printed
        before the pool write, so a crash AT the write exited 1 with
        banner+marker present — filed as parked forever, and Phase 0's
        only action (the supply-gate-open notification) silently dead.
        `system/optim` as a FILE makes the real miner crash exactly
        there."""
        home = _home(tmp_path)
        rec = Path(home) / "system" / "llm_recordings"
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        (Path(home) / "system" / "trajectories").mkdir(exist_ok=True)
        # the crash site: the pool write target's parent is a file
        import shutil as _sh
        _sh.rmtree(Path(home) / "system" / "optim")
        (Path(home) / "system" / "optim").write_text("a file")
        r = _Recorder()
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc is None, (
            f"a crash at the write was filed as a supply verdict "
            f"(rc={rc})")
        assert len(r.notes) == 1 and "instrument failure" in r.notes[0]
        assert not any("still parked" in l for l in r.logs), r.logs

    def test_a_STRING_per_signature_does_not_stall_both_jobs(
            self, tmp_path):
        """⚠ F2: `per_signature: "oops"` raised AttributeError out of
        the judge, aborting the phase BEFORE the supply watch with the
        state never saved — both jobs stalled, hourly ERROR spam, until
        hand repair. The B3 coercion rule, one level below where round 1
        applied it."""
        home = _home(tmp_path, live=("planning.decompose",))
        (Path(home) / "system" / "gepa_autonomy_state.json").write_text(
            json.dumps({"live_judge": {"per_signature": "oops"}}))
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               run_script=_script(0), now=1_000_000)
        assert out == {"planning.decompose": 0}, out

    def test_exit1_with_REVERT_but_no_RETIRED_in_autorevert_mode(
            self, tmp_path):
        """⚠ MUT25: accepting EITHER judge marker on exit 1 survived —
        the narrowest resurrection of lens B's A1 (a crash between the
        verdict print and the rename claims a retirement whose false
        condition swallows the real one)."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        run = _script(JudgeExit.NO_LONGER_WINS,
                      out="REVERT: t 2/20 vs c 15/20")   # no RETIRED
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        assert "RETIRED on disk" not in r.notes[0], r.notes[0]
        assert "crash, not a verdict" in r.notes[0], r.notes[0]

    def test_a_swap_exit3_names_the_promotion_not_a_vanish(self,
                                                           tmp_path):
        """⚠ F3: all three exit-3 causes said "vanished mid-run" — for
        the sha mismatch nothing vanished, a promotion completed; and
        the shared "race" condition meant a later different cause never
        re-notified."""
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        # ⚠ STDERR ONLY, like the real script (the diagnosis prints to
        # stderr) — with the phrase in stdout too, a stdout-only-slice
        # mutant in the consumer survives (peer round-3 watch-item a).
        run = _script(
            JudgeExit.REPORTED_NOT_ACTED,
            out="REVERT: t 2/20",
            err="the artifact at x is no longer the one this verdict "
                "measured (sha aaa != bbb)")
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=run, now=1_000_000)
        assert len(r.notes) == 1
        assert "PROMOTION completed mid-run" in r.notes[0], r.notes[0]
        assert "vanished" not in r.notes[0], r.notes[0]

    def test_the_judge_stood_down_state_carries_a_summary(self, tmp_path,
                                                          monkeypatch):
        """F5: the probe's note points at last_summary, which only the
        supply watch wrote."""
        home = _home(tmp_path, live=("planning.decompose",))
        monkeypatch.setattr(A, "MIN_DISK_FREE_MB", 10 ** 9)
        r = _Recorder()
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(0), now=1_000_000)
        st = json.loads((Path(home) / "system"
                         / "gepa_autonomy_state.json").read_text())
        assert "stood down" in st["live_judge"].get("last_summary", ""), st


class TestTheMarkersHaveOneHome:
    """⚠ MUT21/MUT22/MUT23: the marker STRINGS restated in a script (or
    produced by formatting coincidence) are the §4DA shape-1 defect —
    change the restatement and the consumer files every real verdict as
    an instrument failure. The scripts must PRINT through the contract
    constants, and the placement rules (banner before I/O, done-marker
    after the write) are pinned by real-subprocess behaviour."""

    def test_the_scripts_reference_every_constant_they_print(self):
        glc = Path("scripts/gepa_live_check.py").read_text()
        for const in ("JUDGE_RUN_BANNER", "JUDGE_RETIRED_MARKER",
                      "JUDGE_REVERT_MARKER"):
            assert f"gate_contract.{const}" in glc, (
                f"gepa_live_check.py no longer prints through "
                f"{const} — the marker is a restated string again")
        miner = Path("scripts/mine_tool_fixtures.py").read_text()
        for const in ("MINER_RUN_BANNER", "MINER_RAN_MARKER",
                      "MINER_DONE_MARKER"):
            assert f"gate_contract.{const}" in miner, const

    def test_a_real_park_run_carries_the_done_marker(self, tmp_path):
        """MUT23's behavioural half: a REAL completed mine (park path)
        must be accepted as a verdict — delete the DONE print and this
        run files as an instrument failure."""
        home = _home(tmp_path)
        rec = Path(home) / "system" / "llm_recordings"
        rec.mkdir(parents=True, exist_ok=True)
        # One recording day-file with no minable rows: the mine COMPLETES
        # (parks an empty pool) and must read as PARKED, not failure.
        (rec / "2026-08-01.jsonl").write_text("{}\n")
        (Path(home) / "system" / "trajectories").mkdir(exist_ok=True)
        r = _Recorder()
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                now=1_000_000)
        assert rc == A.MINER_PARKED, (rc, r.logs)
        assert not r.notes, r.notes
        assert any("still parked" in l for l in r.logs), r.logs

    def test_the_banner_precedes_the_trajectory_root_check(self,
                                                           tmp_path):
        """MUT22's behavioural half: a missing trajectory ROOT exits 2
        with a real diagnosis — the banner must already be out, or the
        caller files it as 'did not start'."""
        home = _home(tmp_path, live=("planning.decompose",))
        # no trajectories dir at all
        r = _Recorder()
        out = A.run_live_judge(home, notify=r.notify, log=r.log,
                               now=1_000_000)
        assert out == {"planning.decompose":
                       JudgeExit.COULD_NOT_MEASURE}, out
        assert not r.notes, (
            "a genuine could-not-measure was filed as an instrument "
            "failure — the banner printed after the root check: "
            + str(r.notes))


class TestDistinctInstrumentFailuresEachNotify:
    """⚠ Round 3, F1 — executed on the pre-fix tree: week-1 moved-script
    failure, week-2 crash-at-the-write → ONE notification, because every
    rc-to-None coercion shared the `instrument:None` condition. A
    different later cause is different news."""

    def test_two_different_supply_failures_notify_twice(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        # Week 1: script did not start (no banner).
        A.run_supply_watch(home, notify=r.notify, log=r.log,
                           run_script=_script(2, out="usage:", raw=True),
                           now=1_000_000)
        # Week 2: started but crashed at the write (banner, no DONE).
        A.run_supply_watch(
            home, notify=r.notify, log=r.log,
            run_script=_script(1, out="Labels: x"),
            now=1_000_000 + A.SUPPLY_WATCH_INTERVAL_S)
        assert len(r.notes) == 2, (
            "a second, DIFFERENT instrument failure was swallowed by the "
            "first one's condition: " + str(r.notes))
        # And the SAME cause repeating stays fire-once.
        A.run_supply_watch(
            home, notify=r.notify, log=r.log,
            run_script=_script(1, out="Labels: x"),
            now=1_000_000 + 2 * A.SUPPLY_WATCH_INTERVAL_S)
        assert len(r.notes) == 2, r.notes

    def test_two_different_judge_failures_notify_twice(self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(2, out="usage:", raw=True),
                         now=1_000_000)
        A.run_live_judge(home, notify=r.notify, log=r.log,
                         run_script=_script(JudgeExit.NO_LONGER_WINS,
                                            out=""),
                         now=1_000_000 + A.LIVE_JUDGE_INTERVAL_S)
        assert len(r.notes) == 2, r.notes


class TestRetiredPrintsOnlyAfterTheRename:
    """⚠ Peer round-3 F-C: the RETIRED marker is the consumer's proof of
    the ONE autonomous disk action — printed before the rename, a rename
    that raises (read-only dir; this box has a root-owned-file history)
    would notify "RETIRED on disk" about an artifact still serving."""

    def test_a_failing_rename_leaves_no_RETIRED_marker(self, tmp_path,
                                                       capsys):
        import importlib.util
        import os as _os
        import sys as _sys

        from ghost_agent.optim import live_check as _LC
        home = tmp_path / "home"
        (home / "system" / "trajectories").mkdir(parents=True)
        art_dir = home / "system" / "optim"
        art_dir.mkdir(parents=True)
        art = art_dir / "planning.decompose.json"
        art.write_text(json.dumps({"optimized_instruction": "T"}))
        import hashlib as _hl
        sha = _hl.sha256(b"T").hexdigest()[:8]
        from types import SimpleNamespace as _NS
        rows = []
        for arm, (p_, f_) in (("treatment", (2, 18)),
                              ("control", (15, 5))):
            rows += [_NS(outcome=o,
                         extra={"optim_artifacts": {
                             "planning.decompose": {"sha": sha,
                                                    "arm": arm}}})
                     for o, n in (("passed", p_), ("failed", f_))
                     for _ in range(n)]
        spec = importlib.util.spec_from_file_location(
            "glc_rofail", "scripts/gepa_live_check.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["glc_rofail"] = mod
        spec.loader.exec_module(mod)

        class _Coll:
            def __init__(self, **kw):
                pass

            def iter_trajectories(self):
                return iter(rows)
        mod.TrajectoryCollector = _Coll
        _real_verdict = _LC.verdict

        def _lockdown_verdict(*a, **kw):
            _os.chmod(art_dir, 0o555)     # the rename will raise
            return _real_verdict(*a, **kw)
        _LC.verdict = _lockdown_verdict
        old_argv = _sys.argv
        try:
            _sys.argv = ["glc", "--home", str(home), "--signature",
                         "planning.decompose", "--revert"]
            with pytest.raises(PermissionError):
                mod.main()
        finally:
            _os.chmod(art_dir, 0o755)
            _sys.argv = old_argv
            _LC.verdict = _real_verdict
        out = capsys.readouterr().out
        assert "RETIRED ON DISK" not in out, (
            "the marker printed before the rename — a failed rename "
            "would notify a retirement that never happened:\n" + out)
        assert art.exists()


class TestTheREADYLegIsDoneGatedToo:
    """⚠ Peer round-3 F-D: exit 0 = 'pool written'. Without the DONE
    marker that claim is unverified — a crash after choosing readiness
    but before the write must not fire the SUPPLY GATE OPEN
    notification."""

    def test_exit_0_without_the_done_marker_is_an_instrument_failure(
            self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        rc = A.run_supply_watch(home, notify=r.notify, log=r.log,
                                run_script=_script(A.MINER_READY,
                                                   out="Labels: x"),
                                now=1_000_000)
        assert rc is None, rc
        assert len(r.notes) == 1
        assert "SUPPLY GATE OPEN" not in r.notes[0], r.notes[0]
        assert "did not COMPLETE" in r.notes[0], r.notes[0]


class TestExceptionTypesAreDistinctCauses:
    """⚠ Verification pass over the peer's landing: the cause-keyed
    notify-once pins drove marker-cause pairs, so fixing the EXCEPTION
    path's cause to a constant survived 69 tests in BOTH jobs — a
    TimeoutExpired week one and an OSError week two would share one
    condition and the second would never notify. The exception TYPE is
    the cause, and two types are two conditions."""

    def test_two_exception_types_notify_twice_supply(self, tmp_path):
        home = _home(tmp_path)
        r = _Recorder()
        t = [1_000_000]

        def _raiser(exc):
            def run(script, args, *, home, timeout_s):
                raise exc
            return run
        for exc in (subprocess.TimeoutExpired(cmd="x", timeout=1),
                    OSError("disk went away")):
            t[0] += A.SUPPLY_WATCH_INTERVAL_S
            A.run_supply_watch(home, notify=r.notify, log=r.log,
                               run_script=_raiser(exc), now=t[0])
        assert len(r.notes) == 2, (
            "two DIFFERENT exception types shared one notify-once "
            "condition: " + str(r.notes))

    def test_two_exception_types_notify_twice_judge(self, tmp_path):
        home = _home(tmp_path, live=("planning.decompose",))
        r = _Recorder()
        t = [1_000_000]

        def _raiser(exc):
            def run(script, args, *, home, timeout_s):
                raise exc
            return run
        for exc in (subprocess.TimeoutExpired(cmd="x", timeout=1),
                    OSError("disk went away")):
            t[0] += A.LIVE_JUDGE_INTERVAL_S
            A.run_live_judge(home, notify=r.notify, log=r.log,
                             run_script=_raiser(exc), now=t[0])
        assert len(r.notes) == 2, r.notes

    def test_the_SAME_exception_type_stays_fire_once(self, tmp_path):
        """The pair — cause keying must not become an every-day pager."""
        home = _home(tmp_path)
        r = _Recorder()
        t = [1_000_000]

        def run(script, args, *, home, timeout_s):
            raise OSError("same failure, new day")
        for _ in range(2):
            t[0] += A.SUPPLY_WATCH_INTERVAL_S
            A.run_supply_watch(home, notify=r.notify, log=r.log,
                               run_script=run, now=t[0])
        assert len(r.notes) == 1, r.notes
