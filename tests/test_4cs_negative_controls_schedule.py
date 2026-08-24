"""§4CS item E — the E3 negative controls are SCHEDULED, RECORDED, and LOUD.

§4CO recorded this as still owed. One control run was performed by hand on
2026-08-23 and passed — into a throwaway home, so nothing durable recorded
that it had ever happened. This project's own rule is that a guard which
never demonstrably fires is presumed dead, and E2's entire value is
refusing things.

Three properties, and the third is the one that is easy to get wrong:

  1. a weekly schedule with a DURABLE anchor (an in-memory anchor makes
     the cadence "once per deploy", so on a box that restarts often a
     weekly job runs every boot);
  2. the firing recorded where an operator sees it;
  3. a control that FAILS TO FIRE is LOUD. That failure is NOT a zero —
     a run where two of three controls held has a positive count and is
     the single most important line on the view. `alarm_if_zero` cannot
     express it, which is why `ProbeResult` grew an explicit `alarm`.
"""

import json
import time
from pathlib import Path

import pytest

from ghost_agent.core import liveness as L
from ghost_agent.core.liveness import FIRED, NO_SOURCE, ZERO, probe_all
from ghost_agent.evolve import negative_controls as NC


def _write(home: Path, doc: dict):
    d = home / "system" / "evolve"
    d.mkdir(parents=True, exist_ok=True)
    (d / NC.RESULT_FILE).write_text(json.dumps(doc))


def _iso(epoch):
    import datetime
    return datetime.datetime.utcfromtimestamp(epoch).isoformat() + "Z"


def _doc(*, held=NC.ALL_CONTROLS, failed=(), age_s=0, selected=None,
         unverified=()):
    selected = list(selected if selected is not None
                    else list(held) + list(failed))
    results = ([{"name": n, "ok": True, "verified": True, "rejected": True}
                for n in held]
               + [{"name": n, "ok": False, "verified": True, "rejected": False,
                   "detail": "the candidate was NOT refused"} for n in failed])
    all_ok = (bool(results)
              and not failed
              and set(selected) == set(NC.ALL_CONTROLS)
              and {r["name"] for r in results} == set(selected))
    return {"ts": _iso(time.time() - age_s), "deep": False,
            "all_ok": all_ok, "partial_ok": not failed,
            "selected": selected, "unverified": list(unverified),
            "results": results}


def _row(home):
    return {r["name"]: r for r in probe_all(home)["rows"]}["evolve.negative_controls"]


# ── one cadence, not two ─────────────────────────────────────────────────
class TestCadenceHasOneDefinition:
    def test_the_phase_and_the_probe_share_the_interval(self):
        """Two copies of a cadence is how a monitor ends up quiet about a
        schedule that stopped. Compared against the module constant, not
        against a restated literal."""
        from ghost_agent.core.agent import GhostAgent
        assert GhostAgent._NEGCTRL_COOLDOWN == NC.INTERVAL_S
        assert NC.STALE_AFTER_S == 2 * NC.INTERVAL_S

    def test_the_anchor_is_DURABLE_not_in_memory(self, tmp_path, monkeypatch):
        """An in-memory anchor makes a weekly job run on every boot."""
        from ghost_agent.core.agent import _negctrl_last_run_at
        import datetime
        monkeypatch.setenv("GHOST_HOME", str(tmp_path))
        assert _negctrl_last_run_at() == datetime.datetime.min
        _write(tmp_path, _doc(age_s=3600))
        got = _negctrl_last_run_at()
        assert got != datetime.datetime.min
        assert abs((datetime.datetime.now() - got).total_seconds() - 3600) < 120

    def test_a_malformed_ts_does_not_fake_a_recent_run(self, tmp_path):
        _write(tmp_path, {"ts": "not-a-timestamp", "results": []})
        assert NC.last_run_ts(tmp_path) is None

    def test_the_root_resolves_from_the_PACKAGE_not_the_cwd(self, monkeypatch,
                                                            tmp_path):
        """§4CO #6: resolving against the process CWD is correct from a
        shell in the repo and silently wrong under launchd."""
        from ghost_agent.core.agent import _evolve_canonical_root
        monkeypatch.chdir(tmp_path)
        root = _evolve_canonical_root()
        assert root is not None
        assert (root / "src" / "ghost_agent").is_dir()
        assert (root / "tests").is_dir()
        assert root != tmp_path


# ── the operator can see it fired ────────────────────────────────────────
class TestTheFiringIsVisible:
    def test_never_run_is_NO_SOURCE_not_a_tick(self, tmp_path):
        r = _row(tmp_path)
        assert r["status"] == NO_SOURCE
        assert "presumed dead" in r["note"]
        assert r["name"] in probe_all(tmp_path)["gaps"]

    def test_a_clean_fresh_run_is_FIRED_and_quiet(self, tmp_path):
        _write(tmp_path, _doc())
        r = _row(tmp_path)
        assert r["status"] == FIRED
        assert r["count"] == len(NC.ALL_CONTROLS)
        assert r["alarm"] is False
        assert r["age_h"] is not None


# ── a control that fails to fire is LOUD ─────────────────────────────────
class TestFailureIsLoud:
    def test_a_control_that_did_not_fire_ALARMS_despite_a_positive_count(
            self, tmp_path):
        """The property `alarm_if_zero` cannot express. Two of three
        holding is not a zero — and it is the loudest line on the view."""
        _write(tmp_path, _doc(held=("edits_a_test", "no_op_claiming_improvement"),
                              failed=("deletes_a_guard",)))
        r = _row(tmp_path)
        assert r["count"] == 2 and r["status"] == FIRED
        assert r["alarm"] is True
        assert "DID NOT FIRE" in r["note"] and "deletes_a_guard" in r["note"]
        assert r["name"] in probe_all(tmp_path)["alarms"]

    def test_a_STOPPED_schedule_alarms(self, tmp_path):
        _write(tmp_path, _doc(age_s=NC.STALE_AFTER_S + 86400))
        r = _row(tmp_path)
        assert r["alarm"] is True
        assert "SCHEDULE HAS STOPPED" in r["note"]

    def test_a_run_just_inside_the_bound_does_NOT_alarm(self, tmp_path):
        """One missed window (a box busy every time the idle floor came
        round) must not page — that is how a monitor gets muted."""
        _write(tmp_path, _doc(age_s=NC.STALE_AFTER_S - 3600))
        assert _row(tmp_path)["alarm"] is False

    def test_a_PARTIAL_run_is_not_green(self, tmp_path):
        """A control that was never selected contributes to a green suite
        by being ABSENT — the same rule `verified` exists to prevent."""
        _write(tmp_path, _doc(held=("edits_a_test",), selected=["edits_a_test"]))
        r = _row(tmp_path)
        assert r["alarm"] is True
        assert "PARTIAL run" in r["note"]

    def test_a_control_MISSING_from_its_own_results_alarms(self, tmp_path):
        doc = _doc()
        doc["results"] = [x for x in doc["results"]
                          if x["name"] != "deletes_a_guard"]
        _write(tmp_path, doc)
        r = _row(tmp_path)
        assert r["alarm"] is True
        assert "never ran" in r["note"] and "deletes_a_guard" in r["note"]

    def test_an_UNVERIFIED_control_alarms(self, tmp_path):
        _write(tmp_path, _doc(unverified=("deletes_a_guard",)))
        r = _row(tmp_path)
        assert r["alarm"] is True and "unverified" in r["note"]

    def test_no_traffic_never_excuses_this_probe(self, tmp_path):
        """It runs off the idle clock, not off user traffic, so a quiet
        box is not an excuse — DEN_NONE."""
        den = {p.name: p.denominator for p in L.PROBES}
        assert den["evolve.negative_controls"] == L.DEN_NONE
        _write(tmp_path, _doc(failed=("deletes_a_guard",),
                              held=("edits_a_test", "no_op_claiming_improvement")))
        # tmp_path has no trajectories at all → zero user turns, zero requests
        out = probe_all(tmp_path)
        assert out["user_turns_24h"] in (0, None)
        assert "evolve.negative_controls" in out["alarms"]


# ── the explicit alarm channel itself ────────────────────────────────────
class TestExplicitAlarmChannel:
    def test_a_probe_can_alarm_without_being_ZERO(self, monkeypatch, tmp_path):
        monkeypatch.setattr(L, "PROBES", [
            L.Probe("x", L.EXPECT_PERIODIC, "s",
                    lambda _h: L.ProbeResult(FIRED, count=7, alarm=True),
                    alarm_if_zero=False, denominator=L.DEN_NONE)])
        out = probe_all(tmp_path)
        assert out["rows"][0]["count"] == 7
        assert out["rows"][0]["alarm"] is True
        assert out["alarms"] == ["x"]

    def test_the_legacy_alarm_if_zero_path_still_works(self, monkeypatch,
                                                       tmp_path):
        monkeypatch.setattr(L, "PROBES", [
            L.Probe("y", L.EXPECT_PERIODIC, "s",
                    lambda _h: L.ProbeResult(ZERO, count=0),
                    alarm_if_zero=True, denominator=L.DEN_NONE)])
        assert probe_all(tmp_path)["alarms"] == ["y"]
