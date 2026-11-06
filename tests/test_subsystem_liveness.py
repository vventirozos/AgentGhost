"""Probe-per-mechanism liveness — covering what the activity ledger does NOT.

WHY (2026-08-10). `learning_health.activity_liveness` made absence a ROW rather
than a silence — for the 18 phases that write to `autonomous_activity.jsonl`.
The most important dead loop found that day, **metacog arbitration (0 firings
against 118 below-threshold opportunities)**, does not write there and was
found by grepping the log BY HAND. A dead-loop detector blind to the class of
thing that justified it is the defect it exists to remove, one level up.

THE THIRD STATE is the point. `activity_liveness` cannot distinguish "ran zero
times" from "there is no way to tell": both render as zero. Here:

    FIRED / ZERO / NO_SOURCE / GATED

NO_SOURCE is an instrumentation GAP — a finding, not a diagnostic nicety. It is
the same missing-vs-empty distinction §4AC removed from the response cache.

⚠ THIS FILE PINS FIVE DEFECTS FOUND IN MY OWN PROBES DURING REVIEW, not in the
system they watch. Every one produced a plausible reading:

  R1-a  `\\bverify\\b` does not match "verifier" (verif-Y vs verifi-ER) — a
        FALSE DEAD alarm against 43 real verdict lines.
  R1-b  `float(ts)` on stores that write ISO strings — two healthy stores
        (foresight, rrf) read as ZERO.
  R1-c  probed `experiments.json`, an OPTIONAL OVERRIDE whose absence is
        NORMAL — a healthy system reported as an instrumentation gap.
  R1-d  no turn DENOMINATOR, so turn-driven silence was uninterpretable and
        nearly produced a false MAJOR ("three subsystems died at 02:00").
  R2-a  mtime-based freshness: a `touch` fakes FIRED on a dead mechanism —
        a false GREEN, which nobody investigates.

⚠⚠ AND A SIXTH, FOUND LIVE ON 2026-08-11 — in R1-d, the guard added to stop a
false MAJOR, which by then was arguing FOR one. `_count_user_turns` counted
every `request started` line, but self-play/dream turns enter through the same
`handle_chat` and log the same line. On the live agent that day: **28 "user
turns", of which 28 were self-play and 0 were real**, while foresight / rrf /
trajectories were correctly silent BECAUSE their simulation gates excluded
those same turns. The denominator and the ledgers were measuring the turn
population in opposite directions, and the denominator is what a reader trusts.

Two coupled defects, and the first hid the second:
  D1  the count included self-play, so it could reach 0 (the branch that
      withholds alarms) only on a box where the hourly self-play loop was ALSO
      dead — an unreachable guard, i.e. furniture.
  D2  the withholding cleared EVERY row, but the only two probes that alarm are
      PERIODIC (idle-clock, not turn-driven). Had D1 ever let it fire, it would
      have silenced exactly the alarms a quiet day does not explain. Fixing D1
      ARMS D2, so both move together.
"""

import json
import time
from pathlib import Path

import pytest

from ghost_agent.core import liveness as LV
from ghost_agent.core.liveness import (
    FIRED,
    NO_SOURCE,
    ZERO,
    ProbeResult,
    _parse_ts,
    probe_all,
    render,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    LV._LOG_CACHE.clear()
    yield
    LV._LOG_CACHE.clear()


def _log(home: Path, lines):
    d = home / "system"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ghost-agent.log").write_text("\n".join(lines) + "\n")


def _stamp(offset_h=0.0):
    return time.strftime("%Y-%m-%d %H:%M:%S",
                         time.localtime(time.time() - offset_h * 3600))


def _idle_clock_probe(monkeypatch, home, *, age_h):
    """§4PJ: `calibration.fit` was the last registered PERIODIC, alarming,
    DEN_NONE probe; it left with the calibration refit. The alarm logic it
    exercised is generic, so these pins drive it through a synthetic probe
    of exactly that shape, registered for the test only."""
    import os
    f = home / "system" / "idle_clock.stamp"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x")
    t = time.time() - age_h * 3600
    os.utime(f, (t, t))
    probe = LV.Probe("idle.clock", LV.EXPECT_PERIODIC, "system/idle_clock.stamp",
                     LV._mtime_probe("system/idle_clock.stamp", stale_h=48.0),
                     alarm_if_zero=True)
    monkeypatch.setattr(LV, "PROBES", [*LV.PROBES, probe])
    return probe


# ── R1-b: timestamps come in two shapes ─────────────────────────────────────

@pytest.mark.parametrize("v,ok", [
    (1786316634.0, True), ("1786316634", True),
    ("2026-08-05T11:15:25.881934Z", True),      # the LIVE store format
    ("2026-08-05T11:15:25+00:00", True),
    (None, False), ("", False), ("not-a-time", False), (0, False),
])
def test_ISO_and_epoch_timestamps_both_parse(v, ok):
    """⚠ R1-b. `float(ts)` alone threw on every record in the live stores, so
    foresight and rrf — both healthy — reported ZERO."""
    assert (_parse_ts(v) is not None) is ok


# ── R1-a: the regex that cried wolf ─────────────────────────────────────────

def test_verifier_verdict_lines_are_MATCHED(tmp_path):
    """⚠ R1-a. The live line reads 'verifier — LATE CONFIRMED (95%)'. The
    original `\\bverify\\b` missed it and reported DEAD against 43 real lines.
    A monitor whose first output is a false page gets muted on day one."""
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [abc] verifier — LATE CONFIRMED (95%)",
        f"{_stamp(2)} - GhostStream - INFO - [def] verifier — REFUTED conf=0.9",
    ])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    res = probe.fn(tmp_path)
    assert res.status == FIRED and res.count == 2


def test_verifier_lines_are_matched_REGARDLESS_OF_CASE(tmp_path):
    """⚠ REVIEW ROUND 1 (post-retraction), 2026-08-10. The pattern was
    case-SENSITIVE and skipped **1025 log lines**, almost all
    "Verifier escalation OVERTURNED a cheap-judge refute: … CONFIRMED".
    Those prove verification RAN, so excluding them undercounted the exact
    thing the probe measures: n=1 became n=29 once fixed.

    Found by cross-checking the narrow pattern against a deliberately BROAD
    net and sampling what the narrow one skipped — the same technique that
    should have caught the `verify`/`verifier` bug the first time.
    """
    _log(tmp_path, [
        f"{_stamp(1)} - GhostAgent - WARNING - Verifier escalation OVERTURNED "
        f"a cheap-judge refute: main model says CONFIRMED",
        f"{_stamp(2)} - GhostStream - INFO - [x] verifier — REFUTED conf=0.9",
    ])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    res = probe.fn(tmp_path)
    assert res.count == 2, (
        "capitalised verifier lines are being skipped again — that was a 29x "
        "undercount")


def test_an_unrelated_line_does_not_count_as_a_verdict(tmp_path):
    """Over-matching would be the opposite failure: a permanently green row."""
    _log(tmp_path, [f"{_stamp(1)} - GhostStream - INFO - [abc] dream — cycle done"])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    assert probe.fn(tmp_path).status == ZERO


def test_log_lines_outside_the_window_are_not_counted(tmp_path):
    _log(tmp_path, [
        f"{_stamp(200)} - GhostStream - INFO - [old] verifier — CONFIRMED"])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    res = probe.fn(tmp_path)
    assert res.status == ZERO and res.last_ts is not None, (
        "an out-of-window hit must still report WHEN it last fired")


# ── R2-a: mtime can be faked, file content cannot ───────────────────────────

# ── the gated mechanism that started all this ───────────────────────────────

# ── R1-d: the denominator ───────────────────────────────────────────────────

def test_zero_user_turns_withholds_alarms(tmp_path):
    """⚠ R1-d. Turn-driven mechanisms are correctly silent when nothing asked
    them to run. Without this, three quiet stores read as a triple failure —
    which is exactly the false MAJOR this nearly produced."""
    _log(tmp_path, [f"{_stamp(1)} - GhostStream - INFO - [x] dream — tick"])
    out = probe_all(tmp_path)
    assert out["user_turns_24h"] == 0
    assert out["alarms"] == []


def test_self_play_turns_are_NOT_user_turns(tmp_path):
    """⚠ THE 2026-08-11 DEFECT, pinned on the live shape.

    Self-play enters through the same `handle_chat` and logs the same line. If
    those count, the denominator reports traffic on a box that had none — and
    the turn-driven ledgers below it (foresight, rrf, trajectories) are silent
    precisely BECAUSE their simulation gates dropped those turns. Reverting the
    origin filter turns this into 3.
    """
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=sim",
        f"{_stamp(1)} - GhostStream - INFO - [b] request started — b origin=sim",
        f"{_stamp(1)} - GhostStream - INFO - [c] request started — c origin=sim",
    ])
    out = probe_all(tmp_path)
    assert out["user_turns_24h"] == 0, (
        "three self-play turns are ZERO user turns — the whole point of the "
        "denominator is which population it counts")
    assert out["requests_24h"] == 3, (
        "they ARE requests, and mechanisms that run per request are owed that "
        "denominator instead")
    assert "rrf.observations" not in out["alarms"]


def test_a_prestamp_log_is_UNCLASSIFIED_not_a_number(tmp_path):
    """MISSING IS UNKNOWN, NEVER A COUNT. A log written by a build older than
    the origin stamp cannot be split into user vs self-play. Quoting the
    stamped subset would silently mean something different from what the label
    says; quoting the total is the bug this fixes. Say UNCLASSIFIED instead."""
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a",
        f"{_stamp(1)} - GhostStream - INFO - [b] request started — b origin=user",
    ])
    out = probe_all(tmp_path)
    assert out["user_turns_24h"] is None
    assert "unclassified" in out["user_turns_note"]
    assert "2" not in out["user_turns_note"].split("(")[0]
    txt = render(tmp_path)
    assert "cannot be interpreted" in txt and "unclassified" in txt


def test_UNKNOWN_denominator_withholds_SIM_GATED_alarms_only(tmp_path,
                                                             monkeypatch):
    """An uninterpretable denominator must not license a DEAD verdict on a
    simulation-gated mechanism — that is the false MAJOR. But it must not
    silence a DEN_NONE one either: nothing about user traffic explains an
    idle-clock loop that stopped."""
    # no log at all => user-turn denominator UNKNOWN
    _idle_clock_probe(monkeypatch, tmp_path, age_h=300)
    # a sim-gated probe that alarms, to prove the two are treated differently
    monkeypatch.setattr(
        LV.PROBES[0], "denominator", LV.DEN_USER_TURNS, raising=False)
    monkeypatch.setattr(LV.PROBES[0], "alarm_if_zero", True, raising=False)
    monkeypatch.setattr(LV.PROBES[0], "fn", lambda _h: ProbeResult(ZERO, count=0))

    out = probe_all(tmp_path)
    assert out["user_turns_24h"] is None
    assert LV.PROBES[0].name not in out["alarms"], (
        "sim-gated zero + unknown denominator = cannot say, not DEAD")
    assert "idle.clock" in out["alarms"], (
        "a DEN_NONE mechanism runs off the idle clock — an absent user-turn "
        "count is no excuse for its silence")


def test_turn_origin_agrees_with_the_SIMULATION_GATE_it_borrows_from():
    """⚠ THE SEAM. The stamp is only worth counting if it splits the same
    population the ledgers' own gates split. Both read the `simulated` mark
    (`is_simulated_context`, §4TD); a second, private notion of "real traffic" is what let the
    denominator and the ledgers disagree for a day.

    ⚠ `is True` is load-bearing and this pins it: a MagicMock context (which
    every agent test uses) returns a truthy Mock for ANY attribute, so a plain
    truth test would stamp every mocked turn "sim" — the §4L MagicMock trap,
    already commented at the finalize site for the same reason.
    """
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import turn_origin

    class _Ctx:
        simulated = False

    sim = _Ctx(); sim.simulated = True
    assert turn_origin(sim) == "sim"

    assert turn_origin(_Ctx()) == "user"
    assert turn_origin(object()) == "user", "a context with no marker at all"
    assert turn_origin(MagicMock()) == "user", (
        "a MagicMock's truthy attribute must NOT read as a simulation")


def test_every_alarming_probe_declares_which_absence_excuses_it(tmp_path):
    """⚠ STRUCTURAL. The default is DEN_NONE — a mechanism must EARN its
    excuse — so this cannot be satisfied by forgetting to think about it. It
    fails the moment someone adds an alarming probe without deciding whether a
    quiet box explains its zero, which is the decision that was wrong here."""
    for p in LV.PROBES:
        assert p.denominator in (LV.DEN_NONE, LV.DEN_REQUESTS,
                                 LV.DEN_USER_TURNS), p.name
    # (§4NN: `rrf.observations` and `experiments.stamps`, the user-turn-gated
    # probes, left with their writers.)


def test_no_traffic_does_NOT_silence_a_dead_IDLE_CLOCK_loop(tmp_path, monkeypatch):
    """⚠ D2, armed by the D1 fix. The guard used to clear EVERY row. A quiet
    weekend explains a silent trajectory writer; it does not explain a training
    loop that stopped fitting."""
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=sim"])
    _idle_clock_probe(monkeypatch, tmp_path, age_h=300)

    out = probe_all(tmp_path)
    assert out["user_turns_24h"] == 0
    assert "idle.clock" in out["alarms"], (
        "zero USER turns must not withhold a PERIODIC mechanism's alarm")


def test_the_synthetic_idle_clock_probe_can_stay_quiet(tmp_path, monkeypatch):
    """The control for the two pins around it: a FRESH source on the same
    synthetic probe raises nothing, so their alarm is the stale source's
    doing, not something every run of this probe produces."""
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=user"])
    _idle_clock_probe(monkeypatch, tmp_path, age_h=1)
    assert "idle.clock" not in probe_all(tmp_path)["alarms"]


def test_turns_present_means_alarms_are_LIVE_again(tmp_path, monkeypatch):
    """⚠ OVER-SUPPRESSION GUARD: the no-traffic guard must key on ZERO turns
    only, or it silences every real finding.

    ⚠ The first version of this test asserted an alarm against a BARE home —
    wrong, because a missing store is NO_SOURCE and NO_SOURCE correctly does
    not alarm. The alarm needs a source that EXISTS and is stale, which is the
    only state that means "this really did not run".
    """
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=user",
        f"{_stamp(1)} - GhostStream - INFO - [b] request started — b origin=user",
    ])
    _idle_clock_probe(monkeypatch, tmp_path, age_h=300)

    out = probe_all(tmp_path)
    assert out["user_turns_24h"] == 2
    assert "idle.clock" in out["alarms"], (
        "a periodic mechanism, live source, stale beyond its window, with "
        "traffic present — must alarm")


# ── the view itself ─────────────────────────────────────────────────────────

def test_every_registered_probe_appears_whatever_its_state(tmp_path):
    """Absence is a ROW, not a silence — the founding rule, applied here."""
    out = probe_all(tmp_path)
    assert len(out["rows"]) == len(LV.PROBES)
    assert {r["name"] for r in out["rows"]} == {p.name for p in LV.PROBES}


def test_gaps_are_reported_separately_from_dead(tmp_path):
    out = probe_all(tmp_path)
    assert out["gaps"], "a bare home should report NO_SOURCE gaps"
    for name in out["gaps"]:
        row = next(r for r in out["rows"] if r["name"] == name)
        assert row["alarm"] is False, (
            "NO_SOURCE is a gap to fix, not a dead loop to page about")


def test_a_probe_that_raises_degrades_to_NO_SOURCE(monkeypatch, tmp_path):
    """Instrumentation must never break the view it feeds."""
    def boom(_home):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(LV.PROBES[0], "fn", boom)
    out = probe_all(tmp_path)
    row = next(r for r in out["rows"] if r["name"] == LV.PROBES[0].name)
    assert row["status"] == NO_SOURCE and "raised" in row["note"]


def test_the_render_names_gaps_and_the_turn_denominator(tmp_path):
    """Both halves must reach the SCREEN: what has no evidence source, and the
    denominator without which a zero cannot be read."""
    txt = render(tmp_path)
    assert "SUBSYSTEM LIVENESS" in txt
    assert "NO durable evidence source" in txt
    # bare home => no log => the denominator is UNAVAILABLE, and saying so is
    # the point (an absent denominator must not read as "zero turns").
    assert "user-turn count unavailable" in txt

    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=user"])
    txt2 = render(tmp_path)
    assert "1 real user turns in 24h" in txt2


# ── router signal COVERAGE (the section header used to assert it "never
# reaches the corpus" — retracted: it does, under `extra`; what varies is
# coverage, 0–70%/day) ──────────────────────────────────────────────────────

def _traj(home: Path, day: str, records):
    d = home / "system" / "trajectories" / day
    d.mkdir(parents=True, exist_ok=True)
    (d / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")


def test_learning_health_passes_GHOST_HOME_not_its_system_child(tmp_path):
    """⚠ WIRING DEFECT, caught on first integration. `learning_health` passed
    `Path(memory_dir).parent` — that is $GHOST_HOME/system, but probes resolve
    "system/<x>" relative to $GHOST_HOME, so ALL EIGHT reported NO_SOURCE.

    A pure unit test of the probes cannot catch this: each one works
    perfectly, and only the caller's path is wrong. Pinned at the seam.
    """
    home = tmp_path
    _log(home, [
        f"{_stamp(1)} - GhostStream - INFO - [a] request started — a origin=user",
        f"{_stamp(1)} - GhostStream - INFO - [a] verifier — LATE CONFIRMED (95%)"])

    from ghost_agent.core import learning_health as LH
    out = LH.render_learning_health(home / "system" / "memory")
    assert "SUBSYSTEM LIVENESS" in out
    # (§4PJ: was the calibration.fit row; the verifier's log probe is the
    # registered probe left whose source lives under $GHOST_HOME/system)
    row = [l for l in out.splitlines() if "verifier.outcomes" in l]
    assert row and "fired" in row[0], row
    # The tell for the bug: every probe simultaneously NO_SOURCE.
    assert "8 mechanism(s) have NO durable evidence source" not in out, (
        "the caller is passing the wrong root again — all probes went blind")


def test_the_log_parse_is_shared_not_repeated(tmp_path):
    """⚠ R2 cost: three probes each scanned a 14MB log (1.9s). One parse,
    keyed on size+mtime so an appended log invalidates rather than serving a
    stale count."""
    _log(tmp_path, [f"{_stamp(1)} - GhostStream - INFO - [a] verifier — CONFIRMED"])
    p = tmp_path / "system" / "ghost-agent.log"
    LV._log_entries(p)
    assert len(LV._LOG_CACHE) == 1
    before = list(LV._LOG_CACHE)[0]
    with p.open("a") as fh:
        fh.write(f"{_stamp(0)} - GhostStream - INFO - [b] verifier — REFUTED\n")
    LV._log_entries(p)
    assert list(LV._LOG_CACHE)[0] != before, "an appended log served a stale parse"


def test_agent_PROSE_about_a_mechanism_is_not_evidence_it_RAN(tmp_path):
    """⚠ FALSE-GREEN GUARD, review round 2, 2026-08-10.

    The agent's own reasoning is mirrored at DEBUG, so
    "thinking — the verifier refuted my claim" matches a verifier pattern
    while proving nothing ran. Measured 12 of 1370 matches were prose. Only
    0.9% today — but if verification ever genuinely STOPPED, that prose would
    keep this probe GREEN, and a green row is the one nobody investigates.
    """
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - DEBUG - [x +5s] thinking — "
        f"The verifier refuted my claim that all tests CONFIRMED",
    ])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    assert probe.fn(tmp_path).status == ZERO, (
        "agent prose is being counted as a verification event")


def test_a_REAL_verdict_still_counts_alongside_prose(tmp_path):
    """⚠ OVER-EXCLUSION GUARD: the filter must drop only the prose."""
    _log(tmp_path, [
        f"{_stamp(1)} - GhostStream - DEBUG - [x +5s] thinking — the verifier REFUTED it",
        f"{_stamp(1)} - GhostStream - INFO - [x] verifier — CONFIRMED conf=0.9",
    ])
    probe = next(p for p in LV.PROBES if p.name == "verifier.outcomes")
    res = probe.fn(tmp_path)
    assert res.status == FIRED and res.count == 1
