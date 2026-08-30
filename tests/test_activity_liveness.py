"""Every background mechanism is COUNTED, and a dead one is LOUD.

THE DEFECT (measured 2026-08-10). `BACKGROUND ACTIVITY` enumerated phases
OBSERVATIONALLY — it counted whatever slugs happened to be in the ledger. So
the report could only ever describe mechanisms that were already writing. A
loop that stopped writing, or never wrote at all, produced no key and was
therefore invisible: **indistinguishable from a mechanism that does not
exist**. Against the live ledger it rendered seven green rows while EIGHT of
the fifteen phases instrumented in source had produced nothing in seven days.

Two further silent caps rode along: the renderer sliced `[:8]`, so a phase at
zero — the only kind worth acting on — sorted last and was structurally the
first row dropped; and the reader took `limit=2000` against a ledger already
past 2600 records.

WHY A REGISTRY AND NOT JUST "RENDER THE ZEROS". `reflection` legitimately sits
at 0 (it records only outcome-producing runs and skips unchanged-corpus
ticks); `dream` at 0 would be a dead loop. A benign zero and a fatal zero must
not look alike — that is the same defect one level up. Each phase therefore
declares what ITS zero means, and only PERIODIC can alarm.

⚠ THE LOAD-BEARING TEST IN THIS FILE is `test_a_dead_periodic_loop_ALARMS`
and its siblings: a monitor that has never been observed to alarm proves
nothing at all. This is the mutation test for the alarm, exactly as
scripts/verify_core_math.py mutation-tests the arithmetic it verifies.
"""

import json
import re
import time
from pathlib import Path

import pytest

from ghost_agent.core.autonomous_activity import (
    EXPECT_GATED,
    EXPECT_ON_DEMAND,
    EXPECT_ON_OUTPUT,
    EXPECT_PERIODIC,
    PHASE_EXPECTATION,
    _PHASE_LABELS,
    phase_expectation,
)
from ghost_agent.core.learning_health import activity_liveness

REPO = Path(__file__).resolve().parents[1]
PERIODIC = [p for p, e in PHASE_EXPECTATION.items() if e == EXPECT_PERIODIC]


def _ledger(tmp_path, phases, *, age_h=1.0):
    """A ledger where each named phase fired once, `age_h` hours ago."""
    p = tmp_path / "autonomous_activity.jsonl"
    now = time.time()
    with p.open("w") as fh:
        for ph in phases:
            fh.write(json.dumps({"ts": now - age_h * 3600.0, "phase": ph,
                                 "summary": "x", "severity": "info"}) + "\n")
    return p


# ── the registry covers what the code actually writes ───────────────────────

def test_EVERY_phase_written_in_source_is_registered():
    """⚠ THE ANTI-BLIND-SPOT TEST. A registry that silently omits a mechanism
    rebuilds the exact defect it exists to remove — the omitted loop becomes
    invisible again, and now with a green report vouching for it.

    Measured at the time of writing: 15 phase literals in src/, of which 8 had
    produced ZERO ledger records in seven days.
    """
    # ⚠ `_record_idle_attempt` is a SECOND door into the liveness view
    # (§4DY, the attempt heartbeat). It was added without being added here,
    # so a phase reachable only through it was invisible to this guard — the
    # exact blind spot this test exists to close, re-entered through the new
    # entry point. Any future recorder must be listed here too.
    pat = re.compile(
        r'(?:_record_autonomous_activity|_record_idle_attempt|log\.record'
        r'|_alog\.record|activity_log\.record)\(\s*\n?\s*["\']([a-z_]+)["\']', re.M)
    found = set()
    for f in (REPO / "src").rglob("*.py"):
        found |= set(pat.findall(f.read_text()))
    missing = found - set(PHASE_EXPECTATION)
    assert not missing, (
        f"phase(s) written by the code but absent from PHASE_EXPECTATION: "
        f"{sorted(missing)} — they will never appear in the liveness report, "
        f"which is how a dead loop hides")


def test_every_registered_phase_has_a_human_label():
    """A row the operator cannot read is a row they skip."""
    missing = set(PHASE_EXPECTATION) - set(_PHASE_LABELS)
    assert not missing, f"unlabelled phases render as raw slugs: {sorted(missing)}"


def test_an_unknown_phase_never_manufactures_an_alarm():
    """Unregistered slugs default to ON_DEMAND. An alarm on a phase nobody
    declared is noise, and noise is what gets a monitor ignored."""
    assert phase_expectation("something_nobody_registered") == EXPECT_ON_DEMAND
    assert phase_expectation("") == EXPECT_ON_DEMAND
    assert phase_expectation(None) == EXPECT_ON_DEMAND


# ── absence is a ROW, not a silence ─────────────────────────────────────────

def test_a_mechanism_that_never_wrote_still_appears(tmp_path):
    """THE DEFECT ITSELF. `dream` never wrote here; before the registry it
    simply was not in the output."""
    lv = activity_liveness(_ledger(tmp_path, ["self_play"]))
    phases = {r["phase"] for r in lv["rows"]}
    assert "dream" in phases, "a silent mechanism vanished from the report"
    assert set(PHASE_EXPECTATION) <= phases, "the report is not registry-driven"


def test_zeros_sort_FIRST(tmp_path):
    """The old renderer sorted by count descending and cut to 8 — the dead
    loop was structurally the first row dropped."""
    lv = activity_liveness(_ledger(tmp_path, ["self_play"] * 40))
    assert lv["rows"][0]["n_alarm_window"] == 0
    assert lv["rows"][-1]["phase"] == "self_play"


def test_an_unregistered_phase_in_the_ledger_is_shown_and_flagged(tmp_path):
    """Forward compatibility: a new phase must not be dropped just because
    this registry has not caught up. It renders, marked unregistered — and
    the coverage test above is what makes that state temporary."""
    lv = activity_liveness(_ledger(tmp_path, ["brand_new_loop"]))
    row = next(r for r in lv["rows"] if r["phase"] == "brand_new_loop")
    assert row["registered"] is False and row["alarm"] is False


# ── THE MUTATION TEST: does the alarm actually fire? ────────────────────────

@pytest.mark.parametrize("victim", PERIODIC)
def test_a_dead_periodic_loop_ALARMS(tmp_path, victim):
    """⚠ MUTATION TEST, one per periodic loop. Silence exactly one mechanism
    and require the monitor to name it.

    A monitor that has never been observed to alarm proves nothing — it is
    the "guard that never runs" defect class this codebase keeps finding,
    and building one to detect that class while not testing that it fires
    would be the purest form of the mistake.
    """
    alive = [p for p in PERIODIC if p != victim]
    lv = activity_liveness(_ledger(tmp_path, alive * 3))
    assert lv["alarms"] == [victim], (
        f"silencing {victim} did not raise exactly one alarm: {lv['alarms']}")
    row = next(r for r in lv["rows"] if r["phase"] == victim)
    assert row["alarm"] is True


def test_a_live_periodic_loop_does_not_alarm(tmp_path):
    lv = activity_liveness(_ledger(tmp_path, PERIODIC))
    assert lv["alarms"] == []


@pytest.mark.parametrize("phase,exp", [
    ("reflection", EXPECT_ON_OUTPUT),
    ("open_questions", EXPECT_ON_OUTPUT),
    ("workspace_tidy", EXPECT_ON_OUTPUT),
    ("native_tool_repair", EXPECT_ON_OUTPUT),
    ("scheduled_task", EXPECT_ON_DEMAND),
    ("experiment_verdict", EXPECT_ON_DEMAND),
    ("prm_train", EXPECT_GATED),
    ("postmortem", EXPECT_GATED),
])
def test_a_BENIGN_zero_never_alarms(tmp_path, phase, exp):
    """⚠ THE OTHER HALF, and the one that keeps the monitor usable. These
    sit at zero BY DESIGN: `open_questions` records `if stale:`,
    `workspace_tidy` records `if _tidy_deleted:`, `native_tool_repair` is
    ~0 by design and each occurrence is news, PRM retrain skips while its
    consumers are off. Alarming on them teaches the operator to scroll past
    the section — rebuilding by hand the blindness this exists to remove."""
    assert PHASE_EXPECTATION[phase] == exp
    lv = activity_liveness(_ledger(tmp_path, PERIODIC))   # victim absent
    row = next(r for r in lv["rows"] if r["phase"] == phase)
    assert row["n_alarm_window"] == 0 and row["alarm"] is False


def test_an_ageing_loop_alarms_on_24h_while_the_7d_column_shows_it_ran(tmp_path):
    """Two windows on purpose: 'ran three days ago' and 'dark for a week'
    are different diagnoses, and one boolean cannot carry both.

    ⚠ The other loops must be RECENT here. Ageing everything past the alarm
    window trips the agent-down guard instead — correctly, since that is a
    stopped agent, not five dead loops — which is a different scenario and is
    covered by `test_a_stopped_agent_is_ONE_fact_not_five_alarms`.
    """
    p = tmp_path / "autonomous_activity.jsonl"
    now = time.time()
    with p.open("w") as fh:
        for ph in [x for x in PERIODIC if x != "dream"]:
            fh.write(json.dumps({"ts": now - 3600.0, "phase": ph}) + "\n")
        fh.write(json.dumps({"ts": now - 72 * 3600.0, "phase": "dream"}) + "\n")
    lv = activity_liveness(p)
    assert lv["agent_silent"] is False
    assert lv["alarms"] == ["dream"]
    row = next(r for r in lv["rows"] if r["phase"] == "dream")
    assert row["n_alarm_window"] == 0 and row["n_context_window"] == 1, (
        "the context window must still show the loop ran — otherwise 'slow' "
        "and 'dead' are indistinguishable")


# ── the agent-down guard ────────────────────────────────────────────────────

def test_a_stopped_agent_is_ONE_fact_not_five_alarms(tmp_path):
    """If NOTHING fired, the agent was off or idle. Five 'dead loop' alarms
    for one stopped process is the noise that gets a monitor ignored."""
    lv = activity_liveness(_ledger(tmp_path, []))
    assert lv["agent_silent"] is True
    assert lv["alarms"] == []
    assert all(r["alarm"] is False for r in lv["rows"])


def test_a_MISSING_ledger_is_not_an_idle_agent(tmp_path):
    """⚠ FOUND IN FRESH-EYE REVIEW, 2026-08-10 — in code written the same day
    to detect exactly this defect class.

    A missing ledger and an idle agent rendered IDENTICALLY: every phase at
    zero, `agent_silent` true, no alarms. But one means "nothing ran" and the
    other means "this is pointed at a path that does not exist", in which case
    every zero in the table is meaningless rather than informative.

    It is the same missing-vs-empty conflation removed from the response cache
    hours earlier (an empty cache dir reporting "0 hits", indistinguishable
    from a stale one). Reproducing it here is the sharpest evidence that the
    class is easy to re-introduce and needs a test, not vigilance.
    """
    missing = tmp_path / "nope.jsonl"
    lv = activity_liveness(missing)
    assert lv["ledger_missing"] is True
    assert str(missing) in lv["ledger_path"]

    empty = _ledger(tmp_path, [])
    lv2 = activity_liveness(empty)
    assert lv2["ledger_missing"] is False, "an EXISTING empty ledger is idle"
    assert lv2["agent_silent"] is True

    # The two states must be distinguishable, which is the entire point.
    assert (lv["ledger_missing"], lv["agent_silent"]) != \
           (lv2["ledger_missing"], lv2["agent_silent"])


def test_a_populated_ledger_is_never_reported_missing(tmp_path):
    """Over-firing guard: the notice must key on the FILE, not on the counts."""
    lv = activity_liveness(_ledger(tmp_path, PERIODIC))
    assert lv["ledger_missing"] is False


def test_the_renderer_says_MISSING_INSTRUMENT_not_idle(tmp_path, monkeypatch):
    """The distinction has to reach the screen, and the missing-instrument
    line must WIN over the idle line — a missing ledger is also silent, so
    reporting idleness first would describe the symptom and hide the cause."""
    from ghost_agent.core import learning_health as LH
    lv = activity_liveness(tmp_path / "gone.jsonl")
    monkeypatch.setattr(LH, "collect_learning_health", lambda md: {"liveness": lv})
    out = LH.render_learning_health(tmp_path)
    assert "LEDGER NOT FOUND" in out and "MISSING INSTRUMENT" in out
    assert "off or fully idle" not in out, "the idle line masked the real cause"


def test_the_guard_does_not_swallow_a_REAL_dead_loop(tmp_path):
    """⚠ OVER-SUPPRESSION GUARD. The agent-down guard must key on TOTAL
    silence only. One surviving record means the agent was up, so a silent
    periodic loop is genuinely dead and must still alarm."""
    lv = activity_liveness(_ledger(tmp_path, ["calibration"]))
    assert lv["agent_silent"] is False
    assert "dream" in lv["alarms"]


# ── the silent caps ─────────────────────────────────────────────────────────

def test_truncation_is_reported_rather_than_silent(tmp_path):
    """A tail-read cap that bites must not leave the report confident."""
    from ghost_agent.core import learning_health as LH
    led = _ledger(tmp_path, ["self_play"] * 30)
    assert activity_liveness(led)["read_truncated"] is False
    orig = LH._ACTIVITY_READ_LIMIT
    try:
        LH._ACTIVITY_READ_LIMIT = 5
        assert activity_liveness(led)["read_truncated"] is True
    finally:
        LH._ACTIVITY_READ_LIMIT = orig


def test_the_background_line_no_longer_cuts_to_the_top_N():
    """The old slice dropped zero-count phases first — the one row worth
    acting on.

    ⚠ ANCHORED ON THE STATEMENT, not on a window of surrounding text. The
    first version scanned 400 chars before the render call and matched the
    slice quoted inside the explanatory COMMENT, failing on correct code —
    the "don't grep for the anti-pattern you just wrote about" trap this
    repo already records for the critic-heartbeat guard.
    """
    src = (REPO / "src" / "ghost_agent" / "core" / "learning_health.py").read_text()
    stmt = next(ln for ln in src.splitlines()
                if "top = sorted(act.items()" in ln)
    assert "[:" not in stmt, f"the top-N cut is back: {stmt.strip()}"


def test_the_renderer_marks_a_dead_loop_visibly(tmp_path, monkeypatch):
    """Structural: the alarm has to reach the SCREEN, not just the dict."""
    from ghost_agent.core import learning_health as LH
    lv = activity_liveness(_ledger(tmp_path, [p for p in PERIODIC
                                              if p != "dream"] * 3))
    monkeypatch.setattr(LH, "collect_learning_health",
                        lambda md: {"liveness": lv})
    out = LH.render_learning_health(tmp_path)
    assert "DEAD" in out and "dream" in out
    assert "MECHANISM LIVENESS" in out
