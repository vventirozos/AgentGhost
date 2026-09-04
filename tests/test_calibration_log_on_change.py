"""A standing condition is not an event: `fit` warns on CHANGE, not per refit.

2026-09-04. Three warnings in `CalibrationTracker.fit` describe properties of
the CORPUS — "bench rows rank backwards", "the map is not in force",
"indistinguishable from the base rate". Each was re-emitted at WARNING on
every refit. The refit is hourly and the corpus is append-only, so the same
rows produced the same verdict ~11x a day: measured on the live operator log,
99 INDISTINGUISHABLE lines and 80 bench-EXCLUDED lines across five days,
identical but for the fourth decimal.

The property under review: **a verdict reaches the operator's alarm channel
when it is NEW, and only then.** The steady state must stay READABLE
elsewhere (it is a field on the stored params, and rides the per-refit metacog
CALIB line at INFO), and a genuine transition must still be loud.
"""

import json
import logging
import random

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE, BEATS_NO,
                                          BEATS_UNKNOWN, BEATS_YES,
                                          NOTHING_RECORDED, CalibrationTracker,
                                          announce_level)

BENCH_LINE = "EXCLUDED from the fit"
BASE_RATE_LINES = ("INDISTINGUISHABLE from always", "is WORSE than always")
MAP_LINE = "so the shipped predictor is the RAW"


# ── corpora ────────────────────────────────────────────────────────────────

def _feed_ranking(t, n, *, seed, origin="user", invert=False):
    """Rows whose composite ranks the outcome (or, inverted, misranks it)."""
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        if invert:
            c = 1.0 - c
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4),
                 origin=origin)


def _feed_live_regime(t, n, *, seed):
    """THE LIVE REGIME: composites clustered high, base rate ~0.86, a
    separation that ORDERS turns but cannot beat a constant as a probability.
    Reproduced from `test_calibration_rank_verdict.py`, which grid-searched
    this window."""
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.86
        c = rng.gauss(0.80 + (0.08 if good else -0.08), 0.26)
        c = min(0.99, max(0.01, c))
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))


def _lines(caplog, level, needle):
    return [r for r in caplog.records
            if r.levelno == level and needle in r.getMessage()]


def _any_line(caplog, level, needles):
    return [r for r in caplog.records if r.levelno == level
            and any(n in r.getMessage() for n in needles)]


# ── the rule itself ────────────────────────────────────────────────────────

def test_announce_level_is_warning_only_for_news():
    """The world it fails in: the comparison is inverted, or made
    unconditional — either way the operator's alarm channel stops tracking
    what changed."""
    assert announce_level(BEATS_NO, BEATS_NO) == logging.DEBUG
    assert announce_level(BEATS_NO, BEATS_INDISTINGUISHABLE) == logging.WARNING
    assert announce_level(BEATS_INDISTINGUISHABLE, BEATS_NO) == logging.WARNING
    # Cold start: nothing on record is not the same as "unchanged". No verdict
    # in the vocabulary is None, so this always announces — it fails toward
    # saying too much.
    assert announce_level(None, BEATS_UNKNOWN) == logging.WARNING
    assert announce_level(None, BEATS_NO) == logging.WARNING
    # `map_status` rides the same rule with a different vocabulary.
    assert announce_level("applied", "applied") == logging.DEBUG
    assert announce_level("applied", "rejected_inverted") == logging.WARNING


# ── the bench-exclusion line, from the operator's side ─────────────────────

def test_a_standing_bench_exclusion_is_announced_once(tmp_path, caplog):
    """The defect, reproduced: refit an UNCHANGING corpus and count the
    alarms.

    The world this fails in: `fit` warns unconditionally, so five days of
    hourly refits over one unchanged corpus spend 80 WARNING lines saying the
    same thing.
    """
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 200, seed=1, origin="user")
    _feed_ranking(t, 120, seed=2, origin="bench", invert=True)

    # Refit 1 — COLD START. The gate scores bench rows with the previously
    # fitted weights and there are none, so nothing is excluded yet.
    assert t.fit().bench_verdict == BEATS_UNKNOWN

    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        second = t.fit()          # the exclusion STARTS here: news
        caplog.clear()
        for _ in range(4):        # …and then simply persists
            later = t.fit()

    assert second.bench_verdict == BEATS_NO and second.n_bench_misaligned == 120
    assert later.bench_verdict == BEATS_NO, "fixture stopped exercising the branch"
    assert not _lines(caplog, logging.WARNING, BENCH_LINE), (
        "a standing exclusion re-announced at WARNING")
    # NOT SILENCED — demoted. The line still exists, unchanged, for anyone
    # reading the debug stream or grepping the log.
    persisted = _lines(caplog, logging.DEBUG, BENCH_LINE)
    assert len(persisted) == 4, f"the line vanished instead of moving: {caplog.text!r}"
    assert "rank outcomes backwards" in persisted[0].getMessage()
    assert "AUC" in persisted[0].getMessage()


def test_the_first_bench_exclusion_is_loud(tmp_path, caplog):
    """The world this fails in: a fix that suppresses by "have I logged this
    before", which would swallow the transition that matters."""
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 200, seed=1, origin="user")
    _feed_ranking(t, 120, seed=2, origin="bench", invert=True)
    t.fit()
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.bench_verdict == BEATS_NO
    assert len(_lines(caplog, logging.WARNING, BENCH_LINE)) == 1


def test_the_growing_excluded_COUNT_is_not_the_trigger(tmp_path, caplog):
    """⚠ THE VERDICT IS THE EVENT, NOT THE COUNT.

    The live count climbed 253 -> 309 over five days as nightly bench solves
    landed. The world this fails in: the level is decided by
    `n_bench_misaligned` changing, so a standing exclusion re-announces under
    a new number — the same noise wearing a disguise.
    """
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 400, seed=1, origin="user")
    _feed_ranking(t, 100, seed=2, origin="bench", invert=True)
    t.fit()
    first = t.fit()
    assert first.bench_verdict == BEATS_NO

    _feed_ranking(t, 60, seed=3, origin="bench", invert=True)   # more bench
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        grown = t.fit()

    assert grown.bench_verdict == BEATS_NO
    assert grown.n_bench_misaligned > first.n_bench_misaligned, (
        "fixture no longer grows the count it is about")
    assert not _lines(caplog, logging.WARNING, BENCH_LINE)
    assert _lines(caplog, logging.DEBUG, BENCH_LINE)


def test_a_bench_verdict_that_CHANGES_is_loud_again(tmp_path, caplog):
    """A corpus that stops ranking backwards retires the exclusion — and the
    operator must hear that the fit's population just changed shape.

    The world this fails in: the level is decided once and cached, so the
    exclusion silently lifts and 120 rows rejoin the fit unannounced.
    """
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 200, seed=1, origin="user")
    _feed_ranking(t, 120, seed=2, origin="bench", invert=True)
    t.fit()
    assert t.fit().bench_verdict == BEATS_NO
    assert t.fit().bench_verdict == BEATS_NO          # now steady

    # Swamp the auxiliary population with rows that rank the right way.
    _feed_ranking(t, 600, seed=4, origin="bench")
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        flipped = t.fit()

    assert flipped.bench_verdict != BEATS_NO, (
        f"fixture no longer flips the verdict: {flipped.bench_verdict}")
    assert flipped.n_bench_misaligned == 0
    # The exclusion line does not fire at all now (nothing was excluded); the
    # transition is visible in the params every consumer already reads.
    assert not _lines(caplog, logging.DEBUG, BENCH_LINE)
    assert not _lines(caplog, logging.WARNING, BENCH_LINE)


# ── the base-rate line ─────────────────────────────────────────────────────

def test_a_standing_base_rate_verdict_is_announced_once(tmp_path, caplog):
    """The live state: 99 INDISTINGUISHABLE lines over five days, one fact.

    No bench rows here, so every refit sees an identical population and the
    verdict is genuinely unchanged rather than coincidentally equal.
    """
    t = CalibrationTracker(tmp_path)
    _feed_live_regime(t, 900, seed=21)
    first = t.fit()
    assert first.beats_base_rate in C.NOT_INFORMATIVE, (
        f"fixture no longer reaches the branch under test: {first.beats_base_rate}")

    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        for _ in range(3):
            again = t.fit()

    assert again.beats_base_rate == first.beats_base_rate
    assert not _any_line(caplog, logging.WARNING, BASE_RATE_LINES)
    kept = _any_line(caplog, logging.DEBUG, BASE_RATE_LINES)
    assert len(kept) == 3, f"the line vanished instead of moving: {caplog.text!r}"
    # The measurements the warning exists to carry are still in it.
    assert "AS A PROBABILITY" in kept[0].getMessage()


def test_the_first_base_rate_verdict_is_loud(tmp_path, caplog):
    t = CalibrationTracker(tmp_path)
    _feed_live_regime(t, 900, seed=21)
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.beats_base_rate in C.NOT_INFORMATIVE
    assert len(_any_line(caplog, logging.WARNING, BASE_RATE_LINES)) == 1


def test_a_flip_BETWEEN_the_two_bad_verdicts_is_loud(tmp_path, caplog):
    """⚠ ONE DECISION FOR BOTH BRANCHES.

    `no` and `indistinguishable` are two values of ONE verdict and are logged
    from two branches. The world this fails in: each branch compares against
    itself, so a `no` -> `indistinguishable` flip — the score going from
    demonstrably worse than a constant to merely unproven, which changes what
    it may be used for — lands in the other branch and reads as steady state.
    """
    t = CalibrationTracker(tmp_path)
    _feed_live_regime(t, 900, seed=21)
    t.fit()
    # Rewrite the recorded verdict to the SIBLING bad value, exactly as a
    # previous refit on a worse corpus would have left it.
    path = tmp_path / "calibration_params.json"
    d = json.loads(path.read_text())
    assert d["beats_base_rate"] == BEATS_INDISTINGUISHABLE
    d["beats_base_rate"] = BEATS_NO
    path.write_text(json.dumps(d))

    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.beats_base_rate == BEATS_INDISTINGUISHABLE
    assert len(_any_line(caplog, logging.WARNING, BASE_RATE_LINES)) == 1, (
        "a verdict flip between the two branches read as steady state")


def test_a_verdict_that_becomes_a_LICENCE_stops_warning(tmp_path, caplog):
    """The good news case: nothing to warn about, and nothing left over."""
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 600, seed=7, origin="user")
    t.fit()
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.beats_base_rate == BEATS_YES, (
        f"fixture no longer clears the base rate: {p.beats_base_rate}")
    assert not _any_line(caplog, logging.WARNING, BASE_RATE_LINES)
    assert not _any_line(caplog, logging.DEBUG, BASE_RATE_LINES)


# ── where the baseline lives ───────────────────────────────────────────────

def test_the_baseline_is_the_params_FILE_not_process_memory(tmp_path, caplog):
    """A restart must not re-announce a standing condition.

    The world this fails in: the last-announced verdict is remembered on the
    tracker instance, so every agent restart replays every standing warning —
    and there are then two records of "what did we last say", free to
    disagree with the file every other consumer reads.
    """
    t = CalibrationTracker(tmp_path)
    _feed_live_regime(t, 900, seed=21)
    t.fit()
    t.fit()
    fresh = CalibrationTracker(tmp_path)          # a new process, same disk
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = fresh.fit()
    assert p.beats_base_rate in C.NOT_INFORMATIVE
    assert not _any_line(caplog, logging.WARNING, BASE_RATE_LINES)
    assert _any_line(caplog, logging.DEBUG, BASE_RATE_LINES)


def test_an_unreadable_params_file_announces(tmp_path, caplog):
    """`load_params` returns None on a corrupt file. That is "no verdict on
    record", which is news — never "unchanged".

    The world this fails in: a missing baseline is treated as equal to the
    current verdict, so a corpus whose params file is being destroyed every
    cycle goes quiet at exactly the wrong moment.
    """
    t = CalibrationTracker(tmp_path)
    _feed_live_regime(t, 900, seed=21)
    t.fit()
    (tmp_path / "calibration_params.json").write_text("{ not json")
    caplog.clear()   # `at_level` sets the LEVEL; capture was already on
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.beats_base_rate in C.NOT_INFORMATIVE
    assert len(_any_line(caplog, logging.WARNING, BASE_RATE_LINES)) == 1


# ── the justification, checked like a claim ────────────────────────────────

def test_the_demoted_state_is_still_published(tmp_path):
    """The docstring's promise: demoting the log does not make the state
    unreadable, because the verdict is a FIELD every consumer already asks
    for.

    The world this fails in: the log was the only surface carrying it, and
    quieting the log is how the operator loses the fact entirely.
    """
    t = CalibrationTracker(tmp_path)
    _feed_ranking(t, 200, seed=1, origin="user")
    _feed_ranking(t, 120, seed=2, origin="bench", invert=True)
    t.fit()
    t.fit()
    steady = t.fit()
    assert steady.bench_verdict == BEATS_NO
    assert steady.n_bench_misaligned == 120
    stored = json.loads((tmp_path / "calibration_params.json").read_text())
    assert stored["bench_verdict"] == BEATS_NO
    assert stored["n_bench_misaligned"] == 120
    assert stored["beats_base_rate"] == steady.beats_base_rate
    # …and the loader reconstructs them, which is the path the metacog CALIB
    # line and `introspect learning` read.
    reloaded = CalibrationTracker(tmp_path).load_params()
    assert reloaded.bench_verdict == BEATS_NO
    assert reloaded.beats_base_rate == steady.beats_base_rate


# ── the map-not-applied line, the sibling that has never fired live ────────

def _feed_anticorrelated(t, n, *, seed):
    """Good turns score LOW, so the Platt slope comes out negative and the map
    is refused — `map_status="rejected_inverted"`."""
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.7
        c = rng.uniform(0.05, 0.4) if good else rng.uniform(0.6, 0.95)
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4))


def test_a_standing_rejected_map_is_announced_once(tmp_path, caplog):
    """The third warning on the same rule.

    It has never fired on the live log, which is exactly why it is easy to
    leave behind — the sibling one revision back. The world it fails in: this
    corpus arrives, and the operator gets an hourly line about a map that was
    refused once and has stayed refused ever since.
    """
    t = CalibrationTracker(tmp_path)
    _feed_anticorrelated(t, 400, seed=31)
    first = t.fit()
    assert first.map_status == "rejected_inverted", (
        f"fixture no longer reaches the branch under test: {first.map_status}")

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        for _ in range(3):
            again = t.fit()

    assert again.map_status == "rejected_inverted"
    assert not _lines(caplog, logging.WARNING, MAP_LINE)
    kept = _lines(caplog, logging.DEBUG, MAP_LINE)
    assert len(kept) == 3, f"the line vanished instead of moving: {caplog.text!r}"


def test_the_first_rejected_map_is_loud(tmp_path, caplog):
    t = CalibrationTracker(tmp_path)
    _feed_anticorrelated(t, 400, seed=31)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        p = t.fit()
    assert p.map_status == "rejected_inverted"
    assert len(_lines(caplog, logging.WARNING, MAP_LINE)) == 1


def test_a_map_that_comes_BACK_is_loud(tmp_path, caplog):
    """`rejected_inverted` -> `applied` is the news that the shipped predictor
    just changed from the raw composite to the calibrated one.

    The world it fails in: the level is decided from `map_status` itself
    rather than against what was recorded, so no map transition is ever
    announced in either direction.
    """
    t = CalibrationTracker(tmp_path)
    _feed_anticorrelated(t, 400, seed=31)
    t.fit()
    assert t.fit().map_status == "rejected_inverted"      # now steady
    _feed_ranking(t, 1200, seed=41, origin="user")        # swamp it, right way up
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=C.logger.name):
        back = t.fit()
    assert back.map_status == "applied", (
        f"fixture no longer flips the map: {back.map_status}")
    # Nothing to warn about any more, and no leftover line either.
    assert not _lines(caplog, logging.WARNING, MAP_LINE)
    assert not _lines(caplog, logging.DEBUG, MAP_LINE)


# ── the cold-start baseline ────────────────────────────────────────────────

def test_the_cold_start_sentinel_carries_no_verdict():
    """`NOTHING_RECORDED` must not look like a fit that decided something.

    The world it fails in: it is given a plausible default (say
    `BEATS_UNKNOWN`, which IS in the vocabulary), and the first refit after a
    corpus starts failing its base-rate comparison matches that default and
    goes out at DEBUG — the one announcement that mattered, lost.
    """
    for verdict in (NOTHING_RECORDED.beats_base_rate,
                    NOTHING_RECORDED.bench_verdict,
                    NOTHING_RECORDED.map_status):
        assert verdict is None
        assert verdict not in C._VERDICTS
        assert announce_level(verdict, BEATS_UNKNOWN) == logging.WARNING
        assert announce_level(verdict, BEATS_NO) == logging.WARNING


def test_the_bench_line_cannot_fire_before_a_first_fit():
    """The reachability the sentinel's equivalence argument rests on — pinned,
    because a justification is a claim.

    With no params there is no instrument, so the direction gate measures
    nothing and excludes nothing: the bench warning cannot run at cold start.
    ⚠ If this test ever fails because the gate learned a cold-start
    instrument, the bench site's cold-start branch becomes REACHABLE and
    `NOTHING_RECORDED.bench_verdict` starts deciding a real announcement —
    go read that sentinel's note before changing anything else.
    """
    rows = []
    rng = random.Random(5)
    for _ in range(120):
        good = rng.random() < 0.8
        c = 1.0 - (rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4))
        rows.append(C.CalibrationSample(
            composite=round(c, 4), entropy_component=0.5,
            competence_component=round(c, 4), uncertainty_pressure=0.0,
            outcome=1.0 if good else 0.0, domain="", ts="2026-08-01T00:00:00Z",
            source="turn", req_id="", epoch=C.CURRENT_EPOCH, origin="bench"))
    kept, verdict, ci, _dg = C._apply_bench_direction_gate(rows, None)
    assert verdict == BEATS_UNKNOWN and ci is None
    assert len(kept) == len(rows), "rows were excluded with no instrument to judge them"
