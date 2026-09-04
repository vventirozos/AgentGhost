"""The score's one licensed use, and asking where the corpus is blind (§4ER).

Confidence was computed 865 times over 209 metacog summaries, `below_threshold`
fired 118 times, and its ONLY consumer — metacog arbitration — is hard-gated off
by `_METACOG_ARBITER_ENABLED`. Arbitrations: **zero**. Meanwhile §4EO/§4EP
established what the score may and may not be read as: `beats_base_rate =
indistinguishable` (and provably unresolvable at this corpus), `ranks_outcomes =
yes` (AUC 0.634). Ordering turns is the one thing it is entitled to do.

And the human channel is not the bottleneck — the web console, the `ghost` CLI
and Slack all take one click — yet 39 days produced **7 labels**, because nobody
is ever asked.

The property under review: **the score is used only for ranking, the licence
gates that use, and the operator is asked exactly where a machine label is
missing.**
"""

import datetime
import logging
import types

import pytest

from ghost_agent.core import calibration as C
from ghost_agent.core.calibration import (BEATS_INDISTINGUISHABLE, BEATS_NO,
                                          BEATS_UNKNOWN, BEATS_YES,
                                          CalibrationTracker)


def _fill(t, n, *, seed=3, days_ago=1):
    import random
    rng = random.Random(seed)
    for i in range(n):
        good = rng.random() < 0.8
        c = rng.uniform(0.6, 0.95) if good else rng.uniform(0.05, 0.4)
        t.record(composite=round(c, 4), outcome=1.0 if good else 0.0,
                 entropy_component=0.5, competence_component=round(c, 4),
                 req_id=f"req{i:04d}")


# ── the ranking, gated on the licence ──────────────────────────────────────

def test_the_ranking_is_withheld_without_an_affirmative_licence(tmp_path):
    """⚠ TEST FOR THE LICENCE, NOT FOR ITS ABSENCE. `in NOT_INFORMATIVE`
    fails OPEN on a typo or a future verdict word; only `yes` may unlock a
    use of this score.

    The world it fails in: the corpus stops ranking, the verdict flips to
    `no` — "orders turns BACKWARDS" — and the report keeps printing a list
    that now surfaces the agent's *best* answers as its shakiest.
    """
    t = CalibrationTracker(tmp_path)
    rows, why = t.lowest_confidence_turns()
    assert rows == [] and "no fitted params" in why

    _fill(t, 300)
    p = t.fit()
    assert p.ranks_outcomes == BEATS_YES, "fixture no longer ranks"
    rows, why = t.lowest_confidence_turns(limit=5, days=3650)
    assert rows and not why

    import json
    path = tmp_path / "calibration_params.json"
    for verdict in (BEATS_NO, BEATS_INDISTINGUISHABLE, BEATS_UNKNOWN,
                    "a-typo", None):
        d = json.loads(path.read_text())
        d["ranks_outcomes"] = verdict
        path.write_text(json.dumps(d))
        rows, why = CalibrationTracker(tmp_path).lowest_confidence_turns(
            days=3650)
        assert rows == [], verdict
        assert "withheld" in why, (verdict, why)


def test_an_uncoerced_verdict_object_is_still_refused(tmp_path):
    """The defensive half of the licence test, reachable only in memory.

    `load_params` coerces an unrecognised word on disk to `unknown`, so a
    typo in the FILE can never reach the comparison raw — which makes
    `_known_verdict(...) != BEATS_YES` and `in NOT_INFORMATIVE` behave
    identically for every file. They do NOT behave identically for a params
    object built in memory: `in NOT_INFORMATIVE` is False for an unknown
    word, i.e. it reads as LICENSED.

    The world it fails in: a caller constructs `FittedParams` directly (an
    eval harness, a replay, a future in-memory refit) with a verdict this
    build does not recognise, and the ranking is shown on a licence nobody
    granted.
    """
    t = CalibrationTracker(tmp_path)
    _fill(t, 300)
    assert t.fit().ranks_outcomes == BEATS_YES
    good = t.load_params()
    for raw in ("a-typo", "YES", "probably", ""):
        import dataclasses
        t.load_params = lambda _r=raw: dataclasses.replace(good,
                                                           ranks_outcomes=_r)
        rows, why = t.lowest_confidence_turns(days=3650)
        assert rows == [], raw
        assert "withheld" in why, (raw, why)


def test_absent_is_not_withheld(tmp_path):
    """An empty list and a refusal are different answers. The world it fails
    in: the report prints nothing and the operator reads "no shaky turns this
    week" when the truth is "the score is not allowed to say"."""
    t = CalibrationTracker(tmp_path)
    _fill(t, 300)
    t.fit()
    rows, why = t.lowest_confidence_turns(days=3650)
    assert rows and why == ""
    # A window with no rows says so, distinctly from a licence refusal.
    # ⚠ Reached by emptying the population, not by shrinking the window: the
    # fixture stamps rows with `now`, so even `days=1` contains them — the
    # first version of this test asserted an empty window it could not
    # actually produce.
    t2 = CalibrationTracker(tmp_path)
    t2._load_epoch = lambda *a, **k: []
    rows2, why2 = t2.lowest_confidence_turns(days=1)
    assert rows2 == [] and "no rows" in why2 and "withheld" not in why2


def test_it_ranks_lowest_first_and_honours_the_limit(tmp_path):
    t = CalibrationTracker(tmp_path)
    _fill(t, 300)
    t.fit()
    rows, _ = t.lowest_confidence_turns(limit=7, days=3650)
    assert len(rows) == 7
    assert [r.composite for r in rows] == sorted(r.composite for r in rows)
    allrows = [s for s in t._load_epoch() if s.origin != "bench"]
    assert rows[0].composite == min(s.composite for s in allrows)


def test_bench_rows_are_not_offered_for_review(tmp_path):
    """The operator reviews their own turns; a bench solve is not one."""
    t = CalibrationTracker(tmp_path)
    _fill(t, 300)
    for i in range(40):
        t.record(composite=0.01, outcome=1.0, entropy_component=0.5,
                 competence_component=0.01, origin="bench",
                 req_id=f"bench-{i}")
    t.fit()
    rows, _ = t.lowest_confidence_turns(limit=10, days=3650)
    assert all(getattr(r, "origin", "user") != "bench" for r in rows)


# ── asking for a label ─────────────────────────────────────────────────────

def _agent(tracker, *, ranks=BEATS_YES):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = types.SimpleNamespace(calibration_tracker=tracker)
    return a


def _reading(c):
    return types.SimpleNamespace(raw_pre_penalty_composite=c, composite=c)


@pytest.fixture
def _fitted(tmp_path):
    t = CalibrationTracker(tmp_path)
    _fill(t, 300)
    assert t.fit().ranks_outcomes == BEATS_YES
    return t


def test_a_verified_turn_is_never_asked_about(_fitted):
    """⚠ ASK WHERE THE CORPUS IS BLIND. A turn that already carries a verdict
    has a label; spending the operator's attention there buys a duplicate.

    The world it fails in: the agent asks about turns the verifier already
    settled, the operator learns the asks are noise, and the one channel that
    produces ground truth goes unanswered.
    """
    a = _agent(_fitted)
    assert a._label_request_note(_reading(0.01), ("failed", "x")) == ""
    assert a._label_request_note(_reading(0.01), ("passed", "")) == ""


def test_only_the_bottom_band_is_asked_about(_fitted):
    a = _agent(_fitted)
    assert a._label_request_note(_reading(0.01), None), "the lowest scored turn was not asked about"
    a2 = _agent(_fitted)
    assert a2._label_request_note(_reading(0.99), None) == ""


def test_it_asks_at_most_once_a_day(_fitted):
    a = _agent(_fitted)
    assert a._label_request_note(_reading(0.01), None)
    assert a._label_request_note(_reading(0.01), None) == "", "asked twice"
    a._last_label_ask_at = datetime.datetime.now() - datetime.timedelta(hours=48)
    assert a._label_request_note(_reading(0.01), None)


def test_the_ask_is_disarmable(_fitted, monkeypatch):
    monkeypatch.setenv("GHOST_LABEL_ASK_HOURS", "0")
    assert _agent(_fitted)._label_request_note(_reading(0.01), None) == ""


def test_the_ask_never_states_a_probability(_fitted):
    """`beats_base_rate` is `indistinguishable`, so the score may not say "I
    am 42% sure". It may only rank — and the wording has to reflect that.

    The world it fails in: the ask quotes the number, and the operator
    reasonably reads it as a calibrated probability the module explicitly
    refuses to claim.
    """
    note = _agent(_fitted)._label_request_note(_reading(0.01), None)
    assert "shakier" in note
    assert "%" not in note
    assert "0.0" not in note and "confidence of" not in note


def test_the_ask_is_gated_on_the_same_licence(_fitted, tmp_path):
    import json
    path = tmp_path / "calibration_params.json"
    d = json.loads(path.read_text())
    d["ranks_outcomes"] = BEATS_NO
    path.write_text(json.dumps(d))
    a = _agent(CalibrationTracker(tmp_path))
    assert a._label_request_note(_reading(0.01), None) == ""


def test_a_thin_corpus_is_not_asked_about(tmp_path):
    """"Low" is meaningless on five rows: the band is computed from the rows
    themselves, so it needs enough of them to have a bottom."""
    t = CalibrationTracker(tmp_path)
    _fill(t, 300, seed=9)
    assert t.fit().ranks_outcomes == BEATS_YES
    a = _agent(t)
    # ⚠ ASSERT THE PRECONDITION, DO NOT BRANCH ON IT. The first version wrapped
    # the assertion in `if len(rows) < 10:` — so a fixture that was not thin
    # asserted NOTHING, and a mutation deleting the floor survived.
    # ⚠ Real rows, not `object()`. A stub without `.composite` makes the
    # helper raise into its own catch-all and return "" for the WRONG reason
    # — the mutant that deletes the floor then survives, which is exactly
    # what a mutation sweep caught here.
    t.lowest_confidence_turns = lambda **kw: (
        [types.SimpleNamespace(composite=0.9) for _ in range(9)], "")
    assert a._label_request_note(_reading(0.01), None) == "", (
        "asked for a label off a corpus too thin to know what 'low' means")
    t.lowest_confidence_turns = lambda **kw: (
        [types.SimpleNamespace(composite=0.9)] * 20, "")
    assert a._label_request_note(_reading(0.01), None), (
        "fixture no longer reaches the ask at all")


def test_the_helper_never_raises():
    """A solicitation that breaks a turn is not worth a label."""
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = types.SimpleNamespace(calibration_tracker=None)
    assert a._label_request_note(_reading(0.1), None) == ""
    a.context = types.SimpleNamespace()
    assert a._label_request_note(None, None) == ""
    a.context = types.SimpleNamespace(
        calibration_tracker=types.SimpleNamespace(
            load_params=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert a._label_request_note(_reading(0.1), None) == ""
