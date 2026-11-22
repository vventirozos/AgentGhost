"""§4VF (operator 2026-11-22: "i want mirror to show accurate data, not data based on old rules that no longer
apply"): every automatic move carries the rule version that made it; a path made under another rule is worked out
again from the start under the current one (``character.replay``), and ``mirror`` shows only that path."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import loop as LOOP

UTC = datetime.timezone.utc
_PULL = {"playful": {"humour": 1}, "serious": {"humour": -1}, "formal": {"humour": -1, "directness": -1}}


def _stated_pulls(days):
    return {str(d["date"]): dict(d["pulls"]) for d in C._by_date(days) if d.get("pulls")}


def _stated_shifts(days):
    return {str(d["date"]): {k: float(v) for k, v in d["pulls"].items()} for d in C._by_date(days)
            if d.get("pulls")}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # the same seam as test_4ug_character: each day states its pulls, read as one spread each
    monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    monkeypatch.setattr(C, "day_pulls", _stated_pulls)
    monkeypatch.setattr(C, "day_shifts", _stated_shifts)
    monkeypatch.setattr(C, "SHIFT_OUT", C.SHIFT_IN)
    C._MEMO.clear()
    yield


def _days(*names_by_day, start=datetime.date(2026, 11, 1)):
    out = []
    for i, names in enumerate(names_by_day):
        pulls = {}
        for n in names:
            pulls.update(_PULL[n])
        out.append({"date": (start + datetime.timedelta(days=i)).isoformat(), "pulls": pulls})
    return out


def _sh(home) -> Path:
    p = home / C.SELFHOOD_REL
    p.mkdir(parents=True, exist_ok=True)
    return p


def _old_home(tmp_path, rows, dials, *, through, pins=(), changed=None) -> Path:
    """A character as rule 1 left it: a file with no rule version and a log whose moves carry none."""
    root = _sh(tmp_path)
    (root / C.CHARACTER_FILENAME).write_text(json.dumps(
        {"version": 1, "dials": dials, "pins": list(pins), "daily_through": through, "changed": changed or {}}))
    (root / C.LOG_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


_START = {"at": "2026-11-01T10:00:00Z", "kind": C.KIND_START, "dials": dict(C.START), "why": "the starting character"}
_OLD_MOVE = {"at": "2026-11-04T03:00:00Z", "kind": C.KIND_MOVE, "dial": "humour", "from": 1, "to": 2,
             "why": "3 days clearly better than its usual against 0 the other way (OLD RULE WORDS)",
             "days_up": ["2026-11-01"], "days_down": []}


def test_a_path_made_by_the_old_rule_is_worked_out_again_under_the_current_one(tmp_path):
    # catches: a replay that keeps the old move or its reason, one that moves nothing, one that leaves the dials
    # where the old rule put them, and one that throws the old log away instead of keeping it beside the new one
    days = _days(["formal"], ["formal"], ["formal"])
    home = _old_home(tmp_path, [_START, _OLD_MOVE], {"warmth": -1, "humour": 2, "directness": 0},
                     through="2026-11-03", changed={"humour": "2026-11-03"})
    assert C.load(home)["rule"] == 0 and C.needs_replay(C.load(home), C.log_rows(home))
    said = C.replay(home, days, now=datetime.datetime(2026, 11, 22, 10, tzinfo=UTC))
    assert said.startswith("the rule that moves the dials changed (now rule 2)")
    assert "1 automatic move of the old rule replaced by 2" in said
    state = C.load(home)
    # the current rule over the same three days: humour back toward 0, directness away from it
    assert state["dials"] == {"warmth": -1, "humour": 0, "directness": -1} and state["rule"] == C.RULE_VERSION
    assert state["changed"] == {"humour": "2026-11-03", "directness": "2026-11-03"}
    assert state["daily_through"] == "2026-11-03"
    rows = C.log_rows(home)
    assert [r["kind"] for r in rows] == [C.KIND_START, C.KIND_MOVE, C.KIND_MOVE, C.KIND_REPLAY]
    moves = [r for r in rows if r["kind"] == C.KIND_MOVE]
    assert all(r["rule"] == C.RULE_VERSION and r["through"] == "2026-11-03" for r in moves)
    assert all(r["why"].startswith("3 days read worse than its usual") for r in moves)
    assert "OLD RULE WORDS" not in (home / C.SELFHOOD_REL / C.LOG_FILENAME).read_text()
    kept = list((home / C.SELFHOOD_REL).glob("character.log.before-rule2.*.jsonl"))
    assert len(kept) == 1 and "OLD RULE WORDS" in kept[0].read_text()
    # the move is stamped at the close of the day it read, as the daily rule would have written it
    assert C._mood_day(datetime.datetime.fromisoformat(moves[0]["at"].replace("Z", "+00:00"))) == "2026-11-04"
    # done once: nothing more to replay
    assert not C.needs_replay(C.load(home), C.log_rows(home)) and C.replay(home, days) == ""


def test_mirror_shows_the_replayed_path_and_never_the_old_reasons(tmp_path):
    days = _days(["formal"], ["formal"], ["formal"])
    home = _old_home(tmp_path, [_START, _OLD_MOVE], {"warmth": -1, "humour": 2, "directness": 0},
                     through="2026-11-03")
    C.replay(home, days, now=datetime.datetime(2026, 11, 22, 10, tzinfo=UTC))
    b = LOOP.banner(home)
    assert "OLD RULE WORDS" not in b and "humour +1 → +2" not in b
    assert "(2 moves by itself so far)" in b
    assert "humour +1 → 0, directness 0 → -1, by itself" in b and "3 days read worse than its usual" in b
    assert ("the rule that moves the dials changed, so the path was worked out again under the current one "
            "(humour +2 → 0, directness 0 → -1)") in b
    assert C.story(home).count("moved by itself") == 2 and "worked out again" in C.story(home)


def test_the_owners_word_stands_through_a_replay(tmp_path):
    # catches: a replay that drops the owner's setting or pin, one that keeps a veto of a move the replayed path
    # never made (a ✓ for +1 → +1), and one that keeps the "from" the OLD path had
    rows = [_START,
            {"at": "2026-11-02T10:00:00Z", "kind": C.KIND_OWNER, "dial": "warmth", "from": -1, "to": 1,
             "why": "the owner said so"},
            {"at": "2026-11-02T11:00:00Z", "kind": C.KIND_PIN, "dial": "warmth"},
            _OLD_MOVE,
            {"at": "2026-11-04T12:00:00Z", "kind": C.KIND_VETO, "dial": "humour", "from": 2, "to": 1,
             "of": _OLD_MOVE["at"], "of_to": 2, "why": "the owner took the last automatic move back"},
            {"at": "2026-11-05T12:00:00Z", "kind": C.KIND_OWNER, "dial": "directness", "from": 0, "to": 2}]
    home = _old_home(tmp_path, rows, {"warmth": 1, "humour": 1, "directness": 2}, through="2026-11-05",
                     pins=["warmth"])
    C.replay(home, _days([], [], [], [], []))
    state = C.load(home)
    assert state["dials"] == {"warmth": 1, "humour": 1, "directness": 2} and state["pins"] == ["warmth"]
    got = C.log_rows(home)
    assert [r["kind"] for r in got] == [C.KIND_START, C.KIND_OWNER, C.KIND_PIN, C.KIND_OWNER, C.KIND_REPLAY]
    assert got[1]["from"] == -1 and got[3]["dial"] == "directness" and got[3]["from"] == 0
    # a veto whose dial the replayed path left elsewhere stands as the owner's setting
    home2 = _old_home(tmp_path / "b", [_START, _OLD_MOVE, dict(rows[4], to=0)],
                      {"warmth": -1, "humour": 0, "directness": 0}, through="2026-11-05")
    C.replay(home2, _days([], [], [], [], []))
    veto = [r for r in C.log_rows(home2) if r["kind"] == C.KIND_OWNER]
    assert len(veto) == 1 and (veto[0]["from"], veto[0]["to"]) == (1, 0) and "of" not in veto[0]
    assert "kept as the owner's setting" in veto[0]["why"]
    assert C.load(home2)["dials"]["humour"] == 0


def test_a_reset_re_bases_the_replayed_path(tmp_path):
    reset = {"at": "2026-11-05T12:00:00Z", "kind": C.KIND_START, "by": C.BY_RESET, "dials": dict(C.START),
             "from": {"warmth": -1, "humour": 2, "directness": 0}, "pins_from": [], "why": "the owner reset"}
    home = _old_home(tmp_path, [_START, _OLD_MOVE, reset], dict(C.START), through="2026-11-07",
                     changed={d: "2026-11-05" for d in C.DIALS})
    # three playful days before the reset (humour away from +1 at the 3rd) and two after it (not enough)
    C.replay(home, _days(["playful"], ["playful"], ["playful"], [], [], ["playful"], ["playful"]))
    rows = C.log_rows(home)
    assert [r["kind"] for r in rows] == [C.KIND_START, C.KIND_MOVE, C.KIND_START, C.KIND_REPLAY]
    assert rows[1]["rule"] == C.RULE_VERSION and C.is_reset_row(rows[2])
    assert rows[2]["from"] == {"warmth": -1, "humour": 2, "directness": 0}
    assert C.load(home)["dials"] == C.START


def test_a_replay_cut_short_is_done_whole_at_the_next_try(tmp_path, monkeypatch):
    # catches: a replay that writes the log first and the dials never (a log of the new rule over dials of the
    # old one, and nothing left to say a replay is due), and one that marks the file done before the log is
    days = _days(["formal"], ["formal"], ["formal"])
    home = _old_home(tmp_path, [_START, _OLD_MOVE], {"warmth": -1, "humour": 2, "directness": 0},
                     through="2026-11-03")
    before = (home / C.SELFHOOD_REL / C.LOG_FILENAME).read_text()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(C.os, "replace", boom)
    with pytest.raises(OSError):
        C.replay(home, days)
    monkeypatch.undo()
    monkeypatch.setattr(C, "day_pulls", _stated_pulls)
    monkeypatch.setattr(C, "day_shifts", _stated_shifts)
    monkeypatch.setattr(C, "SHIFT_OUT", C.SHIFT_IN)
    assert (home / C.SELFHOOD_REL / C.LOG_FILENAME).read_text() == before
    assert C.needs_replay(C.load(home), C.log_rows(home))
    C.replay(home, days)
    assert C.load(home)["dials"] == {"warmth": -1, "humour": 0, "directness": -1}
    assert not C.needs_replay(C.load(home), C.log_rows(home))
    assert [r["kind"] for r in C.log_rows(home)] == [C.KIND_START, C.KIND_MOVE, C.KIND_MOVE, C.KIND_REPLAY]


def test_the_daily_rule_replays_first_and_stamps_its_own_moves(tmp_path):
    days = _days(["formal"], ["formal"], ["formal"])
    home = _old_home(tmp_path, [_START, _OLD_MOVE], {"warmth": -1, "humour": 2, "directness": 0},
                     through="2026-11-03")
    assert C.daily(home, days, now=datetime.datetime(2026, 11, 6, 12, tzinfo=UTC)) == []
    assert C.load(home)["dials"] == {"warmth": -1, "humour": 0, "directness": -1}
    # the rule runs on from there, and what it writes carries its version
    more = days + _days(["playful"], ["playful"], ["playful"], start=datetime.date(2026, 11, 4))
    said = C.daily(home, more, now=datetime.datetime(2026, 11, 7, 12, tzinfo=UTC))
    assert said and said[0].startswith("humour +0 → +1")
    last = C.log_rows(home)[-1]
    assert last["kind"] == C.KIND_MOVE and last["rule"] == C.RULE_VERSION


def test_a_new_rule_version_replays_and_a_file_of_no_moves_does_not(tmp_path, monkeypatch):
    home = _sh(tmp_path) and tmp_path
    C.daily(home, _days(["playful"], ["playful"], ["playful"]))
    assert C.load(home)["rule"] == C.RULE_VERSION and not C.needs_replay(C.load(home), C.log_rows(home))
    monkeypatch.setattr(C, "RULE_VERSION", C.RULE_VERSION + 1)
    assert C.needs_replay(C.load(home), C.log_rows(home))          # the file and the move say the old version
    # a file written before versions, with no automatic move in its log: nothing to replay; the rule stamps it
    monkeypatch.undo()
    home2 = _old_home(tmp_path / "b", [_START], dict(C.START), through="")
    assert not C.needs_replay(C.load(home2), C.log_rows(home2))
    C.daily(home2, _days([], [], []))
    assert C.load(home2)["rule"] == C.RULE_VERSION


def test_a_log_with_no_start_row_starts_at_its_first_row_and_the_rule_runs_on(tmp_path):
    # catches (review r1, MAJOR): a log whose start row was lost raised on every idle window — the daily rule
    # never ran again and the old moves stayed
    home = _old_home(tmp_path, [_OLD_MOVE], {"warmth": -1, "humour": 2, "directness": 0}, through="2026-11-03")
    assert C.daily(home, _days(["formal"], ["formal"], ["formal"]),
                   now=datetime.datetime(2026, 11, 6, 12, tzinfo=UTC)) == []
    rows = C.log_rows(home)
    assert rows[0]["kind"] == C.KIND_START and rows[0]["at"] == _OLD_MOVE["at"] and "missing" in rows[0]["why"]
    assert C.load(home)["dials"] == {"warmth": -1, "humour": 0, "directness": -1}
    assert not C.needs_replay(C.load(home), rows)


def _written(days, stamps):
    return [dict(d, written_at=t) for d, t in zip(days, stamps)]


def test_closes_are_replayed_at_the_moments_they_ran(tmp_path):
    # catches (review r1): the first close (the start's own, over the days before it) skipped; one close per day
    # where the agent closed several days in one window; an owner's setting on the morning after a day applied
    # AFTER that day's close
    start = dict(_START, at="2026-11-03T18:00:00Z")
    # three playful days, all written at the start (the deploy's first window): ONE close at the start moves humour
    days = _written(_days(["playful"], ["playful"], ["playful"]), ["2026-11-03T18:00:00Z"] * 3)
    home = _old_home(tmp_path, [start, dict(_OLD_MOVE, at="2026-11-05T03:00:00Z")], dict(C.START),
                     through="2026-11-03")
    C.replay(home, days)
    moves = [r for r in C.log_rows(home) if r["kind"] == C.KIND_MOVE]
    assert [(m["dial"], m["at"], m["through"]) for m in moves] == [("humour", "2026-11-03T18:00:00Z", "2026-11-03")]
    # two closes in one window are one close: three formal days written together read once, not day by day
    days2 = _written(_days(["formal"], ["formal"], ["formal"], ["formal"]),
                     ["2026-11-03T18:00:00Z"] * 2 + ["2026-11-06T02:00:00Z"] * 2)
    home2 = _old_home(tmp_path / "b", [start, dict(_OLD_MOVE, at="2026-11-05T03:00:00Z")], dict(C.START),
                      through="2026-11-04")
    C.replay(home2, days2)
    moves2 = [r for r in C.log_rows(home2) if r["kind"] == C.KIND_MOVE]
    assert {m["at"] for m in moves2} == {"2026-11-06T02:00:00Z"} and {m["through"] for m in moves2} == {"2026-11-04"}
    # the owner set humour on the morning after the third day, before that night's close: the close sees it
    days3 = _written(_days(["playful"], ["playful"], ["playful"]),
                     ["2026-11-02T02:00:00Z", "2026-11-03T02:00:00Z", "2026-11-04T02:00:00Z"])
    own = {"at": "2026-11-03T20:00:00Z", "kind": C.KIND_OWNER, "dial": "humour", "from": 1, "to": 0}
    home3 = _old_home(tmp_path / "c", [dict(_START, at="2026-11-01T10:00:00Z"), own,
                                       dict(_OLD_MOVE, at="2026-11-04T02:00:00Z")],
                      {"warmth": -1, "humour": 0, "directness": 0}, through="2026-11-03")
    C.replay(home3, days3)
    assert [r["kind"] for r in C.log_rows(home3)] == [C.KIND_START, C.KIND_OWNER, C.KIND_REPLAY]
    assert C.load(home3)["dials"]["humour"] == 0


def test_a_plain_start_lets_the_pins_go_as_a_removed_file_does(tmp_path):
    rows = [_START, {"at": "2026-11-01T11:00:00Z", "kind": C.KIND_PIN, "dial": "humour"},
            dict(_START, at="2026-11-01T12:00:00Z"), _OLD_MOVE]
    home = _old_home(tmp_path, rows, {"warmth": -1, "humour": 2, "directness": 0}, through="2026-11-03")
    C.replay(home, _days(["playful"], ["playful"], ["playful"]))
    assert C.load(home)["dials"]["humour"] == 2                     # the second start freed humour: it moves


def test_a_veto_of_a_move_the_replayed_path_also_made_stays_linked_to_it(tmp_path):
    # catches (review r2): a replay that dropped every veto's link to its move — the owner's two vetoes of two
    # moves, and a third `character veto` undid the last of them ("no automatic move still stands" live)
    def mv(at, a, b):
        return {"at": at, "kind": C.KIND_MOVE, "dial": "humour", "from": a, "to": b, "why": "old"}
    def vt(at, a, b, of, of_to):
        return {"at": at, "kind": C.KIND_VETO, "dial": "humour", "from": a, "to": b, "of": of, "of_to": of_to}
    rows = [_START, mv("2026-11-04T02:00:00Z", 1, 0), mv("2026-11-07T02:00:00Z", 0, 1),
            vt("2026-11-07T12:00:00Z", 1, 0, "2026-11-07T02:00:00Z", 1),
            vt("2026-11-07T13:00:00Z", 0, 1, "2026-11-04T02:00:00Z", 0)]
    home = _old_home(tmp_path, rows, dict(C.START), through="2026-11-06")
    C.replay(home, _days(["serious"], ["serious"], ["serious"], ["playful"], ["playful"], ["playful"]))
    got = C.log_rows(home)
    assert [r["kind"] for r in got] == [C.KIND_START, C.KIND_MOVE, C.KIND_MOVE, C.KIND_VETO, C.KIND_VETO,
                                        C.KIND_REPLAY]
    assert got[3]["of"] == got[2]["at"] and got[4]["of"] == got[1]["at"]
    assert C.load(home)["dials"]["humour"] == 1
    assert "no automatic move still stands" in C.veto(home)[1]
    # replaying the replayed log changes nothing
    C._MEMO.clear()
    again, _end = C.replay_path(got[:-1], _days(["serious"], ["serious"], ["serious"], ["playful"], ["playful"],
                                                ["playful"]), "2026-11-06")
    assert again == got[:-1]
