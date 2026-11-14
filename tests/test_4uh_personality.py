"""§4UH: the character moves from the agent's own days; it has tastes; and it leans on how hard the events
of a day land. (Its daily look at its own replies, ``selfview``, is removed — §4UH-noself, 2026-11-14;
`test_4uh_noself.py` pins that.)"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import mood as M

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def _home(tmp_path) -> Path:
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "selfhood").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    C._MEMO.clear()


# ── tastes ──────────────────────────────────────────────────────────────────

from ghost_agent.selfhood import tastes as T  # noqa: E402


def test_a_turn_counts_once_for_each_kind_of_work_it_was():
    got = T.count_turn(["web_search", "browser", "execute", "introspect"], failed_calls=0, calls=4)
    assert got == {"coding": (1.0, 0.0), "research": (1.0, 0.0)}
    assert T.count_turn(["introspect"], 0, 1) == {}                         # not work: no kind
    assert T.count_turn(["web_search"], 1, 2) == {"research": (0.0, 1.0)}  # half the calls failed: bad
    assert T.count_turn(["web_search"], 1, 3) == {"research": (1.0, 0.0)}
    assert T.count_turn(["web_search"], 0, 1, delivered=False) == {"research": (0.0, 1.0)}
    assert T.count_turn(["web_search"], 0, 1, thumb=1) == {"research": (1.0 + T.THUMB_WEIGHT, 0.0)}
    assert T.count_turn(["web_search"], 0, 1, thumb=-1) == {"research": (1.0, T.THUMB_WEIGHT)}
    assert T.count_turn(["web_search"], "x", None) == {"research": (1.0, 0.0)}   # junk counts: no crash


def test_what_was_counted_fades_with_the_days_gone():
    s = T.fold({"kinds": {}, "through": ""}, "2026-11-01", [{"names": ["recall"], "failed": 0, "calls": 1}] * 4)
    assert s == {"kinds": {"memory": {"good": 4.0, "bad": 0.0}}, "through": "2026-11-01"}
    later = T.fold(s, "2026-12-01", [], days_since=int(T.HALF_LIFE_DAYS))
    assert later["kinds"]["memory"]["good"] == pytest.approx(2.0) and later["through"] == "2026-12-01"
    gone = T.fold(s, "2027-11-01", [], days_since=400)
    assert gone["kinds"] == {}                                              # faded below a trace: dropped


def _state(**kinds):
    return {"kinds": {k.replace("_", " "): {"good": g, "bad": b} for k, (g, b) in kinds.items()}, "through": "2026-11-11"}


def test_a_taste_stands_out_only_against_the_agents_own_average():
    s = _state(research=(60, 1), coding=(40, 4), pictures=(10, 6), memory=(25, 10))
    got = T.standing(s)
    assert got["liked"] == ["research"] and got["rough"] == ["pictures", "memory"]
    assert T.standing(_state(research=(60, 0), coding=(40, 0))) == {"liked": [], "rough": []}   # nothing went badly
    assert T.standing(_state(research=(5, 0), pictures=(2, 2))) == {"liked": [], "rough": []}   # too little
    few = T.standing(_state(research=(60, 1), coding=(40, 4), pictures=(3, 4)))
    assert "pictures" not in few["rough"]                                  # under MIN_TURNS: no taste
    # rough needs the floor too: when almost nothing goes badly, twice almost nothing is not rough
    assert T.standing(_state(research=(1000, 1), coding=(100, 0.3)))["rough"] == []


def test_a_reply_hears_only_what_is_liked_and_the_view_says_both(tmp_path):
    home = _home(tmp_path)
    assert T.liked_line(home) == "" and T.line(home) == ""
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(
        _state(research=(60, 1), the_system=(30, 0), pictures=(14, 4), memory=(30, 6))))
    heard = T.liked_line(home)
    assert "research and looking after the system" in heard and "never a reason to choose a tool" in heard
    assert "picture" not in heard and "memory" not in heard and "rough" not in heard
    assert T.line(home) == "Lately: I have come to enjoy research and looking after the system."   # the default
    assert T.line(home, liked_only=False) == ("Lately: I have come to enjoy research and looking after the system; "
                            "making and reading pictures and memory work have been rough.")
    text, stamp = C.section(home)
    assert heard in text and "rough" not in text and stamp == C.stamp(C.load(home)["dials"])
    # review r2: the view is a system REPLY, which goes back to the model as history — liked only
    assert "come to enjoy research" in C.view(home) and "rough" not in C.view(home)


def test_the_section_memo_is_keyed_on_the_tastes_file_too(tmp_path):
    home = _home(tmp_path)
    before, _ = C.section(home)
    assert "enjoy" not in before
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(research=(60, 1), coding=(40, 4),
                                                                                   memory=(30, 6))))
    after, _ = C.section(home)
    assert "Lately you have come to enjoy research" in after


@pytest.mark.parametrize("raw", ["{", "[]", '{"kinds": 3}', '{"kinds": {"research": {"good": "x"}}}',
                                 '{"kinds": {"research": {"good": -5, "bad": 1}}, "through": "nope"}'])
def test_a_bad_tastes_file_is_no_taste_and_never_breaks_the_character(tmp_path, raw):
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(raw)
    assert T.load(home)["kinds"] == {} and T.load(home)["through"] == ""
    assert T.liked_line(home) == "" and C.section(home)[0]


def _turn(at, names, *, failed=0, tid="t", calls=None):
    calls = calls if calls is not None else len(names)
    tc = [{"name": n, "error": "boom" if i < failed else ""} for i, n in enumerate(names)]
    return {"id": tid, "timestamp": at, "tool_calls": tc[:calls] if calls <= len(tc) else tc}


def test_closing_days_reads_owner_turns_and_thumbs_once_and_is_idempotent(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    good = [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a"),
            _turn("2026-11-10T13:00:00Z", ["web_search", "execute"], failed=2, tid="b"),
            _turn("2026-11-11T12:00:00Z", ["recall"], tid="c")]
    bad = [_turn("2026-11-11T13:00:00Z", ["image_generation"], tid="d")]
    reads = []
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: reads.append(window) or (good, bad))
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"trajectory_id": "a", "source": "human_feedback:web", "outcome": "passed"},
        {"trajectory_id": "c", "source": "human_feedback:slack:member", "outcome": "failed"},   # a member's thumb
        {"trajectory_id": "c", "source": "auto", "outcome": "failed"})) + "\nnot json\n")
    done = T.close_days(home, ["2026-11-10", "2026-11-11", "bogus"], tz=UTC)
    assert done == ["2026-11-10", "2026-11-11"] and len(reads) == 1
    k = T.load(home)["kinds"]
    fade = 0.5 ** (1 / T.HALF_LIFE_DAYS)
    assert k["research"]["good"] == pytest.approx((1 + T.THUMB_WEIGHT) * fade, abs=1e-3)
    assert k["research"]["bad"] == pytest.approx(1 * fade, abs=1e-3)       # b: both calls failed
    assert k["coding"]["bad"] == pytest.approx(fade, abs=1e-3)
    assert k["memory"] == {"good": 1.0, "bad": 0.0}                        # only the owner's thumbs count
    assert k["pictures"] == {"good": 0.0, "bad": 1.0}                      # never delivered
    assert T.load(home)["through"] == "2026-11-11"
    assert T.close_days(home, ["2026-11-10", "2026-11-11"], tz=UTC) == [] and len(reads) == 1


def test_closing_days_never_raises_and_writes_nothing_on_a_failure(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)

    def boom(sysd, window):
        raise RuntimeError("no trajectories")
    monkeypatch.setattr(FC, "_owner_turns", boom)
    assert T.close_days(home, ["2026-11-10"], tz=UTC) == []
    assert not (home / "system" / "selfhood" / "tastes.json").exists()
    assert T.close_days(None, ["2026-11-10"]) == []


def test_a_first_run_folds_at_most_the_backfill(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: ([], []))
    start = datetime.date(2026, 6, 1)
    dates = [(start + datetime.timedelta(days=i)).isoformat() for i in range(T.BACKFILL_DAYS + 20)]
    done = T.close_days(home, dates, tz=UTC)
    assert len(done) == T.BACKFILL_DAYS and done[-1] == dates[-1]


@pytest.mark.asyncio
async def test_the_idle_phase_counts_the_closed_days_into_tastes(tmp_path, monkeypatch):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    monkeypatch.setattr(D, "load_days", lambda h: [{"date": "2026-11-10"}, {"date": "2026-11-11"}])
    got = []
    monkeypatch.setattr(T, "close_days", lambda h, dates, **k: got.append(list(dates)) or list(dates))
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=None, args=None, self_model=None)
    ran, notes = [], {}
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, ran, notes, ctx)
    assert got == [["2026-11-10", "2026-11-11"]] and "tastes counted for 2 day(s)" in notes.get("character", "")


# ── the character leans on how hard things land ─────────────────────────────

_REAL_LEAN_DIALS = C.lean_dials


@pytest.fixture
def real_lean(monkeypatch):
    """conftest pins the lean's dials at 0 for every test; these tests read the stored dials."""
    monkeypatch.setattr(C, "lean_dials", _REAL_LEAN_DIALS)


def _ev(kind, dp=-0.2, de=0.1):
    return M.Event(dp, de, "why", kind)


# §4UH-lean: LEAN_STEP 0.25 (was 0.1) — the factors are written out, not derived, so a changed step fails here
@pytest.mark.parametrize("warmth,factor", [(-2, 0.5), (-1, 0.75), (0, 1.0), (1, 1.25), (2, 1.5)])
def test_warmth_scales_what_comes_from_the_owner_both_ways(warmth, factor):
    dials = {"warmth": warmth, "humour": 2, "directness": -2}
    for kind in (M.KIND_THUMB_UP, M.KIND_THUMB_DOWN, M.KIND_CORRECTION, M.KIND_PRAISE, M.KIND_SENTIMENT):
        for dp in (-0.2, 0.2):
            got = C.lean(_ev(kind, dp), dials)
            assert got.dp == pytest.approx(dp * factor) and got.de == pytest.approx(0.1 * factor)
            assert got.kind == kind and got.cause == "why"


def test_humour_lightens_or_weighs_only_the_agents_own_setbacks():
    playful, serious = {"warmth": 0, "humour": 2, "directness": 0}, {"warmth": 0, "humour": -2, "directness": 0}
    assert C.lean(_ev(M.KIND_TURN, -0.3), playful).dp == pytest.approx(-0.15)
    assert C.lean(_ev(M.KIND_TURN, -0.3), serious).dp == pytest.approx(-0.45)
    good = _ev(M.KIND_TURN, 0.3)
    assert C.lean(good, playful) is good and C.lean(good, serious) is good
    assert C.lean(_ev(M.KIND_THUMB_DOWN), {"warmth": 0, "humour": 2, "directness": 0}).dp == pytest.approx(-0.2)


def test_nothing_else_leans_and_the_bound_holds():
    wild = {"warmth": 9, "humour": -9, "directness": 9}
    for kind in (M.KIND_OWNER, M.KIND_QUIET, M.KIND_QUIET_END, M.KIND_DECAY, M.KIND_MANNER, M.KIND_THUMB_OFF):
        ev = _ev(kind)
        assert C.lean(ev, wild) is ev
    assert C.lean(_ev(M.KIND_PRAISE, 0.5), wild).dp == pytest.approx(0.5 * (1 + 2 * C.LEAN_STEP))
    assert C.lean(_ev(M.KIND_TURN, -0.5), wild).dp == pytest.approx(-0.5 * (1 + 2 * C.LEAN_STEP))
    assert C.lean(None, wild) is None


def _sm(home):
    return SelfModel(root=home / "system" / "selfhood")


def test_the_mood_books_the_leaned_size_on_arrival(tmp_path, real_lean):
    home = _home(tmp_path)
    C.set_dial(home, "warmth", 2, now=NOW)
    plain = _home(tmp_path / "plain")
    C.set_dial(plain, "warmth", 0, now=NOW)
    warm, flat = _sm(home), _sm(plain)
    a = warm.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1")
    b = flat.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1")
    assert a is not None and b is not None
    assert a.moved[0] == pytest.approx(b.moved[0] * 1.5, abs=1e-3)
    assert warm._applied["r1"][0] == pytest.approx(a.base_moved[0])          # the ledger holds what moved
    # the same thumb again changes nothing: it was booked at its leaned size
    assert warm.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="r1") is None


def test_a_bad_turn_lands_lighter_on_a_playful_character(tmp_path, real_lean):
    home, plain = _home(tmp_path / "p"), _home(tmp_path / "s")
    C.set_dial(home, "humour", 2, now=NOW)
    C.set_dial(plain, "humour", 0, now=NOW)
    sig = M.TurnSignals(tool_calls=6, failures=5)
    a = _sm(home).note_owner_turn(sig, now=NOW)
    b = _sm(plain).note_owner_turn(sig, now=NOW)
    assert a.moved[0] < 0 and a.moved[0] == pytest.approx(b.moved[0] * 0.5, abs=1e-3)


def test_no_lean_when_it_is_switched_off_or_the_character_is_off_or_unreadable(tmp_path, monkeypatch, real_lean):
    plain = _home(tmp_path / "f")
    C.set_dial(plain, "warmth", 0, now=NOW)
    base = _sm(plain).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0]
    seen = []
    for env in ({"GHOST_CHARACTER_LEAN": "0"}, {"GHOST_CHARACTER_LEAN": "1", "GHOST_CHARACTER": "0"}):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        h = _home(tmp_path / f"case{len(seen)}")
        C.set_dial(h, "warmth", 2, now=NOW)
        seen.append(_sm(h).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0])
        monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    monkeypatch.setenv("GHOST_CHARACTER_LEAN", "1")
    h = _home(tmp_path / "broken")
    (h / "system" / "selfhood" / C.CHARACTER_FILENAME).write_text("{not json")
    C._MEMO.clear()
    seen.append(_sm(h).note_owner_reaction(M.KIND_PRAISE, now=NOW, reply_id="x").moved[0])
    assert seen == [pytest.approx(base)] * 3


def test_day_records_read_in_date_order_once_each(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    f = home / "system" / "selfhood" / D.DAYS_FILENAME
    f.write_text("".join(json.dumps(r) + "\n" for r in (
        {"date": "2026-11-09", "n": 1}, {"date": "2026-10-01", "n": 2, "seeded": True},
        {"date": "2026-11-09", "n": 3}, {"date": "2026-10-02", "n": 4, "seeded": True})))
    assert [(r["date"], r["n"]) for r in D.load_days(home)] == [("2026-10-01", 2), ("2026-10-02", 4), ("2026-11-09", 1)]
    assert [r["date"] for r in D.load_days(home, limit=1)] == ["2026-11-09"]


# ── review r1 ───────────────────────────────────────────────────────────────

_TASTY = dict(research=(60, 1), the_system=(30, 0), pictures=(14, 4), memory=(30, 6))


def test_the_introspect_line_carries_liked_tastes_only(tmp_path):
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(**_TASTY)))
    said = C.line(home)
    assert "come to enjoy research" in said and "rough" not in said and "picture" not in said
    assert "rough" not in C.view(home)
    from ghost_agent.tools import introspect as I
    assert "rough" not in (I._self_image(str(home / "system" / "memory")) or "")


def test_the_handhelds_thumbs_count_for_tastes_and_a_scripts_do_not(tmp_path, monkeypatch):
    from ghost_agent.core import failure_census as FC
    home = _home(tmp_path)
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, window: (
        [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a"), _turn("2026-11-10T12:30:00Z", ["recall"], tid="b")], []))
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(
        json.dumps({"trajectory_id": "a", "source": "human_feedback:clockwork", "outcome": "passed"}) + "\n"
        + json.dumps({"trajectory_id": "b", "source": "human_feedback:api", "outcome": "failed"}) + "\n")
    T.close_days(home, ["2026-11-10"], tz=UTC)
    k = T.load(home)["kinds"]
    # §4UH-e2e review: "api" names no client (a script) — the mood refuses it, so tastes do too
    assert k["research"]["good"] == 1 + T.THUMB_WEIGHT and k["memory"]["bad"] == 0


@pytest.mark.parametrize("humour,raw,trouble", [(2, -0.12, True), (-2, -0.09, False)])
def test_trouble_is_judged_on_the_turns_own_size_and_only_the_booking_leans(tmp_path, monkeypatch, real_lean,
                                                                              humour, raw, trouble):
    home = _home(tmp_path)
    C.set_dial(home, "humour", humour, now=NOW)
    monkeypatch.setattr(M, "turn_events", lambda sig: [M.Event(raw, 0.0, "a call failed in this turn", M.KIND_TURN)])
    sm = _sm(home)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=2, failures=1), now=NOW)
    assert after.moved[0] == pytest.approx(raw * (1 - 0.25 * humour), abs=1e-3)  # leaned once, not twice
    assert (sm._trouble_at is not None) is trouble and (sm._own_trouble_at is not None) is trouble


def test_a_turn_that_moves_only_energy_is_not_leaned():
    ev = M.Event(0.0, -0.3, "a long turn", M.KIND_TURN)
    assert C.lean(ev, {"warmth": 2, "humour": 2, "directness": 2}) is ev


def _journal(home, rows):
    p = home / "system" / "selfhood" / "mood.events.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_days_thumbs_are_the_owners_last_thumb_per_turn_from_the_label_store(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    _journal(home, [{"at": "2026-11-10T12:01:00Z", "kind": M.KIND_THUMB_UP, "dp": 0.25, "de": 0.0, "cause": "x"}] * 3)
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    rows = [("a", "passed", "human_feedback:web", "2026-11-10T10:00:00Z"),
            ("a", "passed", "human_feedback:web", "2026-11-10T11:00:00Z"),      # sent again: one thumb
            ("b", "passed", "human_feedback:clockwork", "2026-11-10T10:00:00Z"),
            ("b", "failed", "human_feedback:clockwork", "2026-11-10T10:01:00Z"),  # changed: the last counts
            ("c", "failed", "human_feedback:slack:member", "2026-11-10T10:00:00Z"),  # a member's
            ("d", "failed", "user_correction", "2026-11-10T10:00:00Z"),          # not a thumb
            ("e", "passed", "human_feedback:api", "2026-11-09T10:00:00Z"),       # the day before
            ("f", "passed", "human_feedback:cli", "2026-11-10T10:00:00Z"),
            ("f", "failed", "human_feedback:cli", "2026-11-11T10:00:00Z")]       # changed the next day
    (tdir / "corrections.jsonl").write_text("".join(json.dumps(
        {"trajectory_id": t, "outcome": o, "source": src, "timestamp": ts}) + "\n" for t, o, src, ts in rows))
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC, now=NOW)
    assert (rec["thumbs_up"], rec["thumbs_down"]) == (1, 1)                # the journal's three rows are not thumbs
    assert D.owner_thumbs(home, datetime.date(2026, 11, 11), tz=UTC) == (0, 1)
    assert D.owner_thumbs(home, datetime.date(2026, 11, 9), tz=UTC) == (0, 0)       # "api" is not the owner's hand
    assert D.owner_thumbs(None, datetime.date(2026, 11, 11)) == (0, 0)

def _pulls_for(monkeypatch, prev, today):
    vals = list(prev) + [today]
    days = [{"date": (datetime.date(2026, 10, 1) + datetime.timedelta(days=i)).isoformat(), "v": v}
            for i, v in enumerate(vals)]
    monkeypatch.setattr(C, "day_signals", lambda rec: {"warmth": None, "humour": None, "directness": rec["v"]})
    return C.day_pulls(days).get(days[-1]["date"], {}).get("directness")


def test_a_pull_is_measured_in_the_agents_own_spread(monkeypatch):
    prev = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0]                    # mean 2, spread 2
    assert _pulls_for(monkeypatch, prev, 2.8) is None        # 0.4 deviations (0.8 unscaled would pull)
    assert _pulls_for(monkeypatch, prev, 3.2) == 1
    assert _pulls_for(monkeypatch, prev, 0.8) == -1


def test_the_band_is_strict_and_needs_enough_days(monkeypatch):
    prev = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]                    # mean 1, spread 1
    assert _pulls_for(monkeypatch, prev, 1.5) is None        # exactly 0.5 deviations: no pull
    assert _pulls_for(monkeypatch, prev, 0.5) is None
    assert _pulls_for(monkeypatch, prev, 1.5001) == 1
    assert _pulls_for(monkeypatch, prev[:C.BASE_MIN - 1], 9.0) is None
    assert _pulls_for(monkeypatch, [0.0, 2.0, 0.0, 2.0, 0.0][:C.BASE_MIN], 9.0) == 1


def test_liked_is_at_most_half_the_average_bad_share_exactly():
    # average bad share 10/200 = 0.05; research 1/40 = 0.025 is exactly half of it: liked
    s = {"kinds": {"research": {"good": 39.0, "bad": 1.0}, "coding": {"good": 151.0, "bad": 9.0}}, "through": "x"}
    assert T.standing(s)["liked"] == ["research"]
    s["kinds"]["research"] = {"good": 38.9, "bad": 1.1}
    s["kinds"]["coding"] = {"good": 151.1, "bad": 8.9}       # same average, research just over half
    assert T.standing(s)["liked"] == []


def test_rough_is_at_least_twice_the_average_exactly():
    # average 20/400 = 0.05, twice it 0.1 (over the 0.08 floor); pictures 10/100 = 0.1 exactly: rough
    s = {"kinds": {"pictures": {"good": 90.0, "bad": 10.0}, "coding": {"good": 290.0, "bad": 10.0}}, "through": "x"}
    assert T.standing(s)["rough"] == ["pictures"]
    s["kinds"]["pictures"] = {"good": 90.1, "bad": 9.9}
    s["kinds"]["coding"] = {"good": 289.9, "bad": 10.1}
    assert T.standing(s)["rough"] == []


def test_humour_needs_three_turns_of_work():
    assert C.day_signals({"own": {"clean_turns": 2, "trouble": 1}})["humour"] == pytest.approx(0.0)
    assert C.day_signals({"own": {"clean_turns": 2, "trouble": 0}})["humour"] is None


def test_no_taste_until_there_are_twice_min_turns_in_all():
    # research alone is enough turns for a taste, but all kinds together are under 2 × MIN_TURNS
    s = {"kinds": {"research": {"good": 8.0, "bad": 0.0}, "coding": {"good": 2.0, "bad": 3.0}}, "through": "x"}
    assert T.standing(s) == {"liked": [], "rough": []}
    s["kinds"]["coding"] = {"good": 5.0, "bad": 3.0}
    assert T.standing(s)["liked"] == ["research"]


@pytest.mark.asyncio
async def test_the_rough_tastes_are_said_in_the_log_when_days_are_counted(tmp_path, monkeypatch):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "tastes.json").write_text(json.dumps(_state(**_TASTY)))
    monkeypatch.setattr(D, "load_days", lambda h: [{"date": "2026-11-10"}])
    monkeypatch.setattr(T, "close_days", lambda h, dates, **k: ["2026-11-10"])
    logged = []
    monkeypatch.setattr(A, "pretty_log", lambda title, msg="", **k: logged.append((title, msg)))
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=None, args=None, self_model=None)
    notes = {}
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, [], notes, ctx)
    assert any(t == "Tastes" and "rough" in m for t, m in logged)
    assert "rough" not in notes.get("character", "")


def test_only_a_thumbs_outcome_counts(tmp_path):
    from ghost_agent.selfhood import days as D
    home = _home(tmp_path)
    tdir = home / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(json.dumps({"trajectory_id": "a", "outcome": "unknown",
                                                        "source": "human_feedback:web",
                                                        "timestamp": "2026-11-10T10:00:00Z"}) + "\n")
    assert D.owner_thumbs(home, datetime.date(2026, 11, 10), tz=UTC) == (0, 0)


# ── the `mirror` command shows the loop's path (§4UH-m, §4UH-loop) ─────────


def test_the_mirror_command_shows_no_diary_even_when_the_old_file_is_there(tmp_path):
    """§4UH-loop: the Mirror is removed; a leftover diary file is never read or shown."""
    from ghost_agent.selfhood import loop as MI
    p = tmp_path / "system" / "selfhood" / "mirror.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": "2026-11-09", "sentence": "I am bright.", "replies": [], "rest": [1, -1]})
                 + "\n", encoding="utf-8")
    b = MI.banner(tmp_path)
    assert "I am bright." not in b and "Until 2026-11-09" not in b
    assert "my mood rests" not in b                             # the starting dials rest even


def test_the_mirror_command_with_an_unreadable_character_says_so(tmp_path):
    from ghost_agent.selfhood import loop as MI
    p = tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    b = MI.banner(tmp_path)
    assert "cannot be read" in b and "STARTING character" in b


def test_a_broken_path_says_so_and_keeps_now(tmp_path, monkeypatch):
    from ghost_agent.selfhood import loop as MI
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(C, "log_rows", boom)
    b = MI.banner(tmp_path)
    assert b.startswith("**The strange loop**") and "path could not be read just now" in b and "diary" not in b
    assert "**Now:**" in b                                      # the character's part still renders


# ── §4UH-e2e: the dials set where the mood rests ─────────────────────────────────────────────────


@pytest.mark.parametrize("dials, units", [
    ({"warmth": -1, "humour": 1, "directness": 0}, (0, 0)),          # the start: even
    ({"warmth": 0, "humour": 1, "directness": 1}, (1, 1)),           # live 11-12: a little sunny and brisk
    ({"warmth": -1, "humour": 1, "directness": 2}, (0, 0)),          # blunt is not a mood: no "restless" rest
    ({"warmth": 2, "humour": -1, "directness": 0}, (1, -1)),         # warm + serious: a little sunny and calm
    ({"warmth": 2, "humour": 2, "directness": 2}, (2, 2)),
    ({"warmth": -2, "humour": -2, "directness": -2}, (0, 0)),        # dry + serious is not gloomy
    ({"warmth": 1, "humour": 0, "directness": 0}, (1, 0)),
    ({"warmth": 9, "humour": "x"}, (1, 0)),                          # brought into range
])
def test_the_dials_set_where_the_mood_rests(dials, units):
    assert C.rest_units(dials) == units


@pytest.mark.parametrize("w", range(-2, 3))
@pytest.mark.parametrize("h", range(-2, 3))
@pytest.mark.parametrize("d", range(-2, 3))
def test_a_rest_never_sits_in_an_unhappy_restless_or_bored_region(w, h, d):
    from ghost_agent.selfhood import mood as MM
    from ghost_agent.selfhood import loop as MI
    a, b = C.rest_units({"warmth": w, "humour": h, "directness": d})
    assert MM.rest_label((a * MI.REST_UNIT, b * MI.REST_UNIT)) in ("neutral", "content", "cheerful", "calm", "curious")


def test_the_rest_pair_follows_the_character_and_its_switches(tmp_path, monkeypatch):
    from ghost_agent.selfhood import loop as MI
    C.set_dial(tmp_path, "directness", 1)
    assert C.rest_pair(tmp_path) == (0.0, 0.0)                       # the start + blunt: even
    C.set_dial(tmp_path, "warmth", 2)
    C.set_dial(tmp_path, "humour", 2)
    assert C.rest_pair(tmp_path) == (2 * MI.REST_UNIT, 2 * MI.REST_UNIT)
    assert "With nothing happening, my mood rests at pleasantness sunny, energy lively." in MI.banner(tmp_path)
    monkeypatch.setenv(MI.REST_ENV, "0")
    assert C.rest_pair(tmp_path) == (0.0, 0.0) and "my mood rests" not in MI.banner(tmp_path)
    monkeypatch.delenv(MI.REST_ENV)
    monkeypatch.setenv("GHOST_CHARACTER", "0")                       # §4UH-loop: off is no rest (no Mirror)
    assert C.rest_pair(tmp_path) == (0.0, 0.0)
    monkeypatch.delenv("GHOST_CHARACTER")
    p = tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME
    p.write_text("{broken", encoding="utf-8")
    assert C.rest_pair(tmp_path) == (0.0, 0.0)
    assert C.rest_pair(None) == (0.0, 0.0)


def test_main_attaches_the_characters_rest_to_the_mood():
    """Parsed, not a text pin: the one `attach_rest(...)` call in main.py passes a lambda that calls
    `<character module>.rest_pair(...)`, where that name is imported from `.selfhood.character`."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/ghost_agent/main.py").read_text())
    char_names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                  and n.module == "selfhood" and n.level == 1 for a in n.names if a.name == "character"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "attach_rest"]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Lambda)
    body = calls[0].args[0].body
    assert isinstance(body, ast.Call) and isinstance(body.func, ast.Attribute) and body.func.attr == "rest_pair"
    assert isinstance(body.func.value, ast.Name) and body.func.value.id in char_names


def test_rest_units_round_toward_zero_both_ways_and_a_broken_file_rests_even(tmp_path, monkeypatch):
    assert C.rest_units({"warmth": 1, "humour": 0, "directness": -1}) == (1, 0)       # half rounds away from 0
    assert C.rest_units({"warmth": -2, "humour": -1, "directness": 0}) == (0, 0)      # never below even
    C.set_dial(tmp_path, "warmth", 2)
    C.set_dial(tmp_path, "humour", 2)
    assert C.rest_pair(tmp_path) != (0.0, 0.0)
    # warm starting dials, so a broken file falling back to them would NOT rest even (review r2: else equivalent)
    monkeypatch.setattr(C, "START", {"warmth": 2, "humour": 2, "directness": 2})
    (tmp_path / C.SELFHOOD_REL / C.CHARACTER_FILENAME).write_text("{broken", encoding="utf-8")
    assert C.unreadable(tmp_path) and C.rest_units(C.dials(tmp_path)) != (0, 0)
    assert C.rest_pair(tmp_path) == (0.0, 0.0)                     # not the starting dials' rest


def test_a_capitalised_owner_source_counts_as_the_mood_counts_it(tmp_path):
    from ghost_agent.selfhood import days as D
    tdir = tmp_path / "system" / "trajectories"
    tdir.mkdir(parents=True)
    (tdir / "corrections.jsonl").write_text(json.dumps({"trajectory_id": "a", "outcome": "passed",
                                                        "source": " Human_Feedback:Web ",
                                                        "timestamp": "2026-11-10T10:00:00Z"}) + "\n")
    assert D.owner_thumbs(tmp_path, datetime.date(2026, 11, 10), tz=UTC) == (1, 0)
    # … and tastes read the same row as the owner's thumb
    import ghost_agent.core.failure_census as FC
    from unittest import mock
    with mock.patch.object(FC, "_owner_turns", lambda sysd, window: (
            [_turn("2026-11-10T12:00:00Z", ["web_search"], tid="a")], [])):
        T.close_days(tmp_path, ["2026-11-10"], tz=UTC)
    assert T.load(tmp_path)["kinds"]["research"]["good"] == 1 + T.THUMB_WEIGHT
