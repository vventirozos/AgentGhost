"""§4UG: the character (three dials, one sentence that is a function of them), the day record it moves
from, and the mood's shade — and how each reaches a reply, the owner and the clients."""
from __future__ import annotations

import datetime
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import days as D
from ghost_agent.selfhood import loop as LOOP
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import shade as S
from ghost_agent.utils.logging import (
    prompt_arm_context, reply_surface_context, request_id_context, request_origin_context,
    requester_role_context)

UTC = datetime.timezone.utc


def _home(tmp_path) -> Path:
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "selfhood").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _ctx(home, sm=None):
    return SimpleNamespace(memory_dir=str(home / "system" / "memory"), turn_origin_label=None,
                           owner_memory_isolated=False, _replay_forbidden_tools=None, self_model=sm)


class _req:
    def __init__(self, rid="web-4ug", role="", surface="", origin="", arm=""):
        self.v = [(request_id_context, rid), (requester_role_context, role), (reply_surface_context, surface),
                  (request_origin_context, origin), (prompt_arm_context, arm)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


_REAL_DAY_PULLS = C.day_pulls


def _stated_pulls(days):
    """Day records written by `_days` state their pulls; anything else goes to the real rule."""
    if any(isinstance(d, dict) and "pulls" in d for d in days or []):
        return {str(d["date"]): dict(d["pulls"]) for d in C._by_date(days) if d.get("pulls")}
    return _REAL_DAY_PULLS(days)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("GHOST_CHARACTER", raising=False)
    # §4UH: the tests of the counting (leads, pins, vetoes, evidence) state each day's pulls and use one
    # lead for both directions; the signals behind the pulls and the larger lead away from 0 are tested
    # on their own, below
    monkeypatch.setattr(C, "day_pulls", _stated_pulls)
    monkeypatch.setattr(C, "LEAD_OUT", C.LEAD_IN)
    C._MEMO.clear()
    A._SELF_IMAGE_BY_REQ.clear()
    A._MOOD_SHADE_BY_REQ.clear()
    yield
    A._SELF_IMAGE_BY_REQ.clear()
    A._MOOD_SHADE_BY_REQ.clear()


# ── the dials and their sentence ────────────────────────────────────────────

def test_the_character_is_on_unless_switched_off(monkeypatch):
    assert C.enabled() is True
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    assert C.enabled() is False
    monkeypatch.setenv("GHOST_CHARACTER", "1")
    assert C.enabled() is True


def test_three_voice_dials_and_where_they_start():
    assert C.DIALS == ("warmth", "humour", "directness") and C.LIMIT == 2
    assert C.START == {"warmth": -1, "humour": 1, "directness": 0}
    assert C.sentence(C.START) == "I am a little dry and fond of a light joke."
    assert C.stamp(C.START) == "character:-1,1,0"


def test_the_sentence_is_a_function_of_the_dials_and_says_nothing_at_zero():
    assert C.sentence({"warmth": 0, "humour": 0, "directness": 0}) == ""
    assert C.section_text({"warmth": 0, "humour": 0, "directness": 0}) == ""
    assert C.sentence({"directness": 1}) == "I am straight about what I think."
    assert C.sentence({"warmth": 1, "humour": -1, "directness": 1}) == \
        "I am quietly warm, mostly serious and straight about what I think."
    # a part with its own comma: semicolons keep the three apart
    assert C.sentence({"warmth": 2, "humour": 2, "directness": 2}) == \
        "I am openly warm and fond; playful, joking wherever there is room; blunt, saying it as it is."
    seen = set()
    for combo in itertools.product(range(-2, 3), repeat=3):
        d = dict(zip(C.DIALS, combo))
        s = C.sentence(d)
        assert s == C.sentence(dict(d)) and (s == "") == (combo == (0, 0, 0))
        assert len(s) <= 300 and "\n" not in s
        seen.add(s)
    assert len(seen) == 125                                    # every setting reads differently


def test_no_dial_and_no_phrase_is_about_length_or_effort():
    """§4UF: "short and dry" cut tool calls by a third. The character is voice only."""
    words = " ".join(p for d in C.PHRASES.values() for p in d.values()).lower()
    for banned in ("short", "brief", "terse", "concise", "long", "thorough", "lazy", "careful", "quick"):
        assert banned not in words.split() and banned not in " ".join(C.DIALS)
    assert "never how much you do" in C.RULE and "the same checks and tool use" in C.RULE
    assert "Never name or announce these traits" in C.RULE


def test_values_are_clamped_and_names_resolved():
    assert [C.clamp(v) for v in (-9, -2, 0, 2, 9, "1", None, True, "x", 1.9)] == [-2, -2, 0, 2, 2, 1, 0, 0, 0, 1]
    assert C.resolve_dial("Humor") == "humour" and C.resolve_dial(" Warmth ") == "warmth"
    # a pole's word is not a dial: "dry up" would have set warmth UP (review r1)
    for pole in ("warm", "dry", "playful", "serious", "blunt", "diplomatic", "tactful", "direct", "speed"):
        assert C.resolve_dial(pole) == ""
    assert C.clamp(float("inf")) == 0 and C.clamp(1e999) == 0


# ── the store and the owner's word ──────────────────────────────────────────

def test_reading_never_writes_and_the_first_write_logs_the_start(tmp_path):
    home = _home(tmp_path)
    assert C.load(home)["dials"] == C.START and C.load(home)["stored"] is False
    assert C.section(home) == (C.section_text(C.START), "character:-1,1,0")
    assert not (home / "system/selfhood" / C.CHARACTER_FILENAME).exists()
    state, said = C.set_dial(home, "humor", 2)
    assert said == "humour +1 → +2" and state["dials"]["humour"] == 2
    kinds = [r["kind"] for r in C.log_rows(home)]
    assert kinds == [C.KIND_START, C.KIND_OWNER]
    assert C.load(home)["dials"] == {"warmth": -1, "humour": 2, "directness": 0}
    assert C.section(home)[1] == "character:-1,2,0"            # the memo follows the file
    assert C.set_dial(home, "humour", 2)[1] == "humour is already at +2"
    assert C.set_dial(home, "speed", 1)[1].startswith("no such dial")
    assert C.set_dial(home, "warmth", 99)[0]["dials"]["warmth"] == 2


def test_an_unreadable_file_is_the_starting_character(tmp_path):
    home = _home(tmp_path)
    (home / "system/selfhood" / C.CHARACTER_FILENAME).write_text("{not json", encoding="utf-8")
    assert C.load(home)["dials"] == C.START
    (home / "system/selfhood" / C.CHARACTER_FILENAME).write_text('{"dials": {"warmth": 7, "x": 1}}', encoding="utf-8")
    assert C.load(home)["dials"] == {"warmth": 2, "humour": 1, "directness": 0}
    assert C.load(None)["dials"] == C.START and C.section(None) == ("", "") and C.dials(None) == C.START


def test_pins_hold_a_dial_and_say_so(tmp_path):
    home = _home(tmp_path)
    assert C.set_pin(home, "warmth", True)[1] == "warmth is pinned at -1"
    assert C.load(home)["pins"] == ["warmth"]
    assert "already" in C.set_pin(home, "warmth", True)[1]
    assert C.set_pin(home, "warmth", False)[1].startswith("warmth is free to move again — the count toward its next move")
    assert C.load(home)["pins"] == []


# ── the daily rule ──────────────────────────────────────────────────────────

#: a short way to write a day's pulls in these tests (the names are the manners the first rule counted)
_PULL = {"playful": {"humour": 1}, "sarcastic": {"humour": 1, "warmth": -1}, "formal": {"humour": -1, "directness": -1},
         "terse": {"directness": 1, "warmth": -1}, "gentle": {"warmth": 1, "directness": -1}}


def _days(*names_by_day, start=datetime.date(2026, 11, 1)):
    out = []
    for i, names in enumerate(names_by_day):
        pulls = {}
        for dial in C.DIALS:
            got = {_PULL[n][dial] for n in names if dial in _PULL[n]}
            if len(got) == 1:
                pulls[dial] = got.pop()
        out.append({"date": (start + datetime.timedelta(days=i)).isoformat(), "pulls": pulls})
    return out


def test_a_dial_moves_one_step_when_the_days_lead_by_three(tmp_path):
    home = _home(tmp_path)
    days = _days(["playful"], ["sarcastic"], [], ["playful"])
    moves = C.propose(days, C.load(home), [], "2026-11-04")
    assert [(m["dial"], m["from"], m["to"]) for m in moves] == [("humour", 1, 2)]
    assert moves[0]["why"].startswith("3 days clearly better than its usual against 0 the other way, in how its own work went")
    # two days are not enough; a day pulling the other way cancels one
    assert C.propose(days[:3], C.load(home), [], "2026-11-03") == []
    assert C.propose(_days(["playful"], ["playful"], ["playful"], ["formal"]), C.load(home), [], "2026-11-04") == []
    # a day with manners pulling a dial both ways counts for neither
    assert C.propose(_days(["playful", "formal"], ["playful"], ["playful"]), C.load(home), [], "2026-11-03") == []


def test_a_day_pulls_the_dials_it_states():
    d = _days(["terse"], ["terse"], ["terse"])
    got = {m["dial"]: (m["from"], m["to"]) for m in C.propose(d, {"dials": dict(C.START), "pins": []}, [], "2026-11-03")}
    assert got == {"warmth": (-1, -2), "directness": (0, 1)}


def test_a_pin_a_limit_and_the_last_change_stop_a_move(tmp_path):
    home = _home(tmp_path)
    days = _days(["playful"], ["playful"], ["playful"])
    assert C.propose(days, {"dials": dict(C.START), "pins": ["humour"]}, [], "2026-11-03") == []
    assert C.propose(days, {"dials": {"warmth": -1, "humour": 2, "directness": 0}, "pins": []}, [], "2026-11-03") == []
    # only days AFTER the dial's last change count: the same evidence never moves it twice
    rows = [{"at": "2026-11-02T10:00:00Z", "kind": C.KIND_OWNER, "dial": "humour", "from": 0, "to": 1}]
    assert C.propose(days, {"dials": dict(C.START), "pins": []}, rows, "2026-11-03") == []
    # … and only the last EVIDENCE_DAYS recorded days
    old = _days(["playful"], ["playful"], ["playful"], *[[] for _ in range(C.EVIDENCE_DAYS)])
    assert C.propose(old, {"dials": dict(C.START), "pins": []}, [], old[-1]["date"]) == []


def test_the_daily_rule_runs_once_per_recorded_day_and_logs_what_it_counted(tmp_path):
    home = _home(tmp_path)
    days = _days(["playful"], ["playful"], ["playful"])
    said = C.daily(home, days)
    assert said and said[0].startswith("humour +1 → +2: 3 days clearly better than its usual against 0")
    assert C.load(home)["dials"]["humour"] == 2 and C.load(home)["daily_through"] == "2026-11-03"
    assert C.daily(home, days) == []                           # the same days again: nothing
    row = [r for r in C.log_rows(home) if r["kind"] == C.KIND_MOVE][0]
    assert row["days_up"] == ["2026-11-01", "2026-11-02", "2026-11-03"] and row["days_down"] == []
    assert C.daily(home, []) == [] and C.daily(None, days) == []


def test_a_veto_takes_the_last_automatic_move_back_once(tmp_path):
    home = _home(tmp_path)
    assert C.veto(home)[1] == "there is no automatic move to take back"
    C.daily(home, _days(["playful"], ["playful"], ["playful"]))
    state, said = C.veto(home)
    assert said.startswith("humour +2 → +1") and state["dials"]["humour"] == 1
    assert "no automatic move still stands" in C.veto(home)[1]
    # the evidence that made the move is behind the veto: it does not move again from it
    assert C.daily(home, _days(["playful"], ["playful"], ["playful"], [])) == []
    # a dial the owner set since is not vetoed either
    home2 = _home(tmp_path / "b")
    C.daily(home2, _days(["playful"], ["playful"], ["playful"]))
    C.set_dial(home2, "humour", 0)
    assert "no automatic move still stands" in C.veto(home2)[1] and C.load(home2)["dials"]["humour"] == 0


def test_the_story_and_the_view_are_the_log_in_words(tmp_path):
    home = _home(tmp_path)
    assert "It has not changed yet" in C.view(home)
    C.set_dial(home, "directness", 1, now=datetime.datetime(2026, 11, 12, 9, tzinfo=UTC))
    C.set_pin(home, "warmth", True, now=datetime.datetime(2026, 12, 1, 9, tzinfo=UTC))
    v = C.view(home)
    assert "Character: «I am a little dry, fond of a light joke and straight about what I think.»" in v
    assert "directness +1 (straight about what I think)" in v and "Pinned" in v and "warmth" in v
    assert "2026-11:\n" in v and "2026-12:\n" in v and "directness +0 → +1, set by the owner" in v
    assert "character: «I am a little dry" in C.line(home) and "not a report of work" in C.line(home)


# ── the owner's command ─────────────────────────────────────────────────────

def test_the_command_shows_sets_pins_and_vetoes(tmp_path):
    home = _home(tmp_path)
    assert C.is_command("character") and C.is_command(" /Character humour up. ") and not C.is_command("characters")
    assert not C.is_command("what is your character like?") and C.command("tell me a story", home) is None
    assert C.command("character", home).startswith("Character: «I am a little dry")
    assert "Commands:" in C.command("character", home)
    assert C.command("character humour up", home).startswith("✓ humour +1 → +2")
    assert C.command("character humour up", home) == "humour is already at its end (+2). Nothing changed."
    assert C.command("character set warmth to +1", home).startswith("✓ warmth -1 → +1")
    assert C.command("character directness −1", home).startswith("✓ directness +0 → -1")
    assert C.command("character pin humor", home).startswith("✓ humour is pinned at +2")
    assert C.command("character pin humour", home) == "Humour is already pinned. Nothing changed."
    assert C.command("character unpin humour", home).startswith("✓ humour is free to move again")
    assert C.command("character veto", home) == "There is no automatic move to take back."
    assert C.command("character help", home).startswith("Character: «") and C.parse_command("character undo") == ("veto",)
    # the WHOLE message must be one of the command's shapes: anything else is the owner talking to the model
    for talk in ("character development in this novel is weak", "character count of this file?",
                 "character be nicer please", "character foo 3", "characters", "character warmth 5",
                 "character warmth is what I need", "Character set design tips?", "character set in UTF-8?",
                 "character pin the tail", "character pin", "character serious flaws", "character dry up",
                 "character humour in Dickens novels", "character help me write", "character humour up now please",
                 "warmth is what I want from you, every single day of the week"):
        assert not C.is_command(talk) and C.command(talk, home) is None, talk
    assert C.load(home)["dials"] == {"warmth": 1, "humour": 2, "directness": -1}


# ── the day record ──────────────────────────────────────────────────────────

def _journal(home, name, rows):
    p = home / "system/selfhood" / name
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_a_day_is_counted_by_kind_from_the_moods_journals(tmp_path):
    home = _home(tmp_path)
    tz = UTC
    day = datetime.date(2026, 11, 10)
    cause_bad = M.merge_events(M.turn_events(M.TurnSignals(tool_calls=3, failures=3))).cause
    cause_good = M.merge_events(M.turn_events(M.TurnSignals(tool_calls=6, tasks_done=1, sources=4))).cause
    _journal(home, "mood.events.jsonl", [
        {"at": "2026-11-10T03:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "cause": "x"},       # the day before (04:00)
        {"at": "2026-11-10T09:00:00Z", "kind": "turn", "dp": -0.36, "de": 0.24, "cause": cause_bad},
        {"at": "2026-11-10T09:20:00Z", "kind": "turn", "dp": 0.46, "de": 0.3, "cause": cause_good},
        {"at": "2026-11-10T10:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "cause": "the owner gave a thumbs-up"},
        {"at": "2026-11-10T10:05:00Z", "kind": "thumb_down", "dp": -0.3, "de": 0.0, "cause": "c", "held": "held"},
        {"at": "2026-11-10T11:00:00Z", "kind": "correction", "dp": -0.2, "de": 0.0, "cause": "c"},
        {"at": "2026-11-10T12:00:00Z", "kind": "sentiment", "dp": -0.08, "de": -0.02, "cause": "c"},
        {"at": "2026-11-10T13:00:00Z", "kind": "owner", "dp": 1.5, "de": 0.2, "cause": "the owner said so (set_mood)",
         "after": "cheerful, 2 of 3"},
        {"at": "2026-11-10T14:00:00Z", "kind": "manner", "dp": 0.0, "de": 0.0,
         "cause": "the owner set the manner: sarcastic, 2 of 3"},
        {"at": "2026-11-10T15:00:00Z", "kind": "manner", "dp": 0.0, "de": 0.0,
         "cause": "the owner cleared the manner (sarcastic)"},
        {"at": "2026-11-11T02:00:00Z", "kind": "praise", "dp": 0.1, "de": 0.0, "cause": "the owner said thanks"},  # still the 10th
        {"at": "2026-11-11T05:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "cause": "x"},       # the 11th
    ])
    _journal(home, "mood.history.jsonl", [
        {"at": "2026-11-09T20:00:00Z", "label": "content", "intensity": 1, "kind": "thumb_up"},
        {"at": "2026-11-10T09:00:00Z", "label": "irritable", "intensity": 2, "kind": "turn"},
        {"at": "2026-11-10T09:20:00Z", "label": "neutral", "intensity": 0, "kind": "turn"},
        {"at": "2026-11-10T13:00:00Z", "label": "cheerful", "intensity": 2, "kind": "owner"},
        {"at": "2026-11-10T14:00:00Z", "label": "cheerful", "intensity": 2, "kind": "manner"},
        {"at": "2026-11-10T19:00:00Z", "label": "bored", "intensity": 3, "kind": "quiet"},
    ])
    _journal(home, "shades.jsonl", [{"shade": "relieved", "at": "2026-11-10T09:20:00Z", "cause": "x"},
                                    {"shade": "proud", "at": "2026-11-12T09:20:00Z", "cause": "x"}])
    rec = D.build_day(home, day, tz=tz)
    assert rec["date"] == "2026-11-10" and rec["events"] == 9 and rec["held"] == 1
    # §4UH review r2: thumbs come from the label store, not the journal (none here)
    assert (rec["thumbs_up"], rec["thumbs_down"], rec["corrections"], rec["praise"]) == (0, 0, 1, 1)
    assert (rec["tone_up"], rec["tone_down"]) == (0, 1)
    assert rec["owner_moods"] == ["cheerful"] and rec["manners"] == ["sarcastic"]
    assert rec["own"] == {"trouble": 1, "clean_turns": 1, "research": 1, "tasks_done": 1, "long_turns": 0}
    assert rec["by_kind"]["turn"] == {"n": 2, "dp": 0.1, "de": 0.54}
    # the arc: content 04:00–09:00 (5 h), irritable 20 min, cheerful 13:00–19:00 (6 h), bored 19:00–04:00 (9 h)
    assert rec["arc"] == {"content": 300, "irritable": 20, "cheerful": 360, "bored": 540}
    assert rec["peak"]["label"] == "irritable"                 # boredom is never the peak; the first 2 stands
    assert [b["kind"] for b in rec["biggest"]] == ["turn", "turn", "thumb_up"]   # never the owner's own word
    assert rec["shades"] == {"relieved": 1}
    line = D.day_line(rec)
    assert "bored 9 h" in line and "my turns: 1 ran into trouble, 1 went fine, 1 task(s) done, 1 research" in line
    assert "(shades: relieved)" in line and line.startswith(rec["date"] + " · moods: ")
    assert "from you: 1 correction(s), 1 thank-you(s), set my mood to cheerful, asked for a sarcastic manner" in line


def test_the_parsers_read_what_the_mood_really_writes(tmp_path):
    """The day record reads two sentences other modules write — pinned against the writers themselves."""
    from ghost_agent.selfhood.state import SelfStateThread
    st = SelfStateThread(tmp_path / "self")
    st.set_manner("playful", 2)
    rows = [json.loads(x) for x in (tmp_path / "self" / "mood.events.jsonl").read_text().splitlines()]
    assert any(D._MANNER_SET_RE.match(r["cause"]) and r["kind"] == M.KIND_MANNER for r in rows)
    for sig, want in ((M.TurnSignals(tool_calls=4, failures=4), "trouble"),
                      (M.TurnSignals(tool_calls=2, guards=1), "trouble"),
                      (M.TurnSignals(tool_calls=5, sources=3), "research"),
                      (M.TurnSignals(tool_calls=5, tasks_done=2), "tasks_done"),
                      (M.TurnSignals(tool_calls=30, duration_s=1000), "long_turns"),
                      (M.TurnSignals(tool_calls=5), "clean_turns")):
        ev = M.merge_events(M.turn_events(sig))
        assert D._own_counts(ev.cause).get(want) == 1, (sig, ev.cause)


def test_days_are_closed_once_and_only_when_over(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    assert D.close_days(home, now=now, tz=UTC) == []            # no journal: nothing can be told
    _journal(home, "mood.events.jsonl", [
        {"at": "2026-11-10T09:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "cause": "x"},
        {"at": "2026-11-12T09:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "cause": "x"}])
    assert D.close_days(home, now=now, tz=UTC) == ["2026-11-10", "2026-11-11"]     # not today, not before the journal
    assert D.close_days(home, now=now, tz=UTC) == []
    assert [d["date"] for d in D.load_days(home)] == ["2026-11-10", "2026-11-11"]
    assert D.load_days(home, limit=1)[0]["date"] == "2026-11-11" and D.load_days(None) == []
    # 03:00 on the 13th is still the 12th: nothing new to close
    assert D.close_days(home, now=datetime.datetime(2026, 11, 13, 3, 0, tzinfo=UTC), tz=UTC) == []
    assert D.close_days(home, now=datetime.datetime(2026, 11, 13, 5, 0, tzinfo=UTC), tz=UTC) == ["2026-11-12"]


# ── the shade ───────────────────────────────────────────────────────────────

def test_which_shade_an_event_sets():
    assert S.for_turn(M.TurnSignals(tasks_done=1), moved_dp=0.2, own_trouble_age_s=None) == S.PROUD
    assert S.for_turn(M.TurnSignals(tool_calls=5), moved_dp=0.1, own_trouble_age_s=600) == S.RELIEVED
    assert S.for_turn(M.TurnSignals(tool_calls=5), moved_dp=0.1, own_trouble_age_s=S.RELIEF_WINDOW_S + 1) == ""
    assert S.for_turn(M.TurnSignals(tool_calls=5), moved_dp=0.1, own_trouble_age_s=None) == ""
    assert S.for_turn(M.TurnSignals(tool_calls=5, failures=5), moved_dp=-0.4, own_trouble_age_s=60) == ""
    assert S.for_reaction(M.KIND_THUMB_DOWN) == S.for_reaction(M.KIND_CORRECTION) == S.CHASTENED
    assert S.for_reaction(M.KIND_THUMB_UP) == S.for_reaction(M.KIND_PRAISE) == ""
    for name in S.SHADES:
        assert "tool use" in S.line(name) and "do not announce it" in S.line(name) and S.line(name).endswith("\n\n")
    assert S.line("grumpy") == "" and S.line("") == ""


def test_a_shade_lasts_a_while_is_taken_once_and_survives_a_restart(tmp_path):
    root = tmp_path / "self"
    t0 = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    st = S.ShadeStore(root)
    assert st.current(now=t0) is None and st.take(now=t0) == "" and st.set("grumpy") is False
    assert st.set(S.RELIEVED, "a turn went through", now=t0)
    assert st.current(now=t0 + datetime.timedelta(minutes=29))["shade"] == S.RELIEVED
    assert st.current(now=t0 + datetime.timedelta(seconds=S.TTL_S + 1)) is None
    again = S.ShadeStore(root)                                  # a restart
    assert again.take(now=t0 + datetime.timedelta(minutes=1)) == S.RELIEVED
    assert again.take(now=t0 + datetime.timedelta(minutes=2)) == ""          # one reply only
    assert S.ShadeStore(root).take(now=t0 + datetime.timedelta(minutes=3)) == ""   # … across a restart too
    assert again.set(S.PROUD, now=t0 + datetime.timedelta(minutes=4))        # a later shade replaces it
    assert again.take(now=t0 + datetime.timedelta(minutes=5)) == S.PROUD
    rows = [json.loads(x) for x in (root / S.SHADES_LOG).read_text().splitlines()]
    assert [r["shade"] for r in rows] == [S.RELIEVED, S.PROUD]
    off = S.ShadeStore(None)
    assert off.set(S.PROUD) is True and off.take() == S.PROUD   # no home: in memory only, never an error


def test_the_self_model_sets_the_shade_from_what_moved_the_mood(tmp_path):
    sm = SelfModel(root=tmp_path / "self")
    t0 = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    sm.note_owner_turn(M.TurnSignals(tool_calls=3, failures=3), now=t0)
    assert sm.shade.current(now=t0) is None                     # trouble alone is no shade
    sm.note_owner_turn(M.TurnSignals(tool_calls=5), now=t0 + datetime.timedelta(minutes=5))
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=5))["shade"] == S.RELIEVED
    sm.note_owner_turn(M.TurnSignals(tool_calls=5), now=t0 + datetime.timedelta(minutes=9))
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=9))["at"].startswith("2026-11-12T10:05:00")   # relief once
    sm.note_owner_turn(M.TurnSignals(tool_calls=5, tasks_done=1), now=t0 + datetime.timedelta(minutes=12))
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=12))["shade"] == S.PROUD
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=t0 + datetime.timedelta(minutes=15), reply_id="r1")
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=15))["shade"] == S.CHASTENED
    sm.note_owner_reaction(M.KIND_THUMB_UP, now=t0 + datetime.timedelta(minutes=16), reply_id="r2")
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=16))["shade"] == S.CHASTENED   # a thumb up sets none


def test_an_event_the_owners_set_mood_holds_sets_no_shade(tmp_path):
    sm = SelfModel(root=tmp_path / "self")
    t0 = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    sm.state.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, now=t0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=t0 + datetime.timedelta(minutes=1), reply_id="r1")
    sm.note_owner_turn(M.TurnSignals(tool_calls=5, tasks_done=1), now=t0 + datetime.timedelta(minutes=2))
    assert sm.shade.current(now=t0 + datetime.timedelta(minutes=2)) is None


# ── what a reply hears ──────────────────────────────────────────────────────

@pytest.mark.parametrize("rid", ["web-4ug-1", "slack-4ug-1", "probe-4ug-1"])
def test_an_owner_reply_hears_the_character(tmp_path, rid):
    home = _home(tmp_path)
    with _req(rid):
        got = A._self_image_block(_ctx(home))
        assert got == f"{C.HEADER}\n«I am a little dry and fond of a light joke.»\n{C.RULE}\n\n"
        assert A._self_image_block(_ctx(home)) == got
    assert A.self_image_heard(rid) == "character:-1,1,0"        # the record says which dials the reply heard


def test_the_switch_off_is_no_self_image_at_all(tmp_path, monkeypatch):
    """§4UH-loop: with GHOST_CHARACTER=0 nothing is heard — the Mirror's sentence it once put back is removed;
    a leftover diary file is not read."""
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "mirror.jsonl").write_text(json.dumps({
        "at": "2026-11-09T00:30:00", "date": "2026-11-09", "sentence": "I am the Mirror's old sentence.",
        "replies": [{"at": "2026-11-09T09:00:00Z", "excerpt": "x"}]}) + "\n")
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    with _req("web-4ug-2"):
        assert A._self_image_block(_ctx(home)) == ""
    assert A.self_image_heard("web-4ug-2") == ""
    assert C.rest_pair(home) == (0.0, 0.0)


@pytest.mark.parametrize("who", [{"rid": "slack-4ug-m", "role": "member"}, {"rid": "slack-4ug-p", "surface": "public"},
                                 {"rid": "sched-4ug"}, {"rid": "job-4"}, {"rid": "sub-4"},
                                 {"rid": "probe-4ug-off", "arm": "no_self_image"}])
def test_nobody_else_hears_the_character(tmp_path, who):
    home = _home(tmp_path)
    with _req(**who):
        assert A._self_image_block(_ctx(home)) == ""
    assert A.self_image_heard(who["rid"]) == ""


def test_a_dial_the_owner_moves_is_heard_by_the_next_request_not_mid_request(tmp_path):
    home = _home(tmp_path)
    with _req("web-4ug-3"):
        first = A._self_image_block(_ctx(home))
        C.set_dial(home, "directness", 2)
        assert A._self_image_block(_ctx(home)) == first          # frozen for the request
    with _req("web-4ug-4"):
        assert "blunt, saying it as it is" in A._self_image_block(_ctx(home))
    assert A.self_image_heard("web-4ug-4") == "character:-1,1,2"


def test_every_dial_at_zero_is_zero_bytes(tmp_path):
    home = _home(tmp_path)
    for d in C.DIALS:
        C.set_dial(home, d, 0)
    with _req("web-4ug-5"):
        assert A._self_image_block(_ctx(home)) == ""
    assert A.self_image_heard("web-4ug-5") == ""


def test_the_shade_is_heard_by_one_owner_reply_and_recorded(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    sm.shade.set(S.RELIEVED, "x")
    for who in ({"rid": "probe-4ug-s"}, {"rid": "slack-4ug-sm", "role": "member"},
                {"rid": "slack-4ug-sp", "surface": "public"}, {"rid": "sched-4ug-s"}, {"rid": "job-9"}):
        with _req(**who):
            assert A._mood_shade_block(ctx) == ""               # … and none of them uses the moment up
        assert A.mood_shade_heard(who["rid"]) == ""
    with _req("web-4ug-s1"):
        got = A._mood_shade_block(ctx)
        assert got == S.line(S.RELIEVED) and got.startswith("RIGHT NOW, FOR THIS REPLY ONLY: relief")
        assert A._mood_shade_block(ctx) == got                  # every step of the request: the same bytes
    assert A.mood_shade_heard("web-4ug-s1") == S.RELIEVED
    with _req("web-4ug-s2"):
        assert A._mood_shade_block(ctx) == ""                   # the next reply: the moment has passed
    assert A.mood_shade_heard("web-4ug-s2") == ""
    with _req(""):
        assert A._mood_shade_block(ctx) == ""
    with _req("web-4ug-s3"):
        assert A._mood_shade_block(_ctx(home, None)) == ""      # no self-model: nothing, never an error


# ── the owner's command, the views, the clients ─────────────────────────────

def test_introspect_shows_the_characters_line_to_the_owner_only(tmp_path, monkeypatch):
    from ghost_agent.tools import introspect as I
    home = _home(tmp_path)
    ctx = _ctx(home)
    assert I._self_image(ctx).startswith("character: «I am a little dry and fond of a light joke.»")
    with _req("slack-4ug-pub", surface="public"):
        assert I._self_image(ctx) == ""
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    assert I._self_image(ctx) == ""                             # the character off: nothing (no Mirror)


def test_health_carries_the_dials_the_numbers_and_the_shade(tmp_path, monkeypatch):
    from ghost_agent.api import routes as R
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    assert R._health_character(ctx) == {"warmth": -1, "humour": 1, "directness": 0}
    C.set_dial(home, "warmth", 2)
    assert R._health_character(ctx)["warmth"] == 2
    assert R._health_character(SimpleNamespace(memory_dir=None)) == {}
    mood = R._health_mood(ctx)
    assert (mood["p"], mood["e"], mood["shade"], mood["shade_at"]) == (0.0, 0.0, "", "")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1")
    mood = R._health_mood(ctx)
    assert mood["p"] < 0 and -1.0 <= mood["e"] <= 1.0 and mood["shade"] == S.CHASTENED and mood["shade_at"].endswith("Z")
    assert set(R._MOOD_NEUTRAL) >= {"p", "e", "shade", "shade_at"}
    assert R._health_mood(SimpleNamespace())["shade"] == ""     # no self-model: the neutral block
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    assert R._health_character(ctx) == {}


@pytest.mark.asyncio
async def test_the_idle_phase_closes_the_days_and_moves_a_dial_without_a_model(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setattr(C, "day_pulls", lambda days: {str(d["date"]): {"humour": 1} for d in C._by_date(days)})
    today = M.mood_day()
    rows = []
    for back in (3, 2, 1):
        day = today - datetime.timedelta(days=back)
        at = datetime.datetime.combine(day, datetime.time(12, 0)).astimezone().astimezone(UTC)
        rows.append({"at": at.isoformat().replace("+00:00", "Z"), "kind": "manner", "dp": 0.0, "de": 0.0,
                     "cause": "the owner set the manner: playful, 2 of 3"})
    _journal(home, "mood.events.jsonl", rows)
    agent = A.GhostAgent.__new__(A.GhostAgent)
    ran, notes = [], {}
    await agent._character_phase(home, ran, notes)
    assert ran == ["character"] and "day record written for" in notes["character"]
    assert "humour +1 → +2" in notes["character"]
    assert C.load(home)["dials"]["humour"] == 2 and len(D.load_days(home)) == 3
    ran2, notes2 = [], {}
    await agent._character_phase(home, ran2, notes2)            # the same idle window again: nothing to do
    assert ran2 == [] and notes2 == {}


@pytest.mark.parametrize("kind,public,answered", [("owner", False, True), ("owner", True, False),
                                                  ("member", False, False), ("probe", False, False),
                                                  ("background", False, False)])
def test_the_owners_command_is_answered_by_the_system_on_a_private_surface_only(tmp_path, monkeypatch, kind,
                                                                               public, answered):
    import asyncio
    from ghost_agent.utils import logging as L
    home = _home(tmp_path)
    monkeypatch.setattr(L, "request_kind", lambda: kind)
    monkeypatch.setattr(A, "reply_is_public", lambda: public)
    fake = SimpleNamespace(context=SimpleNamespace(memory_dir=str(home / "system" / "memory")))
    body = {"messages": [{"role": "user", "content": "character humour up"}]}
    note = asyncio.run(A.GhostAgent._owner_rule_note(fake, body))
    if answered:
        assert "do not call any tool" in str(note) and note.banner.startswith("✓ humour +1 → +2")
        assert C.load(home)["dials"]["humour"] == 2
    else:
        assert not note and C.load(home)["dials"]["humour"] == 1       # nobody else can set a dial


def test_with_the_character_off_the_command_is_not_one(tmp_path, monkeypatch):
    import asyncio
    from ghost_agent.utils import logging as L
    home = _home(tmp_path)
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    monkeypatch.setattr(L, "request_kind", lambda: "owner")
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    fake = SimpleNamespace(context=SimpleNamespace(memory_dir=str(home / "system" / "memory")))
    note = asyncio.run(A.GhostAgent._owner_rule_note(fake, {"messages": [{"role": "user", "content": "character humour up"}]}))
    assert not getattr(note, "banner", "") and C.load(home)["dials"]["humour"] == 1


def test_the_mirror_banner_says_what_a_reply_hears_now(tmp_path, monkeypatch):
    # §4UH-loop: `mirror` is the strange loop's path, with the sentence replies hear now (§4UH-noself: and no
    # daily look)
    b = LOOP.banner(tmp_path)
    assert f"**Now:** «{C.sentence(C.START)}»" in b and "look" not in b.lower()
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    b = LOOP.banner(tmp_path)
    assert "**Now:**" not in b and "switched off (GHOST_CHARACTER=0)" in b and "diary" not in b


# ── review r1: the defects it found, and the mutants that had survived ──────

def test_a_reused_request_id_never_hands_the_owners_shade_to_someone_else(tmp_path):
    """The memo was read before the gate: the same id from a member, a public surface or a probe got the
    owner's line."""
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    sm.shade.set(S.CHASTENED, "x")
    with _req("web-4ug-reuse"):
        assert A._mood_shade_block(ctx) == S.line(S.CHASTENED)
    for who in ({"role": "member"}, {"surface": "public"}, {"origin": "probe"}):
        with _req("web-4ug-reuse", **who):
            assert A._mood_shade_block(ctx) == "", who
        assert A.mood_shade_heard("web-4ug-reuse") == "", who


def test_a_long_request_keeps_its_shade_line_while_others_come_and_go(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    sm.shade.set(S.PROUD, "x")
    with _req("web-4ug-long"):
        first = A._mood_shade_block(ctx)
    for i in range(A._MOOD_MEMO_MAX + 5):
        with _req(f"web-4ug-other-{i}"):
            A._mood_shade_block(ctx)
        if i % 100 == 0:
            with _req("web-4ug-long"):                          # the long request's next step
                assert A._mood_shade_block(ctx) == first == S.line(S.PROUD)
    with _req("web-4ug-long"):
        assert A._mood_shade_block(ctx) == first
    assert len(A._MOOD_SHADE_BY_REQ) <= A._MOOD_MEMO_MAX


def test_two_shades_in_one_second_are_two_moments(tmp_path):
    st = S.ShadeStore(tmp_path / "self")
    t0 = datetime.datetime(2026, 11, 12, 10, 0, 0, 100, tzinfo=UTC)
    st.set(S.RELIEVED, now=t0)
    assert st.take(now=t0) == S.RELIEVED
    st.set(S.PROUD, now=t0 + datetime.timedelta(milliseconds=200))
    assert st.take(now=t0 + datetime.timedelta(milliseconds=300)) == S.PROUD


def test_a_bad_turn_the_owners_mood_held_is_nothing_to_be_relieved_about(tmp_path):
    sm = SelfModel(root=tmp_path / "self")
    t0 = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    sm.state.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, now=t0)
    sm.note_owner_turn(M.TurnSignals(tool_calls=3, failures=3), now=t0 + datetime.timedelta(minutes=1))   # held
    assert sm._own_trouble_at is None
    sm2 = SelfModel(root=tmp_path / "self2")
    sm2.note_owner_turn(M.TurnSignals(tool_calls=3, failures=3), now=t0)
    assert sm2._own_trouble_at is not None
    sm2.note_owner_reaction(M.KIND_THUMB_DOWN, now=t0 + datetime.timedelta(minutes=1), reply_id="r")
    assert sm2._own_trouble_at == t0                            # a thumb is not the agent's OWN trouble
    sm3 = SelfModel(root=tmp_path / "self3")
    sm3.note_owner_reaction(M.KIND_THUMB_DOWN, now=t0, reply_id="r")
    sm3.shade.take(now=t0)
    sm3.note_owner_turn(M.TurnSignals(tool_calls=5), now=t0 + datetime.timedelta(minutes=5))
    assert (sm3.shade.current(now=t0 + datetime.timedelta(minutes=5)) or {}).get("shade") == S.CHASTENED   # no relief


def test_a_job_that_finishes_a_task_is_proud_and_a_task_with_no_lift_is_not(tmp_path):
    sm = SelfModel(root=tmp_path / "self")
    t0 = datetime.datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    sm.note_job_tasks_done(1, now=t0)
    assert sm.shade.current(now=t0)["shade"] == S.PROUD
    assert S.for_turn(M.TurnSignals(tasks_done=1), moved_dp=0.0, own_trouble_age_s=None) == ""
    assert S.for_turn(M.TurnSignals(tasks_done=1), moved_dp=-0.2, own_trouble_age_s=None) == ""


def test_a_file_that_cannot_be_read_is_never_written_over(tmp_path):
    home = _home(tmp_path)
    C.set_dial(home, "warmth", 2)
    C.set_pin(home, "warmth", True)
    p = home / "system/selfhood" / C.CHARACTER_FILENAME
    good = p.read_text()
    for bad in (good[:-5], '["a list"]', '{"dials": 3}'):
        p.write_text(bad, encoding="utf-8")
        C._MEMO.clear()
        assert C.load(home)["unreadable"] and C.load(home)["dials"] == C.START
        assert C.daily(home, _days(["playful"], ["playful"], ["playful"])) == []
        assert C.set_dial(home, "humour", 2)[1] == C.UNREADABLE and C.set_pin(home, "humour", True)[1] == C.UNREADABLE
        assert C.veto(home)[1] == C.UNREADABLE and "cannot be read" in C.command("character humour up", home)
        assert p.read_text() == bad                              # still there, untouched
    assert [r["kind"] for r in C.log_rows(home)].count(C.KIND_START) == 1
    p.write_text('{"dials": {"warmth": 1e999, "humour": "x"}, "pins": 5}', encoding="utf-8")
    assert C.load(home)["dials"] == {"warmth": 0, "humour": 0, "directness": 0} and C.load(home)["pins"] == []


def test_each_of_two_moves_made_in_one_run_can_be_taken_back(tmp_path):
    home = _home(tmp_path)
    said = C.daily(home, _days(["terse"], ["terse"], ["terse"]))
    assert len(said) == 2 and C.load(home)["dials"] == {"warmth": -2, "humour": 1, "directness": 1}
    assert C.veto(home)[1].startswith("directness +1 → +0")
    assert C.veto(home)[1].startswith("warmth -2 → -1")
    assert "no automatic move still stands" in C.veto(home)[1]
    assert C.load(home)["dials"] == C.START


def test_evidence_is_used_once_whatever_changed_the_dial(tmp_path):
    days = _days(["playful"], ["playful"], ["playful"], [], [])
    for kind in (C.KIND_MOVE, C.KIND_VETO, C.KIND_OWNER, C.KIND_UNPIN):
        rows = [{"at": "2026-11-03T23:00:00Z", "kind": kind, "dial": "humour", "from": 0, "to": 1}]
        assert C.propose(days, {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": []}, rows,
                         "2026-11-05") == [], kind
    # the day OF the change is behind it; the day after counts
    rows = [{"at": "2026-11-01T23:00:00Z", "kind": C.KIND_OWNER, "dial": "humour", "from": 0, "to": 1}]
    later = _days([], ["playful"], ["playful"], ["playful"])
    assert [m["dial"] for m in C.propose(later, {"dials": dict(C.START), "pins": []}, rows, "2026-11-04")] == ["humour"]
    assert C.propose(_days(["playful"], ["playful"], ["playful"]), {"dials": dict(C.START), "pins": []}, rows,
                     "2026-11-03") == []                       # only two days after it
    # a pin row is not a change: evidence before a pin still counts once it was never unpinned
    assert C._last_change([{"at": "2026-11-02T00:00:00Z", "kind": C.KIND_PIN, "dial": "humour"}], "humour") == ""


def test_the_day_records_are_trusted_for_nothing(tmp_path):
    home = _home(tmp_path)
    start = {"dials": dict(C.START), "pins": []}
    one = {"date": "2026-11-01", "pulls": {"humour": 1}}
    assert C.propose([one, dict(one), dict(one)], start, [], "2026-11-01") == []        # one date three times
    shuffled = list(reversed(_days(["playful"], ["playful"], ["playful"])))
    assert [m["dial"] for m in C.propose(shuffled, start, [], "2026-11-03")] == ["humour"]
    assert C.propose(shuffled, start, [], "2026-11-02") == []                           # `through` cuts
    junk = [{"date": "zzz", "pulls": {"humour": 1}}, {"pulls": {"humour": 1}}, "x", None,
            {"date": "2026-11-01", "pulls": None}]
    assert C.propose(junk, start, [], "2026-11-30") == []
    assert C.daily(home, junk + _days(["playful"], ["playful"], ["playful"])) != []
    assert C.load(home)["daily_through"] == "2026-11-03"                                # never "zzz"
    # the rule waits for a NEWER recorded day: the same last day again does nothing
    C.set_dial(home, "humour", 0)
    assert C.daily(home, _days(["playful"], ["playful"], ["playful"])) == []


def test_an_edit_from_outside_is_seen(tmp_path):
    import os
    home = _home(tmp_path)
    C.set_dial(home, "humour", 2)
    assert C.section(home)[1] == "character:-1,2,0" and C.dials(home)["humour"] == 2
    p = home / "system/selfhood" / C.CHARACTER_FILENAME
    raw = json.loads(p.read_text())
    raw["dials"]["humour"] = -2
    p.write_text(json.dumps(raw))
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000))
    assert C.section(home)[1] == "character:-1,-2,0" and C.dials(home)["humour"] == -2


def test_a_clean_turn_with_one_failed_call_is_a_clean_turn(tmp_path):
    ev = M.merge_events(M.turn_events(M.TurnSignals(tool_calls=10, failures=1)))
    # two parts: the failed call (a small trouble), and the turn that went through — the second part
    # also says "failed", and was counted as a second trouble instead of the clean turn it is
    assert ev.dp > 0 and D._own_counts(ev.cause) == {"trouble": 1, "clean_turns": 1}
    both = M.merge_events(M.turn_events(M.TurnSignals(tool_calls=4, failures=4, guards=1, pressure=True)))
    assert D._own_counts(both.cause) == {"trouble": 1}
    job = M.merge_events(M.turn_events(M.TurnSignals(tasks_done=2))).cause + " (in a background job)"
    assert D._own_counts(job) == {"tasks_done": 1}
    assert D._own_counts("something nobody wrote") == {}


def test_the_arc_is_a_half_open_day_and_ignores_manner_rows(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _journal(home, "mood.history.jsonl", [
        {"at": "2026-11-09T20:00:00Z", "label": "cheerful", "intensity": 3, "kind": "owner"},
        {"at": "2026-11-10T04:00:00Z", "label": "content", "intensity": 1, "kind": "decay"},      # exactly the boundary
        {"at": "2026-11-10T06:00:00Z", "label": "irritable", "intensity": 3, "kind": "manner"},   # a manner row: not a mood
        {"at": "2026-11-10T10:00:00Z", "label": "neutral", "intensity": 0, "kind": "owner"},
        {"at": "2026-11-11T04:00:00Z", "label": "down", "intensity": 2, "kind": "turn"},          # the next day's
    ])
    _journal(home, "mood.events.jsonl", [
        {"at": "2026-11-10T10:00:00Z", "kind": "owner", "dp": -0.2, "de": 0.0, "cause": "c", "after": "neutral"},
        {"at": "2026-11-10T11:00:00Z", "kind": "sentiment", "dp": 0.0, "de": 0.0, "cause": "c"}])
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC, now=now)
    assert rec["arc"] == {"content": 360} and rec["peak"]["label"] == "content"
    assert rec["owner_moods"] == [] and (rec["tone_up"], rec["tone_down"]) == (0, 0)   # "neutral" is not a mood asked for
    before = D.build_day(home, datetime.date(2026, 11, 9), tz=UTC, now=now)
    assert before["arc"] == {"cheerful": 480} and before["peak"]["label"] == "cheerful"
    after = D.build_day(home, datetime.date(2026, 11, 11), tz=UTC, now=now)
    assert after["peak"]["label"] == "down"
    # a mood carried in from the day before is that day's peak when nothing tops it
    _journal(home, "mood.history.jsonl", [{"at": "2026-11-12T12:00:00Z", "label": "content", "intensity": 1, "kind": "turn"}])
    assert D.build_day(home, datetime.date(2026, 11, 12), tz=UTC, now=now)["peak"]["label"] == "down"


def test_health_sends_no_cause_of_a_shade_and_no_other_new_key(tmp_path):
    from ghost_agent.api import routes as R
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1")
    mood = R._health_mood(_ctx(home, sm))
    assert set(mood) == set(R._MOOD_NEUTRAL)
    assert "thumbs-down" not in json.dumps({k: v for k, v in mood.items() if k != "cause"})
    assert set(R._health_character(_ctx(home, sm))) == set(C.DIALS)


# ── through the real loop: what the model is sent ───────────────────────────

def _loop_agent(home):
    from unittest.mock import MagicMock
    context = MagicMock()
    context.args.model = "test-model"
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 32000
    context.args.perfect_it = False
    context.args.native_tools = True
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(home / "system" / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.scratchpad.list_all.return_value = "nothing yet"
    context.turn_origin_label = None
    context.owner_memory_isolated = False
    context._replay_forbidden_tools = None
    context.self_model = SelfModel(root=home / "system" / "selfhood")
    agent = A.GhostAgent(context)
    agent.sent = []

    def stream(payload, *a, **k):
        agent.sent.append(json.loads(json.dumps(payload, default=str)))

        async def gen():
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": "Bruno is nine."}}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return gen()
    context.llm_client.stream_chat_completion = stream
    return agent


async def _send(agent, rid):
    agent.sent.clear()
    await agent.handle_chat({"messages": [{"role": "user", "content": "how old is Bruno?"}], "stream": False},
                            [], request_id=rid)
    assert len(agent.sent) == 1
    return agent.sent[0]


def _head(payload) -> str:
    import hashlib
    return hashlib.sha256((payload["messages"][0]["content"] + "\x00"
                           + json.dumps(payload.get("tools") or [], sort_keys=True)).encode()).hexdigest()


@pytest.mark.asyncio
async def test_the_model_is_sent_the_character_and_one_shade_after_the_head(tmp_path, monkeypatch):
    home = _home(tmp_path)
    agent = _loop_agent(home)
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    with _req("web-4ug-l0"):
        off = await _send(agent, "web-4ug-l0")
    monkeypatch.delenv("GHOST_CHARACTER")
    agent.context.self_model.shade.set(S.RELIEVED, "x")
    asked, real_heard = [], A.mood_shade_heard
    monkeypatch.setattr(A, "mood_shade_heard", lambda rid: asked.append((rid, real_heard(rid))) or asked[-1][1])
    with _req("web-4ug-l1"):
        on = await _send(agent, "web-4ug-l1")
    state = on["messages"][-1]["content"]
    sentence, shade = "«I am a little dry and fond of a light joke.»", "RIGHT NOW, FOR THIS REPLY ONLY: relief"
    assert state.count(sentence) == 1 and state.count(shade) == 1
    assert state.index("### DYNAMIC SYSTEM STATE") < state.index(shade) < state.index(sentence) \
        < state.index("</system_state_update>")
    for m in on["messages"][:-1]:                                # never the system prompt or an earlier message
        text = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        assert sentence not in text and shade not in text
    assert "little dry" not in json.dumps(on.get("tools") or [])
    assert A.self_image_heard("web-4ug-l1") == "character:-1,1,0"
    assert ("web-4ug-l1", S.RELIEVED) in asked                   # what the turn record's writer read for this request
    C.set_dial(home, "directness", 2)
    with _req("web-4ug-l2"):
        moved = await _send(agent, "web-4ug-l2")
    assert "blunt, saying it as it is" in moved["messages"][-1]["content"]
    assert "RIGHT NOW, FOR THIS REPLY ONLY" not in moved["messages"][-1]["content"]      # the moment has passed
    assert _head(off) == _head(on) == _head(moved)               # the cached head: the same bytes, any setting


@pytest.mark.asyncio
async def test_the_turn_record_says_which_shade_the_reply_carried(tmp_path, monkeypatch):
    home = _home(tmp_path)
    agent = _loop_agent(home)
    seen = {}
    real = agent.context.self_model.capture_turn

    def capture(*a, **k):
        seen.update(extra=dict(k.get("extra") or {}))
        return real(*a, **k)
    monkeypatch.setattr(agent.context.self_model, "capture_turn", capture)
    kept = []
    monkeypatch.setattr(A, "mood_shade_heard", lambda rid: kept.append(rid) or S.PROUD)
    with _req("web-4ug-rec"):
        await _send(agent, "web-4ug-rec")
    assert "web-4ug-rec" in kept                                 # the record's writer asked for this request


@pytest.mark.asyncio
async def test_the_idle_window_runs_the_character_phase_on_or_off(tmp_path, monkeypatch):
    """§4UH-loop: the idle phase is the character's turn alone (the Mirror's look is removed); its cooldown
    holds, and the days are closed with the character off too."""
    home = _home(tmp_path)
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent._bio_cooldown = lambda s: s
    calls = []

    async def phase(h, ran, notes, ctx=None):
        calls.append(("character", Path(h)))
    monkeypatch.setattr(agent, "_character_phase", phase)
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=object(), args=None)
    await agent._self_phase(ctx, [], {})
    assert calls == [("character", home)]
    await agent._self_phase(ctx, [], {})
    assert len(calls) == 1                                       # one try an hour
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    agent._last_mirror_try_at = datetime.datetime.min
    await agent._self_phase(ctx, [], {})
    assert len(calls) == 2
    assert not (home / "system" / "selfhood" / "mirror.jsonl").exists()


@pytest.mark.asyncio
async def test_with_the_character_off_the_days_are_still_closed_and_no_dial_moves(tmp_path, monkeypatch):
    home = _home(tmp_path)
    today = M.mood_day()
    rows = []
    for back in (3, 2, 1):
        at = datetime.datetime.combine(today - datetime.timedelta(days=back), datetime.time(12, 0)).astimezone().astimezone(UTC)
        rows.append({"at": at.isoformat().replace("+00:00", "Z"), "kind": "manner", "dp": 0.0, "de": 0.0,
                     "cause": "the owner set the manner: playful, 2 of 3"})
    _journal(home, "mood.events.jsonl", rows)
    monkeypatch.setenv("GHOST_CHARACTER", "0")
    agent = A.GhostAgent.__new__(A.GhostAgent)
    ran, notes = [], {}
    await agent._character_phase(home, ran, notes)
    assert len(D.load_days(home)) == 3 and C.load(home)["stored"] is False
    assert "dial" not in notes["character"]


@pytest.mark.asyncio
async def test_the_mood_is_settled_before_a_day_is_closed(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    settled = []
    real = sm.state.settle
    sm.state.settle = lambda *a, **k: settled.append(1) or real(*a, **k)
    agent = A.GhostAgent.__new__(A.GhostAgent)
    order = []
    import ghost_agent.selfhood.days as _D
    real_close = _D.close_days
    try:
        _D.close_days = lambda h, **k: order.append(("close", len(settled))) or real_close(h, **k)
        await agent._character_phase(home, [], {}, _ctx(home, sm))
    finally:
        _D.close_days = real_close
    assert order == [("close", 1)]


def test_the_same_last_day_never_runs_the_rule_twice(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    days = _days(["terse"], ["terse"], ["terse"])
    assert len(C.daily(home, days, now=now)) == 2 and C.load(home)["dials"]["directness"] == 1
    assert C.load(home)["changed"] == {"warmth": "2026-11-03", "directness": "2026-11-03"}   # the evidence is used up
    assert C.daily(home, days, now=now) == [] and C.load(home)["dials"]["directness"] == 1
    # a newer recorded day moves the gate on; the old evidence does not count a second time
    assert C.daily(home, days + _days([], start=datetime.date(2026, 11, 4)), now=now) == []
    assert C.load(home)["daily_through"] == "2026-11-04" and C.load(home)["dials"]["directness"] == 1
    # three MORE days of the same pull: the next step
    more = days + _days([], ["terse"], ["terse"], ["terse"], start=datetime.date(2026, 11, 4))
    assert [x.split(":")[0] for x in C.daily(home, more, now=now)] == ["directness +1 → +2"]   # warmth is at its end


# ── review r2 ───────────────────────────────────────────────────────────────

def test_a_file_that_cannot_be_read_is_said_everywhere_the_character_is_shown(tmp_path):
    from ghost_agent.api import routes as R
    home = _home(tmp_path)
    C.set_dial(home, "warmth", 2)
    C.set_pin(home, "warmth", True)
    p = home / "system/selfhood" / C.CHARACTER_FILENAME
    p.write_text(p.read_text()[:20], encoding="utf-8")
    C._MEMO.clear()
    v = C.view(home)
    assert v.startswith("⚠ The character file") and "STARTING character" in v and "Nothing is pinned" not in v
    assert "warmth pinned by the owner" in v                    # the log is shown as what happened BEFORE
    assert "STARTING character" in C.line(home) and "cannot be read" in C.line(home)
    assert C.unreadable(home) is True and R._health_character(_ctx(home)) == {}
    assert C.command("character", home).startswith("⚠ The character file")
    p.unlink()
    assert C.unreadable(home) is False and R._health_character(_ctx(home)) == C.START
    C.set_dial(home, "humour", 2)
    assert C.unreadable(home) is False and R._health_character(_ctx(home))["humour"] == 2


def test_the_owners_ways_of_typing_a_command_are_the_command(tmp_path):
    for text, want in (("character humour up please", ("set", "humour", "up")),
                       ("Character: humour up", ("set", "humour", "up")),
                       ("character, humour = +2", ("set", "humour", "+2")),
                       ("character humour + 1", ("set", "humour", "+1")),
                       ("character humour –1", ("set", "humour", "-1")),      # an en dash
                       ("character humour −2.", ("set", "humour", "-2")),     # a real minus sign
                       ("character warmth 0", ("set", "warmth", "0")),
                       ("character set directness at 1", ("set", "directness", "1")),
                       ("character directness less", ("set", "directness", "less")),
                       ("character veto please", ("veto",)), ("character show", ("view",)),
                       ("/character pin humor pls", ("pin", "humour"))):
        assert C.parse_command(text) == want, text
    for text in ("character humour 2x", "character humour 3", "character pin speed", "character humour\nup now ok ok ok ok"):
        assert C.parse_command(text) is None, text


def test_a_near_miss_is_told_to_the_model_and_ordinary_talk_is_not(tmp_path):
    for text in ("character humour 3", "character humour +3", "character reset", "character humour",
                 "character pin", "character humour up now", "character set warmth", "character undo it",
                 "character a b c d e humour"):                  # six words: the longest that is hinted
        hint = C.command_hint(text)
        assert "was NOT changed" in hint and "character humour +1" in hint and "no reset" in hint, text
        assert "just answer" in hint                              # … and the model may answer something else
    for text in ("character", "character humour up", "character veto",                 # real commands: no hint
                 "character development in this novel is weak", "character count of this file?",
                 "Character set design tips for a fantasy novel with seven point of view narrators",
                 "tell me about your character", "characters", "", "character set to UTF-8 please",
                 "Character set?", "character: set in stone", "character show me the dials", "character dials?",
                 "character a b c d e f humour"):                 # seven words: talk
        assert C.command_hint(text) == "", text


def test_the_agent_hands_the_near_miss_note_to_the_model(tmp_path, monkeypatch):
    import asyncio
    from ghost_agent.utils import logging as L
    home = _home(tmp_path)
    monkeypatch.setattr(L, "request_kind", lambda: "owner")
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    fake = SimpleNamespace(context=SimpleNamespace(memory_dir=str(home / "system" / "memory")))
    note = asyncio.run(A.GhostAgent._owner_rule_note(fake, {"messages": [{"role": "user", "content": "character humour 3"}]}))
    assert "was NOT changed" in str(note) and not getattr(note, "banner", "")      # a note for the model, not a system reply
    assert C.load(home)["dials"] == C.START
    monkeypatch.setattr(L, "request_kind", lambda: "probe")
    assert not asyncio.run(A.GhostAgent._owner_rule_note(fake, {"messages": [{"role": "user", "content": "character humour 3"}]}))


def test_each_move_of_one_dial_can_be_taken_back_in_turn(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    d1 = _days(["playful"], ["playful"], ["playful"])
    C.set_dial(home, "humour", 0, now=datetime.datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    assert C.daily(home, d1, now=now) and C.load(home)["dials"]["humour"] == 1
    d2 = d1 + _days(["playful"], ["playful"], ["playful"], start=datetime.date(2026, 11, 4))
    assert C.daily(home, d2, now=now) and C.load(home)["dials"]["humour"] == 2
    assert C.veto(home, now=now)[1].startswith("humour +2 → +1")
    assert C.veto(home, now=now)[1].startswith("humour +1 → +0")
    assert "no automatic move still stands" in C.veto(home, now=now)[1]
    # the owner set the dial BEFORE the rule moved it: that move can still be taken back
    home2 = _home(tmp_path / "b")
    C.set_dial(home2, "humour", 0, now=datetime.datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    C.daily(home2, d1, now=now)
    assert C.veto(home2, now=now)[1].startswith("humour +1 → +0")
    # a move row that does not say where it came from is not something to go back to
    home3 = _home(tmp_path / "c")
    C.ensure(home3)
    C._log(home3, {"at": "2026-11-05T05:00:00Z", "kind": C.KIND_MOVE, "dial": "warmth", "to": 0})
    assert C.veto(home3)[1] == "there is no automatic move to take back" and C.load(home3)["dials"] == C.START


def test_a_change_and_its_evidence_are_compared_on_the_mood_days_clock(tmp_path):
    home = _home(tmp_path)
    tz = datetime.timezone(datetime.timedelta(hours=2))          # Athens in winter
    # a veto at 02:30 local on the 12th is still mood day the 11th: the 12th's evidence counts
    C.ensure(home)
    C._log(home, {"at": "2026-11-09T05:00:00Z", "kind": C.KIND_MOVE, "dial": "humour", "from": 1, "to": 2})
    state = C.load(home); state["dials"]["humour"] = 2; C._write(home, state)
    at = datetime.datetime(2026, 11, 12, 2, 30, tzinfo=tz)
    import ghost_agent.selfhood.mood as _M
    real = _M.mood_day
    _M.mood_day = lambda when=None, *, tz=None: real(when, tz=datetime.timezone(datetime.timedelta(hours=2)))
    try:
        C.veto(home, now=at)
    finally:
        _M.mood_day = real
    assert C.load(home)["changed"]["humour"] == "2026-11-11"
    days = _days(["playful"], ["playful"], ["playful"], start=datetime.date(2026, 11, 12))
    assert [m["dial"] for m in C.propose(days, C.load(home), C.log_rows(home), "2026-11-14")] == ["humour"]


def test_dates_that_are_not_dates_never_stop_the_rule(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    junk = [{"date": "9999-99-99", "pulls": {"humour": 1}}, {"date": "2026-13-40", "pulls": {"humour": 1}},
            {"date": "2031-01-01", "pulls": {"humour": 1}},      # a real date, but after today
            {"date": 20261101, "pulls": {"humour": 1}}]
    assert C.daily(home, junk + _days(["playful"], ["playful"], ["playful"]), now=now) != []
    assert C.load(home)["daily_through"] == "2026-11-03"
    p = home / "system/selfhood" / C.CHARACTER_FILENAME
    raw = json.loads(p.read_text())
    for bad in (20261110, ["x"], "9999-99-99", "soon"):
        raw["daily_through"] = bad
        raw["changed"] = {"humour": bad, "speed": "2026-11-01"}
        p.write_text(json.dumps(raw))
        C._MEMO.clear()
        assert C.load(home)["daily_through"] == "" and C.load(home)["changed"] == {}
    assert C._real_date("2026-02-29") == "" and C._real_date("2028-02-29") == "2028-02-29" and C._real_date(None) == ""


def test_an_unpin_says_that_the_count_starts_again(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    C.set_pin(home, "humour", True, now=now)
    said = C.command("character unpin humour", home, now=now)
    assert said.startswith("✓ humour is free to move again — the count toward its next move starts from today")
    assert C.load(home)["changed"]["humour"] == M.mood_day(now).isoformat()
    assert C.daily(home, _days(["playful"], ["playful"], ["playful"]), now=now) == []


def test_the_shade_line_stays_for_a_request_however_long_it_runs(tmp_path, monkeypatch):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    sm.shade.set(S.RELIEVED, "x")
    clock = [1000.0]
    monkeypatch.setattr(A.time, "monotonic", lambda: clock[0])
    with _req("web-4ug-age"):
        first = A._mood_shade_block(ctx)
        for _ in range(5):                                      # five hours, a step an hour
            clock[0] += 3600
            assert A._mood_shade_block(ctx) == first == S.line(S.RELIEVED)
    assert A.mood_shade_heard("web-4ug-age") == S.RELIEVED
    clock[0] += A._SELF_IMAGE_FREEZE_S + 1                      # a client reusing the id long after
    sm.shade.set(S.PROUD, "y")
    with _req("web-4ug-age"):
        assert A._mood_shade_block(ctx) == S.line(S.PROUD)      # … is another request


def test_a_mood_held_under_a_minute_is_in_neither_the_arc_nor_the_peak(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _journal(home, "mood.history.jsonl", [
        {"at": "2026-11-09T22:00:00Z", "label": "bored", "intensity": 2, "kind": "quiet"},     # carried in: boredom
        {"at": "2026-11-10T09:00:00Z", "label": "irritable", "intensity": 3, "kind": "turn"},
        {"at": "2026-11-10T09:00:30Z", "label": "content", "intensity": 1, "kind": "thumb_up"},
        {"at": "2026-11-10T11:00:00Z", "label": "neutral", "intensity": 0, "kind": "decay"}])
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC, now=now)
    assert rec["arc"] == {"bored": 300, "content": 119} and rec["peak"]["label"] == "content"


@pytest.mark.asyncio
async def test_the_boredom_of_the_night_is_written_before_the_day_is_closed(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    calls = []
    real_settle, real_mood = sm.state.settle, sm.state.mood
    sm.state.settle = lambda *a, **k: calls.append("settle") or real_settle(*a, **k)
    sm.state.mood = lambda *a, **k: calls.append("read") or real_mood(*a, **k)
    import ghost_agent.selfhood.days as _D
    real_close = _D.close_days
    try:
        _D.close_days = lambda h, **k: calls.append("close") or real_close(h, **k)
        await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, [], {}, _ctx(home, sm))
    finally:
        _D.close_days = real_close
    assert calls[:3] == ["settle", "read", "close"]


# ── review r3 ───────────────────────────────────────────────────────────────

def test_the_command_is_read_without_a_regex_over_the_message():
    """A whole-message regex was cubic on a run of blanks: 1,600 spaces took four seconds, before the turn."""
    import time
    for text in ("character x" + " " * 200_000 + "y", "character " + "." * 200_000 + "x", "character\n" * 50_000,
                 "character humour up" + " " * 200):
        t0 = time.perf_counter()
        assert not C.is_command(text) and C.command_hint(text) == "" and C.command(text, None) is None
        assert time.perf_counter() - t0 < 0.5
    assert len("character humour up") < C.COMMAND_MAX_CHARS
    assert C.parse_command("character") == ("view",) and C.parse_command(" /Character. ") == ("view",)
    assert C.parse_command("character\n\nhumour up") == ("set", "humour", "up")
    assert C.parse_command("character veto please thanks") == ("veto",)            # every closing word goes
    for talk in ("characters", "characterwarmth up", "character_count = 5", "character-driven stories",
                 "character2", "characterς humour up"):
        assert C.parse_command(talk) is None and C.command_hint(talk) == "", talk


def test_a_veto_goes_back_only_from_where_the_move_left_the_dial(tmp_path):
    now = datetime.datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    gentle = _days(["gentle"], ["gentle"], ["gentle"])
    # the file removed, the log kept: the character restarts, and nothing before the restart stands
    home = _home(tmp_path)
    C.set_dial(home, "warmth", 0, now=datetime.datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    assert len(C.daily(home, gentle, now=now)) == 2            # warmth 0→1, directness 0→-1
    (home / "system/selfhood" / C.CHARACTER_FILENAME).unlink()
    C._MEMO.clear()
    assert C.veto(home, now=now)[1] in ("no automatic move still stands (each was taken back, or you set that "
                                        "dial since)", "there is no automatic move to take back")
    assert C.load(home)["dials"] == C.START and not any(r["kind"] == C.KIND_VETO for r in C.log_rows(home))
    # a move whose log row was lost: the dial is two steps on, and one veto does not undo both
    home2 = _home(tmp_path / "b")
    C.ensure(home2, now=now)
    C._log(home2, {"at": "2026-11-05T05:00:00Z", "kind": C.KIND_MOVE, "dial": "directness", "from": 0, "to": -1})
    st = C.load(home2); st["dials"]["directness"] = -2; C._write(home2, st)
    assert "no automatic move still stands" in C.veto(home2, now=now)[1] and C.load(home2)["dials"]["directness"] == -2
    # an old veto row that does not name its move, and a row with a truth value for a number
    home3 = _home(tmp_path / "c")
    C.ensure(home3, now=now)
    C._log(home3, {"at": "2026-11-05T05:00:00Z", "kind": C.KIND_MOVE, "dial": "humour", "from": 1, "to": 2})
    C._log(home3, {"at": "2026-11-06T05:00:00Z", "kind": C.KIND_VETO, "dial": "humour", "from": 2, "to": 1})
    C._log(home3, {"at": "2026-11-07T05:00:00Z", "kind": C.KIND_MOVE, "dial": "warmth", "from": True, "to": -1})
    assert "no automatic move still stands" in C.veto(home3, now=now)[1] and C.load(home3)["dials"] == C.START


def test_a_date_written_under_a_clock_that_was_ahead_does_not_stop_the_rule(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    C.ensure(home, now=now)
    st = C.load(home); st["daily_through"] = "2099-01-01"; C._write(home, st)
    assert C.daily(home, _days(["playful"], ["playful"], ["playful"]), now=now) != []
    assert C.load(home)["daily_through"] == "2026-11-03"


def test_a_change_is_stamped_with_its_mood_day(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setattr(C, "_mood_day", lambda now=None: "2026-01-02")
    C.set_dial(home, "humour", 2)
    C.set_pin(home, "warmth", True)
    C.set_pin(home, "warmth", False)
    assert C.load(home)["changed"] == {"humour": "2026-01-02", "warmth": "2026-01-02"}


def test_a_broken_file_is_warned_about_once_not_at_every_poll(tmp_path, caplog):
    import logging
    home = _home(tmp_path)
    (home / "system/selfhood" / C.CHARACTER_FILENAME).write_text("{broken", encoding="utf-8")
    C._MEMO.clear()
    with caplog.at_level(logging.WARNING, logger="GhostSelfhood"):
        for _ in range(5):
            assert C.unreadable(home) is True and C.dials(home) == C.START and C.section(home)[0]
    assert sum("unreadable" in r.getMessage() for r in caplog.records) == 1


def test_a_move_whose_log_row_fails_still_stands_and_is_said(tmp_path, monkeypatch):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
    C.ensure(home, now=now)
    monkeypatch.setattr(C, "_log", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    said = C.daily(home, _days(["playful"], ["playful"], ["playful"]), now=now)
    assert said and said[0].startswith("humour +1 → +2") and C.load(home)["dials"]["humour"] == 2


def test_the_request_s_shade_entry_leaves_with_its_record(tmp_path):
    import inspect
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    sm.shade.set(S.RELIEVED, "x")
    with _req("web-4ug-same-id"):
        assert A._mood_shade_block(ctx) == S.line(S.RELIEVED)
    assert "web-4ug-same-id" in A._MOOD_SHADE_BY_REQ


@pytest.mark.asyncio
async def test_a_client_that_sends_one_request_id_for_everything_is_not_stuck_on_the_first_shade(tmp_path):
    home = _home(tmp_path)
    agent = _loop_agent(home)
    agent.context.self_model.shade.set(S.RELIEVED, "x")
    with _req("web-4ug-fixed"):
        first = await _send(agent, "web-4ug-fixed")
    assert "RIGHT NOW, FOR THIS REPLY ONLY" in first["messages"][-1]["content"]
    assert "web-4ug-fixed" not in A._MOOD_SHADE_BY_REQ          # the record took it
    with _req("web-4ug-fixed"):
        second = await _send(agent, "web-4ug-fixed")
    assert "RIGHT NOW, FOR THIS REPLY ONLY" not in second["messages"][-1]["content"]


def test_a_mood_held_exactly_a_minute_counts(tmp_path):
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    _journal(home, "mood.history.jsonl", [
        {"at": "2026-11-10T09:00:00Z", "label": "irritable", "intensity": 3, "kind": "turn"},
        {"at": "2026-11-10T09:01:00Z", "label": "neutral", "intensity": 0, "kind": "decay"}])
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC, now=now)
    assert rec["arc"] == {"irritable": 1} and rec["peak"]["label"] == "irritable"


# ── §4UH: the dials move from the agent's own days, not from what the owner asks for ───────

def _rec(date, **kw):
    rec = {"date": date, "thumbs_up": 0, "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0,
           "tone_down": 0, "manners": [], "own": {"trouble": 0, "clean_turns": 0}, "shades": {}}
    for k, v in kw.items():
        if k in ("trouble", "clean_turns"):
            rec["own"][k] = v
        elif k in ("relieved", "proud", "chastened"):
            rec["shades"][k] = v
        else:
            rec[k] = v
    return rec


def _run(*recs, start=datetime.date(2026, 10, 1)):
    return [dict(r, date=(start + datetime.timedelta(days=i)).isoformat()) for i, r in enumerate(recs)]


def test_each_dial_listens_to_one_number_of_the_day():
    s = C.day_signals(_rec("d", thumbs_up=2, praise=1, tone_up=1, thumbs_down=1, clean_turns=5, trouble=1, relieved=1))
    assert s == {"warmth": 2.0, "humour": (5 + 1 - 2) / 6, "directness": (2 - 2) / 3}
    quiet = C.day_signals(_rec("d"))
    assert quiet == {"warmth": None, "humour": None, "directness": None}      # a day that says nothing
    assert C.day_signals(_rec("d", praise=40))["warmth"] == C.WARMTH_CAP      # one very good day is a good day
    assert C.day_signals(_rec("d", tone_down=9))["warmth"] == -C.WARMTH_CAP
    # §4UH-warmth: thumbs and corrections judge an ANSWER — directness only, never warmth
    assert C.day_signals(_rec("d", thumbs_up=40, thumbs_down=9))["warmth"] is None
    assert C.day_signals(_rec("d", clean_turns=2))["humour"] is None          # too little work to say
    assert C.day_signals(_rec("d", tone_up=1))["directness"] is None          # tone does not judge an answer
    assert C.day_signals(_rec("d", corrections=1)) == {"warmth": None, "humour": None, "directness": -2.0}
    # a day with messages the emotion model read: warmth is their warm and annoyed counts, nothing else
    assert C.day_signals(_rec("d", msgs_rated=4, msgs_warm=2, msgs_cold=1, msgs_thanks=2, praise=2))["warmth"] == 0.0
    # … plus the thanks it did not read (Greek): praise beyond the gratitude it saw
    assert C.day_signals(_rec("d", msgs_rated=4, msgs_warm=2, msgs_cold=1, msgs_thanks=2, praise=3))["warmth"] == 1.0
    assert C.day_signals(_rec("d", msgs_rated=3))["warmth"] == 0.0           # read, and nothing warm: a 0, not None
    assert C.day_signals(_rec("d", msgs_rated=9, msgs_warm=9))["warmth"] == C.WARMTH_CAP
    assert C.day_signals(_rec("d", msgs_rated=9, msgs_cold=9))["warmth"] == -C.WARMTH_CAP
    assert C.day_signals(_rec("d", msgs_rated=0, praise=1))["warmth"] == 1.0  # none read: the older counts
    assert C.day_signals({"own": "x", "shades": 3, "thumbs_up": True, "praise": "2", "tone_up": None})["warmth"] == 2.0
    assert set(C.SIGNAL_WORDS) == set(C.DIALS)
    # the manners the owner asks for move no dial: they are not in any signal
    assert C.day_signals(_rec("d", manners=["playful", "sarcastic"])) == quiet and not hasattr(C, "MANNER_PULL")


def test_a_day_counts_only_against_the_agents_own_norm(monkeypatch):
    monkeypatch.setattr(C, "day_pulls", _REAL_DAY_PULLS)
    usual = [_rec("d", praise=1, thumbs_up=1, clean_turns=4)] * 6
    days = _run(*usual, _rec("d", praise=3, thumbs_up=1, clean_turns=4),
                _rec("d", tone_down=1, thumbs_down=1, clean_turns=1, trouble=3),
                _rec("d"), _rec("d", praise=1, thumbs_up=1, clean_turns=4))
    pulls = C.day_pulls(days)
    assert pulls == {"2026-10-07": {"warmth": 1}, "2026-10-08": {"warmth": -1, "humour": -1, "directness": -1}}
    # the same good day means nothing to an agent whose days are all like it …
    always = _run(*[_rec("d", praise=3, clean_turns=4)] * 8)
    assert C.day_pulls(always) == {}
    # … and nothing before it has ``BASE_MIN`` days to compare with
    assert C.day_pulls(_run(*usual[:4], _rec("d", praise=3))) == {}
    # only the ``BASE_DAYS`` before a day are its norm: an old stretch of bad days is forgotten
    old_bad = _run(*[_rec("d", tone_down=2)] * 6, *[_rec("d", praise=1)] * C.BASE_DAYS, _rec("d", praise=1))
    assert old_bad[-1]["date"] not in C.day_pulls(old_bad)
    assert C.day_pulls([]) == {} and C.day_pulls([None, "x", {"date": "zzz"}]) == {}


def test_a_step_away_from_zero_needs_a_longer_lead_than_a_step_back(monkeypatch):
    monkeypatch.setattr(C, "LEAD_OUT", 4)
    assert (C.LEAD_IN, C.MOVE_MIN_DAYS) == (3, 3)
    three = _days(["playful"], ["playful"], ["playful"])
    four = _days(["playful"], ["playful"], ["playful"], ["playful"])
    at = lambda h: {"dials": {"warmth": 0, "humour": h, "directness": 0}, "pins": []}   # noqa: E731
    assert C.propose(three, at(1), [], "2026-11-03") == []                 # +1 → +2: away from 0
    assert [(m["from"], m["to"]) for m in C.propose(four, at(1), [], "2026-11-04")] == [(1, 2)]
    assert [(m["from"], m["to"]) for m in C.propose(three, at(-1), [], "2026-11-03")] == [(-1, 0)]   # back toward 0
    assert C.propose(three, at(0), [], "2026-11-03") == []                 # 0 → +1 is away too
    assert [(m["from"], m["to"]) for m in C.propose(four, at(0), [], "2026-11-04")] == [(0, 1)]


def test_the_real_rule_moves_a_dial_from_a_run_of_better_days(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "day_pulls", _REAL_DAY_PULLS)
    monkeypatch.setattr(C, "LEAD_OUT", 4)
    home = _home(tmp_path)
    now = datetime.datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
    days = _run(*[_rec("d", msgs_rated=4, msgs_warm=1)] * 6, *[_rec("d", msgs_rated=4, msgs_warm=3)] * 3)
    said = C.daily(home, days, now=now)
    assert [x.split(":")[0] for x in said] == ["warmth -1 → +0"]            # three better days: a step back toward 0
    assert f"in {C.SIGNAL_WORDS['warmth']}" in said[0] and "thumbs" not in said[0]
    row = [r for r in C.log_rows(home) if r["kind"] == C.KIND_MOVE][0]
    assert row["days_up"] == ["2026-10-07", "2026-10-08", "2026-10-09"] and row["days_down"] == []
    assert "moves by itself" in C.view(home) and "manner" not in C.view(home).split("How it got here")[0]


def test_health_sends_the_shades_age_by_its_own_clock_and_null_without_one(tmp_path):
    from ghost_agent.api import routes as R
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    assert R._MOOD_NEUTRAL["shade_age_s"] is None
    assert R._health_mood(_ctx(home, sm))["shade_age_s"] is None             # no shade: no age
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1")
    mood = R._health_mood(_ctx(home, sm))
    assert mood["shade"] and isinstance(mood["shade_age_s"], float) and 0.0 <= mood["shade_age_s"] < 60.0


# ── §4UH-e2e review (MAJOR): a day the owner set the mood still counts what happened ─────────────────

def test_a_set_mood_day_still_counts_its_corrections_tones_and_turns(tmp_path):
    from ghost_agent.selfhood.state import SelfStateThread
    from ghost_agent.selfhood.model import SelfModel as _SM
    home = _home(tmp_path)
    hold, full = SelfStateThread.HELD_OWNER_HOLD, _SM.HELD_SENTIMENT_FULL
    assert D.EVIDENCE_HELD == {hold, SelfStateThread.HELD_OWNER_LATER, full}
    _journal(home, "mood.events.jsonl", [
        {"at": "2026-11-10T09:00:00Z", "kind": "owner", "dp": 0.3, "de": 0.1, "cause": "c", "after": "cheerful"},
        {"at": "2026-11-10T10:00:00Z", "kind": "correction", "dp": -0.2, "de": 0.0, "cause": "c", "held": hold},
        {"at": "2026-11-10T10:05:00Z", "kind": "correction", "dp": -0.2, "de": 0.0, "cause": "c", "held": hold},
        {"at": "2026-11-10T10:10:00Z", "kind": "turn", "dp": -0.3, "de": 0.0,
         "cause": "2 of 2 tool calls failed in this turn", "held": hold},
        {"at": "2026-11-10T10:20:00Z", "kind": "turn", "dp": 0.05, "de": 0.0,
         "cause": "a turn of 3 tool calls went through without a failure", "held": hold},
        {"at": "2026-11-10T10:30:00Z", "kind": "sentiment", "dp": -0.08, "de": 0.0, "cause": "c", "held": hold},
        # a repeat is not counted twice
        {"at": "2026-11-10T10:40:00Z", "kind": "sentiment", "dp": -0.08, "de": 0.0, "cause": "c",
         "held": "already counted for this reply"},
        # a full tone share is a cap on new messages, not a repeat: counted (review)
        {"at": "2026-11-10T10:50:00Z", "kind": "sentiment", "dp": -0.08, "de": 0.0, "cause": "c", "held": full},
    ])
    rec = D.build_day(home, datetime.date(2026, 11, 10), tz=UTC,
                      now=datetime.datetime(2026, 11, 11, 12, tzinfo=UTC))
    assert rec["held"] == 7 and rec["events"] == 1                   # what MOVED the mood is still the owner's
    assert rec["corrections"] == 2 and rec["tone_down"] == 2 and rec["tone_up"] == 0
    assert rec["own"]["trouble"] == 1 and rec["own"]["clean_turns"] == 1
    assert set(rec["by_kind"]) == {"owner"} and rec["owner_moods"] == ["cheerful"]
    assert all(b["kind"] == "owner" for b in rec["biggest"]) or rec["biggest"] == []
    sig = C.day_signals(rec)
    assert sig["warmth"] is not None and sig["warmth"] < 0                # a day of corrections pulls DOWN


def test_the_chastened_line_does_not_claim_it_was_the_last_reply():
    # §4UH-e2e review: a correction's shade is heard one reply after the corrected one
    assert "one of your replies down" in S.line(S.CHASTENED) and "last reply" not in S.line(S.CHASTENED)
