"""A manner lasts until the end of the day it was set on — §4TS (2026-11-08).

Operator: "build the stepped fade for manners, i think until the end of day
makes the most sense". A manner used to stay until cleared: one "be sarcastic"
said once and forgotten coloured every reply for weeks, for everyone who hears
the tone. Now it ends at the first local 04:00 at least an hour after it was
set, stepping down 3 → 2 → 1 → off on the way (each step 2 h, spread evenly
when it was set later than that). Read-time only: nothing is stored when it
steps; saying it again starts a new day.

The pure tests use explicit time zones, so they hold on any machine; the
state tests compare against the module's own schedule in local time.
"""
from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pytest

from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.state import SelfStateThread
from ghost_agent.tools import introspect as I

ATH = ZoneInfo("Europe/Athens")
UTC = dt.timezone.utc


def _stamp(y, mo, d, h, mi=0, tz=ATH) -> str:
    """The on-disk stamp (naive UTC + "Z") for a local wall time in ``tz``."""
    t = dt.datetime(y, mo, d, h, mi, tzinfo=tz).astimezone(UTC).replace(tzinfo=None)
    return t.isoformat() + "Z"


def _local(steps, tz=ATH):
    return [(at.astimezone(tz).strftime("%d %H:%M"), lvl) for at, lvl in steps]


# ── the schedule (pure) ─────────────────────────────────────────────────────

def test_a_morning_manner_holds_all_day_and_steps_down_in_the_night():
    s = _stamp(2026, 11, 8, 10)
    assert _local(M.manner_schedule(3, s, tz=ATH)) == [("09 00:00", 2), ("09 02:00", 1), ("09 04:00", 0)]
    assert _local(M.manner_schedule(2, s, tz=ATH)) == [("09 02:00", 1), ("09 04:00", 0)]
    assert _local(M.manner_schedule(1, s, tz=ATH)) == [("09 04:00", 0)]


def test_a_late_manner_spreads_its_steps_over_the_time_it_has():
    """Set at 00:30 at 3: 3.5 h left, three even steps — never a drop at the
    moment it is set."""
    s = _stamp(2026, 11, 9, 0, 30)
    assert _local(M.manner_schedule(3, s, tz=ATH)) == [("09 01:40", 2), ("09 02:50", 1), ("09 04:00", 0)]
    at_set = dt.datetime(2026, 11, 9, 0, 30, tzinfo=ATH)
    assert M.manner_level(3, s, now=at_set, tz=ATH) == 3


def test_a_manner_set_just_before_the_end_lasts_until_the_next_day():
    # 03:15 — less than an hour before 04:00: the next day's end, not 45 min
    s = _stamp(2026, 11, 9, 3, 15)
    assert _local(M.manner_schedule(1, s, tz=ATH)) == [("10 04:00", 0)]
    # 02:59 — just over an hour: tonight's end
    s = _stamp(2026, 11, 9, 2, 59)
    assert _local(M.manner_schedule(1, s, tz=ATH)) == [("09 04:00", 0)]
    # 04:00 sharp and 04:30 — a new day has begun: tomorrow's end
    for h, mi in ((4, 0), (4, 30)):
        assert _local(M.manner_schedule(1, _stamp(2026, 11, 9, h, mi), tz=ATH)) == [("10 04:00", 0)]


@pytest.mark.parametrize("hh,mm,want", [
    (10, 0, 3), (23, 59, 3), (0, 0, 2), (1, 59, 2), (2, 0, 1), (3, 59, 1), (4, 0, 0), (12, 0, 0),
])
def test_the_level_at_each_moment(hh, mm, want):
    s = _stamp(2026, 11, 8, 10)
    day = 8 if hh >= 10 else 9
    if (hh, mm) == (12, 0):
        day = 9
    now = dt.datetime(2026, 11, day, hh, mm, tzinfo=ATH)
    assert M.manner_level(3, s, now=now, tz=ATH) == want


def test_the_day_end_is_local_04_00_across_a_clock_change():
    """Europe/Athens left summer time at 04:00 → 03:00 on 2026-10-25: the end
    is 04:00 EET (02:00 UTC), not 04:00 EEST (01:00 UTC)."""
    end = M.manner_ends_at(_stamp(2026, 10, 24, 12), tz=ATH)
    assert end == dt.datetime(2026, 10, 25, 2, 0, tzinfo=UTC)


def test_a_fixed_offset_zone_works_too():
    tz = dt.timezone(dt.timedelta(hours=-5))
    s = _stamp(2026, 11, 8, 20, tz=tz)
    assert _local(M.manner_schedule(2, s, tz=tz), tz=tz) == [("09 02:00", 1), ("09 04:00", 0)]


@pytest.mark.parametrize("bad", ["", "yesterday", None, 17, "2026-13-45T00:00:00Z"])
def test_a_stamp_nobody_can_read_is_no_manner(bad):
    assert M.manner_schedule(2, bad, tz=ATH) == [] and M.manner_level(2, bad, tz=ATH) == 0
    assert M.manner_ends_at(bad if isinstance(bad, str) else "", tz=ATH) is None


@pytest.mark.parametrize("n", [0, None, "x"])
def test_no_intensity_is_no_manner(n):
    assert M.manner_schedule(n, _stamp(2026, 11, 8, 10), tz=ATH) == []


def test_the_level_never_rises_and_never_skips_more_than_one_step():
    s = _stamp(2026, 11, 8, 22, 47)
    start = dt.datetime(2026, 11, 8, 22, 47, tzinfo=ATH)
    last = 3
    for k in range(0, 8 * 60):                                 # minute by minute to past the end
        lvl = M.manner_level(3, s, now=start + dt.timedelta(minutes=k), tz=ATH)
        assert lvl <= last and last - lvl <= 1, (k, last, lvl)
        last = lvl
    assert last == 0


# ── the state thread (local time, as the agent runs) ────────────────────────

T0 = dt.datetime(2026, 11, 6, 9, 0, tzinfo=UTC)


def _steps_local(set_at: dt.datetime, n: int):
    stamp = set_at.astimezone(UTC).replace(tzinfo=None).isoformat() + "Z"
    return M.manner_schedule(n, stamp)


def test_a_read_shows_each_step_and_then_no_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("sarcastic", 3, now=T0)
    steps = _steps_local(T0, 3)
    assert [lvl for _at, lvl in steps] == [2, 1, 0]
    assert s.mood(now=T0).manner_intensity == 3
    for at, lvl in steps:
        before = s.mood(now=at - dt.timedelta(seconds=1))
        v = s.mood(now=at)
        assert v.manner_intensity == lvl and before.manner_intensity == lvl + 1
        assert v.manner == ("sarcastic" if lvl else "")
    end = steps[-1][0]
    gone = s.mood(now=end)
    assert (gone.manner, gone.manner_intensity, gone.manner_set_at, gone.manner_steps) == ("", 0, "", ())
    # the tone line follows the level, and is gone with it (a neutral mood)
    assert "MANNER (set by the owner): sarcastic, 2 of 3." in s.mood(now=steps[0][0]).tone()
    assert gone.tone() == ""


def test_a_read_never_rewrites_the_stored_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("formal", 2, now=T0)
    raw = s.path.read_text()
    s.mood(now=T0 + dt.timedelta(days=3))
    s.settle(now=T0 + dt.timedelta(days=3))
    d = json.loads(s.path.read_text())["mood"]
    assert (d["manner"], d["manner_intensity"]) == ("formal", 2)          # the level is a function of when
    assert json.loads(raw)["mood"]["manner_set_at"] == d["manner_set_at"]


def test_saying_it_again_starts_a_new_day(tmp_path):
    """Before §4TS the same manner said again kept its first stamp; with a
    day that would have left it over at once."""
    s = SelfStateThread(tmp_path)
    s.set_manner("terse", 2, now=T0)
    later = T0 + dt.timedelta(days=2)
    assert s.mood(now=later).manner == ""
    _b, after = s.set_manner("terse", 2, now=later)
    assert after.manner == "terse" and after.manner_intensity == 2
    assert after.manner_set_at == later.replace(tzinfo=None).isoformat() + "Z"
    # and within the day: again resets the steps to the new time
    mid = later + dt.timedelta(hours=1)
    _b, again = s.set_manner("terse", 2, now=mid)
    assert again.manner_set_at == mid.replace(tzinfo=None).isoformat() + "Z"


def test_clearing_a_manner_whose_day_is_over_clears_nothing(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("gentle", 1, now=T0)
    before, after = s.set_manner("", 0, now=T0 + dt.timedelta(days=2))
    assert before.manner == "" and after.manner == ""


def test_neutral_over_an_expired_manner_says_there_was_none(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("gentle", 1, now=T0)
    before, _after = s.reset_mood(now=T0 + dt.timedelta(days=2))
    assert before.manner == ""


def test_an_event_keeps_the_manners_day(tmp_path):
    """A nudge, an owner mood and a settle copy the manner as it was set —
    none of them starts a new day for it."""
    s = SelfStateThread(tmp_path)
    s.set_manner("playful", 3, now=T0)
    s.nudge(-0.5, 0.3, "2 tool calls failed in this turn", M.KIND_TURN, now=T0 + dt.timedelta(hours=1))
    s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, now=T0 + dt.timedelta(hours=2))
    v = s.mood(now=T0 + dt.timedelta(hours=2))
    assert v.manner_set_at == "2026-11-06T09:00:00Z"
    assert s.mood(now=T0 + dt.timedelta(days=2)).manner == ""


def test_the_prefix_drops_the_manner_with_its_day(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("formal", 2, now=T0)
    assert "My manner (set by the owner): formal, 2 of 3." in s.format_as_prefix(now=T0)
    assert "manner" not in s.format_as_prefix(now=T0 + dt.timedelta(days=2))


# ── what the owner and the model are told ───────────────────────────────────

def test_the_fade_text_names_each_step_in_local_time():
    s = _stamp(2026, 11, 8, 10)
    steps = [(at.astimezone(UTC).replace(tzinfo=None).isoformat() + "Z", lvl)
             for at, lvl in M.manner_schedule(3, s, tz=ATH)]
    hh = [M.parse_iso_utc(a).astimezone().strftime("%H:%M") for a, _ in steps]
    assert I.manner_fade_text(steps) == f"down to 2 at {hh[0]}, down to 1 at {hh[1]}, off at {hh[2]}"
    assert I.manner_fade_text([]) == "" and I.manner_fade_text(None) == ""
    assert I.manner_fade_text([("garbage", 1)]) == ""


async def test_the_set_mood_reply_tells_the_owner_when_it_ends(tmp_path, monkeypatch):
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "mood_moves_on_this_request", lambda ctx: True)
    sm = SelfModel(root=tmp_path)
    out = await I.tool_introspect(action="set_mood", manner="sarcastic", intensity=2, self_model=sm)
    steps = sm.state.mood().manner_steps
    assert str(out) == ("Manner set to 'sarcastic', 2 of 3 for the rest of the day ("
                        f"{I.manner_fade_text(steps)}; saying it again starts a new day).")


def test_the_rollback_note_names_the_old_behaviour():
    """The docstrings that said a manner never decays were rewritten — the
    model-facing prose must not keep the old promise (lesson: a workaround
    outlives its bug)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    for rel in ("selfhood/mood.py", "selfhood/state.py", "selfhood/schema.py", "tools/introspect.py",
                "tools/registry.py", "core/prompts.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for stale in ("never decays", "It does not decay", "does not decay;", "stays until the owner"):
            assert stale not in text, (rel, stale)


# ── review fixes ────────────────────────────────────────────────────────────

def test_neutral_over_an_expired_manner_records_no_owner_word(tmp_path):
    """Review: "be neutral" over a manner whose day was over wrote a new state
    with a "the owner said so" event — later shown as a reason in Why.
    Round 2: it must still drop the expired manner off the file (a clock set
    back would revive it)."""
    s = SelfStateThread(tmp_path)
    s.set_manner("gentle", 1, now=T0)
    before, after = s.reset_mood(now=T0 + dt.timedelta(days=2))
    assert before.manner == "" and after.manner == ""
    assert not any(r.get("kind") == M.KIND_OWNER for r in s.state.mood.recent)
    assert s.state.mood.manner == "" and s.state.mood.manner_set_at == ""
    assert s.mood(now=T0 + dt.timedelta(hours=1)).manner == ""      # clock set back


def test_clearing_an_expired_manner_drops_it_off_the_file(tmp_path):
    """Round-2 review: set_manner("") over an expired manner returned early
    and left it stored; a clock set back brought it back."""
    s = SelfStateThread(tmp_path)
    s.set_manner("sarcastic", 2, now=T0)
    s.set_manner("", 0, now=T0 + dt.timedelta(days=2))
    assert s.state.mood.manner == ""
    assert s.mood(now=T0 + dt.timedelta(hours=1)).manner == ""


def test_neutral_with_nothing_on_file_writes_nothing(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("gentle", 1, now=T0)
    s.set_manner("", 0, now=T0 + dt.timedelta(minutes=5))
    raw = s.path.read_text()
    before, after = s.reset_mood(now=T0 + dt.timedelta(days=2))
    assert before == after and s.path.read_text() == raw
    raw = s.path.read_text()
    s.set_manner("", 0, now=T0 + dt.timedelta(days=2))
    assert s.path.read_text() == raw


def test_the_diary_block_does_not_step_with_the_clock(tmp_path):
    """Review: the diary's prompt hash is its regenerate guard; the manner's
    level in it changed three times a night."""
    s = SelfStateThread(tmp_path)
    s.set_manner("sarcastic", 3, now=T0)
    steps = _steps_local(T0, 3)
    blocks = {s.format_as_prefix(now=at - dt.timedelta(seconds=1), include_mood=False) for at, _ in steps}
    assert len(blocks) == 1 and "My manner (set by the owner): sarcastic." in blocks.pop()
    assert "sarcastic, 3 of 3." in s.format_as_prefix(now=T0)               # the full view keeps the level


def test_a_step_more_than_a_day_ahead_carries_its_date():
    now = dt.datetime.now(UTC)
    near = (now + dt.timedelta(hours=5)).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
    far = (now + dt.timedelta(hours=25)).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
    far_local = M.parse_iso_utc(far).astimezone()
    assert I.manner_fade_text([(near, 1), (far, 0)]) == (               # dated: the day ends tomorrow
        f"down to 1 at {M.parse_iso_utc(near).astimezone().strftime('%b %d %H:%M')}, "
        f"off at {far_local.strftime('%b %d %H:%M')}")
    assert I.manner_fade_text([(near, 0)]) == f"off at {M.parse_iso_utc(near).astimezone().strftime('%H:%M')}"



# ── the evil manner (2026-11-08, operator: "i'd like to have an evil tone") ──

@pytest.mark.parametrize("word", ["evil", "Evil", " villainous ", "sinister", "wicked", "menacing"])
def test_evil_and_its_synonyms_resolve(word):
    assert M.resolve_manner(word) == "evil"


def test_no_manner_word_is_a_mood_word():
    words = set(M.MANNERS) | set(M.MANNER_SYNONYMS)
    assert not words & (set(M.LABELS) | set(M.SYNONYMS) | M.CLEAR_WORDS)
    assert set(M.MANNER_SYNONYMS.values()) <= set(M.MANNERS) and set(M.MANNER_GLOSS) <= set(M.MANNERS)


def test_the_evil_tone_line_says_it_is_a_voice_and_keeps_the_limits():
    for line in (M.tone_line("neutral", 0, "evil", 2), M.tone_line("irritable", 2, "evil", 3)):
        assert "evil" in line and "theatrical villain's voice" in line
        assert "in the wording only" in line and "never at a person" in line
        assert "no insults or threats toward anyone" in line and "changes no answer and no refusal" in line
        assert "the same facts" in line and "no refusing" in line
    assert "villain" not in M.tone_line("neutral", 0, "sarcastic", 2)       # the gloss is evil's only


def test_the_tool_description_lists_every_manner():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    d = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "introspect")["function"]["description"]
    for m in M.MANNERS:
        assert m in d, m


async def test_the_owner_can_ask_for_a_sinister_manner(tmp_path, monkeypatch):
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "mood_moves_on_this_request", lambda ctx: True)
    sm = SelfModel(root=tmp_path)
    out = await I.tool_introspect(action="set_mood", manner="sinister", intensity=3, self_model=sm)
    assert str(out).startswith("Manner set to 'evil', 3 of 3")
    assert sm.state.mood().manner == "evil" and "evil, 3 of 3" in sm.state.mood().tone()


# ── closing review ───────────────────────────────────────────────────────────

def test_neutral_over_an_expired_manner_after_a_real_mood_records_no_owner_event(tmp_path):
    """Closing review (MAJOR): the first fix only held for a pair of exactly
    0, which a mood that ever moved never decays to — "neutral" then still
    wrote "the owner said so" while replying "nothing to clear"."""
    s = SelfStateThread(tmp_path)
    s.nudge(-0.5, 0.3, "2 tool calls failed in this turn", M.KIND_TURN, now=T0)
    s.set_manner("evil", 3, now=T0)
    s.settle(now=T0 + dt.timedelta(hours=8))
    before, after = s.reset_mood(now=T0 + dt.timedelta(hours=30))
    assert before.neutral and before.manner == "" and after.manner == ""
    assert not any(r.get("kind") == M.KIND_OWNER for r in s.state.mood.recent)
    assert (s.state.mood.p, s.state.mood.e) == (0.0, 0.0)            # the remainder is still zeroed
    assert s.state.mood.manner == ""                                  # and the expired manner is off the file


def test_neutral_over_a_live_manner_is_still_the_owners_word(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("evil", 2, now=T0)
    before, _after = s.reset_mood(now=T0 + dt.timedelta(minutes=5))
    assert before.manner == "evil"
    assert s.state.mood.recent[-1]["kind"] == M.KIND_OWNER


def test_a_write_drops_an_expired_manner_so_a_clock_set_back_cannot_revive_it(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("evil", 3, now=T0)
    s.nudge(0.1, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0 + dt.timedelta(days=5))
    assert s.state.mood.manner == ""
    assert s.mood(now=T0 + dt.timedelta(hours=1)).manner == ""          # the clock went back


def test_a_write_keeps_a_live_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("formal", 2, now=T0)
    s.nudge(0.1, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0 + dt.timedelta(minutes=10))
    assert (s.state.mood.manner, s.state.mood.manner_set_at) == ("formal", "2026-11-06T09:00:00Z")


def test_a_manner_stamped_far_in_the_future_is_no_manner():
    now = dt.datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
    future = (now + dt.timedelta(days=30)).replace(tzinfo=None).isoformat() + "Z"
    soon = (now + dt.timedelta(minutes=20)).replace(tzinfo=None).isoformat() + "Z"
    assert M.manner_level(3, future, now=now, tz=ATH) == 0
    assert M.manner_level(3, soon, now=now, tz=ATH) == 3                # minutes of skew are harmless


def test_when_the_day_ends_tomorrow_every_step_carries_its_date():
    now = dt.datetime.now(UTC)
    a = (now + dt.timedelta(hours=20)).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
    b = (now + dt.timedelta(hours=25)).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
    fa, fb = (M.parse_iso_utc(x).astimezone().strftime("%b %d %H:%M") for x in (a, b))
    assert I.manner_fade_text([(a, 1), (b, 0)]) == f"down to 1 at {fa}, off at {fb}"
