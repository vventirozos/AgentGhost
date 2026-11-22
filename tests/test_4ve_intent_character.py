"""§4VE WP-4: the day record's intent counts and the character's intent scales.

* the day record gains ``intent_rated``, ``intent`` {label: n}, ``urgent`` and the ``talk_*`` counts — only on a
  day the intent model read a message of; any other record is exactly what it was;
* the intent's own mood events (kind ``sentiment``) are not counted as the sentiment's tones;
* with ``intent.acts("character")`` on, warmth reads its intent scale and humour its banter scale on those days;
  off, every record reads exactly as before. A scale's name says what its number was computed from (the
  counting version, whether emotions were read) and norms never mix names.
"""
from __future__ import annotations

import datetime
import itertools
import json
import random

import pytest

from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import days as D
from ghost_agent.selfhood import intent as I
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import sentiment as SE

UTC = datetime.timezone.utc
DAY = datetime.date(2026, 11, 13)

#: the record's keys before §4VE (a record without an intent reading must carry exactly these)
OLD_KEYS = {"date", "events", "held", "by_kind", "thumbs_up", "thumbs_down", "corrections", "praise", "tone_up",
            "tone_down", "msgs_rated", "msgs_warm", "msgs_cold", "msgs_thanks", "owner_moods", "manners", "own",
            "arc", "peak", "biggest", "shades"}
INTENT_KEYS = {"intent_rated", "intent", "urgent", "talk_readable", "talk_read", "talk_emotions", "talk_warm",
               "talk_thanks", "talk_cold", "talk_banter", "talk_warm_words", "talk_thanks_words", "talk_banter_words",
               "talk_version"}


def _journal(home, *rows, name=SE.JOURNAL_FILENAME):
    p = home / D.SELFHOOD_REL / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dist(label, p=0.95):
    """A reading as the service sends it: each label scored ON ITS OWN (not a distribution) — the chosen label
    high, the others low."""
    return {k: (p if k == label else 0.05) for k in I.LABELS}


_NO = object()


def _row(ts, label=None, *, text="ok", emotions=None, held="", sent="neutral", urgent=None, intent=None,
         effect=_NO, iheld=""):
    """A journal row as ``score_message`` writes it. ``effect``: the live path's ``intent_effect`` (absent: a row
    from before the field, whose witnesses are read at day close)."""
    r = {"ts": ts, "label": sent, "held": held, "text": text}
    if emotions is not None:
        r["emotions"] = emotions
    if intent is not None:
        r["intent"] = intent
    elif label is not None:
        r["intent"] = _dist(label)
    if urgent is not None:
        r["urgent"] = urgent
    if effect is not _NO:
        r["intent_effect"], r["intent_held"], r["register"] = effect, iheld, ""
    return r


def _at(minute):
    return f"2026-11-13T09:{minute:02d}:00+00:00"


@pytest.fixture(autouse=True)
def _intent_env(monkeypatch):
    for k in ("GHOST_INTENT", "GHOST_INTENT_CHARACTER"):
        monkeypatch.delenv(k, raising=False)


def _count(home, *rows):
    _journal(home, *rows)
    return D.owner_intents(home, DAY, tz=UTC)


# ── the day record ──────────────────────────────────────────────────────────

def test_a_day_without_an_intent_reading_has_exactly_the_old_keys(tmp_path):
    # catches: intent keys written with zeros on every day (old and new records would differ in shape)
    _journal(tmp_path, {"ts": "2026-11-13T09:00:00+00:00", "label": "positive", "emotions": {"gratitude": 0.9},
                        "held": "", "text": "thanks Elena's note helped"},
             _row("2026-11-14T09:00:00+00:00", "banter", text="haha"))       # a reading, but another day
    rec = D.build_day(tmp_path, DAY, tz=UTC)
    assert set(rec) == OLD_KEYS
    assert D.owner_intents(tmp_path, DAY, tz=UTC) is None


def test_a_day_with_readings_carries_the_counts_and_no_words(tmp_path):
    # catches: counts missing from the record, a word of a message in it, the urgent cut ignored
    _journal(tmp_path,
             _row(_at(0), "thanks", text="thanks, that fixed it", emotions={"gratitude": 0.95}),
             _row(_at(10), "banter", text="haha nice one Markos would say", emotions={}),
             _row(_at(20), "advice", text="should I move to the new flat?", urgent=0.9),
             _row(_at(30), "question", text="what time is it in Kosmas's city", urgent=0.5),
             _row(_at(40), "instruction", text="run it", emotions={"annoyance": 0.7}))
    rec = D.build_day(tmp_path, DAY, tz=UTC)
    assert set(rec) == OLD_KEYS | INTENT_KEYS
    assert rec["intent_rated"] == 5 and rec["urgent"] == 1 and rec["talk_version"] == D.TALK_COUNT_VERSION
    # "thanks" is never a counted label (intent.counted): the thanks is counted as warm from its gratitude
    assert rec["intent"] == {**{k: 0 for k in I.LABELS}, "banter": 1, "advice": 1, "question": 1, "instruction": 1}
    assert (rec["talk_read"], rec["talk_emotions"], rec["talk_warm"], rec["talk_thanks"], rec["talk_cold"],
            rec["talk_banter"]) == (5, 3, 1, 1, 1, 1)
    blob = json.dumps(rec)
    assert "Markos" not in blob and "flat" not in blob
    line = D.day_line(rec)
    # review r2 (MINOR 3): what a message was READ as — not that it was accepted
    assert "read for intent: 5" in line and "1 joking" in line and "1 read as asking my view" in line
    assert "1 warm (" in line and "1 read as urgent" in line


def test_day_boundary_at_four_and_the_rotated_journal(tmp_path):
    # catches: a calendar day instead of the mood day; the rotated `.1` journal not read
    _journal(tmp_path, _row("2026-11-13T09:00:00+00:00", "question"))
    p = tmp_path / D.SELFHOOD_REL / SE.JOURNAL_FILENAME
    p.rename(p.with_name(p.name + ".1"))
    got = _count(tmp_path,
                 _row("2026-11-14T03:30:00+00:00", "question"),     # before 04:00: still the 13th
                 _row("2026-11-14T04:30:00+00:00", "question"),     # the 14th
                 _row("2026-11-13T03:59:00+00:00", "question"))     # the 12th
    assert got["intent_rated"] == 2 and got["intent"]["question"] == 2


def test_thanks_comes_from_the_praise_check_or_gratitude_never_the_label(tmp_path):
    # catches: thanks counted on the intent label (the model never predicts it reliably: intent.counted refuses
    # it); either witness dropped — the praise check (``has_praise``) or gratitude felt under any label
    got = _count(tmp_path,
                 _row(_at(0), "thanks", text="thanks for the file from Elena"),       # the label alone
                 _row(_at(1), "thanks", text="the report from Elena"),
                 _row(_at(2), "thanks", text="great job"),                            # the praise check
                 _row(_at(3), "question", text="what is the time", emotions={"gratitude": 0.9}))  # gratitude
    assert (got["talk_read"], got["talk_warm"], got["talk_thanks"]) == (4, 2, 2)
    assert got["intent"]["thanks"] == 0


def test_only_the_top_label_at_its_own_floor_is_counted(tmp_path):
    # catches: a second-place label counted (the scores are independent; a greeting reads banter 0.83 too); a
    # label under its floor counted; a floor re-implemented here instead of read from intent.counted
    greet = {**{k: 0.01 for k in I.LABELS}, "greeting": 1.0, "banter": 0.83}
    floor = I.LABEL_MIN["advice"]
    under = {**{k: 0.01 for k in I.LABELS}, "advice": round(floor - 0.01, 4)}
    at = {**{k: 0.01 for k in I.LABELS}, "advice": floor}
    got = _count(tmp_path, _row(_at(0), intent=greet, text="hi there haha"),
                 _row(_at(1), intent=under, text="should I?"), _row(_at(2), intent=at, text="should I?"))
    assert got["intent"]["greeting"] == 1 and got["intent"]["banter"] == 0 and got["talk_banter"] == 0
    assert got["intent"]["advice"] == 1 and got["intent_rated"] == 3


def test_urgency_counts_without_an_intent_label(tmp_path):
    # catches: urgency counted only on rows with an intent; an urgent-only day dropped as "no reading"
    got = _count(tmp_path, _row(_at(0), urgent=0.95, text="now please"),
                 _row(_at(1), intent={"flirting": 0.9}, urgent=0.9))           # a bad intent, a good urgency
    assert got is not None and got["urgent"] == 2 and got["intent_rated"] == 0
    rec = D.build_day(tmp_path, DAY, tz=UTC)
    assert set(rec) == OLD_KEYS | INTENT_KEYS


def test_a_row_held_as_the_wording_of_a_task_is_read_and_counts_for_nothing_else(tmp_path):
    # catches: a task's words counted as the owner's annoyance (cold) or as a message of the talk scale
    got = _count(tmp_path, _row(_at(0), "instruction", text="kill the failed jobs", emotions={"anger": 0.9},
                                held=SE.HELD_TASK, sent="negative"))
    assert got["intent_rated"] == 1 and got["intent"]["instruction"] == 1
    assert (got["talk_read"], got["talk_cold"], got["talk_warm"]) == (0, 0, 0)


def test_sarcasm_keeps_its_annoyance_and_is_never_warm_or_banter(tmp_path):
    # catches: "thanks a lot, you broke it again" counted as warm thanks; "lol it crashed again" as a joke
    got = _count(tmp_path,
                 _row(_at(0), "thanks", text="thanks a lot, you broke it again",
                      emotions={"gratitude": 0.8, "annoyance": 0.6}, held=SE.HELD_SARCASM, sent="positive"),
                 _row(_at(5), "banter", text="lol it crashed again", sent="positive"))
    assert (got["talk_read"], got["talk_warm"], got["talk_cold"], got["talk_banter"]) == (2, 0, 1, 0)


def test_banter_needs_its_witness_and_an_annoyed_joke_stays_annoyed(tmp_path):
    # catches: banter counted from the classifier alone, or over an annoyed emotion; an annoyed joke not cold
    got = _count(tmp_path,
                 _row(_at(0), "banter", text="haha good one"),
                 _row(_at(1), "banter", text="good one"),                       # no laugh
                 _row(_at(2), "banter", text="haha sure", emotions={"annoyance": 0.7}))
    assert (got["talk_banter"], got["talk_cold"]) == (1, 1)


def test_what_the_live_path_doubted_counts_for_no_joke_or_warmth(tmp_path):
    # MINOR 4. catches: day close re-reading a joke the live path refused right after the agent's trouble (its
    # witnesses read without ``after_trouble``); a doubt recorded on the row ignored
    got = _count(tmp_path,
                 _row(_at(0), "banter", text="haha ok then", emotions={"amusement": 0.8}, effect="",
                      iheld=SE.HELD_SARCASM),
                 _row(_at(1), "banter", text="haha thanks for the file", emotions={"gratitude": 0.9,
                      "annoyance": 0.7}, effect="", iheld=I.WHY_COLD),
                 _row(_at(2), "banter", text="haha ok then", emotions={"amusement": 0.8},
                      effect=I.CAUSE_BANTER, iheld=SE.INTENT_HELD_JOURNAL))
    assert (got["talk_read"], got["talk_banter"], got["talk_warm"]) == (3, 1, 0)


@pytest.mark.parametrize("row,warm,joke", [
    (_row(_at(0), "confiding", text="I feel worn out about my sister's move"), 1, 0),   # no field: re-read
    (_row(_at(1), "confiding", text="fix the login page"), 0, 0),                     # a task's wording
    (_row(_at(2), "confiding", text="I feel worn out today", effect=""), 0, 0),       # the live path said no
    (_row(_at(3), "confiding", text="ok", effect=I.CAUSE_CONFIDED), 1, 0),            # … and here yes
    (_row(_at(4), "banter", text="ok", effect=I.CAUSE_BANTER), 0, 1),
    (_row(_at(5), "banter", text="haha", effect=""), 0, 0),                           # the live path said no
    (_row(_at(6), "banter", text="haha"), 0, 1),                                      # no field: re-read
])
def test_the_live_paths_verdict_decides_a_confiding_and_a_joke(tmp_path, row, warm, joke):
    # catches: a confiding counted without its witness (``confided = True``); the row's ``intent_effect``
    # ignored in favour of a re-read (each row on its own: in one day a yes and a no would cancel out)
    got = _count(tmp_path, row)
    assert (got["talk_warm"], got["talk_banter"]) == (warm, joke)


def test_a_confiding_counts_alike_whatever_the_sentiments_own_hold(tmp_path):
    # review r2 (MAJOR). catches: the sentiment's ``held`` read as the intent's refusal — a confident tone with no
    # emotion behind it (HELD_UNBACKED) refused the confiding the live path ACCEPTED (it booked its mood event),
    # while the same message at a lower score ("below 0.85") counted. Lack of backing refuses only a label that
    # is itself a tone (``intent.INTENT_TONE``): the joke below is refused, the confiding is not
    text = "I finally told my sister Elena about the move, I feel lighter"
    emo = {"realization": 0.6}
    assert M.sentiment_backing("positive", emo) is False
    got = _count(tmp_path,
                 _row(_at(0), "confiding", text=text, emotions=emo, sent="positive", held=SE.HELD_UNBACKED,
                      effect=I.CAUSE_CONFIDED),
                 _row(_at(1), "confiding", text=text, emotions=emo, sent="positive", held="below 0.85",
                      effect=I.CAUSE_CONFIDED),
                 _row(_at(2), "banter", text="haha nice", emotions=emo, sent="positive", held=SE.HELD_UNBACKED,
                      effect="", iheld=SE.HELD_UNBACKED))
    assert (got["talk_read"], got["talk_warm"], got["talk_banter"]) == (3, 2, 0)
    assert (got["talk_warm_words"], got["talk_banter_words"]) == (2, 1)    # no emotions read: nothing unbacked


#: the class: readings × tones × emotions × words, each journalled as the live path writes it
_CLASS_TEXTS = ("haha nice one", "I feel lighter after telling Elena", "great job, thanks!", "lol it crashed again",
                "fix the login page")
_CLASS_EMOTIONS = (None, {"amusement": 0.9}, {"annoyance": 0.8}, {"gratitude": 0.9}, {"realization": 0.6})
_SENT_HOLDS = ("", SE.HELD_UNBACKED, "below 0.85", "neutral")


def test_warm_and_joking_are_counted_exactly_when_the_rows_intent_was_not_refused(tmp_path):
    # review r2 (MAJOR, the class). catches: a second refusal rule beside intent.py's (the sentiment's ``held``
    # read as a refusal; a refusal reason missing), a joke counted the live path did not accept or the other way
    # round. For every row: the joke is counted exactly when its ``intent_effect`` is banter's event; nothing is
    # warm or a joke when the row's intent fields say refused (``intent_held`` is a refusal reason for a label
    # with a mood event, ``intent.refusal`` for the others); a thanks or a gratitude is warm when not refused;
    # and the sentiment's own hold changes nothing
    reasons = set()
    cases = []
    for lab, tone, emo, text in itertools.product(I.LABELS, ("positive", "negative", "neutral"), _CLASS_EMOTIONS,
                                                  _CLASS_TEXTS):
        reading = {"label": tone, "intent": _dist(lab), **({"emotions": emo} if emo is not None else {})}
        ev, why = I.mood_event(reading, text)
        counted = I.counted(reading)
        refused = I.refusal(reading, counted, text)
        reasons.add(refused)
        words = {"label": tone, "intent": reading["intent"]}           # the same reading without the emotions
        wev = I.mood_event(words, text)[0]
        cases.append((lab, tone, emo, text, ev.cause if ev else "", why, counted, refused,
                      I.refusal(words, counted, text), wev.cause if wev else ""))
    reasons.discard("")
    n = 0
    for lab, tone, emo, text, effect, why, counted, refused, refused_w, effect_w in cases:
        seen = set()
        _feeling, thanked = D._warm_routes(tone, text)                  # §4VE functional run: the thanks route
        for held in _SENT_HOLDS:
            n += 1
            home = tmp_path / str(n)
            got = _count(home, _row(_at(0), lab, text=text, emotions=emo, sent=tone, held=held, effect=effect,
                                    iheld=why if not effect else SE.INTENT_HELD_JOURNAL))
            seen.add((got["talk_warm"], got["talk_banter"]))
            says_refused = (why in reasons) if I.MOOD_OUTCOMES.get(counted) else bool(refused)
            assert got["talk_banter"] == (1 if effect == I.CAUSE_BANTER else 0), (lab, tone, emo, text)
            if says_refused:
                assert (got["talk_warm"], got["talk_banter"]) == (0, 0), (lab, tone, emo, text, why)
            elif (thanked and (M.has_praise(text) or (emo or {}).get("gratitude"))) or effect == I.CAUSE_CONFIDED:
                assert got["talk_warm"] == 1, (lab, tone, emo, text)
            # the words-only counts: the same rule, on the reading without its emotions
            if why in (SE.HELD_SARCASM, SE.HELD_TASK):
                continue                                                 # the live path's doubt: nothing counts
            assert got["talk_banter_words"] == (1 if effect_w == I.CAUSE_BANTER else 0), (lab, tone, emo, text)
            if refused_w:
                assert (got["talk_warm_words"], got["talk_banter_words"]) == (0, 0), (lab, tone, emo, text)
            else:
                assert got["talk_warm_words"] == (1 if (thanked and M.has_praise(text)) or effect_w == I.CAUSE_CONFIDED
                                                  else 0)
        assert len(seen) == 1, (lab, tone, emo, text)          # the sentiment's own hold: no difference
    assert reasons >= {SE.HELD_UNBACKED, I.WHY_TONE_AGAINST, I.WHY_COLD, SE.HELD_SARCASM}


def test_a_joke_feeds_humour_not_also_warmth_on_its_laugh(tmp_path):
    # MINOR 5. catches: amusement/joy (warm emotions) counting a joke as warm too — one message, two dials
    got = _count(tmp_path,
                 _row(_at(0), "banter", text="haha good one", emotions={"amusement": 0.9, "joy": 0.6}),
                 _row(_at(1), "banter", text="haha good one", emotions={"amusement": 0.9, "admiration": 0.8}))
    assert (got["talk_banter"], got["talk_warm"]) == (2, 1)


def test_praise_read_under_another_label_is_counted_as_thanks(tmp_path):
    # MINOR 7. catches: "great job!" (the praise check fires; the top label is not thanks; admiration, no
    # gratitude) counted warm and again as the praise check's extra — one message, counted twice
    got = _count(tmp_path, *[_row(_at(i), "banter", text="great job!", emotions={"admiration": 0.9})
                             for i in range(3)])
    assert (got["talk_warm"], got["talk_thanks"]) == (3, 3)
    rec = {**got, "praise": 3, "date": DAY.isoformat()}
    assert C.day_signals(rec, intent=True)["warmth"] == pytest.approx(1.0)
    assert C._talk(rec)[0] == 3                                   # three messages, not six


def test_a_damaged_or_unclear_reading_is_not_a_label(tmp_path):
    # catches: a journal row trusted (an unknown label, a number out of range); an unclear top label counted
    _journal(tmp_path,
             _row(_at(0), intent={"flirting": 0.9}),
             _row(_at(1), intent={"banter": 7}))
    assert D.owner_intents(tmp_path, DAY, tz=UTC) is None
    got = _count(tmp_path, _row(_at(2), intent={**{k: 0.0 for k in I.LABELS}, "advice": 0.5, "question": 0.45}))
    assert got["intent_rated"] == 1 and sum(got["intent"].values()) == 0


def test_every_label_is_decided_for_the_character():
    # catches: a label added to intent.LABELS (or renamed) with no decision about which dial it feeds
    groups = (D.TALK_WARM_INTENTS, D.TALK_BANTER_INTENTS, D.TALK_NO_SIGNAL_INTENTS)
    named = [lab for g in groups for lab in g]
    assert sorted(named) == sorted(I.LABELS) and len(named) == len(set(named))


#: words that pass each label's witness, and words that do not
_PASS = {"confiding": "I feel worn out about my sister's move", "thanks": "thanks for the file", "banter": "haha"}
_FAIL = {"confiding": "fix the login page", "thanks": "the file from Elena", "banter": "good one"}


@pytest.mark.parametrize("label", I.LABELS)
def test_each_label_counts_where_its_group_says(tmp_path, label):
    # catches: a group constant that does not match what the counting really does; a witness that never fails
    got = _count(tmp_path, _row(_at(0), label, text=_PASS.get(label, "ok then")),
                 _row(_at(1), label, text=_FAIL.get(label, "ok then")))
    assert got["talk_warm"] == (1 if label in D.TALK_WARM_INTENTS else 0)
    assert got["talk_banter"] == (1 if label in D.TALK_BANTER_INTENTS else 0)


#: a fixed table of rows and what ``owner_intents`` counts from it, per ``TALK_COUNT_VERSION``
_PIN_ROWS = [
    _row(_at(0), "thanks", text="thanks, that fixed it", emotions={"gratitude": 0.95}, effect=""),
    _row(_at(1), "thanks", text="the report from Elena"),
    _row(_at(2), "banter", text="haha nice one", emotions={"amusement": 0.9}, effect=I.CAUSE_BANTER),
    _row(_at(3), "banter", text="haha ok then", emotions={"amusement": 0.8}, effect="", iheld=SE.HELD_SARCASM),
    _row(_at(4), "banter", text="lol it crashed again", sent="positive"),
    _row(_at(5), "confiding", text="I feel worn out about my sister's move", emotions={"sadness": 0.7}),
    _row(_at(6), "confiding", text="fix the login page"),
    _row(_at(7), "instruction", text="kill the failed jobs", emotions={"anger": 0.9}, held=SE.HELD_TASK,
         sent="negative"),
    _row(_at(8), "question", text="why is it broken again", emotions={"annoyance": 0.8}, sent="negative"),
    _row(_at(9), "banter", text="great job!", emotions={"admiration": 0.9}),
    _row(_at(10), "advice", text="should I take the job?", urgent=0.9),
    _row(_at(11), urgent=0.8),
    _row(_at(12), intent={**{k: 0.0 for k in I.LABELS}, "advice": 0.5, "question": 0.45}),
    _row(_at(13), "venting", text="my boss is awful", emotions={"anger": 0.8}, sent="negative"),
    _row(_at(14), "thanks", text="thanks a lot, you broke it", emotions={"gratitude": 0.8, "annoyance": 0.6},
         held=SE.HELD_SARCASM, sent="positive"),
    _row(_at(15), text="ok thanks", sent="unknown", held="not English"),     # the language gate: not readable
    # version 2: the real service's reading of a sarcastic insult (negative tone, admiration)
    _row(_at(16), "banter", text="great, you broke it again", emotions={"admiration": 0.93}, sent="negative",
         held="below 0.85", effect="", iheld=I.WHY_UNCLEAR),
    _row(_at(17), "question", text="I'm so sad but thank you for staying with me",
         emotions={"sadness": 0.7, "gratitude": 0.8}, sent="negative"),
]
#: version 1's line was counted on rows 0–15 (it stays as the record of what version 1 counted)
_PINNED = {1: {"intent_rated": 14, "urgent": 2, "talk_readable": 14, "talk_read": 13, "talk_emotions": 8,
               "talk_warm": 3, "talk_thanks": 2, "talk_cold": 3, "talk_banter": 1,
               # without the emotion model: "great job!" (the praise check) and the confiding; the laugh's joke
               "talk_warm_words": 2, "talk_thanks_words": 1, "talk_banter_words": 1,
               "intent": {**{k: 0 for k in I.LABELS}, "banter": 4, "confiding": 2, "instruction": 1, "question": 1,
                          "advice": 1, "venting": 1}},
           # version 2 (§4VE functional run 11-21, rows 16–17 added): the sarcastic insult is no longer warm (under
           # version 1 it was: talk_warm 5); the sad thanks is warm by its gratitude. The same day's review refined
           # the thanks route before version 2 shipped (thanks about trouble close only under a negative tone): no
           # row of this table changes under it — "thanks a lot, you broke it" is held as sarcasm, "lol it crashed
           # again" has no thanks, the insult is negative — so the line stays
           2: {"intent_rated": 16, "urgent": 2, "talk_readable": 16, "talk_read": 15, "talk_emotions": 10,
               "talk_warm": 4, "talk_thanks": 3, "talk_cold": 3, "talk_banter": 1,
               "talk_warm_words": 2, "talk_thanks_words": 1, "talk_banter_words": 1,
               "intent": {**{k: 0 for k in I.LABELS}, "banter": 5, "confiding": 2, "instruction": 1, "question": 2,
                          "advice": 1, "venting": 1}}}


def test_the_counting_version_moves_with_what_is_counted(tmp_path):
    # catches: the counting rules (or a witness they call in intent.py) changed and TALK_COUNT_VERSION not bumped
    # — days counted the old way would share a norm with days counted the new way. Bump the version and add its
    # line to _PINNED (never edit an old line)
    got = _count(tmp_path, *_PIN_ROWS)
    assert got["talk_version"] == D.TALK_COUNT_VERSION
    assert D.TALK_COUNT_VERSION in _PINNED, "a new TALK_COUNT_VERSION needs its pinned counts"
    assert {k: v for k, v in got.items() if k != "talk_version"} == _PINNED[D.TALK_COUNT_VERSION]


def test_intent_mood_events_are_not_the_sentiments_tones(tmp_path):
    # catches: venting (the owner's own bad news) counted as tone_down — on a "counts" day it lowered warmth;
    # an intent cause matched by any prefix (a sentiment cause that merely starts with the same words is a tone)
    events = [{"at": "2026-11-13T09:00:00Z", "kind": "sentiment", "dp": -0.05, "de": -0.0125,
               "cause": I.CAUSE_VENTED},
              {"at": "2026-11-13T09:05:00Z", "kind": "sentiment", "dp": 0.05, "de": 0.0125,
               "cause": I.CAUSE_CONFIDED + " (in a background job)"},
              {"at": "2026-11-13T09:10:00Z", "kind": "sentiment", "dp": 0.08, "de": 0.02,
               "cause": "the owner sounded pleased"},
              {"at": "2026-11-13T09:15:00Z", "kind": "sentiment", "dp": 0.08, "de": 0.02,
               "cause": I.CAUSE_BANTER + "ly cheerful"},
              {"at": "2026-11-13T09:20:00Z", "kind": "praise", "dp": 0.1, "de": 0.0, "cause": "thanks"}]
    _journal(tmp_path, *events, name="mood.events.jsonl")
    rec = D.build_day(tmp_path, DAY, tz=UTC)
    assert (rec["tone_up"], rec["tone_down"]) == (2, 0)
    assert rec["events"] == 5                                   # they still moved the mood
    # counted as tones, the day's warmth would read 1+3−2 = 2, not 3
    assert C.day_signals(rec, intent=False)["warmth"] == 3.0


# ── the character ───────────────────────────────────────────────────────────

def _rec(i, **kw):
    base = {"date": (datetime.date(2026, 9, 1) + datetime.timedelta(days=i)).isoformat(), "thumbs_up": 0,
            "thumbs_down": 0, "corrections": 0, "praise": 0, "tone_up": 0, "tone_down": 0, "msgs_rated": 0,
            "msgs_warm": 0, "msgs_cold": 0, "msgs_thanks": 0, "own": {}, "shades": {}}
    base.update(kw)
    return base


def _talk(i, read=10, warm=2, cold=0, banter=1, emotions=None, version=None, **kw):
    """A covered intent day (every message read for intent; the words-only counts equal to the full ones unless
    given)."""
    got = dict(intent_rated=read, intent={k: 0 for k in I.LABELS}, urgent=0, talk_readable=read, talk_read=read,
               talk_emotions=read if emotions is None else emotions, talk_warm=warm, talk_thanks=0, talk_cold=cold,
               talk_banter=banter, talk_warm_words=warm, talk_thanks_words=0, talk_banter_words=banter,
               talk_version=D.TALK_COUNT_VERSION if version is None else version)
    got.update(kw)
    return _rec(i, **got)


def _strip(rec):
    return {k: v for k, v in rec.items() if k not in INTENT_KEYS}


TABLE = [
    _rec(0),
    _rec(1, praise=2, tone_down=1),
    _rec(2, msgs_rated=6, msgs_warm=2, msgs_cold=1, praise=1),
    _rec(3, own={"clean_turns": 5, "trouble": 1}, shades={"proud": 1}, thumbs_up=2, corrections=1),
    _talk(4, msgs_rated=8, msgs_warm=3, own={"clean_turns": 4}),
    _talk(5, read=2, warm=2, banter=2, praise=1, own={"clean_turns": 6}),
    _talk(6, read=12, warm=0, cold=3, banter=0, thumbs_down=1, own={"trouble": 4}),
]


def _with_intent(on):
    import os
    old = os.environ.get("GHOST_INTENT")
    os.environ["GHOST_INTENT"] = "1" if on else "journal"
    return old


@pytest.mark.parametrize("mode", ["0", "journal", None])
def test_switch_off_every_record_reads_as_before(monkeypatch, mode):
    # catches: the new keys read with the switch off; journal mode (the default) acting on the character
    if mode is not None:
        monkeypatch.setenv("GHOST_INTENT", mode)
    for rec in TABLE:
        assert C.day_signals(rec) == C.day_signals(_strip(rec), intent=False)
        assert C.signal_scales(rec) == C.signal_scales(_strip(rec), intent=False)
        for dial in C.DIALS:
            assert C._day_weight(rec, dial) == C._day_weight(_strip(rec), dial, intent=False)
    monkeypatch.setenv("GHOST_INTENT", "1")
    monkeypatch.setenv("GHOST_INTENT_CHARACTER", "0")
    assert C.signal_scales(TABLE[4]) == C.signal_scales(_strip(TABLE[4]), intent=False)


def test_switch_on_reads_the_new_scales_on_days_with_readings(monkeypatch):
    # catches: the new scales on a day without readings; a thin day given a number; work still read
    monkeypatch.setenv("GHOST_INTENT", "1")
    v = f"{D.TALK_COUNT_VERSION}/emotions"
    assert C.signal_scales(TABLE[2]) == C.signal_scales(TABLE[2], intent=False)
    assert C.signal_scales(TABLE[4]) == {"warmth": f"intent/{v}", "humour": f"banter/{v}", "directness": ""}
    s = C.day_signals(TABLE[4])
    assert s["warmth"] == pytest.approx(0.2) and s["humour"] == pytest.approx(0.1)
    # review r2 (MINOR 1): 2 messages read for intent is too few to cover a day — it reads the OLD scales, under
    # their old names (the praise count; the work's 6 clean turns), never the banter's name over the work
    assert C.signal_scales(TABLE[5]) == C.signal_scales(TABLE[5], intent=False) == {
        "warmth": "counts", "humour": "", "directness": ""}
    assert C.day_signals(TABLE[5]) == C.day_signals(TABLE[5], intent=False)
    s6 = C.day_signals(TABLE[6])
    assert s6["warmth"] == pytest.approx(-0.25) and s6["humour"] == 0.0          # an annoyed message counts once
    assert s6["directness"] == C.day_signals(TABLE[6], intent=False)["directness"]


def test_the_counts_are_clamped_to_the_messages_read():
    # catches: a damaged record (more warm or annoyed messages than read) reading past ±1; a count that is not a
    # whole number taken as an intent day
    assert C.day_signals(_talk(0, read=4, warm=9, cold=0), intent=True)["warmth"] == 1.0
    assert C.day_signals(_talk(0, read=4, warm=0, cold=9), intent=True)["warmth"] == -1.0
    assert C.day_signals(_talk(0, read=4, warm=0, cold=0, banter=9), intent=True)["humour"] == 1.0
    odd = dict(_talk(0), intent_rated="10")
    assert C.signal_scales(odd, intent=True) == C.signal_scales(odd, intent=False)


def test_a_thanks_counted_by_the_praise_check_and_the_reading_is_one_message():
    # catches: praise added on top of the thanks the reading already counted (counted twice)
    day = _talk(0, read=4, warm=1, banter=0, praise=1)
    day["talk_thanks"] = 1
    assert C.day_signals(day, intent=True)["warmth"] == pytest.approx(0.25)


# the reference: how each scale's number is computed, by its NAME — a scale name the code can return that is not
# here, or a number that is not computed the way its name says, fails
def _n(rec, k):
    v = rec.get(k)
    return max(0, int(v)) if isinstance(v, int) and not isinstance(v, bool) else 0


def _ref_covered(rec):
    rows, read = rec.get("talk_readable"), _n(rec, "talk_read")
    return (isinstance(rec.get("intent_rated"), int) and _n(rec, "intent_rated") > 0 and isinstance(rows, int)
            and read >= C.WARMTH_MIN_MSGS and 3 * read >= 2 * max(rows, read))


def _ref_variant(rec):
    v = rec.get("talk_version")
    v = v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0
    read, emo = _n(rec, "talk_read"), _n(rec, "talk_emotions")
    return f"{v}/" + ("emotions" if emo >= read else "words")


def _ref_intent_warmth(rec, sfx=""):
    extra = max(0, _n(rec, "praise") - _n(rec, "talk_thanks" + sfx))
    read = _n(rec, "talk_read") + extra
    warm = min(_n(rec, "talk_read"), _n(rec, "talk_warm" + sfx)) + extra
    cold = 0 if sfx else min(_n(rec, "talk_read"), _n(rec, "talk_cold"))
    return (warm - cold) / read if read >= C.WARMTH_MIN_MSGS else None


def _ref_banter(rec, sfx=""):
    read = _n(rec, "talk_read")
    return min(read, _n(rec, "talk_banter" + sfx)) / read if read >= C.WARMTH_MIN_MSGS else None


def _ref_emotions(rec):
    extra = max(0, _n(rec, "praise") - _n(rec, "msgs_thanks"))
    read = _n(rec, "msgs_rated") + extra
    return max(-2.0, min(1.0, (_n(rec, "msgs_warm") + extra - 2 * _n(rec, "msgs_cold")) / read)) \
        if read >= C.WARMTH_MIN_MSGS else None


def _ref_counts(rec):
    good, bad = _n(rec, "praise") + _n(rec, "tone_up"), _n(rec, "tone_down")
    return max(-C.WARMTH_CAP, min(C.WARMTH_CAP, float(good - 2 * bad))) if good + bad else None


def _ref_work(rec):
    own, sh = rec.get("own") or {}, rec.get("shades") or {}
    work = _n(own, "clean_turns") + _n(own, "trouble")
    return ((_n(own, "clean_turns") + _n(sh, "relieved") + _n(sh, "proud") - 2 * _n(own, "trouble")) / work
            if work >= C.WORK_MIN_TURNS else None)


#: keyed by (dial, the name without its counting version): review r2 (MINOR 2) — one name, one computation
SCALE_REFERENCE = {
    ("warmth", "emotions"): _ref_emotions, ("warmth", "counts"): _ref_counts,
    ("warmth", "intent/emotions"): _ref_intent_warmth,
    ("warmth", "intent/words"): lambda r: _ref_intent_warmth(r, "_words"),
    ("humour", ""): _ref_work, ("humour", "banter/emotions"): _ref_banter,
    ("humour", "banter/words"): lambda r: _ref_banter(r, "_words"),
}


def _unversioned(name):
    parts = str(name).split("/")
    return f"{parts[0]}/{parts[2]}" if len(parts) == 3 else name


def _fuzz_records(k=600):
    rng = random.Random(4)
    out = []
    for i in range(k):
        read = rng.choice([0, 1, 2, 3, 4, 6, 9, 15])
        rec = _rec(i, praise=rng.randint(0, 3), tone_up=rng.randint(0, 2), tone_down=rng.randint(0, 2),
                   msgs_rated=rng.choice([0, 0, 2, 3, 7]), msgs_warm=rng.randint(0, 4), msgs_cold=rng.randint(0, 2),
                   msgs_thanks=rng.randint(0, 2),
                   own={"clean_turns": rng.randint(0, 6), "trouble": rng.randint(0, 3)},
                   shades={"proud": rng.randint(0, 1)})
        if rng.random() < 0.7:
            rec.update(intent_rated=rng.choice([read, read, 0, 1]), talk_read=read,
                       talk_readable=rng.choice([read, read, read + 1, read + 2, 2 * read + 1, None]),
                       talk_emotions=rng.choice([read, 0, max(0, read - 1)]), talk_warm=rng.randint(0, read + 1),
                       talk_cold=rng.randint(0, 2), talk_thanks=rng.randint(0, 2), talk_banter=rng.randint(0, read + 1),
                       talk_warm_words=rng.randint(0, read + 1), talk_thanks_words=rng.randint(0, 2),
                       talk_banter_words=rng.randint(0, read + 1), talk_version=rng.choice([1, 2, None, "x"]))
        out.append(rec)
    return out


@pytest.mark.parametrize("on", [False, True])
def test_every_scale_name_says_how_its_number_was_computed(on):
    # MAJOR 2. catches: a fallback that returns another scale's number under this scale's name (a thin banter day
    # reading the work, a thin intent day the emotions), and a new scale name nobody decided how to compute
    seen = set()
    for rec in _fuzz_records():
        scales, sig = C.signal_scales(rec, intent=on), C.day_signals(rec, intent=on)
        for dial in ("warmth", "humour"):
            name = scales[dial]
            base = name.split("/")[0]
            key = (dial, _unversioned(name))
            assert key in SCALE_REFERENCE, name
            # MINOR 1: the intent scales exactly on a day the readings cover
            assert (base in ("intent", "banter")) == (on and _ref_covered(rec)), (name, rec)
            if base in ("intent", "banter"):
                assert name == f"{base}/{_ref_variant(rec)}"
            want = SCALE_REFERENCE[key](rec)
            assert sig[dial] == (pytest.approx(want) if want is not None else None), (dial, name, rec)
            seen.add(key)
    assert seen == ({k for k in SCALE_REFERENCE if "/" not in k[1]} if not on else set(SCALE_REFERENCE))


def test_a_day_without_emotions_or_on_another_count_version_starts_its_own_norm():
    # MAJOR 2. catches: an emotion-model outage (warm only from words, nothing annoyed) judged against days read
    # with emotions — a warm owner read as colder; days counted under old witness rules in the new rules' norm
    emo = [_talk(i, read=10, warm=4, cold=1) for i in range(21)]
    words = [_talk(21 + i, read=10, warm=1, cold=0, emotions=0) for i in range(6)]
    shifts = C.day_shifts(emo + words, intent=True)
    assert not any("warmth" in shifts.get(d["date"], {}) for d in words[:C.BASE_MIN])
    assert C.signal_scales(words[0], intent=True)["warmth"].endswith("/words")
    assert C.signal_scales(_talk(0, emotions=5), intent=True)["warmth"].endswith("/words")    # MINOR 2: no "mixed"
    bumped = [_talk(21 + i, read=10, warm=1, cold=0, version=D.TALK_COUNT_VERSION + 1) for i in range(6)]
    shifts = C.day_shifts(emo + bumped, intent=True)
    assert not any("warmth" in shifts.get(d["date"], {}) for d in bumped[:C.BASE_MIN])
    state = {"dials": {"warmth": 1, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    old = _with_intent(True)
    try:
        assert [m for m in C.propose(emo + words, state, [], words[-1]["date"]) if m["dial"] == "warmth"] == []
    finally:
        import os
        os.environ.pop("GHOST_INTENT", None) if old is None else os.environ.__setitem__("GHOST_INTENT", old)


def test_norms_never_mix_scales():
    # catches: an intent day judged against emotion days (warmth), a banter day against work days (humour). The
    # old days read warmth 0.8 and humour 1.0; the new days a steady 0.2 and 0.1. Mixed, every new day would read
    # far below and pull both down
    old = [_rec(i, msgs_rated=5, msgs_warm=4, own={"clean_turns": 5}) for i in range(21)]
    new = [_talk(21 + i, read=10, warm=2, banter=1) for i in range(8)]
    days = old + new
    pulls = C.day_pulls(days, intent=True)
    assert not any(pulls.get(d["date"]) for d in new)
    shifts = C.day_shifts(days, intent=True)
    assert not any(d["date"] in shifts for d in new[:C.BASE_MIN])
    assert shifts[new[C.BASE_MIN]["date"]] == {"warmth": 0.0, "humour": 0.0}
    state = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    old_env = _with_intent(True)
    try:
        assert C.propose(days, state, [], days[-1]["date"]) == []
    finally:
        import os
        os.environ.pop("GHOST_INTENT", None) if old_env is None else os.environ.__setitem__("GHOST_INTENT", old_env)


def test_no_dial_moves_on_a_new_scale_before_base_min_days(monkeypatch):
    # catches: a norm of fewer than BASE_MIN days on the new scale (a dial moved on its first intent week)
    monkeypatch.setenv("GHOST_INTENT", "1")
    days = [_talk(i, banter=0) for i in range(C.BASE_MIN)] + [_talk(C.BASE_MIN + i, banter=6) for i in range(3)]
    state = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    early = days[:C.BASE_MIN]
    assert C.day_shifts(early) == {}
    assert C.propose(early + [_talk(C.BASE_MIN - 1 + 50, banter=6)], state, [], "2026-12-31") == []
    moved = C.propose(days, state, [], days[-1]["date"])
    assert [m["dial"] for m in moved] == ["humour"] and moved[0]["to"] == 1


def test_the_switch_flip_rereads_recorded_days_on_the_new_scale():
    # decided: the record holds the counts whatever the switch said; the scale is chosen when it is READ. Ten
    # days recorded with the switch off read the old scales then, and the new ones once it is on — the norm on
    # the new scale is there at once (days 6–10 have shifts), and a dial's `changed` stamp still bounds the
    # evidence. catches: a flip that leaves the recorded days unreadable for three weeks, or reads them twice
    days = [_talk(i, msgs_rated=6, msgs_warm=2, banter=1) for i in range(10)]
    assert {C.signal_scales(d, intent=False)["warmth"] for d in days} == {"emotions"}
    assert {C.signal_scales(d, intent=True)["warmth"] for d in days} == {f"intent/{D.TALK_COUNT_VERSION}/emotions"}
    on = C.day_shifts(days, intent=True)
    assert sorted(on) == [d["date"] for d in days[C.BASE_MIN:]]
    assert all(set(v) >= {"warmth", "humour"} for v in on.values())


def test_one_message_never_makes_a_clear_day():
    # MINOR 3. catches: one message making a day CLEARLY off its usual (a pull, one of the two a move needs) —
    # against the WORST norm (identical days, no spread but the floor) at EVERY share and day size: an annoyed
    # message added to an all-warm day moves warmth by (1 + v)/(n + 1), a warm one read as annoyed by 2/n. The
    # shift itself may reach (1/3)/0.3 = 1.11 spreads (warmth) and (1/6)/0.15 = 1.11 (banter): stated, pinned
    worst = {"warmth": 0.0, "humour": 0.0}
    pulled = []
    for n in range(C.WARMTH_MIN_MSGS, 31):
        for w, c in itertools.product(range(n + 1), range(3)):
            if w + c > n:
                continue
            norm = [_talk(i, read=n, warm=w, cold=c, banter=w) for i in range(C.BASE_DAYS)]
            days = [_talk(C.BASE_DAYS, read=n + 1, warm=w, cold=c + 1, banter=w),            # one more annoyed
                    _talk(C.BASE_DAYS, read=n + 1, warm=w + 1, cold=c, banter=w + 1),        # one more warm / joke
                    _talk(C.BASE_DAYS, read=n + 1, warm=w, cold=c, banter=w)]               # one more plain
            if w:
                days.append(_talk(C.BASE_DAYS, read=n, warm=w - 1, cold=c + 1, banter=w - 1))  # warm → annoyed
            if w < n - c:
                days.append(_talk(C.BASE_DAYS, read=n, warm=w + 1, cold=c, banter=w + 1))     # plain → warm / joke
            for day in days:
                for dial, z in (C.day_shifts(norm + [day], intent=True).get(day["date"]) or {}).items():
                    worst[dial] = max(worst[dial], abs(z))
                if C.day_pulls(norm + [day], intent=True):
                    pulled.append((n, w, c, day["talk_read"], day["talk_warm"], day["talk_cold"]))
    assert pulled == []
    assert worst["warmth"] == pytest.approx((1 / 3) / C.SPREAD_FLOOR["warmth"]["intent"])
    assert worst["humour"] == pytest.approx((1 / 6) / C.SPREAD_FLOOR["humour"]["banter"])


def test_two_messages_off_the_usual_make_a_clear_day():
    # catches: the one-message rule set so high that no thin day can ever pull (the dial frozen on 3–6 messages)
    norm = [_talk(i, read=4, warm=1, cold=0, banter=0) for i in range(C.BASE_DAYS)]
    jokes = _talk(C.BASE_DAYS, read=4, warm=1, cold=0, banter=2)
    warmer = _talk(C.BASE_DAYS, read=4, warm=4, cold=0, banter=0)
    assert C.day_pulls(norm + [jokes], intent=True) == {jokes["date"]: {"humour": 1}}
    assert C.day_pulls(norm + [warmer], intent=True) == {warmer["date"]: {"warmth": 1}}


def test_one_joke_from_an_owner_who_never_jokes_moves_nothing():
    # catches: norm 0, spread 0 → the scale-free fallback (one joke = a huge shift); a move carried by one day
    days = [_talk(i, banter=0) for i in range(40)]
    days[30] = _talk(30, banter=1)
    state = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    old = _with_intent(True)
    try:
        for k in range(31, 41):
            assert C.propose(days[:k], state, [], days[k - 1]["date"]) == []
    finally:
        import os
        os.environ.pop("GHOST_INTENT", None) if old is None else os.environ.__setitem__("GHOST_INTENT", old)


def test_a_thin_day_counts_in_proportion_to_its_messages():
    # catches: a 3-message day carrying as much as a full one; the warmth weight not counting the praise check's
    # thanks the reading missed (they are messages read too)
    full_n = C.INTENT_FULL_MSGS
    assert C._day_weight(_talk(0, read=3), "humour", intent=True) == pytest.approx(3 / full_n)
    assert C._day_weight(_talk(0, read=3), "warmth", intent=True) == pytest.approx(3 / full_n)
    assert C._day_weight(_talk(0, read=3, praise=2), "warmth", intent=True) == pytest.approx(5 / full_n)
    assert C._day_weight(_talk(0, read=3, praise=2), "humour", intent=True) == pytest.approx(3 / full_n)
    assert C._day_weight(_talk(0, read=full_n + 5), "humour", intent=True) == 1.0
    assert C._day_weight(_talk(0, read=3), "directness", intent=True) == 1.0
    # the same share (a third joking) on a thin day and a full one: the thin one carries its share of messages
    norm = [_talk(i, read=full_n, banter=0) for i in range(10)]
    thin = C.day_shifts(norm + [_talk(10, read=3, banter=1)], intent=True)[_talk(10)["date"]]["humour"]
    full = C.day_shifts(norm + [_talk(10, read=2 * full_n, banter=2 * full_n // 3)],
                        intent=True)[_talk(10)["date"]]["humour"]
    assert 0 < thin == pytest.approx(full * 3 / full_n)


def test_the_view_says_what_humour_listens_to(tmp_path, monkeypatch):
    # catches: the view still saying only "my own work" when the joking is read (and the other way round), or
    # claiming the joking is all it reads
    assert "how my own work goes" in C.view(tmp_path) and "how often you joke" not in C.view(tmp_path)
    monkeypatch.setenv("GHOST_INTENT", "1")
    v = C.view(tmp_path)
    assert "how often you joke with me" in v and "how my own work goes on a day" in v


def test_a_moves_words_say_what_its_days_were_read_on(monkeypatch):
    # MINOR 6. catches: a warmth move with the switch OFF saying it read "confiding" (not read then), and an
    # intent move saying it read the work
    assert C.signal_words("warmth", ["emotions", "counts"]) == C.SIGNAL_WORDS["warmth"]
    assert "confiding" not in C.SIGNAL_WORDS["warmth"]
    assert C.signal_words("humour", ["banter/1/emotions"]) == C.INTENT_SIGNAL_WORDS["humour"]
    mixed = C.signal_words("warmth", ["emotions", "intent/1/emotions"])
    assert C.SIGNAL_WORDS["warmth"] in mixed and C.INTENT_SIGNAL_WORDS["warmth"] in mixed
    assert C.signal_words("directness", [""]) == C.SIGNAL_WORDS["directness"]
    monkeypatch.setenv("GHOST_INTENT", "1")
    days = [_talk(i, banter=0) for i in range(C.BASE_MIN)] + [_talk(C.BASE_MIN + i, banter=6) for i in range(3)]
    state = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    (mv,) = C.propose(days, state, [], days[-1]["date"])
    assert mv["why"].endswith("in " + C.INTENT_SIGNAL_WORDS["humour"])


def test_after_the_flip_a_dials_last_change_still_bounds_the_evidence(monkeypatch):
    # catches: re-read days from before the dial's last change counted again once the scale changes
    monkeypatch.setenv("GHOST_INTENT", "1")
    days = [_talk(i, banter=0) for i in range(C.BASE_MIN)] + [_talk(C.BASE_MIN + i, banter=6) for i in range(4)]
    fresh = {"dials": {"warmth": 0, "humour": 0, "directness": 0}, "pins": [], "changed": {}}
    assert [m["dial"] for m in C.propose(days, fresh, [], days[-1]["date"])] == ["humour"]
    moved_late = dict(fresh, changed={"humour": days[-3]["date"]})        # only two days after it
    assert C.propose(days, moved_late, [], days[-1]["date"]) == []


def test_the_log_line_says_a_thin_intent_day_counts_less(monkeypatch):
    # catches: the log line's weight read on the old scale (the work turns) on a day read by its messages
    monkeypatch.setenv("GHOST_INTENT", "1")
    days = [_talk(i) for i in range(C.BASE_MIN)] + [_talk(C.BASE_MIN, read=4, warm=1, banter=1)]
    line = C.pull_line(days, days[-1]["date"])
    assert f"a thin day: counts {4 / C.INTENT_FULL_MSGS:.1f}" in line
    monkeypatch.setenv("GHOST_INTENT", "journal")
    assert "thin day" not in C.pull_line(days, days[-1]["date"])


# ── review r2 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("read,rows,covered", [(3, 3, True), (3, 4, True), (4, 5, True), (3, 5, False),
                                               (4, 6, True), (3, 6, False), (2, 2, False), (6, 9, True),
                                               (5, 8, False)])
def test_a_day_reads_the_intent_scales_only_when_its_readings_cover_it(read, rows, covered):
    # MINOR 1. catches: a day read on the intent scales from a few of its messages (the intent model out for the
    # rest of the day), the coverage share or the message minimum off by one — and a day under the cover read on
    # anything but the old scales under the old names
    rec = _talk(0, read=read, talk_readable=rows, msgs_rated=5, msgs_warm=2, own={"clean_turns": 5})
    old = C.signal_scales(rec, intent=False)
    if covered:
        assert C.signal_scales(rec, intent=True)["warmth"].startswith("intent/")
        assert C.signal_scales(rec, intent=True)["humour"].startswith("banter/")
    else:
        assert C.signal_scales(rec, intent=True) == old == {"warmth": "emotions", "humour": "", "directness": ""}
        assert C.day_signals(rec, intent=True) == C.day_signals(rec, intent=False)


def test_the_day_counts_every_owner_message_for_the_cover(tmp_path):
    # MINOR 1. catches: ``talk_readable`` counting only the rows with a reading (every day "covered"), or counting a
    # task's wording (not the owner's)
    # review r3: … or counting a message the service's language gate refused (label "unknown": never an intent)
    got = _count(tmp_path, _row(_at(0), "question"), _row(_at(1), "question"), _row(_at(2), "question"),
                 _row(_at(3)), _row(_at(4)),
                 _row(_at(5), "instruction", held=SE.HELD_TASK), _row(_at(6), held=SE.HELD_TASK),
                 _row(_at(7), text="ok thanks", sent="unknown", held="not English"))
    assert (got["talk_readable"], got["talk_read"]) == (5, 3)
    assert not C._has_intent({**got, "date": DAY.isoformat()})
    assert C._has_intent({**got, "talk_readable": 4, "date": DAY.isoformat()})


def _strip_emotions(rows):
    return [{k: v for k, v in r.items() if k != "emotions"} for r in rows]


#: rows of every kind the counts read, with and without emotions
_COVER_ROWS = [
    _row(_at(0), "banter", text="haha nice one", emotions={"amusement": 0.9}, effect=I.CAUSE_BANTER),
    _row(_at(1), "banter", text="haha ok", emotions={"annoyance": 0.8}, sent="positive", effect="", iheld=I.WHY_COLD),
    _row(_at(2), "confiding", text="I feel worn out about my sister's move", emotions={"sadness": 0.7}),
    _row(_at(3), "question", text="great job, what next?", emotions={"admiration": 0.9}),
    _row(_at(4), "instruction", text="run it", emotions={"annoyance": 0.9}),
    _row(_at(5), "thanks", text="that fixed it", emotions={"gratitude": 0.9}),
]


@pytest.mark.parametrize("pattern", list(itertools.product([True, False], repeat=len(_COVER_ROWS))))
def test_two_days_with_the_same_scale_name_are_computed_the_same_way(tmp_path, pattern):
    # MINOR 2. catches: a third bucket (some messages read with emotions, some without, in one number), or a
    # "words" day that reads the emotions its messages happened to have. For every pattern of which messages
    # the emotion model read: a day named "words" reads exactly what the same day reads with NO emotions at all;
    # a day named "emotions" had every message read by it
    rows = [r if keep else _strip_emotions([r])[0] for r, keep in zip(_COVER_ROWS, pattern)]
    rec = {**_count(tmp_path / "a", *rows), "date": DAY.isoformat(), "praise": 1}
    bare = {**_count(tmp_path / "b", *_strip_emotions(rows)), "date": DAY.isoformat(), "praise": 1}
    scales, sig = C.signal_scales(rec, intent=True), C.day_signals(rec, intent=True)
    assert {scales["warmth"].split("/")[-1], scales["humour"].split("/")[-1]} == \
        {"emotions" if all(pattern) else "words"}
    if not all(pattern):
        assert scales == C.signal_scales(bare, intent=True)
        assert (sig["warmth"], sig["humour"]) == (C.day_signals(bare, intent=True)["warmth"],
                                                  C.day_signals(bare, intent=True)["humour"])


def test_the_records_keys_stay_after_the_switch_is_turned_off(tmp_path, monkeypatch):
    # MINOR 4 (decided: kept). catches: a day closed after GHOST_INTENT=0 dropping the readings the journal mode
    # made that day — they were made; the character reads them only while its switch is on
    _journal(tmp_path, _row(_at(0), "question"), _row(_at(1), "banter", text="haha"), _row(_at(2), "advice"))
    monkeypatch.setenv("GHOST_INTENT", "0")
    rec = D.build_day(tmp_path, DAY, tz=UTC)
    assert set(rec) == OLD_KEYS | INTENT_KEYS and rec["intent_rated"] == 3
    assert C.signal_scales(rec) == C.signal_scales(_strip(rec), intent=False)


def test_the_one_message_rule_counts_the_praise_checks_thanks_as_messages():
    # MINOR 5. catches: the one-message rule reading ``talk_read`` for warmth — the praise check's thanks the
    # reading missed are messages read too: 3 read + 2 such thanks, warm 1 + 2 → 0.6 over 5 messages = 3
    # messages off a usual 0 (clear); over 3 it would be 1.8, under one message's reach
    norm = [_talk(i, read=4, warm=0, cold=0, banter=0) for i in range(C.BASE_DAYS)]
    day = _talk(C.BASE_DAYS, read=3, warm=1, cold=0, banter=0, praise=2)
    assert C.day_signals(day, intent=True)["warmth"] == pytest.approx(0.6)
    assert C.day_pulls(norm + [day], intent=True) == {day["date"]: {"warmth": 1}}


def _gate(minute, text="ok thanks"):
    """A message the service's language gate refused: sentiment "unknown", never an intent reading."""
    return _row(_at(minute), text=text, sent="unknown", held="not English")


@pytest.mark.parametrize("rows,new", [
    ([_row(_at(i), "question") for i in range(4)] + [_gate(10 + i) for i in range(3)], True),
    ([_row(_at(i), "question") for i in range(2)] + [_row(_at(10 + i)) for i in range(4)], False),
    ([_row(_at(i), "question") for i in range(3)] + [_row(_at(10 + i)) for i in range(2)], False),
    ([_row(_at(i), "question") for i in range(3)] + [_row(_at(10))] + [_gate(20 + i) for i in range(5)], True),
    ([_row(_at(i), "question") for i in range(2)] + [_gate(10 + i) for i in range(4)], False),
    ([_row(_at(i), "question") for i in range(2)] + [_row(_at(10))], False),      # 2 of 3 covered, but 2 read
])
def test_the_cover_counts_the_messages_the_service_could_read(tmp_path, rows, new):
    # review r3 (1). catches: the gate's refusals ("great", "do it", "ok thanks" — never an intent, by design)
    # counted as unread messages, so a day of short messages falls back to the old scales; the minimum counting
    # unread or refused messages instead of the ones read (2 read is never enough, whatever else the day held)
    rec = {**_count(tmp_path, *rows), "date": DAY.isoformat(), "own": {"clean_turns": 5}}
    scales = C.signal_scales(rec, intent=True)
    assert scales["humour"].startswith("banter/") is new and scales["warmth"].startswith("intent/") is new
    if not new:
        assert scales == C.signal_scales(rec, intent=False)


@pytest.mark.parametrize("emotions", [None, 0])
@pytest.mark.parametrize("read", [3, 4])
def test_humour_has_a_number_from_three_messages_on_either_count(read, emotions):
    # review r3 (2). catches: the banter scale's floor of WARMTH_MIN_MSGS messages moved (on the emotion counts
    # or on the words-only counts): a 3-message day is the owner's usual and must have its number
    rec = _talk(0, read=read, banter=2, emotions=emotions, talk_banter_words=1)
    name = C.signal_scales(rec, intent=True)["humour"]
    assert name.endswith("/emotions" if emotions is None else "/words")
    assert C.day_signals(rec, intent=True)["humour"] == pytest.approx((2 if emotions is None else 1) / read)
    assert C.day_signals(rec, intent=True)["warmth"] is not None


@pytest.mark.parametrize("emotions", [None, {"neutral": 0.9}, {"amusement": 0.9}])
def test_a_joke_doubted_after_the_agents_trouble_counts_alike_whatever_the_emotions(tmp_path, emotions):
    # review r3. catches: the words-only count re-reading a joke the live path doubted right after the agent's
    # trouble (only the live path saw it: the row's ``intent_held`` carries the doubt) — with no emotion
    # backing the tone, the backing's reason hid the doubt and the words count read a joke (0 / 1 / 0 by reading)
    text = "lol nice one"
    reading = {"label": "positive", "intent": _dist("banter"), **({"emotions": emotions} if emotions else {})}
    ev, why = I.mood_event(reading, text, after_trouble=True)
    assert ev is None and why == SE.HELD_SARCASM
    got = _count(tmp_path, _row(_at(0), "banter", text=text, emotions=emotions, sent="positive", effect="",
                                iheld=why))
    assert (got["talk_banter"], got["talk_banter_words"], got["talk_warm_words"]) == (0, 0, 0)



# ── §4VE functional run 11-21: what may make a message warm ─────────────────

def _tones(home, *rows):
    _journal(home, *rows)
    return D.owner_tones(home, DAY, tz=UTC)


#: (sentiment label, emotions, text, warm on both scales, why)
_WARM_CASES = [
    ("negative", {"admiration": 0.93, "approval": 0.05}, "great, you broke it again", False,
     "the real service's reading of a sarcastic insult: about trouble, and a warm emotion under a negative tone"),
    ("positive", {"love": 0.74, "admiration": 0.45}, "Oh wonderful, another error. Love that for me.", False,
     "upbeat words about trouble under a positive tone — below the mood's gate, so nothing held it"),
    ("positive", {"admiration": 0.95}, "Great job on the summary, really clear.", True, "the control"),
    ("negative", {"sadness": 0.7, "gratitude": 0.8}, "I'm so sad but thank you for staying with me", True,
     "thanks under a negative tone still count (gratitude)"),
    ("negative", {"amusement": 0.7}, "hehe you'd make a terrible weatherman", False,
     "a warm emotion under a negative tone: the tone and the emotion disagree"),
    # review 11-21: thanks about trouble under a neutral tone count (bugs are the owner's subject matter); under
    # version 2's first draft this row was NOT warm — the refinement changed it on purpose
    ("neutral", {"gratitude": 0.9}, "thanks, but it crashed again", True,
     "thanks about trouble that is not gone, under a neutral tone: the thanks count"),
    ("positive", {"gratitude": 0.9}, "thanks, you fixed the bug", True, "trouble gone: the thanks count"),
]


@pytest.mark.parametrize("label,emotions,text,warm,why", _WARM_CASES)
def test_a_warm_emotion_counts_only_where_the_reading_agrees_with_itself(tmp_path, label, emotions, text, warm, why):
    # §4VE functional run. catches: "great, you broke it again" (negative 0.72, admiration 0.93 from the real
    # service) counted WARM on either warmth scale — the old one (``owner_tones``: msgs_warm) or the new one
    # (``owner_intents``: talk_warm); a thanks under a negative tone lost; the rule copied instead of the
    # sentiment module's own trouble test
    rated, w, cold, thanks = _tones(tmp_path / "old", _row(_at(0), "question", text=text, emotions=emotions,
                                                            sent=label))
    assert (rated, w) == (1, 1 if warm else 0), why
    got = _count(tmp_path / "new", _row(_at(0), "question", text=text, emotions=emotions, sent=label))
    assert got["talk_warm"] == (1 if warm else 0), why
    assert got["talk_cold"] == cold                           # the annoyed count is untouched by the rule


# ── §4VE review 11-21: thanks about trouble are warm, unless the tone says complaint ─────────────

@pytest.mark.parametrize("label,trouble,routes", [
    ("negative", True, (False, False)), ("negative", False, (False, True)),
    ("positive", True, (False, True)), ("neutral", True, (False, True)),
    ("positive", False, (True, True)), ("neutral", False, (True, True))])
def test_the_warm_routes_by_tone_and_trouble(label, trouble, routes):
    # catches: trouble closing the thanks under a positive/neutral tone (the first draft); trouble under a
    # negative tone leaving the thanks open; the warm emotion reopened on a message about trouble
    text = "you broke the build" if trouble else "you wrote the summary"
    assert D._warm_routes(label, text) == routes


#: (case, sentiment label, emotions, text, the row's ``held`` as the live path would write it, warm on the old
#: scale ``owner_tones``, warm on the intent scale ``owner_intents``). CONSTRUCTED readings, not the service's
#: (the scores the brief gives; the live path's hold derived from them: a positive tone at 0.85+ backed by an
#: emotion and not the praise check is held as sarcasm about trouble — ``sentiment._feed`` / ``doubtful``)
_ROUTE_CASES = [
    ("a", "negative", {"admiration": 0.93}, "great, you broke it again", "below 0.85", 0, 0),
    ("b", "positive", {"admiration": 0.8}, "Oh wonderful, another error. Love that for me.", "below 0.85", 0, 0),
    # positive 0.9 + gratitude: a row nothing held is warm on the old scale; the intent scale refuses it
    # (``intent.refusal`` runs ``doubtful`` on the positive tone: trouble words) — and on the live path the row
    # IS held as sarcasm (c-live): neither scale
    ("c", "positive", {"gratitude": 0.9}, "thank you for catching that bug!", "", 1, 0),
    ("c-live", "positive", {"gratitude": 0.9}, "thank you for catching that bug!", SE.HELD_SARCASM, 0, 0),
    ("c-neutral", "neutral", {"gratitude": 0.9}, "thank you for catching that bug!", "neutral", 1, 1),
    ("d", "positive", {"gratitude": 0.9}, "you saved me from a huge mess, thank you!!", "", 1, 0),
    ("d-neutral", "neutral", {"gratitude": 0.9}, "you saved me from a huge mess, thank you!!", "neutral", 1, 1),
    ("e", "negative", {"gratitude": 0.9}, "thank you for catching that bug!", "", 0, 0),
    ("f", "negative", {"sadness": 0.7, "gratitude": 0.8}, "I'm so sad but thank you for staying with me", "", 1, 1),
    # the ACCEPTED COST: praise of the work about trouble, admiration and no thanks
    ("g", "positive", {"admiration": 0.9}, "great debugging, the error was subtle", "", 0, 0),
]


@pytest.mark.parametrize("case,label,emotions,text,held,old,new", _ROUTE_CASES, ids=[c[0] for c in _ROUTE_CASES])
def test_thanks_about_trouble_are_warm_unless_the_tone_says_complaint(tmp_path, case, label, emotions, text, held,
                                                                      old, new):
    # §4VE review 11-21. catches: trouble closing the thanks route under a positive or neutral tone (the first
    # draft: c, c-neutral, d, d-neutral lost); trouble under a NEGATIVE tone leaving the thanks open (e warm); the
    # warm-emotion route reopened on a message about trouble (b, g warm); a negative tone alone closing the
    # thanks (f lost); the sarcastic insult warm again (a)
    for il in ("thanks", "question"):           # what the intent model read does not decide warmth
        row = _row(_at(0), il, text=text, emotions=emotions, sent=label, held=held)
        rated, w, cold, thanks = _tones(tmp_path / f"old-{il}", row)
        assert (rated, w, thanks) == (1, old, old), case
        got = _count(tmp_path / f"new-{il}", row)
        assert (got["talk_warm"], got["talk_thanks"]) == (new, new), case


@pytest.mark.parametrize("case,label,emotions,text,held,old,new", _ROUTE_CASES, ids=[c[0] for c in _ROUTE_CASES])
def test_the_warm_routes_never_change_the_annoyed_count(tmp_path, case, label, emotions, text, held, old, new):
    # catches: the refinement reaching the cold count — a closed (or reopened) route dropping or adding an
    # annoyed message on either scale
    for annoyed in (False, True):
        emo = {**emotions, "annoyance": 0.7} if annoyed else emotions
        row = _row(_at(0), "question", text=text, emotions=emo, sent=label, held=held)
        assert _tones(tmp_path / f"old-{annoyed}", row)[2] == int(annoyed), case
        assert _count(tmp_path / f"new-{annoyed}", row)["talk_cold"] == int(annoyed), case


#: (h) the KNOWN LEAK, pinned: sarcastic thanks about trouble, gratitude 0.6. (row held, old scale, intent scale)
_SARCASTIC_THANKS = [
    # positive 0.7: under the mood's gate (0.85), so the live path never ran ``doubtful`` and held nothing —
    # the old scale counts it warm; the intent scale's ``intent.refusal`` doubts the positive tone: not warm
    ("positive", "below 0.85", 1, 0),
    # positive at 0.85+ (gratitude backs it): the live path holds it as sarcasm — neither scale
    ("positive", SE.HELD_SARCASM, 0, 0),
    # neutral: nothing doubts a neutral tone — warm on both scales
    ("neutral", "neutral", 1, 1),
]


@pytest.mark.parametrize("label,held,old,new", _SARCASTIC_THANKS)
def test_sarcastic_thanks_about_trouble_is_the_known_leak(tmp_path, label, held, old, new):
    # §4VE review 11-21: PINNED so a change to it is deliberate. catches: a new rule (or a change to the live
    # path's hold, ``intent.refusal`` or ``_warm_routes``) moving what "thanks for breaking the build again"
    # counts without this table being updated
    row = _row(_at(0), "thanks", text="thanks for breaking the build again", emotions={"gratitude": 0.6},
               sent=label, held=held)
    assert _tones(tmp_path / "old", row)[1] == old
    assert _count(tmp_path / "new", row)["talk_warm"] == new
