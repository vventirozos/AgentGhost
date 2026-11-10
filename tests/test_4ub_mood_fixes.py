"""§4UB — the fixes from the 11-10 end-to-end evaluation of the mood / tone / Mirror / sentiment system.

What the evaluation found (live journals, 84 synthetic messages scored against the live model, code reading):

* the automatic mood was a thumbs-up counter: most owner turns use 0–1 tool calls and earned nothing;
* the owner's thumbs fed the Mirror's "lasting mood" note — the owner's reactions read as the agent's character;
* sentiment misread sarcasm ("great, you broke it again" +0.94), task wording ("delete the failed jobs" −0.87),
  pasted logs and articles, and capitals ("WHY IS THIS STILL BROKEN": "language unclear"); its outage was
  logged at DEBUG only; three upbeat messages alone made a heard mood;
* corrections never fired in Greek; a member's labelled row could reach the correction judge; queued
  messages were never read for the mood;
* a member heard "MANNER (set by the owner)" with no rule for "why"; scheduled reports remarked on the mood
  every run; a mood that went neutral mid-request stayed in the turn record;
* a thumb during the owner's hold was dropped without a word; a Mirror look with only strong-mood replies
  said "no reply of yours"; project tasks a job closed never counted.

All names below are fictional.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import feedback as F
from ghost_agent.distill.user_correction import has_correction_phrase, is_correction_candidate
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mirror as MI
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import sentiment as C
from ghost_agent.selfhood.state import MOOD_HISTORY_FILENAME
from ghost_agent.utils.logging import FOREIGN_MESSAGE_LABELS
from tests.test_4tc_mood_hooks import _labelled
from tests.test_4tx_mood_review import _ctx, _req

T0 = dt.datetime(2026, 11, 6, 9, 0, tzinfo=dt.timezone.utc)


def _at(minutes: float = 0.0, hours: float = 0.0) -> dt.datetime:
    return T0 + dt.timedelta(hours=hours, minutes=minutes)


def _sm(tmp_path):
    return SelfModel(root=tmp_path / "self")


@pytest.fixture(autouse=True)
def _fresh():
    A._MOOD_HEARD_BY_CONV.clear()
    A._MOOD_TONE_BY_REQ.clear()
    C.BREAKER.reset()
    yield
    A._MOOD_HEARD_BY_CONV.clear()
    A._MOOD_TONE_BY_REQ.clear()
    C.BREAKER.reset()


# ── 1. small turns count; the automatic mood still has no cap ───────────────

@pytest.mark.parametrize("calls,dp", [(1, 0.03), (2, 0.04)])
def test_a_clean_turn_under_three_calls_earns_half(calls, dp):
    assert M.CLEAN_TURN_SMALL_SHARE == 0.5
    evs = M.turn_events(M.TurnSignals(tool_calls=calls))
    assert [e.dp for e in evs] == [pytest.approx(dp)]


def test_one_call_is_said_in_the_singular():
    assert M.turn_events(M.TurnSignals(tool_calls=1))[0].cause == \
        "a turn of 1 tool call went through without a failure"


@pytest.mark.parametrize("sig", [
    M.TurnSignals(tool_calls=3, excused=2),            # short: 1 counted call, 2 excused
    M.TurnSignals(tool_calls=30, excused=29),          # long: a slog
])
def test_a_turn_of_mostly_excused_calls_earns_nothing(sig):
    # fails if the small-turn reward pays the one counted call of a turn of Tor timeouts
    assert all(e.dp <= 0 for e in M.turn_events(sig))


def test_a_plain_chat_turn_still_earns_nothing():
    assert M.turn_events(M.TurnSignals(tool_calls=0)) == []


def test_the_automatic_mood_still_reaches_three(tmp_path):
    """The operator's design (11-06): no cap — the evaluation's "limit own turns to level 2" was not built."""
    sm = _sm(tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=6), now=T0)
    after = sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=6), now=_at(minutes=1))
    assert (after.label, after.intensity) == ("irritable", 3)
    assert not hasattr(M, "TURN_LEVEL_LIMIT")


# ── 2. sentiment's share of the mood ────────────────────────────────────────

def test_the_owners_tone_alone_never_makes_a_heard_mood(tmp_path):
    sm = _sm(tmp_path)
    # review r3: clearly under the hold-down of level 1 (0.20 − 0.04), not on it
    assert M.SENTIMENT_SHARE_MAX == 0.12 < M.INTENSITY_THRESHOLDS[0] - M.INTENSITY_MARGIN
    for i in range(5):
        sm.note_owner_sentiment("positive", 0.97, request_id=f"m{i}", now=_at(minutes=i))
    v = sm.mood(now=_at(minutes=5))
    assert v.p == pytest.approx(0.12, abs=1e-3) and v.intensity == 0


def test_a_full_share_is_held_and_says_why(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_sentiment("positive", 0.97, request_id="m1", now=T0)
    sm.note_owner_sentiment("positive", 0.97, request_id="m2", now=T0)
    after, dp, why = sm.note_owner_sentiment("positive", 0.97, request_id="m3", now=T0)
    assert (after, dp, why) == (None, 0.0, sm.HELD_SENTIMENT_FULL)


def test_the_share_comes_back_as_it_decays(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_sentiment("positive", 0.97, request_id="m1", now=T0)
    sm.note_owner_sentiment("positive", 0.97, request_id="m2", now=T0)
    _a, dp, why = sm.note_owner_sentiment("positive", 0.97, request_id="m3", now=_at(hours=2))
    assert why == "" and dp == pytest.approx(0.06, abs=1e-3)       # 0.12 halved to 0.06: room for 0.06


def test_the_other_side_has_its_own_share(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_sentiment("positive", 0.97, request_id="m1", now=T0)
    sm.note_owner_sentiment("positive", 0.97, request_id="m2", now=T0)
    _a, dp, why = sm.note_owner_sentiment("negative", 0.97, request_id="m3", now=T0)
    assert why == "" and dp == pytest.approx(-0.08)


# ── 3. what is sent to the sentiment service ────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("WHY IS THIS STILL BROKEN", "why is this still broken"),
    ("summarise this: the market crashed and everyone lost money", "summarise this"),
    ("Translate this for Markos: we are thrilled to announce", "Translate this for Markos"),
    ("look:\n```\nError: boom\n```\nugh", "look:\nugh"),
    ("Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: bad\nwhy?", "why?"),
    ("2026-11-10 12:00:01 ERROR db down\n[WARN] retry\nthis keeps failing", "this keeps failing"),
    ("> you said it was fine\nit was not", "it was not"),
    ("I love it", "I love it"),
    ("OK", "OK"),                                                   # too short to be shouting
    # review r1: the owner's own words stay
    ("check this out: amazing news, it works!", "check this out: amazing news, it works!"),
    ("Look at this: you nailed it, thank you so much", "Look at this: you nailed it, thank you so much"),
    ("I read your summary: brilliant work", "I read your summary: brilliant work"),
    ("10:30 works for me, great", "10:30 works for me, great"),
    ("2026-11-10 10:00 sounds perfect", "2026-11-10 10:00 sounds perfect"),
    ("at least(ish) it works now, great", "at least(ish) it works now, great"),
    ("12:00:01.123 boom\n  at com.foo.Bar(Bar.java:1)\nugh, again", "ugh, again"),
    # review r2
    ("summarise this\n\nThe market crashed.", "summarise this"),
    ("Hey, can you summarise this: I hate it", "Hey, can you summarise this"),
    ("ERROR: nothing works and I'm furious", "ERROR: nothing works and I'm furious"),
    ("[ERROR] boom\nERROR:root:boom\nugh", "ugh"),
])
def test_pasted_text_is_taken_out_and_capitals_are_read_in_lower_case(raw, want):
    assert C.prepare_text(raw) == want


@pytest.mark.parametrize("raw", ["```\nprint('hi')\n```", "Traceback (most recent call last):", "", None, 5])
def test_nothing_of_the_owners_own_is_nothing_to_score(raw):
    assert C.prepare_text(raw) == ""


@pytest.mark.parametrize("label,text,after,why", [
    ("positive", "great, you broke it again", False, C.HELD_SARCASM),
    # review r1: "deleted" is a task's result as often as trouble ("Deleted the old ones? perfect") — not a cue
    ("positive", "wow, it deleted everything. perfect", False, ""),
    ("positive", "brilliant", True, C.HELD_SARCASM),                  # right after the agent's trouble
    ("positive", "thanks again!", False, ""),
    ("positive", "debugging done, great", False, ""),
    ("positive", "you fixed the bug, awesome", False, ""),                # review r1: a problem gone
    ("positive", "no errors this time, perfect", False, ""),
    ("positive", "Deleted the old ones? perfect, thanks", False, ""),
    ("positive", "great, you broke it again, no surprise", False, C.HELD_SARCASM),
    ("positive", "Nice try, but that's not what I asked", False, C.HELD_SARCASM),   # corrects the reply
    ("negative", "Look, this is awful and I hate it", False, ""),          # an interjection, not a task
    ("negative", "Stop, this is terrible. You keep ignoring me", False, ""),
    ("negative", "Check the date, it is wrong again, I am fed up", False, ""),     # a task WITH a feeling
    # review r2
    ("positive", "Great, you broke it again. Fix it.", False, C.HELD_SARCASM),     # "fix" is no "fixed"
    ("positive", "Let's try again tomorrow, great day today!", False, ""),
    ("positive", "no, try again", False, C.HELD_SARCASM),
    ("positive", "you fixed the bug, awesome", True, ""),                 # good news right after trouble
    ("negative", "Look at this mess they made of the city", False, ""),
    ("negative", "Fix this, it sucks", False, ""),
    ("negative", "Search for flights, I'm so stressed and worried", False, ""),
    ("negative", "delete the failed jobs", False, C.HELD_TASK),
    ("negative", "please, can you kill the stuck runs", False, C.HELD_TASK),
    ("negative", "Fix the broken import", False, C.HELD_TASK),
    ("negative", "this is awful, I lost the whole evening", False, ""),
    ("negative", "the fix you wrote is wrong", False, ""),
    ("neutral", "delete it", False, ""),
])
def test_a_doubtful_score_is_named(label, text, after, why):
    assert C.doubtful(label, text, after_trouble=after) == why


def test_a_doubted_score_does_not_move_the_mood_and_the_journal_says_why(tmp_path):
    sm = _sm(tmp_path)
    nudge, held = C._feed_mood({"label": "positive", "score": 0.95}, sm, False, None, "m1",
                               "great, you broke it again")
    assert (nudge, held) == (0.0, C.HELD_SARCASM) and sm.mood().p == 0.0
    nudge, held = C._feed_mood({"label": "negative", "score": 0.95}, sm, False, None, "m2",
                               "delete the failed jobs")
    assert (nudge, held) == (0.0, C.HELD_TASK) and sm.mood().p == 0.0


def test_an_upbeat_message_right_after_trouble_is_doubted(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=3, failures=3), now=T0)
    before = sm.mood(now=_at(minutes=5)).p
    nudge, held = C._feed_mood({"label": "positive", "score": 0.95}, sm, False, _at(minutes=5), "m1", "nice")
    assert (nudge, held) == (0.0, C.HELD_SARCASM)
    assert sm.mood(now=_at(minutes=5)).p == pytest.approx(before)


# ── 4. the agent's recent trouble ───────────────────────────────────────────

@pytest.mark.parametrize("how", ["turn", "thumb_down", "correction"])
def test_trouble_the_owner_saw_is_remembered_for_half_an_hour(tmp_path, how):
    sm = _sm(tmp_path)
    assert not sm.recent_trouble(T0)
    if how == "turn":
        sm.note_owner_turn(M.TurnSignals(tool_calls=2, failures=2), now=T0)
    else:
        sm.note_owner_reaction(how, reply_id="r1", now=T0)
    assert sm.recent_trouble(_at(minutes=29)) and not sm.recent_trouble(_at(minutes=31))


@pytest.mark.parametrize("how", ["good turn", "thumb_up", "praise"])
def test_good_news_is_no_trouble(tmp_path, how):
    sm = _sm(tmp_path)
    if how == "good turn":
        sm.note_owner_turn(M.TurnSignals(tool_calls=4), now=T0)
    else:
        sm.note_owner_reaction(how, reply_id="r1", now=T0)
    assert not sm.recent_trouble(_at(minutes=1))


# ── 5. the sentiment service's outage is said ───────────────────────────────

def test_the_pause_is_said_once_at_warning_and_the_return_at_info(caplog):
    caplog.set_level(logging.INFO, logger="GhostAgent")
    for i in range(C.BREAKER_FAILS):
        C._failed("ConnectError", 100.0 + i)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "sentiment service" in r.message]
    assert len(warns) == 1 and "ConnectError" in warns[0].message
    for i in range(C.BREAKER_FAILS):                   # the next round after the pause fails too: not said again
        C._failed("ConnectError", 1000.0 + i)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert C.BREAKER.ok(2000.0) is True and C.BREAKER.ok(2001.0) is False


def test_a_failure_or_two_is_not_an_outage(caplog):
    caplog.set_level(logging.INFO, logger="GhostAgent")
    C._failed("ConnectError", 1.0)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert C.BREAKER.ok(2.0) is False


# ── 6. the Mirror's lasting note counts the agent's own work only ───────────

NOW = dt.datetime(2026, 11, 10, 15, 0).astimezone()


def _journal(home, kind, label="content"):
    p = home / "system" / "selfhood" / MOOD_HISTORY_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in (0, 2, 4):
        local = dt.datetime.combine(NOW.date() - dt.timedelta(days=d), dt.time(14)).astimezone()
        at = local.astimezone(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        rows.append({"label": label, "intensity": 2, "at": at, "cause": "x", "kind": kind})
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("kind", ["thumb_up", "praise", "sentiment", "thumb_down", "correction", "owner",
                                  "manner", "decay", "quiet", "quiet_end", "", "something new"])
def test_the_owners_reactions_are_not_the_agents_character(tmp_path, kind):
    _journal(tmp_path, kind)
    assert MI.lasting_mood_note(tmp_path, now=NOW) == ""


def test_the_agents_own_turns_are(tmp_path):
    _journal(tmp_path, "turn")
    assert "content (2 or more of 3) on 3 of them" in MI.lasting_mood_note(tmp_path, now=NOW)


# ── 7. project tasks a job closed ───────────────────────────────────────────

def test_a_jobs_finished_tasks_lift_the_mood_by_the_task_event_only(tmp_path):
    sm = _sm(tmp_path)
    v = sm.note_job_tasks_done(2, now=T0)
    assert (v.p, v.e) == M.EV_TASK_DONE
    assert v.recent[-1]["cause"] == "finished 2 project tasks (in a background job)"


@pytest.mark.parametrize("done", [0, -1, None, "x"])
def test_no_task_no_move(tmp_path, done):
    sm = _sm(tmp_path)
    assert sm.note_job_tasks_done(done, now=T0) is None and sm.mood(now=T0).p == 0.0


@pytest.mark.parametrize("rid,counts", [("job-4ub1", True), ("sub-job-4ub1", False), ("web-elena-1", False),
                                        ("sched-4ub", False), ("probe-4ub", False)])
def test_only_the_wake_up_of_an_owners_job_counts(rid, counts):
    with _req(rid=rid, conv=""):
        assert A.mood_job_tasks_count(SimpleNamespace(turn_origin_label=None, owner_memory_isolated=False)) \
            is counts


def test_a_contained_job_wake_does_not_count():
    with _req(rid="job-4ub2", conv=""):
        assert A.mood_job_tasks_count(SimpleNamespace(turn_origin_label=None, owner_memory_isolated=True)) \
            is False


# ── 8. who set the manner is the owner's to know ────────────────────────────

@pytest.mark.parametrize("label,n", [("neutral", 0), ("down", 2)])
@pytest.mark.parametrize("audience", ["others", "public"])
def test_others_hear_the_manner_without_who_set_it(label, n, audience):
    line = M.tone_line(label, n, "sarcastic", 2, audience=audience)
    assert "MANNER: sarcastic, 2 of 3." in line and "set by the owner" not in line
    assert M.MANNER_WHY_OTHERS in line and M.MANNER_WHY_OWNER not in line


def test_the_owner_hears_who_set_it():
    line = M.tone_line("neutral", 0, "terse", 1)
    assert "MANNER (set by the owner): terse" in line and M.MANNER_WHY_OWNER in line
    both = M.tone_line("down", 1, "terse", 1)
    assert M.MANNER_WHY_OTHERS not in both and M.MANNER_WHY_OWNER in both            # review r1 n10


@pytest.mark.parametrize("mn", [2, 3])
def test_a_manner_already_heard_is_not_remarked_on_again(mn):
    line = M.tone_line("neutral", 0, "formal", mn, remarked=True)
    assert M.MANNER_REMARKED_SENTENCE[mn] in line and M.MANNER_LEVEL_SENTENCE[mn] not in line


def test_a_manner_trims_padding_never_content():
    assert "every point the answer needs (a manner trims only padding)" in M.tone_line("neutral", 0, "terse", 3)


# ── 9. the remark of a run with no conversation; the record of a mood gone ──

def test_scheduled_runs_remark_once(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    seen = []
    for rid in ("sched-report-1", "sched-report-2"):
        with _req(rid=rid, conv=""):
            seen.append("a remark or two" in A._mood_tone_block(_ctx(sm)))
    assert seen == [True, False]


@pytest.mark.parametrize("rids", [("probe-1", "probe-2"), ("web-noconv-1", "web-noconv-2")])
def test_a_probe_or_an_owner_request_with_no_key_reads_a_first_turn(tmp_path, rids):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    for rid in rids:
        with _req(rid=rid, conv=""):
            assert "a remark or two" in A._mood_tone_block(_ctx(sm))


def test_a_mood_gone_mid_request_leaves_the_record(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    with _req(rid="web-elena-9"):
        assert A._mood_tone_block(_ctx(sm))
        assert A.mood_tone_heard("web-elena-9")
        sm.state.reset_mood()
        assert A._mood_tone_block(_ctx(sm)) == ""
        assert A.mood_tone_heard("web-elena-9") is None


# ── 10. the owner's words: Greek corrections, labelled rows, queued messages ──

@pytest.mark.parametrize("text", ["όχι, λάθος", "Όχι αυτό", "οχι", "Λάθος", "δεν είναι αυτό που ζήτησα",
                                  "Δεν εννοούσα αυτό", "ξαναπροσπάθησε", "δεν δουλεύει", "δεν λειτουργεί ακόμα",
                                  "δεν είπα αυτό", "Δεν λειτουργεί."])
def test_a_greek_correction_is_a_candidate(text):
    assert has_correction_phrase(text) and is_correction_candidate(text, "")


@pytest.mark.parametrize("text", ["καλημέρα", "οχυρό", "όχιμα", "πες μου για τον Κοσμά", "λαθρεμπόριο",
                                  # review r1: statements, not corrections
                                  "Μάλλον δεν θα έρθω αύριο", "δεν δουλεύει το wifi στο σπίτι, βρες μου τεχνικό",
                                  "δεν είπα στη μαμά",
                                  # review r2
                                  "Δεν είναι αυτονόητο", "Δεν ρώτησα τον γιατρό ακόμα", "δεν ζήτησα πολλά από τη ζωή",
                                  "Δεν λειτουργεί ακόμα το νέο μετρό στη Θεσσαλονίκη", "λάθος μου, συγγνώμη"])
def test_greek_that_is_no_correction_is_not(text):
    assert not has_correction_phrase(text)


def test_a_labelled_row_is_another_persons():
    label = FOREIGN_MESSAGE_LABELS[0] if isinstance(FOREIGN_MESSAGE_LABELS, tuple) else FOREIGN_MESSAGE_LABELS
    assert A._is_foreign_row(f"{label} no, that's wrong")
    assert A._is_foreign_row([{"type": "text", "text": f"  {label} no"}])
    assert not A._is_foreign_row("no, that's wrong") and not A._is_foreign_row(None)


class _Words:
    """A GhostAgent stand-in for the owner-words hook: records the reactions it was given."""

    def __init__(self, sm):
        self.context = SimpleNamespace(self_model=sm, turn_origin_label=None, owner_memory_isolated=False,
                                       tor_proxy=None)
        self.reactions = []

    def _mood_owner_reaction(self, kind):
        self.reactions.append(kind)


def test_the_owners_words_are_read_for_praise(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    w = _Words(_sm(tmp_path))
    with _req(rid="web-elena-2"):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect")
    assert w.reactions == [M.KIND_PRAISE]


@pytest.mark.parametrize("rid,role", [("web-elena-3", "member"), ("probe-4ub", ""), ("sched-4ub", "")])
def test_not_a_members_probes_or_runs_words(tmp_path, monkeypatch, rid, role):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    w = _Words(_sm(tmp_path))
    with _req(rid=rid, role=role):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect")
    assert w.reactions == []


def test_not_a_labelled_row(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    label = FOREIGN_MESSAGE_LABELS[0] if isinstance(FOREIGN_MESSAGE_LABELS, tuple) else FOREIGN_MESSAGE_LABELS
    w = _Words(_sm(tmp_path))
    with _req(rid="web-elena-4"):
        A.GhostAgent._mood_read_owner_words(w, f"{label} thanks, perfect")
    assert w.reactions == []


def test_a_queued_message_is_read_for_the_mood():
    seen = []
    fake = SimpleNamespace(_mood_read_owner_words=lambda t, **kw: seen.append((t, kw)))
    A.GhostAgent._on_queued_delivery(fake, "thanks, that's it", "web-elena-5")
    assert seen == [("thanks, that's it", {"queued": True})]


# ── 11. a thumb during the owner's hold is answered ─────────────────────────

def test_a_thumb_while_the_set_mood_holds_says_so(tmp_path):
    agent, sm = _labelled(tmp_path)
    sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2)
    out = F.apply_human_label(agent, "r1", "positive", source="web")
    assert out["ok"] and out["mood_note"].startswith("Your set mood holds until ")
    assert "this thumb is saved but doesn't move the mood" in out["mood_note"]


def test_a_thumb_with_no_hold_says_nothing_more(tmp_path):
    agent, _sm_ = _labelled(tmp_path)
    out = F.apply_human_label(agent, "r1", "positive", source="web")
    assert out["ok"] and "mood_note" not in out


def test_the_web_client_shows_the_note_once():
    """The page's own ``sendFeedback`` run by node (the §4SZ harness): the note is a system line, once."""
    from tests.test_4sz_thumbs_down_hint import _run
    note = "Your set mood holds until 04:00 — this thumb is saved but doesn't move the mood."
    out = _run([{"ok": True, "mood_note": note}, {"ok": True, "mood_note": note}, {"ok": True, "mood_note": 7}],
               ["positive", "negative", "positive"])
    assert out == {"shown": [["system", note]], "latched": ["positive", "negative", "positive"]}


# ── 12. the Mirror says why a look had nothing to read ──────────────────────

def test_a_look_with_only_strong_mood_replies_says_so(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(MI, "pick_replies", lambda h, now=None, rng=None, after="": [])
    monkeypatch.setattr(MI, "strong_mood_replies_since", lambda h, now=None, after="": 3)
    said = asyncio.run(MI.look(tmp_path, object(), now=NOW))
    assert said.startswith("skipped — all 3 replies of yours since the last look were written under a strong")


def test_a_look_with_no_reply_at_all_says_that(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(MI, "pick_replies", lambda h, now=None, rng=None, after="": [])
    said = asyncio.run(MI.look(tmp_path, object(), now=NOW))
    assert said == "skipped — no reply of yours since the last look (2 days back at most)"


# ── 13. introspect quotes what a reply hears ────────────────────────────────

def test_the_quote_is_never_a_boredom_no_reply_hears():
    from ghost_agent.tools import introspect as I
    assert "boredom of a quiet, which replies never hear" in I.TONE_NONE


# ── 14. pins through the real paths (battery) ───────────────────────────────

def test_a_job_wake_that_closed_a_task_lifts_the_mood_by_that_alone(tmp_path):
    from ghost_agent.tools.outcome import ToolOutcome
    from tests.test_4tc_mood_hooks import ASK, BAD, _record, _traj_agent
    agent = _traj_agent(tmp_path)
    closed = ToolOutcome.ok(json.dumps({"updated": [{"id": "t1", "status": "DONE", "result_summary": "x"}],
                                        "count": 1, "closed": ["t1"]}))
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "p1", "type": "function",
                "function": {"name": "manage_projects", "arguments": json.dumps({"action": "task_update"})}}]},
            {"role": "tool", "tool_call_id": "p1", "name": "manage_projects", "content": closed}]
    for i in range(3):                                   # the job's own failures are not the owner's turn
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{
                     "id": f"x{i}", "type": "function",
                     "function": {"name": "execute", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"x{i}", "name": "execute", "content": BAD}]
    with _req(rid="job-4ub-wake", conv=""):
        _record(agent, msgs, rid="job-4ub-wake")
    v = agent.context.self_model.mood()
    assert (v.p, v.e) == M.EV_TASK_DONE
    assert v.recent[-1]["cause"] == "finished 1 project task (in a background job)"


def test_a_sub_agents_job_wake_moves_nothing(tmp_path):
    from tests.test_4tc_mood_hooks import ASK, _record, _traj_agent
    from ghost_agent.tools.outcome import ToolOutcome
    agent = _traj_agent(tmp_path)
    closed = ToolOutcome.ok(json.dumps({"updated": [], "count": 1, "closed": ["t1"]}))
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "p1", "type": "function",
                "function": {"name": "manage_projects", "arguments": json.dumps({"action": "task_update"})}}]},
            {"role": "tool", "tool_call_id": "p1", "name": "manage_projects", "content": closed}]
    with _req(rid="sub-job-4ub-wake", conv=""):
        _record(agent, msgs, rid="sub-job-4ub-wake")
    assert agent.context.self_model.mood().p == 0.0


def test_introspect_quotes_the_reply_view_not_the_quiets_boredom(tmp_path):
    import dataclasses
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    real = sm.state.mood

    def mood(**kw):
        v = real(**kw)
        return v if kw.get("for_reply") else dataclasses.replace(v, label=M.BORED, intensity=2)
    sm.state.mood = mood
    with _req(rid="web-elena-7"):
        text = I._render_mood(sm)
    assert "My mood now: bored, 2 of 3" in text and I.TONE_NONE in text and "YOUR MOOD" not in text


# ── 15. review r1 ───────────────────────────────────────────────────────────

def test_the_share_survives_a_restart(tmp_path):
    """r1 MAJOR: the share was a list in memory — after a restart two more tones made "down, 1"."""
    sm = _sm(tmp_path)
    sm.note_owner_sentiment("negative", 0.97, request_id="m1", now=T0)
    sm.note_owner_sentiment("negative", 0.97, request_id="m2", now=T0)
    again = SelfModel(root=tmp_path / "self")
    for i in range(3):
        again.note_owner_sentiment("negative", 0.97, request_id=f"n{i}", now=_at(minutes=i + 1))
    v = again.mood(now=_at(minutes=5))
    assert v.intensity == 0 and v.p >= -0.12 - 1e-3


def test_a_tone_a_thumb_replaced_gives_its_room_back(tmp_path):
    """r1: the booked move stayed after a thumb became the reply's verdict."""
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.owner_arrived(request_id="r2", now=T0)
    sm.note_owner_sentiment("positive", 0.97, request_id="r2", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=T0)
    assert sm.state.private_share(now=T0) == (0.0, 0.0)
    _a, dp, why = sm.note_owner_sentiment("positive", 0.97, request_id="m3", now=T0)
    assert why == "" and dp == pytest.approx(0.08)


def test_a_handed_back_queued_message_is_not_read_twice(tmp_path, monkeypatch):
    """r1: read at the drain, handed back after a cancel, re-sent — and read again."""
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    A._MOOD_QUEUED_READ.clear()
    w = _Words(_sm(tmp_path))
    with _req(rid="web-elena-11", conv="conv-elena"):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect", queued=True)
        turn = SimpleNamespace(_q_items=[{"text": "thanks, perfect"}], _q_read=0, _q_skipped=(),
                               session_id="s", kind="web", req_id="web-elena-11")
        reg = SimpleNamespace(add_leftovers=lambda *a: 1)
        assert A.GhostAgent._hand_back_unread(reg, turn, []) == 1
    with _req(rid="web-elena-12", conv="conv-elena"):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect")      # the re-send: not again
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect")      # said again later: read
    assert w.reactions == [M.KIND_PRAISE, M.KIND_PRAISE]


def test_a_queued_message_that_was_answered_does_not_block_the_same_words_later(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_SENTIMENT_URL", raising=False)
    A._MOOD_QUEUED_READ.clear()
    w = _Words(_sm(tmp_path))
    with _req(rid="web-elena-13", conv="conv-elena"):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect", queued=True)
    with _req(rid="web-elena-14", conv="conv-elena"):
        A.GhostAgent._mood_read_owner_words(w, "thanks, perfect")
    assert w.reactions == [M.KIND_PRAISE, M.KIND_PRAISE]


# ── 16. review r2 ───────────────────────────────────────────────────────────

def _two_tones_one_reply(sm, first, second, *, at=T0):
    sm.owner_turn_ended(request_id="r3", now=at)
    sm.owner_arrived(request_id="r4", now=at)
    sm.note_owner_sentiment(first, 0.97, request_id="r4", now=at)
    return sm.note_owner_sentiment(second, 0.97, request_id="r4", now=at)    # a queued message: the same reply


def test_a_second_tone_on_one_reply_cannot_pass_the_cap(tmp_path):
    """r2 MAJOR: two unhappy replies, an upbeat request, then a queued unhappy message — "down, 1"."""
    sm = _sm(tmp_path)
    for i, rid in enumerate(("r1", "r2")):
        sm.owner_turn_ended(request_id=f"p{i}", now=T0)
        sm.owner_arrived(request_id=rid, now=T0)
        sm.note_owner_sentiment("negative", 0.97, request_id=rid, now=T0)
    _two_tones_one_reply(sm, "positive", "negative")
    v = sm.mood(now=T0)
    assert v.intensity == 0 and sm.state.private_share(now=T0)[0] >= -0.12 - 1e-3


def test_a_cut_tone_never_moves_the_mood_the_other_way(tmp_path):
    """r2 MAJOR: a second upbeat message cut to the room replaced the reply's +0.08 and moved −0.078."""
    sm = _sm(tmp_path)
    for i, rid in enumerate(("r1", "r2")):
        sm.owner_turn_ended(request_id=f"p{i}", now=T0)
        sm.owner_arrived(request_id=rid, now=T0)
        sm.note_owner_sentiment("positive", 0.97, request_id=rid, now=T0)
    before = sm.mood(now=T0).p
    _after, dp, _why = _two_tones_one_reply(sm, "positive", "positive")
    assert dp >= 0.0 and sm.mood(now=T0).p >= before - 1e-9


def test_a_small_failure_in_a_good_turn_is_no_trouble(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=6, failures=1), now=T0)
    assert not sm.recent_trouble(_at(minutes=1))
    sm.note_owner_turn(M.TurnSignals(tool_calls=1, failures=1), now=T0)
    assert sm.recent_trouble(_at(minutes=1))


# ── 17. review r3 ───────────────────────────────────────────────────────────

def test_a_steady_tone_does_not_keep_a_level_it_did_not_make_past_its_own_cause(tmp_path):
    """r3 MAJOR: at a cap of 0.16 (the hold-down of level 1) a tone every two minutes kept a thumb's "content, 1"
    for 11 hours; at 0.12 the level goes once the thumb's own part is under 0.04."""
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r0", now=T0)
    for k in range(0, 6 * 60, 2):
        rid = f"q{k}"
        sm.owner_turn_ended(request_id=f"p{k}", now=_at(minutes=k))
        sm.owner_arrived(request_id=rid, now=_at(minutes=k))
        sm.note_owner_sentiment("positive", 0.97, request_id=rid, now=_at(minutes=k))
    assert sm.mood(now=_at(hours=6)).intensity == 0


def test_a_good_turn_or_a_thumbs_up_ends_the_trouble(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_turn(M.TurnSignals(tool_calls=2, failures=2), now=T0)
    sm.note_owner_turn(M.TurnSignals(tool_calls=3), now=_at(minutes=5))
    assert not sm.recent_trouble(_at(minutes=6))
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=_at(minutes=7))
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r2", now=_at(minutes=8))
    assert not sm.recent_trouble(_at(minutes=9))


@pytest.mark.parametrize("text,why", [
    ("great, the errors are gone", ""), ("Awesome, it no longer crashes", ""),
    ("Wow, nothing went wrong this time!", ""), ("cool, try again later then", ""),
])
def test_review_r3_upbeat_cases(text, why):
    assert C.doubtful("positive", text) == why


@pytest.mark.parametrize("text", ["please fix this, it's ruining my day",
                                  "restart the server, it keeps crashing and I'm losing customers",
                                  "find the bug please, nothing works and I'm desperate"])
def test_review_r3_a_task_with_a_feeling_is_the_feeling(text):
    assert C.doubtful("negative", text) == ""


def test_a_line_that_names_a_file_is_the_owners():
    assert C.prepare_text('File "x.py" is missing, I\'m upset') == 'File "x.py" is missing, I\'m upset'


# ── 18. review r4 ───────────────────────────────────────────────────────────

def test_a_tone_a_changed_thumb_brings_back_fits_the_room(tmp_path):
    """r4 MAJOR: tone on X, 👍 on X, tones on Y and Z, 👎 on X → share +0.196; two clean turns later the owner
    heard "content, 1" that only the tone made."""
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="X", now=T0)
    sm.owner_arrived(request_id="mX", now=T0)
    sm.note_owner_sentiment("positive", 0.97, request_id="mX", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="X", now=T0)
    for rid in ("Y", "Z"):
        sm.owner_turn_ended(request_id=rid, now=T0)
        sm.owner_arrived(request_id="m" + rid, now=T0)
        sm.note_owner_sentiment("positive", 0.97, request_id="m" + rid, now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="X", now=T0)
    assert sm.state.private_share(now=T0)[0] <= M.SENTIMENT_SHARE_MAX + 1e-3


@pytest.mark.parametrize("text,trouble", [
    ("great, you found the error", False), ("great, the error went away", False),
    ("great work on fixing the crash", False), ("great, it's working", True),
    ("awesome, everything works", True), ("excellent, the tests pass now", True),
])
def test_review_r4_good_news_after_a_fix(text, trouble):
    assert C.doubtful("positive", text, after_trouble=trouble) == ""


@pytest.mark.parametrize("raw", ["rewrite the intro\nI'm really unhappy with how it reads",
                                 "ERROR 404 again, I'm so annoyed", "INFO 3 tests failed and I'm sick of it"])
def test_review_r4_the_owners_own_line_stays(raw):
    assert C.prepare_text(raw) == raw


# ── 19. battery pins ────────────────────────────────────────────────────────

from tests.test_4tw_sentiment import client, service  # noqa: E402,F401 — the fake service fixtures


async def test_the_service_is_sent_the_prepared_text(client, tmp_path):
    _url, clf = client
    sm = SelfModel(tmp_path / "selfhood")
    raw = "WHY IS THIS STILL BROKEN\n```\nTraceback (most recent call last):\n```"
    assert await C.score_message(raw, sm.root, self_model=sm) is True
    assert await C.score_message("summarise this: the war began", sm.root, self_model=sm) is True
    assert clf.seen == ["why is this still broken", "summarise this"]
    rows = C.journal_rows(sm.root)
    assert rows[0]["text"] == "why is this still broken" and "war" not in json.dumps(rows)


async def test_nothing_of_the_owners_own_is_not_sent(client, tmp_path):
    _url, clf = client
    sm = SelfModel(tmp_path / "selfhood")
    assert await C.score_message("```\nprint('x')\n```", sm.root, self_model=sm) is False
    assert clf.seen == []


# ── 20. review r5: the cap lives where the share is written ─────────────────

def test_the_share_never_passes_the_cap_whatever_the_ledger_does(tmp_path):
    """Review r5 (MAJOR): a tone that stopped counting exposed another reply's upbeat share — the gross share
    reached 0.16+ and the owner heard a level made by sentiment alone. The state caps it at the write."""
    sm = _sm(tmp_path)
    t = NOW
    for i in range(6):
        sm.note_owner_sentiment("positive", 0.95, now=t, request_id=f"web-elena-cap{i}")
    share = sm.state.private_share(now=t)[0]
    assert 0 < share <= M.SENTIMENT_SHARE_MAX + 1e-9
    # a direct write past the cap (a stored share from an older version, a ledger adjust) is cut too
    sm.state.nudge(0.10, 0.0, "the owner's message sounded pleased", M.KIND_SENTIMENT, now=t,
                   private_adjust=(0.10, 0.0))
    share2 = sm.state.private_share(now=t)[0]
    assert share2 <= M.SENTIMENT_SHARE_MAX + 1e-9


def test_what_the_cap_cuts_leaves_the_pair_too(tmp_path):
    from ghost_agent.selfhood.state import SelfStateThread
    owner = {"private_p": M.SENTIMENT_SHARE_MAX + 0.05, "private_e": 0.0}
    p, e = SelfStateThread._cap_share_locked(owner, 0.30, 0.10)
    assert abs(p - 0.25) < 1e-9 and e == 0.10 and owner["private_p"] == M.SENTIMENT_SHARE_MAX
    owner = {"private_p": -(M.SENTIMENT_SHARE_MAX + 0.05), "private_e": 0.0}
    p, _ = SelfStateThread._cap_share_locked(owner, -0.30, 0.0)
    assert abs(p + 0.25) < 1e-9 and owner["private_p"] == -M.SENTIMENT_SHARE_MAX
    cap_e = M.SENTIMENT_SHARE_MAX * abs(M.EV_SENTIMENT_POSITIVE[1] / M.EV_SENTIMENT_POSITIVE[0])
    owner = {"private_p": 0.0, "private_e": cap_e + 0.02}
    _, e = SelfStateThread._cap_share_locked(owner, 0.0, 0.10)
    assert abs(e - 0.08) < 1e-9 and abs(owner["private_e"] - cap_e) < 1e-9
    owner = {"private_p": 0.05, "private_e": 0.0}                   # under the cap: untouched
    assert SelfStateThread._cap_share_locked(owner, 0.30, 0.10) == (0.30, 0.10) and owner["private_p"] == 0.05


# ── 21. battery 2 pins ──────────────────────────────────────────────────────

async def test_sarcasm_is_read_off_the_owners_own_words(client, tmp_path):
    """Battery: the doubt check must see the message text (an upbeat score on "still broken" is sarcasm)."""
    sm = SelfModel(tmp_path / "selfhood")
    assert await C.score_message("great, it's still broken", sm.root, self_model=sm) is True
    row = C.journal_rows(sm.root)[-1]
    assert row["label"] == "positive" and row["nudge"] == 0.0 and row["held"] == C.HELD_SARCASM


def test_only_strong_mood_replies_are_counted_as_left_out(tmp_path, monkeypatch):
    import ghost_agent.core.failure_census as FC
    from tests.test_4ua_character_mood_tone import NOW as N2, _turn
    turns = [_turn(1, {"mood": "irritable, 2 of 3", "manner": ""}), _turn(2), _turn(3, {"mood": "neutral",
                                                                                         "manner": ""})]
    monkeypatch.setattr(FC, "_owner_turns", lambda sysd, days: (turns, []))
    assert MI.strong_mood_replies_since(tmp_path, now=N2) == 1


def test_a_labelled_row_is_never_sent_for_scoring(tmp_path, monkeypatch):
    """Battery: has_praise never matches a labelled row, so the praise check alone did not pin the guard —
    the sentiment spawn does."""
    import ghost_agent.utils.logging as L
    spawned = []

    def _spawn(coro, name=""):
        spawned.append(name)
        coro.close()
    monkeypatch.setattr(L, "spawn_bg", _spawn)
    monkeypatch.setenv("GHOST_SENTIMENT_URL", "http://127.0.0.1:9")
    label = FOREIGN_MESSAGE_LABELS[0]
    w = _Words(_sm(tmp_path))
    with _req(rid="web-elena-5"):
        A.GhostAgent._mood_read_owner_words(w, f"{label}] this is wonderful, thank you so much")
    assert spawned == []
    with _req(rid="web-elena-6"):
        A.GhostAgent._mood_read_owner_words(w, "this is wonderful, thank you so much")
    assert spawned == ["sentiment"]


def test_the_correction_judge_skips_a_labelled_row():
    """Battery: the gate before the correction judge names the labelled-row test on the user's message."""
    import ast
    import inspect
    hc = next(n for n in ast.walk(ast.parse(inspect.getsource(A.GhostAgent)))
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
    gates = [n for n in ast.walk(hc) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
             and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "requester_is_member"
                     for c in ast.walk(n.test))
             and any(isinstance(s, ast.Raise) and "SkipMemberCorrection" in ast.dump(s) for s in n.body)]
    assert len(gates) == 1
    calls = [c for c in ast.walk(gates[0].test) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "_is_foreign_row"]
    assert calls and getattr(calls[0].args[0], "id", "") == "last_user_content"


# ── 22. review r6: the cap's cut belongs to the tones that held the share ───

def _ledger_share(sm, t):
    w = sm._owner_word()
    return sum(sm._booked_now(v, t, w)[0] for v in sm._private_booked.values())


def test_a_cut_by_the_cap_is_charged_to_the_tones_not_the_thumbed_reply(tmp_path):
    """Review r6 (MAJOR): tone − on A, tone + on B and C, then a thumbs-down on A — A's tone stops counting,
    B's and C's 0.16 is cut to 0.12 at the write, and that cut was booked as A's thumb move. The state's share
    and the ledger's then parted, and members later heard part of the owner's tone."""
    sm = _sm(tmp_path)
    t = NOW
    for r in "ABC":
        sm._answers["m" + r] = r
    sm.note_owner_sentiment("negative", 0.95, now=t, request_id="mA")
    sm.note_owner_sentiment("positive", 0.95, now=t, request_id="mB")
    sm.note_owner_sentiment("positive", 0.95, now=t, request_id="mC")
    _, after = None, sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=t, reply_id="A")
    assert sm.state.private_share(now=t)[0] == pytest.approx(M.SENTIMENT_SHARE_MAX)
    assert _ledger_share(sm, t) == pytest.approx(M.SENTIMENT_SHARE_MAX, abs=1e-4)
    w = sm._owner_word()
    assert sm._booked_now(sm._applied["A"], t, w)[0] == pytest.approx(-0.30, abs=1e-4)   # A: its own move only
    assert sm._booked_now(sm._private_booked["B"], t, w)[0] == pytest.approx(0.06, abs=1e-4)
    for kind, rid in ((M.KIND_THUMB_UP, "B"), (M.KIND_THUMB_UP, "A"), (M.KIND_THUMB_UP, "C")):
        sm.note_owner_reaction(kind, now=t, reply_id=rid)
        assert sm.state.private_share(now=t)[0] == pytest.approx(_ledger_share(sm, t), abs=1e-4)
    # every tone but A's (counting again under its thumbs-up) has stopped: A's own −0.08 is the whole share
    assert sm.state.private_share(now=t)[0] == pytest.approx(-0.08, abs=1e-4)


def test_after_view_says_what_the_cap_took(tmp_path):
    from ghost_agent.selfhood.state import SelfStateThread
    s = SelfStateThread(tmp_path)
    s.nudge(0.10, 0.025, "upbeat", M.KIND_SENTIMENT, now=NOW)
    _b, a = s.nudge(0.08, 0.02, "upbeat", M.KIND_SENTIMENT, now=NOW)
    assert a.capped == (pytest.approx(0.06), pytest.approx(0.015)) and a.moved == (pytest.approx(0.02),
                                                                                    pytest.approx(0.005))


def test_a_step_down_never_grows_the_share():
    """Review r6: a step down carried the share over by new/old — an anchor larger than the pair on an axis
    grew it (0.12 → 0.192)."""
    from ghost_agent.selfhood.state import Mood, SelfStateThread
    base = Mood(p=0.20, e=-0.50, label="calm", intensity=2, private_p=0.12, private_e=-0.03,
                updated_at=NOW.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"))
    out = SelfStateThread._share_after_step(base, 0.32, -0.24, NOW, True)
    assert 0 < out["private_p"] <= 0.12 and -0.03 <= out["private_e"] < 0


# ── 23. battery 3 pins ──────────────────────────────────────────────────────

def test_a_queued_read_never_uses_up_the_hand_back_record():
    A._MOOD_QUEUED_READ.clear()
    try:
        assert A._mood_queued_read("c-4ub", "thanks, perfect", handed_back=True, now=100.0) is True
        assert A._mood_queued_read("c-4ub", "thanks, perfect", queued=True, now=101.0) is True
        assert A._mood_queued_read("c-4ub", "thanks, perfect", now=102.0) is False     # the re-send: once
    finally:
        A._MOOD_QUEUED_READ.clear()


@pytest.mark.parametrize("text", ["Stop!", "stop! stop!", "Delete!!"])
def test_an_interjection_is_a_feeling_not_a_task(text):
    assert C.doubtful("negative", text) == ""


def test_a_task_is_still_a_task():
    assert C.doubtful("negative", "stop the server and restart it") == C.HELD_TASK


def test_a_second_tone_a_moment_later_is_already_counted(tmp_path):
    """Review r3: the same tone re-booked 30 s later differs from the decayed booking by ~0.0002 — rounding,
    not a reaction; it must not move the mood."""
    sm = _sm(tmp_path)
    sm._answers["m1"] = sm._answers["m2"] = "B"
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="m1")
    _after, dp, why = sm.note_owner_sentiment("positive", 0.95, now=NOW + dt.timedelta(seconds=30),
                                              request_id="m2")
    assert dp == 0.0 and why == sm.HELD_ALREADY_COUNTED


def _booked(sm, rid, t):
    return sm._booked_now(sm._private_booked.get(rid), t, sm._owner_word())[0]


def test_a_new_tone_never_shaves_an_earlier_replys_tone(tmp_path):
    """The room is cut BEFORE the write: B's tone stays B's; the new reply gets what is left."""
    sm = _sm(tmp_path)
    sm._answers["mB"], sm._answers["mD"] = "B", "D"
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="mB")
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="mD")
    assert _booked(sm, "B", NOW) == pytest.approx(0.08, abs=1e-4)
    assert _booked(sm, "D", NOW) == pytest.approx(M.SENTIMENT_SHARE_MAX - 0.08, abs=1e-4)


def test_a_returning_tone_never_shaves_the_others(tmp_path):
    """A thumb taken back brings a reply's tone back no bigger than the room left (`_fit_returning_tone`)."""
    sm = _sm(tmp_path)
    for r in "BC":
        sm._answers["m" + r] = r
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="mB")
    sm.note_owner_reaction(M.KIND_THUMB_UP, now=NOW, reply_id="B")          # B's tone stops counting
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="mC")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=NOW, reply_id="B")        # B's tone counts again
    assert _booked(sm, "C", NOW) == pytest.approx(0.08, abs=1e-4)
    assert _booked(sm, "B", NOW) == pytest.approx(M.SENTIMENT_SHARE_MAX - 0.08, abs=1e-4)


def test_a_second_tone_on_a_reply_whose_tone_decayed_gets_its_full_size(tmp_path):
    """The room is net of this reply's own booking (review r2): a fresh tone on the same reply re-books it at
    full size, not cut by its own decayed share."""
    sm = _sm(tmp_path)
    sm._answers["m1"] = sm._answers["m2"] = "B"
    sm.note_owner_sentiment("positive", 0.95, now=NOW, request_id="m1")
    later = NOW + dt.timedelta(hours=1.36)                       # 0.08 → about 0.05
    sm.note_owner_sentiment("positive", 0.95, now=later, request_id="m2")
    assert _booked(sm, "B", later) == pytest.approx(0.08, abs=1e-4)
