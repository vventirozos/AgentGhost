"""§4TX — the fixes of the 11-09 review of the whole mood and tone system.

What was wrong, measured on the live journal and the code:

* failures outweighed successes and drifted the mood gloomy — every failed
  call counted in full, transient ones, guards' refusals and failing test runs
  included, and a clean turn earned a flat 0.08 a long turn cancelled;
* the owner's reactions to ONE reply (a thumbs-down, the correction that came
  with it, the unhappy tone of that message) counted three times, and toggling
  a thumb drifted the mood down;
* a mood the owner set faded in about 40 minutes;
* the tone line asked for "a remark or two" on every reply, never said where
  the mood must stay out (code, tool arguments), had no order of precedence,
  and boredom reached the members and the morning report every night;
* the owner's own sentiment reached the members and the public channels;
* nothing recorded which events were refused, which tone a reply was given,
  or when the manner changed.

All names below are fictional.
"""

from __future__ import annotations

import datetime as dt
import json

from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import feedback as F
from ghost_agent.distill.schema import ToolCall
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood.state import SelfStateThread
from ghost_agent.tools.introspect import tool_introspect
from ghost_agent.tools.outcome import ToolOutcome
from ghost_agent.utils.logging import (
    conversation_key_context, reply_surface_context, request_id_context, requester_role_context)
from tests.test_4tc_mood_hooks import (ASK, BAD, OK, OWNER_RID, _labelled, _record, _traj_agent, _turn,
                                       owner)  # noqa: F401 — the fixture

T0 = dt.datetime(2026, 11, 6, 9, 0, tzinfo=dt.timezone.utc)


def _at(minutes: float = 0.0, hours: float = 0.0) -> dt.datetime:
    return T0 + dt.timedelta(hours=hours, minutes=minutes)


def _req(rid="web-elena-1", conv="conv-elena", role="", surface=""):
    return _Req(rid, conv, role, surface)


class _Req:
    def __init__(self, rid, conv, role, surface):
        self.v = [(request_id_context, rid), (conversation_key_context, conv),
                  (requester_role_context, role), (reply_surface_context, surface)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


def _ctx(sm):
    return SimpleNamespace(self_model=sm, turn_origin_label=None, owner_memory_isolated=False,
                           _replay_forbidden_tools=None)


@pytest.fixture(autouse=True)
def _fresh_memos():
    A._MOOD_HEARD_BY_CONV.clear()
    A._MOOD_TONE_BY_REQ.clear()
    yield
    A._MOOD_HEARD_BY_CONV.clear()
    A._MOOD_TONE_BY_REQ.clear()


# ── A. which failures count ─────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "SYSTEM BLOCK: that path is outside the project.",                        # a guard's refusal
    "REJECTED: the file was not read first.",
    "EXIT CODE: 1\nSTDOUT/STDERR:\nFAILED tests/test_bruno.py::test_age - assert 3 == 4\n1 failed in 0.2s",
    "SUCCESS: Wrote 120 bytes to notes/bruno.md",                             # booked failed, reads success
    "Error: Read timed out. (read timeout=30)",                               # transient
])
def test_a_failure_the_mood_does_not_count_against_itself(text):
    assert A.mood_excuses_failure(text) is True


@pytest.mark.parametrize("text", [
    "EXIT CODE: 1\nSTDOUT/STDERR:\nTraceback (most recent call last):\nValueError: x",
    "Error: no such file or directory: notes/max.md",
    "", None,
])
def test_a_real_failure_still_counts(text):
    assert A.mood_excuses_failure(text) is False


def test_the_turn_signals_split_the_failures(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    timeout = ToolOutcome.failed("URL: https://example.org/vet\nERROR: timeout")
    sig = agent._mood_turn_signals(_turn([("execute", BAD), ("browser", timeout), ("execute", OK)]),
                                   ASK, OWNER_RID)
    assert (sig.tool_calls, sig.failures, sig.excused) == (3, 1, 1)


# ── A. a finished project task ──────────────────────────────────────────────

def _pm(action="task_update", payload=None, error=""):
    c = ToolCall(name="manage_projects", arguments={"action": action})
    c.result = json.dumps(payload) if payload is not None else ""
    c.error = error
    return c


def test_tasks_closed_done_are_counted_from_the_call():
    done = {"updated": [{"id": "t1", "status": "DONE"}, {"id": "t2", "status": "IN_PROGRESS"},
                        {"id": "t3", "status": "DONE"}], "count": 3, "closed": ["t1", "t3"]}
    assert A.mood_tasks_done(_pm(payload=done)) == 2
    # review r2: a re-close of a task already DONE is in "updated" as DONE, not in "closed"
    reclose = {"updated": [{"id": "t1", "status": "DONE"}], "count": 1}
    assert A.mood_tasks_done(_pm(payload=reclose)) == 0
    # review r2: the row's result is cut; the full result is read
    long = {"updated": [{"id": "t1", "status": "DONE", "result_summary": "x" * 5000}], "count": 1, "closed": ["t1"]}
    cut = _pm(payload=long)
    cut.full_result, cut.result = cut.result, cut.result[:4000]
    assert A.mood_tasks_done(cut) == 1
    assert A.mood_tasks_done(_pm(payload={"updated": [], "refused": [{"task_id": "t1"}]})) == 0
    assert A.mood_tasks_done(_pm(action="status", payload=done)) == 0
    assert A.mood_tasks_done(_pm(payload=done, error="failed")) == 0
    assert A.mood_tasks_done(_pm(payload=None)) == 0
    assert A.mood_tasks_done(ToolCall(name="execute", arguments={"action": "task_update"})) == 0


def test_a_turn_that_closed_a_task_lifts_the_mood(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    closed = ToolOutcome.ok(json.dumps({"updated": [{"id": "t1", "status": "DONE", "result_summary": "x"}],
                                        "count": 1, "closed": ["t1"]}))
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "p1", "type": "function",
                "function": {"name": "manage_projects", "arguments": json.dumps({"action": "task_update"})}}]},
            {"role": "tool", "tool_call_id": "p1", "name": "manage_projects", "content": closed}]
    assert agent._mood_turn_signals(msgs, ASK, OWNER_RID).tasks_done == 1
    _record(agent, msgs)
    v = agent.context.self_model.mood()
    assert (v.p, v.e) == (pytest.approx(0.15), pytest.approx(0.10))
    assert v.recent[-1]["cause"] == "finished 1 project task"


# ── B. the reaction ledger ──────────────────────────────────────────────────

def _sm(tmp_path):
    return SelfModel(root=tmp_path / "self")


def test_a_correction_and_the_unhappy_tone_of_the_same_message_are_one_verdict(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1")                       # the reply r1
    sm.owner_arrived(request_id="r2")                          # the owner's next message answers it
    after, dp, why = sm.note_owner_sentiment("negative", 0.95, request_id="r2")
    assert (dp, why) == (-0.08, "") and after.p == pytest.approx(-0.08)
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2")                # the stronger of the two
    assert sm.mood().p == pytest.approx(-0.20, abs=1e-3)
    _a, dp, why = sm.note_owner_sentiment("negative", 0.99, request_id="r2")  # again: already counted
    assert (dp, why) == (0.0, SelfModel.HELD_ALREADY_COUNTED)
    assert sm.state.mood_events()[-1]["held"] == SelfModel.HELD_ALREADY_COUNTED


def test_the_sentiment_feed_books_the_tone_against_the_reply_the_message_answers(tmp_path):
    from ghost_agent.selfhood import sentiment as S
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1")
    sm.owner_arrived(request_id="r2")
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2")
    dp, why = S._feed_mood({"label": "negative", "score": 0.95}, sm, False, None, "r2")
    assert (dp, why) == (0.0, SelfModel.HELD_ALREADY_COUNTED)          # the correction already said it
    assert sm.mood().p == pytest.approx(-0.20, abs=1e-3)


def test_a_thumbs_down_and_the_correction_of_the_same_reply_are_one_verdict(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1")
    sm.owner_arrived(request_id="r2")
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2")
    assert sm.mood().p == pytest.approx(-0.30, abs=1e-3)


def test_thanks_and_a_thumbs_up_for_the_same_reply_are_one_verdict_and_mixed_verdicts_add(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1")
    sm.owner_arrived(request_id="r2")
    sm.note_owner_reaction(M.KIND_PRAISE, request_id="r2")
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1")
    assert sm.mood().p == pytest.approx(0.25, abs=1e-3)
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2")     # pleased and corrected: both stand
    assert sm.mood().p == pytest.approx(0.05, abs=1e-3)


def test_reactions_to_different_replies_each_count(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1")
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r2")
    assert sm.mood().p == pytest.approx(-0.60, abs=1e-3)


def test_a_thumb_toggled_through_the_real_feedback_path_does_not_drift(tmp_path):
    agent, sm = _labelled(tmp_path)
    for _ in range(4):
        F.apply_human_label(agent, "r1", "positive", source="web")
        F.apply_human_label(agent, "r1", "negative", source="web")
    assert sm.mood().p == pytest.approx(-0.30, abs=0.01)


def test_a_reaction_with_no_reply_known_counts_on_its_own(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_PRAISE)
    sm.note_owner_reaction(M.KIND_PRAISE)
    assert sm.mood().p == pytest.approx(0.20, abs=1e-3)


def test_the_ledger_is_bounded(tmp_path):
    sm = _sm(tmp_path)
    for i in range(SelfModel.LEDGER_REPLIES + 30):
        sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id=f"r{i}")
    assert len(sm._standing) == SelfModel.LEDGER_REPLIES and len(sm._applied) == SelfModel.LEDGER_REPLIES


def test_the_agent_hands_the_ledger_its_request_ids(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    sm = agent.context.self_model
    _record(agent, _turn([("execute", OK)]))                          # the owner's reply, this request id
    assert sm._last_reply_id == OWNER_RID
    with _Req("web-elena-next", "", "", ""):
        agent._mood_owner_arrived()
        assert sm._answers["web-elena-next"] == OWNER_RID
        agent._mood_owner_reaction(M.KIND_CORRECTION)
    assert OWNER_RID in sm._standing and M.KIND_CORRECTION in sm._standing[OWNER_RID]


# ── C. praise words ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "excellent", "amazing, ty", "brilliant!", "nice one", "good stuff",
    "τέλειο", "τελειο", "Ωραία!", "άψογα", "εξαιρετικό", "καλή δουλειά", "σ' ευχαριστώ πολύ", "υπέροχο",
    "👍", "🙏🙏", "👏🏽", "❤️", "🔥 🔥",
])
def test_more_ways_of_saying_thanks_are_praise(text):
    assert M.has_praise(text) is True


@pytest.mark.parametrize("text", [
    "τέλειο;", "excellent question about Max", "👍 but the date is wrong", "🙂", "ok", "", "!!!",
])
def test_and_these_are_not(text):
    assert M.has_praise(text) is False


# ── D. the hold ─────────────────────────────────────────────────────────────

def test_the_hold_survives_a_restart(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    again = SelfStateThread(tmp_path)
    assert again.mood(now=_at(hours=3)).held_until == s.state.mood.held_until != ""
    assert again.hold_reason(now=_at(hours=3)) == SelfStateThread.HELD_OWNER_HOLD


def test_a_hold_on_an_automatic_record_is_ignored(tmp_path):
    """A hand edit cannot freeze an automatic mood: only an owner-set record carries a hold."""
    from ghost_agent.selfhood.schema import Mood
    m = Mood.from_dict({"p": 0.5, "e": 0.1, "source": "auto", "held_until": "2099-01-01T00:00:00Z"})
    assert m.held_until == ""


def test_a_turn_held_by_the_owners_word_is_in_the_events_log_not_the_history(tmp_path):
    sm = _sm(tmp_path)
    sm.state.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2)
    rows_before = sm.state.mood_history()
    assert sm.note_owner_turn(M.TurnSignals(tool_calls=4, failures=4)) is not None
    assert sm.mood().label == "cheerful" and sm.state.mood_history() == rows_before
    assert sm.state.mood_events()[-1]["held"] == SelfStateThread.HELD_OWNER_HOLD


# ── E. what other people hear ───────────────────────────────────────────────

def test_the_owners_sentiment_is_left_out_of_what_others_hear(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.10, 0.02, "a clean turn", M.KIND_TURN, now=T0)
    s.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    s.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    s.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    s.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    own = s.mood(now=_at(minutes=1))
    assert (own.label, own.intensity) == ("down", 1)
    assert (s.state.mood.private_p, s.state.mood.private_e) == (pytest.approx(-0.32), pytest.approx(-0.08))
    others = s.mood(now=_at(minutes=1), audience="others")
    # review r2: the share comes out only toward zero — others never hear a mood the agent does not have
    assert (others.label, others.p) == ("neutral", 0.0)
    assert others.cause == "" and others.recent == ()


def test_the_private_share_decays_with_the_pair_and_the_owners_word_clears_it(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(-0.5, -0.2, "unhappy", M.KIND_SENTIMENT, now=T0)
    assert SelfStateThread._private_now(s.state.mood, _at(hours=2))[0] == pytest.approx(-0.25)
    s.nudge(0.0, 0.1, "a turn", M.KIND_TURN, now=_at(hours=2))                # carried, decayed
    assert s.state.mood.private_p == pytest.approx(-0.25, abs=1e-3)
    s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, now=_at(hours=2))
    assert (s.state.mood.private_p, s.state.mood.private_e) == (0.0, 0.0)


def test_the_private_share_is_what_was_really_added_after_the_clamp(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.98, 0.0, "very good", M.KIND_TURN, now=T0)
    s.nudge(0.08, 0.04, "upbeat", M.KIND_SENTIMENT, now=T0)
    assert s.state.mood.private_p == pytest.approx(0.02, abs=1e-3)


def test_a_member_hears_the_mood_without_the_owners_sentiment_and_is_told_it_is_not_about_them(tmp_path):
    sm = _sm(tmp_path)
    for _ in range(5):
        sm.state.nudge(*M.EV_SENTIMENT_NEGATIVE, "unhappy", M.KIND_SENTIMENT)
    ctx = _ctx(sm)
    with _req(rid="slack-m-1", role="member", conv="slack-c1"):
        assert A._mood_tone_block(ctx) == ""
    with _req(rid="web-elena-2"):
        assert A._mood_tone_block(ctx).startswith("YOUR MOOD RIGHT NOW: down, 1 of 3.")
    sm.state.nudge(-0.5, 0.40, "4 tool calls failed in this turn", M.KIND_TURN)
    with _req(rid="slack-m-2", role="member", conv="slack-c1"):
        line = A._mood_tone_block(ctx)
    assert line.startswith("YOUR MOOD RIGHT NOW: irritable") and M.WHY_OTHERS in line and M.WHY_OWNER not in line
    with _req(rid="slack-elena-3", surface="public", conv="slack-c2"):
        line = A._mood_tone_block(ctx)
    # review r2: the owner on a public channel is not told "it is not about the person you are talking to"
    assert M.WHY_PUBLIC in line and M.WHY_OTHERS not in line and M.WHY_OWNER not in line


# ── F. one remark per conversation ──────────────────────────────────────────

def test_the_same_mood_is_remarked_on_once_per_conversation(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    ctx = _ctx(sm)
    with _req(rid="web-1", conv="c-a"):
        first = A._mood_tone_block(ctx)
        assert A._mood_tone_block(ctx) == first                        # the same request, every call: same bytes
    assert "a remark or two" in first
    with _req(rid="web-2", conv="c-a"):
        second = A._mood_tone_block(ctx)
    assert "no remark about it" in second and "a remark or two" not in second
    with _req(rid="web-3", conv="c-b"):                                # another conversation: a remark
        assert "a remark or two" in A._mood_tone_block(ctx)
    sm.state.nudge(-0.4, 0.2, "more trouble", M.KIND_TURN)             # a different mood: a remark again
    with _req(rid="web-4", conv="c-a"):
        assert "no remark" not in A._mood_tone_block(ctx)


def test_with_no_conversation_key_every_reply_may_remark(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    for rid in ("probe-1", "probe-2"):
        with _req(rid=rid, conv=""):
            assert "a remark or two" in A._mood_tone_block(_ctx(sm))


def test_the_memo_is_bounded():
    for i in range(A._MOOD_MEMO_MAX + 40):
        A._mood_remarked(f"c{i}", "down:1::0", f"r{i}")
    assert len(A._MOOD_HEARD_BY_CONV) == A._MOOD_MEMO_MAX


def test_the_line_says_code_and_tool_arguments_carry_no_tone_and_names_the_precedence():
    line = M.tone_line("irritable", 2)
    assert "Code, file contents, tool arguments and commands carry no tone" in line
    assert "Precedence: the owner's rules, then correctness, then the manner, then the mood." in line
    assert "the mood changes no answer and no refusal" in line and "no new topics or asides" in line


# ── H. what is recorded ─────────────────────────────────────────────────────

def test_the_turn_record_carries_the_tone_the_reply_was_given(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    agent.context.self_model.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    with _req(rid=OWNER_RID, conv="c-rec"):
        A._mood_tone_block(agent.context)
    assert A.mood_tone_heard(OWNER_RID) == {"mood": "irritable, 2 of 3", "manner": "", "remarked": False,
                                            "audience": "owner"}
    traj = _record(agent, [{"role": "user", "content": ASK}])
    assert traj.extra["mood_tone"]["mood"] == "irritable, 2 of 3"
    assert A.mood_tone_heard("web-nobody") is None


def test_every_event_applied_or_held_is_one_row_of_the_events_log(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.1, 0.0, "a clean turn", M.KIND_TURN, now=T0)
    s.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, now=_at(minutes=1))
    s.nudge(-0.3, 0.0, "the owner gave a thumbs-down", M.KIND_THUMB_DOWN, now=_at(minutes=2))
    s.set_manner("terse", 2, now=_at(minutes=3))
    s.reset_mood(now=_at(minutes=4))
    rows = s.mood_events()
    assert [(r["kind"], r.get("held", "")) for r in rows] == [
        ("turn", ""), ("owner", ""), ("thumb_down", SelfStateThread.HELD_OWNER_HOLD), ("manner", ""),
        ("owner", ""), ("manner", "")]
    assert rows[0]["after"] == "neutral" and rows[1]["after"] == "calm, 1 of 3" and rows[2]["dp"] == -0.3
    assert all(set(r) <= {"at", "kind", "dp", "de", "cause", "held", "after"} for r in rows)


def test_the_events_log_is_capped(tmp_path, monkeypatch):
    import ghost_agent.selfhood.state as S
    monkeypatch.setattr(S, "MOOD_EVENTS_MAX_ROWS", 10)
    monkeypatch.setattr(S, "MOOD_EVENTS_KEEP_ROWS", 6)
    s = SelfStateThread(tmp_path)
    for i in range(14):
        s.nudge(0.01, 0.0, f"event {i}", M.KIND_TURN, now=_at(minutes=i))
    rows = s.mood_events(limit=100)
    assert 6 <= len(rows) <= 10 and rows[-1]["cause"] == "event 13"


def test_clearing_the_manner_says_so_in_the_history_and_the_log(tmp_path, monkeypatch):
    logged = []
    import ghost_agent.utils.logging as L
    monkeypatch.setattr(L, "pretty_log", lambda title, msg, **kw: logged.append(msg))
    s = SelfStateThread(tmp_path)
    s.set_manner("playful", 1, now=T0)
    s.set_manner("", 0, now=_at(minutes=1))
    assert [r["cause"] for r in s.mood_history()] == ["the owner set the manner: playful, 1 of 3",
                                                     "the owner cleared the manner (playful)"]
    assert any("the owner cleared the manner (playful)" in m for m in logged)


# ── I. a clock set back ─────────────────────────────────────────────────────

def test_a_clock_set_back_keeps_the_manner_at_its_level():
    future = "2026-11-06T12:00:00Z"
    assert M.manner_level(3, future, now=dt.datetime(2026, 11, 6, 9, 0, tzinfo=dt.timezone.utc)) == 3
    assert M.manner_level(2, future, now=dt.datetime(2026, 11, 5, 9, 0, tzinfo=dt.timezone.utc)) == 0  # a day+


def test_a_write_under_a_clock_set_back_does_not_drop_the_manner(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_manner("formal", 2, now=_at(hours=3))
    s.nudge(0.1, 0.0, "a clean turn", M.KIND_TURN, now=T0)            # the clock went back three hours
    assert (s.state.mood.manner, s.mood(now=T0).manner_intensity) == ("formal", 2)


# ── G. set_mood ─────────────────────────────────────────────────────────────

async def _set(sm, **kw):
    return await tool_introspect(action="set_mood", self_model=sm, **kw)


async def test_the_manner_takes_its_own_intensity(tmp_path, owner):
    sm = _sm(tmp_path)
    out = await _set(sm, mood="calm", intensity=1, manner="playful", manner_intensity=3)
    assert out.startswith("Mood set to 'calm', 1 of 3; it holds as set until ")
    assert "Manner set to 'playful', 3 of 3" in out
    v = sm.mood()
    assert (v.label, v.intensity, v.manner, v.manner_intensity) == ("calm", 1, "playful", 3)


async def test_a_step_moves_what_is_there(tmp_path, owner):
    sm = _sm(tmp_path)
    await _set(sm, mood="cheerful", intensity=2)
    await _set(sm, mood="cheerful", step=1)
    assert (sm.mood().label, sm.mood().intensity) == ("cheerful", 3)
    await _set(sm, mood="cheerful", step=5)                              # kept in range
    assert sm.mood().intensity == 3
    await _set(sm, step=-1)                                              # alone: the mood, no manner set
    assert sm.mood().intensity == 2
    await _set(sm, manner="sarcastic", manner_intensity=3)
    await _set(sm, step=-1)                                              # alone: the manner, when one is set
    assert (sm.mood().intensity, sm.mood().manner_intensity) == (2, 2)
    await _set(sm, manner="sarcastic", step=-1)
    assert sm.mood().manner_intensity == 1


async def test_a_step_with_nothing_set_is_still_an_argument_error(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    assert await _set(sm, step=1) == I.MOOD_REQUIRED
    assert await _set(sm, step="x", mood="calm") == await _set(_sm(tmp_path / "b"), mood="calm")


async def test_the_reply_says_until_when_the_mood_holds(tmp_path, owner):
    sm = _sm(tmp_path)
    out = await _set(sm, mood="down", intensity=1)
    end = M.parse_iso_utc(sm.mood().held_until).astimezone()
    assert "it holds as set until " in out and end.strftime("%H:%M") in out
    assert end.hour == M.MANNER_DAY_END_HOUR
    assert "'neutral' takes it off" in out


def test_the_tool_definition_tells_the_model_only_on_the_owners_ask():
    from ghost_agent.tools import registry as R
    d = next(t for t in R.TOOL_DEFINITIONS if t["function"]["name"] == "introspect")["function"]
    assert "only when the owner asks" in d["description"]
    assert {"manner_intensity", "step"} <= set(d["parameters"]["properties"])


# ── J. the chip says it in words: tests/test_interface_face_signals_2026_09_11.py (§4TX) ──


# ── H. review round 2 ───────────────────────────────────────────────────────

def test_a_thumb_changed_after_the_first_wore_off_is_a_plain_thumb(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)
    later = _at(hours=8)                                   # the thumbs-down has all but worn off
    after = sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=later)
    fresh = M.reaction_event(M.KIND_THUMB_UP).dp
    assert after.p == pytest.approx(fresh, abs=0.01)       # not +0.55 on top of a mood near zero
    assert (after.label, after.intensity) == ("content", 1)


def test_the_ledger_books_what_really_moved_when_the_clamp_absorbed_it(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-1.0, 0.0, "a terrible turn", M.KIND_TURN, now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)          # at the floor: moves nothing
    assert sm._applied["r1"][:2] == (0.0, 0.0)
    after = sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=T0)
    assert after.p == pytest.approx(-1.0 + M.reaction_event(M.KIND_THUMB_UP).dp, abs=1e-3)


def test_a_reaction_at_the_limit_says_so(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-1.0, 0.0, "a terrible turn", M.KIND_TURN, now=T0)
    sm.state.nudge(0.0, -1.0, "and long", M.KIND_TURN, now=T0)
    _after, dp, why = sm.note_owner_sentiment("negative", 0.95, request_id="r9", now=T0)
    assert (dp, why) == (0.0, SelfModel.HELD_AT_LIMIT)


def test_the_owners_word_voids_what_the_ledger_booked_before_it(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)
    sm.state.reset_mood(now=_at(minutes=1))                                    # "neutral": the pair is gone
    after = sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=_at(minutes=2))
    assert after.p == pytest.approx(M.reaction_event(M.KIND_THUMB_UP).dp, abs=1e-3)


def test_a_correction_that_replaces_the_messages_tone_takes_it_out_of_the_private_share(tmp_path):
    for order in ("sentiment first", "correction first"):
        sm = _sm(tmp_path / order.replace(" ", "_"))
        sm.owner_turn_ended(request_id="r1", now=T0)
        sm.owner_arrived(request_id="r2", now=T0)
        steps = [lambda: sm.note_owner_sentiment("negative", 0.95, request_id="r2", now=T0),
                 lambda: sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2", now=T0)]
        for f in (steps if order == "sentiment first" else steps[::-1]):
            f()
        own = sm.mood(now=T0)
        others = sm.mood(now=T0, audience="others")
        assert own.p == pytest.approx(-0.20, abs=1e-3), order
        assert (others.label, others.intensity) == (own.label, own.intensity), order
        assert (sm.state._state.mood.private_p, sm.state._state.mood.private_e) == (0.0, 0.0), order


def test_others_never_get_the_cause_even_when_the_label_is_the_same(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(-0.5, 0.25, "4 tool calls failed in this turn", M.KIND_TURN, now=T0)
    s.nudge(-0.08, -0.02, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    own, others = s.mood(now=T0), s.mood(now=T0, audience="others")
    assert (own.label, own.intensity) == (others.label, others.intensity)
    assert own.cause and others.cause == "" and others.recent == ()
    assert others.p == pytest.approx(own.p + 0.08, abs=1e-3)
    assert own.set_at and others.set_at == ""                     # r5: the owner's moment is not shown either


def test_excused_failures_are_out_of_the_count_and_earn_no_clean_reward():
    assert M.turn_events(M.TurnSignals(tool_calls=3, excused=3)) == []
    evs = M.turn_events(M.TurnSignals(tool_calls=5, excused=2))
    assert [e.cause for e in evs] == ["a turn of 3 tool calls went through without a failure"]
    long_ = M.turn_events(M.TurnSignals(tool_calls=26, excused=26))          # long, and nothing done
    assert [(e.dp, e.de) for e in long_] == [M.EV_LONG_TURN]
    assert long_[0].cause == "a very long turn (0 min, 26 tool calls)"


def test_a_long_clean_turn_of_few_calls_is_good_long_work():
    evs = M.turn_events(M.TurnSignals(tool_calls=2, duration_s=1200.0))
    assert [(e.dp, e.de) for e in evs] == [M.EV_LONG_CLEAN_TURN, (pytest.approx(M.CLEAN_TURN_MAX), 0.0)]
    m = M.merge_events(evs)
    assert M.read_off(m.dp, m.de)[:2] == ("calm", 1)


@pytest.mark.parametrize("text", [
    "Error: request failed: 2 failed attempts to write file",
    "upload aborted, 3 failed chunks",
])
def test_text_that_only_mentions_failed_counts_is_not_a_test_run(text):
    assert A.mood_excuses_failure(text) is False


async def test_a_step_on_a_mood_that_is_not_on_starts_from_nothing(tmp_path, owner):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    assert (sm.mood().label, sm.mood().intensity) == ("irritable", 2)
    await _set(sm, mood="calm", step=1)                                  # "a bit calmer"
    assert (sm.mood().label, sm.mood().intensity) == ("calm", 1)


async def test_a_step_down_to_nothing_clears_only_what_it_names(tmp_path, owner):
    sm = _sm(tmp_path)
    await _set(sm, mood="cheerful", intensity=1, manner="formal", manner_intensity=1)
    out = await _set(sm, mood="cheerful", step=-1)
    assert out.startswith("Mood stepped down to neutral (it was 'cheerful', 1 of 3)")
    v = sm.mood()
    assert v.neutral and (v.manner, v.manner_intensity) == ("formal", 1)  # the manner stays
    await _set(sm, manner="formal", step=-1)
    assert sm.mood().manner == ""
    assert (await _set(sm, mood="down", step=-1)).startswith("Nothing to step down: the mood is not 'down'")
    assert sm.mood().neutral


async def test_a_step_down_on_a_mood_is_not_held_for_the_day(tmp_path, owner):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    out = await _set(sm, mood="irritable", step=-1)                      # "be less irritable"
    v = sm.mood()
    assert (v.label, v.intensity, v.held_until) == ("irritable", 1, "")
    assert "holds as set" not in out
    assert sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1") is not None   # reacts at once
    await _set(sm, mood="irritable", step=1)                             # a step UP is the owner's word: held
    assert sm.mood().held_until


def test_the_reply_a_message_answers_is_its_own_conversations(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="web-r1", conversation="web:s1")
    sm.owner_turn_ended(request_id="slack-r1", conversation="slack:t1:u1")
    sm.owner_arrived(request_id="web-r2", conversation="web:s1")
    sm.owner_arrived(request_id="slack-r2", conversation="slack:t1:u1")
    assert sm._answers["web-r2"] == "web-r1" and sm._answers["slack-r2"] == "slack-r1"


def test_the_agent_keys_the_conversation_by_session_and_member():
    from ghost_agent.utils.logging import requester_member_context, slot_session_context
    tok = slot_session_context.set("sess-1")
    try:
        with _req(rid="web-1", conv="fp-x"):
            assert A._mood_conversation() == "sess-1"
        tok_m = requester_member_context.set("U-MARKOS")
        try:
            with _req(rid="slack-m", conv="fp-x", role="member"):
                assert A._mood_conversation() == "member:U-MARKOS:sess-1"
        finally:
            requester_member_context.reset(tok_m)
        with _req(rid="slack-m", conv="fp-x", role="member"):
            assert A._mood_conversation() == ""                          # an unknown member: no memo
    finally:
        slot_session_context.reset(tok)
    with _req(rid="web-2", conv="fp-y"):
        assert A._mood_conversation() == "fp-y"


async def test_the_public_timeline_leaves_out_the_manner_rows(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    sm.state.set_manner("formal", 2)
    rows = sm.state.mood_timeline()
    assert [r["kind"] for r in rows][-1] == M.KIND_MANNER
    with _req(rid="slack-elena-9", surface="public"):
        public = await tool_introspect(action="mood", self_model=sm)
    private = await tool_introspect(action="mood", self_model=sm)
    assert f"Today: {I.MOOD_STEPS_PRIVATE}." in public and "Today: " + I.MOOD_STEPS_PRIVATE not in private
    assert "irritable 2" not in public and "irritable 2" in private          # r4: no rows of the owner's pair


def test_a_share_of_the_other_sign_is_not_taken_out(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.5, 0.0, "good turns", M.KIND_TURN, now=T0)
    s.nudge(-0.08, -0.02, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    own, others = s.mood(now=T0), s.mood(now=T0, audience="others")
    assert others.p == pytest.approx(own.p)                    # a negative share over +0.42: never stronger
    assert (own.e, others.e) == (pytest.approx(-0.02), 0.0)      # the same-sign energy share does come out


def test_a_tone_taken_out_of_the_private_share_is_taken_out_once(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.owner_arrived(request_id="r2", now=T0)
    sm.note_owner_sentiment("negative", 0.95, request_id="r2", now=T0)
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)          # a third reaction, same reply
    assert (sm.state._state.mood.private_p, sm.state._state.mood.private_e) == (0.0, 0.0)


# ── I. review round 3 ───────────────────────────────────────────────────────

def test_setting_the_manner_does_not_void_the_ledger(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)
    sm.state.set_manner("terse", 2, now=T0)                                    # "be terse": the pair stays
    after = sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=T0)
    assert after.p == pytest.approx(M.reaction_event(M.KIND_THUMB_UP).dp, abs=1e-3)


def test_a_weaker_reaction_does_not_bring_a_worn_off_stronger_one_back(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)
    sm.owner_arrived(request_id="r2", now=_at(hours=4))
    before = sm.mood(now=_at(hours=4)).p
    after = sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2", now=_at(hours=4))
    moved = after.p - before
    assert moved == pytest.approx(M.reaction_event(M.KIND_CORRECTION).dp - before, abs=1e-3)
    assert abs(moved) <= abs(M.reaction_event(M.KIND_CORRECTION).dp) + 1e-9


def test_others_get_a_fresh_read_not_the_owners_level_held_by_the_margin(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(-0.17, 0.0, "trouble", M.KIND_TURN, now=T0)
    s.nudge(-0.08, -0.02, "the owner's message sounded unhappy", M.KIND_SENTIMENT, now=T0)
    assert (s.mood(now=T0).label, s.mood(now=T0).intensity) == ("down", 1)
    others = s.mood(now=T0, audience="others")
    assert others.p == pytest.approx(-0.17) and others.neutral
    assert M.read_off(others.p, others.e)[:2] == ("neutral", 0)


async def test_a_step_down_on_a_mood_the_owner_set_keeps_his_hold(tmp_path, owner):
    sm = _sm(tmp_path)
    await _set(sm, mood="cheerful", intensity=3)
    held = sm.mood().held_until
    await _set(sm, mood="cheerful", step=-1)                             # "a bit less cheerful"
    v = sm.mood()
    assert (v.label, v.intensity) == ("cheerful", 2) and v.held_until == held


def test_a_thumb_changed_back_brings_the_tone_back_into_the_private_share(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.owner_arrived(request_id="r2", now=T0)
    sm.note_owner_sentiment("negative", 0.95, request_id="r2", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)         # the tone no longer counts
    assert sm.state._state.mood.private_p == 0.0
    sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=T0)           # it counts again
    assert sm.state._state.mood.private_p == pytest.approx(-0.08)
    others = sm.mood(now=T0, audience="others")
    assert others.p == pytest.approx(sm.mood(now=T0).p)       # a share of the other sign: never made stronger


def test_a_long_turn_of_mostly_excused_calls_is_a_slog():
    evs = M.turn_events(M.TurnSignals(tool_calls=30, excused=29))
    assert [(e.dp, e.de) for e in evs][0] == M.EV_LONG_TURN
    assert all(e.dp <= 0 for e in evs)


async def test_introspect_on_a_public_surface_shows_the_public_mood(tmp_path, owner):
    sm = _sm(tmp_path)
    for _ in range(3):
        sm.state.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT)
    assert (sm.mood().label, sm.mood().intensity) == ("down", 1)
    with _req(rid="slack-elena-9", surface="public"):
        out = await tool_introspect(action="mood", self_model=sm)
        stats = await tool_introspect(action="stats", self_model=sm)
    assert out.startswith("My mood now: neutral.") and "down 1" not in out
    assert "Mood: neutral" in stats
    private = await tool_introspect(action="mood", self_model=sm)
    assert private.startswith("My mood now: down, 1 of 3")


def test_others_get_a_fresh_label_too_not_the_owners_held_by_the_margin(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(-0.9, -0.52, "a terrible long turn", M.KIND_TURN, now=T0)
    s.nudge(0.0, 0.16, "a rest", M.KIND_TURN, now=T0)                      # still tired, held by the margin
    s.nudge(0.0, -0.08, "the owner's message", M.KIND_SENTIMENT, now=T0)
    own, others = s.mood(now=T0), s.mood(now=T0, audience="others")
    assert own.label == "tired"
    assert others.label == M.read_off(others.p, others.e)[0] == "down"


def test_a_mood_the_owner_set_without_a_hold_voids_the_ledger_too(tmp_path):
    sm = _sm(tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="r1", now=T0)
    sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, now=T0, hold=False)   # a step down
    base = sm.mood(now=T0).p
    after = sm.note_owner_reaction(M.KIND_THUMB_UP, reply_id="r1", now=T0)
    assert after.p - base == pytest.approx(M.reaction_event(M.KIND_THUMB_UP).dp, abs=1e-3)


# ── J. review round 4 ───────────────────────────────────────────────────────

def test_a_message_sent_while_the_previous_turn_runs_answers_that_turn(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_arrived(request_id="R0", conversation="slack:t1:u1")
    sm.owner_turn_ended(request_id="R0", conversation="slack:t1:u1")
    sm.owner_arrived(request_id="R1", conversation="slack:t1:u1")
    sm.owner_arrived(request_id="M2", conversation="slack:t1:u1")          # R1 is still running
    assert sm._answers["M2"] == "R1"
    sm.owner_turn_ended(request_id="R1", conversation="slack:t1:u1")
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="M2", now=T0)
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, reply_id="R1", now=T0)
    assert sm.mood(now=T0).p == pytest.approx(-0.30, abs=1e-3)              # one verdict, not two
    sm.owner_turn_ended(request_id="M2", conversation="slack:t1:u1")
    sm.owner_arrived(request_id="M3", conversation="slack:t1:u1")
    assert sm._answers["M3"] == "M2"


def test_the_owners_tone_counts_as_private_only_as_far_as_it_moved_the_pair(tmp_path):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.99, 0.0, "a terrible turn", M.KIND_TURN, now=T0)
    sm.note_owner_sentiment("negative", 0.95, request_id="r1", now=T0)
    moved = sm.state._state.mood.p + 0.99
    assert sm.state._state.mood.private_p == pytest.approx(moved, abs=1e-3)
    assert sm.mood(now=T0, audience="others").p == pytest.approx(-0.99, abs=1e-3)


def test_any_share_gives_others_a_fresh_read(tmp_path):
    """r6 (r4/r5 kept the owner's held level for a small share; others then heard it rise while the mood
    decayed): with any share taken out others get a fresh read — they may step down a little before the
    owner, never hear more."""
    s = SelfStateThread(tmp_path)
    s.nudge(-0.9, -0.52, "a terrible long turn", M.KIND_TURN, now=T0)
    s.nudge(0.0, 0.18, "a rest", M.KIND_TURN, now=T0)
    s.nudge(0.0, -0.01, "the owner's message", M.KIND_SENTIMENT, now=T0)
    own, others = s.mood(now=T0), s.mood(now=T0, audience="others")
    assert own.label == "tired" and M.read_off(own.p, own.e)[0] == "down"          # held by the margin
    assert (others.label, others.intensity) == M.read_off(others.p, others.e)[:2]
    assert others.label == "down"


async def test_a_step_down_on_a_held_mood_keeps_the_hold_end(tmp_path, owner, monkeypatch):
    sm = _sm(tmp_path)
    await _set(sm, mood="cheerful", intensity=3)
    end = sm.mood().held_until
    # at 03:30 a NEW hold would run to the next day's 04:00
    monkeypatch.setattr(SelfStateThread, "_hold_end", staticmethod(lambda at: "2099-01-01T03:00:00Z"))
    await _set(sm, mood="cheerful", step=-1)
    assert sm.mood().intensity == 2 and sm.mood().held_until == end


def test_the_state_keeps_the_hold_end_it_is_given(tmp_path):
    s = SelfStateThread(tmp_path)
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 3, now=T0)
    end = s._state.mood.held_until
    late = M.parse_iso_utc(end) - dt.timedelta(minutes=30)               # half an hour before the hold ends
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, now=late, hold_until=end)
    assert s._state.mood.held_until == end
    s.set_owner_mood("cheerful", M.ANCHOR_DIRECTION["cheerful"], 2, now=late)
    assert s._state.mood.held_until != end                                # a new hold: the next day's end


def test_turns_that_end_out_of_order_answer_the_reply_that_came_last(tmp_path):
    sm = _sm(tmp_path)
    c = "slack:t1:u1"
    sm.owner_arrived(request_id="R1", conversation=c)
    sm.owner_arrived(request_id="R2", conversation=c)
    sm.owner_turn_ended(request_id="R2", conversation=c)
    sm.owner_turn_ended(request_id="R1", conversation=c)                  # R1's reply is the last one shown
    sm.owner_arrived(request_id="R3", conversation=c)
    assert sm._answers["R3"] == "R1"


async def test_a_step_that_changed_nothing_is_not_the_owners_word(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    out = await _set(sm, mood="down", step=-1)
    assert str(out).startswith(I.STEP_NOTHING) and getattr(out, "world_changed", None) is False
    agent = _traj_agent(tmp_path / "a")
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "i1", "type": "function",
                "function": {"name": "introspect",
                             "arguments": json.dumps({"action": "set_mood", "mood": "down", "step": -1})}}]},
            {"role": "tool", "tool_call_id": "i1", "name": "introspect", "content": out}]
    assert agent._mood_turn_signals(msgs, ASK, OWNER_RID).owner_set is False


# ── K. review round 5 ───────────────────────────────────────────────────────

def test_a_crumb_that_alone_flips_the_fresh_read_is_taken_out(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(-0.9, -0.52, "a terrible long turn", M.KIND_TURN, now=T0)
    s.nudge(0.0, 0.16, "a rest", M.KIND_TURN, now=T0)
    s.nudge(0.0, -0.01, "the owner's message", M.KIND_SENTIMENT, now=T0)
    own, others = s.mood(now=T0), s.mood(now=T0, audience="others")
    assert M.read_off(own.p, own.e)[0] == "tired" and M.read_off(others.p, others.e)[0] == "down"
    assert others.label == "down" and others.set_at == ""                  # not the owner's moment either


async def test_a_mixed_step_that_changed_the_mood_is_the_owners_word(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    out = await _set(sm, mood="irritable", manner="terse", step=-1)      # no manner is on
    assert str(out).startswith("Mood set to 'irritable', 1 of 3") and str(out).rstrip(".").endswith(
        f"{I.STEP_NOTHING}: the manner is not 'terse'")
    assert getattr(out, "world_changed", True) is not False
    agent = _traj_agent(tmp_path / "a")
    msgs = [{"role": "user", "content": ASK},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "i1", "type": "function",
                "function": {"name": "introspect", "arguments": json.dumps(
                    {"action": "set_mood", "mood": "irritable", "manner": "terse", "step": -1})}}]},
            {"role": "tool", "tool_call_id": "i1", "name": "introspect", "content": out}]
    assert agent._mood_turn_signals(msgs, ASK, OWNER_RID).owner_set is True


def test_the_private_share_booked_is_what_the_state_stored(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.state.nudge(-0.95, 0.0, "a terrible turn", M.KIND_TURN, now=T0)
    sm.owner_arrived(request_id="r2", now=T0)
    sm.note_owner_sentiment("negative", 0.95, request_id="r2", now=T0)   # moves only -0.05
    assert sm.state._state.mood.private_p == pytest.approx(-0.05, abs=1e-3)
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2", now=T0)   # the tone no longer counts
    assert sm.state._state.mood.private_p == pytest.approx(0.0, abs=1e-6)


def test_the_tool_description_teaches_a_step_for_a_relative_ask():
    from ghost_agent.tools import registry as R
    d = next(t["function"] for t in R.TOOL_DEFINITIONS if t["function"]["name"] == "introspect")
    # r6: the mood is NAMED — a bare step would move a manner that is on instead
    assert "don't be so grumpy\" → mood='irritable', step=-1" in d["description"]


# ── L. review round 6 ───────────────────────────────────────────────────────

def test_others_never_hear_a_level_only_the_owners_share_reached_as_it_decays(tmp_path):
    s = SelfStateThread(tmp_path)
    s.nudge(0.08, 0.02, "the owner's message sounded pleased", M.KIND_SENTIMENT, now=T0)
    later = _at(hours=4)
    s.nudge(0.43, 0.10, "good turns", M.KIND_TURN, now=later)
    for minutes in range(0, 30, 3):
        now = later + dt.timedelta(minutes=minutes)
        own, others = s.mood(now=now), s.mood(now=now, audience="others")
        assert others.intensity <= own.intensity
        assert (others.label, others.intensity) == M.read_off(others.p, others.e)[:2]


async def test_a_grumpy_ask_steps_the_mood_not_the_manner(tmp_path, owner):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, 0.25, "trouble", M.KIND_TURN)
    await _set(sm, manner="sarcastic", manner_intensity=2)
    await _set(sm, mood="irritable", step=-1)                            # the example the description teaches
    v = sm.mood()
    assert (v.label, v.intensity, v.manner, v.manner_intensity) == ("irritable", 1, "sarcastic", 2)


async def test_neutral_on_a_neutral_mood_is_still_the_owners_word(tmp_path, owner):
    sm = _sm(tmp_path)
    out = await _set(sm, mood="neutral")
    assert getattr(out, "world_changed", True) is not False


def test_a_second_tone_on_one_reply_at_the_clamp_is_booked_as_the_state_stored_it(tmp_path):
    sm = _sm(tmp_path)
    sm.owner_turn_ended(request_id="r1", now=T0)
    sm.state.nudge(-0.95, 0.0, "a terrible turn", M.KIND_TURN, now=T0)
    sm.owner_arrived(request_id="r2", now=T0)
    sm.note_owner_sentiment("positive", 0.95, request_id="r2", now=T0)
    sm.note_owner_sentiment("negative", 0.95, request_id="r2", now=T0)   # replaces it; the clamp absorbs part
    stored = sm.state._state.mood.private_p
    assert sm._private_booked["r1"][0] == pytest.approx(stored, abs=1e-4)
    sm.note_owner_reaction(M.KIND_CORRECTION, request_id="r2", now=T0)   # the tone no longer counts
    assert sm.state._state.mood.private_p == pytest.approx(0.0, abs=1e-4)


# ── M. review round 7 ───────────────────────────────────────────────────────

async def test_a_step_down_does_not_hand_others_a_mood_the_owners_share_made(tmp_path, owner):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.12, 0.08, "1 tool call failed in this turn", M.KIND_TURN)
    for _ in range(6):
        sm.state.nudge(*M.EV_SENTIMENT_NEGATIVE, "the owner's message sounded unhappy", M.KIND_SENTIMENT)
    assert (sm.mood().label, sm.mood().intensity) == ("down", 2)
    assert sm.mood(audience="others").neutral
    old_p, old_share = sm.state._state.mood.p, sm.state._state.mood.private_p
    await _set(sm, mood="down", step=-1)                                 # "don't be so down"
    assert (sm.mood().label, sm.mood().intensity) == ("down", 1)
    assert sm.mood(audience="others").neutral                            # not raised by the step
    new_p = sm.state._state.mood.p
    assert sm.state._state.mood.private_p == pytest.approx(old_share * new_p / old_p, abs=2e-3)   # the same PART


async def test_an_explicit_set_is_the_owners_word_and_clears_the_share(tmp_path, owner):
    sm = _sm(tmp_path)
    for _ in range(3):
        sm.state.nudge(*M.EV_SENTIMENT_NEGATIVE, "unhappy", M.KIND_SENTIMENT)
    await _set(sm, mood="down", intensity=1)
    assert (sm.state._state.mood.private_p, sm.state._state.mood.private_e) == (0.0, 0.0)


async def test_less_grumpy_while_tired_steps_the_tired_mood_down(tmp_path, owner):
    sm = _sm(tmp_path)
    sm.state.nudge(-0.5, -0.30, "a slog", M.KIND_TURN)
    assert (sm.mood().label, sm.mood().intensity) == ("tired", 2)
    await _set(sm, mood="irritable", step=-1)                            # the example the description teaches
    assert (sm.mood().label, sm.mood().intensity) == ("tired", 1)


async def test_less_grumpy_while_cheerful_changes_nothing(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    sm.state.nudge(0.5, 0.25, "good turns", M.KIND_TURN)
    before = (sm.mood().label, sm.mood().intensity)
    out = await _set(sm, mood="irritable", step=-1)
    assert str(out).startswith(I.STEP_NOTHING) and (sm.mood().label, sm.mood().intensity) == before


# ── N. review round 8 ───────────────────────────────────────────────────────

async def test_less_calm_while_cheerful_changes_nothing(tmp_path, owner):
    from ghost_agent.tools import introspect as I
    sm = _sm(tmp_path)
    sm.state.nudge(0.5, 0.3, "good turns", M.KIND_TURN)
    before = (sm.mood().label, sm.mood().intensity)
    assert before[0] == "cheerful"
    out = await _set(sm, mood="calm", step=-1)                           # the fallback is for unpleasant moods only
    assert str(out).startswith(I.STEP_NOTHING) and (sm.mood().label, sm.mood().intensity) == before
