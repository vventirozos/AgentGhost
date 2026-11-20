"""§4UW (2026-11-18): the fixes of the character / mood / tone review.

* work done through an autoadvance batch reaches the mood (tasks its task turns closed, tasks they left FAILED);
* a turn the record files FAILED earns no reward and weighs at least one unit; a turn cut from outside keeps
  the trouble it already had and the tasks it finished, and nothing else;
* the correction judge is also reached by a message the emotion model read as displeased — for the mood only;
* the bound of intensity 3 is 0.90 (a good afternoon reads 2); the pair stays a plain sum;
* the reply that reports a finished task carries its pride; a shade set while a reply is being worked on is
  heard by that reply;
* the lift of a return is not given to a message that opens like a correction.

All owner data here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill import user_correction as U
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import days as D
from ghost_agent.selfhood import mood as M
from ghost_agent.selfhood import sentiment as SENT
from ghost_agent.selfhood import shade as S
from ghost_agent.selfhood.state import SelfStateThread
from ghost_agent.tools.outcome import ToolOutcome
from ghost_agent.utils.logging import (
    reply_surface_context, request_id_context, request_origin_context, requester_role_context)

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 11, 18, 10, 0, 0, tzinfo=UTC)
RID = "web-4uw"
ASK = "carry on with Bruno's vet-visit tracker"


class _req:
    def __init__(self, rid=RID, role="", surface="", origin=""):
        self.v = [(request_id_context, rid), (requester_role_context, role), (reply_surface_context, surface),
                  (request_origin_context, origin)]

    def __enter__(self):
        self.t = [(var, var.set(val)) for var, val in self.v]

    def __exit__(self, *exc):
        for var, tok in reversed(self.t):
            var.reset(tok)


def _agent(tmp_path) -> GhostAgent:
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.trajectory_collector = MagicMock()
    agent.context.trajectory_collector.append = MagicMock(return_value=None)
    agent.context._ctx_pressure_lockdown = False
    agent.context._guard_events = None
    agent.context.turn_origin_label = None
    agent.context.owner_memory_isolated = False
    agent.context._replay_forbidden_tools = None
    agent.context.self_model = SelfModel(root=tmp_path)
    return agent


def _call(i, name="execute", args=None):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {"command": "ls"})}}


def _turn(results, ask=ASK):
    """One turn's messages: the request, then one assistant call + tool row per (name, content[, args])."""
    msgs = [{"role": "user", "content": ask}]
    for i, r in enumerate(results):
        name, content, args = (r + (None,))[:3]
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_call(i, name, args)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": content})
    return msgs


OK = ToolOutcome.ok("EXIT CODE: 0\nSTDOUT/STDERR:\nfine")
BAD = ToolOutcome.failed("EXIT CODE: 1\nSTDOUT/STDERR:\nTraceback (most recent call last):\nValueError: x")
ADVANCE = {"action": "autoadvance", "count": "all"}


def _batch(statuses, stop="completed"):
    return json.dumps({"advanced": [{"task_id": f"t{i}", "classification": "coding", "status": s}
                                    for i, s in enumerate(statuses)],
                       "count": len(statuses), "ran": len(statuses), "requested": "all", "stop_reason": stop,
                       "agent_instruction": "report what was done"})


def _record(agent, msgs, rid=RID, **kw):
    return GhostAgent._record_turn_trajectory(
        agent, messages=msgs, final_content="done", req_id=rid, model="m", user_request=ASK, **kw)


def _events(agent):
    return agent.context.self_model.state.mood_events()


# ── the bound of 3 ──────────────────────────────────────────────────────────

def test_three_is_for_when_nearly_everything_went_one_way():
    assert M.INTENSITY_THRESHOLDS == (0.20, 0.45, 0.90)
    assert M.level_for(0.89) == 2 and M.level_for(0.90) == 3
    assert M.level_for(0.87, held=3) == 3 and M.level_for(0.85, held=3) == 2        # held down to 0.86
    # the owner's "3" starts clearly inside the band, and stays 3 while it is held
    for label, direction in M.ANCHOR_DIRECTION.items():
        p, e = M.anchor_for(direction, 3)
        assert M.level_for(max(abs(p), abs(e))) == 3 and max(abs(p), abs(e)) == pytest.approx(0.96), label
    assert [M.level_for(max(abs(v) for v in M.anchor_for((1.0, 0.0), n))) for n in (1, 2, 3)] == [1, 2, 3]


def test_a_good_afternoon_reads_two_and_the_automatic_mood_still_reaches_three(tmp_path):
    """Live 11-16: tasks finished (+0.31), a tone (+0.08), a thumb, a thumb, thanks — 11 of 35 replies heard
    "3 of 3". The pair is a plain sum (the reaction ledger takes out exactly what went in)."""
    st = SelfStateThread(tmp_path)
    for size, kind in ((0.31, M.KIND_TURN), (0.25, M.KIND_THUMB_UP), (0.25, M.KIND_THUMB_UP)):
        st.nudge(size, 0.0, "x", kind, now=T0)
    assert st.state.mood.p == pytest.approx(0.81) and st.mood(now=T0).intensity == 2
    st.nudge(0.10, 0.0, "the owner said thanks", M.KIND_PRAISE, now=T0)
    assert st.state.mood.p == pytest.approx(0.91) and st.mood(now=T0).intensity == 3
    assert "moved" not in st.mood_events()[-1]                      # the size is what moved
    st.nudge(0.25, 0.0, "x", M.KIND_THUMB_UP, now=T0)               # … and the limit shows as what it absorbed
    assert st.mood_events()[-1]["moved"] == [0.09, 0.0]


# ── what one turn's events are ──────────────────────────────────────────────

def test_a_failed_project_task_is_a_unit_of_trouble_each_and_ends_the_reward():
    evs = M.turn_events(M.TurnSignals(tool_calls=1, tasks_done=2, tasks_failed=2))
    assert [(e.dp, e.de, e.cause) for e in evs] == [
        (2 * M.EV_TROUBLE[0], 2 * M.EV_TROUBLE[1], "2 project tasks failed in this turn"),
        (*M.EV_TASK_DONE, "finished 2 project tasks")]
    one = M.turn_events(M.TurnSignals(tool_calls=1, tasks_failed=1))
    assert [e.cause for e in one] == ["1 project task failed in this turn"]
    assert M.turn_events(M.TurnSignals(tool_calls=1, tasks_failed=9))[0].dp == pytest.approx(
        M.TROUBLE_CAP * M.EV_TROUBLE[0])


def test_a_turn_filed_failed_earns_nothing_and_weighs_at_least_one_unit():
    clean = M.turn_events(M.TurnSignals(tool_calls=6))
    assert [e.dp for e in clean] == [0.16]
    failed = M.turn_events(M.TurnSignals(tool_calls=6, failed=True))
    assert [(e.dp, e.de, e.cause) for e in failed] == [(*M.EV_TROUBLE, "the request as a whole failed in this turn")]
    # one failed call of 59 was a quarter of a unit and a reward beside it (11-14: +0.06); filed FAILED it is one
    few = M.turn_events(M.TurnSignals(tool_calls=59, failures=1, failed=True))
    assert [(e.dp, e.cause) for e in few] == [
        (M.EV_TROUBLE[0], "1 tool call failed (of 59) and the request as a whole failed in this turn"),
        (M.EV_LONG_TURN[0], "a very long turn (0 min, 59 tool calls)")]
    # trouble that already weighs more than a unit is not counted again
    many = M.turn_events(M.TurnSignals(tool_calls=4, failures=3, failed=True))
    assert many[0].dp == pytest.approx(3 * M.EV_TROUBLE[0])


def test_a_cut_turn_keeps_its_trouble_and_its_finished_tasks_and_nothing_else():
    cut = M.TurnSignals(tool_calls=26, failures=9, sources=5, duration_s=2400.0, tasks_done=1, completed=False,
                        failed=True)
    evs = M.turn_events(cut)
    assert [e.cause for e in evs] == ["9 tool calls failed (of 26) in this turn", "finished 1 project task",
                                      M.CUT_SHORT_CAUSE]
    assert evs[0].dp < -0.3 and (evs[2].dp, evs[2].de) == (0.0, 0.0)
    assert not any(e.novelty for e in evs)                                       # no research lift either
    # going fine when it was cut: no reward, no blame, no row
    assert M.turn_events(M.TurnSignals(tool_calls=30, duration_s=2400.0, completed=False)) == []
    assert M.turn_events(M.TurnSignals(tool_calls=5, completed=False, failed=True)) == []
    whole = M.merge_events(evs)
    assert whole.cause.endswith("; the turn was cut short") and whole.kind == M.KIND_TURN


def test_the_day_record_counts_the_new_sentences_as_what_they_are():
    for cause, want in [
        ("2 project tasks failed in this turn; finished 1 project task", {"trouble": 1, "tasks_done": 1}),
        ("the request as a whole failed in this turn", {"trouble": 1}),
        ("1 tool call failed (of 59) and the request as a whole failed in this turn; a very long turn "
         "(173 min, 59 tool calls)", {"trouble": 1, "long_turns": 1}),
        ("9 tool calls failed (of 26) in this turn; finished 1 project task; the turn was cut short",
         {"trouble": 1, "tasks_done": 1}),
    ]:
        got = {k: v for k, v in D._own_counts(cause).items() if v}
        assert got == want, cause


# ── an autoadvance batch reaches the mood ───────────────────────────────────

def test_what_a_manage_projects_result_says_about_tasks():
    f = A.mood_payload_tasks
    assert f({"closed": ["t1", "t2", "t1"]}, "task_update") == (2, 0)
    assert f(json.loads(_batch(["DONE", "FAILED", "DONE", "PENDING"])), "autoadvance") == (2, 1)
    assert f(json.loads(_batch(["done", " Failed "])), "autoadvance") == (1, 1)
    # one task counts once, by its last entry (a retry inside the batch)
    again = {"advanced": [{"task_id": "t1", "status": "FAILED"}, {"task_id": "t1", "status": "DONE"}],
             "stop_reason": "completed"}
    assert f(again, "autoadvance") == (1, 0)
    # with no action the shape decides: a batch has its entries AND a stop reason
    assert f(json.loads(_batch(["DONE"]))) == (1, 0) and f({"closed": ["t1"]}) == (1, 0)
    assert f({"advanced": [{"task_id": "t1", "status": "DONE"}]}) == (0, 0)
    # any other action, and anything that is not the tool's object: nothing
    assert f(json.loads(_batch(["DONE"])), "task_list") == (0, 0)
    assert f({"closed": ["t1"]}, "ledger") == (0, 0)
    for junk in (None, [], "DONE", {"advanced": "DONE", "stop_reason": "x"}, {"closed": "t1"},
                 {"advanced": ["DONE", 3, None], "stop_reason": "x"}):
        assert f(junk, "autoadvance") == (0, 0) and f(junk) == (0, 0)


def test_a_batch_that_closed_a_task_is_a_finished_task_and_its_pride(tmp_path):
    """Live 11-16 (req a211f0db): a 27-minute task turn built and closed a task; the mood booked "a very long
    turn (33 min, 1 tool call); a turn of 1 tool call went through" and no task."""
    agent = _agent(tmp_path)
    with _req():
        _record(agent, _turn([("manage_projects", ToolOutcome.ok(_batch(["DONE"])), ADVANCE)]), elapsed_s=1980.0)
    cause = _events(agent)[-1]["cause"]
    assert "finished 1 project task" in cause and "a very long turn (33 min" in cause
    assert agent.context.self_model.shade.current()["shade"] == S.PROUD


def test_a_batch_that_left_tasks_failed_is_their_trouble_not_one_failed_call(tmp_path):
    agent = _agent(tmp_path)
    row = ToolOutcome.partial(_batch(["DONE", "FAILED", "FAILED"], "completed_with_failures"),
                              reason_code="autoadvance_completed_with_failures")
    msgs = _turn([("manage_projects", row, ADVANCE)])
    with _req():
        sig = agent._mood_turn_signals(msgs, ASK, RID)
        assert (sig.tool_calls, sig.failures, sig.tasks_done, sig.tasks_failed) == (1, 0, 1, 2)
        _record(agent, msgs)
    ev = _events(agent)[-1]
    assert ev["cause"].startswith("2 project tasks failed") and "finished 1 project task" in ev["cause"]
    assert "went through" not in ev["cause"]
    assert ev["dp"] == pytest.approx(2 * M.EV_TROUBLE[0] + M.EV_TASK_DONE[0])


def test_a_failed_task_update_closes_nothing_and_a_long_result_is_read_whole(tmp_path):
    agent = _agent(tmp_path)
    bad = ToolOutcome.failed(json.dumps({"closed": ["t1"], "error": "refused"}))
    long = ToolOutcome.ok(json.dumps({"pad": "x" * 6000, **json.loads(_batch(["DONE", "DONE"]))}))
    with _req():
        assert agent._mood_turn_signals(
            _turn([("manage_projects", bad, {"action": "task_update"})]), ASK, RID).tasks_done == 0
        assert agent._mood_turn_signals(_turn([("manage_projects", long, ADVANCE)]), ASK, RID).tasks_done == 2


def test_the_wake_of_the_owners_job_counts_a_batch_too(tmp_path):
    agent = _agent(tmp_path)
    with _req("job-4uw"):
        _record(agent, _turn([("manage_projects", ToolOutcome.ok(_batch(["DONE", "DONE"])), ADVANCE)]),
                rid="job-4uw")
    assert [e["cause"] for e in _events(agent)] == ["finished 2 project tasks (in a background job)"]


@pytest.mark.parametrize("who", [{"rid": "probe-4uw"}, {"rid": "sched-4uw"}, {"rid": "sub-ptask-t1-ab12"},
                                 {"rid": "slack-4uw-m", "role": "member"}])
def test_nobody_elses_batch_moves_the_mood(tmp_path, who):
    agent = _agent(tmp_path)
    with _req(**who):
        _record(agent, _turn([("manage_projects", ToolOutcome.ok(_batch(["DONE"])), ADVANCE)]), rid=who["rid"])
        GhostAgent._record_aborted_turn(agent, req_id=who["rid"], reason="cancelled",
                                        messages=_turn([("execute", BAD)] * 4), user_request=ASK, model="m")
    assert agent.context.self_model.state.state.mood is None and _events(agent) == []


# ── failed and cut turns, through the turn record ───────────────────────────

def test_a_cut_turn_that_had_gone_wrong_lowers_the_mood(tmp_path):
    """Live 11-12: a turn cut after 9 failed calls of 26 moved nothing."""
    agent = _agent(tmp_path)
    with _req():
        GhostAgent._record_aborted_turn(agent, req_id=RID, reason="cancelled: the stop button",
                                        messages=_turn([("execute", BAD)] * 9 + [("execute", OK)] * 17),
                                        user_request=ASK, model="m", elapsed_s=2400.0)
    ev = _events(agent)[-1]
    assert ev["cause"] == "9 tool calls failed (of 26) in this turn; the turn was cut short"
    assert ev["dp"] < -0.3 and agent.context.self_model.mood(for_reply=True).label == "irritable"


def test_a_cut_turn_is_not_blamed_for_the_lockdown_or_the_label_of_the_cut(tmp_path):
    agent = _agent(tmp_path)
    agent.context._ctx_pressure_lockdown = True
    with _req():
        GhostAgent._record_aborted_turn(agent, req_id=RID, reason="cancelled: client disconnected",
                                        messages=_turn([("execute", OK)] * 6), user_request=ASK, model="m")
    assert _events(agent) == [] and agent.context.self_model.state.state.mood is None


def test_a_finished_turn_the_record_files_failed_earns_no_reward(tmp_path, monkeypatch):
    from ghost_agent.distill import outcome_heuristics as OH
    from ghost_agent.distill.schema import Outcome

    def _filed_failed(traj):
        traj.outcome, traj.failure_reason = Outcome.FAILED.value, "browser interact sequence aborted (failed goto)"
        return True
    good, bad = _agent(tmp_path / "good"), _agent(tmp_path / "bad")
    with _req():
        _record(good, _turn([("execute", OK)] * 6))
        monkeypatch.setattr(OH, "apply_chat_outcome_heuristics", _filed_failed)
        _record(bad, _turn([("execute", OK)] * 6))
    assert _events(good)[-1]["dp"] > 0
    assert [(e["dp"], e["cause"]) for e in _events(bad)] == [
        (M.EV_TROUBLE[0], "the request as a whole failed in this turn")]


# ── the correction judge is also reached by a displeased reading ────────────

def test_which_readings_sound_displeased():
    assert M.sounds_displeased({"disapproval": 0.77, "annoyance": 0.05}) is True
    for k in sorted(M.SENTIMENT_COLD):
        assert M.sounds_displeased({k: M.SENTIMENT_EMOTION_MIN}) and not M.sounds_displeased({k: 0.49})
    # the owner's own bad news, a warm message, no reading, junk: no
    for no in ({"sadness": 0.9, "fear": 0.8, "grief": 0.9}, {"gratitude": 0.99}, {}, None, "anger",
               {"anger": True}, {"anger": "0.9"}, ["anger"]):
        assert M.sounds_displeased(no) is False


def test_a_displeased_reading_lets_a_message_reach_the_judge_and_never_relabels_a_turn():
    prev = "I rebuilt the tracker page; the dashboard now lists every vet visit."
    for text in ("the dashboard is still empty", "there is no page at that address", "the numbers are off"):
        assert not U.is_correction_candidate(text, prev)
        assert U.is_correction_candidate(text, prev, displeased=True)
        # … for the MOOD only: with the judge's yes, the prior turn is still not promoted
        v = U.classify_user_correction(prev_user_request="rebuild the tracker page", prev_assistant_response=prev,
                                       current_user_text=text, contradicts=True)
        assert v.is_correction is False
    assert not U.is_correction_candidate("", prev, displeased=True)
    assert not U.is_correction_candidate(None, prev, displeased=True)


def test_the_reading_of_a_requests_message_is_kept_for_whoever_waits():
    async def go():
        assert await SENT.reading_for("web-4uw-none", wait_s=5) is None        # none expected: at once
        SENT.expect_reading("web-4uw-r")
        waiter = asyncio.ensure_future(SENT.reading_for("web-4uw-r", wait_s=5))
        await asyncio.sleep(0.12)
        assert not waiter.done()                                              # on its way: it waits
        SENT._note_reading("web-4uw-r", {"disapproval": 0.8})
        assert await waiter == {"disapproval": 0.8}
        SENT._note_reading("web-4uw-r", {"joy": 0.9})                           # a queued message's: not the request's
        assert await SENT.reading_for("web-4uw-r") == {"disapproval": 0.8}
        SENT._note_reading("web-4uw-unasked", {"anger": 0.9})                   # never expected: not kept
        assert await SENT.reading_for("web-4uw-unasked") is None
        SENT.expect_reading("web-4uw-slow")
        t0 = asyncio.get_event_loop().time()
        assert await asyncio.wait_for(SENT.reading_for("web-4uw-slow", wait_s=0.15), 5) is None   # … and gives up
        assert 0.1 < asyncio.get_event_loop().time() - t0 < 2.0
        SENT.expect_reading("web-4uw-off")
        SENT._note_reading("web-4uw-off", None)                                 # scored, no emotion model
        assert await SENT.reading_for("web-4uw-off", wait_s=5) is None
    asyncio.run(go())
    for i in range(SENT.READINGS_MAX + 20):
        SENT.expect_reading(f"web-4uw-many-{i}")
    assert len(SENT._READINGS) <= SENT.READINGS_MAX


def test_scoring_a_message_ends_the_wait_on_every_way_out(tmp_path, monkeypatch):
    async def go():
        async def answer(text, **k):
            return {"label": "negative", "score": 0.5, "probs": {}, "reason": "", "emotions": {"disapproval": 0.8}}

        async def nothing(text, **k):
            return None

        async def boom(text, **k):
            raise RuntimeError("down")
        for rid, fn, want in (("web-4uw-s1", answer, {"disapproval": 0.8}), ("web-4uw-s2", nothing, None),
                              ("web-4uw-s3", boom, None)):
            monkeypatch.setattr(SENT, "classify", fn)
            SENT.expect_reading(rid)
            await SENT.score_message("the tracker page is still empty", tmp_path, request_id=rid)
            assert await asyncio.wait_for(SENT.reading_for(rid, wait_s=30), 2) == want
        SENT.expect_reading("web-4uw-s4")
        await SENT.score_message("   ", tmp_path, request_id="web-4uw-s4")        # nothing to score
        assert await asyncio.wait_for(SENT.reading_for("web-4uw-s4", wait_s=30), 2) is None
    asyncio.run(go())


class _Judge:
    llm_client = object()

    def __init__(self, answer='{"corrects": true}'):
        self.calls, self.answer = 0, answer

    async def _call_llm(self, *a, **k):
        self.calls += 1
        return {"choices": [{"message": {"content": self.answer}}]}


def _judged(text, reading, rid="web-4uw-j"):
    judge = _Judge()
    a = GhostAgent.__new__(GhostAgent)
    a.context = type("C", (), {"verifier": judge, "trajectory_collector": None})()
    msgs = [{"role": "user", "content": "rebuild Bruno's tracker page"},
            {"role": "assistant", "content": "Done — the tracker page lists every vet visit now."},
            {"role": "user", "content": text}]

    async def go():
        with _req(rid):
            if reading is not _Judge:
                SENT.expect_reading(rid)
                SENT._note_reading(rid, reading)
            return await a._adjudicate_correction(msgs, text)
    return asyncio.run(go()), judge.calls


def test_the_judge_is_asked_about_a_message_that_reads_displeased():
    assert _judged("the tracker page is still empty", {"disappointment": 0.7}, "web-4uw-j1") == (True, 1)
    # no reading on its way, a reading that is not displeased, the owner's own bad news: the judge is not asked
    assert _judged("the tracker page is still empty", _Judge, "web-4uw-j2") == (None, 0)
    assert _judged("the tracker page is still empty", {"curiosity": 0.9}, "web-4uw-j3") == (None, 0)
    assert _judged("Bruno is limping again, I am worried", {"sadness": 0.9, "fear": 0.7}, "web-4uw-j4") == (None, 0)
    # the phrase gate still lets its own in without any reading
    assert _judged("no, that's wrong — the page is empty", _Judge, "web-4uw-j5") == (True, 1)


def test_the_request_asks_for_its_reading_before_the_scoring_starts(tmp_path, monkeypatch):
    """The judge's task is spawned right after the scoring: the slot must be there already."""
    seen = []
    agent = _agent(tmp_path)
    monkeypatch.setattr(SENT, "settings", lambda: ("http://nova.example:8097", "", 3.0))
    import ghost_agent.utils.logging as GL

    def spawn(coro, name=""):
        seen.append((name, "web-4uw-q" in SENT._READINGS, "web-4uw-q2" in SENT._READINGS))
        coro.close()
    monkeypatch.setattr(GL, "spawn_bg", spawn)
    monkeypatch.setattr(A, "_mood_queued_read", lambda *a, **k: True)
    with _req("web-4uw-q"):
        agent._mood_read_owner_words("the tracker page is still empty")
    with _req("web-4uw-q2"):
        agent._mood_read_owner_words("and the chart too", queued=True)          # a mid-turn message: no slot
    assert seen == [("sentiment", True, False), ("sentiment", True, False)]


# ── the reply that reports the work carries its shade ───────────────────────

def _home(tmp_path):
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "selfhood").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _ctx(home, sm=None):
    return SimpleNamespace(memory_dir=str(home / "system" / "memory"), turn_origin_label=None,
                           owner_memory_isolated=False, _replay_forbidden_tools=None, self_model=sm)


def test_a_tool_result_marks_the_request_that_closed_a_task():
    f = A.mood_note_task_result
    assert f("web-4uw-n1", "manage_projects", json.dumps({"closed": ["t1"]})) == 1
    assert f("web-4uw-n1", "manage_projects", _batch(["DONE", "DONE", "FAILED"])) == 2
    assert A.mood_tasks_done_so_far("web-4uw-n1") == 3
    # a note the loop appended after the tool's JSON is not part of it
    assert f("web-4uw-n2", "manage_projects", json.dumps({"closed": ["t1"]}) + "\n\n[NOTE: 2 tasks remain]") == 1
    for rid, name, text in (("web-4uw-n3", "file_system", json.dumps({"closed": ["t1"]})),
                            ("web-4uw-n3", "manage_projects", "closed: t1, advanced nicely"),
                            ("web-4uw-n3", "manage_projects", json.dumps({"closed": []})),
                            ("web-4uw-n3", "manage_projects", _batch(["FAILED", "PENDING"])),
                            ("web-4uw-n3", "manage_projects", None),
                            ("", "manage_projects", json.dumps({"closed": ["t1"]}))):
        assert f(rid, name, text) == 0
    assert A.mood_tasks_done_so_far("web-4uw-n3") == 0 and A.mood_tasks_done_so_far("") == 0


def test_the_reply_that_reports_a_finished_task_is_the_proud_one(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    with _req("web-4uw-p1"):
        assert A._mood_shade_block(ctx) == ""                       # step 1: nothing has happened yet
        A.mood_note_task_result("web-4uw-p1", "manage_projects", json.dumps({"closed": ["t1"]}))
        got = A._mood_shade_block(ctx)                              # the step after the task closed
        assert got == S.line(S.PROUD)
        assert A._mood_shade_block(ctx) == got                      # … and the same bytes to the request's end
        assert A.mood_shade_heard("web-4uw-p1") == S.PROUD
        # the end of the turn: the shade is set (the face, the day record) and is already heard
        sm.note_owner_turn(M.TurnSignals(tool_calls=4, tasks_done=1), shade_heard=A.mood_shade_heard("web-4uw-p1"))
    assert sm.shade.current()["shade"] == S.PROUD
    with _req("web-4uw-p2"):
        assert A._mood_shade_block(ctx) == ""                       # the next reply does not carry it again


def test_without_the_reply_having_heard_it_the_next_reply_does(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    sm.note_owner_turn(M.TurnSignals(tool_calls=4, tasks_done=1), shade_heard="")
    with _req("web-4uw-p3"):
        assert A._mood_shade_block(_ctx(home, sm)) == S.line(S.PROUD)
    # … and a reply that heard ANOTHER shade does not use the turn's own up
    sm2 = SelfModel(root=home / "system" / "selfhood2")
    sm2.note_owner_turn(M.TurnSignals(tool_calls=4, tasks_done=1), shade_heard=S.RELIEVED)
    assert sm2.shade.take() == S.PROUD


@pytest.mark.parametrize("who", [{"rid": "probe-4uw-p"}, {"rid": "slack-4uw-pm", "role": "member"},
                                 {"rid": "slack-4uw-pp", "surface": "public"}, {"rid": "sched-4uw-p"},
                                 {"rid": "sub-ptask-t1-cd34"}])
def test_only_the_owners_private_reply_is_proud_of_its_task(tmp_path, who):
    home = _home(tmp_path)
    ctx = _ctx(home, SelfModel(root=home / "system" / "selfhood"))
    with _req(**who):
        A.mood_note_task_result(who["rid"], "manage_projects", json.dumps({"closed": ["t1"]}))
        assert A._mood_shade_block(ctx) == "" and A.mood_shade_heard(who["rid"]) == ""


def test_no_pride_while_the_owners_set_mood_holds(tmp_path):
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    sm.state.set_owner_mood("down", M.ANCHOR_DIRECTION["down"], 2)
    assert sm.may_shade_now() is False
    with _req("web-4uw-h"):
        A.mood_note_task_result("web-4uw-h", "manage_projects", json.dumps({"closed": ["t1"]}))
        assert A._mood_shade_block(_ctx(home, sm)) == ""
    assert SelfModel(root=home / "system" / "fresh").may_shade_now() is True
    assert SelfModel(root=home / "system" / "off", enabled=False).may_shade_now() is False


def test_a_shade_set_while_the_reply_is_being_worked_on_is_heard_by_that_reply(tmp_path):
    """A correction is judged beside the turn: its "chastened" used to wait for the reply after."""
    home = _home(tmp_path)
    sm = SelfModel(root=home / "system" / "selfhood")
    ctx = _ctx(home, sm)
    with _req("web-4uw-c1"):
        assert A._mood_shade_block(ctx) == ""
        sm.shade.set(S.CHASTENED, "the owner corrected one of my replies")
        assert A._mood_shade_block(ctx) == S.line(S.CHASTENED)
        sm.shade.set(S.PROUD, "x")                                  # a later one does not replace the line heard
        assert A._mood_shade_block(ctx) == S.line(S.CHASTENED)
    with _req("web-4uw-c2"):
        assert A._mood_shade_block(ctx) == S.line(S.PROUD)          # … it is the next reply's
    # a shade waiting from before wins over this turn's own pride (one line a reply)
    sm.shade.set(S.RELIEVED, "x")
    with _req("web-4uw-c3"):
        A.mood_note_task_result("web-4uw-c3", "manage_projects", json.dumps({"closed": ["t9"]}))
        assert A._mood_shade_block(ctx) == S.line(S.RELIEVED)


# ── the owner back, and not pleased ─────────────────────────────────────────

def test_the_lift_of_a_return_is_not_given_to_a_complaint(tmp_path):
    back, cross = SelfModel(root=tmp_path / "a"), SelfModel(root=tmp_path / "b")
    for sm in (back, cross):
        sm.state.mark_owner_turn(now=T0 - dt.timedelta(hours=9))
    back.owner_arrived(now=T0)
    cross.owner_arrived(now=T0, complaint=True)
    assert back.state.mood(now=T0).label == "cheerful"
    assert cross.state.state.mood is None and cross.state.mood_events() == []
    # the quiet is over all the same: the next, ordinary message is no late lift
    cross.owner_arrived(now=T0 + dt.timedelta(minutes=2))
    assert cross.state.state.mood is None


@pytest.mark.parametrize("text,lift", [("good morning ghost, how are things?", True),
                                       ("no, that's wrong — Bruno's appointment is on Friday", False),
                                       ("you didn't send the reminder", False),
                                       ([{"type": "text", "text": "that is wrong, check the date"}], False),
                                       (None, True), ("", True)])
def test_the_agent_reads_the_arriving_message_for_a_complaint(tmp_path, text, lift):
    agent = _agent(tmp_path)
    st = agent.context.self_model.state
    st.mark_owner_turn(now=dt.datetime.now(UTC) - dt.timedelta(hours=9))
    with _req():
        agent._mood_owner_arrived(text)
    assert (st.state.mood is not None) is lift


# ── the journal says which part is the rest ─────────────────────────────────

def test_a_journal_row_names_the_resting_point_it_was_read_on(tmp_path):
    st = SelfStateThread(tmp_path / "rest")
    st.attach_rest(lambda: (0.06, 0.06))
    st.nudge(0.25, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP, now=T0)
    row = json.loads(st.mood_history_path.read_text().splitlines()[-1])
    assert row["p"] == pytest.approx(0.31) and row["rest"] == [0.06, 0.06]
    assert st.mood_history(limit=1)[0]["p"] == pytest.approx(0.31)      # the reader takes the row as before
    plain = SelfStateThread(tmp_path / "plain")
    plain.nudge(0.25, 0.0, "the owner gave a thumbs-up", M.KIND_THUMB_UP, now=T0)
    assert "rest" not in json.loads(plain.mood_history_path.read_text().splitlines()[-1])


# ── through the real turn loop ──────────────────────────────────────────────

def _said(call) -> str:
    """Everything one model call was sent, as text."""
    payload = call.kwargs.get("payload") if "payload" in call.kwargs else (call.args[0] if call.args else call.kwargs)
    return json.dumps(payload, default=str, ensure_ascii=False)


@pytest.mark.asyncio
async def test_the_turn_that_closes_a_task_reports_it_with_its_pride_and_the_next_reply_has_none(tmp_path, monkeypatch):
    """The whole path: the tool's own result marks the request in the dispatch loop, the next step's state
    block carries the line, the turn's end sets the shade already heard."""
    from unittest.mock import AsyncMock
    from tests.helpers import make_context, FakeBgTasks
    from tests.test_4ji_request_echo_narration import _resp, _tc
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Bruno tracker", goal="vet visits")
    tid = store.add_task(pid, "Collect the visit notes")
    store.add_task(pid, "Summarise the notes")
    ctx = make_context()
    ctx.project_store = store
    store.set_focus("session:s4uw", pid)
    ctx.self_model = SelfModel(root=tmp_path / "self")
    ctx.trajectory_collector = MagicMock()                    # the turn record carries the end-of-turn mood hook
    ctx.trajectory_collector.append = MagicMock(return_value=None)
    agent = GhostAgent(ctx)

    async def _mp(**kw):
        return await tool_manage_projects_for_model(ctx, **kw)

    agent.available_tools = {"manage_projects": _mp}
    final = "The visit notes are collected and the task is closed."
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("t0", "manage_projects",
                       {"action": "task_update", "task_id": tid, "status": "done", "result": "notes collected"})]),
    ] + [_resp(final)] * 4)
    await agent.handle_chat({"messages": [{"role": "user", "content": f"proceed with {pid}"}],
                             "session_id": "s4uw"}, FakeBgTasks())
    assert store.get_task(tid)["status"] == "DONE"
    calls = ctx.llm_client.chat_completion.call_args_list
    pride = "a quiet pride"
    assert pride not in _said(calls[0])                       # before the task closed: no line
    assert any(pride in _said(c) for c in calls[1:])          # the step that writes the report hears it
    assert ctx.self_model.shade.current()["shade"] == S.PROUD  # the turn's shade is set …
    assert ctx.self_model.shade.take() == ""                  # … and already heard: the next reply has none
    # the request is over: its id, sent again, has closed no task yet
    rid = next(c for c in ctx.trajectory_collector.append.call_args_list).args[0].extra.get("req_id")
    assert rid and A.mood_tasks_done_so_far(rid) == 0


# ── review r1: one reader of a task result, a turn read once ──

NOTE = "\n\n[CHECKLIST 1/3 passed · next: the chart renders]"


def test_the_end_of_the_turn_reads_a_task_result_the_way_the_loop_did(tmp_path):
    """The loop appends the open checklist line to a batch's last row: the reply heard the pride of a task the
    mood then never booked."""
    agent = _agent(tmp_path)
    closed = ToolOutcome.ok(json.dumps({"updated": [{"id": "t1", "status": "DONE"}], "closed": ["t1"]}) + NOTE)
    batch = ToolOutcome.partial(_batch(["DONE", "DONE", "FAILED"], "completed_with_failures") + NOTE,
                                reason_code="autoadvance_completed_with_failures")
    with _req():
        one = agent._mood_turn_signals(_turn([("execute", OK), ("manage_projects", closed, {"action": "task_update"})]),
                                       ASK, RID)
        assert one.tasks_done == 1 == A.mood_note_task_result("web-4uw-same", "manage_projects", str(closed))
        many = agent._mood_turn_signals(_turn([("manage_projects", batch, ADVANCE)]), ASK, RID)
        assert (many.tasks_done, many.tasks_failed, many.failures) == (2, 1, 0)
    assert A._mood_tool_json("not json") is None and A._mood_tool_json(None) is None


def test_a_turn_recorded_as_finished_and_then_as_cut_is_read_once(tmp_path):
    agent = _agent(tmp_path)
    msgs = _turn([("execute", BAD)] * 3 + [("execute", OK)])
    with _req("web-4uw-twice"):
        _record(agent, msgs, rid="web-4uw-twice")
        GhostAgent._record_aborted_turn(agent, req_id="web-4uw-twice", reason="cancelled: client disconnected",
                                        messages=msgs, user_request=ASK, model="m")
        assert len(_events(agent)) == 1                             # the bookkeeping was cut, not the turn
        _record(agent, msgs, rid="web-4uw-twice")                   # the same id sent again: a new turn
        assert len(_events(agent)) == 2
        agent._mood_owner_arrived("carry on")                       # … and sent once more, and this one is cut
        GhostAgent._record_aborted_turn(agent, req_id="web-4uw-twice", reason="cancelled: the stop button",
                                        messages=msgs, user_request=ASK, model="m")
        assert len(_events(agent)) == 3
    with _req("web-4uw-cut-only"):
        GhostAgent._record_aborted_turn(agent, req_id="web-4uw-cut-only", reason="cancelled",
                                        messages=msgs, user_request=ASK, model="m")
    assert len(_events(agent)) == 4


def test_a_request_id_sent_again_starts_with_no_reading(tmp_path, monkeypatch):
    agent = _agent(tmp_path)
    monkeypatch.setattr(SENT, "settings", lambda: ("", "", 3.0))                 # the service is off
    monkeypatch.setattr(A, "_mood_queued_read", lambda *a, **k: True)
    SENT.expect_reading("web-4uw-re")
    SENT._note_reading("web-4uw-re", {"disapproval": 0.9})
    with _req("web-4uw-re"):
        agent._mood_read_owner_words("and the chart?", queued=True)              # a mid-turn message keeps it
        assert asyncio.run(SENT.reading_for("web-4uw-re")) == {"disapproval": 0.9}
        agent._mood_read_owner_words("what time is Bruno's appointment?")
        assert asyncio.run(SENT.reading_for("web-4uw-re")) is None


def test_the_judges_gate_waits_for_a_reading_that_is_on_its_way():
    judge = _Judge()
    a = GhostAgent.__new__(GhostAgent)
    a.context = type("C", (), {"verifier": judge, "trajectory_collector": None})()
    text = "the tracker page is still empty"
    msgs = [{"role": "user", "content": "rebuild Bruno's tracker page"},
            {"role": "assistant", "content": "Done — the tracker page lists every vet visit now."},
            {"role": "user", "content": text}]

    async def go():
        with _req("web-4uw-late"):
            SENT.expect_reading("web-4uw-late")
            asyncio.get_event_loop().call_later(0.2, SENT._note_reading, "web-4uw-late", {"annoyance": 0.8})
            return await a._adjudicate_correction(msgs, text)
    assert asyncio.run(go()) is True and judge.calls == 1


def test_the_days_sums_are_what_moved_and_its_evidence_what_happened(tmp_path):
    root = tmp_path / "system" / "selfhood"
    root.mkdir(parents=True)
    rows = [{"at": "2026-11-17T10:00:00Z", "kind": "thumb_up", "dp": 0.25, "de": 0.0, "moved": [0.1, 0.0],
             "cause": "the owner gave a thumbs-up", "after": "content, 2 of 3"},
            {"at": "2026-11-17T11:00:00Z", "kind": "sentiment", "dp": 0.08, "de": 0.02, "moved": [0.0, 0.0],
             "cause": "the owner's message sounded upbeat", "after": "content, 2 of 3"},
            {"at": "2026-11-17T12:00:00Z", "kind": "praise", "dp": 0.1, "de": 0.0, "moved": "junk",
             "cause": "the owner said thanks", "after": "content, 2 of 3"},
            {"at": "2026-11-17T13:00:00Z", "kind": "return", "dp": 0.2, "de": 0.1, "moved": [float("inf"), 0.0],
             "cause": "the owner came back after 9 hours", "after": "content, 2 of 3"}]
    (root / "mood.events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = D.build_day(tmp_path, dt.date(2026, 11, 17), tz=UTC, now=dt.datetime(2026, 11, 18, 6, 0, tzinfo=UTC))
    assert rec["by_kind"]["thumb_up"]["dp"] == 0.1 and rec["by_kind"]["praise"]["dp"] == 0.1
    assert rec["by_kind"]["sentiment"]["dp"] == 0.0 and rec["tone_up"] == 1 and rec["praise"] == 1
    assert rec["by_kind"]["return"]["dp"] == 0.2                    # a number nobody can read: the size stands
    assert [b["kind"] for b in rec["biggest"]][0] == "return"


@pytest.mark.asyncio
@pytest.mark.parametrize("text,lifted", [("no, that's wrong — Bruno's appointment is on Friday", False),
                                         ("good morning, what is on today?", True)])
async def test_the_arriving_message_reaches_the_mood_through_the_real_request(tmp_path, monkeypatch, text, lifted):
    from unittest.mock import AsyncMock
    from tests.helpers import make_context, FakeBgTasks
    from tests.test_4ji_request_echo_narration import _resp
    for k, v in (("GHOST_CRITIC_ASYNC", "1"), ("GHOST_CRITIC_REPAIR_BUDGET", "0"), ("GHOST_EVIDENCE_GATE", "0")):
        monkeypatch.setenv(k, v)
    ctx = make_context()
    ctx.self_model = SelfModel(root=tmp_path / "self")
    ctx.self_model.state.mark_owner_turn(now=dt.datetime.now(UTC) - dt.timedelta(hours=9))
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Friday at ten, noted.")] * 4)
    await agent.handle_chat({"messages": [{"role": "user", "content": text}], "session_id": "s4uw-arr"},
                            FakeBgTasks())
    kinds = [e["kind"] for e in ctx.self_model.state.mood_events()]
    assert (M.KIND_RETURN in kinds) is lifted


# ── §4UW-b: an undo brings the ledger of the pair it puts back with it ──────

def test_a_thumb_from_before_a_set_mood_can_be_changed_after_its_undo(tmp_path):
    """Review r3: 👍, a set mood, its undo, then 👎 on the same reply left −0.05 (the 👍 stayed in the pair and
    the 👎 was added to it) where the changed thumb alone gives −0.30."""
    sm = SelfModel(root=tmp_path / "a")
    sm.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    assert sm.state.undo_owner_mood(now=T0) is not None
    assert sm.state.state.mood.p == pytest.approx(M.EV_THUMB_UP[0])
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=T0, reply_id="r1")
    assert sm.state.state.mood.p == pytest.approx(M.EV_THUMB_DOWN[0])
    # … or taken back
    sm2 = SelfModel(root=tmp_path / "b")
    sm2.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    sm2.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    sm2.state.undo_owner_mood(now=T0)
    sm2.retract_owner_reaction(True, now=T0, reply_id="r1")
    assert sm2.state.state.mood.p == pytest.approx(0.0, abs=1e-6)
    # the same thumb again is not a second thumb
    sm3 = SelfModel(root=tmp_path / "c")
    sm3.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    sm3.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    sm3.state.undo_owner_mood(now=T0)
    sm3.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    assert sm3.state.state.mood.p == pytest.approx(M.EV_THUMB_UP[0])


def test_without_an_undo_a_set_mood_still_voids_the_ledger(tmp_path):
    sm = SelfModel(root=tmp_path)
    sm.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0, hold=False)
    p = sm.state.state.mood.p
    assert sm.retract_owner_reaction(True, now=T0, reply_id="r1")[2] == sm.NOTHING_TO_TAKE_BACK
    sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=T0, reply_id="r1")            # a new verdict on the new pair
    assert sm.state.state.mood.p == pytest.approx(p + M.EV_THUMB_DOWN[0])
    # a "neutral" voids it too, and leaves nothing an undo could bring back
    sm.state.reset_mood(now=T0)
    assert sm.state.undo_owner_mood(now=T0) is None
    sm.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
    assert sm.state.state.mood.p == pytest.approx(M.EV_THUMB_UP[0])


def test_an_epoch_that_was_left_is_never_handed_out_again(tmp_path):
    """Set, undo, set: the second set's pair is not the first set's — nothing booked under the first counts."""
    st = SelfStateThread(tmp_path)
    seen = [st.pair_epoch]
    for _ in range(3):
        st.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
        during = st.pair_epoch
        st.undo_owner_mood(now=T0)
        assert st.pair_epoch == seen[0] and during not in seen
        seen.append(during)
    # set over a set: the undo goes back to the FIRST set's pair, not to the one before it
    st.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    first = st.pair_epoch
    st.set_owner_mood("down", M.ANCHOR_DIRECTION["down"], 2, now=T0)
    st.undo_owner_mood(now=T0)
    assert st.pair_epoch == first and st.state.mood.label == "calm"
    assert st.undo_owner_mood(now=T0) is None


def test_an_undo_does_not_bring_the_tone_books_back(tmp_path):
    """The tone shares stay out of what an undo puts back: a tone booked before the set holds nothing, so a
    thumb that replaces it must not "take out" a share the state no longer has."""
    a, b = SelfModel(root=tmp_path / "a"), SelfModel(root=tmp_path / "b")
    for sm, tone in ((a, True), (b, False)):
        sm.state.nudge(0.3, 0.0, "a turn went well", M.KIND_TURN, now=T0)
        if tone:
            sm._react(M.sentiment_event("positive", 0.95), now=T0, reply_id="r1")
        sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
        sm.state.undo_owner_mood(now=T0)
        sm.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
        sm.retract_owner_reaction(True, now=T0, reply_id="r1")
    ma, mb = a.state.state.mood, b.state.state.mood
    assert (ma.private_up_p, ma.private_dn_p) == (0.0, 0.0)
    assert ma.p == pytest.approx(mb.p, abs=1e-6) and mb.p == pytest.approx(0.3, abs=1e-6)
    assert a.state.tone_epoch == 2 and a.state.pair_epoch == 0


def test_an_undo_after_a_restart_starts_a_new_epoch(tmp_path):
    """The set is on file, the process that made it is gone, and so is its ledger: nothing to go back to."""
    st = SelfStateThread(tmp_path)
    st.nudge(0.3, 0.0, "a turn went well", M.KIND_TURN, now=T0)
    st.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 2, now=T0)
    again = SelfStateThread(tmp_path)
    assert again.pair_epoch == 0 and again.undo_owner_mood(now=T0) is not None
    assert again.pair_epoch == 1 and again.tone_epoch == 1 and again.state.mood.p == pytest.approx(0.3)


def test_a_neutral_or_a_set_voids_the_tone_books_too(tmp_path):
    """A tone booked before the owner replaced the pair holds no share of the new one: a thumb on that reply
    must not be netted against it, and the tones that came after are untouched."""
    for how in ("neutral", "set"):
        sm = SelfModel(root=tmp_path / how)
        sm._react(M.sentiment_event("positive", 0.95), now=T0, reply_id="r1")
        if how == "neutral":
            sm.state.reset_mood(now=T0)
        else:
            sm.state.set_owner_mood("calm", M.ANCHOR_DIRECTION["calm"], 1, now=T0, hold=False)
            sm.state.nudge(-sm.state.state.mood.p, -sm.state.state.mood.e, "x", M.KIND_TURN, now=T0)
        sm._react(M.sentiment_event("positive", 0.95), now=T0, reply_id="r2")
        base = sm.state.state.mood.p - sm.state.state.mood.private_up_p
        sm.note_owner_reaction(M.KIND_THUMB_UP, now=T0, reply_id="r1")
        m = sm.state.state.mood
        assert m.private_up_p == pytest.approx(M.EV_SENTIMENT_POSITIVE[0]), how      # r2's tone stays in
        assert m.p - m.private_up_p == pytest.approx(base + M.EV_THUMB_UP[0]), how


def test_a_neutral_over_a_pair_at_exactly_zero_starts_the_ledger_afresh_like_any_other(tmp_path):
    """Review: +0.2, a correction (−0.2): the pair is exactly 0 and "neutral" took an early way out that kept
    the ledger — the same history with a message tone in it took the other way and lost it."""
    ends = []
    for tone in (False, True):
        sm = SelfModel(root=tmp_path / str(tone))
        sm.state.nudge(0.2, 0.0, "a turn went well", M.KIND_TURN, now=T0)
        sm.note_owner_reaction(M.KIND_CORRECTION, now=T0, reply_id="r1")
        if tone:
            sm._react(M.sentiment_event("positive", 0.95), now=T0, reply_id="r2")
        sm.state.reset_mood(now=T0)
        sm.note_owner_reaction(M.KIND_THUMB_DOWN, now=T0, reply_id="r1")
        m = sm.state.state.mood
        ends.append(m.p - m.private_up_p + m.private_dn_p)
    assert ends[0] == pytest.approx(ends[1]) == pytest.approx(M.EV_THUMB_DOWN[0])
