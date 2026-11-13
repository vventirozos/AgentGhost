"""§4PS: live 15f54e1b — "forget yt-skVFTeB4gEk.captions.el": the model
previewed, then confirmed in the same turn; the gate refused ("the user has
not answered yet") and the system answered with the preview's question —
the designed behaviour — yet the stream printed "Execution Fail" and
"turn outcome failed". The forget / reset_all confirm refusals are a
DESIGNED stop, like the project's `confirm_dead_end`: no strike, no failed
turn, no corpus failure flag.

§4UL: the §4PN takeover that produced `forget_wait_shown` is gone with
§4UK's previews; the codes stay designed stops for rows already recorded,
and a `forget_not_confirmed` wall refusal is the model's to explain."""
from __future__ import annotations

from ghost_agent.distill.outcome_heuristics import (DESIGNED_STOP_REASONS as H, is_designed_stop,
                                                    tool_failure_flags)
from ghost_agent.tools.outcome import DESIGNED_STOP_REASONS as T, ToolOutcome as TO
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.utils.logging import request_id_context

WAIT = "NOT deleted: the user has not answered yet — show them the list and wait for their reply."


async def _batches(batches, tools, rid="web-4ps-1"):
    """Run each batch through the REAL dispatch, sharing one turn's rows."""
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    agent.available_tools = tools
    from ghost_agent.utils.logging import conversation_key_context
    tok = request_id_context.set(rid)
    ck = conversation_key_context.set("conv-4ps")
    rows = []
    try:
        for calls in batches:
            ts = H._ts(calls, StrikeLedger(), set())
            ts.tools_run_this_turn[:] = rows
            await agent._dispatch_and_process_tool_batch(ts)
            rows = list(ts.tools_run_this_turn)
            if ts.force_stop:
                break
    finally:
        conversation_key_context.reset(ck)
        request_id_context.reset(tok)
    return ts


def test_only_an_answered_wait_is_a_designed_stop():
    assert T == H
    for code in ("forget_wait_shown", "reset_wait_shown"):
        assert is_designed_stop(TO.rejected(WAIT, reason_code=code))
    # r1 MAJOR: the gate's code itself covers refusals the MODEL must explain
    # (a narrowed yes, outside content read) — those stay failures
    for code in ("forget_not_confirmed", "reset_not_confirmed", "forget_bad_selection"):
        assert not is_designed_stop(TO.rejected("NOT deleted: x", reason_code=code))


def test_the_corpus_reads_no_failure_for_an_answered_wait_only():
    shown = TO.rejected(WAIT, reason_code="forget_wait_shown")
    assert tool_failure_flags([{"name": "knowledge_base", "content": shown}]) == []
    assert is_designed_stop(WAIT + "\n[designed stop: forget_wait_shown]")
    narrowed = TO.rejected("NOT deleted: the user's yes carries a condition.", reason_code="forget_not_confirmed")
    assert tool_failure_flags([{"name": "knowledge_base", "content": narrowed}]) == [True]


def test_the_census_class_is_asked_user():
    from ghost_agent.core.failure_census import tool_class
    out = TO.rejected(WAIT, reason_code="forget_wait_shown")
    assert tool_class({"name": "knowledge_base", "content": out}) == "asked_user"
    assert tool_class({"name": "knowledge_base",
                       "result": WAIT + "\n[designed stop: forget_wait_shown]"}) == "asked_user"


async def test_a_wall_refused_forget_is_the_models_to_explain():
    """§4UL: no system takeover is left — a forget a wall refused keeps its
    failure flag and its strike, and the model writes the reply."""
    async def kb(**kw):
        return TO.rejected("NOT deleted: a background turn may not delete. That list is withdrawn.",
                           reason_code="forget_not_confirmed")
    ts = await _batches([[("knowledge_base", {"action": "forget", "confirm": "aaaa1111", "items": "all"})]],
                        {"knowledge_base": kb})
    assert not ts.force_stop and ts.execution_failure_count == 1
    assert ts.tools_run_this_turn[-1]["content"].reason_code == "forget_not_confirmed"


async def test_any_designed_stop_is_not_a_failed_last_call():
    """r1 m2, the project side: a `confirm_dead_end` (the project's own wait)
    as the last call does not make an earlier strike terminal."""
    async def mp(**kw):
        return TO.rejected("Error: already previewed in this turn — STOP calling delete in this turn.",
                           world_changed=False, reason_code="confirm_dead_end")
    ts = await _batches([[("manage_projects", {"action": "delete", "project_id": "p1"})]],
                        {"manage_projects": mp})
    assert ts.execution_failure_count == 0 and ts.last_was_failure is False


