"""§4PS: live 15f54e1b — "forget yt-skVFTeB4gEk.captions.el": the model
previewed, then confirmed in the same turn; the gate refused ("the user has
not answered yet") and the system answered with the preview's question —
the designed behaviour — yet the stream printed "Execution Fail" and
"turn outcome failed". The forget / reset_all confirm refusals are a
DESIGNED stop, like the project's `confirm_dead_end`: no strike, no failed
turn, no corpus failure flag."""
from __future__ import annotations

from ghost_agent.distill.outcome_heuristics import (DESIGNED_STOP_REASONS as H, is_designed_stop,
                                                    tool_failure_flags)
from ghost_agent.tools.outcome import DESIGNED_STOP_REASONS as T, ToolOutcome as TO
from tests.test_4pn_confirm_wait import _batches, real_kb  # noqa: F401 — fixture

WAIT = "NOT deleted: the user has not answered yet — show them the list and wait for their reply."


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


async def test_a_wait_the_takeover_declines_stays_a_strike():
    """r1 MAJOR: two previews → the model writes the reply (it said "deleted"
    3/3 in §4PN), so the refusal keeps its failure flag and its strike."""
    from tests.test_4pn_confirm_wait import _pv
    a = _pv("A", main=("fact 'a'",)).replace("tok1", "aaaa1111")
    b = _pv("B", main=("fact 'b'",)).replace("tok1", "bbbb2222")

    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected(WAIT, reason_code="forget_not_confirmed")
        return a if kw.get("target") == "A" else b
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "A"}),
                          ("knowledge_base", {"action": "forget", "target": "B"})],
                         [("knowledge_base", {"action": "forget", "confirm": "aaaa1111", "items": "all"})]],
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


