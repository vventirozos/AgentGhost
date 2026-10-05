"""§4LW — the three open items: bad skill arguments, ambiguous document
names, retries after a dead-end refusal."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import request_id_context


# ── 1. the model's bad arguments are refused before the skill runs ──

_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}, "q": {"type": "string"}},
           "required": ["q"], "additionalProperties": False}


@pytest.mark.parametrize("args,why", [
    ({}, "missing required"), ({"q": "x", "zzz": 1}, "unknown argument"),
    ({"q": "x", "count": "ten"}, "must be a integer"), ({"q": "x", "count": True}, "must be a integer"),
    ({"q": "x", "count": 5}, None),
])
def test_skill_arguments_are_checked_against_the_stored_schema(args, why):
    from ghost_agent.tools.registry import _skill_args_error
    got = _skill_args_error(_SCHEMA, args)
    assert (got is None) if why is None else (why in got)


def test_a_missing_or_string_schema_is_handled():
    from ghost_agent.tools.registry import _skill_args_error
    assert _skill_args_error(None, {"a": 1}) is None
    assert "missing required" in _skill_args_error(json.dumps(_SCHEMA), {})


# ── 2. a shortened name that fits several documents is an error ──

@pytest.mark.parametrize("name,library,match,amb", [
    ("q3", ["q3.md", "q3.txt"], None, True),
    ("q3.txt", ["q3.md", "q3.txt"], "q3.txt", False),
    ("Q3.TXT", ["q3.md", "q3.txt"], "q3.txt", False),
    ("report", ["report.pdf", "other.md"], "report.pdf", False),
    ("nope", ["report.pdf"], None, False),
])
def test_document_names_resolve_or_say_which(name, library, match, amb):
    from ghost_agent.tools.memory import _match_library_name
    got, err = _match_library_name(name, library)
    assert got == match and bool(err) is amb


@pytest.mark.parametrize("action", ["query", "outline"])
async def test_an_ambiguous_name_never_picks_silently(action):
    import ghost_agent.tools.memory as M
    ms = MagicMock()
    ms.get_library.return_value = ["q3.md", "q3.txt"]
    out = await M.tool_knowledge_base(action=action, filename="q3", question="what?",
                                      sandbox_dir=None, memory_system=ms)
    assert "matches several documents" in str(out)
    ms.search_document.assert_not_called()


# ── 3. a dead-end refusal ends the tool phase on the first refusal ──

async def test_a_probe_write_refusal_carries_its_reason(tmp_path):
    from ghost_agent.tools.composed_skills import tool_manage_composed_skills
    ctx = type("C", (), {"memory_dir": tmp_path, "sandbox_dir": tmp_path})()
    tok = request_id_context.set("probe-1")
    try:
        out = await tool_manage_composed_skills(context=ctx, action="delete", name="x")
    finally:
        request_id_context.reset(tok)
    assert getattr(out, "reason_code", None) == "not_owner_write"


async def test_the_turn_loop_stops_calling_tools_after_a_dead_end():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def refuse(**kw):
        return ToolOutcome.rejected("Error: a probe … nothing changed.", world_changed=False,
                                    reason_code="not_owner_write")
    agent.available_tools = {"manage_skills": refuse}
    ts = H._ts([("manage_skills", {"action": "delete", "skill_name": "x"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    # §4LX: answered directly — no model-written reply to say "Let me retry"
    assert ts.force_stop is True
    assert ts.final_ai_content.startswith("Not done:") and "STOP" not in ts.final_ai_content
    assert "retry" not in ts.final_ai_content.lower()


async def test_a_confirm_dead_end_lets_the_model_ask_the_user():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def refuse(**kw):
        return ToolOutcome.rejected("Error: NOT done: the user has not answered yet. STOP calling it.",
                                    world_changed=False, reason_code="confirm_dead_end")
    agent.available_tools = {"manage_projects": refuse}
    ts = H._ts([("manage_projects", {"action": "delete", "project_id": "x", "confirm_token": "t"})],
               StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is True and not ts.force_stop
    assert any("Do NOT say you will retry" in str(m.get("content")) for m in ts.messages)


def test_the_direct_reply_drops_the_tool_facing_parts():
    from ghost_agent.core.agent import _dead_end_reply
    out = _dead_end_reply("Error: a probe request does not write the owner's scratchpad — nothing "
                          "changed. STOP: do not retry this in this turn.")
    assert out == "Not done: a probe request does not write the owner's scratchpad — nothing changed."


async def test_an_ordinary_refusal_does_not_end_the_turn():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def refuse(**kw):
        return ToolOutcome.rejected("Error: 'name' is required.", world_changed=False,
                                    reason_code="manage_skills_delete_refused")
    agent.available_tools = {"manage_skills": refuse}
    ts = H._ts([("manage_skills", {"action": "delete"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.force_final_response is False


async def test_the_real_skill_runner_refuses_bad_arguments_without_running_or_charging(tmp_path, monkeypatch):
    import ghost_agent.tools.registry as R
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    from tests.helpers import make_context
    ctx = make_context()
    ctx.sandbox_dir = tmp_path
    ctx.memory_dir = tmp_path
    ctx.memory_system = MagicMock()
    ctx.sandbox_manager = MagicMock()
    mgr = AcquiredSkillManager.get_shared(tmp_path, ctx.memory_system, legacy_sandbox_dir=tmp_path)
    mgr.save_skill("headlines", "desc", _SCHEMA, "def run(args):\n    return 1\n")
    ran = []

    async def fake_execute(**kw):
        ran.append(kw)
        return "EXIT CODE: 0\nok"
    monkeypatch.setattr(R, "tool_execute", fake_execute)
    runner = R.get_available_tools(ctx)["headlines"]
    out = await runner(count="ten")
    assert out.status is OutcomeStatus.REJECTED and ran == []
    assert mgr.get_all_skills()["headlines"]["failure_count"] == 0
    await runner(q="news", count=3)
    assert len(ran) == 1


async def test_two_refusals_in_one_batch_keep_the_first_reason():
    from ghost_agent.core.strikes import StrikeLedger
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()

    async def first(**kw):
        return ToolOutcome.rejected("Error: FIRST reason — nothing changed.", world_changed=False,
                                    reason_code="not_owner_write")

    async def second(**kw):
        return ToolOutcome.rejected("Error: SECOND reason — nothing changed.", world_changed=False,
                                    reason_code="not_owner_write")
    agent.available_tools = {"scratchpad": first, "manage_skills": second}
    ts = H._ts([("scratchpad", {"action": "set", "key": "k", "value": "v"}),
                ("manage_skills", {"action": "delete", "skill_name": "x"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert "FIRST" in ts.final_ai_content and "SECOND" not in ts.final_ai_content
