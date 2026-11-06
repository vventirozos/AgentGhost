"""§4LS — skills: behaviour pins for the review's fixes.

The macro half of this file (M1 a macro declares how it went, M2 no meta tool as a step, M5 approve re-runs
the mint rules) went with the macro engine (§4TA, 2026-11-05). Kept: the acquired-skill pins, and M3 in its
new shape — the one built-in that calls another tool is contained wherever that tool is."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghost_agent.utils.logging import request_id_context


class _Ctx:
    def __init__(self, base):
        self.memory_dir = base
        self.sandbox_dir = base


def _as(rid):
    return request_id_context.set(rid)


# ── M3: a tool that calls another tool is forbidden under containment wherever the inner tool is ──

async def _tool(**kw):
    return "ok"


@pytest.mark.parametrize("forbidden,kept", [
    ({"knowledge_base"}, False),          # its handler would reach the forbidden tool through the full dispatch dict
    ({"web_search"}, True),               # a surface that keeps knowledge_base keeps it
])
def test_containment_drops_youtube_transcribe_where_knowledge_base_is_forbidden(tmp_path, forbidden, kept):
    from ghost_agent.core.isolation import restrict_tool_surface
    agent = SimpleNamespace(context=_Ctx(tmp_path), disabled_tools=set(),
                            available_tools={"youtube_transcribe": _tool, "knowledge_base": _tool,
                                             "web_search": _tool, "file_system": _tool})
    keep = restrict_tool_surface(agent, forbidden=set(forbidden))
    assert ("youtube_transcribe" in keep) is kept
    assert ("youtube_transcribe" in agent.available_tools) is kept
    assert ("youtube_transcribe" in agent.disabled_tools) is (not kept)
    assert "file_system" in agent.available_tools


def test_the_default_replay_surface_has_neither(tmp_path):
    from ghost_agent.core.isolation import restrict_tool_surface
    agent = SimpleNamespace(context=_Ctx(tmp_path), disabled_tools=set(),
                            available_tools={"youtube_transcribe": _tool, "knowledge_base": _tool,
                                             "file_system": _tool})
    keep = restrict_tool_surface(agent)
    assert not {"youtube_transcribe", "knowledge_base"} & set(keep)
    assert "file_system" in keep


# ── M4: probe / background requests never change the stores ──

async def test_a_probe_cannot_delete_or_create_an_acquired_skill(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_create_skill, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("gen_pw", "desc", {}, "def run(args):\n    return 1\n")
    tok = _as("probe-2")
    try:
        d = await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="delete", skill_name="gen_pw")
        c = await tool_create_skill(memory_dir=tmp_path, sandbox_dir=tmp_path, name="x", description="d",
                                    parameters_schema="{}", python_code="x", test_payload="{}")
    finally:
        request_id_context.reset(tok)
    assert "nothing changed" in d and "nothing changed" in c
    assert "gen_pw" in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


async def test_a_probe_list_never_retires_a_skill(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("flaky", "desc", {}, "def run(args):\n    return 1\n")
    for _ in range(3):
        mgr.log_telemetry("flaky", success=False)
    tok = _as("probe-3")
    try:
        await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="delete", skill_name="nope")
    finally:
        request_id_context.reset(tok)
    assert "flaky" in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


async def test_an_owner_list_still_sweeps_degraded_skills(tmp_path):
    # the tool has only list/delete — a list-exempt sweep never ran (fix review N2)
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("flaky", "desc", {}, "def run(args):\n    return 1\n")
    for _ in range(3):
        mgr.log_telemetry("flaky", success=False)
    tok = _as("req-owner")
    try:
        await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="list")
    finally:
        request_id_context.reset(tok)
    assert "flaky" not in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()


async def test_a_job_wake_turn_may_delete_an_acquired_skill(tmp_path):
    # a `job-` wake resumes the owner's own work (§4LS fix review) — the macro store's twin of this pin
    # went with the macro engine; the acquired-skill store has the same rule
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager, tool_manage_skills
    AcquiredSkillManager(tmp_path, memory_system=None).save_skill("gen_pw", "desc", {}, "def run(args):\n    return 1\n")
    tok = _as("job-1234")
    try:
        out = await tool_manage_skills(memory_dir=tmp_path, sandbox_dir=tmp_path, action="delete", skill_name="gen_pw")
    finally:
        request_id_context.reset(tok)
    assert "has been deleted" in out
    assert "gen_pw" not in AcquiredSkillManager(tmp_path, memory_system=None).get_all_skills()
