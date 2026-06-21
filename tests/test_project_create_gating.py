"""Regression tests for PROJECT MODE GATING.

The agent was observed creating a project unprompted for a self-contained
single-file deliverable (a one-file browser OS), purely on the model's own
judgement — nudged by the broad `manage_projects` tool description and by RAG
surfacing prior similar projects. Project creation is 100% model-driven (no
code path auto-creates one), so the fix is explicit gating guidance in the
system prompt AND the tool description: create a project ONLY on an explicit
user ask OR for genuine multi-file + multi-session work; single-file
deliverables stay in free-chat.

These tests pin that guidance so it cannot silently regress.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core.prompts import SYSTEM_PROMPT
from ghost_agent.tools.projects import MANAGE_PROJECTS_TOOL_DEF


def test_system_prompt_has_project_gating_rule():
    assert "PROJECT MODE GATING" in SYSTEM_PROMPT


def test_system_prompt_gating_requires_explicit_or_multifile():
    # Both arms of the chosen policy must be spelled out.
    assert "EXPLICITLY" in SYSTEM_PROMPT
    assert "MULTIPLE files" in SYSTEM_PROMPT
    assert "MULTIPLE turns" in SYSTEM_PROMPT


def test_system_prompt_single_file_is_one_shot():
    # A single-file deliverable must be called out as a free-chat one-shot.
    seg = SYSTEM_PROMPT[SYSTEM_PROMPT.index("PROJECT MODE GATING"):]
    seg = seg[: seg.index("- MEMORY:")]
    assert "ONE-SHOT" in seg
    assert "free-chat" in seg
    # RAG surfacing past projects must be explicitly disqualified.
    assert "RAG" in seg and "NOT a reason" in seg


def test_tool_description_has_create_gating():
    desc = MANAGE_PROJECTS_TOOL_DEF["function"]["description"]
    assert "GATING for `create`" in desc
    assert "SINGLE-FILE" in desc
    # The two legitimate triggers and the single-file exclusion.
    assert "EXPLICITLY asks" in desc
    assert "MULTIPLE files/modules" in desc
    assert "do NOT create a project for it" in desc


def test_tool_description_disqualifies_past_projects():
    desc = MANAGE_PROJECTS_TOOL_DEF["function"]["description"]
    assert "Past similar projects in memory are NOT a reason" in desc
