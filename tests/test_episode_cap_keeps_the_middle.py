"""The episode cap goes up (§4HI, 2026-09-16).

31 of 404 stored episodes were capped at 20 actions, the median one losing
44% of its actions — and the elided MIDDLE is where a long research run does
its reading (ep:402 lost 6 of its 10 substantive page reads, ep:403 4 of 7),
the part the agent expands on the next run of the same task. Each action's
result is already capped at 1,000 chars and `expand` renders ~300 chars per
action, so the cap saved almost nothing. 20 → 60; head-5/tail semantics kept
beyond it. Each pin names the world it fails in.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.episodes import EpisodicMemory


@pytest.fixture
def em(tmp_path):
    return EpisodicMemory(tmp_path)


def _actions(n):
    return [{"tool": "browser" if i % 3 else "web_search", "args": {"i": i},
             "result": f"page {i} read: 4000 chars of article", "success": True}
            for i in range(n)]


def test_a_forty_one_action_research_run_is_stored_whole(em):
    """FAILS IF: the cap is still 20 — the live world (5fa6aa97 ran 41
    tools; its episode kept 19 and a marker)."""
    ep_id = em.record_episode(trigger="investigate", actions=_actions(41), success=True)
    stored = em.get_episode(ep_id)["actions"]
    assert len(stored) == 41
    assert all(a["tool_name"] != EpisodicMemory.TRUNCATION_MARKER_TOOL for a in stored)
    # the middle — the reads — is exactly what is kept now
    assert stored[20]["tool_args"] == '{"i": 20}'


def test_sixty_is_the_last_whole_size_and_sixty_one_truncates(em):
    """FAILS IF: the cap is off by one in either direction, or truncation is
    gone altogether (an unbounded episode would eventually be the whole
    trajectory)."""
    whole = em.get_episode(em.record_episode(trigger="t", actions=_actions(60), success=True))["actions"]
    assert len(whole) == 60
    cut = em.get_episode(em.record_episode(trigger="t", actions=_actions(61), success=True))["actions"]
    assert len(cut) == 60
    names = [a["tool_name"] for a in cut]
    assert names.count(EpisodicMemory.TRUNCATION_MARKER_TOOL) == 1
    # head 5, marker, tail 54: the first action and the last are both there
    assert cut[0]["tool_args"] == '{"i": 0}' and cut[-1]["tool_args"] == '{"i": 60}'
    assert names[5] == EpisodicMemory.TRUNCATION_MARKER_TOOL


def test_the_expanded_episode_carries_every_read(em):
    """FAILS IF: the store keeps 45 actions but the reader still renders a
    capped view — the consumer is `knowledge_base(action='expand')`."""
    import asyncio
    from ghost_agent.tools.memory import tool_expand_evidence
    ep_id = em.record_episode(trigger="investigate", actions=_actions(45), success=True)
    out = asyncio.run(tool_expand_evidence(ref=f"ep:{ep_id}", episodic_memory=em))
    assert out.count("\n  ") == 45
    assert "page 30 read" in out and "elided" not in out
