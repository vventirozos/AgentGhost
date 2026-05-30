"""Tests for the MCTS Reasoning module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from ghost_agent.core.mcts import MCTSReasoner, ActionCandidate, MCTSNode


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    client.route = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mcts(mock_llm_client):
    return MCTSReasoner(llm_client=mock_llm_client, max_candidates=3)


class TestActionCandidate:
    def test_to_dict(self):
        c = ActionCandidate(
            description="Run the script",
            tool_name="execute",
            score=0.8,
            selected=True,
        )
        d = c.to_dict()
        assert d["description"] == "Run the script"
        assert d["score"] == 0.8
        assert d["selected"] is True


class TestMCTSNode:
    def test_avg_score_zero_visits(self):
        node = MCTSNode(action=ActionCandidate(description="test"))
        assert node.avg_score == 0.0

    def test_avg_score_with_visits(self):
        node = MCTSNode(
            action=ActionCandidate(description="test"),
            visits=4, total_score=2.0,
        )
        assert node.avg_score == 0.5


class TestMCTSReasoner:
    async def test_select_best_action(self, mcts, mock_llm_client):
        # Mock expansion response
        mock_llm_client.chat_completion.side_effect = [
            # Expansion call
            {"choices": [{"message": {"content": json.dumps({
                "candidates": [
                    {"description": "Read the file", "tool_name": "file_system", "risk": "low"},
                    {"description": "Run analysis script", "tool_name": "execute", "risk": "medium"},
                    {"description": "Search web", "tool_name": "web_search", "risk": "low"},
                ],
            })}}]},
            # Simulation calls (3x)
            {"choices": [{"message": {"content": json.dumps({
                "predicted_outcome": "File contents loaded",
                "progress": 0.3, "cost": 0.1, "risk": 0.1,
            })}}]},
            {"choices": [{"message": {"content": json.dumps({
                "predicted_outcome": "Analysis complete",
                "progress": 0.9, "cost": 0.4, "risk": 0.3,
            })}}]},
            {"choices": [{"message": {"content": json.dumps({
                "predicted_outcome": "Web results found",
                "progress": 0.2, "cost": 0.2, "risk": 0.5,
            })}}]},
        ]

        result = await mcts.select_best_action(
            task="Analyze the CSV",
            plan_state="No progress yet",
            available_tools=["file_system", "execute", "web_search"],
        )
        assert result is not None
        assert result.selected is True
        assert result.score > 0

    async def test_select_best_no_llm(self):
        mcts = MCTSReasoner(llm_client=None)
        result = await mcts.select_best_action("task", "state", ["tools"])
        assert result is None

    def test_clear(self, mcts):
        mcts._backtrack_stack.append([ActionCandidate(description="alt")])
        mcts.clear()
        assert not mcts._backtrack_stack

    async def test_expand_handles_exception(self, mcts, mock_llm_client):
        mock_llm_client.chat_completion.side_effect = Exception("boom")
        result = await mcts._expand("task", "state", ["tools"], "context")
        assert result == []

    async def test_simulate_with_worker_route(self, mcts, mock_llm_client):
        mock_llm_client.route.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "progress": 0.7, "cost": 0.3, "risk": 0.2,
            })}}],
        }
        candidates = [ActionCandidate(description="test action", tool_name="execute")]
        result = await mcts._simulate_parallel(candidates, "context")
        assert result[0].score > 0

    def test_parse_json_valid(self):
        assert MCTSReasoner._parse_json('{"a": 1}') == {"a": 1}

    def test_parse_json_embedded(self):
        text = 'Here is the result: {"a": 1} done'
        assert MCTSReasoner._parse_json(text) == {"a": 1}

    def test_parse_json_empty(self):
        assert MCTSReasoner._parse_json("") == {}
        assert MCTSReasoner._parse_json("no json here") == {}
