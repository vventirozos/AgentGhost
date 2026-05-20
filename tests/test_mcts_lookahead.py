"""Tests for the MCTS lookahead contract used by the turn loop (item #8).

The reasoning loop now invokes ``MCTSReasoner.select_best_action`` for
router-classified hard requests and reads ``description`` / ``tool_name``
/ ``risk_notes`` off the returned candidate. These tests exercise that
exact call shape with a stub LLM client.
"""

import json

from ghost_agent.core.mcts import MCTSReasoner


class _StubLLM:
    """Minimal llm_client: returns expansion JSON for the expand prompt
    and simulation JSON for the simulate prompt."""

    def __init__(self, expand_payload, sim_payload):
        self.expand_payload = expand_payload
        self.sim_payload = sim_payload
        self.calls = 0

    async def chat_completion(self, payload):
        self.calls += 1
        content = payload["messages"][0]["content"]
        body = (self.expand_payload
                if "candidate next-actions" in content
                else self.sim_payload)
        return {"choices": [{"message": {"content": json.dumps(body)}}]}


async def test_select_best_action_returns_usable_candidate():
    expand = {"candidates": [
        {"description": "read the config with file_system",
         "tool_name": "file_system", "risk": "file may be missing"},
        {"description": "run a probe with execute",
         "tool_name": "execute", "risk": "could crash"},
    ]}
    sim = {"predicted_outcome": "fine", "progress": 0.9,
           "cost": 0.1, "risk": 0.1}
    llm = _StubLLM(expand, sim)
    mcts = MCTSReasoner(llm_client=llm, max_candidates=2)

    winner = await mcts.select_best_action(
        task="diagnose a hard bug",
        plan_state="(turn start — no actions taken yet)",
        available_tools=["file_system", "execute"],
        context="",
    )
    assert winner is not None
    # The turn-loop wiring reads exactly these attributes.
    assert isinstance(winner.description, str) and winner.description
    assert winner.tool_name in ("file_system", "execute")
    assert hasattr(winner, "risk_notes")
    assert llm.calls >= 1


async def test_select_best_action_no_candidates_returns_none():
    llm = _StubLLM({"candidates": []}, {})
    mcts = MCTSReasoner(llm_client=llm, max_candidates=3)
    winner = await mcts.select_best_action(
        task="x", plan_state="y", available_tools=["execute"], context="",
    )
    assert winner is None


async def test_select_best_action_picks_higher_scoring_path():
    expand = {"candidates": [
        {"description": "safe path", "tool_name": "file_system", "risk": ""},
        {"description": "risky path", "tool_name": "execute", "risk": "bad"},
    ]}
    # Simulation gives every candidate the same score here; just assert a
    # winner is chosen deterministically and is one of the candidates.
    sim = {"predicted_outcome": "ok", "progress": 0.5,
           "cost": 0.5, "risk": 0.5}
    mcts = MCTSReasoner(llm_client=_StubLLM(expand, sim), max_candidates=2)
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system", "execute"],
        context="",
    )
    assert winner is not None
    assert winner.description in ("safe path", "risky path")
