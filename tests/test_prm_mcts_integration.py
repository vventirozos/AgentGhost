"""Integration tests: MCTSReasoner + PRMScorer.

Verifies the contract documented on ``MCTSReasoner.select_best_action``:

  * When ``prm_scorer`` is attached AND ``prm_state`` is passed AND the
    scorer has a trained model → candidates are scored by the PRM,
    NO LLM simulation calls are issued.
  * When any of those conditions is missing → fall back to legacy LLM
    simulation. Existing callers continue to work.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.mcts import ActionCandidate, MCTSReasoner
from ghost_agent.prm.features import ActionFeatures, PlanState, extract_step_features
from ghost_agent.prm.model import StepValueModel
from ghost_agent.prm.scorer import PRMScorer


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_scorer():
    """A scorer whose model deliberately scores 'execute' high and
    'browser' low. Lets tests assert the right candidate wins by
    construction rather than by training-noise luck."""
    from ghost_agent.prm.features import PRM_FEATURE_NAMES
    import numpy as np

    model = StepValueModel()
    # Manually craft a weight vector: positive weight on
    # tool_is_lightweight, negative on tool_is_heavyweight.
    w = np.zeros(len(PRM_FEATURE_NAMES))
    w[PRM_FEATURE_NAMES.index("tool_is_lightweight")] = 5.0
    w[PRM_FEATURE_NAMES.index("tool_is_heavyweight")] = -5.0
    model.weights_ = w
    model.bias_ = 0.0
    return PRMScorer(model=model)


@pytest.fixture
def expansion_response():
    """Three candidates: file_system (lightweight), browser (heavy),
    web_search (external)."""
    return {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "Read the file",
                 "tool_name": "file_system", "risk": "low"},
                {"description": "Open browser",
                 "tool_name": "browser", "risk": "high"},
                {"description": "Search the web",
                 "tool_name": "web_search", "risk": "medium"},
            ],
        })}}]
    }


@pytest.fixture
def fake_llm(expansion_response):
    client = MagicMock()
    client.chat_completion = AsyncMock(return_value=expansion_response)
    client.route = AsyncMock(return_value=None)
    return client


# ──────────────────────────────────────────────────────────────────────
# PRM fast-path activation
# ──────────────────────────────────────────────────────────────────────

async def test_prm_path_engages_when_scorer_and_state_provided(fake_llm, trained_scorer):
    """When PRM is attached and prm_state is passed, candidate scoring
    must use the PRM and skip the LLM simulation step."""
    # Reset the call_count *after* fixture init so we count expansion-only.
    fake_llm.chat_completion.reset_mock()
    fake_llm.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "lightweight", "tool_name": "file_system"},
                {"description": "heavy", "tool_name": "browser"},
            ],
        })}}]
    }

    mcts = MCTSReasoner(
        llm_client=fake_llm,
        max_candidates=2,
        prm_scorer=trained_scorer,
    )
    state = PlanState(user_request="x")
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system", "browser"],
        prm_state=state,
    )
    # Exactly one LLM call (expansion). Simulation is skipped.
    assert fake_llm.chat_completion.call_count == 1
    assert winner is not None
    assert winner.tool_name == "file_system"  # lightweight wins under crafted weights


async def test_prm_path_skipped_when_scorer_has_no_model(fake_llm):
    """An untrained scorer (has_model=False) must NOT engage PRM scoring
    — every candidate would tie at 0.5. Falls back to LLM sim."""
    empty_scorer = PRMScorer()
    assert empty_scorer.has_model is False

    mcts = MCTSReasoner(
        llm_client=fake_llm,
        max_candidates=3,
        prm_scorer=empty_scorer,
    )
    state = PlanState(user_request="x")
    # Stub the simulation responses too.
    expansion = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "a", "tool_name": "file_system"},
                {"description": "b", "tool_name": "browser"},
                {"description": "c", "tool_name": "web_search"},
            ],
        })}}]
    }
    sim = lambda p, c, r: {  # one stub call shape per simulation
        "choices": [{"message": {"content": json.dumps({
            "predicted_outcome": "ok", "progress": p, "cost": c, "risk": r,
        })}}]
    }
    fake_llm.chat_completion.side_effect = [
        expansion, sim(0.5, 0.3, 0.2), sim(0.4, 0.5, 0.4), sim(0.3, 0.4, 0.3),
    ]
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system", "browser", "web_search"],
        prm_state=state,
    )
    # Expansion (1) + 3 simulation calls = 4
    assert fake_llm.chat_completion.call_count == 4
    assert winner is not None


async def test_prm_path_skipped_when_state_missing(fake_llm, trained_scorer):
    """Even with a trained PRM, omitting ``prm_state`` keeps the legacy
    simulation path. Backwards-compat for existing callers."""
    expansion = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "a", "tool_name": "file_system"},
                {"description": "b", "tool_name": "browser"},
            ],
        })}}]
    }
    sim_a = {"choices": [{"message": {"content": json.dumps({
        "progress": 0.6, "cost": 0.2, "risk": 0.1,
    })}}]}
    sim_b = {"choices": [{"message": {"content": json.dumps({
        "progress": 0.4, "cost": 0.5, "risk": 0.6,
    })}}]}
    fake_llm.chat_completion.side_effect = [expansion, sim_a, sim_b]
    mcts = MCTSReasoner(
        llm_client=fake_llm, max_candidates=2, prm_scorer=trained_scorer,
    )
    # prm_state OMITTED.
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system", "browser"],
    )
    # 1 expansion + 2 simulations = 3 LLM calls (PRM skipped).
    assert fake_llm.chat_completion.call_count == 3
    assert winner is not None


# ──────────────────────────────────────────────────────────────────────
# Score quality
# ──────────────────────────────────────────────────────────────────────

async def test_prm_path_assigns_scores_within_unit_interval(fake_llm, trained_scorer):
    fake_llm.chat_completion.reset_mock()
    fake_llm.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "a", "tool_name": "file_system"},
                {"description": "b", "tool_name": "browser"},
                {"description": "c", "tool_name": "web_search"},
            ],
        })}}]
    }
    mcts = MCTSReasoner(
        llm_client=fake_llm, max_candidates=3, prm_scorer=trained_scorer,
    )
    state = PlanState(user_request="x")
    winner = await mcts.select_best_action(
        task="t", plan_state="s",
        available_tools=["file_system", "browser", "web_search"],
        prm_state=state,
    )
    assert winner is not None
    # All cached alternatives must also have scores in [0, 1] — the
    # backtrack stack is guaranteed to be sane.
    for stack in mcts._backtrack_stack:
        for cand in stack:
            assert 0.0 <= cand.score <= 1.0


async def test_prm_path_records_provenance_in_simulated_outcome(fake_llm, trained_scorer):
    """Easy way to debug 'why did this candidate win': the
    simulated_outcome string carries the PRM's score, with a 'PRM:'
    prefix to distinguish from LLM-simulated outcomes."""
    fake_llm.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "x", "tool_name": "file_system"},
            ],
        })}}]
    }
    mcts = MCTSReasoner(
        llm_client=fake_llm, max_candidates=1, prm_scorer=trained_scorer,
    )
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system"],
        prm_state=PlanState(user_request="x"),
    )
    assert winner is not None
    assert winner.simulated_outcome.startswith("PRM:")


# ──────────────────────────────────────────────────────────────────────
# Failure isolation
# ──────────────────────────────────────────────────────────────────────

async def test_prm_score_exception_does_not_crash_selection(fake_llm):
    """A scorer that raises on every call must not bring down plan
    selection. Per-candidate try/except keeps the others scorable."""
    bad_scorer = MagicMock()
    bad_scorer.has_model = True
    bad_scorer.score = MagicMock(side_effect=RuntimeError("boom"))

    fake_llm.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "candidates": [
                {"description": "x", "tool_name": "file_system"},
                {"description": "y", "tool_name": "browser"},
            ],
        })}}]
    }
    mcts = MCTSReasoner(
        llm_client=fake_llm, max_candidates=2, prm_scorer=bad_scorer,
    )
    winner = await mcts.select_best_action(
        task="t", plan_state="s", available_tools=["file_system", "browser"],
        prm_state=PlanState(user_request="x"),
    )
    # Each candidate fell back to neutral 0.5; one of them wins.
    assert winner is not None
    assert winner.score == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────
# Hot-swap pattern (biological retrain)
# ──────────────────────────────────────────────────────────────────────

def test_scorer_hot_swap_picked_up_immediately(trained_scorer):
    """The biological retrain phase calls ``scorer.set_model(new_model)``.
    The very next ``score`` call must use the new weights."""
    state = PlanState(user_request="x")
    action = ActionFeatures(tool_name="file_system")

    initial_score = trained_scorer.score(state, action)
    # Replace with a model that has all-zero weights → returns 0.5.
    import numpy as np
    from ghost_agent.prm.features import PRM_FEATURE_NAMES
    new_model = StepValueModel()
    new_model.weights_ = np.zeros(len(PRM_FEATURE_NAMES))
    new_model.bias_ = 0.0
    trained_scorer.set_model(new_model)
    swapped_score = trained_scorer.score(state, action)
    assert swapped_score == pytest.approx(0.5)
    assert swapped_score != initial_score
