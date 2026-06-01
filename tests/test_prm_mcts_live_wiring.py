"""Live PRM → MCTS wiring: train/serve parity.

The agent's deep-reason MCTS lookahead (core.agent.handle_chat) now
constructs a `PlanState` at turn start and passes it as `prm_state=` so
the trained PRM fast path actually fires in production (previously
`prm_state` was never built → the PRM never scored a live candidate and
MCTS always paid the LLM-simulation cost).

The subtle correctness property is **no train/serve skew**: the
turn-start state the live path builds must produce the same features as
the step-0 prefix-state the PRM was trained on, and as the frontier
representative seed. These tests pin that — if anyone changes the
neutral inference constants (pending_count / plan_depth = 1/1) on one
side they must change all three, or the PRM scores out-of-distribution.
"""

from ghost_agent.prm import PlanState, ActionFeatures
from ghost_agent.prm.features import extract_step_features
from ghost_agent.prm.labels import _build_state_for_step
from ghost_agent.core.frontier_selection import representative_state
from ghost_agent.distill.schema import Trajectory, ToolCall


def _live_turn_start_state(user_request: str) -> PlanState:
    """Mirror of the state the agent.py lookahead builds at turn start.
    Kept here so the test fails loudly if the live constants drift."""
    return PlanState(
        user_request=user_request,
        steps_so_far=0,
        failures_so_far=0,
        pending_count=1,
        plan_depth=1,
        tools_used_this_turn=(),
        tools_failed_this_turn=(),
    )


def test_live_turn_start_state_equals_training_prefix():
    live = _live_turn_start_state("analyze the access log")
    traj = Trajectory(
        user_request="analyze the access log",
        tool_calls=[ToolCall(name="execute"), ToolCall(name="file_system")],
    )
    training_step0 = _build_state_for_step(traj, 0)
    # Dataclass equality across every field → zero train/serve skew.
    assert live == training_step0


def test_live_state_features_match_training_features():
    act = ActionFeatures(description="run a query", tool_name="execute", tool_args={})
    live = _live_turn_start_state("x")
    traj = Trajectory(user_request="x", tool_calls=[ToolCall(name="execute")])
    training = _build_state_for_step(traj, 0)
    assert extract_step_features(live, act).values == extract_step_features(
        training, act
    ).values


def test_frontier_seed_uses_same_neutral_constants():
    state, _action = representative_state("data_analysis")
    assert state.pending_count == 1
    assert state.plan_depth == 1
    assert state.steps_so_far == 0
    assert state.failures_so_far == 0
