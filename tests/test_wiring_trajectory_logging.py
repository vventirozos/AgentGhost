"""Tests for the GhostAgent._record_turn_trajectory wiring.

This is the hook inside handle_chat that converts a completed turn
into a Trajectory and persists it via `ctx.trajectory_collector`.
"""

from unittest.mock import MagicMock

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome


def _agent_with_collector(collector):
    """Build an almost-empty GhostAgent whose context carries just what
    `_record_turn_trajectory` needs."""
    ctx = MagicMock()
    ctx.trajectory_collector = collector
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    return agent


def test_record_trajectory_noop_when_collector_missing():
    """No collector → method returns silently. This is the default
    state in deployments that opt out of trajectory logging."""
    ctx = MagicMock()
    ctx.trajectory_collector = None
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Should not raise regardless of messages shape.
    agent._record_turn_trajectory(
        messages=[{"role": "user", "content": "hi"}],
        final_content="hello",
        req_id="r",
        model="m",
    )


def test_record_trajectory_writes_basic_turn(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="turn-test")
    agent = _agent_with_collector(collector)
    agent._record_turn_trajectory(
        messages=[
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        final_content="4",
        req_id="req-1",
        model="m",
    )
    trajs = list(collector.iter_trajectories())
    assert len(trajs) == 1
    t = trajs[0]
    assert t.user_request == "what is 2+2?"
    assert t.final_response == "4"
    assert t.system_prompt == "you are helpful"
    assert t.session_id == "req-1"
    assert t.task_kind == "user_request"
    assert t.outcome == Outcome.UNKNOWN.value
    assert t.n_steps == 1  # one assistant message


def test_record_trajectory_extracts_tool_calls(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="tc-test")
    agent = _agent_with_collector(collector)
    agent._record_turn_trajectory(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "file_system", "arguments": '{"action": "list"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"},
            {"role": "assistant", "content": "Found two files."},
        ],
        final_content="Found two files.",
        req_id="req-2",
        model="m",
    )
    [t] = list(collector.iter_trajectories())
    assert len(t.tool_calls) == 1
    tc = t.tool_calls[0]
    assert tc.name == "file_system"
    assert tc.arguments == {"action": "list"}
    assert "a.py" in tc.result
    assert t.n_steps == 2  # two assistant messages


def test_record_trajectory_handles_malformed_tool_arguments(tmp_path):
    """A runtime slip-up in argument serialization must not break
    logging — we capture whatever we can and carry on."""
    collector = TrajectoryCollector(root=tmp_path, session_id="mal")
    agent = _agent_with_collector(collector)
    agent._record_turn_trajectory(
        messages=[
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "execute", "arguments": "{this is broken"},
                    },
                ],
            },
        ],
        final_content="",
        req_id="req-3",
        model="m",
    )
    [t] = list(collector.iter_trajectories())
    assert len(t.tool_calls) == 1
    assert t.tool_calls[0].name == "execute"
    # Malformed args survive as a _raw capture (truncated to 500 chars).
    assert "_raw" in t.tool_calls[0].arguments


def test_record_trajectory_non_string_final_content_stored_empty(tmp_path):
    """Streaming returns an async generator instead of a string — the
    trajectory records an empty `final_response` in that case, because
    the full text lives in the SSE frames not here."""
    collector = TrajectoryCollector(root=tmp_path, session_id="stream")
    agent = _agent_with_collector(collector)

    async def _gen():
        yield b"chunk"

    agent._record_turn_trajectory(
        messages=[{"role": "user", "content": "q"}],
        final_content=_gen(),
        req_id="req-4",
        model="m",
    )
    [t] = list(collector.iter_trajectories())
    assert t.final_response == ""
    assert t.user_request == "q"


def test_record_trajectory_non_dict_messages_ignored(tmp_path):
    """Defensive: the caller's messages list can contain weird shapes
    if a test fixture is sloppy. We just skip them."""
    collector = TrajectoryCollector(root=tmp_path, session_id="def")
    agent = _agent_with_collector(collector)
    agent._record_turn_trajectory(
        messages=[
            "not a dict",
            None,
            {"role": "user", "content": "real"},
        ],
        final_content="ok",
        req_id="req-5",
        model="m",
    )
    [t] = list(collector.iter_trajectories())
    assert t.user_request == "real"


def test_record_trajectory_list_content_gets_flattened(tmp_path):
    """OpenAI-style list-of-content blocks (vision payloads) flatten
    into a single string for the trajectory."""
    collector = TrajectoryCollector(root=tmp_path, session_id="list")
    agent = _agent_with_collector(collector)
    agent._record_turn_trajectory(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe the image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ],
        final_content="a cat",
        req_id="req-6",
        model="m",
    )
    [t] = list(collector.iter_trajectories())
    assert "describe the image" in t.user_request
