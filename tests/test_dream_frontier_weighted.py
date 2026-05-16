"""Integration test for the frontier-weighted self-play seed pick.

Exercises the full wire-up in ``Dreamer.synthetic_self_play``:
  * a real PRMScorer with a trained model
  * a real TrajectoryCollector with seeded trajectories
  * a real FrontierTracker
  * args.frontier_selfplay = True

Verifies the new path engages (mode=frontier_weighted), and that
disabling the flag or removing either dependency falls back cleanly to
the existing brittle-pool pick_seed without behavioural drift.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ghost_agent.core.dream import Dreamer
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.memory.frontier import FrontierTracker
from ghost_agent.prm import PRMScorer, PRMTrainer


def _dict_to_xml(d):
    return "".join(f"<{k}>{v}</{k}>\n" for k, v in d.items())


def _llm_response(challenge="c", validator="v", setup=""):
    payload = {"challenge_prompt": challenge, "validation_script": validator}
    if setup:
        payload["setup_script"] = setup
    return {"choices": [{"message": {"content": _dict_to_xml(payload)}}]}


def _balanced_corpus():
    pass_t = [
        Trajectory(
            user_request=f"pass {i}",
            outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="scratchpad")],
            n_steps=1,
        )
        for i in range(8)
    ]
    fail_t = [
        Trajectory(
            user_request=f"fail {i}",
            outcome=Outcome.FAILED.value,
            tool_calls=[ToolCall(name="execute", error="boom")],
            n_steps=1,
        )
        for i in range(8)
    ]
    return pass_t + fail_t


def _trained_scorer():
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    trainer.run(_balanced_corpus())
    return PRMScorer(model=trainer.model)


def _seed_collector(tmp_path, cluster_counts: dict):
    """Build a TrajectoryCollector and append synthetic trajectories so
    its `iter_trajectories()` yields N entries per cluster."""
    col = TrajectoryCollector(root=tmp_path / "trajectories")
    for cluster, n in cluster_counts.items():
        for i in range(n):
            t = Trajectory(
                user_request=f"{cluster} {i}",
                outcome=Outcome.PASSED.value,
                tool_calls=[ToolCall(name="scratchpad")],
                n_steps=1,
                cluster=cluster,
            )
            col.append(t)
    return col


def _make_context(tmp_path, *, prm_scorer=None, traj_collector=None,
                  frontier_enabled=True, uniform_prob=0.0):
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_recent_failures.return_value = "No failures"
    ctx.llm_client = MagicMock()
    # Use real argparse-like Namespace so attribute reads return the
    # actual primitive values, not Mocks.
    import argparse
    ctx.args = argparse.Namespace(
        perfect_it=False,
        smart_memory=0.0,
        model="test-model",
        frontier_selfplay=frontier_enabled,
        frontier_uniform_sample_prob=uniform_prob,
    )
    ctx.sandbox_manager = MagicMock()
    ctx.sandbox_dir = str(tmp_path)
    ctx.tor_proxy = None
    ctx.scratchpad = MagicMock()
    ctx.frontier_tracker = FrontierTracker(tmp_path)
    ctx.prm_scorer = prm_scorer
    ctx.trajectory_collector = traj_collector
    return ctx


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_frontier_weighted_path_engages_with_real_prm_and_collector(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates,
):
    """When PRMScorer.has_model is True AND a real TrajectoryCollector
    is attached, the frontier-weighted path must produce a seed whose
    hint reads 'FRONTIER TARGET (PRM-weighted)'."""
    scorer = _trained_scorer()
    assert scorer.has_model
    collector = _seed_collector(tmp_path, {"sql": 1, "bash": 50, "algo": 100})
    ctx = _make_context(
        tmp_path,
        prm_scorer=scorer,
        traj_collector=collector,
        uniform_prob=0.0,  # turn off sanity sampling so we observe the new path deterministically
    )

    # Capture EVERY chat_completion call so we can scan them all for
    # the frontier marker — synthetic_self_play makes a judge call
    # after the challenge call and would overwrite a single-slot capture.
    captured = {"calls": []}

    async def fake_chat(payload, *a, **kw):
        # Concatenate every message's content. The challenge-gen call
        # has a hardcoded "AI training coordinator" system prompt and
        # the frontier hint lives in the USER message — checking only
        # messages[0] misses it.
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        merged = "\n".join((m.get("content") or "") for m in messages)
        captured["calls"].append(merged)
        return _llm_response()

    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)

    dreamer = Dreamer(ctx)
    try:
        await dreamer.synthetic_self_play(model_name="test-model", is_background=True)
    except Exception:
        # We only care about the prompt assembly, not the full E2E path.
        pass

    joined = "\n".join(captured["calls"])
    assert "PRM-weighted" in joined, (
        f"Expected frontier-weighted hint in some chat_completion call, "
        f"got {len(captured['calls'])} calls; last 500 chars:\n{joined[-500:]}"
    )


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_frontier_disabled_flag_falls_back_to_pick_seed(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates,
):
    """args.frontier_selfplay=False must skip the new path even when
    PRM + collector are wired — kill switch for ops / A/B comparison."""
    scorer = _trained_scorer()
    collector = _seed_collector(tmp_path, {"sql": 1, "bash": 50})
    ctx = _make_context(
        tmp_path,
        prm_scorer=scorer,
        traj_collector=collector,
        frontier_enabled=False,
    )
    captured = {"calls": []}

    async def fake_chat(payload, *a, **kw):
        # Concatenate every message's content. The challenge-gen call
        # has a hardcoded "AI training coordinator" system prompt and
        # the frontier hint lives in the USER message — checking only
        # messages[0] misses it.
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        merged = "\n".join((m.get("content") or "") for m in messages)
        captured["calls"].append(merged)
        return _llm_response()

    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)

    dreamer = Dreamer(ctx)
    try:
        await dreamer.synthetic_self_play(model_name="test-model", is_background=True)
    except Exception:
        pass

    joined = "\n".join(captured["calls"])
    assert "PRM-weighted" not in joined


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_untrained_prm_falls_back_to_pick_seed(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates,
):
    """PRMScorer with no loaded model has has_model=False — the
    frontier path must skip and fall back to pick_seed regardless."""
    empty_scorer = PRMScorer()  # no model
    assert not empty_scorer.has_model
    collector = _seed_collector(tmp_path, {"sql": 1})
    ctx = _make_context(
        tmp_path,
        prm_scorer=empty_scorer,
        traj_collector=collector,
    )
    captured = {"calls": []}

    async def fake_chat(payload, *a, **kw):
        # Concatenate every message's content. The challenge-gen call
        # has a hardcoded "AI training coordinator" system prompt and
        # the frontier hint lives in the USER message — checking only
        # messages[0] misses it.
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        merged = "\n".join((m.get("content") or "") for m in messages)
        captured["calls"].append(merged)
        return _llm_response()

    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)

    dreamer = Dreamer(ctx)
    try:
        await dreamer.synthetic_self_play(model_name="test-model", is_background=True)
    except Exception:
        pass

    joined = "\n".join(captured["calls"])
    assert "PRM-weighted" not in joined


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_missing_trajectory_collector_falls_back(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates,
):
    """No trajectory_collector attached → fall back. The isinstance
    check must fail closed (a MagicMock auto-attr is not a real
    TrajectoryCollector)."""
    scorer = _trained_scorer()
    ctx = _make_context(tmp_path, prm_scorer=scorer, traj_collector=None)
    captured = {"calls": []}

    async def fake_chat(payload, *a, **kw):
        # Concatenate every message's content. The challenge-gen call
        # has a hardcoded "AI training coordinator" system prompt and
        # the frontier hint lives in the USER message — checking only
        # messages[0] misses it.
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        merged = "\n".join((m.get("content") or "") for m in messages)
        captured["calls"].append(merged)
        return _llm_response()

    ctx.llm_client.chat_completion = AsyncMock(side_effect=fake_chat)

    dreamer = Dreamer(ctx)
    try:
        await dreamer.synthetic_self_play(model_name="test-model", is_background=True)
    except Exception:
        pass

    joined = "\n".join(captured["calls"])
    assert "PRM-weighted" not in joined
