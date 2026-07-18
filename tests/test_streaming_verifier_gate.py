"""The verifier gate on the STREAMING path (wired 2026-07-18).

The gated verdict was historically invoked only from
``_finalize_and_return`` — the non-streaming finalization chain — so
every streamed answer shipped unverified, with no log line of any kind
(found when the operator asked why a project turn had no verdict: the
late recorder logs every outcome yet the log had none, and the USR2
task dump showed no verdict task alive). The streaming closure now
spawns the pure verdict computation post-drain and hands it to the late
handler with ``force_correction=True``, because a streamed reply never
carries an inline Verifier note — the correction banner surfaced by
``_take_active_correction`` at the next stream start is the ONLY way a
refuted streamed answer reaches the user.

These tests pin:
  * ``force_correction=True`` queues the next-turn correction on a
    high-confidence late REFUTED even with async-critic mode OFF;
  * the default (``force_correction=False``) keeps the old behaviour —
    no queue outside async mode — so inline-note paths don't double-up;
  * ``_attach_late_verdict_handler`` threads the flag through to the
    recorder;
  * source-level wiring: the stream gate block exists inside the
    streaming region, uses the eager full-list captures (messages[-10:]
    loses the first user message the conversation fingerprint is keyed
    on), strips <think> from the claim, and passes force_correction.
"""

import asyncio
import inspect

import pytest
from unittest.mock import MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


def _refuted(conf=0.9, issues=None):
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=conf,
                        reasoning="reason", issues=issues or ["bad fact"])


# ---------- force_correction on the recorder ----------


def test_force_correction_queues_banner_outside_async_mode(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _refuted(issues=["claim contradicts tool output"]),
            trajectory_id="traj-s1", conv_fp="fp-abc",
            force_correction=True,
        )
    assert agent._pending_corrections
    entry = agent._pending_corrections[0]
    assert "claim contradicts tool output" in entry["note"]
    assert entry["conv"] == "fp-abc"


def test_default_still_does_not_queue_outside_async_mode(agent, monkeypatch):
    """Regression pin: without the flag, non-async behaviour is unchanged
    (the non-streaming inline path already appended a Verifier note; a
    banner on top would double-report)."""
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _refuted(), trajectory_id="traj-s2", conv_fp="fp-abc",
        )
    assert agent._pending_corrections == []


def test_low_confidence_refuted_never_queues_even_forced(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _refuted(conf=0.5), trajectory_id="traj-s3", conv_fp="fp-abc",
            force_correction=True,
        )
    assert agent._pending_corrections == []


# ---------- correction dedup (churn brake) ----------


def test_identical_correction_is_not_stacked(agent, monkeypatch):
    """Observed churn (2026-07-18, Rick Dangerous): repeated refutes on
    the same conversation queued the same banner repeatedly; each banner
    led the next turn and drove more 'corrective' grinding. An identical
    (note, conversation) pair must queue at most once."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        for _ in range(3):
            agent._record_late_verdict(
                _refuted(issues=["same problem"]),
                trajectory_id="traj-d1", conv_fp="fp-same",
            )
    assert len(agent._pending_corrections) == 1


def test_distinct_corrections_still_queue(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _refuted(issues=["problem A"]),
            trajectory_id="t1", conv_fp="fp-x")
        agent._record_late_verdict(
            _refuted(issues=["problem B"]),
            trajectory_id="t2", conv_fp="fp-x")
        # Same note but a DIFFERENT conversation is not a duplicate.
        agent._record_late_verdict(
            _refuted(issues=["problem A"]),
            trajectory_id="t3", conv_fp="fp-y")
    assert len(agent._pending_corrections) == 3


# ---------- handler threads the flag ----------


async def test_attach_handler_threads_force_correction(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []

    async def _verdict_coro():
        return _refuted(issues=["late stream refute"]), {"name": "web_search"}

    task = asyncio.ensure_future(_verdict_coro())
    await task  # complete before attaching: callback fires on next tick
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._attach_late_verdict_handler(
            task, "traj-s4", "fp-stream", force_correction=True)
        await asyncio.sleep(0)  # let add_done_callback run
    assert agent._pending_corrections
    assert agent._pending_corrections[0]["conv"] == "fp-stream"


# ---------- source-level wiring pins ----------


def _agent_source():
    import ghost_agent.core.agent as agent_mod
    return inspect.getsource(agent_mod)


def test_stream_gate_block_is_wired():
    src = _agent_source()
    assert "VERIFIER GATE (STREAM)" in src
    gate = src.split("VERIFIER GATE (STREAM)")[1][:6000]
    # Late-only: spawns the pure computation, never the gated/inline form.
    assert "_compute_verifier_verdict(" in gate
    assert "_compute_verifier_verdict_gated" not in gate
    # Uses the eager captures, not the deleted outer variables.
    assert "tools_run_this_turn=stream_tools_snapshot" in gate
    assert "messages=stream_verify_messages" in gate
    assert "stream_conv_fp" in gate
    # Streamed replies have no inline note — the banner must be forced.
    assert "force_correction=True" in gate
    # Toolless turns are skipped (no no-op task, no noise line).
    assert "_find_substantive_tool_for_verifier" in gate


def test_stream_gate_captures_are_eager():
    """The fingerprint must be computed from the FULL live message list
    BEFORE the outer finally deletes it — messages[-10:] loses the first
    user message that fingerprints the conversation, and a mismatched
    fingerprint silently drops every queued correction."""
    src = _agent_source()
    assert "stream_verify_messages = list(messages)" in src
    assert ("stream_conv_fp = self._conversation_fingerprint(messages)"
            in src)
    # Captures happen at closure-creation time, i.e. before stream_wrapper.
    assert (src.index("stream_conv_fp = self._conversation_fingerprint")
            < src.index("async def stream_wrapper"))


def test_stream_gate_strips_think_from_claim():
    src = _agent_source()
    gate = src.split("VERIFIER GATE (STREAM)")[1][:6000]
    assert "<think>" in gate
    assert "final_ai_content=_sv_claim" in gate


def test_stream_gate_every_branch_is_loud():
    """The first live deploy was totally silent on streamed project turns
    and no debug channel is captured — a gate whose skip reasons are
    invisible cannot be distinguished from a gate that never ran. Every
    skip branch and the exception path must log at a visible level."""
    src = _agent_source()
    gate = src.split("VERIFIER GATE (STREAM)")[1][:6000]
    assert gate.count("stream gate:") >= 4  # 3 skips + deferred + failed
    assert "stream gate spawn failed" in gate
    assert 'level="WARNING"' in gate
