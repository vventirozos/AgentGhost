"""Async-critic mode (GHOST_CRITIC_ASYNC=1).

In async mode the verifier never sits on the critical path: the in-loop
repair-before-ship is skipped and the verdict runs purely AFTER the
response ships. A high-confidence REFUTED then (a) scrubs the turn's
lessons and (b) queues a correction surfaced at the top of the NEXT turn.

These tests cover the knobs, the late-verdict recorder, the next-turn
surfacing, and a full handle_chat run proving the repair is skipped and a
correction is queued.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


# --------------------------------------------------------------------------
# Fixtures / helpers (mirror test_verifier_auto_repair's harness)
# --------------------------------------------------------------------------

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


def _verdict(v, conf=0.95, issues=None, reasoning="reason"):
    return VerifyResult(verdict=v, confidence=conf, reasoning=reasoning, issues=issues or [])


def _make_verifier(verdicts):
    v = MagicMock()
    v.llm_client = MagicMock()
    shared = AsyncMock(side_effect=list(verdicts))
    v.verify_claim = shared
    v.verify_code_output = shared
    v.verify_visual = AsyncMock(return_value=None)
    return v, shared


def _tool_call(name="execute", args='{"content": "print(40+2)"}', tid="t1"):
    return {"choices": [{"message": {"content": "working", "tool_calls": [
        {"id": tid, "function": {"name": name, "arguments": args}}]}}]}


def _final(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


async def _run(agent, user, llm_side_effects):
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=list(llm_side_effects))
    body = {"messages": [{"role": "user", "content": user}], "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        result, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())
    return result


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------

def test_async_flag_off_by_default(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    assert agent._critic_async_enabled() is False


def test_async_flag_on(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    assert agent._critic_async_enabled() is True


def test_async_forces_pure_async_gate(agent, monkeypatch):
    # Even a positive explicit gate budget is overridden to 0 in async mode.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "30")
    assert agent._critic_gate_timeout() == 0.0


# --------------------------------------------------------------------------
# _record_late_verdict
# --------------------------------------------------------------------------

def test_late_refuted_queues_correction_in_async(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    # _record_late_verdict spawns the lesson-retraction task; in production it
    # runs inside a task's done-callback (live loop), so stub the spawn here.
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _verdict(VerifyVerdict.REFUTED, conf=0.9, issues=["wrong number"]),
            trajectory_id="traj-1",
        )
    assert agent._pending_corrections
    assert "wrong number" in agent._pending_corrections[0]["note"]


def test_late_refuted_does_not_queue_when_not_async(agent, monkeypatch):
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task") as _spawn:
        agent._record_late_verdict(
            _verdict(VerifyVerdict.REFUTED, conf=0.9, issues=["wrong"]),
            trajectory_id="traj-1",
        )
    # Sync mode: retraction still scheduled, but no next-turn correction queued.
    assert agent._pending_corrections == []
    _spawn.assert_called()  # retraction (lesson scrub) still happens in sync mode


def test_late_confirmed_queues_nothing(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_late_verdict(_verdict(VerifyVerdict.CONFIRMED), trajectory_id="t")
    assert agent._pending_corrections == []


def test_late_low_confidence_refuted_queues_nothing(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._record_late_verdict(
            _verdict(VerifyVerdict.REFUTED, conf=0.5, issues=["maybe"]), trajectory_id="t",
        )
    assert agent._pending_corrections == []


# --------------------------------------------------------------------------
# Late-verdict → trajectory-corpus backfill (the skills-auto producer fix:
# in async mode the verdict lands after the trajectory was recorded
# UNKNOWN, so without this the corpus NEVER contains a passed-with-tools
# chat turn and graduation has zero input — 2058 UNKNOWN / 0 eligible in
# production on 2026-07-05)
# --------------------------------------------------------------------------

def _traj(tid="t1", outcome="unknown"):
    from ghost_agent.distill.schema import Trajectory
    return Trajectory(id=tid, user_request="do the thing",
                      final_response="done", outcome=outcome)


def _wire_corpus(agent, traj):
    """Attach a mock collector + the correction cache holding `traj`."""
    from collections import OrderedDict
    collector = MagicMock()
    agent.context.trajectory_collector = collector
    agent.context._recent_trajectories_for_correction = OrderedDict(
        [("fp", traj)] if traj is not None else [])
    return collector


def _run_sync(agent, verdict, tid="t1"):
    """Drive _record_late_verdict with spawn_task executing inline so the
    sidecar write can be asserted synchronously."""
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task",
               new=lambda coro: asyncio.run(coro)):
        agent._record_late_verdict(verdict, trajectory_id=tid)


def test_late_confirmed_backfills_passed_outcome(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    traj = _traj()
    collector = _wire_corpus(agent, traj)
    _run_sync(agent, _verdict(VerifyVerdict.CONFIRMED, conf=0.95))
    collector.update_outcome.assert_called_once_with(
        "t1", "passed", reason="", source="verifier_late")
    assert traj.outcome == "passed"   # in-process cache updated too


def test_late_confirmed_never_upgrades_failed(agent, monkeypatch):
    # A shape-heuristic / structural FAILED must not be upgraded by a
    # late text CONFIRMED — mirrors resolve_turn_outcome's priorities.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    traj = _traj(outcome="failed")
    collector = _wire_corpus(agent, traj)
    _run_sync(agent, _verdict(VerifyVerdict.CONFIRMED, conf=0.95))
    collector.update_outcome.assert_not_called()
    assert traj.outcome == "failed"


def test_late_confirmed_cache_miss_skips_promotion(agent, monkeypatch):
    # Can't prove the recorded outcome was UNKNOWN → conservative skip.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    collector = _wire_corpus(agent, None)
    _run_sync(agent, _verdict(VerifyVerdict.CONFIRMED, conf=0.95))
    collector.update_outcome.assert_not_called()


def test_late_refuted_backfills_failed_even_on_cache_miss(agent, monkeypatch):
    # Failure direction is fail-safe: lands regardless of cache state,
    # so the Reflector / PRM get their negative.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent._pending_corrections = []
    collector = _wire_corpus(agent, None)
    _run_sync(agent, _verdict(VerifyVerdict.REFUTED, conf=0.9, issues=["wrong"]))
    collector.update_outcome.assert_called_once()
    args, kwargs = collector.update_outcome.call_args
    assert args == ("t1", "failed")
    assert "verifier refuted (late)" in kwargs["reason"]
    assert kwargs["source"] == "verifier_late"


def test_late_low_confidence_no_backfill(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent._pending_corrections = []
    traj = _traj()
    collector = _wire_corpus(agent, traj)
    _run_sync(agent, _verdict(VerifyVerdict.CONFIRMED, conf=0.6))
    _run_sync(agent, _verdict(VerifyVerdict.REFUTED, conf=0.5, issues=["maybe"]))
    collector.update_outcome.assert_not_called()
    assert traj.outcome == "unknown"


# --------------------------------------------------------------------------
# _consume_pending_corrections
# --------------------------------------------------------------------------

def test_consume_stages_banner_without_touching_messages(agent):
    agent._pending_corrections = ["the script printed 7, not 42"]
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "next question"},
    ]
    with patch("ghost_agent.core.agent.pretty_log"):
        out = agent._consume_pending_corrections(msgs)
    # Messages are NOT mutated (no fragile model-instruction injection)...
    assert out == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "next question"},
    ]
    # ...the banner is staged for deterministic prepend, and the queue cleared.
    assert "the script printed 7, not 42" in agent._active_correction
    assert agent._active_correction.startswith("⚠️ **Correction to my previous answer:**")
    assert agent._pending_corrections == []


def test_take_active_correction_is_one_shot(agent):
    agent._active_correction = "BANNER\n\n"
    assert agent._take_active_correction() == "BANNER\n\n"
    # Cleared after taking, so it prepends to exactly one reply.
    assert agent._take_active_correction() == ""


def test_consume_corrections_noop_when_empty(agent):
    agent._pending_corrections = []
    msgs = [{"role": "user", "content": "hi"}]
    with patch("ghost_agent.core.agent.pretty_log"):
        out = agent._consume_pending_corrections(msgs)
    assert out == [{"role": "user", "content": "hi"}]
    assert agent._take_active_correction() == ""


def test_consume_sets_turn_flag_to_skip_trivial_path(agent):
    """A pending correction sets the turn flag so the trivial-chat fast path
    (which returns via its own route and would bypass the prepend) is skipped;
    no pending → flag off."""
    agent._pending_corrections = ["prev answer was truncated"]
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections([{"role": "user", "content": "q"}])
    assert agent._correction_active_this_turn is True

    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections([{"role": "user", "content": "q"}])
    assert agent._correction_active_this_turn is False


# --------------------------------------------------------------------------
# Integration: async mode skips the in-loop repair, ships the answer,
# and queues a correction from the post-response verdict.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_mode_ships_unrepaired_and_queues_correction(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 7")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.REFUTED, issues=["wrong number"])])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()

    # Only TWO llm turns provided: tool call + final. If async wrongly
    # triggered a repair it would demand a third and raise StopAsyncIteration.
    result = await _run(agent, "compute the answer and report it", [
        _tool_call(),
        _final("The answer is 7."),
    ])

    # The (refuted) answer shipped as-is — no before-ship repair.
    assert "The answer is 7." in result

    # The verdict runs AFTER the response: let the spawned task + done
    # callback drain, then assert the correction was queued for next turn.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if agent._pending_corrections:
            break
    assert agent._pending_corrections, "async verdict should have queued a correction"
    assert "wrong number" in agent._pending_corrections[0]["note"]
    assert vmock.await_count == 1  # exactly one verdict, off the critical path


@pytest.mark.asyncio
async def test_staged_correction_is_prepended_to_next_reply(agent, monkeypatch):
    """End-to-end: a correction queued by a prior turn is deterministically
    prepended to THIS turn's reply (not left to the model to weave in)."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent._pending_corrections = ["previous reply was truncated mid-code"]

    result = await _run(agent, "what else can you tell me", [
        _final("Here is more information."),
    ])

    assert result.startswith("⚠️ **Correction to my previous answer:**")
    assert "previous reply was truncated mid-code" in result
    assert "Here is more information." in result
    # One-shot: the queue and stage are both cleared.
    assert agent._pending_corrections == []
    assert agent._take_active_correction() == ""


@pytest.mark.asyncio
async def test_sync_mode_still_repairs(agent, monkeypatch):
    """Control: with async OFF, the same REFUTED verdict DOES trigger the
    in-loop repair (a third llm turn), proving the flag is what gates it."""
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 7")
    verifier, vmock = _make_verifier([
        _verdict(VerifyVerdict.REFUTED, issues=["wrong number"]),
        _verdict(VerifyVerdict.CONFIRMED),
    ])
    agent.context.verifier = verifier

    result = await _run(agent, "compute the answer and report it", [
        _tool_call(),
        _final("The answer is 7."),          # refuted
        _final("Corrected: the answer is 42."),  # repaired
    ])
    assert "42" in result
    assert "The answer is 7." not in result
    assert agent._pending_corrections == []  # sync mode never queues


# --------------------------------------------------------------------------
# Conversation scoping + TTL + cap (the shared-singleton pollution fix)
# --------------------------------------------------------------------------

def _queue(agent, note, conv, ts=None):
    """Directly enqueue a tagged correction as _record_late_verdict would."""
    import time as _t
    if not isinstance(getattr(agent, "_pending_corrections", None), list):
        agent._pending_corrections = []
    agent._pending_corrections.append(
        {"note": note, "conv": conv, "ts": ts if ts is not None else _t.monotonic()}
    )


def _msgs(first_user):
    return [{"role": "system", "content": "SYS"}, {"role": "user", "content": first_user}]


def test_record_late_verdict_tags_conversation(agent, monkeypatch):
    """A late REFUTED verdict stores a conversation-tagged, timestamped dict."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(
            _verdict(VerifyVerdict.REFUTED, conf=0.9, issues=["bad claim"]),
            trajectory_id="t", conv_fp="CONV_A",
        )
    entry = agent._pending_corrections[0]
    assert entry["note"] == "bad claim"
    assert entry["conv"] == "CONV_A"
    assert isinstance(entry["ts"], float)


def test_correction_surfaces_only_in_originating_conversation(agent):
    """The core fix: a correction tagged for conversation A surfaces on A's
    next turn but NOT on an unrelated conversation B's turn (it stays queued)."""
    fp_a = agent._conversation_fingerprint(_msgs("first question in A"))
    _queue(agent, "A's answer was wrong", fp_a)

    # A DIFFERENT conversation must NOT see it, and it must remain queued.
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections(_msgs("unrelated question in B"))
    assert agent._take_active_correction() == ""
    assert agent._correction_active_this_turn is False
    assert len(agent._pending_corrections) == 1  # held for conversation A

    # A's own next turn surfaces it and clears it.
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections(_msgs("first question in A"))
    banner = agent._take_active_correction()
    assert "A's answer was wrong" in banner
    assert agent._pending_corrections == []


def test_correction_expires_after_ttl(agent):
    """A correction whose conversation never returns is dropped after the TTL
    rather than lingering — even when its own conversation finally shows up."""
    from ghost_agent.core.agent import _CORRECTION_TTL
    import time as _t
    fp = agent._conversation_fingerprint(_msgs("stale conversation"))
    _queue(agent, "too old to matter", fp, ts=_t.monotonic() - _CORRECTION_TTL - 5)
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections(_msgs("stale conversation"))
    assert agent._take_active_correction() == ""
    assert agent._pending_corrections == []  # expired → dropped


def test_correction_queue_is_capped(agent, monkeypatch):
    """Queueing more than _CORRECTION_MAX corrections keeps only the newest."""
    from ghost_agent.core.agent import _CORRECTION_MAX
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent._glog.spawn_task"):
        for i in range(_CORRECTION_MAX + 3):
            agent._record_late_verdict(
                _verdict(VerifyVerdict.REFUTED, conf=0.9, issues=[f"issue {i}"]),
                trajectory_id="t", conv_fp="CONV",
            )
    assert len(agent._pending_corrections) == _CORRECTION_MAX
    # The newest survive; the oldest are evicted.
    notes = [c["note"] for c in agent._pending_corrections]
    assert notes[-1] == f"issue {_CORRECTION_MAX + 2}"
    assert "issue 0" not in notes


def test_untagged_string_correction_still_surfaces(agent):
    """Back-compat: a bare-string correction (direct injection / legacy) is
    unscoped and surfaces regardless of conversation."""
    agent._pending_corrections = ["legacy untagged note"]
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._consume_pending_corrections(_msgs("any conversation"))
    assert "legacy untagged note" in agent._take_active_correction()
    assert agent._pending_corrections == []
