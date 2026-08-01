"""§4F Phase 3 — stable-prefix-hash regression with Phase-3 features ACTIVE.

Prerequisite for any Phase-3 default flip (journal §4F "STILL TODO"): enabling
the test-time-scaling features must not perturb the KV-pinned stable prefix.
Under GHOST_PIN_TOOL_SCHEMAS=1 (prod) the stable injection rides the FIRST
user message and must stay byte-identical whether or not
GHOST_TTS_ADAPTIVE_BON / GHOST_VERIFY_LOGIT_EXPECT are set:

- the BoN pass (core/tts.py via _adaptive_bon_final) runs at loop exit and
  builds its candidate payloads on a COPY of the conversation — the live
  message list, and with it the pinned prefix, must never be mutated;
- the logit-expectation probe (core/verifier.py) issues its own off-main
  probe request and must never touch the main payload;
- nothing in the prompt-assembly region may read a Phase-3 env switch.

Harness mirrors test_kv_pin_stable_prefix.py (mocked GhostContext, payload
captured off the AsyncMock llm client).
"""
import copy
import hashlib
import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext

PHASE3_ENV = {
    "GHOST_TTS_ADAPTIVE_BON": "1",
    "GHOST_TTS_BON_K": "3",
    "GHOST_VERIFY_LOGIT_EXPECT": "1",
}


def _make_agent(*, playbook=""):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.native_tools = True
    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hello there", "tool_calls": []}}]})
    ctx.llm_client.worker_clients = None
    ctx.llm_client.vision_clients = None
    ctx.llm_client.critic_clients = None
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="profile-data")
    ctx.profile_memory.load = MagicMock(return_value={})
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.graph_memory = MagicMock()
    ctx.graph_memory.get_neighborhood = MagicMock(return_value=[])
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value=playbook)
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)


def _last_nonstream_payload(agent):
    return agent.context.llm_client.chat_completion.await_args.args[0]


def _first_user(payload):
    return next(m for m in payload["messages"] if m["role"] == "user")


def _system_slot(payload):
    return next(m for m in payload["messages"] if m["role"] == "system")


_BODY = {
    "messages": [{
        "role": "user",
        "content": "Explain the tradeoffs between two database migration strategies for me?",
    }],
    "model": "test",
    "stream": False,
}


async def _run_once(monkeypatch, *, phase3: bool):
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    for key in PHASE3_ENV:
        monkeypatch.delenv(key, raising=False)
    if phase3:
        for key, val in PHASE3_ENV.items():
            monkeypatch.setenv(key, val)
    agent = _make_agent(playbook="ALWAYS wrap paths in quotes.")
    await agent.handle_chat(copy.deepcopy(_BODY), MagicMock())
    return _last_nonstream_payload(agent)


@pytest.mark.asyncio
async def test_pinned_prefix_byte_identical_with_phase3_flags_on(monkeypatch):
    """Same request, flags off vs on: pinned first-user message and system
    slot must be byte-identical — the sha1 mirror of the live
    'prefill cache · stable-prefix h=' line must not move."""
    base = await _run_once(monkeypatch, phase3=False)
    with_p3 = await _run_once(monkeypatch, phase3=True)

    base_pin = _first_user(base)["content"]
    p3_pin = _first_user(with_p3)["content"]
    assert "<session_context>" in base_pin
    assert base_pin == p3_pin
    assert (hashlib.sha1(base_pin.encode("utf-8", "ignore")).hexdigest()[:8]
            == hashlib.sha1(p3_pin.encode("utf-8", "ignore")).hexdigest()[:8])
    assert _system_slot(base)["content"] == _system_slot(with_p3)["content"]


@pytest.mark.asyncio
async def test_pinned_first_message_stable_across_in_request_turns(monkeypatch):
    """Flags ON, one request, two loop turns (tool call then final): every
    in-request payload must carry the exact same pinned first-user bytes —
    the invariant the KV pin exists for. (Across REQUESTS the block may
    legitimately change: persona/steering reclassify per request.)"""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", "1")
    for key, val in PHASE3_ENV.items():
        monkeypatch.setenv(key, val)

    agent = _make_agent(playbook="ALWAYS wrap paths in quotes.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "no_such_tool", "arguments": "{}"},
        }]}}]},
        {"choices": [{"message": {"content": "final answer", "tool_calls": []}}]},
    ])
    await agent.handle_chat(copy.deepcopy(_BODY), MagicMock())

    calls = agent.context.llm_client.chat_completion.await_args_list
    pins = []
    for c in calls:
        payload = c.args[0]
        first_user = next(
            (m for m in payload.get("messages", []) if m.get("role") == "user"), None)
        if first_user and "<session_context>" in first_user.get("content", ""):
            pins.append(first_user["content"])
    assert len(pins) >= 2, "expected at least two in-request main-turn payloads"
    assert all(p == pins[0] for p in pins)


@pytest.mark.asyncio
async def test_adaptive_bon_never_mutates_live_messages(monkeypatch):
    """Drive the BoN pass directly (the hook only fires on a wobble verdict):
    candidates must be generated on a COPY sharing the pinned prefix, and the
    live conversation list must come back untouched."""
    for key, val in PHASE3_ENV.items():
        monkeypatch.setenv(key, val)
    agent = _make_agent()
    live = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "<session_context>stable</session_context>\n\nquestion?"},
        {"role": "assistant", "content": "draft answer"},
    ]
    snapshot = copy.deepcopy(live)

    winner, meta = await agent._adaptive_bon_final(
        messages=live, final_ai_content="draft answer",
        last_user_content="question?", model="test-model")

    # The live list is byte-identical — no in-place append, no edits.
    assert live == snapshot
    # 3 candidate generations (GHOST_TTS_BON_K=3) + 1 judge call.
    calls = agent.context.llm_client.chat_completion.await_args_list
    gen_calls = [c for c in calls if not c.kwargs]
    judge_calls = [c for c in calls if c.kwargs]
    assert len(gen_calls) == 3
    assert len(judge_calls) == 1
    # Every candidate payload extends the SAME pinned prefix (KV reuse) and
    # only APPENDS its instruction.
    for c in gen_calls:
        msgs = c.args[0]["messages"]
        assert msgs[:-1] == snapshot
        assert msgs[-1]["role"] == "user"
    # Judge rides the cheap pool, not the main slot.
    assert judge_calls[0].kwargs.get("use_worker") or judge_calls[0].kwargs.get("use_critic")
    # Judge reply here is unparseable — the original must ship untouched.
    assert winner == "draft answer"
    assert meta["substituted"] is False


class TestSourceInvariants:
    def test_assembly_region_reads_no_phase3_env(self):
        """No Phase-3 switch may be consulted before the injection is
        composed — the stable prefix must be a function of the request
        alone, never of TTS/probe configuration."""
        src = inspect.getsource(GhostAgent.handle_chat)
        assembly, _, _ = src.partition("self._compose_injection(")
        assert "self._compose_injection(" not in assembly  # partition hit the call
        assert "GHOST_TTS_" not in assembly
        assert "GHOST_VERIFY_LOGIT" not in assembly

    def test_bon_hook_is_gated_off_the_repair_path(self):
        # The call must live INSIDE the `if not _do_repair:` guard — two
        # independent substring checks would pass even if it were moved
        # out, so assert on the guarded region itself.
        src = inspect.getsource(GhostAgent.handle_chat)
        assert "if not _do_repair:" in src
        guarded = src.split("if not _do_repair:", 1)[1].split("if _do_repair:", 1)[0]
        assert "_adaptive_bon_final" in guarded

    def test_bon_candidates_built_on_a_copy(self):
        src = inspect.getsource(GhostAgent._adaptive_bon_final)
        assert '"messages": list(messages) + [{' in src

    def test_probe_lives_in_verifier_not_agent(self):
        src = inspect.getsource(GhostAgent.handle_chat)
        assert "GHOST_VERIFY_LOGIT" not in src
