"""Context-compaction optimisations.

Two related savings on the per-turn payload:

  #1 — Skip the XML tool schema entirely on final-generation turns.
      When the planner has set `force_final_response=True` (or the
      target tool is "none"), the model is being asked to answer in
      plain text and any tool_calls it emits are dropped downstream.
      Shipping the ~7.4K-token tool schema in that case is wasted
      bytes and pollutes the model's attention with options it can't
      use. The header is replaced with a tiny "no tools this turn"
      stanza that preserves the think-budget guidance.

  #2 — Don't double-ship schemas under --native-tools. When
      `args.native_tools=True`, schemas are advertised through the
      OpenAI-style `payload["tools"]` channel. Re-emitting the same
      definitions in the prompt XML is pure duplication. The XML
      *format* scaffolding (parsing rules, parallel-call guidance,
      CDATA hint) is preserved so the agent's XML parser still works
      as a fallback for models that emit the legacy shape; only the
      `<tool_def>...</tool_def>` block is suppressed.

Cross-cutting invariant: on a final-generation turn the native
schema is also dropped from `payload["tools"]` — the model is told
to answer, sending tools tempts it to call something instead.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext


def _make_agent(*, llm_response="ok", native_tools=False, use_planning=False):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = use_planning
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.native_tools = native_tools

    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": llm_response, "tool_calls": []}}]}
    )
    ctx.llm_client.worker_clients = None

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
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)


def _payload_after(agent):
    return agent.context.llm_client.chat_completion.await_args.args[0]


def _all_content(payload):
    return "\n".join(m.get("content", "") for m in payload["messages"])


# =====================================================================
# #2 — Don't double-ship schemas under --native-tools
# =====================================================================


@pytest.mark.asyncio
async def test_native_tools_suppresses_xml_schema_in_prompt():
    """With native_tools=True, `<tool_def>` blocks must NOT appear in
    the prompt — they're delivered via payload['tools'] instead."""
    agent = _make_agent(native_tools=True, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = _payload_after(agent)

    all_content = _all_content(payload)
    assert "<tool_def>" not in all_content, (
        "Native-tools mode must NOT ship XML <tool_def> blocks in the "
        "prompt — that's the duplication this optimisation removes."
    )
    # Native channel is wired up.
    assert "tools" in payload, "Native tools must be attached when native_tools=True"
    assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_native_tools_keeps_format_scaffolding():
    """Suppress only the schema, not the XML parsing rules — the agent's
    XML parser is the documented fallback path. Models that emit the
    legacy `<tool_call>` shape must still be supported."""
    agent = _make_agent(native_tools=True, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = _payload_after(agent)

    all_content = _all_content(payload)
    # Format scaffolding stays put.
    assert "<tool_call>" in all_content
    assert "PARALLEL EXECUTION" in all_content
    # The pointer that replaces the schema is informative.
    assert "advertised via the native" in all_content


@pytest.mark.asyncio
async def test_xml_only_mode_keeps_full_schema():
    """With native_tools=False, the full XML schema must still be in
    the prompt — that's the only way the model learns the tool set."""
    agent = _make_agent(native_tools=False, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = _payload_after(agent)

    all_content = _all_content(payload)
    assert "<tool_def>" in all_content, (
        "XML-only mode must ship the schema in the prompt; that's the "
        "only channel the model gets it from."
    )
    assert "tools" not in payload, (
        "Native tools must NOT be attached when native_tools=False"
    )


@pytest.mark.asyncio
async def test_native_tools_savings_are_real():
    """Sanity: the native-tools prompt is meaningfully shorter than the
    XML-only prompt. We're targeting a 5K+ char reduction; assert at
    least 4000 to leave slack for tool-list churn."""
    agent_xml = _make_agent(native_tools=False, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent_xml.handle_chat(body, MagicMock())
    xml_size = len(_all_content(_payload_after(agent_xml)))

    agent_native = _make_agent(native_tools=True, llm_response="answer")
    await agent_native.handle_chat(body, MagicMock())
    native_size = len(_all_content(_payload_after(agent_native)))

    saved = xml_size - native_size
    assert saved > 4000, (
        f"Expected at least 4000 chars saved by suppressing XML schema "
        f"under native_tools, got {saved} (xml={xml_size}, native={native_size})"
    )


# =====================================================================
# #1 — Skip schema on final-generation turns
# =====================================================================


@pytest.mark.asyncio
async def test_final_generation_turn_drops_xml_schema():
    """When the agent decides this turn is a final-answer turn (planner
    set required_tool=none, or force_final_response=True), the XML
    schema must NOT appear — the model is being told to answer the
    user, not to call a tool."""
    agent = _make_agent(native_tools=False, use_planning=False, llm_response="answer")
    # Patch the predicate by feeding a response that has no tool_calls
    # — the planner is off, so the canonical path runs without setting
    # force_final_response. To exercise the schema-skip we set the
    # disabled_tools/required_tool directly via a planner-style fixture.
    # The cleanest way is: plant `required_tool='none'` in the agent's
    # locals via a one-shot patch.
    #
    # Simpler: use_planning=True + a planner stub that returns required_tool='none'.
    pass  # Replaced by the explicit-planner test below.


@pytest.mark.asyncio
async def test_final_generation_turn_via_planner_drops_schema():
    """Drive the schema-skip through the planner: configure use_planning
    so the planner runs, and stub the planner output to return
    required_tool='none' / next_action_id='none'."""
    agent = _make_agent(native_tools=False, use_planning=True, llm_response="answer")

    # The planner runs as a separate chat_completion call BEFORE the
    # main turn. Return a planner JSON that says "answer directly".
    planner_response = {
        "choices": [{"message": {"content": (
            '{"thought": "user is asking conceptually, no tool needed",'
            ' "tree_update": {"id": "root", "description": "answer", "status": "DONE",'
            '   "children": []},'
            ' "next_action_id": "none",'
            ' "required_tool": "none"}'
        ), "tool_calls": []}}]
    }
    main_response = {"choices": [{"message": {"content": "Direct answer.", "tool_calls": []}}]}

    call_count = {"n": 0}

    async def mock_chat(payload, *a, **kw):
        call_count["n"] += 1
        # First call is planner; subsequent are the main turn.
        if call_count["n"] == 1:
            return planner_response
        return main_response

    agent.context.llm_client.chat_completion = mock_chat
    body = {"messages": [{"role": "user", "content": "Run a quick calculation for me"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    # The MAIN turn payload (call #2) must have NO <tool_def>.
    # We can't re-grab via await_args here because we replaced with a
    # plain async function, so capture inside the mock.
    captured = []

    async def mock_chat_capture(payload, *a, **kw):
        captured.append(payload)
        if len(captured) == 1:
            return planner_response
        return main_response

    agent2 = _make_agent(native_tools=False, use_planning=True, llm_response="answer")
    agent2.context.llm_client.chat_completion = mock_chat_capture
    await agent2.handle_chat(body, MagicMock())
    assert len(captured) >= 2, "Expected at least planner + main calls"
    main_payload = captured[1]
    main_content = _all_content(main_payload)
    assert "<tool_def>" not in main_content, (
        "Final-generation turn must drop the XML schema."
    )
    # The slim header is in place.
    assert "Final-generation turn" in main_content
    assert "DO NOT emit any <tool_call>" in main_content


@pytest.mark.asyncio
async def test_final_generation_turn_drops_native_tools_too():
    """Cross-cutting: on a final-generation turn the native `tools`
    array must also be suppressed. Sending tools tempts the model to
    call one instead of answering."""
    agent = _make_agent(native_tools=True, use_planning=True, llm_response="answer")
    planner_response = {
        "choices": [{"message": {"content": (
            '{"thought": "explain conceptually",'
            ' "tree_update": {"id": "root", "description": "x", "status": "DONE", "children": []},'
            ' "next_action_id": "none",'
            ' "required_tool": "none"}'
        ), "tool_calls": []}}]
    }
    main_response = {"choices": [{"message": {"content": "Direct answer.", "tool_calls": []}}]}
    captured = []

    async def mock_chat_capture(payload, *a, **kw):
        captured.append(payload)
        return planner_response if len(captured) == 1 else main_response

    agent.context.llm_client.chat_completion = mock_chat_capture
    body = {"messages": [{"role": "user", "content": "Run a quick lookup"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    assert len(captured) >= 2
    main_payload = captured[1]
    assert "tools" not in main_payload, (
        "Final-generation turn must NOT attach native tools, even when "
        "native_tools=True. Sending tools tempts the model to call one."
    )


@pytest.mark.asyncio
async def test_non_final_turn_keeps_native_tools_when_flag_on():
    """Regression guard for #2: on a normal tool-using turn with
    native_tools=True, payload['tools'] must still be attached."""
    agent = _make_agent(native_tools=True, use_planning=False, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a calculation"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = _payload_after(agent)
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"


# =====================================================================
# Predicate sync between hoisted-schema decision and canonical site
# =====================================================================


@pytest.mark.asyncio
async def test_schema_skip_picks_up_dynamic_state_force_final_via_required_tool():
    """The dynamic_state assembly later in the loop sets
    `force_final_response=True` when next_action_id=='none'. The
    schema-skip predicate runs BEFORE that line, so it must read the
    same signal directly from required_tool / next_action_id (which
    the planner already set). Otherwise we'd ship the schema on
    final-answer turns even though the model can't tool-call.
    """
    agent = _make_agent(native_tools=False, use_planning=True, llm_response="answer")
    captured = []

    async def mock_chat_capture(payload, *a, **kw):
        captured.append(payload)
        if len(captured) == 1:
            return {"choices": [{"message": {"content": (
                '{"thought": "answer",'
                ' "tree_update": {"id": "root", "description": "x", "status": "DONE", "children": []},'
                ' "next_action_id": "none",'
                ' "required_tool": "none"}'
            ), "tool_calls": []}}]}
        return {"choices": [{"message": {"content": "ans", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat_capture
    body = {"messages": [{"role": "user", "content": "Run an explanation request"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    main_payload = captured[1]
    assert "<tool_def>" not in _all_content(main_payload)
