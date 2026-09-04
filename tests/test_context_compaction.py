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

Cross-cutting invariant (⚠ REVISED 2026-09-04, §4ET): on a
final-generation turn the native schema STAYS in `payload["tools"]`
and the call is suppressed with `tool_choice: "none"` instead. The
old rule dropped the key, which cost a full re-prefill: this
template renders `# Tools` BEFORE the system text, so an absent key
leaves a 19-character common prefix and the whole prompt — at its
largest of the request, on the turn the user is waiting for —
re-evaluates from token 3 (measured live: 6,568 tokens / 5.8 s on a
two-turn greeting). `tool_choice: "none"` renders byte-identical
bytes, so the temptation is removed for free, and the text-only
promise is kept by the drop guard rather than by hiding the schema.
NOTE #1 above (the XML/prompt-side schema skip) is unaffected — that
block sits mid-prompt, not ahead of the system slot.
"""

import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, GhostContext


@pytest.fixture(autouse=True)
def _force_use_planning_treatment(monkeypatch):
    """The use_planning experiment arm (2026-08-07) withholds the planner
    on control/unenrolled turns by design; tests here that set
    ``args.use_planning = True`` are exercising the PLANNER PATH, so pin
    this one arm to treatment. Other experiments keep their real
    (unenrolled → control) behavior."""
    import importlib

    # Patch BOTH module identities: tests import `ghost_agent.*` while some
    # files (and prod) import `src.ghost_agent.*` — two distinct module
    # objects with separate `arm_for` attributes (the production-import-
    # shape trap). Patching only one leaves the other gate live.
    for _modname in ("ghost_agent.core.experiments",
                     "src.ghost_agent.core.experiments"):
        try:
            _exp = importlib.import_module(_modname)
        except ImportError:
            continue
        _real = _exp.arm_for
        monkeypatch.setattr(
            _exp, "arm_for",
            lambda ctx, name, req_id="", _e=_exp, _r=_real: (
                _e.TREATMENT if name == "use_planning"
                else _r(ctx, name, req_id)))


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
async def test_native_tools_drops_xml_format_scaffolding():
    """Contract REVERSED 2026-07-31 (corruption root-cause fix): the old
    rule ("suppress only the schema, keep the XML parsing rules as the
    fallback contract") was ablation-proven to be the trigger of the
    native tool_call corruption — teaching the XML dialect alongside the
    template's native format made the model emit hybrid XML, and the
    upstream parser merged every stacked call (8/8 corrupt with the
    scaffolding, 13/13 clean without; journal §6). The agent's XML PARSER
    remains available for stragglers — but the prompt must no longer
    TEACH the dialect on the native path. See test_native_tool_header.py."""
    agent = _make_agent(native_tools=True, llm_response="answer")
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = _payload_after(agent)

    all_content = _all_content(payload)
    # The XML dialect is GONE from the native prompt…
    assert "<function name=" not in all_content
    assert "CDATA" not in all_content
    # …while parallelism stays invited and the schema pointer is informative.
    assert "PARALLEL EXECUTION" in all_content
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
async def test_final_generation_turn_suppresses_the_call_not_the_schema():
    """Cross-cutting: on a final-generation turn the model must not call a
    tool — but the native `tools` array must STAY ATTACHED.

    ⚠ REVERSED 2026-09-04 (§4ET). This used to assert `"tools" not in
    main_payload`, on the rationale that "sending tools tempts the model to
    call one instead of answering". The intent was right; the mechanism was
    catastrophic for latency. The Ornith/Qwen chat template renders the
    `# Tools` block BEFORE the system text, so removing the key leaves a
    19-character common prefix and the WHOLE prompt — at its largest of the
    request, on the turn the user is waiting for — re-prefills from token 3.
    Measured live (req 5d15ffb9): 6,568 tokens / 5.8s thrown away on a
    two-turn greeting.

    Suppression moved to `tool_choice: "none"`, which renders a
    byte-identical prompt (verified on /v1/chat/completions: 7,529
    prompt_tokens either way, 7,525 of them cached) so the temptation is
    removed without the re-prefill. The original intent is now pinned
    BEHAVIOURALLY below — a tool call emitted on such a turn is dropped
    rather than dispatched — which is a stronger guarantee than hiding the
    schema ever was.
    """
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
    assert "tools" in main_payload, (
        "Final-generation turn dropped payload['tools'] — the template renders "
        "# Tools before the system text, so an absent key re-prefills the "
        "entire prompt from token 3 (§4ET)."
    )
    assert main_payload["tool_choice"] == "none", (
        "the call must be suppressed via tool_choice — the one channel that "
        "changes no rendered bytes"
    )
    # The prompt still tells the model to answer in prose.
    assert "DO NOT emit any <tool_call>" in _all_content(main_payload)


@pytest.mark.asyncio
async def test_final_generation_drops_a_tool_call_instead_of_dispatching_it(caplog):
    """The behavioural half of the guarantee above.

    Hiding the schema was never what kept a final-generation turn text-only —
    the drop guard was. Pin the guard directly, because the schema is now
    visible to the model on exactly these turns: `tool_choice:"none"` stops
    llama.cpp PARSING a call, but the model can still emit `<tool_call>` XML
    into `content` (measured against the live server), which this agent's own
    XML parser turns back into calls.
    """
    agent = _make_agent(native_tools=True, use_planning=True, llm_response="answer")
    planner_response = {
        "choices": [{"message": {"content": (
            '{"thought": "explain conceptually",'
            ' "tree_update": {"id": "root", "description": "x", "status": "DONE", "children": []},'
            ' "next_action_id": "none",'
            ' "required_tool": "none"}'
        ), "tool_calls": []}}]
    }
    # The model disobeys and calls a tool on the text-only turn.
    main_response = {"choices": [{"message": {
        "content": "Let me check.",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "execute",
                                     "arguments": '{"command": "rm -rf /tmp/x"}'}}],
    }}]}
    captured = []

    async def mock_chat_capture(payload, *a, **kw):
        captured.append(payload)
        return planner_response if len(captured) == 1 else main_response

    agent.context.llm_client.chat_completion = mock_chat_capture
    body = {"messages": [{"role": "user", "content": "Run a quick lookup"}], "model": "test"}
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        await agent.handle_chat(body, MagicMock())

    assert any("Dropping" in r.message and "tool_call" in r.message
               for r in caplog.records), (
        "a tool_call emitted on a final-generation turn was NOT dropped — with "
        "the schema now attached on these turns, this guard is what keeps the "
        "promise that the schema's absence used to keep")


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
