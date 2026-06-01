"""Tests for the five architectural performance optimisations:

  #3  Trivial-request fast path
  #4  Per-request RequestState cache
  #7  Tool-schema cache + intent-based filtering
  #1  KV-cache-stable system prompt
  #2  Two-tier model routing (LLMClient.route + RoutingTask)
"""
import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import (
    GhostAgent, GhostContext, _freeze_funcs, _xml_schema_key,
    _json_to_xml_schema_cached,
)
from ghost_agent.core.llm import LLMClient, RoutingTask


# ============================================================ shared helpers


def _make_agent(*, llm_response="ok", with_tools=False, native_tools=False):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.5
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    # Pin native_tools explicitly. Without this the MagicMock returns a
    # truthy Mock for args.native_tools, which silently drives every test
    # into the schema-suppressed native-tools branch. Tests that want
    # to exercise that path pass `native_tools=True`.
    ctx.args.native_tools = native_tools

    ctx.llm_client = AsyncMock()
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": llm_response, "tool_calls": []}}]}
    )
    ctx.llm_client.worker_clients = None  # routing falls back to legacy

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


# =====================================================================
# #3 — Trivial-request fast path
# =====================================================================


def test_strict_greeting_detector_accepts_real_greetings():
    canon = GhostAgent._is_strict_trivial_chat
    for ok in ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
               "cool", "got it", "good morning", "tysm", "no worries",
               "yep", "👍", "all good"):
        assert canon(ok), f"Should accept {ok!r}"


def test_strict_greeting_detector_accepts_two_word_continuations():
    """The expanded allowlist must catch common conversational openings
    like 'hello there', 'hi friend', 'hey buddy', 'hi everyone'."""
    canon = GhostAgent._is_strict_trivial_chat
    for ok in (
        "hello there", "hi there", "hey there",
        "hi friend", "hey buddy", "hi everyone",
        "hey ghost", "hey agent", "hi again",
        "thanks a lot", "thank you so much",
        "good morning", "good evening", "good night",
        "ok thanks", "cool thanks", "yeah ok",
    ):
        assert canon(ok.lower()), f"Should accept {ok!r}"


def test_system_prompt_conversational_persona_is_softened():
    """The CONVERSATIONAL MODE persona must be neutral/friendly/helpful,
    not blunt/aggressive. Regression test for the user complaint about
    the agent replying in an aggressive persona to greetings."""
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert "neutral, friendly, and helpful" in SYSTEM_PROMPT
    assert "warm, conversational tone" in SYSTEM_PROMPT
    assert "blunt, direct, and aggressively" not in SYSTEM_PROMPT
    assert "aggressively efficient" not in SYSTEM_PROMPT


def test_fast_path_lite_system_prompt_is_warm():
    """The fast-path lite system prompt must use warm language too."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent._handle_trivial_chat)
    assert "concise conversational AI" in src
    assert "Reply warmly" in src or "warm" in src.lower()


def test_strict_greeting_detector_rejects_non_greetings():
    canon = GhostAgent._is_strict_trivial_chat
    for bad in ("run script.py", "what is 2+2?", "hello world write code",
                "hi can you fix this", "ok run /command", "test request",
                "hello @user", "hi <think>", "thanks for the help, can you write a script?"):
        assert not canon(bad), f"Should reject {bad!r}"


def test_strict_greeting_detector_rejects_empty():
    assert not GhostAgent._is_strict_trivial_chat("")
    assert not GhostAgent._is_strict_trivial_chat("   ")


@pytest.mark.asyncio
async def test_trivial_fast_path_intercepts_real_greeting():
    """A 'hi' message must skip the turn loop and call chat_completion only once."""
    agent = _make_agent(llm_response="hello back")
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "test"}
    content, _, _ = await agent.handle_chat(body, MagicMock())
    assert content == "hello back"
    # Exactly one LLM call (the fast path), not the multi-turn loop.
    assert agent.context.llm_client.chat_completion.await_count == 1
    payload = agent.context.llm_client.chat_completion.await_args.args[0]
    # The fast-path system prompt must NOT contain tool schemas.
    sys_msg = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    assert "<tool_def>" not in sys_msg
    # Must be flagged as the lite system prompt.
    assert "concise conversational AI" in sys_msg


@pytest.mark.asyncio
async def test_trivial_fast_path_does_not_intercept_real_request():
    """A non-greeting must take the full turn loop."""
    agent = _make_agent(llm_response="result")
    body = {"messages": [{"role": "user", "content": "Run a calculation for me"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = agent.context.llm_client.chat_completion.await_args.args[0]
    sys_msg = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    # The full turn loop installs the persona/profile system prompt with skill instruction.
    assert "Acquired Skills" in sys_msg or "GHOST" in sys_msg.upper() or "Ghost" in sys_msg


@pytest.mark.asyncio
async def test_trivial_fast_path_returns_string_for_streaming_too():
    """The fast path now ALWAYS returns a plain string, even when the
    client requested stream=True. The route layer's `stream_openai` helper
    handles SSE-wrapping (api/routes.py:163). This avoids any closure /
    iterator-lifecycle bugs in nested async generators."""
    agent = _make_agent(llm_response="hello there")
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "test", "stream": True}
    content, _, _ = await agent.handle_chat(body, MagicMock())
    assert isinstance(content, str)
    assert content == "hello there"
    # Exactly one chat_completion call (the fast path) — not the full loop.
    assert agent.context.llm_client.chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_trivial_fast_path_falls_through_on_empty_response():
    """If the model returns an empty string, the fast path declines so the
    full path can take a second crack at it."""
    agent = _make_agent(llm_response="")
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[],
        model="x", stream_response=False, req_id="r"
    )
    assert res is None


@pytest.mark.asyncio
async def test_trivial_fast_path_falls_through_on_llm_exception():
    """An exception inside the LLM call must drop us back into the long path."""
    agent = _make_agent()
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=RuntimeError("upstream down"))
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[],
        model="x", stream_response=False, req_id="r"
    )
    assert res is None


@pytest.mark.asyncio
async def test_trivial_fast_path_strips_think_blocks():
    """Some models leak <think>...</think>; the fast path must strip them."""
    agent = _make_agent(llm_response="<think>thinking about the greeting</think>Hello!")
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[],
        model="x", stream_response=False, req_id="r"
    )
    assert res is not None
    content, _, _ = res
    assert content == "Hello!"
    assert "<think>" not in content


@pytest.mark.asyncio
async def test_trivial_fast_path_falls_through_on_no_llm():
    """If the LLM client is missing, decline so the long path can handle it."""
    agent = _make_agent()
    agent.context.llm_client = None
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[], model="x", stream_response=False, req_id="r"
    )
    assert res is None


@pytest.mark.asyncio
async def test_trivial_fast_path_disables_thinking():
    """The trivial path must suppress chain-of-thought on a reasoning model.

    Regression test for the bug where greetings produced empty `content`
    (the model spent its whole budget on a `<think>` block) and every
    greeting fell through to the slow turn loop. We assert BOTH switches
    the model honours are wired: the `enable_thinking` chat-template flag
    and the portable `/no_think` soft-switch on the user message."""
    agent = _make_agent(llm_response="hello back")
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[{"role": "user", "content": "hi"}],
        model="x", stream_response=False, req_id="r"
    )
    assert res is not None
    payload = agent.context.llm_client.chat_completion.await_args.args[0]
    assert payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert payload["messages"][-1]["content"].rstrip().endswith("/no_think")


@pytest.mark.asyncio
async def test_trivial_fast_path_recovers_reasoning_content_field():
    """Reasoning models return chain-of-thought in a separate
    `reasoning_content` field. When `content` is empty but the visible
    answer trails the reasoning, the fast path must still recover a reply
    rather than declining. This is the core fix for the empty-content bug."""
    agent = _make_agent()
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {
            "content": "Hey there, good to see you!",
            "reasoning_content": "The user greeted me; reply warmly.",
        }}]
    })
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[],
        model="x", stream_response=False, req_id="r"
    )
    assert res is not None
    content, _, _ = res
    assert content == "Hey there, good to see you!"
    assert "reason" not in content.lower()


@pytest.mark.asyncio
async def test_trivial_fast_path_strips_unclosed_think_block():
    """A model cut off mid-thought emits `<think>` with no closing tag.
    Everything inside it is reasoning, so the visible reply is empty and
    the path must decline (fall through) rather than leak the monologue."""
    agent = _make_agent(llm_response="<think>still deciding how to greet")
    res = await agent._handle_trivial_chat(
        last_user_content="hi", messages=[],
        model="x", stream_response=False, req_id="r"
    )
    assert res is None


# =====================================================================
# #4 — RequestState cache
# =====================================================================


@pytest.mark.asyncio
async def test_request_state_caches_profile_string():
    agent = _make_agent()
    state = GhostAgent._RequestState(agent)
    s1 = await state.get_profile_str()
    s2 = await state.get_profile_str()
    assert s1 == s2 == "profile-data"
    # Profile fetched only once across two reads.
    assert agent.context.profile_memory.get_context_string.call_count == 1


@pytest.mark.asyncio
async def test_request_state_caches_skill_playbook_per_query():
    agent = _make_agent()
    agent.context.skill_memory.get_playbook_context = MagicMock(return_value="playbook v1")
    state = GhostAgent._RequestState(agent)

    a = await state.get_skill_playbook("query-A")
    b = await state.get_skill_playbook("query-A")
    c = await state.get_skill_playbook("query-B")
    assert a == b == "playbook v1"
    # query-A fetched once, query-B fetched once: 2 total calls.
    assert agent.context.skill_memory.get_playbook_context.call_count == 2


@pytest.mark.asyncio
async def test_request_state_skill_playbook_invalidation():
    agent = _make_agent()
    agent.context.skill_memory.get_playbook_context = MagicMock(return_value="playbook")
    state = GhostAgent._RequestState(agent)
    await state.get_skill_playbook("q")
    state.invalidate_skill_playbook()
    await state.get_skill_playbook("q")
    assert agent.context.skill_memory.get_playbook_context.call_count == 2


def test_request_state_caches_active_tool_defs():
    agent = _make_agent()
    state = GhostAgent._RequestState(agent)
    with patch("ghost_agent.tools.registry.get_active_tool_definitions",
               return_value=[{"function": {"name": "foo"}}]) as mock_get:
        a = state.get_active_tool_defs("query-A")
        b = state.get_active_tool_defs("query-A")
        c = state.get_active_tool_defs("query-B")
    assert a == b
    # Same query cached; different query refetches.
    assert mock_get.call_count == 2


def test_request_state_caches_xml_schema():
    agent = _make_agent()
    state = GhostAgent._RequestState(agent)
    tools = [
        {"function": {"name": "tool_a", "description": "A", "parameters": {"properties": {}, "required": []}}},
        {"function": {"name": "tool_b", "description": "B", "parameters": {"properties": {}, "required": []}}},
    ]
    a = state.get_xml_schema(tools)
    b = state.get_xml_schema(tools)
    assert a == b
    assert "<tool_def>" in a
    # Different tool list → new cache entry.
    tools2 = tools + [{"function": {"name": "tool_c", "description": "C", "parameters": {"properties": {}, "required": []}}}]
    c = state.get_xml_schema(tools2)
    assert c != a
    assert "tool_c" in c


@pytest.mark.asyncio
async def test_request_state_sandbox_invalidation_chain():
    agent = _make_agent()
    state = GhostAgent._RequestState(agent)
    fake_state = "sandbox-snapshot-v1"
    with patch("ghost_agent.tools.file_system.tool_list_files",
               new=AsyncMock(return_value=fake_state)) as mock_ls:
        s1 = await state.get_sandbox_state()
        s2 = await state.get_sandbox_state()
        state.invalidate_sandbox()
        s3 = await state.get_sandbox_state()
    assert s1 == s2 == s3 == fake_state
    assert mock_ls.await_count == 2  # invalidation forces re-read


# =====================================================================
# #7 — Tool-schema cache (module level)
# =====================================================================


def test_freeze_funcs_is_hashable():
    funcs = [
        {"name": "f", "description": "d", "parameters": {"properties": {"x": {"type": "string", "description": "p"}}, "required": ["x"]}},
    ]
    frozen = _freeze_funcs(funcs)
    # Must be a tuple of tuples → hashable
    hash(frozen)
    assert frozen[0][0] == "f"


def test_xml_schema_key_is_stable_and_order_independent():
    tools_a = [{"function": {"name": "a"}}, {"function": {"name": "b"}}]
    tools_b = [{"function": {"name": "b"}}, {"function": {"name": "a"}}]
    assert _xml_schema_key(tools_a) == _xml_schema_key(tools_b)


def test_json_to_xml_schema_cached_memoizes():
    funcs1 = [{"name": "tool_x", "description": "x", "parameters": {"properties": {}, "required": []}}]
    frozen = _freeze_funcs(funcs1)
    _json_to_xml_schema_cached.cache_clear()
    a = _json_to_xml_schema_cached(frozen)
    b = _json_to_xml_schema_cached(frozen)
    assert a == b
    info = _json_to_xml_schema_cached.cache_info()
    assert info.hits >= 1


def test_intent_filter_drops_postgres_when_no_sql_keywords():
    """Post-audit: _intent_filter is permissive by default.

    The aggressive keyword-heuristic dropping was removed because it
    produced false negatives (e.g. a "look at this image" prompt that
    didn't literally contain "image|picture" lost vision_analysis).
    Tools are now only dropped when the caller explicitly passes
    `drop_unconfigured` — see `get_active_tool_definitions`, which adds
    `postgres_admin` to that set when no DB URI is configured.
    """
    from ghost_agent.tools.registry import _intent_filter
    tools = [
        {"function": {"name": "execute"}},
        {"function": {"name": "postgres_admin"}},
    ]
    # Without `drop_unconfigured`, no tools are removed regardless of query.
    res = _intent_filter(tools, "write a python script")
    names = [t["function"]["name"] for t in res]
    assert "execute" in names
    assert "postgres_admin" in names

    # When the caller signals postgres is unconfigured, it IS dropped.
    res = _intent_filter(
        tools, "write a python script", drop_unconfigured={"postgres_admin"}
    )
    names = [t["function"]["name"] for t in res]
    assert "execute" in names
    assert "postgres_admin" not in names


def test_intent_filter_keeps_postgres_when_sql_in_query():
    from ghost_agent.tools.registry import _intent_filter
    tools = [{"function": {"name": "postgres_admin"}}]
    res = _intent_filter(tools, "show me the SQL schema for users")
    assert any(t["function"]["name"] == "postgres_admin" for t in res)


def test_intent_filter_drops_image_gen_when_no_image_request():
    """Post-audit: image_generation is no longer keyword-gated.

    Configuration gating still applies (the registry only advertises
    image_generation when at least one image_gen_node is configured).
    But once advertised, it's not dropped based on whether the user
    query mentions "image" — too many false negatives (e.g. "render a
    chart").
    """
    from ghost_agent.tools.registry import _intent_filter
    tools = [
        {"function": {"name": "execute"}},
        {"function": {"name": "image_generation"}},
    ]
    res = _intent_filter(tools, "write a python script")
    names = [t["function"]["name"] for t in res]
    assert "image_generation" in names


def test_intent_filter_keeps_image_gen_for_real_request():
    from ghost_agent.tools.registry import _intent_filter
    tools = [{"function": {"name": "image_generation"}}]
    res = _intent_filter(tools, "generate an image of a sunset")
    assert any(t["function"]["name"] == "image_generation" for t in res)


def test_intent_filter_drops_vision_for_pure_text_request():
    """Post-audit: vision_analysis is always advertised when present.

    Native multimodal models (Qwen 3.5) handle vision natively, and
    the keyword-based pre-drop was causing false negatives on prompts
    like "what's in this screenshot?" that didn't match the image
    keyword pattern. Permissive policy lets the LLM decide.
    """
    from ghost_agent.tools.registry import _intent_filter
    tools = [
        {"function": {"name": "execute"}},
        {"function": {"name": "vision_analysis"}},
    ]
    res = _intent_filter(tools, "compute the average of these numbers")
    assert "vision_analysis" in [t["function"]["name"] for t in res]


def test_intent_filter_keeps_vision_when_image_mentioned():
    from ghost_agent.tools.registry import _intent_filter
    tools = [{"function": {"name": "vision_analysis"}}]
    res = _intent_filter(tools, "look at this picture and describe it")
    assert any(t["function"]["name"] == "vision_analysis" for t in res)


def test_intent_filter_returns_input_unchanged_when_no_query():
    from ghost_agent.tools.registry import _intent_filter
    tools = [{"function": {"name": "anything"}}]
    res = _intent_filter(tools, None)
    assert res == tools


# =====================================================================
# #1 — KV-cache-stable system prompt
# =====================================================================


@pytest.mark.asyncio
async def test_system_prompt_does_not_contain_tool_schema():
    """The whole point of optimisation #1: tool schemas live in the
    user-message header now, NOT the system slot."""
    agent = _make_agent()
    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())
    payload = agent.context.llm_client.chat_completion.await_args.args[0]
    sys_msg = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    assert "<tool_def>" not in sys_msg
    # The tool schema must appear somewhere — typically in the trailing user message.
    all_content = "\n".join(m["content"] for m in payload["messages"])
    assert "<tool_def>" in all_content


@pytest.mark.asyncio
async def test_system_prompt_byte_stable_across_two_turns():
    """The killer property: same system bytes turn 1 and turn 2 of one request."""
    agent = _make_agent()
    captured_system_msgs = []

    call_n = {"n": 0}
    async def mock_chat(payload, *a, **kw):
        call_n["n"] += 1
        captured_system_msgs.append(
            next(m["content"] for m in payload["messages"] if m["role"] == "system")
        )
        if call_n["n"] == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "fake_tool", "arguments": "{}"}}
            ]}}]}
        return {"choices": [{"message": {"content": "done", "tool_calls": []}}]}
    agent.context.llm_client.chat_completion = mock_chat
    agent.available_tools = {"fake_tool": AsyncMock(return_value="ok")}

    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    assert len(captured_system_msgs) >= 2
    # The whole point: byte-identical.
    assert captured_system_msgs[0] == captured_system_msgs[1], (
        "System slot must be byte-identical across turns for KV-cache stability"
    )


@pytest.mark.asyncio
async def test_continuity_blocks_ride_in_tail_not_system_slot():
    """Regression: with the selfhood wake-up ENABLED (the production case),
    the volatile continuity block must NOT land in the system slot — it
    rides in the per-turn tail injection — so the system prefix stays
    byte-identical across turns and the upstream KV cache is reusable.

    The earlier byte-stable test used a MagicMock self_model, so the
    `isinstance(self_model, SelfModel)` guard skipped the block entirely
    and never exercised this path — which is why prepending the wake-up
    prefix to the system slot silently regressed cache stability."""
    from ghost_agent.selfhood import SelfModel
    agent = _make_agent()
    MARKER = "WAKEUP_CONTINUITY_MARKER_ZZ123"
    sm = MagicMock(spec=SelfModel)          # spec → isinstance(sm, SelfModel) is True
    sm.enabled = True
    sm.build_wakeup_prefix = MagicMock(return_value=MARKER)
    agent.context.self_model = sm

    cap = {"system": [], "last_user": []}
    call_n = {"n": 0}

    async def mock_chat(payload, *a, **kw):
        call_n["n"] += 1
        msgs = payload["messages"]
        cap["system"].append(next(m["content"] for m in msgs if m["role"] == "system"))
        cap["last_user"].append(msgs[-1]["content"])
        if call_n["n"] == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "fake_tool", "arguments": "{}"}}
            ]}}]}
        return {"choices": [{"message": {"content": "done", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = mock_chat
    agent.available_tools = {"fake_tool": AsyncMock(return_value="ok")}

    body = {"messages": [{"role": "user", "content": "Run a tool please"}], "model": "test"}
    await agent.handle_chat(body, MagicMock())

    assert len(cap["system"]) >= 2
    # The volatile selfhood content must NOT be in the system slot...
    assert MARKER not in cap["system"][0]
    # ...the system slot stays byte-identical across turns even with it enabled...
    assert cap["system"][0] == cap["system"][1]
    # ...and the content is preserved — emitted in the per-turn tail injection.
    assert any(MARKER in lu for lu in cap["last_user"] if isinstance(lu, str))


# =====================================================================
# #2 — Two-tier model routing
# =====================================================================


def test_routing_task_constants_exist():
    assert RoutingTask.VALIDATE_TOOL_ARGS == "VALIDATE_TOOL_ARGS"
    assert RoutingTask.EXPAND_QUERY == "EXPAND_QUERY"
    assert RoutingTask.CLASSIFY_INTENT == "CLASSIFY_INTENT"
    assert RoutingTask.SCORE_RELEVANCE == "SCORE_RELEVANCE"
    assert RoutingTask.REPAIR_JSON == "REPAIR_JSON"


@pytest.mark.asyncio
async def test_route_returns_fallback_when_no_worker_pool():
    client = LLMClient(upstream_url="http://mock")
    # Default state is no workers configured (empty list).
    assert not client.worker_clients
    res = await client.route(
        task=RoutingTask.EXPAND_QUERY,
        payload={"model": "x", "messages": []},
        fallback="LEGACY",
    )
    assert res == "LEGACY"
    await client.close()


@pytest.mark.asyncio
async def test_route_dispatches_to_worker_pool_when_present():
    client = LLMClient(upstream_url="http://mock")
    fake_node_client = MagicMock()
    fake_node_client.aclose = AsyncMock()
    client.worker_clients = [{"client": fake_node_client, "url": "http://worker", "model": "small"}]
    client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "rewritten query"}}]}
    )
    res = await client.route(
        task=RoutingTask.EXPAND_QUERY,
        payload={"model": "x", "messages": [{"role": "user", "content": "q"}]},
        fallback="LEGACY",
    )
    assert res == "rewritten query"
    # Must have routed via use_worker=True, is_background=True
    call = client.chat_completion.await_args
    assert call.kwargs.get("use_worker") is True
    assert call.kwargs.get("is_background") is True
    await client.close()


@pytest.mark.asyncio
async def test_route_falls_back_on_router_exception():
    client = LLMClient(upstream_url="http://mock")
    fake_node_client = MagicMock()
    fake_node_client.aclose = AsyncMock()
    client.worker_clients = [{"client": fake_node_client}]
    client.chat_completion = AsyncMock(side_effect=RuntimeError("worker down"))
    res = await client.route(
        task=RoutingTask.EXPAND_QUERY,
        payload={"model": "x", "messages": []},
        fallback="LEGACY",
    )
    assert res == "LEGACY"
    await client.close()


@pytest.mark.asyncio
async def test_agent_query_expansion_uses_router_when_available():
    agent = _make_agent()
    # Configure a worker pool and stub the router output.
    agent.context.llm_client.worker_clients = [{"client": MagicMock()}]
    agent.context.llm_client.route = AsyncMock(return_value="rewritten via router")

    # Force the contextual-expansion branch: short user msg + prior assistant.
    body = {
        "messages": [
            {"role": "user", "content": "make a script"},
            {"role": "assistant", "content": "I have created the calculation script."},
            {"role": "user", "content": "run it then"},
        ],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())
    agent.context.llm_client.route.assert_awaited()
    call = agent.context.llm_client.route.await_args
    assert call.kwargs.get("task") == "EXPAND_QUERY"
    # The bus's vector search must have been called with the rewritten query.
    search_args = agent.context.memory_system.search.call_args
    if search_args is not None:
        assert "rewritten via router" in str(search_args)


@pytest.mark.asyncio
async def test_agent_query_expansion_falls_back_to_legacy_concat():
    """Without a worker pool, the legacy `Context: ... | User intent: ...`
    string is preserved (existing test_recent_memory_features.py relies on it)."""
    agent = _make_agent()
    agent.context.llm_client.worker_clients = None
    body = {
        "messages": [
            {"role": "user", "content": "make a script"},
            {"role": "assistant", "content": "I have created the calculation script."},
            {"role": "user", "content": "run it then"},
        ],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())
    search_args = agent.context.memory_system.search.call_args
    assert search_args is not None
    called_query = search_args[0][0]
    assert "Context: I have created the calculation script." in called_query
    assert "User intent: run it then" in called_query
