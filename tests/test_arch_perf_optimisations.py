"""Tests for the five architectural performance optimisations:

  #3  Trivial-request fast path (removed §4RP-c; the greeting detector stays)
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
    # Bus prefers the per-item search_items entry point (2026-07-07); keep
    # the legacy search stub too for any direct callers.
    ctx.memory_system.search_items = MagicMock(return_value=[])
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.graph_memory = MagicMock()
    ctx.graph_memory.get_neighborhood = MagicMock(return_value=[])
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.sandbox_dir = "/tmp/sandbox"
    return GhostAgent(ctx)


# =====================================================================
# #3 — the strict greeting detector (its fast path was removed, §4RP-c)
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
async def test_a_greeting_takes_the_full_turn():
    """§4RP-c: the greeting fast path is gone — "hi" runs the normal turn
    with the full system prompt (the head the slot keeps), not a lite one."""
    agent = _make_agent(llm_response="hello back")
    body = {"messages": [{"role": "user", "content": "hi"}], "model": "test"}
    content, _, _ = await agent.handle_chat(body, MagicMock())
    assert "hello back" in content
    payload = agent.context.llm_client.chat_completion.await_args_list[0].args[0]
    sys_msg = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    assert "concise conversational AI" not in sys_msg
    assert "### ROLE AND IDENTITY" in sys_msg
    assert not hasattr(GhostAgent, "_handle_trivial_chat")


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
    # ⚠ SINGLE-SLOT NOW, NOT PER-KEY (§4DA post-redesign, lens B F1b):
    # serving tuned descriptions is not a pure read — it draws arms,
    # stamps the request and prunes on the ceiling — and a per-key cache
    # let one request SERVE TWICE with different tool sets, the last
    # call owning the stamps. The FIRST resolution is the advertised set
    # for the whole request, which is also optimisation #7's stated goal
    # (a byte-stable tool header across the request's turns).
    assert c is a
    assert mock_get.call_count == 1


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
async def test_continuity_blocks_ride_in_tail_not_system_slot(monkeypatch):
    """Regression: a volatile continuity block must NOT land in the system
    slot — it rides in the per-turn tail injection — so the system prefix
    stays byte-identical across turns and the upstream KV cache is reusable.

    Driven through the workspace wake-up prefix (the selfhood prefix was
    deleted in §4NM and the uncertainty block in §4NN); the cache-stability
    invariant this regression guards is unchanged."""
    from ghost_agent.workspace import WorkspaceModel
    agent = _make_agent()
    MARKER = "WAKEUP_CONTINUITY_MARKER_ZZ123"
    wm = MagicMock(spec=WorkspaceModel)
    wm.enabled = True
    wm.build_wakeup_prefix = MagicMock(return_value=MARKER)
    agent.context.workspace_model = wm
    agent.context.current_project_id = "p-continuity"
    # keep the bound project: the conversation reconcile would clear it here
    import ghost_agent.tools.projects as _pj
    monkeypatch.setattr(_pj, "reconcile_conversation", lambda *a, **k: None)

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
    # The volatile continuity content must NOT be in the system slot...
    assert MARKER not in cap["system"][0]
    # ...the system slot stays byte-identical across turns even with it enabled...
    assert cap["system"][0] == cap["system"][1]
    # ...and the content is preserved — emitted in the per-turn tail injection.
    assert any(MARKER in lu for lu in cap["last_user"] if isinstance(lu, str))


# =====================================================================
# #2 — Two-tier model routing
# =====================================================================


def test_every_routing_task_member_has_a_live_caller():
    """⚠ REPLACED A TAUTOLOGY. This asserted `RoutingTask.X == "X"` for five
    members — a constant compared with itself, which cannot fail and taught
    nobody anything. Four of those five had ZERO production callers: the class
    read as an inventory of what the routing layer does and was an
    aspirational list (R5 lens B). What is worth pinning is the opposite
    direction — that a member here corresponds to a real dispatch."""
    import re
    from pathlib import Path as _P

    src_root = _P(__file__).resolve().parents[1] / "src" / "ghost_agent"
    members = [n for n in vars(RoutingTask)
               if n.isupper() and not n.startswith("_")]
    assert members, "RoutingTask lost every member"

    # ⚠ AST, not a regex over raw text — `# was RoutingTask.EXPAND_QUERY`
    # counted as a caller (R7 lens C, A02).
    import ast as _ast
    used = set()
    for f in src_root.rglob("*.py"):
        if f.name == "llm.py":
            continue
        try:
            tree = _ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in _ast.walk(tree):
            if isinstance(n, _ast.Attribute) and isinstance(n.value, _ast.Name) \
                    and n.value.id == "RoutingTask":
                used.add(n.attr)
    orphans = [m for m in members if m not in used]
    assert not orphans, (
        f"RoutingTask members with no caller outside llm.py: {orphans}. "
        f"Either wire them up or delete them — a label nothing dispatches "
        f"describes a routing layer that does not exist.")


@pytest.mark.asyncio
async def test_route_returns_fallback_when_no_worker_pool():
    client = LLMClient(upstream_url="http://mock")
    # Default state is no workers configured (empty list).
    assert not client.worker_clients
    res = await client.route(
        task=RoutingTask.CLASSIFY_FAILURE,
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
        task=RoutingTask.CLASSIFY_FAILURE,
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
        task=RoutingTask.CLASSIFY_FAILURE,
        payload={"model": "x", "messages": []},
        fallback="LEGACY",
    )
    assert res == "LEGACY"
    await client.close()






@pytest.mark.asyncio
async def test_a_short_follow_up_is_retrieved_on_the_users_own_words():
    """§4NQ: no worker rewrite before the reply. With a worker pool present,
    a short anaphoric follow-up used to be rewritten by Nova (EXPAND_QUERY)
    and a compound request decomposed (DECOMPOSE_QUERY) — both were removed
    after a replay of real owner turns showed them changing retrieval mostly
    for the worse. The lesson fetch must get the message as typed."""
    agent = _make_agent()
    agent.context.llm_client.worker_clients = [{"client": MagicMock()}]
    agent.context.llm_client.route = AsyncMock(return_value="rewritten via router")
    body = {
        "messages": [
            {"role": "user", "content": "make a script"},
            {"role": "assistant", "content": "I have created the calculation script."},
            {"role": "user", "content": "run it then"},
        ],
        "model": "test",
    }
    await agent.handle_chat(body, MagicMock())
    tasks = [str(c.kwargs.get("task") or (c.args[0] if c.args else ""))
             for c in agent.context.llm_client.route.await_args_list]
    assert not [t for t in tasks if t in ("EXPAND_QUERY", "DECOMPOSE_QUERY")], tasks
    search_args = agent.context.skill_memory.get_playbook_items.call_args
    assert search_args is not None
    assert search_args[0][0] == "run it then"
