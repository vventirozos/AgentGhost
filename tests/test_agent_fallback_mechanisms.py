import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    
    mock_llm = AsyncMock()
    context.llm_client = mock_llm
    
    mock_args = MagicMock()
    mock_args.max_context = 10000
    mock_args.smart_memory = 0.0
    mock_args.temperature = 0.7
    mock_args.perfect_it = False
    context.args = mock_args
    
    context.journal = MagicMock()
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = lambda: ""
    context.sandbox_dir = "/tmp/sandbox"
    context.cached_sandbox_state = None
    
    return context

@pytest.mark.asyncio
async def test_agent_raw_json_fallback(mock_context):
    mock_llm = mock_context.llm_client
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": '{"name": "execute", "arguments": {"filename": "test.py", "content": "print(1)"}}'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS: I have finished."}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    # Verify the tool was extracted and called successfully
    tools_called = [m["name"] for m in body["messages"] if m.get("role") == "tool"]
    assert "execute" in tools_called
    # Check that tool response says executed_successfully
    tool_responses = [m["content"] for m in body["messages"] if m.get("role") == "tool" and m.get("name") == "execute"]
    assert len(tool_responses) > 0
    assert tool_responses[0] == "executed_successfully"

@pytest.mark.asyncio
async def test_agent_conversational_filler_fallback(mock_context):
    mock_llm = mock_context.llm_client
    # Simulate conversational filler mentioning execute but not doing it
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": 'I will use the execute tool.'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS"}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    # We expect our fallback to have appended the SYSTEM ALERT message warning against conversational filler
    alert_messages = [m for m in body["messages"] if m.get("role") == "user" and "SYSTEM ALERT: You provided conversational text mentioning the tool" in m.get("content", "")]
    assert len(alert_messages) == 1
    assert "`execute`" in alert_messages[0]["content"]
    assert "but you DID NOT output the actual XML `<tool_call>` block" in alert_messages[0]["content"]


# ── filler guard: req 8ecb1424 (11-09) false positives ──────────────────
from ghost_agent.core.agent import _filler_promised_tool

_TOOLS = ["run", "portfolio", "execute", "web_search", "jobs", "browser",
          "notify_operator", "file_system"]


@pytest.mark.parametrize("text", [
    # req 8ecb1424 shape: casual offer naming plain-word tools
    "Good evening! I'll be here — we could run through the news, take a "
    "little portfolio look, or just chat.",
    "Whenever you're ready I'll pull up the portfolio or run something. "
    "Want me to?",
    # 10-10 probe-mz shape: saying a tool is NOT available
    "Let me be straightforward: on this channel I don't have memory tools, "
    "only web_search and a few others.",
    # intent word far from the tool name
    "I'll be honest, it has been a long day and the web_search results "
    "were thin.",
    # last sentence is an offer / question
    "I'll keep it short. Should I check web_search for that?",
    "Let me know if you want me to use web_search.",
    # plain English verb, not written as a tool
    "Let me run through the highlights: markets up, weather mild.",
    "I'll execute the plan tomorrow as you said.",
    "",
])
def test_filler_guard_ignores_casual_mentions(text):
    assert _filler_promised_tool(text, _TOOLS) == ""


@pytest.mark.parametrize("text,tool", [
    ("I will use the execute tool.", "execute"),
    ("Let me run `web_search` for the latest numbers.", "web_search"),
    ("Let me search with web search now.", "web_search"),
    ("I'll call `portfolio` to fetch your positions.", "portfolio"),
    ("Running notify_operator now.", "notify_operator"),
    ("Sure. Let me check the file_system for that config.", "file_system"),
])
def test_filler_guard_catches_real_promises(text, tool):
    assert _filler_promised_tool(text, _TOOLS) == tool


def test_filler_guard_sentence_boundary():
    # the tool name in the NEXT sentence is not the promise's object
    assert _filler_promised_tool(
        "I'll keep it brief. web_search found nothing new.", _TOOLS) == ""


@pytest.mark.asyncio
async def test_filler_guard_casual_reply_ships_without_alert(mock_context):
    """req 8ecb1424: the chat reply must ship on the first generation."""
    mock_llm = mock_context.llm_client
    reply = ("The night is young! I'll be here — a news catch-up, a little "
             "portfolio curiosity, or we could run through tomorrow. "
             "Whatever you like.")
    mock_llm.chat_completion.side_effect = [
        {"choices": [{"message": {"content": reply}}]},
        {"choices": [{"message": {"content": "SHOULD NOT BE REACHED"}}]},
    ]
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"run": AsyncMock(), "portfolio": AsyncMock()}
    body = {"messages": [{"role": "user", "content": "i got some time till bed"}],
            "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req8ec")
    alerts = [m for m in body["messages"] if m.get("role") == "user"
              and "SYSTEM ALERT: You provided conversational text" in str(m.get("content"))]
    assert alerts == []
    assert mock_llm.chat_completion.call_count == 1


# ── review round 1 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("text,tool", [
    # sign-off after a real promise is not an offer
    ("I will run web_search now. Let me know if you need more.", "web_search"),
    # curly apostrophes, I'm going / about to
    ("let’s use web_search for that.", "web_search"),
    ("I’m going to call `portfolio` for you.", "portfolio"),
    ("I'm about to run web_search.", "web_search"),
    # backticked underscored name at word 4 still counts
    ("Let me check it with `web_search` now.", "web_search"),
    ("Let me run it with web_search.", "web_search"),
    # "not only" is not a negation (review r2)
    ("I'll not only run web_search but also check the news.", "web_search"),
])
def test_filler_guard_review_catches(text, tool):
    assert _filler_promised_tool(text, _TOOLS) == tool


@pytest.mark.parametrize("text", [
    # word 5 is past the window
    "Let me run it quickly with the web_search.",
    # negation
    "I will not call web_search.",
    "I'll never use web_search without asking.",
    # a colon ends the clause (10-10 probe-mz shape, tool close to marker)
    "Let me be straightforward: web_search is not on this channel.",
    # "ill" is not "I'll"
    "Sorry you felt ill web_search can wait till tomorrow.",
    # a question followed by an emoji is still a question
    "Let me run `web_search` for you. Want me to? 😊",
])
def test_filler_guard_review_silent(text):
    assert _filler_promised_tool(text, _TOOLS) == ""


@pytest.mark.asyncio
async def test_filler_guard_nudges_once_per_request(mock_context):
    """review r1: with no strike behind it the nudge must not repeat — a
    model that repeats the promise ships on the second generation."""
    mock_llm = mock_context.llm_client
    promise = {"choices": [{"message": {"content": "I will use the execute tool."}}]}
    mock_llm.chat_completion.side_effect = [promise] * 6
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="ok")}
    body = {"messages": [{"role": "user", "content": "Run script"}],
            "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req-once")
    alerts = [m for m in body["messages"] if m.get("role") == "user"
              and "SYSTEM ALERT: You provided conversational text" in str(m.get("content"))]
    assert len(alerts) == 1
    assert mock_llm.chat_completion.call_count == 2


@pytest.mark.asyncio
async def test_filler_alert_from_an_earlier_request_does_not_suppress(mock_context):
    """review r2: an alert row from an EARLIER request (resent history)
    must not count as this request's nudge."""
    mock_llm = mock_context.llm_client
    mock_llm.chat_completion.side_effect = [
        {"choices": [{"message": {"content": "I will use the execute tool."}}]},
        {"choices": [{"message": {"content": "SUCCESS"}}]},
    ]
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="ok")}
    old_alert = ("SYSTEM ALERT: You provided conversational text mentioning "
                 "the tool `execute`, but you DID NOT output the actual XML.")
    body = {"messages": [
        {"role": "user", "content": "earlier ask"},
        {"role": "user", "content": old_alert},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Run script", "_request": True},
    ], "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req-hist")
    alerts = [m for m in body["messages"] if m.get("role") == "user"
              and "SYSTEM ALERT: You provided conversational text" in str(m.get("content"))]
    assert len(alerts) == 2  # the old row plus this request's own nudge
