"""§4UI (2026-11-12): the coding "Specialist" persona is gone, and the sandbox
listing + cwd pin follow what the request DOES, not keywords.

Measured before the change: of 189 requests that wrote or ran code
(10-07..11-12), the keyword classifier picked the persona for 82 — the
webOS request 8646c6c8 ("implement a webOS…") got neither the persona nor
the sandbox listing."""
from unittest.mock import AsyncMock, MagicMock

import pytest

import ghost_agent.core.prompts as P
from ghost_agent.core.agent import GhostAgent, GhostContext, _request_touches_sandbox


def test_the_persona_and_its_tool_xml_variants_are_gone():
    for name in ("SPECIALIST_SYSTEM_PROMPT", "SPECIALIST_TOOL_XML_LEGACY", "SPECIALIST_TOOL_XML_NATIVE"):
        assert not hasattr(P, name)


@pytest.mark.parametrize("intent,project,tools,want", [
    (False, None, [], False),                                         # plain chat
    (True, None, [], True),                                           # keywords still count
    (False, "abc123def456", [], True),                                # a bound project
    (False, None, [{"name": "file_system"}], True),                   # touched the sandbox
    (False, None, [{"name": "execute", "_synthetic": True}], True),   # a refused try counts
    (False, None, [{"name": "web_search"}], False),                   # other tools do not
    (False, None, [{"name": "browser"}], False),                      # r1 m1: web research is not the sandbox
    (False, None, [{"name": "workspace"}], False),                    # r1 m1: a read-only ledger
    (False, None, [{"name": "lab"}], True),                           # r1 m1: lab writes the workspace
    (False, None, ["file_system", None], False),                      # malformed rows ignored
])
def test_sandbox_context_follows_what_the_request_does(intent, project, tools, want):
    assert _request_touches_sandbox(intent, project, tools) is want


@pytest.fixture
def ctx():
    c = MagicMock(spec=GhostContext)
    c.args = MagicMock(); c.args.temperature = 0.7; c.args.max_context = 4096; c.args.smart_memory = 0.0
    c.sandbox_dir = "/tmp/sandbox"; c.memory_dir = "/tmp/memory"; c.tor_proxy = None
    c.llm_client = AsyncMock()
    c.llm_client.chat_completion.return_value = {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    c.profile_memory = MagicMock(); c.profile_memory.get_context_string.return_value = ""
    c.scratchpad = MagicMock(); c.scratchpad.list_all.return_value = ""
    c.memory_system = MagicMock(); c.cached_sandbox_state = None
    return c


async def _blob(ctx, text):
    agent = GhostAgent(ctx); agent.available_tools = {}
    await agent.handle_chat({"messages": [{"role": "user", "content": text}], "model": "m"}, MagicMock())
    msgs = ctx.llm_client.chat_completion.call_args[0][0]["messages"]
    return "\n".join(str(m.get("content")) for m in msgs)


async def test_a_project_request_without_coding_words_gets_the_sandbox_listing(ctx, monkeypatch):
    # the conversation's focus row binds the project at request start
    import ghost_agent.tools.projects as TP
    monkeypatch.setattr(TP, "begin_request_focus",
                        lambda c: setattr(c, "current_project_id", "cd174b7ffc68"))
    blob = await _blob(ctx, "make it look awesome and give it a couple of dummy apps")
    assert "CURRENT SANDBOX STATE" in blob
    assert "SPECIALIST SUBSYSTEM" not in blob


async def test_plain_chat_gets_no_sandbox_listing(ctx, monkeypatch):
    import ghost_agent.tools.projects as TP
    monkeypatch.setattr(TP, "begin_request_focus", lambda c: setattr(c, "current_project_id", None))
    blob = await _blob(ctx, "Tell me about the history of jazz music.")
    assert "CURRENT SANDBOX STATE" not in blob


async def test_a_public_reply_gets_no_bound_project_listing(ctx, monkeypatch):
    # r1 m6: the briefing is withheld on a public reply — the listing and the
    # project id in the cwd pin were not
    import ghost_agent.tools.projects as TP
    from ghost_agent.core import agent as A
    monkeypatch.setattr(TP, "begin_request_focus",
                        lambda c: setattr(c, "current_project_id", "cd174b7ffc68"))
    monkeypatch.setattr(A, "reply_is_public", lambda: True)
    blob = await _blob(ctx, "make it look awesome and give it a couple of dummy apps")
    assert "cd174b7ffc68" not in blob
    assert "CURRENT SANDBOX STATE" not in blob
