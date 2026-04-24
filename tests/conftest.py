import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test Response", "tool_calls": []}}]
    })
    return client


@pytest.fixture
def disable_self_play_templates(monkeypatch):
    """Opt-in fixture: disable the synthetic-self-play template fast path
    so a test that mocks `chat_completion` for challenge generation
    actually sees the LLM call it expects.

    By default, `synthetic_self_play` skips the LLM when either (a) the
    cluster_key matches a deterministic template in
    `ghost_agent.core.challenge_templates.TEMPLATES`, or (b) the
    frontier tracker reports cold-start and `pick_random_template` can
    provide a fallback. Tests that pre-date the template bank expect
    the LLM path — they request this fixture to switch templates off.
    Tests that actually exercise template behavior should NOT request
    it (the template functions stay live)."""
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *args, **kwargs: None,
    )

@pytest.fixture(autouse=True)
def inject_global_stream_adapter(monkeypatch):
    """Injects a `stream_chat_completion` adapter on MagicMock LLM clients so
    tests that only stub `chat_completion` still work when the agent tries
    to stream.

    Real LLM clients are NEVER overwritten — we only replace
    `stream_chat_completion` when the attribute is missing OR is itself a
    `MagicMock` / `AsyncMock`. The wrap is one-shot per test (monkeypatch
    auto-reverts), so a re-entrant guard is unnecessary.
    """
    from ghost_agent.core.agent import GhostAgent
    original_init = GhostAgent.__init__

    def wrapped_init(self, context, *args, **kwargs):
        original_init(self, context, *args, **kwargs)

        async def mock_stream_chat_completion(*a, **kw):
            import json
            try:
                res = await context.llm_client.chat_completion(*a, **kw)
                msg = res.get("choices", [{}])[0].get("message", {})
                delta = dict(msg)
                if "tool_calls" in delta and isinstance(delta["tool_calls"], list):
                    for i, tc in enumerate(delta["tool_calls"]):
                        tc["index"] = i
                chunk = {"choices": [{"delta": delta}]}
                yield f"data: {json.dumps(chunk)}\n".encode('utf-8')
            except Exception as e:
                raise e

        if context and hasattr(context, "llm_client") and context.llm_client is not None:
            if not hasattr(context.llm_client, "stream_chat_completion") or isinstance(context.llm_client.stream_chat_completion, (MagicMock, AsyncMock)):
                context.llm_client.stream_chat_completion = mock_stream_chat_completion

    monkeypatch.setattr(GhostAgent, '__init__', wrapped_init)

@pytest.fixture
def temp_dirs():
    base = Path(tempfile.mkdtemp())
    sandbox = base / "sandbox"
    memory = base / "memory"
    sandbox.mkdir()
    memory.mkdir()
    yield {"base": base, "sandbox": sandbox, "memory": memory}
    shutil.rmtree(base)

@pytest.fixture
def mock_context(temp_dirs, mock_llm):
    context = MagicMock()
    context.sandbox_dir = temp_dirs["sandbox"]
    context.memory_dir = temp_dirs["memory"]
    context.llm_client = mock_llm
    context.args = MagicMock()
    context.args.anonymous = True
    context.args.max_context = 32768
    context.args.smart_memory = 0.0 # Prevent comparison error
    context.args.verbose = False
    context.args.temperature = 0.1
    
    # Mock return values as strings to prevent TypeErrors in string manipulation
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = "User Profile Data"
    
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "Scratchpad Data"
    
    return context
