"""Canonical test builders (IMPROVEMENTS.md #26).

Measured duplication before this: 22 separate `def agent()` fixtures, ~94 files
constructing `GhostAgent` by hand, 233 `SimpleNamespace` fakes, 17 copies of a
`FakeBgTasks` shim — all re-deriving the same GhostContext shape. That coupled
the suite to the monolith's internals and made any constructor change fan out
across ~90 files (which actively deterred the agent.py refactor, #5).

`make_context(**overrides)` and `make_agent(**overrides)` centralize that shape.
New tests should use them; the two are also exposed as the `agent_context` /
`built_agent` conftest fixtures. Every field is overridable so a test that needs
a specific stub passes it in rather than rebuilding the whole context.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def make_context(**overrides):
    """Build a MagicMock GhostContext with the fields nearly every agent test
    needs, pre-wired with sensible inert defaults. Pass keyword overrides to
    replace any field (e.g. ``make_context(memory_system=None)``)."""
    from ghost_agent.core.agent import GhostContext

    ctx = MagicMock(spec=GhostContext)

    args = MagicMock()
    args.temperature = 0.7
    args.max_context = 8000
    args.smart_memory = 0.0
    args.use_planning = False
    args.model = "Qwen-Test"
    args.perfect_it = False
    args.native_tools = False
    args.api_key = "test-key"
    args.no_memory = False
    args.bio_time_scale = 1.0
    args.bio_deterministic = False
    ctx.args = args

    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test Response", "tool_calls": []}}]
    })
    llm.foreground_requests = 0
    llm.foreground_tasks = 0
    ctx.llm_client = llm

    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.memory_system.search_items = MagicMock(return_value=[])
    ctx.graph_memory = MagicMock()
    ctx.graph_memory.get_neighborhood = MagicMock(return_value=[])
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="")
    ctx.scratchpad._data = {}
    ctx.verifier = None
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"

    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def make_agent(context=None, **overrides):
    """Build a real ``GhostAgent`` over ``make_context(**overrides)`` (or the
    provided context). Constructs via the normal ``__init__`` so init-time
    wiring (semaphores, flag reads, guards) is exercised."""
    from ghost_agent.core.agent import GhostAgent

    ctx = context if context is not None else make_context(**overrides)
    return GhostAgent(ctx)


class FakeBgTasks:
    """The recurring FastAPI ``BackgroundTasks`` shim — records scheduled
    callables without running them. Was hand-copied into ~17 test files."""

    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append((fn, args, kwargs))
