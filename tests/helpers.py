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

import re

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


def strip_js_comments(js: str) -> str:
    """Comment-free view of a JS source, for the text-assertion suites over
    ``interface/static/*`` (there is no JS runtime here).

    ⚠ ORDER MATTERS, and the natural order is WRONG. Stripping ``/* */``
    first lets a line comment containing ``/api/*`` open a phantom block
    comment that closes at the next real ``*/`` — measured 5,655 characters
    and 206 lines of app.js deleted, including ``installAuthFetch`` and
    ``X-Ghost-Key``. Positive assertions then fail loudly, but NEGATIVE ones
    ("this removed feature is gone") pass VACUOUSLY for anything inside the
    phantom span. Line comments first.

    Trailing comments must go too: stripping only line-initial comments left
    ``foo();  // r.status === 429`` satisfying assertions about the code.
    Two wrong versions preceded this one — ``(?<!:)`` truncated
    ``` `${wsProtocol}//${host}/ws` ``` mid-template-literal, and adding
    ``[;)}]`` made it worse because ``}`` is exactly the character before
    that ``//``. The invariant a real trailing comment always has and a
    protocol-relative URL never has is WHITESPACE (or line start) directly
    before the slashes.

    Three copies of this function existed; ``TestTheStripperItself`` in
    tests/test_webui_feedback_client.py pins THIS one."""
    js = re.sub(r"(?:(?<=\s)|^)//.*$", "", js, flags=re.M)
    return re.sub(r"/\*.*?\*/", "", js, flags=re.S)


def eval_js(source: str, expr: str, timeout: float = 20.0):
    """Run ``source`` under node and return ``JSON.parse``-able ``expr``.

    Text assertions over ``interface/static/*`` cannot see SCOPE, and that
    blind spot has teeth: a review fix that read ``res.status`` from a catch
    block in a *different* function than the one declaring ``res`` satisfied
    every text pin while raising a ReferenceError inside the error handler —
    i.e. the diagnostic path crashed and took the caller's retry with it.
    For small pure helpers, extract and EXECUTE instead.

    Raises RuntimeError with node's stderr on failure, so a ReferenceError
    is a test failure rather than a silent empty result. Skips (loudly) when
    node is absent."""
    import json
    import shutil
    import subprocess

    import pytest as _pytest

    node = shutil.which("node")
    if not node:
        # A skip here silently deletes EVERY behavioural JS pin at once —
        # `test_webui_console_review.py` reports "11 passed, 15 skipped" and
        # reads green while 100% of its execution coverage is gone (R2 lens
        # B/C). `test_node_is_available` below turns that into ONE loud
        # failure, which is why a skip is tolerable here.
        _pytest.skip("node not on PATH — JS EXECUTION pins are inert here")
    script = f"{source}\nprocess.stdout.write(JSON.stringify({expr}));\n"
    proc = subprocess.run([node, "--input-type=module", "-e", script],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"node failed: {proc.stderr.strip()[:2000]}")
    return json.loads(proc.stdout or "null")


def extract_js_function(source: str, name: str) -> str:
    """Source text of ``function <name>(…) { … }``, brace-matched.

    Deliberately naive about braces inside strings/regexes/comments — it is
    for extracting small helpers to run under :func:`eval_js`, and a
    mis-extraction fails loudly at parse time rather than passing weakly."""
    needle = f"function {name}("
    i = source.index(needle)
    # Include an `async` prefix: extracting only from `function` produced a
    # body full of `await` in a non-async function, i.e. a SyntaxError that
    # looks like a broken harness rather than a broken assertion.
    if source[max(0, i - 6):i] == "async ":
        i -= 6
    j = source.index("{", i)
    depth = 0
    for k in range(j, len(source)):
        if source[k] == "{":
            depth += 1
        elif source[k] == "}":
            depth -= 1
            if depth == 0:
                return source[i:k + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def extract_js_listener(source: str, event: str) -> str:
    """Body of ``window.addEventListener('<event>', (e) => { … })`` wrapped as
    ``function handler(e) { … }``, brace-matched.

    Inline listeners cannot be reached by :func:`extract_js_function`, and
    token assertions over them are defeated by the cheapest possible mutation
    (`if (false)` keeps every token). Extract and RUN them instead."""
    needle = f"window.addEventListener('{event}'"
    i = source.index(needle)
    j = source.index("{", source.index("=>", i))
    depth = 0
    for k in range(j, len(source)):
        if source[k] == "{":
            depth += 1
        elif source[k] == "}":
            depth -= 1
            if depth == 0:
                return "function handler(e) " + source[j:k + 1]
    raise AssertionError(f"unbalanced braces extracting the {event} listener")
