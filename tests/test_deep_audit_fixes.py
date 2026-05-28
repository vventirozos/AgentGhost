"""Regression tests for the deep-audit bug-fix sweep.

Each section corresponds to a concrete bug that was identified during the
end-to-end audit and has now been fixed. The tests are written to fail in a
targeted way if someone reverts the fix, not to exhaustively exercise the
subsystem. Keep them cheap — no Docker, no ChromaDB, no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --------------------------------------------------------------------------- #
# tool_failure — HTTP status regex word boundaries
# --------------------------------------------------------------------------- #


def test_tool_failure_retryable_http_statuses_require_word_boundaries():
    """Previously `r"503|502|504"` matched substrings like `5021` in an
    unrelated error code. After the fix we use `\\b(?:502|503|504)\\b`."""
    from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass

    cls, _ = classify_tool_failure("HTTP 503 Service Unavailable")
    assert cls == FailureClass.RETRYABLE

    cls, _ = classify_tool_failure("error code 5021 reported")
    assert cls != FailureClass.RETRYABLE

    cls, _ = classify_tool_failure("connector reset on 50234 port")
    assert cls != FailureClass.RETRYABLE


def test_tool_failure_fatal_vs_diagnostic_precedence():
    from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass

    cls, _ = classify_tool_failure("Permission denied writing /etc/passwd")
    assert cls == FailureClass.FATAL

    cls, _ = classify_tool_failure("Traceback (most recent call last):\n  ValueError: x")
    assert cls == FailureClass.DIAGNOSTIC


def test_tool_failure_empty_error_is_unknown():
    from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass

    cls, reason = classify_tool_failure("")
    assert cls == FailureClass.UNKNOWN
    assert "empty" in reason


# --------------------------------------------------------------------------- #
# journal — push_front truncation keeps newest entries
# --------------------------------------------------------------------------- #


def test_journal_push_front_truncation_preserves_requeued_items():
    """push_front is called to preserve unprocessed items across a
    consolidation interruption, so when truncating we must drop the
    newly appended tail, NOT the re-queued head we were trying to save."""
    from ghost_agent.memory.journal import MemoryJournal

    with tempfile.TemporaryDirectory() as td:
        j = MemoryJournal(Path(td), max_capacity=4)
        # Existing journal has items 1..3 at the tail
        j.append("note", {"n": 1})
        j.append("note", {"n": 2})
        j.append("note", {"n": 3})

        # Re-queue items -3, -2, -1 (ordered oldest→newest) at the head.
        # Total would be 6 — the cap is 4, so we drop 2 from somewhere.
        old = [
            {"type": "note", "data": {"n": -3}},
            {"type": "note", "data": {"n": -2}},
            {"type": "note", "data": {"n": -1}},
        ]
        j.push_front(old)

        data = j.load()
        assert len(data) == 4
        # Re-queued items must all survive: -3, -2, -1. The tail (most
        # recent append) is the one that gets dropped — 2 and 3 go.
        ns = [d["data"]["n"] for d in data]
        assert ns == [-3, -2, -1, 1]


# --------------------------------------------------------------------------- #
# file_system — streaming replace skips multi-line searches
# --------------------------------------------------------------------------- #


async def test_streaming_replace_skips_multiline_search():
    """A multi-line old_text can never match line-by-line. Ensure the
    replace tool falls back to the full-file path instead of silently
    returning 0 replacements."""
    from ghost_agent.tools.file_system import tool_replace_text

    with tempfile.TemporaryDirectory() as td:
        sandbox_dir = Path(td)
        target = sandbox_dir / "big.txt"
        # >1 MB so the streaming threshold triggers
        filler = ("padding line\n" * 100_000)
        content = filler + "BLOCK_START\nfoo\nBLOCK_END\n" + filler
        target.write_text(content)
        assert target.stat().st_size > 1_000_000

        result = await tool_replace_text(
            "big.txt",
            "BLOCK_START\nfoo\nBLOCK_END",
            "BLOCK_START\nbar\nBLOCK_END",
            sandbox_dir,
        )
        assert "SUCCESS" in result, result
        new_text = target.read_text()
        assert "bar" in new_text
        assert "BLOCK_START\nfoo\nBLOCK_END" not in new_text


# --------------------------------------------------------------------------- #
# file_system — new copy operation
# --------------------------------------------------------------------------- #


async def test_file_copy_operation_copies_and_preserves_source():
    from ghost_agent.tools.file_system import tool_file_system

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td)
        (sandbox / "src.txt").write_text("hello")
        result = await tool_file_system(
            operation="copy",
            sandbox_dir=sandbox,
            path="src.txt",
            destination="dest.txt",
        )
        assert "SUCCESS" in result
        assert (sandbox / "src.txt").exists()
        assert (sandbox / "dest.txt").read_text() == "hello"


async def test_file_copy_refuses_overwrite():
    from ghost_agent.tools.file_system import tool_file_system

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td)
        (sandbox / "src.txt").write_text("a")
        (sandbox / "dest.txt").write_text("b")
        result = await tool_file_system(
            operation="copy",
            sandbox_dir=sandbox,
            path="src.txt",
            destination="dest.txt",
        )
        assert "already exists" in result
        # Destination remains untouched
        assert (sandbox / "dest.txt").read_text() == "b"


async def test_file_copy_requires_destination():
    from ghost_agent.tools.file_system import tool_file_system

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td)
        (sandbox / "src.txt").write_text("a")
        result = await tool_file_system(
            operation="copy",
            sandbox_dir=sandbox,
            path="src.txt",
        )
        assert "destination" in result.lower()


# --------------------------------------------------------------------------- #
# qwen_bridge — safe sync-from-async coroutine runner
# --------------------------------------------------------------------------- #


def _load_qwen_bridge_module():
    """Import ``ghost_agent.tools.qwen_bridge`` while stubbing out the
    qwen-agent dependency tree (which pulls optional libs like
    ``soundfile`` that aren't part of the agent's own test requirements).

    We don't need the BaseTool machinery for these tests — only the
    ``_run_coro_blocking`` helper — so we inject minimal stand-ins.
    """
    import sys
    import types

    qa = types.ModuleType("qwen_agent")
    qa_tools = types.ModuleType("qwen_agent.tools")
    qa_tools_base = types.ModuleType("qwen_agent.tools.base")

    class _BaseTool:  # pragma: no cover - trivial stub
        description = ""
        parameters: list = []

    def _register_tool(_name):
        def _wrap(cls):
            return cls
        return _wrap

    qa_tools_base.BaseTool = _BaseTool
    qa_tools_base.register_tool = _register_tool
    sys.modules.setdefault("qwen_agent", qa)
    sys.modules.setdefault("qwen_agent.tools", qa_tools)
    sys.modules["qwen_agent.tools.base"] = qa_tools_base

    # Fresh import so the stubs take effect.
    if "ghost_agent.tools.qwen_bridge" in sys.modules:
        del sys.modules["ghost_agent.tools.qwen_bridge"]
    import importlib
    return importlib.import_module("ghost_agent.tools.qwen_bridge")


async def test_qwen_bridge_run_coro_from_running_loop():
    """`_run_coro_blocking` must succeed when called from inside an active
    event loop — the previous direct `asyncio.run` call crashed with
    `RuntimeError: asyncio.run() cannot be called from a running event loop`
    whenever a qwen-agent tool was dispatched inside a FastAPI request."""
    qwen_bridge = _load_qwen_bridge_module()

    async def _hello():
        await asyncio.sleep(0)
        return "world"

    # We're inside a running asyncio test; the helper must still return.
    # It off-loads to a worker thread with its own loop.
    def _blocking_call():
        return qwen_bridge._run_coro_blocking(_hello())

    result = await asyncio.to_thread(_blocking_call)
    assert result == "world"


def test_qwen_bridge_run_coro_outside_loop():
    qwen_bridge = _load_qwen_bridge_module()

    async def _ten():
        return 10

    assert qwen_bridge._run_coro_blocking(_ten()) == 10


# --------------------------------------------------------------------------- #
# execute — Python stubbornness guard keeps indentation-sensitive compare
# --------------------------------------------------------------------------- #


async def test_execute_python_stubbornness_guard_is_indent_sensitive():
    """For .py files the guard must NOT treat two scripts that differ only
    in indentation as identical — indentation is semantic in Python."""
    from ghost_agent.tools import execute as execute_module

    with tempfile.TemporaryDirectory() as td:
        sandbox_dir = Path(td)

        # Stub out the actual sandbox execution — we only care about the
        # stubbornness decision path (is_new_code → write-or-skip).
        mock_mgr = MagicMock()
        mock_mgr.execute = MagicMock(return_value=("", 0))

        original_code = (
            "def foo():\n"
            "    if True:\n"
            "        return 1\n"
        )
        # Same tokens, different indentation → semantically different.
        reformatted_code = (
            "def foo():\n"
            " if True:\n"
            "  return 1\n"
        )
        target = sandbox_dir / "script.py"
        target.write_text(original_code)

        # Submit the reformatted version. The guard must treat it as new
        # code and overwrite — otherwise Python semantics change silently.
        await execute_module.tool_execute(
            filename="script.py",
            content=reformatted_code,
            sandbox_dir=sandbox_dir,
            sandbox_manager=mock_mgr,
        )
        written = target.read_text()
        # The file must have been rewritten with the new indentation —
        # we don't assert exact equality because the write pipeline may
        # normalise trailing newlines, but the 4-space indent from the
        # original must be gone and the 1-space/2-space indent from the
        # reformatted version must be present.
        assert " if True:" in written
        assert "  return 1" in written
        assert "    if True:" not in written
        assert "    return 1" not in written


# --------------------------------------------------------------------------- #
# API routes — bare-except narrowing + non-streaming error handling
# --------------------------------------------------------------------------- #


async def test_api_generate_rejects_bad_json_with_400():
    """/api/generate now returns 400 with a specific JSON error instead
    of swallowing *all* exceptions behind a bare `except`."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Lazy import — the module has heavy deps.
    from ghost_agent.api import routes as routes_module

    app = FastAPI()
    # Fake the agent + API key so the endpoint can authenticate.
    app.state.agent = MagicMock()
    app.state.agent.context.args.api_key = None  # disable auth for the test
    app.state.args = MagicMock()
    app.state.args.model = "test-model"

    app.include_router(routes_module.router)

    client = TestClient(app)
    resp = client.post(
        "/api/generate",
        data=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert "Invalid JSON" in body["error"]


async def test_api_chat_non_streaming_returns_json_error_on_exception():
    """Previously an exception in `handle_chat` raised a raw 500 with an
    HTML body. The fix returns an OpenAI-shaped JSON error."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api import routes as routes_module

    app = FastAPI()
    fake_agent = MagicMock()
    fake_agent.context.args.api_key = None
    fake_agent.context.args.model = "test-model"
    fake_agent.handle_chat = AsyncMock(side_effect=RuntimeError("boom"))
    app.state.agent = fake_agent
    app.state.args = MagicMock()
    app.state.args.model = "test-model"
    app.include_router(routes_module.router)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 500
    body = resp.json()
    # Error envelope is intentionally generic — the exception type
    # name and its repr must NOT leak to the wire (info-disclosure
    # hardening). The error_id lets operators correlate to the log
    # entry where the full stack lives.
    assert body.get("error", {}).get("type") == "InternalError"
    assert "RuntimeError" not in body["error"]["message"]
    assert "boom" not in body["error"]["message"]
    assert "error_id=" in body["error"]["message"]


# --------------------------------------------------------------------------- #
# DockerSandbox.close() — safe no-op when container missing
# --------------------------------------------------------------------------- #


def test_docker_sandbox_close_handles_missing_container_gracefully():
    """`close()` must never raise, even when no container has ever been
    provisioned. We don't actually need Docker running for this test —
    we construct a minimal stub instance."""
    import types
    from ghost_agent.sandbox import docker as docker_mod

    # Build a naked instance without invoking __init__ (which requires docker).
    sandbox = docker_mod.DockerSandbox.__new__(docker_mod.DockerSandbox)
    sandbox.container = None
    sandbox.container_name = "ghost-agent-sandbox-xxxxxxxx"

    # Fake NotFound / APIError classes and a client whose .get raises NotFound.
    class _NotFound(Exception):
        pass

    class _APIError(Exception):
        pass

    sandbox.NotFound = _NotFound
    sandbox.APIError = _APIError
    client = MagicMock()
    client.containers.get.side_effect = _NotFound("no such container")
    sandbox.client = client

    # Must not raise.
    sandbox.close(remove=False)


# --------------------------------------------------------------------------- #
# --no-memory gating — all four stores must be skipped
# --------------------------------------------------------------------------- #


def test_no_memory_flag_gates_all_memory_subsystems():
    """Scan ``main.py`` to ensure the ``--no-memory`` guard surrounds
    profile, graph, AND vector initialization — the Tier-1 bug was that
    only VectorMemory was gated."""
    main_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "main.py"
    )
    src = main_path.read_text()
    # Locate the `if not args.no_memory:` block and everything up to the
    # matching `else:` / dedent. Blank lines inside the block are part
    # of the body, so the regex must allow them.
    lines = src.splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*if not args\.no_memory:\s*$", ln):
            start = i
            indent = len(ln) - len(ln.lstrip())
            break
    assert start is not None, "expected `if not args.no_memory:` in main.py"

    # Collect the block body — subsequent lines more-indented than the
    # `if` itself (blank lines count as part of the body).
    body_lines = []
    for ln in lines[start + 1:]:
        if not ln.strip():
            body_lines.append(ln)
            continue
        cur_indent = len(ln) - len(ln.lstrip())
        if cur_indent <= indent:
            break
        body_lines.append(ln)
    body = "".join(body_lines)

    # All three stores must live INSIDE the gate.
    assert "ProfileMemory(" in body
    assert "GraphMemory(" in body
    assert "VectorMemory(" in body
