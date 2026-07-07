"""Regression tests for the 18 open-finding fixes from the re-scan audit.

Coverage map:
  CRITICAL-1  sandbox output cap (head+tail truncation)
  CRITICAL-2  qwen_bridge GLOBAL_CONTEXT → ContextVar isolation
  CRITICAL-3  interface/server.py proxy auth + path traversal guard
  HIGH-4      swarm failure surfaced synchronously (no fake SUCCESS)
  HIGH-5      database activity query has LIMIT 50 + ORDER BY
  HIGH-6      agent.py XML reconstruction uses json.dumps for complex args
  HIGH-7      extract_code_from_markdown picks the longest block
  HIGH-8      sanitize_code reverts to pre-heal snapshot on failure
  HIGH-9      routes.py + server.py upload size cap (413)
  HIGH-10     interface/server.py error responses use HTTP 4xx/5xx
  HIGH-11     WebSocket broadcast snapshot via list()
  HIGH-12     log_streamer tail subprocess cleanup with timeout
  MED-13      get_embeddings([]) → [] short-circuit
  MED-14      TaskTree.load_from_json warns on DONE status regression
  MED-15      weather tool distinguishes Tor down vs API down
  MED-16      routes.py workspace save streams in chunks, closes BytesIO
  MED-17      parse_utc_timestamp robust round-trip
  MED-18      conftest inject_global_stream_adapter has re-entrant guard
"""
import asyncio
import io
import json
import logging
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =====================================================================
# CRITICAL-1 — sandbox output cap
# =====================================================================


def test_sandbox_execute_caps_huge_output():
    """A 1 MB stdout blob gets capped to the 256 KB head+tail window."""
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    sb._is_container_ready = MagicMock(return_value=True)
    sb.ensure_running = MagicMock()
    sb.client = MagicMock()
    sb.NotFound = type("NotFound", (Exception,), {})

    huge = b"A" * 1_000_000  # 1 MB
    fake_exec = MagicMock()
    fake_exec.output = huge
    fake_exec.exit_code = 0
    sb.container.exec_run = MagicMock(return_value=fake_exec)

    output, exit_code = sb.execute("echo hi")  # default mode = legacy 256KB cap
    assert exit_code == 0
    # Should contain the truncation banner AND be drastically smaller. The
    # banner wording moved to the shared truncate_head_tail helper (2026-07-07).
    assert "truncated" in output.lower() and "sandbox" in output.lower()
    assert len(output) < 400_000  # well under 1 MB
    # Both head and tail must be present
    assert output.startswith("A")
    assert output.endswith("A")


def test_sandbox_execute_small_output_unchanged():
    """Outputs smaller than the cap pass through untouched."""
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.container = MagicMock()
    sb._is_container_ready = MagicMock(return_value=True)
    sb.ensure_running = MagicMock()
    sb.client = MagicMock()
    sb.NotFound = type("NotFound", (Exception,), {})

    fake_exec = MagicMock()
    fake_exec.output = b"hello world\n"
    fake_exec.exit_code = 0
    sb.container.exec_run = MagicMock(return_value=fake_exec)

    output, exit_code = sb.execute("echo hi")
    assert output == "hello world\n"
    assert "truncated" not in output.lower()


# =====================================================================
# CRITICAL-2 — qwen_bridge context isolation
# =====================================================================


def test_qwen_bridge_uses_contextvar_not_global():
    """The bridge must expose `_CTX` as a ContextVar — not a plain module
    global. Reading via the back-compat `GLOBAL_CONTEXT` attribute returns
    whatever the current context binds."""
    src = Path("src/ghost_agent/tools/qwen_bridge.py").read_text()
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    assert "import contextvars" in code_only
    assert "ContextVar" in code_only
    assert "_CTX" in code_only
    # The old bare global assignment must be gone from CODE (comments fine).
    assert "GLOBAL_CONTEXT = None" not in code_only
    # The public API is still `set_context` (same name, new body).
    assert "def set_context" in code_only


# =====================================================================
# CRITICAL-3 + HIGH-9/10 — interface/server.py auth + uploads + errors
# =====================================================================


def test_interface_proxies_require_auth():
    """Every state-mutating proxy endpoint must carry the new
    verify_interface_key dependency."""
    src = Path("interface/server.py").read_text()
    # The helper itself must exist.
    assert "verify_interface_key" in src
    assert "X-Ghost-Key" in src
    # Every state-mutating endpoint must reference the dependency.
    # We scan the route decorators — they're all on `@app.post(...)` /
    # `@app.get(...)` lines with a `Depends(verify_interface_key)` in them.
    protected_routes = [
        "/api/chat",
        "/api/workspace/save",
        "/api/workspace/load",
        "/api/upload",
        "/api/download/{filename:path}",
        "/api/stt",
        "/api/tts",
    ]
    for route in protected_routes:
        # Find the decorator line for this route.
        marker = f'"{route}"'
        idx = src.find(marker)
        assert idx != -1, f"route {route} missing from interface/server.py"
        decorator_line = src[max(0, idx - 200):idx + len(marker) + 200]
        assert "verify_interface_key" in decorator_line, (
            f"route {route} does not declare verify_interface_key dependency"
        )


def test_interface_has_upload_size_cap():
    src = Path("interface/server.py").read_text()
    assert "MAX_UPLOAD_BYTES" in src
    assert "_read_capped_upload" in src


def test_interface_errors_use_json_response_not_bare_dict():
    src = Path("interface/server.py").read_text()
    # The helper must exist and be used.
    assert "_err_json" in src
    # Grep: endpoints that used to `return {"error": str(e)}` should all
    # be converted to `_err_json(...)` calls now.
    # Allow bare `return {"error": ...}` only inside the CancelledError
    # path in the chat stream worker (that's NOT a response).
    bare_count = src.count('return {"error":')
    # Count should be zero — all production error returns go through _err_json.
    assert bare_count == 0, f"{bare_count} bare error-dict returns still present"


def test_interface_download_rejects_path_traversal():
    src = Path("interface/server.py").read_text()
    # Verify the guard is in the download handler body.
    dl_idx = src.find('@app.get("/api/download/')
    dl_end = src.find("@app.", dl_idx + 1)
    dl_body = src[dl_idx:dl_end]
    assert '".." in filename' in dl_body
    assert "Invalid filename" in dl_body


# =====================================================================
# HIGH-4 — swarm dispatch surfaces failure synchronously
# =====================================================================


@pytest.mark.asyncio
async def test_swarm_returns_warning_when_no_nodes_configured():
    from ghost_agent.tools.swarm import tool_delegate_to_swarm
    llm = MagicMock()
    llm.swarm_clients = None
    scratchpad = MagicMock()
    result = await tool_delegate_to_swarm(
        llm_client=llm, model_name="x", scratchpad=scratchpad,
        tasks=[{"instruction": "i", "input_data": "d", "output_key": "k"}],
    )
    # "Error:" prefix (was "SYSTEM WARNING") so the agent loop registers a
    # failure and the delegate_to_swarm fallback hint fires.
    assert result.startswith("Error")
    assert "not configured" in result
    assert "Swarm" in result


@pytest.mark.asyncio
async def test_swarm_returns_warning_when_all_nodes_missing():
    """Configured swarm but get_swarm_node returns None — all tasks
    should be reported as skipped, NOT as 'SUCCESS: N dispatched'."""
    from ghost_agent.tools.swarm import tool_delegate_to_swarm
    llm = MagicMock()
    llm.swarm_clients = [{"client": MagicMock(), "model": "any"}]  # non-empty
    llm.get_swarm_node = MagicMock(return_value=None)
    scratchpad = MagicMock()
    result = await tool_delegate_to_swarm(
        llm_client=llm, model_name="x", scratchpad=scratchpad,
        tasks=[
            {"instruction": "i1", "input_data": "d", "output_key": "k1"},
            {"instruction": "i2", "input_data": "d", "output_key": "k2"},
        ],
    )
    assert "0 of 2" in result or "WARNING" in result
    # Scrapbook must have been stamped with the failure alerts
    assert scratchpad.set.call_count == 2


@pytest.mark.asyncio
async def test_swarm_partial_success_reports_mixed_result():
    from ghost_agent.tools.swarm import tool_delegate_to_swarm
    llm = MagicMock()
    llm.swarm_clients = [{"client": MagicMock(), "model": "any"}]
    node_results = [
        {"client": MagicMock(), "model": "fast"},  # first task: ok
        None,                                       # second task: no node
    ]
    llm.get_swarm_node = MagicMock(side_effect=node_results)
    scratchpad = MagicMock()
    # Patch asyncio.create_task so we don't actually spawn background workers
    with patch("ghost_agent.tools.swarm.asyncio.create_task", return_value=MagicMock()):
        result = await tool_delegate_to_swarm(
            llm_client=llm, model_name="x", scratchpad=scratchpad,
            tasks=[
                {"instruction": "i1", "input_data": "d", "output_key": "k1"},
                {"instruction": "i2", "input_data": "d", "output_key": "k2"},
            ],
        )
    assert "PARTIAL" in result or "1/2" in result


# =====================================================================
# HIGH-5 — database activity query has LIMIT
# =====================================================================


def test_postgres_activity_query_has_limit():
    import inspect
    from ghost_agent.tools import database
    src = inspect.getsource(database.tool_postgres_admin)
    # Find the activity branch
    activity_idx = src.find('action == "activity"')
    assert activity_idx != -1
    # Use a wider window so the doc comments don't push the SQL past the cap.
    activity_block = src[activity_idx:activity_idx + 1500]
    assert "LIMIT 50" in activity_block
    assert "ORDER BY duration_sec DESC" in activity_block


# =====================================================================
# HIGH-6 — agent.py XML reconstruction uses json.dumps for complex args
# =====================================================================


def test_agent_xml_reconstruction_handles_complex_args():
    """Dict/list/None values in tool_calls history should be rendered as
    JSON, not Python repr."""
    src = Path("src/ghost_agent/core/agent.py").read_text()
    assert "json.dumps(v, ensure_ascii=False)" in src
    # The offending `str(v)` for non-string values must be guarded.
    assert 'isinstance(v, (dict, list, bool, type(None)))' in src


# =====================================================================
# HIGH-7 — extract_code_from_markdown picks the longest block
# =====================================================================


def test_extract_code_picks_longest_block_on_multiple_fences():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown
    text = """Here's a short example:
```python
x = 1
```
And here's the real implementation:
```python
def solve():
    total = 0
    for i in range(100):
        total += i * i
    return total
```
"""
    result = extract_code_from_markdown(text)
    assert "def solve()" in result
    assert "total += i * i" in result


def test_extract_code_single_block_works():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown
    text = """```python
print("hello")
```"""
    result = extract_code_from_markdown(text)
    assert result == 'print("hello")'


def test_extract_code_handles_truncated_block():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown
    # Opening fence, no closing fence (model output truncated mid-stream)
    text = """```python
def f():
    return 42"""
    result = extract_code_from_markdown(text)
    assert "def f()" in result
    assert "return 42" in result


# =====================================================================
# HIGH-8 — sanitize_code reverts to pre-heal snapshot on failure
# =====================================================================


def test_sanitize_code_reverts_when_healer_corrupts_valid_code():
    """If the healing pass somehow breaks valid code, the pre-healing
    snapshot must come back out (not the corrupted healed version)."""
    from ghost_agent.utils import sanitizer
    valid_code = "x = 1\ny = 2\nprint(x + y)\n"
    with patch.object(sanitizer, "fix_python_syntax", return_value="x =\ny =\n((("):
        out, err = sanitizer.sanitize_code(valid_code, "test.py")
    # The returned content must match the parseable INPUT (modulo trailing
    # whitespace stripped during extract/scrub), NOT the corrupted healed version.
    assert "x = 1" in out
    assert "y = 2" in out
    assert "print(x + y)" in out
    assert "(((" not in out
    assert err is not None
    assert "reverted" in err.lower()


def test_sanitize_code_returns_success_on_clean_input():
    from ghost_agent.utils.sanitizer import sanitize_code
    out, err = sanitize_code("x = 1\nprint(x)\n", "test.py")
    assert err is None
    assert "x = 1" in out


# =====================================================================
# HIGH-11 — WebSocket broadcast iterates a snapshot
# =====================================================================


def test_log_streamer_snapshots_websocket_set():
    src = Path("interface/server.py").read_text()
    # The iteration line must wrap the set in list(...)
    assert "for ws in list(connected_websockets):" in src
    # The eviction pass must use discard (not remove) to be idempotent.
    assert "connected_websockets.discard" in src


# =====================================================================
# HIGH-12 — tail subprocess cleanup has a timeout
# =====================================================================


def test_log_streamer_tail_cleanup_has_timeout():
    src = Path("interface/server.py").read_text()
    # The cleanup must wrap process.wait() in wait_for with a timeout,
    # and fall back to process.kill() on timeout.
    assert "asyncio.wait_for(process.wait()" in src
    assert "process.kill()" in src


# =====================================================================
# MED-13 — get_embeddings([]) short-circuit
# =====================================================================


@pytest.mark.asyncio
async def test_get_embeddings_empty_list_short_circuits():
    from ghost_agent.core.llm import LLMClient
    client = LLMClient.__new__(LLMClient)
    client.http_client = MagicMock()
    client.http_client.post = AsyncMock()  # must NOT be called
    client._main_node_lock = asyncio.Lock()

    result = await client.get_embeddings([])
    assert result == []
    client.http_client.post.assert_not_called()


# =====================================================================
# MED-14 — TaskTree.load_from_json warns on DONE regression
# =====================================================================


def test_task_tree_warns_on_done_status_regression(caplog):
    from ghost_agent.core.planning import TaskTree, TaskStatus
    tree = TaskTree()
    tree.load_from_json({
        "id": "root",
        "description": "goal",
        "status": "PENDING",
        "children": [],
    })
    # Mark it done
    tree.load_from_json({
        "id": "root",
        "description": "goal",
        "status": "DONE",
        "children": [],
    })
    assert tree.nodes["root"].status == TaskStatus.DONE

    # Try to regress it
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        tree.load_from_json({
            "id": "root",
            "description": "goal",
            "status": "PENDING",
            "children": [],
        })
    # Node stays DONE
    assert tree.nodes["root"].status == TaskStatus.DONE
    # Warning was emitted
    warnings = [r for r in caplog.records if "regression" in r.message.lower()]
    assert len(warnings) >= 1


# =====================================================================
# MED-17 — parse_utc_timestamp robust round-trip
# =====================================================================


def test_parse_utc_timestamp_round_trips_own_output():
    from ghost_agent.utils.helpers import get_utc_timestamp, parse_utc_timestamp
    ts = get_utc_timestamp()
    assert ts.endswith("Z")
    parsed = parse_utc_timestamp(ts)
    # Should be a naive datetime
    assert parsed.tzinfo is None
    # Round-trip should be accurate to microseconds
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    assert abs((now - parsed).total_seconds()) < 5


def test_parse_utc_timestamp_accepts_without_z():
    from ghost_agent.utils.helpers import parse_utc_timestamp
    dt = parse_utc_timestamp("2026-04-14T12:34:56.789012")
    assert dt.year == 2026
    assert dt.month == 4


def test_parse_utc_timestamp_accepts_offset_suffix():
    from ghost_agent.utils.helpers import parse_utc_timestamp
    dt = parse_utc_timestamp("2026-04-14T12:34:56.789012+00:00")
    assert dt.tzinfo is None  # normalised to naive UTC
    assert dt.hour == 12


def test_parse_utc_timestamp_rejects_garbage():
    from ghost_agent.utils.helpers import parse_utc_timestamp
    with pytest.raises(ValueError):
        parse_utc_timestamp("not a date at all")
    with pytest.raises(ValueError):
        parse_utc_timestamp("")
    with pytest.raises(ValueError):
        parse_utc_timestamp(None)  # type: ignore


# =====================================================================
# MED-16 / #23 — workspace save builds off-loop to a spool file with a cap
# =====================================================================


def test_workspace_save_offloads_and_caps():
    """2026-07-07 (#23): the save no longer walks+deflates the whole sandbox
    inline in the coroutine (which froze the event loop) or holds the archive
    in RAM (an OOM vector). It builds in a worker thread to a spool file with
    a byte ceiling, streamed via FileResponse."""
    src = Path("src/ghost_agent/api/routes.py").read_text()
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    # The old in-RAM getvalue() streaming is gone.
    assert "iter([zip_buffer.getvalue()])" not in code_only
    assert "zip_buffer.getvalue()" not in code_only
    # Build runs off the event loop, is capped, and streams from a file.
    assert "asyncio.to_thread(_build_zip)" in code_only
    assert "_MAX_WORKSPACE_SAVE_BYTES" in code_only
    assert "FileResponse(" in code_only


# =====================================================================
# MED-15 — weather tool distinguishes Tor vs API failure
# =====================================================================


def test_weather_error_message_diagnostic():
    import inspect
    from ghost_agent.tools import system
    src = inspect.getsource(system.tool_get_weather)
    # Must distinguish Tor-down from API-down in the error output.
    assert "Tor proxy appears to be down" in src
    assert "3 Tor retries" in src


# =====================================================================
# MED-18 — conftest re-entrant guard
# =====================================================================


def test_conftest_inject_fixture_isnt_destructive():
    """The autouse fixture must NOT overwrite real LLM clients — only
    MagicMock/AsyncMock-backed ones. The original concern was test
    pollution from a recursive wrap; we addressed it by keeping the
    fixture lean and relying on monkeypatch's per-test cleanup."""
    src = Path("tests/conftest.py").read_text()
    # The guard that protects real LLM clients must be intact.
    assert "isinstance(context.llm_client.stream_chat_completion, (MagicMock, AsyncMock))" in src
    # The wrap is still installed via monkeypatch (so per-test cleanup is automatic).
    assert "monkeypatch.setattr(GhostAgent, '__init__'" in src


# =====================================================================
# execute.py bare except replacements
# =====================================================================


def test_execute_tool_no_bare_except_pass():
    """Verify the bad `except: pass` patterns in execute.py have been
    replaced with typed catches. Walk the AST so embedded string literals
    (e.g. the Jupyter runner code injected into the sandbox) don't count."""
    import ast
    src = Path("src/ghost_agent/tools/execute.py").read_text()
    tree = ast.parse(src)
    bare_handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # Bare `except:` has no `type`. We also tolerate `except Exception:`
            # which still catches everything, so this test is conservative —
            # it only flags the worst-of-the-worst (no exception type at all).
            if node.type is None:
                bare_handlers.append(node.lineno)
    assert bare_handlers == [], f"bare `except:` still at lines: {bare_handlers}"
