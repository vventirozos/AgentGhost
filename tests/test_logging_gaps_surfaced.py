"""Tier 2: previously-invisible failures now surface on the monitored stream.

Each test captures the module's pretty_log and asserts the new line fires.
(The many existing `logger.warning/error` sites are surfaced automatically by
the Tier-1 PrettyLogHandler and are covered in test_logging_framework.py.)
"""

import sys
import threading

import pytest
from unittest.mock import MagicMock


def _capture(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "pretty_log", lambda *a, **k: calls.append(a))
    return calls


def _titled(calls, title):
    return any(a and title in str(a[0]) for a in calls)


def test_profile_corrupt_logs(tmp_path, monkeypatch):
    from ghost_agent.memory import profile as pmod
    calls = _capture(monkeypatch, pmod)
    p = pmod.ProfileMemory(tmp_path)          # writes a valid default file
    p.file_path.write_text("{ broken json")   # now corrupt it
    out = p.load()
    assert out["root"]["name"] == "User"       # safe default returned
    assert _titled(calls, "Profile Corrupt")   # ...and the silent reset is surfaced


def test_context_compaction_logs_on_escalation(monkeypatch):
    from ghost_agent.core import context_manager as cm
    calls = _capture(monkeypatch, cm)
    mgr = cm.ContextManager(max_tokens=200)
    mgr.compress_if_needed([{"role": "user", "content": "word " * 500}])  # >> budget
    assert _titled(calls, "Context Compaction")


def test_context_compaction_silent_below_threshold(monkeypatch):
    from ghost_agent.core import context_manager as cm
    calls = _capture(monkeypatch, cm)
    mgr = cm.ContextManager(max_tokens=100000)
    mgr.compress_if_needed([{"role": "user", "content": "small"}])  # well under budget
    assert not _titled(calls, "Context Compaction")  # no spam when nothing compacts


@pytest.mark.asyncio
async def test_auth_reject_logs(monkeypatch):
    from ghost_agent.api import routes
    from fastapi import HTTPException
    calls = _capture(monkeypatch, routes)
    req = MagicMock()
    req.app.state.agent.context.args.api_key = "the-secret"
    req.url.path = "/v1/chat/completions"
    with pytest.raises(HTTPException):
        await routes.verify_api_key(req, api_key="wrong-key")
    assert _titled(calls, "Auth Rejected")


@pytest.mark.asyncio
async def test_postgres_error_logs(monkeypatch):
    from ghost_agent.tools import database as db
    calls = _capture(monkeypatch, db)
    fake = MagicMock()
    fake.connect.side_effect = Exception("connection refused")
    monkeypatch.setitem(sys.modules, "psycopg2", fake)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", MagicMock())
    monkeypatch.setitem(sys.modules, "tabulate", MagicMock())
    db._evict_connection("postgres://err-test")
    await db.tool_postgres_admin("query", "postgres://err-test", "SELECT 1")
    db._evict_connection("postgres://err-test")
    assert _titled(calls, "Postgres Error")


def test_sandbox_exec_failure_logs(monkeypatch):
    from ghost_agent.sandbox import docker as dk
    calls = _capture(monkeypatch, dk)
    sb = dk.DockerSandbox.__new__(dk.DockerSandbox)
    sb._lock = threading.Lock()
    sb.tor_proxy = None
    sb.container = MagicMock()
    sb.container.exec_run.side_effect = Exception("docker daemon gone")
    monkeypatch.setattr(sb, "ensure_running", lambda: None)
    monkeypatch.setattr(sb, "_is_container_ready", lambda: True)
    out, code = sb.execute("echo hi")
    assert code == 1
    assert _titled(calls, "Sandbox Exec Failed")
