"""Sandbox egress guard (2026-07-05, request C5 post-mortem).

In-sandbox probes of the agent's own ports are intercepted BEFORE
execution. The sandbox container has its own loopback, so 127.0.0.1:8000
in there is not the agent's API — but three separate incidents (07-02,
07-04, 07-05) showed the model curl-testing it anyway, reading the
connection failure as "the server is down on the user's machine", and
writing a forbidden mock server. Prompt-side warnings did not stop it;
the guard answers the probe with ground truth instead of letting the
misleading failure happen.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock

import pytest

from ghost_agent.tools.execute import tool_execute


@pytest.mark.asyncio
async def test_curl_probe_of_agent_api_port_blocked(tmp_path):
    sm = MagicMock()
    res = await tool_execute(
        command="curl -s -o /dev/null http://127.0.0.1:8000/",
        sandbox_dir=tmp_path, sandbox_manager=sm)
    assert "SANDBOX EGRESS BLOCKED" in res
    assert not sm.execute.called


@pytest.mark.asyncio
async def test_localhost_and_llm_port_variants_blocked(tmp_path):
    sm = MagicMock()
    sm.execute = MagicMock(return_value=("", 0))
    for cmd in ("wget http://localhost:8000/api/game/move",
                "curl http://127.0.0.1:8088/v1/models",
                "nc -z 0.0.0.0 8000 && echo up"):
        # "nc -z 0.0.0.0 8000" has no colon — only the colon forms are
        # guarded; the nc form goes through (documented narrowness).
        res = await tool_execute(command=cmd, sandbox_dir=tmp_path,
                                 sandbox_manager=sm)
        if ":8000" in cmd or ":8088" in cmd:
            assert "SANDBOX EGRESS BLOCKED" in res


@pytest.mark.asyncio
async def test_inline_python_probe_blocked(tmp_path):
    sm = MagicMock()
    res = await tool_execute(
        "probe.py",
        "import urllib.request\n"
        "urllib.request.urlopen('http://127.0.0.1:8000/api/game/move')\n",
        tmp_path, sm)
    assert "SANDBOX EGRESS BLOCKED" in res
    assert not sm.execute.called
    assert not (tmp_path / "probe.py").exists()   # nothing written either


@pytest.mark.asyncio
async def test_message_teaches_the_alternatives(tmp_path):
    res = await tool_execute(
        command="curl http://127.0.0.1:8000/", sandbox_dir=tmp_path,
        sandbox_manager=MagicMock())
    assert "browser" in res                    # host-side verification path
    assert "mock" in res.lower()               # names the forbidden move
    assert "8081" in res                       # in-sandbox alternative


@pytest.mark.asyncio
async def test_other_ports_still_execute(tmp_path):
    sm = MagicMock()
    sm.execute = MagicMock(return_value=("ok", 0))
    res = await tool_execute(
        command="curl -s http://127.0.0.1:8081/health",
        sandbox_dir=tmp_path, sandbox_manager=sm)
    assert "SANDBOX EGRESS BLOCKED" not in res
    assert sm.execute.called


@pytest.mark.asyncio
async def test_unrelated_commands_unaffected(tmp_path):
    sm = MagicMock()
    sm.execute = MagicMock(return_value=("hello", 0))
    res = await tool_execute(command="echo hello", sandbox_dir=tmp_path,
                             sandbox_manager=sm)
    assert "SANDBOX EGRESS BLOCKED" not in res
    assert sm.execute.called
