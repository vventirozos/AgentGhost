"""Upstream streaming idle-timeout: the time-to-FIRST-byte (prompt prefill on
a large context / slow node) gets a generous budget, while the inter-token gap
stays tight to catch a genuine mid-stream hang. Regression for the false
"stall → forced retry" that fired when a turn's prefill exceeded the old flat
30s with zero bytes emitted."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import llm as llm_mod
from ghost_agent.core.llm import LLMClient


class _FakeResp:
    status_code = 200

    def __init__(self, lines_with_delays):
        self._items = lines_with_delays

    def raise_for_status(self):
        return None

    async def aread(self):
        return b""

    async def aclose(self):
        return None

    def aiter_lines(self):
        items = self._items

        async def _gen():
            for delay, line in items:
                await asyncio.sleep(delay)
                yield line
        return _gen()


class _FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def build_request(self, *a, **k):
        return MagicMock()

    async def send(self, req, stream=True):
        return self._resp


def _client(lines_with_delays):
    c = LLMClient.__new__(LLMClient)
    c.http_client = _FakeClient(_FakeResp(lines_with_delays))
    c.coding_clients = None
    return c


async def _collect(client):
    chunks = []
    async for b in client._do_stream_chat_completion({"model": "m", "messages": []}):
        chunks.append(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b)
    return "".join(chunks)


@pytest.mark.asyncio
async def test_slow_first_byte_within_budget_is_not_aborted(monkeypatch):
    """Prefill slower than the inter-token gap, but within the first-byte
    budget → stream proceeds (no false stall)."""
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 0.4)
    monkeypatch.setattr(llm_mod, "_STREAM_IDLE_TIMEOUT", 0.1)
    out = await _collect(_client([
        (0.25, "data: hello"),     # first byte: 0.25 < 0.4 → OK (would've tripped a 0.1 gap)
        (0.05, "data: world"),
        (0.0, "data: [DONE]"),
    ]))
    assert "hello" in out and "world" in out
    assert "stalled" not in out.lower()


@pytest.mark.asyncio
async def test_first_byte_over_budget_aborts_as_prefill(monkeypatch):
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 0.2)
    monkeypatch.setattr(llm_mod, "_STREAM_IDLE_TIMEOUT", 0.1)
    out = await _collect(_client([
        (0.5, "data: hello"),      # first byte too slow → abort
        (0.0, "data: [DONE]"),
    ]))
    assert "stalled" in out.lower()
    assert "prefill" in out.lower()
    assert "hello" not in out


@pytest.mark.asyncio
async def test_midstream_gap_over_idle_aborts(monkeypatch):
    monkeypatch.setattr(llm_mod, "_STREAM_FIRST_BYTE_TIMEOUT", 0.4)
    monkeypatch.setattr(llm_mod, "_STREAM_IDLE_TIMEOUT", 0.1)
    out = await _collect(_client([
        (0.05, "data: hello"),     # first byte fast
        (0.3, "data: world"),      # mid-stream gap 0.3 > 0.1 idle → abort
        (0.0, "data: [DONE]"),
    ]))
    assert "hello" in out          # first token made it through
    assert "stalled" in out.lower()
    assert "mid-stream" in out.lower()


def test_default_timeouts_are_generous_for_prefill():
    # First-byte default must comfortably exceed the inter-token default and
    # the old flat 30s, so a large-context prefill isn't a false stall.
    assert llm_mod._STREAM_FIRST_BYTE_TIMEOUT >= 120
    assert llm_mod._STREAM_IDLE_TIMEOUT >= 30
    assert llm_mod._STREAM_FIRST_BYTE_TIMEOUT > llm_mod._STREAM_IDLE_TIMEOUT


def test_timeouts_are_env_tunable():
    # Process-isolated (a reload in-process would rebind LLMClient and break
    # isinstance checks elsewhere in the suite).
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONPATH": "src",
           "GHOST_STREAM_FIRST_BYTE_TIMEOUT": "240",
           "GHOST_STREAM_IDLE_TIMEOUT": "45"}
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("from ghost_agent.core import llm; "
            "print(llm._STREAM_FIRST_BYTE_TIMEOUT, llm._STREAM_IDLE_TIMEOUT)")
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=repo_root)
    assert out.decode().strip() == "240.0 45.0"
