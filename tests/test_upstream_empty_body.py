"""Crash-proof handling of an empty / non-JSON upstream response.

Regression: after a context overflow, the upstream returned a 200 with a
0-byte body while the emergency-prune retry was in flight. `resp.json()`
raised `json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`,
which the generic handler logged as "Upstream Fatal" and re-raised — a
recoverable state became a hard crash with an opaque message.

The fix wraps the decode: an empty/non-JSON body is retried once, and on a
second failure raises a clean, explanatory RuntimeError (no bare decoder
traceback). A transient empty body followed by a good body self-heals.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import asyncio

import pytest

from ghost_agent.core.llm import LLMClient


class _FakeResp:
    """Minimal stand-in for an httpx.Response."""
    def __init__(self, *, text, status=200, good=None):
        self.text = text
        self.status_code = status
        self._good = good

    def raise_for_status(self):
        return None

    def json(self):
        if self._good is not None:
            return self._good
        raise json.JSONDecodeError("Expecting value", self.text or "", 0)


class _FakePost:
    """Async callable returning queued responses, one per call."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


@pytest.fixture
def client():
    c = LLMClient("http://upstream.invalid")
    return c


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # The retry path awaits asyncio.sleep(2); skip the real wait.
    async def _instant(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", _instant)


async def test_empty_body_then_success_self_heals(client):
    good = {"choices": [{"message": {"content": "recovered"}}]}
    client.http_client.post = _FakePost([
        _FakeResp(text=""),                       # 1st attempt: empty body
        _FakeResp(text="{}", good=good),          # retry: valid JSON
    ])
    data = await client._do_chat_completion({"messages": []})
    assert data == good
    assert client.http_client.post.calls == 2     # retried exactly once


async def test_empty_body_twice_raises_clean_error(client):
    client.http_client.post = _FakePost([
        _FakeResp(text=""),                       # 1st attempt: empty
        _FakeResp(text=""),                       # retry: still empty
    ])
    with pytest.raises(RuntimeError) as ei:
        await client._do_chat_completion({"messages": []})
    msg = str(ei.value)
    # Clean, explanatory message — NOT the raw "Expecting value: line 1..." .
    assert "empty/non-JSON response" in msg
    assert "Expecting value" not in msg
    assert client.http_client.post.calls == 2


async def test_valid_body_first_try_unaffected(client):
    good = {"choices": [{"message": {"content": "ok"}}]}
    client.http_client.post = _FakePost([_FakeResp(text="{}", good=good)])
    data = await client._do_chat_completion({"messages": []})
    assert data == good
    assert client.http_client.post.calls == 1     # no retry needed
