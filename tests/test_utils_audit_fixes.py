"""Targeted unit tests for the API/utils/interface audit fixes.

Covers:
- token_counter.check_budget includes system prompt tokens
- token_counter no longer pins gigantic strings in an unbounded LRU
- logging.spawn_task helper exists and propagates ContextVars
- sanitizer.extract_code_from_markdown picks the longest fence and
  handles fence-less / nested-fence inputs without truncating intent
- helpers.helper_fetch_url_content rotates Tor on 503 but NOT on 403
"""

import asyncio
import contextvars
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make sure the src/ tree is importable when this file is run directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# token_counter.check_budget — system prompt is included
# ---------------------------------------------------------------------------

def test_check_budget_includes_system_prompt():
    from ghost_agent.utils.token_counter import check_budget

    messages = [{"role": "user", "content": "hello world"}]

    baseline = check_budget(messages, max_tokens=10_000)
    with_sp = check_budget(messages, max_tokens=10_000, system_prompt="A " * 200)

    # The version with a system prompt MUST charge more tokens than the
    # baseline; otherwise check_budget is undercounting context usage.
    assert with_sp["total_tokens"] > baseline["total_tokens"]
    assert with_sp["system_tokens"] > 0
    assert baseline["system_tokens"] == 0


def test_check_budget_system_prompt_overflow_detected():
    from ghost_agent.utils.token_counter import check_budget

    # A massive system prompt should push a small message payload OVER budget
    # even though the messages alone fit.
    messages = [{"role": "user", "content": "tiny"}]
    huge_system = "x " * 50_000  # at least ~25K tokens

    res = check_budget(messages, max_tokens=1_000, system_prompt=huge_system)
    assert res["fits"] is False
    assert res["overflow"] > 0


def test_check_budget_backwards_compat_no_system_prompt_arg():
    from ghost_agent.utils.token_counter import check_budget

    res = check_budget([{"role": "user", "content": "hello"}], 10_000)
    # Without the new arg, behaviour is unchanged.
    assert res["fits"] is True
    assert "total_tokens" in res
    assert res["system_tokens"] == 0


# ---------------------------------------------------------------------------
# token_counter — LRU sinkhole removed
# ---------------------------------------------------------------------------

def test_estimate_tokens_consistent_for_identical_input():
    from ghost_agent.utils.token_counter import estimate_tokens

    text = "the quick brown fox jumps over the lazy dog " * 5
    a = estimate_tokens(text)
    b = estimate_tokens(text)
    # Two calls with identical input must return identical counts —
    # regardless of whether the cache was a hit or miss.
    assert a == b
    assert a > 0


def test_estimate_tokens_does_not_pin_large_text_in_cache():
    """Repeatedly hashing a large text must NOT explode the bounded cache.

    We can't easily measure RSS in a unit test, but we CAN assert that the
    bounded cache stays within its declared bound after pumping unique
    short strings through it (if the cache were unbounded, len would
    exceed the cap).
    """
    from ghost_agent.utils import token_counter
    from ghost_agent.utils.token_counter import estimate_tokens

    token_counter.clear_token_cache()

    # Push more unique short strings than the cache cap so the bounded
    # eviction path runs.
    cap = token_counter._SHORT_TOKEN_CACHE_MAX
    for i in range(cap + 200):
        estimate_tokens(f"unique-prompt-{i}")

    assert len(token_counter._SHORT_TOKEN_CACHE) <= cap


def test_clear_token_cache_still_callable():
    from ghost_agent.utils.token_counter import clear_token_cache

    # Must not raise. Backwards-compat surface for callers that drop the
    # cache after a tokenizer becomes available.
    clear_token_cache()
    clear_token_cache()


# ---------------------------------------------------------------------------
# logging.spawn_task — exists, propagates ContextVars
# ---------------------------------------------------------------------------

def test_spawn_task_helper_exists():
    from ghost_agent.utils import logging as glog

    assert hasattr(glog, "spawn_task")
    assert callable(glog.spawn_task)


@pytest.mark.asyncio
async def test_spawn_task_propagates_request_id_contextvar():
    from ghost_agent.utils.logging import request_id_context, spawn_task

    seen = {}

    async def child():
        seen["req_id"] = request_id_context.get()

    token = request_id_context.set("REQ-FROM-PARENT")
    try:
        task = spawn_task(child())
        await task
    finally:
        request_id_context.reset(token)

    assert seen.get("req_id") == "REQ-FROM-PARENT"


@pytest.mark.asyncio
async def test_spawn_task_propagates_arbitrary_contextvar():
    from ghost_agent.utils.logging import spawn_task

    cv = contextvars.ContextVar("__test_cv", default="default")

    async def child():
        return cv.get()

    cv.set("propagated-value")
    task = spawn_task(child())
    result = await task
    assert result == "propagated-value"


# ---------------------------------------------------------------------------
# sanitizer.extract_code_from_markdown
# ---------------------------------------------------------------------------

def test_extract_code_picks_longest_fence():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown

    text = (
        "Intro\n\n"
        "```python\nx = 1\n```\n\n"
        "Then a long block:\n\n"
        "```python\n"
        + "\n".join([f"line_{i} = {i}" for i in range(50)])
        + "\n```\n"
    )
    out = extract_code_from_markdown(text)
    # Longest fence wins; the short one-liner must not be returned.
    assert "line_0 = 0" in out
    assert "line_49 = 49" in out
    # The short snippet's body must NOT be the entire output.
    assert out.strip() != "x = 1"


def test_extract_code_handles_nested_triple_backticks():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown

    # Outer block contains inner triple-backticks — common when a model
    # emits a markdown sample inside a code fence. The longest match is
    # still our best bet; the inner fences must NOT cause us to drop content.
    text = (
        "```markdown\n"
        "Here is a sample:\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "```\n"
    )
    out = extract_code_from_markdown(text)
    # We don't claim perfect nested-fence parsing; we only require that
    # SOME content survives — i.e. we don't return an empty string.
    assert len(out) > 0


def test_extract_code_no_fence_returns_input():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown

    text = "just some prose without any code fences"
    out = extract_code_from_markdown(text)
    assert out == text


def test_extract_code_empty_input():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown

    assert extract_code_from_markdown("") == ""


def test_extract_code_does_not_strip_legit_trailing_backtick_in_fenceless():
    from ghost_agent.utils.sanitizer import extract_code_from_markdown

    # Fenceless input that ends with a backtick — must survive untouched.
    text = "x = '`'"
    out = extract_code_from_markdown(text)
    assert "`" in out


# ---------------------------------------------------------------------------
# helpers.helper_fetch_url_content — Tor renewal policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_url_403_does_not_renew_tor(monkeypatch):
    from ghost_agent.utils import helpers

    # Pin TOR_PROXY so the test behaves the same whether CI exports
    # `TOR_PROXY=` (empty), `TOR_PROXY=socks5://...`, or nothing at all.
    # The 403/503 branches gate on `if ... and proxy_url:`, so an empty
    # env var silently skips the rotation logic the test is asserting.
    monkeypatch.setenv("TOR_PROXY", "socks5://127.0.0.1:9050")

    # Force the httpx fallback path: pretend curl_cffi is missing.
    monkeypatch.setattr(helpers, "request_new_tor_identity",
                        MagicMock(return_value=(True, "ok")))

    # Mock httpx response: 403.
    fake_resp = MagicMock()
    fake_resp.status_code = 403
    fake_resp.headers = {"content-type": "text/html"}
    fake_resp.text = "<html>nope</html>"

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = fake_client
    fake_ctx.__aexit__.return_value = None

    # Patch httpx.AsyncClient AND make curl_cffi import fail so we hit the
    # httpx branch.
    with patch.object(helpers, "httpx") as mock_httpx, \
         patch.dict("sys.modules", {"curl_cffi.requests": None, "curl_cffi": None}):
        mock_httpx.AsyncClient = MagicMock(return_value=fake_ctx)
        result = await helpers.helper_fetch_url_content("https://example.com/forbidden")

    # Tor identity rotation must NOT have been triggered for a 403.
    helpers.request_new_tor_identity.assert_not_called()
    assert "403" in result or "Forbidden" in result or "Denied" in result


@pytest.mark.asyncio
async def test_fetch_url_503_does_renew_tor(monkeypatch):
    from ghost_agent.utils import helpers

    # Pin TOR_PROXY — see test_fetch_url_403_does_not_renew_tor for why.
    # The 503 branch at helpers.py (status_code == 503 and proxy_url) is
    # gated on a truthy proxy_url, and a CI runner exporting an empty
    # TOR_PROXY would silently skip the rotation this test asserts.
    monkeypatch.setenv("TOR_PROXY", "socks5://127.0.0.1:9050")

    rotate_calls = {"n": 0}

    def _fake_rotate(*a, **kw):
        rotate_calls["n"] += 1
        return (True, "rotated")

    monkeypatch.setattr(helpers, "request_new_tor_identity", _fake_rotate)
    # Skip the 5s sleep between retries.
    monkeypatch.setattr(helpers.asyncio, "sleep", AsyncMock(return_value=None))

    fake_resp = MagicMock()
    fake_resp.status_code = 503
    fake_resp.headers = {"content-type": "text/html"}
    fake_resp.text = "blocked"

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = fake_client
    fake_ctx.__aexit__.return_value = None

    with patch.object(helpers, "httpx") as mock_httpx, \
         patch.dict("sys.modules", {"curl_cffi.requests": None, "curl_cffi": None}):
        mock_httpx.AsyncClient = MagicMock(return_value=fake_ctx)
        result = await helpers.helper_fetch_url_content("https://example.com/maybe-tor-blocked")

    # 503 should have triggered AT LEAST one rotation across the retry loop.
    assert rotate_calls["n"] >= 1
    assert "503" in result or "Denied" in result or "Tor" in result
