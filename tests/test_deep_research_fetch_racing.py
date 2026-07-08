"""deep_research per-URL Tor fetch racing (2026-07-08).

Search reachability was fixed by per-engine circuits; the page-FETCH stage
still shared one circuit across all URLs, capped the outer timeout BELOW
the client's own (killing slow-but-live Tor fetches), and never retried a
failed URL. Result: ~2 of 8 sources survived (chess-session research).

Fix: each URL rides its own Tor circuit, a circuit-retryable failure
(timeout / 503 / connection error) is retried on a FRESH exit, the outer
timeout sits above the client's 20s, and concurrency rose 2→3.
"""
from unittest.mock import AsyncMock, patch

import pytest

from ghost_agent.tools.search import (
    _FETCH_ATTEMPTS,
    _FETCH_ATTEMPT_TIMEOUT,
    _fetch_error_is_retryable,
    _proxy_for_attempt,
    tool_deep_research,
)


class TestRetryableClassifier:
    @pytest.mark.parametrize("err", [
        "Error: Fetch of x timed out after 22s",
        "Error: Access Denied (503) via Tor. The site blocks Tor exit nodes.",
        "Error reading http://x: ConnectionError('closed')",
        "Error: Received status 502 from x",
        "Error: some never-before-seen failure",  # unknown → retryable
    ])
    def test_circuit_dependent_errors_retry(self, err):
        assert _fetch_error_is_retryable(err) is True

    @pytest.mark.parametrize("err", [
        "Error: URL points to a binary file.",
        "Error: Access Denied (403) from x. Application-level forbidden.",
        "Error: Access Denied (401) from x.",
        "Error: response from x is 9 MB; refusing to read more than 5 MB.",
        "Error: Received status 404 from x",
        "Error: scheme 'file' not allowed",
        "Error: host resolves to internal address",
    ])
    def test_definitive_errors_do_not_retry(self, err):
        assert _fetch_error_is_retryable(err) is False


class TestTimeoutAlignment:
    def test_outer_timeout_above_client_20s(self):
        # The mojeek-bug twin: a 15s outer wait_for guillotined fetches the
        # curl_cffi client (20s) would have completed. The outer budget must
        # sit ABOVE the client timeout.
        assert _FETCH_ATTEMPT_TIMEOUT > 20.0
        assert _FETCH_ATTEMPTS >= 2


class TestPerUrlCircuit:
    def test_distinct_circuit_per_url_and_attempt(self):
        base = "socks5h://127.0.0.1:9050"
        a0 = _proxy_for_attempt(base, "http://a.com", 0, salt="fetch")
        a1 = _proxy_for_attempt(base, "http://a.com", 1, salt="fetch")
        b0 = _proxy_for_attempt(base, "http://b.com", 0, salt="fetch")
        assert a0 != a1  # fresh exit on retry
        assert a0 != b0  # different URL, different circuit
        assert _proxy_for_attempt(base, "http://a.com", 0, salt="fetch") == a0
        assert _proxy_for_attempt(None, "http://a.com", 0, salt="fetch") is None


def _ddgs_mock(hrefs):
    inst = AsyncMock()
    # DDGS is used sync inside a thread; return a normal MagicMock chain.
    from unittest.mock import MagicMock
    m = MagicMock()
    m.return_value.__enter__.return_value.text.return_value = [
        {"href": h} for h in hrefs]
    return m


class TestFetchRetryInDeepResearch:
    @pytest.mark.asyncio
    @patch("ddgs.DDGS")
    @patch("ghost_agent.tools.search.importlib.util.find_spec")
    async def test_retryable_failure_recovers_on_second_circuit(
            self, mock_find_spec, mock_ddgs):
        mock_find_spec.return_value = True
        from unittest.mock import MagicMock
        inst = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = inst
        inst.text.return_value = [{"href": "https://real-site.com/a"}]

        calls = {"n": 0}

        def flaky_fetch(url, **kw):  # sync side_effect — AsyncMock awaits the return
            calls["n"] += 1
            if calls["n"] == 1:
                return "Error: Fetch timed out after 22s"  # retryable
            return "Recovered page body about postgres."

        with patch("ghost_agent.tools.search.helper_fetch_url_content",
                   new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = flaky_fetch
            out = await tool_deep_research("pg", False, "socks5://127.0.0.1:9050",
                                           llm_client=None, max_context=8192)
        assert "Recovered page body" in out
        assert calls["n"] == 2  # first circuit failed, second recovered

    @pytest.mark.asyncio
    @patch("ddgs.DDGS")
    @patch("ghost_agent.tools.search.importlib.util.find_spec")
    async def test_definitive_failure_is_not_retried(
            self, mock_find_spec, mock_ddgs):
        mock_find_spec.return_value = True
        from unittest.mock import MagicMock
        inst = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = inst
        inst.text.return_value = [{"href": "https://real-site.com/a"}]

        calls = {"n": 0}

        def forbidden(url, **kw):  # sync side_effect — AsyncMock awaits the return
            calls["n"] += 1
            return "Error: Access Denied (403) from x. Tor rotation will not help."

        with patch("ghost_agent.tools.search.helper_fetch_url_content",
                   new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = forbidden
            await tool_deep_research("pg", False, "socks5://127.0.0.1:9050",
                                     llm_client=None, max_context=8192)
        assert calls["n"] == 1  # 403 is definitive — no wasted second circuit
