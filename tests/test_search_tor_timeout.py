"""Search reliability over Tor — ddgs request timeout (log-audit fix,
2026-06-11).

Measured directly over Tor: the engine that actually returns results
(usually mojeek) responds in ~10-18s through a circuit, while the others
fail fast. The previous 8s ddgs timeout killed mojeek mid-request
("error sending request for url (mojeek...)"), so every engine came back
empty and the search failed even though results were reachable — the
agent then burned minutes on retries that could never win. The timeout is
now a measured-safe constant; these tests pin that it's adequate and that
every ddgs call site uses it.
"""

from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.tools.search import (
    _DDGS_TOR_TIMEOUT,
    tool_search_ddgs,
    tool_deep_research,
)


def test_timeout_clears_mojeek_tor_latency():
    # mojeek over Tor was measured at ~10-18s; 8s killed it. The constant
    # must comfortably clear that — never regress below the safe floor.
    assert _DDGS_TOR_TIMEOUT >= 15


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_web_search_passes_safe_timeout_to_ddgs(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = [{"title": "T", "body": "B", "href": "http://ok.com"}]

    await tool_search_ddgs("integrated information theory transformers", None)

    # DDGS(...) constructor must receive the measured-safe timeout.
    assert mock_ddgs.call_args is not None
    assert mock_ddgs.call_args.kwargs.get("timeout") == _DDGS_TOR_TIMEOUT


@pytest.mark.asyncio
@patch("src.ghost_agent.tools.search.helper_fetch_url_content")
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_deep_research_passes_safe_timeout_to_ddgs(
    mock_find_spec, mock_ddgs, mock_fetch
):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    inst.text.return_value = [{"title": "T", "body": "B", "href": "http://ok.com"}]

    async def _fake_fetch(url):
        return "some page text"

    mock_fetch.side_effect = _fake_fetch

    await tool_deep_research("ai consciousness theories", tor_proxy="socks5://127.0.0.1:9050")

    assert mock_ddgs.call_args is not None
    assert mock_ddgs.call_args.kwargs.get("timeout") == _DDGS_TOR_TIMEOUT


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_reformulation_also_uses_safe_timeout(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    inst = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = inst
    # primary attempts return nothing → reformulation path runs, also empty
    inst.text.return_value = []

    await tool_search_ddgs("a query that yields nothing anywhere", None)

    # EVERY DDGS() construction (primary attempts + reformulations) used the
    # safe timeout — no call site left on the old 8s ceiling.
    timeouts = {c.kwargs.get("timeout") for c in mock_ddgs.call_args_list}
    assert timeouts == {_DDGS_TOR_TIMEOUT}
