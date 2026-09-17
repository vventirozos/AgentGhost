"""Errors an identical retry cannot fix name their next action (2026-09-15, §4HB).

Live (req 6a7882f5): `net::ERR_HTTP2_PROTOCOL_ERROR` on one site and
"Execution context was destroyed" on another were each re-issued unchanged
on the very next turn and failed the same way. The generic hint mentions
neither. The route table is one implementation with one consumer; this
file walks it.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.tools.browser import tool_browser
from ghost_agent.tools.browser_routes import (
    _DETERMINISTIC_ERROR_ROUTES,
    _deterministic_error_route,
)


def _stub(err_text):
    stub = MagicMock()
    stub.execute = lambda cmd, timeout=300, **kw: (f"[BROWSER_ERR] {err_text}", 1)
    return stub


@pytest.mark.parametrize("marker,route", _DETERMINISTIC_ERROR_ROUTES)
def test_every_table_row_reaches_the_hint(marker, route):
    """R1 — FAILS IF: a row is added that the lookup cannot match, or a
    route that says nothing actionable."""
    assert _deterministic_error_route(f"Error: Page.goto: {marker} at https://x/") == route
    assert len(route) > 40
    assert "retry" in route.lower() or "re-run" in route.lower()


@pytest.mark.asyncio
async def test_the_live_http2_error_gets_the_no_retry_route(tmp_path):
    """FAILS IF: the route is computed but not threaded into the failure hint."""
    out = await tool_browser(
        operation="extract_text",
        url="https://pasqualepillitteri.it/en/news/15859/x",
        sandbox_dir=tmp_path,
        sandbox_manager=_stub("Error: Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://pasqualepillitteri.it/x"))
    text = str(out)
    assert "STATUS: ERROR" in text
    assert "do not retry this URL" in text
    # …and it comes BEFORE the generic advice, where the model reads first.
    assert text.index("do not retry this URL") < text.index("If this is a navigation timeout")


@pytest.mark.asyncio
async def test_the_live_context_destroyed_error_gets_the_interact_route(tmp_path):
    out = await tool_browser(
        operation="extract_text", url="https://cybersecuritynews.com/revolut-data-breach/",
        sandbox_dir=tmp_path,
        sandbox_manager=_stub("Error: Page.evaluate: Execution context was destroyed, most likely because of a navigation"))
    text = str(out)
    assert "networkidle" in text and "operation='interact'" in text


@pytest.mark.asyncio
async def test_an_unlisted_error_gets_only_the_generic_hint(tmp_path):
    """FAILS IF: a route leaks onto errors it was not written for."""
    out = await tool_browser(
        operation="extract_text", url="https://example.com/",
        sandbox_dir=tmp_path,
        sandbox_manager=_stub("ValueError: selector 'main' did not match any element"))
    text = str(out)
    assert "do not retry this URL" not in text
    assert "networkidle" not in text
    assert "If this is a navigation timeout" in text


def test_socks_errors_are_deliberately_not_in_the_table():
    """FAILS IF: someone adds a circuit-level error — those ARE worth a retry
    on a fresh circuit, and the dead-onion memo already owns them."""
    for marker, _ in _DETERMINISTIC_ERROR_ROUTES:
        assert "SOCKS" not in marker
        assert "CONNECTION_REFUSED" not in marker
