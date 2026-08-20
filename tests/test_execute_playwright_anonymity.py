"""§4BX/#5 — raw Playwright in stateful execute is flagged when it omits the
Tor proxy. The `browser` tool enforces the proxy; this escape hatch does not,
so a public navigation from a proxyless launch dials cleartext from the real
IP. Non-blocking (a file:// launch legitimately needs none), so a loud notice.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.tools.execute import _proxyless_browser_launch, tool_execute


@pytest.mark.parametrize("code,expected", [
    ("chromium.launch(headless=True, args=['--no-sandbox'])", True),
    ("await p.firefox.launch(headless=True)", True),
    ("async with async_playwright() as p:", True),
    ("await p.chromium.launch(proxy={'server': os.environ['TOR_PROXY']})", False),
    ("args = a if os.environ.get('TOR_PROXY') else b", False),  # references it
    ("import pandas as pd; pd.read_csv('x.csv')", False),        # no browser
    ("", False),
])
def test_detection(code, expected):
    assert _proxyless_browser_launch(code) is expected


def _run_stateful(content, tor_proxy):
    """Drive tool_execute(stateful=True) with the jupyter machinery mocked,
    returning the result string."""
    sandbox_dir = Path("/tmp/workspace")
    sm = MagicMock()
    sm.tor_proxy = tor_proxy

    with patch("ghost_agent.tools.execute._get_safe_path") as msp, \
         patch("ghost_agent.tools.execute.asyncio.to_thread") as mtt:
        orig = MagicMock()
        orig.stat.return_value.st_size = 0
        wrapper = MagicMock()

        def _safe(sb, fn):
            return wrapper if str(fn) == ".jupyter_runner.py" else orig
        msp.side_effect = _safe

        async def _tt(func, *a, **k):
            if func == sm.execute:
                return ("hello", 0)
            if func in (orig.write_text, wrapper.write_text,
                        orig.parent.mkdir, wrapper.parent.mkdir,
                        wrapper.unlink, orig.unlink):
                return None
            return func(*a, **k)
        mtt.side_effect = _tt

        import asyncio
        return asyncio.run(tool_execute(
            filename="sol.py", content=content, sandbox_dir=sandbox_dir,
            sandbox_manager=sm, stateful=True))


def test_proxyless_launch_gets_the_anonymity_notice():
    r = _run_stateful(
        "from playwright.async_api import async_playwright\n"
        "p = await async_playwright().start()\n"
        "b = await p.chromium.launch(headless=True, args=['--no-sandbox'])\n",
        tor_proxy="socks5://127.0.0.1:9050")
    assert "ANONYMITY" in r, (
        f"a proxyless raw-Playwright launch was not flagged: {r[:200]!r}")


def test_a_proxied_launch_is_NOT_flagged():
    r = _run_stateful(
        "b = await p.chromium.launch(proxy={'server': os.environ['TOR_PROXY']})\n",
        tor_proxy="socks5://127.0.0.1:9050")
    assert "ANONYMITY" not in r


def test_non_browser_code_is_NOT_flagged():
    r = _run_stateful("import pandas as pd\nprint(pd.__version__)\n",
                      tor_proxy="socks5://127.0.0.1:9050")
    assert "ANONYMITY" not in r


def test_no_tor_proxy_configured_no_notice():
    # If the sandbox has no Tor proxy at all, there is nothing to leak past.
    r = _run_stateful(
        "b = await p.chromium.launch(headless=True)\n", tor_proxy=None)
    assert "ANONYMITY" not in r
