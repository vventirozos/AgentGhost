"""The owner's IBKR portfolio, read from the separate ibkr-service over the tailnet (§4TI).

The agent holds no IBKR secret and makes no IBKR connection: the service does that, on the normal internet,
by the operator's choice. This tool only reads the service's text, and every figure in it is computed there.
It has no way to place an order, because the service has no such route.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..utils.logging import requester_is_member

DEFAULT_KEY_FILE = "~/Data/AI/.ibkr_service_key"
ACTIONS = ("summary", "position", "health", "forecast", "review", "status")
_SYMBOL_RE = re.compile(r"[A-Za-z0-9 ./\-]{1,32}")

PORTFOLIO_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "portfolio",
        "description": (
            "The owner's own brokerage holdings, from their read-only IBKR service. Use it for their stocks, "
            "portfolio, gains and losses, and for what deserves a look. Actions: 'summary' (holdings, value, "
            "gains, cash, dividends), 'position' (one holding; needs symbol), 'health' (concentration, currency "
            "mix, volatility, past falls, comparison with an index), 'forecast' (ranges from past returns, for "
            "one symbol or the whole mix), 'review' (fixed-rule prompts, each with its reason and the case "
            "against acting), 'status' (is the service working). It reads only and cannot trade. Report its "
            "figures, dates and caveats as they are: never compute your own, and never turn a prompt or a range "
            "into a buy or sell instruction or a price prediction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "symbol": {"type": "string", "description": "A ticker the owner holds. Required for 'position', optional for 'forecast'."},
            },
            "required": ["action"],
        },
    },
}


def portfolio_settings() -> tuple:
    return (os.environ.get("GHOST_PORTFOLIO_URL", "").strip().rstrip("/"),
            os.environ.get("GHOST_PORTFOLIO_KEY_FILE", DEFAULT_KEY_FILE).strip())


def _read_key(key_file: str) -> str:
    try:
        return Path(os.path.expanduser(key_file)).read_text().strip()
    except OSError:
        return ""


def portfolio_configured() -> bool:
    """True when the URL is set and the key file holds a key. Configuration only, never reachability."""
    url, key_file = portfolio_settings()
    return bool(url and _read_key(key_file))


async def tool_portfolio(action: str = "summary", symbol: str = "", context=None, **_ignored) -> str:
    if requester_is_member():  # the allowlist's wall, held at the data too
        return "Error: the portfolio is the owner's private data and isn't available on this channel."
    action = str(action or "summary").strip().lower()
    if action not in ACTIONS:
        return f"Error: unknown portfolio action {action!r}. Use one of: {', '.join(ACTIONS)}."
    symbol = str(symbol or "").strip()
    if symbol and not _SYMBOL_RE.fullmatch(symbol):
        return "Error: that doesn't look like a ticker symbol."
    if action == "position" and not symbol:
        return "Error: portfolio action 'position' needs a symbol."
    url, key_file = portfolio_settings()
    key = _read_key(key_file)
    if not url or not key:
        return "Error: the portfolio service isn't configured on this agent."
    import httpx
    from ..core.llm import compute_tor_proxy
    target = f"{url}/v1/{action}"
    try:
        # A tailnet address stays direct and anything public goes through Tor: the agent's normal rule.
        proxy = compute_tor_proxy(target, getattr(context, "tor_proxy", None))
        async with httpx.AsyncClient(proxy=proxy, timeout=20.0, follow_redirects=False) as client:
            response = await client.get(target, params={"symbol": symbol} if symbol else None,
                                        headers={"X-Ibkr-Service-Key": key})
    except (httpx.HTTPError, OSError) as exc:
        return f"Error: the portfolio service didn't answer ({type(exc).__name__}). No figures are available right now."
    if response.status_code == 401:
        return "Error: the portfolio service refused this agent's key."
    try:
        body = response.json()
        text = str(body["text"])
        ok = response.status_code == 200 and body.get("ok") is True
    except (ValueError, KeyError, TypeError):
        return f"Error: the portfolio service sent an unreadable answer (HTTP {response.status_code})."
    return text if ok else f"Error: {text}"
