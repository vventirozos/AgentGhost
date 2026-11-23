"""The owner's IBKR portfolio, read from the separate ibkr-service over the tailnet (§4TI).

The agent holds no IBKR secret and makes no IBKR connection: the service does that, on the normal internet,
by the operator's choice. This tool only reads the service's text, and every figure in it is computed there.
It has no way to place an order, because the service has no such route.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from ..utils.logging import requester_is_member

DEFAULT_KEY_FILE = "~/Data/AI/.ibkr_service_key"
ACTIONS = ("summary", "position", "health", "forecast", "review", "advise", "status")
MAX_AMOUNT = 10_000_000
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
            "against acting), 'advise' (how to split a sum the owner is about to invest: needs amount, in the "
            "account's currency; optional reserve, opportunity cash they already hold; optional carry, the core's "
            "leftover from last time), 'status' (is the service "
            "working). It reads only and cannot trade. Report its figures, names, dates and caveats as they are: "
            "never compute your own or name a holding from memory. Only 'advise' suggests buys, and only its own "
            "amounts: never change them, and never turn a prompt or a range into a buy or sell instruction or a "
            "price prediction. When 'advise' names a hot fall, search the news for why that name fell, open and read "
            "at least one page from an established news or financial outlet before naming a cause, and name that "
            "source. A claim seen only in search snippets or on one fringe site is an unconfirmed rumour: say so. "
            "Say when you found no reason. Relay the reply's NEXT TIME line: the service doesn't remember the "
            "carry and the opportunity cash, so the owner must give them next time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "symbol": {"type": "string", "description": "A ticker the owner holds. Required for 'position', optional for 'forecast'."},
                "amount": {"type": "number", "description": "For 'advise': the sum to invest now, in the account's currency."},
                "reserve": {"type": "number", "description": "For 'advise', optional: opportunity cash the owner already holds aside."},
                "carry": {"type": "number", "description": "For 'advise', optional: the core's leftover the owner carried over from last time (goes to the core only)."},
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


_FOREIGN_MONEY_RE = re.compile(r"[$£¥]|\b(?:USD|GBP|CHF|JPY|dollars?|pounds?)\b", re.IGNORECASE)


def _plain_number(text: str):
    """"500", "500 EUR", "€1,500", "1.500", "1.500,50", "2,50" -> a float string, or None.
    The LAST separator followed by 1-2 digits is the decimal mark; any other separator groups thousands.
    §4VH-c: "1.500" read as 1.50 and "2,50" as 250 before."""
    body = re.sub(r"(?i)\s*(?:€|euros?|eur)\s*", "", text.strip())  # §4VH-f: "euros" before "eur", or "500 euros" -> "500os"
    if not re.fullmatch(r"-?[0-9][0-9.,' ]*", body):
        return None
    body = body.replace(" ", "").replace("'", "")
    decimal = re.search(r"[.,]([0-9]{1,2})$", body)
    whole = body[:decimal.start()] if decimal else body
    if whole.count(".") and whole.count(","):
        return None  # mixed grouping marks: ambiguous
    if re.search(r"[.,]", whole) and not re.fullmatch(r"-?[1-9][0-9]{0,2}(?:[.,][0-9]{3})+", whole):  # "0.004" isn't 4
        return None
    whole = re.sub(r"[.,]", "", whole)
    return f"{whole}.{decimal.group(1)}" if decimal else whole


def _number(value, low: float):
    """A finite number in [low, MAX_AMOUNT], or None. Text must be an amount in the account's currency: a sum in
    dollars or pounds is refused rather than read as euros."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if _FOREIGN_MONEY_RE.search(value):
            return None
        value = _plain_number(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and low <= number <= MAX_AMOUNT else None


async def tool_portfolio(action: str = "summary", symbol: str = "", amount=None, reserve=None, carry=None, context=None,
                         **_ignored) -> str:
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
    params = {"symbol": symbol} if symbol else {}
    if action == "advise":
        money = _number(amount, 0.01)
        if money is None:
            return ("Error: portfolio action 'advise' needs amount, a number above 0 in the account's currency (convert a sum "
                    "in another currency first; write 1500 or 1500.50).")
        params = {"amount": f"{money:.2f}"}
        for name, value in (("reserve", reserve), ("carry", carry)):
            if value not in (None, "", 0):
                held = _number(value, 0.0)
                if held is None:
                    return f"Error: portfolio '{name}' must be a number of 0 or more."
                params[name] = f"{held:.2f}"
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
            response = await client.get(target, params=params or None,
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
