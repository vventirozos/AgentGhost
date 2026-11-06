"""Public prices and history from Yahoo's chart endpoint. Only ticker symbols ever leave this machine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

#: IBKR listing exchange -> Yahoo suffix. An exchange not listed here keeps IBKR's own price (with a note);
#: `symbol_overrides` in config.json maps a holding to any Yahoo symbol.
EXCHANGE_SUFFIX = {
    **dict.fromkeys(("NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "IEX", "PINK", "ISLAND", "NYSENAT"), ""),
    **dict.fromkeys(("IBIS", "IBIS2", "XETRA", "GETTEX", "TGATE"), ".DE"),
    "FWB": ".F", "FWB2": ".F", "SWB": ".SG",
    **dict.fromkeys(("LSE", "LSEETF"), ".L"),
    "AEB": ".AS", "SBF": ".PA", "ENEXT.BE": ".BR",
    **dict.fromkeys(("BVME", "BVME.ETF"), ".MI"),
    "BM": ".MC", "EBS": ".SW", "VSE": ".VI", "SFB": ".ST", "CPH": ".CO", "OSE": ".OL", "HEX": ".HE",
    **dict.fromkeys(("ATH", "ATHEX", "ASE.GR"), ".AT"),
    "ASX": ".AX", "TSE": ".TO", "TSEJ": ".T", "SEHK": ".HK",
}
#: A listing quoted in minor units, and the unit IBKR uses for it.
MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}
#: Further from IBKR's last mark than this, the Yahoo line is probably another listing, not a real move.
MAX_MARK_GAP = 0.25


class RateLimited(OSError):
    """The source answered HTTP 429. Only this stops a round: other failures concern one symbol."""


def yahoo_symbol(position: dict, overrides: dict | None = None) -> str | None:
    """The Yahoo symbol of a stock or ETF holding, or None when there is no reliable mapping."""
    symbol = position.get("symbol") or ""
    if position.get("asset") != "STK" or not symbol:
        return None
    for key in (position.get("isin"), symbol):
        if key and (overrides or {}).get(key):
            return overrides[key]
    suffix = EXCHANGE_SUFFIX.get(position.get("exchange") or "")
    if suffix is None:
        return None
    name = symbol.strip().replace(" ", "-")
    if suffix == ".HK":
        name = name.zfill(4)  # IBKR writes 700, Yahoo 0700.HK
    return name + suffix


def local_date(stamp: float, offset: int = 0) -> str:
    return (datetime.fromtimestamp(stamp, timezone.utc) + timedelta(seconds=offset)).date().isoformat()


def fetch_chart(symbol: str, get: Callable[[str, dict], str], loads, span: str = "5d") -> dict:
    """Price, previous session's close and dated adjusted closes for one symbol, in major currency units.
    Dates are the exchange's own, so a morning in Sydney is not filed under yesterday."""
    body = loads(get(CHART_URL + symbol, {"range": span, "interval": "1d"}))
    chart = body.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"no data for {symbol}: {(chart.get('error') or {}).get('code', 'empty answer')}")
    result = results[0]
    meta = result.get("meta") or {}
    currency, factor = MINOR_UNITS.get(meta.get("currency"), (meta.get("currency") or "", 1.0))
    if not isinstance(meta.get("regularMarketPrice"), (int, float)) or meta["regularMarketPrice"] <= 0:
        raise ValueError(f"no price for {symbol}")
    offset = int(meta.get("gmtoffset") or 0)
    at = int(meta.get("regularMarketTime") or 0)
    session = local_date(at, offset) if at else ""
    stamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    raw_closes = ((indicators.get("quote") or [{}])[0].get("close") or [])
    adj_closes = ((indicators.get("adjclose") or [{}])[0].get("adjclose") or raw_closes)
    closes = [(local_date(ts, offset), float(px) * factor)
              for ts, px in zip(stamps, adj_closes) if isinstance(px, (int, float)) and px > 0]
    plain = [(local_date(ts, offset), float(px) * factor)
             for ts, px in zip(stamps, raw_closes) if isinstance(px, (int, float)) and px > 0]
    earlier = [px for day, px in plain if session and day < session]
    return {
        "symbol": symbol, "price": float(meta["regularMarketPrice"]) * factor, "currency": currency, "at": at,
        "offset": offset, "session": session, "previous_close": earlier[-1] if earlier else None, "closes": closes,
    }


def quote_problem(position: dict, quote: dict) -> str | None:
    """Why this Yahoo quote can't be trusted for this holding, or None. Guards against a wrong listing."""
    if position.get("currency") and quote.get("currency") != position["currency"]:
        return f"Yahoo quotes {quote.get('symbol')} in {quote.get('currency')}, the holding is in {position['currency']}"
    mark = position.get("mark")
    if mark and abs(quote["price"] / mark - 1) > MAX_MARK_GAP:
        return f"Yahoo's {quote.get('symbol')} is over {MAX_MARK_GAP:.0%} from IBKR's last price (other listing or a split)"
    return None
