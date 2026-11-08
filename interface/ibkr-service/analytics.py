"""Every number the agent reports is computed here, from plain data, with no network and no clock of its own."""
from __future__ import annotations

import math
from datetime import date
from statistics import mean, pstdev

from flex import NOTIONAL_ASSETS
from market import local_date

TRADING_DAYS = 252
HORIZONS = ((21, "1 month"), (63, "3 months"), (252, "1 year"))
MIN_SERIES_DAYS = 120
#: A public price or FX rate older than this is no longer used.
PRICE_STALE = 4 * 86400
#: A price history whose last close is older than this says so, and its fall-from-high prompt stays quiet.
HISTORY_STALE_DAYS = 7
MAX_FX_GAP = 0.25
#: Yahoo instrument types that hold many companies. A fund is held to `max_fund` instead of the one-company
#: limit (§4TI-d: an S&P 500 fund at 88% was flagged against the 20% single-stock limit, noise the owner learns
#: to ignore). A type can't tell a broad index fund from a one-sector one, so the review says so.
FUND_KINDS = frozenset({"ETF", "MUTUALFUND"})


def is_fund(row: dict) -> bool:
    return row.get("kind") in FUND_KINDS


def percentile(values: list, q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def daily_returns(prices: list) -> list:
    return [b / a - 1 for a, b in zip(prices, prices[1:]) if a > 0]


def max_drawdown(prices: list) -> float:
    """The worst fall from a previous high, as a negative fraction."""
    peak, worst = 0.0, 0.0
    for price in prices:
        peak = max(peak, price)
        if peak > 0:
            worst = min(worst, price / peak - 1)
    return worst


def annual_volatility(prices: list) -> float | None:
    returns = daily_returns(prices)
    return pstdev(returns) * math.sqrt(TRADING_DAYS) if len(returns) >= 20 else None


def horizon_band(prices: list, days: int) -> dict | None:
    """What past `days`-long stretches returned. Needs at least 2 x days of history, or it says nothing."""
    if len(prices) < 2 * days + 1:
        return None
    windows = [prices[i + days] / prices[i] - 1 for i in range(len(prices) - days)]
    return {"low": percentile(windows, 0.10), "mid": percentile(windows, 0.50), "high": percentile(windows, 0.90),
            "periods": len(prices) // days}


def bands(prices: list) -> list:
    rows = []
    for days, label in HORIZONS:
        band = horizon_band(prices, days)
        if band:
            rows.append({"horizon": label, **band})
    return rows


def days_old(day: str, now: float) -> int | None:
    try:
        return (date.fromisoformat(local_date(now)) - date.fromisoformat(day)).days
    except (TypeError, ValueError):
        return None


def blended_series(histories: dict, weights: dict, all_weight: float | None = None) -> dict | None:
    """The value of today's long holdings at constant weights over their shared dates, starting at 1.0.
    Hypothetical by design: it shows how this mix would have behaved, not what the account really did.
    `all_weight` is the weight of every long holding, including those with no history to weigh."""
    usable = {s: dict(histories[s]) for s, w in weights.items() if w > 0 and len(histories.get(s) or []) > MIN_SERIES_DAYS}
    total_weight = sum(weights.get(s, 0) for s in usable)
    every = all_weight if all_weight is not None else sum(w for w in weights.values() if w > 0)
    if not usable or total_weight <= 0:
        return None
    dates = sorted(set.intersection(*(set(h) for h in usable.values())))
    if len(dates) <= MIN_SERIES_DAYS:
        return None
    level, series = 1.0, [1.0]
    for previous, day in zip(dates, dates[1:]):
        level *= 1 + sum(weights[s] / total_weight * (h[day] / h[previous] - 1) for s, h in usable.items())
        series.append(level)
    return {"dates": dates, "values": series, "symbols": sorted(usable),
            "coverage": total_weight / every if every else 0.0}


def correlation(a: list, b: list) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    mean_a, mean_b = mean(a), mean(b)
    sd_a, sd_b = pstdev(a), pstdev(b)
    if sd_a == 0 or sd_b == 0:
        return None
    return sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (len(a) * sd_a * sd_b)


def average_correlation(histories: dict, symbols: list, days: int = TRADING_DAYS, step: int = 5) -> float | None:
    """Mean pairwise correlation of WEEKLY returns over the last shared year. Daily closes in different time
    zones happen hours apart, which makes a US stock and a European one look less related than they are."""
    maps = {s: dict(histories[s]) for s in symbols if len(histories.get(s) or []) > MIN_SERIES_DAYS}
    if len(maps) < 2:
        return None
    dates = sorted(set.intersection(*(set(h) for h in maps.values())))[-(days + 1):]
    if len(dates) <= MIN_SERIES_DAYS:
        return None
    weekly = dates[::-1][::step][::-1]
    returns = {s: daily_returns([h[d] for d in weekly]) for s, h in maps.items()}
    names = sorted(returns)
    pairs = [correlation(returns[a], returns[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    pairs = [p for p in pairs if p is not None]
    return mean(pairs) if pairs else None


def _fresh(stamp, now: float) -> bool:
    return bool(stamp) and now - stamp <= PRICE_STALE


def _usable_quote(pos: dict, quote: dict | None, statement_date: str, now: float, problem) -> tuple:
    """(quote or None, why not, whether a quote has confirmed this listing is the holding's)."""
    if not quote:
        return None, "no public price yet", False
    trouble = problem(pos, quote)
    if trouble:
        return None, trouble, False
    if not _fresh(quote.get("at"), now):
        return None, "the last public price is too old", True
    if statement_date and quote.get("session") and quote["session"] < statement_date:
        return None, "the statement is newer than the last public price", True
    return quote, None, True


def _rate(pos: dict, base: str, fx: dict, now: float, use_statement: bool) -> tuple:
    """(rate to base or None, note). A public rate must be fresh, no older than the statement, and close to
    the statement's own rate."""
    currency = pos.get("currency")
    if currency == base:
        return 1.0, None
    statement = pos.get("fx_to_base") if use_statement else None
    entry = fx.get(f"{currency}{base}") or {}
    public = entry.get("rate") if _fresh(entry.get("at"), now) else None
    if public and statement:
        older = pos.get("report_date") and local_date(entry["at"]) < pos["report_date"]
        if older or abs(public / statement - 1) > MAX_FX_GAP:
            public = None
    if public:
        return public, None
    return (statement, "IBKR's statement rate") if statement else (None, None)


def label(row: dict) -> str:
    """The ticker with IBKR's own name for it, for every line that names a holding (§4TI-b). With a bare ticker
    a model fills the name in from memory, and on 11-08 gave a fund the wrong index."""
    symbol = str(row.get("symbol") or "")
    # no "|" (the line's own separator) and no brackets of its own: "EMIM (… USD (Acc)): " read as two labels
    name = " ".join(str(row.get("description") or "").replace("|", "/").replace("(", " ").replace(")", " ").split())
    return f"{symbol} ({name})" if name and name.upper() != symbol.upper() else symbol


def value_positions(flex: dict, quotes: dict, fx: dict, symbols: dict, base: str | None, problem, now: float) -> dict:
    """Value each holding and the whole account. A holding that can't be valued is listed, never guessed.
    `symbols` maps a holding's key to its Yahoo symbol."""
    rows, incomplete, gain_unknown, notional = [], [], [], []
    value_sum, gain_sum, gain_cost, day_sum, day_base = 0.0, 0.0, 0.0, 0.0, 0.0
    statement_date = flex.get("report_date") or ""
    for pos in flex.get("positions") or []:
        symbol, qty, asset = pos.get("symbol"), pos.get("qty"), pos.get("asset")
        row = {key: pos.get(key) for key in ("key", "symbol", "description", "asset", "currency", "qty", "cost", "cost_price", "isin")}
        yahoo = symbols.get(pos.get("key"))
        row["yahoo"] = yahoo
        row_date = pos.get("report_date") or statement_date
        quote, why_not, confirmed = _usable_quote(pos, quotes.get(yahoo) if yahoo else None, row_date, now, problem)
        row["history_symbol"] = yahoo if confirmed else None
        # a listing a quote has not confirmed as this holding says nothing about what the holding is
        row["kind"] = ((quotes.get(yahoo) or {}).get("kind") or "") if confirmed else ""
        # §4TI-e: the full public name when a quote confirmed the listing, else IBKR's own
        public = ((quotes.get(yahoo) or {}).get("name") or "") if confirmed else ""
        row["public_name"] = public
        row["label"] = label({**pos, "description": public or pos.get("description")})
        if quote:
            price, source, previous = quote["price"], "public", quote.get("previous_close")
            row["price_at"] = quote.get("at")
        else:
            price, source, previous = pos.get("mark"), "IBKR", None
            row["price_date"] = row_date
            row["price_note"] = "no public listing mapped" if not yahoo else why_not
        row.update(price=price, source=source)
        if asset in NOTIONAL_ASSETS:
            notional.append(symbol)
            row["notional"] = True
            rows.append(row)
            continue
        # The statement's rates are to the statement's base, so they say nothing about a base from config.json.
        rate, note = (None, None) if base is None else _rate(pos, base, fx, now, flex.get("base") == base)
        if note:
            row["fx_note"] = note
        # A share is quantity x price. For anything else (bonds quoted in percent, option multipliers) only
        # IBKR's own position value is safe.
        local = (qty * price if qty is not None and price is not None else None) if asset == "STK" else pos.get("position_value")
        if local is None or rate is None:
            incomplete.append(symbol)
            gain_unknown.append(symbol)
            rows.append(row)
            continue
        row.update(value_local=local, value_base=local * rate, fx=rate)
        value_sum += local * rate
        if pos.get("cost"):  # a zero cost (a spin-off, a gift) has no percentage and would inflate the total's
            gain = local - pos["cost"]
            row.update(gain_local=gain, gain_pct=gain / abs(pos["cost"]), gain_base=gain * rate)
            gain_sum += gain * rate
            gain_cost += abs(pos["cost"]) * rate
        else:
            gain_unknown.append(symbol)
        if previous and quote:
            is_today = quote.get("session") == local_date(now, quote.get("offset") or 0)
            row.update(move_pct=price / previous - 1, move_label="today" if is_today else f"last session {quote.get('session')}")
            if is_today:
                day_sum += qty * (price - previous) * rate
                day_base += abs(qty * previous * rate)
        rows.append(row)
    gross = sum(abs(r.get("value_base") or 0) for r in rows)
    for row in rows:
        if row.get("value_base") is not None and gross:
            row["weight"] = abs(row["value_base"]) / gross
    rows.sort(key=lambda r: abs(r.get("value_base") or 0), reverse=True)
    real = [r for r in rows if not r.get("notional")]
    valued_any = any(r.get("value_base") is not None for r in real)
    known = flex.get("positions_known", True)
    invested = value_sum if known and (valued_any or not real) else None
    cash = flex.get("cash")
    return {
        "base": base, "positions": rows, "incomplete": incomplete, "gain_unknown": gain_unknown, "notional": notional, "cash": cash,
        "invested": None if base is None else invested,
        "total": None if base is None or cash is None or invested is None else invested + cash,
        "gain": gain_sum if gain_cost else None, "gain_pct": gain_sum / gain_cost if gain_cost else None,
        "day": day_sum if day_base else None, "day_pct": day_sum / day_base if day_base else None,
    }


def portfolio_health(valued: dict, histories: dict, benchmark: str, benchmark_history: list, now: float) -> dict:
    """Concentration, currency mix and how today's mix would have behaved. Shorts stay out of the blend."""
    rows = [r for r in valued["positions"] if r.get("weight") is not None]
    currencies: dict = {}
    for row in rows:
        currencies[row["currency"]] = currencies.get(row["currency"], 0.0) + row["weight"]
    health = {"count": len(rows), "largest": rows[0] if rows else None,
              "top3": sum(r["weight"] for r in rows[:3]), "currencies": sorted(currencies.items(), key=lambda kv: -kv[1])}
    longs = [r for r in rows if (r.get("qty") or 0) > 0]
    weights: dict = {}
    for row in longs:
        if row.get("history_symbol"):
            weights[row["history_symbol"]] = weights.get(row["history_symbol"], 0.0) + row["weight"]
    blend = blended_series(histories, weights, sum(r["weight"] for r in longs))
    if blend:
        values = blend["values"]
        health.update(volatility=annual_volatility(values), drawdown=max_drawdown(values), history_days=len(values),
                      history_from=blend["dates"][0], history_to=blend["dates"][-1], coverage=blend["coverage"],
                      history_stale=(days_old(blend["dates"][-1], now) or 0) > HISTORY_STALE_DAYS,
                      correlation=average_correlation(histories, blend["symbols"]))
        by_date = dict(zip(blend["dates"], values))
        shared = [d for d, _ in benchmark_history if d in by_date][-(TRADING_DAYS + 1):]
        if len(shared) > TRADING_DAYS // 2:
            bench = dict(benchmark_history)
            health["versus"] = {"benchmark": benchmark, "days": len(shared) - 1, "from": shared[0], "to": shared[-1],
                                "mix": by_date[shared[-1]] / by_date[shared[0]] - 1,
                                "index": bench[shared[-1]] / bench[shared[0]] - 1}
    return health


def review_items(valued: dict, health: dict, histories: dict, limits: dict, now: float) -> tuple:
    """(items, unchecked). Fixed, visible rules: each item is a prompt to look, with its reason and what argues
    against acting. `unchecked` names the rules that couldn't run, so silence isn't read as all clear."""
    items, no_cost, no_history = [], [], []
    for row in valued["positions"]:
        symbol, weight, gain_pct = row["symbol"], row.get("weight"), row.get("gain_pct")
        named = row.get("label") or symbol
        fund = is_fund(row)
        limit = limits["max_fund"] if fund else limits["max_position"]
        if weight is not None and weight > limit:
            if fund:
                items.append({"rule": "concentration", "symbol": symbol,
                              "why": f"{named} is a fund at {weight:.1%} of your holdings, above your {limit:.0%} limit for one fund.",
                              "ask": "Does this fund spread your money over many companies and countries, or one sector or region?",
                              "against": "A broad index fund can reasonably be most of a portfolio, and selling may trigger tax."})
            elif row.get("kind") == "EQUITY":
                items.append({"rule": "concentration", "symbol": symbol,
                              "why": f"{named} is one company at {weight:.1%} of your holdings, above your {limit:.0%} limit.",
                              "ask": "Would you be comfortable if this one company fell by half?",
                              "against": "It may be a company you know well, and selling may trigger tax."})
            else:  # type unknown: held to the one-company limit, worded for either
                items.append({"rule": "concentration", "symbol": symbol,
                              "why": f"{named} is {weight:.1%} of your holdings, above your {limit:.0%} limit.",
                              "ask": "Would you be comfortable if this one company or fund fell by half?",
                              "against": "A broad index fund can reasonably be this large, and selling may trigger tax."})
        if gain_pct is None:
            no_cost.append(symbol)
        elif gain_pct <= limits["loss_review"]:
            items.append({"rule": "large_loss", "symbol": symbol,
                          "why": f"{named} is {gain_pct:.1%} against what you paid.",
                          "ask": "Has your reason for owning it changed, or only the price?",
                          "against": "What you paid says nothing about what happens next, so a loss alone is no reason to sell."})
        elif gain_pct >= limits["gain_review"] and (weight or 0) >= limits["gain_review_weight"] and not fund:
            # a fund that rose is the market that rose: "does it still deserve this much" is the size rule's
            # question, asked against the fund limit above (§4TI-d review)
            items.append({"rule": "large_gain", "symbol": symbol,
                          "why": f"{named} is up {gain_pct:.1%} and is now {weight:.1%} of your holdings.",
                          "ask": "Does it still deserve this much of your money at today's price?",
                          "against": "Strong companies can keep rising, and selling may trigger tax."})
        if (row.get("qty") or 0) <= 0:
            continue
        history = histories.get(row.get("history_symbol")) or []
        year = [price for _, price in history[-TRADING_DAYS:]]
        age = days_old(history[-1][0], now) if history else None
        if len(year) < TRADING_DAYS // 2 or age is None or age > HISTORY_STALE_DAYS:
            no_history.append(symbol)
            continue
        fall = year[-1] / max(year) - 1
        if fall <= limits["drawdown_review"]:
            items.append({"rule": "drawdown", "symbol": symbol,
                          "why": f"{named} closed {fall:.1%} below its high of the past year on {history[-1][0]}.",
                          "ask": "Is there news that explains the fall, and does it change your view?",
                          "against": ("A fund falling this much usually means its whole market fell, which has happened "
                                      "many times before." if fund else "Large falls are normal for single stocks and often pass.")})
    correlated = health.get("correlation")
    if correlated is not None and correlated >= limits["correlation_review"] and health.get("count", 0) >= 3:
        items.append({"rule": "correlation", "symbol": None,
                      "why": f"Your holdings with enough history have moved closely together (average correlation {correlated:.2f}, recent weekly moves).",
                      "ask": "Are they really different bets, or several versions of one?",
                      "against": "Most stocks rise and fall together in a sharp market move."})
    unchecked = {}
    if no_cost:
        unchecked["gain or loss against cost"] = no_cost
    if no_history:
        unchecked["fall from the past year's high"] = no_history
    return items, unchecked
