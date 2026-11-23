"""The buy advisor (§4VH): how to split a sum the owner is about to invest, by opportunity, not fixed percentages.

Every figure is computed here from plain data, with no network and no clock of its own. The plan it serves:
most of each sum goes to a few core funds with a rough target split; a slice goes to a "hot opportunity" (an
unusually sharp fall in a large, liquid name) or, when nothing qualifies, is kept as cash for the next one.

What "cheap" means here: today compared with the fund's OWN past five years, on three plain measures (fall
from its 52-week high, gap to its 200-day average, last month's return). It describes where the price sits,
not where it goes next: for broad index funds buying dips is a small edge at best. The value is discipline,
putting more in when prices are down instead of chasing.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

from analytics import FUND_KINDS, TRADING_DAYS, days_old, HISTORY_STALE_DAYS

#: A score needs a 52-week high and a 200-day average that themselves have a past to be ranked against.
MIN_SCORE_DAYS = 2 * TRADING_DAYS
MA_DAYS = 200
MONTH = 21
#: A score this far from 0.5 multiplies a fund's share by up to TILT (cheapest) or 1/TILT (dearest).
TILT = 3.0
#: A drifted split is pulled back by sqrt(aim / held), within these bounds.
DRIFT_BOUNDS = (0.5, 2.0)
#: The safe fund takes the whole core sum when it is at least this cheap and the cheapest core fund.
SAFE_ALL_IN = 0.75
#: Fall horizons for the hot rule, in trading days.
HOT_HORIZONS = ((1, "1 day"), (5, "5 days"), (21, "1 month"))
MIN_HOT_DAYS = TRADING_DAYS + MONTH + 1
#: A live quote joins a history only when the history ends at most this many days before it (a weekend + a holiday).
QUOTE_JOIN_DAYS = 4

#: A tech-leaning default watchlist of UCITS funds quoted in EUR on Xetra (Yahoo symbols, checked 2026-11-23: type ETF,
#: currency EUR, five years of history). UCITS because an EU retail account generally can't buy US-domiciled funds
#: (no KID); funds because a 2022-2026 replay of the hot rule on single stocks ended well behind plain 60/30 (§4VH-b).
#: Nasdaq-100, semiconductors, S&P 500 IT, World IT, Europe tech, AI & big data, robotics, Japan, China, Europe 600.
#: Share prices under about 120 EUR, so a capped slice can buy one (iShares' Nasdaq-100 at 1,361 and an AI fund at 187
#: were swapped out or dropped for that).
DEFAULT_WATCHLIST = ("XNAS.DE", "VVSM.DE", "QDVE.DE", "XDWT.DE", "EXV3.DE", "2B76.DE", "EUNN.DE", "XCS6.DE", "EXSA.DE")
#: A quote this far from the history's last close is a split or another listing, not a move: never joined (§4VH-c).
MAX_JUMP = 0.25


def _rank(values: list, today: float) -> float:
    """The share of past values at or below today's: 0 = lowest ever seen, 1 = highest."""
    return sum(1 for v in values if v <= today) / len(values)


def cheapness(prices: list) -> dict | None:
    """Where today's price sits against the series' own past, or None when the series is too short.
    score = 1 - mean(rank of today's fall-from-high, gap-to-average, month return): 1 = cheapest ever seen."""
    if len(prices) < MIN_SCORE_DAYS or any(p <= 0 for p in prices):
        return None
    falls, gaps, months = [], [], []
    first = TRADING_DAYS - 1  # >= MA_DAYS - 1, so every ranked day has a full 200-day average
    running = sum(prices[first - MA_DAYS + 1:first + 1])
    for t in range(first, len(prices)):
        if t > first:
            running += prices[t] - prices[t - MA_DAYS]
        gaps.append(prices[t] / (running / MA_DAYS) - 1)
        falls.append(prices[t] / max(prices[t - TRADING_DAYS + 1:t + 1]) - 1)
        months.append(prices[t] / prices[t - MONTH] - 1)
    fall, gap, month = falls[-1], gaps[-1], months[-1]
    ranks = {"fall": _rank(falls, fall), "gap": _rank(gaps, gap), "month": _rank(months, month)}
    return {"score": 1 - mean(ranks.values()), "fall": fall, "gap": gap, "month": month, "ranks": ranks, "days": len(prices)}


def describe_score(score: float) -> str:
    if score >= 0.8:
        return "clearly cheap against its own past"
    if score >= 0.65:
        return "somewhat cheap"
    if score > 0.35:
        return "about usual"
    if score > 0.2:
        return "somewhat dear"
    return "clearly dear against its own past"


def hot_signal(prices: list) -> dict | None:
    """How unusual the latest fall is for this name: each horizon's return in units of its usual move
    (daily volatility of the past year x sqrt(days)). None when the series is too short or flat."""
    if len(prices) < MIN_HOT_DAYS or any(p <= 0 for p in prices):
        return None
    year = prices[-(TRADING_DAYS + 1):]
    returns = [b / a - 1 for a, b in zip(year, year[1:])]
    sigma = pstdev(returns)
    if sigma <= 0:
        return None
    moves = []
    for days, label in HOT_HORIZONS:
        change = prices[-1] / prices[-1 - days] - 1
        moves.append({"days": days, "label": label, "change": change, "z": change / (sigma * math.sqrt(days))})
    worst = min(moves, key=lambda m: m["z"])
    return {"worst": worst, "moves": moves, "fall": prices[-1] / max(prices[-TRADING_DAYS:]) - 1,
            "volatility": sigma * math.sqrt(TRADING_DAYS)}


def whole_shares(money: dict, prices: dict, budget: float) -> tuple:
    """Whole shares per fund close to the money split, never above the budget. (counts, cash left).
    Each fund rounds to its NEAREST whole count: a share is added only when at least half of it is the fund's own
    money (§4VH-c: topping up with whatever still fit put 92% of a 150 EUR core into the cheaper-per-share fund)."""
    counts = {s: (int(money[s] // prices[s]) if prices.get(s) else 0) for s in money}
    left = budget - sum(counts[s] * prices[s] for s in counts if prices.get(s))
    while True:
        short = [(money[s] - counts[s] * prices[s], s) for s in counts
                 if money[s] > 0 and prices.get(s) and prices[s] <= left + 1e-9
                 and money[s] - counts[s] * prices[s] >= prices[s] / 2]
        if not short:
            break
        _, pick = max(short)
        counts[pick] += 1
        left -= prices[pick]
    return counts, max(left, 0.0)


def core_split(funds: list, max_drift: float) -> dict:
    """Money shares (summing to 1) for the core funds, and the reason for each.
    Each fund: symbol, aim (share of the core, sums to 1), held (current share of the core, or None), score
    (cheapness or None), safe (bool)."""
    reasons, raw = {}, {}
    any_held = any(f["held"] is not None for f in funds)
    for f in funds:
        held = (f["held"] or 0.0) if any_held else f["aim"]
        tilt = TILT ** (2 * (f["score"] - 0.5)) if f["score"] is not None else 1.0
        drift = min(max(math.sqrt(f["aim"] / max(held, 0.01)), DRIFT_BOUNDS[0]), DRIFT_BOUNDS[1])
        raw[f["symbol"]] = f["aim"] * tilt * drift
        f["over"] = held - f["aim"] > max_drift
    capped = [f for f in funds if f["over"]]
    if capped and len(capped) < len(funds):
        for f in capped:
            raw[f["symbol"]] = 0.0
            reasons[f["symbol"]] = (f"left out this time: it is more than {max_drift:.0%} above its aim within the core, "
                                    "so the others catch up first")
    scored = [f for f in funds if f["score"] is not None and not f["over"]]
    safe = next((f for f in scored if f["safe"]), None)
    every_scored = all(f["score"] is not None for f in funds if not f["over"])
    if safe and every_scored and safe["score"] >= SAFE_ALL_IN and all(safe["score"] >= f["score"] for f in scored):
        for f in funds:
            if f is not safe:
                reasons.setdefault(f["symbol"], f"none this time: {safe['symbol']} is the safer fund and at least as cheap")
        reasons[safe["symbol"]] = (f"all of the core: it is the safer fund, {describe_score(safe['score'])} "
                                   f"(score {safe['score']:.2f} of 1), and no core fund that can be bought this time is cheaper")
        return {"shares": {f["symbol"]: (1.0 if f is safe else 0.0) for f in funds}, "reasons": reasons, "rule": "safe"}
    total = sum(raw.values())
    if total <= 0:
        return {"shares": {f["symbol"]: f["aim"] for f in funds}, "reasons": reasons, "rule": "aim"}
    return {"shares": {s: v / total for s, v in raw.items()}, "reasons": reasons, "rule": "tilt"}


def advise(amount: float, reserve: float, core: list, rows: dict, histories: dict, watch: list, quotes: dict,
           fx_to_base, base: str, settings: dict, now: float, holdings_known: bool = True, carry: float = 0.0) -> dict:
    """The full advice as data. `core`: [{symbol, yahoo, target, safe}]. `rows`: core symbol -> the valued holding
    row, if held. `watch`: Yahoo symbols. `quotes`: Yahoo symbol -> latest quote. `fx_to_base(currency)` -> rate
    or None. `settings`: opportunity (fraction), max_drift (fraction), hot_z (negative), hot_fall (negative),
    funds_only (bool), cap_slices (how many slices of opportunity cash may wait; above that it joins the core)."""
    gaps = []
    opportunity_share = settings["opportunity"]
    core_money = round(amount * (1 - opportunity_share), 2)
    slice_money = round(amount - core_money, 2)
    # §4VH-b: waiting cash is capped. Uncapped, 2022-2026 piled up EUR 950 and then put it into one falling name.
    cap = round(slice_money * settings.get("cap_slices", 3), 2)
    overflow = round(max(0.0, slice_money + reserve - cap), 2)
    # §4VH-f: last time's whole-share leftover is the core's: it joins the core directly, none of it goes to the slice
    core_money = round(core_money + overflow + carry, 2)

    jumps = []

    def series(symbol: str, latest: tuple | None = None) -> tuple:
        """(closes, last date). None for the date when the history is missing or stale on its own (a fresh quote
        on a week-old history would make its "1 day" move a week's), or when the quote jumps more than MAX_JUMP
        from the last close (a split or another listing would read as a crash, §4VH-c). The quote REPLACES any
        history bar of its own session or later: Yahoo's daily series carries today's live bar, not a close."""
        history = list(histories.get(symbol) or [])
        if not history or (days_old(history[-1][0], now) or 0) > HISTORY_STALE_DAYS:
            return [], None
        if latest and latest[0] and latest[1]:
            before = [h for h in history if h[0] < latest[0]]
            if before and days_old(before[-1][0], now) - (days_old(latest[0], now) or 0) <= QUOTE_JOIN_DAYS:
                if abs(latest[1] / before[-1][1] - 1) > MAX_JUMP:
                    jumps.append(symbol)
                    return [], None
                history = before + [latest]
        return [p for _, p in history], history[-1][0]

    # ---- the core
    total_aim = sum(c["target"] for c in core)
    held_values = {c["symbol"]: (rows.get(c["symbol"]) or {}).get("value_base") for c in core}
    unvalued = [c["symbol"] for c in core if c["symbol"] in rows and held_values[c["symbol"]] is None]
    if not holdings_known:
        gaps.append("your holdings are unknown, so the split ignores how far they have drifted from the aim")
    elif unvalued:
        gaps.append(f"{', '.join(unvalued)}: held but without a value, so the split ignores drift for every core fund")
    drift_known = holdings_known and not unvalued
    held_total = sum(v for v in held_values.values() if v) if drift_known else 0.0
    funds, prices_base, info = [], {}, {}
    for c in core:
        row = rows.get(c["symbol"]) or {}
        yahoo = row.get("history_symbol") or c.get("yahoo")
        quote = quotes.get(yahoo) or {}
        if row.get("price") is not None and row.get("fx"):
            price_local, currency, rate = row["price"], row.get("currency"), row["fx"]
        elif quote.get("price"):
            price_local, currency = quote["price"], quote.get("currency")
            rate = fx_to_base(currency)
        else:
            price_local = currency = rate = None
        latest = (quote.get("session"), quote.get("price")) if quote.get("currency") == currency else None
        closes, last = series(yahoo, latest) if yahoo else ([], None)
        stale = last is None
        score = None if stale else cheapness(closes)
        if score is None:
            gaps.append(f"{c['symbol']}: no score, " + (
                "its price moved more than 25% from its last close (a split or another listing?), so it is left out until "
                "the history refreshes" if yahoo in jumps else
                "its price history is missing or stale" if stale else "its price history is shorter than two years"))
        price_base = price_local * rate if price_local and rate else None
        if price_base is None:
            gaps.append(f"{c['symbol']}: no price in {base}, so no share count")
        prices_base[c["symbol"]] = price_base
        held = (held_values[c["symbol"]] or 0.0) / held_total if held_total else None  # None: nothing held, or unknown
        info[c["symbol"]] = {"yahoo": yahoo, "price_local": price_local, "currency": currency, "price_base": price_base,
                             "score": score, "held": held, "aim": c["target"] / total_aim,
                             "name": row.get("label") or (quote.get("name") and f"{c['symbol']} ({quote['name']})") or c["symbol"],
                             "last": last}
        funds.append({"symbol": c["symbol"], "aim": c["target"] / total_aim, "held": held,
                      "score": score["score"] if score else None, "safe": bool(c.get("safe"))})
    split = core_split(funds, settings["max_drift"])
    money = {s: core_money * share for s, share in split["shares"].items()}
    priced = {s: p for s, p in prices_base.items() if p}
    counts, left = whole_shares({s: money[s] for s in priced}, priced, sum(money[s] for s in priced))
    unplaced = round(sum(money[s] for s in money if s not in priced), 2)  # no price: no buy, and not "left over" either

    # ---- the opportunity slice: the capped slice. The core's leftover is the core's, carried to the next sum.
    pot = round(slice_money + reserve - overflow, 2)
    candidates, checked, unusable, not_fund = [], 0, [], []
    core_yahoo = {(rows.get(c["symbol"]) or {}).get("history_symbol") or c.get("yahoo") for c in core}
    for symbol in list(dict.fromkeys([*watch, *[y for y in core_yahoo if y]])):
        quote = quotes.get(symbol) or {}
        closes, last = series(symbol, (quote.get("session"), quote.get("price")) if quote.get("price") else None)
        if last is None or (days_old(last, now) or 0) > QUOTE_JOIN_DAYS:  # "hot now" needs prices from this week
            unusable.append(symbol)
            continue
        signal = hot_signal(closes)
        if signal is None:
            unusable.append(symbol)
            continue
        if not quote and symbol not in core_yahoo:  # no price yet: its type is unknown too, but that isn't why
            unusable.append(symbol)
            continue
        is_fund = symbol in core_yahoo or quote.get("kind") in FUND_KINDS
        if settings.get("funds_only", True) and not is_fund:  # unknown type counts as not a fund
            not_fund.append(symbol)
            continue
        checked += 1
        hot = signal["worst"]["z"] <= settings["hot_z"] and signal["fall"] <= settings["hot_fall"]
        # the live quote when there is one: the history's last close can be a day behind it
        price, currency = (quote["price"], quote.get("currency")) if quote.get("price") else (closes[-1], None)
        rate = fx_to_base(currency) if currency else None
        candidates.append({"symbol": symbol, "name": quote.get("name") or "", "hot": hot, "price": price,
                           "currency": currency, "price_base": price * rate if rate else None, "last": last, **signal})
    candidates.sort(key=lambda c: c["worst"]["z"])
    hot = [c for c in candidates if c["hot"]]
    # whole shares: European funds generally can't be bought in fractions. The most unusual fall the pot can buy.
    buy = next((c for c in hot if c["price_base"] and c["price_base"] <= pot), None)
    buy_shares = int(pot // buy["price_base"]) if buy else 0
    buy_cost = round(buy_shares * buy["price_base"], 2) if buy else 0.0
    return {"amount": amount, "reserve": reserve, "base": base, "core_money": core_money, "slice_money": slice_money,
            "opportunity_share": opportunity_share, "funds": info, "split": split, "money": money, "counts": counts,
            "left": left, "unplaced": unplaced, "carry": carry, "pot": pot, "buy": buy, "buy_shares": buy_shares, "buy_cost": buy_cost, "jumps": jumps, "cap": cap, "overflow": overflow, "not_fund": not_fund, "hot": hot, "near": [c for c in candidates if not c["hot"]][:3],
            "checked": checked, "drift_known": drift_known, "unusable": unusable, "gaps": gaps, "settings": settings}
