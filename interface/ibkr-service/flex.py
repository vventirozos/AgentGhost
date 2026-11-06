"""IBKR Flex Web Service: download one statement and reduce it to plain data.

Read only by construction: Flex can only hand out reports. The account id is used only to tell statements
apart while parsing and is never copied out, and the raw XML is never written to disk.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Callable

BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

#: The statement is still being generated, or IBKR asks us to slow down. Worth asking again.
RETRY_CODES = frozenset({"1001", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1018", "1019", "1021"})
#: What the owner has to do when IBKR refuses for good. Shown as is in the service status.
ADVICE = {
    "1012": "the Flex token expired: generate a new one in IBKR and run setup-flex.sh",
    "1013": "the token has an IP restriction that excludes this machine",
    "1014": "the query id is not valid: check it in IBKR and run setup-flex.sh",
    "1015": "the Flex token is not valid: run setup-flex.sh",
    "1020": "IBKR refused the token and query id: run setup-flex.sh",
}
TRADES_PER_SYMBOL = 5
#: Holdings plus cash may differ from IBKR's own account value by this much before the service says so.
NAV_TOLERANCE = 0.02
#: A position value here is a contract's size, not money the account holds, so it stays out of every total.
NOTIONAL_ASSETS = frozenset({"FUT", "CFD"})


class FlexError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.retryable = code, message, retryable


def _xml(text: str) -> ET.Element:
    if "<!DOCTYPE" in text[:2000].upper():  # no entity tricks: Flex never sends a doctype
        raise FlexError("parse", "unexpected DOCTYPE in the Flex response")
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise FlexError("parse", f"the Flex response is not XML ({exc})", retryable=True) from None


def _failure(root: ET.Element) -> FlexError:
    code = (root.findtext("ErrorCode") or "unknown").strip()
    message = ADVICE.get(code) or (root.findtext("ErrorMessage") or "no message").strip()
    return FlexError(code, message, retryable=code in RETRY_CODES)


def fetch_statement(token: str, query_id: str, get: Callable[[str, dict], str],
                    sleep: Callable[[float], None] = time.sleep, polls: int = 12, wait: float = 10.0) -> str:
    """SendRequest, then poll GetStatement until the statement is ready. `get(url, params)` returns text."""
    root = _xml(get(f"{BASE_URL}/SendRequest", {"t": token, "q": query_id, "v": "3"}))
    if (root.findtext("Status") or "").strip() != "Success":
        raise _failure(root)
    reference = (root.findtext("ReferenceCode") or "").strip()
    if not reference:
        raise FlexError("parse", "IBKR returned no reference code", retryable=True)
    last = FlexError("timeout", "the statement was not ready in time", retryable=True)
    for attempt in range(polls):
        if attempt:
            sleep(wait)
        text = get(f"{BASE_URL}/GetStatement", {"t": token, "q": reference, "v": "3"})
        root = _xml(text)
        if root.tag == "FlexQueryResponse":
            return text
        last = _failure(root)
        if not last.retryable:
            raise last
    raise last


def _num(value) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _date(value) -> str:
    """ISO date from IBKR's 20261106, 20261106;093000 or 2026-11-06; "" for any other format."""
    match = re.match(r"(\d{4})-?(\d{2})-?(\d{2})(?:\D|$)", str(value or "").strip())
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def _position_key(el: ET.Element) -> str:
    return el.get("conid") or f"{el.get('symbol')}|{el.get('currency')}"


def _merge_position(merged: dict, el: ET.Element) -> None:
    get = el.get
    key = _position_key(el)
    row = merged.get(key)
    asset = get("assetCategory") or ""
    if row is None:
        merged[key] = {
            "key": key, "symbol": (get("symbol") or "").strip(), "description": (get("description") or "").strip(),
            "asset": asset, "currency": get("currency") or "", "isin": get("isin") or "",
            "exchange": get("listingExchange") or "", "qty": _num(get("position")), "mark": _num(get("markPrice")),
            "position_value": _num(get("positionValue")),
            "cost": _num(get("costBasisMoney")), "cost_price": _num(get("costBasisPrice")),
            "fx_to_base": _num(get("fxRateToBase")), "report_date": _date(get("reportDate")),
        }
        return
    # The same holding in a second account: quantities and cost add up; a missing number stays unknown.
    for field, attr in (("qty", "position"), ("cost", "costBasisMoney"), ("position_value", "positionValue")):
        value = _num(get(attr))
        row[field] = None if row[field] is None or value is None else row[field] + value
    row["cost_price"] = row["cost"] / row["qty"] if asset == "STK" and row["qty"] and row["cost"] is not None else None


def _nav_gap(rows: list, cash, nav: list, mixed: bool, positions_known: bool = True) -> float | None:
    """The statement checked against itself: its holdings and cash, as a fraction away from its account value.
    None when a figure needed for the check is missing."""
    if mixed or not positions_known or cash is None or not nav or not nav[-1][1]:
        return None
    if any(row["report_date"] and row["report_date"] != nav[-1][0] for row in rows):
        return None  # holdings of one day against the account value of another prove nothing
    total = cash
    for row in rows:
        if row["asset"] in NOTIONAL_ASSETS:
            continue
        local = row["qty"] * row["mark"] if row["asset"] == "STK" and row["qty"] is not None and row["mark"] is not None else row["position_value"]
        if local is None or row["fx_to_base"] is None:
            return None
        total += local * row["fx_to_base"]
    return total / nav[-1][1] - 1


def parse_statement(text: str) -> dict:
    """Plain data from a Flex XML statement, plus `warnings` naming what the query does not include."""
    root = _xml(text)
    latest: dict = {}
    every = list(root.iter("FlexStatement"))
    for statement in every:
        account = statement.get("accountId") or str(id(statement))
        if account not in latest or _date(statement.get("toDate")) > _date(latest[account].get("toDate")):
            latest[account] = statement
    if not latest:
        raise FlexError("parse", "the Flex response holds no statement")
    statements = list(latest.values())
    warnings = []
    if len(every) > len(statements):
        warnings.append("the Flex query returned several periods for one account, so only the latest is used")
    report_date = max((_date(st.get("toDate")) for st in statements), default="")
    if not report_date:
        warnings.append("the statement's dates aren't readable: set the query's Date Format to yyyyMMdd")
    year = report_date[:4]

    positions, nav, trades, bases, doubled = {}, {}, [], set(), set()
    cash, cash_accounts, nav_days, dividends, withheld = None, 0, {}, 0.0, 0.0
    dividends_known, starts = bool(report_date), []
    for statement in statements:
        base = next((el.get("currency") for el in statement.iter("AccountInformation") if el.get("currency")), None)
        if base:
            bases.add(base)
        starts.append(_date(statement.get("fromDate")))
        rows = list(statement.iter("OpenPosition"))
        latest_row = max((_date(el.get("reportDate")) for el in rows), default="")
        rows = [el for el in rows if _date(el.get("reportDate")) in ("", latest_row)]
        has_summary = any(el.get("levelOfDetail", "").upper() == "SUMMARY" for el in rows)
        seen = set()
        for el in rows:
            if has_summary and el.get("levelOfDetail", "").upper() != "SUMMARY":
                continue
            key = _position_key(el)
            if key in seen:  # lots without a Level Of Detail field: adding them would double the holding
                doubled.add(key)
                continue
            seen.add(key)
            _merge_position(positions, el)
        for el in statement.iter("EquitySummaryByReportDateInBase"):
            day, total = _date(el.get("reportDate")), _num(el.get("total"))
            if day and total is not None:
                nav[day] = nav.get(day, 0.0) + total
                nav_days[day] = nav_days.get(day, 0) + 1
        for el in statement.iter("CashReportCurrency"):
            if el.get("currency") == "BASE_SUMMARY" and _num(el.get("endingCash")) is not None:
                cash = (cash or 0.0) + _num(el.get("endingCash"))
                cash_accounts += 1
        for el in statement.iter("CashTransaction"):
            amount, rate = _num(el.get("amount")), _num(el.get("fxRateToBase"))
            kind = (el.get("type") or "").lower()
            if el.get("currency") == "BASE_SUMMARY":
                continue
            if (el.get("levelOfDetail") or "DETAIL").upper() != "DETAIL" or not kind:
                dividends_known = False  # a summary row or an unnamed one can't be counted, and can't be ignored
                continue
            is_dividend, is_withholding = "dividend" in kind or "payment in lieu" in kind, "withholding" in kind
            if not (is_dividend or is_withholding):
                continue
            day = _date(el.get("dateTime") or el.get("reportDate"))
            if rate is None and el.get("currency") == base:
                rate = 1.0
            if amount is None or rate is None or not day:
                dividends_known = False
                continue
            if day[:4] != year:
                continue
            if is_dividend:
                dividends += amount * rate
            else:
                withheld += amount * rate
        for el in statement.iter("Trade"):
            if (el.get("levelOfDetail") or "EXECUTION").upper() == "EXECUTION":
                trades.append({"date": _date(el.get("tradeDate")), "symbol": el.get("symbol") or "",
                               "side": el.get("buySell") or "", "qty": _num(el.get("quantity")),
                               "price": _num(el.get("tradePrice")), "currency": el.get("currency") or ""})

    for key in doubled:
        positions[key].update(qty=None, cost=None, cost_price=None, position_value=None)
    trades.sort(key=lambda t: t["date"], reverse=True)
    kept, counts = [], {}
    for trade in trades:
        counts[trade["symbol"]] = counts.get(trade["symbol"], 0) + 1
        if counts[trade["symbol"]] <= TRADES_PER_SYMBOL:
            kept.append(trade)

    rows = list(positions.values())
    mixed = len(bases) > 1
    has_cash_rows = root.find(".//CashTransactions") is not None
    if cash_accounts != len(statements):
        cash = None  # one account's cash is not the account total
    nav = {day: total for day, total in nav.items() if nav_days[day] == len(statements)}
    positions_known = root.find(".//OpenPositions") is not None
    if not positions_known:
        warnings.append("the Flex query has no Open Positions section")
    if doubled:
        warnings.append("Open Positions repeats a holding: select Summary and add the Level Of Detail field. Those quantities are left unknown")
    if any(r["cost"] is None and r["key"] not in doubled for r in rows):
        warnings.append("the Flex query lacks Cost Basis Money in Open Positions, so some gains are unknown")
    if any(r["mark"] is None for r in rows):
        warnings.append("the Flex query lacks Mark Price in Open Positions")
    if any(r["asset"] != "STK" and r["asset"] not in NOTIONAL_ASSETS and r["position_value"] is None and r["key"] not in doubled for r in rows):
        warnings.append("the Flex query lacks Position Value in Open Positions, so assets other than shares can't be valued")
    if not bases:
        warnings.append("the Flex query lacks Account Information > Currency (base_currency in config.json also works)")
    if mixed:
        warnings.append("the accounts use different base currencies, so cash, account value and dividends are left out")
    if cash is None:
        warnings.append("the Flex query lacks Cash Report (Base Summary, Ending Cash), so cash is left out")
    if not nav:
        warnings.append("the Flex query lacks Net Asset Value (NAV) in Base")
    if not has_cash_rows:
        warnings.append("the Flex query has no Cash Transactions section, so dividends are left out")
    elif not dividends_known and not mixed:
        warnings.append("some dividend rows can't be read (select Detail, with Type, Date/Time, Amount and FX Rate To Base), so the dividend total is left out")
    gap = _nav_gap(rows, cash, sorted(nav.items()), mixed, positions_known)
    if gap is not None and abs(gap) > NAV_TOLERANCE:
        warnings.append(f"holdings plus cash differ from IBKR's own account value by {gap:+.1%}, so treat the totals with caution (accrued dividends and interest can explain a small gap)")
    usable = has_cash_rows and dividends_known and not mixed
    return {
        "positions_known": positions_known, "report_date": report_date, "base": sorted(bases)[0] if len(bases) == 1 else None, "mixed_base": mixed,
        "positions": rows, "cash": None if mixed else cash, "nav": [] if mixed else sorted(nav.items()),
        "dividends_ytd": dividends if usable else None, "withholding_ytd": withheld if usable else None,
        "dividends_from": max(f"{year}-01-01", min(starts, default="")), "trades": kept, "warnings": warnings,
    }
