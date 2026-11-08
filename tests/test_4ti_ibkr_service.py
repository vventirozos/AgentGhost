"""§4TI: the IBKR service and the agent's `portfolio` tool. Each test names a world where it would fail.
All data is fictional. The service modules are imported from interface/ibkr-service with their real code."""
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1] / "interface" / "ibkr-service"
sys.path.insert(0, str(SERVICE_DIR))

import analytics  # noqa: E402
import flex  # noqa: E402
import market  # noqa: E402
import service as svc  # noqa: E402

ACCOUNT = "U9999999"
STATEMENT = f"""<FlexQueryResponse queryName="q" type="AF"><FlexStatements count="1">
<FlexStatement accountId="{ACCOUNT}" fromDate="20260101" toDate="20261105">
<AccountInformation accountId="{ACCOUNT}" currency="EUR"/>
<EquitySummaryInBase>
 <EquitySummaryByReportDateInBase reportDate="20260101" total="9000"/>
 <EquitySummaryByReportDateInBase reportDate="20261105" total="12000"/>
</EquitySummaryInBase>
<CashReport><CashReportCurrency currency="BASE_SUMMARY" endingCash="500"/><CashReportCurrency currency="USD" endingCash="99999"/></CashReport>
<OpenPositions>
 <OpenPosition accountId="{ACCOUNT}" assetCategory="STK" symbol="ACME" description="ACME CORP" conid="1" isin="US0000000001" currency="USD" listingExchange="NASDAQ" position="10" markPrice="100" costBasisMoney="800" costBasisPrice="80" fxRateToBase="0.9" reportDate="20261105" levelOfDetail="SUMMARY"/>
 <OpenPosition accountId="{ACCOUNT}" assetCategory="STK" symbol="ACME" conid="1" currency="USD" position="10" levelOfDetail="LOT"/>
 <OpenPosition accountId="{ACCOUNT}" assetCategory="STK" symbol="WRLD" description="WORLD ETF" conid="2" currency="EUR" listingExchange="IBIS2" position="20" markPrice="50" costBasisMoney="1500" costBasisPrice="75" fxRateToBase="1" reportDate="20261105" levelOfDetail="SUMMARY"/>
 <OpenPosition accountId="{ACCOUNT}" assetCategory="STK" symbol="ODD" description="ODD LISTING" conid="3" currency="EUR" listingExchange="NOWHERE" position="1" markPrice="10" costBasisMoney="10" fxRateToBase="1" reportDate="20261105" levelOfDetail="SUMMARY"/>
</OpenPositions>
<CashTransactions>
 <CashTransaction type="Dividends" currency="USD" amount="10" fxRateToBase="0.9" dateTime="20260301;000000" levelOfDetail="DETAIL"/>
 <CashTransaction type="Withholding Tax" currency="USD" amount="-1.5" fxRateToBase="0.9" dateTime="20260301;000000" levelOfDetail="DETAIL"/>
 <CashTransaction type="Dividends" currency="USD" amount="500" fxRateToBase="0.9" dateTime="20250301;000000" levelOfDetail="DETAIL"/>
 <CashTransaction type="Dividends" currency="BASE_SUMMARY" amount="777" fxRateToBase="1" dateTime="20260301;000000" levelOfDetail="DETAIL"/>
</CashTransactions>
<Trades><Trade symbol="ACME" tradeDate="20260201" buySell="BUY" quantity="10" tradePrice="80" currency="USD" ibCommission="-1" levelOfDetail="EXECUTION"/></Trades>
</FlexStatement></FlexStatements></FlexQueryResponse>"""


def _reply(status, **fields):
    inner = "".join(f"<{k}>{v}</{k}>" for k, v in fields.items())
    return f"<FlexStatementResponse><Status>{status}</Status>{inner}</FlexStatementResponse>"


NOW = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc).timestamp()


def _chart(price, currency="USD", closes=(90.0, 100.0), end=NOW, offset=0):
    stamps = [int(end - (len(closes) - 1 - i) * 86400) for i in range(len(closes))]
    return json.dumps({"chart": {"result": [{
        "meta": {"regularMarketPrice": price, "currency": currency, "regularMarketTime": int(end), "gmtoffset": offset},
        "timestamp": stamps,
        "indicators": {"quote": [{"close": list(closes)}], "adjclose": [{"adjclose": list(closes)}]},
    }]}})


class FakeWeb:
    """IBKR and Yahoo with scripted answers. Records every request."""

    def __init__(self, statement=STATEMENT):
        self.statement, self.calls, self.prices = statement, [], {
            "ACME": (110.0, "USD"), "WRLD.DE": (55.0, "EUR"), "USDEUR=X": (0.8, "EUR"), "VT": (100.0, "USD")}
        self.history = None

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        if url.endswith("/SendRequest"):
            return _reply("Success", ReferenceCode="REF1", Url="x")
        if url.endswith("/GetStatement"):
            return self.statement
        name = url.rsplit("/", 1)[1]
        if name not in self.prices:
            raise OSError("HTTP 404")
        price, currency = self.prices[name]
        closes = self.history.get(name) if self.history and params.get("range") == "5y" else None
        return _chart(price, currency, closes=closes or (price * 0.9, price))


@pytest.fixture
def world(tmp_path):
    web = FakeWeb()
    token = tmp_path / "token"
    token.write_text("SECRET-TOKEN-123\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"flex_query_id": "4242"}))
    clock = {"now": NOW}
    service = svc.Service(home, token, get=web, clock=lambda: clock["now"], sleep=lambda _s: None)
    return service, web, clock, home


# ---------------------------------------------------------------- Flex protocol

def test_flex_polls_until_ready_and_uses_the_reference_code():
    answers = [_reply("Success", ReferenceCode="REF7"), _reply("Warn", ErrorCode="1019", ErrorMessage="wait"), STATEMENT]
    seen = []

    def get(url, params):
        seen.append((url.rsplit("/", 1)[1], params["q"]))
        return answers.pop(0)
    assert flex.fetch_statement("tok", "55", get, sleep=lambda _s: None) == STATEMENT
    assert seen == [("SendRequest", "55"), ("GetStatement", "REF7"), ("GetStatement", "REF7")]


def test_flex_real_refusal_stops_at_once_and_gives_advice():
    calls = []

    def get(url, params):
        calls.append(url)
        return _reply("Fail", ErrorCode="1012", ErrorMessage="Token has expired.") if len(calls) == 2 else _reply("Success", ReferenceCode="R")
    with pytest.raises(flex.FlexError) as caught:
        flex.fetch_statement("tok", "55", get, sleep=lambda _s: None)
    assert caught.value.code == "1012" and "setup-flex.sh" in caught.value.message and len(calls) == 2


def test_flex_gives_up_after_its_poll_limit():
    calls = []

    def get(url, params):
        calls.append(url)
        return _reply("Success", ReferenceCode="R") if len(calls) == 1 else _reply("Warn", ErrorCode="1019", ErrorMessage="wait")
    with pytest.raises(flex.FlexError):
        flex.fetch_statement("tok", "55", get, sleep=lambda _s: None, polls=3)
    assert len(calls) == 4


def test_doctype_is_refused():
    with pytest.raises(flex.FlexError):
        flex.parse_statement('<!DOCTYPE x [<!ENTITY a "b">]><FlexQueryResponse/>')


# ---------------------------------------------------------------- parsing

def test_parse_uses_summary_rows_and_never_keeps_the_account_id():
    parsed = flex.parse_statement(STATEMENT)
    acme = next(p for p in parsed["positions"] if p["symbol"] == "ACME")
    assert len(parsed["positions"]) == 3 and acme["qty"] == 10  # a counted LOT row would make 20
    assert parsed["base"] == "EUR" and parsed["cash"] == 500  # the USD row is not the base summary
    assert parsed["nav"][-1] == ("2026-11-05", 12000.0)
    assert ACCOUNT not in json.dumps(parsed)


def test_dividends_count_this_year_only_and_skip_summary_rows():
    parsed = flex.parse_statement(STATEMENT)
    assert parsed["dividends_ytd"] == pytest.approx(9.0)
    assert parsed["withholding_ytd"] == pytest.approx(-1.35)


def test_same_holding_in_two_accounts_adds_up():
    second = """<FlexStatement accountId="U2" toDate="20261105"><AccountInformation currency="EUR"/><OpenPositions><OpenPosition assetCategory="STK" symbol="ACME" conid="1" currency="USD" position="5" markPrice="100" costBasisMoney="500" levelOfDetail="SUMMARY"/></OpenPositions></FlexStatement></FlexStatements>"""
    two = STATEMENT.replace("</FlexStatements>", second)
    acme = next(p for p in flex.parse_statement(two)["positions"] if p["symbol"] == "ACME")
    assert acme["qty"] == 15 and acme["cost"] == 1300 and acme["cost_price"] == pytest.approx(1300 / 15)


def test_a_query_missing_a_field_says_so():
    bare = STATEMENT.replace('costBasisMoney="800" ', "").replace('endingCash="500"', "")
    warnings = " ".join(flex.parse_statement(bare)["warnings"])
    assert "Cost Basis Money" in warnings and "Cash Report" in warnings


# ---------------------------------------------------------------- market

@pytest.mark.parametrize("exchange,symbol,expected", [
    ("NASDAQ", "ACME", "ACME"), ("IBIS2", "WRLD", "WRLD.DE"), ("LSE", "ABC", "ABC.L"),
    ("NYSE", "BRK B", "BRK-B"), ("NOWHERE", "ODD", None),
])
def test_yahoo_symbol(exchange, symbol, expected):
    assert market.yahoo_symbol({"asset": "STK", "symbol": symbol, "exchange": exchange}) == expected


def test_only_stocks_get_public_prices_and_overrides_apply_to_them():
    assert market.yahoo_symbol({"asset": "OPT", "symbol": "ACME", "exchange": "NASDAQ"}, {"ACME": "ACME.X"}) is None
    assert market.yahoo_symbol({"asset": "STK", "symbol": "ACME", "exchange": "NOWHERE"}, {"ACME": "ACME.X"}) == "ACME.X"


def test_pence_quotes_become_pounds():
    quote = market.fetch_chart("ABC.L", lambda u, p: _chart(1000.0, "GBp", closes=(900.0, 1000.0)), json.loads)
    assert quote["price"] == 10.0 and quote["currency"] == "GBP" and quote["previous_close"] == 9.0


def test_previous_close_is_not_the_latest_close():
    quote = market.fetch_chart("A", lambda u, p: _chart(100.0, closes=(80.0, 90.0, 100.0)), json.loads)
    assert quote["previous_close"] == 90.0


def test_previous_close_is_the_session_before_when_today_has_no_bar():
    def get(url, params):
        body = json.loads(_chart(100.0, closes=(80.0, 90.0), end=NOW - 86400))
        body["chart"]["result"][0]["meta"]["regularMarketTime"] = int(NOW)
        return json.dumps(body)
    assert market.fetch_chart("A", get, json.loads)["previous_close"] == 90.0


def test_dates_are_the_exchanges_own():
    late_utc = datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc).timestamp()
    quote = market.fetch_chart("A.AX", lambda u, p: _chart(10.0, "AUD", closes=(9.0, 10.0), end=late_utc, offset=39600), json.loads)
    assert quote["session"] == "2026-11-06" and quote["closes"][-1][0] == "2026-11-06"


def test_hong_kong_symbols_are_padded():
    assert market.yahoo_symbol({"asset": "STK", "symbol": "700", "exchange": "SEHK"}) == "0700.HK"


@pytest.mark.parametrize("quote,bad", [
    ({"price": 105.0, "currency": "USD"}, False),
    ({"price": 105.0, "currency": "EUR"}, True),
    ({"price": 130.0, "currency": "USD"}, True),
])
def test_wrong_listing_is_caught(quote, bad):
    position = {"currency": "USD", "mark": 100.0}
    assert bool(market.quote_problem(position, quote)) is bad


# ---------------------------------------------------------------- valuation

def _value(service):
    service.tick()
    return service.valued(service.config())


def _row(valued, symbol):
    return next(r for r in valued["positions"] if r["symbol"] == symbol)


def test_values_use_the_public_price_and_the_public_fx_rate(world):
    service, *_ = world
    valued = _value(service)
    acme = _row(valued, "ACME")
    # The statement's rate is 0.9 and the public one 0.8: only the public one gives 880.
    assert acme["value_local"] == 1100 and acme["value_base"] == pytest.approx(880)
    assert acme["gain_local"] == 300 and acme["gain_pct"] == pytest.approx(0.375)
    assert valued["invested"] == pytest.approx(880 + 1100 + 10)
    assert valued["total"] == pytest.approx(880 + 1100 + 10 + 500)
    assert sum(r["weight"] for r in valued["positions"]) == pytest.approx(1)


def test_unmapped_holding_keeps_ibkr_price_and_says_so(world):
    service, *_ = world
    odd = _row(_value(service), "ODD")
    assert odd["source"] == "IBKR" and odd["price"] == 10 and "no public listing" in odd["price_note"]
    assert odd["history_symbol"] is None


def test_a_wrong_listing_is_used_neither_for_price_nor_for_history(world):
    service, web, *_ = world
    web.prices["ACME"] = (500.0, "USD")
    acme = _row(_value(service), "ACME")
    assert acme["price"] == 100 and acme["source"] == "IBKR" and acme["history_symbol"] is None
    service.histories["ACME"] = [(f"2026-{i:04d}", 100.0 + i) for i in range(600)]
    assert "not enough price history" in service.answer("forecast", "ACME")[1]


def test_missing_fx_uses_the_statement_rate_and_says_so(world):
    service, web, *_ = world
    del web.prices["USDEUR=X"]
    acme = _row(_value(service), "ACME")
    assert acme["fx"] == 0.9 and acme["fx_note"]


@pytest.mark.parametrize("entry", [
    {"rate": 0.8, "at": NOW - analytics.PRICE_STALE - 1},  # too old
    {"rate": 0.5, "at": NOW},  # far from the statement's 0.9
])
def test_a_doubtful_fx_rate_gives_way_to_the_statement(world, entry):
    service, *_ = world
    service.tick()
    service.state["fx"]["USDEUR"] = entry
    assert _row(service.valued(service.config()), "ACME")["fx"] == 0.9


def test_an_unvaluable_holding_is_named_and_kept_out_of_the_total(world):
    service, web, *_ = world
    web.statement = STATEMENT.replace('position="1" markPrice="10"', 'position="1"')
    service.tick()
    ok, text = service.answer("summary")
    assert ok and "PARTIAL: leaves out ODD" in text
    assert service.valued(service.config())["invested"] == pytest.approx(1980)


def test_short_position_gain_and_weight():
    data = {"positions": [{"key": "1", "symbol": "S", "qty": -10, "mark": 90.0, "cost": -1000.0, "currency": "EUR", "asset": "STK"}]}
    row = analytics.value_positions(data, {}, {}, {}, "EUR", lambda p, q: None, NOW)["positions"][0]
    assert row["value_local"] == -900 and row["gain_local"] == 100 and row["weight"] == 1


def _other(asset="OPT", **extra):
    return {"key": "9", "symbol": "X", "qty": 2.0, "mark": 5.0, "cost": 800.0, "currency": "EUR", "asset": asset, "position_value": 1000.0, **extra}


def test_non_stock_uses_ibkr_position_value_never_quantity_times_price():
    row = analytics.value_positions({"positions": [_other()]}, {}, {}, {}, "EUR", lambda p, q: None, NOW)["positions"][0]
    assert row["value_local"] == 1000 and row["gain_local"] == 200 and row["gain_pct"] == pytest.approx(0.25)
    bond = _other("BOND", qty=10000.0, mark=98.5, cost=9700.0, position_value=9850.0)
    assert analytics.value_positions({"positions": [bond]}, {}, {}, {}, "EUR", lambda p, q: None, NOW)["invested"] == 9850


def test_a_non_stock_without_a_position_value_is_unknown_not_guessed():
    valued = analytics.value_positions({"positions": [_other(position_value=None)]}, {}, {}, {}, "EUR", lambda p, q: None, NOW)
    assert valued["incomplete"] == ["X"] and valued["invested"] is None


@pytest.mark.parametrize("asset", ["FUT", "CFD"])
def test_contract_size_is_never_money_held(asset):
    fut = _other(asset, qty=1.0, mark=6000.0, position_value=300000.0)
    stock = {"key": "1", "symbol": "S", "qty": 10.0, "mark": 100.0, "cost": 900.0, "currency": "EUR", "asset": "STK"}
    valued = analytics.value_positions({"positions": [fut, stock], "cash": 500.0}, {}, {}, {}, "EUR", lambda p, q: None, NOW)
    assert valued["invested"] == 1000 and valued["total"] == 1500 and valued["notional"] == ["X"]
    assert next(r for r in valued["positions"] if r["symbol"] == "S")["weight"] == 1


def test_a_statement_that_disagrees_with_itself_says_so():
    clean = flex.parse_statement(STATEMENT.replace('total="12000"', 'total="2910"'))  # 1000*0.9 + 1000 + 10 + 500 = 2410
    assert any("account value" in w for w in clean["warnings"])
    right = flex.parse_statement(STATEMENT.replace('total="12000"', 'total="2410"'))
    assert not any("account value" in w for w in right["warnings"])


def test_no_positions_section_is_unknown_but_an_empty_one_is_zero():
    i, j = STATEMENT.index("<OpenPositions>"), STATEMENT.index("</OpenPositions>") + len("</OpenPositions>")
    absent = flex.parse_statement(STATEMENT[:i] + STATEMENT[j:])
    empty = flex.parse_statement(STATEMENT[:i] + "<OpenPositions/>" + STATEMENT[j:])
    assert analytics.value_positions(absent, {}, {}, {}, "EUR", lambda p, q: None, NOW)["invested"] is None
    assert analytics.value_positions(empty, {}, {}, {}, "EUR", lambda p, q: None, NOW)["invested"] == 0


@pytest.mark.parametrize("old,new", [
    ('type="Dividends" currency="USD" amount="10" fxRateToBase="0.9" dateTime="20260301;000000" levelOfDetail="DETAIL"', 'type="Dividends" currency="USD" amount="10" fxRateToBase="0.9" dateTime="20260301;000000" levelOfDetail="SUMMARY"'),
    ('type="Dividends" currency="USD" amount="10"', 'currency="USD" amount="10"'),
])
def test_unreadable_dividend_rows_are_never_a_confident_zero(old, new):
    parsed = flex.parse_statement(STATEMENT.replace(old, new))
    assert parsed["dividends_ytd"] is None and any("dividend rows" in w for w in parsed["warnings"])


def test_empty_level_of_detail_still_counts():
    parsed = flex.parse_statement(STATEMENT.replace('dateTime="20260301;000000" levelOfDetail="DETAIL"', 'dateTime="20260301;000000" levelOfDetail=""'))
    assert parsed["dividends_ytd"] == pytest.approx(9.0)


def test_no_cash_section_says_so():
    i, j = STATEMENT.index("<CashTransactions>"), STATEMENT.index("</CashTransactions>") + len("</CashTransactions>")
    parsed = flex.parse_statement(STATEMENT[:i] + STATEMENT[j:])
    assert parsed["dividends_ytd"] is None and any("Cash Transactions" in w for w in parsed["warnings"])


def test_a_holding_on_two_report_dates_uses_the_latest_row():
    extra = '<OpenPosition assetCategory="STK" symbol="ACME" conid="1" currency="USD" position="99" markPrice="1" reportDate="20261001" levelOfDetail="SUMMARY"/>'
    parsed = flex.parse_statement(STATEMENT.replace("<OpenPositions>", "<OpenPositions>" + extra))
    assert next(p for p in parsed["positions"] if p["symbol"] == "ACME")["qty"] == 10


def test_a_second_account_without_cash_or_nav_leaves_them_unknown():
    second = '<FlexStatement accountId="U2" toDate="20261105"><AccountInformation currency="EUR"/></FlexStatement></FlexStatements>'
    parsed = flex.parse_statement(STATEMENT.replace("</FlexStatements>", second))
    assert parsed["cash"] is None and parsed["nav"] == []


def test_zero_cost_holding_stays_out_of_the_gain_percentage():
    rows = [{"key": "1", "symbol": "SPUN", "qty": 10.0, "mark": 10.0, "cost": 0.0, "currency": "EUR", "asset": "STK"},
            {"key": "2", "symbol": "S", "qty": 10.0, "mark": 110.0, "cost": 1000.0, "currency": "EUR", "asset": "STK"}]
    valued = analytics.value_positions({"positions": rows}, {}, {}, {}, "EUR", lambda p, q: None, NOW)
    assert valued["gain_pct"] == pytest.approx(0.1) and valued["gain_unknown"] == ["SPUN"]


def test_a_listing_no_quote_has_confirmed_stays_out_of_history():
    pos = {"key": "1", "symbol": "S", "qty": 1.0, "mark": 10.0, "cost": 5.0, "currency": "EUR", "asset": "STK"}
    row = analytics.value_positions({"positions": [pos]}, {}, {}, {"1": "S"}, "EUR", market.quote_problem, NOW)["positions"][0]
    assert row["history_symbol"] is None


def test_fx_rates_are_kept_by_pair_so_a_base_change_cannot_reuse_one():
    pos = {"key": "1", "symbol": "S", "qty": 1.0, "mark": 10.0, "cost": 5.0, "currency": "GBP", "asset": "STK", "fx_to_base": 1.15}
    old = {"GBPUSD": {"rate": 1.27, "at": NOW}}
    data = {"base": "EUR", "positions": [pos]}
    assert analytics.value_positions(data, {}, old, {}, "EUR", lambda p, q: None, NOW)["positions"][0]["fx"] == 1.15


def test_statement_rates_are_not_used_for_a_base_from_config():
    pos = {"key": "1", "symbol": "S", "qty": 1.0, "mark": 10.0, "cost": 5.0, "currency": "EUR", "asset": "STK", "fx_to_base": 1.0}
    assert analytics.value_positions({"positions": [pos]}, {}, {}, {}, "USD", lambda p, q: None, NOW)["incomplete"] == ["S"]


def test_a_429_like_string_is_not_a_rate_limit(world):
    service, web, *_ = world
    service.tick()
    seen = []

    def get(url, params):
        seen.append(url.rsplit("/", 1)[1])
        raise OSError("no data for 4290.T")
    service.get = get
    service.refresh_prices(service.config())
    assert len(seen) == 3 and "not tried" not in json.dumps(service.state["price_errors"])
    service.state["history_at"], service.state["history_try_at"] = {}, {}
    service.refresh_history(service.config())
    assert service.state["history_pause_until"] == 0


@pytest.mark.parametrize("raw,field,value", [
    ({"loss_review_pct": 30}, "loss_review_pct", -30), ({"max_position_pct": float("nan")}, "max_position_pct", 20),
    ({"max_position_pct": "20"}, "max_position_pct", 20), ({"base_currency": " eur "}, "base_currency", "EUR"),
])
def test_invalid_settings_fall_back(raw, field, value):
    cleaned = svc.clean_config(raw)
    assert cleaned[field] == value


def test_rejected_settings_are_named_in_status(world):
    service, _web, _clock, home = world
    (home / "config.json").write_text(json.dumps({"flex_query_id": "4242", "loss_review_pct": 30}))
    assert "Settings ignored as invalid: loss_review_pct" in service.answer("status")[1]


def test_a_future_stamp_from_a_clock_jump_cannot_block_refreshes(world):
    service, web, clock, home = world
    (home / "state.json").write_text(json.dumps({"history_pause_until": NOW + 10 ** 8, "flex_try_at": NOW + 10 ** 8, "flex_ok_at": None}))
    restarted = svc.Service(home, service.token_file, get=web, clock=lambda: clock["now"], sleep=lambda _s: None)
    assert restarted.state["history_pause_until"] == 0 and restarted.state["flex_try_at"] == 0 and restarted.state["flex_ok_at"] == 0


def test_figures_without_a_base_say_the_currency_is_unknown(world):
    service, web, *_ = world
    web.statement = STATEMENT.replace('<AccountInformation accountId="U9999999" currency="EUR"/>', "")
    service.tick()
    assert "Cash: 500.00 (currency unknown)" in service.answer("summary")[1]


def test_a_base_from_config_is_labelled(world):
    service, web, _clock, home = world
    web.statement = STATEMENT.replace('<AccountInformation accountId="U9999999" currency="EUR"/>', "")
    (home / "config.json").write_text(json.dumps({"flex_query_id": "4242", "base_currency": "EUR"}))
    service.tick()
    assert "comes from config.json" in service.answer("summary")[1]


def test_config_base_cannot_relabel_the_statement(world):
    service, _web, _clock, home = world
    (home / "config.json").write_text(json.dumps({"flex_query_id": "4242", "base_currency": "USD"}))
    service.tick()
    ok, text = service.answer("summary")
    assert ok and "Cash: 500.00 EUR" in text and "differs from the statement" in text


def test_unknown_base_prints_no_invented_total(world):
    service, web, *_ = world
    web.statement = STATEMENT.replace('<AccountInformation accountId="U9999999" currency="EUR"/>', "")
    service.tick()
    text = service.answer("summary")[1]
    assert "Holdings: unknown" in text and "Total: unknown" in text


def test_mixed_bases_leave_out_cash_nav_and_dividends():
    second = """<FlexStatement accountId="U2" toDate="20261105"><AccountInformation currency="USD"/><CashTransactions><CashTransaction type="Dividends" currency="USD" amount="100" fxRateToBase="1" dateTime="20260301" levelOfDetail="DETAIL"/></CashTransactions></FlexStatement></FlexStatements>"""
    parsed = flex.parse_statement(STATEMENT.replace("</FlexStatements>", second))
    assert parsed["mixed_base"] and parsed["base"] is None
    assert parsed["cash"] is None and parsed["nav"] == [] and parsed["dividends_ytd"] is None


def test_mixed_bases_never_use_statement_fx():
    data = {"mixed_base": True, "positions": [{"key": "1", "symbol": "A", "qty": 1.0, "mark": 10.0, "cost": 5.0, "currency": "USD", "asset": "STK", "fx_to_base": 0.9}]}
    assert analytics.value_positions(data, {}, {}, {}, "EUR", lambda p, q: None, NOW)["incomplete"] == ["A"]


def test_dividend_without_a_rate_is_not_guessed():
    missing = STATEMENT.replace('amount="10" fxRateToBase="0.9"', 'amount="10"')
    parsed = flex.parse_statement(missing)
    assert parsed["dividends_ytd"] is None and any("dividend" in w for w in parsed["warnings"])
    same_currency = STATEMENT.replace('type="Dividends" currency="USD" amount="10" fxRateToBase="0.9"', 'type="Dividends" currency="EUR" amount="10"')
    assert flex.parse_statement(same_currency)["dividends_ytd"] == pytest.approx(10)


def test_dividends_say_the_period_they_cover():
    short = STATEMENT.replace('fromDate="20260101"', 'fromDate="20260201"')
    assert flex.parse_statement(short)["dividends_from"] == "2026-02-01"


def test_repeated_lots_leave_the_quantity_unknown():
    rows = STATEMENT.replace('levelOfDetail="SUMMARY"', "").replace('levelOfDetail="LOT"', "")
    parsed = flex.parse_statement(rows)
    assert next(p for p in parsed["positions"] if p["symbol"] == "ACME")["qty"] is None
    assert any("repeats a holding" in w for w in parsed["warnings"])


def test_two_periods_for_one_account_use_the_latest_only():
    old = f"""<FlexStatement accountId="{ACCOUNT}" toDate="20251105"><OpenPositions><OpenPosition assetCategory="STK" symbol="ACME" conid="1" currency="USD" position="99" levelOfDetail="SUMMARY"/></OpenPositions></FlexStatement>"""
    parsed = flex.parse_statement(STATEMENT.replace("<FlexStatements count=\"1\">", "<FlexStatements>" + old))
    assert next(p for p in parsed["positions"] if p["symbol"] == "ACME")["qty"] == 10
    assert any("several periods" in w for w in parsed["warnings"])


def test_unreadable_dates_are_flagged_not_misread():
    assert flex._date("11/05/2026") == ""
    assert flex._date("20261105;093000") == "2026-11-05"
    parsed = flex.parse_statement(STATEMENT.replace('toDate="20261105"', 'toDate="11/05/2026"'))
    assert any("Date Format" in w for w in parsed["warnings"]) and parsed["dividends_ytd"] is None


def test_merge_with_unknown_cost_drops_the_average_cost():
    second = """<FlexStatement accountId="U2" toDate="20261105"><OpenPositions><OpenPosition assetCategory="STK" symbol="ACME" conid="1" currency="USD" position="5" levelOfDetail="SUMMARY"/></OpenPositions></FlexStatement></FlexStatements>"""
    acme = next(p for p in flex.parse_statement(STATEMENT.replace("</FlexStatements>", second))["positions"] if p["symbol"] == "ACME")
    assert acme["cost"] is None and acme["cost_price"] is None


def test_trades_are_capped_per_symbol_not_across_the_account():
    many = "".join(f'<Trade symbol="BUSY" tradeDate="202603{i:02d}" buySell="BUY" quantity="1" tradePrice="1" currency="USD"/>' for i in range(1, 30))
    trades = flex.parse_statement(STATEMENT.replace("</Trades>", many + "</Trades>"))["trades"]
    assert any(t["symbol"] == "ACME" for t in trades) and sum(t["symbol"] == "BUSY" for t in trades) == 5


# ---------------------------------------------------------------- what "today" means, and old prices

def test_friday_close_on_a_weekend_is_not_called_today(world):
    service, _web, clock, _home = world
    service.tick()
    clock["now"] += 86400
    valued = service.valued(service.config())
    assert _row(valued, "ACME")["move_label"] == "last session 2026-11-06" and valued["day"] is None
    assert "Today, only" not in service.answer("summary")[1]


def test_a_session_of_today_is_called_today(world):
    service, *_ = world
    valued = _value(service)
    assert _row(valued, "ACME")["move_label"] == "today" and valued["day"] is not None


def test_a_quote_past_its_age_gives_way_to_ibkr(world):
    service, _web, clock, _home = world
    service.tick()
    clock["now"] += analytics.PRICE_STALE + 60
    acme = _row(service.valued(service.config()), "ACME")
    assert acme["source"] == "IBKR" and acme["price"] == 100


def test_a_quote_older_than_the_statement_gives_way_to_ibkr(world):
    service, *_ = world
    service.tick()
    for position in service.state["flex"]["positions"]:
        position["report_date"] = "2026-11-07"
    assert _row(service.valued(service.config()), "ACME")["source"] == "IBKR"


def test_an_old_statement_is_stale_even_when_just_fetched(world):
    service, web, *_ = world
    web.statement = STATEMENT.replace('toDate="20261105"', 'toDate="20250105"')
    service.tick()
    assert "STALE" in service.answer("summary")[1]


# ---------------------------------------------------------------- analytics

def test_percentile_interpolates():
    assert analytics.percentile([0, 10], 0.5) == 5
    assert analytics.percentile([1, 2, 3, 4, 5], 0.1) == pytest.approx(1.4)


def test_max_drawdown_is_peak_to_trough():
    assert analytics.max_drawdown([100, 120, 60, 200, 150]) == pytest.approx(-0.5)


def test_band_refuses_short_history():
    assert analytics.horizon_band([1.0] * 42, 21) is None
    assert analytics.horizon_band([1.0] * 43, 21) is not None


def test_band_values_for_steady_growth():
    prices = [1.01 ** i for i in range(300)]
    band = analytics.horizon_band(prices, 21)
    assert band["low"] == pytest.approx(1.01 ** 21 - 1) and band["high"] == pytest.approx(1.01 ** 21 - 1)


def test_blend_weights_returns_and_reports_coverage():
    dates = [f"2026-{i:03d}" for i in range(130)]
    histories = {"A": [(d, 100.0 * 1.01 ** i) for i, d in enumerate(dates)], "B": [(d, 50.0) for d in dates],
                 "C": [(dates[0], 1.0)]}
    blend = analytics.blended_series(histories, {"A": 0.5, "B": 0.25, "C": 0.25})
    assert blend["coverage"] == pytest.approx(0.75)
    assert blend["values"][1] == pytest.approx(1 + 0.01 * 2 / 3)


def test_coverage_counts_longs_with_no_mapping_at_all():
    dates = [f"2026-{i:03d}" for i in range(130)]
    blend = analytics.blended_series({"A": [(d, 1.0 + i) for i, d in enumerate(dates)]}, {"A": 0.2}, all_weight=1.0)
    assert blend["coverage"] == pytest.approx(0.2)


def test_correlation_sign():
    up = [1.0, 2.0, 3.0, 5.0]
    assert analytics.correlation(up, up) == pytest.approx(1)
    assert analytics.correlation(up, [-x for x in up]) == pytest.approx(-1)
    assert analytics.correlation(up, [1.0, 1.0, 1.0, 1.0]) is None


LIMITS = {"max_position": 0.2, "loss_review": -0.3, "gain_review": 1.0, "gain_review_weight": 0.1,
          "drawdown_review": -0.25, "correlation_review": 0.7}


@pytest.mark.parametrize("row,rule,fires", [
    ({"weight": 0.21}, "concentration", True), ({"weight": 0.20}, "concentration", False),
    ({"gain_pct": -0.30}, "large_loss", True), ({"gain_pct": -0.29}, "large_loss", False),
    ({"gain_pct": 1.0, "weight": 0.1}, "large_gain", True), ({"gain_pct": 1.0, "weight": 0.09}, "large_gain", False),
])
def test_review_rule_edges(row, rule, fires):
    items, _ = analytics.review_items({"positions": [{"symbol": "X", **row}]}, {}, {}, LIMITS, NOW)
    assert any(i["rule"] == rule for i in items) is fires


def test_every_review_item_has_a_reason_and_a_counterargument():
    row = {"symbol": "X", "weight": 0.5, "gain_pct": -0.5}
    items, _ = analytics.review_items({"positions": [row]}, {}, {}, LIMITS, NOW)
    assert items and all(i["why"] and i["ask"] and i["against"] for i in items)


def _closes(prices, last="2026-11-05"):
    end = datetime.fromisoformat(last)
    return [((end - timedelta(days=len(prices) - 1 - i)).date().isoformat(), p) for i, p in enumerate(prices)]


def test_drawdown_rule_needs_a_real_fall_and_a_current_history():
    row = {"symbol": "X", "history_symbol": "X", "price": 70.0, "qty": 1, "gain_pct": 0.0}
    fell = _closes([100.0] * 200 + [70.0])
    items, _ = analytics.review_items({"positions": [row]}, {}, {"X": fell}, LIMITS, NOW)
    assert [i["rule"] for i in items] == ["drawdown"] and "2026-11-05" in items[0]["why"]
    assert not analytics.review_items({"positions": [row]}, {}, {"X": _closes([100.0] * 201)}, LIMITS, NOW)[0]
    old = _closes([100.0] * 200 + [70.0], last="2026-01-05")
    items, unchecked = analytics.review_items({"positions": [row]}, {}, {"X": old}, LIMITS, NOW)
    assert not items and unchecked == {"fall from the past year's high": ["X"]}


@pytest.mark.parametrize("asked", ["BRK B", "BRK.B", "brk-b", "US0846707026"])
def test_a_holding_is_found_however_it_is_written(world, asked):
    service, web, *_ = world
    web.statement = STATEMENT.replace('symbol="ACME"', 'symbol="BRK B"', 1).replace('isin="US0000000001"', 'isin="US0846707026"')
    web.prices["BRK-B"] = (110.0, "USD")
    service.tick()
    assert "ACME CORP" in service.answer("position", asked)[1]


def test_a_symbol_not_held_lists_what_is(world):
    service, *_ = world
    service.tick()
    text = service.answer("position", "NOPE")[1]
    assert "no holding matched NOPE" in text and "ACME" in text and "Statement date" in text


@pytest.mark.parametrize("view", ["position", "health", "review", "forecast"])
def test_unknown_holdings_are_not_reported_as_none(world, view):
    service, web, *_ = world
    i, j = STATEMENT.index("<OpenPositions>"), STATEMENT.index("</OpenPositions>") + len("</OpenPositions>")
    web.statement = STATEMENT[:i] + STATEMENT[j:]
    service.tick()
    assert "holdings are unknown" in service.answer(view, "ACME")[1]


def test_nav_check_is_silent_without_positions_and_across_dates():
    i, j = STATEMENT.index("<OpenPositions>"), STATEMENT.index("</OpenPositions>") + len("</OpenPositions>")
    assert not any("account value" in w for w in flex.parse_statement(STATEMENT[:i] + STATEMENT[j:])["warnings"])
    other_day = STATEMENT.replace('reportDate="20261105" levelOfDetail', 'reportDate="20261104" levelOfDetail')
    assert not any("account value" in w for w in flex.parse_statement(other_day)["warnings"])


def test_repeated_lots_clear_a_non_share_value_too():
    rows = """<OpenPosition assetCategory="BOND" symbol="B" conid="5" currency="EUR" position="1" positionValue="4900" fxRateToBase="1"/>""" * 2
    plain = STATEMENT.replace(' levelOfDetail="SUMMARY"', "").replace(' levelOfDetail="LOT"', "")
    parsed = flex.parse_statement(plain.replace("</OpenPositions>", rows + "</OpenPositions>"))
    bond = next(p for p in parsed["positions"] if p["symbol"] == "B")
    assert bond["position_value"] is None
    assert not any("Position Value" in w for w in parsed["warnings"])


def test_a_public_rate_older_than_the_statement_gives_way():
    pos = {"key": "1", "symbol": "S", "qty": 1.0, "mark": 10.0, "cost": 5.0, "currency": "USD", "asset": "STK", "fx_to_base": 0.9, "report_date": "2026-11-05"}
    fx = {"USDEUR": {"rate": 0.8, "at": NOW - 3 * 86400}}
    row = analytics.value_positions({"base": "EUR", "positions": [pos]}, {}, fx, {}, "EUR", lambda p, q: None, NOW)["positions"][0]
    assert row["fx"] == 0.9 and row["fx_note"]


def test_a_failed_price_round_shows_in_the_summary(world):
    service, *_ = world
    service.tick()
    service.get = lambda u, p: (_ for _ in ()).throw(OSError("HTTP 500"))
    service.refresh_prices(service.config())
    assert "the last price round got 0 of" in service.answer("summary")[1]


def test_forecast_shows_the_statement_state(world):
    service, _web, clock, _home = world
    service.tick()
    service.histories["ACME"] = _closes([100.0 + i for i in range(600)])
    clock["now"] += 30 * 86400
    assert "STALE: the statement" in service.answer("forecast", "ACME")[1]


def test_futures_only_account_has_zero_holdings_not_unknown():
    fut = {"key": "1", "symbol": "F", "qty": 1.0, "mark": 6000.0, "currency": "EUR", "asset": "FUT"}
    valued = analytics.value_positions({"positions": [fut], "cash": 5000.0}, {}, {}, {}, "EUR", lambda p, q: None, NOW)
    assert valued["invested"] == 0 and valued["total"] == 5000


@pytest.mark.parametrize("raw,field,value", [
    ({"correlation_review": 70}, "correlation_review", 0.7), ({"loss_review_pct": -3000}, "loss_review_pct", -30),
    ({"max_position_pct": 0.2}, "max_position_pct", 20), ({"max_position_pct": 35}, "max_position_pct", 35),
])
def test_limit_ranges(raw, field, value):
    assert svc.clean_config(raw)[field] == value


def test_a_quiet_review_names_what_it_could_not_check(world):
    service, *_ = world
    service.tick()
    text = service.answer("review")[1]
    assert "Not checked for fall from the past year's high" in text and "ACME" in text


def test_stale_history_is_labelled_in_a_forecast(world):
    service, *_ = world
    service.tick()
    service.histories["ACME"] = _closes([100.0 + i for i in range(600)], last="2026-01-05")
    text = service.answer("forecast", "ACME")[1]
    assert "STALE" in text and "up to 2026-01-05" in text


# ---------------------------------------------------------------- refresh, failure, secrets

def test_failed_refresh_keeps_last_data_flags_it_and_hides_the_token(world, caplog):
    service, web, clock, _home = world
    service.tick()
    assert service.state["flex"]

    def broken(url, params):
        raise OSError(f"failed for {url}?t=SECRET-TOKEN-123&q=4242")
    service.get = broken
    clock["now"] += svc.FLEX_EVERY + 1
    service.tick()
    assert service.state["flex"] and "SECRET-TOKEN-123" not in service.state["flex_error"]
    ok, text = service.answer("summary")
    assert ok and "Last statement refresh did not complete" in text and "SECRET-TOKEN-123" not in text
    assert "SECRET-TOKEN-123" not in caplog.text and "SECRET-TOKEN-123" not in service.state_path.read_text()


def test_stale_statement_is_labelled(world):
    service, _web, clock, _home = world
    service.tick()
    assert "STALE" not in service.answer("summary")[1]
    clock["now"] += (svc.FLEX_STALE_DAYS + 2) * 86400
    assert "STALE" in service.answer("summary")[1]


def _sends(web):
    return sum(u.endswith("/SendRequest") for u, _ in web.calls)


def _utc(day, hour=4, minute=23):
    return datetime(2026, 11, day, hour, minute, tzinfo=timezone.utc).timestamp()


def test_flex_is_not_fetched_again_inside_its_interval(world):
    service, web, clock, _home = world
    clock["now"] = _utc(6, 2)  # stays on Friday, so Thursday's statement is still the newest one due
    service.tick()
    clock["now"] += svc.FLEX_EVERY - 5
    service.tick()
    assert _sends(web) == 1


def test_a_fetch_just_before_the_new_statement_is_retried_within_the_hour(world):
    """Req 2026-11-07: fetched at 04:23 UTC Saturday, IBKR still served Thursday, and Friday's trades waited 12 h."""
    service, web, clock, _home = world
    clock["now"] = _utc(7)
    service.tick()
    assert service.state["flex"]["report_date"] == "2026-11-05" and _sends(web) == 1
    web.statement = STATEMENT.replace("20261105", "20261106")
    clock["now"] += svc.FLEX_RETRY - 5
    service.tick()
    assert _sends(web) == 1  # not every minute
    clock["now"] += 10
    service.tick()
    assert service.state["flex"]["report_date"] == "2026-11-06" and _sends(web) == 2
    clock["now"] += svc.FLEX_RETRY + 5  # caught up: back to the long interval
    service.tick()
    assert _sends(web) == 2


@pytest.mark.parametrize("day", [7, 8, 9])  # Saturday, Sunday, Monday
def test_friday_is_the_newest_statement_due_over_the_weekend(world, day):
    service, web, clock, _home = world
    web.statement = STATEMENT.replace("20261105", "20261106")
    clock["now"] = _utc(day)
    service.tick()
    clock["now"] += svc.FLEX_RETRY + 5
    service.tick()
    assert _sends(web) == 1


def test_mondays_statement_is_due_on_tuesday(world):
    service, web, clock, _home = world
    web.statement = STATEMENT.replace("20261105", "20261106")
    clock["now"] = _utc(10)
    service.tick()
    clock["now"] += svc.FLEX_RETRY + 5
    service.tick()
    assert _sends(web) == 2


@pytest.mark.parametrize("day, hourly", [(10, True), (12, False)])  # 4 days behind Monday; 6 behind Wednesday
def test_a_statement_that_stops_advancing_ends_the_hourly_fetches(world, day, hourly):
    service, web, clock, _home = world
    clock["now"] = _utc(day)
    service.tick()
    clock["now"] += svc.FLEX_RETRY + 5
    service.tick()
    assert _sends(web) == (2 if hourly else 1)


def test_the_last_weekday_is_counted_in_utc(world):
    service, _web, _clock, _home = world
    service.state["flex"] = {"report_date": "2026-11-06"}
    assert not service._statement_behind(_utc(9, 23, 30))  # Monday in UTC, Tuesday east of it: Friday is still newest
    assert service._statement_behind(_utc(10, 0, 30))


@pytest.mark.parametrize("stored", [20261105, ["2026-11-05"], True, None, ""])
def test_a_corrupt_stored_date_does_not_stop_the_refresh_cycle(world, stored):
    service, _web, _clock, _home = world
    service.state["flex"] = {"report_date": stored}
    assert service._statement_behind(_utc(7)) is False


def test_a_failure_while_catching_up_keeps_the_statement_and_the_hourly_gate(world):
    service, _web, clock, _home = world
    clock["now"] = _utc(7)
    service.tick()
    service.get = lambda u, p: (_ for _ in ()).throw(OSError("HTTP 500"))
    tried = service.state["flex_try_at"]
    clock["now"] += svc.FLEX_RETRY - 5
    service.tick()
    assert service.state["flex_try_at"] == tried and not service.state["flex_error"]
    clock["now"] += 10
    service.tick()
    assert service.state["flex_error"] and service.state["flex"]["report_date"] == "2026-11-05"


def test_an_undated_statement_keeps_the_long_interval(world):
    service, web, clock, _home = world
    web.statement = STATEMENT.replace('toDate="20261105"', 'toDate="soon"')
    clock["now"] = _utc(7)
    service.tick()
    assert service.state["flex"]["report_date"] == ""
    clock["now"] += svc.FLEX_RETRY + 5
    service.tick()
    assert _sends(web) == 1


def test_failed_flex_waits_before_trying_again(world):
    service, web, clock, _home = world
    service.get = lambda u, p: (_ for _ in ()).throw(OSError("HTTP 500"))
    service.tick()
    tried = service.state["flex_try_at"]
    clock["now"] += svc.FLEX_RETRY - 5
    service.tick()
    assert service.state["flex_try_at"] == tried
    clock["now"] += 10
    service.tick()
    assert service.state["flex_try_at"] > tried


def test_no_setup_makes_no_request_and_says_so(world):
    service, web, _clock, home = world
    (home / "config.json").write_text("{}")
    service.tick()
    assert web.calls == []
    ok, text = service.answer("summary")
    assert not ok and "not set up" in text


def test_only_tickers_reach_yahoo(world):
    service, web, *_ = world
    service.tick()
    names = set()
    for url, params in web.calls:
        if "yahoo" in url:
            assert set(params) == {"range", "interval"}
            names.add(url.rsplit("/", 1)[1])
    assert names == {"ACME", "WRLD.DE", "USDEUR=X", "VT"}


def test_state_survives_a_restart_without_the_account_id(world):
    service, web, clock, home = world
    service.tick()
    restarted = svc.Service(home, service.token_file, get=web, clock=lambda: clock["now"], sleep=lambda _s: None)
    assert restarted.answer("summary")[0]
    stored = (home / "state.json").read_text()
    assert ACCOUNT not in stored and "SECRET-TOKEN-123" not in stored
    assert (home / "state.json").stat().st_mode & 0o077 == 0


def test_a_429_stops_the_price_batch_and_keeps_old_quotes(world):
    service, web, *_ = world
    service.tick()
    old = dict(service.state["quotes"])
    calls = []

    def limited(url, params):
        calls.append(url)
        raise market.RateLimited("HTTP 429")
    service.get = limited
    service.refresh_prices(service.config())
    assert len(calls) == 1 and service.state["quotes"] == old


def test_forecast_refuses_without_history_and_names_the_past_with_it(world):
    service, web, *_ = world
    service.tick()
    assert "not enough price history" in service.answer("forecast", "ACME")[1]
    service.histories["ACME"] = [(f"d{i:04d}", 100.0 + i) for i in range(600)]
    text = service.answer("forecast", "ACME")[1]
    assert "1 month" in text and "not a prediction" in text


def test_review_always_says_it_is_not_advice(world):
    service, *_ = world
    service.tick()
    ok, text = service.answer("review")
    assert ok and "not financial advice" in text and ("None is an instruction" in text or "no prompt to act" in text)


def test_failed_history_waits_before_trying_again(world):
    service, web, clock, _home = world
    service.tick()
    del web.prices["VT"]
    service.state["history_at"].pop("VT")
    service.state["history_try_at"].pop("VT")
    before = len(web.calls)
    service.refresh_history(service.config())
    service.refresh_history(service.config())
    service.refresh_history(service.config())
    assert len(web.calls) == before + 1
    clock["now"] += svc.HISTORY_RETRY + 1
    service.refresh_history(service.config())
    assert len(web.calls) == before + 2


def test_a_429_pauses_all_history_requests(world):
    service, _web, clock, _home = world
    service.tick()
    calls = []

    def limited(url, params):
        calls.append(url)
        raise market.RateLimited("HTTP 429")
    service.get = limited
    service.state["history_at"], service.state["history_try_at"] = {}, {}
    service.refresh_history(service.config())
    service.state["history_try_at"] = {}
    service.refresh_history(service.config())
    assert len(calls) == 1


def test_status_tells_a_failed_price_round_from_a_good_one(world):
    service, web, *_ = world
    service.tick()
    assert "4 of 4 arrived" in service.answer("status")[1] or "3 of 3 arrived" in service.answer("status")[1]
    service.get = lambda u, p: (_ for _ in ()).throw(market.RateLimited("HTTP 429"))
    service.refresh_prices(service.config())
    text = service.answer("status")[1]
    assert " 0 of " in text


def test_new_credentials_are_tried_at_once(world):
    service, web, clock, _home = world
    service.get = lambda u, p: (_ for _ in ()).throw(OSError("HTTP 500"))
    service.tick()
    assert service.state["flex"] is None
    service.get = web
    service.token_file.write_text("NEW-TOKEN")
    os.utime(service.token_file, (NOW + 5, NOW + 5))
    clock["now"] += 40
    service.tick()
    assert service.state["flex"]


@pytest.mark.parametrize("name,content", [("state.json", "[1]"), ("config.json", "[1]"), ("config.json", '{"benchmark": null, "max_position_pct": "x"}')])
def test_wrong_shaped_files_do_not_stop_the_service(world, name, content):
    service, web, clock, home = world
    (home / name).write_text(content)
    restarted = svc.Service(home, service.token_file, get=web, clock=lambda: clock["now"], sleep=lambda _s: None)
    restarted.tick()
    assert restarted.answer("status")[0]


def test_wrong_shaped_history_is_ignored(world):
    service, web, clock, home = world
    (home / "history" / "ACME.json").write_text("[1, 2]")
    restarted = svc.Service(home, service.token_file, get=web, clock=lambda: clock["now"], sleep=lambda _s: None)
    assert "ACME" not in restarted.histories


def test_small_prices_and_large_quantities_read_correctly(world):
    service, *_ = world
    assert service._price(0.0034) == "0.0034" and service._quantity(1234567.5) == "1,234,567.5"


def test_two_listings_with_one_symbol_keep_their_own_prices():
    positions = [{"key": "1", "symbol": "SAP", "currency": "EUR", "exchange": "IBIS", "asset": "STK"},
                 {"key": "2", "symbol": "SAP", "currency": "USD", "exchange": "NYSE", "asset": "STK"}]
    service = svc.Service.__new__(svc.Service)
    service.state = {"flex": {"positions": positions}}
    assert service.symbol_map({"symbol_overrides": {}}) == {"1": "SAP.DE", "2": "SAP"}


def test_start_script_takes_only_a_tailnet_interface(tmp_path):
    fake = tmp_path / "ifconfig"
    body = "en0: flags=1\n\tinet 100.70.0.9 netmask 0xff\nutun4: flags=1\n\tinet 100.70.0.5 netmask 0xff\n"
    script = (SERVICE_DIR / "deploy" / "start-ibkr-service.sh").read_text()
    program = script.split("/usr/bin/awk '", 1)[1].split("')", 1)[0]
    fake.write_text(body)
    out = subprocess.run(["/usr/bin/awk", program, str(fake)], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "100.70.0.5"
    fake.write_text("en0: flags=1\n\tinet 100.70.0.9 netmask 0xff\nutun4: flags=1\n\tinet 100.200.0.5 netmask 0xff\n")
    assert subprocess.run(["/usr/bin/awk", program, str(fake)], capture_output=True, text=True, check=True).stdout.strip() == ""


# ---------------------------------------------------------------- HTTP surface

@pytest.fixture
def server(world):
    service, *_ = world
    service.tick()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), svc.make_handler(service, "the-key"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _http(url, key=None, method="GET"):
    request = urllib.request.Request(url, method=method, headers={"X-Ibkr-Service-Key": key} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_every_route_needs_the_key(server):
    assert _http(f"{server}/v1/summary")[0] == 401
    assert _http(f"{server}/v1/summary", "wrong")[0] == 401
    assert _http(f"{server}/v1/status")[0] == 401
    status, body = _http(f"{server}/v1/summary", "the-key")
    assert status == 200 and body["ok"] and "ACME" in body["text"]


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_the_service_accepts_no_writes(server, method):
    assert _http(f"{server}/v1/summary", "the-key", method=method)[0] == 501


def test_no_order_code_exists():
    source = "\n".join(p.read_text() for p in SERVICE_DIR.glob("*.py")).lower()
    for word in ("placeorder", "place_order", "submit_order", "/order", "/iserver", "ib_insync", "ib_async", "ibapi", "do_post", "do_put", "do_delete"):
        assert word not in source


# ---------------------------------------------------------------- the agent's tool

@pytest.fixture
def tool_env(monkeypatch, tmp_path, server):
    key_file = tmp_path / "key"
    key_file.write_text("the-key\n")
    monkeypatch.setenv("GHOST_PORTFOLIO_URL", server)
    monkeypatch.setenv("GHOST_PORTFOLIO_KEY_FILE", str(key_file))
    return server


async def test_tool_returns_the_service_text(tool_env):
    from ghost_agent.tools.portfolio import tool_portfolio
    assert "ACME" in await tool_portfolio(action="summary")
    assert "ACME CORP" in await tool_portfolio(action="position", symbol="acme")


async def test_tool_marks_a_failure_so_the_loop_sees_one(tool_env, monkeypatch):
    from ghost_agent.tools.portfolio import tool_portfolio
    from ghost_agent.tools.tool_failure import result_is_failure
    monkeypatch.setenv("GHOST_PORTFOLIO_URL", "http://127.0.0.1:9")
    down = await tool_portfolio(action="summary")
    assert result_is_failure(down)


async def test_tool_success_is_not_read_as_a_failure(tool_env):
    from ghost_agent.core.strikes import error_line
    from ghost_agent.tools.portfolio import tool_portfolio
    for action, symbol in (("summary", ""), ("position", "ACME"), ("position", "NOPE"), ("health", ""), ("review", ""),
                           ("forecast", ""), ("forecast", "ACME"), ("status", "")):
        result = await tool_portfolio(action=action, symbol=symbol)
        assert error_line(result, tool="portfolio") == "", action


async def test_member_gets_nothing_even_by_direct_call(tool_env, monkeypatch):
    import ghost_agent.tools.portfolio as tool_module
    from ghost_agent.tools.portfolio import tool_portfolio
    monkeypatch.setattr(tool_module, "requester_is_member", lambda: True)
    result = await tool_portfolio(action="summary")
    assert "ACME" not in result and result.startswith("Error:")


def test_tool_is_outside_the_member_allowlist_and_contained_contexts():
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS
    assert "portfolio" not in _MEMBER_ALLOWED_TOOLS
    from ghost_agent.core import isolation
    assert any("portfolio" in value for name, value in vars(isolation).items()
               if isinstance(value, frozenset) and "lab" in value and "manage_tasks" in value)


def test_tool_is_hidden_until_configured(monkeypatch, tmp_path):
    from ghost_agent.tools.portfolio import portfolio_configured
    monkeypatch.delenv("GHOST_PORTFOLIO_URL", raising=False)
    assert not portfolio_configured()
    key = tmp_path / "key"
    key.write_text("")
    monkeypatch.setenv("GHOST_PORTFOLIO_URL", "http://100.64.0.2:8095")
    monkeypatch.setenv("GHOST_PORTFOLIO_KEY_FILE", str(key))
    assert not portfolio_configured()
    key.write_text("k")
    assert portfolio_configured()


def test_the_registry_really_hides_it_until_configured(mock_context, monkeypatch, tmp_path):
    from ghost_agent.tools import registry as R

    def advertised():
        return {(d.get("function") or {}).get("name") for d in R.get_active_tool_definitions(mock_context, None, serve_tuned=False)}
    monkeypatch.delenv("GHOST_PORTFOLIO_URL", raising=False)
    assert "portfolio" not in advertised()
    key = tmp_path / "key"
    key.write_text("k" * 40)
    monkeypatch.setenv("GHOST_PORTFOLIO_URL", "http://100.64.0.2:8095")
    monkeypatch.setenv("GHOST_PORTFOLIO_KEY_FILE", str(key))
    assert "portfolio" in advertised()


def test_key_is_redacted_from_stored_text(monkeypatch, tmp_path):
    from ghost_agent.distill import redact
    secret = "ibkr-service-key-abcdefghijklmnop"
    key = tmp_path / "key"
    key.write_text(secret)
    monkeypatch.setenv("GHOST_PORTFOLIO_KEY_FILE", str(key))
    rule = redact._portfolio_key_rule()
    assert rule and rule[1].sub(rule[2], f"header {secret}") == "header <REDACTED_PORTFOLIO_KEY>"
