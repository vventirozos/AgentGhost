"""§4VH: the ibkr-service buy advisor and the `portfolio` tool's advise action. Each test names a world where it would
fail. All data is fictional."""
import json
import math
import sys
import threading
import urllib.request
from datetime import date, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interface" / "ibkr-service"))
sys.path.insert(0, str(ROOT / "tests"))

import advisor  # noqa: E402
import service as svc  # noqa: E402
from test_4ti_ibkr_service import NOW, FakeWeb  # noqa: E402

END = date(2026, 11, 6)


def wave(days=700, drift=0.0004, amp=0.01, start=100.0):
    """A steady climb with a fixed wiggle: no randomness, so every test sees the same series."""
    return [start * (1 + drift) ** t * (1 + amp * math.sin(t * 1.7)) for t in range(days)]


def crash(prices, fall=0.15, days=5):
    """The same series, with the last `days` closes stepping down to `fall` below where it stood."""
    top = prices[-days - 1]
    return prices[:-days] + [top * (1 - fall * (i + 1) / days) for i in range(days)]


def to(prices, last):
    """The same series scaled so it ends at `last`."""
    return [p * last / prices[-1] for p in prices]


def dated(prices, end=END):
    return [((end - timedelta(days=len(prices) - 1 - i)).isoformat(), p) for i, p in enumerate(prices)]


# ---------------------------------------------------------------- cheapness

def test_a_fund_at_its_high_scores_dear_and_after_a_fall_scores_cheap():
    """Fails if the ranks are inverted: a fall would read as dear and the advisor would buy the top."""
    rising = advisor.cheapness(wave())
    fallen = advisor.cheapness(crash(wave(), 0.12, 10))
    assert rising["score"] < 0.5 < fallen["score"]
    assert fallen["fall"] < -0.10 and fallen["score"] >= 0.8


def test_the_200_day_average_is_a_true_average():
    """Fails if the running sum skips days before the first ranked one (the gap would be measured against a stale sum)."""
    prices = wave(600)
    got = advisor.cheapness(prices)["gap"]
    assert got == pytest.approx(prices[-1] / (sum(prices[-200:]) / 200) - 1)


def test_a_short_history_gives_no_score():
    assert advisor.cheapness(wave(advisor.MIN_SCORE_DAYS - 1)) is None
    assert advisor.cheapness(wave(advisor.MIN_SCORE_DAYS)) is not None


# ---------------------------------------------------------------- hot rule

def test_a_sharp_fall_is_hot_and_a_normal_wiggle_is_not():
    calm = advisor.hot_signal(wave())
    sharp = advisor.hot_signal(crash(wave(), 0.15, 5))
    assert calm["worst"]["z"] > -2.0
    assert sharp["worst"]["z"] < -2.0 and sharp["fall"] < -0.10


def test_usual_move_scales_with_the_name_volatility():
    """The same -15% is far more unusual for a calm name than for a jumpy one."""
    calm = advisor.hot_signal(crash(wave(amp=0.005), 0.15, 5))["worst"]["z"]
    jumpy = advisor.hot_signal(crash(wave(amp=0.04), 0.15, 5))["worst"]["z"]
    assert calm < jumpy < 0


# ---------------------------------------------------------------- core split

def _funds(a_score, b_score, a_held=2 / 3, b_held=1 / 3, safe="A"):
    return [{"symbol": "A", "aim": 2 / 3, "held": a_held, "score": a_score, "safe": safe == "A"},
            {"symbol": "B", "aim": 1 / 3, "held": b_held, "score": b_score, "safe": safe == "B"}]


def test_neutral_scores_and_no_drift_follow_the_aim():
    split = advisor.core_split(_funds(0.5, 0.5), 0.25)
    assert split["shares"]["A"] == pytest.approx(2 / 3) and split["rule"] == "tilt"


def test_a_cheap_fund_gets_more_than_its_aim():
    split = advisor.core_split(_funds(0.5, 0.85), 0.25)
    assert split["shares"]["B"] > 0.5


def test_the_safe_fund_takes_everything_when_cheap_and_cheapest():
    split = advisor.core_split(_funds(0.8, 0.6), 0.25)
    assert split["shares"] == {"A": 1.0, "B": 0.0} and split["rule"] == "safe" and "safer" in split["reasons"]["A"]


def test_the_safe_fund_does_not_take_everything_when_the_other_is_cheaper():
    split = advisor.core_split(_funds(0.8, 0.9), 0.25)
    assert split["rule"] == "tilt" and 0 < split["shares"]["A"] < 1


def test_a_fund_far_above_its_aim_is_left_out_even_when_cheap():
    """Fails if the drift cap is missing: the safe fund would swallow every buy and 60/30 would be gone."""
    split = advisor.core_split(_funds(0.9, 0.5, a_held=0.95, b_held=0.05), 0.25)
    assert split["shares"]["A"] == 0 and split["shares"]["B"] == pytest.approx(1.0)
    assert "above its aim" in split["reasons"]["A"]


def test_drift_pulls_toward_the_aim():
    under = advisor.core_split(_funds(0.5, 0.5, a_held=0.8, b_held=0.2), 0.25)["shares"]["B"]
    assert under > 1 / 3


def test_nothing_held_means_no_drift():
    split = advisor.core_split(_funds(0.5, 0.5, a_held=None, b_held=None), 0.25)
    assert split["shares"]["A"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------- whole shares

@pytest.mark.parametrize("money,prices,budget", [
    ({"A": 300.0, "B": 150.0}, {"A": 110.0, "B": 38.0}, 450.0),
    ({"A": 450.0, "B": 0.0}, {"A": 110.0, "B": 38.0}, 450.0),
    ({"A": 10.0, "B": 10.0}, {"A": 110.0, "B": 38.0}, 20.0),
])
def test_whole_shares_never_spend_more_than_the_budget(money, prices, budget):
    counts, left = advisor.whole_shares(money, prices, budget)
    spent = sum(counts[s] * prices[s] for s in counts)
    assert spent <= budget + 1e-9 and left == pytest.approx(budget - spent)
    assert left < min(prices.values())  # a share that still fits is bought


# ---------------------------------------------------------------- the service view

CORE = [{"symbol": "WRLD", "target": 60, "safe": True}, {"symbol": "EMRG", "target": 30, "yahoo": "EMRG.DE"}]


@pytest.fixture
def adv(tmp_path):
    web = FakeWeb()
    web.prices.update({"EMRG.DE": (20.0, "EUR"), "CHIPX": (30.0, "USD"), "CALM": (50.0, "USD")})
    web.names.update({"CHIPX": "Chipmaker Example Fund", "CALM": "Calm Example Fund"})
    web.kinds.update({"CHIPX": "ETF", "CALM": "ETF", "EMRG.DE": "ETF", "WRLD.DE": "ETF"})
    # each history ends at its quote's price, as Yahoo's does: the split guard rejects a quote 25% off the last close
    web.history = {"WRLD.DE": to(wave(), 55.0), "EMRG.DE": to(wave(), 20.0), "CHIPX": to(crash(wave(), 0.15, 5), 30.0),
                   "CALM": to(wave(), 50.0), "ACME": to(wave(), 110.0), "VT": to(wave(), 100.0)}
    token = tmp_path / "token"
    token.write_text("SECRET-TOKEN-123\n")
    home = tmp_path / "home"
    home.mkdir()
    config = {"flex_query_id": "4242", "advisor_core": CORE, "watchlist": ["CHIPX", "CALM"], "max_drift_pts": 100}
    (home / "config.json").write_text(json.dumps(config))
    service = svc.Service(home, token, get=web, clock=lambda: NOW, sleep=lambda _s: None)
    service.tick()
    return service, web, home, config


def _set(home, config, **changes):
    (home / "config.json").write_text(json.dumps({**config, **changes}))


def test_advice_splits_the_sum_and_names_the_hot_fall(adv):
    service, _, _, _ = adv
    ok, text = service.answer("advise", amount=500.0)
    assert ok, text
    assert "Core 450.00 EUR (90%), opportunity slice 50.00 EUR (10%)" in text
    assert "Hot now: 1." in text and "CHIPX (Chipmaker Example Fund)" in text
    assert "whole share(s) of CHIPX" in text and "find out WHY" in text and "buy it once" in text
    assert "SECRET" not in text


def _spent(text):
    """What the buy lines spend, from their own counts and prices: independent of the reply's leftover figure."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("- Buy ") and " share(s) at about " in line:
            count = int(line.split(": ", 1)[1].split(" ")[0])
            total += count * float(line.split("at about ")[1].split(" ")[0])
    return total


def _num(text, before, after=" EUR"):
    return float(text.split(before)[1].split(after)[0].replace(",", ""))


def test_every_euro_is_in_exactly_one_place(adv):
    """Core buys + the core's leftover + the hot buy + what waits = amount + reserve. §4VH-c: the core's leftover is
    the core's (carried to the next sum), no longer opportunity money."""
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0, reserve=70.0)
    core_left = _num(text, "Left over after whole shares: ")
    hot_cost = _num(text, "whole share(s) of CHIPX at about 30.00 USD (traded in USD, so a conversion), about ")
    pot = _num(text, "the opportunity money (")
    waits = _num(text, "The other ")
    assert pot == pytest.approx(50 + 70) and hot_cost + waits == pytest.approx(pot)
    assert _spent(text) + core_left == pytest.approx(450, abs=0.011)


def test_nothing_hot_keeps_the_slice_as_cash(adv):
    service, web, home, config = adv
    _set(home, config, watchlist=["CALM"])
    _, text = service.answer("advise", amount=500.0, reserve=30.0)
    assert "Hot now: none" in text and "CALM" in text and _num(text, "for the next one: ") == pytest.approx(80.0)
    assert _spent(text) + _num(text, "Left over after whole shares: ") == pytest.approx(450, abs=0.011)


def test_core_buy_is_whole_shares_within_the_core_sum(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0)
    lines = [l for l in text.splitlines() if l.startswith("- Buy ")]
    spent = 0.0
    for line in lines:
        count = int(line.split(": ", 1)[1].split(" ")[0])
        price = float(line.split("at about ")[1].split(" ")[0])
        spent += count * price
    assert len(lines) == 2 and 0 < spent <= 450.0


def test_drift_cap_sends_the_core_to_the_fund_not_yet_held(adv):
    """Only WRLD is held, so it is 100% of the core against a 67% aim: with a 25-point cap, all goes to EMRG."""
    service, _, home, config = adv
    _set(home, config, max_drift_pts=25)
    _, text = service.answer("advise", amount=500.0)
    wrld = next(l for l in text.splitlines() if l.startswith("- Buy WRLD:"))
    assert wrld.startswith("- Buy WRLD: 0 share(s)") and "left out this time" in wrld


def test_no_core_set_gives_no_split(adv):
    service, _, home, config = adv
    _set(home, config, advisor_core=[])
    ok, text = service.answer("advise", amount=500.0)
    assert ok and "no core funds are set" in text and "Give no split" in text


@pytest.mark.parametrize("bad", [
    [{"symbol": "A", "target": -1}], [{"symbol": "A", "target": 60, "safe": True}, {"symbol": "B", "target": 30, "safe": True}],
    [{"symbol": "A", "target": 60}, {"symbol": "a", "target": 30}], "WRLD", [{"symbol": "A", "target": 1, "yahoo": "x y"}],
])
def test_a_malformed_core_is_rejected_and_named(bad):
    cleaned = svc.clean_config({"advisor_core": bad})
    assert cleaned["advisor_core"] == [] and "advisor_core" in cleaned["rejected"]


def test_watchlist_quotes_never_value_a_holding(adv):
    """Fails if the watch round wrote into the holdings' quotes: a watchlist listing would price a holding."""
    service, _, _, _ = adv
    assert "CHIPX" in service.state["watch_quotes"] and "CHIPX" not in service.state["quotes"]


def test_the_history_round_covers_watchlist_and_core(adv):
    service, _, _, _ = adv
    assert {"CHIPX", "CALM", "EMRG.DE"} <= set(service.histories)


def test_advise_needs_an_amount(adv):
    service, _, _, _ = adv
    ok, text = service.answer("advise")
    assert not ok and "needs an amount" in text


@pytest.mark.parametrize("query,ok", [("amount=500", True), ("amount=-5", False), ("amount=nan", False),
                                      ("amount=abc", False), ("amount=500&reserve=-1", False), ("amount=1e12", False)])
def test_http_amount_checks(adv, query, ok):
    service, _, _, _ = adv
    server = ThreadingHTTPServer(("127.0.0.1", 0), svc.make_handler(service, "k"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/advise?{query}", headers={"X-Ibkr-Service-Key": "k"})
        body = json.loads(urllib.request.urlopen(request, timeout=10).read())
    finally:
        server.shutdown()
    assert body["ok"] is ok


def test_no_core_means_no_watchlist_traffic(adv):
    """Fails if the default watchlist is fetched for an owner who never set the advisor up."""
    service, web, home, config = adv
    _set(home, config, advisor_core=[])
    before = len(web.calls)
    service.refresh_watch(service.config())
    assert len(web.calls) == before and service.watch_symbols(service.config()) == []


def test_a_fund_the_split_gave_nothing_gets_no_rounding_share():
    """Review r1: a capped fund (money 0) was handed a leftover share because its shortfall 0 beat negatives."""
    counts, left = advisor.whole_shares({"A": 0.0, "B": 450.0}, {"A": 40.0, "B": 400.0}, 450.0)
    assert counts["A"] == 0 and counts["B"] == 1 and left == pytest.approx(50.0)


def test_the_safe_rule_needs_every_core_fund_scored():
    """Review r1: with the other fund unscored, "no other core fund is cheaper" was unknown, yet safe took all."""
    split = advisor.core_split(_funds(None, 0.8, safe="B"), 0.25)
    assert split["rule"] != "safe" and 0 < split["shares"]["B"] < 1


def _direct(rows, quotes, fx=lambda c: 1.0 if c == "EUR" else None, holdings_known=True):
    history = {"A.DE": dated(to(wave(), 55.0)), "B.X": dated(to(wave(), 20.0))}
    core = [{"symbol": "A", "target": 60, "yahoo": "A.DE", "safe": True}, {"symbol": "B", "target": 30, "yahoo": "B.X"}]
    settings = {"opportunity": 0.1, "max_drift": 0.25, "hot_z": -2.0, "hot_fall": -0.1}
    return advisor.advise(500.0, 0.0, core, rows, history, [], quotes, fx, "EUR", settings, NOW, holdings_known=holdings_known)


def _quote(price, currency):
    return {"price": price, "currency": currency, "session": END.isoformat(), "at": NOW, "kind": "ETF"}


def test_an_unpriced_fund_keeps_its_money_and_is_not_also_in_the_pot():
    """Review r1: B (USD, no rate) was suggested by the split AND its money added to the opportunity pot."""
    a = _direct({}, {"A.DE": _quote(55.0, "EUR"), "B.X": _quote(20.0, "USD")})
    assert a["money"]["B"] > 0 and "B" not in a["counts"]
    assert a["pot"] == pytest.approx(a["slice_money"]) and a["left"] == pytest.approx(a["money"]["A"] - a["counts"]["A"] * 55.0)
    assert any("B: no price in EUR" in g for g in a["gaps"])


def test_a_held_fund_without_a_value_makes_drift_unknown():
    """Review r1: B held but unvalued read as 0% held, so A was capped and the whole core went to B."""
    rows = {"A": {"value_base": 1000.0, "price": 55.0, "fx": 1.0, "currency": "EUR", "history_symbol": "A.DE"},
            "B": {"value_base": None, "price": 20.0, "fx": None, "currency": "USD", "history_symbol": "B.X"}}
    a = _direct(rows, {"A.DE": _quote(55.0, "EUR")})
    assert not a["drift_known"] and a["funds"]["A"]["held"] is None and a["money"]["A"] > 0
    assert any("held but without a value" in g for g in a["gaps"])


def test_unknown_holdings_are_not_read_as_none_held():
    a = _direct({}, {"A.DE": _quote(55.0, "EUR"), "B.X": _quote(20.0, "EUR")}, holdings_known=False)
    assert not a["drift_known"] and any("holdings are unknown" in g for g in a["gaps"])


def test_a_stale_history_is_unusable_and_a_late_quote_does_not_stretch_a_day():
    """Review r1: a quote appended to a week-old history made its "1 day" move a week's. Hot needs this week's prices."""
    settings = {"opportunity": 0.1, "max_drift": 0.25, "hot_z": -2.0, "hot_fall": -0.1}
    core = [{"symbol": "A", "target": 1, "yahoo": "A.DE"}]
    history = {"A.DE": dated(wave(), END - timedelta(days=6)), "HOT": dated(crash(wave(), 0.15, 5), END - timedelta(days=5)),
               "FRESH": dated(crash(wave(), 0.15, 5), END - timedelta(days=2))}
    quotes = {name: _quote(history[name][-1][1], "EUR") for name in ("A.DE", "HOT", "FRESH")}  # today = last close
    a = advisor.advise(100.0, 0.0, core, {}, history, ["HOT", "FRESH"], quotes, lambda c: 1.0, "EUR", settings, NOW)
    assert a["funds"]["A"]["last"] == history["A.DE"][-1][0]  # a 6-day gap: the quote is not joined on
    assert "HOT" in a["unusable"] and "A.DE" in a["unusable"]
    assert [c["symbol"] for c in a["hot"]] == ["FRESH"]  # 2 days: joined, and still hot


def test_a_rejected_core_is_named(adv):
    service, _, home, config = adv
    _set(home, config, advisor_core=[{"symbol": "A", "target": -1}])
    _, text = service.answer("advise", amount=500.0)
    assert "advisor_core setting in config.json is invalid" in text


def test_unknown_holdings_say_so_in_the_reply(adv):
    service, _, _, _ = adv
    service.state["flex"]["positions_known"] = False
    _, text = service.answer("advise", amount=500.0)
    assert "holding unknown" in text and "holdings are unknown" in text


def test_a_failed_watch_round_does_not_look_fresh(adv):
    """Review r1: watch_at moved on every round, so days of failures still read as fresh prices."""
    service, web, _, _ = adv
    first = service.state["watch_ok_at"]
    for name in ("CHIPX", "CALM", "EMRG.DE", "WRLD.DE"):  # WRLD.DE: a held core fund's listing is watched too
        web.prices.pop(name)
    service.clock = lambda: NOW + 3 * 86400
    service.refresh_watch(service.config())
    assert service.state["watch_ok_at"] == first
    _, text = service.answer("advise", amount=500.0)
    assert "fetched in the last 6 hours: 0 of 4" in text and "(STALE" in text


def test_a_quote_past_its_market_age_is_not_used(adv):
    """Review r2 pin: with the history still fresh, a 5-day-old quote must not price EMRG (held nowhere, so the
    quote is its only price) nor CHIPX's last price."""
    service, _, _, _ = adv
    for quote in service.state["watch_quotes"].values():
        quote["at"] = NOW - 5 * 86400
    _, text = service.answer("advise", amount=500.0)
    assert "EMRG: no price in EUR" in text and "30.00 USD" not in text
    for quote in service.state["watch_quotes"].values():
        quote["at"] = NOW
    _, fresh = service.answer("advise", amount=500.0)
    assert "EMRG: no price in EUR" not in fresh and "30.00 USD" in fresh


def test_one_fund_on_two_listings_is_one_holding(adv, monkeypatch):
    """Review r2: max() over the rows dropped an unvalued second listing, so drift read as known."""
    service, _, _, _ = adv
    real = service._holding_rows

    def two(valued, symbol):
        rows = real(valued, symbol)
        return rows + [{**rows[0], "value_base": None, "fx": None}] if rows else rows
    monkeypatch.setattr(service, "_holding_rows", two)
    _, text = service.answer("advise", amount=500.0)
    assert "WRLD: held but without a value" in text and "holding unknown" in text
    assert "Drift from the aim is unknown" in text and "pulled back toward the aim" not in text


def test_two_valued_listings_add_up(adv, monkeypatch):
    service, _, _, _ = adv
    real = service._holding_rows
    monkeypatch.setattr(service, "_holding_rows", lambda v, s: real(v, s) + real(v, s))
    _, text = service.answer("advise", amount=500.0)
    assert "WRLD (WORLD ETF) (the safer fund): aim 66.7% of the core, 100.0% of your core now" in text


def test_watch_fx_merges_and_keeps_the_fresher_rate(adv):
    service, _, _, _ = adv
    service.state["fx"]["USDEUR"] = {"rate": 0.5, "at": NOW + 10}
    service.refresh_watch(service.config())
    assert service.state["fx"]["USDEUR"]["rate"] == 0.5


# ---------------------------------------------------------------- the agent's tool

@pytest.fixture
def adv_tool(adv, monkeypatch, tmp_path):
    service, *_ = adv
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), svc.make_handler(service, "the-key"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    key = tmp_path / "key"
    key.write_text("the-key\n")
    monkeypatch.setenv("GHOST_PORTFOLIO_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setenv("GHOST_PORTFOLIO_KEY_FILE", str(key))
    yield
    httpd.shutdown()


@pytest.mark.parametrize("amount", [500, 500.0, "500", "500 EUR", "€500"])
async def test_tool_advise_passes_the_amount_through(adv_tool, amount):
    from ghost_agent.tools.portfolio import tool_portfolio
    result = await tool_portfolio(action="advise", amount=amount, reserve="20")
    assert result.startswith("PORTFOLIO ADVICE for 500.00 EUR") and "plus 20.00 EUR of opportunity cash" in result


async def test_tool_advise_success_is_not_read_as_a_failure(adv_tool):
    """tool-result prose is parsed: the advice's "Data gap"/"fall"/"left out" lines must not read as an error."""
    from ghost_agent.core.strikes import error_line
    from ghost_agent.tools.portfolio import tool_portfolio
    from ghost_agent.tools.tool_failure import result_is_failure
    for amount, reserve in ((500, None), (500, 100), (12.5, None)):
        result = await tool_portfolio(action="advise", amount=amount, reserve=reserve)
        assert error_line(result, tool="portfolio") == "" and not result_is_failure(result), result[:200]


@pytest.mark.parametrize("amount,reserve", [(None, None), ("", None), ("lots", None), (0, None), (-5, None),
                                            (True, None), (float("nan"), None), (1e12, None), (500, -1), (500, "x")])
async def test_tool_advise_refuses_a_bad_number_before_asking(adv_tool, amount, reserve):
    from ghost_agent.tools.portfolio import tool_portfolio
    from ghost_agent.tools.tool_failure import result_is_failure
    result = await tool_portfolio(action="advise", amount=amount, reserve=reserve)
    assert result.startswith("Error:") and result_is_failure(result)


def test_tool_definition_offers_advise_with_its_numbers():
    from ghost_agent.tools.portfolio import ACTIONS, PORTFOLIO_TOOL_DEFINITION
    params = PORTFOLIO_TOOL_DEFINITION["function"]["parameters"]["properties"]
    assert "advise" in ACTIONS and set(params["action"]["enum"]) == set(ACTIONS)
    assert params["amount"]["type"] == "number" and params["reserve"]["type"] == "number"


def test_many_hot_names_at_once_are_named_a_market_wide_fall(adv):
    service, web, home, config = adv
    _, one = service.answer("advise", amount=500.0)
    assert "market-wide fall" not in one  # 1 of 4
    web.history.update({"CALM": to(crash(wave(), 0.15, 5), 50.0), "EMRG.DE": to(crash(wave(), 0.15, 5), 20.0)})
    for name in ("CALM", "EMRG.DE"):
        service.state["history_at"].pop(name)
        service.state["history_try_at"].pop(name)
    service.refresh_history(service.config())
    _, many = service.answer("advise", amount=500.0)
    assert "Hot now: 3." in many and "3 of 4 names meet the rule at once" in many


def test_progress_has_a_summary_for_advise():
    from ghost_agent.core import progress as P
    assert "portfolio.advise" in P._VERBS


def test_the_tool_asks_for_a_read_page_before_a_cause_is_named():
    """Live probe-4vh-advise-1: the model named a cause for a hot fall from search snippets of a fringe site."""
    from ghost_agent.tools.portfolio import PORTFOLIO_TOOL_DEFINITION
    text = PORTFOLIO_TOOL_DEFINITION["function"]["description"]
    assert "open and read" in text and "unconfirmed rumour" in text and "name that source" in text



# ---------------------------------------------------------------- §4VH-b: funds only, capped waiting cash

def test_a_single_stock_is_no_candidate_when_funds_only(adv):
    """§4VH-b: the 2022-2026 replay put the slice into single falling stocks and ended well behind 60/30."""
    service, web, home, config = adv
    web.kinds["CHIPX"] = "EQUITY"
    service.state["watch_quotes"]["CHIPX"]["kind"] = "EQUITY"
    _, text = service.answer("advise", amount=500.0)
    assert "Hot now: none" in text and "Left out as not a fund (or type unknown): CHIPX" in text
    _set(home, config, hot_funds_only=False)
    _, text = service.answer("advise", amount=500.0)
    assert "Hot now: 1." in text and "whole share(s) of CHIPX" in text


def test_a_name_of_unknown_type_is_not_a_fund(adv):
    service, _, _, _ = adv
    service.state["watch_quotes"]["CHIPX"].pop("kind", None)
    _, text = service.answer("advise", amount=500.0)
    assert "Left out as not a fund (or type unknown): CHIPX" in text


def test_waiting_cash_is_capped_and_the_rest_joins_the_core(adv):
    """§4VH-b: uncapped, the replay piled up 950 EUR and put it into one falling name."""
    service, _, home, config = adv
    _set(home, config, watchlist=["CALM"])  # nothing hot
    _, text = service.answer("advise", amount=500.0, reserve=500.0)
    assert "may wait up to 150.00 EUR (3 slices); the 400.00 EUR above that joins the core now, so the core is 850.00 EUR" in text
    assert _num(text, "for the next one: ") == pytest.approx(150.0)
    assert _spent(text) + _num(text, "Left over after whole shares: ") == pytest.approx(850, abs=0.011)


def test_the_cap_also_bounds_a_hot_buy(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0, reserve=2000.0)
    assert _num(text, "the opportunity money (") == pytest.approx(150.0)
    assert _spent(text) + _num(text, "Left over after whole shares: ") == pytest.approx(2350, abs=0.011)


def test_cap_zero_sends_everything_to_the_core():
    data = advisor.advise(500.0, 100.0, [], {}, {}, [], {}, lambda c: 1.0, "EUR",
                          {"opportunity": 0.1, "max_drift": 0.25, "hot_z": -2.0, "hot_fall": -0.1, "cap_slices": 0}, NOW)
    assert data["cap"] == 0 and data["overflow"] == 150.0 and data["core_money"] == 600.0 and data["pot"] == 0


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_a_non_bool_funds_only_is_rejected(bad):
    cleaned = svc.clean_config({"hot_funds_only": bad})
    assert cleaned["hot_funds_only"] is True and "hot_funds_only" in cleaned["rejected"]


def test_the_reply_says_buys_after_the_statement_are_not_counted(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0)
    assert "Holdings are as of IBKR's statement of 2026-11-05: buys after it aren't counted yet" in text


def test_default_watchlist_is_eur_ucits_funds():
    """An EU retail account generally can't buy US-domiciled funds; every default must be a Xetra (.DE) listing."""
    assert advisor.DEFAULT_WATCHLIST and all(s.endswith(".DE") for s in advisor.DEFAULT_WATCHLIST)


def test_a_hot_fund_dearer_than_the_pot_keeps_the_cash(adv):
    """§4VH-c: whole shares only (European funds generally can't be bought in fractions)."""
    service, _, _, _ = adv
    service.state["watch_quotes"]["CHIPX"]["price"] = 30.0
    service.histories["CHIPX"] = [(d, p * 4) for d, p in service.histories["CHIPX"]]
    service.state["watch_quotes"]["CHIPX"]["price"] = 120.0  # 96 EUR a share, the pot is 50
    _, text = service.answer("advise", amount=500.0)
    assert "Hot now: 1." in text and "keep the opportunity money (50.00 EUR) as cash" in text
    _, text = service.answer("advise", amount=500.0, reserve=60.0)
    assert "1 whole share(s) of CHIPX" in text


# ---------------------------------------------------------------- §4VH-c: the real-money review

def test_a_small_sum_rounds_to_nearest_not_into_the_cheaper_fund():
    """Review M1 (real prices): 150 EUR put 3 EM shares (124.80) against a 55.70 split, 92% of the core."""
    counts, left = advisor.whole_shares({"A": 79.30, "B": 55.70}, {"A": 124.89, "B": 41.60}, 135.0)
    assert counts == {"A": 0, "B": 1} and left == pytest.approx(135.0 - 41.60)


def test_a_small_sum_says_whole_shares_cannot_follow_the_split(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=60.0)
    assert "whole shares can't follow the split closely" in text


def test_a_mistyped_core_symbol_is_flagged_with_the_held_funds(adv):
    """Review M2: a typo made the other fund look 100% held, so every core sum went to the unmatched one, silently."""
    service, _, home, config = adv
    _set(home, config, advisor_core=[{"symbol": "WRLDX", "target": 60, "safe": True, "yahoo": "WRLD.DE"},
                                     {"symbol": "EMRG", "target": 30, "yahoo": "EMRG.DE"}])
    _, text = service.answer("advise", amount=500.0)
    assert "Check: core fund WRLDX matches no holding" in text and "Funds you hold that aren't in the core: WRLD" in text


def test_a_split_is_not_read_as_a_crash():
    """Review M3: a 1:5 split quote joined onto the pre-split history scored 1.0 (all-in) and topped the hot list."""
    history = {"A.DE": dated(to(wave(), 55.0)), "B.X": dated(to(wave(), 20.0))}
    split_quote = {**_quote(11.0, "EUR"), "session": (END + timedelta(days=1)).isoformat()}
    core = [{"symbol": "A", "target": 60, "yahoo": "A.DE", "safe": True}, {"symbol": "B", "target": 30, "yahoo": "B.X"}]
    settings = {"opportunity": 0.1, "max_drift": 0.25, "hot_z": -2.0, "hot_fall": -0.1}
    a = advisor.advise(500.0, 0.0, core, {}, history, [], {"A.DE": split_quote, "B.X": _quote(20.0, "EUR")},
                       lambda c: 1.0, "EUR", settings, NOW + 86400)
    assert a["funds"]["A"]["score"] is None and a["split"]["rule"] != "safe" and not a["hot"]
    assert "A.DE" in a["jumps"] and any("split or another listing" in g for g in a["gaps"])


def test_todays_live_bar_is_replaced_by_the_quote():
    """Review m6: Yahoo's daily series carries today's live bar; a quote of the same session must replace it."""
    prices = to(wave(), 20.0)
    history = {"F": dated(prices[:-1] + [prices[-1] * 0.8])}  # a stale intraday snapshot 20% down
    settings = {"opportunity": 0.1, "max_drift": 0.25, "hot_z": -2.0, "hot_fall": -0.1}
    a = advisor.advise(100.0, 0.0, [], {}, history, ["F"], {"F": _quote(prices[-1], "EUR")}, lambda c: 1.0, "EUR", settings, NOW)
    assert not a["hot"] and a["near"][0]["price"] == pytest.approx(prices[-1])


def test_buys_above_the_statement_cash_are_flagged(adv):
    """Review m4: a large reserve moved into the core with no check against the account's cash (500 on the statement)."""
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0, reserve=3000.0)
    assert "more than the cash on IBKR's statement of 2026-11-05 (500.00 EUR)" in text
    _, small = service.answer("advise", amount=200.0)
    assert "more than the cash" not in small


def test_a_safe_flag_must_be_a_real_bool():
    """Review n9: "safe": "true" silently meant not safe."""
    cleaned = svc.clean_config({"advisor_core": [{"symbol": "A", "target": 60, "safe": "true"}]})
    assert cleaned["advisor_core"] == [] and "advisor_core" in cleaned["rejected"]


def test_the_reply_names_costs(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0)
    assert "commission" in text and "minimum per order" in text


@pytest.mark.parametrize("text,expected", [
    ("500", 500.0), ("500 EUR", 500.0), ("€1,500", 1500.0), ("1.500", 1500.0), ("1.500,50", 1500.5),
    ("1,500.50", 1500.5), ("2,50", 2.5), ("12,5", 12.5), ("1 500", 1500.0),
    ("$500", None), ("500 USD", None), ("£50", None), ("1.50.0", None), ("1,5000", None), ("1.500.50,5", None),
    ("500 euros", 500.0), ("500 euro", 500.0), ("0.004", None),  # §4VH-f review: "euros" -> "os"; "0.004" read as 4
])
def test_amount_text_in_european_and_english_formats(text, expected):
    """Review m5: "1.500" read as 1.50 and "2,50" as 250; "$500" was taken as 500 euros."""
    from ghost_agent.tools.portfolio import _number
    assert _number(text, 0.01) == expected


def test_a_watch_symbol_without_a_price_is_not_called_a_non_fund(adv):
    """Live 11-23: right after a watchlist change every new fund read "not a fund (or type unknown)"; it had no price yet."""
    service, _, _, _ = adv
    service.state["watch_quotes"].pop("CALM")
    _, text = service.answer("advise", amount=500.0)
    assert "CALM" not in text.split("Left out as not a fund")[-1].split("\n")[0] if "Left out as not a fund" in text else True
    assert "no fresh price yet" in text and "CALM" in text.split("no fresh price yet")[1].split("\n")[0]


def test_a_new_watch_symbol_is_fetched_within_minutes(adv):
    service, web, home, config = adv
    web.prices["NEWF"] = (10.0, "EUR")
    _set(home, config, watchlist=["CHIPX", "CALM", "NEWF"])
    service.clock = lambda: NOW + 301
    service.tick()
    assert "NEWF" in service.state["watch_quotes"]



def test_each_buy_line_states_its_trading_currency(adv):
    """Live probe-4vh-advise-3: the model called a EUR listing "USD-denominated" from "USD Acc" in its name."""
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0)
    emrg = next(l for l in text.splitlines() if l.startswith("- Buy EMRG:"))
    assert "traded in EUR, the account's currency: no conversion" in emrg
    assert "(traded in USD, so a conversion)" in text  # the hot USD fund


# ---------------------------------------------------------------- §4VH-d: one world fund, earlier funds kept

def test_one_world_fund_takes_the_whole_core_and_kept_funds_are_not_typos(adv):
    """Operator 11-23: switch to one world fund (90%) + 10% opportunity; the earlier funds stay, no longer bought.
    Fails if a kept fund is reported as a possible symbol typo, or the single fund gets less than the core."""
    service, _, home, config = adv
    _set(home, config, advisor_core=[{"symbol": "EMRG", "target": 90, "yahoo": "EMRG.DE"}], advisor_kept=["WRLD"])
    _, text = service.answer("advise", amount=500.0)
    assert "Kept, not bought any more (your earlier plan; selling isn't suggested): WRLD (WORLD ETF)" in text
    assert "Funds you hold that aren't in the core" not in text
    assert "One core fund: it takes the whole core sum every time" in text
    assert "Buy EMRG: 22 share(s)" in text and "Buy WRLD" not in text  # 450 / 20 EUR
    assert "safer fund" not in text and "That is expected until your first buy of it shows on a statement" in text


def test_market_wide_line_names_no_single_stock_risk_when_funds_only(adv):
    service, web, _, _ = adv
    web.history.update({"CALM": to(crash(wave(), 0.15, 5), 50.0), "EMRG.DE": to(crash(wave(), 0.15, 5), 20.0)})
    for name in ("CALM", "EMRG.DE"):
        service.state["history_at"].pop(name)
        service.state["history_try_at"].pop(name)
    service.refresh_history(service.config())
    _, text = service.answer("advise", amount=500.0)
    assert "market-wide fall" in text and "single-company risk" not in text


@pytest.mark.parametrize("bad", ["WRLD", [1], [""], ["x" * 40]])
def test_a_malformed_kept_list_is_rejected(bad):
    cleaned = svc.clean_config({"advisor_kept": bad})
    assert cleaned["advisor_kept"] == [] and "advisor_kept" in cleaned["rejected"]


# ---------------------------------------------------------------- §4VH-f: final review (carry, unplaced, next time)

def test_carry_joins_the_core_only_and_next_time_names_both_amounts(adv):
    """Review §4VH-f M1: nothing told the owner to pass the waiting cash back, so the cap never applied; and a leftover
    carried inside the next amount leaked 10% into the opportunity slice."""
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0, carry=20.0, reserve=70.0)
    assert "Core 470.00 EUR (90% of the sum, 450.00 EUR + 20.00 EUR carried over from last time), opportunity slice 50.00 EUR" in text
    assert _spent(text) + _num(text, "Left over after whole shares: ") == pytest.approx(470, abs=0.011)
    waits = _num(text, "The other ")
    nxt = text.split("NEXT TIME: ")[1].split("\n")[0]
    assert f"plus {_num(text, 'Left over after whole shares: '):.2f} EUR carried over and {waits:.2f} EUR of opportunity cash" in nxt


def test_an_unpriced_core_fund_is_not_placed_rather_than_left_over():
    """Review §4VH-f m2: with no price the reply said "Left over: 0.00" while the fund's whole share was unplaced."""
    a = _direct({}, {"A.DE": _quote(55.0, "EUR"), "B.X": _quote(20.0, "USD")})
    assert a["unplaced"] == pytest.approx(a["money"]["B"]) and a["unplaced"] > 0


def test_core_buy_names_the_exchange(adv):
    service, _, _, _ = adv
    _, text = service.answer("advise", amount=500.0)
    assert "Buy EMRG: " in text and "20.00 EUR on Xetra (ticker EMRG)" in text


def test_the_hot_pick_is_always_described(adv):
    """Review §4VH-f m5: only the first 3 hot names were described; a cheaper 4th could be picked with no figures."""
    service, web, home, config = adv
    for name in ("BIGA", "BIGB", "BIGC"):  # unusually fallen funds costing more than the pot
        web.prices[name] = (400.0, "EUR")
        web.kinds[name] = "ETF"
        web.history[name] = to(crash(wave(), 0.35, 5), 400.0)  # in steps: one 35% day reads as a split
    _set(home, config, watchlist=["BIGA", "BIGB", "BIGC", "CHIPX"])
    service.clock = lambda: NOW + 600  # past the 5-minute wait for watch symbols that have no price yet
    service.tick()
    _, text = service.answer("advise", amount=500.0)
    hot_lines = [ln for ln in text.splitlines() if ln.startswith("- ") and "times its usual move" in ln]
    assert "Hot now: 4." in text and [ln[2:6] for ln in hot_lines[:3]] == ["BIGA", "BIGB", "BIGC"] or \
        sorted(ln[2:6] for ln in hot_lines[:3]) == ["BIGA", "BIGB", "BIGC"], text  # the pick is NOT among the first 3
    assert "whole share(s) of CHIPX" in text and any(ln.startswith("- CHIPX") for ln in hot_lines), text


async def test_tool_passes_carry(adv_tool):
    from ghost_agent.tools.portfolio import tool_portfolio
    result = await tool_portfolio(action="advise", amount="500 euros", carry="12,40")
    assert "12.40 EUR carried over from last time" in result


def test_unplaced_core_money_is_named_and_carried_in_next_time(adv):
    """Review §4VH-f m2 at the reply: a core fund without a price is "Not placed", and NEXT TIME counts it."""
    service, _, _, _ = adv
    service.state["watch_quotes"].pop("EMRG.DE")
    _, text = service.answer("advise", amount=500.0)
    unplaced = _num(text, "Not placed: ")
    assert unplaced > 0 and "Buy EMRG: " in text and "no share count" in text
    left = _num(text, "Left over after whole shares: ")
    assert f"plus {left + unplaced:.2f} EUR carried over" in text.split("NEXT TIME: ")[1]


def test_exchange_is_named_only_when_that_listing_priced_the_fund(adv):
    """Review §4VH-f r2: a core fund priced in USD by IBKR, with yahoo X.DE configured, read "100 USD on Xetra"."""
    service, _, home, config = adv
    _, text = service.answer("advise", amount=500.0)
    assert "Buy WRLD: " in text and "on Xetra (ticker WRLD)" in text
    service.state["watch_quotes"]["WRLD.DE"]["currency"] = "GBP"  # the configured listing no longer matches the holding's
    _, text = service.answer("advise", amount=500.0)
    assert "Buy WRLD: " in text and "on Xetra (ticker WRLD)" not in text
