# ibkr-service

A small read-only service that downloads an IBKR Flex statement, adds public prices, computes the figures and serves them as text to the agent's `portfolio` tool over the tailnet.

The agent stays on Tor and holds no IBKR secret. This service uses the normal internet and a Flex token, by the operator's choice. It can't trade: there is no order code, Flex only hands out reports, and every route is a GET.

## Set up the Flex query in IBKR

In Client Portal, open Performance & Reports > Flex Queries (the labels may differ slightly).

1. Create an **Activity Flex Query**, format **XML**, period **Last 365 Calendar Days**.
2. Select these sections and fields:
   - **Account Information**: Currency.
   - **Open Positions**, with the **Summary** level of detail: Symbol, Description, Conid, ISIN, Asset Class, Currency, Listing Exchange, Quantity, Mark Price, Position Value, Cost Basis Money, Cost Basis Price, FX Rate To Base, Report Date, Level Of Detail.
   - **Net Asset Value (NAV) in Base**: Report Date, Total.
   - **Cash Report**: Currency, Ending Cash. Include the base currency summary.
   - **Cash Transactions**, with **Detail**: Type, Currency, Amount, FX Rate To Base, Date/Time, Level Of Detail.
   - **Trades**, with **Execution**: Symbol, Trade Date, Buy/Sell, Quantity, Trade Price, Currency, IB Commission, Level Of Detail.
3. Note the **Query ID**, then generate a token under **Flex Web Service Configuration**.

The service names a missing field in its replies as a "Data gap", and leaves the figure unknown instead of guessing. Select the Summary level for Open Positions: Lot rows without the Level Of Detail field make it leave those quantities unknown.

## Give the service the token

In a real terminal, so the token stays out of any transcript:

```
ssh -t nova '~/Data/AI/ibkr-service/setup-flex.sh'
```

The service tries new credentials within a minute. Tokens expire: when one does, the service keeps the last statement, says so, and asks for this step again.

## Files on the service machine

| Path | What it holds |
|---|---|
| `~/Data/AI/ibkr-service/app/` | This code |
| `~/Data/AI/ibkr-service/data/config.json` | Query id and the settings below |
| `~/Data/AI/ibkr-service/data/state.json`, `history/` | Parsed holdings and prices, mode 0600. No account id, no raw statement |
| `~/Data/AI/.ibkr_flex_token` | The Flex token, 0600. Never leaves this machine |
| `~/Data/AI/.ibkr_service_key` | The key the agent presents. Also on the agent machine |
| `~/Data/AI/Logs/ibkr-service.log` | Refresh results. No tickers, holdings, token or key |

## Settings in `config.json`

`base_currency`, `benchmark` (default `VT`), `symbol_overrides` (ISIN or IBKR symbol to Yahoo symbol), `max_position_pct` (20, the limit for one company or a holding of unknown type), `max_fund_pct` (100, the limit for one fund: Yahoo's instrument type ETF or MUTUALFUND; the type can't tell a broad index fund from a one-sector fund, and the review says so in one line when a fund is above `max_position_pct` and at or below `max_fund_pct`; with the default 100 that line replaces the size prompt for funds; funds get no large-gain prompt), `loss_review_pct` (-30), `gain_review_pct` (100), `gain_review_weight_pct` (10), `drawdown_review_pct` (-25), `correlation_review` (0.7).

The buy advisor's settings: `advisor_core` (empty, so the advisor is off and fetches nothing: a list such as `[{"symbol": "<ticker>", "target": 60, "safe": true}, {"symbol": "<ticker>", "target": 30, "yahoo": "<yahoo symbol>"}]`; `symbol` matches a holding like the `position` view does, `yahoo` is needed only for a fund not held yet, at most one is `safe`; one fund alone takes the whole core), `advisor_kept` (holdings from an earlier plan that are kept but no longer bought: named as kept, never as a possible symbol typo), `watchlist` (9 tech-leaning UCITS funds in EUR on Xetra: Nasdaq-100, semiconductors, S&P 500 IT, World IT, Europe tech, robotics, Japan, China, Europe 600), `opportunity_pct` (10), `max_drift_pts` (25), `hot_z` (-2.0), `hot_fall_pct` (-10), `hot_funds_only` (true), `opportunity_cap_slices` (3). An invalid `advisor_core` or `watchlist` (a `safe` that isn't a real true/false included) is named and ignored, never half-used.

## The buy advisor

`/v1/advise?amount=500&reserve=0&carry=0` splits a sum about to be invested. It suggests buys and still places none. The service remembers nothing between asks: `carry` is last time's whole-share leftover (and any core money not placed for want of a price), which joins the core only; `reserve` is the opportunity cash still waiting. Every reply ends with a NEXT TIME line giving both figures. Buy lines name the exchange ("on Xetra (ticker SPYI)").

- **Core** (`100 - opportunity_pct` %): each core fund gets a cheapness score, 0 to 1, against its own five years: its fall from the 52-week high, its gap to the 200-day average and its last month's return, each ranked against every past day. Its money share is its aim (`target` over the sum of targets), multiplied by up to 3 when cheap and down to 1/3 when dear, and by `sqrt(aim / held)` (0.5 to 2) when the holdings have drifted. A fund more than `max_drift_pts` above its aim within the core gets nothing that time. The `safe` fund takes the whole core when its score is 0.75 or more, every core fund has a score, and none is cheaper. The money is then rounded to each fund's nearest whole number of shares, never above the core sum; what is left is the core's, carried to the next sum. A core fund that matches no holding is named, with the funds held outside the core, since a mistyped symbol would otherwise send it every core sum.
- **Opportunity** (`opportunity_pct` %, plus `reserve`, up to `opportunity_cap_slices` slices; above that the money joins the core): with `hot_funds_only`, only funds (Yahoo type ETF or mutual fund, and the core funds) are candidates. Every one is checked for a fall of at least `|hot_z|` times its usual move (the past year's daily volatility x sqrt(days)) over 1 day, 5 days or 1 month, while at least `|hot_fall_pct|`% below its 52-week high. The most unusual fall the pot can buy in whole shares gets it, and the rest waits; with none, the reply says to keep it as cash. The service sees prices only, so the reply says to find out why a name fell before buying.
- A core fund needs two years of history for a score, a hot candidate prices from the last 4 days. A quote replaces the history's bars of its own session (Yahoo's daily series carries today's live bar), and joins only when the history ends at most 4 days before it and the quote is within 25% of the last close: a bigger jump is a split or another listing, so that name is left out until the history refreshes. Watchlist prices refresh every 30 minutes, and a price older than 4 days of market time is dropped.
- A held fund without a value, or a statement without holdings, makes drift unknown: the split then uses the aims and the cheapness only, and says so.
- The reply also says: holdings are as of the statement (asking again before the next one repeats the split), the buys against the statement's cash, and order costs.

**Why funds only, and the cap (§4VH-b).** A month-by-month replay on 2022-2026 prices, with no look-ahead: the slice as first built (single stocks, cash waiting without limit) ended EUR 500-1,000 behind putting the same money into the core, from every start date; waiting cash once reached EUR 950 and went into one falling stock. Funds only with the cap roughly matched the core. The final plan as a whole (through `advise` itself, whole shares, carries) ended within 0.3% of plain 60/30 whole-share buying from three start dates. The cheapness tilt and the hot slice are discipline, not a measured edge. US-domiciled funds (QQQ and the like) are left out because an EU retail account generally can't buy them.

## Running it

`com.local.ibkr-service` is a system LaunchDaemon with `RunAtLoad` and `KeepAlive`, so it starts at boot and restarts after a crash. `start-ibkr-service.sh` waits for the tailnet address and binds only that.

```
sudo launchctl print system/com.local.ibkr-service
sudo launchctl kickstart -k system/com.local.ibkr-service
```

To deploy, copy the five `.py` files to `app/` and run the `kickstart` line. If the plist changes, use `bootout` then `bootstrap`. Replace `__USER__` and `__HOME__` in the plist template before installing it.

## Timing

| Data | Refreshed | Failure |
|---|---|---|
| Flex statement | Every 12 hours, and hourly while the statement is 1 to 4 days older than the last weekday (UTC, holidays not known) | Retried hourly; the last statement stays, marked |
| Advisor watchlist prices and their FX | Every 30 minutes (within 5 minutes for a symbol new to the list), only while `advisor_core` is set | The last prices stay; the reply counts how many are recent |
| Prices and FX | Every 15 minutes | Falls back to IBKR's close and statement rate after 4 days, or when the statement is newer |
| 5-year history | Every 20 hours | Retried hourly, paused an hour after a 429; the last history stays, with its end date |

## Limits

- Holdings update only from the statement, which covers days already closed. A trade shows within an hour of IBKR publishing that day's statement. The one time this was observed, Friday's statement appeared between 04:23 and 07:04 UTC on Saturday.
- Public prices are delayed and unofficial. A quote in the wrong currency or over 25% from IBKR's last price is rejected, and that holding's history goes unused too. This also rejects real moves that large until the next statement.
- Yahoo sees the ticker symbols and this machine's IP address. It never sees quantities or cost.
- `base_currency` only fills in when the statement has no base. It can't convert an account's figures.
- Shares are quantity x price. Other assets use IBKR's Position Value, or stay unknown without it, so they update daily. Futures and CFDs are listed but left out of every total.
- Each statement is checked against itself: holdings plus cash more than 2% from IBKR's account value prints a Data gap.
- A recent split can undervalue a holding until the next statement if the price moved less than 25%.
- Gains in the base currency use today's FX rate, so they leave out past currency moves.
- History-based figures describe today's mix as if held constant. They aren't the account's real performance.
- No sector data, earnings dates, backtested rules or scoring of past ranges yet.
