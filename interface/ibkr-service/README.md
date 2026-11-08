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

`base_currency`, `benchmark` (default `VT`), `symbol_overrides` (ISIN or IBKR symbol to Yahoo symbol), `max_position_pct` (20), `loss_review_pct` (-30), `gain_review_pct` (100), `gain_review_weight_pct` (10), `drawdown_review_pct` (-25), `correlation_review` (0.7).

## Running it

`com.local.ibkr-service` is a system LaunchDaemon with `RunAtLoad` and `KeepAlive`, so it starts at boot and restarts after a crash. `start-ibkr-service.sh` waits for the tailnet address and binds only that.

```
sudo launchctl print system/com.local.ibkr-service
sudo launchctl kickstart -k system/com.local.ibkr-service
```

To deploy, copy the four `.py` files to `app/` and run the `kickstart` line. If the plist changes, use `bootout` then `bootstrap`. Replace `__USER__` and `__HOME__` in the plist template before installing it.

## Timing

| Data | Refreshed | Failure |
|---|---|---|
| Flex statement | Every 12 hours, and hourly while the statement is 1 to 4 days older than the last weekday (UTC, holidays not known) | Retried hourly; the last statement stays, marked |
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
