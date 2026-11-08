#!/usr/bin/env python3
"""ibkr-service: one machine's read-only view of an IBKR account, served to the agent over the tailnet.

The agent stays on Tor and holds no IBKR secret. This process uses the normal internet and a Flex token.
It can't trade: there is no order code, Flex only hands out reports, and every route here is a GET.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import analytics
import flex
import market

log = logging.getLogger("ibkr-service")

FLEX_EVERY = 12 * 3600
FLEX_RETRY = 3600
PRICES_EVERY = 15 * 60
HISTORY_EVERY = 20 * 3600
HISTORY_RETRY = 3600
RATE_LIMIT_PAUSE = 3600
FLEX_STALE_DAYS = 4
TICK = 30
DEFAULT_CONFIG = {
    "flex_query_id": "", "base_currency": None, "benchmark": "VT", "symbol_overrides": {},
    "max_position_pct": 20, "loss_review_pct": -30, "gain_review_pct": 100, "gain_review_weight_pct": 10,
    "drawdown_review_pct": -25, "correlation_review": 0.7,
}
NOTE_NOT_ADVICE = ("These are prompts for your own judgement from fixed rules, not financial advice. "
                   "The service doesn't know your tax position or your plans.")


def default_get(url: str, params: dict) -> str:
    """Browser TLS: Yahoo answers 429 to plain clients. IBKR accepts it too."""
    from curl_cffi import requests
    response = requests.get(url, params=params, impersonate="chrome", timeout=40)
    if response.status_code == 429:
        raise market.RateLimited("HTTP 429")
    if response.status_code != 200:
        raise OSError(f"HTTP {response.status_code}")
    return response.text


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)


def scrub(text, *secrets: str) -> str:
    """A library error can quote the request URL, and the URL carries the token."""
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[hidden]")
    return re.sub(r"([?&](?:t|q)=)[^&\s'\"]+", r"\1[hidden]", text)


#: Where a limit must lie, so a mistyped value can't turn a rule upside down or switch it off.
LIMIT_RANGES = {"max_position_pct": (1, 100), "loss_review_pct": (-100, -1), "gain_review_pct": (1, 10000),
                "gain_review_weight_pct": (1, 100), "drawdown_review_pct": (-100, -1), "correlation_review": (0.01, 1)}


def clean_config(raw) -> dict:
    """Defaults, plus only those settings that are valid. A bad value can't stop the service, and `rejected`
    names a bad limit in the status and review views instead of it being dropped silently."""
    config = {**DEFAULT_CONFIG, "rejected": []}
    if not isinstance(raw, dict):
        return config
    for name, default in DEFAULT_CONFIG.items():
        value = raw.get(name)
        if name == "base_currency":
            config[name] = value.strip().upper() if isinstance(value, str) and value.strip() else None
        elif name == "flex_query_id":
            config[name] = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
        elif isinstance(default, dict):
            config[name] = {str(k): v for k, v in value.items() if isinstance(v, str)} if isinstance(value, dict) else default
        elif isinstance(default, str):
            config[name] = value if isinstance(value, str) and value else default
        elif name in raw:
            low, high = LIMIT_RANGES[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and low <= value <= high:
                config[name] = value
            else:
                config["rejected"].append(name)
    return config


def _load_pairs(raw) -> list:
    return [(str(pair[0]), float(pair[1])) for pair in raw]


class Service:
    def __init__(self, home: Path, token_file: Path, get=default_get, clock=time.time, sleep=time.sleep):
        self.home, self.token_file, self.get, self.clock, self.sleep = Path(home), Path(token_file), get, clock, sleep
        self.lock = threading.RLock()
        self.history_dir = self.home / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_path = self.home / "state.json"
        self.state = self._load_state()
        self.histories: dict = {}
        for path in self.history_dir.glob("*.json"):
            try:
                self.histories[path.stem.replace("__", "=")] = _load_pairs(json.loads(path.read_text()))
            except (OSError, ValueError, TypeError, IndexError, KeyError):
                log.warning("a history file is unreadable and will be fetched again")

    def _load_state(self) -> dict:
        blank = {"flex": None, "flex_ok_at": 0, "flex_try_at": 0, "flex_error": None, "quotes": {}, "fx": {},
                 "prices_at": 0, "price_ok": 0, "price_wanted": 0, "price_errors": {}, "history_at": {},
                 "history_try_at": {}, "history_pause_until": 0, "credentials_seen": None, "flex_due_now": False}
        try:
            loaded = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            return blank
        except (OSError, ValueError):
            log.warning("state file unreadable, starting empty")
            return blank
        if not isinstance(loaded, dict):
            log.warning("state file has the wrong shape, starting empty")
            return blank
        state = {**blank, **loaded}
        now = self.clock()
        for name in ("flex_ok_at", "flex_try_at", "prices_at", "history_pause_until"):  # a clock jump can't block refreshes
            value = state[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value > now + 86400:
                state[name] = 0
        for name in ("quotes", "fx", "price_errors", "history_at", "history_try_at"):
            if not isinstance(state[name], dict):
                state[name] = {}
        if state["flex"] is not None and not isinstance(state["flex"], dict):
            state["flex"] = None
        return state

    def _save(self) -> None:
        atomic_write(self.state_path, json.dumps(self.state))

    def config(self) -> dict:
        try:
            return clean_config(json.loads((self.home / "config.json").read_text()))
        except FileNotFoundError:
            return clean_config({})
        except (OSError, ValueError) as exc:
            log.warning("config.json unreadable, using defaults: %s", type(exc).__name__)
            return clean_config({})

    def token(self) -> str:
        try:
            return self.token_file.read_text().strip()
        except OSError:
            return ""

    # ------------------------------------------------------------------ refresh

    def refresh_flex(self, cfg: dict) -> None:
        token, query = self.token(), str(cfg.get("flex_query_id") or "").strip()
        now = self.clock()
        with self.lock:
            self.state["flex_try_at"] = now
        try:
            text = flex.fetch_statement(token, query, self.get, sleep=self.sleep)
            parsed = flex.parse_statement(text)
        except Exception as exc:  # noqa: BLE001 - a failed fetch keeps the last good statement
            message = scrub(exc, token, query)
            log.warning("flex refresh did not complete: %s", message)
            with self.lock:
                self.state["flex_error"] = message
                self.state["flex_due_now"] = False
                self._save()
            return
        with self.lock:
            self.state.update(flex=parsed, flex_ok_at=now, flex_error=None, flex_due_now=False)
            self._save()
        log.info("flex refreshed: statement %s", parsed["report_date"])

    def _base(self, cfg: dict) -> str | None:
        """The statement's own base wins: every cash and account figure is in it. The setting only fills a gap."""
        data = self.state.get("flex") or {}
        return data.get("base") or cfg.get("base_currency")

    def symbol_map(self, cfg: dict) -> dict:
        positions = (self.state.get("flex") or {}).get("positions") or []
        return {p.get("key"): market.yahoo_symbol(p, cfg.get("symbol_overrides")) for p in positions}

    def refresh_prices(self, cfg: dict) -> None:
        quotes, errors, fx = dict(self.state["quotes"]), {}, dict(self.state["fx"])
        positions = (self.state.get("flex") or {}).get("positions") or []
        base = self._base(cfg)
        wanted = sorted({s for s in self.symbol_map(cfg).values() if s})
        pairs = sorted({p["currency"] for p in positions if base and p.get("currency") not in ("", base)})
        names, done, limited = wanted + [f"{c}{base}=X" for c in pairs], 0, False
        for name in names:
            if limited:
                errors[name] = "not tried: rate limited"
                continue
            try:
                quote = market.fetch_chart(name, self.get, json.loads)
            except Exception as exc:  # noqa: BLE001 - one symbol must not stop the others
                errors[name] = scrub(exc)[:160]
                limited = isinstance(exc, market.RateLimited)
                continue
            done += 1
            if name.endswith("=X"):
                fx[name[:-2]] = {"rate": quote["price"], "at": quote["at"]}
            else:
                quote.pop("closes", None)
                quotes[name] = quote
            self.sleep(0.3)
        with self.lock:
            self.state.update(quotes=quotes, fx=fx, price_errors=errors, prices_at=self.clock(),
                              price_ok=done, price_wanted=len(names))
            self._save()

    def refresh_history(self, cfg: dict) -> None:
        now = self.clock()
        if now < self.state["history_pause_until"]:
            return
        names = {s for s in self.symbol_map(cfg).values() if s} | {cfg["benchmark"]}
        for name in sorted(names):
            if now - self.state["history_at"].get(name, 0) < HISTORY_EVERY:
                continue
            if now - self.state["history_try_at"].get(name, 0) < HISTORY_RETRY:
                continue
            with self.lock:
                self.state["history_try_at"][name] = now
            try:
                closes = market.fetch_chart(name, self.get, json.loads, span="5y")["closes"]
            except Exception as exc:  # noqa: BLE001 - the last good history stays
                log.warning("a price history did not arrive: %s", type(exc).__name__)
                limited = isinstance(exc, market.RateLimited)
                with self.lock:
                    if limited:
                        self.state["history_pause_until"] = now + RATE_LIMIT_PAUSE
                    self._save()
                if limited:
                    return
                continue
            if closes:
                atomic_write(self.history_dir / (name.replace("=", "__") + ".json"), json.dumps(closes))
                with self.lock:
                    self.histories[name] = [tuple(pair) for pair in closes]
                    self.state["history_at"][name] = now
                    self._save()
            self.sleep(0.3)

    def _note_new_credentials(self, cfg: dict) -> None:
        """New credentials from setup-flex.sh are tried now, not after the failure or refresh interval."""
        try:
            stamp = self.token_file.stat().st_mtime
        except OSError:
            stamp = 0
        seen = hashlib.sha256(f"{stamp}|{cfg.get('flex_query_id')}".encode()).hexdigest()
        with self.lock:
            if seen != self.state["credentials_seen"]:
                self.state.update(credentials_seen=seen, flex_due_now=True)
                self._save()

    def _statement_behind(self, now: float) -> bool:
        """IBKR publishes a day's statement some time after that day ends, so a fetch can land just before it.
        Older than the last weekday means a newer one is due. A market holiday only costs some hourly fetches.
        Further behind than FLEX_STALE_DAYS, the date is not advancing (a fixed query period, a closed account),
        so this stops and the 12-hour interval and the STALE label take over."""
        report_date = (self.state.get("flex") or {}).get("report_date")
        if not report_date or not isinstance(report_date, str):  # undated: a newer one could not be recognised either
            return False
        day = datetime.fromtimestamp(now, timezone.utc).date() - timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return (day - timedelta(days=FLEX_STALE_DAYS)).isoformat() <= report_date < day.isoformat()

    def tick(self) -> None:
        cfg = self.config()
        self._note_new_credentials(cfg)
        now = self.clock()
        if self.token() and cfg.get("flex_query_id"):
            wanted = now - self.state["flex_ok_at"] >= FLEX_EVERY or self._statement_behind(now)
            due = wanted and now - self.state["flex_try_at"] >= FLEX_RETRY
            if due or self.state["flex_due_now"]:
                self.refresh_flex(cfg)
        if self.state.get("flex"):
            if self.clock() - self.state["prices_at"] >= PRICES_EVERY:
                self.refresh_prices(cfg)
            self.refresh_history(cfg)

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - never let one bad cycle end the service
                log.exception("refresh cycle stopped early")
            stop.wait(TICK)

    # ------------------------------------------------------------------ views

    def valued(self, cfg: dict) -> dict:
        return analytics.value_positions(self.state["flex"], self.state["quotes"], self.state["fx"],
                                         self.symbol_map(cfg), self._base(cfg), market.quote_problem, self.clock())

    def _limits(self, cfg: dict) -> dict:
        return {"max_position": cfg["max_position_pct"] / 100, "loss_review": cfg["loss_review_pct"] / 100,
                "gain_review": cfg["gain_review_pct"] / 100, "gain_review_weight": cfg["gain_review_weight_pct"] / 100,
                "drawdown_review": cfg["drawdown_review_pct"] / 100, "correlation_review": cfg["correlation_review"]}

    def _money(self, value, currency: str | None) -> str:
        return "unknown" if value is None else f"{value:,.2f} {currency or '(currency unknown)'}"

    def _price(self, value: float) -> str:
        return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"

    def _quantity(self, value: float) -> str:
        return f"{value:,.4f}".rstrip("0").rstrip(".")

    def _when(self, stamp) -> str:
        return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if stamp else "never"

    def _freshness(self, cfg: dict) -> list:
        data, now, lines = self.state["flex"], self.clock(), []
        report_date = data.get("report_date")
        lines.append(f"Statement date: {report_date or 'unknown'} (fetched {self._when(self.state['flex_ok_at'])}).")
        age = analytics.days_old(report_date, now) if report_date else None
        if age is None or age > FLEX_STALE_DAYS:
            lines.append("STALE: the statement is several days old or undated, so holdings may have changed.")
        if self.state.get("flex_error"):
            lines.append(f"Last statement refresh did not complete: {self.state['flex_error']}")
        if not data.get("base") and not data.get("mixed_base") and cfg.get("base_currency"):
            lines.append(f"Data gap: the base currency {cfg['base_currency']} comes from config.json, and the statement doesn't confirm it.")
        if data.get("base") and cfg.get("base_currency") and cfg["base_currency"] != data["base"]:
            lines.append(f"Data gap: base_currency in config.json ({cfg['base_currency']}) differs from the statement's "
                         f"{data['base']}. The statement's is used.")
        for warning in data.get("warnings") or []:
            lines.append(f"Data gap: {warning}.")
        if self.state["price_ok"] < self.state["price_wanted"]:
            lines.append(f"Data gap: the last price round got {self.state['price_ok']} of {self.state['price_wanted']} "
                         "public prices and rates, so some figures use older ones.")
        return lines

    def _price_line(self, row: dict) -> str:
        if row.get("price") is None:
            return "price unknown"
        if row["source"] == "public":
            return f"{self._price(row['price'])} {row['currency']} (public, delayed, {self._when(row.get('price_at'))})"
        return f"{self._price(row['price'])} {row['currency']} (IBKR close {row.get('price_date')}; {row.get('price_note')})"

    def _position_line(self, row: dict, base: str | None) -> str:
        if row.get("notional"):
            return f"{row['symbol']}: {self._quantity(row['qty']) if row.get('qty') is not None else 'quantity unknown'} {row['asset']} contract(s), last price {self._price_line(row)}"
        parts = [f"{row['symbol']}: {self._quantity(row['qty'])} x {self._price_line(row)}" if row.get("qty") is not None
                 else f"{row['symbol']}: quantity unknown"]
        if row.get("value_base") is not None:
            parts.append(f"value {self._money(row['value_base'], base)}")
        if row.get("weight") is not None:
            parts.append(f"{row['weight']:.1%} of holdings")
        if row.get("gain_local") is not None:
            pct = f" ({row['gain_pct']:+.1%})" if row.get("gain_pct") is not None else ""
            parts.append(f"gain {row['gain_local']:+,.2f} {row['currency']}{pct}")
        else:
            parts.append("gain unknown")
        if row.get("move_pct") is not None:
            parts.append(f"{row['move_label']} {row['move_pct']:+.2%}")
        if row.get("fx_note"):
            parts.append(f"FX: {row['fx_note']}")
        return " | ".join(parts)

    def _summary(self, cfg: dict) -> str:
        valued, data = self.valued(cfg), self.state["flex"]
        base = valued["base"]
        missing = valued["incomplete"]
        partial = f" (PARTIAL: leaves out {', '.join(missing)})" if missing else ""
        lines = ["PORTFOLIO (all figures computed by the service; report them as they are)",
                 f"Holdings: {self._money(valued['invested'], base)}{partial}",
                 f"Cash: {self._money(valued['cash'], base)}",
                 f"Total: {self._money(valued['total'], base)}{partial if valued['total'] is not None else ''}"]
        if valued["gain"] is not None:
            left_out = f"; leaves out {', '.join(valued['gain_unknown'])}" if valued["gain_unknown"] else ""
            lines.append(f"Gain against cost: {self._money(valued['gain'], base)} ({valued['gain_pct']:+.1%}), "
                         f"at today's FX, so it leaves out past currency moves{left_out}")
        if valued["day"] is not None:
            lines.append(f"Today, only holdings that have traded today: {self._money(valued['day'], base)} ({valued['day_pct']:+.2%})")
        nav = data.get("nav") or []
        if nav:
            lines.append(f"Account value on IBKR's statement {nav[-1][0]}: {self._money(nav[-1][1], base)}")
            if len(nav) > 1:
                lines.append(f"The same statement on {nav[0][0]}: {self._money(nav[0][1], base)} "
                             "(deposits and withdrawals are included in this change)")
        if data.get("dividends_ytd") is not None:
            lines.append(f"Dividends from {data.get('dividends_from')} to {data.get('report_date')}: "
                         f"{self._money(data['dividends_ytd'], base)}, withholding tax {self._money(data['withholding_ytd'], base)}")
        lines.extend(self._position_line(row, base) for row in valued["positions"])
        if valued["notional"]:
            lines.append("Not in any total, because a contract's size isn't money held: " + ", ".join(valued["notional"]))
        return "\n".join(lines + self._freshness(cfg))

    def _holding_rows(self, valued: dict, symbol: str) -> list:
        """IBKR writes BRK B, Yahoo BRK-B and people BRK.B: compare without separators, and accept ISIN too."""
        fold = lambda text: re.sub(r"[^A-Z0-9]", "", str(text or "").upper())  # noqa: E731
        wanted = fold(symbol)
        return [r for r in valued["positions"] if wanted and wanted in (fold(r.get("symbol")), fold(r.get("isin")), fold(r.get("yahoo")))]

    def _not_held(self, valued: dict, symbol: str, cfg: dict) -> str:
        held = ", ".join(r["symbol"] for r in valued["positions"]) or "none"
        return "\n".join([f"PORTFOLIO: no holding matched {symbol.strip().upper()} on this statement. Holdings: {held}."] + self._freshness(cfg))

    def _position(self, cfg: dict, symbol: str) -> str:
        valued = self.valued(cfg)
        rows = self._holding_rows(valued, symbol)
        if not rows:
            return self._not_held(valued, symbol, cfg)
        lines = []
        for row in rows:
            lines += [self._position_line(row, valued["base"]), f"Name: {row.get('description')}"]
            if row.get("cost_price") is not None:
                lines.append(f"Average cost: {self._price(row['cost_price'])} {row['currency']}")
        symbols = {r["symbol"] for r in rows}
        trades = [t for t in self.state["flex"].get("trades") or [] if t["symbol"] in symbols]
        lines.extend(f"Trade {t['date']}: {t['side']} {self._quantity(t['qty'])} at {self._price(t['price'])} {t['currency']}"
                     for t in trades if t.get("qty") is not None and t.get("price") is not None)
        return "\n".join(lines + self._freshness(cfg))

    def _health_data(self, cfg: dict, valued: dict) -> dict:
        return analytics.portfolio_health(valued, self.histories, cfg["benchmark"],
                                          self.histories.get(cfg["benchmark"]) or [], self.clock())

    def _health(self, cfg: dict) -> str:
        valued = self.valued(cfg)
        health = self._health_data(cfg, valued)
        lines = [f"PORTFOLIO HEALTH: {health['count']} holdings valued"]
        if valued["incomplete"]:
            lines.append("Left out of these figures, value unknown: " + ", ".join(valued["incomplete"]))
        if health["largest"]:
            lines.append(f"Largest: {health['largest']['symbol']} at {health['largest']['weight']:.1%} "
                         f"(your limit {cfg['max_position_pct']}%). Top 3 together: {health['top3']:.1%}.")
            lines.append("Currencies: " + ", ".join(f"{c} {w:.1%}" for c, w in health["currencies"]))
        if "volatility" in health:
            stale = " STALE: the shared history stops well before today." if health["history_stale"] else ""
            lines.append(f"How today's mix would have behaved from {health['history_from']} to {health['history_to']} "
                         f"({health['coverage']:.0%} of long holdings have enough history; hypothetical, in each "
                         f"holding's own currency, dividends included).{stale}")
            if health["volatility"] is not None:
                lines.append(f"Yearly volatility {health['volatility']:.1%}. Worst fall from a high {health['drawdown']:.1%}.")
            if health.get("correlation") is not None:
                lines.append(f"Average correlation of recent weekly moves between holdings with enough history: {health['correlation']:.2f}.")
            versus = health.get("versus")
            if versus:
                lines.append(f"Over {versus['days']} trading days from {versus['from']} to {versus['to']}: this mix "
                             f"{versus['mix']:+.1%}, {versus['benchmark']} {versus['index']:+.1%}.")
        else:
            lines.append("Price history is too short so far for volatility, falls or comparison with an index.")
        lines.append("Sector exposure isn't available from these sources.")
        return "\n".join(lines + self._freshness(cfg))

    def _forecast(self, cfg: dict, symbol: str) -> str:
        valued, now = self.valued(cfg), self.clock()
        wanted = symbol.strip().upper()
        if wanted:
            rows = self._holding_rows(valued, symbol)
            if not rows:
                return self._not_held(valued, symbol, cfg)
            history = next((self.histories[r["history_symbol"]] for r in rows if self.histories.get(r.get("history_symbol"))), [])
            closes = [p for _, p in history]
            last = history[-1][0] if history else None
            name, basis = wanted, f"{wanted} closes with dividends reinvested"
        else:
            longs = [r for r in valued["positions"] if r.get("weight") and (r.get("qty") or 0) > 0]
            weights: dict = {}
            for row in longs:
                if row.get("history_symbol"):
                    weights[row["history_symbol"]] = weights.get(row["history_symbol"], 0.0) + row["weight"]
            blend = analytics.blended_series(self.histories, weights, sum(r["weight"] for r in longs))
            closes = blend["values"] if blend else []
            last = blend["dates"][-1] if blend else None
            name = "your current mix"
            basis = f"today's weights held constant, {blend['coverage']:.0%} of long holdings covered" if blend else ""
        tail = (["Left out, value unknown: " + ", ".join(valued["incomplete"])] if valued["incomplete"] and not wanted else []) + self._freshness(cfg)
        rows = analytics.bands(closes)
        if not rows:
            return "\n".join([f"PORTFOLIO RANGES: not enough price history for {name} (a horizon needs at least twice its "
                              "length). Give no forecast for it."] + tail)
        lines = [f"PORTFOLIO RANGES for {name}: what past periods of the same length returned, based on {basis}, up to {last}."]
        age = analytics.days_old(last, now)
        if age is None or age > analytics.HISTORY_STALE_DAYS:
            lines.append("STALE: this history stops well before today.")
        lines.extend(f"{r['horizon']}: 8 in 10 periods between {r['low']:+.1%} and {r['high']:+.1%}, middle {r['mid']:+.1%} "
                     f"(about {r['periods']} separate periods)" for r in rows)
        lines.append("This is a description of the past, not a prediction. The real outcome can fall outside it, "
                     "and few separate periods means a rough figure.")
        return "\n".join(lines + tail)

    def _review(self, cfg: dict) -> str:
        valued = self.valued(cfg)
        items, unchecked = analytics.review_items(valued, self._health_data(cfg, valued), self.histories,
                                                  self._limits(cfg), self.clock())
        lines = ["PORTFOLIO REVIEW: none of the rules that could run is triggered, so there is no prompt to act."] if not items else \
                [f"PORTFOLIO REVIEW: {len(items)} item(s) to look at. None is an instruction to buy or sell."]
        for item in items:
            lines.append(f"- {item['why']} Ask: {item['ask']} On the other hand: {item['against']}")
        if valued["incomplete"]:
            lines.append("Share of holdings not checked, value unknown: " + ", ".join(valued["incomplete"]))
        for rule, symbols in unchecked.items():
            lines.append(f"Not checked for {rule}, data missing: {', '.join(symbols)}")
        if cfg.get("rejected"):
            lines.append("Settings ignored as invalid, so their defaults are used: " + ", ".join(cfg["rejected"]))
        lines.append(NOTE_NOT_ADVICE)
        return "\n".join(lines + self._freshness(cfg))

    def _status(self, cfg: dict) -> str:
        data = self.state.get("flex") or {}
        return "\n".join([
            "PORTFOLIO SERVICE",
            f"Flex token present: {'yes' if self.token() else 'no'}. Query id present: {'yes' if cfg.get('flex_query_id') else 'no'}.",
            f"Statement date: {data.get('report_date') or 'none yet'}. Fetched: {self._when(self.state['flex_ok_at'])}.",
            f"Last statement error: {self.state.get('flex_error') or 'none'}.",
            f"Holdings: {len(data.get('positions') or [])}. Last price round: {self._when(self.state['prices_at'])}, "
            f"{self.state['price_ok']} of {self.state['price_wanted']} arrived.",
            "Price errors: " + (", ".join(self.state["price_errors"]) or "none") + ".",
            "Settings ignored as invalid: " + (", ".join(cfg.get("rejected") or []) or "none") + ".",
            f"Histories held: {len(self.histories)}.",
        ])

    def answer(self, view: str, symbol: str = "") -> tuple:
        """(ok, text) for one read-only view."""
        with self.lock:
            cfg = self.config()
            if view == "status":
                return True, self._status(cfg)
            if view not in ("summary", "position", "health", "forecast", "review"):
                return False, f"unknown view {view!r}"
            if not self.state.get("flex"):
                if not (self.token() and cfg.get("flex_query_id")):
                    return False, "the IBKR service is not set up yet: its Flex token and query id are missing on the service machine"
                return False, f"no IBKR statement has arrived yet. Last error: {self.state.get('flex_error') or 'none, the first fetch is due'}"
            if view == "summary":
                return True, self._summary(cfg)
            if not self.state["flex"].get("positions_known", True):
                return True, "\n".join(["PORTFOLIO: holdings are unknown, because the Flex query has no Open Positions section. Give no holding figures."] + self._freshness(cfg))
            if view == "position":
                return (True, self._position(cfg, symbol)) if symbol.strip() else (False, "position needs a symbol")
            if view == "health":
                return True, self._health(cfg)
            if view == "forecast":
                return True, self._forecast(cfg, symbol)
            return True, self._review(cfg)


def make_handler(service: Service, key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ibkr-service"

        def _send(self, code: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802 - the only verb: this service changes nothing
            supplied = self.headers.get("X-Ibkr-Service-Key", "")
            if not key or not hmac.compare_digest(supplied.encode(), key.encode()):
                return self._send(401, {"ok": False, "text": "unauthorized"})
            url = urlparse(self.path)
            match = re.fullmatch(r"/v1/(\w+)", url.path)
            if not match:
                return self._send(404, {"ok": False, "text": "not found"})
            symbol = (parse_qs(url.query).get("symbol") or [""])[0][:32]
            try:
                ok, text = service.answer(match.group(1), symbol)
            except Exception as exc:  # noqa: BLE001 - say so rather than drop the connection
                log.exception("a view did not complete")
                return self._send(500, {"ok": False, "text": f"the service hit an internal problem ({type(exc).__name__})"})
            return self._send(200, {"ok": ok, "text": text})

        def log_message(self, fmt, *args):  # routes only: no symbols and no key in the log
            log.debug("%s %s", self.address_string(), urlparse(self.path).path)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="the tailnet address to bind; never a public one")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--home", default="~/Data/AI/ibkr-service/data")
    parser.add_argument("--token-file", default="~/Data/AI/.ibkr_flex_token")
    parser.add_argument("--key-file", default="~/Data/AI/.ibkr_service_key")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    home = Path(args.home).expanduser()
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        raise SystemExit("the service key file is empty")
    service = Service(home, Path(args.token_file).expanduser())
    stop = threading.Event()
    threading.Thread(target=service.run_forever, args=(stop,), daemon=True, name="refresh").start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, key))
    log.info("listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
