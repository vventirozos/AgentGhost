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

import advisor
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
WATCH_EVERY = 30 * 60
MAX_AMOUNT = 10_000_000
FLEX_STALE_DAYS = 4
TICK = 30
DEFAULT_CONFIG = {
    "flex_query_id": "", "base_currency": None, "benchmark": "VT", "symbol_overrides": {},
    # max_fund_pct 100: no fund is a size prompt by default; the review's fund line says why instead (§4TI-d)
    "max_position_pct": 20, "max_fund_pct": 100, "loss_review_pct": -30, "gain_review_pct": 100, "gain_review_weight_pct": 10,
    "drawdown_review_pct": -25, "correlation_review": 0.7,
    # §4VH the buy advisor. advisor_core: [{"symbol", "target", "yahoo"?, "safe"?}], empty until the owner sets it
    "advisor_core": [], "advisor_kept": [], "watchlist": list(advisor.DEFAULT_WATCHLIST), "opportunity_pct": 10, "max_drift_pts": 25,
    "hot_z": -2.0, "hot_fall_pct": -10, "hot_funds_only": True, "opportunity_cap_slices": 3,
}
#: Yahoo suffix -> the exchange's name as a broker shows it (§4VH-f: "XCS6.DE" means nothing to a new investor).
EXCHANGE_NAMES = {".DE": "Xetra", ".MI": "Borsa Italiana (Milan)", ".AS": "Euronext Amsterdam", ".PA": "Euronext Paris",
                  ".L": "London Stock Exchange", ".SW": "SIX Swiss Exchange"}


def exchange(yahoo: str | None) -> str:
    """' on Xetra (ticker XCS6)' for a known Yahoo suffix, else ''."""
    for suffix, name in EXCHANGE_NAMES.items():
        if yahoo and yahoo.upper().endswith(suffix.upper()):
            return f" on {name} (ticker {yahoo[:-len(suffix)]})"
    return ""


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
LIMIT_RANGES = {"max_position_pct": (1, 100), "max_fund_pct": (1, 100), "loss_review_pct": (-100, -1), "gain_review_pct": (1, 10000),
                "gain_review_weight_pct": (1, 100), "drawdown_review_pct": (-100, -1), "correlation_review": (0.01, 1),
                "opportunity_pct": (0, 50), "max_drift_pts": (1, 100), "hot_z": (-10, -0.5), "hot_fall_pct": (-90, -1),
                "opportunity_cap_slices": (0, 24)}
_YAHOO_RE = re.compile(r"[A-Za-z0-9.\-=^]{1,20}")


def _clean_core(value):
    """The core funds, or None when the setting is malformed (then it is named as rejected, never half-used)."""
    if not isinstance(value, list) or len(value) > 10:
        return None
    core = []
    for item in value:
        if not isinstance(item, dict):
            return None
        symbol, target, yahoo = item.get("symbol"), item.get("target"), item.get("yahoo")
        if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 32:
            return None
        if not isinstance(target, (int, float)) or isinstance(target, bool) or not math.isfinite(target) or target <= 0:
            return None
        if yahoo is not None and not (isinstance(yahoo, str) and _YAHOO_RE.fullmatch(yahoo)):
            return None
        if "safe" in item and not isinstance(item["safe"], bool):  # "true" or 1 would silently mean not safe
            return None
        core.append({"symbol": symbol.strip(), "target": float(target), "yahoo": yahoo, "safe": item.get("safe") is True})
    if len({c["symbol"].upper() for c in core}) != len(core) or sum(c["safe"] for c in core) > 1:
        return None
    return core


def _clean_kept(value):
    """Holdings the owner keeps but no longer buys (an earlier plan's funds): named as kept, never as a symbol typo."""
    if not isinstance(value, list) or len(value) > 30 or not all(isinstance(v, str) and 0 < len(v.strip()) <= 32 for v in value):
        return None
    return [v.strip() for v in value]


def _clean_watchlist(value):
    if not isinstance(value, list) or len(value) > 60 or not all(isinstance(v, str) and _YAHOO_RE.fullmatch(v) for v in value):
        return None
    return list(dict.fromkeys(v.upper() for v in value))


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
        elif name in ("advisor_core", "watchlist", "advisor_kept"):
            if name in raw:
                cleaned = {"advisor_core": _clean_core, "watchlist": _clean_watchlist, "advisor_kept": _clean_kept}[name](value)
                if cleaned is None:
                    config["rejected"].append(name)
                else:
                    config[name] = cleaned
            else:
                config[name] = list(default)
        elif isinstance(default, bool):
            if name in raw:
                if isinstance(value, bool):
                    config[name] = value
                else:
                    config["rejected"].append(name)
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
                 "history_try_at": {}, "history_pause_until": 0, "credentials_seen": None, "flex_due_now": False,
                 "watch_quotes": {}, "watch_at": 0, "watch_ok_at": 0}
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
        for name in ("flex_ok_at", "flex_try_at", "prices_at", "history_pause_until", "watch_at", "watch_ok_at"):  # a clock jump can't block refreshes
            value = state[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value > now + 86400:
                state[name] = 0
        for name in ("quotes", "fx", "price_errors", "history_at", "history_try_at", "watch_quotes"):
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
        names = {s for s in self.symbol_map(cfg).values() if s} | {cfg["benchmark"]} | set(self.watch_symbols(cfg))
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

    def watch_symbols(self, cfg: dict) -> list:
        """The advisor's Yahoo symbols: the watchlist and every core fund's own symbol (§4VH). None until the owner
        sets advisor_core: an unused advisor sends Yahoo no extra requests."""
        if not cfg["advisor_core"]:
            return []
        fold = lambda text: re.sub(r"[^A-Z0-9]", "", str(text or "").upper())  # noqa: E731 - as _holding_rows
        wanted = {fold(c["symbol"]) for c in cfg["advisor_core"]}
        held = [y for p in (self.state.get("flex") or {}).get("positions") or []
                if wanted & {fold(p.get("symbol")), fold(p.get("isin"))}
                and (y := market.yahoo_symbol(p, cfg.get("symbol_overrides")))]
        return list(dict.fromkeys([*cfg["watchlist"], *[c["yahoo"] for c in cfg["advisor_core"] if c.get("yahoo")], *held]))

    def refresh_watch(self, cfg: dict) -> None:
        """Latest prices for the advisor's symbols, and FX for their currencies. Kept apart from the holdings'
        quotes, so a watchlist symbol never values a holding."""
        base = self._base(cfg)
        quotes, fx, limited, arrived = dict(self.state["watch_quotes"]), dict(self.state["fx"]), False, 0
        for name in self.watch_symbols(cfg):
            try:
                quote = market.fetch_chart(name, self.get, json.loads)
            except Exception as exc:  # noqa: BLE001 - one symbol must not stop the others
                limited = isinstance(exc, market.RateLimited)
                if limited:
                    break
                continue
            quote.pop("closes", None)
            quote["fetched"] = self.clock()  # "at" is market time: a weekend's quotes are days old though fetched now
            quotes[name] = quote
            arrived += 1
            self.sleep(0.3)
        if base and not limited:
            have = {p.get("currency") for p in (self.state.get("flex") or {}).get("positions") or []}
            for currency in sorted({q.get("currency") for q in quotes.values() if q.get("currency")} - have - {base}):
                try:
                    quote = market.fetch_chart(f"{currency}{base}=X", self.get, json.loads)
                except Exception:  # noqa: BLE001 - the rate stays unknown and the reply says so
                    continue
                fx[f"{currency}{base}"] = {"rate": quote["price"], "at": quote["at"]}
                self.sleep(0.3)
        with self.lock:
            # merge, never replace: the holdings' round may have written fresher rates meanwhile
            self.state["fx"] = {**self.state["fx"], **{k: v for k, v in fx.items() if k not in self.state["fx"]
                                                       or v["at"] > self.state["fx"][k].get("at", 0)}}
            self.state.update(watch_quotes=quotes, watch_at=self.clock())
            if arrived:
                self.state["watch_ok_at"] = self.clock()
            self._save()

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
            missing = any(name not in self.state["watch_quotes"] for name in self.watch_symbols(cfg))
            if self.clock() - self.state["watch_at"] >= WATCH_EVERY or (missing and self.clock() - self.state["watch_at"] >= 300):
                self.refresh_watch(cfg)  # a symbol new to the watchlist is fetched within minutes, not half an hour
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
        return {"max_position": cfg["max_position_pct"] / 100, "max_fund": cfg["max_fund_pct"] / 100, "loss_review": cfg["loss_review_pct"] / 100,
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

    def _names(self, valued: dict, symbols: list) -> str:
        """A list of tickers as named lines: each with IBKR's name (§4TI-b)."""
        labels = {r.get("symbol"): r.get("label") or r.get("symbol") for r in valued["positions"]}
        return ", ".join(labels.get(s) or str(s) for s in symbols)

    def _position_line(self, row: dict, base: str | None) -> str:
        label = row.get("label") or analytics.label(row)
        if row.get("notional"):
            return f"{label}: {self._quantity(row['qty']) if row.get('qty') is not None else 'quantity unknown'} {row['asset']} contract(s), last price {self._price_line(row)}"
        parts = [f"{label}: {self._quantity(row['qty'])} x {self._price_line(row)}" if row.get("qty") is not None
                 else f"{label}: quantity unknown"]
        if row.get("value_base") is not None:
            parts.append(f"value {self._money(row['value_base'], base)}")
        if row.get("weight") is not None:
            parts.append(f"{row['weight']:.1%} of holdings")
        if row.get("gain_local") is not None:
            pct = f" ({row['gain_pct']:+.1%})" if row.get("gain_pct") is not None else ""
            if base and row["currency"] != base and row.get("gain_base") is not None:
                # §4TI-e: in the account's currency first, like every other money figure in the line
                parts.append(f"gain {row['gain_base']:+,.2f} {base} at today's FX ({row['gain_local']:+,.2f} "
                             f"{row['currency']}{', ' + pct.strip(' ()') if pct else ''})")
            else:
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
        partial = f" (PARTIAL: leaves out {self._names(valued, missing)})" if missing else ""
        lines = ["PORTFOLIO (all figures computed by the service; report them as they are)",
                 f"Holdings: {self._money(valued['invested'], base)}{partial}",
                 f"Cash: {self._money(valued['cash'], base)}",
                 f"Total: {self._money(valued['total'], base)}{partial if valued['total'] is not None else ''}"]
        if valued["gain"] is not None:
            left_out = f"; leaves out {self._names(valued, valued['gain_unknown'])}" if valued["gain_unknown"] else ""
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
            lines.append("Not in any total, because a contract's size isn't money held: " + self._names(valued, valued["notional"]))
        return "\n".join(lines + self._freshness(cfg))

    def _holding_rows(self, valued: dict, symbol: str) -> list:
        """IBKR writes BRK B, Yahoo BRK-B and people BRK.B: compare without separators, and accept ISIN too."""
        fold = lambda text: re.sub(r"[^A-Z0-9]", "", str(text or "").upper())  # noqa: E731
        wanted = fold(symbol)
        return [r for r in valued["positions"] if wanted and wanted in (fold(r.get("symbol")), fold(r.get("isin")), fold(r.get("yahoo")))]

    def _not_held(self, valued: dict, symbol: str, cfg: dict) -> str:
        held = ", ".join(r.get("label") or r["symbol"] for r in valued["positions"]) or "none"
        return "\n".join([f"PORTFOLIO: no holding matched {symbol.strip().upper()} on this statement. Holdings: {held}."] + self._freshness(cfg))

    def _position(self, cfg: dict, symbol: str) -> str:
        valued = self.valued(cfg)
        rows = self._holding_rows(valued, symbol)
        if not rows:
            return self._not_held(valued, symbol, cfg)
        lines = []
        for row in rows:
            lines.append(self._position_line(row, valued["base"]))
            if row.get("public_name") and row.get("description") and row["public_name"].upper() != row["description"].upper():
                lines.append(f"IBKR's name for it: {row['description']}")
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
            lines.append("Left out of these figures, value unknown: " + self._names(valued, valued["incomplete"]))
        if health["largest"]:
            big = health["largest"]
            limit = (f"a fund; your limit for one fund {cfg['max_fund_pct']}%" if analytics.is_fund(big)
                     else f"your limit {cfg['max_position_pct']}%")
            lines.append(f"Largest: {big.get('label') or big['symbol']} at {big['weight']:.1%} "
                         f"({limit}). Top 3 together: {health['top3']:.1%}.")
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
            # name the listing whose closes are used (§4TI-b review r2: rows[0] named one, another's history ran)
            row = next((r for r in rows if self.histories.get(r.get("history_symbol"))), rows[0])
            history = self.histories.get(row.get("history_symbol")) or []
            closes = [p for _, p in history]
            last = history[-1][0] if history else None
            name, basis = row.get("label") or wanted, f"{row['symbol']} closes with dividends reinvested"
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
        tail = (["Left out, value unknown: " + self._names(valued, valued["incomplete"])] if valued["incomplete"] and not wanted else []) + self._freshness(cfg)
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
        limits = self._limits(cfg)
        funds = [r for r in valued["positions"] if analytics.is_fund(r) and (r.get("weight") or 0) > limits["max_position"]
                 and r["weight"] <= limits["max_fund"]]
        if funds:
            lines.append(f"A note, not a prompt: funds are held to your limit for one fund ({cfg['max_fund_pct']}%, setting "
                         f"max_fund_pct), not the {cfg['max_position_pct']}% limit for one company: "
                         f"{self._names(valued, [r['symbol'] for r in funds])}. The service can't tell a broad index fund "
                         "from a one-sector or one-country fund; the fund's name says which it is.")
        if valued["incomplete"]:
            lines.append("Share of holdings not checked, value unknown: " + self._names(valued, valued["incomplete"]))
        for rule, symbols in unchecked.items():
            lines.append(f"Not checked for {rule}, data missing: {self._names(valued, symbols)}")
        if cfg.get("rejected"):
            lines.append("Settings ignored as invalid, so their defaults are used: " + ", ".join(cfg["rejected"]))
        lines.append(NOTE_NOT_ADVICE)
        return "\n".join(lines + self._freshness(cfg))

    def _fx_to_base(self, base: str | None):
        def rate(currency):
            if not base or not currency:
                return None
            if currency == base:
                return 1.0
            entry = self.state["fx"].get(f"{currency}{base}") or {}
            return entry.get("rate") if analytics._fresh(entry.get("at"), self.clock()) else None
        return rate

    def _advise(self, cfg: dict, amount: float, reserve: float, carry: float = 0.0) -> str:
        core = cfg["advisor_core"]
        if not core and "advisor_core" in (cfg.get("rejected") or []):
            return ("PORTFOLIO ADVICE: the advisor_core setting in config.json is invalid, so it is ignored (each fund needs "
                    "a symbol and a target above 0, symbols differ, at most one is safe). Give no split.")
        if not core:
            return ("PORTFOLIO ADVICE: no core funds are set. The owner sets advisor_core in config.json on the service "
                    "machine, for example [{\"symbol\": \"<ticker>\", \"target\": 60, \"safe\": true}, ...]. Give no split.")
        positions_known = self.state["flex"].get("positions_known", True)
        valued = self.valued(cfg) if positions_known else {"positions": [], "base": self._base(cfg)}
        base = valued["base"] or self._base(cfg)
        if not base:
            return "PORTFOLIO ADVICE: the account's base currency is unknown, so no amounts can be given."
        rows = {}
        for c in core:
            matched = self._holding_rows(valued, c["symbol"])
            if matched:
                # one fund on several listings is one holding: its value is the sum, and unknown when any part is
                # (review r2: taking the largest row dropped an unvalued listing and its drift gap)
                values = [r.get("value_base") for r in matched]
                priced = next((r for r in matched if r.get("price") is not None and r.get("fx")), matched[0])
                rows[c["symbol"]] = {**priced, "value_base": None if None in values else sum(values)}
        unmatched = [c["symbol"] for c in core if c["symbol"] not in rows] if positions_known else []
        matched_keys = {r.get("key") for c in core for r in self._holding_rows(valued, c["symbol"])}
        kept_rows = [r for k in cfg["advisor_kept"] for r in self._holding_rows(valued, k)]
        kept_keys = {r.get("key") for r in kept_rows}
        other_funds = [r.get("label") or r["symbol"] for r in valued["positions"]
                       if analytics.is_fund(r) and r.get("key") not in matched_keys | kept_keys and (r.get("qty") or 0) > 0]
        now = self.clock()
        quotes = {k: q for k, q in self.state["watch_quotes"].items() if analytics._fresh(q.get("at"), now)}
        settings = {"opportunity": cfg["opportunity_pct"] / 100, "max_drift": cfg["max_drift_pts"] / 100,
                    "hot_z": cfg["hot_z"], "hot_fall": cfg["hot_fall_pct"] / 100, "funds_only": cfg["hot_funds_only"],
                    "cap_slices": cfg["opportunity_cap_slices"]}
        a = advisor.advise(amount, reserve, core, rows, self.histories, cfg["watchlist"], quotes,
                           self._fx_to_base(base), base, settings, now, holdings_known=positions_known, carry=carry)
        m = lambda v: self._money(v, base)  # noqa: E731
        share = 1 - a["opportunity_share"]
        parts = [f"{share:.0%} of the sum, {m(amount * share)}"] + ([f"{m(carry)} carried over from last time"] if carry else []) \
            + ([f"{m(a['overflow'])} of opportunity cash above its cap"] if a["overflow"] else [])
        lines = [f"PORTFOLIO ADVICE for {m(amount)} (every figure computed by the service; report them as they are)",
                 f"Core {m(a['core_money'])}" + (f" ({' + '.join(parts)})" if len(parts) > 1 else f" ({share:.0%})")
                 + f", opportunity slice {m(a['slice_money'])} ({a['opportunity_share']:.0%})"
                 + (f", plus {m(reserve)} of opportunity cash you said you hold." if reserve else "."),
                 *([f"Opportunity cash may wait up to {m(a['cap'])} ({cfg['opportunity_cap_slices']:g} slices); the "
                    f"{m(a['overflow'])} above that joins the core now, so the core is {m(a['core_money'])}."] if a["overflow"] else []),
                 "Core funds, cheapness against each fund's own past 5 years (0 = dearest seen, 1 = cheapest, 0.5 = usual):"]
        for c in core:
            f = a["funds"][c["symbol"]]
            held = (f"{f['held']:.1%} of your core now" if f["held"] is not None else
                    "not held yet" if a["drift_known"] else "holding unknown")
            head = f"- {f['name']}{' (the safer fund)' if c['safe'] else ''}: aim {f['aim']:.1%} of the core, {held}."
            sc = f["score"]
            if sc:
                r = sc["ranks"]
                head += (f" Score {sc['score']:.2f}, {advisor.describe_score(sc['score'])}: {abs(sc['fall']):.1%} below its 52-week high "
                         f"(a deeper fall than on {1 - r['fall']:.0%} of past days), {sc['gap']:+.1%} against its 200-day average "
                         f"(lower than on {1 - r['gap']:.0%} of days), past month {sc['month']:+.1%} (weaker than {1 - r['month']:.0%} "
                         f"of months). Closes up to {f['last']}, in {f['currency'] or 'its listing currency'}.")
            lines.append(head)
        statement = self.state["flex"].get("report_date") or "unknown"
        lines.append(f"Holdings are as of IBKR's statement of {statement}: buys after it aren't counted yet, so asking again "
                     "before the next statement repeats the same split.")
        if kept_rows:
            share = sum(r.get("weight") or 0 for r in kept_rows)
            lines.append(f"Kept, not bought any more (your earlier plan; selling isn't suggested): "
                         f"{', '.join(r.get('label') or r['symbol'] for r in kept_rows)}, {share:.1%} of your holdings.")
        lines.append("Suggested core buy:")
        for c in core:
            s, f = c["symbol"], a["funds"][c["symbol"]]
            money, count = a["money"][s], a["counts"].get(s)
            why = a["split"]["reasons"].get(s)
            if f["price_base"] and count is not None:
                # §4VH-c live: the model read "USD Acc" in a fund's NAME as a currency conversion; say the trading currency
                in_base = (f" ({m(f['price_base'])}; traded in {f['currency']}, so the order needs a conversion to {base})"
                           if f["currency"] != base else f" (traded in {base}, the account's currency: no conversion; a "
                           "currency in the fund's name is only its internal one)")
                text = (f"- Buy {s}: {count} share(s) at about {self._price(f['price_local'])} {f['currency']}{exchange(f.get('yahoo')) if (quotes.get(f.get('yahoo')) or {}).get('currency') == f['currency'] else ''}{in_base}, "
                        f"about {m(count * f['price_base'])} (its share of the split: {m(money)})")
            else:
                text = f"- Buy {s}: {m(money)} by the split; no share count, its price in {base} is unknown"
            lines.append(text + (f"; {why}." if why else "."))
        if len(core) == 1:
            lines.append("One core fund: it takes the whole core sum every time. Its cheapness is shown for interest and "
                         "doesn't change the amount.")
        elif a["split"]["rule"] == "tilt" and a["drift_known"]:
            lines.append("How the split was made: each fund's aim, multiplied up when it is cheap against its own past and down when "
                         "it is dear, and pulled back toward the aim when your holdings have drifted.")
        elif a["split"]["rule"] == "tilt":
            lines.append("How the split was made: each fund's aim, multiplied up when it is cheap against its own past and down when "
                         "it is dear. Drift from the aim is unknown, so it is not used.")
        spent_core = sum(a["counts"].get(c["symbol"], 0) * (a["funds"][c["symbol"]]["price_base"] or 0) for c in core)
        lines.append(f"Left over after whole shares: {m(a['left'])}. It is the core's: give it as carried over next time.")
        if a["unplaced"]:
            lines.append(f"Not placed: {m(a['unplaced'])}, for core funds with no price in {base} right now. Don't buy them blind: add "
                         "this sum as carried over next time (NEXT TIME below counts it).")
        if a["core_money"] and a["left"] > 0.25 * a["core_money"]:
            lines.append("This sum is small next to the share prices, so whole shares can't follow the split closely. Investing "
                         "every other month, or a larger sum, keeps closer to it.")
        hot, rule = a["hot"], (f"Hot rule: a fall of at least {abs(cfg['hot_z']):.1f} times the name's usual move over 1 day, 5 days or "
                               f"1 month, while {abs(cfg['hot_fall_pct']):g}% or more below its 52-week high.")
        kind = "funds" if cfg["hot_funds_only"] else "names"
        lines.append(f"OPPORTUNITY: {a['checked']} {kind} with usable prices, from {len(cfg['watchlist'])} on the watchlist plus the core funds. {rule}"
                     + (" Funds only: single stocks are not candidates." if cfg["hot_funds_only"] else ""))
        def cand(c):
            w = c["worst"]
            name = f"{c['symbol']} ({c['name']})" if c["name"] else c["symbol"]
            price = (f"{self._price(c['price'])} {c['currency']}" if c["currency"] else self._price(c["price"]))
            return (f"{name}: {w['change']:+.1%} over {w['label']}, {abs(w['z']):.1f} times its usual move for that span, "
                    f"{abs(c['fall']):.1%} below its 52-week high, yearly volatility {c['volatility']:.0%}, last price {price} ({c['last']})")
        buy = a["buy"]
        if hot:
            lines.append(f"Hot now: {len(hot)}.")
            lines.extend(f"- {cand(c)}" for c in hot[:3])
            if buy and buy not in hot[:3]:  # §4VH-f: the pick is always described, so its fall can be checked in the news
                lines.append(f"- {cand(buy)}")
            pot = a["pot"]
            if buy:
                conversion = ("no conversion" if buy["currency"] == base else f"traded in {buy['currency']}, so a conversion")
                lines.append(f"Suggested: {a['buy_shares']} whole share(s) of {buy['symbol']}{exchange(buy['symbol'])} at about {self._price(buy['price'])} "
                             f"{buy['currency'] or ''} ({conversion}), about {m(a['buy_cost'])}, the most unusual fall the opportunity money "
                             f"({m(pot)}) can buy. The other {m(pot - a['buy_cost'])} waits as opportunity cash.")
                if buy is not hot[0]:
                    lines.append(f"{hot[0]['symbol']} fell more unusually, but one share costs more than the opportunity money.")
                lines.append("Asking again before buying gives the same pick: buy it once.")
            else:
                lines.append(f"Suggested: keep the opportunity money ({m(pot)}) as cash: one share of each hot fund costs more "
                             "than that, or its price in the account's currency is unknown.")
            if len(hot) * 2 >= a["checked"] >= 4:
                lines.append(f"{len(hot)} of {a['checked']} names meet the rule at once: this looks like a market-wide fall, not "
                             "one company's or one sector's." + ("" if cfg["hot_funds_only"] else
                                                                 " A broad fund on the list carries less single-company risk than one stock."))
            lines.append("Before buying, find out WHY it fell: a fall from news that changes the business (results, a lawsuit, "
                         "an export ban) can keep going; a fall from market mood or a rumour that turns out false often recovers. "
                         "The service sees prices only, not news.")
        else:
            lines.append(f"Hot now: none. Suggested: keep the opportunity money as cash for the next one: {m(a['pot'])} in total "
                         f"(this slice and the cash you said you hold, up to {m(a['cap'])}).")
            if a["near"]:
                lines.append("Closest to the rule:")
                lines.extend(f"- {cand(c)}" for c in a["near"])
        if a["not_fund"]:
            lines.append("Left out as not a fund (or type unknown): " + ", ".join(a["not_fund"]))
        if a["unusable"]:
            lines.append("Not checked, no fresh price yet, or price history missing, short or stale: " + ", ".join(a["unusable"]))
        lines.extend(f"Data gap: {g}." for g in a["gaps"])
        for symbol in unmatched:
            lines.append(f"Check: core fund {symbol} matches no holding on IBKR's statement, so it counts as not held"
                         + (f". Funds you hold that aren't in the core: {', '.join(other_funds)}. If one of them is {symbol} under "
                            "another symbol, fix advisor_core in config.json: until then the split sends it everything." if other_funds
                            else ". That is expected until your first buy of it shows on a statement."))
        total_buy = spent_core + a["buy_cost"]
        cash = valued.get("cash")
        if cash is not None and total_buy > cash:
            lines.append(f"Check: these buys come to about {m(total_buy)}, more than the cash on IBKR's statement of "
                         f"{self.state['flex'].get('report_date') or 'unknown date'} ({m(cash)}). Unless you deposited since, deposit "
                         "first: a margin account would borrow the difference.")
        if cfg.get("rejected"):
            lines.append("Settings ignored as invalid, so their defaults are used: " + ", ".join(cfg["rejected"]))
        # per symbol, not the newest arrival: one fresh quote out of 25 must not clear the label (review r2)
        wanted = self.watch_symbols(cfg)
        recent = sum(1 for name in wanted if now - ((self.state["watch_quotes"].get(name) or {}).get("fetched") or 0) <= 6 * 3600)
        lines.append(f"Watchlist prices fetched in the last 6 hours: {recent} of {len(wanted)}; last arrival {self._when(self.state['watch_ok_at'])}" +
                     (" (STALE: an older price is used for up to 4 days of market time, then dropped)." if recent < len(wanted) else "."))
        waiting = round(a["pot"] - a["buy_cost"], 2)
        lines.append(f"NEXT TIME: say the new sum, plus {m(a['left'] + a['unplaced'])} carried over and {m(waiting)} of opportunity cash"
                     " (if you buy as suggested). The service doesn't remember either amount.")
        lines.append("Costs: every order pays IBKR's commission, which has a minimum per order, and a listing traded in "
                     "another currency pays a conversion too (each buy line says which). On small orders that can be a few percent: fewer, larger orders cost less. Prices "
                     "here are delayed, so the fill can differ.")
        lines.append("Cheapness says where a price sits against its own past, not where it goes next: for broad index funds "
                     "buying dips is a small edge at best, the real gain is buying steadily. A crashed stock can keep falling. "
                     "Public prices are delayed. " + NOTE_NOT_ADVICE)
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

    def answer(self, view: str, symbol: str = "", amount=None, reserve=None, carry=None) -> tuple:
        """(ok, text) for one read-only view."""
        with self.lock:
            cfg = self.config()
            if view == "status":
                return True, self._status(cfg)
            if view not in ("summary", "position", "health", "forecast", "review", "advise"):
                return False, f"unknown view {view!r}"
            if not self.state.get("flex"):
                if not (self.token() and cfg.get("flex_query_id")):
                    return False, "the IBKR service is not set up yet: its Flex token and query id are missing on the service machine"
                return False, f"no IBKR statement has arrived yet. Last error: {self.state.get('flex_error') or 'none, the first fetch is due'}"
            if view == "summary":
                return True, self._summary(cfg)
            if view == "advise":
                if amount is None:
                    return False, "advise needs an amount: a number above 0, in the account's base currency"
                return True, self._advise(cfg, amount, reserve or 0.0, carry or 0.0)
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
            query = parse_qs(url.query)
            symbol = (query.get("symbol") or [""])[0][:32]
            numbers = {}
            for name in ("amount", "reserve", "carry"):
                raw = (query.get(name) or [None])[0]
                if raw is None:
                    continue
                try:
                    value = float(raw[:24])
                except ValueError:
                    value = math.nan
                low = 0.01 if name == "amount" else 0.0
                if not math.isfinite(value) or not low <= value <= MAX_AMOUNT:
                    return self._send(200, {"ok": False, "text": f"{name} must be a number from {low:g} to {MAX_AMOUNT:,}"})
                numbers[name] = value
            try:
                ok, text = service.answer(match.group(1), symbol, **numbers)
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
