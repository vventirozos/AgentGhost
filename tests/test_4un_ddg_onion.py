"""§4UN-onion: DuckDuckGo's onion service races as an EXTRA engine.

Measured 2026-11-13 (primp, impersonate=random, a fresh SOCKS identity per request, POST /html/):
clearnet html.duckduckgo.com through Tor exits answered 1 of 35 searches (34 × HTTP 202); the onion
service answered 7 of 35 (20 × 202, 8 timeouts at 30 s; replies in 7–17 s). It joins the wave on its own
circuit; the wave deadline is still sized off the clearnet engines, so it never lengthens a round —
a late onion answer is dropped.
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from unittest.mock import patch

import pytest

from ghost_agent.tools import search as S


@pytest.fixture(autouse=True)
def _onion_on(monkeypatch):
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "1")
    yield


PROXY = "socks5h://127.0.0.1:9050"


def test_the_onion_address_passes_its_checksum():
    from ghost_agent.core.link_grounding import onion_v3_valid
    assert S.DDG_ONION_HOST.endswith(".onion")
    assert onion_v3_valid(S.DDG_ONION_HOST)


def test_the_engine_is_registered_with_ddgs_and_asks_the_onion():
    assert S.register_ddg_onion() is True
    from ddgs import DDGS
    from ddgs.engines import ENGINES
    from ddgs.engines.duckduckgo import Duckduckgo
    cls = ENGINES["text"][S.DDG_ONION_ENGINE]
    assert issubclass(cls, Duckduckgo) and cls.name == S.DDG_ONION_ENGINE
    assert cls.search_url == f"https://{S.DDG_ONION_HOST}/html/" and cls.search_method == "POST"
    engines = DDGS(proxy=PROXY, timeout=5)._get_engines("text", S.DDG_ONION_ENGINE)
    assert [type(e) for e in engines] == [cls]                   # not ddgs's "auto" fallback
    assert S.register_ddg_onion() is True                        # idempotent


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text, self.content = status, text, text.encode()


def _engine_with(resp, seen=None):
    S.register_ddg_onion()
    S._patch_ddgs_block_detection()
    from ddgs.engines import ENGINES
    eng = ENGINES["text"][S.DDG_ONION_ENGINE](proxy=PROXY, timeout=5)

    def _request(method, url, **kw):
        if seen is not None:
            seen.append((method, url, kw))
        return resp
    eng.http_client = types.SimpleNamespace(request=_request)
    return eng


def test_a_202_from_the_onion_is_a_block_under_its_own_name():
    with pytest.raises(RuntimeError, match=r"^blocked: HTTP 202 from duckduckgo_onion"):
        _engine_with(_Resp(202, "")).search("postgresql 18 release date")
    assert S._failure_category("blocked: HTTP 202 from duckduckgo_onion") == "blocked"


_HTML = f"""<html><body>
<div class="result results_links web-result"><div class="links_main links_deep result__body">
<h2 class="result__title"><a class="result__a" href="https://www.postgresql.org/about/news/pg18/">PostgreSQL 18 Released</a></h2>
<a class="result__snippet" href="//{S.DDG_ONION_HOST}/l/?uddg=https%3A%2F%2Fwww.postgresql.org%2Fabout%2Fnews%2Fpg18%2F&amp;rut=abc">PostgreSQL 18 was released on 25 September 2025.</a>
</div></div>
<div class="result results_links web-result result--ad"><div class="links_main result__body">
<h2 class="result__title"><a class="result__a" href="https://{S.DDG_ONION_HOST}/y.js?ad=1">Ad</a></h2>
<a class="result__snippet" href="https://{S.DDG_ONION_HOST}/y.js?ad=1">Buy now</a>
</div></div>
</body></html>"""


def test_a_results_page_from_the_onion_gives_the_pages_own_addresses():
    seen = []
    rows = _engine_with(_Resp(200, _HTML), seen).search("postgresql 18 release date")
    assert seen[0][0] == "POST" and seen[0][1] == f"https://{S.DDG_ONION_HOST}/html/"
    assert seen[0][2]["data"]["q"] == "postgresql 18 release date"
    hrefs = [r.href for r in rows]
    assert hrefs == ["https://www.postgresql.org/about/news/pg18/"]     # redirect unwrapped, ad dropped


def test_the_switch_and_the_proxy_decide_the_engine_list(monkeypatch):
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES + (S.DDG_ONION_ENGINE,)
    assert S.raced_engines(None) == S._RACE_ENGINES                      # no Tor: no onion
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "0")
    assert S.raced_engines(PROXY) == S._RACE_ENGINES                     # the exact old list
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "1")
    monkeypatch.setattr(S, "register_ddg_onion", lambda: False)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES                     # never ddgs's "auto" fallback


def test_the_wave_deadline_and_the_pool_account_for_it():
    assert S._engine_timeout(S.DDG_ONION_ENGINE) == 20
    # the deadline is sized off the clearnet engines: the onion never lengthens a round
    assert S._race_wave_deadline() == S._DDGS_FAST_ENGINE_TIMEOUT + S._RACE_WAVE_GRACE
    assert S._RACE_POOL._max_workers >= (len(S._RACE_ENGINES) + 1 + len(S._RACE_HEDGE_ENGINES)) * 8


# ── in the race ─────────────────────────────────────────────────────────────

_ROWS = [{"title": "PostgreSQL 18 release date", "href": "https://www.postgresql.org/about/news/pg18/",
          "body": "PostgreSQL 18 release date: 25 September 2025, with asynchronous I/O."}]


def _fake_ddgs(by_engine, calls):
    class _DDGS:
        def __init__(self, **kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, backend=None, **kw):
            calls.append((backend, self.kw.get("proxy")))
            return by_engine[backend]()
    return types.SimpleNamespace(DDGS=_DDGS)


def _race(monkeypatch, by_engine, stats=None, deadline=None):
    calls = []
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.0)                 # no hedge in these
    if deadline is not None:
        monkeypatch.setattr(S, "_race_wave_deadline", lambda: deadline)
    with patch.dict(sys.modules, {"ddgs": _fake_ddgs(by_engine, calls)}):
        t0 = time.monotonic()
        rows = asyncio.run(S._race_search_wave("postgresql 18 release date", PROXY, 0, stats=stats))
        return rows, calls, time.monotonic() - t0


def _empty():
    return []


def _blocked(engine):
    def f():
        raise RuntimeError(f"blocked: HTTP 202 from {engine}")
    return f


def test_the_onion_can_win_a_wave_on_its_own_circuit(monkeypatch):
    lines = []
    monkeypatch.setattr(S, "pretty_log", lambda t, c, **k: lines.append(str(c)))
    # every clearnet engine refused (r1: an engine that ANSWERED empty ends the wave without the onion)
    by = {e: _blocked(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(0.2), [dict(r) for r in _ROWS])[1]
    rows, calls, _ = _race(monkeypatch, by)
    assert [r["href"] for r in rows] == [_ROWS[0]["href"]]
    assert any(line.startswith("duckduckgo_onion won wave 0") for line in lines)
    proxies = dict(calls)
    assert proxies[S.DDG_ONION_ENGINE].startswith("socks5h://")
    assert proxies[S.DDG_ONION_ENGINE] != proxies["duckduckgo"]          # its own circuit


def test_a_win_elsewhere_does_not_wait_for_the_onion(monkeypatch):
    by = {e: _empty for e in S._RACE_ENGINES}
    by["yandex"] = lambda: [dict(r) for r in _ROWS]
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(3), [])[1]
    rows, _, took = _race(monkeypatch, by)
    assert rows and took < 1.5


def test_a_late_onion_is_dropped_at_the_unchanged_deadline_and_named(monkeypatch):
    lines = []
    monkeypatch.setattr(S, "pretty_log", lambda t, c, **k: lines.append(str(c)))
    stats = {}
    by = {e: _blocked(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(3), [dict(r) for r in _ROWS])[1]
    rows, _, took = _race(monkeypatch, by, stats=stats, deadline=0.6)
    assert rows == [] and took < 1.5
    assert stats["engines"][S.DDG_ONION_ENGINE] == {"abandoned": 1}
    assert stats["engines"]["yandex"] == {"blocked": 1}
    wave = next(line for line in lines if "no winner" in line)
    assert "duckduckgo_onion abandoned" in wave
    assert S.search_failure_counts(stats) == (len(S._RACE_ENGINES), 0, 0)  # not a failure kind


def test_an_onion_block_counts_as_a_block_under_its_name(monkeypatch):
    lines = []
    monkeypatch.setattr(S, "pretty_log", lambda t, c, **k: lines.append(str(c)))
    stats = {}
    # the clearnet engines answer a beat AFTER the onion's refusal: with every fake instant, which finished
    # first was thread scheduling, and under suite load the onion read "abandoned" (flaky 1 run in 3)
    by = {e: (lambda: (time.sleep(0.3), [])[1]) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = _blocked(S.DDG_ONION_ENGINE)
    rows, _, _ = _race(monkeypatch, by, stats=stats)
    assert rows == []
    assert stats["engines"][S.DDG_ONION_ENGINE] == {"blocked": 1}
    # final review: the onion stays out of the retry-decision counts; it is measured per engine
    assert S.search_failure_counts(stats) == (0, 0, len(S._RACE_ENGINES))
    assert any("duckduckgo_onion blocked" in line for line in lines)


def test_switch_off_races_exactly_the_old_engines(monkeypatch):
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "0")
    by = {e: _empty for e in S._RACE_ENGINES}
    _, calls, _ = _race(monkeypatch, by)
    assert sorted(c[0] for c in calls) == sorted(S._RACE_ENGINES)


# ── review: no-winner waves, links, measurement ─────────────────────────────

def _answered_empty():
    time.sleep(0.05)
    raise RuntimeError("No results found.")


def _blocked_fast(engine):
    def f():
        time.sleep(0.05)
        raise RuntimeError(f"blocked: HTTP 403 from {engine}")
    return f


def test_r1_an_answered_empty_wave_does_not_wait_for_the_onion(monkeypatch):
    by = {e: _answered_empty for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(2.0), [])[1]
    _, _, on = _race(monkeypatch, by, deadline=1.6)
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "0")
    _, _, off = _race(monkeypatch, by, deadline=1.6)
    assert on < off + 0.3 and on < 0.8


def test_r1_when_every_clearnet_engine_is_refused_the_onion_is_waited_for(monkeypatch):
    by = {e: _blocked_fast(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(0.5), [dict(r) for r in _ROWS])[1]
    rows, _, took = _race(monkeypatch, by, deadline=1.6)
    assert [r["href"] for r in rows] == [_ROWS[0]["href"]] and took < 1.2


def test_r1_the_onion_alone_never_starts_the_hedge(monkeypatch):
    by = {e: _blocked_fast(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(0.6), [dict(r) for r in _ROWS])[1]
    calls = []
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.2)
    monkeypatch.setattr(S, "_race_wave_deadline", lambda: 1.6)
    with patch.dict(sys.modules, {"ddgs": _fake_ddgs(by, calls)}):
        rows = asyncio.run(S._race_search_wave("postgresql 18 release date", PROXY, 0))
    assert rows and len(calls) == len(S._RACE_ENGINES) + 1           # no hedge engines launched


def test_r1_the_wave_lines_carry_every_engines_outcome(monkeypatch):
    lines = []
    monkeypatch.setattr(S, "_log_wave_engines",
                        lambda wave, token: lines.append(("won wave " if "=win" in token else "no winner ")
                                                               + token))
    by = {e: _answered_empty for e in S._RACE_ENGINES}
    by["brave"] = _blocked_fast("brave")
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(2.0), [])[1]
    _race(monkeypatch, by, deadline=1.6)
    wave = next(line for line in lines if line.startswith("no winner"))
    tokens = dict(t.split("=") for t in wave.split("engines: ", 1)[1].split())
    assert tokens == {"duckduckgo": "empty", "yandex": "empty", "brave": "blocked", "yahoo": "empty",
                      "duckduckgo_onion": "abandoned"}
    lines.clear()
    by["yandex"] = lambda: [dict(r) for r in _ROWS]
    _race(monkeypatch, by, deadline=1.6)
    won = next(line for line in lines if line.startswith("won wave"))
    assert "yandex=win" in won.split("engines: ", 1)[1]


@pytest.mark.parametrize("href, want", [
    ("//duckduckgo.com/l/?uddg=https://x.example/a%26b&rut=abc", "https://x.example/a%26b"),
    ("/l/?uddg=https://x.example/p?a=1&b=2&rut=abc", "https://x.example/p?a=1&b=2"),
    ("/l/?rut=abc&uddg=https://x.example/q", "https://x.example/q"),
    (f"//{S.DDG_ONION_HOST}/l/?uddg=https://x.example/r&rut=1", "https://x.example/r"),
    ("https://plain.example/page", "https://plain.example/page"),
])
def test_r1_redirect_links_are_unwrapped_once(href, want):
    assert S._unwrap_ddg_href(href) == want


def test_r1_a_page_of_relative_links_gives_results_and_keeps_percent_escapes():
    html = """<html><body>
<div class="result__body"><h2><a class="result__a" href="/l/?uddg=https%3A%2F%2Fx.example%2Fa%2526b&amp;rut=1">A</a></h2>
<a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fx.example%2Fa%2526b&amp;rut=1">PostgreSQL 18 release date text.</a></div>
<div class="result__body"><h2><a class="result__a" href="/l/?rut=2&amp;uddg=https%3A%2F%2Fy.example%2F">B</a></h2>
<a class="result__snippet" href="/l/?rut=2&amp;uddg=https%3A%2F%2Fy.example%2F">More PostgreSQL 18 text.</a></div>
</body></html>"""
    rows = _engine_with(_Resp(200, html)).search("postgresql 18 release date")
    assert [r.href for r in rows] == ["https://x.example/a%26b", "https://y.example/"]


def test_r1_the_stats_script_tallies_engines_and_page_reads(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "search_engine_stats", Path(__file__).resolve().parents[1] / "scripts" / "search_engine_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = tmp_path / "ghost-agent.log"
    log.write_text(
        "2026-11-01 10:00:00 - GhostAgent - INFO - search wave engines — wave 0 ‹old q› | engines: duckduckgo=blocked yandex=win\n"
        "2026-11-13 10:00:00 - GhostAgent - INFO - search wave engines — wave 0 ‹a=b q› | engines: duckduckgo=blocked "
        "yandex=win yahoo=unfinished duckduckgo_onion=unfinished\n"
        "2026-11-13 10:00:05 - GhostAgent - INFO - search wave engines — wave 1 ‹q› | engines: duckduckgo=empty "
        "yandex=timeout duckduckgo_onion=late\n"
        "2026-11-13 10:00:09 - GhostAgent - INFO - search wave engines — wave 2 | engines: duckduckgo=empty "
        "duckduckgo_onion=abandoned\n"
        "2026-11-13 10:00:06 - GhostStream - INFO - [abc +7.0s] search page read — read python.org in 900 ms — 2500 of 9000 chars\n"
        "2026-11-13 10:00:06 - GhostStream - INFO - [abc +7.0s] search page read — timeout b.example — not read within 12s\n"
        "2026-11-13 10:00:07 - GhostStream - INFO - unrelated line\n", encoding="utf-8")
    import datetime as dt
    stats = mod.tally([str(log)], since=dt.date(2026, 11, 10))
    assert stats["engines"]["duckduckgo"] == {"blocked": 1, "empty": 2}
    assert stats["engines"]["yandex"] == {"win": 1, "timeout": 1}
    assert stats["engines"]["duckduckgo_onion"] == {"unfinished": 1, "late": 1, "abandoned": 1}
    assert stats["page_reads"] == {"read": 1, "timeout": 1}
    assert stats["waves"] == 3
    assert "duckduckgo_onion" in mod.render(stats) and "abandoned" in mod.render(stats)


# ── final review ────────────────────────────────────────────────────────────

def _blocked_q(engine):
    def f():
        time.sleep(0.03)
        raise RuntimeError(f"blocked: HTTP 202 from {engine}")
    return f


def _search_total(monkeypatch, by, onion: str) -> float:
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", onion)
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.0)
    monkeypatch.setattr(S, "_race_wave_deadline", lambda: 0.6)
    monkeypatch.setattr(S, "_proxy_reachable", lambda *a, **k: True)
    monkeypatch.setattr(S, "wiki_supplement_wanted", lambda q: False)
    monkeypatch.setattr(S, "pretty_log", lambda *a, **k: None)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s, *a, **k: real_sleep(0))
    S._SEARCH_CACHE.clear()
    import importlib.machinery
    mod = types.ModuleType("ddgs")
    mod.DDGS = _fake_ddgs(by, []).DDGS
    mod.__spec__ = importlib.machinery.ModuleSpec("ddgs", None)
    with patch.dict(sys.modules, {"ddgs": mod}):
        t0 = time.monotonic()
        asyncio.run(S.tool_search_ddgs("zzqx frobnicator widget count", PROXY))
        return time.monotonic() - t0


def test_f_the_onion_only_wait_happens_once_per_search(monkeypatch):
    by = {e: _blocked_q(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(1.5), [])[1]          # hangs past every deadline
    off = _search_total(monkeypatch, by, "0")
    on = _search_total(monkeypatch, by, "1")
    assert on <= off + 0.6 + 0.35, (on, off)                           # one wait, not one per wave


def test_f_no_onion_wait_once_it_came_back_late_in_this_search(monkeypatch):
    by = {e: _blocked_q(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(1.5), [])[1]
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.0)
    monkeypatch.setattr(S, "_race_wave_deadline", lambda: 0.6)

    async def go():
        tok = S._ONION_SEARCH.set({"onion_spent": True})               # it was late earlier in this search
        try:
            t0 = time.monotonic()
            await S._race_search_wave("zzqx frobnicator widget count", PROXY, 1)
            return time.monotonic() - t0
        finally:
            S._ONION_SEARCH.reset(tok)
    with patch.dict(sys.modules, {"ddgs": _fake_ddgs(by, [])}):
        assert asyncio.run(go()) < 0.4


def test_f_the_onion_stays_out_of_the_retry_decision(monkeypatch):
    by = {"duckduckgo": _answered_empty, "yandex": _answered_empty,
          "brave": _blocked_q("brave"), "yahoo": _blocked_q("yahoo"),
          S.DDG_ONION_ENGINE: _blocked_q(S.DDG_ONION_ENGINE)}
    on = {}
    _race(monkeypatch, by, stats=on, deadline=1.0)
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", "0")
    off = {}
    _race(monkeypatch, by, stats=off, deadline=1.0)
    assert S.search_failure_counts(on) == S.search_failure_counts(off) == (2, 0, 2)
    assert on["engines"][S.DDG_ONION_ENGINE] == {"blocked": 1}              # still measured


def test_f_the_targets_own_rut_parameter_is_kept():
    assert S._unwrap_ddg_href("/l/?uddg=https://x.example/p?rut=keep&a=1&rut=ddghash") == \
        "https://x.example/p?rut=keep&a=1"
    assert S._unwrap_ddg_href("/l/?rut=ddghash&uddg=https://x.example/p?a=1&rut=keep") == \
        "https://x.example/p?a=1&rut=keep"


def test_f_the_engines_line_carries_no_query(monkeypatch, caplog):
    import logging
    by = {e: _answered_empty for e in S._RACE_ENGINES}
    by["yandex"] = lambda: [dict(r) for r in _ROWS]
    with caplog.at_level(logging.INFO):
        _race(monkeypatch, by, deadline=1.0)
    lines = [r.getMessage() for r in caplog.records if "search wave engines" in r.getMessage()]
    assert lines and all("postgresql" not in m and "engines: " in m for m in lines)


# ── last round: "spent" only by a real timeout or the one wait ──────────────

def _wave_state(monkeypatch, by, deadline=1.0):
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    monkeypatch.setattr(S, "_race_hedge_at", lambda: 0.0)
    monkeypatch.setattr(S, "_race_wave_deadline", lambda: deadline)
    monkeypatch.setattr(S, "pretty_log", lambda *a, **k: None)

    async def go():
        state = {}
        tok = S._ONION_SEARCH.set(state)
        try:
            await S._race_search_wave("zzqx frobnicator widget count", PROXY, 0, stats={})
        finally:
            S._ONION_SEARCH.reset(tok)
        return state
    with patch.dict(sys.modules, {"ddgs": _fake_ddgs(by, [])}):
        return asyncio.run(go())


def test_l_an_onion_left_running_when_clearnet_answered_is_abandoned_not_spent(monkeypatch):
    by = {e: _answered_empty for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(1.5), [])[1]
    assert not _wave_state(monkeypatch, by).get("onion_spent")        # the next wave may still wait once


def test_l_the_one_wait_spends_it(monkeypatch):
    by = {e: _blocked_q(e) for e in S._RACE_ENGINES}
    by[S.DDG_ONION_ENGINE] = lambda: (time.sleep(1.5), [])[1]
    assert _wave_state(monkeypatch, by, deadline=0.5).get("onion_spent") is True


def test_l_an_onion_timeout_spends_it(monkeypatch):
    by = {e: _answered_empty for e in S._RACE_ENGINES}

    def _onion_timeout():
        raise TimeoutError("timed out after 20s")
    by[S.DDG_ONION_ENGINE] = _onion_timeout
    assert _wave_state(monkeypatch, by).get("onion_spent") is True


# ── ships OFF (operator, 11-13: 7 of 35 early, then 0 of 72 — all 202 blocks) ──

def test_ships_off_unset_races_the_old_list(monkeypatch):
    monkeypatch.delenv("GHOST_SEARCH_DDG_ONION", raising=False)
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    assert S._ddg_onion_on() is False
    assert S.raced_engines(PROXY) == S._RACE_ENGINES


@pytest.mark.parametrize("val", ["1", "on", "true", "yes", " TRUE ", "Yes"])
def test_ships_off_only_a_truthy_value_turns_it_on(monkeypatch, val):
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", val)
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES + (S.DDG_ONION_ENGINE,)


@pytest.mark.parametrize("val", ["", "0", "off", "false", "no", "2", "maybe"])
def test_ships_off_anything_else_keeps_it_off(monkeypatch, val):
    monkeypatch.setenv("GHOST_SEARCH_DDG_ONION", val)
    monkeypatch.setattr(S, "register_ddg_onion", lambda: True)
    assert S.raced_engines(PROXY) == S._RACE_ENGINES
