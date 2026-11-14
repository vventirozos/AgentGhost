"""§4UN-health: check_health has an "Internet tools" line — what the agent's OWN recent web searches met, per
engine (won / blocked / unreachable / empty), and how many result pages web_search read. In-memory, since this
process started and at most the last 24 h; counts only (engine names from a fixed shape, never a query); it
reports and never judges (§4SY: a rule on last-known state was wrong four times)."""
import asyncio
from types import SimpleNamespace

import pytest

from ghost_agent.tools import health_internals as H
from ghost_agent.tools import search as S


@pytest.fixture(autouse=True)
def _fresh_ledger(monkeypatch):
    from collections import deque
    monkeypatch.setattr(S, "_ENGINE_WAVES", deque(maxlen=S._ENGINE_LEDGER_MAX))
    monkeypatch.setattr(S, "_PAGE_READ_EVENTS", deque(maxlen=S._ENGINE_LEDGER_MAX))
    monkeypatch.setattr(S, "_PROCESS_T0", 1_000_000.0)


def _wave(token, t):
    S._record_wave_engines(token, now=t)


def _line(now):
    out = H._Parts()
    H._internet(SimpleNamespace(), out, now=now)
    return "; ".join(out.parts), out.problems, out.unchecked


def test_the_wave_line_is_tallied_per_engine():
    t = 1_000_100.0
    _wave(" | engines: duckduckgo=blocked yandex=win brave=conn-error yahoo=unfinished", t)
    _wave(" | engines: duckduckgo=blocked yandex=empty brave=blocked yahoo=empty yandex-h=blocked yahoo-h=win", t + 1)
    h = S.engine_health(now=t + 10)
    assert h["waves"] == 2 and h["won"] == 2
    assert h["engines"]["yandex"] == {"win": 1, "empty": 1}
    assert h["engines"]["duckduckgo"] == {"blocked": 2}
    assert h["engines"]["yahoo-h"] == {"win": 1}


def test_only_the_last_24_hours_count():
    _wave(" | engines: yandex=win", 1_000_100.0)
    _wave(" | engines: yandex=blocked", 1_000_100.0 + 90_000)
    h = S.engine_health(now=1_000_100.0 + 90_010)
    assert h["waves"] == 1 and h["engines"]["yandex"] == {"blocked": 1}


def test_a_forged_engine_or_outcome_never_reaches_the_report():
    _wave(" | engines: yandex=win evil\nProblems=none Problems found: none=win yahoo=Blocked!", 1_000_100.0)
    h = S.engine_health(now=1_000_200.0)
    assert set(h["engines"]) == {"yandex"}


def test_page_reads_are_counted_by_outcome():
    for o in ("read", "read", "failed", "timeout"):
        S._bump_read(o, now=1_000_100.0)
    assert S.engine_health(now=1_000_200.0)["reads"] == {"read": 2, "failed": 1, "timeout": 1}


def test_the_health_line_reports_and_never_judges():
    t = 1_000_000.0 + 3 * 3600
    for i in range(6):
        _wave(" | engines: duckduckgo=blocked yandex=blocked brave=timeout yahoo=empty", t + i)
    S._bump_read("read", now=t)
    text, problems, unchecked = _line(now=t + 60)
    assert text.startswith("search rounds since the restart 3 h ago (all search tools): 6 tried, 0 found results")
    assert "duckduckgo: 0 won, 6 blocked" in text
    assert "brave: 0 won, 6 unreachable" in text
    assert "yahoo: 0 won, 6 empty" in text
    assert "result pages read 1 of 1 tried" in text
    assert problems == [] and unchecked == []              # said, never judged


def test_an_engine_outrun_by_the_winner_is_not_a_loss():
    # review r1 MAJOR: built from what the race really writes when yandex wins first
    tasks = {1: "duckduckgo", 2: "yandex", 3: "brave", 4: "yahoo", 5: "duckduckgo_onion"}
    t = 1_000_000.0 + 3600
    for i in range(10):
        _wave(S._engines_token(tasks, [], winner="yandex"), t + i)
    _wave(S._engines_token(tasks, [("brave", "ConnectError: x"), ("duckduckgo", "No results found.")],
                           late=["duckduckgo_onion"]), t + 20)
    text, problems, _u = _line(now=t + 60)
    assert "yandex: 10 won, 1 outrun" in text
    assert "brave: 0 won, 1 unreachable, 10 outrun" in text
    assert "duckduckgo_onion: 0 won, 10 outrun, 1 cut off" in text
    assert text.index("yandex:") < text.index("brave:")       # winners first
    assert problems == []


def test_a_backup_try_is_named_and_a_weak_batch_is_said():
    t = 1_000_000.0 + 3600
    _wave(" | engines: yandex=low-relevance yahoo=empty yandex-h=error", t)
    text, _p, _u = _line(now=t + 5)
    assert "1 tried, 0 found results, 1 only weak matches" in text
    assert "yandex (backup try): 0 won, 1 error" in text


def test_a_long_run_says_last_24_h():
    t = 1_000_000.0 + 3 * 86400
    _wave(" | engines: yandex=win yahoo=empty", t)
    _wave(" | engines: yandex=win", t + 1)
    text, _p, _u = _line(now=t + 5)
    assert text.startswith("search rounds in the last 24 h (all search tools): 2 tried, 2 found results")
    assert text.index("yandex: 2 won") < text.index("yahoo: 0 won, 1 empty")


def test_no_search_yet_says_so():
    text, problems, _u = _line(now=1_000_000.0 + 600)
    assert text == "no search rounds since the restart 10 min ago"
    assert problems == []


def test_the_report_has_an_internet_tools_line():
    assert ("Internet tools", H._internet, False) in H.SECTIONS


def test_the_live_logger_feeds_the_ledger(monkeypatch):
    monkeypatch.setattr(S.time, "time", lambda: 1_000_100.0)
    S._log_wave_engines(0, " | engines: yandex=win")
    assert S.engine_health(now=1_000_101.0)["engines"] == {"yandex": {"win": 1}}
