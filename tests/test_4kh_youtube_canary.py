"""§4KH (2026-09-24): the YouTube route canary.

The route already said "YouTube may have rotated its challenge" — but only
when someone used it. The canary probes on its own schedule and speaks only
on a TRANSITION, through the activity ledger the Slack bot and the push
transports already drain. Every pin drives the code with injected probe
results; nothing here touches the network.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ghost_agent.memory import youtube_canary as yc  # noqa: E402
from ghost_agent.memory import youtube_ingest as yi  # noqa: E402


# ── the probe: classification from injected dependencies ───────────────────

def _resolver(outcome, attempts_used=1, walls=0):
    def _r(url, vid, proxy, stats, n):
        stats.attempts, stats.bot_walls = attempts_used, walls
        if outcome == "ok":
            return {"title": "Me at the zoo", "duration": 19}, "", proxy
        return None, outcome, None
    return _r


def test_the_probe_names_each_failure_by_its_cause():
    ok = yc.probe("socks5://127.0.0.1:9050", ping=lambda: "2.0.0", tor_ok=lambda p: True,
                  resolve=_resolver("ok", attempts_used=2, walls=1))
    assert ok.state == "ok" and ok.attempts == 2 and ok.title == "Me at the zoo"
    walled = yc.probe("socks5://127.0.0.1:9050", ping=lambda: "2.0.0", tor_ok=lambda p: True,
                      resolve=_resolver("ERROR: Sign in to confirm you're not a bot", attempts_used=4, walls=4))
    assert walled.state == "walled" and walled.bot_walls == 4
    partial = yc.probe("socks5://127.0.0.1:9050", ping=lambda: "2.0.0", tor_ok=lambda p: True,
                       resolve=_resolver("ERROR: Private video", attempts_used=1, walls=0))
    assert partial.state == "error", "a rotation that STOPPED EARLY hit a terminal fault"
    mixed = yc.probe("socks5://127.0.0.1:9050", ping=lambda: "2.0.0", tor_ok=lambda p: True,
                     resolve=_resolver("ERROR: HTTP Error 403", attempts_used=4, walls=3))
    assert mixed.state == "unstable", ("an EXHAUSTED rotation with mixed faults is a bad night for Tor or the "
                                      "wall plus noise — held, never paged from one run (review MAJOR-1)")
    helper = yc.probe("socks5://127.0.0.1:9050", ping=lambda: None, tor_ok=lambda p: True,
                      resolve=lambda *a: pytest.fail("must not resolve with the helper down"))
    assert helper.state == "helper_down" and "ping" in helper.reason
    tor = yc.probe(None, ping=lambda: pytest.fail("Tor is checked first"), tor_ok=lambda p: False,
                   resolve=lambda *a: pytest.fail("no"))
    assert tor.state == "tor_down"
    assert yc.probe("socks5://x", ping=lambda: "x", tor_ok=lambda p: False, resolve=lambda *a: None).state == "tor_down"


def test_the_probe_is_resolve_only_and_bounded():
    """The production resolver is fetch_info with the canary's attempt cap —
    no captions, no audio, no store. PARSED via the call it builds."""
    seen = {}
    def fake_fetch_info(url, vid, proxy, *, run, stats, attempts):
        seen.update(url=url, vid=vid, attempts=attempts, run=run)
        stats.attempts = 1
        return {"title": "t"}, "", proxy
    with patch.object(yi, "fetch_info", fake_fetch_info), patch.object(yi, "pot_ping", lambda *a, **k: "2.0.0"):
        r = yc.probe("socks5://127.0.0.1:9050", tor_ok=lambda p: True)
    assert r.state == "ok"
    assert seen["vid"] == yc.CANARY_VIDEO_ID and seen["url"] == yc.CANARY_URL
    assert seen["attempts"] == yc.CANARY_ATTEMPTS and seen["run"] is yi._run_subprocess


# ── the ledger: transitions announced once ─────────────────────────────────

def _run(state, ts, attempts=1, walls=0):
    return yc.CanaryRun(ts=ts, state=state, attempts=attempts, bot_walls=walls, reason="r", duration_s=3.0)


def _fold(runs):
    st = yc.CanaryState()
    records, logs = [], []
    for r in runs:
        st = yc.apply_run(st, r, record=lambda ph, msg, sev, meta: records.append((ph, msg, sev, meta)),
                          log=lambda lvl, msg: logs.append((lvl, msg)))
    return st, records, logs


def test_a_steady_state_is_silent_and_a_transition_speaks_once():
    st, records, _ = _fold([_run("ok", 1), _run("ok", 2), _run("ok", 3)])
    assert st.state == "ok" and records == [], "a healthy route says nothing, ever"
    st, records, _ = _fold([_run("ok", 1), _run("helper_down", 2), _run("helper_down", 3), _run("helper_down", 4)])
    assert [r[2] for r in records] == ["notify"], "ONE notification for three failing runs"
    assert records[0][0] == yc.PHASE and "helper" in records[0][1] and records[0][3]["state"] == "helper_down"
    assert st.since == 2 and st.state == "helper_down"


def test_recovery_is_announced_once_and_names_what_it_recovered_from():
    st, records, _ = _fold([_run("ok", 1), _run("tor_down", 2), _run("ok", 3), _run("ok", 4)])
    sev = [r[2] for r in records]
    assert sev == ["notify", "notify"]
    assert "RECOVERED" in records[1][1] and "tor_down" in records[1][1]
    assert st.announced == "ok"


def test_a_single_all_walled_run_is_rechecked_before_it_pages_the_operator():
    """Four circuits failing by chance is ≈13% with a good token. The first
    all-fail is noted, not announced; the second consecutive one is."""
    st, records, logs = _fold([_run("ok", 1), _run("walled", 2, 4, 4)])
    assert records == [] and st.state == "ok", "held: not confirmed yet"
    assert any("re-checking" in m for _l, m in logs)
    st, records, _ = _fold([_run("ok", 1), _run("walled", 2, 4, 4), _run("walled", 3, 4, 4)])
    assert len(records) == 1 and "WALLED" in records[0][1] and yc.REMEDY in records[0][1]
    assert records[0][3]["remedy"] == yc.REMEDY and st.since == 3
    # a lucky single all-fail between healthy runs never pages
    st, records, _ = _fold([_run("ok", 1), _run("walled", 2, 4, 4), _run("ok", 3)])
    assert records == [] and st.state == "ok"
    # the same hold covers the mixed-fault verdict; two in a row page once
    st, records, _ = _fold([_run("ok", 1), _run("unstable", 2, 4, 3), _run("ok", 3)])
    assert records == []
    st, records, _ = _fold([_run("ok", 1), _run("unstable", 2, 4, 3), _run("unstable", 3, 4, 2)])
    assert len(records) == 1 and "UNSTABLE" in records[0][1] and st.state == "unstable"
    # a held FIRST-EVER run (fresh deploy after a rotation) is also held, then confirmed
    st, records, _ = _fold([_run("walled", 1, 4, 4)])
    assert records == [] and st.state == "" and yc.pending_verdict(st) == "walled"
    st, records, _ = _fold([_run("walled", 1, 4, 4), _run("walled", 2, 4, 4)])
    assert len(records) == 1 and st.state == "walled"


def test_a_held_verdict_is_rechecked_soon_not_next_day():
    """Hold + 24 h interval meant 24–48 h to confirm a rotation (review
    MAJOR-2). A pending verdict shortens the wait to RECHECK_S."""
    st, _, _ = _fold([_run("ok", 1000.0), _run("walled", 2000.0, 4, 4)])
    assert yc.pending_verdict(st) == "walled"
    assert not yc.due(st, boot_monotonic=None, now=2000.0 + 60, monotonic_now=0.0, interval_s=86400, recheck_s=900)
    assert yc.due(st, boot_monotonic=None, now=2000.0 + 901, monotonic_now=0.0, interval_s=86400, recheck_s=900)
    settled, _, _ = _fold([_run("ok", 1000.0), _run("ok", 2000.0)])
    assert not yc.due(settled, boot_monotonic=None, now=2000.0 + 901, monotonic_now=0.0, interval_s=86400, recheck_s=900)
    future = yc.CanaryState(last_run=10 ** 12)
    assert yc.due(future, boot_monotonic=None, now=2000.0, monotonic_now=0.0, interval_s=86400), "a future last_run is not waited out"


def test_an_unaccepted_record_is_retried_on_the_next_run():
    """`announced` moves only when the ledger accepted the message; a lost DM
    (ledger write failed) is retried on the next run instead of forgotten."""
    st = yc.CanaryState()
    calls = []
    st = yc.apply_run(st, _run("helper_down", 1), record=lambda *a: (calls.append(a), False)[1])
    assert st.state == "helper_down" and st.announced == "" and len(calls) == 1
    st = yc.apply_run(st, _run("helper_down", 2), record=lambda *a: (calls.append(a), True)[1])
    assert st.announced == "helper_down" and len(calls) == 2
    st = yc.apply_run(st, _run("helper_down", 3), record=lambda *a: (calls.append(a), True)[1])
    assert len(calls) == 2, "once accepted, silent"


def test_the_first_ever_run_that_is_healthy_makes_no_noise_but_a_failing_first_run_does():
    st, records, _ = _fold([_run("ok", 1)])
    assert records == [] and st.announced == "ok"
    st, records, _ = _fold([_run("helper_down", 1)])
    assert len(records) == 1, "a fresh deploy with the helper down is worth one message"


def test_history_is_bounded_and_the_state_round_trips(tmp_path):
    st, _, _ = _fold([_run("ok", float(i)) for i in range(60)])
    assert len(st.history) == yc.HISTORY_KEEP
    p = tmp_path / "youtube_canary.json"
    yc.save_state(p, st)
    back = yc.load_state(p)
    assert back.state == "ok" and back.last_run == 59.0 and len(back.history) == yc.HISTORY_KEEP
    p.write_text("{not json")
    assert yc.load_state(p).state == "", "a corrupt file reads as never-run, never raises"


# ── scheduling from the tick ───────────────────────────────────────────────

def test_due_honours_boot_delay_interval_and_the_off_switch():
    st = yc.CanaryState(last_run=0.0)
    assert not yc.due(st, boot_monotonic=100.0, now=1000.0, monotonic_now=150.0, boot_delay_s=600, interval_s=600), "boot delay"
    assert yc.due(st, boot_monotonic=100.0, now=1000.0, monotonic_now=800.0, boot_delay_s=600, interval_s=600)
    st.last_run = 1000.0 - 3600.0
    assert not yc.due(st, boot_monotonic=None, now=1000.0, monotonic_now=0.0, interval_s=24 * 3600)
    assert yc.due(st, boot_monotonic=None, now=1000.0 + 24 * 3600, monotonic_now=0.0, interval_s=24 * 3600)
    assert not yc.due(st, boot_monotonic=None, now=10 ** 9, monotonic_now=0.0, interval_s=0), "0 disables"


def test_interval_env_zero_disables_and_a_typo_falls_back(monkeypatch):
    monkeypatch.setenv("GHOST_YT_CANARY_HOURS", "0")
    assert yc._interval_from_env() == 0.0
    monkeypatch.setenv("GHOST_YT_CANARY_HOURS", "abc")
    assert yc._interval_from_env() == 24 * 3600.0
    monkeypatch.setenv("GHOST_YT_CANARY_HOURS", "6")
    assert yc._interval_from_env() == 6 * 3600.0


def _agent(tmp_path, *, foreground=0):
    ctx = SimpleNamespace(memory_dir=tmp_path / "memory", tor_proxy="socks5://127.0.0.1:9050",
                          llm_client=SimpleNamespace(foreground_tasks=foreground, foreground_requests=0),
                          boot_monotonic=time.monotonic() - 10 ** 6)
    (tmp_path / "memory").mkdir(exist_ok=True)
    ag = SimpleNamespace(context=ctx, records=[])
    ag._record_autonomous_activity = lambda ph, msg, severity="info", **meta: ag.records.append((ph, msg, severity, meta))
    return ag


def _wait_idle():
    for _ in range(600):   # 12 s: the probe stub may hold a gate for up to 5 s
        with yc._lock:
            if not yc._running:
                return
        time.sleep(0.02)
    pytest.fail("canary worker did not finish")


def test_maybe_run_probes_off_loop_once_persists_and_announces(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    calls = []
    gate = threading.Event()

    def fake_probe(tor_proxy, **kw):
        calls.append(threading.current_thread().name)
        gate.wait(5)                       # hold the worker so the overlap check is real, not a race
        return yc.CanaryRun(ts=time.time(), state="helper_down", reason="no ping")
    monkeypatch.setattr(yc, "probe", fake_probe)
    assert yc.maybe_run(ag) is True
    for _ in range(100):
        if calls:
            break
        time.sleep(0.01)
    assert yc.maybe_run(ag, force=True) is False, "a second call while the first RUNS must not overlap"
    gate.set()
    _wait_idle()
    assert calls == ["youtube-canary"], "the probe ran once, on its own thread"
    st = yc.load_state(yc.state_path(ag.context))
    assert st.state == "helper_down" and st.last_run > 0
    assert [r[2] for r in ag.records] == ["notify"]
    # (the gate stays SET: the forced re-run below must not block on it)
    assert yc.maybe_run(ag) is False, "not due again for a day"
    assert yc.maybe_run(ag, force=True) is True, "the operator trigger ignores the cadence"
    _wait_idle()
    assert [r[2] for r in ag.records] == ["notify"], "same state again: no second message"


def test_a_probe_that_raises_is_a_verdict_not_a_retry_every_minute(tmp_path, monkeypatch):
    """Without this, a broken resolver (yt-dlp gone, a bad proxy string) left
    `last_run` untouched and the tick re-probed every 60 s for ever, silently."""
    ag = _agent(tmp_path)
    def boom(tor_proxy, **kw):
        raise RuntimeError("yt_dlp module vanished")
    monkeypatch.setattr(yc, "probe", boom)
    assert yc.maybe_run(ag) is True
    _wait_idle()
    st = yc.load_state(yc.state_path(ag.context))
    assert st.state == "error" and st.last_run > 0 and "vanished" in (st.last or {}).get("reason", "")
    assert [r[2] for r in ag.records] == ["notify"]
    assert yc.maybe_run(ag) is False, "last_run advanced: not due again until the interval"


def test_the_boot_delay_applies_even_when_the_context_has_no_boot_clock(tmp_path, monkeypatch):
    """`boot_monotonic` lives on app.state, not on the context; without a
    fallback the first probe fired at the first tick — while Tor may still be
    bootstrapping. The module's import time stands in."""
    ag = _agent(tmp_path)
    del ag.context.boot_monotonic
    monkeypatch.setattr(yc, "probe", lambda *a, **k: pytest.fail("must not probe inside the boot delay"))
    monkeypatch.setattr(yc, "_process_start_monotonic", time.monotonic())
    assert yc.maybe_run(ag) is False
    monkeypatch.setattr(yc, "_process_start_monotonic", time.monotonic() - 10 ** 6)
    monkeypatch.setattr(yc, "probe", lambda *a, **k: yc.CanaryRun(ts=time.time(), state="ok", attempts=1))
    assert yc.maybe_run(ag) is True
    _wait_idle()


def test_maybe_run_yields_to_a_foreground_turn_and_to_the_off_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(yc, "probe", lambda *a, **k: pytest.fail("must not probe"))
    assert yc.maybe_run(_agent(tmp_path, foreground=1)) is False
    monkeypatch.setattr(yc, "INTERVAL_S", 0.0)
    assert yc.maybe_run(_agent(tmp_path)) is False


def test_the_tick_calls_the_canary_before_the_memory_guard():
    """PARSED: `_biological_tick` calls `maybe_run` and does so BEFORE the
    `memory_system` early return — the canary needs no memory and a degraded
    boot is exactly when the operator wants it running."""
    import ast, inspect, textwrap
    from ghost_agent.core import agent as amod
    tree = ast.parse(textwrap.dedent(inspect.getsource(amod.GhostAgent._biological_tick)))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)).body
    def _has_call(node, name):
        return any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == name for n in ast.walk(node))
    def _is_memory_guard(node):
        return (isinstance(node, ast.If) and any(isinstance(n, ast.Constant) and n.value == "memory_system"
                                                  for n in ast.walk(node.test)))
    idx_call = next(i for i, n in enumerate(body) if _has_call(n, "maybe_run"))
    idx_guard = next(i for i, n in enumerate(body) if _is_memory_guard(n))
    assert idx_call < idx_guard


def test_health_view_is_keyed_on_the_last_run_and_shows_a_held_verdict(tmp_path):
    """A held first run used to read `not_yet_run` for a day (review MAJOR-2)
    and a held wall was indistinguishable from "ok on attempt 4" (MEDIUM-1)."""
    ctx = SimpleNamespace(memory_dir=tmp_path / "m"); (tmp_path / "m").mkdir()
    assert yc.health_view(ctx)["state"] == "not_yet_run"
    held, _, _ = _fold([_run("walled", 5.0, 4, 4)])
    yc.save_state(yc.state_path(ctx), held)
    v = yc.health_view(ctx)
    assert v["state"] == "pending" and v["last_state"] == "walled" and v["pending"] == "walled"
    assert v["last_run"] == 5.0 and v["last_run_iso"].startswith("1970-01-01T00:00:05") and v["recheck_in_s"] == yc.RECHECK_S
    assert v["remedy"] == yc.REMEDY
    held_ok, _, _ = _fold([_run("ok", 5.0), _run("walled", 6.0, 4, 4)])
    yc.save_state(yc.state_path(ctx), held_ok)
    v = yc.health_view(ctx)
    assert v["state"] == "ok" and v["pending"] == "walled" and v["last_state"] == "walled"
    st, _, _ = _fold([_run("ok", 5.0), _run("walled", 6.0, 4, 4), _run("walled", 7.0, 4, 4)])
    yc.save_state(yc.state_path(ctx), st)
    v = yc.health_view(ctx)
    assert v["state"] == "walled" and v["since"] == 7.0 and v["pending"] == "" and v["remedy"] == yc.REMEDY


def test_the_phase_is_registered_as_output_driven():
    """A steady healthy route is silent for months; a PERIODIC expectation
    would alarm on that silence. Liveness lives on /api/health instead."""
    from ghost_agent.core import autonomous_activity as aa
    assert aa.phase_expectation(yc.PHASE) == aa.EXPECT_ON_OUTPUT
    assert yc.PHASE in aa._PHASE_LABELS, "every registered phase needs a human label"


def test_the_remedy_script_exists_is_executable_and_verifies_live():
    p = _ROOT / "bin" / "update-youtube-stack.sh"
    assert p.exists() and p.stat().st_mode & 0o111, "the remedy the canary names must exist and be runnable"
    text = p.read_text(encoding="utf-8")
    for step in ("pip", "install", "yt-dlp", "bgutil-ytdlp-pot-provider", "launchctl kickstart", "com.local.ghost-pot",
                 "--verify", "socks5h://", "youtubepot-bgutilhttp:base_url"):
        assert step in text, step
    assert yc.REMEDY == "bin/update-youtube-stack.sh"
