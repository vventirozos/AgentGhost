"""§4RX (2026-10-27, operator: "the web browser tool sometimes gives errors
and timeouts … can we make this more reliable?" → "fix 1 and 2").

  1. interact: the runner's whole-op clock was the PER-ACTION share, so a
     21-action sequence got 27 s of its 570 s budget (live 43fe95a2 /
     ff6c2096: "exceeded its 27142 ms budget"), and an overrun threw away
     the steps that had already run.
  2. ERR_SOCKS_CONNECTION_FAILED — how a guessed clearnet domain fails over
     Tor — had no hint and was never remembered by the host memo, so one
     made-up domain was dialled 11 times (37 of 80 failures in 11 days).
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from ghost_agent.tools import browser as B
from ghost_agent.tools import browser_runner as R
from ghost_agent.tools import host_memo as HM
from ghost_agent.tools.browser_support import _runner_failure_hint
from ghost_agent.tools.outcome import OutcomeStatus
from tests.test_4ln_browser import _op_payload, _run, _stub


# ── 1. the interact clock is the TOTAL, and an overrun keeps what ran ─────────

class _Page:
    url = "file:///workspace/webos/index.html"
    def set_default_timeout(self, ms): pass
    async def wait_for_timeout(self, ms): await asyncio.sleep(ms / 1000.0)
    async def evaluate(self, js, *a):
        if "slow" in str(js):
            await asyncio.sleep(60)
        return 7
    async def title(self): return "NebulaOS"
    async def goto(self, url, **kw): return None


def _fake_playwright(monkeypatch):
    class _Ctx:
        pages = [_Page()]
        async def route(self, *a, **k): pass
        async def route_web_socket(self, *a, **k): pass
        def on(self, *a, **k): pass
        async def close(self): pass
    class _Chromium:
        async def launch_persistent_context(self, **kw): return _Ctx()
    class _PW:
        chromium = _Chromium()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    async def _fake_guard(ctx, sandbox_root=None, anonymous=False):
        return None
    monkeypatch.setattr(R, "async_playwright", lambda: _PW())
    monkeypatch.setattr(R, "_attach_diagnostics", lambda page: ([], []))
    monkeypatch.setattr(R, "_install_ssrf_guard", _fake_guard)


def _record_clock(monkeypatch, scale=None):
    """Record the timeout the runner's whole-op `wait_for` is given; with
    `scale`, actually wait that fraction of it (keeps the +5 s slack out of
    the test's wall clock)."""
    seen = []
    real = asyncio.wait_for

    async def _wf(aw, timeout):
        seen.append(timeout)
        # only the whole-op clock (budget + 5 s) is scaled; a step's own wait stays real
        return await real(aw, timeout * scale if scale and timeout >= 5 else timeout)
    monkeypatch.setattr(R.asyncio, "wait_for", _wf)
    return seen


def test_the_interact_clock_is_the_total_not_one_actions_share(monkeypatch, tmp_path):
    """Fails in the measured world: the clock was timeout_ms (27142) + 5 s."""
    _fake_playwright(monkeypatch)
    seen = _record_clock(monkeypatch)
    out = asyncio.run(R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 27142, "total_timeout_ms": 570000,
        "url": "file:///workspace/webos/index.html", "actions": [{"action": "evaluate", "js": "1"}]}))
    assert out["actions"][0]["ok"]
    assert seen[0] == pytest.approx(575.0)


def test_an_op_without_a_total_keeps_its_own_budget(monkeypatch):
    _fake_playwright(monkeypatch)
    seen = _record_clock(monkeypatch)

    async def fine(page):
        return {"url": page.url}
    asyncio.run(R._with_context("/tmp/p", None, 30000, fine))
    assert seen[0] == pytest.approx(35.0)


def test_an_interact_overrun_returns_the_steps_that_ran(monkeypatch, tmp_path):
    """Fails where the overrun raised: the evaluate that had already
    answered was lost behind a bare `runner exit 1`."""
    _fake_playwright(monkeypatch)
    _record_clock(monkeypatch, scale=0.05)            # 5.2 s → 0.26 s
    out = asyncio.run(R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
        "url": "file:///workspace/webos/index.html",
        "actions": [{"action": "evaluate", "js": "1"}, {"action": "sleep", "ms": 60000},
                    {"action": "evaluate", "js": "2"}, {"action": "evaluate", "js": "3"}]}))
    assert out["budget_exhausted"] and out["budget_ms"] == 200 and not out["aborted"]
    assert [a["ok"] for a in out["actions"]] == [True, False]
    assert out["actions"][0]["value"] == "7"
    assert out["actions"][1]["action"] == "sleep" and "200 ms total budget" in out["actions"][1]["error"]
    assert out["not_run"] == 2


async def test_an_evaluate_in_flight_at_the_overrun_is_not_your_script_threw(monkeypatch, tmp_path):
    """Review MAJOR: §4RJ reads a failed evaluate without "timed out" as the
    model's own script throwing — the budget partial read STATUS: OK."""
    _fake_playwright(monkeypatch)
    real_wait_for = asyncio.wait_for
    _record_clock(monkeypatch, scale=0.05)
    out = await R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
        "url": "file:///workspace/webos/index.html",
        "actions": [{"action": "evaluate", "js": "1"}, {"action": "evaluate", "js": "slow"},
                    {"action": "evaluate", "js": "2"}]})
    monkeypatch.setattr(R.asyncio, "wait_for", real_wait_for)   # the host below runs on the real clock
    assert "may already have taken effect" in out["actions"][-1]["error"]
    res = await _run(tmp_path, out, operation="interact", url="file:///workspace/webos/index.html",
                     actions=[{"action": "evaluate", "js": "1"}] * 3)
    assert "STATUS: PARTIAL (1 of 2 actions failed)" in str(res) and "YOUR script" not in str(res)
    assert res.status != OutcomeStatus.OK


def test_an_overrun_on_the_final_snapshot_is_a_whole_sequence(monkeypatch, tmp_path):
    """Review MINOR: every step ran — no in-flight entry, nothing 'not run'."""
    _fake_playwright(monkeypatch)
    _record_clock(monkeypatch, scale=0.05)

    calls = []

    async def slow_title(self):                       # the implicit goto's read is fast; the final one is not
        calls.append(1)
        if len(calls) > 1:
            await asyncio.sleep(60)
        return "T"
    monkeypatch.setattr(_Page, "title", slow_title)
    out = asyncio.run(R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
        "url": "file:///workspace/webos/index.html", "actions": [{"action": "evaluate", "js": "1"}]}))
    assert [a["ok"] for a in out["actions"]] == [True]
    assert not out["budget_exhausted"] and out["not_run"] == 0


def test_a_goto_abort_keeps_its_banner_through_an_overrun(monkeypatch, tmp_path):
    """Review r2: `aborted` was rebuilt as False by the partial."""
    _fake_playwright(monkeypatch)
    _record_clock(monkeypatch, scale=0.05)
    calls = []

    async def goto(self, url, **kw):
        calls.append(url)
        if "bad" in url:
            raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")

    async def slow_title(self):
        await asyncio.sleep(60)
    monkeypatch.setattr(_Page, "goto", goto)
    monkeypatch.setattr(_Page, "title", slow_title)
    out = asyncio.run(R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
        "actions": [{"action": "goto", "url": "https://bad.example/"}, {"action": "evaluate", "js": "1"}]}))
    assert out["aborted"] and out["abort_reason"] == "goto_failed"


def test_an_in_flight_goto_keeps_the_page_it_asked_for_in_the_sidecar(monkeypatch, tmp_path):
    _fake_playwright(monkeypatch)
    _record_clock(monkeypatch, scale=0.05)
    written = []
    monkeypatch.setattr(R, "_write_last_url", lambda d, u: written.append(u))

    async def goto(self, url, **kw):
        await asyncio.sleep(60)
    monkeypatch.setattr(_Page, "goto", goto)
    out = asyncio.run(R.op_interact({
        "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
        "actions": [{"action": "goto", "url": "https://slow.example/"}]}))
    assert out["budget_exhausted"] and written == ["https://slow.example/"]   # not page.url after it


def test_an_overrun_before_any_action_is_still_an_error(monkeypatch, tmp_path):
    """The implicit first goto never finished: nothing was read, so no
    partial result may pretend otherwise."""
    _fake_playwright(monkeypatch)
    _record_clock(monkeypatch, scale=0.05)

    async def slow_goto(self, url, **kw):
        await asyncio.sleep(60)
    monkeypatch.setattr(_Page, "goto", slow_goto)
    with pytest.raises(RuntimeError, match="exceeded its 200 ms budget"):
        asyncio.run(R.op_interact({
            "profile_dir": str(tmp_path / "p"), "timeout_ms": 200, "total_timeout_ms": 200,
            "url": "https://a.org/", "actions": [{"action": "evaluate", "js": "1"}]}))


async def test_the_host_sends_the_total_and_keeps_the_per_action_share(tmp_path):
    stub = _stub({"final_url": "file:///workspace/x.html", "final_title": "X",
                  "actions": [{"index": i, "action": "evaluate", "ok": True, "value": "1"} for i in range(21)]})
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=stub, operation="interact",
                         url="file:///workspace/x.html",
                         actions=[{"action": "evaluate", "js": "1"}] * 21)
    p = _op_payload(stub)
    assert p["timeout_ms"] == 27142                                  # the share: 570 s / 21
    assert p["total_timeout_ms"] == 570000                            # the whole budget, as the host planned it


async def test_a_one_action_interact_keeps_its_full_clock(tmp_path):
    """Review r2 MAJOR: a launch allowance cut the 1-action clock to 20 s,
    under the 30 s share and shorter than before §4RX."""
    stub = _stub({"final_url": "file:///workspace/x.html", "final_title": "X",
                  "actions": [{"index": 0, "action": "goto", "ok": True}]})
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=stub, operation="interact",
                         url="file:///workspace/x.html", actions=[{"action": "sleep", "ms": 30000}])
    p = _op_payload(stub)
    assert p["timeout_ms"] == p["total_timeout_ms"] == 30000
    assert p["actions"][0]["ms"] <= p["total_timeout_ms"]


async def test_a_queued_call_gets_the_time_it_has_not_one_second(tmp_path, monkeypatch):
    """Review r2 MAJOR: 40 s left in the queue gave a 1000 ms clock."""
    stub = _stub({"final_url": "file:///workspace/x.html", "final_title": "X", "actions": []})
    monkeypatch.setattr(B, "_exec_timeout_for", lambda deadline, want, now=None: 40)
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=stub, operation="interact",
                         url="file:///workspace/x.html", actions=[{"action": "evaluate", "js": "1"}])
    p = _op_payload(stub)
    assert p["total_timeout_ms"] == 25000 and p["timeout_ms"] == 25000      # the share never exceeds it


async def test_the_launch_race_retry_gets_a_total_sized_to_its_own_window(tmp_path, monkeypatch):
    """Review MAJOR: the retry reused the first exec's 560 s total under a
    shorter exec, so the sandbox could kill it with no result."""
    from unittest.mock import MagicMock
    ok = ('[BROWSER_OK] {"final_url": "file:///workspace/x.html", "final_title": "X", '
          '"actions": [{"index": 0, "action": "evaluate", "ok": true, "value": "1"}]}\n')
    outs = iter([("[BROWSER_ERR] TargetClosedError: Target page, context or browser has been closed\n", 1),
                 (ok, 0)])
    sb = MagicMock()
    sb.cmds = []
    sb.execute = lambda cmd, timeout=300, **kw: (sb.cmds.append(cmd), next(outs))[1]
    windows = iter([600, 100])
    monkeypatch.setattr(B, "_exec_timeout_for", lambda deadline, want, now=None: next(windows))
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(B.asyncio, "sleep", lambda s: _real_sleep(0))
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=sb, operation="interact",
                         url="file:///workspace/x.html", actions=[{"action": "evaluate", "js": "1"}] * 21)
    assert len(sb.cmds) == 2
    assert _op_payload(SimpleNamespace(cmds=sb.cmds[:1]))["total_timeout_ms"] == 570000
    assert _op_payload(SimpleNamespace(cmds=sb.cmds))["total_timeout_ms"] == 85000     # (100 - 15) s


async def test_a_single_op_carries_no_total(tmp_path):
    stub = _stub({"status": 200, "url": "https://a.org/", "title": "A", "text": "x", "length": 1})
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=stub, operation="navigate",
                         url="https://a.org/")
    assert "total_timeout_ms" not in _op_payload(stub)


async def test_a_budget_partial_is_shown_as_partial_with_what_did_not_run(tmp_path):
    res = await _run(tmp_path, {
        "final_url": "file:///workspace/x.html", "final_title": "", "aborted": False,
        "budget_exhausted": True, "budget_ms": 570000, "not_run": 3,
        "actions": [{"index": 0, "action": "evaluate", "ok": True, "value": "1"},
                    {"index": 1, "action": "sleep", "ok": False,
                     "error": "TimeoutError: the sequence ran out of its 570000 ms total budget during this step"}]},
        operation="interact", url="file:///workspace/x.html",
        actions=[{"action": "evaluate", "js": "1"}] + [{"action": "sleep", "ms": 1}] * 4)
    s = str(res)
    assert "STATUS: PARTIAL (1 of 2 actions failed)" in s
    assert "BUDGET USED UP" in s and "the 3 actions after it did NOT run" in s
    assert "SEQUENCE ABORTED" not in s                              # the goto-abort banner is not this


# ── 2. a SOCKS failure has a hint and strikes the host ─────────────────────────

def test_a_socks_failure_gets_its_own_hint():
    h = _runner_failure_hint("Error: Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://www.bx.bg/")
    assert "does not exist" in h and "search results" in h and "spelling variants" in h
    assert "identical retry usually fails" not in h                  # not the generic line
    assert "EVERY site fails" in h and "Tor is down" in h            # review: it may BE Tor


@pytest.fixture
def _memo(monkeypatch):
    monkeypatch.setenv("GHOST_DEAD_HOST_MEMO", "1")
    monkeypatch.setattr(HM, "_LAST_CLEARNET_OK", [0.0])
    HM._HOST_FAILS.clear()
    yield
    HM._HOST_FAILS.clear()


_SOCKS = "Error: Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://www.bx.bg/"


def test_two_socks_failures_mark_a_clearnet_host_dead(_memo):
    """Fails in the measured world: www.bx.bg was dialled 11 times."""
    cause = _SOCKS
    HM._mark_host_failed("https://www.bx.bg/", cause)
    assert HM._dead_host_notice("https://www.bx.bg/") is None        # one strike: a Tor hiccup is allowed
    HM._mark_host_ok("https://example.com/")                         # Tor works: another site loaded
    HM._mark_host_failed("https://www.bx.bg/x", cause)
    note = HM._dead_host_notice("https://www.bx.bg/other")
    assert note and "www.bx.bg failed 2 times" in note
    assert HM._HOST_FAILS["www.bx.bg"][3] == HM._HOST_TIMEOUT_TTL    # Tor-side: the short TTL


def test_a_tor_outage_bans_no_host(_memo):
    """Review MAJOR: with Tor unable to build circuits every host fails this
    way; nothing loaded between the failures, so nothing is banned — and a
    load after recovery does not turn the outage strikes into a ban."""
    for h in ("www.bx.bg", "en.wikipedia.org", "www.bbc.com"):
        HM._mark_host_failed(f"https://{h}/", _SOCKS)
        HM._mark_host_failed(f"https://{h}/a", _SOCKS)
        HM._mark_host_failed(f"https://{h}/b", _SOCKS)
    HM._mark_host_ok("https://example.com/")                         # Tor is back
    assert all(HM._dead_host_notice(f"https://{h}/") is None
               for h in ("www.bx.bg", "en.wikipedia.org", "www.bbc.com"))


def test_a_socks_then_timeout_outage_bans_nothing(_memo):
    """Review r2 MINOR: only the LAST cause was checked, so a SOCKS failure
    followed by a timeout skipped the load-between gate."""
    t = "Runner failed (exit 1): Timeout Page.goto: Timeout 30000ms exceeded."
    HM._mark_host_failed("https://www.bbc.com/", _SOCKS)
    HM._mark_host_failed("https://www.bbc.com/", t)
    HM._mark_host_failed("https://www.cnn.com/", t)                  # and the other order
    HM._mark_host_failed("https://www.cnn.com/", _SOCKS)
    assert HM._dead_host_notice("https://www.bbc.com/") is None
    assert HM._dead_host_notice("https://www.cnn.com/") is None


async def test_an_aborted_interact_is_not_proof_that_tor_works(tmp_path, _memo):
    """Review r2 MAJOR: an interact whose first goto failed returns a
    [BROWSER_OK] payload, and `_mark_host_ok` ran on the host that failed —
    "Tor works" during an outage, and the host's own strikes erased."""
    HM._mark_host_failed("https://a.example.com/", _SOCKS)
    await _run(tmp_path, {
        "actions": [{"index": -1, "action": "goto", "ok": False, "url": "https://a.example.com/",
                     "error": "initial navigation failed (Error): Page.goto: net::ERR_SOCKS_CONNECTION_FAILED"}],
        "aborted": True, "abort_reason": "initial_goto_failed", "final_url": "https://a.example.com/",
        "final_title": ""}, operation="interact", url="https://a.example.com/",
        actions=[{"action": "evaluate", "js": "1"}])
    assert HM._LAST_CLEARNET_OK[0] == 0.0 and not HM.tor_recently_ok()
    assert "a.example.com" in HM._HOST_FAILS                        # its strike survives


def test_a_dotless_host_is_never_clearnet():
    assert HM._clearnet_host("http://web:3000/") == "" and HM._clearnet_host("https://www.bx.bg/") == "www.bx.bg"


def test_a_timeout_still_bans_without_a_load_between(_memo):
    t = "Runner failed (exit 1): Timeout Page.goto: Timeout 30000ms exceeded."
    HM._mark_host_failed("https://www.bse.bg/", t)
    HM._mark_host_failed("https://www.bse.bg/", t)
    assert HM._dead_host_notice("https://www.bse.bg/")


def test_a_socks_failure_on_an_onion_stays_out_of_the_clearnet_memo(_memo):
    url = "http://bww2yrsf4gooxcwwnkek7dzzbnq3rvf3w4vm4pys5zid4r2xwqj5wfqd.onion/"
    HM._mark_host_failed(url, "net::ERR_SOCKS_CONNECTION_FAILED")
    HM._mark_host_failed(url, "net::ERR_SOCKS_CONNECTION_FAILED")
    assert not HM._HOST_FAILS


# ── 3. a page outcome logs as information; a mechanical failure stays a warning ─

from ghost_agent.tools.browser_support import browser_outcome_note   # noqa: E402


@pytest.fixture
def _tor_ok(monkeypatch):
    monkeypatch.setattr(HM, "_LAST_CLEARNET_OK", [time.monotonic()])


@pytest.mark.parametrize("text,note", [
    ("Runner failed (exit 1): Timeout Page.goto: Timeout 30000ms exceeded.", "did not load in time"),
    ("Error: Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://www.bx.bg/", "Tor could not reach"),
    ("RuntimeError: the op exceeded its 30000 ms budget — the page kept", "past its budget"),
    ("ValueError: selector '.price' did not match any element", "no such element"),
    ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 5 actions failed)", "some actions failed"),
    ("--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 403 — bot challenge)", "refused the page"),
])
def test_page_outcomes_are_named(text, note, _tor_ok):
    assert note in browser_outcome_note(text)


@pytest.mark.parametrize("text", [
    "(exit 124) Timeout Page.goto: Timeout 30000ms exceeded.",       # killed at the wall clock: ours
    "Runner failed (exit 1): BrowserType.launch: Executable doesn't exist at /root/headless_shell",
    "[SANDBOX INFRA ERROR] docker daemon not reachable",
    "Runner failed (exit 1): TargetClosedError: Target page, context or browser has been closed",
    "Unknown operation 'evaluate'. Valid: click, close, extract_text",
    # review MINORs: a crash that names a locator, a SIGTERM kill, the user's own app down
    "Locator.click: Target page, context or browser has been closed\nCall log:\n  - waiting for locator('#a')",
    "(exit 143) Timeout Page.goto: Timeout 30000ms exceeded.",
    "Page.goto: net::ERR_CONNECTION_REFUSED at http://127.0.0.1:8101/index.html",
    "",
])
def test_mechanical_failures_are_not_page_outcomes(text, _tor_ok):
    assert browser_outcome_note(text) == ""


def test_page_text_does_not_pick_the_class(_tor_ok):
    """Review r2: the classifier read page content too."""
    t = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\n--- PER-ACTION RESULTS ---\n"
         "  [0] OK extract_text: len=40 sel='#a'\n      TEXT: an article about headless_shell and (exit 137)\n"
         "  [1] ERR click '#b': Timeout waiting for locator('#b')")
    assert browser_outcome_note(t) == "the page has no such element"    # the ERR line names it
    nav = "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): Page.goto: Timeout 30000ms\n--- TEXT ---\nTarget crashed"
    assert browser_outcome_note(nav) == "the page did not load in time"


def test_a_crash_inside_a_partial_interact_stays_a_warning(_tor_ok):
    t = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\n--- PER-ACTION RESULTS ---\n"
         "  [1] ERR click '#b': Target page, context or browser has been closed")
    assert browser_outcome_note(t) == ""


def test_a_socks_failure_with_no_recent_load_stays_a_warning(_memo):
    """Review MAJOR: no clean load in 10 minutes — it may be Tor, not the host."""
    assert browser_outcome_note(_SOCKS) == ""
    HM._mark_host_ok("https://example.com/")
    assert "Tor could not reach" in browser_outcome_note(_SOCKS)


async def _logged(tmp_path, monkeypatch, out, rc):
    from unittest.mock import MagicMock
    calls = []
    monkeypatch.setattr(B, "pretty_log", lambda *a, **k: calls.append((a, k)))
    sb = MagicMock()
    sb.execute = lambda cmd, timeout=300, **kw: (out, rc)
    await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=sb, operation="navigate",
                         url="https://www.bse.bg/en/")
    return [(a[0], k.get("level")) for a, k in calls if a and a[0] in ("Browser Note", "Browser Failed")]


async def test_a_page_timeout_logs_a_browser_note(tmp_path, monkeypatch, _memo):
    got = await _logged(tmp_path, monkeypatch,
                        "[BROWSER_ERR] TimeoutError: Page.goto: Timeout 30000ms exceeded.\n", 1)
    assert got == [("Browser Note", "INFO")]


async def test_a_crashed_runner_still_logs_browser_failed(tmp_path, monkeypatch, _memo):
    got = await _logged(tmp_path, monkeypatch,
                        "[BROWSER_ERR] BrowserType.launch: Executable doesn't exist at /root/headless_shell\n", 1)
    assert got == [("Browser Failed", "WARNING")]


# ── the turn loop's lines (review: no test drove the agent.py sites) ──────────

from unittest.mock import MagicMock, patch                                   # noqa: E402
from ghost_agent.tools.outcome import ToolOutcome                           # noqa: E402
from tests.test_blocked_pages_are_not_strikes import _REQ, _drive, agent    # noqa: E402,F401

_WARN_TITLES = ("Tool Warning", "Transient Fail", "Execution Fail")


async def _turn_lines(agent, text):
    _drive(agent, ToolOutcome.failed(text, world_changed=False, reason_code="browser_error"))
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        await agent.handle_chat(_REQ, background_tasks=MagicMock())
    return [(c.args[0], c.kwargs.get("level"), str(c.args[1]) if len(c.args) > 1 else "")
            for c in pl.call_args_list if c.args]


@pytest.mark.parametrize("err", [
    "Runner failed (exit 1):  Page.goto: net::ERR_SOCKS_CONNECTION_FAILED at https://www.bx.bg/",  # unknown class
    "Runner failed (exit 1): ValueError: selector '.price' did not match any element",           # diagnostic
    "Runner failed (exit 1): Timeout Page.goto: Timeout 30000ms exceeded.",                      # retryable
])
async def test_a_page_outcome_draws_no_warning_line_in_the_turn_loop(agent, err, _tor_ok):
    lines = await _turn_lines(agent, f"--- BROWSER RESULT ---\nSTATUS: ERROR\n{err}\n")
    assert not [l for l in lines if l[0] in _WARN_TITLES]
    notes = [l for l in lines if l[0] == "Tool Note"]
    assert notes and all(l[1] == "INFO" for l in notes)
    assert any(("failure " in l[2]) for l in notes)                   # the strike is still counted and shown


async def test_a_mechanical_failure_keeps_its_warning_lines(agent, _tor_ok):
    lines = await _turn_lines(agent, "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): "
                                     "BrowserType.launch: Executable doesn't exist at /root/headless_shell\n")
    assert [l for l in lines if l[0] in _WARN_TITLES and l[1] == "WARNING"]
    assert not [l for l in lines if l[0] == "Tool Note"]
