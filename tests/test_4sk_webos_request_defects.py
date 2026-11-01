"""§4SK — request a60aa2de (2026-10-31, a 91-minute webOS build): the five
mechanical defects found in its log, pinned.

  1 services   a `start` naming a DIFFERENT command replaces the running
               service in one call; an entry whose served directory is gone
               is dead whatever its pid says; the note names the directory
               the command serves (`cd X &&`, `--directory X`), not the
               workdir
  2 brake      the composed-URL brake ignores the agent's own hosts (loopback,
               LAN, file://): a 404 on 127.0.0.1:8101 blocked 127.0.0.1:8100
  3 launch     the runner waits (bounded) for the previous Chromium to leave
               the shared profile before launching; the retry is logged as
               information with the runner's text, WARNING only when the
               retry fails too
  4 selector   a selector that tests an attribute, on a page where no element
               has that value, is answered with the values the attribute DOES
               take
  5 links      an HTML page's asset URLs (stylesheets, fonts, scripts, images,
               CSS url()) are not citations; <a href> still is

World where each pin fails: the model is told "already running … use it as
it is" about a server whose directory was deleted (4 turns); its own fresh
server is refused as "a composed address" (2 turns); two browser calls in
one turn trip the launch race every time; a `data-app="calculator"` typo is
blamed on timing (2 turns); a Google Fonts <link> is reported as an unread
citation twice.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.sandbox import services as S
from ghost_agent.sandbox.services import ServiceSupervisor
from tests.test_sandbox_services import FakeSandbox, happy_handler


# ── 1 services ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("entry,expect", [
    ({"workdir": "/workspace", "command": "cd /workspace/webos && python3 -m http.server ${PORT:-8100}"}, "/workspace/webos"),
    ({"workdir": "/workspace", "command": "cd webos && python3 -m http.server 8100"}, "/workspace/webos"),
    ({"workdir": "/workspace/projects/p1", "command": "python3 -m http.server --directory site 8100"}, "/workspace/projects/p1/site"),
    ({"workdir": "/workspace/projects/p1", "command": "python3 -m http.server -d ../shared 8100"}, "/workspace/projects/shared"),
    ({"workdir": "/workspace", "command": "python3 app.py"}, "/workspace"),
    ({"command": "node server.js"}, "/workspace"),
])
def test_the_served_directory_follows_cd_and_directory(entry, expect):
    assert S.served_dir(entry) == expect


def test_the_note_names_the_served_directory_not_the_workdir():
    e = {"pid": 758, "port": 8101, "workdir": "/workspace",
         "command": "cd /workspace/webos && python3 -m http.server ${PORT:-8100}"}
    msg = S.already_running_message("webos", e, e["command"])
    assert "serving /workspace/webos at http://127.0.0.1:8101/" in msg
    assert "serving /workspace at" not in msg


def test_a_different_command_replaces_the_service_in_one_call(tmp_path):
    sb = FakeSandbox(tmp_path, happy_handler())
    sup = ServiceSupervisor(sb)
    first = sup.start("webos", "cd /workspace/webos && python3 -m http.server 8101", port=8101)
    assert "RUNNING" in first
    n_calls = len(sb.calls)
    out = sup.start("webos", "python3 -m http.server 8100", port=8100)
    assert out.startswith("Replaced the running 'webos' (pid 12345, `cd /workspace/webos && python3 -m http.server 8101`"
                          " on port 8101, serving /workspace/webos) — it ran a different command.")
    assert "RUNNING" in out and "http://127.0.0.1:8100" in out
    assert "is already running" not in out and "do action='stop'" not in out
    assert any("TERM" in c for c in sb.calls[n_calls:]), "the old process was not stopped"
    reg = sup._load()
    assert len(reg) == 1 and reg["webos"]["command"].startswith("python3 -m http.server 8100")


def test_a_vanished_directory_makes_the_entry_dead_even_for_the_same_command(tmp_path):
    gone = {"/workspace/webos"}
    inner = happy_handler()

    def handler(cmd):
        if "test -d" in cmd:
            return ("no\n" if any(g in cmd for g in gone) else "yes\n", 0)
        return inner(cmd)
    sb = FakeSandbox(tmp_path, handler)
    sup = ServiceSupervisor(sb)
    cmd = "cd webos && python3 -m http.server 8101"
    gone.clear()
    assert "RUNNING" in sup.start("webos", cmd, port=8101)
    gone.add("/workspace/webos")                      # the directory is deleted later
    out = sup.start("webos", cmd, port=8101)
    assert out.startswith("Replaced the running 'webos'") and "its directory /workspace/webos no longer exists" in out
    assert "already running" not in out


def test_the_same_command_with_its_directory_present_is_still_a_note(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    cmd = "cd webos && python3 -m http.server 8101"
    sup.start("webos", cmd, port=8101)
    out = sup.start("webos", cmd, port=8101)
    assert out.startswith("Note: service 'webos' is already running") and "same command" in out


@pytest.mark.parametrize("answer", [("", 126), ("", 1), ("[SANDBOX INFRA ERROR] daemon wedged", 1), ("yes\n", 0)])
def test_only_the_probes_own_no_reads_as_missing(tmp_path, answer):
    """The safe direction: an infra fault must not call a live service stale."""
    def handler(cmd):
        if "test -d" in cmd:
            return answer
        return happy_handler()(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
    assert sup._dir_exists("/workspace/x") is True
    sup2 = ServiceSupervisor(FakeSandbox(tmp_path, lambda c: ("no\n", 0) if "test -d" in c else ("", 0)))
    assert sup2._dir_exists("/workspace/x") is False


def test_unknown_liveness_is_never_replaced(tmp_path):
    """§4GI: an infra fault on the pid probe is UNKNOWN, read as alive. A
    different command does not replace what cannot be confirmed killed."""
    def infra(cmd):
        if "kill -0" in cmd:
            return ("[SANDBOX INFRA ERROR] daemon wedged", 1)
        return ("", 0)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, infra))
    sup._save({"web": {"name": "web", "command": "python3 app.py", "pid": 4242, "port": 8100, "workdir": "/workspace"}})
    out = sup.start("web", "python3 -m http.server 8100", port=8100)
    assert out.startswith("Note: service 'web' is already running") and "DIFFERENT command" in out
    assert "web" in sup._load() and sup._load()["web"]["pid"] == 4242


@pytest.mark.parametrize("stored,new,same", [
    ("python3 -m http.server 8101", "python3 -m http.server 8101", True),
    ("python3 -m http.server 8101", "python3 -m http.server", True),           # the start adds the port
    ("python3 -m http.server 8101", "python3 -m http.server 8100", False),      # another port literal
    ("python3 -m http.server 8100", "cd webos && python3 -m http.server 8100", False),
])
def test_same_command(stored, new, same):
    e = {"command": stored, "port": 8101 if "8101" in stored else 8100, "workdir": "/workspace"}
    assert S.same_command(e, new) is same


# ── 2 brake ───────────────────────────────────────────────────────────────────

def test_the_brake_ignores_the_agents_own_hosts():
    from ghost_agent.core.agent import _UrlProvenance
    P = _UrlProvenance()
    P.note_failure({"operation": "navigate", "url": "http://127.0.0.1:8101/index.html"},
                   "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404) OP: navigate")
    assert not P.hosts_404
    assert P.refusal("browser", {"operation": "navigate", "url": "http://127.0.0.1:8100/index.html"}) is None
    # a public host still brakes
    P.note_failure({"operation": "navigate", "url": "https://www.postgresql.org/docs/17/x.html"},
                   "STATUS: BLOCKED (HTTP 404)")
    assert P.refusal("browser", {"operation": "navigate", "url": "https://www.postgresql.org/docs/17/y.html"})
    # and a loopback address is never refused, even with the host marked by hand
    P.hosts_404.add("127.0.0.1")
    assert P.refusal("browser", {"operation": "navigate", "url": "http://127.0.0.1:8100/"}) is None
    assert P.refusal("browser", {"operation": "navigate", "url": "http://localhost:8100/a"}) is None
    assert P.refusal("browser", {"operation": "navigate", "url": "file:///workspace/index.html"}) is None


# ── 3 launch race ─────────────────────────────────────────────────────────────

def _runner():
    from ghost_agent.tools import browser_runner as R
    return R


def test_profile_holders_reads_proc_cmdlines(tmp_path, monkeypatch):
    R = _runner()
    proc = tmp_path / "proc"
    for pid, argv in ((101, b"/usr/lib/chromium/chrome\0--user-data-dir=/workspace/.browser_profile\0--headless"),
                      (102, b"python3\0-u\0.browser_runner.py"),
                      (103, b"chrome\0--user-data-dir=/workspace/other")):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(argv)
    (proc / "self").mkdir()
    monkeypatch.setattr(R, "_PROC", str(proc))
    assert R._profile_holders("/workspace/.browser_profile") == 1
    assert R._profile_holders("/workspace/other") == 1
    assert R._profile_holders("/workspace/none") == 0
    monkeypatch.setattr(R, "_PROC", str(tmp_path / "no-proc"))
    assert R._profile_holders("/workspace/.browser_profile") == 0        # no /proc (macOS) → never waits


def test_the_launch_waits_bounded_until_the_profile_is_free(monkeypatch):
    R = _runner()
    seq = iter([1, 1, 0])
    monkeypatch.setattr(R, "_profile_holders", lambda d: next(seq, 0))
    slept = []

    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(R.asyncio, "sleep", fake_sleep)
    waited = asyncio.run(R._settle_profile("/workspace/.browser_profile"))
    assert slept == [0.5, 0.5] and waited >= 0.0
    # never more than the bound, even when the holder never leaves
    monkeypatch.setattr(R, "_profile_holders", lambda d: 1)
    slept.clear()
    asyncio.run(R._settle_profile("/workspace/.browser_profile"))
    assert len(slept) == R._PROFILE_SETTLE_STEPS
    # a free profile costs nothing
    monkeypatch.setattr(R, "_profile_holders", lambda d: 0)
    slept.clear()
    assert asyncio.run(R._settle_profile("/workspace/.browser_profile")) == 0.0 and slept == []


def test_the_with_context_launch_is_preceded_by_the_settle(monkeypatch, tmp_path):
    R = _runner()
    order = []

    async def settle(d):
        order.append("settle")
        return 1.04
    monkeypatch.setattr(R, "_settle_profile", settle)

    async def no_guard(ctx, **kw):
        return None
    monkeypatch.setattr(R, "_install_ssrf_guard", no_guard)
    monkeypatch.setattr(R, "_attach_diagnostics", lambda page: ([], []))

    class Page:
        def set_default_timeout(self, ms): pass
        async def wait_for_timeout(self, ms): pass

    class Ctx:
        pages = [Page()]
        async def close(self): order.append("close")

    class Chromium:
        async def launch_persistent_context(self, **kw):
            order.append("launch")
            return Ctx()

    class PW:
        chromium = Chromium()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(R, "async_playwright", lambda: PW())

    async def op(page):
        return {"status": 200}
    out = asyncio.run(R._with_context(str(tmp_path / "prof"), None, 1000, op))
    assert order == ["settle", "launch", "close"]
    assert out["profile_settle_s"] == 1.0
    # a wait under 0.4 s is noise and is not reported
    async def quick(d):
        return 0.2
    monkeypatch.setattr(R, "_settle_profile", quick)
    assert "profile_settle_s" not in asyncio.run(R._with_context(str(tmp_path / "prof"), None, 1000, op))


async def test_a_recovered_retry_is_information_and_names_the_runner_text(tmp_path, monkeypatch):
    import json
    from ghost_agent.tools import browser as B
    lines = []
    monkeypatch.setattr(B, "pretty_log", lambda title, msg, **kw: lines.append((title, msg, kw.get("level", "INFO"))))
    tce = ("[BROWSER_ERR] TargetClosedError: BrowserType.launch_persistent_context: "
           "Target page, context or browser has been closed\n")
    ok = {"status": 200, "url": "file:///workspace/x.html", "title": "X", "text": "hi"}
    outputs = [(tce, 1), (f"[BROWSER_OK] {json.dumps(ok)}\n", 0)]
    calls = {"n": 0}

    def execute(cmd, timeout=300, **kw):
        out = outputs[min(calls["n"], len(outputs) - 1)]
        calls["n"] += 1
        return out
    from unittest.mock import MagicMock
    sm = MagicMock(); sm.execute = execute
    result = await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                                  sandbox_dir=tmp_path, sandbox_manager=sm)
    assert "STATUS: OK" in result
    retry = [l for l in lines if l[0] == "Browser Retry"]
    assert len(retry) == 1 and retry[0][2] == "INFO" and "runner said: TargetClosedError" in retry[0][1]


async def test_a_retry_that_fails_too_is_a_warning(tmp_path, monkeypatch):
    from ghost_agent.tools import browser as B
    lines = []
    monkeypatch.setattr(B, "pretty_log", lambda title, msg, **kw: lines.append((title, msg, kw.get("level", "INFO"))))
    tce = ("[BROWSER_ERR] TargetClosedError: BrowserType.launch_persistent_context: "
           "Target page, context or browser has been closed\n")
    from unittest.mock import MagicMock
    sm = MagicMock(); sm.execute = lambda cmd, timeout=300, **kw: (tce, 1)
    result = await B.tool_browser(operation="navigate", url="file:///workspace/x.html",
                                  sandbox_dir=tmp_path, sandbox_manager=sm)
    assert "STATUS: ERROR" in result
    retry = [l for l in lines if l[0] == "Browser Retry"]
    assert [l[2] for l in retry] == ["INFO", "WARNING"] and "retry failed too" in retry[1][1]


async def test_the_host_logs_a_settle_the_runner_reports(tmp_path, monkeypatch):
    import json
    from ghost_agent.tools import browser as B
    lines = []
    monkeypatch.setattr(B, "pretty_log", lambda title, msg, **kw: lines.append((title, msg)))
    ok = {"status": 200, "url": "file:///workspace/x.html", "title": "X", "text": "hi", "profile_settle_s": 1.0}
    from unittest.mock import MagicMock
    sm = MagicMock(); sm.execute = lambda cmd, timeout=300, **kw: (f"[BROWSER_OK] {json.dumps(ok)}\n", 0)
    await B.tool_browser(operation="navigate", url="file:///workspace/x.html", sandbox_dir=tmp_path, sandbox_manager=sm)
    assert any("waited 1.0s for the previous Chromium" in m for _, m in lines)


# ── 4 selector hint ───────────────────────────────────────────────────────────

class _Page:
    """The page answers the runner's question for ONE attribute test: does it
    match, and which values does the attribute take. `=` tests match on an
    exact value; other operators are judged by Python's own reading."""
    def __init__(self, attrs, more=False):
        self.attrs = attrs            # {attr: [values on the page]}
        self.asked = []
        self.more = more

    async def evaluate(self, js, arg=None):
        import re
        test, attr = arg
        self.asked.append(test)
        vals = list(self.attrs.get(attr, []))
        m = _runner()._ATTR_TEST_RE.match(test)
        want = next((g for g in m.groups()[1:] if g is not None), None) if m else None
        op = re.search(r"([~|^$*]?)=", test).group(1) if want is not None else None
        if want is None:
            hit = bool(vals)
        elif op == "":
            hit = want in vals
        elif op == "^":
            hit = any(v.startswith(want) for v in vals)
        elif op == "~":
            hit = any(want in v.split() for v in vals)
        else:
            hit = any(want in v for v in vals)
        return {"hit": hit, "vals": vals, "more": self.more}


def test_the_attribute_hint_names_the_values_the_page_has():
    R = _runner()
    page = _Page({"data-app": ["finder", "terminal", "notes", "calc", "browser"], "data-k": ["7", "+", "="]})
    hint = asyncio.run(R._attribute_hint(page, '#dock .dock-app[data-app="calculator"]'))
    assert hint == 'no element matches [data-app="calculator"]; data-app values on the page: finder, terminal, notes, calc, browser.'
    # the value exists → the selector failed for another reason; no claim is made
    assert asyncio.run(R._attribute_hint(page, '.window[data-app="calc"] #calc-grid button[data-k="7"]')) == ""
    # an attribute no element carries
    assert asyncio.run(R._attribute_hint(page, 'div[data-tab="about"]')) == "no element on the page has a data-tab attribute."
    # id/class/style are not attribute hints; a presence test without a value says nothing when present
    assert asyncio.run(R._attribute_hint(page, 'a[href] [class="x"]')) == "no element on the page has a href attribute."
    assert asyncio.run(R._attribute_hint(page, "#dock .dock-app")) == ""


def test_the_attribute_hint_survives_a_page_that_cannot_evaluate():
    R = _runner()

    class Broken:
        async def evaluate(self, js, arg=None):
            raise RuntimeError("closed")
    assert asyncio.run(R._attribute_hint(Broken(), '[data-app="x"]')) == ""


def test_the_require_selector_error_carries_the_attribute_hint(monkeypatch):
    R = _runner()

    class TimeoutError_(Exception):
        pass
    TimeoutError_.__name__ = "TimeoutError"

    class Page(_Page):
        async def wait_for_selector(self, sel, **kw):
            raise TimeoutError_("timeout")

        async def evaluate(self, js, arg=None):
            if js == R._CANDIDATES_JS:
                return ["#calc-grid button"]
            return await super().evaluate(js, arg)
    sel = '.dock-app[data-app="calculator"]'
    R._MISSING_SELECTORS.discard(sel)
    with pytest.raises(ValueError) as ei:
        asyncio.run(R._require_selector(Page({"data-app": ["calc", "notes"]}), sel, 1000))
    msg = str(ei.value)
    R._MISSING_SELECTORS.discard(sel)
    assert 'no element matches [data-app="calculator"]; data-app values on the page: calc, notes.' in msg
    assert msg.index("data-app values") < msg.index("Clickable elements on this page: #calc-grid button")


# ── 5 links ───────────────────────────────────────────────────────────────────

_PAGE = ('<!DOCTYPE html><html><head><link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap" rel="stylesheet">'
         '<script src="https://cdn.example.com/lib.js"></script>'
         '<style>@import url(https://x.y/z.css); body{background:url("https://img.host/a.png")}</style>'
         '</head><body><img src="https://img.host/b.png"><p>See <a href="https://cite.org/p">the paper</a> '
         'and https://plain.org/q</p></body></html>')


def test_asset_urls_in_markup_are_not_citations():
    from ghost_agent.core.link_grounding import unfetched_links, is_markup_path
    assert unfetched_links(_PAGE, [], markup=True) == ["https://cite.org/p", "https://plain.org/q"]
    assert unfetched_links(_PAGE, ["https://cite.org/p"], markup=True) == ["https://plain.org/q"]
    # a markdown report is read as before
    assert "https://fonts.googleapis.com" in unfetched_links(_PAGE, [])
    assert is_markup_path("index.html") and is_markup_path("a/B.HTM") and not is_markup_path("report.md")


def test_the_write_note_and_the_source_check_read_html_as_markup():
    from ghost_agent.core.agent import document_link_check_note
    args = {"operation": "write", "path": "index.html", "content": _PAGE}
    note = document_link_check_note(args, [])
    assert "fonts.googleapis.com" not in note and "cite.org/p" in note
    args_md = {"operation": "write", "path": "report.md", "content": _PAGE}
    assert "fonts.googleapis.com" in document_link_check_note(args_md, [])


def test_a_test_the_page_matches_is_never_named():
    """r2: the 24-value list and a dropped `~=`/`^=` named a MATCHING test as
    the cause; the runner asks the page about the test as written."""
    R = _runner()
    page = _Page({"data-tags": ["a b"], "data-app": ["calculator"]})
    assert asyncio.run(R._attribute_hint(page, '#nope [data-tags~="a"]')) == ""
    assert asyncio.run(R._attribute_hint(page, '#nope [data-app^="calc"]')) == ""
    assert page.asked == ['[data-tags~="a"]', '[data-app^="calc"]']
    # a page with more than 24 values says the list is not all of them
    many = _Page({"data-id": [f"item-{i}" for i in range(24)]}, more=True)
    assert asyncio.run(R._attribute_hint(many, '[data-id="zzz"]')).endswith("item-23, ….")
    # a long value is marked as cut, not handed over as if whole
    long = _Page({"data-x": ["v" * 80]})
    assert "…(cut)" in asyncio.run(R._attribute_hint(long, '[data-x="w"]'))


@pytest.mark.parametrize("sel", ['#x:not([data-hidden])', 'text=[foo]', 'xpath=//div[@a="b"]', '//div[@a]',
                                 'li:not(:has([data-k="1"]))'])
def test_absence_tests_and_other_engines_say_nothing(sel):
    R = _runner()
    page = _Page({})
    assert asyncio.run(R._attribute_hint(page, sel)) == "" and page.asked == []


def test_the_hint_reads_an_escaped_quote_and_a_case_flag_and_asks_at_most_three():
    R = _runner()
    page = _Page({"data-app": ["calc"]})
    assert asyncio.run(R._attribute_hint(page, r'[data-app="a\"b"]')) .startswith(
        r'no element matches [data-app="a\"b"]; data-app values')
    page = _Page({"data-app": ["calc"]})
    asyncio.run(R._attribute_hint(page, '[data-app="x" i]'))
    assert page.asked == ['[data-app="x" i]']
    page = _Page({k: ["1"] for k in "abcde"})     # every test matches: only the cap stops it
    asyncio.run(R._attribute_hint(page, "#nope [a][b][c][d][e]"))
    assert len(page.asked) == R._ATTR_HINT_TESTS
    # css= is CSS
    page = _Page({"data-app": ["calc"]})
    assert "no element matches" in asyncio.run(R._attribute_hint(page, 'css=[data-app="x"]'))


def test_profile_holders_match_the_whole_argument(tmp_path, monkeypatch):
    R = _runner()
    proc = tmp_path / "proc"
    for pid, argv in ((201, b"chrome\0--user-data-dir=/workspace/.browser_profile2\0"),
                      (202, b"chrome\0--user-data-dir=/workspace/.browser_profile")):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(argv)
    monkeypatch.setattr(R, "_PROC", str(proc))
    assert R._profile_holders("/workspace/.browser_profile") == 1


# ── r2: services review ───────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd,expect", [
    ("php -S 0.0.0.0:8100 -d display_errors=1", "/workspace"),         # php's -d is an ini option
    ("npx serve -d -l 8100", "/workspace"),                             # serve's -d takes no value
    ("rails s -d -p 3000", "/workspace"),
    ("cd a && cd b && python3 -m http.server 8100", "/workspace/a/b"),
    ('cd "my site" && python3 -m http.server 8100', "/workspace/my site"),
    ("cd ~/site && python3 -m http.server 8100", ""),                  # the shell expands it: unknown
    ("python3 -m http.server 8100 -d $HOME/x", ""),
    ("python3 -m http.server -d site 8100", "/workspace/site"),
])
def test_served_dir_reads_only_what_it_can_know(cmd, expect):
    assert S.served_dir({"workdir": "/workspace", "command": cmd}) == expect


def test_a_refused_workdir_leaves_the_old_service_running(tmp_path):
    """r2: the old service was killed before the workdir check refused, and
    the reply said "Nothing was launched"."""
    inner = happy_handler()

    def h(cmd):
        if "test -d /workspace/nope" in cmd or "test -d /workspace/workspace/nope" in cmd:
            return ("", 1)
        return inner(cmd)
    sb = FakeSandbox(tmp_path, h)
    sup = ServiceSupervisor(sb)
    sup.start("web", "python3 -m http.server 8101", port=8101)
    n = len(sb.calls)
    out = sup.start("web", "python3 -m http.server 8101 -b 0.0.0.0", port=8101, workdir="/workspace/nope")
    assert out.startswith("Error: workdir") and "Nothing was launched" in out
    assert not any("TERM" in c for c in sb.calls[n:]), "the old service was stopped by a refused start"
    assert sup._load()["web"]["pid"] == 12345


def test_a_replacement_that_dies_says_so_and_keeps_the_old_row(tmp_path):
    """r2: the old service was stopped, the new one exited at once, the reply
    never said the old one was gone and the registry lost its command."""
    state = {"dead": False}
    inner = happy_handler()

    def h(cmd):
        if "kill -0 23456" in cmd:
            return ("", 1)                                  # the new pid dies at once
        if "nohup" in cmd and state["dead"]:
            return ("23456\n", 0)
        return inner(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    state["dead"] = True
    (tmp_path / ".services" / "web.pid").unlink()           # the new run writes its own pid
    out = sup.start("web", "python3 broken.py", port=8101)
    assert out.startswith("Replaced the running 'web' (pid 12345")
    assert "The new command did NOT start: Error: service 'web' exited immediately" in out
    assert "action='restart' name='web'" in out
    assert out.is_failure and out.world_changed
    reg = sup._load()
    assert reg["web"]["command"] == "python3 -m http.server 8101" and reg["web"]["pid"] == 12345


def test_the_same_start_after_the_port_was_moved_is_the_same_service(tmp_path):
    """r2: `http.server 8100` stored as `… 8101` (8100 was taken); the same
    start again killed and relaunched a healthy service."""
    inner = happy_handler()

    def h(cmd):
        if "s.bind" in cmd and "8100" in cmd:
            return ("", 1)                                  # an outsider holds 8100
        return inner(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    c = "python3 -m http.server 8100"
    sup.start("web", c)
    assert "8101" in sup._load()["web"]["command"]
    out = sup.start("web", c)
    assert out.startswith("Note: service 'web' is already running") and "Replaced" not in out


def test_php_ini_flag_does_not_churn_a_healthy_service(tmp_path):
    inner = happy_handler()

    def h(cmd):
        if "test -d" in cmd and "display_errors" in cmd:
            return ("no\n", 0)
        return inner(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    c = "php -S 0.0.0.0:8101 -d display_errors=1"
    sup.start("web", c, port=8101)
    assert sup.start("web", c, port=8101).startswith("Note: service 'web' is already running")


@pytest.mark.parametrize("kw,same", [
    ({"workdir": "projects/b"}, False),
    ({"workdir": "projects/a"}, True),
    ({"workdir": "/workspace/projects/a/"}, True),
    ({"workdir": "/projects/a"}, True),                      # the start heals a missing /workspace
    ({"port": 8105}, False),
    ({"port": 8100}, True),
    ({"port": 8099}, True),                                  # the port it asked for and was moved from
])
def test_an_explicit_workdir_or_port_is_part_of_the_command(kw, same):
    e = {"command": "python3 -m http.server 8100", "port": 8100, "asked_port": 8099,
         "workdir": "/workspace/projects/a"}
    assert S.same_command(e, "python3 -m http.server 8100", **kw) is same


def test_unknown_liveness_does_not_probe_the_directory(tmp_path):
    def infra(cmd):
        if "kill -0" in cmd:
            return ("[SANDBOX INFRA ERROR] daemon wedged", 1)
        return ("", 0)
    sb = FakeSandbox(tmp_path, infra)
    sup = ServiceSupervisor(sb)
    sup._save({"web": {"name": "web", "command": "cd site && python3 -m http.server 8100", "pid": 4242,
                       "port": 8100, "workdir": "/workspace"}})
    sup.start("web", "python3 app.py", port=8100)
    assert not any("echo yes || echo no" in c for c in sb.calls)


# ── r2: links review ──────────────────────────────────────────────────────────

def test_script_bodies_and_quoted_attributes_are_not_citations():
    from ghost_agent.core.link_grounding import unfetched_links
    page = ('<script type=module>import x from "https://cdn.jsdelivr.net/npm/a@1/x.js";'
            ' fetch("https://api.open-meteo.com/v1/forecast?x=1")</script>'
            '<script type="application/ld+json">{"url":"https://example.com/about"}</script>'
            '<img src="x" alt="a>b" data-src="https://cdn.x.com/lazy.png">'
            '<svg><image href="https://cdn.x.com/i.png"/></svg><input type=image src="https://cdn.x.com/b.png">'
            '<p><a href="https://cite.org/p">p</a> and https://plain.org/q</p>'
            '<script>var u = "https://after.unclosed/x"')
    # r3: an UNCLOSED script keeps the rest of the page — dropping it would drop citations
    assert unfetched_links(page, [], markup=True) == ["https://cite.org/p", "https://plain.org/q",
                                                      "https://after.unclosed/x"]
    # outside markup the same text is all citations
    assert "https://api.open-meteo.com/v1/forecast?x=1" in unfetched_links(page, [])


def test_pathological_markup_is_linear_enough():
    import time
    from ghost_agent.core.link_grounding import unfetched_links
    for bad in ("url(https://a.b/" * 20000, "<script>" * 20000, '<img alt="' * 20000):
        t0 = time.time()
        unfetched_links(bad, [], markup=True)
        assert time.time() - t0 < 3.0, bad[:20]


def test_the_brake_host_drops_only_a_www_prefix():
    from ghost_agent.core.agent import _UrlProvenance
    assert _UrlProvenance.host("https://web.archive.org/x") == "web.archive.org"
    assert _UrlProvenance.host("https://www.w3.org/TR/") == "w3.org"


def test_an_xhtml_document_is_link_checked():
    from ghost_agent.core.agent import _DOC_LINK_EXT_RE
    assert _DOC_LINK_EXT_RE.search("report.xhtml") and _DOC_LINK_EXT_RE.search("a.HTM")


# ── r2 battery survivors ──────────────────────────────────────────────────────

def test_the_reply_source_check_reads_html_as_markup(tmp_path):
    """C6: the reply-time Source check used the plain reader on an HTML page."""
    from tests.test_4oh_open_items import _write_row, _nav_row
    from ghost_agent.core.agent import unopened_citation_note
    (tmp_path / "index.html").write_text(_PAGE)
    rows = [_nav_row("https://cite.org/p"), _nav_row("https://plain.org/q"), _write_row("index.html", _PAGE)]
    assert unopened_citation_note(rows, (tmp_path,)) == ""
    (tmp_path / "report.md").write_text(_PAGE)
    note = unopened_citation_note(rows[:2] + [_write_row("report.md", _PAGE)], (tmp_path,))
    assert "fonts.googleapis.com" in note


def test_a_successful_replace_is_not_reported_as_a_failure(tmp_path):
    """A7: without the note on the success lines, the wrapper read the run as
    a failed replacement."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("webos", "python3 -m http.server 8101", port=8101)
    out = sup.start("webos", "python3 app.py", port=8101)
    assert out.startswith("Replaced the running 'webos'") and "RUNNING" in out
    assert "did NOT start" not in out and not getattr(out, "is_failure", False)


def test_a_start_in_another_workdir_replaces(tmp_path):
    """A2b: start() must hand its workdir and port to same_command."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("web", "python3 -m http.server 8100", port=8100, workdir="projects/a")
    out = sup.start("web", "python3 -m http.server 8100", port=8100, workdir="projects/b")
    assert out.startswith("Replaced the running 'web'") and sup._load()["web"]["workdir"] == "/workspace/projects/b"


def test_a_replace_whose_kill_fails_keeps_the_old_row(tmp_path):
    """A5/A6: TERM+KILL leave the old process alive → a declared failure that
    changed nothing, and the old row stays registered."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    old = dict(sup._load()["web"])

    def survive(entry, others=()):
        sup._last_kill_survived = True
    sup._kill_service = survive
    out = sup.start("web", "python3 app.py", port=8101)
    assert out.startswith("Service 'web' could NOT be replaced") and out.reason_code == "service_kill_survived"
    assert not out.world_changed
    assert sup._load()["web"]["pid"] == old["pid"] and sup._load()["web"]["command"] == old["command"]


def test_a_directory_probe_that_raises_reads_as_present(tmp_path):
    """A4: an exception from the sandbox exec is an infra fault, not "gone"."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))

    def boom(*a, **k):
        raise RuntimeError("docker socket closed")
    sup._exec = boom
    assert sup._dir_exists("/workspace/x") is True


def test_directory_equals_form():
    """A10: `--directory=X` moves the served directory too."""
    assert S.served_dir({"workdir": "/workspace", "command": "python3 -m http.server --directory=site 8100"}) \
        == "/workspace/site"


def test_profile_holders_skip_the_runner_and_its_shell(tmp_path, monkeypatch):
    """D4: the runner's own argv (and its parent shell's) can carry the
    profile path; they are not a previous Chromium."""
    R = _runner()
    proc = tmp_path / "proc"
    argv = b"python3\0--user-data-dir=/workspace/.browser_profile\0"
    for pid in (os.getpid(), os.getppid()):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(argv)
    monkeypatch.setattr(R, "_PROC", str(proc))
    assert R._profile_holders("/workspace/.browser_profile") == 0



# ── r3: review round 3 ────────────────────────────────────────────────────────

def test_a_hijacked_replacement_is_the_service_now(tmp_path):
    """r3: the port-hijacked return lacked the note; the wrapper called it
    "did NOT start" and advised a restart that would run the NEW command."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    (tmp_path / ".services" / "web.pid").unlink()
    sup._holder_pid = lambda port: 999
    sup._pid_ownership = lambda h, p: False
    out = sup.start("web", "python3 app.py", port=8101)
    assert out.startswith("Replaced the running 'web'") and "answered by a DIFFERENT process" in out
    assert "did NOT start" not in out and "runs the old command again" not in out
    assert sup._load()["web"]["command"] == "python3 app.py"


def _taken_8100(inner=None):
    inner = inner or happy_handler()

    def h(cmd):
        if "s.bind" in cmd and "8100" in cmd:
            return ("", 1)
        return inner(cmd)
    return h


def test_a_restart_keeps_the_original_ask(tmp_path):
    """r3: restart re-asked with the stored rewrite (`… 8101`), so the same
    start afterwards killed and relaunched a healthy service."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, _taken_8100()))
    c = "python3 -m http.server 8100"
    sup.start("web", c)
    sup.restart("web")
    e = sup._load()["web"]
    assert e["asked"] == c and e["asked_port"] == 8100
    assert sup.start("web", c).startswith("Note: service 'web' is already running")


def test_the_port_a_command_names_is_the_asked_port(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, _taken_8100()))
    c = "python3 -m http.server 8100"
    sup.start("web", c)
    assert sup._load()["web"]["asked_port"] == 8100
    assert sup.start("web", c, port=8100).startswith("Note: service 'web' is already running")


def test_an_exception_after_the_stop_is_a_declared_failure(tmp_path):
    """r3→r4: an exec that raised after the old service was stopped is a
    declared failure that says what is NOT known — and the registry is left
    as the run left it (a guessed restore orphaned a launched process)."""
    inner = happy_handler()
    state = {"boom": False}

    def h(cmd):
        if state["boom"] and "nohup" in cmd:
            raise RuntimeError("docker socket closed")
        return inner(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    state["boom"] = True
    out = sup.start("web", "python3 app.py", port=8101)
    assert out.startswith("Replaced the running 'web'") and "RuntimeError: docker socket closed" in out
    assert "is NOT known" in out and "may read it DEAD" in out and "runs the old command again" not in out
    assert out.is_failure and out.world_changed


def _boom_after_second_launch(tmp_path, boom_on):
    inner = happy_handler(pid="22222")
    st = {"launched": 0}

    def h(cmd):
        if "nohup" in cmd:
            st["launched"] += 1
        if st["launched"] >= 2 and boom_on(cmd):
            raise RuntimeError("docker socket closed")
        return inner(cmd)
    sb = FakeSandbox(tmp_path, h)
    return ServiceSupervisor(sb), st, sb


def test_an_exception_after_the_launch_does_not_orphan_the_new_process(tmp_path):
    """r4 A: the post-launch probe raised; the wrapper popped the new row and
    restored the old one while the new process ran with no row."""
    sup, st, _ = _boom_after_second_launch(tmp_path, lambda c: "connect_ex" in c)
    sup.start("web", "python3 -m http.server 8101", port=8101)
    (tmp_path / ".services" / "web.pid").unlink()
    out = sup.start("web", "python3 app.py", port=8101)
    assert st["launched"] == 2 and "is NOT known" in out and "did NOT start" not in out
    assert sup._load()["web"]["command"] == "python3 app.py"


def test_an_exception_after_the_save_does_not_contradict_the_registry(tmp_path):
    """r4 B: the report's listener scan raised after the row was saved; the
    reply said "did NOT start … restart runs the old command" over a
    registered running new command."""
    sup, st, sb = _boom_after_second_launch(tmp_path, lambda c: "ss -H -ltn 'sport" in c)
    sb.binds_host_netns = lambda: True
    sup.start("web", "python3 -m http.server 8101", port=8101)
    (tmp_path / ".services" / "web.pid").unlink()
    out = sup.start("web", "python3 app.py", port=8101)
    assert "did NOT start" not in out and "runs the old command again" not in out
    assert sup._load()["web"]["command"] == "python3 app.py"


def test_a_stop_from_another_thread_waits_for_the_verdict(tmp_path):
    """r4 D: the lock was released between the run and the verdict; a stop in
    that window was undone by the restore."""
    import threading
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    in_impl, stopped = threading.Event(), threading.Event()
    orig = sup._start_impl

    def impl(*a, **k):
        out = orig(*a, **k)
        in_impl.set()
        stopped.wait(0.5)            # the other thread's stop would land here without the lock
        return out
    sup._start_impl = impl
    t = threading.Thread(target=lambda: (in_impl.wait(5), sup.stop("web"), stopped.set()))
    t.start()
    out = sup.start("web", "python3 app.py", port=8101)
    t.join(10)
    assert not stopped.is_set() or out.startswith("Replaced the running 'web'")
    assert "did NOT start" not in out


def test_an_exception_without_a_replace_still_raises(tmp_path):
    def h(cmd):
        if "nohup" in cmd:
            raise RuntimeError("docker socket closed")
        return happy_handler()(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    with pytest.raises(RuntimeError):
        sup.start("web", "python3 app.py", port=8101)


def test_past_the_scan_cap_absence_is_not_claimed():
    R = _runner()

    class Capped:
        async def evaluate(self, js, arg=None):
            return {"hit": False, "vals": [], "more": True, "capped": True}
    assert asyncio.run(R._attribute_hint(Capped(), '#dock [data-app="calculator"]')) == ""


@pytest.mark.parametrize("sel", ['a:has-text("[Edit]")', '#dock >> text=[Calc]', '.x:text("[beta]")',
                                 'internal:role=button[name="Save"i] >> nth=3',
                                 'internal:attr=[placeholder="Search"i] >> nth=2',
                                 'div:not(:is(:has([data-app="zzz"]))).nope',
                                 'a:text-is("[x]")'])
def test_text_arguments_engines_and_nested_absence_say_nothing(sel):
    R = _runner()
    page = _Page({})
    assert asyncio.run(R._attribute_hint(page, sel)) == "" and page.asked == []


def test_a_css_part_of_a_chain_is_still_read():
    R = _runner()
    page = _Page({"data-app": ["calc"]})
    out = asyncio.run(R._attribute_hint(page, '#dock >> [data-app="calculator"] >> nth=0'))
    assert out.startswith('no element matches [data-app="calculator"]')


def test_script_stripping_is_exact_on_text_that_lowering_lengthens():
    from ghost_agent.core.link_grounding import unfetched_links
    page = "İstanbul " * 30 + "<script>fetch('https://api.x.com/v1')</script><a href=\"https://cited.org/report\">r</a>"
    assert unfetched_links(page, [], markup=True) == ["https://cited.org/report"]


def test_custom_elements_comments_and_backticks_keep_citations():
    from ghost_agent.core.link_grounding import unfetched_links
    page = ('<!-- TODO add <script> analytics --><p><a href="https://cited.org/a">a</a></p>'
            '<script-note>See https://cited.org/c</script-note><link-card href="https://cited.org/d"></link-card>'
            "<script>var a='`';</script><a href=\"https://cited.org/paper\">p</a><script>var b='`';</script>")
    assert unfetched_links(page, [], markup=True) == ["https://cited.org/a", "https://cited.org/c",
                                                      "https://cited.org/d", "https://cited.org/paper"]


def test_an_unclosed_comment_keeps_the_rest_of_the_page():
    from ghost_agent.core.link_grounding import unfetched_links
    assert unfetched_links('<!-- draft <a href="https://cited.org/x">x</a>', [], markup=True) == ["https://cited.org/x"]


@pytest.mark.parametrize("sel", ['div:not(:is(.a), [data-x="1"])', 'a:has-text("a)b [x]")', "a:has-text('a)b [x]')"])
def test_balanced_removal_counts_depth_and_respects_quotes(sel):
    R = _runner()
    page = _Page({})
    assert asyncio.run(R._attribute_hint(page, sel)) == "" and page.asked == []


# ── r4: one quote-aware pass; comment early-closes ───────────────────────────

@pytest.mark.parametrize("sel,asked", [
    ('[aria-label="Next >>"].missing', ['[aria-label="Next >>"]']),
    ('[data-x=">>"] span', ['[data-x=">>"]']),
    ('[aria-label="Next >>"][data-app=pager] >> nth=5', ['[aria-label="Next >>"]', '[data-app=pager]']),
    ('text="Open >> [data-app=calc]"', []),
    ('[title=":not(x)"] [data-app=calc]', ['[title=":not(x)"]', '[data-app=calc]']),
    ('[title=":text("] [data-app=calc]', ['[title=":text("]', '[data-app=calc]']),
    ('#dock >> [data-app="calculator"] >> nth=0', ['[data-app="calculator"]']),
])
def test_quoted_values_never_leak_into_the_selector_structure(sel, asked):
    R = _runner()
    page = _Page({"aria-label": ["Next >>"], "data-x": [">>"], "data-app": ["pager"], "title": [":not(x)", ":text("]})
    asyncio.run(R._attribute_hint(page, sel))
    assert page.asked == asked


@pytest.mark.parametrize("html", ['<!--> <a href="https://cite.example/a">a</a> <!-- c -->',
                                  '<!---> <a href="https://cite.example/a">a</a> <!-- c -->',
                                  '<!-- x --!> <a href="https://cite.example/a">a</a> <!-- c -->'])
def test_comment_early_closes_follow_the_html_spec(html):
    from ghost_agent.core.link_grounding import unfetched_links
    assert unfetched_links(html, [], markup=True) == ["https://cite.example/a"]


def test_a_refused_port_after_the_stop_is_not_the_service(tmp_path):
    """A16b2: the port lease refused after the old service was stopped; the
    disk still held the OLD row (pid, old token) — it is not the new run."""
    inner = happy_handler()
    st = {"full": False}

    def h(cmd):
        if st["full"] and "s.bind" in cmd:
            return ("", 1)                          # nothing is bindable any more
        return inner(cmd)
    sup = ServiceSupervisor(FakeSandbox(tmp_path, h))
    sup.start("web", "python3 -m http.server 8101", port=8101)
    orig = sup._kill_service

    def kill(entry, others=()):
        orig(entry, others=others)
        st["full"] = True
    sup._kill_service = kill
    out = sup.start("web", "python3 app.py", port=8101)
    assert "The new command did NOT start: Error: could not grant a port" in out
    assert sup._load()["web"]["command"] == "python3 -m http.server 8101"


@pytest.mark.parametrize("sel", ["li:has-text(Item [x])", "li:text-is(Item [x])", 'li:contains("[x]")'])
def test_unquoted_text_arguments_and_unknown_pseudos_with_quoted_brackets(sel):
    R = _runner()
    page = _Page({})
    assert asyncio.run(R._attribute_hint(page, sel)) == "" and page.asked == []
