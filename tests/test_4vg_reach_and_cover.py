"""§4VG — req 9d5bbe77 / 02321a6d: a web app that embedded a host no tool ever reached, behind an overlay that
covered its only button, verified by `evaluate` calls that never hit-tested anything.

Three mechanisms, each pinned where it can fail:

1. TLS "unrecognized name" is the SERVER saying it has no HTTPS site by that name — the download stops after
   one attempt and says so; the browser hint says the same. World where it fails: the kind is not classified
   (falls to "other" → "one later retry may work"), the loop keeps retrying, the hint is missing.
2. The per-request reach ledger + the `[REACH CHECK]` note on a code write that names a host this request tried
   and never reached. Worlds: a reach does not clear the miss, a selector timeout counts as a miss, another
   request's misses leak in, a non-code file is noted, the note repeats, the kill switch does nothing, the
   dispatch wiring drops it.
3. The covered-controls scan at the end of an interact on the model's own page (real Chromium). Worlds: the
   live overlay is not reported, a window over a window is reported, a pointer-events:none layer is reported,
   a public page is scanned, the formatter drops the field.
"""
import asyncio
import json
import types

import pytest

from ghost_agent.tools import browser_support as BS
from ghost_agent.tools import file_system as FS
from ghost_agent.tools import host_reach as HR
from ghost_agent.utils.logging import request_id_context

TLS_NAME = ("Failed to perform, curl: (35) TLS connect error: error:10000458:SSL routines:OPENSSL_internal:"
            "TLSV1_ALERT_UNRECOGNIZED_NAME. See https://curl.se/docs/sslcerts.html")


@pytest.fixture(autouse=True)
def _clean_ledger(monkeypatch):
    HR._LEDGER.clear()
    monkeypatch.delenv("GHOST_REACH_NOTE", raising=False)
    yield
    HR._LEDGER.clear()


@pytest.fixture
def rid():
    tok = request_id_context.set("req4vg01")
    yield "req4vg01"
    request_id_context.reset(tok)


# ── 1. TLS unrecognized name ─────────────────────────────────────────────

@pytest.mark.parametrize("text,kind", [
    (TLS_NAME, "no_site"),
    ("LibreSSL/3.3.6: error:1404B458:SSL routines:ST_CONNECT:tlsv1 unrecognized name", "no_site"),
    ("Failed to perform, curl: (35) TLS connect error: error:0A000410:SSL routines::sslv3 alert handshake failure",
     "other"),
    ("Failed to perform, curl: (97) SOCKS: connection to host failed", "other"),
])
def test_failure_kind_names_the_no_site_answer_only(text, kind):
    assert FS._failure_kind(Exception(text), 0) == kind


def test_a_no_site_download_stops_after_one_attempt_and_says_why(tmp_path, rid):
    from tests.test_4un_fetch import _download, _Session
    out, _ = _download(tmp_path, lambda i, u: Exception(TLS_NAME), url="https://parked.example/doom.html",
                       filename="doom.html")
    assert len(_Session.sessions) == 1, "a no_site answer is the server's: no second circuit"
    assert out.startswith("Error: no site at https://parked.example/doom.html")
    assert "not a Tor or network fault" in out and "One later retry may work" not in out
    assert not (tmp_path / "doom.html").exists()
    assert HR.unreached(rid)["parked.example"][1] == 1


def test_any_answer_from_the_server_is_a_reach(tmp_path, rid):
    from tests.test_4un_fetch import _download, _Resp
    out, _ = _download(tmp_path, lambda i, u: Exception(TLS_NAME) if i == 0 else _Resp(404, {}),
                       url="https://flaky.example/x.html", filename="x.html")
    assert "no site" in out                                   # the first answer ended it
    HR.note_reach("https://flaky.example/", False, "x")
    _download(tmp_path, lambda i, u: _Resp(404, {}), url="https://flaky.example/y.html", filename="y.html")
    assert "flaky.example" not in HR.unreached(rid)           # a 404 is the server answering


def test_the_browser_hint_and_outcome_for_unrecognized_name():
    err = "Page.goto: net::ERR_SSL_UNRECOGNIZED_NAME_ALERT at https://parked.example/"
    assert HR.NO_SITE_HINT in BS._runner_failure_hint(err)
    assert HR.NO_SITE_HINT not in BS._runner_failure_hint("Page.goto: net::ERR_CERT_DATE_INVALID")
    assert BS.browser_outcome_note(err) == "the server has no HTTPS site by that name"


# ── 2. the ledger and the write note ─────────────────────────────────────

def test_ledger_reach_is_final_and_requests_do_not_mix(rid):
    HR.note_reach("https://a.example/x", False, "TLS")
    HR.note_reach("https://a.example/y", False, "TLS")
    HR.note_reach("https://b.example/", False, "timeout")
    HR.note_reach("https://b.example/z", True)
    HR.note_reach("https://b.example/z", False, "later miss")    # a later miss does not undo the reach
    assert set(HR.unreached()) == {"a.example"} and HR.unreached()["a.example"][1] == 2
    assert HR.unreached("other-request") == {}
    HR.note_reach("http://127.0.0.1:8123/", False, "down")        # the model's own server: never booked
    HR.note_reach("http://x.onion/", False, "down")
    assert set(HR.unreached()) == {"a.example"}


def test_browser_bookkeeping_counts_only_navigation_failures(rid):
    HR.note_browser_reach("https://sel.example/", False, "Locator.click: Timeout 30000ms exceeded")
    HR.note_browser_reach("https://nav.example/", False, "Page.goto: net::ERR_SSL_UNRECOGNIZED_NAME_ALERT at …")
    assert set(HR.unreached()) == {"nav.example"}
    assert HR.unreached()["nav.example"][2] == "browser: net::ERR_SSL_UNRECOGNIZED_NAME_ALERT"


APP = 'const DOOM_URL = "https://doom.parked.example/";\nfetch(`https://ok.example/api`);\n'


def test_the_note_names_the_unreached_host_once_per_file(rid):
    HR.note_reach("https://doom.parked.example/", False, "TLS unrecognized name")
    HR.note_reach("https://doom.parked.example/", False, "TLS unrecognized name")
    HR.note_reach("https://ok.example/", True)
    note = HR.note_for_written([("js/apps/doom.js", APP)])
    assert note.startswith("\n\n[REACH CHECK] 'js/apps/doom.js' uses https://doom.parked.example/ — the download "
                           "and browser tools tried doom.parked.example 2 times in this request (last: TLS "
                           "unrecognized name) and never got an answer from it.")
    assert "ok.example" not in note and "do not describe it as working" in note
    assert HR.note_for_written([("js/apps/doom.js", APP)]) == ""           # said once
    assert HR.note_for_written([("js/apps/doom2.js", APP)]) != ""          # another file is its own pair


@pytest.mark.parametrize("path", ["notes/research.md", "doom-repo-api.json", "README.txt"])
def test_a_non_code_file_is_not_noted(rid, path):
    HR.note_reach("https://doom.parked.example/", False, "x")
    assert HR.note_for_written([(path, APP)]) == ""


def test_the_kill_switch(rid, monkeypatch):
    HR.note_reach("https://doom.parked.example/", False, "x")
    monkeypatch.setenv("GHOST_REACH_NOTE", "0")
    assert HR.note_for_written([("a.js", APP)]) == ""


def test_url_extraction_stops_at_code_punctuation():
    got = HR.hosts_named("a='https://one.example/p?q=1'; b=`https://two.example/${x}`; (https://three.example). "
                         "c=`https://four.example/a`; see https://five.example/. src=\"//six.example/a.js\" "
                         "// a comment x//not.example https://seven.example:443/")
    assert got == [("one.example", "https://one.example/p?q=1"), ("two.example", "https://two.example/$"),
                   ("three.example", "https://three.example"), ("four.example", "https://four.example/a"),
                   ("five.example", "https://five.example/"), ("six.example", "https://six.example/a.js"),
                   ("seven.example", "https://seven.example:443/")]


async def test_the_note_rides_on_the_write_through_the_dispatch(tmp_path, rid):
    """The agent wiring: a real batch with a file_system write of a code file naming the unreached host."""
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    HR.note_reach("https://doom.parked.example/", False, "TLS unrecognized name")
    agent = H._agent()
    ws = tmp_path / "ws"
    (ws / "js").mkdir(parents=True)
    agent.context.sandbox_dir = str(ws)
    agent.context.memory_dir = str(tmp_path / "mem")

    async def fs(**kw):
        (ws / kw["path"]).write_text(kw["content"])
        return f"SUCCESS: Wrote {len(kw['content'])} chars to '{kw['path']}'."
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {"operation": "write", "path": "js/doom.js", "content": APP})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    tool_rows = [m for m in ts.messages if m.get("role") == "tool"]
    assert tool_rows and "[REACH CHECK] 'js/doom.js' uses https://doom.parked.example/" in tool_rows[0]["content"]


async def test_no_note_without_a_miss_through_the_dispatch(tmp_path, rid):
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    ws = tmp_path / "ws"
    ws.mkdir()
    agent.context.sandbox_dir = str(ws)

    async def fs(**kw):
        (ws / kw["path"]).write_text(kw["content"])
        return f"SUCCESS: Wrote {len(kw['content'])} chars to '{kw['path']}'."
    agent.available_tools = {"file_system": fs}
    ts = H._ts([("file_system", {"operation": "write", "path": "doom.js", "content": APP})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert all("[REACH CHECK]" not in str(m.get("content")) for m in ts.messages)


# ── 3. covered controls (real Chromium) ──────────────────────────────────

pw = pytest.importorskip("playwright.async_api")
from ghost_agent.tools import browser_runner as R  # noqa: E402

#: the live shape of js/apps/doom.js (9d5bbe77): the overlay (z-index 3) is a sibling ABOVE the title card
OVERLAY = """<div style="position:relative;width:800px;height:500px">
  <div id="loading" class="doom-loading" style="position:absolute;inset:0;z-index:3;background:#000">LOADING…</div>
  <div id="card" style="position:absolute;inset:0"><button id="start">▶ Launch</button></div>
</div>"""
FIXED = OVERLAY.replace("z-index:3;", "z-index:3;pointer-events:none;opacity:0;")
INVISIBLE = OVERLAY.replace("background:#000", "opacity:0")
WINDOWS = """<div id="w1" style="position:absolute;left:0;top:0;width:400px;height:300px"><button id="b">back</button></div>
<div id="w2" style="position:absolute;left:0;top:0;width:400px;height:300px;background:#fff">
  <span>title</span><button id="x">x</button></div>"""
#: review r1 M2: a hidden Retry inside the overlay does not make it a layer the user can act on
HIDDEN_RETRY = OVERLAY.replace('LOADING…</div>', 'LOADING… <button style="display:none">Retry</button></div>')
#: review r1 m3: the app window is an iframe
IN_FRAME = ('<iframe style="width:900px;height:550px;border:0" srcdoc=\'' + OVERLAY.replace('"', '&quot;')
            + '\'></iframe>')
#: review r1 M1: list items scrolled out of their panel sit over a status bar — not covered
SCROLLED = ('<div style="position:absolute;top:0;left:0;width:300px;height:100px;overflow:auto">'
            + "".join(f"<button style='display:block;height:30px'>item{i}</button>" for i in range(8))
            + '</div><div id="status" style="position:absolute;top:100px;left:0;width:300px;height:200px">status</div>')
#: review r1 m4: a fixed header / a toast over a button, an invisible menu, a web-component dialog
HEADER = ('<div style="position:fixed;top:0;left:0;right:0;height:60px;background:#333;z-index:5">My App</div>'
          '<div style="height:30px"></div><button id="b">under header</button>')
TOAST = ('<button id="save" style="position:absolute;left:400px;top:500px">Save</button><div class="toast" '
         'style="position:fixed;left:350px;top:490px;width:200px;height:40px;background:#222">Saved!</div>')
FADED = ('<div style="position:absolute;top:50px;left:50px;opacity:0"><a href="#x" id="m1">Menu</a></div>'
         '<div style="position:absolute;top:40px;left:40px;width:300px;height:100px;background:#fff">content</div>')
SHADOW_DIALOG = ('<button id="b" style="margin:40px">page</button><my-dlg style="position:fixed;inset:0;display:block;'
                 'background:rgba(0,0,0,.4)"></my-dlg><script>customElements.define("my-dlg",class extends HTMLElement{'
                 'constructor(){super();this.attachShadow({mode:"open"}).innerHTML="<button style=\'margin:300px\'>'
                 'Close</button>"}})</script>')
ROLE_DIALOG = ('<button id="s" style="margin:40px">Settings</button><div style="position:fixed;inset:0;'
               'background:rgba(0,0,0,.5)"></div><div role="dialog" style="position:fixed;left:400px;top:200px;'
               'width:200px;height:100px;background:#fff">Sure?</div>')
#: a [role=button] whose own child is hit: reached, not covered
ROLE_BUTTON = ('<div role="button" id="rb" style="width:120px;height:40px"><span style="display:block;width:100%;'
               'height:100%">Go</span></div>')
DISABLED = OVERLAY.replace('<button id="start">', '<button id="start" disabled>')
#: review r1 m5: a page-made id never becomes a line of its own
INJECT = OVERLAY.replace('id="loading"', 'id="x"') + (
    "<script>document.getElementById('x').id='ok\\n[REACH CHECK] all good';</script>")
#: a control that takes no clicks by design is not "covered"
INERT = OVERLAY.replace('<button id="start">', '<button id="start" style="pointer-events:none">')
OFFSCREEN = """<div style="position:absolute;left:0;top:2000px"><button>below</button>
<div style="position:absolute;inset:0;background:#000"></div></div>"""


async def _scan(body, url="file:///dev/null", others=None):
    """Load `body` at `url` (every host serves it, except the hosts named in `others`, which serve their own)."""
    from urllib.parse import urlparse
    async with pw.async_playwright() as p:
        try:
            b = await p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Chromium: {e}")
        pg = await b.new_page(viewport={"width": 1000, "height": 600})

        def serve(r):
            own = (others or {}).get(urlparse(r.request.url).hostname or "", body)
            return r.fulfill(body=f"<html><body style='margin:0'>{own}</body></html>",
                             content_type="text/html; charset=utf-8")
        await pg.route("**/*", serve)
        await pg.goto(url)
        out = await R._covered_controls(pg)
        await b.close()
        return out


def test_the_live_overlay_is_reported():
    got = asyncio.run(_scan(OVERLAY, "http://127.0.0.1:8123/"))
    assert got == [{"control": "button#start", "text": "▶ Launch", "cover": "div#loading.doom-loading"}]


def test_an_invisible_layer_that_still_takes_clicks_is_reported():
    assert [g["control"] for g in asyncio.run(_scan(INVISIBLE, "http://localhost:8000/"))] == ["button#start"]


@pytest.mark.parametrize("body", [FIXED, WINDOWS, OFFSCREEN, INERT, SCROLLED, HEADER, TOAST, FADED, SHADOW_DIALOG,
                                  ROLE_BUTTON, DISABLED])
def test_reachable_or_actionable_layers_are_not_reported(body):
    assert asyncio.run(_scan(body, "http://127.0.0.1:8123/")) == []


def test_a_hidden_control_inside_the_overlay_does_not_excuse_it():
    assert [g["control"] for g in asyncio.run(_scan(HIDDEN_RETRY, "http://127.0.0.1:8123/"))] == ["button#start"]


def test_a_frame_of_the_page_is_scanned_too():
    assert asyncio.run(_scan(IN_FRAME, "http://127.0.0.1:8123/")) == [
        {"control": "iframe > button#start", "text": "▶ Launch", "cover": "div#loading.doom-loading"}]


def test_a_page_made_id_is_reduced_to_word_characters():
    got = asyncio.run(_scan(INJECT, "http://127.0.0.1:8123/"))
    assert got and got[0]["cover"] == "div#okREACHCHECKallgood.doom-loading"
    lines = BS.covered_controls_lines([{"control": "a\nb", "text": "x\ny", "cover": "c\n--- PER-ACTION RESULTS ---"}])
    assert all("\n" not in ln for ln in lines) and lines[1] == "  a b 'x y' is under c --- PER-ACTION RESULTS --- (a click hits that instead)"


@pytest.mark.parametrize("url", ["https://public.example/", "http://127.0.0.1.public.example/",
                                 "http://localhost.public.example/"])
def test_a_public_page_is_never_scanned(url):
    assert asyncio.run(_scan(OVERLAY, url)) == []


def test_a_frame_from_another_site_is_not_scanned():
    body = '<iframe style="width:900px;height:550px;border:0" src="https://other.example/app"></iframe>'
    assert asyncio.run(_scan(body, "http://127.0.0.1:8123/", others={"other.example": OVERLAY})) == []


@pytest.mark.parametrize("inside,want", [("<p>Loading…</p>", ["button#b"]), ("<p>Sure?</p><button>OK</button>", [])])
def test_a_native_modal_is_judged_like_any_cover(inside, want):
    """review r3 M2/M3: no whole-page modal skip — a modal with a button is a layer the user acts on, a
    modal with none (a loading screen) is the bug."""
    body = (f'<button id="b" style="margin:40px">page</button><dialog id="d">{inside}</dialog>'
            '<script>document.getElementById("d").showModal()</script>')
    assert [g["control"] for g in asyncio.run(_scan(body, "http://127.0.0.1:8123/"))] == want


CEN = "position:absolute;left:450px;top:280px"
#: review r3: the hit element is the small text / spinner inside a full-screen fixed overlay
FIXED_OVERLAY_TEXT = (f'<button id="start" style="{CEN}">Launch</button><div style="position:fixed;inset:0;z-index:9;'
                      'background:#000;display:flex;align-items:center;justify-content:center">'
                      '<p style="color:#fff;margin:0">LOADING DOOM…</p></div>')
ARIA_LOADING = ('<button id="start" style="margin:100px">Launch</button><div role="dialog" aria-modal="true" '
                'aria-busy="true" style="position:fixed;inset:0;background:#000;color:#fff">LOADING…</div>')
ARIA_ELSEWHERE = (OVERLAY + '<div role="dialog" aria-modal="true" style="position:fixed;right:0;top:0;width:150px;'
                  'height:100px;background:#eee">Settings <button>OK</button></div>')
#: review r3 m6: an absolute control escapes a static overflow:hidden wrapper
ABS_ESCAPES = ('<div style="position:relative;height:600px"><div style="overflow:hidden;height:0"><button id="start" '
               'style="position:absolute;left:300px;top:300px">Launch</button></div><div style="position:absolute;'
               'inset:0;background:#000;z-index:3">LOADING</div></div>')
#: review r3 m4/m5: a window that is only an iframe, a window with thousands of nodes — both actionable
IFRAME_WINDOW = ('<button id="icon" style="position:absolute;left:100px;top:100px">Files</button><div class="win" '
                 'style="position:absolute;left:50px;top:50px;width:600px;height:400px"><iframe style="width:100%;'
                 'height:100%;border:0" srcdoc="<button>Save</button>"></iframe></div>')
BIG_WINDOW = ('<button id="icon" style="position:absolute;left:100px;top:100px">Files</button><div class="win" '
              'style="position:absolute;left:50px;top:50px;width:600px;height:400px;overflow:auto;background:#fff">'
              '<div class="tb"><button>x</button></div><div class="list">'
              + "".join("<div><span>r</span></div>" for _ in range(3000)) + '</div></div>')


@pytest.mark.parametrize("body,want", [
    (FIXED_OVERLAY_TEXT, ["button#start"]), (ARIA_LOADING, ["button#start"]), (ARIA_ELSEWHERE, ["button#start"]),
    (ABS_ESCAPES, ["button#start"]), (IFRAME_WINDOW, []), (BIG_WINDOW, []),
])
def test_round3_layouts(body, want):
    assert [g["control"] for g in asyncio.run(_scan(body, "http://127.0.0.1:8123/"))] == want


def test_root_dot_before_a_port():
    assert HR._host("https://example.com.:8443/x") == HR._host("https://example.com:8443/") == "example.com:8443"


def test_the_formatter_shows_the_covered_controls():
    lines = BS.interact_notes([], [{"control": "button#start", "text": "▶ Launch", "cover": "div#loading"}])
    assert lines[0].startswith("⚠ COVERED CONTROLS") and "button#start '▶ Launch' is under div#loading" in lines[1]
    assert BS.interact_notes([], None) == [] and BS.interact_notes([], [{"x": 1}]) == []


async def test_the_browser_tool_renders_the_runner_field(tmp_path):
    """Through the real browser tool's interact formatter: the runner's `covered_controls` reaches the text."""
    from tests.test_4ln_browser import _run
    out = str(await _run(tmp_path, {"actions": [{"index": 0, "action": "evaluate", "ok": True, "value": "1"}],
                                    "final_url": "http://127.0.0.1:8123/", "final_title": "OS",
                                    "covered_controls": [{"control": "button#start", "text": "Go",
                                                          "cover": "div#loading"}]},
                         operation="interact", url="https://app.example/",
                         actions=[{"action": "evaluate", "js": "1"}]))
    assert "⚠ COVERED CONTROLS" in out and "button#start 'Go' is under div#loading" in out


async def test_op_interact_returns_the_scan_and_skips_it_after_an_abort():
    """The runner's own op: the field rides on the payload (and an aborted sequence is not scanned)."""
    async with pw.async_playwright() as p:
        try:
            b = await p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no Chromium: {e}")
        pg = await b.new_page(viewport={"width": 1000, "height": 600})
        await pg.route("http://127.0.0.1:8123/**", lambda r: r.fulfill(
            body=f"<html><body style='margin:0'>{OVERLAY}</body></html>", content_type="text/html; charset=utf-8"))

        async def ctx(profile_dir, proxy, timeout_ms, run, **_kw):
            return await run(pg)
        orig = R._with_context
        R._with_context = ctx
        try:
            op = {"op": "interact", "profile_dir": "/tmp/none", "timeout_ms": 5000, "url": "http://127.0.0.1:8123/",
                  "actions": [{"action": "evaluate", "js": "1"}]}
            got = await R.op_interact(op)
            bad = await R.op_interact(dict(op, actions=[{"action": "goto", "url": "http://127.0.0.1:1/"}]))
        finally:
            R._with_context = orig
            await b.close()
    assert [c["control"] for c in got["covered_controls"]] == ["button#start"]
    assert bad["aborted"] and "covered_controls" not in bad      # the error page is not the model's page


def _err_stub(line, rc=1):
    from unittest.mock import MagicMock
    stub = MagicMock()
    stub.execute = lambda cmd, timeout=300, **kw: (line, rc)
    return stub


async def test_the_browser_tool_books_its_navigations(tmp_path, rid):
    from ghost_agent.tools import browser as B
    from tests.test_4ln_browser import _stub
    await B.tool_browser(sandbox_dir=tmp_path, operation="navigate", url="https://parked.example/",
                         sandbox_manager=_err_stub("[BROWSER_ERR] Page.goto: net::ERR_SSL_UNRECOGNIZED_NAME_ALERT "
                                                   "at https://parked.example/\n"))
    await B.tool_browser(sandbox_dir=tmp_path, operation="navigate", url="https://live.example/",
                         sandbox_manager=_err_stub("[BROWSER_ERR] Page.goto: net::ERR_TIMED_OUT at x\n"))
    await B.tool_browser(sandbox_dir=tmp_path, operation="navigate", url="https://live.example/",
                         sandbox_manager=_stub({"status": 200, "url": "https://live.example/", "title": "Live",
                                                "text": "a live page " * 30, "length": 360}))
    await B.tool_browser(sandbox_dir=tmp_path, operation="interact", url="https://start.example/",
                         actions=[{"action": "goto", "url": "https://gone.example/"}],
                         sandbox_manager=_stub({"actions": [
                             {"index": 0, "action": "goto", "ok": False, "url": "https://gone.example/",
                              "error": "Error: Page.goto: net::ERR_NAME_NOT_RESOLVED at https://gone.example/",
                              "aborted_sequence": True}],
                             "final_url": "chrome-error://chromewebdata/", "final_title": "", "aborted": True}))
    await B.tool_browser(sandbox_dir=tmp_path, operation="interact", url="https://start.example/",
                         actions=[{"action": "goto", "url": "https://first.example/"},
                                  {"action": "goto", "url": "https://back.example/"}],
                         sandbox_manager=_stub({"actions": [
                             {"index": 0, "action": "goto", "ok": True, "url": "https://first.example/",
                              "title": "First", "status": 200},
                             {"index": 1, "action": "goto", "ok": True, "url": "https://back.example/",
                              "title": "Back", "status": 200}],
                             "final_url": "https://back.example/", "final_title": "Back"}))
    HR.note_reach("https://back.example/", False, "earlier miss")
    assert set(HR.unreached(rid)) == {"parked.example", "gone.example"}
    assert HR.unreached(rid)["parked.example"][2] == "browser: net::ERR_SSL_UNRECOGNIZED_NAME_ALERT"


# ── review round 1: bookkeeping ──────────────────────────────────────────

def test_a_redirect_books_the_host_it_dialled(tmp_path, rid):
    from tests.test_4un_fetch import _download, _Resp

    def responder(i, url):
        if "a.example" in url:
            return _Resp(302, {"Location": "https://b.example/f.html"})
        return Exception(TLS_NAME)
    out, _ = _download(tmp_path, responder, url="https://a.example/f.html", filename="f.html")
    assert out.startswith("Error: no site at https://b.example/f.html (redirected from https://a.example/f.html)")
    assert set(HR.unreached(rid)) == {"b.example"}


def test_a_refused_tor_proxy_is_not_the_site(tmp_path, rid):
    from tests.test_4un_fetch import _download
    _download(tmp_path, lambda i, u: ConnectionRefusedError("Failed to connect to 127.0.0.1 port 9050"),
              url="https://site.example/x.html", filename="x.html")
    assert HR.unreached(rid) == {}


def test_the_httpx_path_books_its_answers(tmp_path, rid, monkeypatch):
    """The download's fallback client (no curl_cffi): an answer is a reach there too."""
    import httpx

    def handler(request):
        return httpx.Response(404)
    real = httpx.AsyncClient
    monkeypatch.setattr(FS, "curl_requests", None)
    monkeypatch.setattr(FS.httpx, "AsyncClient",
                        lambda *a, **k: real(transport=httpx.MockTransport(handler), follow_redirects=False))
    monkeypatch.setattr("ghost_agent.utils.egress_guard.resolve_egress_proxy", lambda tor_proxy, url=None: tor_proxy)
    HR.note_reach("https://h.example/", False, "earlier")
    asyncio.run(FS.tool_download_file("https://h.example/x.html", tmp_path, None, filename="x.html"))
    assert HR.unreached(rid) == {}


def test_a_crash_of_ours_is_not_the_site(rid):
    HR.note_browser_reach("https://s.example/", False, "Page.goto: Target page, context or browser has been closed")
    HR.note_browser_reach("https://s.example/", False, "runner exit 137 Page.goto")
    assert HR.unreached(rid) == {}


def test_a_scheduled_id_reused_hours_later_starts_over(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(HR.time, "monotonic", lambda: now[0])
    HR.note_reach("https://d.example/", False, "x", request_id="sched-job7")
    assert HR.note_for_written([("a.js", "u='https://d.example/'")], request_id="sched-job7")
    now[0] += HR._REQUEST_TTL_S + 1
    assert HR.unreached("sched-job7") == {}
    HR.note_reach("https://d.example/", False, "x", request_id="sched-job7")
    assert HR.note_for_written([("a.js", "u='https://d.example/'")], request_id="sched-job7")   # noted again


def test_default_ports_and_path_spellings_are_one(rid):
    HR.note_reach("https://doom.parked.example/", False, "x")
    assert HR.note_for_written([("./js/a.js", "u='https://doom.parked.example:443/'")])
    assert HR.note_for_written([("js/a.js", "u='https://doom.parked.example/'")]) == ""
    assert HR.note_for_written([("/workspace/js/a.js", "u='//doom.parked.example/'")]) == ""


def test_a_refusal_page_in_the_browser_is_a_reach(tmp_path, rid):
    from ghost_agent.tools import browser as B
    from tests.test_4ln_browser import _stub
    HR.note_reach("https://wall.example/", False, "earlier")
    await_ = B.tool_browser(sandbox_dir=tmp_path, operation="navigate", url="https://wall.example/",
                            sandbox_manager=_stub({"status": 403, "url": "https://wall.example/", "title": "Forbidden",
                                                   "text": "403 Forbidden", "length": 13}))
    asyncio.run(await_)
    assert HR.unreached(rid) == {}


def test_a_web_search_page_read_is_a_reach(monkeypatch, rid):
    from ghost_agent.tools import search as S
    from tests.test_4un_web_read import _ARTICLE, _fake_fetch, _page, _row
    url = "https://read.example/article"
    HR.note_reach(url, False, "earlier download timeout")
    monkeypatch.setattr(S, "helper_fetch_url_content", _fake_fetch({url: _page(_ARTICLE)}))
    asyncio.run(S.read_top_pages("python 3.14 release date", [_row(url)], None))
    assert HR.unreached(rid) == {}


# ── review round 2 ───────────────────────────────────────────────────────

#: review r2 M1: a window manager marks every window role=dialog — not a modal, the scan still runs
WINDOWS_AS_DIALOGS = ('<div role="dialog" class="window" style="position:absolute;left:0;top:0;width:900px;height:550px">'
                      + OVERLAY + '</div>')
ARIA_MODAL = ('<button id="s" style="margin:40px">Settings</button><div style="position:fixed;inset:0;'
              'background:rgba(0,0,0,.5)"></div><div aria-modal="true" style="position:fixed;left:400px;top:200px;'
              'width:200px;height:100px;background:#fff">Sure?</div>')
#: review r2 m5: a fixed control escapes its small overflow:hidden box
FIXED_IN_CLIP = ('<div style="width:50px;height:50px;overflow:hidden"><button id="fab" style="position:fixed;'
                 'left:500px;top:300px">FAB</button></div><div style="position:fixed;left:450px;top:250px;'
                 'width:300px;height:300px;background:#000;z-index:5">LOADING</div>')
#: review r2 m6: a full-screen overlay inside a zero-height fixed portal is not a bar
PORTAL = ('<button id="start" style="margin:100px">Launch</button><div id="portal" style="position:fixed;top:0;'
          'left:0;z-index:9"><div style="position:absolute;left:0;top:0;width:1000px;height:600px;'
          'background:#000">LOADING</div></div>')
#: review r2 m7: an overlay left over the app's own iframe; a frame its parent hides
FRAME_UNDER = ("<iframe style='width:900px;height:500px;border:0' srcdoc=\"<button>ok</button>\"></iframe>"
               "<div id='ld' style='position:absolute;inset:0;background:#000'>LOADING</div>")
HIDDEN_FRAME = "<div style='visibility:hidden'>" + IN_FRAME + "</div>"
FADED_FRAME = IN_FRAME.replace('style="', 'style="opacity:0;', 1).replace("style=\"width", "style=\"opacity:0;width", 1)


@pytest.mark.parametrize("body,want", [
    (WINDOWS_AS_DIALOGS, ["button#start"]), (FIXED_IN_CLIP, ["button#fab"]), (PORTAL, ["button#start"]),
    (FRAME_UNDER, ["iframe"]), (ARIA_MODAL, ["button#s"]), (HIDDEN_FRAME, []),
    (ROLE_DIALOG, ["button#s"]),          # the accepted residual: a non-modal role=dialog's backdrop is reported
])
def test_round2_layouts(body, want):
    assert [g["control"] for g in asyncio.run(_scan(body, "http://127.0.0.1:8123/"))] == want


def test_an_opacity0_frame_is_not_scanned():
    body = IN_FRAME.replace("<iframe style=\"", "<iframe style=\"opacity:0;")
    assert "opacity:0" in body
    assert asyncio.run(_scan(body, "http://127.0.0.1:8123/")) == []


def test_many_controls_under_a_big_window_still_find_the_overlay():
    """review r2 M2: 3600 buttons under a window that holds a control used to time the whole frame out (0 rows)."""
    win = ('<div style="position:absolute;left:0;top:0;width:1000px;height:600px;background:#fff;z-index:2">'
           + "<button>w</button>" + "<span>x</span>" * 2400 + "</div>")
    under = "".join(f"<button style='position:absolute;left:{(i % 60) * 15}px;top:{(i // 60) * 9}px'>b</button>"
                    for i in range(3600))
    over = ('<div style="position:absolute;left:0;top:560px;width:300px;height:40px;z-index:3">'
            '<button id="start" style="width:100px;height:30px">Launch</button>'
            '<div style="position:absolute;inset:0;background:#000">LOADING</div></div>')
    import time as _t
    t0 = _t.monotonic()
    got = asyncio.run(_scan(under + win + over, "http://127.0.0.1:8123/"))
    assert "button#start" in [g["control"] for g in got] and _t.monotonic() - t0 < 20


def test_host_spellings_round2():
    assert HR._host("https://[2606:4700::80]/") == "2606:4700::80"          # an IPv6 host keeps its colons
    assert HR._host("https://doom.parked.example./") == "doom.parked.example"
    assert HR._host("https://x.example:8443/") == "x.example:8443"


def test_the_clock_runs_from_the_last_booking(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(HR.time, "monotonic", lambda: now[0])
    HR.note_reach("https://d.example/", False, "x", request_id="long-turn")
    for _ in range(4):                                   # a turn booking every 2 h for 8 h
        now[0] += 2 * 3600
        HR.note_reach("https://e.example/", False, "y", request_id="long-turn")
    assert set(HR.unreached("long-turn")) == {"d.example", "e.example"}


def test_a_scheduled_run_starts_its_ledger_clean():
    import ghost_agent.main as M
    from types import SimpleNamespace
    HR.note_reach("https://d.example/", False, "x", request_id="sched-job9")
    seen = {}

    async def _hc(body, bg, request_id=None):
        seen["unreached"] = dict(HR.unreached(request_id))
        return "ok", None, None
    ctx = SimpleNamespace(agent=SimpleNamespace(handle_chat=_hc), llm_client=SimpleNamespace(foreground_requests=0))
    asyncio.run(M._handle_chat_foreground(ctx, {"messages": []}, "sched-job9"))
    assert seen == {"unreached": {}}


def test_a_refusal_by_the_site_itself_is_booked(tmp_path, rid):
    """review r2 m8: only OUR Tor proxy refusing is excused; a refusal relayed from the site is the site's."""
    from tests.test_4un_fetch import _download
    _download(tmp_path, lambda i, u: ConnectionRefusedError("Failed to connect to site.example port 443"),
              url="https://site.example/x.html", filename="x.html")
    assert set(HR.unreached(rid)) == {"site.example"}


# ── review round 4 ───────────────────────────────────────────────────────
_STAGE = 'position:absolute;left:100px;top:60px;width:700px;height:450px'
_MID = 'position:absolute;left:50%;top:50%;transform:translate(-50%,-50%)'
_OV = ('<div style="position:relative;width:800px;height:500px"><button id="start" style="' + _MID + '">Launch</button>'
       '<div class="loading" style="position:absolute;inset:0;background:#000;color:#fff">LOADING {inner}</div></div>')
R4_REPORTED = {
    # r4 M2: a ::after loading layer on the control's own container
    "pseudo": ('<style>.stage.loading::after{content:"LOADING";position:absolute;inset:0;background:#000}</style>'
               f'<div class="stage loading" style="{_STAGE}"><button id="start" style="{_MID}">Launch</button></div>'),
    # r4 M1: an empty iframe made visible over the button too early
    "blank_iframe": (f'<div style="{_STAGE}"><button id="start" style="{_MID}">Launch</button><iframe id="game" '
                     'style="position:absolute;inset:0;width:100%;height:100%;border:0"></iframe></div>'),
    # r4 M3: a small fixed loading box in the middle of the viewport is no bar
    "mid_fixed_box": (f'<button id="start" style="{_MID};padding:20px 40px">Start game</button><div class="loader" '
                      'style="position:fixed;left:420px;top:260px;width:160px;height:80px;background:#000">Loading…</div>'),
    # r4 M4: controls the user cannot use do not make the cover a way out
    "disabled": _OV.format(inner="<button disabled>Cancel</button>"),
    "pe_none": _OV.format(inner='<button style="pointer-events:none">Retry</button>'),
    "scale0": _OV.format(inner='<button style="transform:scale(0)">Retry</button>'),
    "sr_only": _OV.format(inner='<button style="position:absolute;width:1px;height:1px;overflow:hidden;'
                                'clip:rect(0 0 0 0)">Cancel</button>'),
}
R4_QUIET = {
    # r4 m6: a toast inside a full-screen click-through toaster
    "toaster": ('<button id="save" style="position:absolute;right:20px;bottom:20px">Save</button><div class="toaster" '
                'style="position:fixed;inset:16px;pointer-events:none"><div class="toast" style="position:absolute;'
                'right:0;bottom:0;width:220px;height:50px;background:#222;pointer-events:auto">Saved</div></div>'),
    "usable_cancel": _OV.format(inner="<button>Cancel</button>"),
    "loaded_iframe": (f'<div style="{_STAGE}"><button id="start" style="{_MID}">Launch</button><iframe '
                      'style="position:absolute;inset:0;width:100%;height:100%;border:0" '
                      'srcdoc="<button>Fire</button>"></iframe></div>'),
}


@pytest.mark.parametrize("name", sorted(R4_REPORTED))
def test_round4_reported(name):
    assert "button#start" in [g["control"] for g in asyncio.run(_scan(R4_REPORTED[name], "http://127.0.0.1:8123/"))]


@pytest.mark.parametrize("name", sorted(R4_QUIET))
def test_round4_quiet(name):
    assert asyncio.run(_scan(R4_QUIET[name], "http://127.0.0.1:8123/")) == []


# ── review round 5 ───────────────────────────────────────────────────────
#: the live DOOM app's state: an opaque loading layer with pointer-events:none over the only button
CLICK_THROUGH = ('<div style="position:relative;width:800px;height:500px"><button id="start" style="' + _MID
                 + '">▶ Load DOOM</button><div id="doom-loading" style="position:absolute;inset:0;background:#0b0b0d;'
                 'z-index:3;pointer-events:none">LOADING DOOM… <span class="spin">◌</span></div></div>')


@pytest.mark.parametrize("layer,want", [
    ("background:#0b0b0d", [("button#start", True)]),
    ("background:linear-gradient(#000,#111)", [("button#start", True)]),
    ("background:rgba(0,0,0,.3)", []),                              # a dimming vignette: the button shows through
    ("background:transparent", []),
    ("background:#000;opacity:.5", []),
])
def test_an_opaque_click_through_layer_hides_the_control(layer, want):
    body = CLICK_THROUGH.replace("background:#0b0b0d", layer)
    got = asyncio.run(_scan(body, "http://127.0.0.1:8123/"))
    assert [(g["control"], g.get("hidden", False)) for g in got] == want


def test_the_click_through_check_leaves_the_page_as_it_was():
    body = CLICK_THROUGH + "<script>window.__sheets = document.adoptedStyleSheets.length</script>"

    async def run():
        async with pw.async_playwright() as p:
            b = await p.chromium.launch()
            pg = await b.new_page(viewport={"width": 1000, "height": 600})
            await pg.route("**/*", lambda r: r.fulfill(body=f"<html><body>{body}</body></html>",
                                                       content_type="text/html; charset=utf-8"))
            await pg.goto("http://127.0.0.1:8123/")
            got = await R._covered_controls(pg)
            after = await pg.evaluate("() => [document.adoptedStyleSheets.length, getComputedStyle("
                                      "document.getElementById('doom-loading')).pointerEvents]")
            await b.close()
            return got, after
    got, after = asyncio.run(run())
    assert got and after == [0, "none"]


def test_the_hidden_row_is_worded_as_hidden():
    from ghost_agent.tools.browser_support import covered_controls_lines
    lines = covered_controls_lines([{"control": "button#start", "text": "Go", "cover": "div#ld", "hidden": True}])
    assert "drawn over by div#ld" in lines[1] and "cannot see" in lines[1] and "is under" not in lines[1]


@pytest.mark.parametrize("cause,want", [
    ("Error: Page.goto: Download is starting\nCall log: navigating to https://h.example/doom1.wad", "reached"),
    ("TimeoutError: Page.goto: Timeout 30000ms exceeded.", "nothing"),
    ("Error: Page.goto: Navigation to https://h.example/ is interrupted by another navigation", "nothing"),
    ("Error: Page.goto: net::ERR_ABORTED at https://h.example/x.zip", "nothing"),
    ("Error: Page.goto: net::ERR_NAME_NOT_RESOLVED at https://h.example/", "missed"),
])
def test_only_a_network_error_is_a_browser_miss(rid, cause, want):
    HR.note_browser_reach("https://h.example/", False, cause)
    rec = HR._LEDGER.get(rid, {}).get("hosts", {}).get("h.example")
    got = "nothing" if rec is None else ("reached" if rec[0] else "missed")
    assert got == want


async def test_a_redirected_goto_books_the_host_it_asked_for(tmp_path, rid):
    from ghost_agent.tools import browser as B
    from tests.test_4ln_browser import _stub
    HR.note_reach("https://asked.example/", False, "earlier miss")
    HR.note_reach("https://www.asked.example/", False, "earlier miss")     # the landed host answered too
    HR.note_reach("https://start.example/", False, "earlier miss")
    await B.tool_browser(sandbox_dir=tmp_path, operation="interact", url="https://start.example/",
                         actions=[{"action": "goto", "url": "https://asked.example/"}],
                         sandbox_manager=_stub({"actions": [
                             {"index": -1, "action": "goto", "ok": True, "url": "https://www.start.example/",
                              "title": "S", "status": 200},
                             {"index": 0, "action": "goto", "ok": True, "url": "https://www.asked.example/",
                              "title": "A", "status": 200}],
                             "final_url": "https://www.asked.example/", "final_title": "A"}))
    assert HR.unreached(rid) == {}


def test_a_native_modal_at_the_top_edge_is_no_bar():
    """A small modal pinned near the top is still a modal (everything else is inert), not a header bar."""
    body = ('<button id="b" style="position:absolute;left:400px;top:30px">Save</button><dialog id="d" '
            'style="margin:0;top:10px;left:300px;width:300px;height:60px">Loading…</dialog>'
            '<script>document.getElementById("d").showModal()</script>')
    assert [g["control"] for g in asyncio.run(_scan(body, "http://127.0.0.1:8123/"))] == ["button#b"]


# ── review round 6 ───────────────────────────────────────────────────────
_GO = '<button id="go" style="position:absolute;left:450px;top:280px">Go</button>'


@pytest.mark.parametrize("layer,want", [
    # r6 M1: the commonest form — a FIXED full-screen layer (its edges touch the viewport: not a bar)
    ('<div class="ld" style="position:fixed;inset:0;background:#000;pointer-events:none"></div>', True),
    ('<div class="ld" style="position:fixed;inset:0;background:#000;pointer-events:none;display:flex;'
     'align-items:center;justify-content:center"><p style="color:#fff">LOADING</p></div>', True),
    ('<div class="ld" style="position:fixed;inset:0;background:oklch(0.2 0.01 250);pointer-events:none"></div>', True),
    # r6 M2: see-through decorations people put pointer-events:none on
    ('<div class="fade" style="position:fixed;inset:0;pointer-events:none;'
     'background:linear-gradient(transparent,#fff)"></div>', False),
    ('<div class="vig" style="position:fixed;inset:0;pointer-events:none;'
     'background:radial-gradient(transparent 60%,rgba(0,0,0,.7))"></div>', False),
    ('<div class="scan" style="position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient('
     'rgba(0,0,0,.15) 0 1px,transparent 1px 3px)"></div>', False),
    ('<div class="ld" style="position:fixed;inset:0;pointer-events:none;'
     'background:oklch(0.2 0.01 250 / 0.4)"></div>', False),
])
def test_round6_click_through_layers(layer, want):
    got = asyncio.run(_scan(_GO + layer, "http://127.0.0.1:8123/"))
    assert bool([g for g in got if g.get("hidden") and g["control"] == "button#go"]) is want, got


@pytest.mark.parametrize("cause,want", [
    ("Error: Page.goto: net::ERR_PROXY_CONNECTION_FAILED at https://h.example/", "nothing"),
    ("Error: Page.goto: net::ERR_HTTP_RESPONSE_CODE_FAILURE at https://h.example/", "reached"),
    ("Error: Page.goto: net::ERR_CONNECTION_REFUSED at https://h.example/", "missed"),
    # the runner killed mid-goto: the connection error is ours, not the site's
    ("Error: Page.goto: net::ERR_CONNECTION_CLOSED at https://h.example/ (runner exit 137)", "nothing"),
])
def test_round6_browser_causes(rid, cause, want):
    HR.note_browser_reach("https://h.example/", False, cause)
    rec = HR._LEDGER.get(rid, {}).get("hosts", {}).get("h.example")
    assert ("nothing" if rec is None else ("reached" if rec[0] else "missed")) == want


# ── review round 7 ───────────────────────────────────────────────────────
_SVG = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='1' height='1' fill='%23000'/%3E%3C/svg%3E\")"


@pytest.mark.parametrize("layer,want", [
    (f'background-image:{_SVG}', False),                                       # r7 M1: a grain texture
    (f'background:{_SVG} no-repeat right bottom / 80px 80px', False),          # a corner watermark
    ('background:linear-gradient(#000,#000) no-repeat 0 0 / 50px 50px', False),  # a corner patch
    ('background:radial-gradient(circle,transparent,rgba(0,0,0,.5)),linear-gradient(#111,#000)', True),  # r7 m3
    ('background:oklch(0.2 0.1 none / none)', False),                           # r7 n7
])
def test_round7_what_counts_as_solid(layer, want):
    body = _GO + f'<div class="ld" style="position:fixed;inset:0;pointer-events:none;{layer}"></div>'
    got = asyncio.run(_scan(body, "http://127.0.0.1:8123/"))
    assert bool([g for g in got if g.get("hidden")]) is want, got


def test_an_ancestor_hit_between_a_wrapped_links_boxes_is_not_a_cover():
    """The parent answers at the centre of a link wrapped over lines (the point is between its line boxes, or on
    the parent's own text): no layer painted over it."""
    body = ('<p style="width:120px;font:16px monospace;margin:0;padding:0;line-height:30px">x <a href="#" id="l">'
            'aaaaaaaaaaa bbbbbbbbb</a> y</p>')

    async def run():
        async with pw.async_playwright() as p:
            b = await p.chromium.launch()
            pg = await b.new_page(viewport={"width": 1000, "height": 600})
            await pg.route("**/*", lambda r: r.fulfill(body=f"<html><body style='margin:0'>{body}</body></html>",
                                                       content_type="text/html; charset=utf-8"))
            await pg.goto("http://127.0.0.1:8123/")
            hit = await pg.evaluate("() => { const r = document.getElementById('l').getBoundingClientRect();"
                                    " return document.elementFromPoint(r.left + r.width/2, r.top + r.height/2).tagName }")
            got = await R._covered_controls(pg)
            await b.close()
            return hit, got
    hit, got = asyncio.run(run())
    assert hit == "P" and got == []           # the parent IS hit — and that alone is not reported
