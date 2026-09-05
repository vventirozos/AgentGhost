"""Header simplification + timestamp overlap (2026-09-05, operator):

    "the timer on the first request got behind the 'hello' input"
    "the load / save session buttons should go in the sessions screen,
     next to +"
    "the show log button should be removed and instead, when i click or
     tap the status indicator the log window should open"
    "change the status indicator to fit the button theme"
    "make it so everything fits in a single line in mobile (safari on
     iphone)"
    "make the sorting of the icon make sense"

Layout claims are RENDERED, not read: the real index.html (scripts
stripped) and the real style.css are loaded into Playwright's WebKit —
the engine behind iPhone Safari, where emulated-Chromium green has been
wrong for viewport code before — at three iPhone widths plus landscape
and a desktop width, and bounding boxes are measured. The wiring claims
(which button opens what) are text pins over the modules, executed where
a function can be lifted out.
"""

import re
from pathlib import Path

import pytest

from tests.helpers import strip_js_comments

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


def _raw(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _js(name: str) -> str:
    return strip_js_comments(_raw(name))


# ═══════════════════════════════════════════════════════════════════════
#  Wiring (text pins)
# ═══════════════════════════════════════════════════════════════════════

class TestWiring:

    def test_the_terminal_button_is_gone_and_the_chip_toggles_the_console(self):
        html = _raw("index.html")
        assert 'id="logs-btn"' not in html
        chip = re.search(r'<button id="status-indicator"([^>]*)>', html)
        assert chip, "the chip must be a real <button>"
        attrs = chip.group(1)
        assert 'class="icon-btn"' in attrs, "the chip wears the button theme"
        assert 'data-title="Live log"' in attrs and "aria-label=" in attrs
        js = _js("app.js")
        assert "const logToggleBtn = document.getElementById('status-indicator');" in js
        assert "logToggleBtn.addEventListener('click'" in js
        assert "logsBtn" not in js, "a stale reference to the removed button"
        # The bridge the palette uses must click the chip too.
        assert "toggleLogConsole: () => { if (logToggleBtn) logToggleBtn.click(); }" in js
        # Open-state highlight follows the chip.
        assert "#status-indicator.active {" in _raw("style.css")
        assert "#logs-btn" not in _raw("style.css")

    def test_workspace_load_and_save_live_in_the_rail_beside_new_chat(self):
        html = _raw("index.html")
        rail = html[html.index('id="rail-header"'):html.index('id="session-search"')]
        header = html[html.index("<header>"):html.index("</header>")]
        for el_id in ("workspace-load-btn", "workspace-save-btn", "workspace-upload-input", "new-chat-btn"):
            assert f'id="{el_id}"' in rail, f"#{el_id} is not in the rail header"
            assert f'id="{el_id}"' not in header, f"#{el_id} is still in the page header"
        assert 'id="workspace-btn"' not in html, "the old flip-button is back"
        # Order inside the actions cluster: load, save, then "+" at the end
        # where it always was.
        acts = rail[rail.index('id="rail-actions"'):]
        assert acts.index("workspace-load-btn") < acts.index("workspace-save-btn") < acts.index("new-chat-btn")

    def test_two_buttons_two_handlers(self):
        js = _js("app.js")
        assert "workspaceLoadBtn.addEventListener('click'" in js
        assert "workspaceSaveBtn.addEventListener('click'" in js
        assert "workspaceBtn.addEventListener" not in js
        # Load into an open chat asks first; save with nothing to save is inert.
        load = js[js.index("workspaceLoadBtn.addEventListener('click'"):js.index("if (workspaceSaveBtn) {")]
        assert "chatHistory.length > 0 && !confirm(" in load
        assert "workspaceUploadInput.click();" in load
        save = js[js.index("workspaceSaveBtn.addEventListener('click'"):]
        save = save[:save.index("fetch('/api/workspace/save'")]
        assert "if (isProcessingRequest || chatHistory.length === 0) return;" in save
        # The change handler (the actual load) survived the split intact.
        assert "workspaceUploadInput.addEventListener('change'" in js
        assert "fetch('/api/workspace/load'" in js

    def test_the_palette_offers_both(self):
        js = _js("palette.js")
        assert "workspace-save-btn" in js and "workspace-load-btn" in js
        assert "'Load workspace'" in js and "'Save workspace'" in js
        assert "workspace-btn'" not in js

    def test_icon_order_navigation_status_alerts_files_appearance(self):
        html = _raw("index.html")
        header = html[html.index("<header>"):html.index("</header>")]
        ids = re.findall(r'<button id="([\w-]+)"', header)
        assert ids == ["rail-toggle", "status-indicator", "notif-btn", "upload-btn",
                       "download-btn", "face-form-btn", "fullscreen-btn"], ids

    def test_the_chip_restates_none_of_the_button_chrome(self):
        """'Fit the button theme' means INHERIT it: the chip's own rule may
        size and letter itself but must not set background/border/blur, or
        the next .icon-btn tweak leaves the chip behind again."""
        css = _raw("style.css")
        m = re.search(r"\n#status-indicator \{(.*?)\n\}", css, re.S)
        assert m
        body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
        for prop in ("background", "border:", "border-color", "backdrop-filter", "box-shadow"):
            assert prop not in body, f"{prop} restated on #status-indicator"


# ═══════════════════════════════════════════════════════════════════════
#  Rendered layout (WebKit)
# ═══════════════════════════════════════════════════════════════════════

_INDEX_NO_SCRIPTS = re.sub(r"<script\b.*?</script>", "", _raw("index.html"), flags=re.S)
_INDEX_NO_SCRIPTS = re.sub(r"<link[^>]+fonts\.g[^>]*>", "", _INDEX_NO_SCRIPTS)


def _real_playwright():
    """Import the REAL playwright even when a sibling test module in this
    xdist worker has stubbed it. Six `tests/test_browser_*.py` files put a
    fake `playwright` / `playwright.async_api` into `sys.modules` (some via
    setdefault, never removed); under `--dist loadfile` this file shared a
    worker with them and every rendered pin here SKIPPED as "playwright not
    installed" while the run read green — 20 silent skips. Evict the stubs,
    import the package from disk, then put the stubs back so the browser
    tests that still run in this worker keep the doubles they rely on."""
    import importlib
    import sys
    stubbed = {k: v for k, v in sys.modules.items()
               if k == "playwright" or k.startswith("playwright.")}
    real = all(getattr(v, "__file__", None) for v in stubbed.values())
    if stubbed and real and hasattr(sys.modules["playwright"], "__path__"):
        try:
            return importlib.import_module("playwright.sync_api")
        except ImportError:
            pass
    for k in stubbed:
        del sys.modules[k]
    try:
        mod = importlib.import_module("playwright.sync_api")
    finally:
        loaded = {k: v for k, v in sys.modules.items()
                  if k == "playwright" or k.startswith("playwright.")}
        # Restore the doubles by NAME; keep the real modules we loaded under
        # names the stubs never claimed so this module's references stay live.
        for k, v in stubbed.items():
            sys.modules[k] = v
        for k, v in loaded.items():
            sys.modules.setdefault(k, v)
    return mod


@pytest.fixture(scope="module")
def playwright_browser():
    try:
        sync_playwright = _real_playwright().sync_playwright
    except ImportError:               # pragma: no cover
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        # ⚠ Skip ONLY when the engine is genuinely absent. Any other launch
        # failure must FAIL: under the full 8-worker run these 20 rendered
        # pins once reported as 20 silent skips while the summary read
        # green — the "harness that cannot run reports success" trap.
        try:
            browser = p.webkit.launch(timeout=120_000)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                pytest.skip(f"WebKit not installed: {str(e)[:80]}")
            raise
        yield p, browser
        browser.close()


def _serve(page):
    """Route the page's requests to the real static files; no JS runs."""
    def handler(route):
        url = route.request.url
        path = url.split("//", 1)[1].split("/", 1)[1].split("?", 1)[0] if "/" in url.split("//", 1)[1] else ""
        if path == "" or path.startswith("?"):
            route.fulfill(status=200, content_type="text/html", body=_INDEX_NO_SCRIPTS)
        elif path.startswith("static/"):
            f = _STATIC / path[len("static/"):]
            if f.exists():
                ctype = "text/css" if f.suffix == ".css" else "application/octet-stream"
                route.fulfill(status=200, content_type=ctype, body=f.read_bytes())
            else:
                route.fulfill(status=404, body="")
        else:
            route.abort()
    page.route("**/*", handler)


def _open(playwright_browser, **ctx_kwargs):
    p, browser = playwright_browser
    ctx = browser.new_context(**ctx_kwargs)
    page = ctx.new_page()
    _serve(page)
    page.goto("http://ghost.test/", wait_until="load")
    return ctx, page


_HEADER_CONTROLS = ["rail-toggle", "status-indicator", "notif-btn", "upload-btn",
                    "download-btn", "face-form-btn", "fullscreen-btn"]


def _boxes(page, ids):
    return page.evaluate("""(ids) => ids.map(id => {
        const r = document.getElementById(id).getBoundingClientRect();
        return { id, top: r.top, bottom: r.bottom, left: r.left, right: r.right,
                 w: r.width, h: r.height };
    })""", ids)


class TestPhoneHeaderIsOneRow:

    @pytest.mark.parametrize("width,height", [(375, 667), (390, 844), (430, 932), (844, 390)])
    def test_every_control_shares_one_row_inside_the_viewport(self, playwright_browser, width, height):
        ctx, page = _open(playwright_browser, viewport={"width": width, "height": height},
                          has_touch=True, is_mobile=True, device_scale_factor=3)
        try:
            boxes = _boxes(page, _HEADER_CONTROLS)
            tops = [b["top"] for b in boxes]
            assert max(tops) - min(tops) < 2, f"controls on more than one row at {width}px: {boxes}"
            for b in boxes:
                assert b["left"] >= 0 and b["right"] <= width, f"{b['id']} off-screen at {width}px: {b}"
                assert b["w"] >= 40 and b["h"] >= 40, f"{b['id']} below the 40px touch floor: {b}"
            # Left cluster left, right cluster right, and in the pinned order.
            lefts = [b["left"] for b in boxes]
            assert lefts == sorted(lefts), "header order regressed"
            header = page.evaluate("document.querySelector('header').getBoundingClientRect().bottom")
            assert max(b["bottom"] for b in boxes) <= header + 0.5, "a control overflows the header box"
        finally:
            ctx.close()

    def test_on_a_phone_the_chip_is_a_round_light(self, playwright_browser):
        ctx, page = _open(playwright_browser, viewport={"width": 390, "height": 844},
                          has_touch=True, is_mobile=True)
        try:
            b = _boxes(page, ["status-indicator", "notif-btn"])
            chip, bell = b
            assert abs(chip["w"] - chip["h"]) < 1, f"chip is not round on a phone: {chip}"
            assert abs(chip["w"] - bell["w"]) < 1, "chip is not the same size as its neighbours"
            assert page.evaluate("getComputedStyle(document.getElementById('status-text')).display") == "none"
            assert page.evaluate("getComputedStyle(document.getElementById('connection-dot')).display") != "none"
        finally:
            ctx.close()


class TestDesktopChip:

    def test_the_chip_is_a_pill_in_the_button_chrome(self, playwright_browser):
        ctx, page = _open(playwright_browser, viewport={"width": 1280, "height": 800})
        try:
            chip, bell = _boxes(page, ["status-indicator", "notif-btn"])
            assert chip["w"] > chip["h"], "the desktop chip shows the state word"
            assert abs(chip["h"] - bell["h"]) < 1, "chip height differs from the buttons"
            styles = page.evaluate("""(() => {
                const c = getComputedStyle(document.getElementById('status-indicator'));
                const b = getComputedStyle(document.getElementById('notif-btn'));
                return { chip: [c.backgroundColor, c.borderTopColor, c.borderTopWidth],
                         bell: [b.backgroundColor, b.borderTopColor, b.borderTopWidth],
                         text: getComputedStyle(document.getElementById('status-text')).display };
            })()""")
            assert styles["chip"] == styles["bell"], f"chip chrome differs from the buttons: {styles}"
            assert styles["text"] != "none"
            assert page.evaluate("document.getElementById('status-indicator').tagName") == "BUTTON"
        finally:
            ctx.close()

    def test_rail_actions_sit_beside_new_chat_on_one_row(self, playwright_browser):
        ctx, page = _open(playwright_browser, viewport={"width": 1280, "height": 800})
        try:
            page.evaluate("document.getElementById('session-rail').style.marginLeft = '0'")
            boxes = _boxes(page, ["workspace-load-btn", "workspace-save-btn", "new-chat-btn"])
            tops = [b["top"] for b in boxes]
            assert max(tops) - min(tops) < 1, boxes
            assert boxes[0]["right"] <= boxes[1]["left"] <= boxes[2]["left"], boxes
            rail_right = page.evaluate("document.getElementById('session-rail').getBoundingClientRect().right")
            assert boxes[2]["right"] <= rail_right
        finally:
            ctx.close()


_BUBBLE_SCRIPT = """([role, text, time]) => {
    const log = document.getElementById('chat-log');
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.dataset.ts = '1';
    if (role === 'agent') { const p = document.createElement('p'); p.textContent = text; div.appendChild(p); }
    else div.appendChild(document.createTextNode(text));
    const row = document.createElement('div'); row.className = 'msg-actions';
    const t = document.createElement('span'); t.className = 'msg-time'; t.textContent = time; row.appendChild(t);
    const m = document.createElement('button'); m.className = 'msg-menu-btn'; m.type = 'button'; m.textContent = '⋯';
    row.appendChild(m); div.appendChild(row);
    log.appendChild(div);
    const rect = (el) => { const r = el.getBoundingClientRect(); return {l: r.left, t: r.top, r: r.right, b: r.bottom}; };
    const range = document.createRange();
    const textNode = role === 'agent' ? div.querySelector('p').firstChild : div.firstChild;
    range.selectNodeContents(textNode);
    const lines = [...range.getClientRects()].map(r => ({l: r.left, t: r.top, r: r.right, b: r.bottom}));
    return { bubble: rect(div), time: rect(t), menu: rect(m), lines, nLines: lines.length };
}"""


def _intersects(a, b):
    return a["l"] < b["r"] and b["l"] < a["r"] and a["t"] < b["b"] and b["t"] < a["b"]


class TestTimestampNeverSitsOnTheText:
    """The screenshot: 08:39 drawn behind "hello", and a reply's first line
    running under the ⋯ button. Rendered with the real CSS; the text's line
    boxes come from a Range, so 'behind' is measured, not inferred."""

    @pytest.mark.parametrize("mobile", [False, True])
    @pytest.mark.parametrize("time", ["08:39", "12:59 PM"])
    @pytest.mark.parametrize("role,text", [
        ("user", "hello"),
        ("user", "how is the weather?"),
        ("agent", "Right now in Athens it's clear and pleasant — around 25°C with light winds "
                  "(under 4 km/h) and humidity at about 48%. Nice kind of day. Want me to check "
                  "anywhere else, or anything else I can help with?"),
    ])
    def test_overlay_and_text_do_not_overlap(self, playwright_browser, mobile, time, role, text):
        kw = (dict(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
              if mobile else dict(viewport={"width": 1280, "height": 800}))
        ctx, page = _open(playwright_browser, **kw)
        try:
            out = page.evaluate(_BUBBLE_SCRIPT, [role, text, time])
            assert out["nLines"] >= 1
            for line in out["lines"]:
                assert not _intersects(line, out["time"]), f"time on the text: {out}"
                assert not _intersects(line, out["menu"]), f"menu on the text: {out}"
            # The overlay stays inside its bubble.
            for k in ("time", "menu"):
                assert out[k]["r"] <= out["bubble"]["r"] + 0.5 and out[k]["t"] >= out["bubble"]["t"] - 0.5, out
            # A one-word bubble keeps ONE line: word and overlay share it.
            if text == "hello":
                assert out["nLines"] == 1, out
        finally:
            ctx.close()

    def test_the_reservation_only_exists_with_an_overlay(self, playwright_browser):
        """Streaming/thinking bubbles have no overlay yet; they must not pay
        for one (a full-width first line while tokens arrive)."""
        ctx, page = _open(playwright_browser, viewport={"width": 1280, "height": 800})
        try:
            w = page.evaluate("""() => {
                const log = document.getElementById('chat-log');
                const mk = (withRow) => {
                    const d = document.createElement('div'); d.className = 'message agent';
                    d.appendChild(document.createTextNode('short'));
                    if (withRow) { const r = document.createElement('div'); r.className = 'msg-actions'; d.appendChild(r); }
                    log.appendChild(d); return d.getBoundingClientRect().width;
                };
                return [mk(false), mk(true)];
            }""")
            assert w[1] > w[0] + 40, f"the decorated bubble did not reserve the overlay footprint: {w}"
        finally:
            ctx.close()
