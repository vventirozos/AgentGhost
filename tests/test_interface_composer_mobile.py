"""Phone composer (2026-10-25, operator: "the 3 buttons in the input window
seem weird on mobile (iphone) like they are overlapping. change their
placement to be mobile friendly").

Measured before the change (WebKit, 375x667, effort switch on): the text
field had 175px; mic 52px at top 582, effort 28px at top 594, send 44px at
top 590 — three sizes on three baselines — and the pill sat 5px from the
screen edge because content-box sizing made it 30px wider than the footer's
content box. These are RENDERED pins (same WebKit harness as the phone
header pins): a CSS-text check cannot see a baseline.
"""
import pytest

from tests.test_interface_header_simplify import _real_playwright, _serve  # noqa: F401  (harness)

CONTROLS = ["mic-btn", "effort-wrap", "send-btn"]


@pytest.fixture(scope="module")
def browser():
    try:
        sync_playwright = _real_playwright().sync_playwright
    except ImportError:               # pragma: no cover
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            b = p.webkit.launch(timeout=120_000)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                pytest.skip(f"WebKit not installed: {str(e)[:80]}")
            raise
        yield b
        b.close()


def _open(browser, width, height):
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              has_touch=width < 600, is_mobile=width < 600)
    page = ctx.new_page()
    _serve(page)
    page.goto("http://ghost.test/", wait_until="load")
    # production runs the 27B with GHOST_REASONING_EFFORT=1: the effort control is shown
    page.evaluate("document.getElementById('effort-select').classList.remove('hidden')")
    return ctx, page


def _box(page, el_id):
    return page.evaluate("""(id) => { const r = document.getElementById(id).getBoundingClientRect();
        return {id, left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height}; }""", el_id)


@pytest.mark.parametrize("width,height", [(375, 667), (390, 844), (430, 932)])
def test_on_a_phone_the_text_field_has_the_row_and_the_controls_share_one_below(browser, width, height):
    """Fails in the measured world: a 175px text field beside three controls of
    three sizes on three baselines."""
    ctx, page = _open(browser, width, height)
    try:
        pill = _box(page, "input-area")
        field = _box(page, "chat-input")
        boxes = [_box(page, i) for i in CONTROLS]
        # the pill sits inside the footer padding, not 5px from the screen edge
        assert pill["left"] >= 10 and pill["right"] <= width - 10, pill
        # the text field takes the whole first row …
        assert field["w"] >= pill["w"] - 40, (field, pill)
        # … and every control sits under it, on ONE row, at the same size
        assert all(b["top"] >= field["bottom"] - 1 for b in boxes), (field, boxes)
        tops = [b["top"] for b in boxes]
        assert max(tops) - min(tops) < 1, boxes
        assert all(abs(b["w"] - 40) < 1 and abs(b["h"] - 40) < 1 for b in boxes), boxes
        # mic left, effort next to it, send at the right edge of the pill
        mic, eff, send = boxes
        assert mic["left"] < eff["left"] < send["left"]
        assert send["right"] >= pill["right"] - 12, (send, pill)
        assert all(b["left"] >= 0 and b["right"] <= width for b in boxes)
    finally:
        ctx.close()


def test_without_the_effort_switch_the_row_keeps_its_shape(browser):
    """The effort control hides with the switch (display:none); send must stay
    at the right and the mic at the left — margin-left:auto, not a spacer."""
    ctx, page = _open(browser, 390, 844)
    try:
        page.evaluate("document.getElementById('effort-select').classList.add('hidden')")
        assert page.evaluate("getComputedStyle(document.getElementById('effort-wrap')).display") == "none"
        pill, mic, send = _box(page, "input-area"), _box(page, "mic-btn"), _box(page, "send-btn")
        assert abs(mic["top"] - send["top"]) < 1
        assert mic["left"] <= pill["left"] + 12 and send["right"] >= pill["right"] - 12, (pill, mic, send)
    finally:
        ctx.close()


def test_the_desktop_composer_is_still_one_row_at_its_designed_width(browser):
    """box-sizing: border-box must not change the desktop layout except for the
    pill now being its designed 750px (it rendered 786px)."""
    ctx, page = _open(browser, 1280, 800)
    try:
        pill, field = _box(page, "input-area"), _box(page, "chat-input")
        boxes = [_box(page, i) for i in CONTROLS]
        assert abs(pill["w"] - 750) < 1, pill
        assert all(b["left"] > field["right"] for b in boxes), (field, boxes)   # beside, not below
        assert all(b["bottom"] <= pill["bottom"] and b["top"] >= pill["top"] for b in boxes)
    finally:
        ctx.close()


# ── Phone bubbles: the time/menu overlay leaves the text (2026-10-25,
#    operator: "the way it wraps text around the timer, the word placement
#    and line wrapping etc"). Before: an 86px first-line reservation in a
#    ~240px text column left two or three words on the first line. ─────────

_BUBBLES = """() => {
  const log = document.getElementById('chat-log'); log.innerHTML = '';
  const actions = (d, time) => { const row = document.createElement('div'); row.className = 'msg-actions';
    const t = document.createElement('span'); t.className = 'msg-time'; t.textContent = time; row.appendChild(t);
    const m = document.createElement('button'); m.className = 'msg-menu-btn'; m.type = 'button'; m.textContent = '⋯';
    row.appendChild(m); d.appendChild(row); return {t, m}; };
  const fb = (d) => { const r = document.createElement('div'); r.className = 'msg-feedback';
    for (const s of ['positive', 'negative']) { const b = document.createElement('button'); b.className = 'msg-fb-btn'; b.type = 'button'; b.textContent = '+'; r.appendChild(b); }
    d.appendChild(r); return r; };
  const R = el => { const r = el.getBoundingClientRect(); return {l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width}; };
  const box = r => ({l: r.left, t: r.top, r: r.right, b: r.bottom, w: r.width});
  const lines = el => { const rg = document.createRange(); rg.selectNodeContents(el); return [...rg.getClientRects()].map(box); };
  const u = document.createElement('div'); u.className = 'message user';
  u.appendChild(document.createTextNode('what is the latest postgres version and where do i download it?')); log.appendChild(u); const ua = actions(u, '12:59 PM');
  const a = document.createElement('div'); a.className = 'message agent'; const p = document.createElement('p');
  p.textContent = 'Right now in Athens it is clear and pleasant, around 25 degrees with light winds and humidity at about 48 percent. Want me to check anywhere else?';
  a.appendChild(p); log.appendChild(a); const afb = fb(a); const aa = actions(a, '12:59 PM');
  return { user: {lines: lines(u.firstChild), time: R(ua.t), menu: R(ua.m), bubble: R(u)},
           agent: {lines: lines(p.firstChild), time: R(aa.t), menu: R(aa.m), fb: R(afb), bubble: R(a)},
           reservation: getComputedStyle(a, '::before').content };
}"""


@pytest.mark.parametrize("width,height", [(375, 667), (390, 844), (430, 932)])
def test_on_a_phone_the_time_sits_under_the_text_and_the_first_line_is_whole(browser, width, height):
    """Fails in the measured world: the first text line stopped 86px short of
    the others and the time + ⋯ hung over the corner."""
    ctx, page = _open(browser, width, height)
    try:
        out = page.evaluate(_BUBBLES)
        assert out["reservation"] in ("none", "normal"), out["reservation"]
        for role in ("user", "agent"):
            b = out[role]
            assert len(b["lines"]) >= 2, b
            # time and menu are BELOW the last text line, inside the bubble
            last_bottom = max(ln["b"] for ln in b["lines"])
            for k in ("time", "menu"):
                assert b[k]["t"] >= last_bottom - 0.5, (k, b)
                assert b[k]["r"] <= b["bubble"]["r"] + 0.5 and b[k]["b"] <= b["bubble"]["b"] + 0.5, (k, b)
            assert b["menu"]["r"] > b["time"]["r"]            # time, then ⋯, at the right edge
        # agent bubble: thumbs and the time share ONE footer row (one row added, not two)
        ag = out["agent"]
        fb_mid = (ag["fb"]["t"] + ag["fb"]["b"]) / 2
        assert ag["menu"]["t"] <= fb_mid <= ag["menu"]["b"], ag
        assert ag["fb"]["r"] <= ag["time"]["l"], ag
    finally:
        ctx.close()


# ── Token counter beside the effort icon (2026-10-27, operator: "discreet,
#    mobile (iphone safari) compatible, don't clutter the ui ... next to the
#    effort button, under the input text"). Widest realistic text. ─────────

_WIDEST = "↑~128k ↓12.3k"


def _show_counter(page, text=_WIDEST):
    page.evaluate("""(t) => { const e = document.getElementById('token-count');
        e.textContent = t; e.classList.remove('hidden'); }""", text)
    page.wait_for_timeout(600)          # #input-area has `transition: all 0.4s` (anything would animate)


@pytest.mark.parametrize("width,height", [(320, 568), (360, 740), (375, 667), (390, 844), (430, 932)])
def test_on_a_phone_the_counter_is_a_small_line_under_the_buttons_on_the_right(browser, width, height):
    """Operator 10-27: "on the right side, under the buttons … similar in mobile"."""
    ctx, page = _open(browser, width, height)
    try:
        before = _box(page, "input-area")
        b0 = [_box(page, i) for i in CONTROLS]
        _show_counter(page)
        pill, field, cnt = _box(page, "input-area"), _box(page, "chat-input"), _text_box(page, "token-count")
        b1 = [_box(page, i) for i in CONTROLS]
        mic, eff, send = b1
        _assert_under_the_buttons_on_the_right(before, pill, b0, b1, cnt, send)
        assert all(abs(b["w"] - 40) < 1 for b in b1)                    # controls keep their size
        assert all(b["top"] >= field["bottom"] - 1 for b in b1)         # the phone layout is intact
    finally:
        ctx.close()


def _text_box(page, el_id):
    """The rendered TEXT's box (the element spans the row; its text is right-aligned)."""
    return page.evaluate("""(id) => { const rg = document.createRange();
        rg.selectNodeContents(document.getElementById(id)); const r = rg.getBoundingClientRect();
        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height}; }""", el_id)


def _assert_under_the_buttons_on_the_right(before, pill, b0, b1, cnt, send):
    # nothing moves when a count appears (operator 10-27: the resize "looks
    # weird") — the pill always holds the line
    assert abs(pill["h"] - before["h"]) < 0.5 and abs(pill["top"] - before["top"]) < 0.5, (before, pill)
    assert all(abs(a["left"] - b["left"]) < 0.5 and abs(a["top"] - b["top"]) < 0.5
               for a, b in zip(b0, b1)), (b0, b1)
    assert cnt["top"] >= max(b["bottom"] for b in b1) - 0.5, (b1, cnt)   # under, never over Send
    assert send["right"] - 12 <= cnt["right"] <= send["right"] + 0.5, (send, cnt)   # ends under Send
    assert cnt["h"] < 16, cnt                                             # one small line
    assert cnt["bottom"] <= pill["bottom"] and cnt["left"] >= pill["left"]


def test_the_counter_is_quiet_and_not_a_control(browser):
    ctx, page = _open(browser, 390, 844)
    try:
        _show_counter(page)
        st = page.evaluate("""(() => { const s = getComputedStyle(document.getElementById('token-count'));
            return {fs: parseFloat(s.fontSize), us: s.webkitUserSelect || s.userSelect, color: s.color, ws: s.whiteSpace}; })()""")
        assert st["fs"] <= 11.5 and st["us"] == "none" and st["ws"] == "nowrap", st
        assert st["color"] == page.evaluate(
            "getComputedStyle(document.getElementById('effort-wrap')).color"), st   # the effort icon's grey
        page.evaluate("document.getElementById('token-count').classList.add('hidden')")
        assert page.evaluate("getComputedStyle(document.getElementById('token-count')).visibility") == "hidden"
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(1280, 800), (820, 1180)])
def test_on_wider_screens_it_is_a_small_line_under_the_buttons_on_the_right(browser, width, height):
    """Operator 10-27: "in desktop, put the token information on the right
    side, under the buttons" — the text field's width untouched."""
    ctx, page = _open(browser, width, height)
    try:
        before, field0 = _box(page, "input-area"), _box(page, "chat-input")
        b0 = [_box(page, i) for i in CONTROLS]
        _show_counter(page)
        pill, field, cnt = _box(page, "input-area"), _box(page, "chat-input"), _text_box(page, "token-count")
        b1 = [_box(page, i) for i in CONTROLS]
        assert abs(field["w"] - field0["w"]) < 1 and abs(pill["w"] - before["w"]) < 1, (field0, field)
        _assert_under_the_buttons_on_the_right(before, pill, b0, b1, cnt, b1[2])
        # hoverable: its title (the cached part) is the desktop detail (review r2)
        hit = page.evaluate("""(([x, y]) => document.elementFromPoint(x, y).id)""",
                            [cnt["right"] - 4, (cnt["top"] + cnt["bottom"]) / 2])
        assert hit == "token-count", hit
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(1280, 800), (390, 844)])
def test_an_empty_counter_still_holds_its_line(browser, width, height):
    """Hidden = not drawn, still laid out: the pill is the same size empty
    or with a count, and the empty line is a real ~13px row."""
    ctx, page = _open(browser, width, height)
    try:
        e = _box(page, "token-count")
        assert page.evaluate("getComputedStyle(document.getElementById('token-count')).visibility") == "hidden"
        assert 10 < e["h"] < 16, e
        if width > 480:
            assert abs(_box(page, "input-area")["w"] - 750) < 1
    finally:
        ctx.close()
