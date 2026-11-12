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
        mids = [(b["top"] + b["bottom"]) / 2 for b in boxes]
        assert max(mids) - min(mids) < 1, boxes
        # §4UJ toolbar (2026-11-12): 36px icons, 44px Send
        mic_, eff_, send_ = boxes
        assert abs(mic_["w"] - 36) < 1 and abs(eff_["w"] - 36) < 1 and abs(send_["w"] - 44) < 1, boxes
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
        assert abs((mic["top"] + mic["bottom"]) - (send["top"] + send["bottom"])) < 2
        assert mic["left"] <= pill["left"] + 12 and send["right"] >= pill["right"] - 12, (pill, mic, send)
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(1280, 800), (1280, 720), (820, 1180)])
def test_the_desktop_composer_is_the_toolbar_at_its_designed_width(browser, width, height):
    """§4UJ (2026-11-12, operator: the mic, speaker and effort buttons "take too
    much estate … smaller and nicely aligned with send … use 2 rows"). Before:
    52px mic and speaker circles beside the text, ~200px of the row. Now the
    text field has the first row and the controls are a toolbar under it:
    32px borderless icons, a 40px Send at the right edge. 1280x720 is the
    uConsole, whose old 36px shrink must not win over the toolbar."""
    ctx, page = _open(browser, width, height)
    try:
        pill, field = _box(page, "input-area"), _box(page, "chat-input")
        mic, tts, eff, send = (_box(page, i) for i in ("mic-btn", "tts-toggle-btn", "effort-wrap", "send-btn"))
        assert abs(pill["w"] - 750) < 1, pill
        assert field["w"] >= pill["w"] - 40, (field, pill)                      # the whole first row
        assert all(b["top"] >= field["bottom"] - 1 for b in (mic, tts, eff, send))   # below, not beside
        assert all(abs(b["w"] - 32) < 1 and abs(b["h"] - 32) < 1 for b in (mic, tts, eff)), (mic, tts, eff)
        assert abs(send["w"] - 40) < 1 and send["right"] >= pill["right"] - 12, (send, pill)
        assert mic["left"] < tts["left"] < eff["left"] < send["left"]
        assert abs((mic["left"] + 7.5) - field["left"]) < 3, (mic, field)       # the mic glyph under the text
        assert all(b["bottom"] <= pill["bottom"] and b["top"] >= pill["top"] for b in (mic, send))
        st = page.evaluate("""(() => { const s = getComputedStyle(document.getElementById('mic-btn'));
            return {bg: s.backgroundColor, bw: s.borderTopWidth}; })()""")
        assert st["bw"] == "0px" and st["bg"] in ("rgba(0, 0, 0, 0)", "transparent"), st   # borderless
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(1280, 800), (390, 844)])
def test_the_borderless_icons_still_show_recording_and_spoken_replies(browser, width, height):
    """The toolbar rules carry #input-area and beat the single-id state rules;
    the states are restated so a recording mic is still a red circle and an
    active speaker still green."""
    ctx, page = _open(browser, width, height)
    try:
        page.evaluate("""document.getElementById('mic-btn').classList.add('recording');
            document.getElementById('tts-toggle-btn').classList.add('active')""")
        page.wait_for_timeout(400)      # the icons' 0.15s colour transition
        st = page.evaluate("""(() => ({mic: getComputedStyle(document.getElementById('mic-btn')).backgroundColor,
            tts: getComputedStyle(document.getElementById('tts-toggle-btn')).color}))()""")
        assert st["mic"] == "rgb(194, 39, 69)", st
        assert st["tts"] == "rgb(73, 176, 124)", st
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(320, 568), (390, 844)])
def test_on_a_phone_the_icons_have_a_44px_finger_target(browser, width, height):
    ctx, page = _open(browser, width, height)
    try:
        for i in ("mic-btn", "tts-toggle-btn"):
            r = _box(page, i)
            # a point 3px outside the drawn 36px circle still lands on the button
            hit = page.evaluate("""([id, x, y]) => { const e = document.elementFromPoint(x, y);
                return !!(e && e.closest('#' + id)); }""", [i, r["left"] + r["w"] / 2, r["top"] - 3])
            assert hit, (i, r)
    finally:
        ctx.close()


def test_on_the_narrowest_phone_a_running_turn_keeps_send_on_the_toolbar(browser):
    """320px with the tool-stop button and the widest count: the counter is cut
    ("…"), Send does not wrap to a third row."""
    ctx, page = _open(browser, 320, 568)
    try:
        page.evaluate("document.getElementById('interrupt-btn').classList.remove('hidden')")
        _show_counter(page)
        mic, stop, send = (_box(page, i) for i in ("mic-btn", "interrupt-btn", "send-btn"))
        assert abs((mic["top"] + mic["bottom"]) - (send["top"] + send["bottom"])) < 2, (mic, send)
        assert stop["right"] <= send["left"] and send["right"] <= 320
        assert page.evaluate("document.documentElement.scrollWidth") <= 320
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 720), (820, 1180), (375, 667)])
@pytest.mark.parametrize("field_height", [None, 120])
def test_every_composer_control_shares_one_centre_line(browser, width, height, field_height):
    """Operator 2026-11-07, desktop screenshot: the effort icon floated above the
    mic and send. It was centred on the row while they sat at its bottom, so a
    grown text field moved it alone. Also 1280x720, where mic and send are 36px."""
    ctx, page = _open(browser, width, height)
    try:
        page.evaluate("document.getElementById('interrupt-btn').classList.remove('hidden')")
        if field_height:
            page.evaluate("(h) => { document.getElementById('chat-input').style.height = h + 'px'; }", field_height)
        _show_counter(page)
        boxes = [_box(page, i) for i in ("mic-btn", "tts-toggle-btn", "effort-wrap", "interrupt-btn", "send-btn")]
        centres = [(b["top"] + b["bottom"]) / 2 for b in boxes]
        assert max(centres) - min(centres) < 1, boxes
        pill = _box(page, "input-area")
        assert all(b["bottom"] <= pill["bottom"] and b["top"] >= pill["top"] for b in boxes), (pill, boxes)
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
def test_on_a_phone_the_counter_sits_in_the_toolbar_before_send(browser, width, height):
    """Operator 10-27 put it under the buttons on the right; §4UJ (2026-11-12) moved
    it into the toolbar row, its text ending just before Send."""
    ctx, page = _open(browser, width, height)
    try:
        before = _box(page, "input-area")
        b0 = [_box(page, i) for i in CONTROLS]
        _show_counter(page)
        pill, field, cnt = _box(page, "input-area"), _box(page, "chat-input"), _text_box(page, "token-count")
        b1 = [_box(page, i) for i in CONTROLS]
        mic, eff, send = b1
        _assert_in_the_toolbar_before_send(before, pill, b0, b1, cnt, send)
        assert abs(mic["w"] - 36) < 1 and abs(send["w"] - 44) < 1      # controls keep their size
        assert all(b["top"] >= field["bottom"] - 1 for b in b1)         # the phone layout is intact
    finally:
        ctx.close()


def _text_box(page, el_id):
    """The rendered TEXT's box (the element spans the row; its text is right-aligned)."""
    return page.evaluate("""(id) => { const rg = document.createRange();
        rg.selectNodeContents(document.getElementById(id)); const r = rg.getBoundingClientRect();
        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom, w: r.width, h: r.height}; }""", el_id)


def _assert_in_the_toolbar_before_send(before, pill, b0, b1, cnt, send):
    # nothing moves when a count appears (operator 10-27: the resize "looks
    # weird") — the pill always holds the counter's place
    assert abs(pill["h"] - before["h"]) < 0.5 and abs(pill["top"] - before["top"]) < 0.5, (before, pill)
    assert all(abs(a["left"] - b["left"]) < 0.5 and abs(a["top"] - b["top"]) < 0.5
               for a, b in zip(b0, b1)), (b0, b1)
    # §4UJ: on the toolbar's centre line, its text ending just before Send
    assert abs((cnt["top"] + cnt["bottom"]) / 2 - (send["top"] + send["bottom"]) / 2) < 2, (send, cnt)
    assert send["left"] - 16 <= cnt["right"] <= send["left"], (send, cnt)
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
def test_on_wider_screens_it_sits_in_the_toolbar_before_send(browser, width, height):
    """Operator 10-27: "in desktop, put the token information on the right
    side, under the buttons" — the text field's width untouched. §4UJ: now in
    the toolbar row, before Send."""
    ctx, page = _open(browser, width, height)
    try:
        before, field0 = _box(page, "input-area"), _box(page, "chat-input")
        b0 = [_box(page, i) for i in CONTROLS]
        _show_counter(page)
        pill, field, cnt = _box(page, "input-area"), _box(page, "chat-input"), _text_box(page, "token-count")
        b1 = [_box(page, i) for i in CONTROLS]
        assert abs(field["w"] - field0["w"]) < 1 and abs(pill["w"] - before["w"]) < 1, (field0, field)
        _assert_in_the_toolbar_before_send(before, pill, b0, b1, cnt, b1[2])
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


@pytest.mark.parametrize("width,height", [(844, 390), (1024, 768)])
def test_touch_screens_wider_than_a_phone_keep_the_finger_sizes(browser, width, height):
    """Review §4UJ MAJOR: a landscape iPhone or an iPad matched only the desktop
    width rules and got 32px borderless targets (52px circles before) —
    hold-to-talk on 32px. Touch screens get the phone sizes at any width."""
    ctx = browser.new_context(viewport={"width": width, "height": height}, has_touch=True, is_mobile=True)
    page = ctx.new_page()
    try:
        _serve(page)
        page.goto("http://ghost.test/", wait_until="load")
        page.evaluate("document.getElementById('effort-select').classList.remove('hidden')")
        mic, eff, send = (_box(page, i) for i in CONTROLS)
        assert abs(mic["w"] - 36) < 1 and abs(eff["w"] - 36) < 1 and abs(send["w"] - 44) < 1, (mic, eff, send)
        sel = _box(page, "effort-select")
        assert sel["w"] >= 43, sel                  # the picker's own tap area is 44px
    finally:
        ctx.close()


def test_the_tool_stop_button_keeps_its_red_hover(browser):
    ctx, page = _open(browser, 1280, 800)
    try:
        page.evaluate("document.getElementById('interrupt-btn').classList.remove('hidden')")
        page.hover("#interrupt-btn")
        page.wait_for_timeout(400)
        assert page.evaluate("getComputedStyle(document.getElementById('interrupt-btn')).color") == "rgb(194, 39, 69)"
    finally:
        ctx.close()
