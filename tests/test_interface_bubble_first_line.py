"""Short sent messages wrapped (2026-11-07, operator screenshot: "hel / lo ghost.", "how's the / weather ?").
A message renders as <p>; the overlay's first-line reservation sat beside it, where a float adds nothing to
the bubble's fitted width. Measured before the fix: 155px and two lines. Rendered, because only layout shows it."""
import pytest

from tests.test_interface_composer_mobile import _open, browser  # noqa: F401  (harness)

_BUBBLE = """async ([role, html]) => {
  const log = document.getElementById('chat-log'); log.innerHTML = '';
  const d = document.createElement('div'); d.className = 'message ' + role; d.innerHTML = html;
  const row = document.createElement('div'); row.className = 'msg-actions';
  const t = document.createElement('span'); t.className = 'msg-time'; t.textContent = '12:59 PM'; row.appendChild(t);
  const m = document.createElement('button'); m.className = 'msg-menu-btn'; m.textContent = '⋯'; row.appendChild(m);
  d.appendChild(row); log.appendChild(d);
  await new Promise(r => setTimeout(r, 600));
  const first = d.firstElementChild; const rg = document.createRange(); rg.selectNodeContents(first.firstChild);
  const rects = [...rg.getClientRects()];
  const R = e => e.getBoundingClientRect();
  return { lines: new Set(rects.map(r => Math.round(r.top))).size, textRight: Math.max(...rects.map(r => r.right)),
           timeLeft: R(t).left, bubble: R(d).width };
}"""


@pytest.mark.parametrize("role", ["user", "agent"])
@pytest.mark.parametrize("text", ["hello", "hello ghost.", "how's the weather ?"])
@pytest.mark.parametrize("width,height", [(1852, 559), (1280, 800), (768, 1024), (1280, 560), (1000, 700), (844, 390)])
def test_a_short_message_stays_on_one_line_clear_of_the_time(browser, role, text, width, height):  # noqa: F811
    """1280x560: bubble padding is 14px there, and a fixed reservation put the time over the last 9px of text."""
    ctx, page = _open(browser, width, height)
    try:
        out = page.evaluate(_BUBBLE, [role, f"<p>{text}</p>"])
        assert out["lines"] == 1, out
        assert out["textRight"] + 4 <= out["timeLeft"], out
    finally:
        ctx.close()


def test_a_short_heading_gets_the_same(browser):  # noqa: F811
    ctx, page = _open(browser, 1280, 800)
    try:
        assert page.evaluate(_BUBBLE, ["agent", "<h3>Short title</h3><p>Body.</p>"])["lines"] == 1
    finally:
        ctx.close()


def test_a_long_message_still_wraps_inside_its_maximum(browser):  # noqa: F811
    ctx, page = _open(browser, 1280, 800)
    try:
        out = page.evaluate(_BUBBLE, ["user", "<p>" + "many words here " * 30 + "</p>"])
        assert out["lines"] > 2 and out["bubble"] < 640, out
    finally:
        ctx.close()


def test_a_bubble_starting_with_a_list_keeps_a_reservation(browser):  # noqa: F811
    ctx, page = _open(browser, 1280, 800)
    try:
        page.evaluate(_BUBBLE, ["agent", "<ul><li>one</li></ul>"])
        read = "getComputedStyle(document.querySelector('#chat-log .message'), '::before').content"
        assert page.evaluate(read) not in ("none", "normal")
    finally:
        ctx.close()


def test_a_long_first_heading_keeps_its_second_line_whole(browser):  # noqa: F811
    """A reservation taller than the heading's line narrowed line 2 as well (445, 430, 415)."""
    ctx, page = _open(browser, 1280, 800)
    try:
        ends = page.evaluate("""async () => {
            const log = document.getElementById('chat-log'); log.innerHTML = '';
            const d = document.createElement('div'); d.className = 'message agent';
            d.innerHTML = '<h1>' + 'word '.repeat(60) + '</h1>';
            const row = document.createElement('div'); row.className = 'msg-actions'; d.appendChild(row);
            log.appendChild(d); await new Promise(r => setTimeout(r, 600));
            const rg = document.createRange(); rg.selectNodeContents(d.firstChild.firstChild);
            const rows = {}; for (const r of rg.getClientRects()) {
                const k = Math.round(r.top); rows[k] = Math.max(rows[k] || 0, r.right); }
            return Object.keys(rows).map(Number).sort((a, b) => a - b).map(k => rows[k]);
        }""")
        assert len(ends) >= 3, ends
        assert ends[1] > ends[0] + 40, ends
    finally:
        ctx.close()
