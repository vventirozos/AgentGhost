"""§4RN (2026-10-25): the web turn clock across a phone app switch — the REAL page.

Operator: "in the web ui, while on mobile, the request timer resets when i
change focus or open another app". iOS suspends the page and kills the reply
fetch (`TypeError: Load failed`); the teardown stopped the clock. A same-page
reattach then kept the bubble and showed NO clock; a reload/discard rebuilt the
bubble and restarted it at 0:00.

The whole web client boots in Chromium with `fetch` replaced by a double whose
reply streams the test can feed, end or KILL the way iOS does. The node tests
in test_interface_silent_resume.py execute the helpers; these drive the call
sites inside `sendMessage` and the two recovery paths, which no node harness
can boot."""
from pathlib import Path

import pytest

from tests.test_interface_header_simplify import _real_playwright  # noqa: F401  (harness)

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"

# `__turnState`: what /api/chat/task/<id>/state answers ('exists' = the proxy
# still holds the stream; 'gone' = it lost it while the agent still runs).
_FETCH_DOUBLE = r"""
(() => {
  if (window !== window.top) return;      // the page's sandboxed render frames
  const real = window.fetch.bind(window);
  window.__streams = [];
  // a conversation this tab shows (production always has one by the first turn)
  if (!sessionStorage.getItem('ghost_session_id')) sessionStorage.setItem('ghost_session_id', 'sess-1');
  const enc = (s) => new TextEncoder().encode(s);
  const json = (o) => new Response(JSON.stringify(o), {headers: {'Content-Type': 'application/json'}});
  const handleSid = () => {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k.startsWith('ghost_inflight_turn:')) return JSON.parse(localStorage.getItem(k)).sessionId;
    }
    return null;
  };
  window.fetch = async (input, init) => {
    const u = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (!u.pathname.startsWith('/api/')) return real(input, init);
    const flag = (k) => sessionStorage.getItem(k) === '1';
    if (u.pathname.startsWith('/api/chat/resume/') && flag('__resume404'))
      return new Response('', {status: 404});
    if (u.pathname.startsWith('/api/sessions/') && flag('__sessions404'))
      return new Response('', {status: 404});
    if (u.pathname === '/api/chat' || u.pathname.startsWith('/api/chat/resume/')) {
      let ctl;
      const body = new ReadableStream({ start(c) { ctl = c; } });
      window.__streams.push({ path: u.pathname,
        kill() { ctl.error(new TypeError('Load failed')); },
        send(t) { ctl.enqueue(enc('data: ' + JSON.stringify({choices: [{delta: {content: t}}]}) + '\n\n')); },
        end() { ctl.enqueue(enc('data: ' + JSON.stringify({choices: [{delta: {content: 'Done.'}}]}) + '\n\n'));
                ctl.enqueue(enc('data: [DONE]\n\n')); ctl.close(); } });
      return new Response(body, {headers: {'Content-Type': 'text/event-stream', 'X-Task-ID': 'task-1'}});
    }
    if (/^\/api\/chat\/task\/.+\/state$/.test(u.pathname))
      return json(sessionStorage.getItem('__turnState') === 'gone' ? {exists: false} : {exists: true});
    if (u.pathname === '/api/turns' && sessionStorage.getItem('__turnState') === 'gone')
      return json({turns: [{session_id: handleSid(), running: true}]});
    return json({});
  };
})();
"""


def _serve(route):
    path = route.request.url.split("?", 1)[0].split("ghost.test", 1)[1]
    if path == "/":
        return route.fulfill(path=str(_STATIC / "index.html"))
    f = _STATIC / path.removeprefix("/static/")
    if path.startswith("/static/") and f.is_file():
        return route.fulfill(path=str(f), content_type="application/javascript" if f.suffix == ".js" else None)
    return route.abort()


@pytest.fixture(scope="module")
def chromium():
    try:
        sync_playwright = _real_playwright().sync_playwright
    except ImportError:               # pragma: no cover
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(timeout=120_000)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                pytest.skip(f"Chromium not installed: {str(e)[:80]}")
            raise
        yield b
        b.close()


@pytest.fixture
def page(chromium):
    ctx = chromium.new_context()
    ctx.route("http://ghost.test/**", _serve)
    ctx.route_web_socket("**/*", lambda ws: None)
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.add_init_script(_FETCH_DOUBLE)
    yield pg
    ctx.close()
    assert not errors, errors


def _boot(pg, clock=True):
    if clock:
        pg.clock.install()
    pg.goto("http://ghost.test/")
    pg.wait_for_function("document.getElementById('chat-input') !== null")


def _send(pg, text, n):
    pg.fill("#chat-input", text)
    pg.click("#send-btn")
    pg.wait_for_function(f"window.__streams.length === {n}")


def _clock(pg):
    """(clock text, shown) of the turn status line."""
    return tuple(pg.evaluate("[document.getElementById('turn-status-clock').textContent,"
                             " !document.getElementById('turn-status').classList.contains('hidden')]"))


def _secs(text):
    m, s = text.split(":")
    return int(m) * 60 + int(s)


def test_a_killed_stream_reattaches_and_the_clock_keeps_counting(page):
    """Same page, nothing streamed yet: the teardown removed the empty bubble,
    the reattach rebuilds it — old world: the clock restarted at 0:00."""
    _boot(page)
    _send(page, "hello", 1)
    page.clock.fast_forward(10_000)
    assert _clock(page) == ("0:10", True)
    page.evaluate("window.__streams[0].kill()")      # iOS suspended the page
    page.wait_for_timeout(50)
    page.clock.fast_forward(1_500)                        # the 1 s network-resume retry
    page.wait_for_function("window.__streams.length === 2")
    assert page.evaluate("window.__streams[1].path") == "/api/chat/resume/task-1"
    text, shown = _clock(page)
    assert shown and _secs(text) >= 11, (text, shown)
    page.clock.fast_forward(3_000)
    assert _secs(_clock(page)[0]) >= 14


def test_a_reload_mid_turn_continues_the_clock_from_the_send(page):
    """iOS discarded the page: the boot adopts the in-flight handle and the
    rebuilt bubble counts from the send (old world: 0:00)."""
    _boot(page, clock=False)
    _send(page, "hello", 1)
    page.wait_for_timeout(2_300)
    page.reload()
    page.wait_for_function("window.__streams.length === 1 && window.__streams[0].path.includes('/resume/')")
    text, shown = _clock(page)
    assert shown and _secs(text) >= 2, (text, shown)


def test_a_reload_after_the_proxy_lost_the_stream_counts_from_the_send(page):
    """The reconcile path: no replay buffer, the agent still runs — the
    placeholder bubble's clock counts from the handle's send time."""
    _boot(page, clock=False)
    _send(page, "hello", 1)
    page.wait_for_timeout(2_300)
    page.evaluate("sessionStorage.setItem('__turnState', 'gone')")
    page.reload()
    page.wait_for_function("!document.getElementById('turn-status').classList.contains('hidden')")
    text, _ = _clock(page)
    assert _secs(text) >= 2, text


def test_the_next_turn_starts_its_own_clock_at_zero(page):
    _boot(page)
    _send(page, "first", 1)
    page.clock.fast_forward(10_000)
    page.evaluate("window.__streams[0].end()")
    page.wait_for_function("document.getElementById('turn-status').classList.contains('hidden')")
    page.clock.fast_forward(20_000)
    sent_after = page.evaluate("Date.now()")
    _send(page, "second", 2)
    text, shown = _clock(page)
    assert shown and _secs(text) <= 1, text            # not 0:30 from the first turn
    handle = page.evaluate("JSON.parse(localStorage.getItem('ghost_inflight_turn:' + sessionStorage.getItem('ghost_tab_id')))")
    assert sent_after <= handle["startedAt"] <= page.evaluate("Date.now()")


def _agent_bubbles(pg):
    return pg.evaluate("document.querySelectorAll('.message.agent').length")


def _clock_follows(pg):
    """Text of the reply bubble the clock line sits under."""
    return pg.evaluate("(document.getElementById('turn-status').previousElementSibling || {}).textContent || ''")


def test_a_partial_reply_keeps_its_bubble_and_gets_its_clock_back(page):
    """Content had streamed before iOS killed the fetch: the SAME bubble
    survives (no `thinking` class); old world — no clock at all."""
    _boot(page)
    _send(page, "hello", 1)
    page.evaluate("window.__streams[0].send('Part one of the answer. ')")
    page.wait_for_function("document.querySelector('.message.agent').textContent.includes('Part one')")
    page.clock.fast_forward(10_000)
    page.evaluate("window.__streams[0].kill()")
    page.wait_for_timeout(50)
    page.clock.fast_forward(1_500)
    page.wait_for_function("window.__streams.length === 2")
    text, shown = _clock(page)
    assert shown and _secs(text) >= 11, (text, shown)
    assert _agent_bubbles(page) == 1


def test_a_reattach_that_cannot_recover_stops_the_clock(page):
    """Partial reply, kill, the proxy lost the stream (resume 404), the agent
    still runs → placeholder clock; then the turn ends unrecoverable. The
    reviewer's reproduction: the clock kept counting under 'could not be
    recovered'."""
    _boot(page)
    _send(page, "hello", 1)
    page.evaluate("window.__streams[0].send('Part one. ')")
    page.wait_for_function("document.querySelector('.message.agent').textContent.includes('Part one')")
    page.evaluate("sessionStorage.setItem('__resume404', '1'); sessionStorage.setItem('__turnState', 'gone')")
    page.evaluate("window.__streams[0].kill()")
    page.wait_for_timeout(50)
    page.clock.fast_forward(1_500)                    # resume → 404 → reconcile: still running
    page.wait_for_function("!document.getElementById('turn-status').classList.contains('hidden')")
    page.evaluate("sessionStorage.setItem('__turnState', 'done'); sessionStorage.setItem('__sessions404', '1')")
    page.clock.fast_forward(5_500)                    # the next poll: not running, session 404
    page.wait_for_function("document.body.textContent.includes('could not be recovered')")
    assert not _clock(page)[1]
    page.clock.fast_forward(60_000)
    assert not _clock(page)[1]


def test_adopting_another_tabs_turn_never_clocks_a_finished_reply(page):
    """This tab finished a turn; a dead tab's handle for the same session is
    adopted on a visibility change. Old world: the clock ran under 'Done.'."""
    _boot(page)
    _send(page, "first", 1)
    page.evaluate("window.__streams[0].end()")
    page.wait_for_function("document.getElementById('turn-status').classList.contains('hidden')")
    page.clock.fast_forward(60_000)
    page.evaluate("""() => {
        sessionStorage.setItem('__turnState', 'gone');
        const now = Date.now();
        localStorage.setItem('ghost_inflight_turn:dead-tab', JSON.stringify({
            taskId: 'task-9', sessionId: sessionStorage.getItem('ghost_session_id'), tabId: 'dead-tab',
            ts: now - 20000, beat: now - 20000, startedAt: now - 20000}));
        document.dispatchEvent(new Event('visibilitychange'));
    }""")
    page.wait_for_function("!document.getElementById('turn-status').classList.contains('hidden')")
    text, _ = _clock(page)
    assert 20 <= _secs(text) <= 22, text
    assert "Done." not in _clock_follows(page)
    assert _agent_bubbles(page) == 2
