"""Web token counter beside the effort icon (2026-10-27, operator: "just add
it to the webUI, make it discreet and mobile (iphone safari) compatible,
don't clutter the ui ... next to the effort button, under the input text").

The agent's `ghost.usage` frame (one, before [DONE]) drives #token-count.
These boot the REAL page (the §4RN fetch double) and feed frames through
`sendMessage`'s parser; the node test executes tokens.js itself; the layout
pins (WebKit, phone widths) live in test_interface_composer_mobile.py."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_4rn_turn_clock_browser import _FETCH_DOUBLE, _boot, _send, chromium, page  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]

# The §4RN double plus a raw-frame feeder (any JSON object as one SSE frame).
_DOUBLE = _FETCH_DOUBLE.replace(
    "        end() {",
    "        raw(o) { ctl.enqueue(enc('data: ' + JSON.stringify(o) + '\\n\\n')); },\n        end() {", 1)
assert _DOUBLE != _FETCH_DOUBLE


def _usage(rid, p=29582, c=840, cached=28202, **extra):
    return {"id": f"chatcmpl-{rid}", "choices": [],
            "ghost": {"usage": dict(prompt_tokens=p, completion_tokens=c, cached_tokens=cached, calls=1, **extra)}}


def _counter(pg):
    return pg.evaluate("""(() => { const e = document.getElementById('token-count');
        return [e.textContent, !e.classList.contains('hidden'), e.title]; })()""")


def _turn(pg, n, rid, **kw):
    _send(pg, f"q{n}", n)
    pg.evaluate(f"window.__streams[{n - 1}].send('hello')")
    pg.evaluate(f"window.__streams[{n - 1}].raw({json.dumps(_usage(rid, **kw))})")
    pg.evaluate(f"window.__streams[{n - 1}].end()")
    pg.wait_for_function("!document.getElementById('send-btn').disabled && "
                         "document.querySelectorAll('.message.agent').length >= %d" % n)


@pytest.fixture
def tpage(page):
    page.add_init_script(_DOUBLE)       # runs after the base double: the later one wins
    return page


def test_hidden_until_a_turn_reports(tpage):
    _boot(tpage, clock=False)
    text, shown, _ = _counter(tpage)
    assert (text, shown) == ("", False)


def test_a_turn_shows_its_tokens_and_the_frame_is_not_reply_text(tpage):
    _boot(tpage, clock=False)
    _turn(tpage, 1, "r1")
    text, shown, title = _counter(tpage)
    assert shown and text == "↑29.6k ↓840", text
    assert "context: 29,582 tokens (28,202 cached), generated: 840" in title, title
    reply = tpage.evaluate("[...document.querySelectorAll('.message.agent')].pop().textContent")
    assert "hello" in reply and "29582" not in reply and "usage" not in reply
    hist = tpage.evaluate("window.GhostCore.getChatHistory()")
    assert hist[-1]["role"] == "assistant" and "usage" not in json.dumps(hist[-1])


def test_each_turn_replaces_the_last(tpage):
    _boot(tpage, clock=False)
    _turn(tpage, 1, "r1")
    _turn(tpage, 2, "r2", p=30000, c=60)
    assert _counter(tpage)[0] == "↑30.0k ↓60"


def test_a_turn_without_a_frame_shows_nothing(tpage):
    """Review r1 m3 (the CLI's twin): the previous turn's ↑↓ must not stand
    for a turn that reported nothing (an error, a system answer)."""
    _boot(tpage, clock=False)
    _turn(tpage, 1, "r1")
    _send(tpage, "q2", 2)
    assert _counter(tpage)[:2] == ["", False]      # gone at the send …
    tpage.evaluate("window.__streams[1].end()")
    tpage.wait_for_function("!document.getElementById('send-btn').disabled")
    assert _counter(tpage)[:2] == ["", False]      # … and after a frameless turn


def test_unmetered_turn_is_marked_approximate(tpage):
    _boot(tpage, clock=False)
    _turn(tpage, 1, "r1", unmetered_calls=1)
    assert _counter(tpage)[0].startswith("↑~29.6k")


def test_clear_hides_it_and_a_reload_keeps_it(tpage):
    _boot(tpage, clock=False)
    _turn(tpage, 1, "r1")
    tpage.reload()
    tpage.wait_for_function("document.getElementById('chat-input') !== null")
    assert _counter(tpage)[1] is True            # an iOS reload keeps the number
    tpage.fill("#chat-input", "/clear")
    tpage.click("#send-btn")
    assert _counter(tpage)[:2] == ["", False]
    # this double runs with sessions off: the id does not change, and a
    # reload must not bring the cleared turn back (review r2)
    tpage.reload()
    tpage.wait_for_function("document.getElementById('chat-input') !== null")
    assert _counter(tpage)[:2] == ["", False]


_NODE = shutil.which("node")


@pytest.mark.skipif(_NODE is None, reason="node not installed")
def test_the_module(tmp_path):
    mod = (ROOT / "interface/static/tokens.js").as_uri()
    script = tmp_path / "t.mjs"
    script.write_text(f"""
import {{ fmtTokens, readUsage, createTokenCounter }} from '{mod}';
const out = [];
out.push([0, 999, 1000, 31234, 99949, 99999, 128900, -5, 'x'].map(fmtTokens));
out.push(readUsage(null), readUsage({{prompt_tokens: 'bad', completion_tokens: -3, unmetered_calls: 2}}));
const mem = {{}};
const storage = {{ get: k => mem[k] ?? null, set: (k, v) => {{ mem[k] = v; }} }};
const cls = new Set(['hidden']);
const el = {{ textContent: '', title: '', classList: {{ add: c => cls.add(c), remove: c => cls.delete(c),
  contains: c => cls.has(c) }}, removeAttribute(a) {{ this[a] = ''; }} }};
let sid = 's1';
mem.ghost_token_counts = '{{not json';                    // a corrupt store is an empty one
const t = createTokenCounter(el, storage, () => sid);
out.push(cls.has('hidden'));
t.note({{prompt_tokens: 100, completion_tokens: 5, cached_tokens: 500}});   // cached capped at prompt
t.note('junk');                                           // no usage: ignored
out.push(el.textContent, el.title);
t.startTurn(); out.push(el.textContent, cls.has('hidden'));   // a new turn: nothing yet
t.note({{prompt_tokens: 100, completion_tokens: 5}}); out.push(el.textContent);
sid = 's2'; t.render(); out.push(cls.has('hidden'));       // another session: nothing
sid = 's1'; t.render(); out.push(el.textContent);          // back: its own last turn
for (let i = 0; i < 40; i++) {{ sid = 'x' + i; t.note({{prompt_tokens: 1}}); }}
out.push(Object.keys(JSON.parse(mem.ghost_token_counts)).length);   // sessions capped
mem.ghost_token_counts = JSON.stringify({{x39: {{last: {{prompt: '<b>', completion: -1}}}}}});
createTokenCounter(el, storage, () => 'x39'); out.push(el.textContent);   // (a reload) an edited store renders numbers only
const t2 = createTokenCounter(null, storage, () => 'x39'); t2.note({{prompt_tokens: 1}});  // no element: no throw
// review r2: a FULL store (set reports failure, as safeStorage does) must not freeze the counter
const full = {{ get: () => JSON.stringify({{s: {{last: {{prompt: 1000, completion: 10}}, at: 1}}}}), set: () => false }};
const t3 = createTokenCounter(el, full, () => 's');
out.push(el.textContent);
t3.startTurn(); out.push(cls.has('hidden'));
t3.note({{prompt_tokens: 50000, completion_tokens: 900}}); out.push(el.textContent);
// no session id (sessions off, fresh browser): still counted
const t4 = createTokenCounter(el, {{ get: () => null, set: () => true }}, () => null);
t4.note({{prompt_tokens: 7, completion_tokens: 1}}); out.push(el.textContent);
// a hand-edited null entry never throws out of a send
const t5 = createTokenCounter(el, {{ get: () => JSON.stringify({{a: null, b: 3}}), set: () => true }}, () => 'c');
t5.startTurn(); t5.note({{prompt_tokens: 2}}); out.push(el.textContent);
console.log(JSON.stringify(out));
""")
    r = subprocess.run([_NODE, str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [
        ["0", "999", "1.0k", "31.2k", "99.9k", "100k", "129k", "0", "0"],
        None, {"prompt": 0, "completion": 0, "cached": 0, "approx": True},
        True, "↑100 ↓5", "Last reply — context: 100 tokens (100 cached), generated: 5",
        "", True, "↑100 ↓5", True, "↑100 ↓5", 30, "↑0 ↓0",
        "↑1.0k ↓10", True, "↑50.0k ↓900", "↑7 ↓1", "↑2 ↓0"]
