"""The web turn status line shows what the agent is doing and nothing else
(2026-10-27, operator: "show tool calls, image generation, web / deep web
searches, generally... show what it's doing, without showing errors /
warnings / head disk operations things that don't concern the user. when
you don't have anything to put put something like 💭 thinking … just for
the web UI description under the reply box NOT the live log").

Two layers, both executed: `turn_caption.js` imported under node (titles and
contents sampled from the live log), and the REAL page in Chromium with log
lines pushed over a routed WebSocket in the agent's corridor format — the
path `noteTickerLine` reads in production."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_4nv_log_lines import _titles
from tests.test_4rn_turn_clock_browser import _FETCH_DOUBLE, _serve, chromium  # noqa: F401
from tests.test_web_token_counter import _DOUBLE

ROOT = Path(__file__).resolve().parents[1]
MOD = (ROOT / "interface/static/turn_caption.js").as_uri()
_NODE = shutil.which("node")
node = pytest.mark.skipif(_NODE is None, reason="node not installed")


def _node(expr_js: str):
    r = subprocess.run([_NODE, "--input-type=module", "-e",
                        f"import {{ turnCaption, TURN_CAPTIONS }} from '{MOD}';\n"
                        f"console.log(JSON.stringify({expr_js}));"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


SHOWN = [
    (("🧰", "tool call", "youtube_transcribe · url=https://www.youtube.com/watch?v=x"), "using a tool · youtube transcribe", "🧰"),
    (("🌐", "web search", "postgres latest version"), "searching the web · postgres latest version", "🌐"),
    (("🌐", "web search", "yandex won wave 0 in 1.4s (9 results) ‹q›"), "searching the web", "🌐"),
    (("🌐", "web search", "no winner after 6s — hedging yandex"), "searching the web", "🌐"),
    (("🧅", "darkweb search", "german forums | germany dark web"), "searching the dark web · german forums", "🧅"),
    (("🔶", "darkweb engine", "torgle: 503"), "searching the dark web", "🧅"),
    (("🌎", "browser", "navigate https://www.tradingview.com/symbols/BVL-VUAA/"), "browsing · tradingview.com", "🌎"),
    (("🎨", "image generation", "size=768x1024 | prompt (807 chars): A cascade"), "generating an image", "🎨"),
    (("📥", "subject photo", "Joseph Stalin → ref_joseph_stalin.jpg (Joseph Stalin)"), "fetching a reference photo · Joseph Stalin", "📥"),
    (("📥", "download [tor]", "https://upload.wikimedia.org/wikipedia/commons/0/08/x.jpg"), "downloading · upload.wikimedia.org", "📥"),
    (("📖", "file read", "/webos/index.html [190:210]"), "reading · index.html", "📖"),
    (("💾", "file write", "webos/index.html (34262 chars)"), "writing · index.html", "💾"),
    (("🐍", "execution task", ".ephemeral_85bae656.py"), "running code", "🐍"),
    (("🐚", "shell command", "cd /workspace && node -e \"x\""), "running a command", "🐚"),
    (("🐍", "executing skill", "Running custom tool: news_headlines"), "running a skill · news_headlines", "🐍"),
    (("🌐", "system weather", "Location: Athens, Greece"), "checking the weather · Athens, Greece", "⛅"),
    (("🔬", "vision", "describe_picture -> mu_9702.jpg"), "looking at an image", "🔬"),
    (("📚", "youtube", "resolving GZB6T8QOcFs over Tor"), "reading a YouTube video", "📚"),
    (("✅", "verifier", "CONFIRMED"), "checking the answer", "🧪"),
]


@node
@pytest.mark.parametrize("line,want,icon", SHOWN)
def test_what_the_agent_is_doing_is_shown(line, want, icon):
    icon_in, title, content = line
    got = _node(f"turnCaption({json.dumps(title)}, {json.dumps(icon_in)}, {json.dumps(content)})")
    assert got == {"text": want, "icon": icon}, got


HIDDEN = [("🔶", "tool warning"), ("🔶", "transient fail"), ("❌", "execution fail"), ("📀", "head disk"),
          ("💽", "session switch"), ("📼", "conversation"), ("🛑", "thinking loop"), ("🔄", "turn queue"),
          ("⏳", "client deadline"), ("✅", "turn outcome"), ("🧩", "memory bus"), ("📍", "memory match"),
          ("🌐", "search relevance"), ("⏩", "search wave empty"), ("🔗", "constraint check"),
          ("🧪", "verify visual"), ("🔶", "blocked page"), ("🔶", "composed url")]


@node
def test_housekeeping_errors_and_warnings_get_no_caption():
    got = _node("[" + ",".join(f"turnCaption({json.dumps(t)}, {json.dumps(i)}, 'x')" for i, t in HIDDEN) + "]")
    assert got == [None] * len(HIDDEN), list(zip(HIDDEN, got))


@node
def test_every_caption_key_is_a_title_the_agent_prints():
    keys = _node("Object.keys(TURN_CAPTIONS)")
    missing = sorted(k for k in keys if k not in _titles())
    assert keys and not missing, missing


@node
def test_a_long_detail_is_cut_and_a_missing_one_shows_none():
    got = _node("[turnCaption('web search', '🌐', 'x'.repeat(80)).text,"
                " turnCaption('browser', '🌎', 'navigate').text]")
    assert got == ["searching the web · " + "x" * 30 + "…", "browsing"]


# ── the real page: log lines over the socket ──────────────────────────────

_RID = "a8a93a27"


@pytest.fixture
def live(chromium):  # noqa: F811
    ctx = chromium.new_context()
    ctx.route("http://ghost.test/**", _serve)
    sockets = []
    ctx.route_web_socket("**/ws*", lambda ws: sockets.append(ws))
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.add_init_script(_FETCH_DOUBLE)
    # the proxy names the turn's request id up front (X-Request-ID)
    double = _DOUBLE.replace("{'Content-Type': 'text/event-stream', 'X-Task-ID': 'task-1'}",
                             "{'Content-Type': 'text/event-stream', 'X-Task-ID': 'task-1', "
                             f"'X-Request-ID': '{_RID}'}}")
    assert double != _DOUBLE
    pg.add_init_script(double)
    yield pg, sockets
    ctx.close()
    assert not errors, errors


def _caption(pg):
    return pg.evaluate("[document.getElementById('turn-status-icon').textContent,"
                       " document.getElementById('turn-status-desc').textContent]")


def _log(pg, sockets, line):
    sockets[-1].send(json.dumps({"type": "log", "content": line}))
    pg.wait_for_timeout(60)


def test_the_page_shows_steps_hides_housekeeping_and_starts_on_thinking(live):
    pg, sockets = live
    pg.goto("http://ghost.test/")
    pg.wait_for_function("document.getElementById('chat-input') !== null")
    for _ in range(50):
        if sockets:
            break
        pg.wait_for_timeout(100)
    assert sockets, "the page opened no log socket"
    pg.fill("#chat-input", "hello")
    pg.click("#send-btn")
    pg.wait_for_function("window.__streams.length === 1")
    assert _caption(pg) == ["💭", "thinking…"]                      # nothing to show yet
    _log(pg, sockets, f"┌─ 99 {_RID}  request started  15:16:26 ───")
    _log(pg, sockets, "│  99  🌐  +0.1s  web search  postgres latest version")
    assert _caption(pg) == ["🌐", "searching the web · postgres latest version"]
    for icon, title in [("🔶", "tool warning"), ("📀", "head disk"), ("❌", "execution fail"), ("🔄", "turn queue")]:
        _log(pg, sockets, f"│  99  {icon}  +0.1s  {title}  something internal")
    assert _caption(pg) == ["🌐", "searching the web · postgres latest version"]   # untouched
    _log(pg, sockets, "│  99  🧠  +0.1s  thought  x")
    assert _caption(pg) == ["💭", "thinking…"]
    _log(pg, sockets, "│  77  🎨  +0.1s  image generation  someone else's turn")
    assert _caption(pg) == ["💭", "thinking…"]                      # another corridor: never ours
    _log(pg, sockets, "│  99  🎨  +0.1s  image generation  size=768x1024 | prompt")
    assert _caption(pg) == ["🎨", "generating an image"]
