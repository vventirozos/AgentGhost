"""§4PR (2026-10-17) — web client review: the defects four fresh-eye readers
found in app.js, the console modules, the service worker, the stylesheet
and interface/server.py (the faces are pinned in
test_interface_face_review_4pr.py).

Executed where the logic is reachable (node, Playwright Chromium, the
FastAPI TestClient); a few ORDER checks read the source where the code sits
inside a 5,000-line closure no harness boots.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function

_ROOT = Path(__file__).resolve().parent.parent
_STATIC = _ROOT / "interface" / "static"


def _js(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _block(src: str, opener: str) -> str:
    """Brace-matched text from ``opener`` (which ends in ``{``) to its close."""
    i = src.index(opener)
    j = src.index("{", i + len(opener) - 1)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(f"unbalanced: {opener}")


# ── CRIT: mermaid runs in the sandboxed frame, never in the key's document ──

# one statement: in the sandbox `top.PWNED=` throws (cross-origin); in the
# host document it lands on window
_PAYLOAD = ('sequenceDiagram\n  participant A as Alice<img src=x onerror="top.PWNED=1">$$x$$\n'
            '  A->>B: hi')


def _srcdoc(code: str, nonce: str = "n1") -> str:
    app = _js("app.js")
    src = ("const location = { href: 'http://ghost.test/' };\n"
           + re.search(r"const _escAttr = .*?\n.*?\n.*?\n", app, re.S).group(0)
           + extract_js_function(app, "_mermaidSrcdoc"))
    return eval_js(src, f"_mermaidSrcdoc({json.dumps(code)}, {json.dumps(nonce)})")


def test_mermaid_is_not_a_main_document_vendor():
    """Fails where mermaid can still be loaded into the page that holds
    GHOST_API_KEY (10.9.1 runs script from a sequence label, CVE-2025-54881)."""
    app = _js("app.js")
    i = app.index("const _VENDORS = {")
    vendors = app[i:app.index("};", i) + 2]
    out = eval_js(vendors.replace("const _VENDORS", "globalThis._VENDORS"),
                  "Object.keys(_VENDORS)")
    assert "mermaid" not in out and {"chart", "papaparse"} <= set(out)


def test_the_srcdoc_cannot_be_closed_by_the_diagram_source():
    doc = _srcdoc("graph TD\nA-->B</script><script>top.x=1</script>")
    body = doc.split("var src=", 1)[1].split(";var nonce", 1)[0]
    assert "</script" not in body.lower() and "\\u003c/script" in body
    assert 'securityLevel:"strict"' in doc


@pytest.fixture(scope="module")
def chromium():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:               # pragma: no cover
        pytest.skip("playwright not installed")
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:        # pragma: no cover
            if "Executable doesn't exist" in str(e):
                pytest.skip("chromium not installed")
            raise
        yield b
        b.close()


def _host(chromium, body_html):
    ctx = chromium.new_context()
    page = ctx.new_page()

    def handler(route):
        url = route.request.url
        if url.startswith("http://ghost.test/static/vendor/mermaid.min.js"):
            route.fulfill(status=200, content_type="application/javascript; charset=utf-8",
                          body=(_STATIC / "vendor" / "mermaid.min.js").read_bytes())
        elif url.rstrip("/") == "http://ghost.test":
            route.fulfill(status=200, content_type="text/html", body=body_html)
        else:
            route.fulfill(status=404, body="")
    page.route("**/*", handler)
    page.goto("http://ghost.test/")
    return ctx, page


def test_the_cve_payload_fires_in_a_main_document_render(chromium):
    """The harness's own control: the payload DOES run when mermaid renders
    in the host document (what the old renderMermaid did)."""
    ctx, page = _host(chromium, """<!doctype html><body><div id=c></div>
<script src="/static/vendor/mermaid.min.js"></script><script>
mermaid.initialize({startOnLoad:false,theme:'dark'});
mermaid.render('g', %s).then(r => { document.getElementById('c').innerHTML = r.svg; window.DONE = 1; })
  .catch(() => { window.DONE = 1; });
</script></body>""" % json.dumps(_PAYLOAD))
    try:
        page.wait_for_function("window.DONE === 1", timeout=20000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.PWNED") == 1
    finally:
        ctx.close()


def test_the_sandboxed_render_keeps_the_payload_away_from_the_page(chromium):
    """Fails where the diagram renders in the key-holding document again,
    or the frame gets same-origin access."""
    doc = _srcdoc(_PAYLOAD)
    ctx, page = _host(chromium, """<!doctype html><body>
<iframe id=f sandbox="allow-scripts"></iframe><script>
window.GHOST_API_KEY = 'secret';
window.GOT = null;
window.addEventListener('message', (e) => { if (e.data && e.data.ghostMermaid === 'n1') window.GOT = e.data.svg; });
</script></body>""")
    # set as a PROPERTY (as app.js does): inlined into the host's <script>,
    # the srcdoc's own </script> would close it
    page.evaluate("(d) => { document.getElementById('f').srcdoc = d; }", doc)
    try:
        page.wait_for_function("typeof window.GOT === 'string'", timeout=20000)
        page.wait_for_timeout(500)
        assert page.evaluate("window.PWNED") is None
        assert "<svg" in page.evaluate("window.GOT")
    finally:
        ctx.close()


def test_the_index_frame_is_still_scripts_only():
    html = _js("index.html")
    m = re.search(r'<iframe id="render-iframe-sandboxed" sandbox="([^"]*)"', html)
    assert m and m.group(1) == "allow-scripts"


# ── app.js behaviour ─────────────────────────────────────────────────────

def _send_composed_env(draft, processing):
    app = _js("app.js")
    return ("const sent = []; const notes = [];\n"
            f"let isProcessingRequest = {json.dumps(processing)};\n"
            f"const chatInput = {{ value: {json.dumps(draft)} }};\n"
            "function addMessage(r, t) { notes.push(t); }\n"
            "function sendMessage() { sent.push(chatInput.value); chatInput.value = ''; return Promise.resolve(); }\n"
            + extract_js_function(app, "_sendComposed"))


def test_a_composed_send_keeps_the_users_draft():
    """Fails where an upload/voice send overwrote what the user was typing."""
    out = eval_js(_send_composed_env("half a thought", False),
                  "(_sendComposed('voice text'), {sent, left: chatInput.value})")
    assert out == {"sent": ["voice text"], "left": "half a thought"}


def test_a_composed_send_during_a_turn_joins_the_draft():
    out = eval_js(_send_composed_env("draft ", True),
                  "(_sendComposed('voice text'), {sent, left: chatInput.value, notes})")
    assert out["sent"] == [] and out["left"] == "draft voice text" and out["notes"]


def _mic_env():
    app = _js("app.js")
    decls = app[app.index("let isRecording = false;"):app.index("if (micBtn) {")]
    return ("""
const handlers = {}; const msgs = []; let tracksStopped = 0; let started = 0;
let resolveMedia;
const micBtn = { addEventListener: (k, f) => { handlers[k] = f; }, classList: { add() {}, remove() {} } };
const window = {}; let audioCtx = null; let isTTSActive = false; let mediaRecorder = null; let audioChunks = [];
function stopTTS() {} function addMessage(r, t) { msgs.push(t); }
function _startMicAudioPump() {} function _stopMicAudioPump() {}
const activeFace = { setWorkingState() {} };
const navigator = { mediaDevices: { getUserMedia: () => new Promise(r => { resolveMedia = r; }) } };
class MediaRecorder { constructor() { this.state = 'inactive'; } start() { started++; } stop() {} static isTypeSupported() { return true; } }
const stream = { getTracks: () => [{ stop: () => { tracksStopped++; } }] };
""" + decls + _block(app, "if (micBtn) {"))


def test_a_release_during_the_permission_wait_records_nothing():
    """Fails where a quick tap left the mic recording after the release
    (and auto-sent it on the next tap)."""
    out = eval_js(_mic_env(), """await (async () => {
        const ev = { cancelable: false, preventDefault() {} };
        const p = handlers.touchstart(ev);
        await new Promise(r => setTimeout(r, 0));
        handlers.touchend(ev);
        resolveMedia(stream);
        await p;
        return { started, tracksStopped, isRecording, msgs };
    })()""")
    assert out["started"] == 0 and out["tracksStopped"] == 1 and out["isRecording"] is False
    assert not any("Recording started" in m for m in out["msgs"])


def test_a_held_press_still_records():
    out = eval_js(_mic_env(), """await (async () => {
        const ev = { cancelable: false, preventDefault() {} };
        const p = handlers.touchstart(ev);
        await new Promise(r => setTimeout(r, 0));
        resolveMedia(stream);
        await p;
        return { started, isRecording };
    })()""")
    assert out == {"started": 1, "isRecording": True}


def test_concurrent_image_fetches_share_one_request():
    """Fails where every streaming frame fetched the same image again."""
    app = _js("app.js")
    src = ("""
let fetches = 0;
const _authedBlobCache = new Map();
const _authedBlobInflight = new Map();
function _ownApiUrl(u, p) { return u.startsWith(p); }
function _evictAuthedBlobCache() {}
function _safeImageBlobType(t) { return t; }
const URL = { createObjectURL: () => 'blob:' + (++fetches) };
class Blob { constructor(a, o) { this.type = o.type; } }
async function fetch() { fetches += 0; await new Promise(r => setTimeout(r, 5)); return { ok: true, blob: async () => ({ type: 'image/png' }) }; }
const window = { __ghostSessionId: 's1' };
""" + extract_js_function(app, "_authedBlobKey") + "\n"
           + extract_js_function(app, "_toAuthedBlobUrl") + "\n" + extract_js_function(app, "_fetchAuthedBlobUrl"))
    out = eval_js(src, """await (async () => {
        const [a, b] = await Promise.all([_toAuthedBlobUrl('/api/download/x.png'), _toAuthedBlobUrl('/api/download/x.png')]);
        const c = await _toAuthedBlobUrl('/api/download/x.png');
        const made1 = fetches;
        // §4SX B0 r3: another session's bare x.png may be another project's file
        window.__ghostSessionId = 's2';
        const [d, e] = await Promise.all([_toAuthedBlobUrl('/api/download/x.png'), _toAuthedBlobUrl('/api/download/x.png')]);
        return { a, b, c, d, e, made1, made: fetches, inflight: _authedBlobInflight.size };
    })()""")
    assert out["a"] == out["b"] == out["c"] and out["made1"] == 1 and out["inflight"] == 0
    assert out["d"] == out["e"] != out["a"] and out["made"] == 2


def test_the_recovery_stands_down_without_touching_a_live_turn():
    """r2 M1: fails where a live turn kept the recovery poll alive (its
    give-up path then killed the LIVE turn's clock and deleted its resume
    handle), or a session switch stopped a live turn's ticker."""
    app = _js("app.js")
    src = ("""
let scheduled = 0, stopped = 0; let isProcessingRequest = true; let cur = 's1';
let _reconcileGen = 0, _reconcileAnnounced = true, _reconcileTries = 9;
function _sessionStillCurrent(s) { return s === cur; }
function _scheduleReconcilePoll() { scheduled++; }
function stopTurnTicker() { stopped++; }
""" + extract_js_function(app, "_reconcileBlocked"))
    out = eval_js(src, """(() => {
        const a = _reconcileBlocked('s1', null);           // live turn: stand down
        cur = 's2'; const b = _reconcileBlocked('s1', null); // switched, live turn owns the ticker
        isProcessingRequest = false; const c = _reconcileBlocked('s1', null);
        cur = 's1'; const d = _reconcileBlocked('s1', null);
        return { a, b, c, d, scheduled, stopped, tries: _reconcileTries, ann: _reconcileAnnounced };
    })()""")
    # r4: a recovery ended by a switch leaves a full budget for the next one
    assert out == {"a": True, "b": True, "c": True, "d": False, "scheduled": 0, "stopped": 1,
                   "tries": 0, "ann": False}
    body = extract_js_function(app, "reconcileFromSession")
    assert "return; // sessions store hiccup" not in body


def test_a_send_ends_a_pending_recovery_cleanly():
    """Fails where a send left the older turn's placeholder dangling, or
    cleared anything but the OLD handle."""
    app = _js("app.js")
    src = ("""
const cleared = []; const notes = []; let stopped = 0; let removed = 0;
let reconcilePollTimer = 7; let _reconcileHandle = { tabId: 'old' };
let _reconcileAnnounced = true, _reconcileTries = 30;
let currentAgentMessageDiv = { classList: { contains: (c) => c === 'thinking' }, remove() { removed++; } };
let _reconcileGen = 0, _reconcileRunning = 0, _reconcileSid = 's1';
function _sessionStillCurrent(s) { return s === 's1'; }
function clearInflightHandle(h) { cleared.push(h && h.tabId); reconcilePollTimer = null; }
function stopTurnTicker() { stopped++; }
function addMessage(r, t) { notes.push(t); }
""" + extract_js_function(app, "_endRecoveryForNewTurn"))
    out = eval_js(src, """(() => {
        _endRecoveryForNewTurn();
        _endRecoveryForNewTurn();                  // nothing pending: a no-op
        return { cleared, notes: notes.length, stopped, removed, div: currentAgentMessageDiv,
                 tries: _reconcileTries, announced: _reconcileAnnounced, handle: _reconcileHandle };
    })()""")
    assert out == {"cleared": ["old"], "notes": 1, "stopped": 1, "removed": 1, "div": None,
                   "tries": 0, "announced": False, "handle": None}
    send = app[app.index("async function sendMessage"):]
    send = send[:send.index("const _userDiv = addMessage('user', text);")]
    assert "_endRecoveryForNewTurn();" in send


def test_order_a_post_that_never_left_is_reported_not_resumed():
    """ORDER check (sendMessage is a closure no harness boots): the silent
    resume branch requires a task id."""
    app = _js("app.js")
    assert "if (currentTaskId && (errMsg.includes('load failed')" in app


def test_order_a_loaded_workspace_is_saved_after_the_rebind():
    app = _js("app.js")
    i = app.index("chatHistory = Array.isArray(result.chat_history)")
    seg = app[i:i + 2500]
    assert seg.index("new CustomEvent('conversation-cleared')") < seg.index("saveChatState();")


def test_every_open_path_supersedes_an_awaiting_render():
    app = _js("app.js")
    body = extract_js_function(app, "resetRenderSurfaces")
    assert "_renderSeq += 1" in body
    # declared before the top-level boot runs (TDZ)
    boot = app.index("\nloadChatState();")
    for decl in ("let _renderSeq = 0", "const _authedBlobInflight", "const _autoOpenedImgSrcs"):
        assert app.index(decl) < boot, decl


# ── console modules ─────────────────────────────────────────────────────

def test_the_outside_click_closer_spares_its_openers():
    """Fails where a toast / palette-row click opened the bell panel and the
    same click closed it again."""
    js = _js("notifications.js")
    i = js.index("document.addEventListener('click', (e) => {")
    listener = js[i:js.index("});", i) + 3]
    src = ("""
let closed = 0;
const panel = { classList: { contains: () => false }, contains: () => false };
const btn = { contains: () => false };
function closePanel() { closed++; }
const handlers = [];
const document = { addEventListener: (k, f) => handlers.push(f) };
const target = (sel) => ({ closest: (q) => q.split(',').map(s => s.trim()).includes(sel) ? {} : null });
""" + listener)
    out = eval_js(src, """(() => {
        handlers[0]({ target: target('#toast-stack') });
        handlers[0]({ target: target('#cmd-palette') });
        const before = closed;
        handlers[0]({ target: target('#chat-log') });
        return { before, after: closed };
    })()""")
    assert out == {"before": 0, "after": 1}


def test_clear_all_keeps_the_dedupe_set():
    js = _js("notifications.js")
    i = js.index("getElementById('notif-clear')")
    body = js[i:js.index("});", i)]
    assert "seen = new Set()" not in body


def test_palette_will_not_open_under_a_modal():
    """ORDER check: ⌘K stands down while another aria-modal is open (the
    palette shares z-index 150 and opened BEHIND it)."""
    js = _js("palette.js")
    i = js.index("e.key.toLowerCase() === 'k'")
    seg = js[i:js.index("return;", i)]
    assert '[aria-modal="true"]:not(.hidden):not(#cmd-palette)' in seg
    assert "if (isOpen()) close(); else open();" in js[i:i + 600]


def test_prune_keeps_another_tabs_unsaved_chat():
    js = _js("sessions.js")
    fn = extract_js_function(js, "pruneLocalHistories")
    out = eval_js("""
const removed = []; const store = {};
let sessions = [{id: 'a'}], currentId = 'a';
const LISTED_KEY = 'ghost_listed_sessions';
const Core = { localHistoryKeys: () => ['ghost_chat_history:a', 'ghost_chat_history:newtab'],
               safeStorage: { remove: (k) => removed.push(k), get: (k) => store[k] || null, set: (k, v) => { store[k] = v; } } };
""" + fn, """(() => {
        pruneLocalHistories();
        sessions = [];  currentId = null;          // 'a' deleted elsewhere
        pruneLocalHistories();
        return removed;
    })()""")
    assert out == ["ghost_chat_history:a"]


def test_the_service_worker_precaches_from_the_network():
    sw = _js("sw.js")
    i = sw.index("self.addEventListener('install'")
    assert "cache: 'reload'" in sw[i:sw.index("});", i)]


def test_style_css_is_in_the_cachebust_manifest():
    from tests.test_interface_cachebust import _CARRIER
    assert _CARRIER.get("style.css") == "index.html"


def _css():
    return _js("style.css")


def test_toasts_sit_above_modals_and_below_the_message_menu():
    css = _css()
    z = lambda sel: int(re.search(re.escape(sel) + r"\s*\{[^}]*?z-index:\s*(\d+)", css).group(1))
    assert z("#memory-modal") < z("#toast-stack") < z("#msg-menu")


def test_a_closed_rail_is_inert():
    js = _js("workspace.js")
    fn = extract_js_function(js, "setRailOpen")
    out = eval_js("""
const rail = { inert: false, contains: () => false };
const body = { classList: { toggle() {} } };
const document = { body, getElementById: (id) => id === 'session-rail' ? rail : null };
function isDocked() { return false; }
const Core = { safeStorage: { set() {} } };
""" + fn, "(() => { setRailOpen(false); const a = rail.inert; setRailOpen(true); return [a, rail.inert]; })()")
    assert out == [True, False]


# ── server ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    import os
    os.environ.setdefault("GHOST_API_KEY", "testkey")
    import sys
    sys.path.insert(0, str(_ROOT / "interface"))
    import interface.server as server
    from fastapi.testclient import TestClient
    return TestClient(server.app), server


def _h(server):
    return {"X-Ghost-Key": server.GHOST_API_KEY}


def test_a_malformed_stt_upload_is_a_400(client):
    c, server = client
    r = c.post("/api/stt", headers={**_h(server), "Content-Type": "multipart/form-data"}, content=b"x")
    assert r.status_code == 400, r.text


def test_tts_rejects_non_string_text(client):
    c, server = client
    r = c.post("/api/tts", headers=_h(server), json={"text": ["x"]})
    assert r.status_code == 400


@pytest.mark.parametrize("keys", ["abc", ["a"], {"p256dh": 1, "auth": "x"}, {"p256dh": "x"}])
def test_push_subscriptions_need_string_keys(keys, tmp_path, monkeypatch):
    import interface.webpush_notify as wp
    monkeypatch.setattr(wp, "_load_subs", lambda: {})
    saved = []
    monkeypatch.setattr(wp, "_save_subs", lambda s: saved.append(s))
    assert wp.add_subscription({"endpoint": "https://web.push.apple.com/x", "keys": keys}) is False
    assert not saved


def test_a_stored_subscription_carries_only_what_a_send_needs(monkeypatch):
    import interface.webpush_notify as wp
    monkeypatch.setattr(wp, "_load_subs", lambda: {})
    saved = []
    monkeypatch.setattr(wp, "_save_subs", lambda s: saved.append(s))
    assert wp.add_subscription({"endpoint": "https://web.push.apple.com/x",
                                "keys": {"p256dh": "p", "auth": "a"}, "junk": "x" * 10}) is True
    assert saved[0]["https://web.push.apple.com/x"] == {
        "endpoint": "https://web.push.apple.com/x", "keys": {"p256dh": "p", "auth": "a"}}


@pytest.mark.parametrize("size", [2 * 1024 * 1024, 1024 * 1024 + 10])
def test_a_long_log_line_does_not_kill_the_stream(tmp_path, monkeypatch, size):
    """Fails where one line over 64 KiB raised out of the tail loop."""
    import asyncio
    import interface.server as server
    log = tmp_path / "agent.log"
    # 2 MiB: dropped before its newline arrives; limit+10: its newline
    # arrives in the chunk that crosses the limit
    log.write_text("first\n" + "x" * size + "\nafter\n", encoding="utf-8")
    monkeypatch.setattr(server, "AGENT_LOG_PATH", str(log))
    got = []

    class _WS:
        async def send_text(self, m):
            got.append(json.loads(m)["content"])
    monkeypatch.setattr(server, "connected_websockets", {_WS()})

    async def run():
        t = asyncio.create_task(server._log_streamer_once())
        for _ in range(100):
            await asyncio.sleep(0.05)
            if "after" in got:
                break
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    # its own thread: a sync-Playwright fixture in this module may hold a
    # running loop on the test thread
    import threading
    th = threading.Thread(target=lambda: asyncio.run(run()))
    th.start()
    th.join(30)
    assert "first" in got and "after" in got
    # r2: no fragment of the over-long line reaches the console
    assert got == ["first", "after"], [g[:20] for g in got]


def test_a_long_line_is_shortened_for_the_console(tmp_path, monkeypatch):
    import asyncio, threading
    import interface.server as server
    log = tmp_path / "agent.log"
    log.write_text("y" * 100_000 + "\nend\n", encoding="utf-8")
    monkeypatch.setattr(server, "AGENT_LOG_PATH", str(log))
    got = []

    class _WS:
        async def send_text(self, m):
            got.append(json.loads(m)["content"])
    monkeypatch.setattr(server, "connected_websockets", {_WS()})

    async def run():
        t = asyncio.create_task(server._log_streamer_once())
        for _ in range(100):
            await asyncio.sleep(0.05)
            if "end" in got:
                break
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    th = threading.Thread(target=lambda: asyncio.run(run()))
    th.start()
    th.join(30)
    assert got[-1] == "end" and got[0].endswith("[truncated]") and len(got[0]) < 20_000


def test_a_client_that_drops_mid_upload_is_a_400():
    import asyncio
    import interface.server as server
    from starlette.requests import ClientDisconnect

    class _Req:
        async def form(self):
            raise ClientDisconnect()
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", asyncio.run(server.stt_proxy(_Req()))))
    th.start()
    th.join(30)
    assert box["r"].status_code == 400


def test_render_mermaid_uses_the_sandboxed_frame_and_never_the_page():
    """Fails where renderMermaid writes the diagram (or the posted SVG)
    into the key-holding document."""
    from tests.helpers import extract_js_listener
    app = _js("app.js")
    src = ("""
const frame = { style: {}, srcdoc: '', contentWindow: { id: 'frame' } };
const renderIframeSandboxed = frame;
let _renderSeq = 0; let _mermaidNonce = ''; let currentRenderState = null;
const crypto = { randomUUID: () => 'nonce-1' };
function _mermaidSrcdoc(c, n) { return 'DOC:' + n + ':' + c; }
let innerWrites = 0;
const mermaidContainer = { set innerHTML(v) { innerWrites++; } };
""" + extract_js_function(app, "renderMermaid") + "\n"
           + extract_js_listener(app, "message").replace("function handler(e)", "function handler(ev)", 1))
    out = eval_js(src, """(() => {
        renderMermaid('graph TD');
        handler({ data: { ghostMermaid: 'nonce-1', svg: '<svg/>' }, source: frame.contentWindow });
        const svgOk = currentRenderState.svg;
        handler({ data: { ghostMermaid: 'forged', svg: 'x' }, source: frame.contentWindow });
        handler({ data: { ghostMermaid: 'nonce-1', svg: 'y' }, source: { id: 'other' } });
        return { srcdoc: frame.srcdoc, display: frame.style.display, svgOk,
                 svg: currentRenderState.svg, innerWrites };
    })()""")
    assert out == {"srcdoc": "DOC:nonce-1:graph TD", "display": "block", "svgOk": "<svg/>",
                   "svg": "<svg/>", "innerWrites": 0}


def test_an_image_auto_opens_once_per_turn():
    """Fails where every streaming frame re-opened the visualizer."""
    app = _js("app.js")
    src = ("""
let opened = 0; let _restoringHistory = false;
const _autoOpenedImgSrcs = new Set();
function _ownApiUrl(u, p) { return u.startsWith(p); }
async function _toAuthedBlobUrl(u) { return 'blob:1'; }
function _showBrokenImagePlaceholder() {}
function openImageInVisualizer() { opened++; }
const document = { getElementById: () => ({}), createElement: () => ({ style: {} }) };
function mkImg() {
    return { getAttribute: () => '/api/download/a.png', dataset: {}, classList: { add() {} },
             addEventListener() {}, parentNode: { insertBefore() {} }, set src(v) {} };
}
""" + extract_js_function(app, "_handleChatImage"))
    out = eval_js(src, """await (async () => {
        await _handleChatImage(mkImg()); await _handleChatImage(mkImg()); await _handleChatImage(mkImg());
        const first = opened;
        _autoOpenedImgSrcs.clear();                     // a new turn
        await _handleChatImage(mkImg());
        return { first, after: opened };
    })()""")
    assert out == {"first": 1, "after": 2}


@pytest.mark.parametrize("turn_live", [True, False])
def test_a_send_during_a_running_poll_ends_it_too(turn_live):
    """r3 MAJOR: fails where a send landing while a poll awaited
    /api/turns (timer already null) left the placeholder, the old handle
    and a poll that could re-arm during the live turn."""
    app = _js("app.js")
    src = ("""
const said = []; const cleared = []; const scheduled = [];
let reconcilePollTimer = null, _reconcileAnnounced = false, _reconcileTries = 0;
const _RECONCILE_MAX_TRIES = 120;
let _reconcileHandle = null, _reconcileGen = 0, _reconcileRunning = 0, _reconcileSid = null;
const _RECONCILE_FETCH_MS = 15000; function _timeoutSignal() { return {}; } let currentTaskId = null;
let isProcessingRequest = false; let chatHistory = [];
let currentAgentMessageDiv = null;
function addMessage(k, t) { said.push(t); }
function ensureAgentBubbleForResume() { currentAgentMessageDiv = { classList: { contains: (c) => c === 'thinking' }, remove() { said.push('REMOVED'); } }; }
function stopTurnTicker() {}
function clearInflightHandle(h) { cleared.push(h && h.tabId); if (reconcilePollTimer) reconcilePollTimer = null; }
function renderHistoryToLog() {} function saveChatState() {} function mergeClientLabelKeys(m) { return m; }
function _sessionStillCurrent() { return true; }
function setTimeout(fn) { scheduled.push(fn); return scheduled.length; }
let release;
globalThis.fetch = (url) => new Promise((r) => { release = () => r({ ok: true,
    json: async () => ({ turns: [{ session_id: 'S', running: true }] }) }); });
""" + extract_js_function(app, "_scheduleReconcilePoll") + "\n"
           + extract_js_function(app, "_reconcileBlocked") + "\n"
           + extract_js_function(app, "_endRecoveryForNewTurn") + "\n"
           + extract_js_function(app, "reconcileFromSession"))
    out = eval_js(f"const TURN_LIVE = {str(turn_live).lower()};\n" + src, """await (async () => {
        const p = reconcileFromSession('S', { tabId: 'old' });   // awaiting /api/turns
        await new Promise(r => setTimeout === r ? 0 : globalThis.queueMicrotask(r));
        _endRecoveryForNewTurn();                               // the user sends
        // live, or already finished by the time the old poll resumes —
        // only the generation tells the old poll it was ended
        isProcessingRequest = TURN_LIVE;
        release();
        await p;
        return { cleared, said, scheduled: scheduled.length, running: _reconcileRunning };
    })()""")
    assert out["cleared"] == ["old"]
    assert any("Stopped waiting" in t for t in out["said"])
    assert out["scheduled"] == 0 and out["running"] == 0
    assert not any("still working" in t for t in out["said"])


def test_no_poll_is_armed_during_a_live_turn():
    app = _js("app.js")
    src = ("""
let reconcilePollTimer = null, _reconcileTries = 0, _reconcileAnnounced = false, _reconcileHandle = null;
const _RECONCILE_MAX_TRIES = 120; let isProcessingRequest = true; let armed = 0;
function setTimeout() { armed++; return 1; }
function addMessage() {} function stopTurnTicker() {} function clearInflightHandle() {}
""" + extract_js_function(app, "_scheduleReconcilePoll"))
    assert eval_js(src, "(_scheduleReconcilePoll('S', null), armed)") == 0


# ── review round 4: the network-error resume vs the recovery ─────────────

def test_one_recovery_run_at_a_time():
    """r4: fails where a second run beside an in-flight one posted
    "Recovered…" then "could not be recovered"."""
    app = _js("app.js")
    src = ("""
let fetched = 0; let _reconcileRunning = 1, _reconcileGen = 0, _reconcileHandle = null, _reconcileSid = null;
const _RECONCILE_FETCH_MS = 15000; function _timeoutSignal() { return {}; } let currentTaskId = null;
let isProcessingRequest = false;
globalThis.fetch = async () => { fetched++; return { ok: false, status: 500 }; };
function clearInflightHandle() {}
""" + extract_js_function(app, "reconcileFromSession"))
    assert eval_js(src, "await (async () => { await reconcileFromSession('S', null); return fetched; })()") == 0


def test_order_a_resume_does_not_fight_a_recovery():
    """ORDER checks inside sendMessage's closure: the network resume stands
    down while a recovery owns the turn, and a resume the proxy lost hands
    over to the session recovery instead of offering a Retry."""
    app = _js("app.js")
    i = app.index("const _resumeAttempt = () => {")
    head = app[i:i + 700]
    assert "if (reconcilePollTimer || _reconcileRunning) return;" in head
    j = app.index("} else if (resuming) {")
    branch = app[j:j + 900]
    assert "reconcileFromSession(_rsid, _rh)" in branch and "currentTaskId = null;" in branch
    assert "addRetryableSystemMessage" in branch.split("} else {", 1)[1][:200]
    assert "if (!resuming) clearInflightHandle();" in app


def test_edit_and_clear_end_a_recovery_silently():
    app = _js("app.js")
    trunc = extract_js_function(app, "_truncateToUserTurn")
    assert "_endRecoveryForNewTurn(true);" in trunc
    send = app[app.index("async function sendMessage"):]
    clear = send[send.index("if (text === '/clear' && !resuming) {"):]
    assert clear.index("_endRecoveryForNewTurn(true);") < clear.index("chatLog.innerHTML = '';")



def test_recovery_fetches_are_bounded_and_settle_the_task_id():
    """r5 MAJOR: fails where a recovery fetch has no timeout (one hung
    request held `_reconcileRunning` and disabled recovery for the page),
    or a settled turn kept its task id (the network resume then re-ran)."""
    app = _js("app.js")
    src = ("""
const signals = []; const said = [];
let reconcilePollTimer = null, _reconcileAnnounced = false, _reconcileTries = 0;
const _RECONCILE_MAX_TRIES = 120;
let _reconcileHandle = null, _reconcileGen = 0, _reconcileRunning = 0, _reconcileSid = null;
const _RECONCILE_FETCH_MS = 15000; function _timeoutSignal() { return {}; }
let currentTaskId = 'T2'; let isProcessingRequest = false; let chatHistory = [{role:'user', content:'q'}];
let currentAgentMessageDiv = null;
function addMessage(k, t) { said.push(t); }
function ensureAgentBubbleForResume() {} function stopTurnTicker() {}
function clearInflightHandle() {} function renderHistoryToLog() {} function saveChatState() {}
function mergeClientLabelKeys(m) { return m; } function _sessionStillCurrent() { return true; }
function _reconcileWithLocal(a) { return a; } function _adoptionChanges() { return true; }
function setTimeout() { return 1; }
globalThis.fetch = async (url, opts) => {
    signals.push(!!(opts && opts.signal));
    if (url === '/api/turns') return { ok: true, json: async () => ({ turns: [] }) };
    return { ok: true, json: async () => ({ messages: [{role:'user', content:'q'}, {role:'assistant', content:'a'}] }) };
};
""" + extract_js_function(app, "_scheduleReconcilePoll") + "\n"
           + extract_js_function(app, "_reconcileBlocked") + "\n"
           + extract_js_function(app, "reconcileFromSession"))
    out = eval_js(src, """await (async () => {
        await reconcileFromSession('S', { tabId: 't', taskId: 'T2' });
        return { signals, taskId: currentTaskId, said };
    })()""")
    assert out["signals"] == [True, True]
    assert out["taskId"] is None
