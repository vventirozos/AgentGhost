"""Web client batch 3 (2026-09-11) — service worker, face pause, TLS watch.

  1. Service worker: a real fetch handler (network-first "/", cache-first
     "/static/*", nothing else intercepted), a build-stamped cache the
     server renders from the shipped assets, an UPDATE flow the page
     controls (no skipWaiting on install), `pushsubscriptionchange`
     re-subscription, and a notification click that picks the right window
     and tells it.
  2. `/api/push/vapid` accepts the page cookie (read-only public key) so the
     worker can re-subscribe with no page open; every writing route keeps
     the header-only rule. Auth-rejected push endpoints are pruned only when
     the same broadcast proved our VAPID key works.
  3. The WebGL face pauses while the document is hidden.
  4. TLS certificate expiry: `tls_cert_status`, a daily log line, an
     interface-local health route, and an amber chip under 14 days.

Behavioural tests run under node (the rendered worker with a stub global,
the page helpers with a DOM shim) or against the FastAPI app.
"""

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interface.server as server  # noqa: E402
from interface import webpush_notify  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_STATIC = _ROOT / "interface" / "static"
AUTH = {"X-Ghost-Key": server.GHOST_API_KEY}


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js_nc(app_js) -> str:
    return strip_js_comments(app_js)


@pytest.fixture(scope="module")
def sw_js() -> str:
    """The worker AS SERVED — placeholders filled by the server."""
    return server._render_sw()


@pytest.fixture(scope="module")
def sw_js_nc(sw_js) -> str:
    return strip_js_comments(sw_js)


@pytest.fixture(scope="module")
def status_js() -> str:
    return strip_js_comments((_STATIC / "status.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph_js() -> str:
    return strip_js_comments((_STATIC / "matrix_graph.js").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sessions_js() -> str:
    return strip_js_comments((_STATIC / "sessions.js").read_text(encoding="utf-8"))


# Evaluate the whole worker with a recording ServiceWorkerGlobalScope stub.
_SW_GLOBAL = """
const handlers = {};
const cacheStore = new Map();          // name -> Map(url -> Response)
globalThis.caches = {
    open: async (name) => {
        if (!cacheStore.has(name)) cacheStore.set(name, new Map());
        const m = cacheStore.get(name);
        return {
            add: async (req) => { const r = await fetch(req); m.set(new URL(req.url).pathname + new URL(req.url).search, r); },
            put: async (req, res) => { const u = typeof req === 'string' ? req : new URL(req.url).pathname + new URL(req.url).search; m.set(u, res); },
            match: async (req) => { const u = typeof req === 'string' ? req : new URL(req.url).pathname + new URL(req.url).search; return m.get(u) || undefined; },
        };
    },
    keys: async () => [...cacheStore.keys()],
    delete: async (n) => cacheStore.delete(n),
};
const posted = [];
globalThis.self = {
    addEventListener: (ev, fn) => { handlers[ev] = fn; },
    location: { origin: 'https://ghost.test' },
    skipWaiting: () => { posted.push('skipWaiting'); },
    clients: { claim: async () => { posted.push('claim'); }, matchAll: async () => globalThis.__clients || [], openWindow: async (u) => { posted.push(['open', u]); } },
    registration: { showNotification: async () => {}, pushManager: { subscribe: async (o) => { posted.push(['subscribe', !!o.applicationServerKey]); return {}; } } },
};
globalThis.atob = (s) => Buffer.from(s, 'base64').toString('binary');
function evt(extra) {
    const waits = [];
    return Object.assign({
        _waits: waits,
        waitUntil(p) { waits.push(p); },
        respondWith(p) { waits.push(p); this.response = p; },
    }, extra);
}
"""


def _sw_env(sw_js):
    # Strip the module-ish `new URL(request.url)`… nothing to strip: the
    # worker is plain script; run it inside a Function so `self` resolves.
    return _SW_GLOBAL + "\nnew Function(" + json.dumps(sw_js) + ").call(globalThis);\n" \
        + "function h(ev, e) { return handlers[ev](e); }\n"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Service worker
# ═══════════════════════════════════════════════════════════════════════════

class TestRenderedWorker:
    def test_placeholders_are_filled_and_the_list_is_the_shipped_shell(self, sw_js):
        assert "__GHOST_SW" not in sw_js
        build = re.search(r'const BUILD = "([0-9a-f]{12})";', sw_js)
        assert build, "no build stamp"
        pre = json.loads(re.search(r"const PRECACHE = (\[.*?\]);", sw_js).group(1))
        index = (_STATIC / "index.html").read_text()
        app = (_STATIC / "app.js").read_text()
        ws = (_STATIC / "workspace.js").read_text()
        for src, pat in ((index, r"/static/(app\.js\?v=[\d.]+)"), (index, r"/static/(style\.css\?v=[\d.]+)"),
                         (app, r"\./(workspace\.js\?v=[\d.]+)"), (ws, r"\./(sessions\.js\?v=[\d.]+)")):
            m = re.search(pat, src)
            assert m and f"/static/{m.group(1)}" in pre, (m and m.group(1), pre)
        assert pre[0] == "/" and "/static/vendor/purify.min.js" in pre
        assert not any("mermaid" in u for u in pre), "the lazy 3.3 MB library must not be precached"

    def test_build_stamp_moves_when_a_shipped_file_changes(self, tmp_path):
        (tmp_path / "index.html").write_text('<script src="/static/app.js?v=1.0"></script>')
        (tmp_path / "app.js").write_text("a")
        (tmp_path / "workspace.js").write_text("")
        (tmp_path / "sw.js").write_text("")
        urls = server._sw_precache_list(tmp_path)
        assert urls[:2] == ["/", "/static/app.js?v=1.0"]
        a = server._sw_build_stamp(urls, tmp_path)
        assert a == server._sw_build_stamp(urls, tmp_path)
        time.sleep(1.05)                      # mtime resolution is a second
        (tmp_path / "app.js").write_text("ab")   # edit WITHOUT a ?v= bump
        assert server._sw_build_stamp(urls, tmp_path) != a, (
            "an unbumped edit kept the old cache name — stale code for the cache's lifetime")

    def test_route_is_served_fresh_with_no_cache(self):
        r = TestClient(server.app).get("/sw.js")
        assert r.status_code == 200 and "no-cache" in r.headers["cache-control"]
        assert r.headers["content-type"].startswith("application/javascript")
        assert "__GHOST_SW" not in r.text

    def test_worker_evaluates_and_registers_every_handler(self, sw_js):
        out = eval_js(_sw_env(sw_js), "Object.keys(handlers).sort()")
        assert out == sorted(["install", "activate", "message", "fetch", "push",
                              "pushsubscriptionchange", "notificationclick"])


class TestWorkerRouting:
    def test_route_table(self, sw_js):
        fn = extract_js_function(sw_js, "routeFor")
        out = eval_js(fn, "['/', '/static/app.js', '/api/chat', '/ws', '/manifest.webmanifest', '/sw.js', '/static/'].map(routeFor)")
        assert out == ["shell", "static", None, None, None, None, "static"]

    def test_non_get_and_cross_origin_and_api_are_not_intercepted(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
let intercepted = 0;
for (const [method, url] of [['POST', 'https://ghost.test/'], ['GET', 'https://evil.test/static/x.js'], ['GET', 'https://ghost.test/api/chat'], ['GET', 'https://ghost.test/ws']]) {
    const e = evt({ request: new Request(url, { method }) });
    h('fetch', e);
    if (e._waits.length) intercepted++;
}
""", "intercepted")
        assert out == 0

    def test_shell_is_network_first_and_caches_only_a_200(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async (req) => new Response('<html>401</html>', { status: 401 });
const e1 = evt({ request: new Request('https://ghost.test/?key=k') });
h('fetch', e1);
const r1 = await e1.response;
await new Promise(r => setTimeout(r, 5));
const names1 = [...cacheStore.keys()];
const after401 = names1.length ? await (await caches.open(names1[0])).match('/') : undefined;
globalThis.fetch = async (req) => new Response('<html>page</html>', { status: 200 });
const e2 = evt({ request: new Request('https://ghost.test/') });
h('fetch', e2);
const r2 = await e2.response;
await new Promise(r => setTimeout(r, 5));
const cached = await (await caches.open([...cacheStore.keys()][0])).match('/');
""", "await (async () => ({ s1: r1.status, after401: after401 ? await after401.text() : null, s2: r2.status, cached: cached ? await cached.text() : null }))()")
        assert out == {"s1": 401, "after401": None, "s2": 200, "cached": "<html>page</html>"}, (
            "a 401 page was cached as the shell — offline would show the login wall forever")

    def test_offline_shell_falls_back_to_the_cache_then_to_the_offline_page(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
const e1 = evt({ request: new Request('https://ghost.test/') });
h('fetch', e1);
const r1 = await e1.response;
const t1 = await r1.text();
const c = await caches.open([...cacheStore.keys()][0]);
await c.put('/', new Response('<html>last good</html>', { status: 200 }));
const e2 = evt({ request: new Request('https://ghost.test/') });
h('fetch', e2);
const r2 = await e2.response;
""", "await (async () => ({ s1: r1.status, offline: t1.includes('offline') || t1.includes('unreachable'), s2: r2.status, t2: await r2.text() }))()")
        assert out["s1"] == 503 and out["offline"] is True
        assert out["s2"] == 200 and out["t2"] == "<html>last good</html>"

    def test_static_is_cache_first_and_fills_on_miss(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
let fetches = 0;
globalThis.fetch = async (req) => { fetches++; return new Response('js', { status: 200 }); };
const req = () => new Request('https://ghost.test/static/app.js?v=9');
const e1 = evt({ request: req() }); h('fetch', e1); await e1.response;
await new Promise(r => setTimeout(r, 5));
const e2 = evt({ request: req() }); h('fetch', e2); const r2 = await e2.response;
""", "await (async () => ({ fetches, body: await r2.text() }))()")
        assert out == {"fetches": 1, "body": "js"}

    def test_static_offline_miss_is_a_504_not_an_exception(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async () => { throw new TypeError('offline'); };
const e = evt({ request: new Request('https://ghost.test/static/nothing.js') }); h('fetch', e); const r = await e.response;
""", "r.status")
        assert out == 504


class TestWorkerLifecycle:
    def test_install_precaches_but_does_not_skip_waiting(self, sw_js, sw_js_nc):
        i = sw_js_nc.index("self.addEventListener('install'")
        body = sw_js_nc[i:sw_js_nc.index("self.addEventListener('activate'")]
        assert "skipWaiting" not in body, "install still swaps the worker under a live page"
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async (req) => new Response('x', { status: 200 });
const e = evt({}); h('install', e); await Promise.all(e._waits);
const names = [...cacheStore.keys()];
""", "({ names, n: cacheStore.get(names[0]).size, posted })")
        assert len(out["names"]) == 1 and out["names"][0].startswith("ghost-shell-")
        assert out["n"] >= 10, "the shell was not precached"
        assert "skipWaiting" not in out["posted"]

    def test_install_tolerates_one_missing_asset(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async (req) => { if (req.url.includes('palette')) throw new TypeError('404'); return new Response('x'); };
const e = evt({}); h('install', e); let failed = false;
try { await Promise.all(e._waits); } catch (err) { failed = true; }
""", "({ failed, n: cacheStore.get([...cacheStore.keys()][0]).size })")
        assert out["failed"] is False and out["n"] >= 9

    def test_activate_sweeps_old_shells_and_claims(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
cacheStore.set('ghost-shell-old', new Map()); cacheStore.set('other-app', new Map());
const e = evt({}); h('activate', e); await Promise.all(e._waits);
""", "({ names: [...cacheStore.keys()], posted })")
        assert "ghost-shell-old" not in out["names"] and "other-app" in out["names"]
        assert "claim" in out["posted"]

    def test_skip_waiting_only_on_the_pages_message(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
h('message', { data: { type: 'something-else' } });
const before = posted.length;
h('message', { data: { type: 'SKIP_WAITING' } });
""", "({ before, after: posted })")
        assert out["before"] == 0 and out["after"] == ["skipWaiting"]


class TestWorkerPush:
    def test_subscription_change_resubscribes_with_the_old_key_and_tells_pages(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.__clients = [{ postMessage: (m) => posted.push(['msg', m.type]) }];
const e = evt({ oldSubscription: { options: { applicationServerKey: new Uint8Array([1,2,3]) } } });
h('pushsubscriptionchange', e); await Promise.all(e._waits);
""", "posted")
        assert out == [["subscribe", True], ["msg", "push-resubscribed"]]

    def test_subscription_change_without_an_old_key_asks_the_server(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
const urls = [];
globalThis.fetch = async (u, o) => { urls.push([u, o && o.credentials]); return { ok: true, json: async () => ({ enabled: true, key: 'AQID' }) }; };
globalThis.__clients = [];
const e = evt({ oldSubscription: null });
h('pushsubscriptionchange', e); await Promise.all(e._waits);
""", "({ urls, posted })")
        assert out["urls"] == [["/api/push/vapid", "same-origin"]]
        assert out["posted"] == [["subscribe", True]]

    def test_subscription_change_gives_up_quietly_when_push_is_off(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.fetch = async () => ({ ok: true, json: async () => ({ enabled: false, key: null }) });
const e = evt({ oldSubscription: null });
h('pushsubscriptionchange', e); await Promise.all(e._waits);
""", "posted")
        assert out == []

    def test_click_prefers_the_focused_window_and_tells_it(self, sw_js):
        fn = extract_js_function(sw_js, "pickClient")
        assert eval_js(fn, "pickClient([{id:1, visibilityState:'visible'}, {id:2, focused:true}]).id") == 2
        assert eval_js(fn, "pickClient([{id:1, visibilityState:'hidden'}, {id:2, visibilityState:'visible'}]).id") == 2
        assert eval_js(fn, "pickClient([{id:1}, {id:2}]).id") == 1
        assert eval_js(fn, "pickClient([])") is None
        out = eval_js(_sw_env(sw_js) + """
const c = { focused: true, focus: async () => posted.push('focus'), postMessage: (m) => posted.push(['msg', m]) };
globalThis.__clients = [c];
const e = evt({ notification: { close() {}, tag: 'ghost-turn-ab', data: { url: '/?key=k' } } });
h('notificationclick', e); await Promise.all(e._waits);
""", "posted")
        assert out == ["focus", ["msg", {"type": "push-click", "url": "/?key=k", "tag": "ghost-turn-ab"}]]

    def test_click_with_no_window_opens_the_keyed_url(self, sw_js):
        out = eval_js(_sw_env(sw_js) + """
globalThis.__clients = [];
const e = evt({ notification: { close() {}, tag: 't', data: { url: '/?key=k' } } });
h('notificationclick', e); await Promise.all(e._waits);
""", "posted")
        assert out == [["open", "/?key=k"]]


class TestPageSideOfTheWorker:
    def test_worker_messages_are_acted_on(self, app_js):
        fn = extract_js_function(app_js, "_onServiceWorkerMessage")
        out = eval_js("""
const calls = [];
function ensurePushSubscription() { calls.push('subscribe'); }
function resumeOrReconcileInflightTurn() { calls.push('resume'); }
const dispatched = [];
globalThis.window = { GhostCore: { events: { dispatchEvent: (e) => dispatched.push(e.type) } } };
globalThis.CustomEvent = class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } };
""" + fn, """[
            _onServiceWorkerMessage({type: 'push-resubscribed'}),
            _onServiceWorkerMessage({type: 'push-click', url: '/'}),
            _onServiceWorkerMessage({type: 'nope'}), _onServiceWorkerMessage(null),
            calls, dispatched]""")
        assert out == [True, True, False, False, ["subscribe", "resume"], ["push-click"]]

    def test_update_offer_posts_skip_waiting_only_on_reload_click(self, app_js):
        fn = extract_js_function(app_js, "_offerServiceWorkerUpdate")
        out = eval_js("""
let _swReloadPending = false;
const made = [];
function addMessage(role, text) { const d = { role, text, kids: [], appendChild(c) { this.kids.push(c); } }; made.push(d); return d; }
globalThis.document = { createElement: (t) => ({ tag: t, on: {}, addEventListener(ev, fn) { this.on[ev] = fn; }, setAttribute() {} }),
                        createTextNode: (t) => ({ t }) };
const sent = [];
const reg = { waiting: { postMessage: (m) => sent.push(m) } };
globalThis.location = { reload: () => sent.push('reload') };
""" + fn, """(() => {
            const d = _offerServiceWorkerUpdate(reg);
            const before = sent.slice();
            const btn = d.kids.find(k => k.tag === 'button');
            btn.on.click();
            return { role: d.role, before, after: sent, pending: _swReloadPending, label: btn.textContent };
        })()""")
        assert out == {"role": "system", "before": [], "after": [{"type": "SKIP_WAITING"}], "pending": True, "label": "Reload"}

    def test_registration_watches_for_updates_and_listens_to_the_worker(self, app_js_nc):
        i = app_js_nc.index("navigator.serviceWorker.register('/sw.js')")
        body = app_js_nc[i:i + 400]
        assert "_watchServiceWorkerUpdates(reg)" in body
        assert "navigator.serviceWorker.addEventListener('message'" in app_js_nc
        assert "'controllerchange'" in app_js_nc and "if (!_swReloadPending) return;" in app_js_nc

    def test_offline_state_reaches_the_chip(self, app_js_nc):
        assert "window.addEventListener('offline', () => setConnectionState('error', 'OFFLINE'))" in app_js_nc
        assert "window.addEventListener('online'" in app_js_nc

    def test_sessions_realign_on_a_push_click(self, sessions_js):
        i = sessions_js.index("Core.events.addEventListener('push-click'")
        assert "resyncCurrent();" in sessions_js[i:i + 200] and "scheduleRefresh();" in sessions_js[i:i + 200]


# ═══════════════════════════════════════════════════════════════════════════
# 2. VAPID via cookie (read-only) — writes stay header-only; auth prune
# ═══════════════════════════════════════════════════════════════════════════

class TestVapidCookieAndPrune:
    def test_vapid_accepts_header_or_page_cookie_and_nothing_else(self):
        c = TestClient(server.app)
        assert c.get("/api/push/vapid", headers=AUTH).status_code == 200
        c.cookies.set(server._PAGE_COOKIE, server.GHOST_API_KEY)
        assert c.get("/api/push/vapid").status_code == 200
        c.cookies.set(server._PAGE_COOKIE, "wrong")
        assert c.get("/api/push/vapid").status_code == 401
        assert TestClient(server.app).get("/api/push/vapid").status_code == 401

    def test_the_cookie_still_authorises_no_write(self):
        c = TestClient(server.app)
        c.cookies.set(server._PAGE_COOKIE, server.GHOST_API_KEY)
        r = c.post("/api/push/subscribe", json={"subscription": {"endpoint": "https://x/y"}})
        assert r.status_code == 401, "the page cookie authorised a WRITE — CSRF surface reopened"
        assert c.post("/api/push/unsubscribe", json={"endpoint": "https://x/y"}).status_code == 401

    def test_every_route_but_vapid_keeps_the_header_dependency(self):
        src = (_ROOT / "interface" / "server.py").read_text()
        uses = re.findall(r'@app\.(?:get|post|delete)\("([^"]+)"[^\n]*verify_interface_key_or_page_cookie', src)
        assert uses == ["/api/push/vapid"], uses

    def _broadcast(self, monkeypatch, tmp_path, statuses):
        import json as _json
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        import base64
        priv = ec.generate_private_key(ec.SECP256R1())
        pub = priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        vapid = tmp_path / "vapid.json"
        vapid.write_text(_json.dumps({
            "public_key_b64url": base64.urlsafe_b64encode(pub).decode().rstrip("="),
            "private_key_pem": priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()).decode()}))
        subs = tmp_path / "subs.json"
        monkeypatch.setattr(webpush_notify, "_VAPID_FILE", vapid)
        monkeypatch.setattr(webpush_notify, "_SUBS_FILE", subs)
        monkeypatch.setattr(webpush_notify, "_vapid_cache", None)
        monkeypatch.setattr(webpush_notify, "_vapid_signer", None)
        for ep in statuses:
            webpush_notify.add_subscription({"endpoint": f"https://web.push.apple.com/{ep}",
                                             "keys": {"p256dh": "BP" + "A" * 85, "auth": "A" * 22}})

        class FakeWPE(Exception):
            def __init__(self, status):
                self.response = MagicMock(status_code=status)

        def fake_webpush(subscription_info, **kw):
            st = statuses[subscription_info["endpoint"].rsplit("/", 1)[1]]
            if st != 200:
                raise FakeWPE(st)
            return MagicMock()

        fake_mod = MagicMock(webpush=fake_webpush, WebPushException=FakeWPE)
        with patch.dict(sys.modules, {"pywebpush": fake_mod}):
            sent = webpush_notify.broadcast("t", "b", url="/x")
        left = sorted(e.rsplit("/", 1)[1] for e in webpush_notify._load_subs_or_empty())
        return sent, left

    def test_a_rejected_endpoint_is_pruned_when_another_accepted_our_key(self, monkeypatch, tmp_path):
        """FAILS pre-fix: 401/403 was only logged, so a rotated subscription
        stayed in the store and failed on every send forever."""
        sent, left = self._broadcast(monkeypatch, tmp_path, {"alive": 200, "rotated": 403, "gone": 410})
        assert sent == 1 and left == ["alive"]

    def test_all_rejected_means_OUR_config_is_wrong_so_nothing_is_pruned(self, monkeypatch, tmp_path):
        sent, left = self._broadcast(monkeypatch, tmp_path, {"a": 401, "b": 403})
        assert sent == 0 and left == ["a", "b"], "a broken VAPID config wiped the subscription store"


# ═══════════════════════════════════════════════════════════════════════════
# 3. The face pauses while hidden
# ═══════════════════════════════════════════════════════════════════════════

class TestFacePausesWhenHidden:
    def _run(self, graph_js, expr, renderer="{}"):
        fn = extract_js_function(graph_js, "setAnimationPaused").replace("export function", "function")
        return eval_js(f"""
let animationFrameId = 7, renderer = {renderer}, _animationPaused = false;
const cancelled = [], started = [];
globalThis.cancelAnimationFrame = (id) => cancelled.push(id);
function animate() {{ started.push(1); animationFrameId = 9; }}
""" + fn, expr)

    def test_pause_cancels_the_frame_and_resume_restarts_it(self, graph_js):
        out = self._run(graph_js, "[setAnimationPaused(true), animationFrameId, cancelled, setAnimationPaused(false), started.length, animationFrameId]")
        assert out == [False, None, [7], True, 1, 9]

    def test_resume_before_init_does_not_start_a_loop_without_a_renderer(self, graph_js):
        out = self._run(graph_js, "[setAnimationPaused(true), setAnimationPaused(false), started.length]", renderer="null")
        assert out == [False, False, 0]

    def test_resume_is_idempotent_while_running(self, graph_js):
        out = self._run(graph_js, "[setAnimationPaused(false), started.length]")
        assert out == [True, 0], "a running loop was started a second time (two frames per tick)"

    def test_the_loop_guards_itself_and_visibility_drives_it(self, graph_js):
        i = graph_js.index("function animate() {")
        assert graph_js[i:i + 120].count("if (_animationPaused) { animationFrameId = null; return; }") == 1
        assert "document.addEventListener('visibilitychange', () => setAnimationPaused(!!document.hidden))" in graph_js


# ═══════════════════════════════════════════════════════════════════════════
# 4. TLS expiry watch
# ═══════════════════════════════════════════════════════════════════════════

def _self_signed(tmp_path, days):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "t.example")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(1).not_valid_before(now - timedelta(days=30)).not_valid_after(now + timedelta(days=days))
            .sign(key, hashes.SHA256()))
    p = tmp_path / f"{days}.crt"
    p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return p


class TestTlsWatch:
    def test_cert_path_comes_from_uvicorns_argv_or_env(self):
        assert server._tls_cert_path(["uvicorn", "server:app", "--ssl-certfile", "eva.crt"], {}) == Path("eva.crt")
        assert server._tls_cert_path(["x", "--ssl-certfile=y.crt"], {}) == Path("y.crt")
        assert server._tls_cert_path(["x"], {"GHOST_TLS_CERTFILE": "/e/z.crt"}) == Path("/e/z.crt")
        assert server._tls_cert_path(["x"], {}) is None

    def test_status_reports_days_left_and_warns_under_the_threshold(self, tmp_path):
        ok = server.tls_cert_status(_self_signed(tmp_path, 40))
        soon = server.tls_cert_status(_self_signed(tmp_path, 5))
        assert 39 < ok["days_left"] <= 40 and ok["warn"] is False and ok["error"] is None
        assert 4 < soon["days_left"] <= 5 and soon["warn"] is True
        assert soon["not_after"].endswith("+00:00")

    def test_an_expired_cert_is_negative_days(self, tmp_path):
        st = server.tls_cert_status(_self_signed(tmp_path, -3))
        assert st["days_left"] < -2.9 and st["warn"] is True

    def test_missing_or_garbage_cert_reports_not_raises(self, tmp_path):
        assert server.tls_cert_status(tmp_path / "nope.crt")["error"]
        g = tmp_path / "g.crt"; g.write_text("not a cert")
        assert server.tls_cert_status(g)["error"]
        assert server.tls_cert_status(None)["error"] if server._tls_cert_path() is None else True

    def test_the_live_cert_if_any_is_readable(self):
        """Whatever the launcher configured must parse — a misnamed file would
        otherwise report `error` forever and the chip would stay calm."""
        p = server._tls_cert_path()
        if p is None:
            pytest.skip("no --ssl-certfile in this process's argv")
        st = server.tls_cert_status(p)
        assert st["error"] is None and st["days_left"] is not None

    def test_interface_health_route(self, tmp_path, monkeypatch):
        monkeypatch.setattr(server, "_tls_cert_path", lambda *a, **k: _self_signed(tmp_path, 3))
        r = TestClient(server.app).get("/api/interface/health", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        for k in ("ok", "uptime_s", "active_tasks", "buffered_bytes", "ws_clients", "sse_ping_s", "tls"):
            assert k in body, k
        assert body["tls"]["warn"] is True and 2 < body["tls"]["days_left"] <= 3
        assert TestClient(server.app).get("/api/interface/health").status_code == 401

    def test_the_watch_task_is_started(self):
        import inspect
        src = inspect.getsource(server._lifespan)
        assert "_tls_expiry_watch()" in src

    def test_chip_warning_text(self, status_js):
        fn = extract_js_function(status_js, "tlsWarning")
        out = eval_js("const TLS_WARN_DAYS = 14;\n" + fn, """[
            tlsWarning(null), tlsWarning({}), tlsWarning({tls: {error: 'no certificate configured'}}),
            tlsWarning({tls: {days_left: 40}}), tlsWarning({tls: {days_left: 9.7}}), tlsWarning({tls: {days_left: -1.2}})]""")
        assert out[:4] == [None, None, None, None]
        assert "expires in 9 day" in out[4] and "tailscale cert" in out[4]
        assert "EXPIRED 2 day" in out[5]

    def test_chip_goes_amber_on_tls_alone(self, status_js):
        harness = """
const HEALTH_POLL_MS = 25_000, TLS_WARN_DAYS = 14;
const indicator = { classes: new Set(), title: 'Live log', getAttribute: (n) => n === 'data-title' ? 'Live log' : null,
    classList: { toggle(c, on) { on ? indicator.classes.add(c) : indicator.classes.delete(c); } } };
globalThis.document = { getElementById: (id) => id === 'status-indicator' ? indicator : null, addEventListener() {}, visibilityState: 'visible' };
globalThis.window = {}; globalThis.setInterval = () => 0;
globalThis.fetch = async (url) => ({ ok: true, status: 200, json: async () => url.includes('interface')
    ? { ok: true, tls: { days_left: 6 } } : { memory_system_loaded: true, biological_watchdog_alive: true } });
"""
        out = eval_js(harness + extract_js_function(status_js, "initStatus")
                      + "\ninitStatus({});\nawait new Promise(r => setTimeout(r, 30));\n",
                      "({ degraded: indicator.classes.has('degraded'), title: indicator.title })")
        assert out["degraded"] is True
        assert out["title"].startswith("DEGRADED") and "TLS certificate expires in 6 day" in out["title"]
        assert "memory" not in out["title"].lower(), "a healthy agent was blamed alongside the certificate"


def test_touched_modules_bumped():
    index = (_STATIC / "index.html").read_text()
    app = (_STATIC / "app.js").read_text()
    ws = (_STATIC / "workspace.js").read_text()
    assert "app.js?v=12.1" in index
    assert "workspace.js?v=8.5" in app and "matrix_graph.js?v=12.1" in app
    assert "sessions.js?v=7.9" in ws and "status.js?v=7.4" in ws
