// Ghost service worker (rewritten 2026-09-11).
//
// Served by server.py's GET /sw.js, which fills the two placeholders below
// from the assets it actually ships: BUILD is a stamp over the shipped
// files' sizes/mtimes and the precache list, so any static change yields a
// new cache name and the old one is swept on activate; PRECACHE is the app
// shell — "/" plus every versioned module and stylesheet index.html and the
// import chain reference, plus the sanitizer vendors and the icons.
//
// Routing (fetch): GET, same-origin only.
//   "/"          network-first, falling back to the cached copy, then to a
//                small offline page. The cached document carries the page's
//                injected API key — the same secret the page cookie and the
//                installed app's start_url already persist on this device;
//                Cache Storage adds no NEW audience.
//   "/static/*"  cache-first (assets are immutable per ?v=; the build stamp
//                retires the whole cache when anything ships), filled on miss.
//   everything else (/api/*, /ws, /manifest, /sw.js) is NOT intercepted.
//
// Updates: install no longer calls skipWaiting() — that swapped the worker
// under a page mid-turn. The page shows "new version ready — Reload" and
// posts SKIP_WAITING when the operator says so.
const BUILD = '__GHOST_SW_BUILD__';
const PRECACHE = __GHOST_SW_PRECACHE__;
const CACHE = 'ghost-shell-' + BUILD;

const OFFLINE_HTML = '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    + '<meta name="viewport" content="width=device-width, initial-scale=1">'
    + '<title>Ghost — offline</title>'
    + '<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;'
    + 'background:#000;color:#e6e9ef;font:15px/1.5 -apple-system,Inter,system-ui,sans-serif;text-align:center;padding:24px}'
    + 'code{font-family:JetBrains Mono,monospace;color:#b894ff}</style></head>'
    + '<body><div><p><strong>Ghost is unreachable.</strong></p>'
    + '<p>You are offline, or the tailnet is not up yet. Nothing was sent.</p>'
    + '<p><a href="/" style="color:#b894ff">Try again</a></p></div></body></html>';

// Which handler owns a same-origin GET path. Pure; executed under node.
function routeFor(pathname) {
    if (pathname === '/') return 'shell';
    if (pathname.startsWith('/static/')) return 'static';
    return null;
}

// The client a notification click should land on: the focused window,
// else a visible one, else the first. Pure; executed under node.
function pickClient(clients) {
    if (!clients || !clients.length) return null;
    return clients.find((c) => c && c.focused)
        || clients.find((c) => c && c.visibilityState === 'visible')
        || clients[0];
}

// Absolute URL for a same-origin path. A bare `new Request('/')` resolves
// against the worker's scope in a browser but is invalid elsewhere (the
// node harness) — resolve explicitly and the behaviour is one thing.
function _abs(path) {
    return new URL(path, self.location.origin).href;
}

function _b64urlToUint8(base64url) {
    const pad = '='.repeat((4 - (base64url.length % 4)) % 4);
    const b64 = (base64url + pad).replace(/-/g, '+').replace(/_/g, '/');
    const raw = atob(b64);
    return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

self.addEventListener('install', (event) => {
    event.waitUntil((async () => {
        const cache = await caches.open(CACHE);
        // Tolerate a missing entry: one 404 must not veto the whole shell.
        await Promise.allSettled(PRECACHE.map((u) =>
            cache.add(new Request(_abs(u), { credentials: 'same-origin' }))));
    })());
});

self.addEventListener('activate', (event) => {
    event.waitUntil((async () => {
        const names = await caches.keys();
        await Promise.all(names
            .filter((n) => n.startsWith('ghost-shell-') && n !== CACHE)
            .map((n) => caches.delete(n)));
        await self.clients.claim();
    })());
});

self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting();
});

async function shellResponse(request) {
    try {
        const fresh = await fetch(request);
        if (fresh && fresh.ok) {
            // Keyed by the bare path so the ?key= and scrubbed variants share
            // one entry; only a 200 is worth keeping (never a 401 page).
            const cache = await caches.open(CACHE);
            cache.put(new Request(_abs('/')), fresh.clone());
        }
        return fresh;
    } catch (e) {
        const cache = await caches.open(CACHE);
        const hit = await cache.match(new Request(_abs('/')));
        if (hit) return hit;
        return new Response(OFFLINE_HTML, {
            status: 503,
            headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' },
        });
    }
}

async function staticResponse(request) {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(request);
    if (hit) return hit;
    try {
        const fresh = await fetch(request);
        if (fresh && fresh.status === 200) cache.put(request, fresh.clone());
        return fresh;
    } catch (e) {
        return new Response('', { status: 504, statusText: 'offline' });
    }
}

self.addEventListener('fetch', (event) => {
    const request = event.request;
    if (request.method !== 'GET') return;
    let url;
    try { url = new URL(request.url); } catch (e) { return; }
    if (url.origin !== self.location.origin) return;
    const route = routeFor(url.pathname);
    if (route === 'shell') event.respondWith(shellResponse(request));
    else if (route === 'static') event.respondWith(staticResponse(request));
});

// Web push (2026-08-01). Payload is JSON from webpush_notify.broadcast:
// {title, body, url, tag}. iOS REQUIRES a visible notification per push
// (silent pushes revoke the subscription) — always show one, even for an
// unparseable payload.
self.addEventListener('push', (event) => {
    let data = {};
    try { data = event.data ? event.data.json() : {}; } catch (e) { /* show fallback */ }
    const title = data.title || 'Ghost';
    event.waitUntil(self.registration.showNotification(title, {
        body: data.body || 'Ghost has news.',
        tag: data.tag || 'ghost',
        icon: '/static/icons/icon-192.png?v=2',
        badge: '/static/icons/icon-192.png?v=2',
        data: { url: data.url || '/' },
    }));
});

// The browser rotated (or dropped and re-issued) this device's push
// subscription. Before 2026-09-11 nothing handled it: the server kept the
// dead endpoint, every push failed, and the only repair was opening the
// app. Re-subscribe here with the same VAPID key, then tell open pages —
// registering the NEW endpoint needs the API key only the page holds
// (ensurePushSubscription POSTs it on every boot too).
self.addEventListener('pushsubscriptionchange', (event) => {
    event.waitUntil((async () => {
        let key = event.oldSubscription && event.oldSubscription.options
            && event.oldSubscription.options.applicationServerKey;
        if (!key) {
            try {
                const r = await fetch('/api/push/vapid', { credentials: 'same-origin' });
                if (r.ok) {
                    const d = await r.json();
                    if (d && d.enabled && d.key) key = _b64urlToUint8(d.key);
                }
            } catch (e) { /* offline — the page re-subscribes on its next boot */ }
        }
        if (!key) return;
        try {
            await self.registration.pushManager.subscribe(
                { userVisibleOnly: true, applicationServerKey: key });
        } catch (e) { return; }
        const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
        clients.forEach((c) => c.postMessage({ type: 'push-resubscribed' }));
    })());
});

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    const url = (event.notification.data && event.notification.data.url) || null;
    const tag = event.notification.tag || null;
    event.waitUntil((async () => {
        const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
        const target = pickClient(clients);
        if (target) {
            // Focus alone fires no visibilitychange on an already-visible
            // window, so the page's resume/resync chain never ran — say it.
            try { await target.focus(); } catch (e) { /* not focusable */ }
            target.postMessage({ type: 'push-click', url, tag });
            return;
        }
        // No window: open the keyed URL from the notification data; a
        // payload without one would only reach the 401 page.
        if (url) await self.clients.openWindow(url);
    })());
});
