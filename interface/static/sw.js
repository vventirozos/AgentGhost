self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim());
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

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    // An existing window is focused WITHOUT navigation (it re-syncs itself
    // on visibilitychange — resume/reconcile/bell all hang off it). With no
    // window, open the keyed URL from the notification data; a payload
    // without one would only reach the 401 page, so in that case do
    // nothing rather than open a dead tab.
    const url = (event.notification.data && event.notification.data.url) || null;
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then(windowClients => {
            if (windowClients.length > 0) {
                return windowClients[0].focus();
            }
            if (url) return clients.openWindow(url);
            return undefined;
        })
    );
});
