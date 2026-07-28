// ═══════════════════════════════════════════════════════════════
//  Notifications bell (2026-07-28)
//
//  Surfaces the agent's notify-severity autonomous-activity records —
//  the "actionable events" tier — via the watermark contract:
//    GET  /api/notifications/pending?consumer=web-ui  → {records, watermark}
//    POST /api/notifications/ack {consumer, watermark}
//  First contact returns baseline:true with an end-of-ledger watermark
//  (no historical replay); we ack it silently. Records are only marked
//  delivered once acked, so a dropped poll re-serves them — dedupe by
//  (ts, summary). The consumer name is OURS ('web-ui'): watermarks are
//  per-consumer, so this never steals records from the Slack bot.
// ═══════════════════════════════════════════════════════════════

const CONSUMER = 'web-ui';
const POLL_MS = 30_000;
const KEEP = 100;

export function initNotifications(ctx) {
    const { Core, el, toast, relTime } = ctx;

    const btn = document.getElementById('notif-btn');
    const badge = document.getElementById('notif-badge');
    const panel = document.getElementById('notif-panel');
    const listEl = document.getElementById('notif-list');

    let records = [];      // newest first
    let unread = 0;
    let enabled = null;
    let seen = new Set();  // `${ts}|${summary}` dedupe across re-served polls
    let timer = null;

    function updateBadge() {
        if (!badge) return;
        badge.textContent = unread > 99 ? '99+' : String(unread);
        badge.classList.toggle('hidden', unread === 0);
    }

    function renderList() {
        if (!listEl) return;
        listEl.replaceChildren();
        if (enabled === false) {
            listEl.appendChild(el('div', 'notif-empty',
                'The agent’s activity ledger is disabled.'));
            return;
        }
        if (!records.length) {
            listEl.appendChild(el('div', 'notif-empty',
                'Nothing yet. Actionable agent events land here.'));
            return;
        }
        for (const r of records) {
            const item = el('div', 'notif-item');
            const head = el('div', 'notif-item-head');
            head.appendChild(el('span', 'notif-phase', String(r.phase || 'event').toUpperCase()));
            head.appendChild(el('span', 'notif-time', relTime(r.ts)));
            item.appendChild(head);
            item.appendChild(el('div', 'notif-summary', String(r.summary || '')));
            listEl.appendChild(item);
        }
    }

    async function ack(watermark) {
        try {
            await fetch('/api/notifications/ack', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ consumer: CONSUMER, watermark }),
            });
        } catch (e) { /* re-served next poll — by design */ }
    }

    async function poll() {
        try {
            const res = await fetch(
                `/api/notifications/pending?consumer=${encodeURIComponent(CONSUMER)}&limit=50`);
            if (!res.ok) return;
            const data = await res.json();
            enabled = !!data.enabled;
            if (!enabled) { renderList(); return; }
            if (data.baseline) { await ack(data.watermark); return; }
            const fresh = (data.records || []).filter(r => {
                const key = `${r.ts}|${r.summary}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
            if (fresh.length) {
                records = [...fresh.slice().reverse(), ...records].slice(0, KEEP);
                // Records arriving while the panel is OPEN are being seen —
                // don't count them as unread.
                if (panel && panel.classList.contains('hidden')) {
                    unread += fresh.length;
                }
                updateBadge();
                renderList();
                const first = fresh[fresh.length - 1];
                toast(`${String(first.phase || 'agent').toUpperCase()} — ${first.summary}`,
                    'notify', openPanel);
                if (fresh.length > 1) {
                    toast(`+${fresh.length - 1} more agent notifications`, 'notify', openPanel);
                }
            }
            if (seen.size > 500) seen = new Set([...seen].slice(-250));
            // Ack AFTER rendering: a crash before this line re-serves the
            // records on the next poll instead of dropping them.
            await ack(data.watermark);
        } catch (e) { /* agent down — next poll retries */ }
    }

    function openPanel() {
        if (!panel) return;
        renderList();
        panel.classList.remove('hidden');
        unread = 0;
        updateBadge();
    }
    function closePanel() { panel?.classList.add('hidden'); }
    function togglePanel() {
        if (panel?.classList.contains('hidden')) openPanel();
        else closePanel();
    }

    btn?.addEventListener('click', togglePanel);
    document.getElementById('notif-close')?.addEventListener('click', closePanel);
    document.addEventListener('click', (e) => {
        if (panel && !panel.classList.contains('hidden')
            && !panel.contains(e.target) && !btn.contains(e.target)) {
            closePanel();
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && panel && !panel.classList.contains('hidden')) closePanel();
    });

    // iOS Safari suspends timers in background tabs — poll immediately on
    // return so the badge is fresh when the operator looks.
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') poll();
    });

    poll();
    timer = setInterval(poll, POLL_MS);

    return { poll, openPanel, closePanel, togglePanel, count: () => records.length };
}
