// ═══════════════════════════════════════════════════════════════
//  Sessions rail (2026-07-28)
//
//  Durable server-side conversations via the interface proxies:
//    GET    /api/sessions          → {enabled, sessions: [summaries]}
//    GET    /api/sessions/{id}     → full conversation (messages)
//    DELETE /api/sessions/{id}
//  The session itself is created lazily by the AGENT on the first
//  /api/chat that carries a session_id, so "new chat" just mints a
//  fresh id client-side. app.js attaches window.__ghostSessionId to
//  every chat payload; the server merges history so a fat client can
//  never double it.
// ═══════════════════════════════════════════════════════════════

export function initSessions(ctx) {
    const { Core, el, toast, jewelHue, relTime } = ctx;

    const listEl = document.getElementById('session-list');
    const noteEl = document.getElementById('rail-note');
    const searchEl = document.getElementById('session-search');
    const newBtn = document.getElementById('new-chat-btn');

    let enabled = null;          // null = unknown yet
    let sessions = [];           // summaries, most recent first
    let currentId = Core.safeStorage.get('ghost_session_id') || null;
    let filterText = '';
    let refreshTimer = null;

    function mintId() {
        try { return crypto.randomUUID(); }
        catch (e) {
            return 'web-' + Date.now().toString(36) + '-'
                + Math.random().toString(36).slice(2, 10);
        }
    }

    function setCurrent(id) {
        currentId = id;
        window.__ghostSessionId = id;
        if (id) Core.safeStorage.set('ghost_session_id', id);
        else Core.safeStorage.remove('ghost_session_id');
        render();
    }

    function note(text) {
        if (!noteEl) return;
        noteEl.textContent = text || '';
        noteEl.classList.toggle('hidden', !text);
    }

    function render() {
        if (!listEl) return;
        listEl.replaceChildren();
        const shown = sessions.filter(s =>
            !filterText
            || String(s.title || '').toLowerCase().includes(filterText)
            || String(s.id || '').startsWith(filterText));
        if (currentId && !sessions.some(s => s.id === currentId)) {
            // The active conversation hasn't been persisted yet (no turn
            // sent) — show it as a pending row so the rail always reflects
            // where the operator is.
            const row = buildRow({ id: currentId, title: 'New session', pending: true });
            listEl.appendChild(row);
        }
        for (const s of shown) listEl.appendChild(buildRow(s));
        if (!shown.length && !currentId) {
            listEl.appendChild(el('div', 'session-empty',
                'No sessions yet. Start a conversation and it lands here.'));
        }
    }

    function buildRow(s) {
        const row = el('div', 'session-row' + (s.id === currentId ? ' active' : ''));
        row.setAttribute('role', 'listitem');
        const stripe = el('span', 'session-stripe');
        stripe.style.background = jewelHue(s.id);
        row.appendChild(stripe);

        const main = el('button', 'session-main');
        main.type = 'button';
        main.appendChild(el('span', 'session-title',
            s.title || (s.pending ? 'New session' : 'untitled')));
        const meta = s.pending
            ? 'not saved yet'
            : [relTime(s.updated_at), s.message_count != null ? `${s.message_count} msgs` : '']
                .filter(Boolean).join(' · ');
        main.appendChild(el('span', 'session-meta', meta));
        main.addEventListener('click', () => switchTo(s.id));
        row.appendChild(main);

        if (!s.pending) {
            const del = el('button', 'session-delete', '✕');
            del.type = 'button';
            del.title = 'Delete session';
            del.setAttribute('aria-label', `Delete session ${s.title || s.id}`);
            del.addEventListener('click', (ev) => {
                ev.stopPropagation();
                deleteSession(s);
            });
            row.appendChild(del);
        }
        return row;
    }

    async function fetchList() {
        const res = await fetch('/api/sessions');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    }

    async function refresh() {
        if (enabled === false) return;
        try {
            const data = await fetchList();
            enabled = !!data.enabled;
            if (!enabled) {
                note('Sessions are disabled on the agent — conversations stay in this browser only.');
                return;
            }
            note('');
            sessions = data.sessions || [];
            render();
        } catch (e) {
            note('Agent unreachable — session list unavailable.');
        }
    }

    async function load(id) {
        const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const messages = Array.isArray(data.messages) ? data.messages : [];
        Core.setChatHistory(messages);
        Core.renderHistoryToLog(messages);
        if (!messages.length) Core.renderEmptyStateHero();
        return data;
    }

    async function switchTo(id) {
        if (id === currentId) return;
        if (Core.isProcessing()) {
            toast('Wait for the current turn to finish first', 'error');
            return;
        }
        try {
            Core.stopTTS();
            await load(id);
            setCurrent(id);
        } catch (e) {
            toast(`Could not open session: ${e.message}`, 'error');
        }
    }

    function newChat() {
        if (Core.isProcessing()) {
            toast('Wait for the current turn to finish first', 'error');
            return;
        }
        Core.stopTTS();
        Core.clearConversation();
        setCurrent(mintId());
    }

    async function deleteSession(s) {
        // Same gate as switchTo/newChat: deleting the ACTIVE session
        // mid-turn would wipe the streaming bubble, and the in-flight
        // turn's append would silently recreate the session server-side.
        if (Core.isProcessing()) {
            toast('Wait for the current turn to finish first', 'error');
            return;
        }
        const title = s.title || 'untitled session';
        if (!window.confirm(`Delete "${title}"? The stored conversation is removed from the agent.`)) return;
        try {
            const res = await fetch(`/api/sessions/${encodeURIComponent(s.id)}`, { method: 'DELETE' });
            if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
            sessions = sessions.filter(x => x.id !== s.id);
            if (s.id === currentId) {
                Core.clearConversation();
                setCurrent(mintId());
            } else {
                render();
            }
            toast('Session deleted');
        } catch (e) {
            toast(`Delete failed: ${e.message}`, 'error');
        }
    }

    function scheduleRefresh() {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(refresh, 1200);
    }

    async function boot() {
        try {
            const data = await fetchList();
            enabled = !!data.enabled;
        } catch (e) {
            enabled = null;   // unknown — retry on next refresh
            note('Agent unreachable — session list unavailable.');
            setTimeout(boot, 30_000);
            return;
        }
        if (!enabled) {
            note('Sessions are disabled on the agent — conversations stay in this browser only.');
            return;
        }
        await refresh();
        if (currentId && sessions.some(s => s.id === currentId)) {
            // Server is the source of truth for a known session: replace
            // the localStorage restore app.js already painted. NEVER
            // mid-turn — wiping the log detaches a streaming bubble and
            // orphans the reply into a replaced history.
            if (!Core.isProcessing()) {
                try { await load(currentId); } catch (e) { /* keep local paint */ }
            }
            setCurrent(currentId);
        } else {
            // No durable session yet — adopt the local conversation under a
            // fresh id; it is persisted server-side on the next turn.
            setCurrent(currentId || mintId());
        }
    }

    // Staleness re-sync: with one session id shared across tabs (and the
    // agent's merge treating a SHORTER replay as a thin-client append), a
    // tab that fell behind must realign from the server before it sends —
    // otherwise its next fat replay concatenates and permanently doubles
    // the session. Re-sync whenever the tab becomes visible, and shortly
    // after an aborted turn (the server may have persisted the full reply
    // while this client kept a "*[Aborted]*" stub).
    async function resyncCurrent() {
        if (!enabled || !currentId || Core.isProcessing()) return;
        try {
            const res = await fetch(`/api/sessions/${encodeURIComponent(currentId)}`);
            if (!res.ok) return;   // not persisted yet (or deleted) — nothing to align to
            const data = await res.json();
            const messages = Array.isArray(data.messages) ? data.messages : [];
            const local = Core.getChatHistory();
            const drifted = messages.length !== local.length
                || JSON.stringify(messages[messages.length - 1] || null)
                    !== JSON.stringify(local[local.length - 1] || null);
            if (drifted && messages.length) {
                Core.setChatHistory(messages);
                Core.renderHistoryToLog(messages);
            }
        } catch (e) { /* offline — next visibility change retries */ }
    }

    newBtn?.addEventListener('click', newChat);
    searchEl?.addEventListener('input', () => {
        filterText = searchEl.value.trim().toLowerCase();
        render();
    });
    // After each turn the agent may have created the session / derived a
    // title — refresh the rail (debounced; also picks up other clients).
    Core.events.addEventListener('turn-complete', scheduleRefresh);
    Core.events.addEventListener('conversation-cleared', () => {
        if (enabled) {
            setCurrent(mintId());
        } else if (enabled === null) {
            // Sessions state unknown (agent unreachable at boot): unbind
            // entirely so a cleared conversation can never merge into the
            // OLD session if the agent comes back mid-conversation.
            currentId = null;
            window.__ghostSessionId = null;
            Core.safeStorage.remove('ghost_session_id');
        }
    });
    Core.events.addEventListener('turn-aborted', () => {
        setTimeout(resyncCurrent, 2000);
    });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') resyncCurrent();
    });

    boot();

    return {
        newChat,
        switchTo,
        refresh,
        isEnabled: () => enabled === true,
        current: () => currentId,
        list: () => sessions.slice(),
    };
}
