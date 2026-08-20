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
    const footerEl = document.getElementById('rail-footer');
    const deleteAllBtn = document.getElementById('delete-all-sessions-btn');

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
        // Every identity change invalidates in-flight loads/resyncs (R5:
        // newChat/delete/clear minted a new id WITHOUT bumping the token,
        // so a resync of the old session could resolve late and paint —
        // then persist — the old conversation under the new id).
        loadSeq++;
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
        // The destructive footer only exists when there is something to
        // destroy (and sessions are enabled at all).
        if (footerEl) {
            footerEl.classList.toggle('hidden', !(enabled && sessions.length > 0));
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
        if (!res.ok) {
            // Carry the status ON the error. Callers need to tell "the key
            // was rejected" from "nothing is listening" (review R1 M9), and
            // `res` is not in their scope — reading it there is a
            // ReferenceError raised inside the handler, which would take out
            // the rail and boot()'s retry with it.
            const err = new Error(`HTTP ${res.status}`);
            err.status = res.status;
            throw err;
        }
        return res.json();
    }

    // 401/403 = the agent is answering and refusing this key (the normal
    // state after a rotation); anything else = it isn't answering. Reporting
    // both as "unreachable" sent the operator to restart a healthy process.
    function _railFailureNote(e) {
        const st = e && e.status;
        return (st === 401 || st === 403)
            ? 'Not authorised — the agent rejected this key (probably '
              + 'rotated). Reload with the current ?key=.'
            : 'Agent unreachable — session list unavailable.';
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
            note(_railFailureNote(e));
        }
    }

    // Monotonic token so overlapping loads (boot's fetch racing a rail
    // click) can't paint the loser's history under the winner's active
    // row (R3 review).
    let loadSeq = 0;

    async function load(id) {
        const seq = ++loadSeq;
        const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        // Superseded by a newer load: return null so the CALLER stands
        // down too (R4: returning data let switchTo's setCurrent bind the
        // loser's id under the winner's painted history — the next turn
        // then replayed conversation B into session A's durable store).
        if (seq !== loadSeq) return null;
        // A turn started during the two awaits above does NOT bump loadSeq,
        // and this function paints unconditionally: the streaming bubble was
        // detached, the user's message dropped, and `setCurrent(id)` rebound
        // the identity so the reply landed in the OTHER session's durable
        // store. Its three siblings (switchTo, boot, resyncCurrent) all
        // re-check processing; this one never did (R2 lens B).
        if (Core.isProcessing()) {
            // Standing down is right, but standing down SILENTLY is not: the
            // rail row the operator clicked would simply never open.
            toast('A turn started — finish it, then reopen this session', 'error');
            return null;
        }
        let messages = Array.isArray(data.messages) ? data.messages : [];
        // Preserve client-only 👍/👎 label keys — ONLY when re-loading the
        // session that is already current (boot/refresh). Merging on a
        // SWITCH would index-align against the OLD conversation, and any
        // short templated reply ("Done.") at the same index would graft
        // the old session's request id onto the new one — a 👍 there
        // writes an operator-authoritative label onto a trajectory the
        // operator never read (R2 review CRIT).
        if (id === currentId) {
            // RE-loading the session already on screen (boot, refresh) — not
            // a switch. The local copy is the same conversation, so it may
            // hold a turn the agent never persisted, and replacing wholesale
            // deletes the user's own message. That is the F2 defect exactly;
            // the fix landed in `resyncCurrent` and this sibling path kept
            // the old behaviour, so the loss simply moved from "2s later" to
            // "on the next reload" (R3 lens B).
            messages = _reconcileWithLocal(messages, Core.getChatHistory());
            if (Core.mergeClientLabelKeys) {
                messages = Core.mergeClientLabelKeys(messages);
            }
        }
        // Commit identity ATOMICALLY with the paint (R6): the old
        // paint-here/bind-in-caller split left a microtask window where a
        // concurrent resync saw the pre-switch id with an unchanged token
        // and repainted the old session over the new one — then the
        // caller bound the new id to it. The merge gate above must keep
        // reading the OLD currentId, so this runs after it.
        setCurrent(id);
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
            // load() commits the identity itself, atomically with the
            // paint (R6); a superseded load returns null having bound
            // nothing.
            await load(id);
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
                // RE-CHECK after the await. The entry gate above ran before
                // window.confirm() and the DELETE round trip — a turn started
                // in between, and clearing here detaches its streaming bubble
                // and orphans the reply into an emptied history, which the
                // agent then persists under a session it just deleted (R2
                // lens B). The row is gone either way; the conversation on
                // screen stays until the turn ends.
                if (Core.isProcessing()) {
                    render();
                    toast('Session deleted — the running turn keeps this view',
                          'error');
                    return;
                }
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

    // ── Delete ALL sessions (2026-07-29) ───────────────────────────
    // Two-step armed confirm instead of window.confirm: the first click
    // arms the button (it turns danger-red and says so), the second
    // click within 4s executes; anything else — timeout, pointer leaving
    // the rail — disarms. A stray click can never wipe the strata, and
    // no browser dialog breaks the UI's flow. Deletion iterates the
    // enumerated per-id proxy (DELETE /api/sessions/{id}) — deliberately
    // no bulk endpoint, the LAN-reachable surface stays as it is.
    let deleteAllArmTimer = null;

    function disarmDeleteAll() {
        clearTimeout(deleteAllArmTimer);
        deleteAllArmTimer = null;
        if (deleteAllBtn) {
            deleteAllBtn.classList.remove('armed');
            deleteAllBtn.textContent = 'Delete all sessions';
        }
    }

    async function deleteAllSessions() {
        disarmDeleteAll();
        if (Core.isProcessing()) {
            toast('Wait for the current turn to finish first', 'error');
            return;
        }
        const targets = sessions.slice();
        if (!targets.length) return;
        if (deleteAllBtn) deleteAllBtn.disabled = true;
        let failed = 0;
        for (const s of targets) {
            try {
                const res = await fetch(`/api/sessions/${encodeURIComponent(s.id)}`,
                    { method: 'DELETE' });
                if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
                sessions = sessions.filter(x => x.id !== s.id);
            } catch (e) {
                failed++;
            }
        }
        if (deleteAllBtn) deleteAllBtn.disabled = false;
        // The active conversation was among the deleted (or never
        // persisted) — start clean either way, same as deleting the
        // current session individually.
        //
        // ⚠ Only the button was disabled during the loop; the COMPOSER
        // stayed live through N sequential DELETEs, so a turn started
        // mid-loop had its bubble detached here and its reply written into
        // an empty history — recreating the session that was just deleted
        // (R2 lens B). Same re-check as deleteSession.
        if (Core.isProcessing()) {
            render();
            toast('Sessions deleted — the running turn keeps this view',
                  'error');
            return;
        }
        Core.clearConversation();
        setCurrent(mintId());
        toast(failed
            ? `Deleted ${targets.length - failed} session(s), ${failed} failed`
            : `Deleted ${targets.length} session(s)`,
            failed ? 'error' : undefined);
    }

    deleteAllBtn?.addEventListener('click', () => {
        if (deleteAllBtn.classList.contains('armed')) {
            deleteAllSessions();
            return;
        }
        deleteAllBtn.classList.add('armed');
        deleteAllBtn.textContent = `Really delete all ${sessions.length}? Click again`;
        clearTimeout(deleteAllArmTimer);
        deleteAllArmTimer = setTimeout(disarmDeleteAll, 4000);
    });
    // Leaving the rail disarms — walking away is a "no". MOUSE ONLY:
    // touch pointers are transient, so EVERY tap ends in a pointerleave —
    // which disarmed the button between tap 1 and tap 2, making the
    // second tap re-arm instead of execute (mobile delete-all was
    // impossible, 2026-08-01). Touch relies on the 4s timeout instead.
    document.getElementById('session-rail')
        ?.addEventListener('pointerleave', (ev) => {
            if (ev.pointerType === 'mouse') disarmDeleteAll();
        });

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
            note(_railFailureNote(e));
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
    // Adopt the server copy WITHOUT deleting messages the server has not
    // got yet.
    //
    // This path used to replace local history unconditionally whenever it
    // "drifted" — including when the server had FEWER messages. The sibling
    // path in app.js requires `msgs.length > chatHistory.length` and is
    // pinned with exactly that reasoning ("silent local truncation"), but
    // this one had no such gate, so any turn the agent failed to persist —
    // an error frame, a Stop that killed the relay before the write, a
    // workspace load — had the user's own typed message DELETED from their
    // screen ~2s later by the resync that exists to help them (R2 lens B).
    //
    // Stable message key. Mirrors the server's `_msg_key`: the ROLE plus a
    // stringified content, so a multimodal payload compares by value rather
    // than by object identity. Key ORDER matters to JSON.stringify, so
    // objects are serialised with sorted keys — two identical multimodal
    // parts built in different key order are the same message (R3 lens B).
    function _msgKey(m) {
        const c = m && m.content;
        let text;
        if (c === null || c === undefined) text = '';
        else if (typeof c === 'string') text = c;
        else text = JSON.stringify(c, (_k, v) =>
            (v && typeof v === 'object' && !Array.isArray(v))
                ? Object.keys(v).sort().reduce((o, kk) => (o[kk] = v[kk], o), {})
                : v);
        return (m && m.role) + '\u0000' + text;
    }

    // Adopt the server copy WITHOUT deleting or REORDERING anything.
    //
    // ⚠ This is the second version. The first took every local user message
    // the server did not account for and `concat`ed them onto the end — an
    // order-blind multiset. Its own comment claimed it "mirrors the server's
    // own merge", and it did not: the server LCS-aligns and appends only the
    // unmatched TAIL, order-preserving. Two consequences, both reproduced
    // (R3 lens B): an abandoned question landed AFTER the reply to a later
    // question, and — because the store truncates from the FRONT at 400
    // while the interface caps at 500 — every message in that window looked
    // "unaccounted" and got pasted onto the tail, where the next turn
    // replayed it and the server appended it for good. The fix inherited the
    // defect class it was written to remove, one layer along.
    //
    // Now: align local against the server with the same LCS the server uses,
    // and re-append only what follows the last aligned position. USER
    // messages only — a local assistant entry the server lacks is an
    // aborted-stream stub, and dropping it is the one thing the original
    // unconditional adopt got right.
    // Alignment PAIRS, the same LCS traceback the server runs.
    function _lcsPairs(serverMsgs, localMsgs) {
        const n = serverMsgs.length, m = localMsgs.length;
        if (!n || !m) return [];
        const sk = serverMsgs.map(_msgKey), lk = localMsgs.map(_msgKey);
        const table = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
        for (let i = n - 1; i >= 0; i--) {
            for (let j = m - 1; j >= 0; j--) {
                table[i][j] = sk[i] === lk[j]
                    ? table[i + 1][j + 1] + 1
                    : Math.max(table[i + 1][j], table[i][j + 1]);
            }
        }
        const pairs = [];
        let i = 0, j = 0;
        while (i < n && j < m) {
            if (sk[i] === lk[j]) { pairs.push([i, j]); i++; j++; }
            else if (table[i + 1][j] >= table[i][j + 1]) i++;
            else j++;
        }
        return pairs;
    }

    function _reconcileWithLocal(serverMsgs, localMsgs) {
        const raw = localMsgs || [];
        if (!raw.length) return serverMsgs;
        const local = raw.map(
            (m) => (m && Core.toWireMessage) ? Core.toWireMessage(m) : m);
        const pairs = _lcsPairs(serverMsgs, local);
        const out = [];
        let i = 0, j = 0;
        const firstAligned = pairs.length ? pairs[0][1] : local.length;
        const takeLocal = (k) => {
            const m = raw[k];
            if (!m) return;
            // BEFORE the first aligned position sits the client's copy of
            // history the store's 400-message cap EVICTED. Keep it whole —
            // it is the operator's own record, and the server puts anything
            // ahead of the alignment into `lead` (prompt only), so it can
            // never be re-persisted from there. AFTER the first alignment,
            // a local-only entry is a turn the agent never stored: keep the
            // USER message (that is the F2 fix) and drop assistant entries,
            // which are aborted-stream stubs.
            if (k < firstAligned || m.role === 'user') out.push(m);
        };
        for (const [pi, pj] of pairs) {
            while (i < pi) out.push(serverMsgs[i++]);   // server-only
            while (j < pj) takeLocal(j++);              // local-only, in place
            out.push(serverMsgs[pi]);
            i = pi + 1;
            j = pj + 1;
        }
        while (i < serverMsgs.length) out.push(serverMsgs[i++]);
        while (j < local.length) takeLocal(j++);
        return out;
    }

    async function resyncCurrent() {
        if (!enabled || !currentId || Core.isProcessing()) return;
        // Participate in the load token (R4) and re-check IDENTITY and
        // PROCESSING after every await (R5): a resync resolving after a
        // rail click, a New-chat/delete/clear (which mint an id without a
        // load), or a just-started turn must not repaint — painting
        // mid-turn detaches the streaming bubble, and painting the old
        // session under a fresh id persists cross-session contamination.
        const seq = loadSeq;
        const id = currentId;
        try {
            const res = await fetch(`/api/sessions/${encodeURIComponent(currentId)}`);
            if (!res.ok) return;   // not persisted yet (or deleted) — nothing to align to
            if (seq !== loadSeq || id !== currentId) return;
            const data = await res.json();
            const messages = Array.isArray(data.messages) ? data.messages : [];
            const local = Core.getChatHistory();
            // Compare on the WIRE shape: local entries carry client-only
            // label keys (reqId/feedback) the server never returns, so a
            // fat-vs-clean JSON compare read "drifted" on EVERY focus after
            // any completed turn — wiping the 👍/👎 feature and re-rendering
            // the log each time the tab came back (R1 review CRIT).
            const wire = (m) => (m && Core.toWireMessage)
                ? Core.toWireMessage(m) : (m || null);
            // ⚠ Compare against the RECONCILED array, not the raw server
            // copy. Once one local turn is legitimately unaccounted (the F2
            // case), local is permanently `server + k` — so a length compare
            // against the server copy read "drifted" on EVERY focus, and the
            // resync rebuilt the whole log (innerHTML='', scroll to bottom)
            // each time, destroying scroll position and selection forever
            // (R3 lens B). Reconciling is cheap and idempotent; rendering is
            // not.
            const adopted = _reconcileWithLocal(messages, local);
            const drifted = adopted.length !== local.length
                || JSON.stringify(adopted[adopted.length - 1] || null)
                    !== JSON.stringify(wire(local[local.length - 1]));
            if (drifted && messages.length && seq === loadSeq
                    && id === currentId && !Core.isProcessing()) {
                const merged = Core.mergeClientLabelKeys
                    ? Core.mergeClientLabelKeys(adopted) : adopted;
                Core.setChatHistory(merged);
                Core.renderHistoryToLog(merged);
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
            // Through setCurrent (R6): the direct writes skipped the token
            // bump AND the rail repaint, leaving a stale .active row.
            setCurrent(null);
        }
    });
    Core.events.addEventListener('turn-aborted', () => {
        // ⚠ RETRY, don't fire once. `resyncCurrent` bails immediately when a
        // turn is in flight (`Core.isProcessing()`), so an operator who hits
        // Stop and retypes within 2s skipped realignment entirely — the tab
        // stayed diverged until it happened to be hidden and shown again.
        // That was the client-side half of the doubling defect: the server's
        // alignment is now robust enough that this is belt rather than brace,
        // but a mitigation that silently no-ops on the fastest path is worth
        // fixing on its own terms.
        let _tries = 0;
        const _attempt = () => {
            if (!enabled || !currentId) return;
            if (Core.isProcessing()) {
                if (_tries++ < 10) setTimeout(_attempt, 2000);
                return;                     // ...wait out the new turn
            }
            resyncCurrent();
        };
        setTimeout(_attempt, 2000);
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
