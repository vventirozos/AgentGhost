// ═══════════════════════════════════════════════════════════════
//  Agent status strip + panel (2026-07-28)
//
//  Polls GET /api/health for the header strip (model pill + degraded
//  flag) and, while the panel is open, GET /api/turns for the live
//  turn queue. Exposes the REAL turn cancel (POST /api/turn/cancel —
//  releases the agent's global turn lock) which the send button's
//  abort never did: that only stopped the proxy-side stream.
//
//  Two silent-failure detectors surfaced on purpose (see api_health):
//  memory_system_loaded=false means a degraded boot disabled the
//  biological phases while HTTP keeps answering;
//  biological_watchdog_alive=false means the self-improvement daemon
//  died. Both show as a red DEGRADED tag the operator can't miss.
// ═══════════════════════════════════════════════════════════════

const HEALTH_POLL_MS = 25_000;
const TURNS_POLL_MS = 5_000;

export function initStatus(ctx) {
    const { Core, el, toast, relTime } = ctx;

    const indicator = document.getElementById('status-indicator');
    const modelPill = document.getElementById('model-pill');
    const panel = document.getElementById('status-panel');
    const bodyEl = document.getElementById('status-body');

    let health = null;
    let healthOk = null;
    let healthFailNote = '';
    let turnsTimer = null;

    function pillText(h) {
        const cfg = (h && h.config) || {};
        // Live shape (2026-07-28): resolved args flattened as "arg.model".
        const model = cfg['arg.model'] || cfg.model || cfg.main_model
            || (cfg.args && cfg.args.model) || '';
        return String(model).split('/').pop().slice(0, 26);
    }

    // 401/403 is an AUTH failure, not an unreachable agent: after a key
    // rotation the injected key is stale and every call fails, while the
    // panel blamed the agent process ("Agent unreachable on :8000") — wrong
    // diagnosis, wrong fix attempted (review R1 M9).
    //
    // A PURE function on purpose: as a boolean computed inline in the catch,
    // a `false && …` mutation survived every text assertion the suite could
    // write. tests/test_webui_console_review.py EXECUTES this under node.
    function _healthFailureNote(e) {
        const msg = String((e && e.message) || e || '');
        if (/\b(401|403)\b/.test(msg)) {
            return 'NOT AUTHORISED (' + (/\b401\b/.test(msg) ? '401' : '403')
                + '). The agent is answering but rejecting this key — it was '
                + 'probably rotated. Reload with the current ?key=, or '
                + 're-open the PWA from a fresh link.';
        }
        return 'Agent unreachable on :8000. The interface is up; the agent '
            + 'process is not answering.';
    }

    function isDegraded(h) {
        // `h === null` means the health poll FAILED — the most abnormal state
        // there is — and `null && …` is falsy, so the header strip showed no
        // abnormality at all with the agent down: the only cue was the model
        // pill quietly vanishing (review R1 M9).
        // `!h`, not `h === null`: an ABSENT health object is the same
        // finding as a null one, and `undefined` reached the property reads
        // and threw a TypeError inside the render (found by executing this
        // function rather than grepping it — R2 lens C).
        if (!h) return true;
        return !!(h.memory_system_loaded === false
            || h.biological_watchdog_alive === false);
    }

    async function pollHealth() {
        try {
            const res = await fetch('/api/health',
                { signal: AbortSignal.timeout(8000) });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            health = await res.json();
            healthOk = true;
            healthFailNote = '';
        } catch (e) {
            health = null;
            healthOk = false;
            healthFailNote = _healthFailureNote(e);
        }
        window.__ghostHealth = health;
        if (modelPill) {
            const text = healthOk ? pillText(health) : '';
            modelPill.textContent = text;
            modelPill.classList.toggle('hidden', !text);
        }
        indicator?.classList.toggle('degraded', !!isDegraded(health));
        if (!panel?.classList.contains('hidden')) renderPanel();
    }

    function row(label, value, cls) {
        const r = el('div', 'status-row' + (cls ? ` ${cls}` : ''));
        r.appendChild(el('span', 'status-label', label));
        r.appendChild(el('span', 'status-value', String(value)));
        return r;
    }

    function fmtUptime(s) {
        s = Number(s);
        if (!Number.isFinite(s)) return '—';
        if (s < 3600) return `${Math.floor(s / 60)}m`;
        if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
        return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
    }

    let _rendering = false;
    async function renderPanel() {
        if (!bodyEl) return;
        // Re-entrancy guard: the 5s turns timer, the 25s health poll and
        // cancelTurn can overlap; interleaved awaits otherwise append
        // duplicate TURNS sections into a body another call just wiped.
        if (_rendering) return;
        _rendering = true;
        try {
            await _renderPanelOnce();
        } finally {
            _rendering = false;
        }
    }

    async function _renderPanelOnce() {
        bodyEl.replaceChildren();

        if (healthOk === false) {
            bodyEl.appendChild(el('div', 'status-unreachable',
                healthFailNote || _healthFailureNote(null)));
            return;
        }
        if (!health) {
            bodyEl.appendChild(el('div', 'status-unreachable', 'Loading…'));
            return;
        }

        const h = health;
        const sec = el('div', 'status-section');
        sec.appendChild(row('MODEL', pillText(h) || '—'));
        sec.appendChild(row('UPTIME', fmtUptime(h.uptime_s)));
        if (h.rss_mb != null) {
            sec.appendChild(row('MEMORY',
                `${h.rss_mb} MB${h.rss_limit_mb ? ` / ${h.rss_limit_mb} MB` : ''}`));
        }
        if (h.asyncio_tasks != null) sec.appendChild(row('TASKS', h.asyncio_tasks));
        sec.appendChild(row('MEMORY SYSTEM',
            h.memory_system_loaded === false ? 'NOT LOADED' : 'loaded',
            h.memory_system_loaded === false ? 'bad' : 'ok'));
        sec.appendChild(row('WATCHDOG',
            h.biological_watchdog_alive === false ? 'DEAD' : 'alive',
            h.biological_watchdog_alive === false ? 'bad' : 'ok'));
        bodyEl.appendChild(sec);

        // Off-main node pools: empty pool = that work runs on the main model.
        const nodes = h.nodes || {};
        const active = Object.entries(nodes).filter(([, v]) => (v || []).length);
        const nsec = el('div', 'status-section');
        nsec.appendChild(el('div', 'status-section-title', 'NODES'));
        if (active.length) {
            for (const [pool, urls] of active) {
                nsec.appendChild(row(pool.toUpperCase(),
                    urls.map(u => u.replace(/^https?:\/\//, '')).join(', ')));
            }
        } else {
            nsec.appendChild(el('div', 'status-note', 'No off-main nodes — all work on the main slot.'));
        }
        bodyEl.appendChild(nsec);

        // Live turn queue — why the agent "appears unresponsive".
        const tsec = el('div', 'status-section');
        tsec.appendChild(el('div', 'status-section-title', 'TURNS'));
        try {
            // ⚠ BOUNDED. `_rendering` releases in a finally, but an
            // unbounded fetch never reaches it: one hung /api/turns
            // froze the panel for the LIFE of the page, including
            // after close/reopen — and /api/turns is what you open
            // when the agent is wedged (R3 lens B).
            const res = await fetch('/api/turns',
                { signal: AbortSignal.timeout(8000) });
            const data = res.ok ? await res.json() : null;
            const turns = (data && data.turns) || [];
            if (!turns.length) {
                tsec.appendChild(el('div', 'status-note', 'Idle — no turn holds the lock.'));
            }
            for (const t of turns) {
                // ActiveTurn.to_dict() emits request_id / age_s / preview /
                // session_id (turns.py) — an earlier draft guessed req_id
                // and every cancel button posted a garbage id.
                const id = t.request_id || t.req_id || t.id || null;
                const line = el('div', 'status-turn' + (t.running ? ' running' : ''));
                const age = Number(t.age_s);
                const preview = String(t.preview || '').slice(0, 42);
                const desc = el('span', 'status-turn-desc',
                    `${t.running ? '▶' : '⧗'} ${String(id || '?').slice(0, 10)}`
                    + (Number.isFinite(age) ? ` · ${Math.round(age)}s` : '')
                    + (preview ? ` · ${preview}` : ''));
                desc.title = String(t.preview || '');
                line.appendChild(desc);
                if (id) {
                    const stop = el('button', 'status-turn-stop', t.running ? 'Stop' : 'Drop');
                    stop.type = 'button';
                    stop.title = t.running
                        ? 'Cooperative cancel — stops at the next boundary, returns partial work'
                        : 'Remove from queue';
                    stop.addEventListener('click', () => cancelTurn(id, false));
                    line.appendChild(stop);
                    if (t.running) {
                        const force = el('button', 'status-turn-stop force', 'Force');
                        force.type = 'button';
                        force.title = 'Hard cancel — guaranteed lock release for a wedged turn';
                        force.addEventListener('click', () => cancelTurn(id, true));
                        line.appendChild(force);
                    }
                }
                tsec.appendChild(line);
            }
        } catch (e) {
            tsec.appendChild(el('div', 'status-note', 'Turn registry unavailable.'));
        }
        bodyEl.appendChild(tsec);
    }

    async function cancelTurn(requestId, hard) {
        try {
            const res = await fetch('/api/turn/cancel', {
                method: 'POST',
                signal: AbortSignal.timeout(8000),
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestId
                    ? { request_id: requestId, hard }
                    : { hard }),
            });
            const data = await res.json().catch(() => ({}));
            if (res.ok && data.cancelled) {
                toast(hard ? 'Turn hard-cancelled — lock released' : 'Turn cancel requested', 'ok');
            } else {
                toast(data.detail || data.error || 'Turn not found (already finished?)', 'error');
            }
        } catch (e) {
            toast(`Cancel failed: ${e.message}`, 'error');
        }
        renderPanel();
    }

    function openPanel() {
        panel?.classList.remove('hidden');
        renderPanel();
        clearInterval(turnsTimer);
        turnsTimer = setInterval(renderPanel, TURNS_POLL_MS);
    }
    function closePanel() {
        panel?.classList.add('hidden');
        clearInterval(turnsTimer);
        turnsTimer = null;
    }
    function togglePanel() {
        if (panel?.classList.contains('hidden')) openPanel();
        else closePanel();
    }

    indicator?.addEventListener('click', togglePanel);
    indicator?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); togglePanel(); }
    });
    document.getElementById('status-close')?.addEventListener('click', closePanel);
    document.addEventListener('click', (e) => {
        if (panel && !panel.classList.contains('hidden')
            && !panel.contains(e.target) && !indicator?.contains(e.target)) {
            closePanel();
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && panel && !panel.classList.contains('hidden')) closePanel();
    });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') pollHealth();
    });

    pollHealth();
    setInterval(pollHealth, HEALTH_POLL_MS);

    return { openPanel, closePanel, togglePanel, cancelTurn, health: () => health };
}
