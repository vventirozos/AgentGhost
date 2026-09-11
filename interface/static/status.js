// ═══════════════════════════════════════════════════════════════
//  Agent health tag (2026-07-28; reduced 2026-09-05)
//
//  Polls GET /api/health every 25s (and on tab-visible) and marks the
//  header's ONLINE chip DEGRADED when either silent-failure detector
//  fires (see api_health): memory_system_loaded=false means a degraded
//  boot disabled the biological phases while HTTP keeps answering;
//  biological_watchdog_alive=false means the self-improvement daemon
//  died. A FAILED poll is DEGRADED too — the agent being down, or the
//  key rotated, is the most abnormal state there is and must never
//  read as calm (§4BU R1 M9).
//
//  2026-09-05 (operator): the model pill beside ONLINE and the agent
//  status panel the chip opened (uptime / RSS / node pools / the live
//  turn queue with per-turn Stop and Force) were REMOVED. Later the same
//  day the chip became the LIVE LOG toggle (app.js owns that click); this
//  module only paints it. What the panel used to explain — WHY the chip
//  is red — rides the chip's `title`, restored to its resting tooltip
//  (`data-title`) when healthy. The composer's Stop still
//  reaches the real cancel (POST /api/turn/cancel — app.js
//  cancelOwnTurn); gone with the panel are the queue LISTING and the
//  Force (hard) cancel of an arbitrary turn.
// ═══════════════════════════════════════════════════════════════

const HEALTH_POLL_MS = 25_000;
// The interface's own probe (2026-09-11): TLS certificate expiry lives
// here. The Let's-Encrypt-via-Tailscale cert the phone PWA depends on had
// a renewal note in the launcher and nothing watching it; on expiry the
// PWA, service worker and web push all die with only a browser
// interstitial to say why. Amber under TLS_WARN_DAYS.
const TLS_WARN_DAYS = 14;

export function initStatus(_ctx) {
    const indicator = document.getElementById('status-indicator');
    const baseTitle = indicator ? (indicator.getAttribute('data-title') || '') : '';

    let health = null;
    let healthOk = null;
    let healthFailNote = '';
    let ihealth = null;      // GET /api/interface/health

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

    // Why the chip is red. The panel that spelled this out is gone, so the
    // reason rides the tooltip instead of leaving the operator to guess
    // between "agent down", "key rotated" and "the watchdog died" — three
    // different fixes. Pure; executed under node with the chip harness.
    function degradedReason(h, ok, failNote) {
        if (ok === false || !h) return failNote || _healthFailureNote(null);
        const parts = [];
        if (h.memory_system_loaded === false) parts.push('memory system NOT LOADED');
        if (h.biological_watchdog_alive === false) parts.push('biological watchdog DEAD');
        return parts.join(' · ');
    }

    // Why the certificate matters, or null. Pure; executed under node.
    // `days_left` is the server's number; a missing/erroring TLS block is
    // NOT a warning (plain-HTTP dev runs have no certificate at all).
    function tlsWarning(ih) {
        const tls = ih && ih.tls;
        if (!tls || typeof tls.days_left !== 'number') return null;
        const d = tls.days_left;
        if (d < 0) {
            return 'TLS certificate EXPIRED ' + Math.ceil(-d) + ' day(s) ago — the phone app, '
                + 'service worker and push are down until it is renewed (tailscale cert <host>).';
        }
        if (d < TLS_WARN_DAYS) {
            return 'TLS certificate expires in ' + Math.floor(d) + ' day(s) — renew now '
                + '(tailscale cert <host>), or the phone app, service worker and push stop.';
        }
        return null;
    }

    function applyToChip() {
        if (!indicator) return;
        const agentDegraded = isDegraded(health);
        const tlsNote = tlsWarning(ihealth);
        const degraded = agentDegraded || !!tlsNote;
        indicator.classList.toggle('degraded', degraded);
        const reasons = [];
        if (agentDegraded) reasons.push(degradedReason(health, healthOk, healthFailNote));
        if (tlsNote) reasons.push(tlsNote);
        indicator.title = degraded ? 'DEGRADED — ' + reasons.join(' · ') : baseTitle;
    }

    async function pollInterfaceHealth() {
        try {
            const res = await fetch('/api/interface/health',
                { signal: AbortSignal.timeout(8000) });
            ihealth = res.ok ? await res.json() : null;
        } catch (e) {
            ihealth = null;    // the AGENT probe owns the "unreachable" verdict
        }
        window.__ghostInterfaceHealth = ihealth;
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
        await pollInterfaceHealth();
        applyToChip();
        feedFace();
    }

    // The face's slow signals (2026-09-11): the agent's functional mood
    // (health.mood.label → a baseline hue shift) and whether a turn that
    // is NOT ours holds the lock (a dream / self-play turn → a second,
    // slower breath at the edge). Pure decision in `backgroundBusyFrom`;
    // executed under node.
    function backgroundBusyFrom(turnsPayload, mySessionId) {
        const turns = (turnsPayload && Array.isArray(turnsPayload.turns)) ? turnsPayload.turns : [];
        return turns.some(t => t && t.running && (t.session_id || null) !== (mySessionId || null));
    }
    async function feedFace() {
        // `_ctx.Core` may be absent (a harness, or a stale bridge): the face
        // is decoration for this probe and must never take the chip down.
        const Core = (_ctx && _ctx.Core) || null;
        const face = Core && Core.activeFace;
        if (!face) return;
        try {
            if (typeof face.setMoodHue === 'function') {
                face.setMoodHue(health && health.mood ? health.mood.label : null);
            }
            if (typeof face.setBackgroundBusy === 'function') {
                const res = await fetch('/api/turns', { signal: AbortSignal.timeout(6000) });
                const data = res.ok ? await res.json() : null;
                const mine = window.__ghostSessionId
                    || (typeof Core.storedSessionId === 'function' ? Core.storedSessionId() : null);
                face.setBackgroundBusy(backgroundBusyFrom(data, mine));
            }
        } catch (e) { /* the face is decoration for this probe — never fail the chip */ }
    }

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') pollHealth();
    });

    pollHealth();
    setInterval(pollHealth, HEALTH_POLL_MS);

    return { health: () => health };
}
