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

export function initStatus(_ctx) {
    const indicator = document.getElementById('status-indicator');
    const baseTitle = indicator ? (indicator.getAttribute('data-title') || '') : '';

    let health = null;
    let healthOk = null;
    let healthFailNote = '';

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

    function applyToChip() {
        if (!indicator) return;
        const degraded = isDegraded(health);
        indicator.classList.toggle('degraded', degraded);
        indicator.title = degraded
            ? 'DEGRADED — ' + degradedReason(health, healthOk, healthFailNote)
            : baseTitle;
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
        applyToChip();
    }

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') pollHealth();
    });

    pollHealth();
    setInterval(pollHealth, HEALTH_POLL_MS);

    return { health: () => health };
}
