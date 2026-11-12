// ═══════════════════════════════════════════════════════════════
//  Agent health tag (2026-07-28; reduced 2026-09-05)
//
//  Polls GET /api/health every 25s (and on tab-visible) and marks the
//  header's ONLINE chip DEGRADED when either silent-failure detector
//  fires (see api_health): memory_system_loaded=false means a degraded
//  boot disabled the idle jobs while HTTP keeps answering;
//  biological_watchdog_alive=false (historical key name, §4NN) means the
//  idle scheduler died. A FAILED poll is DEGRADED too — the agent being down, or the
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
// §4UD F11: a mood change reaches the face at the end of a turn, not at the next
// 25 s poll. The agent moves the mood just AFTER the stream ends (the turn's
// record path), so the one extra health read waits this long; several triggers
// inside the wait make one read. A read that still comes too early is corrected
// by the next poll.
const MOOD_REFRESH_DELAY_MS = 1500;

// ── §4UD: the mood block, read ONCE ────────────────────────────────────
// `moodView` is the only reader of `health.mood`: the face's look
// (`faceMoodFrom`), the visible words (`moodWords`) and the tooltip
// (`moodNote`) are all made from what it returns, so they cannot disagree.
// The handheld has the same rules in Python (agentapi.py `mood_view`,
// `mood_words`, `mood_clock`); tests/test_4ud_mood_ui.py runs one table of
// payloads through both. All pure; executed under node.

// The labels a face has a look for (matrix_graph.js MOOD_LOOKS) + neutral.
export function moodLabels() {
    return ['neutral', 'cheerful', 'content', 'calm', 'restless', 'bored',
        'curious', 'irritable', 'down', 'tired'];
}
// How many polls in a row may come back without a mood before it is unknown.
export function moodStalePolls() { return 3; }
export function moodLevel(v) { return Number.isInteger(v) && v >= 0 && v <= 3 ? v : null; }

// 'HH:MM' in this device's time for an ISO stamp, '' for anything else.
// A stamp without a zone is already a wall-clock time. No Date string
// parsing: browsers differ on fractions and impossible days.
export function moodClock(iso) {
    if (typeof iso !== 'string') return '';
    const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,9})?)?(Z|[+-]\d{2}:\d{2})?$/.exec(iso);
    if (!m) return '';
    const [y, mo, d, h, mi, sec] = [1, 2, 3, 4, 5, 6].map(i => Number(m[i] || 0));
    if (y < 2000 || y > 2999 || mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || sec > 59) return '';
    const two = v => String(v).padStart(2, '0');
    if (!m[7]) return two(h) + ':' + two(mi);
    let off = 0;
    if (m[7] !== 'Z') {
        const oh = Number(m[7].slice(1, 3)), om = Number(m[7].slice(4, 6));
        if (oh > 23 || om > 59) return '';
        off = (m[7][0] === '-' ? -1 : 1) * (oh * 60 + om);
    }
    const at = new Date(Date.UTC(y, mo - 1, d, h, mi, sec) - off * 60000);
    return Number.isNaN(at.getTime()) ? '' : two(at.getHours()) + ':' + two(at.getMinutes());
}

// The mood block as the clients use it, or null when it is absent or
// malformed (not an object, an unknown label, an intensity that is not a
// whole number 0–3, a label that contradicts the intensity). A bad
// OPTIONAL field is dropped on its own. Never throws.
//   face + strength — the look: the server's `face` when it sends one
//   (W1: the engine's rule), else this client's rule for an older agent
//   (the mood, or while it is neutral the resting point). strength is
//   1–3, 'rest' (the resting point, faint) or 0 (neutral).
//   §4UG: p + e — the mood's two numbers (−1..1), both or null (an older
//   agent, a bad value): the face blends its looks by them. shade + shadeAt —
//   'relieved' | 'chastened' | 'proud' with the stamp it was set at, or '' and
//   '' (the face plays one moment per stamp).
export function moodView(m) {
    if (!m || typeof m !== 'object' || Array.isArray(m)) return null;
    const known = moodLabels();
    const isLabel = v => typeof v === 'string' && known.includes(v);
    const n = moodLevel(m.intensity);
    if (n === null || !isLabel(m.label)) return null;
    if ((n > 0) !== (m.label !== 'neutral')) return null;
    // r2: a mood has ONE label. A `face` that names another label than the mood's own (or neutral
    // beside a level) is a block nobody can show truthfully — a miss, not "neutral".
    if (n > 0 && isLabel(m.face) && m.face !== m.label) return null;
    const face = isLabel(m.face) ? m.face : (n ? m.label : (isLabel(m.rest) ? m.rest : 'neutral'));
    const text = (v, max) => (typeof v === 'string' ? v.trim().slice(0, max) : '');
    const manner = (typeof m.manner === 'string' && /^[a-z]{1,24}$/.test(m.manner)) ? m.manner : '';
    const unit = v => typeof v === 'number' && Number.isFinite(v) && v >= -1 && v <= 1;
    const point = unit(m.p) && unit(m.e);
    const shade = (['relieved', 'chastened', 'proud'].includes(m.shade) && moodClock(m.shade_at)) ? m.shade : '';
    return {
        p: point ? m.p : null,
        e: point ? m.e : null,
        shade,
        shadeAt: shade ? m.shade_at : '',
        // §4UG-f: the shade's age by the agent's clock (null without it: the face falls back to its own clock)
        shadeAgeS: (shade && typeof m.shade_age_s === 'number' && Number.isFinite(m.shade_age_s) && m.shade_age_s >= 0) ? m.shade_age_s : null,
        face,
        strength: face === 'neutral' ? 0 : (n || 'rest'),
        problem: (typeof m.problem === 'string' && /[^ \t\r\n]/.test(m.problem))
            ? (text(m.problem, 120) || 'unreadable') : '',
        cause: n ? text(m.cause, 200) : '',
        held: !!(n && text(m.held_until, 40)),
        heldUntil: n ? moodClock(m.held_until) : '',
        manner,
        mannerLevel: manner ? (moodLevel(m.manner_intensity) || 0) : 0,
        mannerUntil: manner ? moodClock(m.manner_until) : '',
    };
}

// "evil, 2 of 3 — until 04:00" (F10): the level and the end when the agent sends them.
export function mannerWords(view) {
    return view.manner + (view.mannerLevel ? ', ' + view.mannerLevel + ' of 3' : '')
        + (view.mannerUntil ? ' — until ' + view.mannerUntil : '');
}

// The mood as one short phrase of visible text (F11: a tooltip cannot be
// reached by touch). Made from the SAME label and strength the face shows
// (W4): "calm, 1 of 3", "rests calm" (neutral, the look is the resting
// point's), "neutral"; "mood unknown" after `moodStalePolls` polls without one.
// F12: when the agent could not read its state file, that is said BESIDE a
// mood the block does carry ("cheerful, 2 of 3 · state file was unreadable at
// start") and INSTEAD of the words only when there is nothing else to say
// ("mood unreadable" — a neutral that may be an emptied state).
export function moodWords(view, unknown) {
    if (unknown) return 'mood unknown';
    if (!view) return '';
    if (view.problem && !view.strength && !view.manner) return 'mood unreadable';
    const parts = [view.strength === 'rest' ? 'rests ' + view.face
        : view.strength ? view.face + ', ' + view.strength + ' of 3' : 'neutral'];
    if (view.manner) parts.push('manner ' + mannerWords(view));
    if (view.problem) parts.push('state file was unreadable at start');
    return parts.join(' · ');
}

// §4UA: what the face shows of one mood block — the mood itself, or, while it
// is neutral, the character's resting point at the faint 'rest' strength.
// §4UD: read through `moodView`; [null, null] for an absent or malformed block.
export function faceMoodFrom(mood) {
    const v = moodView(mood);
    return v ? [v.face, v.strength] : [null, null];
}

// One step of the reading's memory. A poll with a usable mood block resets
// it; a poll without one (agent down, no block, a malformed block) keeps the
// last block for now, and the `moodStalePolls`-th in a row drops it: unknown.
export function moodTrack(prev, mood) {
    if (moodView(mood)) return { misses: 0, mood, unknown: false };
    const misses = ((prev && prev.misses) || 0) + 1;
    const unknown = misses >= moodStalePolls();
    return { misses, mood: unknown ? null : ((prev && prev.mood) || null), unknown };
}

// What the face is told after a poll: the reading's look; with no reading
// yet, [null, null] (it keeps the look it has); once the mood is unknown,
// neutral — the face eases there on its usual path (F11).
export function faceLookFor(track) {
    if (track && track.mood) return faceMoodFrom(track.mood);
    return (track && track.unknown) ? ['neutral', 0] : [null, null];
}

// §4TX: the mood in words on the chip's tooltip, with what the short
// phrase leaves out: the cause, the owner's hold and when it ends.
export function moodNote(h, unknown) {
    if (unknown) return '\nmood: unknown — no reading for ' + moodStalePolls() + ' polls';
    const v = moodView(h && h.mood);
    if (!v) return '';
    const parts = [];
    if (v.problem) parts.push('mood: unreadable — ' + v.problem);
    if (v.strength === 'rest') {
        parts.push('mood: neutral — rests ' + v.face);
    } else if (v.strength) {
        parts.push('mood: ' + v.face + ', ' + v.strength + ' of 3'
            + (v.cause ? ' — ' + v.cause : '')
            + (v.held ? ' (held as the owner set it' + (v.heldUntil ? ' until ' + v.heldUntil : '') + ')' : ''));
    }
    if (v.manner) parts.push('manner: ' + mannerWords(v));
    return parts.length ? '\n' + parts.join('\n') : '';
}

export function initStatus(_ctx) {
    const indicator = document.getElementById('status-indicator');
    const baseTitle = indicator ? (indicator.getAttribute('data-title') || '') : '';

    let health = null;
    let healthOk = null;
    let healthFailNote = '';
    let ihealth = null;      // GET /api/interface/health
    // §4UD: the last mood reading and how many polls in a row had none (`moodTrack`)
    let moodState = { misses: 0, mood: null, unknown: false };
    let moodRefreshTimer = null;

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
        if (h.biological_watchdog_alive === false) parts.push('idle scheduler DEAD');
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
        indicator.title = (degraded ? 'DEGRADED — ' + reasons.join(' · ') : baseTitle)
            + moodNote({ mood: moodState.mood }, moodState.unknown);
    }

    // One extra health read a moment from now; triggers inside the wait share it.
    function scheduleMoodRefresh() {
        if (moodRefreshTimer !== null) return;
        moodRefreshTimer = setTimeout(() => { moodRefreshTimer = null; requestHealth(); }, MOOD_REFRESH_DELAY_MS);
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
        moodState = moodTrack(moodState, health && health.mood);
        // §4RB: every health result, for readers outside this module (effort.js)
        try { window.dispatchEvent(new CustomEvent('ghost-health', { detail: health })); } catch (_) { /* old browser */ }
        await pollInterfaceHealth();
        applyToChip();
        feedFace();
    }

    // §4UD r2: ONE health read at a time. The 25 s poll, the tab coming back and
    // the extra read after a turn all come through here; a trigger while a read
    // is in flight asks for one more read when it ends, never a second request
    // beside it. So answers are applied in the order they were asked (a slow
    // older answer cannot land over a newer one), and a burst of triggers
    // against a dead agent is one missed poll, not three. After a read that
    // brought no mood the asked-for re-read is dropped: the next poll is soon
    // enough, and it would only count the same outage twice.
    let healthReading = false;
    let healthAgain = false;
    async function requestHealth() {
        if (healthReading) { healthAgain = true; return; }
        healthReading = true;
        try {
            do {
                healthAgain = false;
                await pollHealth();
            } while (healthAgain && moodState.misses === 0);
        } finally {
            healthReading = false;
            healthAgain = false;
        }
    }

    // The face's slow signals (2026-09-11): the agent's mood
    // (health.mood.label → a baseline hue shift, scaled by
    // health.mood.intensity — §4TC) and whether a turn that
    // is NOT ours holds the lock (a dream / self-play turn → a second,
    // slower breath at the edge). Pure decision in `backgroundBusyFrom`;
    // executed under node.
    function backgroundBusyFrom(turnsPayload, mySessionId) {
        const turns = (turnsPayload && Array.isArray(turnsPayload.turns)) ? turnsPayload.turns : [];
        return turns.some(t => t && t.running && (t.session_id || null) !== (mySessionId || null));
    }
    // §4UG: the rest of what a poll tells the face — the mood's two numbers
    // (BEFORE the label, so the look is computed once from both), a new shade
    // (the face plays each stamp once) and the character's dials. An older
    // face lacks these functions and keeps today's looks. A reading kept
    // through a failed poll keeps its numbers; a moment and the dials are
    // taken only from a poll that answered.
    function feedPersonality(face) {
        try {
            const v = moodState.mood ? moodView(moodState.mood) : null;
            if (typeof face.setMoodPoint === 'function') {
                if (v) face.setMoodPoint(v.p, v.e);
                else if (moodState.unknown) face.setMoodPoint(null, null);
            }
            if (!health) return;
            if (v && moodState.misses === 0 && typeof face.noteShade === 'function') face.noteShade(v.shade, v.shadeAt, v.shadeAgeS);
            if (typeof face.setCharacter === 'function') {
                const c = health.character;
                face.setCharacter(c && typeof c === 'object' && !Array.isArray(c) ? c : null);
            }
        } catch (e) { /* decoration: the label's look below must still arrive */ }
    }
    async function feedFace() {
        // `_ctx.Core` may be absent (a harness, or a stale bridge): the face
        // is decoration for this probe and must never take the chip down.
        const Core = (_ctx && _ctx.Core) || null;
        const face = Core && Core.activeFace;
        if (!face) return;
        try {
            feedPersonality(face);
            if (typeof face.setMoodHue === 'function') {
                const fm = faceLookFor(moodState);
                face.setMoodHue(fm[0], fm[1]);
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
        if (document.visibilityState === 'visible') requestHealth();
    });

    // §4UD F11: the end of a reply (a `set_mood` turn included) and a recorded
    // thumb are when the mood moves; app.js announces both on Core.events.
    const coreEvents = _ctx && _ctx.Core && _ctx.Core.events;
    if (coreEvents && typeof coreEvents.addEventListener === 'function') {
        for (const name of ['turn-complete', 'mood-changed']) coreEvents.addEventListener(name, scheduleMoodRefresh);
    }

    requestHealth();
    setInterval(requestHealth, HEALTH_POLL_MS);

    return { health: () => health };
}
