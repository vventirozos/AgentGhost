// ── Temperament: the face's inner state (2026-10-30) ─────────────────
//
// Operator: "make the faces more reactive to the agent, more interesting,
// less predictable". Before this module every signal mapped to ONE fixed
// animation with a fixed envelope — the same tool call always kicked the
// same amount, so after a few days the whole vocabulary had been seen.
//
// This is the layer BETWEEN the events and the animation. Events nudge a
// handful of continuous drives that decay with momentum; the renderer
// reads the drives every frame. Four things make it less predictable
// without making it random:
//
//   * habituation — the 20th web search in a turn reacts less than the
//     first; it recovers over ~20 s of quiet;
//   * surprise — a kind that is rare (per-kind rate estimate) reacts more;
//   * bounded noise — every reaction is scaled by a draw in ~[0.7, 1.35]
//     and the continuous outputs carry slow Ornstein–Uhlenbeck wander, so
//     no two reactions are identical but the character stays the same;
//   * session memory — fatigue integrates working time, the night lowers
//     the baseline, and a turn after a long absence WAKES it.
//
// At rest it picks from an idle repertoire (glance, stretch, sigh, twitch,
// look-around, doze) instead of one loop, never the same action twice in a
// row.
//
// Pure: no THREE, no DOM. `createTemperament({seed, now})` returns an
// object whose `step(dt, nowMs)` returns the outputs the renderer reads.
// Executed under node by tests/test_interface_face_temperament.py.

export const TEMPERAMENT_VERSION = '1.3';
export const DEFAULT_INTENSITY = 1.5;

// v1.2 (operator: "can't say i saw a huge difference"): v1.1 kept every
// effect within a few percent and steered a gaze that the camera-static
// forms (tesseract, vortex) never read. The BODY outputs below move the
// whole creature on every form — head turns and tilts, a startle shake,
// a coil while waiting that releases on the first words, sinking into a
// doze, brightness flares — scaled by one knob, `intensity` (default 1;
// localStorage `ghost_face_intensity`, 0 = the v1.1 face, 3 = theatrical).

// Log-line icons (src/ghost_agent/utils/logging.py `Icons`) → what kind of
// work, or what outcome, the line reports.
const WORK_BY_ICON = {
    '🌐': 'web', '🌎': 'web', '🧅': 'web', '🔬': 'web',
    '🐍': 'code', '🐚': 'code', '📦': 'code',
    '💾': 'file', '📖': 'file', '📙': 'file', '🔖': 'file', '🔍': 'file', '👀': 'file', '📥': 'file',
    '🔎': 'memory', '📍': 'memory',
    '🎨': 'image',
    '💭': 'think', '🧠': 'think', '📋': 'think', '🧩': 'think', '🧭': 'think',
};
const OUTCOME_BY_ICON = {
    '✅': 'ok', '❌': 'fail', '🔶': 'warn', '🔄': 'retry', '🛑': 'stop',
    '📀': 'restore', '📛': 'cachefail', '🌙': 'dream', '🌀': 'dream', '📨': 'queued',
};

// Every icon the line carries, in order (a line can name a work kind AND an
// outcome: "🐍 Sandbox Exec … ❌"). Pure.
export function classifyLogLine(content, isError) {
    const s = String(content || '');
    const out = [];
    for (const ch of s.match(/\p{Extended_Pictographic}/gu) || []) {
        if (WORK_BY_ICON[ch]) out.push('work:' + WORK_BY_ICON[ch]);
        else if (OUTCOME_BY_ICON[ch]) out.push('outcome:' + OUTCOME_BY_ICON[ch]);
    }
    if (isError && !out.includes('outcome:fail')) out.push('outcome:fail');
    // one line, one reaction per event kind
    return [...new Set(out)];
}

// How strongly each event nudges each drive, before gain. Drives are 0..1
// except valence (−1..1). Positive = toward 1.
const IMPULSE = {
    'tool':            { arousal: 0.10, focus: 0.05 },
    'recall':          { curiosity: 0.18, arousal: 0.05 },
    'activity':        { arousal: 0.025 },
    'turn:start':      { arousal: 0.22, focus: 0.35 },
    'turn:end':        { focus: -0.25 },
    'verdict:pass':    { valence: 0.35, strain: -0.25, arousal: -0.05 },
    'verdict:refute':  { valence: -0.30, strain: 0.25, arousal: 0.18 },
    'verdict:stop':    { valence: 0.05, strain: -0.10 },
    'error':           { valence: -0.35, strain: 0.35, arousal: 0.30 },
    'work:web':        { curiosity: 0.12, arousal: 0.04 },
    'work:code':       { focus: 0.10, arousal: 0.03 },
    'work:file':       { focus: 0.06 },
    'work:memory':     { curiosity: 0.10 },
    'work:image':      { curiosity: 0.15, arousal: 0.06, valence: 0.05 },
    'work:think':      { focus: 0.05 },
    'outcome:ok':      { valence: 0.08, strain: -0.04 },
    'outcome:fail':    { valence: -0.15, strain: 0.18, arousal: 0.10 },
    'outcome:warn':    { strain: 0.06 },
    'outcome:retry':   { strain: 0.10, arousal: 0.04 },
    'outcome:stop':    { strain: 0.05 },
    'outcome:restore': { valence: 0.05 },
    'outcome:cachefail': { strain: 0.12 },
    'outcome:dream':   { curiosity: 0.06 },
    'outcome:queued':  { arousal: 0.08, focus: 0.05 },
    'stream:first':    { valence: 0.06 },   // relief: the reply started
    'wake':            { arousal: 0.40, curiosity: 0.20, valence: 0.10 },
};
// Kinds whose reaction also produces a visible transient (`startle`).
const STARTLE = { 'error': 1.0, 'verdict:refute': 0.6, 'outcome:fail': 0.35, 'wake': 0.5 };

const DRIVES = ['arousal', 'focus', 'strain', 'curiosity', 'fatigue', 'valence'];
// Decay time constants (s) toward each drive's baseline.
const TAU = { arousal: 9, focus: 25, strain: 45, curiosity: 14, valence: 70 };

export const IDLE_ACTIONS = ['glance', 'stretch', 'sigh', 'twitch', 'lookaround', 'doze'];
// Duration (s) of each idle action.
const IDLE_DUR = { glance: 2.6, stretch: 4.5, sigh: 3.5, twitch: 0.6, lookaround: 6.0, doze: 16.0 };

function mulberry32(a) {
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

// Smooth 1-D value noise in [−1, 1], a pure function of time: the wander
// is the same at 30, 60 or 120 Hz (a per-frame random walk was not —
// tests/test_interface_face_motion_2026_09_20 holds every clock to that).
function makeNoise(rnd) {
    const tab = new Float32Array(256);
    for (let i = 0; i < 256; i++) tab[i] = rnd() * 2 - 1;
    const one = (t) => {
        const i = Math.floor(t), f = t - i, u = f * f * (3 - 2 * f);
        const a = tab[i & 255], b = tab[(i + 1) & 255];
        return a + (b - a) * u;
    };
    return (t) => clamp(0.7 * one(t) + 0.3 * one(t * 2.37 + 17.1), -1, 1);
}

// Night = 23:00–07:00 local, as a smooth 0..1 (1 deep in the night).
export function nightness(date) {
    const h = date.getHours() + date.getMinutes() / 60;
    // distance (hours) from 03:00 on the 24 h circle
    const d = Math.min(Math.abs(h - 3), 24 - Math.abs(h - 3));
    return clamp((5.5 - d) / 2.5, 0, 1);       // 1 within ±3 h of 03:00, 0 beyond ±5.5 h
}

export function createTemperament(opts = {}) {
    const rnd = mulberry32(opts.seed != null ? opts.seed : (Date.now() ^ 0x5eed));
    const gauss = () => {           // Box–Muller, one value
        const u = Math.max(1e-9, rnd()), v = rnd();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    let now = opts.now != null ? opts.now : Date.now();
    // default 1.5 (operator, v1.1 → v1.2: "can't say i saw a huge difference")
    const K = clamp(opts.intensity != null && isFinite(opts.intensity) ? +opts.intensity : DEFAULT_INTENSITY, 0, 3);
    let reliefAt = -1e12, wakeAt = -1e12, startleAt = -1e12, curiousAt = -1e12, curiousSign = 1;
    let syllable = 0;           // a glow kick per streamed chunk, decays ~120 ms

    const d = { arousal: 0.15, focus: 0.0, strain: 0.0, curiosity: 0.1, fatigue: 0.0, valence: 0.0 };
    const hab = {};             // kind → 0..1 habituation
    const rate = {};            // kind → events/min (EMA)
    const lastAt = {};          // kind → ms of the last event
    let startle = 0;            // transient 0..1
    let startleQueued = 0;      // kick the renderer consumes once
    let inTurn = false, turnStartAt = 0, firstChunkAt = 0;
    let lastTurnEndAt = opts.lastTurnEndAt != null ? opts.lastTurnEndAt : now;
    let anticipation = 0;
    // stream
    let inThink = false, thinking = 0, speech = 0, charRate = 0, pendingChars = 0;
    // slow wander channels (seconds per feature: gaze 6–7 s, speed 9 s, hue 30 s)
    const nGX = makeNoise(rnd), nGY = makeNoise(rnd), nSP = makeNoise(rnd), nHU = makeNoise(rnd);
    const ou = { gx: 0, gy: 0, sp: 0, hue: 0 };
    let breathPhase = rnd() * Math.PI * 2;
    // idle repertoire
    let idle = null;            // {name, t0, dur, gx, gy, fresh}
    let nextIdleAt = now + 6000 + rnd() * 8000;
    let lastIdle = null;
    let restSince = now;

    const out = {
        arousal: 0, focus: 0, strain: 0, curiosity: 0, fatigue: 0, valence: 0,
        speedMul: 1, ampMul: 1, breath: 0, lean: 0, gazeX: 0, gazeY: 0,
        hueShift: 0, stillness: 0, jitter: 0, anticipation: 0, speech: 0,
        thinking: 0, startleKick: 0, twitch: false, idle: null, idleEnv: 0,
        // body (radians / scene units / multipliers), all 0 / 1 at rest
        yaw: 0, pitch: 0, roll: 0, offX: 0, offY: 0, scaleMul: 1, bright: 1, glow: 0,
    };

    function _nudge(kind, g) {
        const imp = IMPULSE[kind];
        if (!imp) return;
        for (const k in imp) {
            const v = imp[k] * g;
            if (k === 'valence') d.valence = clamp(d.valence + v, -1, 1);
            else d[k] = clamp(d[k] + v * (v > 0 ? (1 - d[k]) : d[k] + 0.2), 0, 1);
        }
        if (kind === 'wake') wakeAt = now;
        if (kind === 'stream:first') reliefAt = now;
        if (kind === 'recall' || kind === 'work:web' || kind === 'work:image') { curiousAt = now; curiousSign = rnd() < 0.5 ? -1 : 1; }   // a head tilt, one side or the other
        if (STARTLE[kind]) {
            startleAt = now;
            const s = STARTLE[kind] * g;
            startle = Math.min(1, startle + s);
            startleQueued = Math.min(1, startleQueued + s);
        }
    }

    // Record an event; returns its GAIN (~0.3..1.8) — the renderer scales
    // its own kick for the same event by it.
    function event(kind, info) {
        const t = (info && info.now) || now;
        // per-kind rate (events/min), EMA over ~2 min
        const dtMin = lastAt[kind] != null ? Math.max(0.001, (t - lastAt[kind]) / 60000) : 10;
        const inst = Math.min(1 / dtMin, 12);             // a burst reads as "frequent", not infinite
        // the estimate forgets with quiet time (~2 min), then takes this gap in
        if (rate[kind] != null) rate[kind] *= Math.exp(-dtMin / 2);
        rate[kind] = rate[kind] == null ? Math.min(inst, 2) : rate[kind] + (inst - rate[kind]) * Math.min(1, dtMin / 2);
        lastAt[kind] = t;
        const h = hab[kind] || 0;
        const surprise = 1 / (1 + 0.35 * rate[kind]);          // rare → ~1, frequent → small
        const noise = Math.exp(0.18 * gauss());                // ~[0.7, 1.4]
        const g = clamp((1 - 0.72 * h) * (0.55 + 0.75 * surprise) * noise * (0.8 + 0.5 * d.arousal), 0.25, 1.8);
        hab[kind] = Math.min(1, h + 0.22);
        if (kind === 'turn:start') {
            const away = t - lastTurnEndAt;
            if (away > 60 * 60 * 1000) _nudge('wake', clamp(away / (4 * 3600e3), 0.6, 1.4));
            inTurn = true; turnStartAt = t; firstChunkAt = 0; anticipation = 0;
            idle = null;
        } else if (kind === 'turn:end') {
            inTurn = false; lastTurnEndAt = t; restSince = t;
            inThink = false;
        }
        _nudge(kind, g);
        return g;
    }

    function ingestLog(content, isError) {
        const kinds = classifyLogLine(content, isError);
        for (const k of kinds) event(k);
        return kinds;
    }

    function ingestChunk(text) {
        const s = String(text || '');
        if (!s) return;
        // the last tag in the chunk decides (a chunk can close and reopen)
        const o = s.lastIndexOf('<think>'), c = s.lastIndexOf('</think>');
        if (o > c) inThink = true; else if (c > o) inThink = false;
        if (!inThink) {
            pendingChars += s.replace(/<\/?think>/g, '').length;
            syllable = Math.min(1, syllable + 0.35);
            if (inTurn && !firstChunkAt && s.replace(/<\/?think>|\s/g, '').length) {
                firstChunkAt = now;
                event('stream:first');
            }
        }
    }

    function _pickIdle() {
        const night = nightness(new Date(now));
        const w = {
            glance: 3, stretch: 1.4, sigh: 1.4, twitch: 2, lookaround: 1.2,
            doze: 0.25 + 2.5 * Math.max(d.fatigue, night),
        };
        if (lastIdle) w[lastIdle] = 0;                        // never the same twice running
        let sum = 0;
        for (const k of IDLE_ACTIONS) sum += w[k];
        let r = rnd() * sum;
        for (const k of IDLE_ACTIONS) { r -= w[k]; if (r <= 0) return k; }
        return 'glance';
    }

    function step(dt, nowMs) {
        dt = clamp(dt || 0, 0, 0.25);
        now = nowMs != null ? nowMs : now + dt * 1000;
        const night = nightness(new Date(now));

        // habituation recovers (~20 s), startle fades (~1.2 s)
        for (const k in hab) { hab[k] *= Math.exp(-dt / 20); if (hab[k] < 0.01) delete hab[k]; }
        startle *= Math.exp(-dt / 1.2);

        // drives relax toward their baselines
        const base = {
            arousal: 0.12 + 0.18 * (inTurn ? 1 : 0) - 0.08 * night - 0.06 * d.fatigue,
            focus: inTurn ? 0.45 : 0.0,
            strain: 0.0, curiosity: 0.08, valence: 0.0,
        };
        for (const k in TAU) d[k] += (base[k] - d[k]) * (1 - Math.exp(-dt / TAU[k]));
        // a long turn strains (beyond 2 min it creeps up)
        if (inTurn) {
            const el = (now - turnStartAt) / 1000;
            if (el > 120) d.strain = clamp(d.strain + dt * 0.002, 0, 0.6);
        }
        // fatigue: +1 per ~2 h of work, −1 per ~30 min of rest, night adds
        const working = inTurn ? 1 : 0;
        d.fatigue = clamp(d.fatigue + dt * (working / 7200 - (1 - working) / 1800 + night / 14400), 0, 1);

        // stream: char rate (chars/s) → speech; thinking envelope
        const instRate = pendingChars / Math.max(dt, 1e-3);
        pendingChars = 0;
        charRate += (instRate - charRate) * (1 - Math.exp(-dt / 0.6));
        speech += (clamp(charRate / 140, 0, 1) - speech) * (1 - Math.exp(-dt / 0.35));
        thinking += ((inTurn && inThink ? 1 : 0) - thinking) * (1 - Math.exp(-dt / 1.5));
        // anticipation: builds while a turn waits for its first answer text
        const waiting = inTurn && !firstChunkAt;
        const antT = waiting ? 1 - Math.exp(-((now - turnStartAt) / 1000) / 8) : 0;
        anticipation += (antT - anticipation) * (1 - Math.exp(-dt / (waiting ? 0.8 : 0.4)));

        const ts = now / 1000;
        ou.gx = 0.75 * nGX(ts / 6); ou.gy = 0.6 * nGY(ts / 7 + 40);
        ou.sp = 0.09 * nSP(ts / 9 + 80); ou.hue = 0.012 * nHU(ts / 30 + 120);

        // idle repertoire (only at rest: no turn, calm, ≥ 8 s since the last turn)
        let idleEnv = 0, twitch = false;
        const resting = !inTurn && d.arousal < 0.35 && now - restSince > 8000;
        if (idle) {
            const p = (now - idle.t0) / 1000 / idle.dur;
            if (p >= 1 || !resting) { lastIdle = idle.name; idle = null; }
            else idleEnv = Math.sin(Math.PI * clamp(p, 0, 1));
        }
        if (!idle && resting && now >= nextIdleAt) {
            const name = _pickIdle();
            const a = rnd() * Math.PI * 2;
            idle = { name, t0: now, dur: IDLE_DUR[name] * (0.75 + 0.5 * rnd()), gx: Math.cos(a), gy: Math.sin(a) * 0.6 };
            if (name === 'twitch') twitch = true;
            // mean gap 14 s, longer when tired or at night
            nextIdleAt = now + (7000 + rnd() * 14000) * (1 + d.fatigue + 0.6 * night);
        }
        if (!resting && now > nextIdleAt) nextIdleAt = now + 6000 + rnd() * 6000;

        // ── outputs ──
        const N = idle ? idle.name : null;
        const doze = N === 'doze' ? idleEnv : 0;
        out.arousal = d.arousal; out.focus = d.focus; out.strain = d.strain;
        out.curiosity = d.curiosity; out.fatigue = d.fatigue; out.valence = d.valence;
        out.jitter = clamp(0.6 * d.strain + 0.8 * startle, 0, 1);
        // ≈1.0 at rest in daytime, so today's look stays the baseline
        out.speedMul = clamp(0.91 + 0.55 * d.arousal + 0.25 * d.curiosity - 0.28 * d.fatigue
            - 0.12 * night - 0.30 * thinking * (1 - speech) + 0.12 * speech
            - 0.25 * anticipation - 0.45 * doze + (N === 'stretch' ? 0.35 * idleEnv : 0)
            + ou.sp, 0.45, 1.75);
        out.ampMul = clamp(0.85 + 0.45 * d.arousal + 0.35 * out.jitter + 0.15 * speech
            + (N === 'stretch' ? 0.5 * idleEnv : 0) - 0.3 * doze, 0.6, 1.8);
        breathPhase += dt * Math.PI * 2 * (0.10 + 0.12 * d.arousal - 0.03 * d.fatigue - 0.03 * doze);
        out.breath = Math.sin(breathPhase) * (0.4 + 0.6 * Math.max(d.fatigue, doze))
            + (N === 'sigh' ? -0.8 * idleEnv * Math.sin(Math.PI * idleEnv) : 0)
            + (N === 'stretch' ? 1.0 * idleEnv : 0);
        out.lean = clamp(anticipation * 0.9 + 0.35 * d.focus * (inTurn ? 1 : 0), 0, 1);
        const wander = (1 - 0.7 * d.focus) * (1 - doze);
        let gx = 0.10 * ou.gx * wander, gy = 0.07 * ou.gy * wander;
        if (N === 'glance') { gx += 0.22 * idle.gx * idleEnv; gy += 0.18 * idle.gy * idleEnv; }
        if (N === 'lookaround') {
            const p = (now - idle.t0) / 1000 / idle.dur;
            gx += 0.22 * Math.sin(p * Math.PI * 2 + idle.gx) * idleEnv;
            gy += 0.10 * Math.sin(p * Math.PI * 4) * idleEnv;
        }
        if (doze) gy -= 0.12 * doze;                        // head drops
        gy += 0.02 * speech * Math.sin(now / 1000 * 7.0);   // a faint nod while it talks
        out.gazeX = gx; out.gazeY = gy;
        out.hueShift = clamp(-0.025 * d.valence + 0.03 * d.strain + 0.012 * d.curiosity + ou.hue, -0.05, 0.06);
        out.stillness = clamp(0.30 * thinking * (1 - speech) + 0.5 * doze
            + (N === 'sigh' ? 0.35 * idleEnv : 0) + 0.15 * anticipation, 0, 0.85);
        out.anticipation = anticipation; out.speech = speech; out.thinking = thinking;
        // ── body ──
        const since = (t0) => (now - t0) / 1000;
        const pulse = (t0, dur) => { const x = since(t0) / dur; return x < 0 || x > 1 ? 0 : Math.sin(Math.PI * x) * (1 - x); };
        syllable *= Math.exp(-dt / 0.12);
        const shake = startle * Math.sin(now * 0.044) ;                      // ~7 Hz, fades with the startle
        const wake = pulse(wakeAt, 3.0), relief = pulse(reliefAt, 1.6), cur = pulse(curiousAt, 2.5);
        let yaw = 1.6 * gx, pitch = 1.4 * gy;                                // the gaze, as a head turn
        if (N === 'stretch') pitch -= 0.10 * idleEnv;
        if (doze) pitch += 0.20 * doze;                                      // the head drops
        yaw += 0.06 * Math.sin(now / 1000 * 0.9) * thinking;                 // a slow pondering sway
        let roll = 0.10 * cur * curiousSign + 0.03 * shake + 0.05 * Math.sin(now / 1000 * 0.37) * d.curiosity;
        out.yaw = K * yaw; out.pitch = K * pitch; out.roll = K * roll;
        out.offX = K * (0.05 * shake);
        out.offY = K * (-0.16 * doze - 0.04 * (N === 'sigh' ? idleEnv : 0) + 0.05 * wake + 0.03 * relief);
        out.scaleMul = 1 + K * (0.012 * out.breath * 2.5
            - 0.10 * anticipation * (1 - relief)                             // coils while it waits…
            + 0.08 * relief                                                  // …and lets go on the first words
            + 0.09 * (N === 'stretch' ? idleEnv : 0) - 0.05 * (N === 'sigh' ? idleEnv : 0)
            + 0.08 * wake - 0.05 * doze + 0.025 * speech);
        out.bright = clamp(1 + K * (0.35 * startle + 0.35 * wake + 0.18 * speech + 0.10 * cur
            - 0.35 * doze - 0.12 * thinking * (1 - speech) + 0.12 * relief), 0.45, 1.9);
        out.glow = K * Math.min(1, syllable * speech + 0.4 * syllable);      // text rhythm on the bloom
        out.startleKick = startleQueued; startleQueued = 0;
        out.twitch = twitch;
        out.idle = N; out.idleEnv = idleEnv;
        return out;
    }

    function debug() {
        return { drives: { ...d }, hab: { ...hab }, rate: { ...rate }, inTurn, inThink,
                 anticipation, speech, thinking, idle: idle ? idle.name : null, nextIdleAt, now };
    }

    return { event, ingestLog, ingestChunk, step, debug, DRIVES };
}
