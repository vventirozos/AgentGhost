import * as THREE from 'three';
import { createTemperament, logRowIcon } from './temperament.js?v=1.6';
export { logRowIcon };
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// --- SPEED CONFIGURATION ---
export const SPEEDS = {
    idle: 0.75,
    busy: 4.0,
};

// --- COLOR CONFIGURATION ---
// THERMAL two-pole scheme (2026-07-28, operator request — the 5-hue
// jewel wheel read as "too colorful, attracts too much attention"):
// the cloud lives between exactly two poles, deep ocean BLUE (cold
// analysis) and dark arterial RED (hot intent), mutating into each
// other through dark violet — the client chrome's own accent family,
// so the bridge reads intentional. The ring below still has 5 stops
// (the shader's palette() contract is unchanged) but traverses
// blue → indigo → violet → red → plum → back to blue: as uHueDrift
// slides and the spatial hue-wave travels, every node continuously
// MIGRATES between the two poles — the color mutation IS the
// animation. Additive blending + bloom lift these considerably, so
// every stop stays several steps darker than it reads on screen
// (dark-first; the 2026-07-12 envelope smoothing is intact).
export const COLORS = {
    background: new THREE.Color('#000000'),

    // Dim floor under every hue — what a node/line reads as at the
    // bottom of its color breath (a whisper of hue is always added on
    // top, so even the idle graph is tinted, not grey).
    nodeBase: new THREE.Color('#070811'),
    lineBase: new THREE.Color('#05060e'),

    // The thermal ring. Order matters — adjacent stops blend and the
    // ring wraps (last → first). Two bridges (violet down, plum up)
    // keep the blue↔red crossing dark and gradual instead of flashing
    // through a bright mid-tone. Greens/yellows deliberately absent.
    palette: [
        new THREE.Color('#0c1c50'),  // deep ocean blue
        new THREE.Color('#1e2a78'),  // indigo blue
        new THREE.Color('#4a165a'),  // violet bridge (warming)
        new THREE.Color('#701226'),  // dark arterial red
        new THREE.Color('#38173f'),  // plum bridge (cooling)
    ],

    // Errors flush noticeably HOTTER than the palette's resting red —
    // brighter and pinker than any ring stop, so the tint still reads
    // as "wrong" now that red is part of the base scheme.
    nodeError: new THREE.Color('#8f1226'),
    lineError: new THREE.Color('#9c1430'),
};
// ---------------------------

// iOS Safari / mobile adaptive detail. The per-frame proximity loop is
// O(nodeCount^2) on the CPU and runs every tick; at 250 nodes that's
// ~31k distance checks per frame, which is already tight on A-series
// silicon. Halving node count + bloom scale keeps the look readable
// without dropping frames on small screens. Safari's devicePixelRatio
// is also capped more aggressively — bloom cost scales with backing
// store pixels, not CSS pixels.
const _mqMobile = window.matchMedia('(max-width: 768px), (max-height: 600px)');
const IS_MOBILE = _mqMobile.matches;
// Users who ask for reduced motion get a calmer sphere: no orbit and a
// much slower morph. Mirrors the CSS prefers-reduced-motion block.
const PREFERS_REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const NODE_COUNT = IS_MOBILE ? 120 : 250;
// 2026-09-21: the vortex's user-turn web measured 11,942 qualifying links
// on desktop (2,944 mobile) against caps of 10,000 / 2,500 — capped in
// 64% of turn frames (91% under the read gait's thicken). Headroom here
// covers the plain turn; the link BUDGET below (`_linkBudget`) handles
// whatever still overflows, uniformly.
const MAX_LINES = IS_MOBILE ? 3000 : 12000;
const BLOOM_SCALE = IS_MOBILE ? 0.6 : 1.0;
// Backing-store pixel-ratio cap. iOS Safari reports dpr=3 on Retina
// phones; rendering + bloom at 3x backing-store pixels is the single
// biggest cost we can shed with no visual penalty. Read at init AND on
// every resize / DPR change (§4PR: a window dragged between a 1x and a 2x
// monitor kept its first ratio).
const DPR_CAP = IS_MOBILE ? 1.5 : 2.0;
function _pixelRatio() { return Math.min(window.devicePixelRatio || 1, DPR_CAP); }
// Tightened 2.5 → 1.7 with the medusa anatomy: the body is locally
// dense (membrane, strands), so a wide link radius smeared the form
// back into an undifferentiated web. Shorter links trace the anatomy.
const PROXIMITY_SQ = 1.7;
// Link budget (2026-09-21). When the proximity pass finds more pairs
// than MAX_LINES, the emitter used to keep the FIRST pairs by node index
// and silently drop the rest — so during a vortex surge the highest-
// indexed region (the last-built stream nodes) lost its web entirely and
// the count flickered across the cap frame to frame. Now an overflow
// frame tightens the link radius a little for the next frame and an
// under-budget frame relaxes it back: the web thins UNIFORMLY to fit,
// nothing is dropped by position. `_linkBudget` scales proximitySq;
// 1.0 whenever the budget is not binding.
let _linkBudget = 1.0;
const LINK_BUDGET_MIN = 0.45;

let scene, camera, renderer, composer, bloomPass;
let instancedMesh, linesMesh;
let lineGeometry, nodeMaterial, lineMaterial;
let time = 0;
let currentShapeSpeed = SPEEDS.idle;
let animationFrameId;

// ── Shader clock (2026-10-17, §4PR) ───────────────────────────────
// `time` is a float64 that grows forever (~0.3 units/s); the GPU reads
// uniforms as float32, whose resolution at a multi-day value (the
// ClockworkPi runs always-on: 3 days ≈ 78,000 units → ulp ≈ 0.008) is
// coarser than one frame's step (0.005), so every shader phase stair-
// stepped. The shaders now get:
//   uTime        = time wrapped at SHADER_TIME_PERIOD = 4000π. Every
//                  `uTime * f` the shaders use is a sine argument whose f
//                  (× uOrganic ∈ {0.25, 1}) makes f·4000π a whole number of
//                  turns (2000·f ∈ ℤ: 0.087 forces the 1000, 0.35·0.25 the
//                  2), so the wrap is invisible — the phase after it equals
//                  the unwrapped phase. Bounded at ~12,566 (ulp ≈ 0.001).
//                  A new shader frequency must keep 2000·f integral
//                  (tests/test_interface_face_review_4pr.py extracts every
//                  `uTime *` from the shader sources and checks it).
//   uPacketPhase = fract(2·time), uBurstPhase = fract(6·time): the line
//                  shader's two fract() packet trains, computed here in
//                  float64 (a fract argument cannot be wrapped at any
//                  multiple of π, so they never touch uTime).
export const SHADER_TIME_PERIOD = 4000 * Math.PI;
const _shaderClockOut = { uTime: 0.0, packet: 0.0, burst: 0.0 };
export function shaderClock(t, out = _shaderClockOut) {
    const P = SHADER_TIME_PERIOD;
    out.uTime = ((t % P) + P) % P;
    out.packet = _fract(t * 2.0);
    out.burst = _fract(t * 6.0);
    return out;
}

// ── Frame-time scale (2026-09-20) ──────────────────────────────────
// Every increment and ease in animate() was written as "per 60fps
// frame" (`x += 0.005`, `x += (t - x) * 0.05`, `x *= 0.97`, `(1 / 60)`).
// Stepping once per requestAnimationFrame made the whole face run at
// the DISPLAY's rate: twice as fast on a 120Hz ProMotion screen, and
// visibly slowing / lurching whenever the main thread dropped frames
// (a streaming reply re-rendering markdown, a GC pause) — the
// "sometimes erratic" motion. dtF is the elapsed time in 60fps-frame
// units, so every step below multiplies by it (eases and decays are
// exponentiated: `ease(k)`, `decay(r)`). dtF === 1 at 60Hz, which keeps
// every tuned constant exactly as it was; CLAMPED so a long stall (tab
// switch, debugger) is a small catch-up, never a jump.
let _lastFrameTs = null;
const DT_MIN = 0.25, DT_MAX = 3.0;
let dtF = 1.0;
const ease = (k) => 1.0 - Math.pow(1.0 - k, dtF);
const decay = (r) => Math.pow(r, dtF);

// ── Medusa kinematics (2026-07-28) ─────────────────────────────────
// The face is no longer an amorphous point cloud: it is a spectral
// MEDUSA — a bell canopy, trailing tentacle strands, and a hot core —
// because "alive" is a property of ANATOMY + intent-like motion, not
// of timing curves layered on random scatter (operator: "it looks
// like random lines and dots animating"). pulsePhase drives medusan
// propulsion: a sharp contraction propagating apex→margin, then a
// long glide — a muscle envelope, deliberately NOT a sine.
let pulsePhase = 0.0;
let flinch = 0.0;        // error recoil — one sharp agitated pulse, decays
let _bob = 0.0;          // swim bob: rises on contraction, sinks on glide
let _bobTarget = 0.0;
let huePhase = 0.0;      // bounded thermal oscillation (see hueDrift)

// ── cube2 (2026-09-20, EXPERIMENT) — the infinite cube ─────────────
// Wireframe cube shells nested at exponentially spaced scales: each is
// BORN tiny and hot at the kernel, grows outward (self-similar
// renormalisation — the vortex's trick, rectilinear) and dissolves as it
// passes the viewer. THE CAMERA NEVER MOVES; a user turn accelerates the
// birth rate (cube2Travel) exactly as the vortex's swallow surges.
//   cube2Flow   = accumulated depth (a shell's phase d = fract(d0 + flow))
//   cube2Travel = eased user-turn engagement 0..1
let cube2Flow = 0.0;
let cube2Travel = 0.0;
// cube2 v2 (2026-09-21, operator: "fix them all" — the seven upgrades):
// the kernel is a tumbling lattice whose spin is INTEGRATED (kernel
// angle = c2KernelSpin, stepped per frame at the current rate); a shell
// inherits the kernel's angle at birth (`sh.tw`, folded to ±45° — a cube's
// own symmetry — so it is bounded) and unwinds toward the stack's alignment as it grows — the
// kernel's rotation propagates outward as a torsion wave carried by the
// shells and made visible by the ties between them. The per-shell twist
// is a first-order decay of a BOUNDED angle, never an accumulator times
// a per-shell factor (the §4JA/§4JK class).
let c2KernelSpin = 0.0;
const _c2KernelM = [1, 0, 0, 0, 1, 0, 0, 0, 1];   // the kernel lattice's rotation this frame
const _c2Order = [];                              // shell indices sorted by depth phase, per frame
const _c2LineStats = { shells: 0, rulings: 0, ties: 0, kernel: 0 };
let c2Roll = 0.0;                                 // the stack's slow roll about the view axis (integrated)
let _c2Wob = 0.0;                                 // the wobble amplitude in force this frame (debug)
// v4 (operator): expansion flow. Contraction (matter streaming edges→
// center) reads as moving BACKWARD; a black-hole fall needs the hole
// AHEAD and the walls expanding outward past the viewer. The singularity
// (dark-red shadow + accretion ring) sits deep on the view axis; stream
// matter EMERGES from its glow (L small) and blooms outward/past the
// camera (L grows exponentially — still a self-similar infinite fall).
// Spawn AT the ring's edge, not inside it: matter emerges THROUGH the
// accretion ring, so the shadow interior keeps only its dim embers and
// reads as a genuinely dark hole.

// ── AI-form state (2026-07-29) ─────────────────────────────────────
let errorState = 0.0;
let targetErrorState = 0.0;
let workingState = 0.0;
let targetWorkingState = 0.0;
// User-turn state (2026-07-28): a SEPARATE signal from workingState.
// workingState is fed by BOTH the user's in-flight request and ambient
// working-class log icons (app.js updateStateFromIcon), which on an
// autonomously busy agent re-arms almost continuously — so driving the
// immersion dive from it held the camera inside the cloud all night.
// The dive belongs to "working for YOU, right now" only; this state is
// set exclusively by sendMessage's lifecycle. Speed/glow/color keep
// riding workingState + the envelope as before.
let userTurnState = 0.0;
let targetUserTurnState = 0.0;

// Shader "energy" (uPulseT in the shaders): faint traveling charge on
// the lines + a mild node glow. Since 2026-07-12 this is DERIVED from
// the smoothed activity envelope every frame — it is no longer a
// per-event shockwave.
let pulseT = 0.0;

// Accent tint — the graph's slow "mood" color, blended toward the hue
// of recent log activity and drained back to neutral over ~10s.
// Neutral default = the violet bridge between the two thermal poles.
const accentColor = new THREE.Color(0x2a1750);
let accentStrength = 0.0;

// Hue drift — a BOUNDED thermal oscillation (±0.07 of the ring, set
// from huePhase each frame): regions warm and cool into each other
// without the anatomical color placement ever scrambling.
let hueDrift = 0.0;

// Reduced-motion damping for the CPU-side kinematics (sway, ripples).
// The shader-side counterpart is the uOrganic uniform.
const CALM = PREFERS_REDUCED_MOTION ? 0.35 : 1.0;

// The forms: `moodtesseract`, the infinite corridor of nested frames, kernel
// lattice, stream and roll, showing the agent's mood through MOOD_LOOKS, and
// `empty`, no face. The header's button (the handheld's too) switches between them:
// animations enabled / disabled (operator, 2026-11-06, who removed cube,
// tesseract, vortex and descent; rollback in interface/face_backups/20261106-premoodonly).
// The internals keep their cube2/C2 names. Ids stay [a-z]: the server's prefs and
// the handheld's form state refuse anything else.
const FORMS = ['moodtesseract', 'empty'];
// A pick stored before the other faces went opens as the one that stayed.
const FORM_ALIASES = { cube2: 'moodtesseract', tesseract: 'moodtesseract', cube: 'moodtesseract', vortex: 'moodtesseract', descent: 'moodtesseract' };
// Default form: the first. A stored pick overrides it, resolved by
// `resolveInitialForm` below; a name that left the roster falls through to it.
let formIndex = FORMS.indexOf('moodtesseract');

// Which form to boot into (2026-09-05). Precedence, highest first:
//   1. the SERVER's record of the last form picked ANYWHERE — server.py
//      injects it as <meta name="ghost-face-form"> once a pick has been
//      saved (POST /api/ui/prefs, app.js rememberFaceForm). This is what
//      makes the face come back the same in the phone PWA, a LAN-IP tab
//      and the Tailscale-name tab, and after Safari's storage purge —
//      localStorage is per-origin, per-browser, per-device, and none of
//      those is "the last face used".
//   2. this browser's localStorage (the pre-2026-09-05 mechanism; also
//      the fallback for a pick whose save never reached the server).
//   3. the default.
// An unknown name at any level (a form since removed, a tampered value)
// falls through to the next: the roster is the only authority. Pure and
// exported so the precedence is EXECUTED under node
// (tests/test_interface_face_prefs_and_status_chip.py), not read.
export function resolveInitialForm(serverForm, storedForm, fallback) {
    for (const raw of [serverForm, storedForm]) {
        const candidate = (typeof raw === 'string' && FORM_ALIASES[raw]) ? FORM_ALIASES[raw] : raw;
        if (typeof candidate === 'string' && FORMS.indexOf(candidate) >= 0) return candidate;
    }
    return fallback;
}
try {
    const _meta = document.querySelector('meta[name="ghost-face-form"]');
    let _stored = null;
    try { _stored = localStorage.getItem('ghost_face_form'); } catch (e) { /* private mode */ }
    formIndex = FORMS.indexOf(resolveInitialForm(
        _meta ? _meta.getAttribute('content') : null, _stored, FORMS[formIndex]));
} catch (e) { /* no DOM */ }
// Reorganization blend: on a form switch the nodes visibly re-assemble
// from their old positions into the new anatomy over ~1.4s.
let formBlend = 1.0;
const _blendFrom = [];
// The scene + camera ride the same blend (2026-10-17, §4PR). Only node
// positions used to ease: scene rotation / position / scale and the
// camera's z and look target HARD-CUT on a switch — the vortex's roll
// (vortexSpin × 0.22) and the tesseract's c2Roll grow without bound, so
// leaving either after ten minutes turned the whole scene up to 133° in
// one frame, and a switch mid-turn snapped the camera 5.0 → 1.3. setForm
// snapshots what is ON SCREEN into _sceneFrom; while formBlend < 1 the
// frame eases from it to the new form's steady values (_sceneT) with the
// node blend's smoothstep — angles by the shortest wrapped delta — and
// from formBlend === 1 on it writes _sceneT exactly: the blend is only a
// transition; a camera-static form (vortex, tesseract) holds z = 5.0 at
// steady state.
const _sceneFrom = { on: false, rx: 0, ry: 0, rz: 0, px: 0, py: 0, s: 0.9, camZ: 5.0, lx: 0, ly: 0, lz: 0 };
const _sceneT = { rx: 0, ry: 0, rz: 0, px: 0, py: 0, s: 0.9, camZ: 5.0, lx: 0, ly: 0, lz: 0 };
const _lookNow = { x: 0, y: 0, z: 0 };   // the look target applied last frame
// The empty form at rest (§4PR): once its blend has finished and the
// last node has faded out, a frame draws nothing — the O(n²) proximity
// pass, the instance uploads and the bloom composite are skipped. The
// scalar envelopes keep running (the rAF loop never stops), so a form
// switch wakes the full frame on the very next tick.
let _emptySettled = false;

// Immersion (2026-07-13): while a USER request is in flight the grid
// "swallows" the camera — the scene scales up around the viewer and the
// camera dollies from its resting z into the middle of the cloud; on
// completion it drifts back out. Driven by the smoothed workingState
// with a DELIBERATELY slower attack/release than workingState itself:
// a 2-second request produces only a subtle lean inward, and the full
// engulfment develops only on sustained work — that asymmetry is what
// makes it read as "alive" instead of a yo-yo. Background agent
// activity (the envelope) does NOT drive immersion on purpose: it
// animates speed/glow all night; the swallow is reserved for "working
// for YOU, right now". Reduced-motion users get a capped slight lean.
let immersion = 0.0;
const IMMERSION_CAP = PREFERS_REDUCED_MOTION ? 0.15 : 1.0;
const CAMERA_REST_Z = 5.0;
const CAMERA_DIVE_Z = 1.3;   // inside the cloud (node shell radius ~2)


// TTS-driven audio level (0..1). Wired by setAudioLevel() from app.js
// when the TTS engine is active; multiplies node jitter subtly so the
// sphere "breathes with the voice."
let audioLevel = 0.0;

// ── Signal layer (2026-09-11) ──────────────────────────────────────
// The face used to read four scalars (working, your-turn, error, the
// activity envelope). These give it more to SAY without a second colour
// axis — temperature stays the one thing the eye must track; what is
// added is gait and event.
//   phase gaits   — the turn ticker's step class → a motion grammar:
//                   search sweeps outward, read draws inward and slows,
//                   tool is a discrete kick, verify tightens toward
//                   stillness, write is a laminar wave toward the viewer.
//   recall spark  — a comet from the periphery igniting one node.
//   verdict       — pass crystallises and holds, refute shudders, stop
//                   exhales.
//   background    — a second, slower breath at the edge while a dream /
//                   self-play turn holds the lock.
//   mood          — a slow baseline shift of the cold pole (hours).
//   gaze          — the body leans toward the composer while you type.
//   error kind    — network flickers, refusal freezes, timeout fades.
export const PHASES = ['search', 'read', 'tool', 'verify', 'write'];
let phase = null;
const gait = { search: 0, read: 0, tool: 0, verify: 0, write: 0 };
let toolPulse = 0.0;
let sweepAngle = 0.0;
let recallSpark = 0.0;
let recallNode = -1;
let recallDir = [0, 1, 0];
let verdict = null;          // 'pass' | 'refute' | 'stop' | null
let verdictEnv = 0.0;
let backgroundBusy = 0.0, targetBackgroundBusy = 0.0;
let moodHue = 0.0, targetMoodHue = 0.0;
let gazeX = 0.0, gazeY = 0.0, targetGazeX = 0.0, targetGazeY = 0.0;
let errorKind = null, errorKindEnv = 0.0;
let idleTwitch = 0.0, idleTwitchNode = -1, idleTwitchAt = 0.0;
let twitchAmp = 0.0;         // rendered twitch: eased attack, not a pop
// The inner state (temperament.js, 2026-10-30): events nudge its drives
// and every kick below is scaled by the gain it returns; `animate` reads
// its outputs (`TS`) once per frame. Rollback: interface/face_backups/.
// Its clock is FRAME-INTEGRATED on the true frame time (deterministic under
// the node harness, which seeds Math.random) and re-syncs FORWARD to the wall
// clock only after a real absence: a gap too long to be rendering between two
// frames, on the frame stamps or on the wall clock. The wall clock covers a paused
// tab and a system sleep (rAF stamps can stop through one, and document.hidden
// stays false). It used to integrate
// the CLAMPED dtF and re-sync whenever it fell 5 s behind, so below 20 fps it
// jumped 5 s about every 20 s (measured at 15 fps) and the whole body lurched.
let _temperNow = Date.now();
let _temperWall = _temperNow;
const TEMPER_ABSENCE_MS = 1000;
function _readIntensity() {
    try { const v = parseFloat(localStorage.getItem('ghost_face_intensity')); return isFinite(v) ? v : null; }
    catch (e) { return null; }
}
const temper = createTemperament({ seed: Math.floor(Math.random() * 4294967296), now: _temperNow,
                                    intensity: _readIntensity() });
// The body offsets applied to the scene LAST frame — taken back off before
// the per-form code reads scene.rotation/position (some forms keep the
// current value as their target), then re-applied after the blend.
const _bodyPrev = { rx: 0, ry: 0, rz: 0, px: 0, py: 0, s: 1 };
let TS = temper.step(0, _temperNow);

// ── Tunables ───────────────────────────────────────────────────────
// Every amplitude the signal layer uses, in one table (the face lab
// that dragged these live was removed 2026-09-12; the table stays so
// tuning is one edit and the test can prove every key is read).
export const TUNE = Object.freeze({
    radialSearch: 0.04,    // search gait: radial expansion
    radialRead: 0.035,     // read gait: radial contraction
    radialVerify: 0.03,    // verify gait: contraction toward stillness
    toolKick: 0.06,        // tool call: radial kick amplitude
    writeWave: 0.10,       // write gait: z-wave amplitude toward the viewer
    flowWrite: 1.2,        // write gait, 'flow' dialect: extra flow speed
    thickenLinks: 0.35,    // read gait, 'thicken' dialect: link radius growth
    sweepHeat: 0.22,       // search gait: palette warmth of the sweep
    sweepSpeed: 1.4,       // search gait: sweep rate (rad/s or units/s)
    stillPass: 0.85,       // verdict pass: stillness
    stillVerify: 0.5,      // verify gait: stillness
    stillRead: 0.3,        // read gait: stillness
    passDim: 0.18,         // verdict pass: luminance hold
    passHold: 0.992,       // verdict pass: envelope decay per frame (~2s)
    shudder: 0.035,        // verdict refute: displacement amplitude
    exhale: 0.035,         // verdict stop: scene-scale swell
    bgBreath: 0.012,       // background busy: scene breath
    bgEdge: 0.05,          // background busy: outer-node radial breath
    recallReach: 2.6,      // recall comet: streak length
    recallFlare: 2.0,      // recall comet: node size flare (peak ×(1+flare))
    gazeY: 0.22,           // composer gaze: look-target drop
    netFlicker: 0.6,       // network error: flicker depth
    timeoutFade: 0.45,     // timeout error: fade depth
    twitch: 0.08,          // idle twitch: displacement amplitude
    flashGain: 2.0,        // tool 'flash' dialect: core/kernel flare gain
    alignGain: 0.7,        // verify 'align' dialect: jitter/churn suppression
});

// ── Dialects (2026-09-11) ──────────────────────────────────────────
// The gaits are one vocabulary; each anatomy speaks it in its own
// grammar. Fields:
//   radialAxis  which components the radial factor touches:
//               all | xz | y (a sheet heaves) | none (crystals do not
//               breathe)
//   search      sweep (azimuth scan) | plane (a plane scanning along y)
//               | ring (an expanding ring from the centre)
//   read        contract | thicken (flow slows, links thicken) | settle
//   tool        kick | flash (core/kernel flare)
//   write       wave (z-wave toward the viewer) | flow (the form's own
//               flow accelerates)
//   verify      still | align (jitter and churn suppressed → the grid
//               snaps true)
export const DIALECT_VALUES = Object.freeze({
    radialAxis: ['all', 'xz', 'y', 'none'],
    search: ['sweep', 'plane', 'ring'],
    read: ['contract', 'thicken', 'settle'],
    tool: ['kick', 'flash'],
    write: ['wave', 'flow'],
    verify: ['still', 'align'],
});
export const DIALECTS = Object.freeze({
    moodtesseract: { radialAxis: 'none', search: 'ring', read: 'settle',   tool: 'flash',  write: 'flow', verify: 'align' },
    empty:        { radialAxis: 'none', search: 'sweep', read: 'settle',   tool: 'kick',   write: 'wave', verify: 'still' },
});
const _DIALECT_DEFAULT = DIALECTS.moodtesseract;
export function dialectFor(form) { return DIALECTS[form] || _DIALECT_DEFAULT; }
// Per-frame derived gait scalars the form branches read (set in animate).
let gaitFlow = 0, gaitThicken = 0, gaitFlash = 0, gaitAlign = 0;

// --- Activity envelope (2026-07-12) -------------------------------
// The face is ALIVE, not reactive-per-event: log lines feed small
// amounts of energy into `activityTarget`; the rendered `activity`
// follows it with a soft attack (~1s) and a slow release (~8s). All
// visuals read the smoothed envelope — a busy agent makes the graph
// drift faster, glow slightly warmer, and rewire more often; a quiet
// agent lets it settle. This replaced the per-log-line triggerPulse()
// shockwave, which strobed the whole scene several times a second the
// moment the live log stream came back to life.
let activityTarget = 0.0;   // raw accumulated energy, decays on its own
let activity = 0.0;         // smoothed envelope the visuals actually use
let accentTarget = 0.0;     // smoothed accent-tint strength target
const _accentBlend = new THREE.Color();

function _fract(x) { return x - Math.floor(x); }

// Medusan propulsion envelope over one cycle x∈[0,1): fast squeeze
// (0→0.16), long smooth release (0.16→0.62), quiescent glide (0.62→1).
// The asymmetry is what reads as muscle rather than oscillation.
function _pulseShape(x) {
    if (x < 0.16) { const s = x / 0.16; return s * s * (3 - 2 * s); }
    if (x < 0.62) { const r = (x - 0.16) / 0.46; return 1 - r * r * (3 - 2 * r); }
    return 0;
}

// Parallax — camera drift toward cursor / device tilt. Never more than
// ±PARALLAX_RANGE units so the chat never visually shifts.
const PARALLAX_RANGE = 0.15;
let parallaxTargetX = 0.0;
let parallaxTargetY = 0.0;
let parallaxCameraBaseZ = 5.0;

const basePositions = [];
const currentPositions = new Array(NODE_COUNT);
const nodeScales = new Float32Array(NODE_COUNT).fill(1.0);
// One stable palette-wheel position per node. Read by the shader (as
// the aSeed instanced attribute) AND by the per-frame line builder so
// each line can gradient between its endpoints' hues.
const nodeSeeds = new Float32Array(NODE_COUNT);

// (2026-10-17, §4PR) The old `destroy()` export was deleted: nothing called
// it (app.js, the ClockworkPi face.html and every test were grepped) and it
// was wrong as written — it left animationFrameId and renderer set (so
// setAnimationPaused(false) would restart a loop on a disposed renderer),
// never disposed the composer / bloom render targets, and left the pointer /
// orientation listeners installed. The face lives as long as the page.

// Shared 5-stop palette lookup, wrapped (last stop blends back into the
// first). `t` is unbounded — fract() puts it on the wheel — so callers
// can just add the drift offset to a per-node seed.
const paletteGLSL = `
uniform vec3 uPal0;
uniform vec3 uPal1;
uniform vec3 uPal2;
uniform vec3 uPal3;
uniform vec3 uPal4;

vec3 palette(float t) {
    float p = fract(t) * 5.0;
    vec3 col = mix(uPal0, uPal1, clamp(p, 0.0, 1.0));
    col = mix(col, uPal2, clamp(p - 1.0, 0.0, 1.0));
    col = mix(col, uPal3, clamp(p - 2.0, 0.0, 1.0));
    col = mix(col, uPal4, clamp(p - 3.0, 0.0, 1.0));
    col = mix(col, uPal0, clamp(p - 4.0, 0.0, 1.0));
    return col;
}
`;

// Spatial hue wave — shared by all three shaders. A slow sine field over
// model-space position makes hue drift travel THROUGH the cloud as
// coherent currents (convection), instead of every node mutating in
// place. uOrganic damps it for reduced-motion users. NB: uses uTime but
// deliberately does NOT declare it — the line fragment shader already
// declares uTime for its pulse math, and GLSL forbids redeclaration, so
// each including shader declares uTime itself exactly once.
const hueWaveGLSL = `
uniform float uOrganic;
uniform float uWaveAmp;

float hueWave(vec3 p) {
    return uOrganic * uWaveAmp
        * (0.085 * sin(uTime * 0.21 + p.x * 0.5 + p.y * 0.35 + p.z * 0.3)
         + 0.045 * sin(uTime * 0.087 - p.y * 0.6 + p.z * 0.4));
}
`;

const nodeVertexShader = `
attribute float aSeed;
uniform float uTime;
uniform float uSweep;
uniform float uSweepAngle;
uniform float uSweepMode;
uniform float uSweepHeat;
uniform float uSweepReach;
uniform float uCenterDim;
uniform float uCenterXY;
uniform float uFormDim;
uniform float uWorkingState;
uniform float uErrorState;
uniform float uPulseT;
uniform float uAudioLevel;
uniform float uHueDrift;
uniform vec3 uBaseColor;
uniform vec3 uErrorColor;
uniform vec3 uAccentColor;
uniform float uAccentStrength;
${paletteGLSL}
${hueWaveGLSL}
varying vec3 vColor;
varying vec2 vUv;
varying float vDepthFade;
varying float vNearFade;

void main() {
    vUv = uv;

    // Extract instance position
    vec3 instancePos = (instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // Extract scale from instance matrix (assuming isotropic)
    float scale = length(vec3(instanceMatrix[0][0], instanceMatrix[0][1], instanceMatrix[0][2]));

    // Billboard logic: apply local face offset directly in view space.
    // Audio level gently inflates billboard size so the sphere feels
    // like it's breathing with the voice during TTS playback. The tiny
    // per-node shimmer keeps every star faintly restless — organic, not
    // frozen — at an amplitude below conscious notice.
    float sizeBoost = 1.0 + uAudioLevel * 0.25 + uPulseT * 0.15
        + uOrganic * 0.06 * sin(uTime * 1.7 + aSeed * 43.0);
    vec4 mvPosition = modelViewMatrix * vec4(instancePos, 1.0);
    mvPosition.xy += position.xy * scale * sizeBoost;

    gl_Position = projectionMatrix * mvPosition;

    // Depth fade: 0 near the camera .. 1 far. Lets the fragment dim
    // distant nodes so the cloud reads as a 3D volume, not a flat sheet.
    vDepthFade = clamp((-mvPosition.z - 2.5) / 7.0, 0.0, 1.0);

    // Near fade: nodes dissolve as they approach the camera instead of
    // exploding into screen-filling quads (or popping when they cross
    // the camera plane). Essential for the immersion dive, harmless at
    // rest (nothing sits within 1.4 units of the resting camera).
    vNearFade = smoothstep(0.3, 1.4, -mvPosition.z);

    // Per-node thermal hue: aSeed anchors this node on the ring (seeds
    // are SPATIAL at init, so neighbours share a temperature region),
    // uHueDrift slides the whole field between the poles, and hueWave
    // sends slow currents of warmth/cold traveling through the cloud —
    // the blue↔red mutation the theme is built around.
    // Search gait (2026-09-11): a rotating azimuth window warms the
    // nodes it crosses — a scan sweeping the body, no CPU seed writes.
    // Dialects: 0 = azimuth sweep, 1 = a plane scanning along y,
    // 2 = a ring expanding from the centre (uSweepAngle is the scan
    // position in every mode; the CPU advances it).
    float az = atan(instancePos.z, instancePos.x);
    float dAz = abs(mod(az - uSweepAngle + 3.14159, 6.28318) - 3.14159);
    float scan = fract(uSweepAngle / 6.28318);
    float dPlane = abs(instancePos.y - (scan * 4.0 - 2.0));
    float dRing = abs(length(instancePos) - scan * uSweepReach);
    float sweepD = uSweepMode < 0.5 ? dAz : (uSweepMode < 1.5 ? dPlane : dRing);
    float sweepW = uSweepMode < 0.5 ? 0.9 : 0.45;
    // 1 − smoothstep(lo, hi, x): GLSL ES leaves smoothstep with edge0 ≥ edge1
    // undefined (§4PR); this is the reversed-edge form's exact value.
    float sweepHeat = uSweep * uSweepHeat * (1.0 - smoothstep(0.0, sweepW, sweepD));
    vec3 jewel = palette(aSeed + uHueDrift + hueWave(instancePos) + sweepHeat);

    // Life: each node breathes between a dim floor and its full hue.
    // The breath itself travels as a slow luminance wave (uTime term)
    // instead of being a frozen spatial pattern.
    float colorMix = sin(instancePos.x * 1.3 + instancePos.y * 1.1
        + uTime * uOrganic * 0.35 + uWorkingState) * 0.5 + 0.5;
    vec3 dimCol = uBaseColor + jewel * 0.22;
    vec3 mixCol = mix(dimCol, jewel, colorMix);
    // A busy agent saturates slightly toward the full hue.
    mixCol *= (1.0 + uWorkingState * 0.25);

    vec3 col = mix(mixCol, uErrorColor, uErrorState);
    // Per-node falloff for the accent tint: nodes nearer the world
    // origin catch the accent more strongly, so the tint reads as a
    // radiating wave rather than a flat filter. Cheap (no extra uniform
    // needed — we use the instancePos we already have).
    float distFromCenter = length(instancePos);
    float accentFalloff = exp(-distFromCenter * 0.35);
    // Cap lowered 0.85 → 0.5 (2026-07-28): at the old strength a busy
    // spell washed the whole field into one flat accent color, erasing
    // the blue↔red poles the thermal scheme is built on. The mood tint
    // should color the weather, not replace the climate.
    col = mix(col, uAccentColor, clamp(uAccentStrength * accentFalloff, 0.0, 0.5));
    // Brightness boost during an active pulse — subtle; the bloom pass
    // amplifies this considerably.
    col *= (1.0 + uPulseT * 0.4);
    // Center dimming: additive stacking is densest near the origin — a
    // pile of dark-red quads sums past saturation and reads hot PINK.
    // Dimming each contribution radially keeps the red channel dominant
    // so the center stays CRIMSON. Strength is per-form (uCenterDim).
    // uCenterXY switches the dim metric to SCREEN-radial (xy) distance —
    // the vortex's hot zone sits on the view axis at depth, not at the
    // 3D origin.
    float cdist = mix(length(instancePos), length(instancePos.xy), uCenterXY);
    col *= mix(1.0, smoothstep(0.2, 1.3, cdist), uCenterDim);
    // Per-form master luminance: forms meant to BLEND into the
    // background (vortex) emit less light overall — structure and
    // motion carry the visibility, not brightness.
    col *= uFormDim;
    vColor = col;
}
`;

const nodeFragmentShader = `
varying vec3 vColor;
varying vec2 vUv;
varying float vDepthFade;
varying float vNearFade;
void main() {
    float d = distance(vUv, vec2(0.5)) * 2.0; // scaled 0 to 1
    if (d > 1.0) discard;

    // Soft quadratic falloff. Dimmed 0.8→0.68 with the medusa anatomy:
    // the body is locally DENSE (membrane, strands), so additive
    // stacking multiplied apparent brightness — the dark-first theme is
    // held at the fragment level, not just in the palette.
    float intensity = pow(1.0 - d, 2.0) * 0.68;
    // Bright core (also trimmed for the same stacking reason)
    float core = pow(1.0 - d, 8.0) * 1.35;

    float alpha = (intensity + core) * vNearFade;
    // Distance dimming: far nodes glow less and fade out, near nodes pop.
    float depthDim = mix(1.0, 0.35, vDepthFade);
    gl_FragColor = vec4(vColor * alpha * depthDim, alpha * mix(1.0, 0.7, vDepthFade));
}
`;

const lineVertexShader = `
attribute float aLightPass;
attribute float aLineHue;
attribute float aLineFade;
varying float vLightPass;
varying float vLineHue;
varying float vLineFade;
varying float vLineDepth;
varying float vLineNear;
varying vec3 vLinePos;

void main() {
    vLightPass = aLightPass;
    vLineHue = aLineHue;
    vLineFade = aLineFade;
    vLinePos = position;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vLineDepth = clamp((-mvPosition.z - 2.5) / 7.0, 0.0, 1.0);
    // Near fade — see the node shader; lines whose endpoint passes the
    // camera dissolve instead of slashing across the whole screen.
    vLineNear = smoothstep(0.3, 1.4, -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
}
`;

const lineFragmentShader = `
uniform float uTime;
uniform float uPacketPhase;
uniform float uBurstPhase;
uniform float uCenterDim;
uniform float uCenterXY;
uniform float uFormDim;
uniform float uWorkingState;
uniform float uErrorState;
uniform float uPulseT;
uniform float uHueDrift;
uniform float uDive;
uniform vec3 uBaseColor;
uniform vec3 uErrorColor;
uniform vec3 uAccentColor;
uniform float uAccentStrength;
uniform float uSweep;
uniform float uSweepAngle;
uniform float uSweepHeat;
uniform float uSweepReach;
uniform float uLineSweep;
${paletteGLSL}
${hueWaveGLSL}
varying float vLightPass;
varying float vLineHue;
varying float vLineFade;
varying float vLineDepth;
varying float vLineNear;
varying vec3 vLinePos;

void main() {
    // uPacketPhase = fract(2·time), computed in float64 on the CPU (see
    // shaderClock) — fract(a − fract(b)) ≡ fract(a − b).
    float gradient = fract(vLightPass * 1.5 - uPacketPhase);

    // Smooth sharp front tail for data packets traveling along lines.
    // (1 − smoothstep(lo, hi, x) is the reversed-edge smoothstep GLSL ES
    // leaves undefined, written portably — §4PR.)
    float pulse = smoothstep(0.0, 0.5, gradient) * (1.0 - smoothstep(0.9, 1.0, gradient));

    // Shockwave: a second, faster burst rides the lines when pulseT is
    // non-zero. Disappears completely at pulseT==0 so idle lines are
    // unchanged.
    float burstX = fract(vLightPass * 1.5 - uBurstPhase - (1.0 - uPulseT));
    float burst = smoothstep(0.0, 0.25, burstX) * (1.0 - smoothstep(0.85, 1.0, burstX));

    // Each line is a GRADIENT between its two endpoint nodes' thermal
    // hues (vLineHue interpolates the endpoint seeds), riding the same
    // traveling hue wave as the nodes so links and their endpoints stay
    // in the same temperature current.
    // Search scan on the LINES (cube2 v2, uLineSweep = 1): the ring
    // expanding from the kernel lights whole shells as it crosses them
    // — the search gait reads as a scan sweeping the infinite stack.
    float scan = fract(uSweepAngle / 6.28318);
    float ringR = scan * uSweepReach;
    float dRing = abs(length(vLinePos) - ringR);
    // Ring width grows with its radius (a fixed 0.45 is the whole kernel
    // at r 0.5 and a hairline on a shell of half-size 6); warmth at 0.7 of
    // the node sweep's — a passing highlight on the shell it crosses, not
    // a fill (the near shell alone covers most of the screen).
    float ringW = 0.25 + 0.06 * ringR;
    float lineSweep = uLineSweep * uSweep * uSweepHeat * 0.7 * (1.0 - smoothstep(0.0, ringW, dRing));
    vec3 jewel = palette(vLineHue + uHueDrift + hueWave(vLinePos) + lineSweep);
    float colorMix = sin(vLightPass * 10.0 + uTime * uOrganic * 0.4 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor + jewel * 0.18, jewel, colorMix);

    vec3 col = mix(mixCol, uErrorColor, uErrorState * 0.8);
    col = mix(col, uAccentColor, clamp(uAccentStrength * 0.6, 0.0, 0.45));

    float alpha = mix(0.30 + uWorkingState * 0.18, 1.0, pulse);
    alpha = max(alpha, burst * uPulseT);
    alpha = min(1.0, alpha + 1.5 * lineSweep);
    alpha *= vLineNear;
    // Explicit-edge forms (cube2) taper their shells in at birth and out
    // as they pass the viewer; proximity links carry 1.0.
    alpha *= vLineFade;
    // Inside the cloud MANY lines stack additively right in front of
    // the camera — dim each one so the sum stays comfortable.
    float diveDim = 1.0 - 0.30 * uDive;
    // Distance dimming so far links recede behind near ones.
    float depthDim = mix(1.0, 0.4, vLineDepth);
    // Center dimming — see the node shader: the central line hairball is
    // the main additive pile-up; keep it dark so hues stay saturated.
    float ldist = mix(length(vLinePos), length(vLinePos.xy), uCenterXY);
    float centerFade = mix(1.0, smoothstep(0.15, 1.25, ldist), uCenterDim);
    // Master per-form luminance: color scales fully, alpha partially so
    // faint far links don't vanish entirely.
    float formA = mix(1.0, uFormDim, 0.6);
    gl_FragColor = vec4(col * (1.0 + uPulseT * 0.3) * depthDim * vLineNear * diveDim * centerFade * uFormDim, alpha * diveDim * centerFade * formA * mix(1.0, 0.55, vLineDepth));
}
`;


// ── Anatomy builders ───────────────────────────────────────────────
// Each fills basePositions + nodeSeeds for exactly NODE_COUNT nodes.
// The thermal poles become BODY PLAN in every form: cold blue outer
// structure, violet transitions, hot arterial strands and core.

// Nodes a form draws with EXPLICIT edges instead of the proximity web
// (cube2's shells): the O(n²) pass skips them, the edge emitter marks
// them connected. Reset on every build.
const _noProx = new Uint8Array(NODE_COUNT);

function _buildAnatomy() {
    basePositions.length = 0;
    _noProx.fill(0);
    const f = FORMS[formIndex];
    if (f === 'empty') _buildEmpty();
    else _buildCube2();
}

// Form E — EMPTY: no face at all. Nodes park on a sparse far sphere
// (golden-angle spacing ≈ 2.7 units, well past the 1.3-unit link radius,
// so nothing connects and every unlinked node scales to zero). Chosen
// over simply hiding the meshes so the reorganization blend still works:
// cycling INTO empty disperses the face beyond the screen edges, and
// cycling OUT materializes the next form from the void.
function _buildEmpty() {
    for (let i = 0; i < NODE_COUNT; i++) {
        const cosP = 2 * ((i + 0.5) / NODE_COUNT) - 1;
        const sinP = Math.sqrt(Math.max(0, 1 - cosP * cosP));
        const phi = i * 2.399963;
        basePositions.push({
            kind: 8,
            hx: 12.0 * sinP * Math.cos(phi),
            hy: 12.0 * cosP,
            hz: 12.0 * sinP * Math.sin(phi),
        });
        nodeSeeds[i] = 0.3;
    }
}

// Form F — CUBE (named `lattice` until 2026-09-20): the weight tensor.
// A slowly tumbling crystal grid, cell edge tuned just under the
// (form-tightened) link radius so ONLY axis-neighbors weave — a literal
// wireframe tensor, the first angular silhouette. Cold at the corners,
// warming toward the middle; diagonal activation waves and a drifting
// hot attention kernel do the living.
// Form G — CUBE2 (2026-09-20, experiment): the infinite cube.
// C2_SHELLS wireframe cube shells share one exponential depth axis:
// a shell at phase d ∈ [0,1) has half-size L = C2_L0·e^(C2_K·d) —
// born at C2_L0 inside the kernel, grown past the camera by d→1 —
// and its phase advances with cube2Flow, wrapping invisibly (node size
// and edge alpha taper to zero over the last fifth, so the wrap is
// never a pop even on a 21:9 screen where a cube's back face can never
// leave the frame). Each shell carries 8 corner + 12 mid-edge NODES;
// its 12 edges are drawn as 6-segment polylines by _emitCube2Edges,
// bent by a self-similar wobble field (in unit-cube coordinates, so a
// shell keeps its shape as it grows) and pulled toward the kernel
// while small — cubes are born where the kernel is and straighten as
// they grow. Shells twist a little with depth (a slow spiral) and
// breathe on the pulse clock, the breath propagating outward.
// v2 (2026-09-21): ten shells (was eleven — the lattice kernel takes the
// nodes), a 3×3×3 lattice kernel (2×2×2 on mobile), ties between
// consecutive shells, per-edge packet phases, a heat wave on the birth
// clock, the search scan on the lines, and the verdict grammar.
// v3 (2026-09-21, operator: "kinda boring"): the shells are LINE-ONLY
// (a wireframe needs no nodes), so the stack is 24 deep (12 mobile) — a
// long corridor of frames converging on the kernel instead of ten hoops;
// corner nodes ride every other shell; every face carries a "+" ruling
// (paneled data-cubes, not hollow frames); the freed nodes are a STREAM
// of matter riding the expansion at spread speeds and spiralling with
// the twist — the vortex's swallow, inside the cube; the whole stack
// rolls slowly; and the kernel is no longer centre-dimmed.
const C2_SHELLS = IS_MOBILE ? 12 : 24;
const C2_NODE_EVERY = 2;                          // corner nodes on every Nth shell
const C2_RULINGS = !IS_MOBILE;                    // the face "+" rulings
const C2_KERNEL_N = IS_MOBILE ? 2 : 3;            // lattice nodes per axis
const C2_KERNEL = C2_KERNEL_N * C2_KERNEL_N * C2_KERNEL_N;
const C2_KERNEL_R = 0.16;                         // lattice half-size at rest
const C2_TIE_GAIN = 0.65;                         // tie edges vs shell edges
const C2_KERNEL_LINE_GAIN = 2.0;
const C2_RULING_GAIN = 0.55;                      // rulings vs edges
const C2_STREAM_SPIRAL = 0.9;                     // rad of spiral over a particle's life
const C2_TW_DECAY = 0.997;                        // a shell's inherited twist unwinds (per 60fps frame)
const C2_TW_DECAY_PASS = 0.975;                   // …much faster while a pass crystallises
const C2_L0 = 0.08;                       // newborn half-size
const C2_K = Math.log(100);               // L(1) = 8.0 — well past the camera
const C2_TILT_X = 0.24, C2_TILT_Y = 0.42; // base view tilt (3D silhouette)
const C2_TWIST = 0.70;                    // total z-twist across the depth
const C2_LINE_GAIN = 1.8;                 // shell edges are single long lines — lift them
const C2_SEGS = IS_MOBILE ? 4 : 6;        // polyline segments per edge
// Edge table: [axis, sign of the 1st other coord, sign of the 2nd].
const C2_EDGES = [];
for (let a = 0; a < 3; a++)
    for (const sb of [-1, 1]) for (const sc of [-1, 1]) C2_EDGES.push([a, sb, sc]);
const _c2Shells = [];                     // per-shell frame state (per frame)
let _c2KernelBase = 0;                    // index of the first kernel node
let _c2StreamCount = 0;                   // stream grains built (the nodes the shells and kernel leave)
function _buildCube2() {
    _c2Shells.length = 0;
    let n = 0;
    for (let k = 0; k < C2_SHELLS; k++) {
        _c2Shells.push({
            // Near-even phases with a little jitter — a metronomic
            // recession read as machined, not grown.
            d0: (k + 0.5 + (Math.random() - 0.5) * 0.35) / C2_SHELLS,
            ph: Math.random() * Math.PI * 2,  // wobble personality (re-rolled at wrap)
            prevD: 0, d: 0, L: 1, fade: 0, seed: 0.3, wob: 0,
            tw: 0,                            // inherited kernel twist, unwinding (v2)
            cx: 0, cy: 0, cz: 0,
            m: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        });
        // 8 corner nodes on every C2_NODE_EVERY-th shell (v3: the edges
        // are explicit lines and need no nodes; the corners are sparks).
        if (k % C2_NODE_EVERY === 0) {
            for (let c = 0; c < 8; c++, n++) {
                basePositions.push({ kind: 0, k, ux: (c & 1) ? 1 : -1, uy: (c & 2) ? 1 : -1, uz: (c & 4) ? 1 : -1, sz: 1.0 });
                nodeSeeds[n] = 0.3; _noProx[n] = 1;
            }
        }
    }
    // The kernel: a hot LATTICE at the origin every shell is born from —
    // C2_KERNEL_N³ sites in unit coordinates, tumbling on c2KernelSpin;
    // its edges are drawn explicitly (see _emitCube2Edges), so these
    // nodes skip the proximity pass like the shells.
    _c2KernelBase = n;
    for (let iz = 0; iz < C2_KERNEL_N; iz++)
        for (let iy = 0; iy < C2_KERNEL_N; iy++)
            for (let ix = 0; ix < C2_KERNEL_N; ix++, n++) {
                const u = (v) => C2_KERNEL_N > 1 ? (v / (C2_KERNEL_N - 1)) * 2 - 1 : 0;
                basePositions.push({
                    kind: 2, lx: u(ix), ly: u(iy), lz: u(iz),
                    jit: Math.random() * Math.PI * 2,
                    sz: 0.85,
                });
                nodeSeeds[n] = 0.58 + Math.random() * 0.04;
                _noProx[n] = 1;
            }
    // The stream (v3): matter born at the kernel, riding the same
    // exponential expansion as the shells — each grain at its OWN speed
    // (a wide spread, so streams shear past each other) on a spiral
    // about the stack's axis, hot at birth and cooling as it recedes;
    // the vortex's swallow, inside the cube. Fills every node the shells
    // and the kernel leave.
    _c2StreamCount = 0;
    while (n < NODE_COUNT) {
        const r = 0.12 + 0.68 * Math.sqrt(Math.random());
        const phi = Math.random() * Math.PI * 2;
        basePositions.push({
            kind: 3,
            ur: r, uphi: phi,
            uz: (Math.random() * 2 - 1) * 0.9,
            d0: Math.random(),
            flowScale: 0.7 + Math.random() * 1.1,
            jit: Math.random() * Math.PI * 2,
            sz: 0.9,
        });
        nodeSeeds[n] = 0.10 + Math.random() * 0.10;
        _noProx[n] = 1;
        _c2StreamCount++;
        n++;
    }
}

// ── Form switching ─────────────────────────────────────────────────
export function getForm() { return FORMS[formIndex]; }

// The full body-plan roster (copy — callers can't mutate the cycle).
// The header's form MENU builds itself from this, so a new form added
// here appears in the picker with no app.js edit (2026-07-29: the
// cycle button stopped scaling at 9 forms — up to 8 clicks × 1.4s
// blends to reach the one you wanted).
export function getForms() { return FORMS.slice(); }

export function setForm(name) {
    const i = FORMS.indexOf(name);
    if (i < 0 || i === formIndex) return FORMS[formIndex];
    // Snapshot current positions so the switch reads as the creature
    // REORGANIZING itself rather than a hard cut.
    _blendFrom.length = 0;
    for (let k = 0; k < NODE_COUNT; k++) {
        _blendFrom.push(currentPositions[k] ? currentPositions[k].clone() : null);
    }
    // …and the scene + camera exactly as they are on screen now (mid-blend
    // values included, so a second switch mid-blend is continuous too).
    if (scene && camera) {
        const F = _sceneFrom;
        F.on = true;
        // without the temperament's body: it is re-applied on top of the blend
        const B = _bodyPrev;
        F.rx = scene.rotation.x - B.rx; F.ry = scene.rotation.y - B.ry; F.rz = scene.rotation.z - B.rz;
        F.px = scene.position.x - B.px; F.py = scene.position.y - B.py; F.s = scene.scale.x / B.s;
        F.camZ = camera.position.z;
        F.lx = _lookNow.x; F.ly = _lookNow.y; F.lz = _lookNow.z;
    }
    _emptySettled = false;
    formIndex = i;
    _buildAnatomy();
    if (instancedMesh) instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    formBlend = 0.0;
    try { localStorage.setItem('ghost_face_form', FORMS[formIndex]); } catch (e) {}
    return FORMS[formIndex];
}

// ── Signal-layer hooks (2026-09-11) ────────────────────────────────

// The ticker's step class for the running turn, or null between turns.
export function setPhase(name) {
    phase = PHASES.indexOf(name) >= 0 ? name : null;
    return phase;
}

// One discrete kick per tool invocation.
export function noteToolCall() {
    toolPulse = Math.min(1.0, toolPulse + 0.6 * temper.event('tool'));
    return toolPulse;
}

// A memory / knowledge-base hit: a comet from the periphery ignites one
// node. Cross-form.
export function noteRecall() {
    recallSpark = Math.min(1.0, 0.45 + 0.45 * temper.event('recall'));
    recallNode = Math.floor(Math.random() * NODE_COUNT);
    const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    recallDir = [Math.sin(ph) * Math.cos(th), Math.cos(ph), Math.sin(ph) * Math.sin(th)];
    return recallNode;
}

// How the turn ended shapes the release: 'pass' crystallises and holds,
// 'refute' shudders and re-weaves, anything else exhales.
export function noteVerdict(kind) {
    verdict = kind === 'pass' || kind === 'refute' ? kind : 'stop';
    const _g = temper.event('verdict:' + verdict);
    verdictEnv = Math.min(1.0, 0.6 + 0.4 * _g);
    if (verdict === 'refute') flinch = Math.min(1.0, flinch + 0.5 * _g);
    return verdict;
}

// A dream / self-play turn holds the lock: a second, slower breath at
// the edge — "busy with itself", visible without a spinner.
export function setBackgroundBusy(busy) {
    targetBackgroundBusy = busy ? 1.0 : 0.0;
}

// Mood → how the corridor carries itself, on `moodtesseract` only (2026-11-06,
// prototype: three contrasting moods; any other label looks neutral). The mood is
// a slow baseline under the events: only rates and sizes, eased over seconds,
// nothing that moves per frame, so it cannot bring the jitter back (§4TF).
//   flow    shells' birth rate      spin    kernel's turn     pulse   breath's rate
//   breath  breath's depth          kernel  kernel's size     wob     shells' wobble
//   bright  overall light           hue     colour shift      drift   the idle colour swing
//   heat    the corridor's gradient: above 0 the kernel's red reaches further out,
//           below 0 the whole corridor runs colder. Sliding the hue alone did not
//           read: irritable at full strength never looked red.
//   droop   scene height            gap     time between idle actions
//   doze    extra doze weight       react   how hard events land
//   ring    the mood's own five colours, in the order of COLORS.palette: the
//           outer shells sit near the first, the kernel near the fourth. Each
//           stays in one family and dark, as the thermal ring does, because
//           additive blending and bloom lift them a long way.
const MOOD_NEUTRAL = Object.freeze({ flow: 1, spin: 1, pulse: 1, breath: 1, kernel: 1, wob: 1,
    bright: 1, hue: 0, drift: 1, heat: 0, droop: 0, gap: 1, doze: 0, react: 1,
    ring: COLORS.palette.map((c) => new THREE.Color(c)) });
const MOOD_FIELDS = Object.keys(MOOD_NEUTRAL).filter((f) => f !== 'ring');
const MOOD_RINGS = {
    cheerful:  ['#0a4a50', '#125a7a', '#5a2a86', '#8a3a24', '#56204a'],   // sunrise: teal, cerulean, orchid, coral, rose
    irritable: ['#1c0a2c', '#3c0e34', '#6e0c28', '#8e2c0a', '#4e0f12'],   // fire: soot, wine, crimson, ember, oxblood
    tired:     ['#123f4c', '#27436e', '#4c4080', '#5e3a66', '#2e2a50'],   // dusk: slate teal, dusk blue, lavender, mauve
};
const MOOD_LOOKS = {
    cheerful: { flow: 1.5, spin: 1.6, pulse: 1.25, breath: 1.7, kernel: 1.2, wob: 1.2,
        bright: 1.3, hue: 0.03, drift: 0.5, heat: 0.2, droop: 0.06, gap: 0.6, doze: 0, react: 1.1 },
    irritable: { flow: 1.35, spin: 2.6, pulse: 1.7, breath: 1.8, kernel: 0.7, wob: 0.35,
        bright: 1.1, hue: 0.02, drift: 0.3, heat: 0.8, droop: 0, gap: 1, doze: 0, react: 1.5 },
    tired: { flow: 0.4, spin: 0.3, pulse: 0.5, breath: 1.3, kernel: 0.9, wob: 1.0,
        bright: 0.75, hue: 0, drift: 0.2, heat: -0.22, droop: -0.2, gap: 1.5, doze: 3, react: 0.6 },
};
// Share of the look at each intensity: 1 barely there, 2 clear, 3 unmistakable.
const MOOD_STRENGTH = { 1: 0.35, 2: 0.7, 3: 1 };
// Pure. An unknown label or intensity 0 is neutral; an absent intensity reads as 2.
export function moodLook(label, intensity) {
    const key = String(label || '').toLowerCase();
    const look = Object.prototype.hasOwnProperty.call(MOOD_LOOKS, key) ? MOOD_LOOKS[key] : null;
    const n = intensity == null ? 2 : Number(intensity);
    const k = MOOD_STRENGTH[n];
    if (!look || !k) return MOOD_NEUTRAL;
    const out = {};
    for (const f of MOOD_FIELDS) out[f] = MOOD_NEUTRAL[f] + (look[f] - MOOD_NEUTRAL[f]) * k;
    out.ring = MOOD_NEUTRAL.ring.map((c, i) => new THREE.Color(c).lerp(new THREE.Color(MOOD_RINGS[key][i]), k));
    // Reduced motion keeps a steady tempo, as the temperament does; the mood
    // still shows in light and colour.
    if (PREFERS_REDUCED_MOTION) for (const f of ['flow', 'spin', 'pulse', 'breath', 'wob']) out[f] = MOOD_NEUTRAL[f];
    return out;
}
const _moodNow = { ...MOOD_NEUTRAL };     // the eased look, read per frame
let _moodLive = MOOD_NEUTRAL, _moodPreview = null;
// Try a look by hand, from the console: __ghostFace.previewMood('irritable', 3);
// previewMood() goes back to the agent's real mood. Not saved anywhere.
export function previewMood(label, intensity) {
    _moodPreview = label == null ? null : moodLook(label, intensity == null ? 3 : intensity);
    return { ...(_moodPreview || _moodLive) };
}

// Mood → a slow baseline shift of the COLD pole only (never competes
// with the hot pole). Pure mapping; executed under node. §4TC: the labels
// are the agent's nine moods — low energy sits cooler (deeper blue, toward
// the plum bridge), high energy warmer (indigo → violet); "neutral" and any
// unknown label are 0. `moodHueScale` lets the intensity (1–3) scale the
// shift: half at 1, as given at 2, half again at 3 — always within ±0.07.
export function moodHueFor(label) {
    switch (String(label || '').toLowerCase()) {
        case 'content': return -0.015;     // settled blue
        case 'calm': return -0.03;         // deeper, settled blue
        case 'down': return -0.035;        // deeper still
        case 'bored': return -0.04;        // toward the plum bridge — resting
        case 'tired': return -0.045;       // the plum bridge
        case 'cheerful': return 0.025;     // indigo
        case 'curious': return 0.03;       // indigo, leaning violet
        case 'restless': return 0.04;      // violet
        case 'irritable': return 0.045;    // violet, nearly warm
        default: return 0.0;
    }
}
export function moodHueScale(intensity) {
    const n = Number(intensity);
    return n === 1 ? 0.5 : n === 3 ? 1.5 : 1.0;      // 2, absent or unreadable → as given
}
export function setMoodHue(label, intensity) {
    targetMoodHue = moodHueFor(label) * moodHueScale(intensity);
    // A failed health poll passes null: keep the last look through the blip (it
    // would drop to neutral within seconds and come back on the next good poll).
    if (label != null) _moodLive = moodLook(label, intensity);
    return targetMoodHue;
}

// The body leans toward the composer while you type (pre-turn posture).
export function setComposerGaze(active) {
    targetGazeY = active ? -TUNE.gazeY : 0.0;
    targetGazeX = 0.0;
}

// Error KIND shapes the flinch (2026-09-11): a network drop flickers
// with gaps, a refusal freezes, a timeout fades slowly; anything else
// is the generic recoil.
export function errorKindFor(message, type) {
    const m = String(message || '').toLowerCase();
    const t = String(type || '').toLowerCase();
    if (/refus|declin|not allowed|policy|forbidden/.test(m + ' ' + t)) return 'refusal';
    if (/timeout|timed out|deadline/.test(m + ' ' + t)) return 'timeout';
    if (/network|load failed|failed to fetch|disconnect|unreachable|econn|socket/.test(m + ' ' + t)) return 'network';
    return 'generic';
}

export function cycleForm() {
    return setForm(FORMS[(formIndex + 1) % FORMS.length]);
}

export function init() {
    basePositions.length = 0;
    const container = document.getElementById('sphere-container');
    scene = new THREE.Scene();
    scene.background = COLORS.background;

    camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = parallaxCameraBaseZ;

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(container.clientWidth, container.clientHeight);
    // Cap DPR aggressively on mobile (see DPR_CAP).
    _appliedDpr = _pixelRatio();
    renderer.setPixelRatio(_appliedDpr);

    container.appendChild(renderer.domElement);

    const renderTarget = new THREE.WebGLRenderTarget(container.clientWidth, container.clientHeight, {
        type: THREE.HalfFloatType, format: THREE.RGBAFormat, colorSpace: THREE.SRGBColorSpace,
    });

    composer = new EffectComposer(renderer, renderTarget);
    const renderScene = new RenderPass(scene, camera);
    // strength, radius, threshold. Wider radius + lower threshold gives a
    // softer, dreamier halo than the previous tight bloom.
    bloomPass = new UnrealBloomPass(
        new THREE.Vector2(container.clientWidth, container.clientHeight),
        1.3 * BLOOM_SCALE, 0.55, 0.05
    );

    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    // Size the composer through its own setSize (§4PR). EffectComposer
    // takes a caller-built target's width/height AS its CSS size, so the
    // CSS-sized target above made both ping-pong buffers render at 1x on
    // a 2x display until the first resize. setSize(css) re-allocates them
    // (and every pass) at css × the renderer's pixel ratio.
    composer.setSize(container.clientWidth, container.clientHeight);

    _buildAnatomy();
    for (let i = 0; i < NODE_COUNT; i++) currentPositions[i] = new THREE.Vector3();

    // Shared palette uniforms — same THREE.Color objects on both
    // materials, so a future live re-theme (mutating COLORS.palette)
    // propagates everywhere.
    const paletteUniforms = () => ({
        uPal0: { value: COLORS.palette[0] },
        uPal1: { value: COLORS.palette[1] },
        uPal2: { value: COLORS.palette[2] },
        uPal3: { value: COLORS.palette[3] },
        uPal4: { value: COLORS.palette[4] },
        uHueDrift: { value: 0.0 },
    });

    // Nodes (Instanced Mesh)
    const nodeGeom = new THREE.PlaneGeometry(0.12, 0.12);
    // Per-instance palette seed (divisor-1 attribute on the shared quad).
    nodeGeom.setAttribute('aSeed', new THREE.InstancedBufferAttribute(nodeSeeds, 1));
    nodeMaterial = new THREE.ShaderMaterial({
        vertexShader: nodeVertexShader,
        fragmentShader: nodeFragmentShader,
        uniforms: {
            uTime: { value: 0.0 },
            uOrganic: { value: PREFERS_REDUCED_MOTION ? 0.25 : 1.0 },
            uWaveAmp: { value: 1.0 },
            uCenterDim: { value: 0.0 },
            uCenterXY: { value: 0.0 },
            uFormDim: { value: 1.0 },
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uAudioLevel: { value: 0.0 },
            uSweep: { value: 0.0 },
            uSweepAngle: { value: 0.0 },
            uSweepMode: { value: 0.0 },
            uSweepHeat: { value: TUNE.sweepHeat },
            uSweepReach: { value: 2.4 },
            uBaseColor: { value: COLORS.nodeBase },
            uErrorColor: { value: COLORS.nodeError },
            uAccentColor: { value: accentColor },
            uAccentStrength: { value: 0.0 },
            ...paletteUniforms(),
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    instancedMesh = new THREE.InstancedMesh(nodeGeom, nodeMaterial, NODE_COUNT);
    scene.add(instancedMesh);

    // Lines (Line Segments)
    lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(MAX_LINES * 2 * 3);
    const lineUvs = new Float32Array(MAX_LINES * 2);
    const lineHues = new Float32Array(MAX_LINES * 2);
    const lineFades = new Float32Array(MAX_LINES * 2).fill(1.0);

    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('aLightPass', new THREE.BufferAttribute(lineUvs, 1));
    lineGeometry.setAttribute('aLineHue', new THREE.BufferAttribute(lineHues, 1));
    lineGeometry.setAttribute('aLineFade', new THREE.BufferAttribute(lineFades, 1));

    lineMaterial = new THREE.ShaderMaterial({
        vertexShader: lineVertexShader,
        fragmentShader: lineFragmentShader,
        uniforms: {
            uTime: { value: 0 },
            uPacketPhase: { value: 0.0 },
            uBurstPhase: { value: 0.0 },
            uOrganic: { value: PREFERS_REDUCED_MOTION ? 0.25 : 1.0 },
            uWaveAmp: { value: 1.0 },
            uCenterDim: { value: 0.0 },
            uCenterXY: { value: 0.0 },
            uFormDim: { value: 1.0 },
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uDive: { value: 0.0 },
            uSweep: { value: 0.0 },
            uSweepAngle: { value: 0.0 },
            uSweepHeat: { value: TUNE.sweepHeat },
            uSweepReach: { value: 2.4 },
            uLineSweep: { value: 0.0 },
            uBaseColor: { value: COLORS.lineBase },
            uErrorColor: { value: COLORS.lineError },
            uAccentColor: { value: accentColor },
            uAccentStrength: { value: 0.0 },
            ...paletteUniforms(),
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    linesMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(linesMesh);


    // Scale scene down by 10%
    scene.scale.set(0.9, 0.9, 0.9);

    window.addEventListener('resize', handleResize);
    _watchPixelRatio();
    _installParallaxListeners();

    animate();
}

function handleResize() {
    if (!camera || !renderer || !composer) return;
    const container = document.getElementById('sphere-container');
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    // A DPR change (window dragged to another monitor, browser zoom)
    // reaches the renderer, the composer's buffers and the motes' point
    // size — capped exactly as at init (§4PR).
    const pr = _pixelRatio();
    if (pr !== _appliedDpr) {
        _appliedDpr = pr;
        renderer.setPixelRatio(pr);
        if (typeof composer.setPixelRatio === 'function') composer.setPixelRatio(pr);
    }
    renderer.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
    // Resizing clears the canvas; the resting empty form skips its
    // composite, so redraw its (black) frame once here.
    if (_emptySettled) composer.render();
}

// The pixel ratio the renderer, composer and motes were last given.
let _appliedDpr = 1;
// A DPR change does not always fire 'resize' (zoom does; a drag between
// monitors of equal CSS size may not), so also watch the resolution
// media query for the CURRENT ratio and re-arm on every change.
function _watchPixelRatio() {
    if (typeof window.matchMedia !== 'function') return;
    let mq;
    try { mq = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`); } catch (e) { return; }
    if (!mq || typeof mq.addEventListener !== 'function') return;
    mq.addEventListener('change', () => { handleResize(); _watchPixelRatio(); }, { once: true });
}

// --- Parallax -----------------------------------------------------
//
// Desktop gets a pointermove-driven camera offset. iOS Safari requires
// an explicit permission prompt for DeviceOrientationEvent (iOS 13+),
// and the prompt can only fire in response to a user gesture, so we
// wire the listener behind `tryEnableDeviceOrientation()` which is
// called on the first user tap. Stored permission result is cached so
// we don't keep asking.

function _installParallaxListeners() {
    // Pointer parallax — runs on desktop + Mac Safari. iOS touches also
    // produce pointermove events but they're transient; deviceorientation
    // handles the "no finger on screen" case.
    window.addEventListener('pointermove', (e) => {
        // Normalize to [-1, 1]
        const nx = (e.clientX / window.innerWidth) * 2 - 1;
        const ny = (e.clientY / window.innerHeight) * 2 - 1;
        parallaxTargetX = nx * PARALLAX_RANGE;
        parallaxTargetY = -ny * PARALLAX_RANGE;
    }, { passive: true });

    // iOS DeviceOrientation — gated on permission and only enabled once
    // the user has tapped somewhere (Safari requires a transient
    // activation for requestPermission()).
    const needsPermission = typeof DeviceOrientationEvent !== 'undefined'
        && typeof DeviceOrientationEvent.requestPermission === 'function';

    const attachHandler = () => {
        window.addEventListener('deviceorientation', (e) => {
            // gamma: left/right tilt (-90 to 90), beta: front/back (-180 to 180)
            if (e.gamma === null || e.beta === null) return;
            const nx = Math.max(-1, Math.min(1, (e.gamma || 0) / 45));
            const ny = Math.max(-1, Math.min(1, ((e.beta || 0) - 45) / 45));
            parallaxTargetX = nx * PARALLAX_RANGE;
            parallaxTargetY = -ny * PARALLAX_RANGE;
        }, { passive: true });
    };

    if (needsPermission) {
        let asked = false;
        const ask = () => {
            if (asked) return;
            asked = true;
            DeviceOrientationEvent.requestPermission()
                .then(res => { if (res === 'granted') attachHandler(); })
                .catch(() => { /* user declined — fall back to pointer only */ });
        };
        // Piggyback on the first user gesture rather than asking
        // unprompted; Safari denies requests not tied to activation.
        window.addEventListener('touchend', ask, { once: true, passive: true });
        window.addEventListener('click', ask, { once: true, passive: true });
    } else if (typeof window.DeviceOrientationEvent !== 'undefined') {
        // Non-iOS or older iOS that doesn't require permission.
        attachHandler();
    }
}

// --- Public hooks --------------------------------------------------

// Feed energy into the envelope. `weight` is small (a single log line
// is ~0.1-0.2); saturation at 1.0 means "very busy". An optional color
// BLENDS into the accent tint over time — a slow mood shift, never a
// flash cut.
export function noteActivity(weight, color) {
    const w = Math.max(0, Math.min(1, (weight || 0.15) * temper.event('activity')));
    activityTarget = Math.min(1.0, activityTarget + w);
    if (color) {
        try {
            _accentBlend.set(color);
            accentColor.lerp(_accentBlend, 0.18);   // drift, don't jump
        } catch (e) { /* bad color string — keep previous */ }
        accentTarget = Math.min(0.3, accentTarget + w * 0.35);
    }
}

export function updateSphereColor(colorHex) {
    try { accentColor.set(colorHex); } catch (e) { /* ignore */ }
}

// Errors tint the graph toward magenta for a couple of seconds —
// noticeable, but no longer a flashbang: the old spike drove the error
// uniforms to 1.0 (full recolor + bloom x3) AND zeroed the connection
// probability, disintegrating every link. Repeat calls extend the
// window from the latest call.
let _spikeClearTimeout;
export function noteError(kind) {
    if (kind && kind !== 'generic') { errorKind = kind; errorKindEnv = 1.0; }
    targetErrorState = 0.5;
    // Recoil: the body flinches — one sharp agitated pulse cycle — in
    // addition to the hot tint. Motion + color together read as a
    // startle response rather than a rendering glitch.
    const _g = temper.event('error');
    flinch = Math.min(1.0, flinch + 0.8 * Math.min(1.25, _g));
    activityTarget = Math.min(1.0, activityTarget + 0.3);
    if (_spikeClearTimeout) clearTimeout(_spikeClearTimeout);
    _spikeClearTimeout = setTimeout(() => {
        targetErrorState = 0.0;
        _spikeClearTimeout = null;
    }, 2500);
}

// Back-compat aliases: older call sites keep working, but they now feed
// the envelope instead of firing shockwaves.
export function triggerSpike() { noteError(); }
export function triggerNextColor() { /* retained no-op for compatibility */ }
export function triggerPulse(color) { noteActivity(0.3, color); }
export function triggerSmallPulse(color) { noteActivity(0.12, color); }

// Called by app.js on each audio-analyser tick during TTS playback.
// Level is a normalized 0..1 RMS; we low-pass it here to hide the
// per-frame jitter so the sphere doesn't look jittery.
export function setAudioLevel(level) {
    const clamped = Math.max(0, Math.min(1, level || 0));
    audioLevel += (clamped - audioLevel) * 0.35;
}

// Richer signals (2026-10-30): every live log line (its icons say the
// work kind and the outcome) and every streamed reply chunk (the token
// rate, and whether the model is inside <think>) feed the temperament.
export function noteLogLine(content) { return temper.ingestLog(content); }
export function noteStreamChunk(text) {
    temper.ingestChunk(text);
    activityTarget = Math.min(1.0, activityTarget + 0.012);
}
export function getTemperament() { return { out: { ...TS }, ...temper.debug() }; }

// Introspection for headless verification / tuning — never used by the
// render path itself.
export function getDebugState() {
    return {
        immersion,
        workingState,
        userTurn: userTurnState,
        cameraZ: camera ? camera.position.z : null,
        sceneScale: scene ? scene.scale.x : null,
        activity,
        pulse: _fract(pulsePhase),
        bob: _bob,
        anatomy: FORMS[formIndex],
        phase, gait: { ...gait }, toolPulse, recallSpark, verdict, verdictEnv,
        backgroundBusy, moodHue, moodLook: { ..._moodNow }, pulsePhase, gazeY, errorKind, errorKindEnv, body: { ..._bodyPrev },
        dialect: dialectFor(FORMS[formIndex]), gaitFlow, gaitThicken, gaitFlash, gaitAlign,
        dtF, clock: time, cube2Flow, cube2Travel, twitchAmp,
        linkBudget: _linkBudget, maxLines: MAX_LINES,
        tesseract: { lines: { ..._c2LineStats }, breaths: _c2Shells.map((s) => s.L / (C2_L0 * Math.exp(C2_K * s.d))), kernelSpin: c2KernelSpin, wob: _c2Wob,
                 tw: _c2Shells.map((sh) => sh.tw), d: _c2Shells.map((sh) => sh.d),
                 seeds: _c2Shells.map((sh) => sh.seed), kernelBase: _c2KernelBase,
                 shells: C2_SHELLS, kernelN: C2_KERNEL_N, stream: _c2StreamCount, roll: c2Roll },
    };
}

// The immersion dive's ONLY driver — set true when the user's request
// goes in flight, false when the reply completes/aborts. Ambient
// activity must never call this (that's what setWorkingState is for).
export function setUserTurn(isActive) {
    const _was = targetUserTurnState > 0.5;
    if (isActive !== _was) temper.event(isActive ? 'turn:start' : 'turn:end');
    targetUserTurnState = isActive ? 1.0 : 0.0;
}

let workingTimeout;
export function setWorkingState(isWorking) {
    if (isWorking) {
        if (targetWorkingState < 0.5 && !workingTimeout) {
            workingTimeout = setTimeout(() => {
                targetWorkingState = 1.0;
                workingTimeout = null;
            }, 500);
        }
    } else {
        if (workingTimeout) {
            clearTimeout(workingTimeout);
            workingTimeout = null;
        }
        targetWorkingState = 0.0;
    }
}

// ── cube2 helpers ──────────────────────────────────────────────────

// Fill a shell's per-frame frame for phase d: half-size (with the
// outward-travelling breath), rotation (base tilt + depth twist), the
// birth-pull toward the kernel, wobble amplitude and the taper.
function _c2Frame(sh, d, kx, ky, kz, wobBase, pAmp) {
    sh.d = d;
    const breath = 1.0 + 0.10 * pAmp * _pulseShape(_fract(pulsePhase - d * 0.35));
    sh.L = C2_L0 * Math.exp(C2_K * d) * breath;
    // Born where the kernel is; centred by the time it is grown.
    const pull = Math.exp(-sh.L / 0.6);
    sh.cx = kx * pull; sh.cy = ky * pull; sh.cz = kz * pull;
    sh.wob = wobBase;
    // Taper: in over the first 6% (already tiny), out over the last fifth.
    const tin = Math.min(1, d / 0.06);
    const tout = 1.0 - Math.max(0, Math.min(1, (d - 0.78) / 0.22));
    sh.fade = tin * tin * (3 - 2 * tin) * tout * tout * (3 - 2 * tout);
    // Rotation m = Ry(tiltY + slow wander) · Rx(tiltX) · Rz(twist(d)).
    const az = C2_TWIST * (d - 0.5) + sh.tw + 0.04 * Math.sin(time * 0.05);
    const ax = C2_TILT_X + 0.03 * Math.sin(time * 0.041 + 0.7);
    const ay = C2_TILT_Y + 0.05 * Math.sin(time * 0.033 + 1.9);
    const cz = Math.cos(az), sz = Math.sin(az), cx = Math.cos(ax), sx = Math.sin(ax),
        cy = Math.cos(ay), sy = Math.sin(ay);
    const m = sh.m;
    // Rx·Rz
    const r00 = cz, r01 = -sz, r02 = 0;
    const r10 = cx * sz, r11 = cx * cz, r12 = -sx;
    const r20 = sx * sz, r21 = sx * cz, r22 = cx;
    // Ry·(Rx·Rz)
    m[0] = cy * r00 + sy * r20; m[1] = cy * r01 + sy * r21; m[2] = cy * r02 + sy * r22;
    m[3] = r10; m[4] = r11; m[5] = r12;
    m[6] = -sy * r00 + cy * r20; m[7] = -sy * r01 + cy * r21; m[8] = -sy * r02 + cy * r22;
}

// An angle folded into (−π, π].
function _wrapPi(a) {
    const t = Math.PI * 2;
    return ((a + Math.PI) % t + t) % t - Math.PI;
}
// The kernel's angle as a CUBE sees it: a cube is 4-fold symmetric about
// its axis, so a rotation of θ looks like θ mod 90°. A newborn shell
// inherits the kernel's angle folded into (−45°, 45°] — the shortest
// unwind to the stack's alignment, bounded, never an accumulator carried
// into a per-shell factor (the §4JA/§4JK class).
function _wrapQuarter(a) {
    return _wrapPi(4.0 * a) / 4.0;
}

// The kernel lattice's rotation for this frame: a tumble about a tilted
// axis — Ry(spin) · Rx(0.55) · Rz(0.37·spin) — written into _c2KernelM.
function _c2KernelFrame(spin) {
    const ay = spin, ax = 0.55, az = 0.37 * spin;
    const cz = Math.cos(az), sz = Math.sin(az), cx = Math.cos(ax), sx = Math.sin(ax),
        cy = Math.cos(ay), sy = Math.sin(ay);
    const r00 = cz, r01 = -sz, r02 = 0;
    const r10 = cx * sz, r11 = cx * cz, r12 = -sx;
    const r20 = sx * sz, r21 = sx * cz, r22 = cx;
    const m = _c2KernelM;
    m[0] = cy * r00 + sy * r20; m[1] = cy * r01 + sy * r21; m[2] = cy * r02 + sy * r22;
    m[3] = r10; m[4] = r11; m[5] = r12;
    m[6] = -sy * r00 + cy * r20; m[7] = -sy * r01 + cy * r21; m[8] = -sy * r02 + cy * r22;
}

// A unit-cube point (ux,uy,uz ∈ [-1,1]) of shell `sh` → model space,
// through the self-similar wobble (evaluated in unit coordinates so a
// shell keeps its shape as it grows), the shell's rotation and centre.
function _c2Pt(out, sh, ux, uy, uz) {
    const w = sh.wob, t = time, ph = sh.ph;
    const px = ux + w * Math.sin(1.9 * uy + 1.1 * uz + t * 0.27 + ph);
    const py = uy + w * Math.sin(1.7 * uz + 1.3 * ux + t * 0.23 + ph * 2.0);
    const pz = uz + w * Math.sin(2.3 * ux + 0.9 * uy + t * 0.19 + ph * 3.0);
    const L = sh.L, m = sh.m;
    out.x = L * (m[0] * px + m[1] * py + m[2] * pz) + sh.cx;
    out.y = L * (m[3] * px + m[4] * py + m[5] * pz) + sh.cy;
    out.z = L * (m[6] * px + m[7] * py + m[8] * pz) + sh.cz;
    return out;
}

// Draw every shell's 12 edges as C2_SEGS-segment polylines through the
// same _c2Pt the nodes use; marks the explicit-edge nodes connected.
// Edge alpha carries the shell's taper (× the reorganisation blend, so
// a switch INTO cube2 materialises the wireframe as the nodes arrive).
const _c2A = new THREE.Vector3(), _c2B = new THREE.Vector3();
// One line segment into the buffers; returns the next index.
function _c2Line(pos, uv, hue, fade, lineIdx, a, b, u0, u1, h0, h1, f) {
    const o = lineIdx * 6;
    pos[o] = a.x; pos[o + 1] = a.y; pos[o + 2] = a.z;
    pos[o + 3] = b.x; pos[o + 4] = b.y; pos[o + 5] = b.z;
    uv[lineIdx * 2] = u0; uv[lineIdx * 2 + 1] = u1;
    hue[lineIdx * 2] = h0; hue[lineIdx * 2 + 1] = h1;
    fade[lineIdx * 2] = f; fade[lineIdx * 2 + 1] = f;
    return lineIdx + 1;
}
// Draw every shell's 12 edges as C2_SEGS-segment polylines through the
// same _c2Pt the nodes use, the ties between consecutive shells and the
// kernel lattice's edges; marks the explicit-edge nodes connected. Edge
// alpha carries the shell's taper (× the reorganisation blend, so a
// switch INTO cube2 materialises the wireframe as the nodes arrive).
function _emitCube2Edges(pos, uv, hue, fade, connected, lineIdx) {
    const blend = formBlend < 1.0 ? formBlend * formBlend * (3.0 - 2.0 * formBlend) : 1.0;
    _c2LineStats.shells = 0; _c2LineStats.rulings = 0; _c2LineStats.ties = 0; _c2LineStats.kernel = 0;
    // Shell edges. Every edge carries its OWN packet phase (v2): the line
    // shader's travelling charge is `fract(vLightPass·1.5 − 2·time)`, so
    // a per-edge offset spreads the packets over the stack instead of
    // marching every edge in lock-step (invisible as motion).
    for (let k = 0; k < _c2Shells.length; k++) {
        const sh = _c2Shells[k];
        const f = sh.fade * blend * C2_LINE_GAIN;
        if (f < 0.002) continue;
        for (let e = 0; e < C2_EDGES.length; e++) {
            const E = C2_EDGES[e], a = E[0], sb = E[1], sc = E[2];
            const ph0 = _fract(0.37 * e + 0.11 * k);
            const u = _c2U; u[(a + 1) % 3] = sb; u[(a + 2) % 3] = sc;   // all three set below
            u[a] = -1; _c2Pt(_c2A, sh, u[0], u[1], u[2]);
            for (let sgi = 1; sgi <= C2_SEGS; sgi++) {
                if (lineIdx >= MAX_LINES) return lineIdx;
                u[a] = -1 + 2 * sgi / C2_SEGS;
                _c2Pt(_c2B, sh, u[0], u[1], u[2]);
                lineIdx = _c2Line(pos, uv, hue, fade, lineIdx, _c2A, _c2B,
                    ph0 + (sgi - 1) / C2_SEGS, ph0 + sgi / C2_SEGS, sh.seed, sh.seed, f);
                _c2LineStats.shells++;
                _c2A.copy(_c2B);
            }
        }
    }
    // Face rulings (v3): a "+" on every face — the mid-lines of the
    // face, bent by the same wobble field — so a shell reads as a
    // paneled data-cube, not a hollow frame. Dimmer than the edges.
    if (C2_RULINGS) {
        for (let k = 0; k < _c2Shells.length; k++) {
            const sh = _c2Shells[k];
            const f = sh.fade * blend * C2_LINE_GAIN * C2_RULING_GAIN;
            if (f < 0.002) continue;
            for (let face = 0; face < 3; face++) for (let si = 0; si < 2; si++) for (let axis = 0; axis < 2; axis++) {
                const sgn = si === 0 ? -1 : 1;
                // the ruling runs along axis `run` across the face normal to `face`
                const run = (face + 1 + axis) % 3, mid = (face + 2 - axis) % 3;
                const u = _c2U; u[face] = sgn; u[mid] = 0;   // u[run] set below
                const ph0 = _fract(0.29 * (face * 4 + (sgn + 1) + axis) + 0.13 * k);
                u[run] = -1; _c2Pt(_c2A, sh, u[0], u[1], u[2]);
                for (let sgi = 1; sgi <= C2_SEGS; sgi++) {
                    if (lineIdx >= MAX_LINES) return lineIdx;
                    u[run] = -1 + 2 * sgi / C2_SEGS;
                    _c2Pt(_c2B, sh, u[0], u[1], u[2]);
                    lineIdx = _c2Line(pos, uv, hue, fade, lineIdx, _c2A, _c2B,
                        ph0 + (sgi - 1) / C2_SEGS, ph0 + sgi / C2_SEGS, sh.seed, sh.seed, f);
                    _c2LineStats.rulings++;
                    _c2A.copy(_c2B);
                }
            }
        }
    }
    // Ties (v2): the 8 corners of each shell joined to the corners of the
    // next shell out — nested frames become ONE tunnel receding to the
    // kernel, and the shells' differing twists make the ties a visible
    // helix. Consecutive in DEPTH order; the wrap pair (oldest → newborn)
    // is skipped — that would be a line across the whole scene from the
    // faded rim to the kernel.
    _c2Order.length = 0;
    for (let k = 0; k < _c2Shells.length; k++) _c2Order.push(k);
    _c2Order.sort((i, j) => _c2Shells[i].d - _c2Shells[j].d);
    for (let q = 0; q + 1 < _c2Order.length; q++) {
        const A = _c2Shells[_c2Order[q]], B = _c2Shells[_c2Order[q + 1]];
        const f = Math.min(A.fade, B.fade) * blend * C2_LINE_GAIN * C2_TIE_GAIN;
        if (f < 0.002) continue;
        for (let c = 0; c < 8; c++) {
            if (lineIdx >= MAX_LINES) return lineIdx;
            const ux = (c & 1) ? 1 : -1, uy = (c & 2) ? 1 : -1, uz = (c & 4) ? 1 : -1;
            _c2Pt(_c2A, A, ux, uy, uz); _c2Pt(_c2B, B, ux, uy, uz);
            const ph0 = _fract(0.23 * c + 0.31 * q);
            lineIdx = _c2Line(pos, uv, hue, fade, lineIdx, _c2A, _c2B, ph0, ph0 + 1.0, A.seed, B.seed, f);
            _c2LineStats.ties++;
        }
    }
    // The kernel lattice's edges (v2): adjacent sites along each axis,
    // read from the node positions already placed this frame.
    const N = C2_KERNEL_N, kb = _c2KernelBase;
    const site = (ix, iy, iz) => kb + ix + N * (iy + N * iz);
    const fk = blend * C2_KERNEL_LINE_GAIN;
    for (let iz = 0; iz < N; iz++) for (let iy = 0; iy < N; iy++) for (let ix = 0; ix < N; ix++) {
        const i0 = site(ix, iy, iz);
        const p0 = currentPositions[i0], h0 = nodeSeeds[i0];
        const nb = _c2Nb;
        let nbN = 0;
        if (ix + 1 < N) nb[nbN++] = site(ix + 1, iy, iz);
        if (iy + 1 < N) nb[nbN++] = site(ix, iy + 1, iz);
        if (iz + 1 < N) nb[nbN++] = site(ix, iy, iz + 1);
        for (let q = 0; q < nbN; q++) {
            const j = nb[q];
            if (lineIdx >= MAX_LINES) return lineIdx;
            const ph0 = _fract(0.41 * (ix + iy + iz) + 0.17 * j);
            lineIdx = _c2Line(pos, uv, hue, fade, lineIdx, p0, currentPositions[j], ph0, ph0 + 1.0, h0, nodeSeeds[j], fk);
            _c2LineStats.kernel++;
        }
    }
    for (let i = 0; i < NODE_COUNT; i++) if (_noProx[i]) connected[i] = true;
    return lineIdx;
}

// ── Scene/camera form-switch blend helpers (§4PR) ──────────────────
function _smooth01(x) { return x * x * (3.0 - 2.0 * x); }
// The temperament's clock and amplitude multipliers; reduced motion
// keeps the old fixed behaviour (no extra speed-ups or swells).
function _tsSpeed() { return PREFERS_REDUCED_MOTION ? 1.0 : TS.speedMul; }
function _tsAmp() { return PREFERS_REDUCED_MOTION ? 1.0 : TS.ampMul; }
// This frame's steady look target (the form branches call it).
function _lookTarget(x, y, z) { _sceneT.lx = x; _sceneT.ly = y; _sceneT.lz = z; }
// Write the scene transform + camera z / look target: the steady targets
// when e === 1, else the blend from the switch snapshot. Angles take the
// shortest wrapped path and are written as `target − wrapped·(1 − e)`, so
// the last blend frame lands EXACTLY on the target value (no 2π residue).
function _applySceneBlend(e) {
    const T = _sceneT, F = _sceneFrom;
    if (e >= 1.0) {
        F.on = false;
        scene.scale.set(T.s, T.s, T.s);
        scene.rotation.x = T.rx; scene.rotation.y = T.ry; scene.rotation.z = T.rz;
        scene.position.x = T.px; scene.position.y = T.py;
        camera.position.z = T.camZ;
        _lookNow.x = T.lx; _lookNow.y = T.ly; _lookNow.z = T.lz;
    } else {
        const k = 1.0 - e;
        const sc = F.s + (T.s - F.s) * e;
        scene.scale.set(sc, sc, sc);
        scene.rotation.x = T.rx - _wrapPi(T.rx - F.rx) * k;
        scene.rotation.y = T.ry - _wrapPi(T.ry - F.ry) * k;
        scene.rotation.z = T.rz - _wrapPi(T.rz - F.rz) * k;
        scene.position.x = F.px + (T.px - F.px) * e;
        scene.position.y = F.py + (T.py - F.py) * e;
        camera.position.z = F.camZ + (T.camZ - F.camZ) * e;
        _lookNow.x = F.lx + (T.lx - F.lx) * e;
        _lookNow.y = F.ly + (T.ly - F.ly) * e;
        _lookNow.z = F.lz + (T.lz - F.lz) * e;
    }
    camera.lookAt(_lookNow.x, _lookNow.y, _lookNow.z);
}

// Per-frame scratch, hoisted out of animate() (§4PR: the frame allocated
// a 250-entry array, an Object3D, the link table and a dust frame object
// every tick — GC churn on the always-on ClockworkPi).
const _connected = new Array(NODE_COUNT).fill(false);
const _dummy = new THREE.Object3D();
const _c2Dust = { d0: 0, ph: 0, prevD: 0, d: 0, L: 1, fade: 0, seed: 0, wob: 0, tw: 0, cx: 0, cy: 0, cz: 0, m: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
const _c2U = [0, 0, 0];
const _c2Nb = [0, 0, 0];
// Network-error flicker's duty cycle: sin(x) > 0.3 holds for
// (π − 2·asin 0.3) / 2π ≈ 0.40 of the time — the steady reduced-motion dim.
const NET_FLICKER_DUTY = 0.40;
// Per-form link-radius² multipliers: the vortex reads best DENSE
// (1.5× weaves the funnel membrane tighter — "a bit more dense");
// the AI forms need TIGHTER radii — the cube so only axis-neighbors
// weave (a clean wireframe tensor, no diagonals), descent so the
// terrain reads as a mesh surface, not a solid (the mobile sheet is
// sparser, hence the wider radius).
const LINK_MULT = {
    moodtesseract: 0.35,  // nothing links by proximity here (explicit edges)
};

// The thermal oscillation's clock (body below in animate's comment).
function _advanceHue() {
    huePhase += (0.006 + activity * 0.010)
        * (1.0 + 0.35 * Math.sin(time * 0.083))
        * (PREFERS_REDUCED_MOTION ? 0.3 : 1.0) * dtF;
    hueDrift = 0.07 * Math.sin(huePhase);
}

// --- Main animation loop -------------------------------------------

// Pause the render loop while the tab is hidden (2026-09-11). The loop
// self-scheduled forever: a backgrounded phone kept the GPU busy with
// bloom passes nobody could see. Resume restarts it only once init() has
// built a renderer. Returns whether the loop is RUNNING after the call.
let _animationPaused = false;
export function setAnimationPaused(paused) {
    _animationPaused = !!paused;
    if (_animationPaused) {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        _lastFrameTs = null;   // the first resumed frame steps dtF = 1
        return false;
    }
    if (renderer && !animationFrameId) animate();
    return !!animationFrameId;
}
if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('visibilitychange', () => setAnimationPaused(!!document.hidden));
}

function animate(ts) {
    if (_animationPaused) { animationFrameId = null; return; }
    animationFrameId = requestAnimationFrame(animate);

    // Frame-time scale — see the dtF declaration. `ts` is the rAF
    // timestamp; the very first frame (and the first after a pause)
    // has no previous stamp and steps exactly one 60fps frame.
    let frameMs = 1000 / 60;
    if (typeof ts === 'number' && _lastFrameTs !== null) {
        frameMs = ts - _lastFrameTs;
        dtF = Math.min(DT_MAX, Math.max(DT_MIN, frameMs / (1000 / 60)));
    } else {
        dtF = 1.0;
    }
    if (typeof ts === 'number') _lastFrameTs = ts;

    const isWaking = targetWorkingState > workingState + 0.01;
    const transitionSpeed = isWaking ? 0.05 : 0.02;

    workingState += (targetWorkingState - workingState) * ease(transitionSpeed);
    userTurnState += (targetUserTurnState - userTurnState)
        * ease(targetUserTurnState > userTurnState ? 0.05 : 0.02);
    errorState   += (targetErrorState   - errorState)   * ease(0.05);

    // Activity envelope: the raw target decays on its own (half-life
    // ~2.5s at 60fps) and the rendered value follows it with a soft
    // attack and a slower release — so a burst of log lines swells the
    // graph over ~a second and lets it settle over ~8, instead of
    // strobing per event.
    activityTarget *= decay(0.9955);
    if (activityTarget < 0.005) activityTarget = 0;
    const _rising = activityTarget > activity;
    activity += (activityTarget - activity) * ease(_rising ? 0.035 : 0.008);

    // Accent tint follows the same philosophy: drift up with colored
    // activity, drain slowly back to neutral.
    accentTarget *= decay(0.995);
    accentStrength += (accentTarget - accentStrength) * ease(0.02);

    // The shader "energy" (formerly the per-event shockwave) is now the
    // smoothed envelope — faint traveling charge on the lines and a mild
    // node glow when the agent is busy, perfectly still when quiet.
    pulseT = Math.min(1.0, activity * 0.55);

    // Audio-level natural decay: even if setAudioLevel stops being
    // called (TTS queue drained), the residual level dies out fast.
    audioLevel *= decay(0.92);

    // ── Temperament (2026-10-30): one step per frame ───────────────
    const _wall = Date.now(), _wallGap = _wall - _temperWall;
    _temperWall = _wall;
    if (frameMs > TEMPER_ABSENCE_MS || _wallGap > TEMPER_ABSENCE_MS) {
        frameMs = 1000 / 60;
        _temperNow = Math.max(_temperNow, _wall);
    }
    _temperNow += frameMs;
    TS = temper.step(frameMs / 1000, _temperNow);
    if (TS.startleKick > 0) flinch = Math.min(1.0, flinch + 0.45 * TS.startleKick);

    // ── Signal-layer envelopes (2026-09-11) ─────────────────────────
    for (const k of PHASES) gait[k] += ((phase === k ? 1 : 0) - gait[k]) * ease(0.04);
    toolPulse *= decay(0.93);
    if (toolPulse < 0.003) toolPulse = 0;
    recallSpark *= decay(0.972);                // ~1.5s comet
    if (recallSpark < 0.01) { recallSpark = 0; recallNode = -1; }
    verdictEnv *= decay(verdict === 'pass' ? TUNE.passHold : 0.975);   // pass holds ~2s, others ~1s
    if (verdictEnv < 0.01) { verdictEnv = 0; verdict = null; }
    backgroundBusy += (targetBackgroundBusy - backgroundBusy) * ease(0.01);
    // The mood's own form takes its colour from the look alone: adding this shift
    // pushed the kernel past the red stop, and a preview never matched the live mood.
    // The mood form carries no label shift, and sheds one brought over from the
    // plain tesseract as fast as its look arrives, not over the usual minute. With
    // no look showing it keeps the usual rate, so at neutral it stays the tesseract.
    moodHue += ((FORMS[formIndex] === 'moodtesseract' ? 0.0 : targetMoodHue) + TS.hueShift - moodHue) * ease(FORMS[formIndex] === 'moodtesseract' && (_moodPreview || _moodLive) !== MOOD_NEUTRAL ? 0.012 : 0.0008);     // ~1 min to settle
    gazeX += (targetGazeX + TS.gazeX - gazeX) * ease(0.03);
    gazeY += (targetGazeY + TS.gazeY - gazeY) * ease(0.03);
    errorKindEnv *= decay(0.985);
    if (errorKindEnv < 0.01) { errorKindEnv = 0; errorKind = null; }
    sweepAngle += (dtF / 60) * TUNE.sweepSpeed * gait.search;
    // The current form's dialect → the scalars the branches below read.
    const DIAL = dialectFor(FORMS[formIndex]);
    gaitFlow = DIAL.write === 'flow' ? gait.write : 0;
    gaitThicken = DIAL.read === 'thicken' ? gait.read : 0;
    gaitFlash = DIAL.tool === 'flash' ? toolPulse : 0;
    gaitAlign = DIAL.verify === 'align' ? gait.verify : 0;
    // Stillness: a pass crystallises; a refusal freezes; verify tightens
    // toward stillness; read slows. 0 = full motion, 1 = frozen.
    const still = Math.min(1.0, Math.max(
        verdict === 'pass' ? TUNE.stillPass * verdictEnv : 0,
        errorKind === 'refusal' ? 0.9 * errorKindEnv : 0,
        TUNE.stillVerify * gait.verify, TUNE.stillRead * gait.read,
        PREFERS_REDUCED_MOTION ? 0 : TS.stillness));
    const motionMul = 1.0 - still;

    // Immersion follows the USER-TURN state with a slower, asymmetric
    // ease: ~5s to fully swallow (only sustained work gets there), ~10s
    // to drift back out after completion. Short requests barely lean in.
    // Deliberately NOT workingState: ambient log activity re-arms that
    // almost continuously on a busy agent, which kept the camera inside
    // the cloud when the agent wasn't working for the user at all.
    const immersionTarget = userTurnState * IMMERSION_CAP;
    immersion += (immersionTarget - immersion)
        * ease(immersionTarget > immersion ? 0.012 : 0.006);
    // Smoothstepped path so both ends of the dive are gentle.
    const dive = immersion * immersion * (3.0 - 2.0 * immersion);

    // Form name, bound once per frame — every per-form branch below
    // keys on the NAME (index comparisons broke silently whenever the
    // FORMS array grew; 2026-07-29, when the four AI forms joined).
    const FORM = FORMS[formIndex];
    // The mood's look: only its own form has a target; leaving eases back to
    // neutral, and neutral is exactly 1/0 so every other form is untouched.
    const _moodT = FORM === 'moodtesseract' ? (_moodPreview || _moodLive) : MOOD_NEUTRAL;
    for (const f of MOOD_FIELDS) {
        const gap = _moodT[f] - _moodNow[f];
        _moodNow[f] = Math.abs(gap) < 1e-4 ? _moodT[f] : _moodNow[f] + gap * ease(0.012);
    }
    // The ring's stops are the objects all three materials read, so only a mood
    // ever writes them, and at neutral they land back on the exact thermal colours.
    COLORS.palette.forEach((c, i) => {
        const t = _moodT.ring[i];
        if (c.r === t.r && c.g === t.g && c.b === t.b) return;
        c.lerp(t, ease(0.012));
        if (Math.abs(c.r - t.r) + Math.abs(c.g - t.g) + Math.abs(c.b - t.b) < 3e-4) c.set(t);
    });
    temper.setMoodBias(_moodT);

    // cube2: the same engagement law as the vortex — the birth rate is
    // the swallow. Idle ≈ a new shell every ~4.5s; a user turn ≈ every
    // 0.7s, gulping in rhythm with the pulse; completion is a pure
    // intensity morph (no reset, no camera travel).
    if (FORM === 'moodtesseract') {
        if (userTurnState > 0.5) {
            cube2Travel += (IMMERSION_CAP - cube2Travel) * ease(0.010);
        } else if (cube2Travel > 0.001) {
            cube2Travel *= decay(0.985);
        } else {
            cube2Travel = 0.0;
        }
        const c2n = Math.min(cube2Travel / Math.max(IMMERSION_CAP, 0.001), 1.0);
        const c2Turb = 1.0 + CALM * (0.30 * Math.sin(time * 0.21) * Math.sin(time * 0.077 + 1.1)
            + 0.45 * c2n * _pulseShape(_fract(pulsePhase)));
        cube2Flow += (dtF / 60) * (0.020 + 0.020 * Math.max(workingState, activity)
            + 0.10 * c2n) * c2Turb * CALM * (1.0 + TUNE.flowWrite * gaitFlow) * _moodNow.flow;
        // v2: a tool call sends a burst of packets down every edge (the
        // line shader's `burst` term rides uPulseT), and a user turn keeps
        // the charge visibly moving.
        pulseT = Math.min(1.0, pulseT + 0.6 * toolPulse + 0.15 * c2n);
    } else {
        cube2Travel = 0.0;
    }

    // Accumulate time for lines at steady pace — except inside the
    // cloud, where the data pulses rush a little faster: the interior
    // should feel BUSIER than the outside view, not emptier.
    // tStep is THIS frame's advance of the shared clock; the per-form
    // integrated phases below (runner flow, cube churn/spin) step by it
    // so they stay locked to `time` at constant rate.
    const tStep = 0.005 * motionMul * dtF * _tsSpeed();
    time += tStep;

    // Idle breathing: ±1% scene-scale sine at ~0.1Hz. Below the
    // motion-detection threshold on both desktop and mobile; keeps the
    // sphere from looking frozen when nothing else is animating.
    // Immersion scales the whole scene UP around the viewer — combined
    // with the camera dolly below, this is what reads as "the grid
    // swallowing the camera" rather than the camera flying somewhere.
    // Two incommensurate frequencies: the breath never repeats exactly —
    // a metronomic single sine is precisely what read as "not alive".
    const breathe = 1.0 + 0.011 * Math.sin(time * 0.55)
        + 0.006 * Math.sin(time * 0.173 + 1.7)

    // Scale boost trimmed 0.7 → 0.55: the swell is what sells the
    // swallow, but it also DILUTES local node density exactly when the
    // camera is closest — the interior looked empty at 0.7 even with
    // the denser web + motes compensating.
    const baseScale = 0.9 * breathe * (1.0 + dive * 0.55);
    // The vortex's journey is the CAMERA's — swelling the scene under it
    // would double-transform the funnel, so the swell is neutralized.
    // The cube gets a GENTLED swell: the point of its dive is watching
    // the attention kernel work the grid, not entering a cloud — full
    // swell dissolved the silhouette into dots.
    // Verdict release: a stop EXHALES (one soft swell that lets go); the
    // background breath is a second, slower rhythm under everything.
    const exhale = verdict === 'stop' ? TUNE.exhale * Math.sin(Math.PI * (1.0 - verdictEnv)) * verdictEnv : 0;
    const bgBreath = TUNE.bgBreath * backgroundBusy * Math.sin(time * 0.31 + 0.9);
    const sceneScale = (FORM === 'moodtesseract' ? 0.9 * breathe : baseScale) * (1.0 + exhale + bgBreath);
    // Steady targets go into _sceneT; _applySceneBlend writes them (or
    // the blend toward them after a form switch) below — §4PR.
    const T = _sceneT;
    T.s = sceneScale;
    scene.rotation.x -= _bodyPrev.rx; scene.rotation.y -= _bodyPrev.ry; scene.rotation.z -= _bodyPrev.rz;
    scene.position.x -= _bodyPrev.px; scene.position.y -= _bodyPrev.py;

    // Slow continuous orbit (faster while working) plus a gentle tilt
    // gives the network real depth and life at idle. Skipped entirely
    // under reduced-motion (the targets then hold the current values).
    T.rx = scene.rotation.x; T.ry = scene.rotation.y; T.rz = scene.rotation.z;
    T.px = scene.position.x; T.py = scene.position.y;
    if (!PREFERS_REDUCED_MOTION) {
        if (FORM === 'moodtesseract') {
            // Near-still heading: the corridor's own roll is the motion.
            T.ry = 0.09 * Math.sin(time * 0.029);
            T.rx = 0.05 * Math.sin(time * 0.021 + 0.7);
            T.rz = 0.02 * Math.sin(time * 0.017 + 2.0) + c2Roll;   // v3: the corridor's slow roll
            T.px = 0.04 * Math.sin(time * 0.023);
            T.py = _bob * 0.4 + _moodNow.droop;
        } else {
            // Heading wander: the creature holds a loose heading and lets
            // it drift — sums of incommensurate slow sines, so it turns
            // its attention rather than rotating on a turntable. The body
            // also drifts a little, like something suspended in a current.
            T.ry = 0.38 * Math.sin(time * 0.050)
                + 0.22 * Math.sin(time * 0.023 + 1.3);
            T.rx = 0.10 * Math.sin(time * 0.037 + 0.7)
                + 0.05 * Math.sin(time * 0.011);
            T.rz = 0.05 * Math.sin(time * 0.029 + 2.1);
            T.px = 0.10 * Math.sin(time * 0.031);
            T.py = _bob;
        }
    }

    // Parallax: ease camera toward target offset. Z is owned by the
    // immersion dive (rest 5.0 → inside the cloud 1.3); x/y parallax
    // rides on top at any depth.
    camera.position.x += (parallaxTargetX - camera.position.x) * ease(0.04);
    camera.position.y += (parallaxTargetY - camera.position.y) * ease(0.04);
    if (FORM === 'moodtesseract') {
        // cube2: the camera never moves either — the shells' outward
        // bloom is the fall; it studies the kernel at the origin.
        T.camZ = CAMERA_REST_Z;
        _lookTarget(camera.position.x * 0.35, camera.position.y * 0.35, 0.0);
    } else {
        T.camZ = CAMERA_REST_Z - (CAMERA_REST_Z - CAMERA_DIVE_Z) * dive;
        // Look-target blend: at rest the camera studies the origin; deep in
        // the cloud it must look FORWARD through it instead — near the
        // origin, lookAt(0,0,0) turns tiny parallax offsets into wild
        // rotations (the lookAt singularity).
        _lookTarget(gazeX, gazeY * (1.0 - dive), -3.5 * dive);
    }
    // The form-switch blend (see _sceneFrom): the same smoothstep the node
    // blend below applies this frame, or exactly the steady targets.
    _applySceneBlend(formBlend < 1.0 && _sceneFrom.on
        ? _smooth01(Math.min(1.0, formBlend + (dtF / 60) / 1.4)) : 1.0);
    // The temperament's BODY (2026-10-30 v1.2): head turn/tilt, shake,
    // sink, coil/release — on every form, over whatever the form did.
    {
        const B = _bodyPrev, on = !PREFERS_REDUCED_MOTION;
        B.rx = on ? TS.pitch : 0; B.ry = on ? TS.yaw : 0; B.rz = on ? TS.roll : 0;
        B.px = on ? TS.offX : 0; B.py = on ? TS.offY : 0;
        scene.rotation.x += B.rx; scene.rotation.y += B.ry; scene.rotation.z += B.rz;
        scene.position.x += B.px; scene.position.y += B.py;
        const sm = on ? TS.scaleMul : 1;
        B.s = sm;
        scene.scale.set(scene.scale.x * sm, scene.scale.y * sm, scene.scale.z * sm);
    }

    // Structure changes form slowly when idle, faster when busy. "Busy"
    // is the max of an in-flight chat turn (workingState) and ambient
    // log activity (the envelope) — so autonomous background work also
    // animates the graph, just smoothly.
    const drive = Math.min(1.0, Math.max(workingState, activity * 0.85));
    let targetShapeSpeed = SPEEDS.idle + (drive * (SPEEDS.busy - SPEEDS.idle));
    let speedDiff = targetShapeSpeed - currentShapeSpeed;
    if (Math.abs(speedDiff) > 0.001) {
        // Change by 2.0 over ~180 frames (3 seconds at 60fps) -> 0.011 per frame
        currentShapeSpeed += Math.sign(speedDiff) * Math.min(Math.abs(speedDiff), 0.011 * dtF);
    }

    // ── Medusan propulsion ──────────────────────────────────────────
    // The pulse clock: ~4.5s per cycle at idle, tightening toward ~1.8s
    // when the agent is busy (currentShapeSpeed lerps between
    // SPEEDS.idle and SPEEDS.busy above). An error flinch briefly
    // agitates both rate and amplitude — a recoil, not a flash.
    pulsePhase += (dtF / 60) * (0.20 + 0.105 * currentShapeSpeed)
        * (1.0 + flinch * 1.2)
        * (PREFERS_REDUCED_MOTION ? 0.5 : 1.0) * motionMul * _tsSpeed() * _moodNow.pulse;
    flinch *= decay(0.97);

    const pulseAmp = (0.16 + 0.10 * drive + 0.22 * flinch)
        * (PREFERS_REDUCED_MOTION ? 0.45 : 1.0)
        * (1.0 + 0.25 * audioLevel) * _tsAmp();
    // Contraction as felt at the bell margin (drives tentacle tug + the
    // swim bob) — the wave reaches the margin 0.16 cycles after the apex.
    const cMargin = _pulseShape(_fract(pulsePhase - 0.16));

    // Slow vertical mass-drift (NOT a pulse-locked swim bob — that was
    // the jellyfish tell). The body hovers like something suspended.
    _bobTarget = 0.10 * Math.sin(time * 0.043 + 0.5);
    _bob += (_bobTarget - _bob) * ease(0.02);

    if (FORM === 'moodtesseract') {
        // ── CUBE2 (v2): the kernel lattice tumbles on its integrated spin;
        //    each shell's frame (phase → size, twist = stack twist + the
        //    inherited kernel angle unwinding, breath, wobble, birth-pull,
        //    taper), then the corner/mid nodes through the same _c2Pt the
        //    edge emitter uses, the lattice sites, and the dust.
        const c2n = Math.min(cube2Travel / Math.max(IMMERSION_CAP, 0.001), 1.0);
        const kx = 0.22 * Math.sin(time * 0.090), ky = 0.18 * Math.sin(time * 0.070 + 1.3),
            kz = 0.12 * Math.cos(time * 0.050 + 0.4);
        // Verdict grammar (v2): a pass CRYSTALLISES — the wobble collapses
        // and the inherited twists unwind fast (edges a touch brighter via
        // the global formDim pass lift); a refute SHUDDERS — a fast jitter
        // on the wobble that dies with the verdict envelope; a stop exhales
        // through the scene-scale exhale every form shares.
        const passEnv = verdict === 'pass' ? verdictEnv : 0.0;
        const shudder = verdict === 'refute' ? 0.09 * verdictEnv * CALM * Math.sin(time * 45.0) : 0.0;
        const wobBase = 0.07 * CALM * (1.0 - TUNE.alignGain * gaitAlign) * _moodNow.wob;
        const c2Amp = pulseAmp * _moodNow.breath;
        const wob = wobBase * (1.0 - 0.85 * passEnv) + shudder;
        _c2Wob = wob;
        // Kernel spin: integrated at the current rate (idle a slow turn,
        // a user turn faster, a tool call a kick) — see c2KernelSpin.
        c2KernelSpin += (dtF / 60) * (0.10 + 0.50 * c2n + 0.8 * gaitFlash) * CALM * motionMul * _moodNow.spin;
        _c2KernelFrame(c2KernelSpin);
        // The stack's slow roll (v3): the whole corridor screws inward —
        // a full turn in ~5 minutes at idle, three times that on a turn.
        c2Roll += (dtF / 60) * (0.02 + 0.04 * c2n) * CALM * motionMul;
        const twDecay = decay(passEnv > 0.05 ? C2_TW_DECAY_PASS : C2_TW_DECAY);
        const c2Slope = 1.0 - Math.max(0.0, _moodNow.heat), c2Cold = Math.min(0.0, _moodNow.heat);
        for (let k = 0; k < _c2Shells.length; k++) {
            const sh = _c2Shells[k];
            const d = _fract(sh.d0 + cube2Flow);
            if (d < sh.prevD - 0.5) {                     // wrapped: born again
                sh.ph = Math.random() * Math.PI * 2;      // new wobble personality
                sh.tw = _wrapQuarter(c2KernelSpin);       // inherits the kernel's angle (bounded, ≤ 45°)
            }
            sh.prevD = d;
            sh.tw *= twDecay;
            _c2Frame(sh, d, kx, ky, kz, wob, c2Amp);
            // Heat wave (v2): hottest at birth, cooling as it grows, with a
            // pulse of warmth racing outward on the breath clock — stronger
            // while a user turn runs, so busy reads as red pulses leaving
            // the kernel.
            sh.seed = Math.min(0.62, Math.max(0.02, 0.60 - d * 0.52 * c2Slope + c2Cold)
                + 0.10 * (0.35 + c2n) * _pulseShape(_fract(pulsePhase - d * 0.35))
                + 0.08 * c2n * (1.0 - d));
        }
        const g = 1.0 + 0.30 * c2n + 0.25 * gaitFlash * TUNE.flashGain
            + 0.12 * _pulseShape(_fract(pulsePhase + 0.05));
        const tremor = 0.010 + 0.020 * c2n;
        const _dust = _c2Dust;   // scratch frame; _c2Frame rewrites every field it reads
        const km = _c2KernelM, kr = C2_KERNEL_R * g * _moodNow.kernel;
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            const P = currentPositions[i];
            if (bp.kind === 0) {
                const sh = _c2Shells[bp.k];
                _c2Pt(P, sh, bp.ux, bp.uy, bp.uz);
                const corner = bp.ux !== 0 && bp.uy !== 0 && bp.uz !== 0;
                // Corner sparks (v2): a tool call flashes the corners.
                bp.sz = (corner ? 1.0 + 0.6 * gaitFlash : 0.8) * sh.fade;
                nodeSeeds[i] = sh.seed;
            } else if (bp.kind === 2) {
                const lx = bp.lx * kr, ly = bp.ly * kr, lz = bp.lz * kr;
                P.set(
                    kx + km[0] * lx + km[1] * ly + km[2] * lz + tremor * CALM * Math.sin(time * 1.1 + bp.jit),
                    ky + km[3] * lx + km[4] * ly + km[5] * lz + tremor * CALM * Math.cos(time * 0.9 + bp.jit * 2.0),
                    kz + km[6] * lx + km[7] * ly + km[8] * lz + tremor * CALM * Math.sin(time * 0.7 + bp.jit * 3.0));
                bp.sz = 1.15 * (1.0 + 0.5 * c2n + 0.4 * gaitFlash);
                nodeSeeds[i] = Math.min(0.62, 0.58 + 0.03 * c2n + 0.02 * Math.sin(time * 0.8 + bp.jit) + c2Cold);
            } else {
                // Stream grain (v3): its own phase and speed on the shared
                // expansion; spirals about the axis as it recedes; hot at
                // birth, cooling outward; tapered like a shell.
                const d = _fract(bp.d0 + cube2Flow * bp.flowScale);
                _c2Frame(_dust, d, kx, ky, kz, 0.0, c2Amp);
                const phi = bp.uphi + C2_STREAM_SPIRAL * d;
                _c2Pt(P, _dust, bp.ur * Math.cos(phi), bp.ur * Math.sin(phi), bp.uz);
                P.x += 0.02 * CALM * Math.sin(time * 0.6 + bp.jit);
                P.y += 0.02 * CALM * Math.cos(time * 0.5 + bp.jit * 2.0);
                bp.sz = 0.9 * _dust.fade * (1.0 + 0.3 * c2n);
                nodeSeeds[i] = Math.min(0.62, Math.max(0.02, 0.60 - d * 0.55 * c2Slope + c2Cold) + 0.06 * c2n * (1.0 - d));
            }
        }
        instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    } else {
        // ── EMPTY: the dispersed far sphere — static, unlinked,
        //    invisible.
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            currentPositions[i].set(bp.hx, bp.hy, bp.hz);
        }
    }

    // ── Gaits + events, applied on top of every anatomy ─────────────
    // Radial factors compose: search expands, read/verify contract, a tool
    // call kicks, the background breath swells the edge, a refute shudders,
    // an idle twitch shivers one neighbourhood.
    if (FORM !== 'empty') {
        // The dialect decides WHICH components breathe (a terrain sheet
        // heaves in y, a crystal never breathes) and whether write is the
        // z-wave or the form's own flow.
        const axis = DIAL.radialAxis;
        const readRadial = DIAL.read === 'contract' ? TUNE.radialRead * gait.read : 0;
        const toolRadial = DIAL.tool === 'kick' ? TUNE.toolKick * toolPulse : 0;
        const radial = 1.0 + TUNE.radialSearch * gait.search - readRadial
            - TUNE.radialVerify * gait.verify + toolRadial;
        // × CALM like every other CPU-side ripple (§4PR: reduced motion
        // got the full-amplitude shake).
        const shudder = verdict === 'refute' ? TUNE.shudder * verdictEnv * CALM : 0;
        const writeW = DIAL.write === 'wave' ? gait.write : 0;
        // Not on the tesseract: its wireframe is computed apart from the nodes,
        // so a twitch pulled corners off their edges, and the neighbourhood of a
        // node near the centre held the whole kernel (5–18× the rest motion,
        // every ~13 s; operator: "gittery", 2026-11-06).
        const twitchP = idleTwitchNode >= 0 && !FORM === 'moodtesseract' ? currentPositions[idleTwitchNode] : null;
        for (let i = 0; i < NODE_COUNT; i++) {
            const p = currentPositions[i];
            if (basePositions[i].kind === 8) continue;
            const r = p.length();
            let f = axis === 'none' ? 1.0 : radial;
            if (backgroundBusy > 0.01 && r > 1.5) f += TUNE.bgEdge * backgroundBusy * Math.sin(time * 0.9 + r * 2.0);
            if (f !== 1.0) {
                if (axis === 'xz') { p.x *= f; p.z *= f; }
                else if (axis === 'y') { p.y *= f; }
                else if (axis === 'none') { p.multiplyScalar(1.0 + (f - 1.0) * 0.35); }   // breath only, gentled
                else p.multiplyScalar(f);
            }
            if (writeW > 0.01) {
                // Laminar wave toward the viewer: a travelling front on z.
                p.z += TUNE.writeWave * writeW * CALM * Math.sin(time * 2.4 - r * 3.0);
            }
            if (shudder > 0) {
                p.x += shudder * Math.sin(time * 61.0 + i * 1.7);
                p.y += shudder * Math.cos(time * 53.0 + i * 2.3);
            }
            if (twitchP && twitchAmp > 0.01) {
                const dx = p.x - twitchP.x, dy = p.y - twitchP.y, dz = p.z - twitchP.z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 0.5 && d2 > 1e-6) {
                    const w = (1 - d2 / 0.5) * twitchAmp * TUNE.twitch * CALM;
                    const inv = 1 / Math.sqrt(d2);
                    p.x += dx * inv * w; p.y += dy * inv * w; p.z += dz * inv * w;
                }
            }
        }
    }
    // Idle twitch: at rest, every 6–14s one neighbourhood shivers — so
    // "rest" is not one uniform drift. Never while working. The clock
    // advances 0.3 units per second (0.005 × 60), so the interval is
    // 0.3 × seconds — it was written as 0.03 × seconds, firing every
    // ~1.5s (2026-09-20). The rendered amplitude eases up over ~4
    // frames rather than popping to full in one.
    idleTwitch *= decay(0.94);
    twitchAmp += (idleTwitch - twitchAmp) * ease(0.35);
    if (drive < 0.15 && userTurnState < 0.1 && !PREFERS_REDUCED_MOTION
        // a twitch the temperament asks for still waits for the last one to
        // fade (≥ 4 s, amplitude ~0): moving the twitch to a NEW node while
        // the old one is still up pops that node (0.054/frame, measured).
        && ((TS.twitch && time - idleTwitchAt > 0.3 * 4 && twitchAmp < 0.05)
            || (time - idleTwitchAt > 0.3 * (9 + Math.random() * 12) && Math.random() < 0.012 * dtF))) {
        idleTwitchAt = time;
        idleTwitch = 1.0;
        idleTwitchNode = Math.floor(Math.random() * NODE_COUNT);
    }

    // Reorganization blend: after a form switch, ease from the snapshot
    // of the old body into the freshly computed one.
    if (formBlend < 1.0) {
        formBlend = Math.min(1.0, formBlend + (dtF / 60) / 1.4);
        const e = formBlend * formBlend * (3.0 - 2.0 * formBlend);
        for (let i = 0; i < NODE_COUNT; i++) {
            const from = _blendFrom[i];
            if (from) currentPositions[i].lerpVectors(from, currentPositions[i], e);
        }
    }

    // The empty form at rest draws nothing (see _emptySettled): skip the
    // O(n²) proximity pass, the instance uploads and the composite. The
    // hue clock still advances so a switch out resumes it seamlessly.
    if (FORM === 'empty' && formBlend >= 1.0 && _emptySettled) {
        _advanceHue();
        return;
    }

    // 2. Update lines and track connectivity
    const linePosAttr = lineGeometry.attributes.position.array;
    const lineUvAttr = lineGeometry.attributes.aLightPass.array;
    const lineHueAttr = lineGeometry.attributes.aLineHue.array;
    let lineIdx = 0;

    // NB: errors no longer sever connections. The old
    // `errorState > 0.5 → connectionProbability 0` made every link
    // vanish at once (the graph "disintegrated") on any log line
    // containing ERROR — spectacular, but the opposite of "alive".
    const connected = _connected.fill(false);   // reused scratch (§4PR)
    const lineFadeAttr = lineGeometry.attributes.aLineFade.array;

    // cube2 draws its shells as explicit polylines (a wireframe cube's
    // edges span far more than any proximity radius at the outer
    // scales); those nodes are skipped by the O(n²) pass below.
    if (FORM === 'moodtesseract') {
        lineIdx = _emitCube2Edges(linePosAttr, lineUvAttr, lineHueAttr, lineFadeAttr, connected, lineIdx);
    }

    // The web THICKENS while immersed: easing the proximity threshold
    // up with the dive forms more links exactly when the viewer is in
    // the middle of them (the O(n²) distances are computed either way;
    // this only accepts more pairs, bounded by MAX_LINES).
    // (Per-form multipliers: LINK_MULT, module scope.)
    const proximitySq = PROXIMITY_SQ
        * (LINK_MULT[FORM] === undefined ? 1.0 : LINK_MULT[FORM])
        * (1.0 + TUNE.thickenLinks * gaitThicken)    // read, 'thicken' dialect
        * _linkBudget;                                // see LINK_BUDGET_MIN
    let _pairsFound = 0;                              // every qualifying pair, drawn or not

    for (let i = 0; i < NODE_COUNT; i++) {
        if (_noProx[i]) continue;
        for (let j = i + 1; j < NODE_COUNT; j++) {
            if (_noProx[j]) continue;
            const distSq = currentPositions[i].distanceToSquared(currentPositions[j]);
            if (distSq < proximitySq) {
                {
                    connected[i] = true;
                    connected[j] = true;
                    _pairsFound++;

                    if (lineIdx < MAX_LINES) {
                        linePosAttr[lineIdx * 6] = currentPositions[i].x;
                        linePosAttr[lineIdx * 6 + 1] = currentPositions[i].y;
                        linePosAttr[lineIdx * 6 + 2] = currentPositions[i].z;

                        linePosAttr[lineIdx * 6 + 3] = currentPositions[j].x;
                        linePosAttr[lineIdx * 6 + 4] = currentPositions[j].y;
                        linePosAttr[lineIdx * 6 + 5] = currentPositions[j].z;

                        lineUvAttr[lineIdx * 2] = 0;
                        lineUvAttr[lineIdx * 2 + 1] = 1;
                        lineFadeAttr[lineIdx * 2] = 1.0;
                        lineFadeAttr[lineIdx * 2 + 1] = 1.0;

                        // Endpoint hues — the fragment shader gradients
                        // between them along the segment.
                        lineHueAttr[lineIdx * 2] = nodeSeeds[i];
                        lineHueAttr[lineIdx * 2 + 1] = nodeSeeds[j];
                        lineIdx++;
                    }
                }
            }
        }
    }
    // Link budget feedback (see _linkBudget): tighten a little after an
    // overflow frame, relax once comfortably under. Per-frame steps (not
    // dt-scaled on purpose: this is a controller on the emitter's own
    // output, and one frame IS its sampling period).
    if (_pairsFound > MAX_LINES - 1) {
        _linkBudget = Math.max(LINK_BUDGET_MIN, _linkBudget * 0.97);
    } else if (_linkBudget < 1.0 && _pairsFound < MAX_LINES * 0.92) {
        _linkBudget = Math.min(1.0, _linkBudget / 0.99);
    }
    // Recall comet: a hot streak from the periphery into the recalled
    // node, shortening as it arrives; the node itself flares below.
    if (recallSpark > 0 && recallNode >= 0 && recallNode < NODE_COUNT && lineIdx < MAX_LINES
        && FORM !== 'empty') {
        const N = currentPositions[recallNode];
        connected[recallNode] = true;
        const reach = TUNE.recallReach * recallSpark * recallSpark;   // eased approach
        linePosAttr[lineIdx * 6] = N.x + recallDir[0] * reach;
        linePosAttr[lineIdx * 6 + 1] = N.y + recallDir[1] * reach;
        linePosAttr[lineIdx * 6 + 2] = N.z + recallDir[2] * reach;
        linePosAttr[lineIdx * 6 + 3] = N.x; linePosAttr[lineIdx * 6 + 4] = N.y; linePosAttr[lineIdx * 6 + 5] = N.z;
        lineUvAttr[lineIdx * 2] = 0; lineUvAttr[lineIdx * 2 + 1] = 1;
        lineFadeAttr[lineIdx * 2] = 1.0; lineFadeAttr[lineIdx * 2 + 1] = 1.0;
        lineHueAttr[lineIdx * 2] = 0.60; lineHueAttr[lineIdx * 2 + 1] = 0.58;
        lineIdx++;
    }
    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.aLightPass.needsUpdate = true;
    lineGeometry.attributes.aLineHue.needsUpdate = true;
    lineGeometry.attributes.aLineFade.needsUpdate = true;
    lineGeometry.setDrawRange(0, lineIdx * 2);

    // 3. Update nodes meshes (hide unconnected nodes)
    const dummy = _dummy;   // reused scratch (§4PR)
    for (let i = 0; i < NODE_COUNT; i++) {
        const targetScale = connected[i] ? 1.0 : 0.0;
        nodeScales[i] += (targetScale - nodeScales[i]) * ease(0.1); // Smooth scale in and out

        // Per-node size factor (bp.sz): anatomy builders shrink nodes in
        // regions where additive stacking would otherwise wash out the
        // thermal hue (e.g. the horizon core). Default 1. The horizon
        // core additionally FLARES in size when the infall surges land.
        const bpi = basePositions[i];
        const s = nodeScales[i] * (bpi.sz || 1.0)
            * (i === recallNode ? 1.0 + TUNE.recallFlare * 4.0 * recallSpark * (1.0 - recallSpark) : 1.0);   // bell, peaks mid-flight
        if (s < 0.001) {
            dummy.scale.set(0, 0, 0);
            dummy.position.set(9999, 9999, 9999);
        } else {
            dummy.scale.set(s, s, s);
            dummy.position.copy(currentPositions[i]);
        }

        dummy.updateMatrix();
        instancedMesh.setMatrixAt(i, dummy.matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;

    // Thermal breathing: with ANATOMICAL color placement (cold crown,
    // hot tentacles/core), a monotonic wheel drift would rotate the
    // body's colors out of their anatomy — so the drift now OSCILLATES
    // (±0.07 of the ring, ~18s per swing at idle, faster when busy,
    // rate wandering like weather): regions warm and cool into each
    // other without the body plan ever scrambling.
    _advanceHue();

    // Per-form center dimming: horizon's core region is by far the
    // densest additive pile-up; the vortex dims by SCREEN-radial
    // distance (its hot zone sits on the view axis at depth) and damps
    // the hue wave/drift so its center stays anchored DARK RED instead
    // of swinging through the plum stop ("bright purple" report).
    // cube2 v3: the kernel IS the hot focal point — no centre dim there.
    const centerDim = FORM === 'moodtesseract' ? 0.0 : 0.30;
    const centerXY = 0.0;
    const waveAmp = 1.0;
    // Mood rides the drift as a slow baseline (cold-pole shift only;
    // the vortex keeps its anchored centre).
    const hueDriftOut = hueDrift * _moodNow.drift + moodHue + _moodNow.hue;
    // Master luminance: EVERY form emits ~half the light so the face
    // BLENDS into the background (operator: "too bright and
    // distracting", then extended to all faces) — structure and motion
    // carry visibility, not brightness. Error KINDS modulate it: a
    // network drop flickers with gaps, a timeout fades; a pass holds a
    // touch brighter while it crystallises.
    let formDim = 0.55;
    if (errorKind === 'network') {
        // Reduced motion: a STEADY dim at the flicker's duty cycle (the
        // gate is open ~40% of the time) instead of a ~4 Hz blink (§4PR).
        formDim *= 1.0 - TUNE.netFlicker * errorKindEnv
            * (PREFERS_REDUCED_MOTION ? NET_FLICKER_DUTY : (Math.sin(time * 90.0) > 0.3 ? 1.0 : 0.0));
    } else if (errorKind === 'timeout') {
        formDim *= 1.0 - TUNE.timeoutFade * errorKindEnv;
    }
    if (verdict === 'pass') formDim *= 1.0 + TUNE.passDim * verdictEnv;
    // temperament: flares, dozing dim, speech glow — never over an error
    // kind's own dim (network flicker, timeout fade, refusal) and never
    // under reduced motion (a steady face)
    if (!PREFERS_REDUCED_MOTION && !errorKind) formDim *= TS.bright;
    formDim *= _moodNow.bright;

    const nUniforms = instancedMesh.material.uniforms;
    const clk = shaderClock(time);   // wrapped / CPU-side phases (§4PR)
    nUniforms.uTime.value = clk.uTime;
    nUniforms.uCenterDim.value = centerDim;
    nUniforms.uCenterXY.value = centerXY;
    nUniforms.uWaveAmp.value = waveAmp;
    nUniforms.uFormDim.value = formDim;
    nUniforms.uWorkingState.value = workingState;
    nUniforms.uErrorState.value = errorState;
    nUniforms.uPulseT.value = pulseT;
    nUniforms.uAudioLevel.value = audioLevel;
    nUniforms.uSweep.value = gait.search;
    nUniforms.uSweepAngle.value = sweepAngle;
    nUniforms.uSweepMode.value = DIAL.search === 'plane' ? 1.0 : (DIAL.search === 'ring' ? 2.0 : 0.0);
    nUniforms.uSweepHeat.value = TUNE.sweepHeat;
    // The ring scan reaches the whole stack in cube2 (shells grow to ~8).
    const sweepReach = FORM === 'moodtesseract' ? 7.0 : 2.4;
    nUniforms.uSweepReach.value = sweepReach;
    nUniforms.uAccentStrength.value = accentStrength;
    nUniforms.uHueDrift.value = hueDriftOut;

    const lUniforms = lineMaterial.uniforms;
    lUniforms.uTime.value = clk.uTime;
    lUniforms.uPacketPhase.value = clk.packet;
    lUniforms.uBurstPhase.value = clk.burst;
    lUniforms.uCenterDim.value = centerDim;
    lUniforms.uCenterXY.value = centerXY;
    lUniforms.uWaveAmp.value = waveAmp;
    lUniforms.uFormDim.value = formDim;
    lUniforms.uWorkingState.value = workingState;
    lUniforms.uErrorState.value = errorState;
    lUniforms.uPulseT.value = pulseT;
    lUniforms.uAccentStrength.value = accentStrength;
    lUniforms.uHueDrift.value = hueDriftOut;
    lUniforms.uSweep.value = gait.search;
    lUniforms.uSweepAngle.value = sweepAngle;
    lUniforms.uSweepHeat.value = TUNE.sweepHeat;
    lUniforms.uSweepReach.value = sweepReach;
    lUniforms.uLineSweep.value = FORM === 'moodtesseract' ? 1.0 : 0.0;


    // Bloom breathes with the envelope. The old formula added
    // errorState * 2.2 (a 3x glow flashbang on any error line) and
    // pulseT * 0.5 per event — together the main source of "flashing".
    // Immersion DAMPS bloom (×0.65 fully inside): at close range the
    // additive quads are already bright, and the streaming reply is
    // being read right on top of them — inside should feel vast and
    // dim, not blinding.
    // Base lowered 1.15 → 0.95 → 0.88 across the 2026-07-28 passes: the
    // medusa is meant to sit BACK — present, not clamoring — and its
    // locally dense anatomy stacks additive light. Working/activity
    // response unchanged, so a busy agent still visibly glows.
    // Background-blend bloom (×0.65, all forms — see formDim above).
    bloomPass.strength = (0.88 + workingState * 0.3 + activity * 0.35
        + errorState * 0.5 + 0.20 * toolPulse + 0.25 * recallSpark
        + (verdict === 'pass' ? 0.2 * verdictEnv : 0)
        + 0.08 * backgroundBusy * (0.5 + 0.5 * Math.sin(time * 0.31 + 0.9))
        + 0.35 * TS.glow)
        * BLOOM_SCALE * 0.65 * _moodNow.bright;

    composer.render();

    // Did the empty form just draw its last visible pixel? (Blend done, no
    // line, every node faded out — then the frame above was black and the
    // next ones would be too.)
    if (FORM === 'empty' && formBlend >= 1.0 && lineIdx === 0) {
        let lit = false;
        for (let i = 0; i < NODE_COUNT; i++) if (nodeScales[i] >= 0.001) { lit = true; break; }
        _emptySettled = !lit;
    } else {
        _emptySettled = false;
    }
}
