import * as THREE from 'three';
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
const MAX_LINES = IS_MOBILE ? 2500 : 10000;
const BLOOM_SCALE = IS_MOBILE ? 0.6 : 1.0;
// Tightened 2.5 → 1.7 with the medusa anatomy: the body is locally
// dense (membrane, strands), so a wide link radius smeared the form
// back into an undifferentiated web. Shorter links trace the anatomy.
const PROXIMITY_SQ = 1.7;

let scene, camera, renderer, composer, bloomPass;
let instancedMesh, linesMesh;
let lineGeometry, nodeMaterial, lineMaterial;
let motesMesh, motesMaterial;
let time = 0;
let currentShapeSpeed = SPEEDS.idle;
let animationFrameId;

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

// ── Vortex (form 'vortex') — self-similar swallow ──────────────────
// A logarithmic cone converging at the singularity (origin), opening
// toward the camera. Because the taper is EXPONENTIAL the funnel is
// self-similar — the conveyor's depth wrap (fract) is an infinite
// fractal zoom: matter streams in from beyond the screen edges, is
// consumed at the apex, respawns at the rim, forever. THE CAMERA NEVER
// MOVES — busy and idle are the same scene at different intensities,
// so completion needs no reset at all: the view just morphs back as
// the flow calms (operator: "don't pull the camera out; morph the
// default view like an infinite fractal — the main animation is not
// the spinning, it's the swallow").
//   vortexTravel = eased user-turn ENGAGEMENT (0..1): drives flow
//     surge, harmonic shape amplitude, and the singularity's appetite.
//   tunnelFlow   = accumulated conveyor depth (the swallow itself).
//   vortexSpin   = slow ambient swirl — deliberately NOT the star.
// In-falling matter morphs through procedural, never-repeating shapes:
// a drifting field of low-order harmonics deforms the cross-section.
let vortexTravel = 0.0;
let tunnelFlow = 0.0;
let vortexSpin = 0.0;
// v4 (operator): expansion flow. Contraction (matter streaming edges→
// center) reads as moving BACKWARD; a black-hole fall needs the hole
// AHEAD and the walls expanding outward past the viewer. The singularity
// (dark-red shadow + accretion ring) sits deep on the view axis; stream
// matter EMERGES from its glow (L small) and blooms outward/past the
// camera (L grows exponentially — still a self-similar infinite fall).
const VORTEX_APEX_Z = -2.0;   // the hole, deep on the view axis
// Spawn AT the ring's edge, not inside it: matter emerges THROUGH the
// accretion ring, so the shadow interior keeps only its dim embers and
// reads as a genuinely dark hole.
const VORTEX_LMIN = 0.55;
const VORTEX_KOUT = 2.6;      // exponential expansion — self-similarity knob
const VORTEX_COS = 0.60, VORTEX_SIN = 0.80;   // cone half-angle

// ── AI-form state (2026-07-29) ─────────────────────────────────────
// embedding: the query comet's flight state. From/to are cluster
// indices; embT is eased 0..1 along a bezier whose control point is
// pushed outward so the flight arcs through the void between concepts.
let embFrom = 0, embTo = 1, embT = 0.5;
const embExcite = [];        // per-cluster recall glow, decays ~2s
let _embCenters = [];        // rebuilt by _buildEmbedding
// descent: the optimizer bead. True gradient descent on the (moving)
// loss surface — velocity, damping, soft walls, stuck-kick.
let beadX = 0.4, beadZ = 0.3, beadVX = 0.0, beadVZ = 0.0;
let beadStill = 0.0;         // seconds spent near-stationary
const beadTrail = [];        // recent (x,z) surface points, newest first
let _descTick = 0;
// cube: the resident complexities (anchors rebuilt with the form) and
// the active mutation's strength. Growth is driven by the USER turn
// (mirrors vortexTravel — ambient work only adds restlessness), decay
// is a slow organic taming after completion.
let _cubeCx = [];            // [{ax,ay,az, r0, ph}]
let cubeActive = 0;          // index of the mutating complexity
let cubeS = 0.0;             // active mutation strength 0..1
let _cubePrevTurn = false;   // rising-edge detector for picking anchor

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

// ── Forms (roster trimmed 2026-09-12) ─────────────────────────────
// Interchangeable body plans over ONE motion engine (the asymmetric
// _pulseShape propulsion, hot core, flinch, thermal anatomy):
//   vortex    — a black hole ahead of the viewer; self-similar
//               expansion flow, accretion ring, dark shadow.
//   lattice   — the weight tensor: a tumbling crystal grid; diagonal
//               activation waves heat the sites they cross; a hot
//               attention kernel drifts the volume, bending the grid
//               toward itself; charge runners ride the axes.
//   embedding — latent space: cold concept clusters on slow orbits; a
//               hot query comet streaks cluster→cluster, and each
//               arrival IGNITES the recalled cluster.
//   descent   — the loss landscape: an undulating terrain sheet, cold
//               ridges / warm valleys; the optimizer bead rolls
//               downhill trailing heat, kicked to explore again
//               whenever it settles (and on error flinches).
//   cube      — the infinite monolith: a large dark cube with a FEW
//               resident alien complexities quietly deforming it from
//               inside; a user turn wakes one, which spreads like an
//               infection, running crimson, and tames on completion.
//   empty     — no face.
// Removed 2026-09-12 at the operator's request: abyssal, horizon,
// cortex, stack, conversation, toolgraph (and the auto mode + face lab).
// The header's form button opens a picker built from this roster; the
// choice persists (server-side + localStorage).
const FORMS = ['vortex', 'lattice', 'embedding', 'descent', 'cube', 'empty'];
// Default form: VORTEX (operator pick, 2026-07-28 — superseded horizon
// after the black-hole iteration). The form the operator last picked
// overrides it — resolved by `resolveInitialForm` below.
let formIndex = FORMS.indexOf('vortex');

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
    for (const candidate of [serverForm, storedForm]) {
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

// Interior atmosphere: tiny drifting motes that only exist while
// immersed. The dive dilutes local node density (the scene scales up
// around the viewer), so the inside looked empty — the motes are the
// "data nebula" filling that void. Drift is computed in the vertex
// shader from a per-mote seed, so the layer costs one draw call and
// zero per-frame CPU; it is skipped entirely (visible=false) at rest.
const MOTE_COUNT = IS_MOBILE ? 150 : 400;

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
    vortex:       { radialAxis: 'none', search: 'ring',  read: 'thicken',  tool: 'kick',   write: 'flow', verify: 'still' },
    lattice:      { radialAxis: 'none', search: 'plane', read: 'settle',   tool: 'flash',  write: 'flow', verify: 'align' },
    embedding:    { radialAxis: 'all',  search: 'ring',  read: 'contract', tool: 'kick',   write: 'flow', verify: 'still' },
    descent:      { radialAxis: 'y',    search: 'plane', read: 'settle',   tool: 'kick',   write: 'flow', verify: 'still' },
    cube:         { radialAxis: 'none', search: 'plane', read: 'settle',   tool: 'flash',  write: 'flow', verify: 'align' },
    empty:        { radialAxis: 'none', search: 'sweep', read: 'settle',   tool: 'kick',   write: 'wave', verify: 'still' },
});
const _DIALECT_DEFAULT = DIALECTS.vortex;
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

export function destroy() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    const container = document.getElementById('sphere-container');
    if (container && renderer && renderer.domElement) {
        container.removeChild(renderer.domElement);
    }
    basePositions.length = 0;
    if (nodeMaterial) nodeMaterial.dispose();
    if (lineMaterial) lineMaterial.dispose();
    if (motesMaterial) motesMaterial.dispose();
    if (instancedMesh && instancedMesh.geometry) instancedMesh.geometry.dispose();
    if (linesMesh && linesMesh.geometry) linesMesh.geometry.dispose();
    if (motesMesh && motesMesh.geometry) motesMesh.geometry.dispose();
    if (renderer) renderer.dispose();
    window.removeEventListener('resize', handleResize);
}

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
    float dRing = abs(length(instancePos) - scan * 2.4);
    float sweepD = uSweepMode < 0.5 ? dAz : (uSweepMode < 1.5 ? dPlane : dRing);
    float sweepW = uSweepMode < 0.5 ? 0.9 : 0.45;
    float sweepHeat = uSweep * uSweepHeat * smoothstep(sweepW, 0.0, sweepD);
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
varying float vLightPass;
varying float vLineHue;
varying float vLineDepth;
varying float vLineNear;
varying vec3 vLinePos;

void main() {
    vLightPass = aLightPass;
    vLineHue = aLineHue;
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
${paletteGLSL}
${hueWaveGLSL}
varying float vLightPass;
varying float vLineHue;
varying float vLineDepth;
varying float vLineNear;
varying vec3 vLinePos;

void main() {
    float gradient = fract(vLightPass * 1.5 - uTime * 2.0);

    // Smooth sharp front tail for data packets traveling along lines
    float pulse = smoothstep(0.0, 0.5, gradient) * smoothstep(1.0, 0.9, gradient);

    // Shockwave: a second, faster burst rides the lines when pulseT is
    // non-zero. Disappears completely at pulseT==0 so idle lines are
    // unchanged.
    float burst = smoothstep(0.0, 0.25,
        fract(vLightPass * 1.5 - uTime * 6.0 - (1.0 - uPulseT))
    ) * smoothstep(1.0, 0.85,
        fract(vLightPass * 1.5 - uTime * 6.0 - (1.0 - uPulseT))
    );

    // Each line is a GRADIENT between its two endpoint nodes' thermal
    // hues (vLineHue interpolates the endpoint seeds), riding the same
    // traveling hue wave as the nodes so links and their endpoints stay
    // in the same temperature current.
    vec3 jewel = palette(vLineHue + uHueDrift + hueWave(vLinePos));
    float colorMix = sin(vLightPass * 10.0 + uTime * uOrganic * 0.4 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor + jewel * 0.18, jewel, colorMix);

    vec3 col = mix(mixCol, uErrorColor, uErrorState * 0.8);
    col = mix(col, uAccentColor, clamp(uAccentStrength * 0.6, 0.0, 0.45));

    float alpha = mix(0.30 + uWorkingState * 0.18, 1.0, pulse);
    alpha = max(alpha, burst * uPulseT);
    alpha *= vLineNear;
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

const moteVertexShader = `
attribute float aSeed;
uniform float uTime;
uniform float uDive;
uniform float uHueDrift;
uniform float uDpr;
${paletteGLSL}
${hueWaveGLSL}
varying vec3 vMoteCol;
varying float vMoteFade;

void main() {
    // Slow, per-mote orbital drift — pure shader math, no CPU updates.
    vec3 p = position + 0.18 * vec3(
        sin(uTime * 0.40 + aSeed * 37.0),
        cos(uTime * 0.33 + aSeed * 53.0),
        sin(uTime * 0.27 + aSeed * 71.0)
    );
    vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
    // Near fade (don't smear across the lens) + gentle far fade, all
    // gated by the dive: motes are invisible at rest by construction.
    float near = smoothstep(0.12, 0.8, -mvPosition.z);
    float far = 1.0 - clamp((-mvPosition.z - 3.0) / 6.0, 0.0, 0.85);
    vMoteFade = near * far * uDive;
    vMoteCol = palette(aSeed + uHueDrift + hueWave(position));
    // Perspective size attenuation, clamped so close motes stay motes.
    gl_PointSize = min(mix(1.5, 4.5, fract(aSeed * 7.31))
        * (2.2 / max(-mvPosition.z, 0.2)) * uDpr, 9.0 * uDpr);
    gl_Position = projectionMatrix * mvPosition;
}
`;

const moteFragmentShader = `
varying vec3 vMoteCol;
varying float vMoteFade;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c) * 2.0;
    if (d > 1.0) discard;
    float a = pow(1.0 - d, 2.5) * 0.55 * vMoteFade;
    gl_FragColor = vec4(vMoteCol * a, a);
}
`;

// ── Anatomy builders ───────────────────────────────────────────────
// Each fills basePositions + nodeSeeds for exactly NODE_COUNT nodes.
// The thermal poles become BODY PLAN in every form: cold blue outer
// structure, violet transitions, hot arterial strands and core.

function _buildAnatomy() {
    basePositions.length = 0;
    const f = FORMS[formIndex];
    if (f === 'lattice') _buildLattice();
    else if (f === 'embedding') _buildEmbedding();
    else if (f === 'descent') _buildDescent();
    else if (f === 'cube') _buildCube();
    else if (f === 'empty') _buildEmpty();
    else _buildVortex();
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

// Form D — VORTEX: a black hole facing the viewer. Spiral arms wind
// down a funnel whose throat holds the crimson singularity; the whole
// structure swirls (faster with ambient work), matter heats from cold
// blue at the mouth to arterial red at the throat, and user turns send
// the camera traveling down the tunnel (see vortexTravel above).
function _buildVortex() {
    const ARMS = 5;
    const CORE_COUNT = Math.max(8, Math.round(NODE_COUNT * 0.055));
    const RING_COUNT = Math.round(NODE_COUNT * 0.14);
    const HAZE_COUNT = Math.round(NODE_COUNT * 0.32);
    const ARM_COUNT = NODE_COUNT - CORE_COUNT - RING_COUNT - HAZE_COUNT;
    let n = 0;

    const pushStream = (armLocked, count) => {
        for (let b = 0; b < count; b++, n++) {
            const d = Math.random();
            // theta0 is the STATIC arm anchor — spiral wind + swirl are
            // applied per frame from the node's FLOWING progress.
            const theta0 = armLocked
                ? (b % ARMS) * (Math.PI * 2 / ARMS) + (Math.random() - 0.5) * 0.20
                : Math.random() * Math.PI * 2;
            basePositions.push({
                kind: armLocked ? 0 : 1, d0: d, theta0,
                // Wide speed spread: streams genuinely SHEAR past each
                // other instead of riding one conveyor (monotony fix).
                flowScale: 0.6 + Math.random() * 1.0,
                // Per-node trajectory: each grain flies its own cone
                // angle, so the outward bloom has parallax variety.
                sinA: VORTEX_SIN * (0.78 + Math.random() * 0.45),
                cosA: VORTEX_COS * (0.90 + Math.random() * 0.20),
                rJit: 0.94 + Math.random() * 0.12,
                seedJit: Math.random() * 0.03,
                jit: Math.random() * Math.PI * 2,
                sz: armLocked ? 1.0 : 0.85,
            });
            nodeSeeds[n] = 0.60 - d * 0.56;              // overwritten per frame
        }
    };
    pushStream(true, ARM_COUNT);
    pushStream(false, HAZE_COUNT);

    // The accretion ring — the black hole's icon. A dark-red band
    // orbiting the shadow, slightly elliptic for perspective.
    for (let k = 0; k < RING_COUNT; k++, n++) {
        basePositions.push({
            kind: 3,
            th0: (k / RING_COUNT) * Math.PI * 2 + Math.random() * 0.15,
            rRing: 0.62 * (0.92 + Math.random() * 0.16),
            zJit: (Math.random() - 0.5) * 0.10,
            jit: Math.random() * Math.PI * 2,
            sz: 0.9,
        });
        nodeSeeds[n] = 0.595 + Math.random() * 0.02;     // anchored dark red
    }

    // The shadow — dim crimson embers inside the ring.
    for (let c = 0; c < CORE_COUNT; c++, n++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const rr = 0.18 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2,
            hx: rr * Math.sin(ph) * Math.cos(th),
            hy: rr * Math.sin(ph) * Math.sin(th),
            hz: rr * Math.cos(ph) * 0.5,
            jit: Math.random() * Math.PI * 2,
            sz: 0.55,
        });
        nodeSeeds[n] = 0.60 + Math.random() * 0.02;
    }
}

// Form F — LATTICE: the weight tensor. A slowly tumbling crystal grid,
// cell edge tuned just under the (form-tightened) link radius so ONLY
// axis-neighbors weave — a literal wireframe tensor, the first angular
// silhouette. Cold at the corners, warming toward the middle; diagonal
// activation waves and a drifting hot attention kernel do the living.
const LATTICE_N = IS_MOBILE ? [4, 5, 5] : [6, 6, 6];
const LATTICE_A = IS_MOBILE ? 0.80 : 0.72;   // cell edge
function _buildLattice() {
    const [NX, NY, NZ] = LATTICE_N;
    const SITES = NX * NY * NZ;
    const RUNNERS = Math.min(IS_MOBILE ? 12 : 20, NODE_COUNT - SITES - 8);
    const spanOf = (m) => ((m - 1) / 2) * LATTICE_A;
    let n = 0;
    for (let ix = 0; ix < NX; ix++)
        for (let iy = 0; iy < NY; iy++)
            for (let iz = 0; iz < NZ; iz++, n++) {
                // Chebyshev shell 0 (center) → 1 (corner): the cold
                // base gradient — corners coldest, heart warmer.
                const shell = Math.max(
                    Math.abs(ix - (NX - 1) / 2) / Math.max((NX - 1) / 2, 1),
                    Math.abs(iy - (NY - 1) / 2) / Math.max((NY - 1) / 2, 1),
                    Math.abs(iz - (NZ - 1) / 2) / Math.max((NZ - 1) / 2, 1));
                basePositions.push({
                    kind: 0,
                    gx: (ix - (NX - 1) / 2) * LATTICE_A,
                    gy: (iy - (NY - 1) / 2) * LATTICE_A,
                    gz: (iz - (NZ - 1) / 2) * LATTICE_A,
                    // Diagonal wave phase 0..1 across the volume.
                    wp: (ix + iy + iz) / (NX + NY + NZ - 3),
                    seed0: 0.20 - shell * 0.16 + Math.random() * 0.02,
                    jit: Math.random() * Math.PI * 2,
                });
                nodeSeeds[n] = basePositions[n].seed0;   // reheated per frame
            }
    // Charge runners: violet packets riding the grid lines, wrapping
    // face to face (size-tapered at the faces so the wrap hides).
    for (let k = 0; k < RUNNERS; k++, n++) {
        const axis = Math.floor(Math.random() * 3);
        const cell = (m) => (Math.floor(Math.random() * m) - (m - 1) / 2) * LATTICE_A;
        basePositions.push({
            kind: 1,
            dx: axis === 0 ? 1 : 0, dy: axis === 1 ? 1 : 0, dz: axis === 2 ? 1 : 0,
            ox: axis === 0 ? 0 : cell(NX),
            oy: axis === 1 ? 0 : cell(NY),
            oz: axis === 2 ? 0 : cell(NZ),
            span: spanOf(axis === 0 ? NX : axis === 1 ? NY : NZ) + 0.30,
            d0: Math.random(),
            speed: 0.05 + Math.random() * 0.08,
            jit: Math.random() * Math.PI * 2,
            sz: 0.85,
        });
        nodeSeeds[n] = 0.44 + Math.random() * 0.05;
    }
    // The attention kernel: a hot locus scanning the tensor (its
    // drifting center is computed per frame; these are its offsets).
    while (n < NODE_COUNT) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const rr = 0.26 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2,
            hx: rr * Math.sin(ph) * Math.cos(th),
            hy: rr * Math.cos(ph),
            hz: rr * Math.sin(ph) * Math.sin(th),
            jit: Math.random() * Math.PI * 2,
            sz: 0.85,
        });
        nodeSeeds[n] = 0.58 + Math.random() * 0.04;
        n++;
    }
}

// Form H — EMBEDDING: latent space. Cold concept clusters anchored on
// jittered octahedral directions (≥ ~80° apart, so with the tightened
// link radius clusters can never cross-link — the void between
// concepts stays void), each with its own slow orbit and internal
// swirl. The hot query comet streaks cluster→cluster doing recall.
const EMB_CLUSTERS = IS_MOBILE ? 5 : 6;
function _buildEmbedding() {
    const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    _embCenters = [];
    embExcite.length = 0;
    for (let c = 0; c < EMB_CLUSTERS; c++) {
        const d = DIRS[c];
        const R = 1.50 + Math.random() * 0.25;
        _embCenters.push({
            bx: d[0] * R + (Math.random() - 0.5) * 0.30,
            by: d[1] * R + (Math.random() - 0.5) * 0.30,
            bz: d[2] * R + (Math.random() - 0.5) * 0.30,
            oa: Math.random() * Math.PI * 2,          // orbit phase
            ow: 0.04 + Math.random() * 0.03,          // orbit rate
            or: 0.10 + Math.random() * 0.08,          // orbit radius
            spin: 0.08 + Math.random() * 0.07,        // internal swirl
        });
        embExcite.push(0);
    }
    const QUERY = IS_MOBILE ? 10 : 16;
    const CL_TOTAL = NODE_COUNT - QUERY;
    const per = Math.floor(CL_TOTAL / EMB_CLUSTERS);
    let n = 0;
    for (let c = 0; c < EMB_CLUSTERS; c++) {
        const count = c === EMB_CLUSTERS - 1
            ? CL_TOTAL - per * (EMB_CLUSTERS - 1) : per;
        for (let b = 0; b < count; b++, n++) {
            const th = Math.random() * Math.PI * 2;
            const ph = Math.acos(2 * Math.random() - 1);
            basePositions.push({
                kind: 0, ci: c,
                dx: Math.sin(ph) * Math.cos(th),
                dy: Math.cos(ph),
                dz: Math.sin(ph) * Math.sin(th),
                r0: 0.42 * Math.cbrt(Math.random()),
                jit: Math.random() * Math.PI * 2,
                // Per-cluster temperature: distinct concepts sit at
                // slightly different colds.
                seed0: 0.04 + c * 0.024 + Math.random() * 0.03,
            });
            nodeSeeds[n] = basePositions[n].seed0;   // reheated on recall
        }
    }
    while (n < NODE_COUNT) {   // the query comet (head → tail)
        const qi = n - CL_TOTAL;
        const QN = NODE_COUNT - CL_TOTAL;
        basePositions.push({
            kind: 1,
            s: qi / Math.max(QN - 1, 1),
            rr: 0.05 + Math.random() * 0.06,
            jit: Math.random() * Math.PI * 2,
            sz: 1.0,
        });
        nodeSeeds[n] = 0.60 - (qi / Math.max(QN, 1)) * 0.16;
        n++;
    }
    embFrom = 0;
    embTo = EMB_CLUSTERS > 1 ? 1 : 0;
    embT = 0.0;
}

// Form I — DESCENT: the loss landscape. A tilted terrain sheet whose
// height field slowly EVOLVES (the objective is non-stationary —
// training), cold on the ridges, warming in the valleys; the hot
// optimizer bead rolls true gradient descent across it, pressing the
// surface down where it passes, trailing heat, and taking a
// learning-rate kick whenever it settles too long (or on a flinch).
const DESC_NX = IS_MOBILE ? 11 : 15, DESC_NZ = IS_MOBILE ? 9 : 15;
const DESC_SX = 4.4, DESC_SZ = 3.4;   // sheet spans (x, z)
const DESC_H = 0.85;                  // height-field scale
function _lossH(x, z, t) {
    return DESC_H * (0.30 * Math.sin(1.7 * x + 0.9 * z + t * 0.13)
        + 0.22 * Math.sin(2.3 * z - 1.1 * x + t * 0.117)
        + 0.14 * Math.sin(2.2 * x + 3.1 * z + t * 0.071)
        + 0.10 * Math.sin(2.5 * (x + z) - t * 0.093));
}
function _buildDescent() {
    let n = 0;
    for (let ix = 0; ix < DESC_NX; ix++)
        for (let iz = 0; iz < DESC_NZ; iz++, n++) {
            basePositions.push({
                kind: 0,
                gx: -DESC_SX / 2 + DESC_SX * ix / (DESC_NX - 1),
                gz: -DESC_SZ / 2 + DESC_SZ * iz / (DESC_NZ - 1),
                jit: Math.random() * Math.PI * 2,
            });
            nodeSeeds[n] = 0.1;                      // recomputed per frame
        }
    while (n < NODE_COUNT) {   // the optimizer bead + cooling trail
        const bi = n - DESC_NX * DESC_NZ;
        const total = NODE_COUNT - DESC_NX * DESC_NZ;
        basePositions.push({
            kind: 1,
            s: bi / Math.max(total - 1, 1),          // 0 head → 1 tail end
            rr: 0.03 + Math.random() * 0.05,
            jit: Math.random() * Math.PI * 2,
            sz: 1.0,
        });
        nodeSeeds[n] = 0.60;
        n++;
    }
    beadX = 0.4; beadZ = 0.3; beadVX = 0; beadVZ = 0;
    beadStill = 0; beadTrail.length = 0;
}

// Form J — CUBE: the infinite monolith. A large dark grid — bigger
// cell edge than lattice, spanning past the view so the eye never
// finds a boundary — holding a FEW resident alien complexities:
// localized regions where incommensurate sine fields quietly deform
// the grid (not many; just enough to be interesting). One of them is
// the MUTATION: on a user turn it grows (aggressively but not fast),
// spreading through an irregular per-node boundary like an infection,
// re-weaving links as displaced nodes tear and rejoin, running
// crimson at its heart while the rest of the cube stays cold dark
// blue. The immersion dive is focus-translated onto it (same lesson
// as descent/embedding), so the camera rides INTO the mutation as it
// evolves; on completion it tames and the cube re-knits.
const CUBE_N = IS_MOBILE ? [5, 5, 4] : [6, 6, 6];
// Cell edge trimmed 0.88 → 0.80 (v2, operator: "the cube is not
// clear"): the whole silhouette must FIT the view so it reads as a
// monolith, not an endless field.
const CUBE_A = IS_MOBILE ? 0.85 : 0.80;
const CUBE_CX_COUNT = IS_MOBILE ? 2 : 3;
// Partial dive for this form: the point is WATCHING the mutation
// spread across the grid, not being inside a cloud of dots — the
// camera closes to ~3.1 (vs the global 1.3) and the swell is gentled.
const CUBE_DIVE_Z = 3.1;
function _buildCube() {
    const [NX, NY, NZ] = CUBE_N;
    const SITES = NX * NY * NZ;
    // Resident complexities: random interior anchors, kept apart so
    // they read as separate organisms living in the grid.
    _cubeCx = [];
    const spans = [
        ((NX - 1) / 2) * CUBE_A * 0.6,
        ((NY - 1) / 2) * CUBE_A * 0.6,
        ((NZ - 1) / 2) * CUBE_A * 0.6,
    ];
    for (let k = 0; k < CUBE_CX_COUNT; k++) {
        let ax = 0, ay = 0, az = 0;
        for (let attempt = 0; attempt < 24; attempt++) {
            ax = (Math.random() * 2 - 1) * spans[0];
            ay = (Math.random() * 2 - 1) * spans[1];
            az = (Math.random() * 2 - 1) * spans[2];
            const ok = _cubeCx.every(c =>
                Math.hypot(ax - c.ax, ay - c.ay, az - c.az) > 1.4);
            if (ok) break;
        }
        _cubeCx.push({
            ax, ay, az,
            r0: 0.50 + Math.random() * 0.15,      // idle influence radius
            ph: Math.random() * Math.PI * 2,       // personality phase
        });
    }
    cubeActive = 0;
    cubeS = 0.0;
    _cubePrevTurn = false;

    let n = 0;
    for (let ix = 0; ix < NX; ix++)
        for (let iy = 0; iy < NY; iy++)
            for (let iz = 0; iz < NZ; iz++, n++) {
                // How many axes sit on the cube's surface: 0 interior,
                // 1 face, 2 edge, 3 corner. The WIREFRAME SILHOUETTE is
                // what makes the monolith legible (v2): edge and corner
                // nodes render larger, tracing the cube's outline.
                const onFace =
                    (ix === 0 || ix === NX - 1 ? 1 : 0)
                    + (iy === 0 || iy === NY - 1 ? 1 : 0)
                    + (iz === 0 || iz === NZ - 1 ? 1 : 0);
                basePositions.push({
                    kind: 0,
                    gx: (ix - (NX - 1) / 2) * CUBE_A,
                    gy: (iy - (NY - 1) / 2) * CUBE_A,
                    gz: (iz - (NZ - 1) / 2) * CUBE_A,
                    // Darker than lattice — a monolith, not a machine.
                    seed0: 0.03 + onFace * 0.025 + Math.random() * 0.012,
                    sz: onFace >= 2 ? 1.3 : onFace === 1 ? 1.0 : 0.85,
                    // Per-node spread threshold: the mutation's boundary
                    // is IRREGULAR (some nodes resist, some succumb
                    // early) — that irregularity is what reads organic.
                    gate: 0.80 + Math.random() * 0.35,
                    jit: Math.random() * Math.PI * 2,
                });
                nodeSeeds[n] = basePositions[n].seed0;   // reheated per frame
            }
    // Each complexity's visible HEART: a small tangle of nodes orbiting
    // its anchor — CRIMSON already at rest (v2, operator: "the
    // mutations must be red"), burning brighter as its mutation grows.
    while (n < NODE_COUNT) {
        const ci = (n - SITES) % CUBE_CX_COUNT;
        const th = Math.random() * Math.PI * 2;
        const phv = Math.acos(2 * Math.random() - 1);
        const rr = 0.06 + 0.16 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2, ci,
            dx: Math.sin(phv) * Math.cos(th),
            dy: Math.cos(phv),
            dz: Math.sin(phv) * Math.sin(th),
            rr,
            jit: Math.random() * Math.PI * 2,
            sz: 0.8,
        });
        nodeSeeds[n] = 0.54 + Math.random() * 0.04;      // reheated per frame
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
    toolPulse = Math.min(1.0, toolPulse + 0.6);
    return toolPulse;
}

// A memory / knowledge-base hit: a comet from the periphery ignites one
// node. Cross-form (the embedding form already had its own).
export function noteRecall() {
    recallSpark = 1.0;
    recallNode = Math.floor(Math.random() * NODE_COUNT);
    const th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    recallDir = [Math.sin(ph) * Math.cos(th), Math.cos(ph), Math.sin(ph) * Math.sin(th)];
    if (FORMS[formIndex] === 'embedding' && embExcite.length) embExcite[embTo] = 1.0;
    return recallNode;
}

// How the turn ended shapes the release: 'pass' crystallises and holds,
// 'refute' shudders and re-weaves, anything else exhales.
export function noteVerdict(kind) {
    verdict = kind === 'pass' || kind === 'refute' ? kind : 'stop';
    verdictEnv = 1.0;
    if (verdict === 'refute') flinch = Math.min(1.0, flinch + 0.5);
    return verdict;
}

// A dream / self-play turn holds the lock: a second, slower breath at
// the edge — "busy with itself", visible without a spinner.
export function setBackgroundBusy(busy) {
    targetBackgroundBusy = busy ? 1.0 : 0.0;
}

// Mood → a slow baseline shift of the COLD pole only (never competes
// with the hot pole). Pure mapping; executed under node.
export function moodHueFor(label) {
    switch (String(label || '').toLowerCase()) {
        case 'satisfied': return -0.035;   // deeper, settled blue
        case 'idle': return -0.06;         // toward the plum bridge — resting
        case 'curious': return 0.03;       // indigo, leaning violet
        case 'stuck': return 0.05;         // violet
        case 'overloaded': return 0.07;    // violet, nearly warm
        default: return 0.0;
    }
}
export function setMoodHue(label) {
    targetMoodHue = moodHueFor(label);
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
    // Cap DPR aggressively on mobile. iOS Safari reports dpr=3 on
    // Retina phones; rendering + bloom at 3x backing-store pixels is
    // the single biggest cost we can shed with no visual penalty.
    const dprCap = IS_MOBILE ? 1.5 : 2.0;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, dprCap));

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

    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('aLightPass', new THREE.BufferAttribute(lineUvs, 1));
    lineGeometry.setAttribute('aLineHue', new THREE.BufferAttribute(lineHues, 1));

    lineMaterial = new THREE.ShaderMaterial({
        vertexShader: lineVertexShader,
        fragmentShader: lineFragmentShader,
        uniforms: {
            uTime: { value: 0 },
            uOrganic: { value: PREFERS_REDUCED_MOTION ? 0.25 : 1.0 },
            uWaveAmp: { value: 1.0 },
            uCenterDim: { value: 0.0 },
            uCenterXY: { value: 0.0 },
            uFormDim: { value: 1.0 },
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uDive: { value: 0.0 },
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

    // Interior motes (see MOTE_COUNT comment). Distributed with an
    // inward bias so the region the camera dives through is the
    // densest. Added to the scene so they inherit its spin/scale.
    const moteGeom = new THREE.BufferGeometry();
    const motePos = new Float32Array(MOTE_COUNT * 3);
    const moteSeeds = new Float32Array(MOTE_COUNT);
    for (let i = 0; i < MOTE_COUNT; i++) {
        const u = Math.random(), v = Math.random();
        const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
        const r = 2.4 * Math.pow(Math.random(), 0.55); // inward bias
        motePos[i * 3] = r * Math.sin(ph) * Math.cos(th);
        motePos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
        motePos[i * 3 + 2] = r * Math.cos(ph);
        moteSeeds[i] = Math.random();
    }
    moteGeom.setAttribute('position', new THREE.BufferAttribute(motePos, 3));
    moteGeom.setAttribute('aSeed', new THREE.BufferAttribute(moteSeeds, 1));
    motesMaterial = new THREE.ShaderMaterial({
        vertexShader: moteVertexShader,
        fragmentShader: moteFragmentShader,
        uniforms: {
            uTime: { value: 0 },
            uOrganic: { value: PREFERS_REDUCED_MOTION ? 0.25 : 1.0 },
            uWaveAmp: { value: 1.0 },
            uDive: { value: 0 },
            uDpr: { value: Math.min(window.devicePixelRatio, dprCap) },
            ...paletteUniforms(),
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
    });
    motesMesh = new THREE.Points(moteGeom, motesMaterial);
    motesMesh.visible = false;   // skipped entirely at rest
    scene.add(motesMesh);

    // Scale scene down by 10%
    scene.scale.set(0.9, 0.9, 0.9);

    window.addEventListener('resize', handleResize);
    _installParallaxListeners();

    animate();
}

function handleResize() {
    if (!camera || !renderer || !composer) return;
    const container = document.getElementById('sphere-container');
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
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
    const w = Math.max(0, Math.min(1, weight || 0.15));
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
    flinch = Math.min(1.0, flinch + 0.8);
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
        vortexTravel,
        tunnelFlow,
        phase, gait: { ...gait }, toolPulse, recallSpark, verdict, verdictEnv,
        backgroundBusy, moodHue, gazeY, errorKind, errorKindEnv,
        dialect: dialectFor(FORMS[formIndex]), gaitFlow, gaitThicken, gaitFlash, gaitAlign,
    };
}

// The immersion dive's ONLY driver — set true when the user's request
// goes in flight, false when the reply completes/aborts. Ambient
// activity must never call this (that's what setWorkingState is for).
export function setUserTurn(isActive) {
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
        return false;
    }
    if (renderer && !animationFrameId) animate();
    return !!animationFrameId;
}
if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('visibilitychange', () => setAnimationPaused(!!document.hidden));
}

function animate() {
    if (_animationPaused) { animationFrameId = null; return; }
    animationFrameId = requestAnimationFrame(animate);

    const isWaking = targetWorkingState > workingState + 0.01;
    const transitionSpeed = isWaking ? 0.05 : 0.02;

    workingState += (targetWorkingState - workingState) * transitionSpeed;
    userTurnState += (targetUserTurnState - userTurnState)
        * (targetUserTurnState > userTurnState ? 0.05 : 0.02);
    errorState   += (targetErrorState   - errorState)   * 0.05;

    // Activity envelope: the raw target decays on its own (half-life
    // ~2.5s at 60fps) and the rendered value follows it with a soft
    // attack and a slower release — so a burst of log lines swells the
    // graph over ~a second and lets it settle over ~8, instead of
    // strobing per event.
    activityTarget *= 0.9955;
    if (activityTarget < 0.005) activityTarget = 0;
    const _rising = activityTarget > activity;
    activity += (activityTarget - activity) * (_rising ? 0.035 : 0.008);

    // Accent tint follows the same philosophy: drift up with colored
    // activity, drain slowly back to neutral.
    accentTarget *= 0.995;
    accentStrength += (accentTarget - accentStrength) * 0.02;

    // The shader "energy" (formerly the per-event shockwave) is now the
    // smoothed envelope — faint traveling charge on the lines and a mild
    // node glow when the agent is busy, perfectly still when quiet.
    pulseT = Math.min(1.0, activity * 0.55);

    // Audio-level natural decay: even if setAudioLevel stops being
    // called (TTS queue drained), the residual level dies out fast.
    audioLevel *= 0.92;

    // ── Signal-layer envelopes (2026-09-11) ─────────────────────────
    for (const k of PHASES) gait[k] += ((phase === k ? 1 : 0) - gait[k]) * 0.04;
    toolPulse *= 0.93;
    if (toolPulse < 0.003) toolPulse = 0;
    recallSpark *= 0.972;                       // ~1.5s comet
    if (recallSpark < 0.01) { recallSpark = 0; recallNode = -1; }
    verdictEnv *= verdict === 'pass' ? TUNE.passHold : 0.975;   // pass holds ~2s, others ~1s
    if (verdictEnv < 0.01) { verdictEnv = 0; verdict = null; }
    backgroundBusy += (targetBackgroundBusy - backgroundBusy) * 0.01;
    moodHue += (targetMoodHue - moodHue) * 0.0008;     // ~1 min to settle
    gazeX += (targetGazeX - gazeX) * 0.03;
    gazeY += (targetGazeY - gazeY) * 0.03;
    errorKindEnv *= 0.985;
    if (errorKindEnv < 0.01) { errorKindEnv = 0; errorKind = null; }
    sweepAngle += (1 / 60) * TUNE.sweepSpeed * gait.search;
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
        TUNE.stillVerify * gait.verify, TUNE.stillRead * gait.read));
    const motionMul = 1.0 - still;

    // Immersion follows the USER-TURN state with a slower, asymmetric
    // ease: ~5s to fully swallow (only sustained work gets there), ~10s
    // to drift back out after completion. Short requests barely lean in.
    // Deliberately NOT workingState: ambient log activity re-arms that
    // almost continuously on a busy agent, which kept the camera inside
    // the cloud when the agent wasn't working for the user at all.
    const immersionTarget = userTurnState * IMMERSION_CAP;
    immersion += (immersionTarget - immersion)
        * (immersionTarget > immersion ? 0.012 : 0.006);
    // Smoothstepped path so both ends of the dive are gentle.
    const dive = immersion * immersion * (3.0 - 2.0 * immersion);

    // Form name, bound once per frame — every per-form branch below
    // keys on the NAME (index comparisons broke silently whenever the
    // FORMS array grew; 2026-07-29, when the four AI forms joined).
    const FORM = FORMS[formIndex];

    // Vortex engagement + flow (see the vortexTravel declaration).
    // Engagement eases up while a user turn runs and eases back down on
    // completion — since the camera never moves and the cone is
    // self-similar, that ease IS the whole transition: busy morphs into
    // idle with no reset of any kind.
    if (FORM === 'vortex') {
        if (userTurnState > 0.5) {
            vortexTravel += (IMMERSION_CAP - vortexTravel) * 0.010;
        } else if (vortexTravel > 0.001) {
            vortexTravel *= 0.985;                   // calm back down (~5s)
        } else {
            vortexTravel = 0.0;
        }
        const travelNorm = Math.min(vortexTravel / Math.max(IMMERSION_CAP, 0.001), 1.0);
        // THE SWALLOW — the main animation. Slow accretion at idle, a
        // touch quicker under ambient work, a strong endless in-fall
        // while a user request executes (full emergence cycle ≈ 5s).
        // The rate is TURBULENT, not constant: it surges and slackens on
        // slow beats, and while feeding it GULPS in rhythm with the
        // pulse — a constant conveyor read as monotonic.
        const flowTurb = 1.0 + CALM * (0.35 * Math.sin(time * 0.23)
            * Math.sin(time * 0.081 + 1.7)
            + 0.45 * travelNorm * _pulseShape(_fract(pulsePhase)));
        tunnelFlow += (1 / 60) * (0.008
            + 0.014 * Math.max(workingState, activity)
            + 0.19 * travelNorm) * flowTurb * CALM
            * (1.0 + TUNE.flowWrite * gaitFlow) * (1.0 - 0.5 * gaitThicken);
        // Swirl (final direction 2026-07-28): idle keeps its slight
        // rotation (0.125); busy ACCELERATES it moderately (+60%)
        // alongside the flow surge — falling faster AND spinning
        // faster, but nowhere near the early "way too fast" rates
        // (that culprit was wind-coupling, long since fixed).
        vortexSpin += (1 / 60) * (0.125
            + 0.025 * Math.max(workingState, activity, travelNorm)) * CALM;
    } else {
        vortexTravel = 0.0;
    }

    // Accumulate time for lines at steady pace — except inside the
    // cloud, where the data pulses rush a little faster: the interior
    // should feel BUSIER than the outside view, not emptier.
    time += 0.005 * (1.0 + dive * 0.6) * motionMul;

    // Idle breathing: ±1% scene-scale sine at ~0.1Hz. Below the
    // motion-detection threshold on both desktop and mobile; keeps the
    // sphere from looking frozen when nothing else is animating.
    // Immersion scales the whole scene UP around the viewer — combined
    // with the camera dolly below, this is what reads as "the grid
    // swallowing the camera" rather than the camera flying somewhere.
    // Two incommensurate frequencies: the breath never repeats exactly —
    // a metronomic single sine is precisely what read as "not alive".
    const breathe = 1.0 + 0.011 * Math.sin(time * 0.55)
        + 0.006 * Math.sin(time * 0.173 + 1.7);
    // Scale boost trimmed 0.7 → 0.55: the swell is what sells the
    // swallow, but it also DILUTES local node density exactly when the
    // camera is closest — the interior looked empty at 0.7 even with
    // the denser web + motes compensating.
    const baseScale = 0.9 * breathe * (1.0 + dive * 0.55);
    // The vortex's journey is the CAMERA's — swelling the scene under it
    // would double-transform the funnel, so the swell is neutralized.
    // The cube gets a GENTLED swell (v2): the point of its dive is
    // watching the mutation spread across the monolith, not entering a
    // cloud — full swell dissolved the silhouette into dots.
    // Verdict release: a stop EXHALES (one soft swell that lets go); the
    // background breath is a second, slower rhythm under everything.
    const exhale = verdict === 'stop' ? TUNE.exhale * Math.sin(Math.PI * (1.0 - verdictEnv)) * verdictEnv : 0;
    const bgBreath = TUNE.bgBreath * backgroundBusy * Math.sin(time * 0.31 + 0.9);
    const sceneScale = (FORM === 'vortex' ? 0.9 * breathe
        : FORM === 'cube' ? 0.9 * breathe * (1.0 + dive * 0.18)
        : baseScale) * (1.0 + exhale + bgBreath);
    scene.scale.set(sceneScale, sceneScale, sceneScale);

    // Slow continuous orbit (faster while working) plus a gentle tilt
    // gives the network real depth and life at idle. Skipped entirely
    // under reduced-motion.
    if (!PREFERS_REDUCED_MOTION) {
        if (FORM === 'vortex') {
            // Vortex: the funnel must keep facing the viewer, so no big
            // heading wander — instead the whole scene ROLLS around the
            // view axis (global swirl) with only a whisper of tilt.
            scene.rotation.x = 0.06 * Math.sin(time * 0.050);
            scene.rotation.y = 0.05 * Math.sin(time * 0.031 + 1.0);
            scene.rotation.z = vortexSpin * 0.22;
            scene.position.x = 0.05 * Math.sin(time * 0.031);
            scene.position.y = _bob * 0.5;
        } else if (FORM === 'cube') {
            // Monolith: near-still heading — its OWN slow tumble is the
            // motion. The full heading wander compounded with the
            // focus-translation release into the "erratic zoom-out"
            // (v2), and it also kept the silhouette from ever settling
            // into a readable cube.
            scene.rotation.y = 0.09 * Math.sin(time * 0.029);
            scene.rotation.x = 0.05 * Math.sin(time * 0.021 + 0.7);
            scene.rotation.z = 0.02 * Math.sin(time * 0.017 + 2.0);
            scene.position.x = 0.04 * Math.sin(time * 0.023);
            scene.position.y = _bob * 0.4;
        } else {
            // Heading wander: the creature holds a loose heading and lets
            // it drift — sums of incommensurate slow sines, so it turns
            // its attention rather than rotating on a turntable. The body
            // also drifts a little, like something suspended in a current.
            scene.rotation.y = 0.38 * Math.sin(time * 0.050)
                + 0.22 * Math.sin(time * 0.023 + 1.3);
            scene.rotation.x = 0.10 * Math.sin(time * 0.037 + 0.7)
                + 0.05 * Math.sin(time * 0.011);
            scene.rotation.z = 0.05 * Math.sin(time * 0.029 + 2.1);
            scene.position.x = 0.10 * Math.sin(time * 0.031);
            scene.position.y = _bob;
        }
    }

    // Parallax: ease camera toward target offset. Z is owned by the
    // immersion dive (rest 5.0 → inside the cloud 1.3); x/y parallax
    // rides on top at any depth.
    camera.position.x += (parallaxTargetX - camera.position.x) * 0.04;
    camera.position.y += (parallaxTargetY - camera.position.y) * 0.04;
    if (FORM === 'vortex') {
        // Vortex: THE CAMERA NEVER MOVES — the self-similar swallow does
        // all the traveling. It studies the hole itself.
        camera.position.z = CAMERA_REST_Z;
        camera.lookAt(camera.position.x * 0.35, camera.position.y * 0.35,
            VORTEX_APEX_Z);
    } else {
        // Cube (v2): PARTIAL dive only — close enough to watch the
        // mutation spread across the monolith, never inside the cloud
        // ("zoom is too much... just shows random dots").
        const _diveZ = FORM === 'cube' ? CUBE_DIVE_Z : CAMERA_DIVE_Z;
        camera.position.z = CAMERA_REST_Z - (CAMERA_REST_Z - _diveZ) * dive;
        // Look-target blend: at rest the camera studies the origin; deep in
        // the cloud it must look FORWARD through it instead — near the
        // origin, lookAt(0,0,0) turns tiny parallax offsets into wild
        // rotations (the lookAt singularity).
        camera.lookAt(gazeX, gazeY * (1.0 - dive), (FORM === 'cube' ? -1.0 : -3.5) * dive);
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
        currentShapeSpeed += Math.sign(speedDiff) * Math.min(Math.abs(speedDiff), 0.011);
    }

    // ── Medusan propulsion ──────────────────────────────────────────
    // The pulse clock: ~4.5s per cycle at idle, tightening toward ~1.8s
    // when the agent is busy (currentShapeSpeed lerps between
    // SPEEDS.idle and SPEEDS.busy above). An error flinch briefly
    // agitates both rate and amplitude — a recoil, not a flash.
    pulsePhase += (1 / 60) * (0.20 + 0.105 * currentShapeSpeed)
        * (1.0 + flinch * 1.2)
        * (PREFERS_REDUCED_MOTION ? 0.5 : 1.0) * motionMul;
    flinch *= 0.97;

    const pulseAmp = (0.16 + 0.10 * drive + 0.22 * flinch)
        * (PREFERS_REDUCED_MOTION ? 0.45 : 1.0)
        * (1.0 + 0.25 * audioLevel);
    // Contraction as felt at the bell margin (drives tentacle tug + the
    // swim bob) — the wave reaches the margin 0.16 cycles after the apex.
    const cMargin = _pulseShape(_fract(pulsePhase - 0.16));

    // Slow vertical mass-drift (NOT a pulse-locked swim bob — that was
    // the jellyfish tell). The body hovers like something suspended.
    _bobTarget = 0.10 * Math.sin(time * 0.043 + 0.5);
    _bob += (_bobTarget - _bob) * 0.02;

    if (FORM === 'vortex') {
        // ── VORTEX: the self-similar swallow. Each wall node's depth
        //    FLOWS (dEff advances with tunnelFlow); its distance from
        //    the apex is L0·e^(−k·d) so the wrap is an invisible fractal
        //    renormalization; its hue re-heats blue→red along the fall
        //    (seeds recomputed per frame, uploaded below). A drifting
        //    harmonic field morphs the cross-section through organic,
        //    never-repeating shapes — the "procedurally generated
        //    content" being consumed — swelling while a turn runs.
        const engage = Math.min(vortexTravel / Math.max(IMMERSION_CAP, 0.001), 1.0);
        // Pulsing OFF at idle (final direction 2026-07-28): the resting
        // vortex only drifts and turns. A SLIGHT pulse (35%) is allowed
        // while busy — it rides the feeding gulps and keeps the surge
        // from feeling like a conveyor.
        const vPulseScale = 0.35 * Math.max(engage, drive);
        const a2 = (0.10 + 0.20 * engage) * Math.sin(time * 0.110);
        const a3 = (0.08 + 0.16 * engage) * Math.sin(time * 0.073 + 2.1);
        const a5 = (0.05 + 0.11 * engage) * Math.sin(time * 0.047 + 0.7);
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            if (bp.kind === 2) {
                // The shadow's embers: dim, trembling, hungrier while a
                // request runs — the center is never static.
                const c = _pulseShape(_fract(pulsePhase + 0.05));
                const g = 1.0 + (0.14 * vPulseScale + 0.28 * engage) * c;
                const tremor = 0.02 + 0.04 * engage;
                currentPositions[i].set(
                    bp.hx * g + tremor * CALM * Math.sin(time * 1.1 + bp.jit),
                    bp.hy * g + tremor * CALM * Math.cos(time * 0.9 + bp.jit * 2.0),
                    VORTEX_APEX_Z + bp.hz * g
                        + 0.04 * CALM * Math.sin(time * 0.7 + bp.jit));
            } else if (bp.kind === 3) {
                // The accretion ring: the iconic slow orbit, breathing
                // with the pulse, swelling slightly while feeding.
                const th = bp.th0 + vortexSpin * 1.6;
                const rr = bp.rRing
                    * (1.0 + 0.05 * CALM * Math.sin(time * 0.5 + bp.jit)
                        + 0.10 * engage * _pulseShape(_fract(pulsePhase)));
                currentPositions[i].set(
                    rr * Math.cos(th),
                    rr * Math.sin(th) * 0.92,
                    VORTEX_APEX_Z + bp.zJit
                        + 0.03 * CALM * Math.sin(time * 0.8 + bp.jit));
            } else {
                // Stream matter: EMERGES from the hole's glow (L small,
                // screen center) and blooms outward past the viewer —
                // expansion optical flow = falling IN. The wrap at LMAX
                // happens far off-screen; the respawn at LMIN hides
                // inside the ring's glow. Hue cools as matter recedes
                // from the hole (hottest at the center).
                const dOut = _fract(bp.d0 + tunnelFlow * bp.flowScale);
                const L = VORTEX_LMIN * Math.exp(VORTEX_KOUT * dOut);
                // Spiral wind cut 4.2 → 1.2 → 0.35 across three "too
                // fast" reports: the wind is traversed by the FLOW, so
                // any busy surge multiplies into apparent rotation —
                // keep it near-nothing. The differential is flattened
                // too (0.9→0.5 center→rim, was 1.5): the center is
                // where the eye rests, so it must not spin fastest by
                // much. Busy total ≈ 13°/s near the hole vs ~6°/s idle.
                const theta = bp.theta0 + (1.0 - dOut) * 0.35
                    + vortexSpin * (0.9 - 0.4 * dOut);
                // Organic cross-section morph — the procedural content.
                const shape = 1.0
                    + a2 * Math.sin(2.0 * theta + dOut * 5.0)
                    + a3 * Math.sin(3.0 * theta - dOut * 7.0)
                    + a5 * Math.sin(5.0 * theta + dOut * 11.0);
                // Turbulence: grains wobble off their trajectories, more
                // while feeding, scale-relative so spawn stays calm.
                const turb = (0.05 + 0.30 * engage)
                    * Math.sin(time * (1.1 + bp.flowScale) + bp.jit * 3.0 + dOut * 7.0);
                const pinch = _pulseShape(_fract(pulsePhase - dOut * 0.25));
                const rad = L * bp.sinA * bp.rJit * shape
                    * (1.0 - pulseAmp * 0.35 * vPulseScale * pinch)
                    + turb * L * 0.10 * CALM;
                currentPositions[i].set(
                    rad * Math.cos(theta),
                    rad * Math.sin(theta),
                    VORTEX_APEX_Z + L * bp.cosA
                        + turb * 0.12 * CALM
                        + 0.05 * CALM * Math.sin(time * 0.6 + bp.jit));
                nodeSeeds[i] = Math.max(0.02, 0.60 - dOut * 0.56) + bp.seedJit;
            }
        }
        instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    } else if (FORM === 'lattice') {
        // ── LATTICE: the weight tensor tumbles slowly while diagonal
        //    activation waves sweep it (riding the shared pulse clock),
        //    heating the sites they cross; the hot attention kernel
        //    drifts a Lissajous through the volume, bending nearby grid
        //    toward itself; charge runners ride the axis lines.
        const la1 = time * 0.055 * CALM, la2 = time * 0.034 * CALM;
        const lc1 = Math.cos(la1), ls1 = Math.sin(la1);
        const lc2 = Math.cos(la2), ls2 = Math.sin(la2);
        const kx = 0.85 * Math.sin(time * 0.111);
        const ky = 0.80 * Math.sin(time * 0.087 + 1.7);
        const kz = 0.85 * Math.cos(time * 0.067 + 0.6);
        const waveGain = 0.16 + 0.22 * drive + 0.15 * flinch;
        const alignMul = 1.0 - TUNE.alignGain * gaitAlign;   // verify: the grid snaps true
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            let x, y, z;
            if (bp.kind === 0) {
                const w = _pulseShape(_fract(pulsePhase - bp.wp * 0.42));
                const dxk = bp.gx - kx, dyk = bp.gy - ky, dzk = bp.gz - kz;
                const kAtt = Math.exp(-(dxk * dxk + dyk * dyk + dzk * dzk) / 0.55);
                const lean = 0.10 * kAtt * alignMul;            // pulled toward attention
                const wd = 0.055 * pulseAmp * w * CALM * alignMul;   // along (1,1,1)/√3
                x = bp.gx - dxk * lean + wd * 0.577
                    + 0.018 * CALM * alignMul * Math.sin(time * 1.1 + bp.jit);
                y = bp.gy - dyk * lean + wd * 0.577
                    + 0.018 * CALM * alignMul * Math.sin(time * 0.8 + bp.jit * 2.0);
                z = bp.gz - dzk * lean + wd * 0.577;
                nodeSeeds[i] = Math.min(0.60,
                    bp.seed0 + w * waveGain + 0.28 * kAtt * (1.0 + gaitFlash * TUNE.flashGain));
            } else if (bp.kind === 1) {
                const t = _fract(bp.d0 + time * bp.speed * (1.0 + 1.5 * drive + 2.0 * gaitFlow));
                const p = -bp.span + t * 2 * bp.span;
                // Taper at the faces so the wrap hides off-grid.
                bp.sz = 0.85 * Math.min(1, 8 * Math.min(t, 1 - t) + 0.10);
                x = bp.ox + bp.dx * p;
                y = bp.oy + bp.dy * p;
                z = bp.oz + bp.dz * p;
            } else {
                const c = _pulseShape(_fract(pulsePhase + 0.05));
                const g = 1.0 + 0.15 * c + 0.05 * Math.sin(time * 0.8 + bp.jit);
                x = kx + bp.hx * g;
                y = ky + bp.hy * g;
                z = kz + bp.hz * g;
            }
            // Stately two-axis tumble — crystalline, no medusan sway.
            const x1 = x * lc1 + z * ls1;
            const z1 = -x * ls1 + z * lc1;
            currentPositions[i].set(
                x1,
                y * lc2 - z1 * ls2,
                y * ls2 + z1 * lc2);
        }
        instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    } else if (FORM === 'embedding') {
        // ── EMBEDDING: concept clusters drift on slow orbits with
        //    gentle internal swirl; the query comet flies an outward
        //    bezier arc cluster→cluster (faster while the agent is
        //    busy). Arrival IGNITES the recalled cluster — it flares
        //    hot and tightens, then cools back over ~2s.
        const cc = [];
        for (let c = 0; c < _embCenters.length; c++) {
            const e = _embCenters[c];
            const a = e.oa + time * e.ow;
            cc.push([
                e.bx + e.or * Math.sin(a),
                e.by + e.or * 0.6 * Math.sin(a * 1.31 + 1.2),
                e.bz + e.or * Math.cos(a * 0.83),
            ]);
            embExcite[c] *= 0.985;
        }
        embT += (1 / 60) * (0.24 + 0.60 * drive + 0.4 * flinch + 0.8 * gaitFlow) * CALM;
        if (embT >= 1.0) {
            embExcite[embTo] = 1.0;
            embFrom = embTo;
            let next = Math.floor(Math.random() * _embCenters.length);
            if (next === embFrom) next = (next + 1) % _embCenters.length;
            embTo = next;
            embT = 0.0;
        }
        const A = cc[embFrom] || [0, 0, 0], B = cc[embTo] || [0, 0, 0];
        // Control point pushed outward: the flight arcs through the
        // void between concepts instead of cutting through the origin.
        const CXx = (A[0] + B[0]) * 0.5 * 1.75;
        const CXy = (A[1] + B[1]) * 0.5 * 1.75;
        const CXz = (A[2] + B[2]) * 0.5 * 1.75;
        // Dive focus (2026-07-29 operator report, descent's sibling
        // fix): the immersion dive zooms toward the scene ORIGIN, and
        // embedding's origin is DELIBERATELY empty void between the
        // clusters. While diving, translate the whole space so the
        // QUERY HEAD sits at the origin — a user turn rides the recall
        // flight itself. Weighted by `dive`, so at rest nothing moves;
        // the head's path is continuous (bezier, and each new flight
        // starts where the last one landed), so the follow never jumps.
        const _ehE = embT * embT * (3 - 2 * embT);
        const _ehU = 1 - _ehE;
        const _embFx = dive * (_ehU * _ehU * A[0] + 2 * _ehU * _ehE * CXx + _ehE * _ehE * B[0]);
        const _embFy = dive * (_ehU * _ehU * A[1] + 2 * _ehU * _ehE * CXy + _ehE * _ehE * B[1]);
        const _embFz = dive * (_ehU * _ehU * A[2] + 2 * _ehU * _ehE * CXz + _ehE * _ehE * B[2]);
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            let x, y, z;
            if (bp.kind === 0) {
                const ex = embExcite[bp.ci] || 0;
                const C = cc[bp.ci] || [0, 0, 0];
                const sp = time * _embCenters[bp.ci].spin * CALM;
                const cs = Math.cos(sp), sn = Math.sin(sp);
                const ox = bp.dx * cs + bp.dz * sn;
                const oz = -bp.dx * sn + bp.dz * cs;
                const r = bp.r0 * (1.0 - 0.30 * ex)
                    * (1.0 + 0.06 * pulseAmp
                        * _pulseShape(_fract(pulsePhase - bp.ci * 0.13)));
                x = C[0] + ox * r + 0.015 * CALM * Math.sin(time * 1.0 + bp.jit);
                y = C[1] + bp.dy * r + 0.015 * CALM * Math.sin(time * 0.8 + bp.jit * 2.0);
                z = C[2] + oz * r;
                nodeSeeds[i] = Math.min(0.58, bp.seed0 + 0.40 * ex);
            } else {
                // Query comet: head leads, tail nodes trail along the
                // same flight path with a slight lag.
                const tl = Math.max(0, Math.min(1, embT - bp.s * 0.10));
                const e = tl * tl * (3 - 2 * tl);
                const u = 1 - e;
                x = u * u * A[0] + 2 * u * e * CXx + e * e * B[0]
                    + bp.rr * Math.sin(time * 2.1 + bp.jit * 3.0);
                y = u * u * A[1] + 2 * u * e * CXy + e * e * B[1]
                    + bp.rr * Math.cos(time * 1.7 + bp.jit * 2.0);
                z = u * u * A[2] + 2 * u * e * CXz + e * e * B[2]
                    + bp.rr * Math.sin(time * 1.9 + bp.jit * 5.0);
                bp.sz = 1.0 - bp.s * 0.5;
            }
            currentPositions[i].set(x - _embFx, y - _embFy, z - _embFz);
        }
        instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    } else if (FORM === 'descent') {
        // ── DESCENT: true gradient descent on the evolving loss
        //    surface. Numeric ∇h each frame; damping keeps the roll
        //    readable; soft walls; a learning-rate kick fires when the
        //    bead settles (basin found) or the agent flinches (bad
        //    gradient step). The sheet dips under the bead and heats
        //    where it passes; ridges stay cold, valleys warm.
        const ddt = 1 / 60;
        const lr = 1.6 * (0.55 + 0.45 * drive + 0.6 * gaitFlow);
        const eps = 0.06;
        const gX = (_lossH(beadX + eps, beadZ, time)
            - _lossH(beadX - eps, beadZ, time)) / (2 * eps);
        const gZ = (_lossH(beadX, beadZ + eps, time)
            - _lossH(beadX, beadZ - eps, time)) / (2 * eps);
        beadVX = (beadVX - gX * lr * ddt) * 0.965;
        beadVZ = (beadVZ - gZ * lr * ddt) * 0.965;
        beadX += beadVX * ddt * 2.0 * CALM;
        beadZ += beadVZ * ddt * 2.0 * CALM;
        const XL = DESC_SX / 2 - 0.25, ZL = DESC_SZ / 2 - 0.25;
        if (beadX > XL) { beadX = XL; beadVX = -Math.abs(beadVX) * 0.5; }
        if (beadX < -XL) { beadX = -XL; beadVX = Math.abs(beadVX) * 0.5; }
        if (beadZ > ZL) { beadZ = ZL; beadVZ = -Math.abs(beadVZ) * 0.5; }
        if (beadZ < -ZL) { beadZ = -ZL; beadVZ = Math.abs(beadVZ) * 0.5; }
        const spd = Math.hypot(beadVX, beadVZ);
        beadStill = spd < 0.10 ? beadStill + ddt : 0;
        if (beadStill > 2.2 || (flinch > 0.55 && beadStill > 0.4)) {
            const ka = Math.random() * Math.PI * 2;
            const kk = 0.9 + 0.7 * drive;
            beadVX += Math.cos(ka) * kk;
            beadVZ += Math.sin(ka) * kk;
            beadStill = 0;
        }
        if ((_descTick++ % 3) === 0) {
            beadTrail.unshift([beadX, beadZ]);
            if (beadTrail.length > 40) beadTrail.pop();
        }
        // Tilt the whole landscape toward the camera.
        const TC = Math.cos(-0.52), TS = Math.sin(-0.52);
        // Dive focus (2026-07-29 operator report: "descent usually
        // zooms into an uninteresting location when busy"): the
        // immersion dive zooms toward the scene ORIGIN — a generic
        // patch of terrain — while the story (the hot bead) is off
        // wherever it rolled. While diving, translate the whole sheet
        // so the BEAD sits at the origin, slightly below the view axis
        // (+0.30 lift so the camera hovers over the surface instead of
        // clipping into it): a user turn rides the optimizer hunting
        // the minimum. Weighted by `dive` — at rest nothing moves; the
        // bead's position is continuous (velocity physics; kicks jolt
        // velocity, never position), so the follow never jumps.
        const _bwYpre = _lossH(beadX, beadZ, time) + 0.10;
        const _descFx = dive * beadX;
        const _descFy = dive * (_bwYpre * TC - beadZ * TS - 0.15 + 0.30);
        const _descFz = dive * (_bwYpre * TS + beadZ * TC);
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            let x, y, z;
            if (bp.kind === 0) {
                const h = _lossH(bp.gx, bp.gz, time);
                const dxb = bp.gx - beadX, dzb = bp.gz - beadZ;
                const heat = Math.exp(-(dxb * dxb + dzb * dzb) / 0.30);
                x = bp.gx;
                y = h - 0.10 * heat
                    + 0.02 * CALM * Math.sin(time * 0.9 + bp.jit);
                z = bp.gz;
                // Height → thermal: ridges cold, valleys warm, the
                // bead's neighborhood glowing.
                const hn = Math.max(0, Math.min(1, 0.5 + h / (2 * DESC_H * 0.76)));
                nodeSeeds[i] = 0.04 + (1 - hn) * 0.30 + 0.24 * heat;
            } else {
                const ti = Math.min(
                    Math.floor(bp.s * Math.max(beadTrail.length - 1, 0)),
                    Math.max(beadTrail.length - 1, 0));
                const P = beadTrail.length ? beadTrail[ti] : [beadX, beadZ];
                x = P[0] + bp.rr * Math.sin(time * 2.3 + bp.jit * 3.0);
                y = _lossH(P[0], P[1], time) + 0.10
                    + bp.rr * Math.cos(time * 1.9 + bp.jit * 2.0);
                z = P[1] + bp.rr * Math.sin(time * 2.1 + bp.jit * 5.0);
                bp.sz = 1.0 - bp.s * 0.55;
                nodeSeeds[i] = 0.60 - bp.s * 0.22;
            }
            currentPositions[i].set(
                x - _descFx,
                y * TC - z * TS - 0.15 - _descFy,
                y * TS + z * TC - _descFz);
        }
        instancedMesh.geometry.attributes.aSeed.needsUpdate = true;
    } else if (FORM === 'cube') {
        // ── CUBE: the infinite monolith. The grid barely moves — the
        //    LIFE is in the resident complexities, each an alien sine
        //    field quietly warping its neighborhood. A user turn wakes
        //    one (the mutation): it grows aggressively-but-not-fast,
        //    spreads through each node's own irregular gate, churns
        //    faster as it strengthens, runs crimson at its heart — and
        //    the dive translation carries the camera into it. On
        //    completion it tames over ~6s and the cube re-knits.
        const _turnOn = userTurnState > 0.5;
        if (_turnOn && !_cubePrevTurn && cubeS < 0.4) {
            // New turn while calm: a (possibly different) complexity
            // wakes. Mid-decay re-arms keep the SAME one — the heat
            // must never teleport.
            cubeActive = Math.floor(Math.random() * _cubeCx.length);
        }
        _cubePrevTurn = _turnOn;
        if (_turnOn) {
            cubeS += (1.0 - cubeS) * 0.014;   // ~5s to full presence
        } else if (cubeS > 0.001) {
            // Taming tail slowed 0.988 → 0.9935 (v2): the spring-back
            // must pace the ~10s dive-out — nodes releasing faster than
            // the camera retreats was the "erratic zoom-out".
            cubeS *= 0.9935;
        } else {
            cubeS = 0.0;
        }
        // Stately monolith tumble — slower than lattice.
        const ka1 = time * 0.030 * CALM, ka2 = time * 0.019 * CALM;
        const kc1 = Math.cos(ka1), ks1 = Math.sin(ka1);
        const kc2 = Math.cos(ka2), ks2 = Math.sin(ka2);
        // Dive focus: the ACTIVE complexity's post-tumble position —
        // the camera rides into the mutation, not a generic corner
        // (the descent/embedding lesson applied from birth).
        const _A = _cubeCx[cubeActive] || { ax: 0, ay: 0, az: 0 };
        const _awx = _A.ax * kc1 + _A.az * ks1;
        const _awz1 = -_A.ax * ks1 + _A.az * kc1;
        const _awy = _A.ay * kc2 - _awz1 * ks2;
        const _awz = _A.ay * ks2 + _awz1 * kc2;
        const kfx = dive * _awx, kfy = dive * _awy, kfz = dive * _awz;
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            let x, y, z;
            if (bp.kind === 0) {
                x = bp.gx; y = bp.gy; z = bp.gz;
                let heat = 0;
                for (let k = 0; k < _cubeCx.length; k++) {
                    const c = _cubeCx[k];
                    const Sk = k === cubeActive ? cubeS : 0;
                    // Idle life (v2): each complexity is ALREADY a slow
                    // red organism — swelling and shrinking on its own
                    // rhythm — and the active one adds the mutation on
                    // top. S_total drives radius, motion and heat alike.
                    const St = 0.22 + 0.12 * Math.sin(time * 0.11 + c.ph) + Sk;
                    const edge = c.r0 * (0.8 + 1.5 * St) * bp.gate;
                    const dx = bp.gx - c.ax, dy = bp.gy - c.ay, dz = bp.gz - c.az;
                    const d = Math.hypot(dx, dy, dz);
                    if (d > edge + 0.4) continue;
                    const w = 1 - d / (edge + 0.4);
                    const wS = w * w;
                    const amp = (0.06 + 0.30 * St) * wS * CALM
                        * (1 + 0.5 * flinch) * (1.0 - TUNE.alignGain * gaitAlign);
                    // The mutation churns FASTER as it strengthens (and
                    // while the reply streams, in the 'flow' dialect).
                    const tk = time * (0.7 + 1.4 * Sk + 1.0 * gaitFlow);
                    // SPATIALLY COHERENT field (v2, "random dots" fix):
                    // phases keyed to grid POSITION at low frequency, so
                    // neighbors move together — waves rippling through
                    // flesh — with only a whisper of per-node texture.
                    x += amp * (Math.sin(tk * 0.9 + bp.gy * 1.9 + bp.gz * 1.3 + c.ph)
                        + 0.15 * Math.sin(tk * 1.7 + bp.jit * 3.0));
                    y += amp * (Math.sin(tk * 0.75 + bp.gz * 1.7 + bp.gx * 1.1 + c.ph * 2.0)
                        + 0.15 * Math.sin(tk * 1.5 + bp.jit * 5.0));
                    z += amp * (Math.sin(tk * 0.6 + bp.gx * 1.5 + bp.gy * 1.4 + c.ph * 3.0)
                        + 0.15 * Math.sin(tk * 1.3 + bp.jit * 7.0));
                    // ACCRETION, gentled 0.55 → 0.35 (v2): enough to
                    // knit the mutation visibly denser, small enough
                    // that its release can never read erratic.
                    const pull = 0.35 * Sk * w;
                    x -= dx * pull;
                    y -= dy * pull;
                    z -= dz * pull;
                    // RED that reads as a spreading REGION: the heat
                    // front rides the same envelope as the deformation,
                    // so wherever the flesh moves, it is red — idle
                    // cores sit visibly crimson, the mutation drives
                    // the front outward across the grid.
                    heat += wS * (0.30 + 0.34 * St);
                }
                nodeSeeds[i] = Math.min(0.60, bp.seed0 + heat);
            } else {
                // Complexity heart: a small orbiting tangle — violet at
                // rest, swelling and running crimson as ITS mutation
                // grows.
                const c = _cubeCx[bp.ci] || _cubeCx[0] || { ax: 0, ay: 0, az: 0, ph: 0 };
                const Sk = bp.ci === cubeActive ? cubeS : 0;
                const sp = time * (0.35 + 1.3 * Sk) * CALM + bp.jit;
                const cs = Math.cos(sp), sn = Math.sin(sp);
                const r = bp.rr * (1 + 1.1 * Sk + 0.5 * gaitFlash * TUNE.flashGain)
                    * (1 + 0.10 * Math.sin(time * 0.9 + bp.jit * 2.0));
                const ox = bp.dx * cs + bp.dz * sn;
                const oz = -bp.dx * sn + bp.dz * cs;
                x = c.ax + ox * r;
                y = c.ay + bp.dy * r * (1 + 0.3 * Math.sin(time * 0.7 + bp.jit));
                z = c.az + oz * r;
                nodeSeeds[i] = Math.min(0.62,
                    0.52 + 0.08 * Sk + 0.03 * Math.sin(time * 0.8 + bp.jit));
            }
            const x1 = x * kc1 + z * ks1;
            const z1 = -x * ks1 + z * kc1;
            const y2 = y * kc2 - z1 * ks2;
            const z2 = y * ks2 + z1 * kc2;
            currentPositions[i].set(x1 - kfx, y2 - kfy, z2 - kfz);
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
        const shudder = verdict === 'refute' ? TUNE.shudder * verdictEnv : 0;
        const writeW = DIAL.write === 'wave' ? gait.write : 0;
        const twitchP = idleTwitchNode >= 0 ? currentPositions[idleTwitchNode] : null;
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
            if (twitchP && idleTwitch > 0.01) {
                const dx = p.x - twitchP.x, dy = p.y - twitchP.y, dz = p.z - twitchP.z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 0.5 && d2 > 1e-6) {
                    const w = (1 - d2 / 0.5) * idleTwitch * TUNE.twitch * CALM;
                    const inv = 1 / Math.sqrt(d2);
                    p.x += dx * inv * w; p.y += dy * inv * w; p.z += dz * inv * w;
                }
            }
        }
    }
    // Idle twitch: at rest, every 6–14s one neighbourhood shivers — so
    // "rest" is not one uniform drift. Never while working.
    idleTwitch *= 0.94;
    if (drive < 0.15 && userTurnState < 0.1 && !PREFERS_REDUCED_MOTION
        && time - idleTwitchAt > 0.03 * (6 + Math.random() * 8) && Math.random() < 0.02) {
        idleTwitchAt = time;
        idleTwitch = 1.0;
        idleTwitchNode = Math.floor(Math.random() * NODE_COUNT);
    }

    // Reorganization blend: after a form switch, ease from the snapshot
    // of the old body into the freshly computed one.
    if (formBlend < 1.0) {
        formBlend = Math.min(1.0, formBlend + (1 / 60) / 1.4);
        const e = formBlend * formBlend * (3.0 - 2.0 * formBlend);
        for (let i = 0; i < NODE_COUNT; i++) {
            const from = _blendFrom[i];
            if (from) currentPositions[i].lerpVectors(from, currentPositions[i], e);
        }
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
    const connected = new Array(NODE_COUNT).fill(false);

    // The web THICKENS while immersed: easing the proximity threshold
    // up with the dive forms more links exactly when the viewer is in
    // the middle of them (the O(n²) distances are computed either way;
    // this only accepts more pairs, bounded by MAX_LINES).
    // Per-form link-radius² multipliers: the vortex reads best DENSE
    // (1.5× weaves the funnel membrane tighter — "a bit more dense");
    // the AI forms need TIGHTER radii — lattice so only axis-neighbors
    // weave (a clean wireframe tensor, no diagonals), embedding so
    // clusters can never cross-link (the void between concepts stays
    // void), descent so the terrain reads as a mesh surface, not a
    // solid (the mobile sheet is sparser, hence the wider radius).
    const LINK_MULT = {
        vortex: 1.5,
        lattice: 0.45,
        // Cube: neighbors-only like lattice but on the BIGGER cell edge
        // (0.88/0.95); mutation displacement deliberately exceeds the
        // link slack so the grid visibly tears and re-weaves around it.
        cube: 0.62,
        embedding: 0.62,
        // Descent radius must cover a grid step across the WORST-CASE
        // analytic slope of _lossH (all terms aligned), or the sheet
        // tears momentarily on steep ridges — computed invariant pinned
        // in tests/test_interface_face_forms_ai.py.
        descent: IS_MOBILE ? 0.38 : 0.20,
    };
    const proximitySq = PROXIMITY_SQ * (1.0 + dive * 0.15)
        * (LINK_MULT[FORM] === undefined ? 1.0 : LINK_MULT[FORM])
        * (1.0 + TUNE.thickenLinks * gaitThicken);   // read, 'thicken' dialect

    for (let i = 0; i < NODE_COUNT; i++) {
        for (let j = i + 1; j < NODE_COUNT; j++) {
            const distSq = currentPositions[i].distanceToSquared(currentPositions[j]);
            if (distSq < proximitySq) {
                {
                    connected[i] = true;
                    connected[j] = true;

                    if (lineIdx < MAX_LINES) {
                        linePosAttr[lineIdx * 6] = currentPositions[i].x;
                        linePosAttr[lineIdx * 6 + 1] = currentPositions[i].y;
                        linePosAttr[lineIdx * 6 + 2] = currentPositions[i].z;

                        linePosAttr[lineIdx * 6 + 3] = currentPositions[j].x;
                        linePosAttr[lineIdx * 6 + 4] = currentPositions[j].y;
                        linePosAttr[lineIdx * 6 + 5] = currentPositions[j].z;

                        lineUvAttr[lineIdx * 2] = 0;
                        lineUvAttr[lineIdx * 2 + 1] = 1;

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
        lineHueAttr[lineIdx * 2] = 0.60; lineHueAttr[lineIdx * 2 + 1] = 0.58;
        lineIdx++;
    }
    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.aLightPass.needsUpdate = true;
    lineGeometry.attributes.aLineHue.needsUpdate = true;
    lineGeometry.setDrawRange(0, lineIdx * 2);

    // 3. Update nodes meshes (hide unconnected nodes)
    const dummy = new THREE.Object3D();
    for (let i = 0; i < NODE_COUNT; i++) {
        const targetScale = connected[i] ? 1.0 : 0.0;
        nodeScales[i] += (targetScale - nodeScales[i]) * 0.1; // Smooth scale in and out

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
    huePhase += (0.006 + activity * 0.010)
        * (1.0 + 0.35 * Math.sin(time * 0.083))
        * (PREFERS_REDUCED_MOTION ? 0.3 : 1.0);
    hueDrift = 0.07 * Math.sin(huePhase);

    // Per-form center dimming: horizon's core region is by far the
    // densest additive pile-up; the vortex dims by SCREEN-radial
    // distance (its hot zone sits on the view axis at depth) and damps
    // the hue wave/drift so its center stays anchored DARK RED instead
    // of swinging through the plum stop ("bright purple" report).
    const centerDim = FORM === 'vortex' ? 0.85 : 0.30;
    const centerXY = FORM === 'vortex' ? 1.0 : 0.0;
    const waveAmp = FORM === 'vortex' ? 0.3 : 1.0;
    // Mood rides the drift as a slow baseline (cold-pole shift only;
    // the vortex keeps its anchored centre).
    const hueDriftOut = hueDrift * (FORM === 'vortex' ? 0.3 : 1.0)
        + moodHue * (FORM === 'vortex' ? 0.4 : 1.0);
    // Master luminance: EVERY form emits ~half the light so the face
    // BLENDS into the background (operator: "too bright and
    // distracting", then extended to all faces) — structure and motion
    // carry visibility, not brightness. Error KINDS modulate it: a
    // network drop flickers with gaps, a timeout fades; a pass holds a
    // touch brighter while it crystallises.
    let formDim = 0.55;
    if (errorKind === 'network') {
        formDim *= 1.0 - TUNE.netFlicker * errorKindEnv * (Math.sin(time * 90.0) > 0.3 ? 1.0 : 0.0);
    } else if (errorKind === 'timeout') {
        formDim *= 1.0 - TUNE.timeoutFade * errorKindEnv;
    }
    if (verdict === 'pass') formDim *= 1.0 + TUNE.passDim * verdictEnv;

    const nUniforms = instancedMesh.material.uniforms;
    nUniforms.uTime.value = time;
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
    nUniforms.uAccentStrength.value = accentStrength;
    nUniforms.uHueDrift.value = hueDriftOut;

    const lUniforms = lineMaterial.uniforms;
    lUniforms.uTime.value = time;
    lUniforms.uCenterDim.value = centerDim;
    lUniforms.uCenterXY.value = centerXY;
    lUniforms.uWaveAmp.value = waveAmp;
    lUniforms.uFormDim.value = formDim;
    lUniforms.uWorkingState.value = workingState;
    lUniforms.uErrorState.value = errorState;
    lUniforms.uPulseT.value = pulseT;
    lUniforms.uAccentStrength.value = accentStrength;
    lUniforms.uHueDrift.value = hueDriftOut;
    lUniforms.uDive.value = dive;

    // Interior motes: only rendered while actually diving — and never
    // for the empty form (a dive there would summon motes out of a
    // deliberately blank screen).
    motesMesh.visible = dive > 0.01 && FORM !== 'empty';
    if (motesMesh.visible) {
        motesMaterial.uniforms.uTime.value = time;
        motesMaterial.uniforms.uDive.value = dive;
        motesMaterial.uniforms.uHueDrift.value = hueDriftOut;
        motesMaterial.uniforms.uWaveAmp.value = waveAmp;
    }

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
        + 0.08 * backgroundBusy * (0.5 + 0.5 * Math.sin(time * 0.31 + 0.9)))
        * BLOOM_SCALE * (1.0 - 0.5 * dive) * 0.65;

    composer.render();
}
