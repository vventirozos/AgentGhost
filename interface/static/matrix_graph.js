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
// Horizon event choreography: occasional cycles run HOT (deeper
// collapse, brighter core flare, faster infall) — variability is what
// separates a heartbeat from a metronome. coreFlare feeds the core
// quads' render size each frame.
let _lastPulseFract = 0.0;
let eventBoost = 0.0;
let coreFlare = 1.0;

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

// ── Alien forms (2026-07-28) ───────────────────────────────────────
// Three interchangeable body plans over ONE motion engine (the
// asymmetric _pulseShape propulsion, metachronal strand waves, hot
// core, flinch, thermal anatomy):
//   abyssal — asymmetric lobed mantle tilted off-axis, feelers
//             reaching in all directions, off-center core.
//   horizon — eccentric orbital shells collapsing toward a burning
//             core; filament spirals falling inward, heating as
//             they fall.
//   cortex  — asymmetric neural lobes swelling with thought-waves;
//             dendrites radiating outward carrying signal trains.
// The header's form button cycles these; the choice persists.
const FORMS = ['abyssal', 'horizon', 'cortex', 'vortex', 'empty'];
// Default form: VORTEX (operator pick, 2026-07-28 — superseded horizon
// after the black-hole iteration). An explicit button choice still
// overrides via localStorage below.
let formIndex = FORMS.indexOf('vortex');
try {
    const _stored = localStorage.getItem('ghost_face_form');
    const _i = FORMS.indexOf(_stored);
    if (_i >= 0) formIndex = _i;
} catch (e) { /* private mode */ }
// Reorganization blend: on a form switch the nodes visibly re-assemble
// from their old positions into the new anatomy over ~1.4s.
let formBlend = 1.0;
const _blendFrom = [];
// Abyssal fixed off-axis tilt (baked trig, applied per frame).
const _TCX = Math.cos(0.42), _TSX = Math.sin(0.42);
const _TCZ = Math.cos(0.18), _TSZ = Math.sin(0.18);

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
uniform float uCenterDim;
uniform float uCenterXY;
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
    vec3 jewel = palette(aSeed + uHueDrift + hueWave(instancePos));

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
    gl_FragColor = vec4(col * (1.0 + uPulseT * 0.3) * depthDim * vLineNear * diveDim * centerFade, alpha * diveDim * centerFade * mix(1.0, 0.55, vLineDepth));
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
    if (f === 'horizon') _buildHorizon();
    else if (f === 'cortex') _buildCortex();
    else if (f === 'vortex') _buildVortex();
    else if (f === 'empty') _buildEmpty();
    else _buildAbyssal();
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

// Form A — ABYSSAL: an unclassifiable biomechanical mass. The mantle is
// a lobed, UNEQUAL husk (no earthly symmetry) tilted off-axis, feelers
// reach in all directions (down, sideways, some upward — reaching reads
// sentient; hanging read jellyfish), and the core sits off-center.
function _buildAbyssal() {
    const CORE_COUNT = Math.max(8, Math.round(NODE_COUNT * 0.055));
    const STRANDS = IS_MOBILE ? 8 : 11;
    const PER_STRAND = Math.max(5, Math.floor((NODE_COUNT * 0.34) / STRANDS));
    const MANTLE_COUNT = NODE_COUNT - CORE_COUNT - STRANDS * PER_STRAND;
    const BELL_R = 1.5, BELL_SWEEP = 1.9;   // sweep past the equator — a husk, not a bell
    const lobeP1 = Math.random() * Math.PI * 2;
    const lobeP2 = Math.random() * Math.PI * 2;
    const lobeOf = (theta, u) => 1 + 0.24 * Math.sin(2 * theta + lobeP1)
        + 0.15 * Math.sin(3 * theta + lobeP2)
        + 0.08 * Math.sin(5 * theta + u * 4.0);
    let n = 0;

    for (let b = 0; b < MANTLE_COUNT; b++, n++) {
        const u = Math.sqrt((b + 0.5) / MANTLE_COUNT);
        const theta = b * 2.399963;
        const alpha = u * BELL_SWEEP;
        basePositions.push({
            kind: 0, u, theta,
            cos: Math.cos(theta), sin: Math.sin(theta),
            r0: BELL_R * Math.sin(alpha) * lobeOf(theta, u)
                + (Math.random() - 0.5) * 0.07,
            y0: 0.9 - BELL_R * 1.05 * (1 - Math.cos(alpha))
                + (Math.random() - 0.5) * 0.07,
        });
        nodeSeeds[n] = 0.02 + u * 0.30 + Math.random() * 0.04;
    }

    for (let k = 0; k < STRANDS; k++) {
        // Feelers emerge from random mantle latitudes with their own
        // reach directions.
        const thetaA = Math.random() * Math.PI * 2;
        const uA = 0.55 + Math.random() * 0.45;
        const aA = uA * BELL_SWEEP;
        const rA = BELL_R * Math.sin(aA) * lobeOf(thetaA, uA);
        const ax = rA * Math.cos(thetaA);
        const az = rA * Math.sin(thetaA);
        const ay = 0.9 - BELL_R * 1.05 * (1 - Math.cos(aA));
        let dx = Math.cos(thetaA) * (0.5 + Math.random() * 0.5);
        let dz = Math.sin(thetaA) * (0.5 + Math.random() * 0.5);
        let dy = -1.1 + Math.random() * 1.7;          // most drift down, some reach up
        const dl = Math.hypot(dx, dy, dz);
        dx /= dl; dy /= dl; dz /= dl;
        // Wave-plane perpendiculars for the metachronal sway.
        let p1x = -dz, p1z = dx;
        const p1l = Math.hypot(p1x, p1z) || 1;
        p1x /= p1l; p1z /= p1l;
        const p2x = dy * p1z, p2y = dz * p1x - dx * p1z, p2z = -dy * p1x;
        const len = 1.7 + Math.random() * 1.1;
        const swaySpeed = 0.5 + Math.random() * 0.4;
        const swayPhase = Math.random() * Math.PI * 2;
        for (let j = 0; j < PER_STRAND; j++, n++) {
            const s = (j + 1) / PER_STRAND;
            basePositions.push({
                kind: 1, s, ax, ay, az, dx, dy, dz,
                p1x, p1z, p2x, p2y, p2z,
                len, swaySpeed, swayPhase,
                swayAmp: 0.30 * Math.pow(s, 1.3),
            });
            nodeSeeds[n] = 0.40 + s * 0.28 + Math.random() * 0.05;
        }
    }

    for (let c = 0; c < CORE_COUNT; c++, n++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const rr = 0.4 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2,
            hx: 0.28 + rr * Math.sin(ph) * Math.cos(th),   // off-center on purpose
            hy: 0.05 + rr * Math.cos(ph) * 0.7,
            hz: -0.12 + rr * Math.sin(ph) * Math.sin(th),
            jit: Math.random() * Math.PI * 2,
        });
        nodeSeeds[n] = 0.58 + Math.random() * 0.08;
    }
}

// Form B — EVENT HORIZON: information falling into a mind. Eccentric
// orbital shells collapse toward the core (outer first — the wave
// travels inward), filament spirals stream in and HEAT as they fall.
function _buildHorizon() {
    // Core trimmed 10% → 7% of nodes and each core quad rendered at 0.7
    // size (2026-07-28 operator: "too bright at its center"): additive
    // stacking is what bleached the dark red into hot pink — fewer,
    // smaller overlaps keep the center CRIMSON instead of washing out.
    const CORE_COUNT = Math.max(10, Math.round(NODE_COUNT * 0.07));
    const STRANDS = IS_MOBILE ? 5 : 7;
    const PER_STRAND = Math.max(6, Math.floor((NODE_COUNT * 0.30) / STRANDS));
    const SHELL_TOTAL = NODE_COUNT - CORE_COUNT - STRANDS * PER_STRAND;
    const SHELLS = [
        { r: 0.85, tilt: 0.35, omega: 0.20, off: 0.10 },
        { r: 1.45, tilt: -0.22, omega: 0.12, off: 0.18 },
        { r: 2.05, tilt: 0.12, omega: 0.07, off: 0.26 },
    ];
    let n = 0;
    const per = Math.floor(SHELL_TOTAL / SHELLS.length);
    for (let si = 0; si < SHELLS.length; si++) {
        const sh = SHELLS[si];
        const count = si === SHELLS.length - 1
            ? SHELL_TOTAL - per * (SHELLS.length - 1) : per;
        for (let b = 0; b < count; b++, n++) {
            const cosP = 2 * ((b + 0.5) / count) - 1;
            basePositions.push({
                kind: 0, shell: si,
                r: sh.r * (0.94 + Math.random() * 0.12),
                cosP, sinP: Math.sqrt(Math.max(0, 1 - cosP * cosP)),
                phi0: b * 2.399963, omega: sh.omega,
                tiltC: Math.cos(sh.tilt), tiltS: Math.sin(sh.tilt),
                offX: sh.off, jit: Math.random() * Math.PI * 2,
            });
            // Inner shells run HOT — matter heats as it falls. The inner
            // shell sits on the arterial-red stop itself: any blue this
            // close to the core additively mixes into magenta (the "too
            // bright pink center" report), so the center is kept pure red
            // and the cold blues live only in the outer shells.
            nodeSeeds[n] = [0.58, 0.16, 0.04][si] + Math.random() * 0.03;
        }
    }
    for (let k = 0; k < STRANDS; k++) {
        const phi0 = (k / STRANDS) * Math.PI * 2 + Math.random() * 0.4;
        const pitch = (Math.random() < 0.5 ? -1 : 1) * (0.5 + Math.random() * 0.5);
        const flow = 1.6 + Math.random() * 0.8;
        for (let j = 0; j < PER_STRAND; j++, n++) {
            const s = (j + 1) / PER_STRAND;             // 0 rim → 1 core (r floor 0.45)
            basePositions.push({ kind: 1, s, phi0, pitch, flow,
                jit: Math.random() * Math.PI * 2,
                // Inner filament ends shrink a little too — they pile up
                // right where the core already stacks.
                sz: s > 0.75 ? 0.85 : 1.0 });
            // Filaments run red along most of their fall — violet only at
            // the outermost rim — so the bright inner band reads crimson.
            nodeSeeds[n] = 0.42 + s * 0.20 + Math.random() * 0.03;
        }
    }
    for (let c = 0; c < CORE_COUNT; c++, n++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const rr = 0.38 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2,
            hx: rr * Math.sin(ph) * Math.cos(th),
            hy: rr * Math.cos(ph),
            hz: rr * Math.sin(ph) * Math.sin(th),
            jit: Math.random() * Math.PI * 2,
            sz: 0.7,
        });
        // Tight anchor dead on the arterial-red stop — a wide seed spread
        // let half the core drift toward violet/plum and read pink.
        nodeSeeds[n] = 0.595 + Math.random() * 0.03;
    }
}

// Form C — SYNTHETIC CORTEX: an asymmetric cluster of neural lobes.
// Thought-waves swell the lobes in sequence; dendrites radiate outward
// carrying displacement signal-trains toward their tips.
function _buildCortex() {
    const LOBE_CENTERS = [
        [0.75, 0.35, 0.15], [-0.60, 0.45, -0.30],
        [0.15, -0.25, 0.70], [-0.30, -0.40, -0.65],
    ];
    const LOBE_R = [0.78, 0.68, 0.62, 0.72];
    const CORE_COUNT = Math.max(8, Math.round(NODE_COUNT * 0.06));
    const STRANDS = IS_MOBILE ? 10 : 14;
    const PER_STRAND = Math.max(4, Math.floor((NODE_COUNT * 0.30) / STRANDS));
    const LOBE_TOTAL = NODE_COUNT - CORE_COUNT - STRANDS * PER_STRAND;
    let n = 0;
    const per = Math.floor(LOBE_TOTAL / LOBE_CENTERS.length);
    for (let li = 0; li < LOBE_CENTERS.length; li++) {
        const count = li === LOBE_CENTERS.length - 1
            ? LOBE_TOTAL - per * (LOBE_CENTERS.length - 1) : per;
        for (let b = 0; b < count; b++, n++) {
            const th = Math.random() * Math.PI * 2;
            const ph = Math.acos(2 * Math.random() - 1);
            basePositions.push({
                kind: 0, lobe: li,
                cx: LOBE_CENTERS[li][0], cy: LOBE_CENTERS[li][1], cz: LOBE_CENTERS[li][2],
                dx: Math.sin(ph) * Math.cos(th),
                dy: Math.cos(ph),
                dz: Math.sin(ph) * Math.sin(th),
                r0: LOBE_R[li] * (0.88 + Math.random() * 0.28),
                lobePhase: li * 0.18 + Math.random() * 0.05,
                jit: Math.random() * Math.PI * 2,
            });
            nodeSeeds[n] = 0.04 + li * 0.055 + Math.random() * 0.03;
        }
    }
    for (let k = 0; k < STRANDS; k++) {
        const li = k % LOBE_CENTERS.length;
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const dx = Math.sin(ph) * Math.cos(th);
        const dy = Math.cos(ph);
        const dz = Math.sin(ph) * Math.sin(th);
        const ax = LOBE_CENTERS[li][0] + dx * LOBE_R[li];
        const ay = LOBE_CENTERS[li][1] + dy * LOBE_R[li];
        const az = LOBE_CENTERS[li][2] + dz * LOBE_R[li];
        let p1x = -dz, p1z = dx;
        const p1l = Math.hypot(p1x, p1z) || 1;
        p1x /= p1l; p1z /= p1l;
        const p2x = dy * p1z, p2y = dz * p1x - dx * p1z, p2z = -dy * p1x;
        const len = 1.4 + Math.random() * 0.9;
        const swaySpeed = 0.9 + Math.random() * 0.5;
        const swayPhase = Math.random() * Math.PI * 2;
        for (let j = 0; j < PER_STRAND; j++, n++) {
            const s = (j + 1) / PER_STRAND;
            basePositions.push({
                kind: 1, s, ax, ay, az, dx, dy, dz,
                p1x, p1z, p2x, p2y, p2z,
                len, swaySpeed, swayPhase,
                swayAmp: 0.10 * Math.pow(s, 1.2),
                signal: 2.2 + Math.random() * 0.9,
            });
            nodeSeeds[n] = 0.40 + s * 0.28 + Math.random() * 0.05;
        }
    }
    for (let c = 0; c < CORE_COUNT; c++, n++) {
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const rr = 0.35 * Math.cbrt(Math.random());
        basePositions.push({
            kind: 2,
            hx: rr * Math.sin(ph) * Math.cos(th),
            hy: 0.05 + rr * Math.cos(ph),
            hz: rr * Math.sin(ph) * Math.sin(th),
            jit: Math.random() * Math.PI * 2,
        });
        nodeSeeds[n] = 0.58 + Math.random() * 0.08;
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

// ── Form switching ─────────────────────────────────────────────────
export function getForm() { return FORMS[formIndex]; }

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
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uAudioLevel: { value: 0.0 },
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
export function noteError() {
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
        eventBoost,
        coreFlare,
        vortexTravel,
        tunnelFlow,
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

function animate() {
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

    // Vortex engagement + flow (see the vortexTravel declaration).
    // Engagement eases up while a user turn runs and eases back down on
    // completion — since the camera never moves and the cone is
    // self-similar, that ease IS the whole transition: busy morphs into
    // idle with no reset of any kind.
    if (formIndex === 3) {
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
            + 0.19 * travelNorm) * flowTurb * CALM;
        // Swirl (2026-07-28 tuning): idle raised +25% (0.10 → 0.125),
        // and busy HALVES the rate rather than raising it — work
        // de-emphasizes rotation entirely so the surging swallow owns
        // the busy state without competition.
        vortexSpin += (1 / 60) * (0.125
            - 0.050 * Math.max(workingState, activity, travelNorm)) * CALM;
    } else {
        vortexTravel = 0.0;
    }

    // Accumulate time for lines at steady pace — except inside the
    // cloud, where the data pulses rush a little faster: the interior
    // should feel BUSIER than the outside view, not emptier.
    time += 0.005 * (1.0 + dive * 0.6);

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
    const sceneScale = formIndex === 3 ? 0.9 * breathe : baseScale;
    scene.scale.set(sceneScale, sceneScale, sceneScale);

    // Slow continuous orbit (faster while working) plus a gentle tilt
    // gives the network real depth and life at idle. Skipped entirely
    // under reduced-motion.
    if (!PREFERS_REDUCED_MOTION) {
        if (formIndex === 3) {
            // Vortex: the funnel must keep facing the viewer, so no big
            // heading wander — instead the whole scene ROLLS around the
            // view axis (global swirl) with only a whisper of tilt.
            scene.rotation.x = 0.06 * Math.sin(time * 0.050);
            scene.rotation.y = 0.05 * Math.sin(time * 0.031 + 1.0);
            scene.rotation.z = vortexSpin * 0.22;
            scene.position.x = 0.05 * Math.sin(time * 0.031);
            scene.position.y = _bob * 0.5;
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
    if (formIndex === 3) {
        // Vortex: THE CAMERA NEVER MOVES — the self-similar swallow does
        // all the traveling. It studies the hole itself.
        camera.position.z = CAMERA_REST_Z;
        camera.lookAt(camera.position.x * 0.35, camera.position.y * 0.35,
            VORTEX_APEX_Z);
    } else {
        camera.position.z = CAMERA_REST_Z - (CAMERA_REST_Z - CAMERA_DIVE_Z) * dive;
        // Look-target blend: at rest the camera studies the origin; deep in
        // the cloud it must look FORWARD through it instead — near the
        // origin, lookAt(0,0,0) turns tiny parallax offsets into wild
        // rotations (the lookAt singularity).
        camera.lookAt(0, 0, -3.5 * dive);
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
        * (PREFERS_REDUCED_MOTION ? 0.5 : 1.0);
    flinch *= 0.97;

    // Major infall events: on ~1 in 5 cycle wraps, the next cycle runs
    // hot. Decays over ~2s so the event owns most of its cycle.
    const _pf = _fract(pulsePhase);
    if (_pf < _lastPulseFract && !PREFERS_REDUCED_MOTION
        && Math.random() < 0.22) {
        eventBoost = 1.0;
    }
    _lastPulseFract = _pf;
    eventBoost *= 0.995;
    if (eventBoost < 0.02) eventBoost = 0;

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

    if (formIndex === 0) {
        // ── ABYSSAL: lobed husk contraction + omnidirectional feelers,
        //    the whole body baked onto an off-axis tilt (no "up").
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            let x, y, z;
            if (bp.kind === 0) {
                const local = _pulseShape(_fract(pulsePhase - bp.u * 0.16));
                const squeeze = 1.0 - pulseAmp * local * (0.30 + 0.70 * bp.u);
                const r = bp.r0 * squeeze
                    + 0.025 * CALM * Math.sin(time * 1.3 + bp.theta * 3.0 + bp.u * 7.0);
                x = r * bp.cos;
                z = r * bp.sin;
                y = bp.y0 + 0.20 * pulseAmp * local * bp.u
                    + 0.02 * CALM * Math.sin(time * 0.9 + bp.theta * 2.0);
            } else if (bp.kind === 1) {
                // Feelers retract slightly with each contraction —
                // reaching and recoiling, not hanging.
                const reach = bp.len * bp.s * (1.0 - 0.22 * pulseAmp * cMargin);
                const swayA = time * bp.swaySpeed + bp.swayPhase - bp.s * 3.3;
                const lat = bp.swayAmp * CALM * (1.0 + 0.25 * drive) * Math.sin(swayA);
                const lat2 = bp.swayAmp * 0.7 * CALM * Math.cos(swayA * 0.83 + 1.1);
                x = bp.ax + bp.dx * reach + bp.p1x * lat + bp.p2x * lat2;
                y = bp.ay + bp.dy * reach + bp.p2y * lat2;
                z = bp.az + bp.dz * reach + bp.p1z * lat + bp.p2z * lat2;
            } else {
                const c = _pulseShape(_fract(pulsePhase + 0.05));
                const g = 1.0 + 0.12 * c + 0.05 * Math.sin(time * 0.8 + bp.jit);
                x = bp.hx * g;
                y = bp.hy * g + 0.03 * Math.sin(time * 0.6 + bp.jit * 2.0);
                z = bp.hz * g;
            }
            const x1 = x * _TCZ - y * _TSZ;
            const y1 = x * _TSZ + y * _TCZ;
            const y2 = y1 * _TCX - z * _TSX;
            const z2 = y1 * _TSX + z * _TCX;
            currentPositions[i].set(x1, y2, z2);
        }
    } else if (formIndex === 1) {
        // ── EVENT HORIZON — a choreographed feeding cycle, not a beat:
        //    filament surges race rim→core, the core FLARES as they
        //    land, and a rebound ripple travels back out through the
        //    shells. Collapse squeezes hardest along a slowly precessing
        //    TIDAL AXIS (directional deformation reads gravitational;
        //    the old uniform radial shrink read mechanical/boring).
        const hAmp = pulseAmp * (1.0 + 0.55 * eventBoost);
        const tideA = time * 0.16;
        const tideY = 0.34 * Math.sin(time * 0.05);
        const tideX = Math.cos(tideA) * (1.0 - Math.abs(tideY) * 0.5);
        const tideZ = Math.sin(tideA) * (1.0 - Math.abs(tideY) * 0.5);
        const flareEnv = Math.pow(_pulseShape(_fract(pulsePhase - 0.02)), 2.0);
        coreFlare = 1.0 + (0.35 + 0.50 * eventBoost) * flareEnv;

        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            if (bp.kind === 0) {
                const phi = bp.phi0 + time * bp.omega * CALM;
                const dirX = bp.sinP * Math.cos(phi);
                const dirY = bp.cosP;
                const dirZ = bp.sinP * Math.sin(phi);
                const align = dirX * tideX + dirY * tideY + dirZ * tideZ;
                const collapse = _pulseShape(_fract(pulsePhase - bp.shell * 0.12));
                const rebound = _pulseShape(_fract(pulsePhase - 0.34 - (2 - bp.shell) * 0.10));
                const r = bp.r
                    * (1.0 - hAmp * collapse * (0.30 + 0.55 * align * align))
                    * (1.0 + 0.10 * hAmp * rebound)
                    * (1.0 + 0.04 * CALM * Math.sin(time * 0.5 + bp.jit));
                const x = r * dirX + bp.offX * Math.sin(time * 0.07 + bp.shell * 2.1);
                const y = r * dirY;
                const z = r * dirZ;
                currentPositions[i].set(
                    x,
                    y * bp.tiltC - z * bp.tiltS,
                    y * bp.tiltS + z * bp.tiltC);
            } else if (bp.kind === 1) {
                const phi = bp.phi0 + bp.s * 4.4 + time * 0.22 * CALM;
                // Infall surge: a packet races down the spiral each
                // cycle (lag grows toward the rim), pulling the filament
                // inward as it passes — it lands as the core flares.
                const surge = _pulseShape(_fract(pulsePhase - (1.0 - bp.s) * 0.35));
                const rr = (2.3 - 1.85 * bp.s)
                    * (1.0 - 0.10 * (1.0 + eventBoost) * surge)
                    * (1.0 + 0.05 * CALM * Math.sin(
                        bp.s * 9.0 - time * bp.flow * (1.0 + eventBoost)));
                currentPositions[i].set(
                    rr * Math.cos(phi),
                    bp.pitch * (bp.s - 0.5)
                        + 0.05 * CALM * Math.sin(time * 0.7 + bp.jit),
                    rr * Math.sin(phi));
            } else {
                // The core drinks: a small steady beat plus the FLARE
                // when the surges land (size flare rides coreFlare in
                // the instance-matrix pass below).
                const g = 1.0 + 0.10 * _pulseShape(_fract(pulsePhase + 0.05))
                    + 0.30 * flareEnv * (1.0 + eventBoost)
                    + 0.05 * Math.sin(time * 0.8 + bp.jit);
                currentPositions[i].set(bp.hx * g, bp.hy * g, bp.hz * g);
            }
        }
    } else if (formIndex === 3) {
        // ── VORTEX: the self-similar swallow. Each wall node's depth
        //    FLOWS (dEff advances with tunnelFlow); its distance from
        //    the apex is L0·e^(−k·d) so the wrap is an invisible fractal
        //    renormalization; its hue re-heats blue→red along the fall
        //    (seeds recomputed per frame, uploaded below). A drifting
        //    harmonic field morphs the cross-section through organic,
        //    never-repeating shapes — the "procedurally generated
        //    content" being consumed — swelling while a turn runs.
        const engage = Math.min(vortexTravel / Math.max(IMMERSION_CAP, 0.001), 1.0);
        // Idle pulsing cut to 20% (2026-07-28 tuning): at rest the hole
        // barely breathes; the pulse returns in full with engagement or
        // ambient drive.
        const vPulseScale = 0.2 + 0.8 * Math.max(engage, drive);
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
                // Spiral wind cut 4.2 → 1.2: the wind is traversed by the
                // flow, so at busy flow rates it alone corkscrewed the
                // whole scene ~48°/s — THIS was the "fast rotation while
                // busy", not vortexSpin.
                const theta = bp.theta0 + (1.0 - dOut) * 1.2
                    + vortexSpin * (1.5 - dOut);
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
    } else if (formIndex === 4) {
        // ── EMPTY: the dispersed far sphere — static, unlinked,
        //    invisible. (Must stay an explicit branch: the trailing
        //    else belongs to cortex.)
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            currentPositions[i].set(bp.hx, bp.hy, bp.hz);
        }
    } else {
        // ── SYNTHETIC CORTEX: thought-waves swell the lobes in
        //    sequence; dendrites carry displacement signal-trains.
        for (let i = 0; i < NODE_COUNT; i++) {
            const bp = basePositions[i];
            if (bp.kind === 0) {
                const wave = _pulseShape(_fract(pulsePhase - bp.lobePhase));
                const g = 1.0 + 0.55 * pulseAmp * wave;
                const spread = 1.0 + 0.18 * pulseAmp * wave;
                currentPositions[i].set(
                    bp.cx * spread + bp.dx * bp.r0 * g
                        + 0.02 * CALM * Math.sin(time * 0.9 + bp.jit),
                    bp.cy * spread + bp.dy * bp.r0 * g
                        + 0.02 * CALM * Math.sin(time * 0.7 + bp.jit * 2.0),
                    bp.cz * spread + bp.dz * bp.r0 * g);
            } else if (bp.kind === 1) {
                const swayA = time * bp.swaySpeed + bp.swayPhase - bp.s * 2.8;
                const lat = bp.swayAmp * CALM * Math.sin(swayA);
                const lat2 = bp.swayAmp * 0.7 * CALM * Math.cos(swayA * 0.83 + 1.1);
                const sig = 0.05 * CALM * Math.sin(bp.s * 10.0 - time * bp.signal);
                const reach = bp.len * bp.s + sig;
                currentPositions[i].set(
                    bp.ax + bp.dx * reach + bp.p1x * lat + bp.p2x * lat2,
                    bp.ay + bp.dy * reach + bp.p2y * lat2,
                    bp.az + bp.dz * reach + bp.p1z * lat + bp.p2z * lat2);
            } else {
                const c = _pulseShape(_fract(pulsePhase + 0.05));
                const g = 1.0 + 0.14 * c + 0.05 * Math.sin(time * 0.8 + bp.jit);
                currentPositions[i].set(
                    bp.hx * g,
                    bp.hy * g + 0.03 * Math.sin(time * 0.6 + bp.jit * 2.0),
                    bp.hz * g);
            }
        }
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
    // The vortex reads best DENSE — a 1.5× link-radius² multiplier weaves
    // the funnel membrane much tighter (operator: "a bit more dense").
    const proximitySq = PROXIMITY_SQ * (1.0 + dive * 0.15)
        * (formIndex === 3 ? 1.5 : 1.0);

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
            * (formIndex === 1 && bpi.kind === 2 ? coreFlare : 1.0);
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
    const centerDim = formIndex === 1 || formIndex === 3 ? 0.85 : 0.30;
    const centerXY = formIndex === 3 ? 1.0 : 0.0;
    const waveAmp = formIndex === 3 ? 0.3 : 1.0;
    const hueDriftOut = hueDrift * (formIndex === 3 ? 0.3 : 1.0);

    const nUniforms = instancedMesh.material.uniforms;
    nUniforms.uTime.value = time;
    nUniforms.uCenterDim.value = centerDim;
    nUniforms.uCenterXY.value = centerXY;
    nUniforms.uWaveAmp.value = waveAmp;
    nUniforms.uWorkingState.value = workingState;
    nUniforms.uErrorState.value = errorState;
    nUniforms.uPulseT.value = pulseT;
    nUniforms.uAudioLevel.value = audioLevel;
    nUniforms.uAccentStrength.value = accentStrength;
    nUniforms.uHueDrift.value = hueDriftOut;

    const lUniforms = lineMaterial.uniforms;
    lUniforms.uTime.value = time;
    lUniforms.uCenterDim.value = centerDim;
    lUniforms.uCenterXY.value = centerXY;
    lUniforms.uWaveAmp.value = waveAmp;
    lUniforms.uWorkingState.value = workingState;
    lUniforms.uErrorState.value = errorState;
    lUniforms.uPulseT.value = pulseT;
    lUniforms.uAccentStrength.value = accentStrength;
    lUniforms.uHueDrift.value = hueDriftOut;
    lUniforms.uDive.value = dive;

    // Interior motes: only rendered while actually diving — and never
    // for the empty form (a dive there would summon motes out of a
    // deliberately blank screen).
    motesMesh.visible = dive > 0.01 && formIndex !== 4;
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
    bloomPass.strength = (0.88 + workingState * 0.3 + activity * 0.35
        + errorState * 0.5) * BLOOM_SCALE * (1.0 - 0.5 * dive);

    composer.render();
}
