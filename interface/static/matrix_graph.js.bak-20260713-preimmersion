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
// Dark-but-MULTICOLOR (2026-07-13): instead of one active hue, every
// node owns a stable position on a 5-stop jewel-tone wheel (deep violet
// → electric blue → teal → emerald → magenta) and the whole wheel
// drifts slowly, so the graph is iridescent rather than monochrome.
// Lines gradient between their two endpoint hues. Additive blending +
// bloom lift these considerably, so every stop is deliberately several
// stops darker than it will read on screen — the theme stays dark-first
// (no neon, no flashbang; the 2026-07-12 envelope smoothing is intact).
export const COLORS = {
    background: new THREE.Color('#000000'),

    // Dim floor under every hue — what a node/line reads as at the
    // bottom of its color breath (a whisper of its jewel hue is always
    // added on top, so even the idle graph is multicolor, not grey).
    nodeBase: new THREE.Color('#0b0714'),
    lineBase: new THREE.Color('#070a18'),

    // The jewel wheel. Order matters — adjacent stops blend into each
    // other, and the wheel wraps (last → first). Warm yellows/oranges
    // are deliberately absent so the crimson ERROR tint stays unique.
    // Stops dimmed ~18% on 2026-07-13 operator feedback ("a tiny bit
    // too bright") — the dim floor, line gradients, and bloom all scale
    // off these, so this is the one knob for overall face brightness.
    palette: [
        new THREE.Color('#3e187a'),  // deep violet
        new THREE.Color('#1f39a1'),  // electric blue
        new THREE.Color('#0a6675'),  // teal
        new THREE.Color('#0f7143'),  // emerald
        new THREE.Color('#80198f'),  // magenta
    ],

    nodeError: new THREE.Color('#7a0f26'),   // crimson red
    lineError: new THREE.Color('#8f1030'),   // crimson red
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
const PROXIMITY_SQ = 2.5;

let scene, camera, renderer, composer, bloomPass;
let instancedMesh, linesMesh;
let lineGeometry, nodeMaterial, lineMaterial;
let time = 0;
let shapeTime = 0;
let sceneSpin = 0;  // accumulated slow-orbit angle (radians)
let currentShapeSpeed = SPEEDS.idle;
let animationFrameId;

let errorState = 0.0;
let targetErrorState = 0.0;
let workingState = 0.0;
let targetWorkingState = 0.0;

// Shader "energy" (uPulseT in the shaders): faint traveling charge on
// the lines + a mild node glow. Since 2026-07-12 this is DERIVED from
// the smoothed activity envelope every frame — it is no longer a
// per-event shockwave.
let pulseT = 0.0;

// Accent tint — the graph's slow "mood" color, blended toward the hue
// of recent log activity and drained back to neutral over ~10s.
// Neutral default = deep purple, matching the dark theme.
const accentColor = new THREE.Color(0x3d1460);
let accentStrength = 0.0;

// Hue drift — the whole jewel wheel slides slowly around its loop
// (~50s per full cycle at idle, ~15s when busy), so a node's color is
// stable moment-to-moment but the graph's overall cast keeps evolving.
let hueDrift = 0.0;

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

// Structural evolution: a few nodes at a time slowly MIGRATE to new
// home positions, so the graph's topology genuinely evolves (proximity
// links dissolve and re-form along the way) instead of oscillating
// around a fixed skeleton. Rate scales gently with activity.
let migrateCooldown = 4.0;         // seconds until the next migration starts
const MAX_CONCURRENT_MIGRATIONS = 5;

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
    if (instancedMesh && instancedMesh.geometry) instancedMesh.geometry.dispose();
    if (linesMesh && linesMesh.geometry) linesMesh.geometry.dispose();
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

const nodeVertexShader = `
attribute float aSeed;
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
varying vec3 vColor;
varying vec2 vUv;
varying float vDepthFade;

void main() {
    vUv = uv;

    // Extract instance position
    vec3 instancePos = (instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // Extract scale from instance matrix (assuming isotropic)
    float scale = length(vec3(instanceMatrix[0][0], instanceMatrix[0][1], instanceMatrix[0][2]));

    // Billboard logic: apply local face offset directly in view space.
    // Audio level gently inflates billboard size so the sphere feels
    // like it's breathing with the voice during TTS playback.
    float sizeBoost = 1.0 + uAudioLevel * 0.25 + uPulseT * 0.15;
    vec4 mvPosition = modelViewMatrix * vec4(instancePos, 1.0);
    mvPosition.xy += position.xy * scale * sizeBoost;

    gl_Position = projectionMatrix * mvPosition;

    // Depth fade: 0 near the camera .. 1 far. Lets the fragment dim
    // distant nodes so the cloud reads as a 3D volume, not a flat sheet.
    vDepthFade = clamp((-mvPosition.z - 2.5) / 7.0, 0.0, 1.0);

    // Per-node jewel hue: aSeed anchors this node on the palette wheel,
    // uHueDrift slides the whole wheel slowly. Replaces the old
    // two-hue (dark blue <-> deep purple) position mix — the graph is
    // now genuinely multicolor while staying inside the dark family.
    vec3 jewel = palette(aSeed + uHueDrift);

    // Life: each node breathes between a dim floor and its full jewel
    // hue. The floor keeps a whisper of the hue so even the idle graph
    // reads as multicolor, not grey.
    float colorMix = sin(instancePos.x * 2.0 + instancePos.y * 2.0 + uWorkingState) * 0.5 + 0.5;
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
    col = mix(col, uAccentColor, clamp(uAccentStrength * accentFalloff, 0.0, 0.85));
    // Brightness boost during an active pulse — subtle; the bloom pass
    // amplifies this considerably.
    col *= (1.0 + uPulseT * 0.4);
    vColor = col;
}
`;

const nodeFragmentShader = `
varying vec3 vColor;
varying vec2 vUv;
varying float vDepthFade;
void main() {
    float d = distance(vUv, vec2(0.5)) * 2.0; // scaled 0 to 1
    if (d > 1.0) discard;

    // Soft quadratic falloff
    float intensity = pow(1.0 - d, 2.0) * 0.8;
    // Intense bright core
    float core = pow(1.0 - d, 8.0) * 1.5;

    float alpha = intensity + core;
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

void main() {
    vLightPass = aLightPass;
    vLineHue = aLineHue;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vLineDepth = clamp((-mvPosition.z - 2.5) / 7.0, 0.0, 1.0);
    gl_Position = projectionMatrix * mvPosition;
}
`;

const lineFragmentShader = `
uniform float uTime;
uniform float uWorkingState;
uniform float uErrorState;
uniform float uPulseT;
uniform float uHueDrift;
uniform vec3 uBaseColor;
uniform vec3 uErrorColor;
uniform vec3 uAccentColor;
uniform float uAccentStrength;
${paletteGLSL}
varying float vLightPass;
varying float vLineHue;
varying float vLineDepth;

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

    // Each line is a GRADIENT between its two endpoint nodes' jewel
    // hues (vLineHue interpolates the endpoint seeds), breathing
    // between a dim floor and the full hue like the nodes do.
    vec3 jewel = palette(vLineHue + uHueDrift);
    float colorMix = sin(vLightPass * 10.0 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor + jewel * 0.18, jewel, colorMix);

    vec3 col = mix(mixCol, uErrorColor, uErrorState * 0.8);
    col = mix(col, uAccentColor, clamp(uAccentStrength * 0.6, 0.0, 0.7));

    float alpha = mix(0.4 + uWorkingState * 0.2, 1.0, pulse);
    alpha = max(alpha, burst * uPulseT);
    // Distance dimming so far links recede behind near ones.
    float depthDim = mix(1.0, 0.4, vLineDepth);
    gl_FragColor = vec4(col * (1.0 + uPulseT * 0.3) * depthDim, alpha * mix(1.0, 0.55, vLineDepth));
}
`;

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

    // Initialize Base Positions
    for (let i = 0; i < NODE_COUNT; i++) {
        // Uniform spherical distribution
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        const r = 2.0 * Math.cbrt(Math.random()); // Larger radius from center than 1.8

        basePositions.push({
            x: r * Math.sin(phi) * Math.cos(theta),
            y: r * Math.sin(phi) * Math.sin(theta),
            z: r * Math.cos(phi),
            phaseX: Math.random() * Math.PI * 2, // Keep independent phases!
            phaseY: Math.random() * Math.PI * 2,
            phaseZ: Math.random() * Math.PI * 2,
            speed: 0.15 + Math.random() * 0.4
        });
        currentPositions[i] = new THREE.Vector3();
        nodeSeeds[i] = Math.random();
    }

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
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
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
        accentTarget = Math.min(0.4, accentTarget + w * 0.35);
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

    // Accumulate time for lines at steady pace regardless of state
    time += 0.005;

    // Idle breathing: ±1% scene-scale sine at ~0.1Hz. Below the
    // motion-detection threshold on both desktop and mobile; keeps the
    // sphere from looking frozen when nothing else is animating.
    const breathe = 1.0 + 0.012 * Math.sin(time * 0.6);
    const baseScale = 0.9 * breathe;
    scene.scale.set(baseScale, baseScale, baseScale);

    // Slow continuous orbit (faster while working) plus a gentle tilt
    // gives the network real depth and life at idle. Skipped entirely
    // under reduced-motion.
    if (!PREFERS_REDUCED_MOTION) {
        sceneSpin += 0.0006 + workingState * 0.0010;
        scene.rotation.y = sceneSpin;
        scene.rotation.x = 0.12 * Math.sin(time * 0.25);
    }

    // Parallax: ease camera toward target offset. Never changes z, so
    // depth-of-field / chat clipping don't flicker.
    camera.position.x += (parallaxTargetX - camera.position.x) * 0.04;
    camera.position.y += (parallaxTargetY - camera.position.y) * 0.04;
    camera.position.z = parallaxCameraBaseZ;
    camera.lookAt(0, 0, 0);

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

    // Multiply by 0.0025 instead of 0.005 to halve the speed for both idle and busy
    shapeTime += 0.0025 * currentShapeSpeed * (PREFERS_REDUCED_MOTION ? 0.3 : 1.0);

    // Audio reactivity: inflate the morph amplitude subtly during TTS
    // so the sphere visibly resonates with speech. Capped at +30% so it
    // stays subtle — chat overlay must remain readable.
    const morphAmp = 1.5 * (1.0 + audioLevel * 0.3);

    // Structural evolution: start a slow migration for one node every
    // few seconds (sooner when the agent is active). The node's HOME
    // position glides to a fresh point in the volume over ~6s, so its
    // proximity links dissolve and re-form — the graph visibly rewires
    // itself instead of orbiting a fixed skeleton.
    migrateCooldown -= (1 / 60) * (1.0 + activity * 3.0);
    if (migrateCooldown <= 0) {
        migrateCooldown = 3.5 + Math.random() * 3.0;
        let migrating = 0;
        for (let i = 0; i < NODE_COUNT; i++) {
            if (basePositions[i]._mig) migrating++;
        }
        if (migrating < MAX_CONCURRENT_MIGRATIONS) {
            const bp = basePositions[Math.floor(Math.random() * NODE_COUNT)];
            if (!bp._mig) {
                const u = Math.random(), v = Math.random();
                const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
                const r = 2.0 * Math.cbrt(Math.random());
                bp._mig = {
                    sx: bp.x, sy: bp.y, sz: bp.z,
                    tx: r * Math.sin(ph) * Math.cos(th),
                    ty: r * Math.sin(ph) * Math.sin(th),
                    tz: r * Math.cos(ph),
                    p: 0,
                };
            }
        }
    }
    for (let i = 0; i < NODE_COUNT; i++) {
        const m = basePositions[i]._mig;
        if (!m) continue;
        m.p += (1 / 60) / 6.0;   // ~6s per migration
        if (m.p >= 1) {
            basePositions[i].x = m.tx;
            basePositions[i].y = m.ty;
            basePositions[i].z = m.tz;
            basePositions[i]._mig = null;
        } else {
            const e = m.p * m.p * (3 - 2 * m.p);   // smoothstep ease
            basePositions[i].x = m.sx + (m.tx - m.sx) * e;
            basePositions[i].y = m.sy + (m.ty - m.sy) * e;
            basePositions[i].z = m.sz + (m.tz - m.sz) * e;
        }
    }

    for (let i = 0; i < NODE_COUNT; i++) {
        const bp = basePositions[i];

        // Nodes morph with independent symmetric phases across the entire screen
        const t1 = shapeTime * bp.speed + bp.phaseX;
        const t2 = shapeTime * bp.speed * 0.73 + bp.phaseY;
        const t3 = shapeTime * bp.speed * 1.37 + bp.phaseZ;

        const dx = (Math.sin(t1) + Math.sin(t2 * 1.4) * 0.5) * morphAmp;
        const dy = (Math.cos(t2) + Math.cos(t3 * 1.1) * 0.5) * morphAmp;
        const dz = (Math.sin(t3) + Math.cos(t1 * 0.9) * 0.5) * morphAmp;

        currentPositions[i].set(bp.x + dx, bp.y + dy, bp.z + dz);
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

    for (let i = 0; i < NODE_COUNT; i++) {
        for (let j = i + 1; j < NODE_COUNT; j++) {
            const distSq = currentPositions[i].distanceToSquared(currentPositions[j]);
            if (distSq < PROXIMITY_SQ) {
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

        const s = nodeScales[i];
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

    // Advance the palette wheel: ~50s per full hue cycle at idle,
    // tightening toward ~15s when the agent is busy. Slowed under
    // reduced-motion like every other animation.
    hueDrift += (0.00035 + activity * 0.0009)
        * (PREFERS_REDUCED_MOTION ? 0.3 : 1.0);

    const nUniforms = instancedMesh.material.uniforms;
    nUniforms.uWorkingState.value = workingState;
    nUniforms.uErrorState.value = errorState;
    nUniforms.uPulseT.value = pulseT;
    nUniforms.uAudioLevel.value = audioLevel;
    nUniforms.uAccentStrength.value = accentStrength;
    nUniforms.uHueDrift.value = hueDrift;

    const lUniforms = lineMaterial.uniforms;
    lUniforms.uTime.value = time;
    lUniforms.uWorkingState.value = workingState;
    lUniforms.uErrorState.value = errorState;
    lUniforms.uPulseT.value = pulseT;
    lUniforms.uAccentStrength.value = accentStrength;
    lUniforms.uHueDrift.value = hueDrift;

    // Bloom breathes with the envelope. The old formula added
    // errorState * 2.2 (a 3x glow flashbang on any error line) and
    // pulseT * 0.5 per event — together the main source of "flashing".
    bloomPass.strength = (1.15 + workingState * 0.3 + activity * 0.35
        + errorState * 0.5) * BLOOM_SCALE;

    composer.render();
}
