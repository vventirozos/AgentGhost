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
export const COLORS = {
    background: new THREE.Color('#000000'),

    nodeBase: new THREE.Color('#1a0000'),    // Very Dark Red
    nodeActive: new THREE.Color('#005eff'),  // Electric Blue
    nodeError: new THREE.Color('#ff00ee'),   // Magenta

    lineBase: new THREE.Color('#300000'),    // Very Dark Red (lines)
    lineActive: new THREE.Color('#450000'),  // Dark Red (lines)
    lineError: new THREE.Color('#00fff2'),   // Cyan
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

// Pulse (shockwave) state. triggerPulse()/triggerSmallPulse() raise
// pulseT to some peak and then decay toward 0 each frame. The line
// shader uses it to amplify the traveling pulse; the node shader
// boosts brightness.
let pulseT = 0.0;
let pulseDecay = 0.0;  // per-frame decay rate (tuned in setPulse)

// Accent tint — a transient color nudge driven by the log icon. Decays
// on its own so the color lingers slightly longer than the shockwave
// for a softer "afterimage" feel.
const accentColor = new THREE.Color(0x00f3ff);
let accentStrength = 0.0;
const ACCENT_DECAY = 0.015;  // ~1.2s to fall from 1.0 to 0.0 at 60fps

// TTS-driven audio level (0..1). Wired by setAudioLevel() from app.js
// when the TTS engine is active; multiplies node jitter subtly so the
// sphere "breathes with the voice."
let audioLevel = 0.0;

// Parallax — camera drift toward cursor / device tilt. Never more than
// ±PARALLAX_RANGE units so the chat never visually shifts.
const PARALLAX_RANGE = 0.15;
let parallaxTargetX = 0.0;
let parallaxTargetY = 0.0;
let parallaxCameraBaseZ = 5.0;

const basePositions = [];
const currentPositions = new Array(NODE_COUNT);
const nodeScales = new Float32Array(NODE_COUNT).fill(1.0);

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

const nodeVertexShader = `
uniform float uWorkingState;
uniform float uErrorState;
uniform float uPulseT;
uniform float uAudioLevel;
uniform vec3 uBaseColor;
uniform vec3 uActiveColor;
uniform vec3 uErrorColor;
uniform vec3 uAccentColor;
uniform float uAccentStrength;

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

    // Active color drifts electric-blue <-> cyan across the body so the
    // network isn't one flat hue.
    float hueShift = sin(instancePos.y * 1.5 + instancePos.z * 1.5) * 0.5 + 0.5;
    vec3 activeCol = mix(uActiveColor, vec3(0.0, 0.95, 1.0), hueShift);

    // Smoothly shift between shades based on position/time to give it life
    float colorMix = sin(instancePos.x * 2.0 + instancePos.y * 2.0 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor, activeCol, colorMix);

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
varying float vLightPass;
varying float vLineDepth;

void main() {
    vLightPass = aLightPass;
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
uniform vec3 uBaseColor;
uniform vec3 uActiveColor;
uniform vec3 uErrorColor;
uniform vec3 uAccentColor;
uniform float uAccentStrength;

varying float vLightPass;
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

    // Add same color blend along lines
    float colorMix = sin(vLightPass * 10.0 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor, uActiveColor, colorMix);

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
    }

    // Nodes (Instanced Mesh)
    const nodeGeom = new THREE.PlaneGeometry(0.12, 0.12);
    nodeMaterial = new THREE.ShaderMaterial({
        vertexShader: nodeVertexShader,
        fragmentShader: nodeFragmentShader,
        uniforms: {
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uAudioLevel: { value: 0.0 },
            uBaseColor: { value: COLORS.nodeBase },
            uActiveColor: { value: COLORS.nodeActive },
            uErrorColor: { value: COLORS.nodeError },
            uAccentColor: { value: accentColor },
            uAccentStrength: { value: 0.0 },
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

    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute('aLightPass', new THREE.BufferAttribute(lineUvs, 1));

    lineMaterial = new THREE.ShaderMaterial({
        vertexShader: lineVertexShader,
        fragmentShader: lineFragmentShader,
        uniforms: {
            uTime: { value: 0 },
            uWorkingState: { value: 0.0 },
            uErrorState: { value: 0.0 },
            uPulseT: { value: 0.0 },
            uBaseColor: { value: COLORS.lineBase },
            uActiveColor: { value: COLORS.lineActive },
            uErrorColor: { value: COLORS.lineError },
            uAccentColor: { value: accentColor },
            uAccentStrength: { value: 0.0 },
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

function _applyPulse(peak, durationMs, color) {
    pulseT = Math.max(pulseT, peak);
    // Decay so that pulseT reaches 0 after `durationMs`.
    // At 60fps that's durationMs/16.67 frames.
    pulseDecay = peak / Math.max(1, durationMs / 16.67);
    if (color) {
        try {
            accentColor.set(color);
        } catch (e) { /* bad color string — keep previous */ }
        // Accent sticks around a touch longer than the shockwave.
        accentStrength = Math.max(accentStrength, peak);
    }
}

export function updateSphereColor(colorHex) {
    try { accentColor.set(colorHex); } catch (e) { /* ignore */ }
}

// Spike clear-timeout is tracked so rapid repeat spikes each extend
// the error window to +2s from the latest call instead of letting the
// first timeout yank it down while later spikes are still in flight.
let _spikeClearTimeout;
export function triggerSpike() {
    targetErrorState = 1.0;
    if (_spikeClearTimeout) clearTimeout(_spikeClearTimeout);
    _spikeClearTimeout = setTimeout(() => {
        targetErrorState = 0.0;
        _spikeClearTimeout = null;
    }, 2000);
}

export function triggerNextColor() { /* retained no-op for compatibility */ }

export function triggerPulse(color) { _applyPulse(1.0, 700, color); }
export function triggerSmallPulse(color) { _applyPulse(0.45, 500, color); }

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
    errorState   += (targetErrorState   - errorState)   * 0.08;

    // Pulse / accent decay. Independent rates so the color lingers past
    // the shockwave for a softer afterimage.
    pulseT = Math.max(0, pulseT - pulseDecay);
    accentStrength = Math.max(0, accentStrength - ACCENT_DECAY);

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

    // Structure changes form slowly when idle, faster when busy
    let targetShapeSpeed = SPEEDS.idle + (workingState * (SPEEDS.busy - SPEEDS.idle));
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
    let lineIdx = 0;

    const connectionProbability = errorState > 0.5 ? 0.0 : 1.0;
    const connected = new Array(NODE_COUNT).fill(false);

    for (let i = 0; i < NODE_COUNT; i++) {
        for (let j = i + 1; j < NODE_COUNT; j++) {
            const distSq = currentPositions[i].distanceToSquared(currentPositions[j]);
            if (distSq < PROXIMITY_SQ) {
                if (connectionProbability > 0) {
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
                        lineIdx++;
                    }
                }
            }
        }
    }
    lineGeometry.attributes.position.needsUpdate = true;
    lineGeometry.attributes.aLightPass.needsUpdate = true;
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
    const nUniforms = instancedMesh.material.uniforms;
    nUniforms.uWorkingState.value = workingState;
    nUniforms.uErrorState.value = errorState;
    nUniforms.uPulseT.value = pulseT;
    nUniforms.uAudioLevel.value = audioLevel;
    nUniforms.uAccentStrength.value = accentStrength;

    const lUniforms = lineMaterial.uniforms;
    lUniforms.uTime.value = time;
    lUniforms.uWorkingState.value = workingState;
    lUniforms.uErrorState.value = errorState;
    lUniforms.uPulseT.value = pulseT;
    lUniforms.uAccentStrength.value = accentStrength;

    // Bloom tracks work + errors, and kicks briefly on each pulse so the
    // heartbeat reads as a glow swell. Mobile scale keeps the composite
    // pass cheap on A-series GPUs.
    bloomPass.strength = (1.15 + workingState * 0.45 + errorState * 2.2 + pulseT * 0.5) * BLOOM_SCALE;

    composer.render();
}
