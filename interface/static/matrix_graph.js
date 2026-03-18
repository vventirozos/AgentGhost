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
    nodeActive: new THREE.Color('#005eff'),  // Dark Red
    nodeError: new THREE.Color('#ff00ee'),   // Electric Blue

    lineBase: new THREE.Color('#300000'),    // Very Dark Red (lines)
    lineActive: new THREE.Color('#450000'),  // Dark Red (lines)
    lineError: new THREE.Color('#00fff2'),   // Electric Blue
};
// ---------------------------

let scene, camera, renderer, composer, instancedMesh, linesMesh;
let lineGeometry, nodeMaterial, lineMaterial;
let time = 0;
let shapeTime = 0;
let currentShapeSpeed = SPEEDS.idle;
let animationFrameId;

let spikeStrength = 0.2;
let targetSpikeStrength = 0.2;

let errorState = 0.0;
let targetErrorState = 0.0;
let workingState = 0.0;
let targetWorkingState = 0.0;
let waitingState = 0.0;
let targetWaitingState = 0.0;

const nodeCount = 250;
const MAX_LINES = 10000;
const basePositions = [];
const currentPositions = new Array(nodeCount);
const nodeScales = new Float32Array(nodeCount).fill(1.0);

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
uniform vec3 uBaseColor;
uniform vec3 uActiveColor;
uniform vec3 uErrorColor;

varying vec3 vColor;
varying vec2 vUv;

void main() {
    vUv = uv;
    
    // Extract instance position
    vec3 instancePos = (instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    
    // Extract scale from instance matrix (assuming isotropic)
    float scale = length(vec3(instanceMatrix[0][0], instanceMatrix[0][1], instanceMatrix[0][2]));
    
    // Billboard logic: apply local face offset directly in view space
    vec4 mvPosition = modelViewMatrix * vec4(instancePos, 1.0);
    mvPosition.xy += position.xy * scale;
    
    gl_Position = projectionMatrix * mvPosition;
    
    // Smoothly shift between shades based on position/time to give it life
    float colorMix = sin(instancePos.x * 2.0 + instancePos.y * 2.0 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor, uActiveColor, colorMix);
    
    vec3 col = mix(mixCol, uErrorColor, uErrorState);
    vColor = col;
}
`;

const nodeFragmentShader = `
varying vec3 vColor;
varying vec2 vUv;
void main() {
    float d = distance(vUv, vec2(0.5)) * 2.0; // scaled 0 to 1
    if (d > 1.0) discard;
    
    // Soft quadratic falloff
    float intensity = pow(1.0 - d, 2.0) * 0.8;
    // Intense bright core
    float core = pow(1.0 - d, 8.0) * 1.5; 
    
    float alpha = intensity + core;
    gl_FragColor = vec4(vColor * alpha, alpha);
}
`;

const lineVertexShader = `
attribute float aLightPass;
varying float vLightPass;

void main() {
    vLightPass = aLightPass;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const lineFragmentShader = `
uniform float uTime;
uniform float uWorkingState;
uniform float uErrorState;
uniform vec3 uBaseColor;
uniform vec3 uActiveColor;
uniform vec3 uErrorColor;

varying float vLightPass;

void main() {
    float gradient = fract(vLightPass * 1.5 - uTime * 2.0);
    
    // Smooth sharp front tail for data packets traveling along lines
    float pulse = smoothstep(0.0, 0.5, gradient) * smoothstep(1.0, 0.9, gradient);
    
    // Add same color blend along lines
    float colorMix = sin(vLightPass * 10.0 + uWorkingState) * 0.5 + 0.5;
    vec3 mixCol = mix(uBaseColor, uActiveColor, colorMix);
    
    vec3 col = mix(mixCol, uErrorColor, uErrorState * 0.8);
    gl_FragColor = vec4(col, mix(0.4 + uWorkingState * 0.2, 1.0, pulse));
}
`;

export function init() {
    basePositions.length = 0;
    const container = document.getElementById('sphere-container');
    scene = new THREE.Scene();
    scene.background = COLORS.background;

    camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 5.0;

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    container.appendChild(renderer.domElement);

    const renderTarget = new THREE.WebGLRenderTarget(container.clientWidth, container.clientHeight, {
        type: THREE.HalfFloatType, format: THREE.RGBAFormat, colorSpace: THREE.SRGBColorSpace,
    });

    composer = new EffectComposer(renderer, renderTarget);
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight),
        1.5, 0.3, 0.1
    );

    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    // Initialize Base Positions
    for (let i = 0; i < nodeCount; i++) {
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
            uBaseColor: { value: COLORS.nodeBase },
            uActiveColor: { value: COLORS.nodeActive },
            uErrorColor: { value: COLORS.nodeError }
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });

    instancedMesh = new THREE.InstancedMesh(nodeGeom, nodeMaterial, nodeCount);
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
            uBaseColor: { value: COLORS.lineBase },
            uActiveColor: { value: COLORS.lineActive },
            uErrorColor: { value: COLORS.lineError }
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

function animate() {
    animationFrameId = requestAnimationFrame(animate);

    if (targetErrorState > 0.5) {
        targetSpikeStrength = 1.0;
    } else {
        targetSpikeStrength = 0.0;
    }

    let isWaking = (targetWorkingState > workingState + 0.01) || (targetWaitingState > waitingState + 0.01);
    let transitionSpeed = isWaking ? 0.05 : 0.02;

    spikeStrength += (targetSpikeStrength - spikeStrength) * transitionSpeed;
    workingState += (targetWorkingState - workingState) * transitionSpeed;
    waitingState += (targetWaitingState - waitingState) * transitionSpeed;
    errorState += (targetErrorState - errorState) * 0.08;

    // Accumulate time for lines at steady pace regardless of state
    time += 0.005;

    // Structure changes form slowly when idle, faster when busy
    let targetShapeSpeed = SPEEDS.idle + (workingState * (SPEEDS.busy - SPEEDS.idle));
    let speedDiff = targetShapeSpeed - currentShapeSpeed;
    if (Math.abs(speedDiff) > 0.001) {
        // Change by 2.0 over ~180 frames (3 seconds at 60fps) -> 0.011 per frame
        currentShapeSpeed += Math.sign(speedDiff) * Math.min(Math.abs(speedDiff), 0.011);
    }

    // Multiply by 0.0025 instead of 0.005 to halve the speed for both idle and busy
    shapeTime += 0.0025 * currentShapeSpeed;

    for (let i = 0; i < nodeCount; i++) {
        const bp = basePositions[i];

        // Nodes morph with independent symmetric phases across the entire screen
        const t1 = shapeTime * bp.speed + bp.phaseX;
        const t2 = shapeTime * bp.speed * 0.73 + bp.phaseY;
        const t3 = shapeTime * bp.speed * 1.37 + bp.phaseZ;

        const dx = (Math.sin(t1) + Math.sin(t2 * 1.4) * 0.5) * 1.5;
        const dy = (Math.cos(t2) + Math.cos(t3 * 1.1) * 0.5) * 1.5;
        const dz = (Math.sin(t3) + Math.cos(t1 * 0.9) * 0.5) * 1.5;

        currentPositions[i].set(bp.x + dx, bp.y + dy, bp.z + dz);
    }

    // 2. Update lines and track connectivity
    const linePosAttr = lineGeometry.attributes.position.array;
    const lineUvAttr = lineGeometry.attributes.aLightPass.array;
    let lineIdx = 0;

    const connectionProbability = errorState > 0.5 ? 0.0 : 1.0;
    const connected = new Array(nodeCount).fill(false);

    for (let i = 0; i < nodeCount; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
            const distSq = currentPositions[i].distanceToSquared(currentPositions[j]);
            if (distSq < 2.5) { // Increased distance to account for wider screen volume
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
    for (let i = 0; i < nodeCount; i++) {
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
    instancedMesh.material.uniforms.uWorkingState.value = workingState;
    instancedMesh.material.uniforms.uErrorState.value = errorState;

    lineMaterial.uniforms.uTime.value = time;
    lineMaterial.uniforms.uWorkingState.value = workingState;
    lineMaterial.uniforms.uErrorState.value = errorState;

    // Rotation should be very minimal and independent of state
    let rotSpeed = 0.0;
    // scene.rotation.y += rotSpeed;
    // scene.rotation.x += rotSpeed * 0.5;

    composer.passes[1].strength = 1.0 + errorState * 2.5;

    composer.render();
}

export function updateSphereColor(colorHex) { }
export function triggerSpike() { targetErrorState = 1.0; setTimeout(() => { targetErrorState = 0.0; }, 2000); }
export function triggerNextColor() { }

export function triggerPulse() { }
export function triggerSmallPulse() { }

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

let waitingTimeout;
export function setWaitingState(isWaiting) {
    if (isWaiting) {
        if (targetWaitingState < 0.5 && !waitingTimeout) {
            waitingTimeout = setTimeout(() => {
                targetWaitingState = 1.0;
                waitingTimeout = null;
            }, 500);
        }
    } else {
        if (waitingTimeout) {
            clearTimeout(waitingTimeout);
            waitingTimeout = null;
        }
        targetWaitingState = 0.0;
    }
}
