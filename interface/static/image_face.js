import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

let scene, camera, renderer, composer;
let faceMesh;
let animationFrameId;
let time = 0;

let workingState = 0.0;
let targetWorkingState = 0.0;
let errorState = 0.0;
let targetErrorState = 0.0;
let pulseState = 0.0;
let targetPulseState = 0.0;

// Target rotation for random floating
let targetRotX = 0;
let targetRotY = 0;
let currentRotX = 0;
let currentRotY = 0;
let rotChangeTime = 0;

export function init() {
    const container = document.getElementById('sphere-container');
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x02050a);

    // Camera setup
    camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 10);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const renderTarget = new THREE.WebGLRenderTarget(container.clientWidth, container.clientHeight, {
        type: THREE.HalfFloatType, format: THREE.RGBAFormat
    });

    composer = new EffectComposer(renderer, renderTarget);
    const renderPass = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 1.2, 0.4, 0.1);
    composer.addPass(renderPass);
    composer.addPass(bloomPass);

    const textureLoader = new THREE.TextureLoader();
    const loadFace = (url) => {
        textureLoader.load(url, (texture) => {
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            createDisplacementFace(texture);
        }, undefined, (err) => {
            if (url === '/static/cyber_face.png') {
                loadFace('/static/cyber_face.jpg'); // Fallback to jpg
            } else {
                console.error("Could not load /static/cyber_face.png or .jpg");
            }
        });
    };

    loadFace('/static/cyber_face.png');

    window.addEventListener('resize', handleResize);
    animate();
}

function createDisplacementFace(texture) {
    const aspect = texture.image.width / texture.image.height;
    const height = 8.5;
    const width = height * aspect;

    // Subdivide the plane heavily (256x256) so vertices can be pushed smoothly
    const geometry = new THREE.PlaneGeometry(width, height, 256, 256);

    const vertexShader = `
        uniform sampler2D uMap;
        uniform float uTime;
        uniform float uWorking;
        uniform float uError;
        
        varying vec2 vUv;
        varying float vDepth;

        void main() {
            vUv = uv;
            vec4 texColor = texture2D(uMap, uv);
            
            // Calculate luminance to use as a makeshift depth map
            float luminance = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
            
            vec3 pos = position;
            
            // --- ENHANCED WORKING ANIMATIONS ---
            // Pulse extrusion further out rhythmically when working (reduced by 60%)
            float workPulse = (sin(uTime * 8.0) * 0.5 + 0.5) * uWorking * 0.4;
            float depthStr = 1.8 + (workPulse * 2.5); // Extrudes forward during work
            
            // Idle breathing effect
            float breathe = sin(uTime * 1.5 + uv.y * 5.0) * 0.05 * luminance;
            
            // Faster, more aggressive rippling data flow
            float ripple = 0.0;
            if (uWorking > 0.0) {
                ripple = sin(uv.y * 40.0 - uTime * 15.0) * 0.25 * luminance * uWorking;
            }

            // Glitch vertices heavily on error
            float glitch = 0.0;
            if (uError > 0.0) {
                glitch = (fract(sin(dot(uv, vec2(12.9898, 78.233)) + uTime * 10.0) * 43758.5453) * 2.0 - 1.0) * uError * 0.3;
            }
            
            pos.z += (luminance * depthStr) + breathe + ripple + glitch;
            
            // Curve the plane slightly so it feels volumetric like a screen/face
            float distCenter = length(uv - vec2(0.5));
            pos.z -= distCenter * 1.5;

            vDepth = luminance;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
    `;

    const fragmentShader = `
        uniform sampler2D uMap;
        uniform float uTime;
        uniform float uWorking;
        uniform float uError;
        uniform float uPulse;
        
        varying vec2 vUv;
        varying float vDepth;

        // Smoothly cycle through colors: Cyan -> Magenta -> Deep Blue -> Dark Red -> Purple 
        vec3 getColorCycle(float t) {
            float phase = mod(t, 5.0);
            vec3 c1 = vec3(0.0, 1.0, 1.0); // Cyan
            vec3 c2 = vec3(1.0, 0.0, 0.8); // Magenta
            vec3 c3 = vec3(0.2, 0.0, 1.0); // Deep Blue
            vec3 c4 = vec3(0.6, 0.0, 0.1); // Dark Red
            vec3 c5 = vec3(0.5, 0.0, 0.8); // Purple
            
            vec3 col = mix(c1, c2, smoothstep(0.0, 1.0, phase));
            col = mix(col, c3, smoothstep(1.0, 2.0, phase));
            col = mix(col, c4, smoothstep(2.0, 3.0, phase));
            col = mix(col, c5, smoothstep(3.0, 4.0, phase));
            col = mix(col, c1, smoothstep(4.0, 5.0, phase));
            return col;
        }

        void main() {
            vec4 texColor = texture2D(uMap, vUv);
            vec3 color = texColor.rgb;
            
            // --- ACTIVE STATE (Color Shifting & Glow) ---
            vec3 activeColor = getColorCycle(uTime * 1.5);
            
            // Fast scanning lines
            float workScan = sin(vUv.y * 120.0 - uTime * 15.0) * 0.5 + 0.5;
            float throb = sin(uTime * 8.0) * 0.5 + 0.5; // Syncs with 3D depth pulse
            
            // Tint the base image with the cycling color (glow reduced by 50%)
            vec3 tintedBase = mix(color, color * activeColor * 2.0, uWorking * 0.45);
            color = tintedBase;

            // Add intense scanning energy lines overlaid on the glowing parts (reduced by 50%)
            color += activeColor * (uWorking * 0.5) * workScan * vDepth * (1.0 + throb * 2.0);

            // Log Pulse State (White/Cyan flashes on new events)
            color += vec3(0.2, 0.8, 1.0) * uPulse * vDepth * 0.8;

            // Error State: Glitch to Neon Red
            if (uError > 0.0) {
                float glitch = step(0.9, fract(sin(vUv.y * 50.0 + uTime * 20.0)));
                // Intense bright neon red instead of dark red
                color = mix(color, vec3(1.0, 0.1, 0.2) * 2.5, uError * vDepth + glitch * 0.8);
            }
            
            // Fade out the hard edges of the plane
            float edgeFadeX = smoothstep(0.0, 0.1, vUv.x) * smoothstep(1.0, 0.9, vUv.x);
            float edgeFadeY = smoothstep(0.0, 0.1, vUv.y) * smoothstep(1.0, 0.9, vUv.y);
            float alpha = edgeFadeX * edgeFadeY;

            // Mask absolute black background slightly so it merges with the 3D scene
            alpha *= smoothstep(0.01, 0.15, vDepth + 0.05);

            gl_FragColor = vec4(color, alpha);
        }
    `;

    const material = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: {
            uMap: { value: texture },
            uTime: { value: 0 },
            uWorking: { value: 0 },
            uError: { value: 0 },
            uPulse: { value: 0 }
        },
        transparent: true,
        depthWrite: false
    });

    faceMesh = new THREE.Mesh(geometry, material);
    scene.add(faceMesh);
}

// Removed mouse tracking
function onDocumentMouseMove(event) {
    // No longer tracking mouse
}

function handleResize() {
    const container = document.getElementById('sphere-container');
    if (!container || !camera || !renderer) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    if (composer) composer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    animationFrameId = requestAnimationFrame(animate);
    time += 0.015;

    // Smooth state transitions
    workingState += (targetWorkingState - workingState) * 0.1;
    errorState += (targetErrorState - errorState) * 0.1;
    pulseState += (targetPulseState - pulseState) * 0.1;
    if (targetPulseState > 0) targetPulseState -= 0.05;
    if (targetPulseState < 0) targetPulseState = 0;

    // Random slow floating rotation
    if (time > rotChangeTime) {
        // Pick a new random target rotation every 2-5 seconds
        targetRotX = (Math.random() - 0.5) * 0.3; // Slight tilt up/down
        targetRotY = (Math.random() - 0.5) * 0.4; // Slight tilt left/right
        rotChangeTime = time + 2.0 + Math.random() * 3.0;
    }

    // Smoothly interpolate to new rotation
    currentRotX += (targetRotX - currentRotX) * 0.005;
    currentRotY += (targetRotY - currentRotY) * 0.005;

    if (faceMesh) {
        // Apply random floating rotation
        faceMesh.rotation.x = currentRotX;
        faceMesh.rotation.y = currentRotY;

        // Very slow floating motion up/down
        faceMesh.position.y = Math.sin(time * 0.5) * 0.1;

        // Pass uniforms
        faceMesh.material.uniforms.uTime.value = time;
        faceMesh.material.uniforms.uWorking.value = workingState;
        faceMesh.material.uniforms.uError.value = errorState;
        faceMesh.material.uniforms.uPulse.value = pulseState;
    }

    if (composer && composer.passes[1]) {
        // Sync the bloom pulse with the heartbeat wave from the shader (reduced by 80%)
        let heartbeatBloom = 0.0;
        if (workingState > 0.01) {
            heartbeatBloom = (Math.sin(time * 8.0) * 0.5 + 0.5) * 0.4 * workingState;
        }

        composer.passes[1].strength = 1.2 + (workingState * 0.2) + heartbeatBloom + (pulseState * 1.0) + (errorState * 2.5);
    }

    composer.render();
}

export function destroy() {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    window.removeEventListener('resize', handleResize);
    document.removeEventListener('mousemove', onDocumentMouseMove);
    const container = document.getElementById('sphere-container');
    if (container && renderer && renderer.domElement) container.removeChild(renderer.domElement);
    if (renderer) renderer.dispose();
}

export function setWorkingState(isWorking) { targetWorkingState = isWorking ? 1.0 : 0.0; }
export function setWaitingState(isWaiting) { targetWorkingState = isWaiting ? 0.5 : (targetWorkingState > 0.5 ? targetWorkingState : 0.0); }
export function triggerSpike() { targetErrorState = 1.0; targetPulseState = 1.5; setTimeout(() => { targetErrorState = 0.0; }, 1000); }
export function triggerPulse(colorHex) { targetPulseState = 1.0; }
export function triggerSmallPulse() { targetPulseState = 0.5; }
export function updateSphereColor() { }
export function triggerNextColor() { }
