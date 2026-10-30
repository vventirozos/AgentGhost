"""The web face's §4PR review fixes, EXECUTED (2026-10-17).

A reviewer measured nine defects in ``interface/static/matrix_graph.js``;
this module drives the REAL module under node (THREE stubbed, seeded RNG,
fake rAF — the harness of test_interface_face_motion_2026_09_20.py, with a
composer stub that sizes its buffers the way the vendored EffectComposer
does) and pins each fix by its behaviour, not its text:

  1  a form switch HARD-CUT the scene's rotation / position / scale and the
     camera's z and look target (the vortex's unbounded roll: up to 133° in
     one frame after ten minutes; a mid-turn switch snapped the camera
     5.0 → 1.3). Now the scene and camera ease over the node blend and land
     exactly on the new form's values (the camera never moves in vortex /
     tesseract at steady state).
  2  the composer's buffers were built at CSS size: a 2x display rendered at
     1x until the first resize.
  3  a DPR change (monitor drag, zoom) never reached the renderer, the
     composer or the motes' point size.
  4  the shaders read the ever-growing clock as float32 — phases stair-
     stepped on multi-day sessions (the ClockworkPi runs always-on). uTime
     is now wrapped at a period every shader frequency divides, and the
     fract() packet trains are CPU-side phases.
  5  reduced motion still got the full refute shudder and a ~4 Hz network-
     error blink.
  6  the empty form ran the O(n²) pass and the bloom composite every frame.
  7  the dead, wrong ``destroy()`` export (deleted).
  8  per-frame allocations (pinned for the Object3D, the measurable one).
  9  reversed-edge smoothstep (undefined in GLSL ES).

Every scenario is also run against the pre-fix copy (GHOST_FACE_GRAPH=<that
file>) as the negative control; each test here FAILED there.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GRAPH = Path(os.environ.get("GHOST_FACE_GRAPH") or (_ROOT / "interface" / "static" / "matrix_graph.js"))

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_THREE_STUB = r"""
globalThis.__o3d = 0;
export class Color {
  constructor(c) { this.r = 0; this.g = 0; this.b = 0; if (c !== undefined) this.set(c); }
  set(c) {
    if (typeof c === 'number') { this.r = ((c >> 16) & 255) / 255; this.g = ((c >> 8) & 255) / 255; this.b = (c & 255) / 255; }
    else if (typeof c === 'string') { const h = c.replace('#', ''); if (h.length === 6) { this.r = parseInt(h.slice(0, 2), 16) / 255; this.g = parseInt(h.slice(2, 4), 16) / 255; this.b = parseInt(h.slice(4, 6), 16) / 255; } }
    else if (c && typeof c === 'object') { this.r = c.r; this.g = c.g; this.b = c.b; }
    return this;
  }
  lerp(c, a) { this.r += (c.r - this.r) * a; this.g += (c.g - this.g) * a; this.b += (c.b - this.b) * a; return this; }
}
export class Vector2 { constructor(x = 0, y = 0) { this.x = x; this.y = y; } }
export class Vector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  clone() { return new Vector3(this.x, this.y, this.z); }
  length() { return Math.hypot(this.x, this.y, this.z); }
  multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  distanceToSquared(v) { const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z; return dx * dx + dy * dy + dz * dz; }
  lerpVectors(a, b, t) { this.x = a.x + (b.x - a.x) * t; this.y = a.y + (b.y - a.y) * t; this.z = a.z + (b.z - a.z) * t; return this; }
}
class Euler { constructor() { this.x = 0; this.y = 0; this.z = 0; } }
export class Object3D {
  constructor() { globalThis.__o3d++; this.position = new Vector3(); this.scale = new Vector3(1, 1, 1); this.rotation = new Euler(); this.matrix = { position: new Vector3(), scale: new Vector3(1, 1, 1) }; this.children = []; this.visible = true; }
  add(o) { this.children.push(o); }
  updateMatrix() { this.matrix.position.copy(this.position); this.matrix.scale.copy(this.scale); }
  lookAt(x, y, z) { this.lookTarget = new Vector3(x, y, z); }
}
export class Scene extends Object3D { constructor() { super(); this.background = null; globalThis.__scene = this; } }
export class PerspectiveCamera extends Object3D { constructor(fov, aspect, near, far) { super(); this.fov = fov; this.aspect = aspect; this.near = near; this.far = far; globalThis.__camera = this; } updateProjectionMatrix() {} }
export class WebGLRenderer {
  constructor() { this.domElement = { tagName: 'CANVAS' }; this.pr = 1; globalThis.__renderer = this; }
  setSize(w, h) { this.w = w; this.h = h; } setPixelRatio(p) { this.pr = p; } getPixelRatio() { return this.pr; } dispose() {}
}
export class WebGLRenderTarget {
  constructor(w = 1, h = 1) { this.width = w; this.height = h; }
  setSize(w, h) { this.width = w; this.height = h; }
  clone() { return new WebGLRenderTarget(this.width, this.height); }
}
export const HalfFloatType = 1, RGBAFormat = 2, SRGBColorSpace = 3, AdditiveBlending = 4;
export class BufferAttribute { constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; this.needsUpdate = false; } }
export class InstancedBufferAttribute extends BufferAttribute {}
export class BufferGeometry { constructor() { this.attributes = {}; this.drawRange = { start: 0, count: 0 }; } setAttribute(n, a) { this.attributes[n] = a; } setDrawRange(s, c) { this.drawRange = { start: s, count: c }; } dispose() {} }
export class PlaneGeometry extends BufferGeometry {}
export class ShaderMaterial { constructor(o) { Object.assign(this, o); } dispose() {} }
export class InstancedMesh extends Object3D {
  constructor(geometry, material, count) { super(); this.geometry = geometry; this.material = material; this.count = count; this.instanceMatrix = { needsUpdate: false }; this.captured = new Array(count); globalThis.__imesh = this; }
  setMatrixAt(i, m) { const c = this.captured[i] || (this.captured[i] = { p: new Vector3(), s: 0 }); c.p.copy(m.position); c.s = m.scale.x; }
}
export class LineSegments extends Object3D { constructor(g, m) { super(); this.geometry = g; this.material = m; globalThis.__lines = this; } }
export class Points extends Object3D { constructor(g, m) { super(); this.geometry = g; this.material = m; globalThis.__points = this; } }
"""
# Mirrors interface/static/vendor/three/jsm/postprocessing/EffectComposer.js
# sizing: _pixelRatio from the renderer at construction; a caller-built
# target's width/height become the composer's (CSS) size; setSize(css)
# re-allocates both buffers + every pass at css × _pixelRatio.
_COMPOSER_STUB = r"""
export class EffectComposer {
  constructor(renderer, renderTarget) {
    this.renderer = renderer; this._pixelRatio = renderer.getPixelRatio(); this.passes = []; this.renders = 0;
    if (renderTarget === undefined) { this._width = 1; this._height = 1; renderTarget = { width: 1, height: 1, setSize(w, h) { this.width = w; this.height = h; }, clone() { return { ...this }; } }; }
    else { this._width = renderTarget.width; this._height = renderTarget.height; }
    this.renderTarget1 = renderTarget; this.renderTarget2 = renderTarget.clone();
    globalThis.__composer = this;
  }
  addPass(p) { this.passes.push(p); if (p.setSize) p.setSize(this._width * this._pixelRatio, this._height * this._pixelRatio); }
  setSize(w, h) {
    this._width = w; this._height = h;
    const W = w * this._pixelRatio, H = h * this._pixelRatio;
    this.renderTarget1.setSize(W, H); this.renderTarget2.setSize(W, H);
    for (const p of this.passes) if (p.setSize) p.setSize(W, H);
  }
  setPixelRatio(pr) { this._pixelRatio = pr; this.setSize(this._width, this._height); }
  render() { this.renders++; }
}
"""
_RENDERPASS_STUB = "export class RenderPass { constructor() {} }\n"
_BLOOM_STUB = ("export class UnrealBloomPass { constructor(res, s) { this.strength = s; this.w = res.x; this.h = res.y; "
               "globalThis.__bloom = this; } setSize(w, h) { this.w = w; this.h = h; } }\n")

_HARNESS = r"""
const rafQueue = []; const timers = []; let nowMs = 0; const frameMs = 1000 / 60;
globalThis.requestAnimationFrame = (cb) => { rafQueue.push(cb); return rafQueue.length; };
globalThis.cancelAnimationFrame = () => { rafQueue.length = 0; };
const listeners = {};
globalThis.window = {
  matchMedia: (q) => ({ matches: q.includes('reduced') ? process.env.RM === '1' : false }),
  addEventListener(t, cb) { (listeners[t] = listeners[t] || []).push(cb); }, removeEventListener() {},
  devicePixelRatio: Number(process.env.DPR || 2), innerWidth: 1440, innerHeight: 900,
};
globalThis.document = {
  getElementById: () => ({ clientWidth: 1440, clientHeight: 900, appendChild() {}, removeChild() {} }),
  querySelector: () => null, addEventListener() {}, hidden: false,
};
globalThis.localStorage = { getItem: () => process.env.FACE_FORM || null, setItem() {} };
globalThis.setTimeout = (cb, ms) => { timers.push({ cb, at: nowMs + (ms || 0) }); return timers.length; };
globalThis.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].cb = null; };
let seed = 12345;
Math.random = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

export const face = await import('./matrix_graph.js');
face.init();
export const scene = globalThis.__scene, camera = globalThis.__camera, imesh = globalThis.__imesh, lines = globalThis.__lines;
export const fire = (t) => { for (const cb of (listeners[t] || [])) cb({}); };
export function tick(n = 1) {
  for (let k = 0; k < n; k++) {
    nowMs += frameMs;
    for (const t of timers) if (t.cb && t.at <= nowMs) { const cb = t.cb; t.cb = null; cb(); }
    const cbs = rafQueue.splice(0);
    for (const cb of cbs) cb(nowMs);
  }
}
export function snapshot() {
  const out = [];
  for (let i = 0; i < imesh.count; i++) { const c = imesh.captured[i]; out.push(c ? [c.p.x, c.p.y, c.p.z] : null); }
  return out;
}
export function view() {
  const T = camera.lookTarget || { x: 0, y: 0, z: 0 };
  return { rx: scene.rotation.x, ry: scene.rotation.y, rz: scene.rotation.z, px: scene.position.x, py: scene.position.y,
           s: scene.scale.x, cz: camera.position.z, lx: T.x, ly: T.y, lz: T.z };
}
export const emit = (o) => process.stdout.write(JSON.stringify(o));
"""

_SCENARIOS = {
    # 1 — scene + camera continuity across a switch, exact landing after it.
    "blend": r"""
import { face, tick, view, emit } from './harness.mjs';
const wrap = (a) => { const t = 2 * Math.PI; return ((a + Math.PI) % t + t) % t - Math.PI; };
const [from, to, warm, turn] = [process.env.A, process.env.B, Number(process.env.WARM), process.env.TURN === '1'];
face.setForm(from); tick(warm);
if (turn) { face.setUserTurn(true); face.setWorkingState(true); tick(600); }
let prev = view(); const maxD = { rot: 0, pos: 0, s: 0, cz: 0, look: 0 }; const firstD = {};
face.setForm(to);
for (let f = 0; f < 110; f++) {          // the blend is 1.4 s = 84 frames
  tick(1); const v = view();
  const d = { rot: Math.max(Math.abs(wrap(v.rx - prev.rx)), Math.abs(wrap(v.ry - prev.ry)), Math.abs(wrap(v.rz - prev.rz))),
              pos: Math.hypot(v.px - prev.px, v.py - prev.py), s: Math.abs(v.s - prev.s), cz: Math.abs(v.cz - prev.cz),
              look: Math.hypot(v.lx - prev.lx, v.ly - prev.ly, v.lz - prev.lz) };
  if (f === 0) Object.assign(firstD, d);
  for (const k in maxD) maxD[k] = Math.max(maxD[k], d[k]);
  prev = v;
}
// Steady state: the camera-static forms hold z EXACTLY, frame after frame,
// and the vortex roll is exactly its own law (no residue from the blend).
let camHeld = true;
for (let f = 0; f < 240; f++) { tick(1); if ((to === 'vortex' || to === 'tesseract') && camera_z() !== 5.0) camHeld = false; }
function camera_z() { return view().cz; }
const st = face.getDebugState(), v = view();
emit({ maxD, firstD, camHeld, camZ: v.cz, rz: v.rz, rx: v.rx,
       // the temperament's body (2026-10-30) rides on top by design: the law
       // holds once it is taken off (st.body: 0 on a face without it)
       vortexRz: st.vortexSpin * 0.22 + ((st.body || {}).rz || 0),
       vortexRx: 0.06 * Math.sin(st.clock * 0.050) + ((st.body || {}).rx || 0) });
""",
    # 2 / 3 — backing-store sizes at init and after DPR changes.
    "dpr": r"""
import { face, fire, emit } from './harness.mjs';
const R = globalThis.__renderer, C = globalThis.__composer, B = globalThis.__bloom, M = globalThis.__points.material;
const read = () => ({ pr: R.pr, rt1: C.renderTarget1.width, rt2: C.renderTarget2.width, bloom: B.w, uDpr: M.uniforms.uDpr.value });
const init = read();
window.devicePixelRatio = 1; fire('resize'); const one = read();
window.devicePixelRatio = 3; fire('resize'); const three = read();
emit({ init, one, three });
""",
    # 4 — the shader clock: every uTime use wraps cleanly; fract phases CPU-side.
    "clock": r"""
import { face, tick, imesh, lines, emit } from './harness.mjs';
tick(120);
const srcs = [imesh.material.vertexShader, imesh.material.fragmentShader, lines.material.vertexShader,
              lines.material.fragmentShader, globalThis.__points.material.vertexShader, globalThis.__points.material.fragmentShader];
const uses = [], bad = [];
for (const s of srcs) {
  const body = s.replace(/\/\/[^\n]*/g, '').replace(/uniform float uTime;/g, '');   // code only
  for (const m of body.matchAll(/\buTime\b/g)) {          // EVERY token, several per statement
    const rest = body.slice(m.index);
    const mm = /^uTime \* (uOrganic \* )?([0-9.]+)/.exec(rest);
    if (mm) uses.push({ f: Number(mm[2]), organic: !!mm[1] }); else bad.push(rest.slice(0, 60));
  }
  for (const m of body.matchAll(/fract\([^;]*uTime/g)) bad.push('fract:' + m[0].slice(0, 60));
}
const P = face.SHADER_TIME_PERIOD;
const turns = [];
for (const u of uses) for (const org of (u.organic ? [0.25, 1.0] : [1.0])) turns.push(u.f * org * P / (2 * Math.PI));
// Across a wrap: the phase the GPU gets equals the unwrapped phase.
let wrapErr = 0, maxWrapped = 0;
for (const k of [1, 7, 1000]) for (const eps of [-0.01, -1e-4, 0, 1e-4, 0.01]) {
  const t = k * P + eps; const w = face.shaderClock(t).uTime; maxWrapped = Math.max(maxWrapped, w);
  for (const u of uses) for (const org of (u.organic ? [0.25, 1.0] : [1.0])) {
    const f = u.f * org; wrapErr = Math.max(wrapErr, Math.abs(Math.sin(f * w + 0.3) - Math.sin(f * t + 0.3)));
  }
}
const big = face.shaderClock(3.0e7);   // ~3 years of uptime
const clock = face.getDebugState().clock, lu = lines.material.uniforms, nu = imesh.material.uniforms;
const fr = (x) => x - Math.floor(x);
emit({ uses: uses.length, bad, turns, wrapErr, maxWrapped, P, bigU: big.uTime, bigPkt: big.packet,
       live: { n: nu.uTime.value, l: lu.uTime.value, want: face.shaderClock(clock).uTime,
               pkt: lu.uPacketPhase && lu.uPacketPhase.value, wantPkt: fr(clock * 2.0),
               burst: lu.uBurstPhase && lu.uBurstPhase.value, wantBurst: fr(clock * 6.0) } });
""",
    # 5 — reduced motion: shudder amplitude (refute − stop on the descent
    # sheet, whose nodes no flinch reaches) and the network-error dim.
    "rm": r"""
import { face, tick, snapshot, imesh, emit } from './harness.mjs';
tick(240);
const mode = process.env.MODE;
const frames = [];
if (mode === 'net') {
  face.noteError('network');
  for (let f = 0; f < 120; f++) { tick(1); frames.push(imesh.material.uniforms.uFormDim.value); }
} else {
  face.noteVerdict(mode);
  for (let f = 0; f < 60; f++) { tick(1); frames.push(snapshot()); }
}
emit({ frames });
""",
    # 6 — the resting empty form skips the composite; a switch wakes it.
    "empty": r"""
import { face, tick, lines, emit } from './harness.mjs';
const C = globalThis.__composer;
tick(300);
const c0 = C.renders, t0 = face.getDebugState().clock; tick(120); const c1 = C.renders, t1 = face.getDebugState().clock;
face.setForm('cube'); tick(3); const c2 = C.renders;
tick(240); const linesAfter = lines.geometry.drawRange.count / 2;
emit({ restRenders: c1 - c0, clockMoved: t1 > t0, wokeRenders: c2 - c1, linesAfter, form: face.getForm() });
""",
    # 7 / 8 / 9 — dead export gone, no per-frame Object3D, portable smoothstep.
    "misc": r"""
import { face, tick, imesh, lines, emit } from './harness.mjs';
const o0 = globalThis.__o3d; tick(60); const o1 = globalThis.__o3d;
const srcs = [imesh.material.vertexShader, imesh.material.fragmentShader, lines.material.vertexShader,
              lines.material.fragmentShader, globalThis.__points.material.vertexShader];
const reversed = [];
for (const s of srcs.map((x) => x.replace(/\/\/[^\n]*/g, ''))) for (const m of s.matchAll(/smoothstep\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,/g)) {
  const a = Number(m[1]), b = Number(m[2]);
  if (Number.isFinite(a) && Number.isFinite(b) ? a >= b : (Number.isFinite(b) && !Number.isFinite(a))) reversed.push(m[0]);
}
emit({ hasDestroy: typeof face.destroy !== 'undefined', perFrameObject3D: (o1 - o0) / 60, reversed });
""",
}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("face4pr")
    three = root / "node_modules" / "three"
    (three / "addons" / "postprocessing").mkdir(parents=True)
    (three / "package.json").write_text('{ "name": "three", "type": "module", "main": "index.js" }\n')
    (three / "index.js").write_text(_THREE_STUB)
    (three / "addons" / "postprocessing" / "EffectComposer.js").write_text(_COMPOSER_STUB)
    (three / "addons" / "postprocessing" / "RenderPass.js").write_text(_RENDERPASS_STUB)
    (three / "addons" / "postprocessing" / "UnrealBloomPass.js").write_text(_BLOOM_STUB)
    (root / "package.json").write_text('{ "type": "module" }\n')
    (root / "harness.mjs").write_text(_HARNESS)
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src)
    shutil.copyfile(_GRAPH, root / "matrix_graph.js")
    shutil.copyfile(_GRAPH.parent / "temperament.js", root / "temperament.js")
    return root


def _run(root: Path, scenario: str, **env) -> dict:
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    proc = subprocess.run(["node", str(root / f"{scenario}.mjs")], capture_output=True,
                          text=True, timeout=240, env=e, cwd=str(root))
    if proc.returncode != 0:
        raise RuntimeError(f"node failed ({scenario}): {proc.stderr.strip()[:2000]}")
    return json.loads(proc.stdout)


# ═══ 1 — the scene and the camera blend; steady state is exact ═══════════

# (from, to, warm-up frames, mid-turn?). 3600 frames = one minute: the
# vortex roll is then ~1.65 rad — the cut the old code made in one frame.
_SWITCHES = [
    ("vortex", "cube", 3600, False),
    ("tesseract", "descent", 3600, False),
    ("tesseract", "cube", 120, True),      # old: camera 5.0 → 2.2 in one frame
    ("vortex", "descent", 120, True),      # old: 5.0 → 1.3
    ("cube", "vortex", 120, True),         # old: 2.2 → 5.0
    ("descent", "tesseract", 120, True),
]


@pytest.mark.parametrize("a,b,warm,turn", _SWITCHES, ids=[f"{a}->{b}{'-turn' if t else ''}" for a, b, _, t in _SWITCHES])
def test_form_switch_moves_scene_and_camera_continuously(lab, a, b, warm, turn):
    r = _run(lab, "blend", A=a, B=b, WARM=warm, TURN=int(turn))
    d = r["maxD"]
    # Per-frame bounds: the blend's steepest frame (smoothstep slope 1.5 over
    # 84 frames) of the largest legitimate delta (π in angle, 3.7 in z) plus
    # each form's own motion — an order of magnitude under the old cuts
    # (2.3 rad, 3.7 units).
    assert d["rot"] < 0.08, r
    assert d["cz"] < 0.1, r
    assert d["look"] < 0.12, r
    assert d["pos"] < 0.02 and d["s"] < 0.02, r
    # The first frame after the switch barely moves (smoothstep starts flat).
    assert r["firstD"]["rot"] < 0.01 and r["firstD"]["cz"] < 0.01, r


@pytest.mark.parametrize("a,b,warm,turn", [s for s in _SWITCHES if s[1] in ("vortex", "tesseract")],
                         ids=lambda v: str(v))
def test_camera_static_forms_hold_the_camera_exactly_after_the_blend(lab, a, b, warm, turn):
    r = _run(lab, "blend", A=a, B=b, WARM=warm, TURN=int(turn))
    assert r["camHeld"] and r["camZ"] == 5.0, r
    if b == "vortex":
        # The roll is the form's own law, bit for bit — no 2π residue.
        assert abs(r["rz"] - r["vortexRz"]) < 1e-9 and abs(r["rx"] - r["vortexRx"]) < 1e-9, r


# ═══ 2 / 3 — backing-store resolution and DPR changes ═══════════════════

def test_composer_buffers_are_built_at_device_resolution(lab):
    r = _run(lab, "dpr", DPR=2)
    i = r["init"]
    assert i["pr"] == 2
    # 1440 CSS px × DPR 2. Old: the target was built at 1440 and the
    # composer took that as its size until the first resize.
    assert i["rt1"] == 2880 and i["rt2"] == 2880, r
    assert i["bloom"] == 2880 and i["uDpr"] == 2, r


def test_a_dpr_change_reaches_renderer_composer_and_motes_capped(lab):
    r = _run(lab, "dpr", DPR=2)
    assert r["one"] == {"pr": 1, "rt1": 1440, "rt2": 1440, "bloom": 1440, "uDpr": 1}, r
    # DPR 3 is capped at the desktop cap (2), exactly as at init.
    assert r["three"] == {"pr": 2, "rt1": 2880, "rt2": 2880, "bloom": 2880, "uDpr": 2}, r


# ═══ 4 — the shader clock ═══════════════════════════════════════════════

def test_shader_time_wraps_where_every_shader_frequency_completes_whole_turns(lab):
    r = _run(lab, "clock")
    assert r["bad"] == [], f"uTime used outside the `uTime * f` sine form: {r['bad']}"
    assert r["uses"] >= 10, r            # hueWave×2 (3 shaders), sizeBoost, colour breath ×2, motes ×3
    for n in r["turns"]:
        assert abs(n - round(n)) < 1e-6, f"a shader frequency does not divide the wrap period: {r['turns']}"
    # Continuity across the wrap: the GPU's phase IS the unwrapped phase.
    assert r["wrapErr"] < 1e-6, r
    # Bounded: float32 resolution stays ≤ 2^-10 for any uptime.
    assert r["maxWrapped"] < r["P"] <= 16384 and 0 <= r["bigU"] < r["P"], r
    assert 0 <= r["bigPkt"] < 1, r


def test_live_uniforms_carry_the_wrapped_clock_and_the_cpu_packet_phases(lab):
    live = _run(lab, "clock")["live"]
    assert live["n"] == live["want"] and live["l"] == live["want"], live
    assert live["pkt"] == live["wantPkt"] and live["burst"] == live["wantBurst"], live


# ═══ 5 — reduced motion ═════════════════════════════════════════════════

def test_reduced_motion_damps_the_refute_shudder(lab):
    ref = _run(lab, "rm", RM=1, FACE_FORM="descent", MODE="refute")["frames"]
    stop = _run(lab, "rm", RM=1, FACE_FORM="descent", MODE="stop")["frames"]
    worst = 0.0
    for fa, fb in zip(ref, stop):
        for pa, pb in zip(fa, fb):
            if pa is None or pb is None or pa[0] > 9000 or pb[0] > 9000:
                continue
            worst = max(worst, abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))
    # TUNE.shudder 0.035 × CALM 0.35 ≈ 0.012 (old: the full 0.035).
    assert 0.002 < worst < 0.015, worst


def test_reduced_motion_network_error_is_a_steady_dim_not_a_blink(lab):
    dims = _run(lab, "rm", RM=1, MODE="net")["frames"]
    assert min(dims) < 0.55 * 0.9, "the network error must still dim"
    # Monotone recovery as the envelope decays — no on/off gate (old: a
    # ~4 Hz square wave between 0.55 and 0.22).
    assert all(b >= a - 1e-12 for a, b in zip(dims, dims[1:])), dims
    # Without reduced motion the operator's flicker is unchanged.
    full = _run(lab, "rm", RM=0, MODE="net")["frames"]
    flips = sum(1 for a, b in zip(full, full[1:]) if b < a - 0.05)
    assert flips >= 3, full


# ═══ 6 — the resting empty form does no frame work ══════════════════════

def test_empty_form_at_rest_skips_the_composite_and_wakes_on_switch(lab):
    r = _run(lab, "empty", FACE_FORM="empty")
    assert r["restRenders"] == 0, r
    assert r["clockMoved"], "the loop must keep running (envelopes, clock) while empty rests"
    assert r["wokeRenders"] == 3 and r["form"] == "cube", r
    assert r["linesAfter"] > 100, r


# ═══ 7 / 8 / 9 ══════════════════════════════════════════════════════════

def test_dead_destroy_export_is_gone(lab):
    assert _run(lab, "misc")["hasDestroy"] is False


def test_the_frame_allocates_no_object3d(lab):
    assert _run(lab, "misc")["perFrameObject3D"] == 0


def test_no_reversed_edge_smoothstep_in_any_shader(lab):
    assert _run(lab, "misc")["reversed"] == []
