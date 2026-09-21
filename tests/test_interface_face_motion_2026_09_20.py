"""The face's motion contract, EXECUTED (2026-09-20).

Operator: "the faces of the webUI are sometimes moving erratic and the
'zoom to action' doesn't always work right, investigate all faces
(especially lattice) and fix any defects."

Every earlier face pin is a text pin or a builder run; the render loop
itself — where the erratic motion lived — had never been executed under
test. This module runs the REAL ``animate()`` from ``matrix_graph.js``
under node with a minimal THREE stub, drives frames deterministically
(seeded RNG, fake rAF with timestamps), reads every node's position back
through ``InstancedMesh.setMatrixAt`` and projects each form's "action"
through the camera. It pins the seven defects that were measured and
fixed, and — because a harness that cannot see a defect proves nothing —
runs the key scenarios against the pre-fix backup
(``matrix_graph.js.bak-20260920-preframetime``) and asserts they FAIL there.
There are deliberately NO source-text pins here (R4): every mechanism below
is pinned by driving the loop, and the §4JA mutation battery (18 whole-file
mutants, ``GHOST_FACE_GRAPH=<mutant>``) killed each one by an executed pin.

Roster note (same day, later): the operator then deleted the embedding
form and the old monolith ``cube``, and renamed ``lattice`` to ``cube`` (now
the default). The shipped-file pins below use the new names; the negative
controls run the frozen pre-fix backup, whose names are the old ones.

Defects (all measured with this harness before the fix):
  D1  frame-locked stepping — every increment assumed 60fps, so a 120Hz
      display ran the face at 2× and a throttled 30fps tab at ½× (the
      dive-out took 2× / ½× the time). Now dt-scaled (``dtF``).
  D2  ``time × (rate + k·drive)`` phases — lattice runners, cube churn and
      heart spin jumped by ``time × Δrate`` on every drive change, worse the
      longer the page was open (lattice runners crossing the grid on log
      lines; cube 17× more violent after ten minutes). Now integrated.
  D3  lattice dive — plunged to the global 1.3 into the grid's interior
      with the attention kernel projecting OFF SCREEN. Now focus-translated
      onto the kernel with a partial dive (``LATTICE_DIVE_Z``).
  D4  (old monolith cube, since removed) re-arm teleport — a follow-up
      message while the dive lingered re-picked the anchor and jumped the
      whole cube ~1 unit in a frame.
  D5  descent trail — sampled every 3rd frame; the tail stuttered at 20Hz.
  D6  (embedding, since removed) tail — lagged the head and SNAPPED onto
      the cluster at every arrival (the flight wrapped before the tail
      landed).
  D7  idle twitch — fired ~5× more often than its "6–14s" comment (a
      seconds→clock-units slip) and popped a neighbourhood in one frame.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
# GHOST_FACE_GRAPH points the executed pins at a MUTANT copy (mutation
# batteries never touch the deployed file); unset = the shipped face.
_GRAPH = Path(os.environ.get("GHOST_FACE_GRAPH") or (_STATIC / "matrix_graph.js"))
_BACKUP = _STATIC / "matrix_graph.js.bak-20260920-preframetime"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

# ── THREE stub: exactly the surface matrix_graph.js touches ────────────────
_THREE_STUB = r"""
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
  constructor() { this.position = new Vector3(); this.scale = new Vector3(1, 1, 1); this.rotation = new Euler(); this.matrix = { position: new Vector3(), scale: new Vector3(1, 1, 1) }; this.children = []; this.visible = true; }
  add(o) { this.children.push(o); }
  updateMatrix() { this.matrix.position.copy(this.position); this.matrix.scale.copy(this.scale); }
  lookAt(x, y, z) { this.lookTarget = new Vector3(x, y, z); }
}
export class Scene extends Object3D { constructor() { super(); this.background = null; globalThis.__scene = this; } }
export class PerspectiveCamera extends Object3D { constructor(fov, aspect, near, far) { super(); this.fov = fov; this.aspect = aspect; this.near = near; this.far = far; globalThis.__camera = this; } updateProjectionMatrix() {} }
export class WebGLRenderer { constructor() { this.domElement = { tagName: 'CANVAS' }; } setSize() {} setPixelRatio() {} dispose() {} }
export class WebGLRenderTarget { constructor() {} }
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
export class Points extends Object3D { constructor(g, m) { super(); this.geometry = g; this.material = m; } }
"""
_COMPOSER_STUB = "export class EffectComposer { constructor() {} addPass() {} render() {} setSize() {} }\n"
_RENDERPASS_STUB = "export class RenderPass { constructor() {} }\n"
_BLOOM_STUB = "export class UnrealBloomPass { constructor(res, s) { this.strength = s; } }\n"

# ── harness: browser globals, deterministic frames, measurement helpers ───
_HARNESS = r"""
const rafQueue = []; const timers = []; let nowMs = 0; let frameMs = 1000 / 60;
globalThis.requestAnimationFrame = (cb) => { rafQueue.push(cb); return rafQueue.length; };
globalThis.cancelAnimationFrame = () => { rafQueue.length = 0; };
globalThis.window = {
  matchMedia: (q) => ({ matches: false }),
  addEventListener() {}, removeEventListener() {},
  devicePixelRatio: 2, innerWidth: 1440, innerHeight: 900,
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
export function setHz(hz) { frameMs = 1000 / hz; }
export function advanceClock(ms) { nowMs += ms; }
export function tick(n = 1) {
  for (let k = 0; k < n; k++) {
    nowMs += frameMs;
    for (const t of timers) if (t.cb && t.at <= nowMs) { const cb = t.cb; t.cb = null; cb(); }
    const cbs = rafQueue.splice(0);
    for (const cb of cbs) cb(nowMs);
  }
}
export function snapshot() {
  const N = imesh.count, out = new Array(N);
  for (let i = 0; i < N; i++) { const c = imesh.captured[i]; out[i] = c ? { x: c.p.x, y: c.p.y, z: c.p.z, s: c.s } : null; }
  return out;
}
// Largest per-node move between two frames over nodes visible in both.
// Moves > 1.0 are the designed wraps (vortex respawn, lattice runners) and
// are counted separately.
export function displacement(a, b) {
  let max = 0, idx = -1, wraps = 0;
  for (let i = 0; i < a.length; i++) {
    const p = a[i], q = b[i];
    if (!p || !q || p.s < 0.05 || q.s < 0.05 || p.x > 9000 || q.x > 9000) continue;
    const d = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
    if (d > 1.0) { wraps++; continue; }
    if (d > max) { max = d; idx = i; }
  }
  return { max, idx, wraps };
}
export function run(frames, hooks = {}) {
  const stats = []; let prev = snapshot();
  for (let f = 0; f < frames; f++) {
    if (hooks[f]) hooks[f]();
    tick(1);
    const cur = snapshot();
    stats.push({ f, ...displacement(prev, cur) });
    prev = cur;
  }
  return stats;
}
export const maxOf = (stats) => stats.reduce((m, s) => Math.max(m, s.max), 0);
export const p95Of = (stats) => [...stats.map(s => s.max)].sort((a, b) => a - b)[Math.floor(stats.length * 0.95)];
function rotXYZ(p, e) {
  let { x, y, z } = p;
  let c = Math.cos(e.z), s = Math.sin(e.z); [x, y] = [x * c - y * s, x * s + y * c];
  c = Math.cos(e.y); s = Math.sin(e.y); [x, z] = [x * c + z * s, -x * s + z * c];
  c = Math.cos(e.x); s = Math.sin(e.x); [y, z] = [y * c - z * s, y * s + z * c];
  return { x, y, z };
}
export function toWorld(p) {
  const r = rotXYZ({ x: p.x * scene.scale.x, y: p.y * scene.scale.y, z: p.z * scene.scale.z }, scene.rotation);
  return { x: r.x + scene.position.x, y: r.y + scene.position.y, z: r.z + scene.position.z };
}
// Normalised screen coords (|x|,|y| <= 1 is on screen) through the camera.
export function project(w) {
  const C = camera.position, T = camera.lookTarget || { x: 0, y: 0, z: 0 };
  let fx = T.x - C.x, fy = T.y - C.y, fz = T.z - C.z; const fl = Math.hypot(fx, fy, fz); fx /= fl; fy /= fl; fz /= fl;
  let rx = -fz, ry = 0, rz = fx; const rl = Math.hypot(rx, ry, rz); rx /= rl; ry /= rl; rz /= rl;
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
  const dx = w.x - C.x, dy = w.y - C.y, dz = w.z - C.z;
  const x = dx * rx + dy * ry + dz * rz, y = dx * ux + dy * uy + dz * uz, z = dx * fx + dy * fy + dz * fz;
  const t = Math.tan((camera.fov / 2) * Math.PI / 180);
  return { sx: x / (z * t * camera.aspect), sy: y / (z * t), depth: z };
}
export function centroid(snap, idxs) {
  let x = 0, y = 0, z = 0, n = 0;
  for (const i of idxs) { const p = snap[i]; if (!p || p.x > 9000) continue; x += p.x; y += p.y; z += p.z; n++; }
  return n ? { x: x / n, y: y / n, z: z / n } : null;
}
export const range = (a, b) => [...Array(b - a).keys()].map(i => a + i);
export const emit = (o) => process.stdout.write(JSON.stringify(o));
"""

# Desktop node-index map of each form's "action" (NODE_COUNT 250):
#   cube (the renamed lattice; `lattice` in the backup): 216 sites +
#         20 runners, kernel = 236..249
#   descent: 15×15 sheet = 225, bead head = 225
_SCENARIOS = {
    "spikes": r"""
import { face, tick, run, maxOf, p95Of, emit } from './harness.mjs';
tick(240);
let twitches = 0, armed = true;
const idle = run(3600, Object.fromEntries([...Array(3600).keys()].map(f => [f, () => {
  const a = face.getDebugState().twitchAmp;
  if (a !== undefined) { if (armed && a > 0.5) { twitches++; armed = false; } else if (a < 0.2) armed = true; }
}])));
const idleSpikes = idle.filter(s => s.max > 0.03).length;
run(12000);                                   // ~3.3 min of uptime: the time-proportional phases grow
const hooks = {}; for (let f = 0; f < 480; f += 7) hooks[f] = () => face.noteActivity(0.16, '#3a1750');
hooks[0] = () => { face.setUserTurn(true); face.setWorkingState(true); face.noteActivity(0.16, '#3a1750'); };
const on = run(480, hooks);
// Six OFF(3s)→ON cycles: a follow-up message while the dive lingers. (The
// cube's re-pick is random; one cycle could land on the same anchor.)
let follow = [];
for (let k = 0; k < 6; k++) {
  run(180, { 0: () => { face.setUserTurn(false); face.setWorkingState(false); face.noteVerdict('stop'); } });
  follow = follow.concat(run(60, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } }));
}
emit({ form: face.getForm(), idleSpikes, twitches, turnMax: maxOf(on), turnP95: p95Of(on), followMax: maxOf(follow) });
""",
    "jerk": r"""
import { face, tick, snapshot, emit } from './harness.mjs';
tick(240);
let prev = snapshot(); const d = [];
for (let f = 0; f < 900; f++) { tick(1); const cur = snapshot();
  d.push(Math.hypot(cur[240].x - prev[240].x, cur[240].y - prev[240].y, cur[240].z - prev[240].z)); prev = cur; }
let jerk = 0; for (let i = 1; i < d.length; i++) jerk = Math.max(jerk, Math.abs(d[i] - d[i - 1]));
emit({ form: face.getForm(), tailJerk: jerk });
""",
    "framerate": r"""
import { face, tick, setHz, camera, scene, emit } from './harness.mjs';
const hz = Number(process.env.FACE_HZ); setHz(hz);
const F = (sec) => Math.round(sec * hz);
tick(F(4));
face.setUserTurn(true); face.setWorkingState(true);
for (let i = 0; i < F(6); i++) { if (i % F(0.25) === 0) face.noteActivity(0.16); tick(1); }
const a = face.getDebugState(); const camA = camera.position.z;
face.setUserTurn(false); face.setWorkingState(false); face.noteVerdict('stop');
tick(F(6));
const b = face.getDebugState();
emit({ hz, immA: a.immersion, camA, immB: b.immersion, camB: camera.position.z, actB: b.activity, rotB: scene.rotation.y, clockB: b.clock, flowB: b.cube2Flow || 0 });
""",
    "cube2": r"""
import { face, tick, run, snapshot, camera, lines, emit } from './harness.mjs';
face.setForm('cube2'); tick(240);
const N = snapshot().filter(p => p && p.x < 9000).length;
const idle = run(1800);
const f0 = face.getDebugState().cube2Flow;
const on = run(600, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } });
const f1 = face.getDebugState().cube2Flow;
const off = run(600, { 0: () => { face.setUserTurn(false); face.setWorkingState(false); face.noteVerdict('stop'); } });
const f2 = face.getDebugState().cube2Flow;
const sum = (st, k) => st.reduce((m, s) => m + s[k], 0);
const camMoved = [...idle, ...on, ...off].some(() => camera.position.z !== 5.0);
emit({ nodes: N, visibleWraps: sum(idle, 'wraps') + sum(on, 'wraps') + sum(off, 'wraps'),
       idleSpikes: idle.filter(s => s.max > 0.05).length,
       idleRate: (f0 - 0) / 30, turnRate: (f1 - f0) / 10, offRate: (f2 - f1) / 10,
       lines: lines.geometry.drawRange.count / 2, camZ: camera.position.z, camMoved });
""",
    "pause": r"""
import { face, tick, advanceClock, emit } from './harness.mjs';
tick(120);
const running = face.setAnimationPaused(true);
advanceClock(10000);                                  // ten seconds hidden
const resumed = face.setAnimationPaused(false);
tick(1);
emit({ running, resumed, dtFirst: face.getDebugState().dtF });
""",
    "focus": r"""
import { face, tick, run, snapshot, camera, scene, lines, toWorld, project, centroid, range, emit } from './harness.mjs';
const form = face.getForm();
tick(240);
run(480, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } });
const snap = snapshot();
const idxs = form === 'descent' ? [225] : form === 'cube2' ? range(96, 123) : range(236, 250);   // kernel (v3: after 12 shells' corners) / bead / embers
const c = centroid(snap, idxs);
const pr = project(toWorld(c));
emit({ form, sx: pr.sx, sy: pr.sy, depth: pr.depth, camZ: camera.position.z, immersion: face.getDebugState().immersion,
       scale: scene.scale.x, rotY: scene.rotation.y, lines: lines.geometry.drawRange.count / 2, flow: face.getDebugState().cube2Flow || 0 });
""",
}


def _write_harness(root: Path, graph: Path) -> None:
    three = root / "node_modules" / "three"
    (three / "addons" / "postprocessing").mkdir(parents=True, exist_ok=True)
    (three / "package.json").write_text('{ "name": "three", "type": "module", "main": "index.js" }\n')
    (three / "index.js").write_text(_THREE_STUB)
    (three / "addons" / "postprocessing" / "EffectComposer.js").write_text(_COMPOSER_STUB)
    (three / "addons" / "postprocessing" / "RenderPass.js").write_text(_RENDERPASS_STUB)
    (three / "addons" / "postprocessing" / "UnrealBloomPass.js").write_text(_BLOOM_STUB)
    (root / "package.json").write_text('{ "type": "module" }\n')
    (root / "harness.mjs").write_text(_HARNESS)
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src)
    shutil.copyfile(graph, root / "matrix_graph.js")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """Two harness roots: the shipped face and the pre-fix backup (the
    negative control that proves the instrument sees each defect)."""
    base = tmp_path_factory.mktemp("facelab")
    new, old = base / "new", base / "old"
    new.mkdir(); old.mkdir()
    _write_harness(new, _GRAPH)
    _write_harness(old, _BACKUP)
    return {"new": new, "old": old}


def _run(root: Path, scenario: str, form: str, **env) -> dict:
    import os
    e = dict(os.environ, FACE_FORM=form, **{k: str(v) for k, v in env.items()})
    proc = subprocess.run(["node", str(root / f"{scenario}.mjs")], capture_output=True,
                          text=True, timeout=240, env=e, cwd=str(root))
    if proc.returncode != 0:
        raise RuntimeError(f"node failed ({scenario}/{form}): {proc.stderr.strip()[:2000]}")
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════
# D2 / D4 / D7 — no phase jumps, no teleport, no pops (executed)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("form", ["cube", "vortex"])
def test_no_frame_jumps_after_uptime_and_on_follow_up_turn(lab, form):
    r = _run(lab["new"], "spikes", form)
    assert r["form"] == form
    # D2: a turn (with ambient log lines wobbling the drive) after ~3 min
    # of uptime moves no node more than a small fraction of a unit per
    # frame. Old lattice (today's cube): ~0.9 — runners crossing the grid.
    # (The vortex is exempt from the per-frame cap: its exponential
    # expansion legitimately carries OFF-SCREEN matter up to ~1 unit per
    # frame at full flow, unchanged by this fix; it is held to the idle and
    # follow-up checks only.)
    if form == "cube":
        assert r["turnMax"] < 0.05, r
        assert r["turnP95"] < 0.03, r
    # A follow-up message 3s after the reply — the dive still ~0.5 — must
    # not jump anything (old lattice: 0.90, the runner phase jump).
    assert r["followMax"] < (0.5 if form == "vortex" else 0.05), r
    assert r["idleSpikes"] == 0, r


def test_idle_is_pop_free_on_the_still_forms(lab):
    # D7: the cube's grid is otherwise ~still (0.003/frame); every ≥0.03
    # frame at idle was an idle-twitch pop. Old lattice: ~9 per minute.
    r = _run(lab["new"], "spikes", "cube")
    assert r["idleSpikes"] == 0, r
    # ...and the twitch fires on its documented 6–14s cadence (+ the ~0.8s
    # chance gate): 3–9 per minute, never the ~30 the 0.03× slip produced.
    assert 2 <= r["twitches"] <= 12, r


def test_descent_tail_glides(lab):
    # D5: consecutive-frame change of a tail node's speed. Old: 0.08
    # (the 20Hz sample shift); the bead's own rolling stays under 0.03.
    r = _run(lab["new"], "jerk", "descent")
    assert r["tailJerk"] < 0.04, r


# ═══════════════════════════════════════════════════════════════════════════
# D1 — frame-rate invariance (executed at 30 / 60 / 120 Hz)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("form", ["cube", "cube2", "descent"])
def test_same_wall_clock_state_at_30_60_and_120hz(lab, form):
    rs = {hz: _run(lab["new"], "framerate", form, FACE_HZ=hz) for hz in (30, 60, 120)}
    # immersion/camera/activity are ease-driven; the scene heading is
    # `time`-driven — together they cover both halves of the dt scaling.
    for key in ("immA", "camA", "immB", "camB", "actB", "rotB"):
        vals = [rs[hz][key] for hz in (30, 60, 120)]
        assert max(vals) - min(vals) < 0.03, (key, vals)
    clocks = [rs[hz]["clockB"] for hz in (30, 60, 120)]
    assert max(clocks) - min(clocks) < 0.05, clocks           # 16s ≈ 4.8 clock units everywhere
    flows = [rs[hz]["flowB"] for hz in (30, 60, 120)]
    assert max(flows) - min(flows) < 0.02, flows              # cube2's birth clock too


# ═══════════════════════════════════════════════════════════════════════════
# D3 — the dive lands on the action (executed + projected)
# ═══════════════════════════════════════════════════════════════════════════

def test_cube_dive_lands_on_the_kernel_partially(lab):
    r = _run(lab["new"], "focus", "cube")
    assert r["immersion"] > 0.95, r
    assert abs(r["sx"]) < 0.2 and abs(r["sy"]) < 0.2, r      # old: (1.24, 1.72) — off screen
    assert abs(r["camZ"] - 2.2) < 0.05, r                     # CUBE_DIVE_Z (2.7 → 2.2, "too far"), not the global 1.3
    assert r["depth"] > 1.6, r                                # outside the near-fade band (0.3–1.4)
    # the old monolith's treatment in full: gentled swell (0.18, not the
    # global 0.55) and the near-still heading (its own tumble is the motion)
    assert r["scale"] < 1.15, r
    assert abs(r["rotY"]) < 0.15, r


def test_descent_dive_keeps_the_bead_mid_low_in_frame(lab):
    # The hover lift puts the bead below the view axis on purpose; the
    # pitched look target keeps it out of the bottom fifth (old: −0.60,
    # under the composer on a phone).
    r = _run(lab["new"], "focus", "descent")
    assert abs(r["sx"]) < 0.25, r
    assert -0.35 < r["sy"] < -0.02, r      # level look: −0.37…−0.60 across seeds


def test_resume_after_a_pause_steps_one_frame(lab):
    # D1: a hidden tab pauses the loop; on resume the first frame must NOT
    # try to catch up ten seconds (dtF clamps at 3 — still a visible lurch
    # on every tab switch) — the stamp is dropped on pause.
    r = _run(lab["new"], "pause", "vortex")
    assert r["running"] is False and r["resumed"] is True, r
    assert r["dtFirst"] == 1.0, r


def test_cube2_infinite_cube_contract(lab):
    """The experiment's contract (2026-09-20): nested shells wrap from
    huge back to tiny ONLY while faded out (a visible node never jumps),
    the birth rate is the swallow (idle ≈ a shell per 4s, a user turn
    ≈ one per 0.7s, easing back after), the shells are drawn as explicit
    polylines (12 edges × 6 segments × shells) and the camera never moves."""
    r = _run(lab["new"], "cube2", "cube2")
    assert r["visibleWraps"] == 0, r
    assert r["idleSpikes"] == 0, r
    assert 0.018 < r["idleRate"] < 0.035, r
    assert r["turnRate"] > 3 * r["idleRate"], r
    assert r["offRate"] < r["turnRate"], r
    # v3 (2026-09-21): 24 line-only shells × (72 edge + 72 ruling) segments + ≤184 ties + 54 kernel edges
    assert 20 * 144 + 54 <= r["lines"] <= 24 * 144 + 23 * 8 + 54, r
    assert r["camZ"] == 5.0 and r["camMoved"] is False, r


def test_cube2_kernel_sits_on_the_view_axis(lab):
    r = _run(lab["new"], "focus", "cube2")
    assert abs(r["sx"]) < 0.15 and abs(r["sy"]) < 0.15, r
    assert abs(r["camZ"] - 5.0) < 1e-9, r
    assert r["lines"] >= 20 * 144 + 54, r


def test_vortex_camera_never_moves(lab):
    r = _run(lab["new"], "focus", "vortex")
    assert abs(r["camZ"] - 5.0) < 1e-9, r
    assert abs(r["sx"]) < 0.1 and abs(r["sy"]) < 0.1, r


# ═══════════════════════════════════════════════════════════════════════════
# Negative controls — the instrument SEES the defects in the pre-fix file
# ═══════════════════════════════════════════════════════════════════════════

def test_backup_of_prefix_face_exists():
    assert _BACKUP.exists(), "restore point for the 2026-09-20 motion fixes"


def test_control_old_lattice_runners_jumped_and_kernel_was_off_screen(lab):
    r = _run(lab["old"], "spikes", "lattice")
    assert r["turnMax"] > 0.5, r          # D2 seen
    assert r["followMax"] > 0.5, r
    f = _run(lab["old"], "focus", "lattice")
    assert max(abs(f["sx"]), abs(f["sy"])) > 1.0, f   # D3 seen
    assert abs(f["camZ"] - 1.3) < 0.05, f


def test_control_old_lattice_popped_at_idle(lab):
    r = _run(lab["old"], "spikes", "lattice")
    assert r["idleSpikes"] >= 3, r        # D7 seen (9/min under this seed; new file: 0)


def test_control_old_face_ran_at_the_display_rate(lab):
    rs = {hz: _run(lab["old"], "framerate", "lattice", FACE_HZ=hz) for hz in (30, 120)}
    assert abs(rs[30]["camB"] - rs[120]["camB"]) > 0.5, rs   # D1 seen (3.6 vs 5.0)


def test_control_old_descent_stuttered(lab):
    assert _run(lab["old"], "jerk", "descent")["tailJerk"] > 0.06     # D5 seen
