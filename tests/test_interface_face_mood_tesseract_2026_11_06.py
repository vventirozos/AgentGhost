"""mood tesseract: the corridor carrying the agent's mood, EXECUTED (2026-11-06).

Operator: "build it, make it as a copy of tesseract, call it mood teserract".
A prototype of three contrasting moods. Run on the real ``animate()`` under the
face harness. The first thing it must not do is change the plain tesseract or any
other form; the second is bring back the idle jitter fixed the same day (§4TF).
"""
from __future__ import annotations

import ast
import asyncio
import filecmp
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_interface_face_idle_calm_2026_11_06 as CALM
import interface.externals.clockwork_ghost.facestate as fs

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_BEFORE = _ROOT / "interface" / "face_backups" / "20261106-premoodonly"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_ROW = "│  **  12:00:00  ❌  +1.0s  sandbox exec   exit 1"

_SCENARIOS = {
    # An old pick, whether it was the saved one at start or arrives afterwards.
    "forms": r"""
import { tick, emit, face } from './common.mjs';
const opened = face.getForm(); const picked = {};
for (const old of ['cube', 'tesseract', 'vortex', 'descent', 'cube2']) {
  face.setForm('empty'); tick(1); face.setForm(old); tick(1); picked[old] = face.getForm(); }
emit({ opened, picked });
""",
    # Every node's path as one number, with the mood pushed as hard as it goes.
    "digest": r"""
import { tick, emit, face, snapshot, scene } from './common.mjs';
import { imesh, lines } from './harness.mjs';
const mood = process.env.MOOD || 'irritable';
if (process.env.LIVE) face.setMoodHue(mood, 3);
if (process.env.PREVIEW) face.previewMood(mood, 3);
tick(300); let digest = 0; const parts = {};
const add = (k, v) => { if (typeof v === 'number' && isFinite(v)) { digest += v; parts[k] = (parts[k] || 0) + v; } };
const events = { 200: () => face.noteToolCall(), 500: () => face.noteLogLine(process.env.ROW), 800: () => face.noteError(),
  1400: () => face.setUserTurn(true), 2200: () => face.noteRecall(), 2600: () => face.setUserTurn(false) };
for (let f = 0; f < 3600; f++) { if (events[f]) events[f](); tick(1);
  for (const p of snapshot()) if (p && p.x < 9000) add('nodes', p.x + 2 * p.y + 3 * p.z);
  add('scene', scene.position.y + scene.scale.x + scene.rotation.z);
  for (const [k, u] of Object.entries(imesh.material.uniforms)) { add('n.' + k, u.value); if (u.value && typeof u.value.r === 'number') add('n.' + k, u.value.r + 2 * u.value.g + 3 * u.value.b); }
  for (const [k, u] of Object.entries(lines.material.uniforms)) { add('l.' + k, u.value); if (u.value && typeof u.value.r === 'number') add('l.' + k, u.value.r + 2 * u.value.g + 3 * u.value.b); }
  add('drawn', lines.geometry.drawRange.count); add('bloom', globalThis.__bloom.strength);
  for (const [k, v] of Object.entries(face.getTemperament().out)) add('t.' + k, v); }
emit({ form: face.getForm(), digest, parts });
""",
    # What each mood does to the clocks, the light, the colour and the height.
    "looks": r"""
import { tick, emit, face, scene } from './common.mjs';
import { imesh } from './harness.mjs';
const out = {};
for (const [label, n] of JSON.parse(process.env.MOODS)) {
  face.previewMood(label, n); tick(1800);
  const a = face.getDebugState(); const spin0 = a.tesseract.kernelSpin, flow0 = a.cube2Flow, pulse0 = a.pulsePhase;
  const kb = a.tesseract.kernelBase, kn = a.tesseract.kernelN ** 3;
  let dim = 0, y = 0, bloom = 0, hue = 0, hueLo = 9, hueHi = -9, breath = 0, wob = 0, kr = 0, kSeed = 0, sSeed = 0; const F = 7200;
  for (let f = 0; f < F; f++) { tick(1); const U = imesh.material.uniforms; dim += U.uFormDim.value; y += scene.position.y; bloom += globalThis.__bloom.strength;
    const h = U.uHueDrift.value; hue += h; hueLo = Math.min(hueLo, h); hueHi = Math.max(hueHi, h);
    if (f % 10) continue; const D = face.getDebugState().tesseract; breath = Math.max(breath, ...D.breaths); wob += D.wob / (F / 10);
    const c = imesh.captured, m = [0, 0, 0]; for (let i = kb; i < kb + kn; i++) { m[0] += c[i].p.x / kn; m[1] += c[i].p.y / kn; m[2] += c[i].p.z / kn; }
    let sp = 0; for (let i = kb; i < kb + kn; i++) sp = Math.max(sp, Math.hypot(c[i].p.x - m[0], c[i].p.y - m[1], c[i].p.z - m[2])); kr += sp / (F / 10);
    const A = imesh.geometry.attributes.aSeed.array; for (let i = kb; i < kb + kn; i++) kSeed += A[i] / kn / (F / 10);
    let ss = 0, sn = 0; for (let i = kb + kn; i < A.length; i++) { ss += A[i]; sn++; } sSeed += (sn ? ss / sn : 0) / (F / 10); }
  const b = face.getDebugState(); const s = b.tesseract.seeds;
  let kick = 0; for (let k = 0; k < 6; k++) { tick(1500); face.noteToolCall(); kick += face.getDebugState().toolPulse / 6; tick(60); }
  out[label + ':' + n] = { spin: (b.tesseract.kernelSpin - spin0) / 120, flow: (b.cube2Flow - flow0) / 120, pulse: (b.pulsePhase - pulse0) / 120,
    dim: dim / F, y: y / F, bloom: bloom / F, hue: hue / F, hueSwing: hueHi - hueLo, breath, wob, kr, kSeed, sSeed, kick,
    seed: s.reduce((p, q) => p + q, 0) / s.length, look: b.moodLook };
}
emit(out);
""",
    # On-screen movement: resting in each mood, and through each change of mood.
    "steps": r"""
import { step, px, tick, q, emit, face } from './common.mjs';
tick(600); let prev = px(); const out = {};
for (const [label, n] of [['tired', 3], ['cheerful', 3], ['irritable', 3], [null, null]]) {
  face.previewMood(label, n); const change = [], rest = [];
  for (let f = 0; f < 1200; f++) { const F = step(prev); prev = F.cur; change.push(F.med); }
  for (let f = 0; f < 3600; f++) { const F = step(prev); prev = F.cur; rest.push(F.med); }
  out[label || 'neutral'] = { changeMax: Math.max(...change), p99: q(rest, 0.99), max: Math.max(...rest) };
}
emit(out);
""",
    # The console's preview, the real mood underneath it, and leaving the form.
    "control": r"""
import { tick, emit, face } from './common.mjs';
const hue = () => face.getDebugState().moodLook.heat, settle = () => tick(1800);
settle(); const start = hue();
face.setMoodHue('irritable', 3); settle(); const live = hue();
face.previewMood('tired', 3); settle(); const preview = hue();
const back = face.previewMood(); settle(); const restored = hue();
face.previewMood('no such mood', 3); settle(); const unknown = hue();
face.previewMood(); face.setMoodHue('neutral', 0); settle(); const cleared = hue();
face.setMoodHue('irritable', 3); settle(); face.setMoodHue(null, null); tick(120); const failedPoll = hue();
face.setForm('empty'); tick(1800); const left = face.getDebugState().moodLook;
face.setForm('moodtesseract'); settle(); const returned = hue();
const dbg = face.getDebugState();
emit({ start, live, preview, restored, unknown, cleared, failedPoll, left, returned, lines: dbg.tesseract.lines.shells, backHeat: back.heat });
""",
    "pure": r"""
import { emit, face } from './common.mjs';
const l = (a, b) => face.moodLook(a, b).heat;
emit({ i1: l('irritable', 1), i2: l('irritable', 2), i3: l('irritable', 3), absent: l('irritable'), zero: l('irritable', 0),
       weird: l('irritable', 7), other: l('curious', 3), none: l(null, 3), upper: l('IRRITABLE', 3),
       proto: face.moodLook('__proto__', 3).flow, ctor: face.moodLook('constructor', 3).flow,
       reduced: { flow: face.moodLook('irritable', 3).flow, spin: face.moodLook('irritable', 3).spin, heat: face.moodLook('irritable', 3).heat, bright: face.moodLook('irritable', 3).bright } });
""",
    # A cold mood stays on the palette: the lowest seed plus the lowest hue drift.
    "low": r"""
import { tick, emit, face } from './common.mjs';
import { imesh } from './harness.mjs';
const low = () => imesh.material.uniforms.uHueDrift.value + Math.min(...face.getDebugState().tesseract.seeds);
face.previewMood('tired', 3); tick(9000); let lo = 9;
for (let f = 0; f < 7200; f++) { tick(1); if (f % 30 === 0) lo = Math.min(lo, low()); }
face.previewMood(); face.setForm('empty'); face.setMoodHue('tired', 3); tick(9000);
face.setForm('moodtesseract'); tick(600); let carried = 9;
for (let f = 0; f < 3600; f++) { tick(1); if (f % 30 === 0) carried = Math.min(carried, low()); }
emit({ lowest: lo, carried });
""",
    # Each mood's own colours, on every material, and the way back.
    "rings": r"""
import { tick, emit, face } from './common.mjs';
import { imesh, lines } from './harness.mjs';
const read = (m) => [0, 1, 2, 3, 4].map((i) => { const c = m.material.uniforms['uPal' + i].value; return [c.r, c.g, c.b]; });
tick(60); const neutral = read(imesh); const out = { neutral, moods: {}, lines: {} };
for (const [m, n] of [['cheerful', 3], ['irritable', 3], ['tired', 3], ['irritable', 1]]) {
  face.previewMood(m, n); tick(30); out.early = out.early || read(imesh); tick(3570); out.moods[m + n] = read(imesh); out.lines[m + n] = read(lines); }
face.previewMood(); tick(3600); out.back = read(imesh);
face.setMoodHue('irritable', 3); tick(3600); out.live = read(imesh);
face.setForm('empty'); tick(3600); out.left = read(imesh);
emit(out);
""",
    # A failure over each mood, and how hard it lands.
    "gain": r"""
import * as M from './temperament.js';
const t0 = new Date(2026, 9, 30, 15, 0, 0).getTime();
const run = (bias) => { let actions = 0, doze = 0, cur = null, sum = 0;
  for (let seed = 1; seed <= 6; seed++) { const T = M.createTemperament({ seed, now: t0, lastTurnEndAt: t0 }); T.setMoodBias(bias); let t = t0;
    for (let i = 0; i < 40 * 3600; i++) { t += 25; const o = T.step(0.025, t); if (o.idle && o.idle !== cur) { actions++; if (o.idle === 'doze') doze++; } cur = o.idle; }
    for (let k = 0; k < 12; k++) { sum += T.event('error'); t += 600000; T.step(0.25, t); } }
  return { actions, doze, gain: sum / 72 }; };
const N = { gap: 1, doze: 0, react: 1 };
console.log(JSON.stringify({ neutral: run(N), quick: run({ ...N, gap: 0.5 }), sleepy: run({ ...N, doze: 3 }), hard: run({ ...N, react: 1.5 }), soft: run({ ...N, react: 0.6 }) }));
""",
}


def _build(root: Path, static: Path) -> Path:
    CALM._build(root, static)
    bloom = root / "node_modules" / "three" / "addons" / "postprocessing" / "UnrealBloomPass.js"
    if not bloom.exists():
        bloom = next(root.rglob("UnrealBloomPass.js"))
    src = bloom.read_text(encoding="utf-8")
    h = (root / "harness.mjs").read_text(encoding="utf-8")
    assert "matches: false" in h and "setItem() {}" in h
    h = h.replace("matches: false", "matches: !!process.env.REDUCED")
    h = h.replace("setItem() {}", "setItem(k, v) { globalThis.__saved = v; }")   # the harness's own store drops writes
    (root / "harness.mjs").write_text(h, encoding="utf-8")
    needle = "this.strength = s;"
    assert needle in src
    bloom.write_text(src.replace(needle, needle + " globalThis.__bloom = this;"), encoding="utf-8")
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def now(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("mood_now"), _STATIC)


@pytest.fixture(scope="module")
def before(tmp_path_factory):
    root = _build(tmp_path_factory.mktemp("mood_before"), _BEFORE)
    # The dive's motes went with the forms that dived. They were invisible here, but
    # drawing their random numbers at start shifted every node's jitter after them.
    f = next(p for p in root.rglob("matrix_graph.js") if "node_modules" not in str(p))
    src = f.read_text(encoding="utf-8")
    for old, new in (("const u = Math.random(), v = Math.random();", "const u = 0.5, v = 0.5;"),
                     ("Math.pow(Math.random(), 0.55); // inward bias", "0.5; // inward bias"),
                     ("moteSeeds[i] = Math.random();", "moteSeeds[i] = 0.5;")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    f.write_text(src, encoding="utf-8")
    return root


def _run(root, scenario, form="moodtesseract", **env):
    return CALM._run(root, scenario, form=form, **env)


@pytest.mark.parametrize("form,mood", [("moodtesseract", ""), ("moodtesseract", "irritable"),
                                       ("moodtesseract", "cheerful"), ("moodtesseract", "tired"),
                                       ("empty", ""), ("empty", "irritable")])
def test_the_faces_that_stayed_are_frame_for_frame_what_they_were(now, before, form, mood):
    """World: taking cube, tesseract, vortex and descent out moved something the
    two faces that stayed rely on. Everything is summed, not only the nodes:
    uniforms, colours, lines drawn, bloom, the scene and the inner state, through
    a stretch with real events."""
    env = dict(form=form, ROW=_ROW)
    if mood:
        env.update(LIVE="1", MOOD=mood)
    a = _run(before, "digest", **env)
    b = _run(now, "digest", **env)
    assert a["form"] == b["form"] == form
    assert a["parts"]["drawn"] == b["parts"]["drawn"]
    assert {k: v for k, v in b["parts"].items()} == {k: v for k, v in a["parts"].items()}


def test_each_mood_reads_differently_on_every_channel(now):
    """World: the look exists on paper but a channel does not reach the screen.
    Each is read from what the loop actually drives, not from the table."""
    r = _run(now, "looks", MOODS=json.dumps([["neutral", 0], ["cheerful", 3], ["irritable", 3], ["tired", 3]]))
    n, c, i, t = r["neutral:0"], r["cheerful:3"], r["irritable:3"], r["tired:3"]
    assert i["spin"] > 2 * n["spin"] and c["spin"] > 1.3 * n["spin"] and t["spin"] < 0.5 * n["spin"], r
    assert c["flow"] > 1.2 * n["flow"] and i["flow"] > 1.1 * n["flow"] and t["flow"] < 0.6 * n["flow"], r
    assert c["dim"] > 1.15 * n["dim"] and t["dim"] < 0.85 * n["dim"], r
    assert i["seed"] > n["seed"] + 0.1 and t["seed"] < n["seed"] - 0.1, r   # red reaches out; the corridor runs cold
    assert t["y"] < n["y"] - 0.1, r                                        # it sits lower
    assert i["pulse"] > 1.3 * n["pulse"] and t["pulse"] < 0.7 * n["pulse"], r      # breath's rate
    assert i["breath"] - 1 > 1.3 * (n["breath"] - 1), r                            # breath's depth
    assert i["kr"] < 0.85 * n["kr"] and c["kr"] > 1.1 * n["kr"], r                 # the kernel tightens or swells
    assert i["wob"] < 0.6 * n["wob"], r                                            # rigid shells
    assert c["bloom"] > 1.15 * n["bloom"] and t["bloom"] < 0.9 * n["bloom"], r     # the glow follows the light
    assert t["kSeed"] < n["kSeed"] - 0.1, r                                        # the kernel itself runs cold
    assert i["sSeed"] > n["sSeed"] + 0.05 and t["sSeed"] < n["sSeed"] - 0.05, r    # and the stream with it
    assert c["hue"] > n["hue"] + 0.015, r                                          # the warm shift
    assert t["hueSwing"] < 0.5 * n["hueSwing"], r                                  # tired stops wandering
    assert i["hueSwing"] < 0.6 * n["hueSwing"], r                                  # the idle swing calms so it reads
    assert i["kick"] > 1.3 * t["kick"], r                                          # the live loop feeds the bias


def test_intensity_grows_the_look(now):
    """World: every intensity looks the same, or a higher one is weaker."""
    r = _run(now, "looks", MOODS=json.dumps([["irritable", 1], ["irritable", 2], ["irritable", 3]]))
    seeds = [r[f"irritable:{n}"]["seed"] for n in (1, 2, 3)]
    spins = [r[f"irritable:{n}"]["spin"] for n in (1, 2, 3)]
    assert seeds[0] < seeds[1] < seeds[2], r
    assert spins[0] < spins[1] < spins[2], r


def test_moodlook_reads_only_what_it_should(now):
    """World: an out-of-range intensity or an unknown label produces a look."""
    r = _run(now, "pure")
    assert r["i1"] < r["i2"] < r["i3"]
    assert r["absent"] == r["i2"] and r["upper"] == r["i3"]
    for k in ("zero", "weird", "other", "none"):
        assert r[k] == 0, (k, r)
    assert r["proto"] == 1 and r["ctor"] == 1, r     # the table's inherited keys are not moods (NaN for good)
    calm = _run(now, "pure", REDUCED="1")["reduced"]
    assert calm["flow"] == 1 and calm["spin"] == 1, calm           # reduced motion: a steady tempo…
    assert calm["heat"] > 0.7 and calm["bright"] > 1.05, calm      # …and still a mood


def test_no_mood_and_no_change_of_mood_brings_the_jitter_back(now):
    """World: a mood steps the scene as it arrives, or rests with the kind of
    frame-to-frame pop §4TF removed. Irritable must read as tense, not twitchy:
    its tempo is higher, so it gets more room, but never a pop."""
    r = _run(now, "steps")
    for k, v in r.items():
        assert v["changeMax"] < 1.0, (k, v)
        assert v["max"] < 0.7 and v["p99"] < 0.45, (k, v)


def test_preview_and_the_real_mood_stay_apart(now):
    """World: a preview overwrites the agent's mood, or sticks; an unknown
    preview falls through to the real mood instead of showing neutral; the look
    follows the operator onto the empty form, or is lost on the way back."""
    r = _run(now, "control")
    assert r["start"] == 0
    assert r["live"] > 0.7, r
    assert r["preview"] < -0.15, r
    assert r["restored"] == r["live"] and r["backHeat"] == r["live"], r
    assert r["unknown"] == 0, r
    assert r["cleared"] == 0, r
    assert all(r["left"][k] == {"heat": 0, "hue": 0, "droop": 0}.get(k, r["left"][k]) for k in ("heat", "hue", "droop")), r
    assert r["failedPoll"] > 0.7, r                  # one failed health poll is not a change of mood
    assert r["left"]["flow"] == 1 and r["left"]["spin"] == 1 and r["left"]["bright"] == 1, r
    assert r["returned"] > 0.7, r
    assert r["lines"] > 0, r


@pytest.mark.parametrize("mood", ["irritable", "cheerful", "tired"])
def test_a_preview_shows_what_the_live_mood_shows(now, mood):
    """World: the label's own hue shift is added on top of the look when the mood
    is real, so a look tuned with previewMood is not the one that appears, and
    irritable's kernel slides past the red stop toward plum."""
    a = _run(now, "digest", PREVIEW="1", MOOD=mood, ROW=_ROW)
    b = _run(now, "digest", LIVE="1", MOOD=mood, ROW=_ROW)
    assert b["parts"] == pytest.approx(a["parts"], rel=1e-6, abs=1e-6)


def test_each_mood_has_its_own_colours_and_neutral_keeps_the_thermal_ring(now):
    """World: every mood is a shade of the same blue-to-red, or neutral is left a
    shade off the colours the operator wants kept, or one material misses the
    change and the lines and nodes disagree."""
    r = _run(now, "rings")
    ring = [[0x0c, 0x1c, 0x50], [0x1e, 0x2a, 0x78], [0x4a, 0x16, 0x5a], [0x70, 0x12, 0x26], [0x38, 0x17, 0x3f]]
    assert r["neutral"] == [[v / 255 for v in c] for c in ring]
    assert r["back"] == r["neutral"] and r["left"] == r["neutral"], r    # exact, not nearly
    far = lambda a, b: max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q))
    full = {m: r["moods"][m + "3"] for m in ("cheerful", "irritable", "tired")}
    for m, c in full.items():
        assert far(c, r["neutral"]) > 0.1, m
        assert r["lines"][m + "3"] == c, m
        assert max(v for s in c for v in s) < 0.62, m                    # dark-first: bloom lifts them
    # They ease in, and each stays in its family.
    assert 0.002 < far(r["early"], r["neutral"]) < 0.7 * far(full["cheerful"], r["neutral"]), r
    assert all(c[2] > c[0] for c in full["tired"][:4]), full["tired"]              # dusk: nothing warm
    assert all(c[0] > c[2] for c in full["irritable"][2:]), full["irritable"]      # fire: the hot stops
    assert full["cheerful"][0][1] > full["cheerful"][0][0] and full["cheerful"][3][0] > full["cheerful"][3][2]  # teal to coral
    assert far(full["cheerful"], full["irritable"]) > 0.1 and far(full["cheerful"], full["tired"]) > 0.1
    assert far(full["irritable"], full["tired"]) > 0.1
    assert far(r["live"], full["irritable"]) < 1e-3, r                   # the live mood wears the same ring
    assert 0.02 < far(r["moods"]["irritable1"], r["neutral"]) < 0.5 * far(full["irritable"], r["neutral"]), r


def test_a_cold_mood_stays_on_the_palette(now):
    """World: a negative heat plus a downward hue runs the seed past the blue
    end of the palette, and the palette wraps there into plum."""
    r = _run(now, "low")
    assert r["lowest"] >= 0, r
    # Nor when the shift comes along from the plain tesseract: gone within 10 s.
    # (-0.07 if it were kept; the slow drift itself touches about -0.001.)
    assert r["carried"] >= -0.01, r


def test_the_mood_reaches_the_inner_state(now):
    """World: only the look changed — a tired face reacts like any other, an
    irritable one no harder, and neither its gestures nor its dozing move."""
    r = _run(now, "gain")
    n = r["neutral"]
    assert r["quick"]["actions"] > 1.4 * n["actions"], r
    assert r["sleepy"]["doze"] > 2 * max(1, n["doze"]), r
    assert r["hard"]["gain"] > 1.2 * n["gain"] and r["soft"]["gain"] < 0.8 * n["gain"], r


def test_the_id_survives_the_prefs_and_the_handheld():
    """World: "mood tesseract" as an id — the server refuses it, the handheld's
    parser drops it, and the pick silently does not stick."""
    assert "moodtesseract" in fs.known_forms(_STATIC)
    server = ast.parse((_ROOT / "interface" / "server.py").read_text(encoding="utf-8"))
    pat = next(n.value.args[0].value for n in server.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", "") == "_FACE_FORM_RE")
    assert re.fullmatch(pat, "moodtesseract")


# ── the button: animations enabled / disabled (operator, 2026-11-06) ──────────
_HANDHELD = _ROOT / "interface" / "externals" / "clockwork_ghost"

_BUTTON_RUN = r"""
const src = require('fs').readFileSync(process.argv[1], 'utf8');
const i = src.indexOf('const ANIMATIONS_ON_FORM'), j = src.indexOf('// Persist a pick SERVER-side');
const k = src.indexOf('function showAnimationsState()');
let e = src.indexOf('window.__ghostWorkspace.toast;', k); e = src.indexOf('});', e); e = src.indexOf('}', e + 3) + 1;
if (i < 0 || j < i || k < 0 || e <= k) throw new Error('the button code moved');
new Function('start', `
let form = start; const saved = [], toasts = [], classes = new Set(), attrs = {}; let handler;
const animationsBtn0 = { classList: { toggle: (c, v) => v ? classes.add(c) : classes.delete(c) },
  setAttribute: (k, v) => { attrs[k] = v; }, addEventListener: (_, f) => { handler = f; }, title: '' };
const document = { getElementById: (id) => id === 'animations-btn' ? animationsBtn0 : null };
const window = { __ghostWorkspace: { toast: (t) => toasts.push(t) } };
const activeFace = { getForm: () => form, setForm: (n) => (form = n) };
const rememberFaceForm = (n) => saved.push(n);
${src.slice(i, j)}
${src.slice(k, e)}
const snap = () => ({ form, saved: [...saved], toasts: [...toasts], off: classes.has('animations-off'),
  pressed: attrs['aria-pressed'], label: attrs['aria-label'], title: animationsBtn0.title });
const out = [snap()]; handler(); out.push(snap()); handler(); out.push(snap());
console.log(JSON.stringify(out));
`)(process.argv[2]);
"""


def _button_run(start):
    """The button's real code, on a fake page and face: what a click changes."""
    r = subprocess.run(["node", "-e", _BUTTON_RUN, str(_STATIC / "app.js"), start],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout)


class _Page(HTMLParser):
    """The page's element ids, and the classes inside the animations button."""

    def __init__(self):
        super().__init__()
        self.ids, self.in_button, self._inside = set(), set(), False

    def parse(self, text):
        self.feed(text)
        return self

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.ids.add(a.get("id"))
        if a.get("id") == "animations-btn":
            self._inside = True
        elif self._inside:
            self.in_button.update((a.get("class") or "").split())

    def handle_endtag(self, tag):
        if tag == "button":
            self._inside = False


class _Sheet:
    """A stylesheet as {selector: {property: value}}."""

    @staticmethod
    def parse(text):
        rules = {}
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", text, flags=re.S)):
            props = dict((k.strip(), v.strip()) for k, _, v in (d.partition(":") for d in body.split(";")) if v)
            for selector in selectors.split(","):
                rules.setdefault(" ".join(selector.split()), {}).update(props)
        return rules


def test_the_button_switches_the_two_forms_and_says_so():
    """World: the button changes the look but not the face, or forgets the choice
    on the next device, or says enabled while it is off."""
    a, b, c = _button_run("moodtesseract")
    assert a["form"] == "moodtesseract" and a["label"] == "Animations enabled" and a["pressed"] == "true" and not a["off"]
    assert b["form"] == "empty" and b["saved"] == ["empty"] and b["off"] and b["pressed"] == "false"
    assert b["label"] == b["title"] == "Animations disabled" and b["toasts"] == ["Animations disabled"]
    assert c["form"] == "moodtesseract" and c["saved"] == ["empty", "moodtesseract"] and not c["off"]
    assert c["toasts"][-1] == "Animations enabled"


def test_the_button_opens_in_the_state_it_was_left_in():
    first = _button_run("empty")[0]
    assert first["off"] and first["label"] == "Animations disabled" and first["pressed"] == "false"


def test_the_menu_is_gone_and_the_off_state_is_visible():
    page = _Page().parse((_STATIC / "index.html").read_text(encoding="utf-8"))
    assert "animations-btn" in page.ids and "face-form-btn" not in page.ids
    assert "animations-slash" in page.in_button
    sheet = _Sheet.parse((_STATIC / "style.css").read_text(encoding="utf-8"))
    assert sheet["#animations-btn .animations-slash"]["display"] == "none"
    assert sheet["#animations-btn.animations-off .animations-slash"]["display"] == "inline"
    assert not [selector for selector in sheet if "face-form-menu" in selector]


def test_an_old_pick_opens_on_the_face_that_stayed(now):
    assert fs.known_forms(_STATIC) == ["moodtesseract", "empty"]
    for old in ("cube", "tesseract", "vortex", "descent", "cube2"):
        assert _run(now, "forms", form=old)["opened"] == "moodtesseract", old
    assert _run(now, "forms")["picked"] == {old: "moodtesseract" for old in ("cube", "tesseract", "vortex", "descent", "cube2")}


def _client_tree():
    return ast.parse((_HANDHELD / "client.py").read_text(encoding="utf-8"))


def _window_methods(tree, *names):
    """The named methods of the handheld's window, executed without PyQt6 (not
    installed here; the client runs on the device)."""
    found = [n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert sorted(n.name for n in found) == sorted(names)
    return ast.Module(body=found, type_ignores=[])


class _Button:
    def __init__(self):
        self.tip = self.name = None

    def setIcon(self, *_): pass
    def setIconSize(self, *_): pass
    def setToolTip(self, word): self.tip = word
    def setAccessibleName(self, word): self.name = word


def test_the_handheld_opens_on_the_mood_tesseract_and_toggles():
    import sys
    sys.path.insert(0, str(_HANDHELD))
    import facestate
    assert facestate.FALLBACK_FORM == "moodtesseract"
    assert facestate.known_forms(_HANDHELD / "webface") == ["moodtesseract", "empty"]

    tree = _client_tree()
    ns = {"icons": SimpleNamespace(icon=lambda *a: a), "T": SimpleNamespace(TEXT_DIM="dim"), "QSize": lambda *a: a}
    exec(compile(_window_methods(tree, "_show_animations", "toggle_animations"), "client.py", "exec"), ns)
    forms = iter(["empty", "moodtesseract", None])
    win = SimpleNamespace(animations_btn=_Button(),
                          web_face=SimpleNamespace(cycle_form=lambda got: got(next(forms))))
    win._show_animations = lambda on: ns["_show_animations"](win, on)
    ns["toggle_animations"](win)
    assert win.animations_btn.tip == win.animations_btn.name == "Animations off"
    ns["toggle_animations"](win)
    assert win.animations_btn.tip == win.animations_btn.name == "Animations on"
    ns["toggle_animations"](win)                                 # the face reported nothing: the chip stays
    assert win.animations_btn.tip == "Animations on"

    # the mood reaches the face, and main() starts the loop that carries it
    told = []

    class _Client:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False

    class _Stop(Exception):
        pass

    async def fetch_mood(*_):
        return ("irritable", 2)

    async def stop(_):
        raise _Stop

    ns = {"MOOD_POLL_S": 60, "httpx": SimpleNamespace(AsyncClient=_Client), "AGENT_BASE": "a", "GHOST_API_KEY": "k",
          "agentapi": SimpleNamespace(fetch_mood=fetch_mood), "asyncio": SimpleNamespace(sleep=stop)}
    exec(compile(_window_methods(tree, "mood_loop"), "client.py", "exec"), ns)
    win = SimpleNamespace(web_face=SimpleNamespace(set_agent_mood=lambda *m: told.append(m)))
    with pytest.raises(_Stop):
        asyncio.run(ns["mood_loop"](win))
    assert told == [("irritable", 2)]
    started = [ast.unparse(c.args[0]) for c in ast.walk(tree) if isinstance(c, ast.Call)
               and isinstance(c.func, ast.Attribute) and c.func.attr == "create_task" and c.args]
    assert "window.mood_loop()" in started
    assert "switch_face_btn" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    html = (_HANDHELD / "webface" / "face.html").read_text(encoding="utf-8")
    assert "case 'agentMood': fn('setMoodHue', arg[0], arg[1]); break;" in html
    # the copy deploy.sh sends is the web face
    assert filecmp.cmp(_HANDHELD / "webface" / "matrix_graph.js", _STATIC / "matrix_graph.js", shallow=False)


@pytest.mark.parametrize("body,want", [
    ({"mood": {"label": "irritable", "intensity": 2}}, ("irritable", 2)),
    ({"mood": {"label": "neutral", "intensity": 0}}, ("neutral", 0)),
    ({"mood": {"label": "x'); alert(1); //", "intensity": 2}}, None),   # goes into a JS call
    ({"mood": {"label": "calm", "intensity": 9}}, None),
    ({"mood": {"label": "calm", "intensity": True}}, None),
    ({"mood": None}, None), ({}, None),
])
def test_the_handheld_reads_the_mood_and_keeps_the_look_on_nonsense(body, want):
    import asyncio, sys
    sys.path.insert(0, str(_HANDHELD))
    import agentapi

    class R:
        status_code = 200
        def json(self): return body

    class C:
        async def get(self, url, headers=None):
            assert url.endswith("/api/health")
            return R()

    assert asyncio.run(agentapi.fetch_mood(C(), "http://a", "k")) == want
