"""Face §4UG (2026-11-11): the face shows the mood continuously, plays one
moment per shade, and rests and reacts in the character's way.

1  CONTINUOUS MOOD — `moodLookAt(p, e, label, intensity)` blends the region
   looks by where the mood's two numbers sit; at the agent's own anchor points
   it IS today's look for that label and intensity; without numbers it is
   exactly `moodLook(label, intensity)`.
2  MOMENTS — a new `shade_at` plays one eased gesture (relieved / proud /
   chastened) once; the same stamp never again; an old stamp on the first
   reading is not replayed; reduced motion keeps the light and the colour only.
3  CHARACTER — warmth / humour / directness are bounded biases inside the
   temperament; no dials = bit-for-bit today's outputs.

Everything here is executed (node for the JS, the real Python for the
handheld); nothing reads source text for behaviour.
"""
from __future__ import annotations

import ast
import asyncio
import datetime
import filecmp
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.test_4ud_mood_ui as UI
import tests.test_interface_face_idle_calm_2026_11_06 as CALM
import tests.test_interface_face_mood_tesseract_2026_11_06 as MOOD
from ghost_agent.selfhood import mood as M

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_HANDHELD = _ROOT / "interface" / "externals" / "clockwork_ghost"
agentapi = UI.agentapi

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_TEMPO = ("flow", "spin", "pulse", "breath", "wob")

_SCENARIOS = {
    # The pure functions, asked a list of questions.
    "pure4ug": r"""
import { emit, face } from './common.mjs';
const flat = (L) => { const o = {}; for (const [k, v] of Object.entries(L)) if (k !== 'ring') o[k] = v; o.ring = L.ring.map((c) => [c.r, c.g, c.b]); return o; };
const ask = (q) => q.op === 'at' ? flat(face.moodLookAt(...q.a)) : q.op === 'label' ? flat(face.moodLook(...q.a))
    : q.op === 'k' ? face.moodStrengthAt(...q.a) : q.op === 'stamp' ? face.shadeStampMs(...q.a) : null;
const garbage = [NaN, Infinity, -Infinity, '0.3', null, undefined, true, {}, []].map((x) =>
    [flat(face.moodLookAt(x, 0.3, 'irritable', 2)), flat(face.moodLookAt(0.3, x, 'irritable', 2))]);
emit({ answers: JSON.parse(process.env.Q).map(ask), garbage, want: flat(face.moodLook('irritable', 2)) });
""",
    # A walk over the whole plane in steps of 0.001: how far any field moves in one step.
    "walk4ug": r"""
import { emit, face } from './common.mjs';
const N = face.moodLook('neutral', 0);
const fields = Object.keys(N).filter((k) => k !== 'ring');
let maxField = 0, maxRing = 0, worst = null, falls = 0, maxK = 0, steps = 0;
const cmp = (A, B, where) => { steps++;
  for (const f of fields) { const d = Math.abs(A[f] - B[f]); if (d > maxField) { maxField = d; worst = [f, where]; } }
  for (let i = 0; i < 5; i++) for (const c of ['r', 'g', 'b']) maxRing = Math.max(maxRing, Math.abs(A.ring[i][c] - B.ring[i][c])); };
const D = 0.001;
// circles: the direction turns, the seam at ±π included
for (let r = 0.05; r <= 1.0001; r += 0.05) { const n = Math.ceil(2 * Math.PI * r / D); let prev = null;
  for (let i = 0; i <= n + 3; i++) { const a = -Math.PI + (i / n) * 2 * Math.PI; const L = face.moodLookAt(r * Math.cos(a), r * Math.sin(a));
    if (prev) cmp(prev, L, ['circle', r, a]); prev = L; } }
// rays: the size grows; the look may only grow with it
for (let j = 0; j < 64; j++) { const a = -Math.PI + (j + 0.37) / 64 * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a), u = 1 / Math.max(Math.abs(c), Math.abs(s));
  let prev = null, prevK = 0;
  for (let m = 0; m <= 1.00001; m += D) { const L = face.moodLookAt(m * c * u, m * s * u); const k = face.moodStrengthAt(m);
    maxK = Math.max(maxK, k - prevK); prevK = k;
    if (prev) { cmp(prev, L, ['ray', a, m]); for (const f of fields) if (Math.abs(L[f] - N[f]) < Math.abs(prev[f] - N[f]) - 1e-12) falls++; }
    prev = L; } }
// a mood with a level, small: the floor holds it and it is still continuous
let prevL = null, maxLevelled = 0;
for (let m = 0.001; m <= 0.5; m += D) { const L = face.moodLookAt(-m, 0.75 * m, 'irritable', 1);
  if (prevL) for (const f of fields) maxLevelled = Math.max(maxLevelled, Math.abs(L[f] - prevL[f])); prevL = L; }
emit({ maxField, maxRing, worst, falls, maxK, steps, maxLevelled });
""",
    # The numbers through the live loop.
    "live4ug": r"""
import { tick, emit, face } from './common.mjs';
const look = () => { const L = face.getDebugState().moodLook; const o = {}; for (const [k, v] of Object.entries(L)) if (k !== 'ring') o[k] = v; return o; };
const settle = () => tick(1800), out = {};
face.setMoodHue('irritable', 1); settle(); out.old = look();
out.oldPoint = face.getDebugState().moodPoint;
face.setMoodPoint(-0.44, 0.33); face.setMoodHue('irritable', 1); settle(); out.below = look();
face.setMoodPoint(-0.46, 0.345); face.setMoodHue('irritable', 2); settle(); out.above = look();
face.setMoodHue(null, null); tick(120); out.failedPoll = look();
face.setMoodPoint(null, null); face.setMoodHue('irritable', 2); settle(); out.cleared = look();
face.setMoodPoint(-0.9, 0.675); face.setMoodHue('neutral', 0); settle(); out.neutral = look();
out.point = face.getDebugState().moodPoint;
face.setMoodPoint(0.1, -0.02); face.setMoodHue('calm', 'rest'); settle(); out.rest = look();
face.setMoodHue('irritable', 3); face.setMoodPoint(-0.9, 0.675); settle(); out.pointLast = look();   // the other order
face.setMoodPoint('x', 0.3); settle(); out.badPoint = look();
const P = face.previewMoodAt(0.9, 0.675); settle(); out.preview = look(); out.previewReturn = P.bright;
face.previewMoodAt(); settle(); out.previewOff = look();
emit(out);
""",
    # One moment against the same run without it (the harness is deterministic).
    "moment4ug": r"""
import { tick, setHz, emit, face, scene } from './common.mjs';
import { imesh } from './harness.mjs';
const hz = +(process.env.HZ || 60); setHz(hz); tick(hz * 20);
const kind = process.env.KIND || '';
const played = kind ? face.playMoment(kind) : null;
const row = () => { const t = face.getTemperament().out, D = face.getDebugState();
  return [t.scaleMul, t.offY, t.pitch, t.light, scene.scale.x, scene.position.y, scene.rotation.x,
          imesh.material.uniforms.uFormDim.value, globalThis.__bloom.strength, D.tesseract.warm.front, D.tesseract.warm.env,
          D.tesseract.seeds.reduce((a, b) => a + b, 0), t.moment ? 1 : 0, t.speedMul, t.ampMul]; };
const rows = [], grid = [];
for (let f = 0; f < hz * 8; f++) { tick(1); if (hz === 60) rows.push(row()); if ((f + 1) % (hz / 2) === 0) grid.push(row()); }
emit({ played, rows, grid, moments: face.getDebugState().moments, bad: [face.playMoment('smug'), face.playMoment(null), face.playMoment('constructor')] });
""",
    # What a poll's shade does, by case: each needs its own page load.
    "shade4ug": r"""
import { tick, emit, face } from './common.mjs';
const iso = (agoMs) => new Date(Date.now() - agoMs).toISOString().replace(/\.\d+Z$/, 'Z');
const n = () => face.getDebugState().moments, out = {}, c = process.env.CASE;
tick(600);
if (c === 'old') { const s = iso(600000); out.first = face.noteShade('proud', s); tick(300); out.afterFirst = n();
  out.again = face.noteShade('proud', s); out.seen = face.getDebugState().shadeSeen === s;
  out.newer = face.noteShade('relieved', iso(300000)); out.count = n(); }
if (c === 'fresh') { const s = iso(10000); out.first = face.noteShade('relieved', s); const polls = [];
  for (let k = 0; k < 20; k++) { tick(1500); polls.push(face.noteShade('relieved', s)); } out.polls = polls; out.count = n();
  out.otherShadeSameStamp = face.noteShade('proud', s);
  out.bad = [face.noteShade('smug', iso(0)), face.noteShade('proud', ''), face.noteShade('proud', null), face.noteShade('', iso(0)),
             face.noteShade('proud', 12345), face.noteShade(null, null), face.noteShade()]; out.countBad = n();
  out.next = face.noteShade('chastened', iso(0)); out.countNext = n(); out.now = face.getTemperament().out.moment;
  tick(1); out.playing = face.getTemperament().out.moment; }
if (c === 'empty') { out.first = face.noteShade('', ''); out.then = face.noteShade('proud', iso(900000)); out.count = n(); }
if (c === 'edge_in') { out.first = face.noteShade('proud', iso(110000)); out.count = n(); }
if (c === 'edge_out') { out.first = face.noteShade('proud', iso(130000)); out.count = n(); }
if (c === 'ahead') { out.first = face.noteShade('proud', iso(-30000)); out.count = n(); }
if (c === 'paused') { face.setAnimationPaused(true); out.whilePaused = face.noteShade('proud', iso(5000)); out.countPaused = n();
  face.setAnimationPaused(false); out.countResumed = n(); face.setAnimationPaused(true); face.setAnimationPaused(false);
  out.countTwice = n(); }
if (c === 'paused_long') { const real = Date.now; let skew = 0; Date.now = () => real() + skew;
  face.setAnimationPaused(true); face.noteShade('relieved', iso(0), 0); skew += 30000;
  face.setAnimationPaused(false); let peak = 0; const keys = Object.keys(face.getTemperament().out);
  for (let k = 0; k < 240; k++) { tick(1); const o = face.getTemperament().out; peak = Math.max(peak, Number(o.momentEnv || 0)); }
  out.peak = peak; out.keys = keys; out.count = n(); Date.now = real; }
if (c === 'skewed') { out.played = face.noteShade('proud', iso(10 * 60000), 3); out.count = n(); }
if (c === 'paused_stale') { face.setAnimationPaused(true); out.whilePaused = face.noteShade('proud', iso(130000));
  face.setAnimationPaused(false); out.count = n(); }
if (c === 'unreadable') { out.first = face.noteShade('proud', 'yesterday'); out.again = face.noteShade('proud', 'yesterday'); out.count = n(); }
emit(out);
""",
    # The character's light on the real loop, against the same run without it.
    "char4ug": r"""
import { tick, emit, face } from './common.mjs';
import { imesh } from './harness.mjs';
tick(600); const C = JSON.parse(process.env.CHAR || 'null'), P = JSON.parse(process.env.PREVIEW || 'null');
const back = face.setCharacter(C); if (P) face.previewCharacter(P);
let dim = 0, bloom = 0, step = 0, last = imesh.material.uniforms.uFormDim.value; const ser = [];
for (let f = 0; f < 1500; f++) { tick(1); const v = imesh.material.uniforms.uFormDim.value; ser.push(v); if (f >= 900) { dim += v / 600; bloom += globalThis.__bloom.strength / 600; } }
let after = null; const light = face.getTemperament().out.light;
if (P) { face.previewCharacter(); tick(900); after = face.getTemperament().out.light; }
emit({ dim, bloom, ser, back, light, after });
""",
    # Rest, with everything on and the poll repeating: on-screen movement.
    "rest4ug": r"""
import { step, px, tick, q, emit, face, snapshot } from './common.mjs';
tick(600);
const stamp = new Date(Date.now() - 600000).toISOString().replace(/\.\d+Z$/, 'Z'), C = JSON.parse(process.env.CHAR || 'null');
const poll = () => { face.setMoodPoint(+process.env.P, +process.env.E); face.noteShade('proud', stamp); face.setCharacter(C); face.setMoodHue(process.env.LABEL, +process.env.N); };
poll(); let prev = px(); const change = [], rest = []; let digest = 0, moment = 0, speedLo = 9, speedHi = 0, speed = 0;
for (let f = 0; f < 1200; f++) { const F = step(prev); prev = F.cur; change.push(F.med); }
for (let f = 0; f < 5400; f++) { if (process.env.POLLS && f % 1500 === 0) poll(); const F = step(prev); prev = F.cur; rest.push(F.med);
  for (const p of snapshot()) if (p && p.x < 9000) digest += p.x + 2 * p.y + 3 * p.z;
  const t = face.getTemperament().out; if (t.moment) moment++; digest += t.light + t.scaleMul; speedLo = Math.min(speedLo, t.speedMul); speedHi = Math.max(speedHi, t.speedMul); speed += t.speedMul / 5400; }
emit({ changeMax: Math.max(...change), p99: q(rest, 0.99), max: Math.max(...rest), digest, moment, moments: face.getDebugState().moments,
       speedLo, speedHi, speed, twitch: face.getDebugState().twitchAmp || 0, forms: face.getForms() });
""",
}


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = MOOD._build(tmp_path_factory.mktemp("personality_4ug"), _STATIC)
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src, encoding="utf-8")
    return root


def _run(root, scenario, **env):
    return MOOD._run(root, scenario, **{k: str(v) for k, v in env.items()})


def _pure(root, questions, **env):
    return _run(root, "pure4ug", Q=json.dumps(questions), **env)


def _close(a, b, tol=1e-9):
    assert set(a) == set(b), (set(a) ^ set(b))
    for k in a:
        if k == "ring":
            for ca, cb in zip(a[k], b[k]):
                assert all(abs(x - y) <= tol for x, y in zip(ca, cb)), (k, ca, cb)
        else:
            assert abs(a[k] - b[k]) <= tol, (k, a[k], b[k])


def _anchor(label, n):
    """The point the agent itself puts a mood of this label and intensity at, as health sends it."""
    d, m = M.ANCHOR_DIRECTION[label], M.ANCHOR_MAGNITUDE[n]
    return round(d[0] * m, 3), round(d[1] * m, 3)


def _level(p, e):
    m = max(abs(p), abs(e))
    return sum(1 for t in M.INTENSITY_THRESHOLDS if m >= t)


# ── 1. the continuous mood ──────────────────────────────────────────────────

def test_at_the_agents_anchor_points_the_blend_is_todays_look(lab):
    """World: the blend re-tunes the nine looks the operator settled. At each label's own anchor (the
    engine's ANCHOR_DIRECTION × ANCHOR_MAGNITUDE — what `set_mood` stores) and each intensity, the look
    by the numbers must be the label's look, field for field and colour for colour."""
    cases = [(label, n) for label in M.ANCHOR_DIRECTION for n in (1, 2, 3)]
    assert len(cases) == 27 and {c[0] for c in cases} == set(M.LABELS)
    q = []
    for label, n in cases:
        p, e = _anchor(label, n)
        # the anchors are where the engine's own rules say they are: this label's region, this level
        assert _level(p, e) == n, (label, n, p, e)
        assert M.region(p, e) == ("cheerful" if label == M.CURIOUS else label), (label, p, e)
        q += [{"op": "at", "a": [p, e, label, n]}, {"op": "label", "a": [label, n]}, {"op": "at", "a": [p, e]}]
    got = _pure(lab, q)["answers"]
    for i, (label, n) in enumerate(cases):
        blended, labelled, bare = got[3 * i], got[3 * i + 1], got[3 * i + 2]
        _close(blended, labelled)
        if label != M.CURIOUS:                # curious is a tag: the bare point there is cheerful's
            _close(bare, labelled)
    # and the strength passes through today's three shares exactly at the anchors' sizes
    ks = _pure(lab, [{"op": "k", "a": [m]} for m in (0, 0.32, 0.60, 0.90, 1.0, -0.6)])["answers"]
    assert ks == [0, 0.35, 0.7, 1, 1, 0.7], ks


def test_the_look_moves_a_little_when_the_numbers_move_a_little_and_only_grows_outward(lab):
    """World: the look still steps (0.35 → 0.7 is a third of every field at once), jumps where two
    regions meet or at the ±180° seam, or shrinks somewhere as the mood grows."""
    r = _run(lab, "walk4ug")
    assert r["steps"] > 100000, r
    # the largest field range is `doze` (0..3) and `spin` (0.3..2.6): one thousandth of the plane moves them < 1 %
    assert r["maxField"] < 0.012, r
    assert r["maxRing"] < 0.004, r
    assert r["falls"] == 0, r
    assert r["maxK"] < 0.0015, r               # the strength itself: no step anywhere (today's were 0.35)
    assert r["maxLevelled"] < 0.012, r


def test_a_point_between_two_regions_is_a_mix_of_both(lab):
    """World: the direction snaps to the nearest look (still nine looks, only the strength is smooth)."""
    def at(p, e):
        return {"op": "at", "a": [p, e]}
    got = _pure(lab, [at(0.6, 0.0), at(0.6, 0.225), at(0.6, 0.45), at(0.0, 0.6), at(0.3, 0.6), at(-0.6, 0.0),
                      at(-0.6, -0.01), at(-0.6, 0.01)])["answers"]
    content, mid, cheerful, restless, ch_rest, down, down_lo, down_hi = got
    for f in ("spin", "flow", "bright", "gap"):
        lo, hi = sorted((content[f], cheerful[f]))
        assert lo < mid[f] < hi, (f, content[f], mid[f], cheerful[f])
    assert min(cheerful["wob"], restless["wob"]) < ch_rest["wob"] < max(cheerful["wob"], restless["wob"])
    for f in ("heat", "bright", "droop", "spin"):      # the ±180° seam: both sides are `down`
        assert abs(down_lo[f] - down[f]) < 0.05 and abs(down_hi[f] - down[f]) < 0.05, f


def test_without_numbers_the_look_is_exactly_the_labels(lab):
    """World: an older agent (no `p`/`e`), a failed read or a bad value changes what the face shows."""
    r = _pure(lab, [{"op": "at", "a": [None, None, "irritable", 2]}, {"op": "label", "a": ["irritable", 2]},
                    {"op": "at", "a": [0.1, -0.02, "calm", "rest"]}, {"op": "label", "a": ["calm", "rest"]},
                    {"op": "at", "a": [-0.9, 0.675, "neutral", 0]}, {"op": "label", "a": ["neutral", 0]},
                    {"op": "at", "a": [0.5, 0.5, "grumpy", 2]}, {"op": "label", "a": ["grumpy", 2]},
                    {"op": "at", "a": [0.5, 0.5, "cheerful", 7]}, {"op": "label", "a": ["cheerful", 7]},
                    {"op": "at", "a": [0.0, 0.0]}, {"op": "at", "a": [0.0, 0.0, "tired", 2]}, {"op": "label", "a": ["tired", 2]}])
    a = r["answers"]
    for i in (0, 2, 4, 6, 8, 11):
        _close(a[i], a[i + 1], 0)
    _close(a[10], a[5], 0)                              # the origin is neutral
    for by_p, by_e in r["garbage"]:
        _close(by_p, r["want"], 0)
        _close(by_e, r["want"], 0)


def test_the_label_is_honoured_where_the_numbers_alone_would_show_another_mood(lab):
    """World: `curious` (a tag on a pleasant, energetic mood) shows as cheerful; a small mood the resting
    point pushed across a region shows a look the words do not name; a mood with a level fades to nothing."""
    def at(*a):
        return {"op": "at", "a": list(a)}
    def lab_(label, n):
        return {"op": "label", "a": [label, n]}
    a = _pure(lab, [at(0.6, 0.45, "curious", 2), lab_("curious", 2), lab_("cheerful", 2),
                    at(0.6, 0.2, "curious", 2), lab_("content", 2),
                    at(0.05, 0.19, "content", 1), lab_("content", 3), at(0.05, 0.19),
                    at(-0.02, 0.015, "irritable", 1), lab_("irritable", 3),
                    at(-0.3, -0.3, "curious", 2)])["answers"]
    cur_at, cur, cheer, cur_toward_content, content2, pushed, content3, bare, faint, irr3, cur_far = a
    _close(cur_at, cur)
    assert cur["kernel"] > cheer["kernel"] + 0.05 and cur_at["kernel"] == pytest.approx(cur["kernel"])
    lo, hi = sorted((content2["kernel"], cur["kernel"]))
    assert lo < cur_toward_content["kernel"] < hi                      # still continuous toward content
    # the pushed point: content's look (not restless's), at the point's size
    k = pushed["bright"] - 1
    assert k == pytest.approx((content3["bright"] - 1) * 0.2078125, abs=1e-6)       # strengthAt(0.19)
    assert pushed["wob"] < 1 < bare["wob"]                             # content steadies, restless loosens
    # a mood that is still said never drops under a fifth of its look
    assert faint["heat"] == pytest.approx(0.2 * irr3["heat"]) and faint["heat"] > 0.15
    assert cur_far["kernel"] > 1                                       # a curious the numbers contradict: still curious


def test_reduced_motion_keeps_the_blend_in_light_and_colour_only(lab):
    q = [{"op": "at", "a": [-0.9, 0.675, "irritable", 3]}, {"op": "at", "a": [-0.5, 0.6]}, {"op": "label", "a": ["neutral", 0]}]
    calm, moving = _pure(lab, q, REDUCED="1")["answers"], _pure(lab, q)["answers"]
    for look, free in zip(calm[:2], moving[:2]):
        for f in _TEMPO:
            assert look[f] == calm[2][f] == 1, (f, look[f])
            assert free[f] != 1, f
        assert look["heat"] == free["heat"] > 0.3 and look["bright"] == free["bright"] and look["ring"] == free["ring"]
        assert look["ring"] != calm[2]["ring"]


def test_the_live_face_shows_the_numbers_and_falls_back_without_them(lab):
    """World: the functions exist but the loop still shows the label's fixed look; or a mood just over
    an intensity threshold still jumps; or a failed poll / an older agent loses the look."""
    r = _run(lab, "live4ug")
    pure = _pure(lab, [{"op": "label", "a": ["irritable", 1]}, {"op": "label", "a": ["irritable", 2]},
                       {"op": "label", "a": ["irritable", 3]}, {"op": "label", "a": ["calm", "rest"]},
                       {"op": "at", "a": [-0.44, 0.33, "irritable", 1]}, {"op": "label", "a": ["cheerful", 3]}])["answers"]
    i1, i2, i3, rest, below_want, cheerful3 = pure

    def same(got, want, tol=2e-3):
        for f, v in got.items():
            assert abs(v - want[f]) < tol, (f, v, want[f])
    assert r["oldPoint"] is None
    same(r["old"], i1)                                        # no numbers: today's look
    same(r["below"], below_want)
    # 0.44 → 0.46 crosses the engine's second threshold (0.45): the label's look jumps 0.28 in heat…
    assert i2["heat"] - i1["heat"] > 0.25
    # …the look by the numbers moves a fortieth of that, and sits between the two
    assert 0 < r["above"]["heat"] - r["below"]["heat"] < 0.03, r
    assert i1["heat"] < r["below"]["heat"] < r["above"]["heat"] < i2["heat"]
    same(r["failedPoll"], r["above"], 1e-9)                   # a failed poll keeps it
    same(r["cleared"], i2)                                    # the numbers went away: the label's look
    assert r["point"] == [-0.9, 0.675]
    assert r["neutral"]["heat"] == 0 and r["neutral"]["flow"] == 1    # neutral is neutral whatever the numbers say
    same(r["rest"], rest)                                     # the resting point stays the label's faint look
    same(r["pointLast"], i3)                                  # either order of the two calls ends the same
    same(r["badPoint"], i3)                                   # an unreadable number = no numbers
    same(r["preview"], cheerful3)
    assert r["previewReturn"] == pytest.approx(cheerful3["bright"])
    same(r["previewOff"], i3)


# ── 2. moments ──────────────────────────────────────────────────────────────

_COL = {name: i for i, name in enumerate(
    ["scaleMul", "offY", "pitch", "light", "sceneScale", "sceneY", "sceneRx", "dim", "bloom", "front", "env", "seed",
     "moment", "speed", "amp"])}


def _col(rows, name):
    return [r[_COL[name]] for r in rows]


@pytest.fixture(scope="module")
def moments(lab):
    out = {"": _run(lab, "moment4ug")}
    for kind in ("relieved", "proud", "chastened"):
        out[kind] = _run(lab, "moment4ug", KIND=kind)
    return out


def _diff(moments, kind, name):
    return [a - b for a, b in zip(_col(moments[kind]["rows"], name), _col(moments[""]["rows"], name))]


def test_each_moment_is_clearly_there_and_its_own(moments):
    """World: a moment is a few percent (the 0.15 `rest` tint was the example of too faint), or all
    three look alike."""
    assert moments[""]["played"] is None and moments[""]["moments"] == 0
    for kind in ("relieved", "proud", "chastened"):
        assert moments[kind]["played"] is True and moments[kind]["moments"] == 1
        assert moments[kind]["bad"] == [False, False, False]
    ratio = {k: [a / b for a, b in zip(_col(moments[k]["rows"], "dim"), _col(moments[""]["rows"], "dim"))]
             for k in ("relieved", "proud", "chastened")}
    # relieved: lets go — swells, lifts, brightens
    assert max(_diff(moments, "relieved", "scaleMul")) > 0.09 and min(_diff(moments, "relieved", "scaleMul")) > -1e-6
    assert max(_diff(moments, "relieved", "offY")) > 0.04
    assert max(ratio["relieved"]) > 1.18
    assert max(_diff(moments, "relieved", "sceneScale")) > 0.08          # it reaches the scene
    # proud: a warm bright front from the kernel to the rim, standing taller
    assert max(ratio["proud"]) > 1.22
    assert max(_diff(moments, "proud", "scaleMul")) > 0.07 and max(_diff(moments, "proud", "offY")) > 0.05
    assert min(_diff(moments, "proud", "pitch")) < -0.04                 # the head lifts
    env, front = _col(moments["proud"]["rows"], "env"), _col(moments["proud"]["rows"], "front")
    assert max(env) > 0.95 and env[-1] == 0
    live = [f for f, e in zip(front, env) if e > 0]
    assert live[0] < 0.1 and live[-1] > 0.85 and all(b >= a for a, b in zip(live, live[1:]))    # outward, never back
    assert max(_diff(moments, "proud", "seed")) > 1.0                    # the shells it passes go to the hot stop (summed)
    assert max(_col(moments["relieved"]["rows"], "env")) == 0            # only pride has the front
    assert max(_diff(moments, "proud", "bloom")) > 0.1 * max(_col(moments[""]["rows"], "bloom"))
    # chastened: droops and dims
    assert min(ratio["chastened"]) < 0.72
    assert min(_diff(moments, "chastened", "offY")) < -0.10 and min(_diff(moments, "chastened", "scaleMul")) < -0.06
    assert max(_diff(moments, "chastened", "pitch")) > 0.10              # the head drops
    assert min(_diff(moments, "chastened", "sceneY")) < -0.09
    # a moment is a gesture, not a change of tempo
    for kind in ("relieved", "proud", "chastened"):
        assert max(abs(d) for d in _diff(moments, kind, "speed")) == 0
        assert max(abs(d) for d in _diff(moments, kind, "amp")) == 0


def test_a_moment_is_eased_lasts_seconds_and_ends(moments):
    """World: it steps the scene in one frame (the §4TF class), is over before it can be seen, or
    never lets go."""
    for kind in ("relieved", "proud", "chastened"):
        for name, per_frame in (("scaleMul", 0.006), ("offY", 0.004), ("pitch", 0.006), ("light", 0.012),
                                ("sceneScale", 0.006), ("sceneY", 0.004)):
            d = [0.0] + _diff(moments, kind, name)
            step = max(abs(b - a) for a, b in zip(d, d[1:]))
            peak = max(abs(v) for v in d)
            assert step < per_frame, (kind, name, step)
            if peak == 0:
                continue                                                 # this moment does not use the channel
            assert step < 0.08 * peak, (kind, name, step, peak)          # no frame carries a twelfth of it
            assert abs(d[-1]) < 2e-3, (kind, name, d[-1])                # 8 s later it is over
            # it can be seen: over a third of its peak for more than a second
            assert sum(1 for v in d if abs(v) > peak / 3) > 60, (kind, name)
        playing = _col(moments[kind]["rows"], "moment")
        assert 3.5 * 60 <= sum(playing) <= 5.2 * 60, (kind, sum(playing))
        assert playing[-1] == 0


def test_under_reduced_motion_a_moment_is_light_and_colour_only(lab):
    base = _run(lab, "moment4ug", REDUCED="1")
    for kind, lo, hi in (("relieved", 1.18, 9), ("proud", 1.22, 9), ("chastened", 0, 0.72)):
        r = _run(lab, "moment4ug", KIND=kind, REDUCED="1")
        for name in ("sceneScale", "sceneY", "sceneRx"):
            assert _col(r["rows"], name) == _col(base["rows"], name), (kind, name)      # nothing moves
        ratio = [a / b for a, b in zip(_col(r["rows"], "dim"), _col(base["rows"], "dim"))]
        assert (max(ratio) > lo) if hi == 9 else (min(ratio) < hi), (kind, max(ratio), min(ratio))
        if kind == "proud":
            # review (face): nothing travels under reduced motion — the warm front stays off, the light plays
            assert max(_col(r["rows"], "env")) == 0


@pytest.mark.parametrize("kind", ["relieved", "proud", "chastened"])
def test_a_moment_is_the_same_at_30_60_and_120_hz(lab, moments, kind):
    """World: a per-frame envelope — twice as fast on a 120 Hz display."""
    a, c = (_run(lab, "moment4ug", KIND=kind, HZ=hz)["grid"] for hz in (30, 120))
    b = moments[kind]["grid"]
    assert len(a) == len(b) == len(c) == 16
    for name, tol in (("scaleMul", 0.012), ("offY", 0.008), ("pitch", 0.01), ("light", 0.012), ("front", 0.02), ("env", 0.03)):
        d = max(max(abs(x - y), abs(z - y)) for x, y, z in zip(_col(a, name), _col(b, name), _col(c, name)))
        assert d < tol, (name, d)


def _shade(lab, case):
    return _run(lab, "shade4ug", CASE=case)


def test_a_new_shade_plays_once_and_the_same_stamp_never_again(lab):
    """World: the agent sends the shade on every poll for up to 30 minutes — the face replays it every 25 s."""
    r = _shade(lab, "fresh")
    assert r["first"] is True
    assert r["polls"] == [False] * 20 and r["count"] == 1
    assert r["otherShadeSameStamp"] is False                 # one moment per stamp
    assert r["bad"] == [False] * 7 and r["countBad"] == 1    # not a shade, no stamp: nothing, and no throw
    assert r["next"] is True and r["countNext"] == 2         # the next shade plays
    assert r["playing"] == "chastened"


def test_an_old_shade_on_the_first_reading_is_not_replayed(lab):
    """World: every page reload inside the shade's 30 minutes plays the moment again."""
    r = _shade(lab, "old")
    assert r["first"] is False and r["afterFirst"] == 0
    assert r["again"] is False and r["seen"] is True         # it is remembered, not waiting to fire
    assert r["newer"] is True and r["count"] == 1            # a NEW stamp on a later poll plays
    assert _shade(lab, "edge_in") == {"first": True, "count": 1}       # 110 s old: fresh
    assert _shade(lab, "edge_out") == {"first": False, "count": 0}     # 130 s old: not
    assert _shade(lab, "ahead") == {"first": True, "count": 1}         # this clock runs 30 s behind the agent's
    assert _shade(lab, "unreadable") == {"first": False, "again": False, "count": 0}
    # a first reading without a shade uses up "first": what comes later is new
    assert _shade(lab, "empty") == {"first": False, "then": True, "count": 1}


def test_the_stamp_is_read_the_same_in_every_browser(lab):
    """No `Date.parse` (browsers differ on fractions and missing zones): the face's own reader, against Python."""
    def ms(*a, **k):
        return int(datetime.datetime(*a, tzinfo=datetime.timezone.utc, **k).timestamp() * 1000)
    table = [("2026-11-11T22:30:00Z", ms(2026, 11, 11, 22, 30)), ("2026-11-11T22:30:05.123456Z", ms(2026, 11, 11, 22, 30, 5)),
             ("2026-11-12T00:00:00+01:30", ms(2026, 11, 11, 22, 30)), ("2026-11-11T12:30:00-10:00", ms(2026, 11, 11, 22, 30)),
             ("2026-11-11T22:30:00", ms(2026, 11, 11, 22, 30)), ("2026-11-11T22:30Z", ms(2026, 11, 11, 22, 30)),
             ("", None), ("yesterday", None), ("2026-11-11", None), ("2026-11-11 22:30:00Z", None), (None, None), (5, None)]
    got = _pure(lab, [{"op": "stamp", "a": [s]} for s, _ in table])["answers"]
    assert got == [want for _, want in table]
    # and it is the engine's own stamp format
    from ghost_agent.selfhood.shade import ShadeStore
    stamp = ShadeStore._iso(datetime.datetime(2026, 11, 11, 22, 30, 7, tzinfo=datetime.timezone.utc))
    assert _pure(lab, [{"op": "stamp", "a": [stamp]}])["answers"] == [ms(2026, 11, 11, 22, 30, 7)]


# ── 3. the character ────────────────────────────────────────────────────────

_T = """
const mk = (c, seed = 5) => { const T = M.createTemperament({ seed, now: t0, lastTurnEndAt: t0 }); if (c !== undefined) T.setCharacter(c); return T; };
"""


def test_no_character_changes_nothing_at_all(lab):
    """World: the new code moves the face for everyone — `{}` (the character is off), null, all zeros
    or dials nobody can read must give today's outputs to the last bit, events included."""
    r = CALM._temper(lab, _T + """
const trace = (T) => { let t = t0; const rows = []; let light = 0; const ev = { 600: 'turn:start', 700: 'tool', 900: 'error', 1500: 'turn:end', 5000: 'recall', 9000: 'outcome:fail' };
  for (let i = 0; i < 60 * 240; i++) { if (ev[i]) rows.push(T.event(ev[i])); t += 1000 / 60; const o = T.step(1 / 60, t); if (o.light !== 1) light++; if (i % 7 === 0) rows.push(JSON.stringify(o)); }
  return { s: rows.join('|'), light }; };
const base = trace(mk());
const others = [null, {}, { warmth: 0, humour: 0, directness: 0 }, { warmth: '2', humour: NaN, directness: true }, 'warm', [], { warmth: null }];
emit({ same: others.map((c) => trace(mk(c)).s === base.s), len: base.s.length, light: base.light,
       moved: trace(mk({ warmth: 1 })).s !== base.s,
       bias: [M.characterBias({}), M.characterBias({ warmth: 99, humour: -99, directness: 2 }), M.characterBias({ warmth: 2, humour: -2, directness: 2 })] });
""")
    assert r["same"] == [True] * 7 and r["len"] > 100000, r["same"]
    assert r["light"] == 0                       # the light field is exactly 1 the whole way
    assert r["moved"] is True                    # the instrument sees a dial when there is one
    assert r["bias"][0] == {"gap": 1, "light": 1, "heavy": 1, "react": 1, "bright": 0, "hue": 0}
    assert r["bias"][1] == r["bias"][2]          # out of range is brought to the end, not multiplied


def test_humour_changes_how_the_face_idles_and_stays_calm(lab):
    """World: the dial is stored and nothing at rest differs; or playful brings the jitter back
    (bigger or faster idle movement, a resting speed that is no longer ×1)."""
    r = CALM._temper(lab, _T + """
const LIGHT = ['glance', 'lookaround', 'stretch'];
const run = (c, bias) => { let actions = 0, light = 0, maxYaw = 0, maxStep = 0, sp = 0, n = 0, twitchy = 0;
  for (let seed = 1; seed <= 6; seed++) { const T = mk(c, seed); if (bias) T.setMoodBias(bias); let t = t0, cur = null, last = null;
    for (let i = 0; i < 40 * 3600; i++) { t += 25; const o = T.step(0.025, t);
      if (o.idle && o.idle !== cur) { actions++; if (LIGHT.includes(o.idle)) light++; } cur = o.idle;
      maxYaw = Math.max(maxYaw, Math.abs(o.yaw), Math.abs(o.pitch)); sp += o.speedMul; n++; if (o.startleKick) twitchy++;
      if (last) maxStep = Math.max(maxStep, Math.abs(o.yaw - last[0]), Math.abs(o.pitch - last[1]), Math.abs(o.scaleMul - last[2]));
      last = [o.yaw, o.pitch, o.scaleMul]; } }
  return { actions, share: light / actions, maxYaw, maxStep, speed: sp / n, twitchy }; };
emit({ neutral: run(), playful: run({ humour: 2 }), serious: run({ humour: -2 }), one: run({ humour: 1 }),
       quickMood: run(undefined, { gap: 0.5, doze: 0, react: 1 }), both: run({ humour: 2 }, { gap: 0.5, doze: 0, react: 1 }) });
""")
    n, p, s = r["neutral"], r["playful"], r["serious"]
    assert p["actions"] > 1.3 * n["actions"] and s["actions"] < 0.8 * n["actions"], r
    assert n["actions"] < r["one"]["actions"] < p["actions"], r                   # it grows with the dial
    assert p["share"] > n["share"] + 0.08 and s["share"] < n["share"] - 0.08, r   # lighter / heavier gestures
    for v in r.values():
        assert 0.9 <= v["speed"] <= 1.1, v                 # resting speed stays ≈ ×1
        assert v["maxYaw"] <= 1.05 * n["maxYaw"] + 1e-9, v  # no gesture is bigger than today's
        assert v["maxStep"] <= 1.1 * n["maxStep"] + 1e-9, v # …or faster (the eased, third-size idle body)
        assert v["twitchy"] == 0, v
    # a quick mood AND a playful character: bounded, never back-to-back fidgeting
    assert r["both"]["actions"] < 1.25 * r["quickMood"]["actions"], r


def test_warmth_is_a_steady_light_and_nothing_that_moves(lab):
    """World: warm and dry look the same; or warmth changes the tempo or the body (a baseline is light
    and colour only); or it arrives as a step."""
    r = CALM._temper(lab, _T + """
const run = (c, hz = 60) => { const T = mk(c); let t = t0; const body = [], light = []; let hue = 0, maxStep = 0, last = 1;
  for (let i = 0; i < hz * 30; i++) { t += 1000 / hz; const o = T.step(1 / hz, t);
    if (hz === 60) body.push([o.speedMul, o.ampMul, o.yaw, o.pitch, o.roll, o.scaleMul, o.offY, o.bright, o.stillness].join(','));
    maxStep = Math.max(maxStep, Math.abs(o.light - last)); last = o.light; hue += o.hueShift / (hz * 30);
    if ((i + 1) % hz === 0) light.push(o.light); }
  return { body: body.join('|'), light, hue, maxStep }; };
const N = run(), W = run({ warmth: 2 }), D = run({ warmth: -2 }), one = run({ warmth: 1 });
emit({ n: N.light[29], w: W.light[29], d: D.light[29], one: one.light[29], hueW: W.hue - N.hue, hueD: D.hue - N.hue,
       sameBody: [W.body === N.body, D.body === N.body], maxStep: Math.max(W.maxStep, D.maxStep),
       rates: [30, 120].map((hz) => Math.max(...run({ warmth: 2 }, hz).light.map((v, i) => Math.abs(v - W.light[i])))) });
""")
    assert r["n"] == 1
    assert 1.14 < r["w"] < 1.18 and 0.82 < r["d"] < 0.86, r        # ±16 % of the light at the ends
    assert 1.06 < r["one"] < 1.10, r
    assert r["hueW"] > 0.025 and r["hueD"] < -0.025, r             # and warmer / cooler in colour
    assert r["sameBody"] == [True, True]                           # tempo and body untouched, to the bit
    assert r["maxStep"] < 0.002, r                                 # it arrives over seconds
    assert max(r["rates"]) < 1e-6, r                               # the same at 30 and 120 Hz


def test_directness_is_how_hard_events_land_and_nothing_at_rest(lab):
    """World: blunt and diplomatic react alike; or the dial also moves the resting face."""
    r = CALM._temper(lab, _T + """
const run = (c) => { let sum = 0, shake = 0, flare = 0;
  for (let seed = 1; seed <= 6; seed++) { const T = mk(c, seed); let t = t0;
    for (let i = 0; i < 600; i++) { t += 1000 / 60; T.step(1 / 60, t); }
    for (let k = 0; k < 12; k++) { sum += T.event('error'); let pk = 0, fl = 0;
      for (let i = 0; i < 60; i++) { t += 1000 / 60; const o = T.step(1 / 60, t); pk = Math.max(pk, Math.abs(o.offX)); fl = Math.max(fl, o.bright); }
      shake += pk; flare += fl; t += 600000; T.step(0.25, t); } }
  return { gain: sum / 72, shake: shake / 72, flare: flare / 72 }; };
const idle = (c) => { const T = mk(c); let t = t0; const rows = []; for (let i = 0; i < 60 * 300; i++) { t += 1000 / 60; rows.push(JSON.stringify(T.step(1 / 60, t))); } return rows.join('|'); };
emit({ neutral: run(), blunt: run({ directness: 2 }), soft: run({ directness: -2 }), one: run({ directness: 1 }),
       restSame: [idle({ directness: 2 }) === idle(), idle({ directness: -2 }) === idle()] });
""")
    n, b, s = r["neutral"], r["blunt"], r["soft"]
    assert b["gain"] > 1.2 * n["gain"] and s["gain"] < 0.8 * n["gain"], r
    assert n["gain"] < r["one"]["gain"] < b["gain"], r
    assert b["shake"] > 1.15 * n["shake"] and s["shake"] < 0.85 * n["shake"], r     # the startle scales with it
    assert b["flare"] > n["flare"] > s["flare"], r
    assert b["gain"] <= 1.8 and s["gain"] >= 0.25, r                # inside the gain's own bounds
    assert r["restSame"] == [True, True]                            # five idle minutes: identical to the bit


def test_the_characters_light_reaches_the_screen_also_under_reduced_motion(lab):
    base = _run(lab, "char4ug")
    warm = _run(lab, "char4ug", CHAR=json.dumps({"warmth": 2, "humour": 0, "directness": 0}))
    dry = _run(lab, "char4ug", CHAR=json.dumps({"warmth": -2}))
    off = _run(lab, "char4ug", CHAR="{}")
    assert off["ser"] == base["ser"] and off["light"] == 1          # {} = no character: the same frames
    assert warm["dim"] > 1.12 * base["dim"] and dry["dim"] < 0.88 * base["dim"], (warm["dim"], base["dim"], dry["dim"])
    assert warm["bloom"] > 1.1 * base["bloom"] and dry["bloom"] < 0.9 * base["bloom"]
    steps = [abs(b / a - 1) for a, b in zip(warm["ser"], warm["ser"][1:])]
    assert max(steps) < 0.01, max(steps)                            # eased in, never a step
    rb = _run(lab, "char4ug", REDUCED="1")
    rw = _run(lab, "char4ug", CHAR=json.dumps({"warmth": 2}), REDUCED="1")
    assert rw["dim"] > 1.12 * rb["dim"]
    # the console preview holds over the agent's own until it is given back
    held = _run(lab, "char4ug", CHAR=json.dumps({"warmth": -2}), PREVIEW=json.dumps({"warmth": 2}))
    assert held["light"] > 1.14 and held["after"] < 0.86, held


# ── calm at rest, with everything on ────────────────────────────────────────

_STRONG = {"warmth": 2, "humour": 2, "directness": 2}


def test_at_rest_with_a_mood_a_character_and_an_old_shade_nothing_fires(lab):
    """World: the 25 s poll itself moves the face (a restarted ease, a replayed moment, a re-set
    character), or a blended mood with a strong character rests with the pop §4TF removed."""
    env = dict(P=0.5, E=0.6, LABEL="cheerful", N=2, CHAR=json.dumps(_STRONG))
    once, polled = _run(lab, "rest4ug", **env), _run(lab, "rest4ug", POLLS="1", **env)
    assert polled["digest"] == once["digest"]                # four more polls: not one node differs
    assert polled["moments"] == 0 and polled["moment"] == 0  # the old shade never plays
    for r in (once, polled):
        assert r["changeMax"] < 1.0 and r["max"] < 0.7 and r["p99"] < 0.60, r      # the mood test's own limits
        assert 0.9 < r["speed"] < 1.1, r                                            # resting speed ≈ ×1
        assert r["twitch"] == 0 and r["forms"] == ["moodtesseract", "empty"], r
    # and the dials on a neutral mood
    quiet = _run(lab, "rest4ug", P=0, E=0, LABEL="neutral", N=0, CHAR=json.dumps(_STRONG), POLLS="1")
    assert quiet["max"] < 0.7 and quiet["p99"] < 0.60 and quiet["moments"] == 0, quiet
    # the temperament's own speed is the same with and without the personality: nothing here touches it
    plain = _run(lab, "rest4ug", P=0, E=0, LABEL="neutral", N=0)
    assert abs(quiet["speed"] - plain["speed"]) < 0.03 and quiet["speedHi"] <= plain["speedHi"] + 0.05, (quiet, plain)


# ── the web page: what a poll tells the face ────────────────────────────────

_FACE_NEEDLE = "const face = DATA.face ? { setMoodHue: (label, n) => { told.push([label, n]); } } : null;"
_REC_FACE = ("const calls = []; const rec = (name) => (...a) => { calls.push([name, ...a]); };\n"
             "const face = DATA.face === 'old' ? { setMoodHue: rec('setMoodHue') }\n"
             "  : DATA.face === 'broken' ? { setMoodHue: rec('setMoodHue'), setMoodPoint() { throw new Error('boom'); } }\n"
             "  : { setMoodHue: rec('setMoodHue'), setMoodPoint: rec('setMoodPoint'), noteShade: rec('noteShade'), setCharacter: rec('setCharacter') };\n"
             )
_POLLS = ("const per = [calls.splice(0)];\n"
          "while (script.length) { await tick(); await settle(); per.push(calls.splice(0)); }\nout(per);")

_STAMP = "2026-11-11T12:00:00Z"
_FULL = {"label": "cheerful", "intensity": 2, "face": "cheerful", "p": 0.5, "e": 0.41, "shade": "proud", "shade_at": _STAMP,
         "shade_age_s": 42.0}
_DIALS = {"warmth": -1, "humour": 1, "directness": 0}


def _page(polls, face="new"):
    assert _FACE_NEEDLE in UI._PAGE
    return UI._node(UI._PAGE.replace(_FACE_NEEDLE, _REC_FACE) + _POLLS, {"polls": polls, "face": face})


def test_a_poll_tells_the_face_the_numbers_the_shade_and_the_dials_before_the_label():
    """World: the server sends the fields and the page drops them; or a failed poll replays a moment /
    clears the character; or once the mood is unknown the old numbers keep a look alive."""
    old_agent = {"label": "calm", "intensity": 1, "face": "calm"}
    bad = dict(_FULL, p="0.5", shade="smug", shade_at="yesterday", shade_age_s="42")
    per = _page([{"status": "ok", "mood": _FULL, "character": _DIALS}, "DOWN",
                 {"status": "ok", "mood": old_agent},
                 {"status": "ok", "mood": bad, "character": ["warm"]},
                 {"status": "ok", "mood": dict(_FULL, e=7), "character": {}},
                 {"status": "ok", "mood": dict(_FULL, shade="relieved", shade_at="soon")},
                 "DOWN", "DOWN", "DOWN"])
    assert per[0] == [["setMoodPoint", 0.5, 0.41], ["noteShade", "proud", _STAMP, 42], ["setCharacter", _DIALS],
                      ["setMoodHue", "cheerful", 2]]
    assert per[1] == [["setMoodPoint", 0.5, 0.41], ["setMoodHue", "cheerful", 2]]       # kept; no moment, no dials
    assert per[2] == [["setMoodPoint", None, None], ["noteShade", "", "", None], ["setCharacter", None],
                      ["setMoodHue", "calm", 1]]                                         # an older agent
    assert per[3] == [["setMoodPoint", None, None], ["noteShade", "", "", None], ["setCharacter", None],
                      ["setMoodHue", "cheerful", 2]]                                     # bad optional fields drop alone
    assert per[4][:3] == [["setMoodPoint", None, None], ["noteShade", "proud", _STAMP, 42], ["setCharacter", {}]]
    assert per[5][1] == ["noteShade", "", "", None]                                           # a shade needs a real stamp
    assert per[6] == per[7] == [["setMoodPoint", 0.5, 0.41], ["setMoodHue", "cheerful", 2]]
    assert per[8] == [["setMoodPoint", None, None], ["setMoodHue", "neutral", 0]]       # unknown: numbers dropped


def test_an_older_face_and_a_face_that_throws_still_get_the_label():
    polls = [{"status": "ok", "mood": _FULL, "character": _DIALS}]
    assert _page(polls, face="old") == [[["setMoodHue", "cheerful", 2]]]
    assert _page(polls, face="broken") == [[["setMoodHue", "cheerful", 2]]]


_BLOCKS = [
    _FULL, dict(_FULL, p=-1, e=1), dict(_FULL, p=1.01), dict(_FULL, p=None), dict(_FULL, e="0.4"), dict(_FULL, p=True),
    {k: v for k, v in _FULL.items() if k not in ("p", "e")}, {k: v for k, v in _FULL.items() if k != "e"},
    dict(_FULL, shade="chastened"), dict(_FULL, shade="relieved", shade_at="2026-11-11T12:00:00.5+02:00"),
    dict(_FULL, shade="smug"), dict(_FULL, shade=""), dict(_FULL, shade_at=""), dict(_FULL, shade_at="2026-11-11"),
    dict(_FULL, shade_at=None), dict(_FULL, shade=["proud"]), dict(_FULL, shade="PROUD"),
    {"label": "neutral", "intensity": 0, "p": 0.0, "e": 0.0, "shade": "", "shade_at": ""},
    {"label": "neutral", "intensity": 0, "rest": "calm", "face": "calm", "p": 0.1, "e": -0.05},
    dict(_FULL, shade_at="2026-11-11T12:00:00Z'); alert(1); //"), dict(_FULL, p=0, e=-0.0),
]


def test_both_clients_read_the_new_fields_the_same_way(local_tz=None):
    """The web page (status.js `moodView`) and the handheld (agentapi `mood_view`) on one table."""
    web = UI._node("out(DATA.map(m => { const v = S.moodView(m); return v && [v.p, v.e, v.shade, v.shadeAt]; }));", _BLOCKS)
    hand = []
    for m in _BLOCKS:
        v = agentapi.mood_view(m)
        hand.append(v and [v["p"], v["e"], v["shade"], v["shade_at"]])
    assert web == hand
    assert web[0] == [0.5, 0.41, "proud", _STAMP] and web[1][:2] == [-1, 1]
    for i in (2, 3, 4, 5, 6, 7):
        assert web[i][:2] == [None, None], (i, web[i])          # both numbers or neither
    assert web[8][2] == "chastened" and web[9][2:] == ["relieved", "2026-11-11T12:00:00.5+02:00"]
    for i in (10, 11, 12, 13, 14, 15, 16, 19):
        assert web[i][2:] == ["", ""], (i, web[i])
    assert web[17] == [0, 0, "", ""] and web[18][:2] == [0.1, -0.05]


# ── the handheld ────────────────────────────────────────────────────────────

def test_the_handhelds_face_files_are_the_web_faces():
    for name in ("matrix_graph.js", "temperament.js"):
        assert filecmp.cmp(_HANDHELD / "webface" / name, _STATIC / name, shallow=False), name


def test_the_handheld_watch_hands_over_the_extras_with_the_look():
    watch = agentapi.MoodWatch()
    http = UI._Http({"mood": _FULL, "character": dict(_DIALS, warmth=9, extra=1, humour=True)}, "DOWN",
                    {"mood": {"label": "calm", "intensity": 1}}, "DOWN", "DOWN", "DOWN")
    seen = []
    assert watch.extras() is None                                 # before any reading: keep what it has
    for _ in range(6):
        watch.step(asyncio.run(agentapi.fetch_mood_view(http, "http://a", "k")))
        seen.append(watch.extras())
    full = {"p": 0.5, "e": 0.41, "shade": "proud", "shadeAt": _STAMP, "shadeAgeS": 42.0, "character": {"warmth": 2, "directness": 0}}
    assert seen[0] == dict(full, live=True)
    assert seen[1] == dict(full, live=False)                      # kept through a failed poll, marked so
    assert seen[2] == {"p": None, "e": None, "shade": "", "shadeAt": "", "shadeAgeS": None, "character": None, "live": True}
    assert seen[3] == seen[4] == dict(seen[2], live=False)
    assert seen[5] == {"p": None, "e": None, "live": False}       # unknown: the numbers are dropped
    assert agentapi.character_view({}) is None and agentapi.character_view(["warm"]) is None
    assert agentapi.character_view({"warmth": float("nan"), "humour": "2"}) is None
    # the look itself is what it was (fetch_mood: the pinned two-tuple)
    assert asyncio.run(agentapi.fetch_mood(UI._Http({"mood": _FULL}), "http://a", "k")) == ("cheerful", 2)


def test_the_handheld_client_sends_the_extras_before_the_look_and_survives_a_face_without_them():
    told = []
    scope = UI._window("_mood_read", httpx=SimpleNamespace(AsyncClient=lambda **_kw: UI._Http({"mood": _FULL, "character": _DIALS})))

    def win(face):
        return SimpleNamespace(_mood_watch=agentapi.MoodWatch(), _mood_words="", update_stats=lambda: None, web_face=face)
    asyncio.run(scope["_mood_read"](win(SimpleNamespace(set_agent_mood=lambda *look: told.append(("mood",) + look),
                                                         set_agent_extras=lambda x: told.append(("extras", x))))))
    assert [t[0] for t in told] == ["extras", "mood"] and told[1] == ("mood", "cheerful", 2)
    assert told[0][1]["p"] == 0.5 and told[0][1]["character"] == _DIALS and told[0][1]["live"] is True
    told.clear()

    def boom(_x):
        raise RuntimeError("no view")
    asyncio.run(scope["_mood_read"](win(SimpleNamespace(set_agent_mood=lambda *look: told.append(look), set_agent_extras=boom))))
    asyncio.run(scope["_mood_read"](win(SimpleNamespace(set_agent_mood=lambda *look: told.append(look)))))
    assert told == [("cheerful", 2), ("cheerful", 2)]


def _webface_method(name):
    tree = ast.parse((_HANDHELD / "webface.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "webface.py", "exec"), scope)
    return scope[name]


_BRIDGE = r"""
import fs from 'node:fs';
const html = fs.readFileSync(process.argv[1], 'utf8');
const open = '<script type="module">', i = html.indexOf(open), j = html.indexOf('</' + 'script>', i);
let src = html.slice(i + open.length, j);
const needle = "import('./matrix_graph.js')";
if (!src.includes(needle)) throw new Error('bridge changed');
const DATA = JSON.parse(process.argv[2]);
const calls = [], rec = (name) => (...a) => { calls.push([name, ...a]); };
const full = { init() {}, setMoodHue: rec('setMoodHue'), setMoodPoint: rec('setMoodPoint'), noteShade: rec('noteShade'), setCharacter: rec('setCharacter') };
globalThis.__mod = DATA.face === 'old' ? { init() {}, setMoodHue: rec('setMoodHue') } : full;
let release; const gate = new Promise((r) => { release = r; });
globalThis.__load = () => gate.then(() => globalThis.__mod);
globalThis.window = globalThis; globalThis.addEventListener = () => {};
await import('data:text/javascript;base64,' + Buffer.from(src.replace(needle, 'globalThis.__load()')).toString('base64'));
for (const js of DATA.before) (0, eval)(js);          // sent while the face is still loading: queued
release(); for (let k = 0; k < 10; k++) await new Promise((r) => setImmediate(r));
for (const js of DATA.after) (0, eval)(js);
process.stdout.write(JSON.stringify({ calls, ready: !!globalThis.__faceReady, errors: globalThis.__faceErrors }));
"""


def _bridge(before, after, face="new"):
    p = subprocess.run(["node", "--input-type=module", "-e", _BRIDGE, str(_HANDHELD / "webface" / "face.html"),
                        json.dumps({"before": before, "after": after, "face": face})],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def test_the_handhelds_page_passes_the_extras_to_the_face_and_an_old_face_ignores_them():
    """The whole handheld path: `MoodWatch.extras` → `WebFace.set_agent_extras` (the JS it sends) →
    face.html's bridge → the face's functions, in order, queued while the face loads."""
    sent = []
    host = SimpleNamespace(_js=sent.append)
    extras, mood = _webface_method("set_agent_extras"), _webface_method("set_agent_mood")
    live = {"p": 0.5, "e": 0.41, "shade": "proud", "shadeAt": _STAMP, "shadeAgeS": 42.0, "character": _DIALS, "live": True}
    kept = dict(live, live=False)
    for x, look in ((live, ("cheerful", 2)), (kept, ("cheerful", 2)), ({"p": None, "e": None, "live": False}, ("neutral", 0))):
        extras(host, x)
        mood(host, *look)
    extras(host, None)                                             # nothing to say: nothing sent
    assert len(sent) == 6
    r = _bridge(sent[:2], sent[2:])
    assert r["ready"] is True and r["errors"] == []
    assert r["calls"] == [
        ["setMoodPoint", 0.5, 0.41], ["noteShade", "proud", _STAMP, 42], ["setCharacter", _DIALS], ["setMoodHue", "cheerful", 2],
        ["setMoodPoint", 0.5, 0.41], ["setMoodHue", "cheerful", 2],          # a kept reading: no moment, no dials
        ["setMoodPoint", None, None], ["setMoodHue", "neutral", 0]]
    old = _bridge(sent[:2], sent[2:], face="old")
    assert old["errors"] == [] and old["calls"] == [["setMoodHue", "cheerful", 2], ["setMoodHue", "cheerful", 2],
                                                     ["setMoodHue", "neutral", 0]]
    with pytest.raises(ValueError):
        extras(host, {"p": float("nan")})                          # never a NaN into a JS call



def test_a_moment_that_arrives_while_the_face_is_paused_plays_on_resume_once(lab):
    """Review (face): a hidden tab or the handheld's idle tier still polls; the moment was marked seen and
    played into a frozen clock — lost, or a stray front minutes late."""
    r = _shade(lab, "paused")
    assert r["whilePaused"] is False and r["countPaused"] == 0
    assert r["countResumed"] == 1 and r["countTwice"] == 1
    assert _shade(lab, "paused_stale")["count"] == 0         # already old when it came: seen, never played


def test_bored_shows_at_its_own_level_whatever_the_numbers(lab):
    """Review (face): bored is read off the clock; the pair stays near 0, and the look fell to the point's floor
    (a third of "bored, 2 of 3") while the words said bored 2."""
    q = []
    for n in (1, 2, 3):
        for pt in ((0.005, -0.003), (0.0, 0.0), (0.1, -0.08), (-0.12, 0.05)):
            q += [{"op": "at", "a": [pt[0], pt[1], "bored", n]}, {"op": "label", "a": ["bored", n]}]
    got = _pure(lab, q)["answers"]
    for i in range(0, len(got), 2):
        _close(got[i], got[i + 1])



def test_a_held_moment_is_drawn_after_a_long_pause(lab):
    """Review r2 (face): the held moment was stamped with the clock the pause left; the first frame jumped
    the clock across the pause and the moment was over before it was drawn (only the counter moved)."""
    r = _shade(lab, "paused_long")
    assert r["count"] == 1 and r["peak"] > 0.3, r


def test_freshness_uses_the_agents_age_not_this_devices_clock(lab):
    """A device clock 10 min behind: the stamp looks 10 min old, the agent says 3 s — it plays."""
    r = _shade(lab, "skewed")
    assert r["played"] is True and r["count"] == 1
