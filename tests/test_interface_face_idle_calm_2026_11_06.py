"""The face at rest: no jitter, no lurch — EXECUTED (2026-11-06).

Operator: "the face tesseract in the webui is sometimes gittery and erratic,
mostly while idle". Measured on the real ``animate()`` projected to 1440×900:
with the temperament's body and the twitch off, ten idle minutes were smooth;
with them on, whole-scene turns every ~14 s, a pop every ~13 s, and steps in a
single frame whenever an idle action was cancelled or a tilt restarted.

Every scenario also runs on the pre-fix copy (``face_backups/20261106-preidlecalm``)
where it must FAIL: a harness that cannot see the defect proves nothing (R6).
"Step" below is the median visible node's on-screen move in one frame, in px.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tests.test_interface_face_motion_2026_09_20 as MOT

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_BACKUP = _ROOT / "interface" / "face_backups" / "20261106-preidlecalm"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_ROW = "│  **  12:00:00  {}  +1.0s  some title   some text"

_COMMON = r"""
import { face, tick, snapshot, toWorld, project, setHz, advanceClock, scene, emit } from './harness.mjs';
export { face, tick, setHz, advanceClock, scene, emit, snapshot };
export const px = () => { const s = snapshot(), o = new Array(s.length);
  for (let i = 0; i < s.length; i++) { const p = s[i]; if (!p || p.s < 0.05 || p.x > 9000) { o[i] = null; continue; }
    const q = project(toWorld(p)); if (q.depth < 0.6 || Math.abs(q.sx) > 1.1 || Math.abs(q.sy) > 1.1) { o[i] = null; continue; }
    o[i] = [q.sx * 720, q.sy * 450]; } return o; };
export function step(prev) { tick(1); const cur = px(), d = [];
  for (let i = 0; i < cur.length; i++) { if (!cur[i] || !prev[i]) continue;
    const m = Math.hypot(cur[i][0] - prev[i][0], cur[i][1] - prev[i][1]); if (m < 80) d.push(m); }
  d.sort((a, b) => a - b); return { cur, med: d.length ? d[d.length >> 1] : 0 }; }
export const T = () => face.getTemperament().out;
export const q = (a, p) => [...a].sort((x, y) => x - y)[Math.floor(a.length * p)];
"""

_SCENARIOS = {
    # Ten quiet minutes: nothing happens except what the face does on its own.
    "idle": r"""
import { step, px, tick, T, q, emit, face, snapshot } from './common.mjs';
tick(600); let prev = px(); const m = []; let maxYaw = 0, twitches = 0, armed = true, digest = 0;
for (let f = 0; f < 36000; f++) { const F = step(prev); prev = F.cur; m.push(F.med);
  for (const p of snapshot()) if (p && p.x < 9000) digest += p.x + 2 * p.y + 3 * p.z;
  maxYaw = Math.max(maxYaw, Math.abs(T().yaw));
  const a = face.getDebugState().twitchAmp; if (armed && a > 0.5) { twitches++; armed = false; } else if (a < 0.2) armed = true; }
emit({ form: face.getForm(), p99: q(m, 0.99), max: Math.max(...m), maxYaw, twitches, digest });
""",
    # A turn arrives at the peak of each idle action.
    "cancel": r"""
import { step, px, tick, T, emit, face } from './common.mjs';
tick(600); let prev = px(); const out = {};
for (const name of ['glance', 'lookaround', 'stretch', 'doze']) {
  for (let g = 0; g < 400000; g++) { const F = step(prev); prev = F.cur; const t = T();
    if (t.idle === name && t.idleEnv > 0.85) { face.setUserTurn(true); const G = step(prev); prev = G.cur; out[name] = G.med; break; } }
  face.setUserTurn(false); for (let i = 0; i < 2400; i++) { const F = step(prev); prev = F.cur; }
}
emit(out);
""",
    # A 🌐 row every half second, at rest.
    "tilt": r"""
import { step, px, tick, T, emit, face } from './common.mjs';
tick(600); let prev = px(), last = T().roll, maxStep = 0, maxRoll = 0;
for (let f = 0; f < 600; f++) { if (f % 30 === 0) face.noteLogLine(process.env.ROW);
  const F = step(prev); prev = F.cur; const r = T().roll; maxStep = Math.max(maxStep, Math.abs(r - last)); maxRoll = Math.max(maxRoll, Math.abs(r)); last = r; }
emit({ maxStep, maxRoll });
""",
    # A real failure still shakes and flares; quiet before it.
    "startle": r"""
import { step, px, tick, T, emit, face, scene } from './common.mjs';
tick(600); let prev = px(); const before = T().bright; face.noteError(); let flips = 0, lastX = 0, peakX = 0, peakB = 0, rollTurns = 0, lastR = T().roll, lastDR = 0;
for (let f = 0; f < 150; f++) { const F = step(prev); prev = F.cur; const t = T();
  if (t.offX * lastX < 0) flips++; lastX = t.offX; if ((t.roll - lastR) * lastDR < 0) rollTurns++; lastDR = t.roll - lastR; lastR = t.roll; peakX = Math.max(peakX, Math.abs(t.offX)); peakB = Math.max(peakB, t.bright); }
emit({ before, flips, rollTurns, peakX, peakB });
""",
    # What each kind of line does to the inner state.
    "lines": r"""
import { tick, emit, face } from './common.mjs';
const out = {};
for (const [k, line] of Object.entries(JSON.parse(process.env.LINES))) {
  tick(60 * 90); const a = face.getTemperament().drives; const A = { ...a };
  const kinds = face.noteLogLine(line); tick(1); const b = face.getTemperament().drives;
  out[k] = { kinds, dStrain: b.strain - A.strain };
}
emit(out);
""",
    # The temperament's clock under a sustained low frame rate, then absences.
    "clock": r"""
import { face, tick, setHz, advanceClock, T, emit } from './common.mjs';
const lag = () => Date.now() - face.getTemperament().now;
setHz(15); tick(60); let maxD = 0, last = { ...T() }, maxLag = 0;
for (let f = 0; f < 15 * 180; f++) { tick(1); const c = T(); maxD = Math.max(maxD, Math.abs(c.yaw - last.yaw), Math.abs(c.pitch - last.pitch)); last = { ...c }; maxLag = Math.max(maxLag, lag()); }
setHz(60);
face.setAnimationPaused(true); advanceClock(600000); face.setAnimationPaused(false); tick(2); const afterPause = lag();
const y0 = T().yaw; advanceClock(600000); tick(2); const afterGap = lag(), yawStep = Math.abs(T().yaw - y0);
// a short gap, over the absence line but far under the frame's own limits
tick(120); const y1 = T().yaw; advanceClock(30000); tick(1); const shortGapStep = Math.abs(T().yaw - y1);
// the wall jumps and the frame stamps do not: a system sleep
tick(60); globalThis.__wallSkew = 8 * 3600e3; tick(2); const afterSleep = lag();
// the wall goes backwards over a long gap: the clock must not
tick(60); const nowBefore = face.getTemperament().now; globalThis.__wallSkew = 8 * 3600e3 - 3600e3; advanceClock(2000); tick(1);
const backward = face.getTemperament().now - nowBefore;
emit({ maxD, maxLag, afterPause, afterGap, yawStep, shortGapStep, afterSleep, backward });
""",
    # The same wall-clock moment at three display rates.
    "rates": r"""
import { tick, setHz, T, emit } from './common.mjs';
const hz = +process.env.HZ; setHz(hz); tick(hz * 60); const r = [];
for (let s = 0; s < 60; s++) { tick(hz); const t = T(); r.push([t.yaw, t.pitch, t.scaleMul]); }
emit(r);
""",
}


def _build(root: Path, static: Path) -> Path:
    MOT._write_harness(root, static / "matrix_graph.js")
    shutil.copyfile(static / "temperament.js", root / "temperament.js")
    h = (root / "harness.mjs").read_text(encoding="utf-8")
    # Date.now follows the simulated clock from a fixed afternoon: this is about
    # how the face's clock relates to the wall, its noise is a function of that
    # clock, and the night changes its behaviour (a suite run at 03:00 dozes).
    needle = "let seed = 12345;"
    assert needle in h
    h = h.replace(needle, needle + "\nconst _t0 = new Date(2026, 9, 30, 15, 0, 0).getTime(); Date.now = () => _t0 + nowMs + (globalThis.__wallSkew || 0);")
    needle = "getItem: () => process.env.FACE_FORM || null"
    assert needle in h
    h = h.replace(needle, "getItem: (k) => k === 'ghost_face_form' ? (process.env.FACE_FORM || null) : null")
    (root / "harness.mjs").write_text(h, encoding="utf-8")
    (root / "common.mjs").write_text(_COMMON, encoding="utf-8")
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def now(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("calm_now"), _STATIC)


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("calm_old"), _BACKUP)


def _run(root: Path, scenario: str, form: str = "moodtesseract", **env):
    p = subprocess.run(["node", str(root / f"{scenario}.mjs")], capture_output=True, text=True, timeout=280,
                       cwd=str(root), env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "FACE_FORM": form, **env})
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def test_ten_idle_minutes_on_the_tesseract_are_smooth(now, old):
    """World: an idle action or a twitch moves the whole scene or pops the
    centre. The form alone measures p99 0.16, worst 0.20; p99 here must stay
    close to that, and the worst frame must not be a pop."""
    r = _run(now, "idle")
    assert r["form"] == "moodtesseract"
    assert r["p99"] < 0.25 and r["max"] < 0.45, r
    base = _run(old, "idle")
    assert base["p99"] > 0.35 and base["max"] > 1.0, base


def test_the_idle_body_is_a_third_but_still_there(now, old):
    """World: the body is simply off (the operator chose calmer, not dead), or
    still the ~25° turn. The old face reached past 0.4 rad at rest."""
    r, base = _run(now, "idle"), _run(old, "idle")
    assert 0.06 < r["maxYaw"] < 0.25, r
    assert base["maxYaw"] > 0.4, base


def test_a_turn_arriving_mid_gesture_does_not_snap_the_scene(now, old):
    """World: cancelling an idle action drops its envelope to 0 in one frame.
    Was 9–37 px on the median node, against ~0.13 at rest."""
    r = _run(now, "cancel")
    assert set(r) == {"glance", "lookaround", "stretch", "doze"}
    assert max(r.values()) < 1.5, r
    base = _run(old, "cancel")
    assert min(base.values()) > 5, base


def test_a_run_of_web_rows_makes_one_tilt_not_a_flapping_head(now, old):
    """World: every 🌐 line restarts the tilt, on a random side. The reaction
    itself must survive: a real tilt still shows."""
    row = _ROW.format("🌐")
    r = _run(now, "tilt", ROW=row)
    assert r["maxStep"] < 0.01, r
    assert r["maxRoll"] > 0.05, r
    base = _run(old, "tilt", ROW=row)
    assert base["maxStep"] > 0.04, base


def test_a_real_failure_keeps_its_full_startle(now):
    """World: the easing also swallowed the reaction — a 7 Hz shake through a
    0.25 s filter is nearly gone, and the flare becomes a slow glow. This is
    the half of the operator's choice that must NOT get calmer."""
    r = _run(now, "startle")
    assert r["flips"] > 25, r            # ~7 Hz for 2.5 s
    assert r["rollTurns"] > 25, r        # the roll's shake too, not only the offset
    assert r["peakX"] > 0.04, r
    assert r["peakB"] > r["before"] + 0.4, r


def test_only_a_rows_own_icon_is_a_report(now):
    """World: free text moves the face — `ON_ERROR_STOP` in a thought, or a ✅
    list in a reply. A failure row must still strain it."""
    lines = {
        "fail_row": _ROW.format("❌"),
        "fail_row_ansi": "\x1b[38;5;34m│\x1b[0m  \x1b[38;5;34m**\x1b[0m  \x1b[2m12:00:00\x1b[0m  ❌  \x1b[2m+1.0s\x1b[0m  title  text",
        "fail_row_no_clock": "│  26  ❌  +69.7s  title  text",
        "thought_row": "│  AF  08:23:44  💭  + 145s  thinking  did ❌ happen? \\set ON_ERROR_STOP on",
        "reply_row": "│  5B  🤖  + 283s  final reply  ❌ not done, ✅ done",
        "continuation": "❌ this failed ERROR: psql exits with Exception",
        "bar_text": "│  some words about a ❌ failure",
    }
    r = _run(now, "lines", LINES=json.dumps(lines))
    for k in ("fail_row", "fail_row_ansi", "fail_row_no_clock"):
        assert r[k]["kinds"] == ["outcome:fail"] and r[k]["dStrain"] > 0.05, (k, r[k])
    assert r["thought_row"]["kinds"] == ["work:think"], r
    for k in ("reply_row", "continuation", "bar_text"):
        assert r[k]["kinds"] == [], (k, r[k])
    for k in ("thought_row", "reply_row", "continuation", "bar_text"):
        assert r[k]["dStrain"] <= 0, (k, r[k])


def test_the_clients_log_branch_reads_a_rows_icon_not_a_word(now):
    """World: `data.is_error` (the words ERROR/Exception anywhere in the line)
    or an emoji in free text still drives the face's strongest reaction or its
    working state; or the console stopped getting the flag it colours by.
    Runs the REAL ws.onmessage out of app.js."""
    app = (_STATIC / "app.js").read_text(encoding="utf-8")
    i = app.index("ws.onmessage = (event) => {")
    j = app.index("ws.onclose =", i)
    handler = app[i + len("ws.onmessage = "):j].rstrip().rstrip(";")
    (now / "client.mjs").write_text("""
import { logRowIcon } from './temperament.js';
let spikes = 0, states = [], flags = [], errs = [];
const activeFace = { logRowIcon, noteLogLine() {}, noteActivity() {}, triggerSpike() { spikes++; } };
const updateStateFromIcon = (i) => states.push(i), pushLogEntry = (c, e) => flags.push(e);
const noteTickerLine = () => {}, getIconColor = () => '#000', showPlannerMonologue = () => {};
const extractIcon = () => { throw new Error('free-text icon read'); };
console.error = (...a) => errs.push(String(a));
const onmessage = """ + handler + """;
const out = {};
for (const [k, [content, is_error]] of Object.entries(JSON.parse(process.env.LINES))) {
  spikes = 0; states = []; flags = [];
  onmessage({ data: JSON.stringify({ type: 'log', content, is_error }) });
  out[k] = { spikes, states, flags };
}
console.log(JSON.stringify({ out, errs }));
""", encoding="utf-8")
    lines = {
        "fail": [_ROW.format("❌"), False],
        "stop": [_ROW.format("🛑"), False],
        "warn_flagged": ["│  26  🔶  +69.7s  tool warning  web_search -> ERROR: zero results", True],
        "thought_flagged": ["│  AF  08:23:44  💭  + 145s  thinking  \\set ON_ERROR_STOP on", True],
        "text_flagged": ["❌ psql ERROR: stopped", True],
        "text_ok": ["✅ one item of a list", False],
    }
    r = _run(now, "client", LINES=json.dumps(lines))
    assert r["errs"] == [], r
    o = r["out"]
    assert o["fail"]["spikes"] == 1 and o["stop"]["spikes"] == 1, o
    for k in ("warn_flagged", "thought_flagged", "text_flagged", "text_ok"):
        assert o[k]["spikes"] == 0, (k, o[k])
    assert o["fail"]["states"] == ["❌"] and o["thought_flagged"]["states"] == ["💭"], o
    assert o["text_flagged"]["states"] == [] and o["text_ok"]["states"] == [], o
    assert o["warn_flagged"]["flags"] == [True] and o["fail"]["flags"] == [False], o


def test_a_low_frame_rate_does_not_lurch_and_an_absence_still_reads(now, old):
    """World 1: below 20 fps the clock falls behind and re-syncs 5 s at a time
    (every ~20 s), so every noise channel jumps. World 2: fixing that also lost
    the re-sync after a paused tab or a sleeping laptop, so a real absence
    never reads as one."""
    r = _run(now, "clock")
    assert r["maxLag"] < 1000, r
    assert r["maxD"] < 0.02, r
    assert abs(r["afterPause"]) < 200 and abs(r["afterGap"]) < 200, r
    assert r["yawStep"] < 0.02, r        # even the re-sync eases the scene
    assert r["shortGapStep"] < 0.02, r   # a 30 s gap is an absence too, not a long frame
    assert abs(r["afterSleep"]) < 200, r
    assert 0 <= r["backward"] < 100, r
    base = _run(old, "clock")
    assert base["maxLag"] > 4000, base


def test_the_eased_body_is_the_same_at_30_60_and_120_hz(now):
    """World: the easing is per frame (`x += (t − x) · 0.07`), which makes the
    whole body twice as fast on a 120 Hz display."""
    a, b, c = (_run(now, "rates", HZ=str(h)) for h in (30, 60, 120))
    d = 0.0
    for ra, rb, rc in zip(a, b, c):
        for k in range(3):
            d = max(d, abs(ra[k] - rb[k]), abs(rc[k] - rb[k]))
    assert d < 0.012, d


_TEMPER = """
import * as M from './temperament.js';
const t0 = new Date(2026, 9, 30, 15, 0, 0).getTime();
const emit = (x) => console.log(JSON.stringify(x));
"""


def _temper(root: Path, body: str):
    f = root / "temper_case.mjs"
    f.write_text(_TEMPER + body, encoding="utf-8")
    p = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=120, cwd=str(root))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_every_idle_channel_is_at_a_third_not_only_the_turn(now, old):
    """World: yaw got its third, while scale, the sink and the doze's dimming
    still run at full size. Hours of afternoon rest, every action seen."""
    body = """
const T = M.createTemperament({ seed: 5, now: t0, lastTurnEndAt: t0 }); let t = t0, scale = 0, off = 0, dim = 9, pitch = 0; const seen = new Set();
for (let i = 0; i < 40 * 7200; i++) { t += 25; const o = T.step(0.025, t);
  if (o.idle) seen.add(o.idle); scale = Math.max(scale, Math.abs(o.scaleMul - 1)); off = Math.max(off, Math.abs(o.offY)); dim = Math.min(dim, o.bright); pitch = Math.max(pitch, Math.abs(o.pitch)); }
emit({ scale, off, dim, pitch, seen: [...seen] });
"""
    r, base = _temper(now, body), _temper(old, body)
    assert {"stretch", "sigh", "doze"} <= set(r["seen"]), r
    assert r["scale"] < 0.10 and r["off"] < 0.11 and r["dim"] > 0.75 and r["pitch"] < 0.15, r
    assert base["scale"] > 0.15 and base["off"] > 0.2 and base["dim"] < 0.6 and base["pitch"] > 0.2, base


def test_a_cue_during_a_tilt_does_not_restart_it(now):
    """World: the easing hides the steps of a restarted tilt, but the head
    stays up for as long as the cues come. Cues every 0.5 s for 2 s: the tilt
    must still be ending on schedule; and a later cue starts a new one."""
    r = _temper(now, """
const T = M.createTemperament({ seed: 9, now: t0, lastTurnEndAt: t0 }); let t = t0; const run = (s) => { let o; for (let i = 0; i < s * 60; i++) { t += 1000 / 60; o = T.step(1 / 60, t); } return o; };
run(5); let peak = 0;
for (let k = 0; k < 4; k++) { T.event('work:web'); for (let i = 0; i < 30; i++) { t += 1000 / 60; peak = Math.max(peak, Math.abs(T.step(1 / 60, t).roll)); } }
const end = Math.abs(run(0.9).roll);
run(30); T.event('work:web'); const again = Math.max(...[...Array(60)].map(() => { t += 1000 / 60; return Math.abs(T.step(1 / 60, t).roll); }));
emit({ peak, end, again });
""")
    assert r["peak"] > 0.05, r
    assert r["end"] < 0.45 * r["peak"], r
    assert r["again"] > 0.05, r


def test_reactions_keep_their_strength_through_the_easing(now):
    """World: the easing is slow enough to eat them (a 1 s constant), or the
    speech nod went through it and halved (0.084 → 0.042 peak to peak)."""
    r = _temper(now, """
const T = M.createTemperament({ seed: 3, now: t0, lastTurnEndAt: t0 }); let t = t0; const stepf = () => { t += 1000 / 60; return T.step(1 / 60, t); };
for (let i = 0; i < 300; i++) stepf();
T.event('turn:start'); for (let i = 0; i < 360; i++) stepf(); const coil = T.step(0, t).scaleMul;
T.ingestChunk('Hello'); let release = 0; for (let i = 0; i < 90; i++) { T.ingestChunk('word '); release = Math.max(release, stepf().scaleMul); }
let lo = 9, hi = -9; for (let i = 0; i < 54; i++) { T.ingestChunk('word '); const p = stepf().pitch; lo = Math.min(lo, p); hi = Math.max(hi, p); }
emit({ coil, release, nod: hi - lo });
""")
    assert r["coil"] < 0.93, r
    assert r["release"] - r["coil"] > 0.11, r
    assert r["nod"] > 0.06, r


def test_the_rollback_copy_is_the_pre_fix_face():
    """The operator's rollback habit; also the negative control above."""
    for name in ("matrix_graph.js", "temperament.js", "app.js", "index.html"):
        assert (_BACKUP / name).is_file(), name
    assert "IDLE_BODY" not in (_BACKUP / "temperament.js").read_text(encoding="utf-8")
