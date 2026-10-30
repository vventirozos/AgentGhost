"""The face's inner state — interface/static/temperament.js (2026-10-30).

Operator: "make the faces more reactive to the agent, more interesting, less
predictable". Every signal used to map to ONE fixed animation; the temperament
sits between the events and the animation: drives with momentum, habituation,
surprise, bounded noise, session memory (fatigue, night, waking after an
absence) and an idle repertoire. These tests EXECUTE the module under node
(and the real matrix_graph.js under the face harness for the wiring); each
names the world in which it fails.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is required (see tests/helpers.eval_js)")

# 2026-10-30 15:00 local and 03:00 local, as epoch ms (built in node, local tz)
_PRELUDE = """
import * as M from './temperament.js';
const at = (h) => { const d = new Date(2026, 9, 30, h, 0, 0); return d.getTime(); };
const mk = (h = 15, seed = 7, away = 0) => M.createTemperament({ seed, now: at(h), lastTurnEndAt: at(h) - away });
function run(T, seconds, hz = 60, t0) { let o; const n = Math.round(seconds * hz); let t = t0;
  for (let i = 0; i < n; i++) { t += 1000 / hz; o = T.step(1 / hz, t); } return { o, t }; }
const emit = (x) => console.log(JSON.stringify(x));
"""


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("temperament")
    shutil.copyfile(_STATIC / "temperament.js", root / "temperament.js")
    (root / "package.json").write_text('{ "type": "module" }\n')
    return root


def _node(lab: Path, body: str):
    f = lab / "case.mjs"
    f.write_text(_PRELUDE + body)
    p = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=120, cwd=str(lab))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_repeats_habituate_and_quiet_restores(lab):
    """World: every tool call kicks the same — the 12th web search looks
    exactly like the first (the predictability the operator named)."""
    r = _node(lab, """
const T = mk(); let t = at(15); const g = [];
for (let i = 0; i < 12; i++) { g.push(T.event('tool')); ({ t } = run(T, 0.5, 60, t)); }
({ t } = run(T, 180, 60, t)); const back = T.event('tool');
emit({ g, back });
""")
    g = r["g"]
    assert g[0] > 1.0 and sum(g[-4:]) / 4 < 0.45 * g[0], g
    assert r["back"] > 2 * (sum(g[-4:]) / 4), r          # three quiet minutes restore it
    assert all(0.25 <= x <= 1.8 for x in g), g             # bounded either way


def test_no_two_reactions_are_identical_but_the_character_holds(lab):
    """World: noise-free gains (a fixed envelope) or unbounded ones."""
    r = _node(lab, """
const gs = []; for (let s = 1; s <= 40; s++) { const T = mk(15, s); gs.push(T.event('error')); }
emit({ gs });
""")
    gs = r["gs"]
    assert len({round(x, 4) for x in gs}) >= 35, gs
    assert min(gs) >= 0.25 and max(gs) <= 1.8
    assert max(gs) / min(gs) < 2.6, gs                     # same character: ~×0.7–1.4 around one level


def test_rare_events_surprise_more_than_routine_ones(lab):
    """World: a refute after hours reacts like the 30th one in a minute."""
    r = _node(lab, """
const A = mk(); const first = A.event('verdict:refute');
const B = mk(); let t = at(15);
for (let i = 0; i < 30; i++) { B.event('verdict:refute'); ({ t } = run(B, 2, 60, t)); }
({ t } = run(B, 25, 60, t));   // habituation mostly recovered (~20 s), the RATE is still high
emit({ first, routine: B.event('verdict:refute') });
""")
    assert r["first"] > 1.4 * r["routine"], r


def test_rest_is_the_old_baseline_in_daytime(lab):
    """World: the temperament silently re-tunes the face the operator already
    settled (speed and amplitude at rest must stay ≈ ×1)."""
    r = _node(lab, """
const T = mk(15); const { o } = run(T, 5, 60, at(15));
emit({ speed: o.speedMul, amp: o.ampMul, still: o.stillness, lean: o.lean });
""")
    assert 0.88 <= r["speed"] <= 1.12 and 0.8 <= r["amp"] <= 1.1, r
    assert r["still"] == 0 and r["lean"] == 0, r


def test_the_night_is_slower_and_dozes_more(lab):
    r = _node(lab, """
const count = (h) => { const T = mk(h, 3); let t = at(h), doze = 0, sp = 0, n = 0;
  for (let s = 0; s < 1200; s++) { const r = run(T, 1, 30, t); t = r.t; if (r.o.idle === 'doze') doze++; sp += r.o.speedMul; n++; }
  return { doze, speed: sp / n }; };
emit({ day: count(15), night: count(3) });
""")
    assert r["night"]["speed"] < r["day"]["speed"] - 0.05, r
    assert r["night"]["doze"] > 2 * max(1, r["day"]["doze"]), r


def test_the_idle_repertoire_varies_and_never_repeats_back_to_back(lab):
    """World: one idle loop (what 'rest' used to be), or the same action twice running."""
    r = _node(lab, """
const T = mk(15, 11); let t = at(15); const seq = []; let cur = null, busy = 0;
for (let s = 0; s < 4 * 1200; s++) { const r = run(T, 0.25, 40, t); t = r.t;
  if (r.o.idle && r.o.idle !== cur) seq.push(r.o.idle); cur = r.o.idle; }
T.event('turn:start'); for (let s = 0; s < 240; s++) { const r = run(T, 0.25, 40, t); t = r.t; if (r.o.idle) busy++; }
emit({ seq, busy });
""")
    seq = r["seq"]
    assert len(set(seq)) >= 5 and len(seq) >= 30, seq
    assert all(a != b for a, b in zip(seq, seq[1:])), seq
    assert r["busy"] == 0, "no idle acting while a turn runs"


def test_waiting_builds_anticipation_and_the_first_words_release_it(lab):
    r = _node(lab, """
const T = mk(); let t = at(15); T.event('turn:start');
let a = run(T, 6, 60, t); t = a.t; const waiting = a.o.anticipation, lean = a.o.lean;
T.ingestChunk('<think>let me see'); a = run(T, 4, 60, t); t = a.t; const thinking = a.o.thinking, still = a.o.stillness, antThink = a.o.anticipation;
T.ingestChunk('</think>Here'); let o;
for (let i = 0; i < 180; i++) { T.ingestChunk('word '); t += 1000 / 60; o = T.step(1 / 60, t); }
emit({ waiting, lean, thinking, still, antThink, speech: o.speech, antAfter: o.anticipation, thinkAfter: o.thinking });
""")
    assert r["waiting"] > 0.4 and r["lean"] > 0.4, r
    assert r["thinking"] > 0.8 and r["still"] > 0.2, r          # deep thought reads as stillness
    assert r["speech"] > 0.6 and r["antAfter"] < 0.1 and r["thinkAfter"] < 0.2, r


def test_a_long_absence_wakes_it(lab):
    r = _node(lab, """
const near = mk(15, 5, 2 * 60e3); near.event('turn:start');
const far = mk(15, 5, 6 * 3600e3); far.event('turn:start');
emit({ near: near.debug().drives.arousal, far: far.debug().drives.arousal });
""")
    assert r["far"] > r["near"] + 0.15, r


def test_an_error_startles_once_and_failures_strain(lab):
    r = _node(lab, """
const T = mk(); let t = at(15); T.event('error');
const a = { ...T.step(1 / 60, t += 16) }, b = { ...T.step(1 / 60, t += 16) };   // step() reuses one object
const U = mk(); U.ingestLog('12:01 🐍 Sandbox Exec · python x.py ❌ exit 1');
emit({ k1: a.startleKick, k2: b.startleKick, jitter: a.jitter, strain: U.debug().drives.strain, valence: U.debug().drives.valence });
""")
    assert r["k1"] > 0.5 and r["k2"] == 0, r                    # the kick is consumed once
    assert r["jitter"] > 0.5 and r["strain"] > 0.1 and r["valence"] < 0, r


def test_wander_is_the_same_at_30_60_and_120_hz(lab):
    """World: a per-frame random walk — the face drifts differently on a
    120 Hz display (the motion contract every face clock is held to)."""
    r = _node(lab, """
const at60 = (hz) => { const T = mk(15, 9); let t = at(15); const out = [];
  for (let s = 0; s < 60; s++) { const r = run(T, 1, hz, t); t = r.t; if (s === 20) T.event('tool'); out.push([r.o.gazeX, r.o.gazeY, r.o.speedMul]); }
  return out; };
const a = at60(30), b = at60(60), c = at60(120);
let d = 0; for (let i = 0; i < a.length; i++) for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a[i][k] - b[i][k]), Math.abs(c[i][k] - b[i][k]));
emit({ d });
""")
    assert r["d"] < 0.01, r


@pytest.mark.parametrize("line,expect", [
    ("🌐 Web Search · postgres 18 release", ["work:web"]),
    ("🐍 Sandbox Exec · python x.py ❌ exit 1", ["work:code", "outcome:fail"]),
    ("📀 head disk — restored the head in 0.5s", ["outcome:restore"]),
    ("🔄 retry 2/3", ["outcome:retry"]),
    ("plain line, no icon", []),
])
def test_log_lines_say_what_ran_and_how_it_went(lab, line, expect):
    r = _node(lab, f"emit(M.classifyLogLine({json.dumps(line)}, false));")
    assert r == expect


def test_an_error_flag_counts_as_a_failure(lab):
    assert _node(lab, "emit(M.classifyLogLine('something broke', true));") == ["outcome:fail"]


# ── the wiring: the REAL matrix_graph.js under the face harness ──────────

@pytest.fixture(scope="module")
def face_lab(tmp_path_factory):
    import tests.test_interface_face_motion_2026_09_20 as MOT
    root = tmp_path_factory.mktemp("temperface")
    MOT._write_harness(root, _STATIC / "matrix_graph.js")
    (root / "wire.mjs").write_text("""
import { face, tick } from './harness.mjs';
tick(120);
const r0 = face.getTemperament().out;
face.noteLogLine('🐍 Sandbox Exec ❌ exit 1', false); tick(2);
const strain = face.getTemperament().drives.strain;
face.setUserTurn(true); tick(360);
const waiting = face.getTemperament().out.anticipation;
face.noteStreamChunk('<think>hmm'); tick(240);
const thinking = face.getTemperament().out.thinking;
face.noteStreamChunk('</think>Hi'); for (let i = 0; i < 120; i++) { face.noteStreamChunk('word '); tick(1); }
const speech = face.getTemperament().out.speech;
// the tool kick is scaled by the gain: twelve in a row end smaller than the first
const k = []; for (let i = 0; i < 12; i++) { const before = face.getDebugState().toolPulse; face.noteToolCall(); k.push(face.getDebugState().toolPulse - before); tick(40); }
face.setUserTurn(false); tick(2);
console.log(JSON.stringify({ restSpeed: r0.speedMul, strain, waiting, thinking, speech, k, inTurn: face.getTemperament().inTurn }));
""")
    return root


def test_matrix_graph_feeds_and_reads_the_temperament(face_lab):
    """World: the module exists but nothing feeds it (a built-but-unwired
    loop), or the kicks ignore its gain."""
    p = subprocess.run(["node", str(face_lab / "wire.mjs")], capture_output=True, text=True,
                       timeout=120, cwd=str(face_lab))
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert 0.85 <= r["restSpeed"] <= 1.15, r
    assert r["strain"] > 0.1 and r["waiting"] > 0.3 and r["thinking"] > 0.7 and r["speech"] > 0.5, r
    assert r["k"][-1] < 0.7 * r["k"][0], r["k"]
    assert r["inTurn"] is False


def test_app_js_feeds_log_lines_and_stream_chunks():
    """World: the face exports the hooks but the client never calls them."""
    app = (_STATIC / "app.js").read_text(encoding="utf-8")
    i = app.index("noteTickerLine(data.content);")
    assert "activeFace.noteLogLine(data.content, data.is_error)" in app[i:i + 400]
    j = app.index("chunkContent = data.choices[0].delta.content;")
    assert "activeFace.noteStreamChunk(chunkContent)" in app[j:j + 300]


def test_the_rollback_copy_is_kept():
    """The operator asked for backups to roll back to if the change is not liked."""
    b = _ROOT / "interface" / "face_backups" / "20261030-pretemperament"
    for name in ("matrix_graph.js", "app.js", "index.html"):
        assert (b / name).is_file(), name
    assert "temperament" not in (b / "matrix_graph.js").read_text(encoding="utf-8")


# ── v1.2: the BODY (operator: "can't say i saw a huge difference") ───────

def test_the_body_moves_visibly_and_intensity_zero_is_the_quiet_face(lab):
    """World: v1.1 — every effect within a few percent, and a gaze the
    camera-static forms never read. The body must turn the head by a visible
    angle, coil while waiting, let go on the first words, flare on a startle
    and sink into a doze; `intensity: 0` must give back a neutral body."""
    r = _node(lab, """
function arc(intensity) {
  const T = M.createTemperament({ seed: 21, now: at(15), lastTurnEndAt: at(15), intensity });
  let t = at(15), maxYaw = 0, minS = 9, maxS = 0, maxB = 0, sink = 0, rest;
  rest = { ...run(T, 2, 60, t).o }; t += 2000;
  T.event('turn:start'); let r = run(T, 6, 60, t); t = r.t; minS = Math.min(minS, r.o.scaleMul);
  T.ingestChunk('Hello'); for (let i = 0; i < 40; i++) { t += 1000 / 60; const o = T.step(1 / 60, t); maxS = Math.max(maxS, o.scaleMul); }
  T.event('error'); for (let i = 0; i < 20; i++) { t += 1000 / 60; const o = T.step(1 / 60, t); maxB = Math.max(maxB, o.bright); }
  T.event('turn:end');
  for (let s = 0; s < 4 * 600; s++) { const q = run(T, 0.25, 40, t); t = q.t; maxYaw = Math.max(maxYaw, Math.abs(q.o.yaw)); sink = Math.min(sink, q.o.offY); }
  return { rest, minS, maxS, maxB, maxYaw, sink };
}
emit({ on: arc(undefined), off: arc(0) });
""")
    on, off = r["on"], r["off"]
    assert on["maxYaw"] > 0.25, on                      # > ~14° head turn at rest
    assert on["minS"] < 0.95 and on["maxS"] > 1.04, on  # coil, then release
    assert on["maxB"] > 1.3, on                          # a startle flares
    assert abs(on["rest"]["scaleMul"] - 1) < 0.03 and abs(on["rest"]["yaw"]) < 0.1, on["rest"]
    for k in ("yaw", "pitch", "roll", "offX", "offY"):
        assert off["rest"][k] == 0, (k, off["rest"])
    assert off["maxYaw"] == 0 and off["minS"] == 1 and off["maxS"] == 1 and off["maxB"] == 1, off


def test_the_body_never_double_applies_through_a_form_switch(face_lab):
    """World: setForm snapshotted the scene WITH the body on it, so the first
    blend frame applied the body twice (a 0.11 scale jump, measured)."""
    (face_lab / "switch.mjs").write_text("""
import { face, tick, scene } from './harness.mjs';
tick(60); face.setUserTurn(true); tick(400);            // coiled: scaleMul well below 1
let prev = scene.scale.x, maxJump = 0;
face.setForm(face.getForm() === 'cube' ? 'vortex' : 'cube');
for (let f = 0; f < 120; f++) { tick(1); maxJump = Math.max(maxJump, Math.abs(scene.scale.x - prev)); prev = scene.scale.x; }
console.log(JSON.stringify({ maxJump, body: face.getDebugState().body }));
""")
    p = subprocess.run(["node", str(face_lab / "switch.mjs")], capture_output=True, text=True, timeout=120, cwd=str(face_lab))
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["body"]["s"] < 0.97, r          # the body was really on
    assert r["maxJump"] < 0.02, r
