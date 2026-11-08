"""Idle → busy without a shake or a pop — EXECUTED (2026-11-08).

Operator: "web UI face still is jidderish when it transits from idle to busy"
(shaking, Mac browser). Measured on the real ``animate()`` projected to
1440×900: at steady and irregular frame rates the busy motion itself is smooth
(a quadratic fit over nine frames leaves ~0.001 px). The shake was the
temperament's ``wake``: a turn that starts more than an hour after the last
one ended nudged ``wake``, which was in ``STARTLE`` — a ~7 Hz shake of the
whole scene (roll + offset), a flare and a flinch for ~2 s. In the web tab an
hour between messages is the common case, so a message sent more than an hour after
the last one (about 18 % of real turns, with the tab open that long) began
with a shake. The wake keeps its arousal nudge and its smooth
3 s lift/brighten; a real error still startles (idle-calm test).

The scenario also runs on the pre-fix copy (``face_backups/20261108-prewakecalm``)
where it must FAIL (R6).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import tests.test_interface_face_idle_calm_2026_11_06 as CALM

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_BACKUP = _ROOT / "interface" / "face_backups" / "20261108-prewakecalm"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

# High-frequency shake: each visible node's residual against a 9-frame
# Savitzky–Golay quadratic (passes smooth acceleration, keeps > ~5 Hz),
# median over nodes, per frame. Rest is spent at 1 frame/s so 90 minutes run
# in seconds; the clock (Date.now) follows the simulated time.
_WAKE = r"""
import { tick, px, emit, face, advanceClock } from './common.mjs';
const SG = [-21, 14, 39, 54, 59, 54, 39, 14, -21].map(c => c / 231);
function rec(n, hook) { const fr = []; for (let f = 0; f < n; f++) { if (hook) hook(f); tick(1); fr.push(px()); } return fr; }
function hf(fr) { const per = []; for (let t = 4; t < fr.length - 4; t++) { const r = [];
  for (let i = 0; i < fr[t].length; i++) { let ok = true, sx = 0, sy = 0;
    for (let k = -4; k <= 4; k++) { const p = fr[t + k][i]; if (!p) { ok = false; break; } sx += SG[k + 4] * p[0]; sy += SG[k + 4] * p[1]; }
    if (!ok || Math.abs(fr[t][i][0] - fr[t - 1][i][0]) > 40) continue;
    r.push(Math.hypot(fr[t][i][0] - sx, fr[t][i][1] - sy)); }
  r.sort((a, b) => a - b); per.push(r.length ? r[r.length >> 1] : 0); }
  return { mean: per.reduce((a, b) => a + b, 0) / per.length, max: Math.max(...per) }; }
const away = +process.env.AWAY_MIN;
tick(1200);
for (let m = 0; m < away * 60; m++) { advanceClock(1000 - 1000 / 60); tick(1); }
tick(300);
const a0 = face.getTemperament().drives.arousal;
let peakScale = 0, peakBright = 0;
const fr = rec(240, f => { if (f === 0) { face.setWorkingState(true); face.setUserTurn(true); }
  const o = face.getTemperament().out; peakScale = Math.max(peakScale, o.scaleMul); peakBright = Math.max(peakBright, o.bright); });
emit({ away, ...hf(fr.slice(0, 90)), arousalJump: face.getTemperament().drives.arousal - a0, peakScale, peakBright });
"""


# A turn whose first seconds carry a few log rows and two tool calls (a third
# of real turns show a 🧰 row in their first 15 s). The 95th percentile over
# nodes: the kernel and corner sparks are a minority the median cannot see.
_TOOL = r"""
import { tick, px, emit, face } from './common.mjs';
const SG = [-21, 14, 39, 54, 59, 54, 39, 14, -21].map(c => c / 231);
const row = (ic) => '│  50  12:00:00  ' + ic + '  +1.0s  some title   some text';
tick(1200); tick(300);
const fr = []; let peakFlash = 0;
for (let f = 0; f < 150; f++) {
  if (f === 0) { face.setWorkingState(true); face.setUserTurn(true); }
  if (f === 2 || f === 6 || f === 20 || f === 40) { face.noteLogLine(row(f === 2 ? '💬' : '🧠')); face.noteActivity(0.16); }
  if (f === 45 || f === 60) face.noteToolCall();
  tick(1); fr.push(px()); peakFlash = Math.max(peakFlash, face.getDebugState().gaitFlash);
}
let p95 = 0;
for (let t = 4; t < fr.length - 4; t++) { const r = [];
  for (let i = 0; i < fr[t].length; i++) { let ok = true, sx = 0, sy = 0;
    for (let k = -4; k <= 4; k++) { const p = fr[t + k][i]; if (!p) { ok = false; break; } sx += SG[k + 4] * p[0]; sy += SG[k + 4] * p[1]; }
    if (!ok || Math.abs(fr[t][i][0] - fr[t - 1][i][0]) > 40) continue;
    r.push(Math.hypot(fr[t][i][0] - sx, fr[t][i][1] - sy)); }
  r.sort((a, b) => a - b); if (r.length) p95 = Math.max(p95, r[Math.floor(r.length * 0.95)]); }
emit({ p95, peakFlash });
"""


def _build(root: Path, static: Path) -> Path:
    CALM._build(root, static)
    (root / "wake.mjs").write_text(_WAKE, encoding="utf-8")
    (root / "tool.mjs").write_text(_TOOL, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def now(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("wake_now"), _STATIC)


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("wake_old"), _BACKUP)


def test_the_first_turn_after_an_hour_starts_as_smooth_as_after_five_minutes(now, old):
    """World: the wake startle — the first ~1.5 s of a turn after 90 minutes
    away shakes at ~7 Hz (pre-fix: mean 0.38 px, worst 0.87 px; after 5
    minutes away: 0.0003 / 0.003)."""
    near = CALM._run(now, "wake", AWAY_MIN="5")
    far = CALM._run(now, "wake", AWAY_MIN="90")
    assert near["max"] < 0.02 and far["max"] < 0.02, (near, far)
    assert far["mean"] < 3 * near["mean"] + 0.002, (near, far)
    shook = CALM._run(old, "wake", AWAY_MIN="90")
    assert shook["max"] > 0.3 and shook["mean"] > 0.1, shook        # the harness sees the defect


def test_waking_still_reads_as_waking(now):
    """World: the fix took the whole wake out — the first turn after an hour
    looks like any other. It must still lift arousal more and brighten/swell
    (the smooth 3 s `wakeAt` pulse)."""
    near = CALM._run(now, "wake", AWAY_MIN="5")
    far = CALM._run(now, "wake", AWAY_MIN="90")
    assert far["arousalJump"] > near["arousalJump"] + 0.06, (near, far)
    assert far["peakBright"] > near["peakBright"] + 0.1, (near, far)
    assert far["peakScale"] > near["peakScale"] + 0.02, (near, far)


def test_a_tool_call_flashes_without_a_one_frame_pop(now, old):
    """World: the tool flash steps the kernel, corner sparks, node size and
    line bursts in ONE frame (this scenario, pre-fix: worst 95th-percentile node
    residual 4.5 px; fixed: 0.7). The follower rises over ~0.15 s; the flash
    must still be there."""
    r = CALM._run(now, "tool")
    assert r["p95"] < 1.0, r
    assert r["peakFlash"] > 0.2, r                                   # still a visible flash
    o = CALM._run(old, "tool")
    assert o["p95"] > 2.0, o                                         # the harness sees the pop


def test_no_visual_reads_the_one_frame_tool_pulse():
    """World: a consumer the node positions cannot show — pulseT (node size and
    light, the line bursts' phase) or the bloom — goes back to `toolPulse` and
    pops again while the position test still passes. `toolPulse` may only be
    written, decayed, followed, returned and reported."""
    import re
    src = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    reads = [m.start() for m in re.finditer(r"\btoolPulse\b", code)]
    assert len(reads) >= 8, len(reads)          # a rename must not pass the check vacuously
    allowed = ("let toolPulse = 0.0;", "toolPulse = Math.min(1.0, toolPulse + 0.6 * temper.event('tool'));",
               "return toolPulse;", "phase, gait: { ...gait }, toolPulse, toolFlash, recallSpark, verdict, verdictEnv,",
               "toolPulse *= decay(0.93);",
               "if (toolPulse < 0.003) toolPulse = 0;", "toolFlash += (toolPulse - toolFlash) * ease(TOOL_FLASH_RISE);",
               "if (toolFlash < 0.003 && toolPulse === 0) toolFlash = 0;")
    lines = [code[code.rfind("\n", 0, i) + 1:code.find("\n", i)].strip() for i in reads]
    # each line EQUAL to an allowed statement (review: a substring match let an
    # alias ride on an allowed line — `toolPulse *= decay(0.93); const _tp = toolPulse;`)
    assert set(lines) <= set(allowed), sorted(set(lines) - set(allowed))
    # and each allowed line appears ONCE (round 2: a second `return toolPulse;`
    # in a new getter passed a set comparison)
    stmt = [ln.strip() for ln in code.splitlines()]
    for a in allowed:
        assert stmt.count(a) == 1, (a, stmt.count(a))
    # the handheld serves the same face (round 2: only the web copy was scanned)
    hand = _STATIC.parent / "externals" / "clockwork_ghost" / "webface" / "matrix_graph.js"
    import filecmp
    assert filecmp.cmp(hand, _STATIC / "matrix_graph.js", shallow=False), hand
    for needle in ("pulseT + 0.6 * toolFlash + 0.15 * c2n", "+ 0.20 * toolFlash + 0.25 * recallSpark",
                   "gaitFlash = DIAL.tool === 'flash' ? toolFlash : 0;",
                   "TUNE.toolKick * toolFlash : 0",
                   "toolFlash += (toolPulse - toolFlash) * ease(TOOL_FLASH_RISE);"):
        assert needle in code, needle


def test_the_rollback_copy_is_the_pre_fix_face():
    for name in ("matrix_graph.js", "temperament.js", "app.js", "index.html"):
        assert (_BACKUP / name).is_file(), name
    assert "'wake': 0.5 }" in (_BACKUP / "temperament.js").read_text(encoding="utf-8")
    assert "toolFlash" not in (_BACKUP / "matrix_graph.js").read_text(encoding="utf-8")
