"""cube2 v2 — the seven upgrades, EXECUTED (2026-09-21).

Operator, after the defect hunt: "fix them all" — build every suggestion.
Headless measurement had said why cube2 read flat: idle and busy were
pixel-identical (lit 4.2% both), nested wireframes with nothing happening
ON them. Each mechanism below is driven through the real ``animate()``
under node (the §4JA harness) and read back from positions, line buffers,
uniforms and ``getDebugState().tesseract``:

  1 ties        8 corner-to-corner lines between consecutive shells (by
                depth), skipping the wrap pair — one tunnel, not frames
  2 torsion     the kernel lattice spins on an integrated angle; a newborn
                shell inherits it folded to ±45° (a cube's own symmetry)
                and unwinds as it grows — bounded, no uptime growth
  3 runners     every edge carries its own packet phase; a tool call
                bursts the packets (uPulseT) and flashes the corners
  4 heat wave   hottest at birth, a pulse of warmth racing outward on the
                breath clock, stronger on a user turn
  5 scan        the search gait's ring lights the LINES it crosses
                (uLineSweep, reach 7 — the stack is deeper than 2.4)
  6 kernel      a 3×3×3 lattice (2×2×2 mobile) with explicit edges, no
                proximity tangle
  7 verdicts    pass: wobble collapses + twists unwind fast; refute: a
                shudder on the wobble; stop: the shared exhale

v3 (same day, operator: "kinda boring" — functionally better, visually
flat): the shells are LINE-ONLY so the stack is 24 deep (12 mobile) with
corner nodes on every other shell; every face carries a "+" ruling; the
freed nodes are a STREAM riding the expansion at spread speeds on a
spiral (the vortex's swallow inside the cube); the corridor rolls slowly
(integrated); the kernel is no longer centre-dimmed. Headless lit
fraction 4.2% → 7.9% at rest, 9.6% on a turn.

World where each pin fails: a tie/kernel edge stops being emitted, the
kernel stops spinning or the newborn stops inheriting, the twist escapes
its bound or grows with uptime, the packet phases collapse to one, a tool
call stops flashing, the heat stops rising on a turn, the scan uniforms
stop reaching the lines, the verdict grammar goes silent, or a shell
teleports.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests import test_interface_face_motion_2026_09_20 as M

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_GRAPH = Path(os.environ.get("GHOST_FACE_GRAPH") or (_STATIC / "matrix_graph.js"))
_BACKUP = _STATIC / "matrix_graph.js.bak-20260921-precube2v2"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_SCENARIOS = {
    "mech": r"""
import { face, tick, run, snapshot, lines, scene, maxOf, p95Of, emit } from './harness.mjs';
face.setForm('tesseract'); tick(300);
const d0 = face.getDebugState(); const sceneZ0 = scene.rotation.z;
const rest = { lines: { ...d0.tesseract.lines }, wob: d0.tesseract.wob, spin: d0.tesseract.kernelSpin,
               uLineSweep: lines.material.uniforms.uLineSweep.value, reach: lines.material.uniforms.uSweepReach.value,
               seedsMax: Math.max(...d0.tesseract.seeds), twMax: Math.max(...d0.tesseract.tw.map(Math.abs)) };
const idle = run(1200);
const spin1 = face.getDebugState().tesseract.kernelSpin;
const turn = run(900, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } });
const dT = face.getDebugState();
const twAbs = dT.tesseract.tw.map(Math.abs);
const uv = lines.geometry.attributes.aLightPass.array; const bases = new Set();
for (let e = 0; e < 12; e++) bases.add(Math.round(uv[(e * 6) * 2] * 1000) / 1000);
face.noteToolCall(); tick(3);
const dTool = face.getDebugState();
const cornerSizesTool = snapshot().slice(0, 8).map(p => p.s);
tick(400);
face.noteVerdict('pass'); tick(5); const wobPass = face.getDebugState().tesseract.wob;
const pA = face.getDebugState().tesseract; tick(60); const pB = face.getDebugState().tesseract;
// the unwind ratio over the SAME shells (a shell born inside the window carries a fresh twist)
let num = 0, den = 0; for (let k = 0; k < pA.tw.length; k++) { if (pB.d[k] > pA.d[k] && Math.abs(pA.tw[k]) > 0.02) { num += Math.abs(pB.tw[k]); den += Math.abs(pA.tw[k]); } }
const twA = den, twB = num;
tick(400); face.noteVerdict('refute'); let shudder = 0; for (let i = 0; i < 12; i++) { tick(1); shudder = Math.max(shudder, Math.abs(face.getDebugState().tesseract.wob - rest.wob)); }
tick(400);
face.setPhase('search'); tick(120); const sweepOn = lines.material.uniforms.uSweep.value; face.setPhase(null);
face.setUserTurn(false); face.setWorkingState(false); face.noteVerdict('stop'); tick(600);
const snap = snapshot(); let nan = 0; for (const p of snap) if (p && !Number.isFinite(p.x + p.y + p.z + p.s)) nan++;
emit({ rest, idleP95: p95Of(idle), turnP95: p95Of(turn), turnMax: maxOf(turn),
       spinIdleRate: (spin1 - d0.tesseract.kernelSpin) / 20, spinTurnRate: (dT.tesseract.kernelSpin - spin1) / 15,
       twMaxTurn: Math.max(...twAbs), twNonZero: twAbs.filter(t => t > 0.05).length,
       seedsTurnMax: Math.max(...dT.tesseract.seeds), edgePhaseBases: bases.size,
       flashTool: dTool.gaitFlash, cornerSizesTool, wobPass, twPassDecay: twB / Math.max(1e-9, twA), shudder,
       sweepOn, nan, wraps: turn.reduce((m, s) => m + s.wraps, 0), lines: lines.geometry.drawRange.count / 2,
       kernelBase: d0.tesseract.kernelBase, shells: d0.tesseract.shells, kernelN: d0.tesseract.kernelN,
       stream: d0.tesseract.stream, rollIdleRate: (face.getDebugState().tesseract.roll - d0.tesseract.roll) / ((1200 + 900 + 3 + 400 + 5 + 60 + 400 + 12 + 400 + 120 + 600) / 60),
       centerDim: lines.material.uniforms.uCenterDim.value,
       sceneRoll: scene.rotation.z - sceneZ0, rollAcc: face.getDebugState().tesseract.roll - d0.tesseract.roll });
""",
    "stream": r"""
import { face, tick, snapshot, emit } from './harness.mjs';
face.setForm('tesseract'); tick(300);
const d = face.getDebugState().tesseract; const kb = d.kernelBase, kn = d.kernelN * d.kernelN * d.kernelN;
const s0 = kb + kn, N = snapshot().length;                       // stream nodes follow the kernel
const a = snapshot(); tick(30); const b = snapshot();
let out = 0, inn = 0, seen = 0; const speeds = [];
for (let i = s0; i < N; i++) { const p = a[i], q = b[i]; if (!p || !q || p.s < 0.05 || q.s < 0.05) continue;
  const ra = Math.hypot(p.x, p.y, p.z), rb = Math.hypot(q.x, q.y, q.z); if (Math.abs(rb - ra) > 1.0) continue;   // a wrap
  seen++; if (rb > ra) out++; else inn++;
  // exponential growth rate per grain, beyond the kernel's birth-pull (r > 1.5): ∝ its own flowScale
  if (ra > 1.5) speeds.push(Math.log(rb / ra)); }
speeds.sort((x, y) => x - y);
emit({ streamNodes: N - s0, seen, out, inn, spreadRatio: speeds[Math.floor(speeds.length * 0.9)] / Math.max(1e-9, speeds[Math.floor(speeds.length * 0.1)]) });
""",
    "uptime": r"""
import { face, tick, run, p95Of, maxOf, emit } from './harness.mjs';
face.setForm('tesseract'); tick(300);
const early = run(900);
const hooks = {}; for (let f = 0; f < 28800; f += 120) hooks[f] = () => face.noteActivity(0.12, '#3a1750');
run(28800, hooks);                                             // 8 minutes
const late = run(900);
const turn = run(300, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } });
const tw = face.getDebugState().tesseract.tw.map(Math.abs);
emit({ earlyP95: p95Of(early), lateP95: p95Of(late), turnP95: p95Of(turn), turnMax: maxOf(turn), twMax: Math.max(...tw), spin: face.getDebugState().tesseract.kernelSpin });
""",
    "other": r"""
import { face, tick, lines, emit } from './harness.mjs';
face.setForm(process.env.FACE_FORM); tick(120);
const u = lines.material.uniforms;
emit({ form: face.getForm(), uLineSweep: u.uLineSweep ? u.uLineSweep.value : null, reach: u.uSweepReach ? u.uSweepReach.value : null, lines: lines.geometry.drawRange.count / 2 });
""",
}


def _write(root: Path, graph: Path) -> None:
    M._write_harness(root, graph)
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    base = tmp_path_factory.mktemp("cube2v2")
    new, old = base / "new", base / "old"
    new.mkdir(); old.mkdir()
    _write(new, _GRAPH)
    _write(old, _BACKUP)
    return {"new": new, "old": old}


def _run(root: Path, scenario: str, **env) -> dict:
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    r = subprocess.run(["node", f"{scenario}.mjs"], cwd=root, env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def mech(lab):
    return _run(lab["new"], "mech")


def test_one_corridor_edges_rulings_ties_and_a_lattice_kernel(mech):
    L = mech["rest"]["lines"]
    assert mech["shells"] == 24 and mech["kernelN"] == 3
    assert 20 * 72 <= L["shells"] <= 24 * 72, L               # faded shells skip their edges
    assert 20 * 72 <= L["rulings"] <= 24 * 72, L              # a "+" on every face: 12 rulings × 6 segments
    assert 19 * 8 <= L["ties"] <= 23 * 8, L                   # consecutive pairs only, the wrap pair skipped
    assert L["kernel"] == 3 * 3 * 3 * 2, L                    # 54 lattice edges
    assert mech["kernelBase"] == 12 * 8                       # corners on every other shell
    assert mech["stream"] == 250 - 12 * 8 - 27
    assert mech["lines"] == L["shells"] + L["rulings"] + L["ties"] + L["kernel"]


def test_the_stream_rides_the_expansion_at_spread_speeds(lab):
    r = _run(lab["new"], "stream")
    assert r["streamNodes"] == 250 - 12 * 8 - 27, r
    assert r["out"] > 0.9 * r["seen"], r                       # expansion: matter recedes from the kernel
    assert r["spreadRatio"] > 1.6, r                           # a wide speed spread — streams shear


def test_the_corridor_rolls_slowly_and_the_kernel_is_not_centre_dimmed(mech):
    assert 0.015 < mech["rollIdleRate"] < 0.06, mech           # ~0.02 rad/s idle (+turn share in the window)
    assert abs(mech["sceneRoll"] - mech["rollAcc"]) < 0.06, mech   # the SCENE rolls by the accumulator (± the wander sine)
    assert mech["sceneRoll"] > 0.8, mech
    assert mech["centerDim"] == 0.0, mech


def test_the_kernel_spins_and_a_turn_spins_it_faster(mech):
    assert 0.09 < mech["spinIdleRate"] < 0.11, mech
    assert mech["spinTurnRate"] > 4 * mech["spinIdleRate"], mech


def test_newborn_shells_inherit_a_bounded_twist(mech):
    assert mech["twNonZero"] >= 6, mech                        # most of the stack carries a twist on a turn
    assert mech["twMaxTurn"] <= 0.7854 + 1e-9, mech            # folded to the cube's 45° symmetry


def test_busy_is_hotter_than_idle(mech):
    assert mech["seedsTurnMax"] > mech["rest"]["seedsMax"] + 0.02, mech


def test_every_edge_has_its_own_packet_phase(mech):
    assert mech["edgePhaseBases"] == 12, mech


def test_a_tool_call_flashes_the_corners(mech):
    assert mech["flashTool"] > 0.3, mech
    assert all(s > 1.15 for s in mech["cornerSizesTool"]), mech


def test_pass_crystallises_and_refute_shudders(mech):
    assert mech["wobPass"] < 0.3 * mech["rest"]["wob"], mech
    assert mech["twPassDecay"] < 0.75, mech                    # twists unwind fast under a pass
    assert mech["shudder"] > 0.02, mech


def test_the_search_scan_reaches_the_lines_in_cube2_only(mech, lab):
    assert mech["rest"]["uLineSweep"] == 1.0 and mech["rest"]["reach"] == 7.0
    assert mech["sweepOn"] > 0.9
    for form in ("cube", "vortex"):
        r = _run(lab["new"], "other", FACE_FORM=form)
        assert r["uLineSweep"] == 0.0 and r["reach"] == 2.4, r


def test_no_nan_no_visible_wrap_no_teleport(mech):
    assert mech["nan"] == 0 and mech["wraps"] == 0, mech
    assert mech["turnMax"] < 0.35, mech


def test_the_twist_does_not_grow_with_uptime(lab):
    r = _run(lab["new"], "uptime")
    assert r["lateP95"] < 1.5 * r["earlyP95"], r
    assert r["turnMax"] < 0.35 and r["twMax"] <= 0.7854 + 1e-9, r
    assert r["spin"] > 40                                       # the spin itself keeps counting (it is folded at use)


def test_the_backup_drew_shell_edges_only(lab):
    """Negative control, executed: the pre-v2 face draws 11 shells × 72
    segments plus a proximity tangle — no ties, no rulings, no lattice."""
    r = _run(lab["old"], "other", FACE_FORM="tesseract")
    assert 11 * 72 <= r["lines"] < 11 * 72 + 200, r
