"""Three face defects found by driving the real render loop (2026-09-21).

Operator: "search for defects in the faces". The §4JA harness (a THREE
stub, fake rAF, the REAL ``animate()`` under node) was pointed at what the
09-20 pins never measured — 20 minutes of uptime per form, the line
budget under a user turn, and the dive couplings on forms whose camera
never moves. Each pin runs against the shipped face AND the pre-fix backup
(``matrix_graph.js.bak-20260921-prevortexfix``) as a negative control, so
the instrument is shown to see the defect it pins.

  F1  vortex line cap — the user-turn web measured 11,942 qualifying links
      (desktop; 2,944 mobile) against MAX_LINES 10,000 / 2,500: capped in
      64% of turn frames, 91% under the read gait's thicken. The emitter
      kept the FIRST pairs by node index and dropped the rest, so the
      last-built region lost its web and the count flickered across the
      cap. Now: headroom (12,000 / 3,000) + a link BUDGET that tightens the
      radius uniformly after an overflow frame and relaxes when under.
  F2  vortex differential swirl `vortexSpin × (0.9 − 0.4·dOut)` — the
      ever-growing spin angle times a per-node factor that changes as the
      node flows: a `time × variable` phase in disguise. Idle motion grew
      13× over 20 minutes, a user turn after an hour teleported nodes
      (p95 0.998/frame). Now each stream node integrates its own swirl.
  F3  the camera-STATIC forms (vortex, cube2) still took every dive
      coupling built for a camera moving INTO the cloud — bloom ×0.5, line
      dim ×0.7, motes, clock ×1.6, proximity ×1.15 — so the vortex's busy
      state was DIMMER than idle (bloom 0.50 vs 0.57; measured mean
      brightness fell 6.6 → 6.3 on a user turn). Now `camDive` = 0 for them.
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
_BACKUP = _STATIC / "matrix_graph.js.bak-20260921-prevortexfix"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

# The bloom stub registers itself so a scenario can read the strength the
# loop wrote this frame.
_BLOOM_STUB = "export class UnrealBloomPass { constructor(res, s) { this.strength = s; globalThis.__bloom = this; } }\n"

_SCENARIOS = {
    "uptime": r"""
import { face, tick, run, p95Of, maxOf, emit } from './harness.mjs';
face.setForm('vortex'); tick(300);
const early = run(900);
const hooks = {}; for (let f = 0; f < 28800; f += 120) hooks[f] = () => face.noteActivity(0.12, '#3a1750');
run(28800, hooks);                                             // 8 minutes of uptime
const late = run(900);
const turn = run(300, { 0: () => { face.setUserTurn(true); face.setWorkingState(true); } });
emit({ earlyP95: p95Of(early), lateP95: p95Of(late), turnP95: p95Of(turn), turnMax: maxOf(turn) });
""",
    "budget": r"""
import { face, tick, lines, emit } from './harness.mjs';
face.setForm('vortex'); tick(300);
const MAXL = face.getDebugState().maxLines || 10000;
let capped = 0, n = 0, minBudget = 1;
face.setUserTurn(true); face.setWorkingState(true); face.setPhase('read');
for (let f = 0; f < 1200; f++) { if (f % 15 === 0) face.noteActivity(0.16); tick(1); n++;
  if (lines.geometry.drawRange.count / 2 >= MAXL) capped++;
  const b = face.getDebugState().linkBudget; if (b !== undefined) minBudget = Math.min(minBudget, b); }
face.setPhase(null); face.setUserTurn(false); face.setWorkingState(false); face.noteVerdict('stop');
tick(1500);
emit({ maxLines: MAXL, cappedFrac: capped / n, minBudget, budgetAfterRest: face.getDebugState().linkBudget });
""",
    "camdive": r"""
import { face, tick, scene, lines, emit } from './harness.mjs';
const form = process.env.FACE_FORM; face.setForm(form); tick(300);
const bloomRest = globalThis.__bloom.strength;
face.setUserTurn(true); face.setWorkingState(true);
for (let i = 0; i < 480; i++) { if (i % 15 === 0) face.noteActivity(0.16); tick(1); }
const motes = scene.children.find(c => c.constructor.name === 'Points');
emit({ form, bloomRest, bloomTurn: globalThis.__bloom.strength, uDive: lines.material.uniforms.uDive.value,
       motesVisible: motes ? motes.visible : null, immersion: face.getDebugState().immersion });
""",
}


def _write(root: Path, graph: Path) -> None:
    M._write_harness(root, graph)
    (root / "node_modules" / "three" / "addons" / "postprocessing" / "UnrealBloomPass.js").write_text(_BLOOM_STUB)
    for name, src in _SCENARIOS.items():
        (root / f"{name}.mjs").write_text(src)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    base = tmp_path_factory.mktemp("facelab21")
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


# ── F2: the vortex no longer grows with uptime ─────────────────────────

def test_vortex_idle_motion_does_not_grow_with_uptime(lab):
    r = _run(lab["new"], "uptime")
    assert r["lateP95"] < 1.5 * r["earlyP95"], r
    assert r["turnP95"] < 0.35 and r["turnMax"] < 0.5, r      # a turn after 8 min: no teleports


def test_the_instrument_sees_the_uptime_growth_in_the_backup(lab):
    r = _run(lab["old"], "uptime")
    assert r["lateP95"] > 3.0 * r["earlyP95"], r
    assert r["turnP95"] > 0.5, r


# ── F1: the line budget replaces index-ordered dropping ───────────────

def test_vortex_read_turn_stays_under_the_line_cap(lab):
    r = _run(lab["new"], "budget")
    assert r["maxLines"] == 12000
    assert r["cappedFrac"] < 0.06, r                        # transient frames only, while the budget converges
    assert r["minBudget"] < 0.9, r                          # the budget engaged
    assert r["budgetAfterRest"] > 0.99, r                   # and relaxed fully once the surge ended


def test_the_backup_is_capped_most_of_the_read_turn(lab):
    r = _run(lab["old"], "budget")
    assert r["maxLines"] == 10000 and r["cappedFrac"] > 0.5, r


# ── F3: camera-static forms opt out of the dive couplings ─────────────

@pytest.mark.parametrize("form", ["vortex", "tesseract"])
def test_a_static_camera_form_gets_no_dive_dim_and_a_brighter_busy_state(lab, form):
    r = _run(lab["new"], "camdive", FACE_FORM=form)
    assert r["immersion"] > 0.8, r                          # the turn engaged (the dive VALUE still rises)
    assert r["uDive"] == 0.0 and r["motesVisible"] is False, r
    assert r["bloomTurn"] > r["bloomRest"], r               # busy glows more than idle


def test_a_diving_form_keeps_its_interior_couplings(lab):
    r = _run(lab["new"], "camdive", FACE_FORM="cube")
    assert r["uDive"] > 0.5 and r["motesVisible"] is True, r


def test_the_backup_dimmed_the_vortex_on_a_turn(lab):
    r = _run(lab["old"], "camdive", FACE_FORM="vortex")
    assert r["uDive"] > 0.5 and r["bloomTurn"] < r["bloomRest"], r
