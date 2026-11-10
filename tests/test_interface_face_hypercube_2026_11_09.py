"""Face 2026-11-09 (§4TV): the kernel is a hypercube, more pronounced.

Operator: "can this cube be a bit more pronounced and instead of a plain cube
have a hypercube animating ? (keep a backup before any change)".

Same day, operator after seeing it: the 4D shape and the newborn frames look
like "2 completely separate animations, this needs to be more organic", and
"reduce the pulse for the 4D shape to the minimum, it's size can change but
not like pulsing, more steadily, less erratic". So the frames peel off it:
born at its size, in its frame (the corridor's frame at birth), in its
colour, with a birth glow; and its size follows only the eased turn
engagement and the mood — no breath, no tool flash, no tremor, no spin kick.

World where each pin fails: the kernel is still a lattice or a plain cube,
its 4D turn stops (only a 3D tumble, the inner cube never passes through the
outer one), the projection collapses to one cube, it pulses or jumps again,
a tool call kicks it, or a frame is born somewhere other than on it.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests import test_interface_face_motion_2026_09_20 as M

_STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"

_SCEN = r"""
import { face, tick, snapshot, emit } from './harness.mjs';
face.setForm('moodtesseract'); tick(240);
const T = () => face.getDebugState().tesseract;
const samples = [];
for (let k = 0; k < 8; k++) {
  tick(150);
  const t = T(), kb = t.kernelBase, s = snapshot();
  const P = s.slice(kb, kb + t.kernelVerts);
  const c = P.reduce((a, p) => [a[0] + p.x / P.length, a[1] + p.y / P.length, a[2] + p.z / P.length], [0, 0, 0]);
  const r = P.map(p => Math.hypot(p.x - c[0], p.y - c[1], p.z - c[2]));
  // the 8 vertices farthest from the centre this frame = the "outer" cube
  const outer = r.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8).map(x => x[1]).sort((a, b) => a - b);
  samples.push({ xw: t.hyper.xw, rMax: Math.max(...r), rMin: Math.min(...r), outer: outer.join(','),
                 kernelLines: t.lines.kernel, verts: t.kernelVerts, edges: t.kernelEdges,
                 // newborn shells (smallest) vs grown ones: fade carried on corner node sizes
                 shellSeeds: t.seeds.length });
}
// steadiness at rest: the birth size (= the kernel's size) every frame for 10 s
const sizes = []; for (let f = 0; f < 600; f++) { tick(1); sizes.push(T().birthL); }
// a tool call: no spin kick, no size change
const sp0 = T().kernelSpin; tick(60); const rateA = (T().kernelSpin - sp0) / 60;
const L0 = T().birthL; face.noteToolCall(); const sp1 = T().kernelSpin; tick(60);
const rateB = (T().kernelSpin - sp1) / 60, L1 = T().birthL;
// a user turn: the size may grow, but eased
face.setUserTurn(true); face.setWorkingState(true);
const turn = []; for (let f = 0; f < 600; f++) { tick(1); turn.push(T().birthL); }
emit({ samples, sizes, rateA, rateB, L0, L1, turn });
"""


@pytest.fixture(scope="module")
def hyper(tmp_path_factory):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    root = tmp_path_factory.mktemp("hyper")
    M._write_harness(root, _STATIC / "matrix_graph.js")
    (root / "h.mjs").write_text(_SCEN, encoding="utf-8")
    r = subprocess.run(["node", "h.mjs"], cwd=root, env=dict(os.environ), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_it_is_a_hypercube():
    js = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")
    assert "const C2_KERNEL = 16;" in js
    # the edge list: pairs differing in exactly one of four sign bits → 32, degree 4
    edges = [(a, a | (1 << b)) for a in range(16) for b in range(4) if not a & (1 << b)]
    assert len(edges) == 32 and all(sum(v in e for e in edges) == 4 for v in range(16))
    assert "if (!(a & (1 << b))) C2_HYPER_EDGES.push([a, a | (1 << b)]);" in js


def test_all_32_edges_are_drawn_every_frame(hyper):
    assert all(s["verts"] == 16 and s["edges"] == 32 and s["kernelLines"] == 32 for s in hyper["samples"])


def test_the_projection_shows_two_cubes(hyper):
    # perspective in w: the far cube is clearly smaller than the near one
    assert all(s["rMax"] > 1.4 * s["rMin"] for s in hyper["samples"]), [(s["rMax"], s["rMin"]) for s in hyper]


def test_it_turns_in_4d_so_the_cubes_trade_places(hyper):
    xw = [s["xw"] for s in hyper["samples"]]
    assert all(b > a for a, b in zip(xw, xw[1:])), xw          # integrated, monotone
    # which vertices form the outer cube changes over time: the inner cube
    # passes through the outer one (a plain 3D tumble never changes this set)
    assert len({s["outer"] for s in hyper["samples"]}) >= 3, [s["outer"] for s in hyper["samples"]]


def test_more_pronounced():
    js = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")
    import re
    r = float(re.search(r"const C2_KERNEL_R = ([0-9.]+);", js).group(1))
    gain = float(re.search(r"const C2_KERNEL_LINE_GAIN = ([0-9.]+);", js).group(1))
    glow = float(re.search(r"const C2_BIRTH_GLOW = ([0-9.]+);", js).group(1))
    shell = float(re.search(r"const C2_LINE_GAIN = ([0-9.]+);", js).group(1))
    assert r > 0.16 * 1.5 and gain > 2.0 * 1.5            # vs the lattice's 0.16 / 2.0
    # still the brightest thing, but a newborn frame leaves close to it
    assert shell * (1 + glow) < gain < 1.6 * shell * (1 + glow)


def test_the_size_is_steady_at_rest(hyper):
    z = hyper["sizes"]
    assert max(z) - min(z) < 1e-9, (min(z), max(z))         # no breath, no pulse at rest


def test_a_tool_call_does_not_kick_it(hyper):
    assert hyper["rateB"] == pytest.approx(hyper["rateA"], rel=0.02), hyper
    assert hyper["L1"] == pytest.approx(hyper["L0"], rel=1e-6)


def test_a_turn_grows_it_steadily(hyper):
    t = hyper["turn"]
    assert t[-1] > 1.05 * t[0]                                 # it may grow on a turn…
    steps = [b - a for a, b in zip(t, t[1:])]
    assert min(steps) >= -1e-12                                # …monotonically, no pulse…
    assert max(steps) < 0.02 * t[0]                            # …and no frame jumps


def test_frames_emerge_from_its_surface_in_its_frame():
    js = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")
    i = js.index("function _c2Frame(")
    body = js[i:i + 1800]
    # growth unchanged (×100 from 0.08): emerging at the kernel's size with a
    # slower growth moved the newborns 2.5× faster on screen (measured)
    assert "sh.L = C2_L0 * Math.exp(C2_K * d) * breath;" in body
    assert "const C2_L0 = 0.08;" in js and "const C2_K = Math.log(100);" in js
    # hidden inside the kernel, fading in across its surface, on it until it leaves
    assert "(sh.L / _c2KR - 0.85) / 0.4" in body
    assert "Math.exp(-Math.max(0, sh.L - _c2KR) / 0.6)" in body
    assert "Math.exp(-Math.max(0, d - _c2BirthD) / 0.12)" in body
    # the kernel sits on the corridor's frame at the depth a frame reaches it
    assert "_c2KFrame.tw = c2KernelSpin;" in js and "_c2Frame(_c2KFrame, _c2BirthD," in js
    assert "_c2KernelFrame" not in js and "const tremor" not in js and "flashGain" not in js


def test_a_hidden_frame_draws_no_corners():
    # a size of exactly 0 must stay 0 (`bp.sz || 1.0` drew the frames hidden
    # inside the kernel at full size)
    js = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")
    assert "(bpi.sz ?? 1.0)" in js and "(bpi.sz || 1.0)" not in js
