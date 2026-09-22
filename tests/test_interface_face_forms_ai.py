"""Regression guard for the AI face forms (2026-07-29; roster 2026-09-20).

The 2026-07-28 alien forms (abyssal/horizon/cortex/vortex/empty) gained
four AI-native siblings — the machine's own internals as anatomy, over
the SAME motion engine:

    cube      — the weight tensor (named `lattice` until 2026-09-20, when
                the operator deleted embedding and the old monolith
                `cube`, gave the lattice this name and made it the
                DEFAULT): tumbling crystal grid, diagonal activation
                waves, a drifting hot attention kernel.
    stack     — the transformer (removed 2026-09-12).
    embedding — latent space (removed 2026-09-20).
    descent   — the loss landscape: evolving terrain sheet, optimizer
                bead rolling true gradient descent, stuck-kicks.

These pins catch what would silently break them:
- a builder disappearing or FORMS losing a name ('empty' must stay LAST
  so the cycle disperses into the void before restarting),
- the dispatch regressing to brittle formIndex comparisons (adding the
  four forms shifted every index after 'vortex'),
- per-form link-radius multipliers drifting against the geometry they
  were tuned for (the stack's discs weave into a solid cylinder the
  moment the layer gap dips under the link radius — first render),
- a builder filling ≠ NODE_COUNT nodes (breaks the instanced attributes
  at runtime) — checked by EXECUTING the real builder code under node.
"""

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def graph_js() -> str:
    return (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


# ── presence + cycle order ─────────────────────────────────────────

def test_ai_form_builders_present(graph_js):
    for builder in ("_buildCube", "_buildCube2", "_buildDescent"):
        assert builder in graph_js, f"missing {builder}"
        assert f"{builder}()" in graph_js, f"{builder} never dispatched"
    for gone in ("_buildLattice", "_buildEmbedding", "_cubeCx", "embExcite"):
        assert gone not in graph_js, f"{gone} is back (removed 2026-09-20)"


def test_forms_array_contents_and_order(graph_js):
    m = re.search(r"const FORMS = \[(.*?)\];", graph_js, re.DOTALL)
    assert m, "FORMS array missing"
    names = re.findall(r"'(\w+)'", m.group(1))
    # 2026-09-12 (operator): abyssal, horizon, cortex, stack, conversation
    # and toolgraph removed. 2026-09-20 (operator): embedding and the old
    # cube removed, lattice renamed cube and made the DEFAULT (listed
    # first); 'empty' stays last.
    # 2026-09-20 (later): cube2, the infinite-cube EXPERIMENT, sits next
    # to cube while the operator decides whether to keep it.
    assert names == ["cube", "tesseract", "vortex", "descent", "empty"]
    # 'empty' last: cycling INTO it disperses the face beyond the screen
    # edges and the NEXT cycle materializes the default from the void.
    assert names[-1] == "empty"
    assert "let formIndex = FORMS.indexOf('cube');" in graph_js, "cube is the default"


# ── name-based dispatch (the index comparisons broke on growth) ────

def test_no_form_index_number_comparisons(graph_js):
    assert "formIndex === " not in graph_js, (
        "form dispatch must key on FORMS[formIndex] names — numeric "
        "index checks silently mis-dispatch whenever FORMS grows")
    assert "const FORM = FORMS[formIndex];" in graph_js


def test_link_multiplier_map(graph_js):
    m = re.search(r"const LINK_MULT = \{(.*?)\};", graph_js, re.DOTALL)
    assert m, "per-form link multiplier map missing"
    body = m.group(1)
    for key in ("vortex", "cube", "tesseract", "descent"):
        assert re.search(rf"\b{key}:", body), f"LINK_MULT.{key} missing"
    for gone in ("lattice", "embedding"):
        assert not re.search(rf"\b{gone}:", body), f"LINK_MULT.{gone} outlived its form"


def test_new_forms_reheat_seeds_per_frame(graph_js):
    # cube (wave+kernel heat) and descent (height+bead heat) rewrite
    # nodeSeeds per frame — each must flag the attribute upload, like
    # vortex already does (plus the setForm upload).
    assert graph_js.count("aSeed.needsUpdate = true") >= 5


# ── geometry invariants (computed from the shipped constants) ──────

def _const(graph_js, pattern):
    m = re.search(pattern, graph_js)
    assert m, f"constant not found: {pattern}"
    return float(m.group(1))


def test_cube_links_neighbors_but_not_diagonals(graph_js):
    """The tensor must read as a clean wireframe grid: cell edge under
    the link radius, face diagonal over it — no diagonal mush."""
    prox = _const(graph_js, r"const PROXIMITY_SQ = ([\d.]+)")
    mult = _const(graph_js, r"cube: ([\d.]+)")
    edge = _const(graph_js, r"const CUBE_A = IS_MOBILE \? [\d.]+ : ([\d.]+)")
    edge_mobile = _const(graph_js, r"const CUBE_A = IS_MOBILE \? ([\d.]+)")
    radius = math.sqrt(prox * mult)
    for a in (edge, edge_mobile):
        assert a < radius, f"cell edge {a} must link (radius {radius:.3f})"
        assert a * math.sqrt(2) > radius, (
            f"face diagonal {a * math.sqrt(2):.3f} must NOT link")


def test_descent_mesh_holds_on_slopes(graph_js):
    """Adjacent sheet nodes must stay linked even across the steepest
    slope of the height field, or the terrain tears while it evolves."""
    prox = _const(graph_js, r"const PROXIMITY_SQ = ([\d.]+)")
    mult = _const(graph_js, r"descent: IS_MOBILE \? [\d.]+ : ([\d.]+)")
    nx = _const(graph_js, r"const DESC_NX = IS_MOBILE \? \d+ : (\d+)")
    nz = _const(graph_js, r"DESC_NZ = IS_MOBILE \? \d+ : (\d+)")
    sx = _const(graph_js, r"const DESC_SX = ([\d.]+)")
    sz = _const(graph_js, r"DESC_SZ = ([\d.]+)")
    h = _const(graph_js, r"const DESC_H = ([\d.]+)")
    radius = math.sqrt(prox * mult)
    dx = sx / (nx - 1)
    dz = sz / (nz - 1)
    # Worst-case slope of the analytic field: sum of |amp·k| over terms
    # (amplitudes/wavenumbers from _lossH, scaled by DESC_H).
    slope = h * (0.30 * 1.7 + 0.22 * 2.3 + 0.14 * 3.1 + 0.10 * 2.5)
    for d in (dx, dz):
        worst = math.hypot(d, slope * d)
        assert worst < radius, (
            f"grid step {d:.3f} at max slope → {worst:.3f} exceeds link "
            f"radius {radius:.3f}: the sheet would tear")


# ── behavioral micro-pins ──────────────────────────────────────────

def test_descent_is_actual_gradient_descent(graph_js):
    assert "_lossH(beadX + eps" in graph_js, "numeric gradient missing"
    assert "beadStill" in graph_js, "stuck detection / kick missing"
    assert re.search(r"beadV[XZ] = \(beadV[XZ] - g[XZ] \* lr", graph_js), \
        "velocity update must subtract the gradient (descent, not ascent)"


def test_dive_centers_on_the_form_focus(graph_js):
    """2026-07-29 operator report: 'descent usually zooms into an
    uninteresting location when busy.' The immersion dive targets the
    scene ORIGIN — for descent a generic terrain patch, for the cube the
    grid centre with the attention kernel off screen (2026-09-20). Both
    forms translate their space (dive-weighted) so the dive rides the
    form's hot focus: the optimizer bead / the attention kernel. (The
    executed pins live in test_interface_face_motion_2026_09_20.py.)"""
    # descent: bead world position subtracted, dive-weighted, with the
    # +0.30 lift so the camera hovers over the surface.
    assert "const _descFx = dive * beadX" in graph_js
    assert "- _descFx" in graph_js and "- _descFy" in graph_js
    # cube: the kernel's post-tumble position subtracted, dive-weighted.
    assert "const _cubeFx = dive *" in graph_js
    assert "- _cubeFx" in graph_js and "- _cubeFy" in graph_js


def test_form_picker_menu(graph_js):
    """2026-07-29: the blind cycle button stopped scaling at 9 forms (up
    to 8 clicks × 1.4s blends through forms you didn't ask for). The
    header button now opens a picker built from matrix_graph's OWN
    roster, so a new form appears in the menu with no app.js edit."""
    assert "export function getForms()" in graph_js
    assert "FORMS.slice()" in graph_js
    app_js = (_STATIC / "app.js").read_text(encoding="utf-8")
    assert "activeFace.getForms()" in app_js, "menu must build from the roster"
    assert "activeFace.setForm(name)" in app_js, "picking must jump directly"
    assert "face-form-menu" in app_js
    assert "cycleForm()" in app_js, "stale-cache fallback must remain"
    # Every shipped form carries a hint line in the picker.
    for name in ("cube", "tesseract", "vortex", "descent", "empty"):
        assert re.search(rf"\b{name}: '", app_js), f"hint missing for {name}"
    for gone in ("lattice", "embedding"):
        assert not re.search(rf"\b{gone}: '", app_js), f"hint for removed form {gone}"
    css = (_STATIC / "style.css").read_text(encoding="utf-8")
    assert "#face-form-menu" in css
    assert ".face-form-item.active" in css


def test_wrap_fades_hide_respawns(graph_js):
    # The cube's runners wrap; they must taper size at the wrap ends
    # (bp.sz mutation) so respawns don't pop on screen. (The stack's
    # packets, the other wrapper, left with the form on 2026-09-12.)
    assert graph_js.count("Math.min(t, 1 - t)") >= 1


# ── executing the REAL builders (node) ─────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("is_mobile,expected", [(False, 250), (True, 120)])
def test_every_builder_fills_exactly_node_count(graph_js, is_mobile, expected):
    """A builder filling ≠ NODE_COUNT corrupts the instanced attributes
    at runtime with no console error — the one invariant worth running
    the real code for. Extracts the anatomy-builders section verbatim
    and executes every builder under node for both device classes."""
    start = graph_js.index("// ── Anatomy builders")
    end = graph_js.index("// ── Form switching")
    section = graph_js[start:end]
    harness = f"""
const IS_MOBILE = {str(is_mobile).lower()};
const NODE_COUNT = IS_MOBILE ? 120 : 250;
const basePositions = [];
const nodeSeeds = new Float32Array(NODE_COUNT);
const VORTEX_APEX_Z = -2.0, VORTEX_LMIN = 0.55, VORTEX_KOUT = 2.6;
const VORTEX_COS = 0.60, VORTEX_SIN = 0.80;
let beadX = 0, beadZ = 0, beadVX = 0, beadVZ = 0, beadStill = 0;
const beadTrail = []; let _descTick = 0;
const FORMS = ['cube', 'tesseract', 'vortex', 'descent', 'empty'];
let formIndex = 0;
{section}
const builders = [_buildCube, _buildCube2, _buildVortex, _buildDescent, _buildEmpty];
for (const fn of builders) {{
    basePositions.length = 0;
    fn();
    if (basePositions.length !== NODE_COUNT) {{
        console.error(fn.name + ': ' + basePositions.length);
        process.exit(1);
    }}
    for (let i = 0; i < NODE_COUNT; i++) {{
        const s = nodeSeeds[i];
        if (!(s >= 0 && s <= 0.75)) {{
            console.error(fn.name + ' seed[' + i + ']=' + s);
            process.exit(1);
        }}
    }}
}}
console.log('OK');
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "harness.mjs"
        p.write_text(harness, encoding="utf-8")
        r = subprocess.run(["node", str(p)], capture_output=True, text=True,
                           timeout=30)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "OK" in r.stdout
