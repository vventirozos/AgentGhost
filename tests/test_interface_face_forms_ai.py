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

def test_forms_array_contents_and_order(graph_js):
    m = re.search(r"const FORMS = \[(.*?)\];", graph_js, re.DOTALL)
    assert m, "FORMS array missing"
    names = re.findall(r"'(\w+)'", m.group(1))
    # 2026-11-06 (operator): only these two stayed; the button switches between them.
    assert names == ["moodtesseract", "empty"]
    assert "let formIndex = FORMS.indexOf('moodtesseract');" in graph_js


# ── name-based dispatch (the index comparisons broke on growth) ────

def test_no_form_index_number_comparisons(graph_js):
    assert "formIndex === " not in graph_js, (
        "form dispatch must key on FORMS[formIndex] names — numeric "
        "index checks silently mis-dispatch whenever FORMS grows")
    assert "const FORM = FORMS[formIndex];" in graph_js


# ── geometry invariants (computed from the shipped constants) ──────

def _const(graph_js, pattern):
    m = re.search(pattern, graph_js)
    assert m, f"constant not found: {pattern}"
    return float(m.group(1))


# ── behavioral micro-pins ──────────────────────────────────────────

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
const FORMS = ['moodtesseract', 'empty'];
let formIndex = 0;
{section}
const builders = [_buildCube2, _buildEmpty];
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
