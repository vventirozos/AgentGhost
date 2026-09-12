"""Gait dialects (2026-09-11) and the tunables table (trimmed 2026-09-12).

  TUNE      every amplitude the signal layer uses lives in one frozen
            table and every key is read in the loop (set equality both
            ways). The lab that dragged them live was REMOVED 2026-09-12;
            its persistence machinery went with it (pinned absent).
  dialects  DIALECTS covers every form with values from DIALECT_VALUES;
            the per-frame scalars are derived from the current form's
            dialect; each form branch reads the scalar its dialect promises
            (text pins per branch); the shader has the three sweep modes;
            the generic pass honours radialAxis.
"""

import re
import sys
from pathlib import Path

import pytest

from tests.helpers import eval_js, extract_js_function, strip_js_comments

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_STATIC = _ROOT / "interface" / "static"


@pytest.fixture(scope="module")
def graph_js() -> str:
    return (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def graph_nc(graph_js) -> str:
    return strip_js_comments(graph_js)


def _fn(src, *names):
    return "".join(extract_js_function(src, n) for n in names)


def _const_block(src, name):
    """`export const NAME = ...;` as source, brace/paren matched."""
    i = src.index(f"export const {name} = ")
    j = min(x for x in (src.find("\n});", i), src.find("\n};", i)) if x != -1)
    end = src.index("\n", j + 1)
    return src[i:end + 1].replace("export const", "const")


# ═══════════════════════════════════════════════════════════════════════════
# TUNE
# ═══════════════════════════════════════════════════════════════════════════

class TestTune:
    def test_every_key_is_read_and_every_read_has_a_key(self, graph_js, graph_nc):
        keys = set(eval_js(_const_block(graph_js, "TUNE"), "Object.keys(TUNE)"))
        used = set(re.findall(r"\bTUNE\.([A-Za-z]+)", graph_nc))
        assert keys == used, f"orphan keys {keys - used} / undeclared uses {used - keys}"
        assert len(keys) >= 20

    def test_the_table_is_frozen_and_the_lab_machinery_is_gone(self, graph_js, graph_nc):
        assert "export const TUNE = Object.freeze({" in graph_js
        for gone in ("TUNE_DEFAULTS", "setTune", "getTune", "resetTune", "loadTuneOverrides", "ghost_face_tune"):
            assert gone not in graph_nc, f"{gone} is back"

    def test_the_render_loop_reads_the_table_not_literals(self, graph_nc):
        for old in ("0.04 * gait.search", "0.06 * toolPulse;", "0.10 * writeW", "0.035 * verdictEnv",
                    "0.012 * backgroundBusy", "2.6 * recallSpark", "0.22 * smoothstep"):
            assert old not in graph_nc, f"a hard-coded amplitude survived: {old}"
        assert "uniform float uSweepHeat;" in graph_nc and "nUniforms.uSweepHeat.value = TUNE.sweepHeat;" in graph_nc


# ═══════════════════════════════════════════════════════════════════════════
# Dialects
# ═══════════════════════════════════════════════════════════════════════════

class TestDialects:
    def _env(self, graph_js):
        forms = re.search(r"const FORMS = \[(.*?)\];", graph_js, re.DOTALL).group(0)
        return (forms + _const_block(graph_js, "DIALECT_VALUES") + _const_block(graph_js, "DIALECTS")
                + "const _DIALECT_DEFAULT = DIALECTS.vortex;\n" + _fn(graph_js, "dialectFor"))

    def test_every_form_has_a_dialect_with_valid_values(self, graph_js):
        out = eval_js(self._env(graph_js), """(() => {
            const missing = FORMS.filter(f => !DIALECTS[f]);
            const bad = [];
            for (const [f, d] of Object.entries(DIALECTS)) {
                for (const [k, allowed] of Object.entries(DIALECT_VALUES)) {
                    if (!allowed.includes(d[k])) bad.push(f + '.' + k + '=' + d[k]);
                }
                for (const k of Object.keys(d)) if (!(k in DIALECT_VALUES)) bad.push(f + ' extra ' + k);
            }
            const extra = Object.keys(DIALECTS).filter(f => !FORMS.includes(f));
            return { missing, bad, extra, fallback: dialectFor('bogus') === DIALECTS.vortex };
        })()""")
        assert out == {"missing": [], "bad": [], "extra": [], "fallback": True}

    def test_no_value_is_declared_that_no_form_uses(self, graph_js):
        out = eval_js(self._env(graph_js), """(() => {
            const unused = [];
            for (const [k, allowed] of Object.entries(DIALECT_VALUES))
                for (const v of allowed) if (!Object.values(DIALECTS).some(d => d[k] === v)) unused.push(k + '=' + v);
            return unused;
        })()""")
        # 'xz' stays as the vocabulary's ring-breath option; nothing else may dangle.
        assert out == ["radialAxis=xz"], out

    def test_dialects_are_actually_different(self, graph_js):
        out = eval_js(self._env(graph_js), "new Set(Object.values(DIALECTS).map(d => JSON.stringify(d))).size")
        assert out >= 4, "the table exists but most forms speak the same grammar"

    def test_crystals_do_not_breathe(self, graph_js):
        out = eval_js(self._env(graph_js), """[DIALECTS.lattice.radialAxis, DIALECTS.cube.radialAxis,
            DIALECTS.descent.radialAxis, DIALECTS.vortex.read, DIALECTS.lattice.verify, DIALECTS.cube.verify]""")
        assert out == ["none", "none", "y", "thicken", "align", "align"]

    def test_scalars_are_derived_from_the_current_dialect_each_frame(self, graph_nc):
        i = graph_nc.index("const DIAL = dialectFor(FORMS[formIndex]);")
        body = graph_nc[i:i + 400]
        for line in ("gaitFlow = DIAL.write === 'flow' ? gait.write : 0;",
                     "gaitThicken = DIAL.read === 'thicken' ? gait.read : 0;",
                     "gaitFlash = DIAL.tool === 'flash' ? toolPulse : 0;",
                     "gaitAlign = DIAL.verify === 'align' ? gait.verify : 0;"):
            assert line in body, line

    @pytest.mark.parametrize("form,anchor,needles", [
        ("vortex", "tunnelFlow += (1 / 60) * (0.008", ["gaitFlow", "gaitThicken"]),
        ("lattice", "const alignMul = 1.0 - TUNE.alignGain * gaitAlign;", ["alignMul", "gaitFlash", "gaitFlow"]),
        ("embedding", "embT += (1 / 60) * (0.24", ["gaitFlow"]),
        ("descent", "const lr = 1.6 * (0.55", ["gaitFlow"]),
        ("cube", "const tk = time * (0.7 + 1.4 * Sk", ["gaitFlow", "gaitAlign"]),
    ])
    def test_each_branch_speaks_its_dialect(self, graph_nc, form, anchor, needles):
        i = graph_nc.index(anchor)
        window = graph_nc[max(0, i - 700):i + 1500]
        for n in needles:
            assert n in window, f"{form}: {n} not read near its hook"

    def test_generic_pass_honours_radial_axis_and_dialect_choices(self, graph_nc):
        i = graph_nc.index("const axis = DIAL.radialAxis;")
        body = graph_nc[i:i + 1400]
        assert "DIAL.read === 'contract' ? TUNE.radialRead * gait.read : 0" in body
        assert "DIAL.tool === 'kick' ? TUNE.toolKick * toolPulse : 0" in body
        assert "DIAL.write === 'wave' ? gait.write : 0" in body
        assert "else if (axis === 'y') { p.y *= f; }" in body
        assert "let f = axis === 'none' ? 1.0 : radial;" in body

    def test_thicken_widens_the_link_radius(self, graph_nc):
        assert "* (1.0 + TUNE.thickenLinks * gaitThicken);" in graph_nc

    def test_shader_has_three_sweep_modes_and_the_uniform_follows_the_dialect(self, graph_js):
        assert "uniform float uSweepMode;" in graph_js
        assert "float sweepD = uSweepMode < 0.5 ? dAz : (uSweepMode < 1.5 ? dPlane : dRing);" in graph_js
        assert "nUniforms.uSweepMode.value = DIAL.search === 'plane' ? 1.0 : (DIAL.search === 'ring' ? 2.0 : 0.0);" in graph_js
        assert "uSweepMode: { value: 0.0 }" in graph_js

    def test_debug_state_reports_the_dialect(self, graph_nc):
        i = graph_nc.index("export function getDebugState()")
        assert "dialect: dialectFor(FORMS[formIndex]), gaitFlow, gaitThicken, gaitFlash, gaitAlign," in graph_nc[i:i + 1200]
