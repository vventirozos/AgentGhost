"""Face lab + gait dialects (2026-09-11).

  lab       facelab.js: every button names an export the face module
            actually has (cross-checked against the module's `export
            function` list — a renamed hook would otherwise be a dead
            button with a toast); slider specs; the TUNE table with
            persistence and clamping; the palette command + shortcut.
  dialects  DIALECTS covers every form with values from DIALECT_VALUES;
            the per-frame scalars are derived from the current form's
            dialect; each form branch reads the scalar its dialect promises
            (text pins per branch); the shader has the three sweep modes;
            the generic pass honours radialAxis.

Pure pieces executed under node; the render loop pinned as text.
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


@pytest.fixture(scope="module")
def lab_js() -> str:
    return (_STATIC / "facelab.js").read_text(encoding="utf-8")


def _fn(src, *names):
    return "".join(extract_js_function(src, n) for n in names)


def _const_block(src, name):
    """`export const NAME = ...;` as source, brace/paren matched."""
    i = src.index(f"export const {name} = ")
    j = src.index("\n});", i) if "\n});" in src[i:i + 6000] and src[i:i + 6000].index("\n});") < (src[i:i + 6000].find("\n};") if "\n};" in src[i:i + 6000] else 10**9) else src.index("\n};", i)
    end = src.index("\n", j + 1) if j + 1 < len(src) else len(src)
    return src[i:end + 1].replace("export const", "const")


def _tune_env(graph_js):
    return (_const_block(graph_js, "TUNE_DEFAULTS")
            + "const TUNE = { ...TUNE_DEFAULTS };\n"
            + "const stored = {}; globalThis.localStorage = { setItem: (k, v) => { stored[k] = v; }, removeItem: (k) => { delete stored[k]; }, getItem: (k) => stored[k] ?? null };\n"
            + _fn(graph_js, "getTune", "setTune", "resetTune", "_persistTune", "loadTuneOverrides"))


# ═══════════════════════════════════════════════════════════════════════════
# TUNE
# ═══════════════════════════════════════════════════════════════════════════

class TestTune:
    def test_every_default_is_used_and_every_use_has_a_default(self, graph_js, graph_nc):
        keys = set(eval_js(_const_block(graph_js, "TUNE_DEFAULTS"), "Object.keys(TUNE_DEFAULTS)"))
        used = set(re.findall(r"\bTUNE\.([A-Za-z]+)", graph_nc))
        assert keys == used, f"orphan defaults {keys - used} / undeclared uses {used - keys}"
        assert len(keys) >= 20

    def test_set_tune_validates_clamps_and_persists(self, graph_js):
        out = eval_js(_tune_env(graph_js), """(() => {
            const a = setTune('toolKick', 0.09);
            const b = setTune('toolKick', 99);            // clamped to 4× default
            const c = setTune('toolKick', -1);            // clamped to 0
            const d = setTune('nope', 1);                 // unknown key
            const e = setTune('writeWave', 'abc');        // not a number: unchanged
            const f = setTune('passHold', 5);             // decay stays < 1
            return { a, b, c, d, e, f, stored: JSON.parse(stored.ghost_face_tune),
                     keys: Object.keys(TUNE).sort(), defaults: Object.keys(TUNE_DEFAULTS).sort() };
        })()""")
        assert out["a"] == 0.09 and out["b"] == pytest.approx(0.24) and out["c"] == 0
        assert out["d"] is None and out["e"] == 0.10 and out["f"] == 0.999
        assert out["stored"] == {"toolKick": 0, "passHold": 0.999}, "only DIFFS persist"
        # An unknown key must not be CREATED either (a NaN entry serialises as
        # null and slipped past `d is None` — the mutant that dropped the key
        # guard survived until this line).
        assert out["keys"] == out["defaults"], "setTune grew the table"

    def test_reset_restores_and_clears_storage(self, graph_js):
        out = eval_js(_tune_env(graph_js), """(() => {
            setTune('exhale', 0.1);
            const t = resetTune();
            return { exhale: t.exhale, stored: stored.ghost_face_tune ?? null };
        })()""")
        assert out == {"exhale": 0.035, "stored": None}

    def test_overrides_load_ignores_junk(self, graph_js):
        out = eval_js(_tune_env(graph_js), """(() => {
            const n = loadTuneOverrides(JSON.stringify({ shudder: 0.05, bogus: 1, twitch: 'x', passHold: 2 }));
            const bad = loadTuneOverrides('{not json');
            return { n, bad, shudder: TUNE.shudder, twitch: TUNE.twitch, passHold: TUNE.passHold };
        })()""")
        assert out == {"n": 2, "bad": 0, "shudder": 0.05, "twitch": 0.08, "passHold": 0.999}

    def test_the_render_loop_reads_the_table_not_literals(self, graph_nc):
        for old in ("0.04 * gait.search", "0.06 * toolPulse;", "0.10 * writeW", "0.035 * verdictEnv",
                    "0.012 * backgroundBusy", "2.6 * recallSpark", "0.22 * smoothstep"):
            assert old not in graph_nc, f"a hard-coded amplitude survived: {old}"

    def test_boot_loads_overrides_and_the_shader_heat_is_a_uniform(self, graph_nc):
        assert "loadTuneOverrides(localStorage.getItem('ghost_face_tune'))" in graph_nc
        assert "uniform float uSweepHeat;" in graph_nc and "nUniforms.uSweepHeat.value = TUNE.sweepHeat;" in graph_nc


# ═══════════════════════════════════════════════════════════════════════════
# Dialects
# ═══════════════════════════════════════════════════════════════════════════

class TestDialects:
    def _env(self, graph_js):
        forms = re.search(r"const FORMS = \[(.*?)\];", graph_js, re.DOTALL).group(0)
        return (forms + _const_block(graph_js, "DIALECT_VALUES") + _const_block(graph_js, "DIALECTS")
                + "const _DIALECT_DEFAULT = DIALECTS.abyssal;\n" + _fn(graph_js, "dialectFor"))

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
            return { missing, bad, extra, fallback: dialectFor('bogus') === DIALECTS.abyssal };
        })()""")
        assert out == {"missing": [], "bad": [], "extra": [], "fallback": True}

    def test_dialects_are_actually_different(self, graph_js):
        out = eval_js(self._env(graph_js), "new Set(Object.values(DIALECTS).map(d => JSON.stringify(d))).size")
        assert out >= 8, "the table exists but most forms speak the same grammar"

    def test_crystals_do_not_breathe_and_data_forms_pulse(self, graph_js):
        out = eval_js(self._env(graph_js), """[DIALECTS.lattice.radialAxis, DIALECTS.cube.radialAxis, DIALECTS.stack.radialAxis,
            DIALECTS.descent.radialAxis, DIALECTS.conversation.write, DIALECTS.toolgraph.write, DIALECTS.vortex.read, DIALECTS.lattice.verify]""")
        assert out == ["none", "none", "xz", "y", "pulse", "pulse", "thicken", "align"]

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
        ("horizon", "const hAmp = pulseAmp *", ["gaitFlash"]),
        ("lattice", "const alignMul = 1.0 - TUNE.alignGain * gaitAlign;", ["alignMul", "gaitFlash", "gaitFlow"]),
        ("stack", "stackFlow += (1 / 60) * (0.055", ["gaitFlow", "DIAL.tool === 'ripple'"]),
        ("embedding", "embT += (1 / 60) * (0.24", ["gaitFlow"]),
        ("descent", "const lr = 1.6 * (0.55", ["gaitFlow"]),
        ("cube", "const tk = time * (0.7 + 1.4 * Sk", ["gaitFlow", "gaitAlign"]),
        ("cortex", "const g = 1.0 + 0.55 * pulseAmp * wave", ["gaitFlash"]),
        ("conversation", "const live = bp.isLast ?", ["gait.write"]),
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
        assert "if (axis === 'xz') { p.x *= f; p.z *= f; }" in body
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


# ═══════════════════════════════════════════════════════════════════════════
# Face lab
# ═══════════════════════════════════════════════════════════════════════════

class TestFaceLab:
    def test_every_action_names_a_real_export(self, lab_js, graph_js):
        exports = set(re.findall(r"^export function (\w+)\(", graph_js, re.M))
        actions = eval_js(_fn(lab_js, "labActions"), "labActions()")
        assert len(actions) >= 16
        missing = sorted({a["call"] for a in actions} - exports)
        assert missing == [], f"lab buttons call exports that do not exist: {missing}"
        groups = {a["group"] for a in actions}
        assert groups == {"events", "errors", "gait"}
        gaits = [a["args"][0] for a in actions if a["group"] == "gait"]
        assert gaits == [None, "search", "read", "tool", "verify", "write"]

    def test_the_lab_also_uses_these_exports(self, lab_js, graph_js):
        exports = set(re.findall(r"^export (?:function|const) (\w+)", graph_js, re.M))
        for name in ("getTune", "setTune", "resetTune", "TUNE_DEFAULTS", "getForms", "getForm", "setForm",
                     "getDebugState", "setBackgroundBusy", "setComposerGaze", "setMoodHue", "setAudioLevel",
                     "getAutoForm", "setAutoForm", "fireIdleTwitch"):
            assert name in exports, f"the lab relies on {name}, which the face does not export"
            assert f"face.{name}" in lab_js or f"'{name}'" in lab_js, f"the lab never touches {name}"

    def test_slider_spec(self, lab_js):
        fn = _fn(lab_js, "tuneSliderSpec")
        out = eval_js(fn, "[tuneSliderSpec('toolKick', 0.06), tuneSliderSpec('passHold', 0.992)]")
        assert out[0]["min"] == 0 and out[0]["max"] == pytest.approx(0.24) and out[0]["step"] > 0
        assert out[1] == {"min": 0.9, "max": 0.999, "step": pytest.approx(0.099 / 200)}

    def test_open_builds_the_panel_from_the_face_and_wires_the_buttons(self, lab_js):
        out = eval_js("""
const calls = [];
const face = {
    getTune: () => ({ toolKick: 0.06, exhale: 0.035 }), TUNE_DEFAULTS: { toolKick: 0.06, exhale: 0.035 },
    setTune: (k, v) => { calls.push(['tune', k, v]); return v; }, resetTune: () => calls.push(['reset']),
    getForms: () => ['vortex', 'cube'], getForm: () => 'cube', setForm: (n) => calls.push(['form', n]),
    getDebugState: () => ({ anatomy: 'cube', gait: {}, dialect: { a: 'x' } }),
    setBackgroundBusy: (v) => calls.push(['bg', v]), setComposerGaze: (v) => calls.push(['gaze', v]),
    setMoodHue: (m) => calls.push(['mood', m]), setAudioLevel: () => {}, getAutoForm: () => false, setAutoForm: (v) => calls.push(['auto', v]),
    noteToolCall: (t) => calls.push(['tool', t]), noteRecall: () => calls.push(['recall']), noteVerdict: (k) => calls.push(['verdict', k]),
    fireIdleTwitch: () => calls.push(['twitch']), noteError: (k) => calls.push(['error', k]), setPhase: (p) => calls.push(['phase', p]),
};
const nodes = [];
function mk(tag) { const n = { tag, children: [], attrs: {}, classes: new Set(), on: {}, textContent: '', value: '',
    appendChild(c) { this.children.push(c); c.parent = this; return c; }, replaceChildren() { this.children = []; },
    addEventListener(ev, fn) { this.on[ev] = fn; }, setAttribute(k, v) { this.attrs[k] = v; },
    classList: { add: (c) => n.classes.add(c), remove: (c) => n.classes.delete(c), toggle: (c, on) => on ? n.classes.add(c) : n.classes.delete(c), contains: (c) => n.classes.has(c) },
    get isConnected() { return true; } }; nodes.push(n); return n; }
globalThis.document = { createElement: mk, getElementById: () => null, body: mk('body') };
globalThis.setInterval = () => 1; globalThis.clearInterval = () => {};
function el(tag, cls, text) { const n = mk(tag); if (cls) cls.split(' ').filter(Boolean).forEach(c => n.classes.add(c)); if (text !== undefined) n.textContent = text; return n; }
const toasts = [];
""" + _fn(lab_js, "labActions", "tuneSliderSpec", "openFaceLab").replace("const MOODS =", "var MOODS_UNUSED =") + """
const MOODS = ['', 'curious'];
""", """(() => {
            const panel = openFaceLab({ Core: { activeFace: face }, el, toast: (t) => toasts.push(t) });
            const all = nodes;
            const btn = (label) => all.find(n => n.tag === 'button' && n.textContent === label);
            btn('recall comet').on.click();
            btn('verdict: refute').on.click();
            btn('error: timeout').on.click();
            btn('search').on.click();
            btn('background busy').on.click();
            const sliders = all.filter(n => n.tag === 'input' && n.attrs === n.attrs && n.type === 'range' && n.id && n.id.startsWith('face-lab-tune-'));
            sliders[0].value = '0.1'; sliders[0].on.input();
            return { id: panel.id, calls, sliders: sliders.length, readout: all.some(n => n.classes.has('face-lab-readout')) };
        })()""")
        assert out["id"] == "face-lab" and out["readout"] is True
        assert out["sliders"] == 2, "one slider per tunable"
        assert ["recall"] in out["calls"] and ["verdict", "refute"] in out["calls"]
        assert ["error", "timeout"] in out["calls"] and ["phase", "search"] in out["calls"] and ["bg", True] in out["calls"]
        assert ["tune", "toolKick", 0.1] in out["calls"]

    def test_open_refuses_a_stale_face_module(self, lab_js):
        out = eval_js("""
const toasts = [];
globalThis.document = { getElementById: () => null };
""" + _fn(lab_js, "openFaceLab"), "[openFaceLab({ Core: { activeFace: {} }, el: () => ({}), toast: (t) => toasts.push(t) }), toasts.length]")
        assert out == [None, 1]

    def test_palette_and_shortcut_open_it_lazily(self):
        ws = strip_js_comments((_STATIC / "workspace.js").read_text(encoding="utf-8"))
        pal = strip_js_comments((_STATIC / "palette.js").read_text(encoding="utf-8"))
        assert "import('./facelab.js?v=" in ws and "openFaceLab({ Core, el, toast })" in ws
        assert "e.altKey && e.shiftKey" in ws and "faceLab.open();" in ws
        assert "faceLab" in ws[ws.index("const ctx = {"):ws.index("const ctx = {") + 200]
        assert "{ label: 'Face lab'" in pal and "faceLab && faceLab.open()" in pal
        css = (_STATIC / "style.css").read_text(encoding="utf-8")
        assert "#face-lab {" in css and ".face-lab-slider" in css


def test_touched_modules_bumped():
    index = (_STATIC / "index.html").read_text()
    app = (_STATIC / "app.js").read_text()
    ws = (_STATIC / "workspace.js").read_text()
    assert "app.js?v=12.1" in index and "style.css?v=6.3" in index
    assert "matrix_graph.js?v=12.1" in app and "workspace.js?v=8.5" in app
    assert "palette.js?v=7.1" in ws and "facelab.js?v=1.1" in ws
