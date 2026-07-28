"""Regression guard for the face's THERMAL two-pole palette (2026-07-28).

History: the 2026-07-12 anti-flashbang pass muted the graph; 2026-07-13
made it a 5-stop jewel wheel; on 2026-07-28 the operator found the jewel
wheel "too colorful, attracts too much attention" and set the current
direction — exactly two poles, dark ocean blue and dark arterial red,
mutating into each other through dark violet bridges. The ring still has
5 stops (the shader's palette() contract is unchanged) but every stop
must live on the blue↔red axis: no greens, no yellows.

These pins catch regressions that would quietly break the scheme:
- the palette shrinking, brightening, or drifting off the thermal axis,
- the per-node seed / per-line hue attributes being dropped,
- a return to the single uActiveColor uniform,
- the spatial hue-wave (the organic mutation) being dropped,
- cache-bust versions not bumped together (stale-module skew).
"""

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def graph_js() -> str:
    return (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


def test_palette_has_five_stops(graph_js):
    m = re.search(r"palette:\s*\[(.*?)\]", graph_js, re.DOTALL)
    assert m, "COLORS.palette missing from matrix_graph.js"
    stops = re.findall(r"#([0-9a-fA-F]{6})", m.group(1))
    assert len(stops) == 5
    # Multicolor means genuinely distinct hues, not five shades of one.
    assert len(set(s.lower() for s in stops)) == 5


def test_palette_stays_dark(graph_js):
    # Additive blending + bloom lift these several stops; a stop with a
    # near-max channel is the flashbang regression the 2026-07-12 pass
    # fixed. Guard: no palette channel above 0xd0.
    m = re.search(r"palette:\s*\[(.*?)\]", graph_js, re.DOTALL)
    for hexcode in re.findall(r"#([0-9a-fA-F]{6})", m.group(1)):
        channels = [int(hexcode[i:i + 2], 16) for i in (0, 2, 4)]
        assert max(channels) <= 0xD0, f"palette stop #{hexcode} too bright"


def test_palette_stays_on_the_thermal_axis(graph_js):
    # Two-pole scheme: every stop is a blue, a red, or a violet bridge —
    # green must never be the dominant channel (that's the jewel wheel's
    # teal/emerald sneaking back in).
    m = re.search(r"palette:\s*\[(.*?)\]", graph_js, re.DOTALL)
    for hexcode in re.findall(r"#([0-9a-fA-F]{6})", m.group(1)):
        r, g, b = (int(hexcode[i:i + 2], 16) for i in (0, 2, 4))
        assert g < max(r, b), f"stop #{hexcode} has a dominant green channel"


def test_alien_forms_present(graph_js):
    """2026-07-28: 'alive' is a property of anatomy + muscle-like motion.
    Three interchangeable alien body plans (abyssal / horizon / cortex)
    share ONE propulsion engine — the asymmetric _pulseShape envelope,
    strand waves, hot core, flinch — and a header button cycles them."""
    assert "_pulseShape" in graph_js
    assert "kind: 0" in graph_js and "kind: 1" in graph_js and "kind: 2" in graph_js
    for builder in ("_buildAbyssal", "_buildHorizon", "_buildCortex"):
        assert builder in graph_js, f"missing {builder}"
    assert "export function cycleForm" in graph_js
    assert "export function setForm" in graph_js
    assert "ghost_face_form" in graph_js, "form choice must persist"
    assert "swayAmp" in graph_js, "strand wave params missing"
    # The propulsion envelope must stay asymmetric (squeeze ≪ release).
    assert "if (x < 0.16)" in graph_js and "if (x < 0.62)" in graph_js
    # Uniform-random scatter must not come back.
    assert "Uniform spherical distribution" not in graph_js


def test_form_button_wired(graph_js):
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="face-form-btn"' in html
    app_js = (_STATIC / "app.js").read_text(encoding="utf-8")
    assert "cycleForm()" in app_js


def test_hue_wave_travels_spatially(graph_js):
    # The organic mutation: hue drift must travel through the cloud as a
    # spatial wave, not tick uniformly.
    assert "hueWave(" in graph_js
    assert "uOrganic" in graph_js


def test_shaders_use_wheel_not_single_active_color(graph_js):
    # Both shaders sample the wheel with the drift offset.
    assert graph_js.count("palette(") >= 2
    assert "uHueDrift" in graph_js
    # The single-hue uniform must not come back.
    assert "uActiveColor" not in graph_js


def test_per_node_and_per_line_hue_attributes_wired(graph_js):
    assert "aSeed" in graph_js
    assert "InstancedBufferAttribute(nodeSeeds, 1)" in graph_js
    assert "aLineHue" in graph_js
    # The per-frame line builder writes endpoint hues from the seeds.
    assert "nodeSeeds[i]" in graph_js and "nodeSeeds[j]" in graph_js


def test_cache_bust_versions_move_together(app_js):
    # index.html loads app.js?v=N and app.js imports matrix_graph.js?v=N.
    # Editing the module without bumping both leaves one browser-cached —
    # the doc'd cache-bust discipline (docs/interfaces/web_server.html).
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    app_v = re.search(r"app\.js\?v=([\d.]+)", html)
    graph_v = re.search(r"matrix_graph\.js\?v=([\d.]+)", app_js)
    assert app_v and graph_v
    assert app_v.group(1) == graph_v.group(1)
