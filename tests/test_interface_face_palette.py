"""Regression guard for the face's THERMAL two-pole palette (2026-07-28).

History: the 2026-07-12 anti-flashbang pass muted the graph; 2026-07-13
made it a 5-stop jewel wheel; on 2026-07-28 the operator found the jewel
wheel "too colorful, attracts too much attention" and set the current
direction — exactly two poles, dark ocean blue and dark arterial red,
mutating into each other through dark violet bridges. The ring still has
5 stops (the shader's palette() contract is unchanged) but every stop
must live on the blue↔red axis: no greens, no yellows. 2026-11-09: the
operator added ONE dark green, at the rim (stop 0), and the ring became a
clamped line (kernel red → purple → blue → green rim).

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


def test_palette_is_the_depth_scale(graph_js):
    # 2026-11-09 (operator): red kernel → purple → blue → DARK GREEN rim.
    # Exactly one green stop, the FIRST (the rim, nearest the camera); the
    # rest stay on the thermal axis (green never dominant), so the jewel
    # wheel's teal/emerald cannot creep into the middle of the scale.
    m = re.search(r"palette:\s*\[(.*?)\]", graph_js, re.DOTALL)
    stops = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
             for h in re.findall(r"#([0-9a-fA-F]{6})", m.group(1))]
    r, g, b = stops[0]
    assert g > max(r, b), "stop 0 (the rim) must be the green"
    assert g <= 0x50, "the rim green must stay dark"
    for r, g, b in stops[1:]:
        assert g < max(r, b), "only the rim stop may be green"
    blue, violet, red = stops[1], stops[2], stops[3]
    assert blue[2] > blue[0] and red[0] > red[2] and violet[0] > blue[0] and violet[2] > red[2]


def test_palette_clamps_instead_of_wrapping(graph_js):
    # A wrapped ring would blend the red kernel straight into the green rim
    # whenever the drift pushed it past the last stop (red↔green = mud).
    i = graph_js.index("vec3 palette(float t)")
    body = graph_js[i:i + 300]
    assert "clamp(t, 0.0, 0.8)" in body and "fract(t)" not in body


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
    # the doc'd cache-bust discipline (docs/reference/interfaces/web_server.html).
    html = (_STATIC / "index.html").read_text(encoding="utf-8")
    app_v = re.search(r"app\.js\?v=([\d.]+)", html)
    graph_v = re.search(r"matrix_graph\.js\?v=([\d.]+)", app_js)
    assert app_v and graph_v
    assert app_v.group(1) == graph_v.group(1)
