"""Regression guard for the face's dark-multicolor palette (2026-07-13).

The 2026-07-12 anti-flashbang pass muted the whole graph to near-black
tones; the operator then found it dull. The redesign keeps the dark
theme + smoothed activity envelope but makes the graph genuinely
multicolor: every node owns a stable position on a 5-stop jewel wheel
(violet / blue / teal / emerald / magenta) that drifts slowly, and each
line gradients between its endpoints' hues.

These pins catch regressions that would quietly re-monochrome the face:
- the palette shrinking or disappearing,
- the per-node seed / per-line hue attributes being dropped,
- a return to the single uActiveColor uniform,
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
