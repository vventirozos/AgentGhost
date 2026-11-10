"""Face 2026-11-09: green rim + wandering centre with a following camera.

Operator: "the center remains red, gets purple, then blue and at the outer
(closest to the camera) it gets dark green" and "what if the center moved
slightly and the camera followed it".
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
def _js():
    return (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


def test_depth_gradient_reaches_green_inside_the_visible_shells():
    js = _js()
    span = float(re.search(r"const C2_HUE_SPAN = ([0-9.]+);", js).group(1))
    # Kernel seed 0.60 falls to the green stop (seed 0) before the shells
    # start fading past the camera (d = 0.78).
    assert 0.60 / span < 0.78
    assert js.count("d * C2_HUE_SPAN * c2Slope") == 2       # shells and stream grains
    assert "d * 0.52 * c2Slope" not in js and "d * 0.55 * c2Slope" not in js


def test_path_phase_is_integrated_not_time_times_rate():
    # A rate that eases with work must step an integrated phase (the
    # §4JA class: time × variable rate jumps when the rate moves).
    js = _js()
    assert re.search(r"c2PathPhase \+= \(dtF / 60\) \* pathRate", js)
    assert "time * pathRate" not in js


def test_shells_keep_the_path_point_of_their_birth():
    js = _js()
    i = js.index("function _c2Frame(")
    body = js[i:i + 1600]
    assert "_c2PathAt(c2PathPhase - C2_PATH_LAG * d" in body
    assert "c2PathAmp * P.x" in body


def test_camera_follows_the_kernel_and_bank_is_capped():
    js = _js()
    i = js.index("_c2KW.copy(_c2K).applyMatrix4(scene.matrixWorld)")
    block = js[i:i + 2000]
    assert "camera.position.x = _parX + _camFollowX" in block
    assert "C2_BANK_MAX" in block and "camera.up.set(" in block
    bank = float(re.search(r"const C2_BANK_MAX = ([0-9.]+);", js).group(1))
    assert bank <= 0.1
    # Other forms get the plain camera back (no leftover roll or follow).
    assert "camera.up.set(0, 1, 0)" in js and "camera.position.x = _parX;" in js
