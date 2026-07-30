"""Immersion dive (2026-07-13) — source pins.

While a USER request is in flight, the grid "swallows" the camera
(scene scales up + camera dollies from z=5.0 into the cloud at z=1.3)
and drifts back out on completion. Verified live headlessly via the
window.__ghostFace hook: idle camZ=5.00 → swallowed camZ=1.33 /
scale=1.54 → released camZ=4.94, no page errors. These pins guard the
safety-critical parts of the design.
"""

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


@pytest.fixture(scope="module")
def graph_js() -> str:
    return (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


def test_immersion_driven_by_user_turn_only(graph_js):
    """2026-07-28: the dive is gated on the USER-TURN state. workingState
    is re-armed almost continuously by ambient log icons on a busy agent,
    so driving immersion from it held the camera inside the cloud all
    night — the swallow means "working for YOU, right now"."""
    assert "const immersionTarget = userTurnState * IMMERSION_CAP" in graph_js
    assert "workingState * IMMERSION_CAP" not in graph_js
    assert "export function setUserTurn" in graph_js
    app_js = (_STATIC / "app.js").read_text(encoding="utf-8")
    # Only sendMessage's lifecycle may drive the dive (true + false).
    assert app_js.count("setUserTurn(true)") == 1
    assert app_js.count("setUserTurn(false)") == 1


def test_immersion_not_driven_by_activity_envelope(graph_js):
    # The other historical wrong driver: the ambient activity envelope
    # (fires all night on idle-loop work).
    assert "activity * IMMERSION_CAP" not in graph_js


def test_reduced_motion_users_get_capped_lean(graph_js):
    assert "IMMERSION_CAP = PREFERS_REDUCED_MOTION ? 0.15 : 1.0" in graph_js


def test_asymmetric_ease_prevents_yo_yo(graph_js):
    # Attack faster than release, both far slower than workingState's
    # own ramp — short requests only lean, sustained work engulfs.
    assert "immersionTarget > immersion ? 0.012 : 0.006" in graph_js


def test_near_fade_present_in_both_shaders(graph_js):
    # Without near-fade, nodes cross the camera plane as screen-filling
    # quads and lines slash across the whole viewport.
    assert graph_js.count("smoothstep(0.3, 1.4, -mvPosition.z)") == 2
    assert "vNearFade" in graph_js and "vLineNear" in graph_js


def test_bloom_damped_while_inside(graph_js):
    # Inside the cloud the reply is being READ on top of near, bright
    # quads — bloom must scale down with the dive, not up. Deepened
    # 0.35→0.5 plus per-line diveDim after operator feedback
    # ("too bright"): additive line stacking was the brightness driver.
    assert "(1.0 - 0.5 * dive)" in graph_js
    assert "float diveDim = 1.0 - 0.30 * uDive;" in graph_js


def test_lookat_singularity_handled(graph_js):
    # Near the origin, lookAt(0,0,0) turns parallax into wild rotation;
    # the target must blend forward through the cloud with the dive.
    # Since the cube form (2026-07-29) the forward blend is form-aware
    # (the cube's partial dive looks less far through), but it must
    # still scale with `dive` and never target the origin.
    assert re.search(
        r"camera\.lookAt\(0, 0, \(FORM === 'cube' \? -[\d.]+ : -3\.5\) \* dive\)",
        graph_js)
    assert "camera.lookAt(0, 0, 0);" not in graph_js


def test_debug_hook_exported_for_headless_verification(graph_js):
    assert "export function getDebugState()" in graph_js
    app_js = (_STATIC / "app.js").read_text(encoding="utf-8")
    assert "window.__ghostFace = activeFace" in app_js


def test_backup_of_preimmersion_face_exists():
    # Operator-requested restore point for the pre-immersion face.
    assert (_STATIC / "matrix_graph.js.bak-20260713-preimmersion").exists()


def test_interior_enrichment_present(graph_js):
    # Operator feedback: "fully zoomed it looks kind of empty". Three
    # compensators, all gated on the dive so the resting view is
    # untouched: interior motes (shader-drifted, zero per-frame CPU,
    # skipped entirely at rest), a thicker proximity web, and faster
    # data pulses inside.
    assert "MOTE_COUNT = IS_MOBILE ? 150 : 400" in graph_js
    assert "motesMesh.visible = dive > 0.01" in graph_js
    assert "PROXIMITY_SQ * (1.0 + dive * 0.15)" in graph_js
    assert "time += 0.005 * (1.0 + dive * 0.6)" in graph_js
    # Scale boost deliberately trimmed — the swell dilutes local density
    # exactly when the camera is closest (the original emptiness bug).
    assert "dive * 0.55" in graph_js
