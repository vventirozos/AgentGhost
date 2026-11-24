"""§4QR → §4VJ — the image node samples the Turbo checkpoint's distilled
trajectory more densely than the card's 8 steps.

§4QR (2026-10-22) ran 16 steps: the 8-step `sample_sigmas` interpolated in
index space, every original sigma kept (8-step grain on textured surfaces
resolved, faces stayed clean; sd-cli's own 16-step schedule over-sharpens).
§4VJ (2026-11-23) moved to 12 steps on the same trajectory at the 1152x768
envelope; the 16-step list stays as TURBO_SIGMAS_16 and the 8-step list as
TURBO_SIGMAS_8, for rollback. (File name kept: other tooling references it.)"""
from __future__ import annotations

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "interface", "externals", "image_generation", "img_gen_server.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("img_gen_server_4qr", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_schedule_is_twelve_steps_on_the_distilled_trajectory(mod):
    s12, s8 = list(mod.TURBO_SIGMAS), list(mod.TURBO_SIGMAS_8)
    assert len(s12) == 13 and len(s8) == 9 and mod.TURBO_STEPS == 12
    assert s12[0] == 1.0 and s12[-1] == 0.0
    assert all(a > b for a, b in zip(s12, s12[1:]))                      # strictly decreasing
    # on the card's trajectory: its endpoints and inner points are kept verbatim
    assert s8[0] == 1.0 and s8[-1] == 0.0
    for kept in (0.95418, 0.89508, 0.704534):
        assert kept in s8 and kept in s12, kept
    assert s12[::3] == s8[::2]                                            # every third point = the card's even points


def test_the_rollback_schedules_are_kept(mod):
    s16, s8 = list(mod.TURBO_SIGMAS_16), list(mod.TURBO_SIGMAS_8)
    assert len(s16) == 17 and len(s8) == 9
    assert s16[0] == s8[0] == 1.0 and s16[-1] == s8[-1] == 0.0
    assert all(a > b for a, b in zip(s16, s16[1:]))
    assert all(a > b for a, b in zip(s8, s8[1:]))
    assert s16[::2] == s8                                                 # §4QR: every original sigma, at the even positions
    for i in range(1, 17, 2):                                             # the odd ones are the midpoints
        assert abs(s16[i] - (s16[i - 1] + s16[i + 1]) / 2) < 1e-6
    assert s16 != list(mod.TURBO_SIGMAS) != s8                            # neither is the live schedule


def test_every_render_runs_twelve_steps_on_that_schedule(mod):
    for asked, editing in ((None, False), (8, False), (16, False), (30, True), (1, True), ("x", True)):
        assert mod.resolve_steps(asked, editing) == 12
    argv = mod.build_sd_cli_args("a cat", 1152, 768, mod.TURBO_STEPS, "o.png", seed=1)
    assert argv[argv.index("--steps") + 1] == "12"
    assert [float(x) for x in argv[argv.index("--sigmas") + 1].split(",")] == list(mod.TURBO_SIGMAS)
    assert "--sigmas" not in mod.build_sd_cli_args("a cat", 1152, 768, 8, "o.png")    # 8 is not the model's schedule
    assert "--sigmas" not in mod.build_sd_cli_args("a cat", 1152, 768, 16, "o.png")   # nor is 16 any more


def test_the_timeouts_cover_a_twelve_step_1152_edit(mod):
    # §4VJ measured at 1152x768, 12 steps, cfg 1: edit 375 s, 2-subject 369 s
    assert mod.GEN_TIMEOUT_S >= 2 * 375                                   # the same ~2x margin as before
    assert mod.BUSY_WAIT_TIMEOUT > mod.GEN_TIMEOUT_S                      # outlasts one full render ahead
    assert mod.BUSY_WAIT_TIMEOUT + mod.GEN_TIMEOUT_S + mod.RESPONSE_MARGIN_S <= mod.CLIENT_TIMEOUT_S
    assert mod.WORST_GENERATION_S >= 365
    from ghost_agent.core import llm as L
    assert L.IMAGE_NODE_TIMEOUT_S == mod.CLIENT_TIMEOUT_S                 # the node derives from the agent's ceiling
    assert L.IMAGE_NODE_SLOT_WAIT_S == mod.BUSY_WAIT_TIMEOUT


def test_the_agents_defaults_and_wording_follow(mock_context):
    import json
    import ghost_agent.tools.image_gen as IG
    import ghost_agent.tools.registry as R
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert IG.DEFAULT_RENDER_SECONDS == {"edit": 375.0, "generate": 245.0}   # §4VJ measured
    IG.RECENT_RENDER_SECONDS["generate"].clear(); IG.RECENT_RENDER_SECONDS["edit"].clear()
    os.environ.pop(IG.LOOK_ENV, None)
    assert IG.expected_render_seconds({"prompt": "x"}) == 245.0
    tools = R.get_active_tool_definitions(mock_context, None, serve_tuned=False)
    text = json.dumps([t for t in tools if "image_generation" in json.dumps(t)])
    assert "(~4 min)" in text and "an edit, ~6 min" in text and "takes ~6 min" in text
    assert "(~2 min)" not in text and "an edit, ~3 min" not in text and "takes ~3 min" not in text
    assert "about 4 minutes on the node (about 6 with" in SYSTEM_PROMPT and "An edit takes about 6 minutes" in SYSTEM_PROMPT
    assert "`subjects` (about 6 minutes;" in SYSTEM_PROMPT
    assert "about 3 minutes" not in SYSTEM_PROMPT and "about 2 minutes on the node" not in SYSTEM_PROMPT
