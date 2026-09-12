"""The face's signal layer (2026-09-11; trimmed 2026-09-12).

Design rule kept: temperature stays the only thing the eye must track;
what is added is GAIT and EVENT.

  matrix_graph.js  phase gaits (search/read/tool/verify/write), tool kick,
                   recall comet, verdict release (pass/refute/stop),
                   background breath, mood baseline, composer gaze, error
                   kinds (network/refusal/timeout), idle twitch.
  app.js           the ticker → face feed, turn lifecycle → phase/verdict,
                   composer → gaze, mic → audio level, errors → kind.
  status.js        health.mood → hue; /api/turns → background busy.
  routes.py        /api/health carries `mood`.

Removed 2026-09-12 at the operator's request (and pinned as ABSENT here):
the auto form mode, the conversation and tool-graph forms with their hover
labels, and the face lab.

Pure helpers are EXECUTED under node; three.js-bound code is pinned by
the builder harness in test_interface_face_forms_ai.py plus text pins here.
"""

import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
def app_js() -> str:
    return (_STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_nc(app_js) -> str:
    return strip_js_comments(app_js)


@pytest.fixture(scope="module")
def status_js() -> str:
    return strip_js_comments((_STATIC / "status.js").read_text(encoding="utf-8"))


def _fn(src, *names):
    return "".join(extract_js_function(src, n) for n in names)


# ═══════════════════════════════════════════════════════════════════════════
# matrix_graph.js — pure decisions
# ═══════════════════════════════════════════════════════════════════════════

class TestFacePureHelpers:
    def test_phase_setter_accepts_only_the_five_gaits(self, graph_js):
        out = eval_js("const PHASES = ['search','read','tool','verify','write']; let phase = null;\n"
                      + _fn(graph_js, "setPhase"),
                      "[setPhase('search'), setPhase('bogus'), setPhase('write'), setPhase(null)]")
        assert out == ["search", None, "write", None]

    def test_mood_shifts_only_the_cold_pole_and_is_bounded(self, graph_js):
        fn = _fn(graph_js, "moodHueFor")
        out = eval_js(fn, "['satisfied','idle','curious','stuck','overloaded','nonsense',null].map(moodHueFor)")
        assert out[:5] == [-0.035, -0.06, 0.03, 0.05, 0.07]
        assert out[5:] == [0.0, 0.0]
        assert all(abs(v) <= 0.07 for v in out), "a mood offset that large would cross into the hot pole"

    def test_error_kind_classification(self, graph_js):
        fn = _fn(graph_js, "errorKindFor")
        out = eval_js(fn, """[
            errorKindFor('Load failed', null), errorKindFor('Failed to fetch', ''),
            errorKindFor('ReadTimeout contacting the agent', 'UpstreamError'),
            errorKindFor('The request was refused by policy', 'Refusal'),
            errorKindFor('internal server error (error_id=ab12)', 'InternalError')]""")
        assert out == ["network", "network", "timeout", "refusal", "generic"]

    def test_verdict_mapping_and_refute_flinch(self, graph_js):
        out = eval_js("let verdict = null, verdictEnv = 0, flinch = 0;\n" + _fn(graph_js, "noteVerdict"),
                      "[noteVerdict('pass'), verdictEnv, flinch, noteVerdict('refute'), flinch, noteVerdict('whatever')]")
        assert out == ["pass", 1.0, 0, "refute", 0.5, "stop"]

    def test_tool_call_is_a_saturating_kick(self, graph_js):
        out = eval_js("let toolPulse = 0;\n" + _fn(graph_js, "noteToolCall"),
                      "[noteToolCall(), noteToolCall(), toolPulse]")
        assert out == [0.6, 1.0, 1.0], "the kick must saturate at 1"

    def test_the_removed_machinery_is_gone(self, graph_nc):
        """Pin the deletion (2026-09-12): no auto mode, no data forms, no
        hover labels, no lab persistence."""
        for gone in ("setAutoForm", "autoFormFor", "setTaskHint", "setConversation", "describeNodeAt",
                     "_buildConversation", "_buildToolGraph", "_buildStack", "_buildAbyssal", "_buildHorizon",
                     "_buildCortex", "explicitEdges", "nodeLabels", "setTune", "loadTuneOverrides",
                     "fireIdleTwitch", "ghost_face_auto", "ghost_face_tune"):
            assert gone not in graph_nc, f"{gone} is back"
        m = re.search(r"const FORMS = \[(.*?)\];", graph_nc, re.DOTALL)
        assert re.findall(r"'(\w+)'", m.group(1)) == ["vortex", "lattice", "embedding", "descent", "cube", "empty"]


class TestFaceRenderWiring:
    def test_gaits_and_events_are_applied_after_every_anatomy(self, graph_nc):
        i = graph_nc.index("const radial = 1.0 + TUNE.radialSearch * gait.search")
        body = graph_nc[i:i + 2200]
        for needle in ("toolRadial", "backgroundBusy", "writeW", "shudder", "idleTwitch"):
            assert needle in body, needle
        assert graph_nc.index("const radial = 1.0") < graph_nc.index("if (formBlend < 1.0) {"), (
            "gaits must be applied BEFORE the reorganisation blend or a form switch snaps")

    def test_stillness_slows_time_and_the_pulse_clock(self, graph_nc):
        assert "time += 0.005 * (1.0 + dive * 0.6) * motionMul;" in graph_nc
        assert "* (PREFERS_REDUCED_MOTION ? 0.5 : 1.0) * motionMul;" in graph_nc
        i = graph_nc.index("const still = Math.min(1.0, Math.max(")
        body = graph_nc[i:i + 300]
        assert "verdict === 'pass'" in body and "errorKind === 'refusal'" in body and "gait.verify" in body

    def test_search_sweep_rides_the_shader_not_the_seeds(self, graph_js):
        assert "uniform float uSweep;" in graph_js and "uniform float uSweepAngle;" in graph_js
        assert "sweepHeat = uSweep * uSweepHeat * smoothstep(sweepW, 0.0, sweepD)" in graph_js
        assert "nUniforms.uSweep.value = gait.search;" in graph_js
        assert "uSweep: { value: 0.0 }" in graph_js, "the uniform is read by the shader but never declared"

    def test_recall_comet_draws_a_streak_and_flares_the_node(self, graph_nc):
        assert "if (recallSpark > 0 && recallNode >= 0" in graph_nc
        assert "const reach = TUNE.recallReach * recallSpark * recallSpark;" in graph_nc
        assert "i === recallNode ? 1.0 + TUNE.recallFlare * 4.0 * recallSpark * (1.0 - recallSpark) : 1.0" in graph_nc

    def test_mood_and_error_kinds_reach_the_uniforms(self, graph_nc):
        assert "+ moodHue * (FORM === 'vortex' ? 0.4 : 1.0);" in graph_nc
        i = graph_nc.index("let formDim = 0.55;")
        body = graph_nc[i:i + 500]
        assert "errorKind === 'network'" in body and "errorKind === 'timeout'" in body
        assert "verdict === 'pass'" in body

    def test_gaze_moves_the_look_target_not_the_camera(self, graph_nc):
        assert "camera.lookAt(gazeX, gazeY * (1.0 - dive)," in graph_nc
        assert "targetGazeY = active ? -TUNE.gazeY : 0.0;" in graph_nc

    def test_verdict_stop_exhales_and_background_breathes_the_scale(self, graph_nc):
        assert "const exhale = verdict === 'stop' ?" in graph_nc
        assert "const bgBreath = TUNE.bgBreath * backgroundBusy" in graph_nc
        assert "* (1.0 + exhale + bgBreath);" in graph_nc

    def test_debug_state_exposes_the_new_signals(self, graph_nc):
        i = graph_nc.index("export function getDebugState()")
        body = graph_nc[i:i + 900]
        for k in ("phase", "toolPulse", "recallSpark", "verdict", "backgroundBusy", "moodHue", "dialect"):
            assert k in body, k

    def test_the_animate_chain_covers_exactly_the_roster(self, graph_nc):
        # The LAST `if (FORM === 'vortex')` before the gait pass is the anatomy
        # chain (earlier ones pick scene rotation and the camera).
        j = graph_nc.index("const radial = 1.0")
        i = graph_nc.rindex("    if (FORM === 'vortex') {", 0, j)
        chain = graph_nc[i:j]
        branches = re.findall(r"\} else if \(FORM === '(\w+)'\)", chain)
        assert branches == ["lattice", "embedding", "descent", "cube"], branches
        assert "} else if (FORM === 'empty')" not in chain, "empty is the trailing else now"


# ═══════════════════════════════════════════════════════════════════════════
# app.js — the feed
# ═══════════════════════════════════════════════════════════════════════════

class TestTickerFeedsTheFace:
    def test_ticker_signals(self, app_js):
        i = app_js.index("const _FACE_PHASE_BY_TITLE = {")
        table = app_js[i:app_js.index("};", i) + 2]
        fn = table + _fn(app_js, "faceSignalsForTicker")
        out = eval_js(fn, """[
            faceSignalsForTicker('web search', '🌐', 'weather athens'),
            faceSignalsForTicker('tool call', '🧰', 'web_search · 2 args'),
            faceSignalsForTicker('verify', '🧪', 'CONFIRMED conf=0.81 3.2s · answer'),
            faceSignalsForTicker('verify claim', '🧪', 'REFUTED conf=0.30 1.1s'),
            faceSignalsForTicker('memory search', '🔎', '3 hits'),
            faceSignalsForTicker('sandbox exec', '🐚', 'ls'),
            faceSignalsForTicker('something new', '💡', ''),
        ]""")
        assert out[0] == {"phase": "search", "tool": False, "recall": False, "verdict": None}
        assert out[1]["phase"] == "tool" and out[1]["tool"] is True
        assert out[2]["phase"] == "verify" and out[2]["verdict"] == "pass"
        assert out[3]["verdict"] == "refute"
        assert out[4]["recall"] is True and out[4]["phase"] == "search"
        assert out[5] == {"phase": "tool", "tool": False, "recall": False, "verdict": None}
        assert out[6] == {"phase": None, "tool": False, "recall": False, "verdict": None}

    def test_feed_calls_the_face_and_rate_limits_recalls(self, app_js):
        i = app_js.index("const _FACE_PHASE_BY_TITLE = {")
        table = app_js[i:app_js.index("};", i) + 2]
        fns = table + _fn(app_js, "faceSignalsForTicker", "_feedFaceFromTicker")
        out = eval_js("""
const calls = [];
const activeFace = { setPhase: (p) => calls.push(['phase', p]),
    noteToolCall: () => calls.push(['tool']), noteRecall: () => calls.push(['recall']) };
let _turnVerdict = null, _lastRecallAt = 0;
""" + fns, """(() => {
            _feedFaceFromTicker('memory search', '🔎', 'a');
            _feedFaceFromTicker('memory search', '📍', 'b');     // within 1.5s → dropped
            _feedFaceFromTicker('tool call', '🧰', 'file_read · x');
            _feedFaceFromTicker('verify', '🧪', 'REFUTED conf=0.2');
            return { calls, verdict: _turnVerdict };
        })()""")
        assert out["calls"].count(["recall"]) == 1, out["calls"]
        assert ["tool"] in out["calls"] and ["phase", "tool"] in out["calls"]
        assert ["phase", "verify"] in out["calls"] and out["verdict"] == "refute"

    def test_ticker_line_parser_feeds_the_face(self, app_nc):
        i = app_nc.index("function noteTickerLine(raw)")
        body = app_nc[i:i + 2200]
        assert "_feedFaceFromTicker(title, icon, detail);" in body

    def test_turn_lifecycle_sets_phase_and_releases_with_a_verdict(self, app_nc):
        i = app_nc.index("setTurnStatusDesc('writing the reply…', '💬');")
        assert "activeFace.setPhase('write')" in app_nc[i:i + 200]
        i = app_nc.index("activeFace.noteVerdict(_turnVerdict || 'stop');")
        tail = app_nc[i:i + 300]
        assert "_turnVerdict = null;" in tail and "activeFace.setPhase(null)" in tail
        i = app_nc.index("chatHistory.push({ role: \"user\", content: text });")
        head = app_nc[i - 500:i]
        assert "_turnVerdict = null;" in head and "activeFace.setPhase(null)" in head

    def test_composer_gaze(self, app_nc):
        assert "activeFace.setComposerGaze(this.value.trim().length > 0);" in app_nc
        assert "chatInput.addEventListener('blur'" in app_nc

    def test_error_kinds_replace_the_generic_spike_at_both_sites(self, app_nc):
        assert app_nc.count("_faceError(") >= 3      # definition + two call sites
        i = app_nc.index("addRetryableSystemMessage(`Network Error: ${_m}`);")
        assert "_faceError(_m, 'network');" in app_nc[i:i + 200]
        i = app_nc.index("addRetryableSystemMessage(`Error${_type}: ${_msg}${_eid}`);")
        assert "_faceError(_msg, _type);" in app_nc[i:i + 200]
        fn = _fn(app_nc, "_faceError")
        out = eval_js("const activeFace = { errorKindFor: (m) => 'timeout', noteError: (k) => { globalThis.k = k; } };\n" + fn,
                      "[_faceError('x'), globalThis.k]")
        assert out == ["timeout", "timeout"]

    def test_mic_level_reaches_the_face_and_stops_with_the_recorder(self, app_nc):
        assert "_startMicAudioPump(stream);" in app_nc
        i = app_nc.index("mediaRecorder.onstop = async () => {")
        assert "_stopMicAudioPump();" in app_nc[i:i + 120]
        i = app_nc.index("function _startMicAudioPump(stream)")
        body = app_nc[i:i + 1200]
        assert "createMediaStreamSource(stream)" in body and "activeFace.setAudioLevel(" in body
        assert "connect(audioCtx.destination)" not in body, "the microphone must never be routed to the speakers"

    def test_the_removed_wiring_is_gone(self, app_nc):
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        for gone in ("setAutoForm", "getAutoForm", "setTaskHint", "setConversation", "describeNodeAt",
                     "faceTooltip", "_FACE_CHROME", "faceHoverAllowed", "FACE_AUTO_HINT", "'auto-on'",
                     "dataset.form = 'auto'"):
            assert gone not in app_nc, f"{gone} is back in app.js"
        assert 'id="face-tooltip"' not in html
        i = app_nc.index("const FACE_FORM_HINTS = {")
        hints = app_nc[i:app_nc.index("};", i)]
        assert re.findall(r"(\w+): ", hints) == ["vortex", "lattice", "embedding", "descent", "cube", "empty"]
        ws = strip_js_comments((_STATIC / "workspace.js").read_text(encoding="utf-8"))
        pal = strip_js_comments((_STATIC / "palette.js").read_text(encoding="utf-8"))
        assert "facelab" not in ws and "faceLab" not in ws and "faceLab" not in pal and "Face lab" not in pal
        assert not (_STATIC / "facelab.js").exists()
        css = (_STATIC / "style.css").read_text(encoding="utf-8")
        assert "#face-lab" not in css and "#face-tooltip" not in css and "auto-on" not in css


# ═══════════════════════════════════════════════════════════════════════════
# status.js — slow signals
# ═══════════════════════════════════════════════════════════════════════════

class TestStatusFeedsTheFace:
    def test_background_busy_means_someone_elses_running_turn(self, status_js):
        fn = _fn(status_js, "backgroundBusyFrom")
        out = eval_js(fn, """[
            backgroundBusyFrom({turns: [{running: true, session_id: 'dream'}]}, 'mine'),
            backgroundBusyFrom({turns: [{running: true, session_id: 'mine'}]}, 'mine'),
            backgroundBusyFrom({turns: [{running: false, session_id: 'dream'}]}, 'mine'),
            backgroundBusyFrom({turns: [{running: true, session_id: null}]}, null),
            backgroundBusyFrom(null, 'mine'),
        ]""")
        assert out == [True, False, False, False, False]

    def test_feed_face_runs_after_each_poll(self, status_js):
        i = status_js.index("async function pollHealth()")
        assert "feedFace();" in status_js[i:i + 900]
        i = status_js.index("async function feedFace()")
        body = status_js[i:i + 900]
        assert "face.setMoodHue(health && health.mood ? health.mood.label : null)" in body
        assert "fetch('/api/turns'" in body and "face.setBackgroundBusy(backgroundBusyFrom(data, mine))" in body


# ═══════════════════════════════════════════════════════════════════════════
# routes.py — mood on health
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthCarriesMood:
    def _health(self, mood_obj):
        from ghost_agent.api import routes
        state = SimpleNamespace(mood=mood_obj)
        agent = SimpleNamespace(context=SimpleNamespace(self_model=SimpleNamespace(state=state),
                                                         llm_client=None, scheduler=None, memory_system=None))
        app = SimpleNamespace(state=SimpleNamespace(agent=agent, biological_task=None, boot_monotonic=None,
                                                   resolved_config={}))
        req = MagicMock()
        req.app = app
        resp = asyncio.get_event_loop().run_until_complete(routes.api_health(req))
        import json
        return json.loads(resp.body)

    def test_mood_label_and_provenance(self):
        body = self._health(SimpleNamespace(label="curious", source="derived", set_at="2026-09-11T10:00:00Z"))
        assert body["mood"] == {"label": "curious", "source": "derived", "set_at": "2026-09-11T10:00:00Z"}

    def test_callable_mood_and_missing_mood(self):
        body = self._health(lambda: SimpleNamespace(label="stuck", source="self", set_at=""))
        assert body["mood"]["label"] == "stuck"
        assert self._health(None)["mood"] is None
        assert self._health(lambda: None)["mood"] is None


def test_touched_modules_bumped():
    index = (_STATIC / "index.html").read_text()
    app = (_STATIC / "app.js").read_text()
    ws = (_STATIC / "workspace.js").read_text()
    assert "app.js?v=12.2" in index and "style.css?v=6.4" in index
    assert "matrix_graph.js?v=12.2" in app and "workspace.js?v=8.6" in app
    assert "status.js?v=7.4" in ws and "palette.js?v=7.2" in ws
