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
        """§4TC: the nine moods — low energy cooler (negative), high energy
        warmer (positive); "neutral", the labels the removed derived updater
        wrote and anything unknown are 0."""
        fn = _fn(graph_js, "moodHueFor")
        out = eval_js(fn, "['content','calm','down','bored','tired','cheerful','curious','restless','irritable']"
                          ".map(moodHueFor)")
        assert out == [-0.015, -0.03, -0.035, -0.04, -0.045, 0.025, 0.03, 0.04, 0.045]
        rest = eval_js(fn, "['neutral','satisfied','idle','stuck','overloaded','nonsense','',null,undefined,7]"
                           ".map(moodHueFor)")
        assert rest == [0.0] * 10
        assert eval_js(fn, "moodHueFor('IRRITABLE')") == 0.045

    def test_the_intensity_scales_the_mood_shift_and_stays_in_the_cold_pole(self, graph_js):
        fns = ("let targetMoodHue = 0, _moodLive = null; const moodLook = () => null;\n" + _fn(graph_js, "moodHueFor") + "\n"
               + _fn(graph_js, "moodHueScale") + "\n" + _fn(graph_js, "setMoodHue"))
        assert eval_js(fns, "[1, 2, 3, 0, null, undefined, 'x', 9].map(moodHueScale)") == [0.5, 1, 1.5, 1, 1, 1, 1, 1]
        got = eval_js(fns, "[setMoodHue('irritable', 1), setMoodHue('irritable', 2), setMoodHue('irritable', 3),"
                           " setMoodHue('irritable'), setMoodHue('tired', 3), setMoodHue('neutral', 0),"
                           " setMoodHue(null, null)]")
        assert got == pytest.approx([0.0225, 0.045, 0.0675, 0.045, -0.0675, 0.0, 0.0])
        labels = "['content','calm','down','bored','tired','cheerful','curious','restless','irritable']"
        every = eval_js(fns, f"{labels}.flatMap(l => [1, 2, 3].map(n => setMoodHue(l, n)))")
        assert all(abs(v) <= 0.07 for v in every), "a mood offset that large would cross into the hot pole"

    def test_error_kind_classification(self, graph_js):
        fn = _fn(graph_js, "errorKindFor")
        out = eval_js(fn, """[
            errorKindFor('Load failed', null), errorKindFor('Failed to fetch', ''),
            errorKindFor('ReadTimeout contacting the agent', 'UpstreamError'),
            errorKindFor('The request was refused by policy', 'Refusal'),
            errorKindFor('internal server error (error_id=ab12)', 'InternalError')]""")
        assert out == ["network", "network", "timeout", "refusal", "generic"]

    def test_verdict_mapping_and_refute_flinch(self, graph_js):
        out = eval_js("let verdict = null, verdictEnv = 0, flinch = 0, greens = []; const temper = { event: () => 1 };\n"
                      "const SUCCESS_VERIFIED = 1; function noteSuccess(s) { greens.push(s); }\n" + _fn(graph_js, "noteVerdict"),
                      "[noteVerdict('pass'), verdictEnv, flinch, noteVerdict('refute'), flinch, noteVerdict('whatever'), greens]")
        # 2026-11-09: a pass also starts the green front, once; refute and stop do not
        assert out == ["pass", 1.0, 0, "refute", 0.5, "stop", [1]]

    def test_tool_call_is_a_saturating_kick(self, graph_js):
        # the temperament's gain (2026-10-30) is 1 here: this pins the kick itself
        out = eval_js("let toolPulse = 0; const temper = { event: () => 1 };\n" + _fn(graph_js, "noteToolCall"),
                      "[noteToolCall(), noteToolCall(), toolPulse]")
        assert out == [0.6, 1.0, 1.0], "the kick must saturate at 1"

    def test_the_removed_machinery_is_gone(self, graph_nc):
        """Pin the deletion (2026-09-12): no auto mode, no data forms, no
        hover labels, no lab persistence."""
        for gone in ("setAutoForm", "autoFormFor", "setTaskHint", "setConversation", "describeNodeAt",
                     "_buildConversation", "_buildToolGraph", "_buildStack", "_buildAbyssal", "_buildHorizon",
                     "_buildCortex", "explicitEdges", "nodeLabels", "setTune", "loadTuneOverrides",
                     "fireIdleTwitch", "ghost_face_auto", "ghost_face_tune",
                     "idleTwitch", "twitchAmp", "toolKick", "'kick'"):
            assert gone not in graph_nc, f"{gone} is back"
        m = re.search(r"const FORMS = \[(.*?)\];", graph_nc, re.DOTALL)
        assert re.findall(r"'(\w+)'", m.group(1)) == ["moodtesseract", "empty"]


class TestFaceRenderWiring:
    def test_gaits_and_events_are_applied_after_every_anatomy(self, graph_nc):
        i = graph_nc.index("const radial = 1.0 + TUNE.radialSearch * gait.search")
        body = graph_nc[i:i + 2200]
        for needle in ("backgroundBusy", "writeW", "shudder"):   # toolRadial/idleTwitch removed 2026-11-09 (dead)
            assert needle in body, needle
        assert graph_nc.index("const radial = 1.0") < graph_nc.index("if (formBlend < 1.0) {"), (
            "gaits must be applied BEFORE the reorganisation blend or a form switch snaps")

    def test_stillness_slows_time_and_the_pulse_clock(self, graph_nc):
        # dt-scaled since 2026-09-20 (tStep = the clock advance this frame).
        assert "const tStep = 0.005 * motionMul * dtF * _tsSpeed();" in graph_nc   # camDive since 2026-09-21; temperament speed since 2026-10-30
        assert "time += tStep;" in graph_nc
        assert "* (PREFERS_REDUCED_MOTION ? 0.5 : 1.0) * motionMul * _tsSpeed() * _moodNow.pulse;" in graph_nc
        i = graph_nc.index("const still = Math.min(1.0, Math.max(")
        body = graph_nc[i:i + 300]
        assert "verdict === 'pass'" in body and "errorKind === 'refusal'" in body and "gait.verify" in body

    def test_search_sweep_rides_the_shader_not_the_seeds(self, graph_js):
        assert "uniform float uSweep;" in graph_js and "uniform float uSweepAngle;" in graph_js
        # §4PR: the reversed-edge smoothstep(sweepW, 0.0, d) is undefined in
        # GLSL ES; 1 − smoothstep(0, sweepW, d) is its exact value.
        assert "sweepHeat = uSweep * uSweepHeat * (1.0 - smoothstep(0.0, sweepW, sweepD))" in graph_js
        assert "nUniforms.uSweep.value = gait.search;" in graph_js
        assert "uSweep: { value: 0.0 }" in graph_js, "the uniform is read by the shader but never declared"

    def test_recall_comet_draws_a_streak_and_flares_the_node(self, graph_nc):
        assert "if (recallSpark > 0 && recallNode >= 0" in graph_nc
        assert "const reach = TUNE.recallReach * recallSpark * recallSpark;" in graph_nc
        assert "i === recallNode ? 1.0 + TUNE.recallFlare * 4.0 * recallSpark * (1.0 - recallSpark) : 1.0" in graph_nc

    def test_mood_and_error_kinds_reach_the_uniforms(self, graph_nc):
        assert "hueDrift * _moodNow.drift + moodHue + _moodNow.hue;" in graph_nc
        i = graph_nc.index("let formDim = 0.55;")
        body = graph_nc[i:i + 500]
        assert "errorKind === 'network'" in body and "errorKind === 'timeout'" in body
        assert "verdict === 'pass'" in body

    def test_gaze_moves_the_look_target_not_the_camera(self, graph_nc):
        assert "_lookTarget(gazeX, gazeY * (1.0 - dive), -3.5 * dive);" in graph_nc   # §4PR: target, then blended lookAt
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
        # The LAST corridor test before the gait pass is the anatomy chain; empty
        # is its trailing else.
        j = graph_nc.index("const radial = 1.0")
        i = graph_nc.rindex("    if (FORM === 'moodtesseract') {", 0, j)
        assert re.findall(r"\} else if \(", graph_nc[i:j].split("        const axis")[0]) is not None
        assert "} else if (FORM ===" not in graph_nc[i:j]


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
            faceSignalsForTicker('memory recall', '🔎', '3 hits'),
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
            _feedFaceFromTicker('memory recall', '🔎', 'a');
            _feedFaceFromTicker('memory recall', '📍', 'b');     // within 1.5s → dropped
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
        # rindex: §4PR added a no-session fallback in the resume hand-off
        # above; the face signal follows the whole branch
        i = app_nc.rindex("addRetryableSystemMessage(`Network Error: ${_m}`);")
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
        assert "FACE_FORM_HINTS" not in app_nc and "face-form" not in html
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
        assert "face.setMoodHue(health && health.mood ? health.mood.label : null," in body
        assert "health && health.mood ? health.mood.intensity : null);" in body        # §4TC
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

    def test_mood_label_intensity_manner_and_who_set_it(self):
        """§4TC: one shape, always an object."""
        body = self._health(lambda: SimpleNamespace(label="irritable", intensity=2, manner="sarcastic",
                                                    source="owner", set_at="2026-09-11T10:00:00Z",
                                                    cause="the owner said so (set_mood)",
                                                    held_until="2026-09-12T02:00:00Z"))
        assert body["mood"] == {"label": "irritable", "intensity": 2, "manner": "sarcastic",
                                "set_at": "2026-09-11T10:00:00Z", "source": "owner",
                                "cause": "the owner said so (set_mood)", "held_until": "2026-09-12T02:00:00Z"}
        body = self._health(lambda: SimpleNamespace(label="bored", intensity=1, manner="", source="auto", set_at=""))
        assert body["mood"] == {"label": "bored", "intensity": 1, "manner": "", "set_at": "", "source": "auto",
                                "cause": "", "held_until": ""}

    def test_the_cause_is_capped_and_a_neutral_mood_has_none(self):
        """§4TX: the chip's tooltip says why — in the system's own words, capped."""
        body = self._health(lambda: SimpleNamespace(label="down", intensity=1, manner="", source="auto",
                                                    set_at="", cause="x" * 500))
        assert body["mood"]["cause"] == "x" * 200
        body = self._health(lambda: SimpleNamespace(label="neutral", intensity=0, manner="", source="auto",
                                                    set_at="", cause="it wore off", held_until="2026-09-12T02:00:00Z"))
        assert (body["mood"]["cause"], body["mood"]["held_until"]) == ("", "")

    NEUTRAL = {"label": "neutral", "intensity": 0, "manner": "", "set_at": "", "source": "auto", "cause": "",
               "held_until": ""}

    def test_a_neutral_mood_keeps_the_manner_and_names_no_owner(self):
        body = self._health(lambda: SimpleNamespace(label="neutral", intensity=0, manner="formal",
                                                    source="owner", set_at="2026-09-11T10:00:00Z"))
        assert body["mood"] == {**self.NEUTRAL, "manner": "formal"}

    def test_a_missing_or_unreadable_mood_is_the_neutral_object(self):
        def boom():
            raise RuntimeError("store down")
        for bad in (None, lambda: None, boom, "irritable", SimpleNamespace(label="irritable"),
                    lambda: SimpleNamespace(label="irritable", intensity="2"),
                    lambda: SimpleNamespace(label=7, intensity=2),
                    lambda: SimpleNamespace(label="irritable", intensity=True)):
            assert self._health(bad)["mood"] == self.NEUTRAL

    def test_an_out_of_range_intensity_is_brought_into_range(self):
        body = self._health(lambda: SimpleNamespace(label="tired", intensity=9, manner=None, source="x", set_at=None))
        assert body["mood"] == {"label": "tired", "intensity": 3, "manner": "", "set_at": "", "source": "auto",
                                "cause": "", "held_until": ""}

    def test_the_real_state_thread_feeds_the_endpoint(self, tmp_path):
        from ghost_agent.selfhood import SelfModel
        from ghost_agent.selfhood.mood import ANCHOR_DIRECTION
        sm = SelfModel(root=tmp_path)
        assert self._health(sm.state.mood)["mood"] == self.NEUTRAL
        before = sm.state.path.exists()
        sm.state.set_owner_mood("cheerful", ANCHOR_DIRECTION["cheerful"], 3)
        sm.state.set_manner("playful", 1)
        raw = sm.state.path.read_bytes()
        body = self._health(sm.state.mood)["mood"]
        assert (body["label"], body["intensity"], body["manner"], body["source"]) == ("cheerful", 3, "playful", "owner")
        assert body["set_at"].endswith("Z") and body["held_until"].endswith("Z")       # §4TX: held as set
        assert body["cause"] == "the owner said so (set_mood)"
        assert sm.state.path.read_bytes() == raw and before is False       # the endpoint only reads


def test_touched_modules_bumped():
    index = (_STATIC / "index.html").read_text()
    app = (_STATIC / "app.js").read_text()
    ws = (_STATIC / "workspace.js").read_text()
    assert "app.js?v=19.18" in index and "style.css?v=8.7" in index
    assert "matrix_graph.js?v=19.18" in app and "workspace.js?v=9.6" in app
    assert "status.js?v=7.8" in ws and "palette.js?v=7.6" in ws


def test_the_chip_tooltip_names_the_mood_its_cause_and_the_manner(status_js):
    """§4TX: the chip's title says the mood in words, its cause, the hold and the manner."""
    fn = extract_js_function(status_js, "moodNote")
    out = eval_js(fn, "[moodNote({mood: {label: 'irritable', intensity: 2, cause: '3 tool calls failed', "
                      "manner: 'terse', held_until: ''}}), "
                      "moodNote({mood: {label: 'calm', intensity: 1, cause: 'the owner said so (set_mood)', "
                      "held_until: '2026-11-10T02:00:00Z', manner: ''}}), "
                      "moodNote({mood: {label: 'neutral', intensity: 0, manner: ''}}), moodNote(null), moodNote({})]")
    assert out == ["\nmood: irritable, 2 of 3 — 3 tool calls failed\nmanner: terse",
                   "\nmood: calm, 1 of 3 — the owner said so (set_mood) (held as the owner set it)",
                   "", "", ""]
