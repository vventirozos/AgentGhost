"""The face's signal layer (2026-09-11) — "make the faces more interesting".

Fourteen ideas, one design rule kept: temperature stays the only thing the
eye must track; what is added is GAIT and EVENT.

  matrix_graph.js  phase gaits (search/read/tool/verify/write), tool kick,
                   recall comet, verdict release (pass/refute/stop),
                   background breath, mood baseline, composer gaze, error
                   kinds (network/refusal/timeout), idle twitch, auto form,
                   two data forms (conversation, toolgraph) with explicit
                   edges + hover labels.
  app.js           the ticker → face feed, turn lifecycle → phase/verdict,
                   history → conversation form, composer → gaze, mic →
                   audio level, errors → kind, pointer → tooltip, the menu's
                   auto item.
  status.js        health.mood → hue; /api/turns → background busy.
  routes.py        /api/health carries `mood`.

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
    """Extract functions; `export function` becomes a plain function."""
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

    def test_conversation_layout_pairs_question_and_answer(self, graph_js):
        fn = _fn(graph_js, "conversationLayout")
        out = eval_js(fn, """(() => {
            const u = conversationLayout(4, 10, 'user'), a = conversationLayout(5, 10, 'assistant');
            const first = conversationLayout(0, 10, 'user'), last = conversationLayout(9, 10, 'user');
            return { ru: u.r, ra: a.r, climbs: last.y > first.y, single: conversationLayout(0, 1, 'user').y };
        })()""")
        assert out["ru"] > out["ra"], "user turns must ride the OUTER rail"
        assert out["climbs"] is True and out["single"] == 0.0

    def test_tool_graph_layout_is_a_ring(self, graph_js):
        fn = _fn(graph_js, "toolGraphLayout")
        out = eval_js(fn, "[0,1,2,3].map(j => { const p = toolGraphLayout(j, 4); return Math.round(Math.hypot(p.x, p.z) * 100) / 100; })")
        assert out == [1.45, 1.45, 1.45, 1.45]

    def test_mood_shifts_only_the_cold_pole_and_is_bounded(self, graph_js):
        fn = _fn(graph_js, "moodHueFor")
        out = eval_js(fn, "['satisfied','idle','curious','stuck','overloaded','nonsense',null].map(moodHueFor)")
        assert out[:5] == [-0.035, -0.06, 0.03, 0.05, 0.07]
        assert out[5:] == [0.0, 0.0]
        assert all(abs(v) <= 0.07 for v in out), "a mood offset that large would cross into the hot pole"

    def test_auto_form_follows_the_task_and_returns_to_the_pick(self, graph_js):
        fn = _fn(graph_js, "autoFormFor")
        out = eval_js(fn, "['coding','research','verify','memory',null,'unknown'].map(h => autoFormFor(h, 'cube'))")
        assert out == ["lattice", "embedding", "descent", "embedding", "cube", "cube"]
        assert eval_js(fn, "autoFormFor(null, null)") == "vortex"

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

    def test_tool_calls_accumulate_habits(self, graph_js):
        pre = "let toolPulse = 0, dataDirty = false; const toolUsage = new Map(); const toolSeq = [];\n"
        out = eval_js(pre + _fn(graph_js, "noteToolCall"), """(() => {
            noteToolCall('Web_Search'); noteToolCall('file_read'); noteToolCall('web_search'); noteToolCall('');
            return { pulse: toolPulse, tools: [...toolUsage.entries()].map(([k, v]) => [k, v.count]), seq: toolSeq, dirty: dataDirty };
        })()""")
        assert out["pulse"] == 1.0, "pulse must saturate at 1"
        assert out["tools"] == [["web_search", 2], ["file_read", 1]]
        assert out["seq"] == ["web_search", "file_read", "web_search"] and out["dirty"] is True

    def test_set_conversation_keeps_only_dialogue_and_strips_reasoning(self, graph_js):
        pre = "const conversation = []; let dataDirty = false;\n"
        out = eval_js(pre + _fn(graph_js, "setConversation"), """(() => {
            const n = setConversation([
                {role: 'user', content: 'hello there'},
                {role: 'assistant', content: '<think>secret</think>Hi!  How are you'},
                {role: 'tool', content: 'ignored'},
                {role: 'user', content: [{type: 'text', text: 'what is this'}, {type: 'image_url', image_url: {url: 'data:x'}}]},
            ]);
            return { n, c: conversation.map(m => [m.role, m.preview, m.len, m.hidx]) };
        })()""")
        assert out["n"] == 3
        assert out["c"] == [["user", "hello there", 11, 0], ["assistant", "Hi! How are you", 15, 1],
                            ["user", "what is this", 12, 3]]

    def test_auto_form_switching_is_debounced_and_never_persists(self, graph_js):
        pre = """
const FORMS = ['vortex', 'lattice', 'embedding', 'descent'];
let formIndex = 0, autoForm = false, taskHint = null, baseFormName = 'vortex', _lastAutoSwitchAt = -1e9, _autoSwitching = false;
const stored = [];
globalThis.localStorage = { setItem: (k, v) => stored.push([k, v]), getItem: () => null };
const switched = [];
// The REAL setForm runs here (its persistence guard is the property under
// test); its anatomy side effects are stubbed.
const NODE_COUNT = 3, _blendFrom = [], currentPositions = [null, null, null];
let formBlend = 1.0, instancedMesh = null;
function _buildAnatomy() { switched.push(FORMS[formIndex]); }
"""
        fns = _fn(graph_js, "setForm", "autoFormFor", "setAutoForm", "getAutoForm", "setTaskHint", "_applyAutoForm")
        out = eval_js(pre + fns, """(() => {
            setAutoForm(true);                 // no hint yet: stays on the pick
            setTaskHint('coding');             // → lattice
            setTaskHint('research');           // within 20s: debounced
            const mid = FORMS[formIndex];
            setAutoForm(false);                // back to the pick
            return { switched, mid, final: FORMS[formIndex], stored, auto: getAutoForm() };
        })()""")
        assert out["switched"] == ["lattice", "vortex"], out
        assert out["mid"] == "lattice" and out["final"] == "vortex" and out["auto"] is False
        assert all(k != "ghost_face_form" for k, _ in out["stored"]), (
            "an AUTO switch persisted itself as the user's pick")
        assert ["ghost_face_auto", "1"] in out["stored"] and ["ghost_face_auto", "0"] in out["stored"]


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

    def test_data_forms_draw_explicit_edges_and_no_proximity_links(self, graph_nc):
        m = re.search(r"const LINK_MULT = \{(.*?)\};", graph_nc, re.DOTALL)
        assert re.search(r"conversation:\s*0\.0", m.group(1)) and re.search(r"toolgraph:\s*0\.0", m.group(1))
        assert "(LINK_MULT[FORM] === undefined ? 1.0 : LINK_MULT[FORM])" in graph_nc, (
            "`|| 1.0` turns a deliberate 0 multiplier back into 1 — proximity hairball on the data forms")
        i = graph_nc.index("for (let e = 0; e < explicitEdges.length && lineIdx < MAX_LINES; e++)")
        assert "connected[a] = true; connected[b] = true;" in graph_nc[i:i + 400], (
            "explicit-edge endpoints must count as connected or they scale to zero")

    def test_recall_comet_draws_a_streak_and_flares_the_node(self, graph_nc):
        assert "if (recallSpark > 0 && recallNode >= 0" in graph_nc
        assert "const reach = TUNE.recallReach * recallSpark * recallSpark;" in graph_nc
        assert "i === recallNode ? 1.0 + TUNE.recallFlare * 4.0 * recallSpark * (1.0 - recallSpark) : 1.0" in graph_nc

    def test_mood_and_error_kinds_reach_the_uniforms(self, graph_nc):
        assert "+ moodHue * (FORM === 'vortex' ? 0.4 : 1.0);" in graph_nc
        i = graph_nc.index("let formDim = (FORM === 'conversation' || FORM === 'toolgraph') ? 0.95 : 0.55;")
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
        for k in ("phase", "toolPulse", "recallSpark", "verdict", "backgroundBusy", "moodHue", "autoForm"):
            assert k in body, k

    def test_relayout_of_a_data_form_blends_from_current_positions(self, graph_nc):
        i = graph_nc.index("function _relayoutDataForm()")
        body = graph_nc[i:i + 700]
        assert "_blendFrom.push(currentPositions[k]" in body and "formBlend = 0.0;" in body
        assert "_relayoutDataForm();" in graph_nc[graph_nc.index("function animate()"):]


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
        assert out[0] == {"phase": "search", "hint": "research", "tool": None, "recall": False, "verdict": None}
        assert out[1]["phase"] == "tool" and out[1]["tool"] == "web_search"
        assert out[2]["phase"] == "verify" and out[2]["verdict"] == "pass" and out[2]["hint"] == "verify"
        assert out[3]["verdict"] == "refute"
        assert out[4]["recall"] is True and out[4]["phase"] == "search"
        assert out[5] == {"phase": "tool", "hint": "coding", "tool": None, "recall": False, "verdict": None}
        assert out[6] == {"phase": None, "hint": None, "tool": None, "recall": False, "verdict": None}

    def test_feed_calls_the_face_and_rate_limits_recalls(self, app_js):
        i = app_js.index("const _FACE_PHASE_BY_TITLE = {")
        table = app_js[i:app_js.index("};", i) + 2]
        fns = table + _fn(app_js, "faceSignalsForTicker", "_feedFaceFromTicker")
        out = eval_js("""
const calls = [];
const activeFace = { setPhase: (p) => calls.push(['phase', p]), setTaskHint: (h) => calls.push(['hint', h]),
    noteToolCall: (t) => calls.push(['tool', t]), noteRecall: () => calls.push(['recall']) };
let _turnVerdict = null, _lastRecallAt = 0;
""" + fns, """(() => {
            _feedFaceFromTicker('memory search', '🔎', 'a');
            _feedFaceFromTicker('memory search', '📍', 'b');     // within 1.5s → dropped
            _feedFaceFromTicker('tool call', '🧰', 'file_read · x');
            _feedFaceFromTicker('verify', '🧪', 'REFUTED conf=0.2');
            return { calls, verdict: _turnVerdict };
        })()""")
        assert out["calls"].count(["recall"]) == 1, out["calls"]
        assert ["tool", "file_read"] in out["calls"] and ["phase", "tool"] in out["calls"]
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

    def test_history_feeds_the_conversation_form(self, app_nc):
        i = app_nc.index("function saveChatState()")
        assert "activeFace.setConversation(chatHistory)" in app_nc[i:i + 400]

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

    def test_hover_is_suppressed_over_chrome(self, app_js):
        i = app_js.index("const _FACE_CHROME = ")
        consts = app_js[i:app_js.index("\n", i) + 1]
        fn = consts + _fn(app_js, "faceHoverAllowed")
        out = eval_js(fn, """[
            faceHoverAllowed({ closest: (sel) => sel.includes('.message') ? {} : null }),
            faceHoverAllowed({ closest: () => null }),
            faceHoverAllowed(null),
        ]""")
        assert out == [False, True, True]

    def test_tooltip_element_and_listeners(self, app_nc):
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        assert 'id="face-tooltip"' in html
        assert "window.addEventListener('pointermove', _onFacePointer" in app_nc
        assert "activeFace.describeNodeAt(nx, ny)" in app_nc

    def test_menu_offers_auto_and_the_new_hints(self, app_nc):
        i = app_nc.index("function buildFaceFormMenu()")
        body = app_nc[i:i + 2500]
        assert "auto.dataset.form = 'auto';" in body and "activeFace.setAutoForm(!activeFace.getAutoForm())" in body
        i = app_nc.index("function markActiveFaceForm()")
        assert "'auto-on'" in app_nc[i:i + 700]


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
    assert "app.js?v=12.1" in index and "style.css?v=6.3" in index
    assert "matrix_graph.js?v=12.1" in app and "workspace.js?v=8.5" in app
    assert "status.js?v=7.4" in ws
