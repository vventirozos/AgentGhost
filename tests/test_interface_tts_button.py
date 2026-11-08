"""§4TL: the speaker button beside the mic (2026-11-07, operator: "figure out a pretty way to add a
button to the webui"). The switch logic is EXECUTED under node, lifted from app.js; the layout is
rendered in WebKit, because the phone header had no room and only a measurement shows that."""
import pytest

from tests.helpers import eval_js
from tests.test_interface_composer_mobile import _box, _open, browser  # noqa: F401  (harness)
from tests.test_interface_header_simplify import _js, _raw


def _env(stored=None, broken_storage=False):
    app = _js("app.js")
    start = app.index("const ttsToggleBtn = document.getElementById('tts-toggle-btn');")
    engine = app[start:app.index("function stopTTS() {")]
    typed_at = app.index("function typedTurnSpeech(")
    typed = app[typed_at:app.index("if (interruptBtn) interruptBtn", typed_at)]
    storage = ("{ getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } }"
               if broken_storage else
               "{ v: %s, getItem() { return this.v; }, setItem(k, x) { this.v = x; saved.push(x); } }"
               % ("null" if stored is None else repr(stored)))
    return """
const saved = []; let stops = 0, sends = 0, unlocks = 0, handler = null;
const classes = new Set(), attrs = {};
const btn = { title: '', classList: { toggle(c, on) { on ? classes.add(c) : classes.delete(c); } },
              setAttribute(k, v) { attrs[k] = v; }, addEventListener(k, f) { handler = f; } };
const document = { getElementById: () => btn };
const localStorage = """ + storage + """;
let isTTSActive = false, speakReplies = false, audioCtx = null, isProcessingRequest = false;
let _voiceSession = false;
const _bargeCancel = () => {};
const chatInput = { value: 'a message' };
let _speechOffSeq = 0;
function FakeCtx() { unlocks++; this.state = 'running'; this.destination = {};
    this.createBuffer = () => ({}); this.createBufferSource = () => ({ connect() {}, start() {} }); }
const window = { AudioContext: FakeCtx };
function stopTTS() { stops++; }
function sendMessage() { sends++; }
const state = () => ({ on: isTTSActive, setting: speakReplies, lit: classes.has('active'),
                       pressed: attrs['aria-pressed'], label: attrs['aria-label'], title: btn.title,
                       saved, stops, sends, unlocks });
const micSent = () => { isTTSActive = true; showSpeechState(); };
""" + engine + typed


def test_off_by_default_and_a_press_turns_it_on_and_saves():
    out = eval_js(_env(), "(() => { const a = state(); handler(); return [a, state()]; })()")
    assert out[0]["on"] is False and out[0]["lit"] is False and out[0]["pressed"] == "false"
    assert out[0]["label"] == "Spoken replies off"
    after = out[1]
    assert after["on"] is True and after["setting"] is True and after["lit"] is True
    assert after["pressed"] == "true" and after["title"] == "Spoken replies on"
    assert after["saved"] == ["1"] and after["unlocks"] == 1 and after["stops"] == 0


def test_a_second_press_stops_speech_at_once_and_saves():
    out = eval_js(_env(), "(() => { handler(); handler(); return state(); })()")
    assert out["on"] is False and out["lit"] is False and out["stops"] == 1 and out["saved"] == ["1", "0"]


def test_the_setting_survives_a_reload_without_speaking_until_a_turn_is_sent():
    """A page load adopts a running turn and replays it from the start: speaking there is a replay."""
    out = eval_js(_env(stored="1"), "state()")
    assert out["on"] is False and out["setting"] is True and out["lit"] is True


def test_a_press_on_during_a_reply_waits_for_the_next_reply():
    """Part-way through, it would open mid-sentence, or inside reasoning with no opening tag."""
    out = eval_js(_env(), """(() => { isProcessingRequest = true; handler(); const during = state();
        isProcessingRequest = false; sendTypedMessage(); return [during, state()]; })()""")
    assert out[0]["on"] is False and out[0]["lit"] is True and out[0]["saved"] == ["1"]
    assert out[1]["on"] is True


def test_after_a_reload_the_first_typed_send_unlocks_audio():
    """A browser plays nothing until a gesture, and a reload starts with none."""
    out = eval_js(_env(stored="1"), "(() => { sendTypedMessage(); return state(); })()")
    assert out["on"] is True and out["unlocks"] == 1 and out["sends"] == 1


def test_typing_keeps_the_button_on():
    """Typing used to set speech off, which would flip a visible switch by itself."""
    out = eval_js(_env(), "(() => { handler(); sendTypedMessage(); return state(); })()")
    assert out["on"] is True and out["lit"] is True and out["saved"] == ["1"]


def test_a_mic_turn_lights_the_button_and_typing_ends_it_and_its_speech():
    out = eval_js(_env(), """(() => { micSent(); const during = state(); sendTypedMessage();
        return [during, state()]; })()""")
    assert out[0]["on"] is True and out[0]["lit"] is True and out[0]["setting"] is False
    assert out[0]["saved"] == [], "a mic turn is not a change of setting"
    assert out[1]["on"] is False and out[1]["lit"] is False and out[1]["unlocks"] == 0
    assert out[1]["stops"] == 1, "a dim button with speech still playing"


def test_a_press_during_a_mic_turn_silences_it_rather_than_saving_on():
    out = eval_js(_env(), "(() => { micSent(); handler(); return state(); })()")
    assert out["on"] is False and out["setting"] is False and out["stops"] == 1 and out["saved"] == ["0"]


def test_blocked_storage_leaves_a_working_button():
    out = eval_js(_env(broken_storage=True), "(() => { handler(); return state(); })()")
    assert out["on"] is True and out["lit"] is True


def test_audio_refused_still_sends_the_message():
    env = _env(stored="1").replace("const window = { AudioContext: FakeCtx };", "const window = {};")
    out = eval_js(env, "(() => { sendTypedMessage(); return state(); })()")
    assert out["sends"] == 1 and out["on"] is True


def test_only_a_sent_transcript_switches_speech_on():
    """A denied mic, a quick tap or an empty transcript left the button lit with no spoken turn."""
    app = _js("app.js")
    assert app.count("isTTSActive = true") == 2     # the button, and this
    sent = app[app.index("function _voiceWords("):]
    assert sent.index("else spokenTurnSpeech();") < sent.index("_sendVoice(text)")
    assert "_voiceWords(data.text, spokenIn, voiceToken)" in app[app.index("if (data.text) {"):]
    spoken = app[app.index("function spokenTurnSpeech() {"):]
    assert spoken.index("isTTSActive = true") < spoken.index("}")


def test_every_typed_send_applies_the_setting():
    """A held message and an upload went round sendTypedMessage and kept a mic turn's state."""
    app = _js("app.js")
    held = app.index("sendMessage(false, it.text);")
    assert "typedTurnSpeech(false);" in app[held - 40:held]
    upload = app[app.index("async function confirmUploadAsk"):]
    assert upload.index("typedTurnSpeech();") < upload.index("await uploadToSandbox")


def test_a_new_turn_clears_the_resume_mute_and_a_resume_keeps_it():
    """The reply after a resumed turn stayed silent for as many characters as the old one had."""
    app = _js("app.js")
    fresh = app[app.index("currentTurnStartedAt = Date.now();"):]
    assert fresh.index("_replayUpTo = 0;") < fresh.index("currentReqId = null;")
    assert app.count("_noteStreamReplay();") == 2


def _fetch_env():
    app = _js("app.js")
    start = app.index("async function processTTSFetch() {")
    fn = app[start:app.index("let _ttsAnalyser = null;", start)]
    return """
let isFetchingTTS = false, _ttsGeneration = 0, ttsTextQueue = [], ttsAudioQueue = [], isPlayingTTS = false;
const pending = [], started = [];
globalThis.fetch = (u, o) => new Promise(r => { started.push(JSON.parse(o.body).text); pending.push(r); });
const URL = { createObjectURL: () => 'blob' };
function playNextAudio() {} function addMessage() {} async function _httpError() { return {message: ''}; }
function stopTTS() { _ttsGeneration += 1; ttsTextQueue = []; isFetchingTTS = false; }
const ok = { ok: true, blob: async () => ({}) };
""" + fn


def test_a_stale_fetch_does_not_release_the_newer_ones_flag():
    """A quick off then on ran two fetches side by side, so clips could land out of order."""
    out = eval_js(_fetch_env(), """await (async () => {
        ttsTextQueue.push('old'); processTTSFetch();
        stopTTS();
        ttsTextQueue.push('new', 'next'); processTTSFetch();
        pending[0](ok);
        await new Promise(r => setTimeout(r, 0));
        processTTSFetch();
        return { flag: isFetchingTTS, started: [...started] };
    })()""")
    assert out == {"flag": True, "started": ["old", "new"]}


def test_the_button_is_a_labelled_toggle_beside_the_mic():
    html = _raw("index.html")
    assert html.index('id="mic-btn"') < html.index('id="tts-toggle-btn"') < html.index('id="send-btn"')
    tag = html[html.index('<button id="tts-toggle-btn"'):]
    tag = tag[:tag.index(">")]
    assert 'aria-pressed="false"' in tag and 'aria-label="Spoken replies off"' in tag and 'type="button"' in tag


@pytest.mark.parametrize("width,height", [(320, 568), (375, 667), (390, 844), (430, 932)])
def test_on_a_phone_it_sits_on_the_mic_row_at_the_same_size(browser, width, height):  # noqa: F811
    ctx, page = _open(browser, width, height)
    try:
        mic, tts, eff, send = (_box(page, i) for i in ("mic-btn", "tts-toggle-btn", "effort-wrap", "send-btn"))
        assert abs(tts["top"] - mic["top"]) < 1 and abs(tts["w"] - mic["w"]) < 1 and abs(tts["h"] - mic["h"]) < 1
        assert tts["w"] == 40
        assert mic["right"] <= tts["left"] and tts["right"] <= eff["left"] and eff["right"] <= send["left"]
        assert send["right"] <= width
        assert page.evaluate("document.documentElement.scrollWidth") <= width
    finally:
        ctx.close()


@pytest.mark.parametrize("width,height,size", [(1280, 800, 52), (1280, 720, 36)])
def test_on_a_desktop_it_matches_the_mic_and_the_text_field_keeps_its_row(browser, width, height, size):  # noqa: F811
    ctx, page = _open(browser, width, height)
    try:
        field, mic, tts = (_box(page, i) for i in ("chat-input", "mic-btn", "tts-toggle-btn"))
        assert tts["w"] == size and tts["h"] == size and abs(tts["top"] - mic["top"]) < 1
        assert tts["left"] > mic["right"] and field["right"] <= mic["left"] and field["w"] > 300
    finally:
        ctx.close()


def test_waves_and_slash_follow_the_state(browser):  # noqa: F811
    ctx, page = _open(browser, 390, 844)
    try:
        read = """() => { const b = document.getElementById('tts-toggle-btn');
            const d = s => getComputedStyle(b.querySelector(s)).display;
            return { waves: d('.tts-waves'), slash: d('.tts-slash') }; }"""
        assert page.evaluate(read) == {"waves": "none", "slash": "inline"}
        page.evaluate("document.getElementById('tts-toggle-btn').classList.add('active')")
        assert page.evaluate(read) == {"waves": "inline", "slash": "none"}
    finally:
        ctx.close()


def _clip_env():
    app = _js("app.js")
    start = app.index("function _bufferClip(audioUrl, onDone) {")
    fn = app[start:app.index("const BARGE_PREF_KEY", start)]
    return """
const made = [], done = [], notes = [];
let _ttsRefusalShown = false;
const audioCtx = { state: 'running', currentTime: 0, destination: {}, resume: async () => {},
    decodeAudioData: async (b) => ({ decoded: b, duration: 1 }),
    createBufferSource() { const s = { stops: 0, starts: [], connect() {}, disconnect() {},
        start(when, at) { this.starts.push(at); }, stop() { this.stops++; if (this.onended) this.onended(); } };
        made.push(s); return s; } };
globalThis.fetch = async () => ({ arrayBuffer: async () => 'bytes' });
function _ensureTTSAnalyser() { return null; }
function _showAudioRefused() { notes.push('refused'); }
const settle = () => new Promise(r => setTimeout(r, 0));
""" + fn


def test_a_clip_through_the_context_plays_then_reports_its_end():
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        const before = done.length; made[0].onended();
        return { made: made.length, at: made[0].starts, before, after: done.length };
    })()""")
    assert out == {"made": 1, "at": [0], "before": 0, "after": 1}


def test_a_paused_clip_resumes_where_it_was_and_pause_is_not_an_end():
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        audioCtx.currentTime = 0.25; c.pause(); audioCtx.currentTime = 9; await c.play(); await settle();
        const r = { at: made[1].starts, ends: done.length }; c.stop(); return r;
    })()""")
    assert out == {"at": [0.25], "ends": 0}


def test_a_clip_paused_at_its_very_end_finishes_instead_of_hanging():
    """Started at or past its end, a source never fires ended (WebKit, measured), which held the queue."""
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        audioCtx.currentTime = 1; c.pause(); await c.play(); await settle();
        return { made: made.length, ends: done.length };
    })()""")
    assert out == {"made": 1, "ends": 1}


def test_a_context_stopped_mid_clip_frees_the_queue():
    """A call or Siri suspends the context, and then no ended ever comes."""
    env = _clip_env().replace("decoded: b, duration: 1", "decoded: b, duration: 0.01")
    out = eval_js(env, """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        audioCtx.state = 'interrupted';
        await new Promise(r => setTimeout(r, 2700));
        return { ends: done.length, notes };
    })()""")
    assert out == {"ends": 1, "notes": ["refused"]}


def test_a_lost_end_with_the_context_running_still_finishes():
    env = _clip_env().replace("decoded: b, duration: 1", "decoded: b, duration: 0.01")
    out = eval_js(env, """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        audioCtx.currentTime = 5;
        await new Promise(r => setTimeout(r, 1700));
        return { ends: done.length, notes };
    })()""")
    assert out == {"ends": 1, "notes": []}


def test_a_stopped_clip_is_silent_for_good_and_starts_nothing_more():
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); await settle();
        c.stop(); await c.play();
        return { made: made.length, stops: made[0].stops, ends: done.length };
    })()""")
    assert out == {"made": 1, "stops": 1, "ends": 0}


def test_stop_before_the_clip_decoded_never_plays_it():
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); c.stop(); await settle();
        return { made: made.length, ends: done.length };
    })()""")
    assert out == {"made": 0, "ends": 0}


def test_a_pause_while_decoding_waits_for_the_resume():
    out = eval_js(_clip_env(), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play(); c.pause(); await settle();
        const during = made.length; c.play(); await settle();
        const r = { during, after: made.length }; c.stop(); return r;
    })()""")
    assert out == {"during": 0, "after": 1}


def test_a_context_that_will_not_run_says_so_and_frees_the_queue():
    """A suspended context never reaches onended, which would hold every later clip."""
    out = eval_js(_clip_env().replace("state: 'running'", "state: 'suspended'"), """await (async () => {
        const c = _bufferClip('u', () => done.push(1)); c.play();
        await new Promise(r => setTimeout(r, 1100));
        return { made: made.length, ends: done.length, notes };
    })()""")
    assert out == {"made": 0, "ends": 1, "notes": ["refused"]}


def _play_env():
    app = _js("app.js")
    start = app.index("function playNextAudio() {")
    fn = app[start:app.index("function _bufferClip(audioUrl", start)]
    return (_clip_env().replace("function _showAudioRefused() { notes.push('refused'); }", "")
            .replace("let _ttsRefusalShown = false;", "") + """
let _bargeHold = false, isRecording = false, _micPressHeld = false, isPlayingTTS = false, ttsAudioQueue = [];
let currentAudioSource = null, _ttsPlayViaContext = false, _ttsRefusalShown = false, _ttsGeneration = 0;
const elements = [], revoked = [];
class Audio { constructor(u) { this.u = u; elements.push(this); }
    play() { return Promise.reject(Object.assign(new Error('no gesture'), { name: 'NotAllowedError' })); } }
const URL = { revokeObjectURL: (u) => revoked.push(u) };
function addMessage(r, t) { notes.push(t); }
function _bargeListen() {} function _bargeArmGrace() {} function _startTTSAudioPump() {} function _stopTTSAudioPump() {}
""" + fn)


def test_a_refused_element_falls_back_to_the_context_and_stays_there():
    out = eval_js(_play_env(), """await (async () => {
        ttsAudioQueue.push('a', 'b'); playNextAudio(); await settle(); await settle();
        const first = made.length; const refused = elements[0];
        if (refused.onerror) refused.onerror();       // a late element error must not end the clip again
        made[0].onended(); await settle(); await settle();
        const r = { first, elements: elements.length, sources: made.length, sticky: _ttsPlayViaContext,
                    revoked: [...revoked], playing: isPlayingTTS, notes };
        currentAudioSource.stop(); return r;
    })()""")
    assert out == {"first": 1, "elements": 1, "sources": 2, "sticky": True, "revoked": ["a"],
                   "playing": True, "notes": []}


def test_a_stop_before_the_refusal_arrives_plays_nothing():
    out = eval_js(_play_env(), """await (async () => {
        ttsAudioQueue.push('a'); playNextAudio();
        _ttsGeneration += 1; isPlayingTTS = false; currentAudioSource = null;
        await settle(); await settle();
        return { sources: made.length, revoked: [...revoked], playing: isPlayingTTS };
    })()""")
    assert out == {"sources": 0, "revoked": ["a"], "playing": False}
