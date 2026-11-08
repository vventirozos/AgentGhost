"""Talking over a reply on the REAL page: app.js + bargein.js in Chromium, a fake
microphone playing a file, and a local server that streams slowly. Only a browser shows
whether the audio pauses and resumes, and only a slow stream shows what a reply that
is still arriving does after the interruption."""
import io
import json
import random
import struct
import threading
import time
import wave
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
RATE = 48000
SENTENCES = ["First sentence of the reply. ", "Second sentence of the reply. ",
             "Third sentence of the reply. ", "Fourth sentence of the reply. "]
ANSWER = "Metric it is from here on. "
NOTE_HEARS_SPEAKERS = "hearing the speakers"


def _wav(path, parts):
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        rng = random.Random(7)
        w.writeframes(b"".join(
            struct.pack("<h", int(amp * 32767 * rng.uniform(-1, 1)))
            for secs, amp in parts for _ in range(int(secs * RATE))))


class Backend:
    """What the page talks to. Records every call."""

    def __init__(self, stt, *, chunk_gap=0.0, inbox_path="inject", stt_delay=0.0, stt_status=200,
                 inbox_delay=0.0, leftovers=()):
        self.stt_texts, self.chunk_gap, self.inbox_path = list(stt), chunk_gap, inbox_path
        self.stt_delay, self.stt_status, self.inbox_delay = stt_delay, stt_status, inbox_delay
        self.leftovers = list(leftovers)
        self.chat, self.spoken, self.inbox, self.stt_clips = [], [], [], []
        self.retract_on_inject = False
        self.first_delay, self.reasoning = 0.0, ""
        self.hold_speech = False
        self.speech = self._speech()
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _send(self, body, kind="application/json", status=200):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                name = self.path.split("?")[0]
                if name == "/":
                    return self._send((STATIC / "index.html").read_bytes(), "text/html")
                file = STATIC / name.removeprefix("/static/")
                if name.startswith("/static/") and file.is_file():
                    kind = {".js": "text/javascript", ".css": "text/css"}.get(file.suffix, "application/octet-stream")
                    return self._send(file.read_bytes(), kind)
                self._send(b"{}", status=404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                name = self.path.split("?")[0]
                if name == "/api/tts":
                    backend.spoken.append(json.loads(body)["text"])
                    return self._send(backend.speech, "audio/wav")
                if name == "/api/stt":
                    message = BytesParser().parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                    backend.stt_clips.append(message.get_payload()[0].get_payload(decode=True))
                    first = len(backend.stt_clips) == 1
                    text = backend.stt_texts.pop(0) if backend.stt_texts else ""
                    if not first:
                        time.sleep(backend.stt_delay)
                        if backend.stt_status != 200:
                            return self._send(b'{"error":"down"}', status=backend.stt_status)
                    return self._send(json.dumps({"text": text}).encode())
                if name == "/api/turn/inbox/leftovers":
                    out, backend.leftovers = backend.leftovers, []
                    return self._send(json.dumps({"leftovers": out, "messages": out, "texts": out}).encode())
                if name == "/api/turn/inbox":
                    backend.inbox.append(json.loads(body)["text"])
                    time.sleep(backend.inbox_delay)
                    return self._send(json.dumps({"path": backend.inbox_path, "reply": "still going"}).encode())
                if name == "/api/chat":
                    backend.chat.append(json.loads(body)["messages"][-1]["content"])
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    try:
                        def frame(payload):
                            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                            self.wfile.flush()
                        if backend.hold_speech:
                            frame({"id": "chatcmpl-x", "ghost": {"reasoning_unparsed": True}})
                        if backend.reasoning:
                            frame({"id": "chatcmpl-x", "choices": [{"delta": {"content": "<think>" + backend.reasoning}}]})
                        time.sleep(backend.first_delay)
                        if backend.reasoning:
                            frame({"id": "chatcmpl-x", "choices": [{"delta": {"content": "more of it. Yet more. </think>"}}]})
                        for text in SENTENCES:
                            if backend.retract_on_inject and backend.inbox:
                                # a tool step read the inbox: the text so far goes
                                frame({"id": "chatcmpl-x", "ghost": {"retract": True}})
                                frame({"id": "chatcmpl-x", "choices": [{"delta": {"content": ANSWER}}]})
                                break
                            frame({"id": "chatcmpl-x", "choices": [{"delta": {"content": text}}]})
                            time.sleep(backend.chunk_gap)
                        self.wfile.write(b"data: [DONE]\n\n")
                    except OSError:
                        pass
                    return None
                self._send(b"{}", status=404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://localhost:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def _speech():
        out = io.BytesIO()
        _wav(out, [(12, 0.01)])
        return out.getvalue()


# Every Audio the page makes, so a test can read the real element's state.
_SPY = """
const RealAudio = window.Audio;
window.__clips = [];
window.__ghostSessionId = 'session-under-test';
window.__pauses = 0;
window.__heard = 0;
window.__listened = 0;
window.Audio = function (src) {
    const a = new RealAudio(src), pause = a.pause.bind(a);
    a.pause = () => { window.__pauses++; pause(); };
    window.__clips.push(a);
    return a;
};
"""
# Recorded by the page: under load a whole utterance can fit between two polls.
_COUNT_HEARD = """() => { const mic = document.getElementById('mic-btn');
    let was = false;
    new MutationObserver(() => {
        const now = mic.classList.contains('recording');
        if (now && !was && mic.classList.contains('barge-listening')) window.__heard++;
        if (mic.classList.contains('barge-listening')) window.__listened++;
        was = now;
    }).observe(mic, { attributes: true, attributeFilter: ['class'] }); }"""
_CLIP = """() => { const a = window.__clips[window.__clips.length - 1];
    return a ? {paused: a.paused, t: a.currentTime, count: window.__clips.length} : null; }"""
_MIC = """() => { const m = document.getElementById('mic-btn');
    return {listening: m.classList.contains('barge-listening'), recording: m.classList.contains('recording')}; }"""


@pytest.fixture
def run(tmp_path):
    """run(backend, mic=[(seconds, amplitude), ...], storage={...}) opens the page."""
    opened = []

    def _run(backend, *, mic, storage=None, script=None):
        file = tmp_path / f"mic{len(opened)}.wav"
        _wav(file, mic)
        pw = playwright_api.sync_playwright().start()
        try:
            browser = pw.chromium.launch(args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={file}", "--autoplay-policy=no-user-gesture-required",
                # The machine's audio output can stall on a busy box: no clock, no frames.
                "--disable-audio-output"])
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pw.stop()
            pytest.skip(f"chromium unavailable: {exc}")
        opened.append((pw, browser, backend))
        page = browser.new_context(permissions=["microphone"]).new_page()
        page.errors = []
        page.on("pageerror", lambda e: page.errors.append(str(e)))
        page.add_init_script(_SPY)
        if script:
            page.add_init_script(script)
        if storage:
            page.add_init_script("".join(f"if (window === top) localStorage.setItem({json.dumps(k)},{json.dumps(v)});"
                                         for k, v in storage.items()))
        page.goto(backend.url)
        page.wait_for_selector("#mic-btn")
        page.evaluate(_COUNT_HEARD)
        page.wait_for_timeout(400)
        return page

    yield _run
    for pw, browser, backend in opened:
        browser.close()
        pw.stop()
        backend.server.shutdown()


def _hold_mic(page, ms=700, recorder=True):
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    if recorder:        # opening the microphone can take longer than the hold, on a busy box
        page.wait_for_selector("#mic-btn.recording", timeout=30000)
    page.wait_for_timeout(ms)
    page.mouse.up()


def _wait(page, condition, seconds=40.0, what="condition"):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return
        page.wait_for_timeout(50)
    state = page.evaluate("""() => ({mic: document.getElementById('mic-btn').className,
        clips: window.__clips.map(a => [a.paused, a.currentTime, a.ended]), pauses: window.__pauses,
        heard: window.__heard})""")
    raise AssertionError(f"timed out waiting for {what}: {state}, page errors {page.errors}")


# The fake file starts when the microphone opens and loops.
def _heard(page):
    return page.evaluate("() => window.__heard") > 0


TALK_AFTER_2S = [(2, 0.0005), (1.2, 0.3), (9, 0.0005)]
SILENT = [(5, 0.0005)]


def test_words_pause_the_reply_cut_it_and_are_sent(run):
    backend = Backend(["first question", "what about tomorrow"])
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    _wait(page, lambda: _heard(page), what="speech to be heard")
    _wait(page, lambda: page.evaluate(_CLIP)["paused"], what="the reply to pause")
    clips_before = page.evaluate(_CLIP)["count"]
    t_paused = page.evaluate(_CLIP)["t"]
    _wait(page, lambda: len(backend.chat) == 2, what="the words to be sent")
    assert backend.chat == ["first question", "what about tomorrow"]
    # the clip holds the talking, and the lead-in that keeps the first word
    with wave.open(io.BytesIO(backend.stt_clips[1])) as clip:
        assert 2.0 < clip.getnframes() / clip.getframerate() < 3.2
        rate = clip.getframerate()
        pcm = struct.unpack(f"<{clip.getnframes()}h", clip.readframes(clip.getnframes()))
    def level(start, end):
        part = pcm[int(start * rate):int(end * rate)]
        return (sum(x * x for x in part) / len(part)) ** 0.5
    # quiet lead-in, the talking, quiet tail: a clip of the same size in the wrong order fails
    assert level(0, 0.1) < level(0.4, 0.8) / 3 > level(len(pcm) / rate - 0.4, len(pcm) / rate)
    # the interrupted clip never carried on
    assert page.evaluate(f"() => window.__clips[{clips_before - 1}].paused") is True
    assert page.evaluate(f"() => window.__clips[{clips_before - 1}].currentTime") <= t_paused + 0.3
    assert page.errors == []


def test_an_empty_transcript_lets_the_reply_carry_on(run):
    backend = Backend(["first question", ""])
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    assert page.evaluate("() => window.__pauses") >= 1
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription")
    _wait(page, lambda: page.evaluate(_CLIP)["paused"] is False, what="the reply to resume")
    resumed_at = page.evaluate(_CLIP)["t"]
    _wait(page, lambda: page.evaluate(_CLIP)["t"] > resumed_at + 0.2, what="the reply to carry on playing")
    assert backend.chat == ["first question"]
    assert page.errors == []


def test_stop_with_nothing_running_ends_the_speech_and_is_not_a_message(run):
    backend = Backend(["first question", "Stop."])
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription")
    page.wait_for_timeout(600)
    assert backend.chat == ["first question"]
    assert page.evaluate(_CLIP)["paused"] is True
    assert page.evaluate(_MIC) == {"listening": False, "recording": False}


def test_a_press_stops_a_reply_that_is_still_streaming_from_speaking(run):
    """The bug this work found: the press cleared the queue, and the next sentence off
    the stream started talking into the microphone that had just opened."""
    backend = Backend(["first question"], chunk_gap=1.0)
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the first sentence")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    spoken_at_press = len(backend.spoken)
    page.wait_for_timeout(3500)       # the rest of the reply arrives during this
    assert len(backend.spoken) == spoken_at_press
    page.mouse.up()
    assert page.errors == []


def test_words_during_a_running_turn_go_to_that_turn(run):
    """They used to sit in the composer with a note to send them later."""
    backend = Backend(["first question", "use metric units"], chunk_gap=2.0, inbox_path="inject")
    page = run(backend, mic=TALK_AFTER_2S)
    page.fill("#chat-input", "half a typed draft")
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    assert backend.chat == ["first question"]
    assert page.input_value("#chat-input") == "half a typed draft"


def test_the_reply_that_was_cut_stays_cut_and_the_answer_is_spoken(run):
    backend = Backend(["first question", "use metric units"], chunk_gap=2.5, inbox_path="inject")
    backend.retract_on_inject = True
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    _wait(page, lambda: ANSWER.strip() in backend.spoken, what="the answer to be spoken")
    cut_at = backend.spoken.index(ANSWER.strip())
    assert all(s in [t.strip() for t in SENTENCES[:cut_at]] for s in backend.spoken[:cut_at])
    assert backend.spoken[cut_at:] == [ANSWER.strip()]


def test_a_cut_reply_with_no_tool_step_is_not_heard_again(run):
    """Accepted into the inbox while the final text streams: nothing more of it is spoken."""
    backend = Backend(["first question", "use metric units"], chunk_gap=2.5, inbox_path="inject")
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    spoken = len(backend.spoken)
    page.wait_for_timeout(4000)
    assert len(backend.spoken) == spoken


def test_a_message_the_turn_never_read_is_re_sent_with_a_spoken_reply(run):
    backend = Backend(["first question", "use metric units"], chunk_gap=1.5, inbox_path="inject",
                      leftovers=["use metric units"])
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.chat == ["first question", "use metric units"], seconds=20, what="the re-send")
    spoken = len(backend.spoken)
    _wait(page, lambda: len(backend.spoken) > spoken, what="the new reply to be spoken")


def test_a_press_after_the_post_is_not_undone_by_a_late_inbox_answer(run):
    backend = Backend(["first question", "use metric units"], chunk_gap=2.5, inbox_path="inject",
                      inbox_delay=2.0)
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.wait_for_timeout(100)
    spoken = len(backend.spoken)
    page.wait_for_timeout(3000)
    assert len(backend.spoken) == spoken
    page.mouse.up()


@pytest.mark.parametrize("move_on", [
    "document.getElementById('tts-toggle-btn').click()",
    "(() => { const i = document.getElementById('chat-input'); i.value = 'typed instead';"
    " i.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true})); })()",
    "window.GhostCore.clearConversation()",
])
def test_moving_on_during_transcription_keeps_the_words(run, move_on):
    backend = Backend(["first question", "my precious words"], stt_delay=2.0)
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription to start")
    page.evaluate(move_on)
    _wait(page, lambda: "my precious words" in page.evaluate(
        "() => { const k = Object.keys(localStorage).map(n => localStorage.getItem(n)).join(' ');"
        " const s = Object.keys(sessionStorage).map(n => sessionStorage.getItem(n)).join(' ');"
        " return k + s + document.body.innerText; }"), what="the words to be kept")
    assert "my precious words" not in backend.chat


def test_words_spoken_in_one_session_do_not_go_to_another(run):
    backend = Backend(["first question", "for session a"])
    page = run(backend, mic=[(2, 0.0005), (6, 0.3), (6, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    page.evaluate("window.__ghostSessionId = 'session-b'")
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription")
    # kept for the session they were said in, whose tray shows them
    _wait(page, lambda: page.evaluate(
        "() => Object.values(localStorage).some(v => v.includes('for session a') && v.includes('session-under-test'))"),
        what="the words to be kept")
    assert backend.chat == ["first question"]


@pytest.mark.parametrize("stop", ["window.GhostCore.cancelOwnTurn(false)", "window.GhostCore.endVoiceSession()"])
def test_after_a_stop_the_room_cannot_start_a_turn(run, stop):
    backend = Backend(["first question", "said to someone else"], chunk_gap=0.0)
    page = run(backend, mic=[(2.5, 0.0005), (1.2, 0.3), (9, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    page.evaluate(stop)
    _wait(page, lambda: page.evaluate(_MIC) == {"listening": False, "recording": False}, what="the mic to close")
    assert page.evaluate(_CLIP)["paused"] is True
    page.wait_for_timeout(4500)
    assert backend.chat == ["first question"]


def test_a_control_appearing_beside_the_mic_does_not_end_the_hold(run):
    backend = Backend(["first question"], chunk_gap=1.0)
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the reply to be spoken")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _wait(page, lambda: page.evaluate(_MIC)["recording"], what="the recording to start")
    page.mouse.move(5, 5)              # the pointer is off the button, as after a layout shift
    page.wait_for_timeout(600)
    assert page.evaluate(_MIC)["recording"] is True
    assert len(backend.stt_clips) == 1
    page.mouse.up()
    _wait(page, lambda: not page.evaluate(_MIC)["recording"], what="the release to end it")
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the clip to be uploaded")


def test_enter_on_an_empty_composer_leaves_speech_and_the_mic_alone(run):
    backend = Backend(["first question"])
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    page.wait_for_timeout(300)
    page.press("#chat-input", "Enter")
    page.wait_for_timeout(300)
    assert page.evaluate(_MIC)["listening"] is True
    assert page.evaluate(_CLIP)["paused"] is False


def test_the_grace_period_is_counted_from_the_last_clip(run):
    backend = Backend(["first question"])
    short = io.BytesIO()
    _wav(short, [(1.0, 0.01)])
    backend.speech = short.getvalue()
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: len(backend.spoken) == len(SENTENCES), what="all the sentences")
    # four 1 s clips: all played by about 4.5 s, so the mic is open until about 7.5 s
    _wait(page, lambda: page.evaluate(
        f"() => window.__clips.length === {len(SENTENCES)} && window.__clips[{len(SENTENCES) - 1}].ended"),
        what="the last clip to end")
    last_ended = time.monotonic()
    _wait(page, lambda: not page.evaluate(_MIC)["listening"], seconds=8, what="the mic to close")
    assert time.monotonic() - last_ended > 2.5


def test_stt_failures_are_not_blamed_on_the_speakers(run):
    backend = Backend(["first question"], stt_status=502)
    page = run(backend, mic=[(1, 0.0005), (0.7, 0.3), (1.5, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) >= 5, seconds=30, what="several failed transcriptions")
    assert NOTE_HEARS_SPEAKERS not in page.inner_text("#chat-log").lower()
    _wait(page, lambda: page.evaluate(_CLIP)["paused"] is False, what="the reply to carry on")


def test_words_held_for_after_the_turn_get_a_spoken_reply(run):
    """Talking as a reply finishes takes this path: it counted as typed, and silent."""
    backend = Backend(["first question", "and tomorrow"], chunk_gap=1.5, inbox_path="queue")
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["and tomorrow"], what="the inbox post")
    spoken = len(backend.spoken)
    _wait(page, lambda: backend.chat == ["first question", "and tomorrow"], seconds=20, what="the held send")
    _wait(page, lambda: len(backend.spoken) > spoken, what="the new reply to be spoken")


def test_a_status_answer_leaves_the_interrupted_reply_silent(run):
    backend = Backend(["first question", "how is it going"], chunk_gap=2.0, inbox_path="status")
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: len(backend.inbox) == 1, what="the inbox post")
    spoken = len(backend.spoken)
    page.wait_for_timeout(4500)
    assert len(backend.spoken) == spoken


def test_a_typed_stop_ends_the_speech_and_the_listening(run):
    backend = Backend(["first question"], chunk_gap=2.0, inbox_path="cancel")
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    page.fill("#chat-input", "stop")
    page.press("#chat-input", "Enter")
    _wait(page, lambda: backend.inbox == ["stop"], what="the inbox post")
    _wait(page, lambda: not page.evaluate(_MIC)["listening"], seconds=3, what="the mic to close")
    assert page.evaluate(_CLIP)["paused"] is True


def test_a_tap_before_any_reply_text_does_not_silence_the_answer(run):
    backend = Backend(["first question", ""])
    backend.first_delay = 3.0
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: backend.chat == ["first question"], what="the first send")
    _hold_mic(page, 500)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the answer to be spoken")


def test_a_press_during_reasoning_does_not_read_it_out(run):
    backend = Backend(["first question", ""])
    backend.first_delay, backend.reasoning = 3.0, "my secret reasoning here. "
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: backend.chat == ["first question"], what="the first send")
    _hold_mic(page, 500)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the answer to be spoken")
    assert not any("secret" in s or "more of it" in s.lower() for s in backend.spoken)


def test_nothing_plays_while_the_button_is_held_and_it_plays_on_release(run):
    backend = Backend(["first question", ""])
    backend.first_delay = 1.0
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: backend.chat == ["first question"], what="the first send")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _wait(page, lambda: len(backend.spoken) >= 1, what="speech to arrive during the hold")
    page.wait_for_timeout(500)
    assert page.evaluate("() => window.__clips.length") == 0
    page.mouse.up()
    _wait(page, lambda: page.evaluate("() => window.__clips.length") >= 1, what="speech after the release")


def test_turning_the_speaker_off_covers_a_message_still_to_be_re_sent(run):
    backend = Backend(["first question", "use metric units"], chunk_gap=2.5, inbox_path="inject",
                      leftovers=["use metric units"])
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    page.wait_for_timeout(200)
    page.click("#tts-toggle-btn")
    spoken = len(backend.spoken)
    _wait(page, lambda: backend.chat == ["first question", "use metric units"], seconds=20, what="the re-send")
    page.wait_for_timeout(1500)
    assert len(backend.spoken) == spoken
    assert page.evaluate(_MIC)["listening"] is False


def test_a_stop_word_by_the_button_with_nothing_running_is_not_a_message(run):
    backend = Backend(["first question", "Stop."])
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the reply to be speaking")
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription")
    page.wait_for_timeout(600)
    assert backend.chat == ["first question"]


def test_button_words_after_a_new_chat_go_to_the_tray(run):
    backend = Backend(["first question", "my precious words"], stt_delay=2.0)
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the reply to be speaking")
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the upload")
    page.evaluate("window.GhostCore.clearConversation()")
    _wait(page, lambda: "my precious words" in page.evaluate(
        "() => Object.keys(localStorage).map(n => localStorage.getItem(n)).join(' ') + document.body.innerText"),
        what="the words to be kept")
    assert backend.chat == ["first question"]
    assert page.evaluate(_MIC)["listening"] is False


def test_a_press_during_held_reasoning_does_not_read_it_out(run):
    backend = Backend(["first question", ""])
    backend.first_delay, backend.reasoning, backend.hold_speech = 3.0, "my secret reasoning here. ", True
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: backend.chat == ["first question"], what="the first send")
    _hold_mic(page, 500)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the answer to be spoken")
    assert not any("secret" in s for s in backend.spoken)


def test_a_second_press_keeps_the_words_of_the_first(run):
    backend = Backend(["first question", "my precious words"], stt_delay=2.0)
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the reply to be speaking")
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the upload")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _wait(page, lambda: "my precious words" in backend.chat, what="the first words to be sent")
    page.mouse.up()


def test_a_press_mid_sentence_sends_what_was_said(run):
    backend = Backend(["first question", "half a thought"])
    page = run(backend, mic=[(2, 0.0005), (6, 0.3), (2, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _wait(page, lambda: "half a thought" in backend.chat, what="the words to be sent")
    page.mouse.up()


def test_turning_the_speaker_off_covers_a_post_still_in_flight(run):
    backend = Backend(["first question", "use metric units"], chunk_gap=2.5, inbox_path="queue", inbox_delay=1.5)
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["use metric units"], what="the inbox post")
    page.click("#tts-toggle-btn")
    spoken = len(backend.spoken)
    _wait(page, lambda: backend.chat == ["first question", "use metric units"], seconds=20, what="the held send")
    page.wait_for_timeout(1500)
    assert len(backend.spoken) == spoken


def test_a_typed_turn_never_opens_the_microphone(run):
    backend = Backend([])
    page = run(backend, mic=TALK_AFTER_2S, storage={"ghost-speak-replies": "1"})
    page.fill("#chat-input", "typed question")
    page.press("#chat-input", "Enter")
    _wait(page, lambda: len(backend.spoken) >= 1, what="the reply to be spoken")
    for _ in range(40):
        assert page.evaluate(_MIC)["listening"] is False
        page.wait_for_timeout(100)
    assert page.evaluate(_CLIP)["paused"] is False


def test_the_setting_turns_it_off(run):
    backend = Backend(["first question"])
    page = run(backend, mic=TALK_AFTER_2S, storage={"ghost-barge-in": "0"})
    _hold_mic(page)
    _wait(page, lambda: len(backend.spoken) >= 1, what="the reply to be spoken")
    for _ in range(40):
        assert page.evaluate(_MIC)["listening"] is False
        page.wait_for_timeout(100)


def test_the_microphone_closes_after_the_reply_ends(run):
    backend = Backend(["first question"])
    short = io.BytesIO()
    _wav(short, [(0.5, 0.01)])
    backend.speech = short.getvalue()
    page = run(backend, mic=SILENT)
    _hold_mic(page)
    # recorded by the page: with clips this short the mic can open and close between two polls
    _wait(page, lambda: page.evaluate("() => window.__listened") > 0, what="the mic to open")
    _wait(page, lambda: len(backend.spoken) == len(SENTENCES), what="all the sentences")
    _wait(page, lambda: not page.evaluate(_MIC)["listening"], what="the mic to close")


def test_repeated_false_starts_stop_listening_and_say_why(run):
    backend = Backend(["first question", "", "", "", "the next question"])
    page = run(backend, mic=[(1, 0.0005), (0.7, 0.3), (1.5, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: NOTE_HEARS_SPEAKERS in page.inner_text("#chat-log").lower(), seconds=30,
          what="the note about the speakers")
    page.wait_for_timeout(200)
    assert page.evaluate(_MIC) == {"listening": False, "recording": False}
    clips = len(backend.stt_clips)
    assert page.evaluate(_CLIP)["paused"] is False
    page.wait_for_timeout(3000)
    assert len(backend.stt_clips) == clips
    # only until the next reply
    page.evaluate("() => { window.__listened = 0; }")
    _hold_mic(page)
    _wait(page, lambda: page.evaluate("() => window.__listened") > 0, what="the mic to open for the next reply")


# ── a dropped stream ───────────────────────────────────────────────────
def _frames(parts):
    out = []
    for part in parts:
        body = {"ghost": {"retract": True}} if part == "RETRACT" else {"choices": [{"delta": {"content": part}}]}
        out.append(f"data: {json.dumps({'id': 'chatcmpl-x', **body})}\n\n")
    return out


def _dropping_stream(before, replay, resume_delay_ms=0):
    """/api/chat breaks with Safari's error after `before`; the resume replays `replay`."""
    return """(() => {
        const BEFORE = %s, REPLAY = %s, DELAY = %d, real = window.fetch;
        window.__resumes = 0;
        const body = (list, die, gap) => new ReadableStream({ async start(c) {
            const e = new TextEncoder();
            for (const f of list) { c.enqueue(e.encode(f)); await new Promise(r => setTimeout(r, gap)); }
            if (die) { window.__dropped = true; c.error(new TypeError('Load failed')); }
            else { c.enqueue(e.encode('data: [DONE]\\n\\n')); c.close(); }
        } });
        const sse = (b) => new Response(b, { status: 200, headers: { 'Content-Type': 'text/event-stream', 'X-Task-Id': 'task-under-test' } });
        window.fetch = async (u, o) => {
            const url = String((u && u.url) || u);
            if (url.includes('/api/chat/resume/')) {
                window.__resumes++;
                await new Promise(r => setTimeout(r, DELAY));
                return sse(body(REPLAY, false, 30));
            }
            if (url.endsWith('/api/chat')) return sse(body(BEFORE, true, 120));
            if (url.includes('/api/chat/ack') || url.includes('/api/chat/task')) return new Response('{}', { status: 200 });
            return real(u, o);
        };
    })();""" % (json.dumps(_frames(before)), json.dumps(_frames(replay)), resume_delay_ms)


_SPEAKER_ON = {"ghost-speak-replies": "1"}


def _ask(page):
    page.fill("#chat-input", "q")
    page.press("#chat-input", "Enter")


def test_a_drop_during_reasoning_keeps_the_rest_of_it_silent(run):
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["<think>Reasoning one. "],
        ["<think>Reasoning one. ", "Hidden two here. Hidden three here. </think>", "Answer one. Answer two. "]))
    _ask(page)
    _wait(page, lambda: any("Answer" in text for text in backend.spoken), what="the answer to be spoken")
    page.wait_for_timeout(300)
    assert page.evaluate("() => window.__resumes") == 1
    assert not [text for text in backend.spoken if "Hidden" in text or "Reasoning" in text]


def test_a_replayed_retract_leaves_what_is_being_said_alone(run):
    """The replay repeats a retract from before the drop. It used to stop the clip
    playing and say everything after it a second time."""
    backend = Backend([])
    final = ["Final one. ", "Final two. ", "Final three. "]
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["Preface before tool. ", "RETRACT", *final],
        ["Preface before tool. ", "RETRACT", *final, "Final four. "], 1500))   # the wait lets a final clip start
    _ask(page)
    _wait(page, lambda: any("four" in text for text in backend.spoken), what="the new sentence to be spoken")
    page.wait_for_timeout(300)
    for sentence in ("Final one.", "Final two."):
        assert sum(sentence in text for text in backend.spoken) == 1, backend.spoken
    assert page.evaluate("() => window.__pauses") <= 1     # the first pass's retract, never the replay's


def test_a_new_retract_after_a_replay_lets_the_next_text_speak(run):
    """Only history is ignored: past what was seen, a retract works as it always did."""
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["Before the drop. "], ["Before the drop. ", "RETRACT", "After the retract. Second one. "]))
    _ask(page)
    _wait(page, lambda: any("After the retract" in text for text in backend.spoken), what="the replacement to be spoken")


def test_a_press_while_a_resume_catches_up_still_cuts_the_reply(run):
    backend = Backend(["hold on"])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["Said one. ", "Said two. "], ["Said one. ", "Said two. ", "New three. ", "New four. "], 3000))
    _ask(page)
    _wait(page, lambda: page.evaluate("() => window.__resumes") == 1, what="the resume to start")
    _hold_mic(page, 500)
    _wait(page, lambda: backend.inbox == ["hold on"], what="the words to reach the turn")
    page.wait_for_timeout(4000)
    assert not [text for text in backend.spoken if "New" in text], backend.spoken


# ── the microphone and the clips ───────────────────────────────────────
_COUNT_STREAMS = """(() => {
    window.__streams = [];
    const open = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => { const s = await open(c); window.__streams.push(s); return s; };
})();"""
_TRACKS = "() => window.__streams.flatMap(s => s.getTracks().map(t => t.readyState))"


def test_the_button_microphone_goes_off_at_release_not_after_the_transcription(run):
    backend = Backend(["q"])
    script = _COUNT_STREAMS + """(() => { const real = window.fetch; window.fetch = async (u, o) => {
        if (String(u).includes('/api/stt')) await new Promise(r => setTimeout(r, 4000));
        return real(u, o); }; })();"""
    page = run(backend, mic=SILENT, script=script)
    _hold_mic(page, 500)
    _wait(page, lambda: page.evaluate(_TRACKS) == ["ended"], seconds=2, what="the track to end")
    assert backend.chat == []


def test_the_button_microphone_goes_off_when_the_recorder_cannot_start(run):
    script = _COUNT_STREAMS + "window.MediaRecorder = function () { throw new Error('no recorder'); };"
    page = run(Backend([]), mic=SILENT, script=script)
    _hold_mic(page, 300, recorder=False)
    _wait(page, lambda: page.evaluate(_TRACKS) == ["ended"], what="the track to end")


def test_the_switch_holds_when_storage_refuses_it(run):
    script = """(() => { const set = Storage.prototype.setItem;
        Storage.prototype.setItem = function (k, v) {
            if (k === 'ghost-barge-in') throw new DOMException('full', 'QuotaExceededError');
            return set.call(this, k, v); }; })();"""
    backend = Backend(["q"])
    page = run(backend, mic=SILENT, script=script)
    assert page.evaluate("window.GhostCore.toggleBargeIn()") is False
    assert page.evaluate("window.GhostCore.toggleBargeIn()") is True
    assert page.evaluate("window.GhostCore.toggleBargeIn()") is False
    _hold_mic(page, 500)
    _wait(page, lambda: backend.spoken, what="the reply to speak")
    page.wait_for_timeout(1500)
    assert page.evaluate(_MIC)["listening"] is False


def test_a_late_empty_transcript_leaves_a_held_button_alone(run):
    backend = Backend(["first question", ""], stt_delay=2.5)
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription to start")
    box = page.locator("#mic-btn").bounding_box()
    page.mouse.move(box["x"] + 5, box["y"] + 5)
    page.mouse.down()
    _wait(page, lambda: page.evaluate(_MIC)["recording"], what="the recording to start")
    page.wait_for_timeout(3000)
    assert page.evaluate(_MIC)["recording"] is True
    page.mouse.up()


def test_words_being_said_when_speech_is_switched_off_go_to_the_tray(run):
    """They used to vanish, while words already transcribing were kept."""
    backend = Backend(["first question", "half a thought"])
    page = run(backend, mic=[(2, 0.0005), (6, 0.3), (6, 0.0005)])
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    page.wait_for_timeout(600)
    page.evaluate("window.GhostCore.endVoiceSession()")
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the words to be transcribed")
    _wait(page, lambda: "half a thought" in page.inner_text("body"), what="the words to be kept")
    assert backend.chat == ["first question"]


def test_a_clip_that_cannot_load_starts_the_next_one_once(run):
    # Fails late, so the next clips are queued when it reports, and reports both ways as a browser does.
    script = _dropping_stream([], []).replace(
        "if (url.endsWith('/api/chat')) return sse(body(BEFORE, true, 120));",
        "if (url.endsWith('/api/chat')) return sse(body(%s, false, 10));"
        % json.dumps(_frames(["One a. ", "Two b. ", "Three c. ", "Four d. "]))) + """
        (() => { const Spied = window.Audio; let first = true;
            window.Audio = function (src) { const a = new Spied(src);
                if (first) { first = false;
                    a.play = () => new Promise((_, no) => setTimeout(() => {
                        if (a.onerror) a.onerror(); no(new Error('cannot decode')); }, 1500)); }
                return a; }; })();"""
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=script)
    _ask(page)
    _wait(page, lambda: page.evaluate("() => window.__clips.length") >= 2, what="the next clip")
    page.wait_for_timeout(500)
    assert page.evaluate("() => window.__clips.slice(1).filter(a => !a.paused).length") == 1


# ── behaviours only a mutant showed had no test ────────────────────────
_GUM_LOG = """(() => {
    window.__streams = []; window.__constraints = [];
    const open = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => {
        window.__constraints.push(c.audio);
        const s = await open(c); window.__streams.push(s); return s; };
})();"""


def test_listening_uses_echo_cancellation_and_the_button_does_not(run):
    page = run(Backend(["q"]), mic=SILENT, script=_GUM_LOG)
    _hold_mic(page)
    _wait(page, lambda: len(page.evaluate("() => window.__constraints")) >= 2, what="the mic to open")
    button, listening = page.evaluate("() => window.__constraints")[:2]
    assert button["echoCancellation"] is False and listening["echoCancellation"] is True


def test_the_listening_microphone_is_really_released(run):
    page = run(Backend(["q"]), mic=SILENT, script=_GUM_LOG)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    assert "live" in page.evaluate(_TRACKS)
    page.evaluate("window.GhostCore.endVoiceSession()")
    _wait(page, lambda: set(page.evaluate(_TRACKS)) == {"ended"}, what="every track to end")


def test_a_refused_listening_microphone_says_so_once_and_leaves_the_reply_alone(run):
    script = """(() => { const open = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = async (c) => {
            if (c.audio.echoCancellation) throw new DOMException('no', 'NotAllowedError');
            return open(c); }; })();"""
    backend = Backend(["q"])
    backend.speech = _short_clip()
    page = run(backend, mic=SILENT, script=script)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate("() => window.__clips.length") >= 3, what="the clips to play")
    assert page.inner_text("#chat-log").lower().count("no microphone to listen with") == 1


def _short_clip(seconds=1.0):
    out = io.BytesIO()
    _wav(out, [(seconds, 0.01)])
    return out.getvalue()


def test_the_next_clip_waits_while_someone_is_talking(run):
    # The first clip ends before the talking starts; the next one's speech lands during it.
    script = """(() => { const real = window.fetch; let n = 0; window.fetch = async (u, o) => {
        if (String(u).includes('/api/tts') && n++ > 0) await new Promise(r => setTimeout(r, 3500));
        return real(u, o); }; })();"""
    backend = Backend(["q", "words"])
    backend.speech = _short_clip(0.5)
    page = run(backend, mic=[(2, 0.0005), (8, 0.3), (9, 0.0005)], script=script)
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    page.wait_for_timeout(4000)
    assert page.evaluate("() => window.__clips.length") == 1
    assert len(backend.stt_clips) == 1      # still talking, so this says something


def test_a_capture_whose_audio_was_suspended_still_ends(run):
    script = """(() => { const Real = window.AudioContext; window.__contexts = [];
        window.AudioContext = function (...a) { const c = new Real(...a); window.__contexts.push(c); return c; }; })();"""
    backend = Backend(["q", "cut off words"])
    page = run(backend, mic=[(2, 0.0005), (30, 0.3)], script=script)
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    page.wait_for_timeout(600)
    page.evaluate("() => Promise.all(window.__contexts.map(c => c.suspend()))")
    _wait(page, lambda: len(backend.stt_clips) == 2, seconds=15, what="the stalled capture to be sent")


def test_a_spoken_stop_reaches_the_running_turn(run):
    backend = Backend(["q", "stop"], chunk_gap=3.0)      # still running when the word arrives
    page = run(backend, mic=TALK_AFTER_2S)
    _hold_mic(page)
    _wait(page, lambda: backend.inbox == ["stop"], what="the stop to reach the turn")


def test_a_transcription_that_never_answers_does_not_hold_the_reply_for_ever(run):
    script = """(() => { const real = AbortSignal.timeout.bind(AbortSignal);
        AbortSignal.timeout = (ms) => real(Math.min(ms, 1500)); })();"""
    backend = Backend(["q", "too late"], stt_delay=6)
    page = run(backend, mic=TALK_AFTER_2S, script=script)
    _hold_mic(page)
    _wait(page, lambda: _heard(page), what="speech to be heard")
    _wait(page, lambda: len(backend.stt_clips) == 2, what="the transcription")
    _wait(page, lambda: page.evaluate(_CLIP)["paused"] is False, seconds=5, what="the reply to carry on")


def test_switching_talking_over_off_closes_the_microphone_now(run):
    page = run(Backend(["q"]), mic=SILENT)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate(_MIC)["listening"], what="the mic to open")
    assert page.evaluate("window.GhostCore.toggleBargeIn()") is False
    _wait(page, lambda: not page.evaluate(_MIC)["listening"], seconds=3, what="the mic to close")


def test_a_clip_stopped_while_its_speech_was_being_fetched_never_plays(run):
    """stopTTS's generation counter, run for real and not on a test's copy of it."""
    script = """(() => { const real = window.fetch; window.fetch = async (u, o) => {
        const answer = await real(u, o);
        if (String(u).includes('/api/tts')) await new Promise(r => setTimeout(r, 1500));
        return answer; }; })();"""
    backend = Backend(["q"])
    page = run(backend, mic=SILENT, script=script)
    _hold_mic(page)
    _wait(page, lambda: backend.spoken, what="the first sentence to be on its way")
    page.evaluate("window.GhostCore.endVoiceSession()")
    page.wait_for_timeout(2500)
    assert page.evaluate("() => window.__clips.length") == 0


# ── a drop, and what happens in the wait before the resume ────────────
_REASONING_DROP = (["<think>Reasoning one. "],
                   ["<think>Reasoning one. ", "Hidden two here. Hidden three here. </think>", "Answer one. Answer two. "])


# The page waits 1 s before it reattaches. That is too short to act in reliably, on a busy box.
_LONG_RESUME_WAIT = """(() => { const later = window.setTimeout;
    window.setTimeout = (f, ms, ...rest) => later(f, ms === 1000 ? 6000 : ms, ...rest); })();"""


def _spoke_no_reasoning(backend):
    return not [text for text in backend.spoken if "Hidden" in text or "Reasoning" in text]


def test_a_press_in_the_resume_wait_keeps_reasoning_silent(run):
    backend = Backend([""])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_LONG_RESUME_WAIT + _dropping_stream(*_REASONING_DROP))
    _ask(page)
    _wait(page, lambda: page.evaluate("() => window.__dropped"), what="the drop")
    _hold_mic(page, 300)
    assert page.evaluate("() => window.__resumes") == 0
    _wait(page, lambda: any("Answer" in text for text in backend.spoken), what="the answer to be spoken")
    assert _spoke_no_reasoning(backend), backend.spoken


def test_the_speaker_button_in_the_resume_wait_keeps_reasoning_silent(run):
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_LONG_RESUME_WAIT + _dropping_stream(*_REASONING_DROP))
    _ask(page)
    _wait(page, lambda: page.evaluate("() => window.__dropped"), what="the drop")
    page.click("#tts-toggle-btn")
    page.click("#tts-toggle-btn")
    assert page.evaluate("() => window.__resumes") == 0
    _wait(page, lambda: any("Answer" in text for text in backend.spoken), what="the answer to be spoken")
    assert _spoke_no_reasoning(backend), backend.spoken


def test_a_drop_mid_sentence_says_it_once_whole(run):
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["Half a sente"], ["Half a sente", "nce here. Next one here. "]))
    _ask(page)
    _wait(page, lambda: any("Next" in text for text in backend.spoken), what="the next sentence")
    assert backend.spoken[0] == "Half a sentence here."


_ADOPTED = ["Adopted one is here. ", "Adopted two is here. ", "Adopted three is here. "]
_ADOPTABLE = """(() => { const real = window.fetch, list = %s, e = new TextEncoder();
    window.fetch = async (u, o) => { const url = String((u && u.url) || u);
        if (url.includes('/api/chat/task/')) return new Response(JSON.stringify({exists: true, cancelled: false}), {status: 200});
        if (url.includes('/api/chat/ack')) return new Response('{}', {status: 200});
        if (url.includes('/api/chat/resume/')) return new Response(new ReadableStream({ async start(c) {
            for (const f of list) { c.enqueue(e.encode(f)); await new Promise(r => setTimeout(r, 30)); }
            c.enqueue(e.encode('data: [DONE]\\n\\n')); c.close(); } }),
            {status: 200, headers: {'Content-Type': 'text/event-stream'}});
        return real(u, o); }; })();""" % json.dumps(_frames(_ADOPTED))
_DEAD_TAB = """() => {
    localStorage.setItem('ghost_inflight_turn:dead-tab', JSON.stringify({taskId: 'orphan-task',
        sessionId: 'session-under-test', tabId: 'dead-tab', ts: Date.now() - 60000,
        beat: Date.now() - 60000, startedAt: Date.now() - 60000}));
    document.dispatchEvent(new Event('visibilitychange')); }"""


@pytest.mark.parametrize("tap_first", [False, True])
def test_another_tabs_turn_does_not_inherit_this_pages_counts_or_cut(run, tap_first):
    backend = Backend([""])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_ADOPTABLE)
    _ask(page)
    _wait(page, lambda: len(backend.spoken) >= len(SENTENCES), what="the first turn to be spoken")
    page.wait_for_timeout(500)
    if tap_first:
        _hold_mic(page, 300)
        _wait(page, lambda: backend.stt_clips, what="the empty transcript")
    backend.spoken.clear()
    page.evaluate(_DEAD_TAB)
    _wait(page, lambda: any("Adopted three" in text for text in backend.spoken), what="the adopted reply to be spoken")
    assert any("Adopted one" in text for text in backend.spoken), backend.spoken


def test_the_spoken_reply_reaches_the_speakers(run):
    """currentTime advances with nothing connected, so only the graph shows this."""
    script = """(() => { window.__toSpeakers = 0; const connect = AudioNode.prototype.connect;
        AudioNode.prototype.connect = function (to, ...rest) {
            if (this instanceof AnalyserNode && to === this.context.destination) window.__toSpeakers++;
            return connect.call(this, to, ...rest); }; })();"""
    backend = Backend(["q"])
    page = run(backend, mic=SILENT, script=script)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate("() => window.__toSpeakers") > 0, what="the analyser to reach the speakers")


def test_the_listening_microphone_never_reaches_the_speakers(run):
    script = """(() => { window.__micToSpeakers = []; const connect = AudioNode.prototype.connect;
        AudioNode.prototype.connect = function (to, ...rest) {
            if (this instanceof GainNode && to === this.context.destination) window.__micToSpeakers.push(this.gain.value);
            return connect.call(this, to, ...rest); }; })();"""
    page = run(Backend(["q"]), mic=SILENT, script=script)
    _hold_mic(page)
    _wait(page, lambda: page.evaluate("() => window.__micToSpeakers").count(0) >= 1, what="the silent output")
    assert set(page.evaluate("() => window.__micToSpeakers")) == {0}


def test_a_retract_during_open_reasoning_lets_the_replacement_speak(run):
    backend = Backend([])
    page = run(backend, mic=SILENT, storage=_SPEAKER_ON, script=_dropping_stream(
        ["<think>Never closed. ", "RETRACT", "Answer one. Answer two. "],
        ["<think>Never closed. ", "RETRACT", "Answer one. Answer two. "]))
    _ask(page)
    _wait(page, lambda: any("Answer" in text for text in backend.spoken), what="the replacement to be spoken")
    assert not [text for text in backend.spoken if "Never" in text]

