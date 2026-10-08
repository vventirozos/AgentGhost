"""Drive the REAL client window, offscreen, on the device, against a fake agent.

``client.py`` imports PyQt6, which exists only on the handheld — so nothing in
the project's test suite can import it, and for a year its wiring was checked
by reading it. ``deploy.sh`` now runs this in the staging directory before it
installs anything: it builds the actual ``MainWindow`` (only the WebGL face is
replaced by a recorder — an offscreen platform has no GPU), points it at a
fake agent served from a thread in this process, and drives real turns
through it. A failure aborts the deploy with the live client untouched.

What it can see that the unit tests cannot: Qt's own behaviour (a QLabel's
size hint, a scroll bar's range signal, a shortcut's focus rules), and
whether the client's methods actually fit together — the busy guard really
keeps a second message in the input, Stop really reaches ``/api/turn/cancel``
with THIS turn's id, ``/shutdown`` really needs typing twice.

    python3 device_probe.py          # exit 0 = every check passed

Safe to run beside the live client: offscreen, no GPU, no microphone, no
camera, its own session file — and the three things the client does to the
DEVICE are replaced by recorders before the window is built: killing audio
playback (``_kill_playback`` — the first version of this probe ran the real
``pkill aplay`` six times per deploy, cutting off whatever the live client
was saying), running helpers (``_run_quiet`` — it would have woken a sleeping
panel), and the power commands (``_power``, plus ``GHOST_DRY_POWER=1``).
"""

from __future__ import annotations

import asyncio
import atexit
import http.server
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── the fake agent ──────────────────────────────────────────────────────────
STATE = {
    "chats": [], "cancels": [], "feedback": [], "acks": [], "gets": [], "stt": [],
    "chat_delay": 0.0, "chat_status": 200, "cancel": threading.Event(),
    "hold": False,          # after the first piece, wait for a cancel
    "ignore_cancel": False, # answer "cancelled" but keep streaming (a wedged turn)
    "pending": [], "watermark": 7, "sessions": {}, "turns": [],
    "pieces": ["Probe ", "reply. ", "It has ", "two sentences."],
}


class _Agent(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):          # noqa: A003
        pass

    def _json(self, obj, status=200):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):                      # noqa: N802
        path = self.path.split("?", 1)[0]
        STATE["gets"].append(path)
        if path == "/api/turns":
            return self._json({"turns": STATE["turns"]})
        if path == "/api/sessions":
            return self._json({"enabled": True, "sessions": [
                {"id": sid, "title": f"title of {sid}", "updated_at": time.time() - 3600,
                 "message_count": len(msgs)} for sid, msgs in STATE["sessions"].items()]})
        if path.startswith("/api/sessions/"):
            sid = path.rsplit("/", 1)[1]
            if sid not in STATE["sessions"]:
                return self._json({"detail": "session not found"}, 404)
            return self._json({"id": sid, "messages": STATE["sessions"][sid]})
        if path == "/api/notifications/pending":
            records, STATE["pending"] = STATE["pending"], []
            return self._json({"enabled": True, "records": records,
                               "watermark": STATE["watermark"]})
        return self._json({"detail": "not found"}, 404)

    def do_POST(self):                     # noqa: N802
        if self.path == "/api/stt":
            declared = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(declared)
            STATE["stt"].append({"declared": declared, "got": len(raw), "clip": raw.count(b"RIFF")})
            return self._json({"text": "hello from the mic"})
        body = self._body()
        if self.path == "/api/notifications/ack":
            STATE["acks"].append(body)
            return self._json({"ok": True})
        if self.path == "/api/feedback":
            STATE["feedback"].append(body)
            return self._json({"ok": True})
        if self.path == "/api/turn/cancel":
            STATE["cancels"].append(body)
            if not STATE["ignore_cancel"]:
                STATE["cancel"].set()
            return self._json({"cancelled": True, "request_id": body.get("request_id")})
        if self.path == "/api/chat":
            rid = self.headers.get("X-Request-ID") or "norid"
            STATE["chats"].append({"rid": rid, "body": body})
            if STATE["chat_status"] != 200:
                return self._json({"error": {"message": "model is loading"}},
                                  STATE["chat_status"])
            STATE["cancel"].clear()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for i, piece in enumerate(STATE["pieces"]):
                    # In "hold" mode the reply stalls after its first piece
                    # until the client's stop arrives — so "a stop cuts a
                    # reply that has begun" does not depend on timing.
                    wait = 15.0 if (STATE["hold"] and i == 1) else STATE["chat_delay"]
                    if STATE["cancel"].wait(wait):
                        break
                    frame = {"id": f"chatcmpl-{rid}",
                             "choices": [{"index": 0, "delta": {"content": piece}}]}
                    self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                    self.wfile.flush()
                # The shapes that are not content: a usage frame (its `choices`
                # is EMPTY — indexing it raised IndexError out of the stream),
                # a non-object, a keep-alive comment.
                usage = {"id": f"chatcmpl-{rid}", "choices": [],
                         "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
                self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: 12\n\n: keepalive\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass                       # the client hung up (a forced stop)
            return None
        return self._json({"detail": "not found"}, 404)


def _serve():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Agent)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


# ── environment, BEFORE the client is imported ──────────────────────────────
PORT = _serve()
TMP = tempfile.mkdtemp(prefix="ghost-probe-")
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
os.environ.update({
    "QT_QPA_PLATFORM": "offscreen",
    "GHOST_AGENT_BASE": f"http://127.0.0.1:{PORT}",
    "GHOST_API_KEY": "probe-key",
    "GHOST_DRY_POWER": "1",
    "GHOST_SESSION_STATE": os.path.join(TMP, "session_id"),
    "GHOST_LINK_POLL_S": "0",
    "GHOST_NOTIFY_POLL_S": "0",
    "GHOST_CHIME_AFTER_S": "20",
    "GHOST_STT_REVIEW_S": "2.5",
})
for _k in ("GHOST_FACE_IDLE_S", "GHOST_FACE_SLEEP_S", "GHOST_FACE_IDLE_FPS",
           "GHOST_FACE_BATTERY_FPS", "GHOST_BUBBLE_AGENT", "GHOST_BUBBLE_USER"):
    os.environ.pop(_k, None)

# PyQt6 BEFORE qasync, as in client.py: qasync binds to whichever Qt is already
# imported and otherwise tries PyQt5 first — which is installed in this venv
# too. Imported the other way round, the event loop runs on one Qt and the
# widgets on another, and the first await never returns.
from PyQt6.QtCore import QEvent, Qt                       # noqa: E402
from PyQt6.QtGui import QKeyEvent, QTextDocument, QTextOption  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget         # noqa: E402
import qasync                                             # noqa: E402

import agentapi                                           # noqa: E402
import chatlog                                            # noqa: E402
import client                                             # noqa: E402
import markup                                             # noqa: E402
import webface                                            # noqa: E402

FAILS = []
PASSED = 0


def step(title):
    """Section marker — printed with GHOST_PROBE_VERBOSE=1, so a crash inside
    Qt (which leaves no Python traceback) can be placed."""
    if os.environ.get("GHOST_PROBE_VERBOSE") == "1":
        print(f"  … {title}", flush=True)


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILS.append(f"{name}" + (f" — {detail}" if detail else ""))
        print(f"  FAIL  {name}  {detail}", flush=True)


# Every method the client calls on the face, read from its source — so the
# recorder below cannot quietly accept a call the real widget would not.
FACE_CALLS = sorted(set(re.findall(
    r"self\.web_face\.([a-z_]+)\(", open(os.path.join(HERE, "client.py")).read())))


FACE_ERRORS = []


class FakeFace(QWidget):
    """Stands in for the WebGL face: records what the client asks of it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calls = []


def _recorder(name):
    real = getattr(webface.WebFaceWidget, name, None)

    def method(self, *args, **kw):
        # Bound against the REAL method's signature: a call the real widget
        # would reject (wrong arity, a misspelt keyword) must not pass here
        # just because a recorder accepts anything. Collected, not raised —
        # the client wraps some face calls in try/except.
        if real is not None:
            try:
                inspect.signature(real).bind(self, *args, **kw)
            except TypeError as exc:
                FACE_ERRORS.append(f"{name}{args}: {exc}")
        self.calls.append((name,) + args)
    return method


for _name in FACE_CALLS:
    if not hasattr(QWidget, _name):
        setattr(FakeFace, _name, _recorder(_name))


async def until(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return bool(pred())


def bubbles(w):
    return w.chat_display._body.findChildren(chatlog._Bubble)


def texts(w, role=None):
    return [b.text() for b in bubbles(w) if role is None or b._role == role]


def send(w, text):
    w.text_input.setText(text)
    w.handle_input()


async def main(app):
    # ── 0. the client and the face agree on the face's API ───────────────
    step('0' + '. ' + "the client and the face agree on the face's API")
    for name in FACE_CALLS:
        check(f"webface has .{name}()", hasattr(webface.WebFaceWidget, name))
    client.WebFaceWidget = FakeFace
    powered, helpers, kills, popped = [], [], [], []
    client._power = lambda argv: powered.append(list(argv))
    # What the client does to the DEVICE, recorded instead of done.
    client.MainWindow._kill_playback = lambda self: kills.append(1)

    async def _fake_run_quiet(self, *argv, env=None, stdin=None):
        helpers.append(list(argv))
        return 0, ("DSI-1 off\n" if list(argv) == ["wlopm"] else "")
    client.MainWindow._run_quiet = _fake_run_quiet

    async def _fake_download(self, path):
        popped.append(path)
    client.MainWindow._download_and_show_image = _fake_download

    # The microphone: "recording" writes a small clip to the path it is given.
    recorded = []

    def _fake_arecord(self, path):
        recorded.append(path)
        with open(path, "wb") as fh:
            fh.write(b"RIFF" + b"\0" * 4000)
    client.MainWindow._start_arecord = _fake_arecord
    client.MainWindow._stop_arecord = lambda self: None
    client.STT_SERVER_URL = f"http://127.0.0.1:{PORT}/api/stt"

    w = client.MainWindow()
    w.show()
    await asyncio.sleep(0.2)
    face = w.web_face
    log = w.chat_display
    check("the session id was created and remembered",
          agentapi.valid_session_id(w.session_id)
          and agentapi.load_session_id() == w.session_id)
    check("the placeholder is the real one",
          w.text_input.placeholderText() == client.INPUT_PLACEHOLDER)

    # ── 1. a bubble shrinks back after the waiting caption ───────────────
    step('1' + '. ' + 'a bubble shrinks back after the waiting caption')
    fresh = log.add("<p>Hello!</p>", "agent")
    log.start_agent()
    log.update_agent("<span style='font-family:monospace'>0:34</span>&nbsp; reading a file · "
                     "some_long_notes_file.md &nbsp;📖")
    wide = log._current_agent.width()
    log.update_agent("<p>Hello!</p>")
    narrow = log._current_agent.width()
    log.end_agent()
    check("a short reply is not left at the caption's width",
          abs(narrow - fresh.width()) <= 2 and narrow < wide,
          f"fresh={fresh.width()} caption={wide} after={narrow}")

    # ── 2. nothing is clipped: code lines wrap, long runs break ──────────
    step('2' + '. ' + 'nothing is clipped: code lines wrap, long runs break')
    def ideal_width(html, width):
        doc = QTextDocument()
        opt = doc.defaultTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WordWrap)   # what a wrapped QLabel uses
        doc.setDefaultTextOption(opt)
        doc.setHtml(html)
        doc.setTextWidth(width)
        return doc.idealWidth()

    inner = log._max_width("agent") - 40
    code = "```\nx = some_function(argument_one, argument_two, argument_three, " \
           "argument_four, argument_five, argument_six, argument_seven)\n```"
    styled = markup.style_markup(markup.render_reply(code), client.T.FONT,
                                 client.T.ACCENT, client.T.TEXT_DIM)
    check("a long code line wraps inside the bubble",
          ideal_width(styled, inner) <= inner + 1, f"{ideal_width(styled, inner):.0f} > {inner}")
    url = "see https://example.com/" + "a" * 160 + " ok"
    styled = markup.style_markup(markup.render_reply(url), client.T.FONT,
                                 client.T.ACCENT, client.T.TEXT_DIM)
    check("a long unbroken run wraps inside the bubble",
          ideal_width(styled, inner) <= inner + 1, f"{ideal_width(styled, inner):.0f} > {inner}")
    log.clear()

    # ── 3. one ordinary turn ─────────────────────────────────────────────
    step('3' + '. ' + 'one ordinary turn')
    send(w, "hello <there>")
    check("a turn is in flight after Enter", w._busy())
    check("STOP is shown while a turn runs", not w.stop_btn.isHidden())
    check("the turn finished", await until(lambda: not w._busy()))
    chat = STATE["chats"][-1] if STATE["chats"] else {"rid": "", "body": {}}
    check("exactly one chat request was sent", len(STATE["chats"]) == 1)
    check("the turn carries the session id", chat["body"].get("session_id") == w.session_id)
    check("the turn carries a request id minted here",
          bool(re.fullmatch(r"[0-9a-f]{8}", chat["rid"])), chat["rid"])
    check("the operator's text is escaped, not swallowed as markup",
          texts(w, "user") == ["hello &lt;there&gt;"], str(texts(w, "user")))
    check("the reply is on screen", any("two sentences" in t for t in texts(w, "agent")))
    check("the reply is in the conversation",
          [m["role"] for m in w.conversation_history] == ["user", "assistant"]
          and w.conversation_history[-1]["content"] == "Probe reply. It has two sentences.")
    check("the reply can be rated", w._last_reply_rid == chat["rid"]
          and w.good_btn.isEnabled() and w.bad_btn.isEnabled())
    check("frames that carry no text are not a fault (usage, non-object, comment)",
          not any("fault" in t for t in texts(w)), str(texts(w)))
    await until(lambda: ["wlopm", "--on", "*"] in helpers, 3.0)
    check("a reply that lands on a dark panel wakes it",
          ["wlopm"] in helpers and ["wlopm", "--on", "*"] in helpers, str(helpers))
    check("…and chimes, since nobody was watching",
          await until(lambda: any(h[:1] == ["aplay"] for h in helpers), 3.0), str(helpers))
    check("sending silences speech without touching another process's audio here",
          len(kills) >= 1)
    check("STOP is hidden again", w.stop_btn.isHidden())
    check("the face was told the reply is being written",
          ("set_phase", "write") in face.calls and ("set_phase", None) in face.calls)

    # ── 4. rating ────────────────────────────────────────────────────────
    step('4' + '. ' + 'rating')
    w.rate_last("positive")
    check("the rating reached the agent", await until(lambda: STATE["feedback"]))
    fb = STATE["feedback"][-1] if STATE["feedback"] else {}
    check("the rating names this reply",
          fb.get("request_id") == chat["rid"] and fb.get("signal") == "positive"
          and fb.get("source") == "clockwork", str(fb))
    check("the chip latches", await until(lambda: w._rated == "positive"))

    # ── 5. busy guard and stop ───────────────────────────────────────────
    step('5' + '. ' + 'busy guard and stop')
    # The reply's first piece comes after 0.6 s (so the log lines below land
    # while the turn is still "thinking"), then it stalls until stopped.
    STATE["hold"], STATE["chat_delay"] = True, 0.6
    face.calls.clear()
    send(w, "a slow one")
    await asyncio.sleep(0.2)
    check("the rating chips are off while a new reply is arriving",
          not w.good_btn.isEnabled() and w._last_reply_rid is None)
    # §4ML: a corridor that opens FIRST but is not ours (a member's, a
    # probe's, self-play's) is never adopted — only the one carrying the id
    # this client minted
    w.note_log_line("┌─ 7F 1f00d7f3  request started  15:16:25 ─────────────")
    w.note_log_line("│  7F  📖  +1.00s  file read           secret_member_notes.md")
    check("another turn's corridor is not adopted",
          "secret_member_notes" not in str(face.calls) and "secret_member_notes" not in w.ticker.desc,
          str(face.calls[:4]))
    # the log socket's lines for THIS turn drive the face
    w.note_log_line(f"┌─ 99 {w._turn_rid[:8]}  request started  15:16:26 ─────────────")
    w.note_log_line("│  99  📖  +8.06s  file read           notes.md")
    w.note_log_line("│  99  🧪  +9.10s  verifier            CONFIRMED: grounded")
    check("a step line gives the face its gait",
          ("set_phase", "read") in face.calls and ("set_phase", "verify") in face.calls,
          str(face.calls[:8]))
    # Wait for the first piece of the reply: the stop below must cut a reply
    # that has already begun, so there is a partial one to keep.
    await until(lambda: bool(w.current_response_text))
    send(w, "second message")
    check("a second message is NOT sent while a turn runs", len(STATE["chats"]) == 2)
    check("…and its text stays in the input", w.text_input.text() == "second message")
    check("no caller can start a second turn (the camera path uses the same gate)",
          w._submit("from the camera", "from the camera") is False
          and len(STATE["chats"]) == 2 and len(w.conversation_history) == 3,
          f"chats={len(STATE['chats'])} history={len(w.conversation_history)}")
    w.tts_enabled, w.is_recording = True, True
    w._say(["spoken into the open microphone"])
    check("nothing is queued to be spoken while the operator is recording",
          client.audio_queue.empty())
    w.tts_enabled, w.is_recording = False, False
    slow_rid = STATE["chats"][-1]["rid"]
    w.request_stop()
    check("stop reached the agent", await until(lambda: STATE["cancels"]))
    check("stop names THIS turn",
          STATE["cancels"] and STATE["cancels"][-1].get("request_id") == slow_rid,
          str(STATE["cancels"]))
    check("the stopped turn ended", await until(lambda: not w._busy()))
    check("the verdict the verifier logged shapes the release",
          ("note_verdict", "pass") in face.calls, str([c for c in face.calls if c[0] == "note_verdict"]))
    check("a partial reply stays in the conversation",
          w.conversation_history[-1] == {"role": "assistant", "content": "Probe "},
          repr(w.conversation_history[-1]))
    check("a stop the agent took is reported as stopped", "stopped." in texts(w)[-1], texts(w)[-1])
    STATE["hold"], STATE["chat_delay"] = False, 0.0
    w.text_input.clear()
    w.request_stop()
    check("stop with nothing running says so", "nothing is running" in texts(w)[-1])

    # ── 5b. a turn that will not stop is forced ──────────────────────────
    step("5b. a forced stop")
    STATE["hold"], STATE["ignore_cancel"] = True, True
    before = len(w.conversation_history)
    send(w, "one that will not stop")
    await until(lambda: bool(w.current_response_text))
    w.request_stop()
    await asyncio.sleep(0.4)
    check("a stop the agent has not acted on leaves the turn running", w._busy())
    w.request_stop()                               # the second press forces it
    check("a second stop ends the turn here", await until(lambda: not w._busy(), 5.0))
    check("…and tells the agent to cancel outright",
          STATE["cancels"][-1].get("hard") is True, str(STATE["cancels"][-1]))
    check("a reply cut by a forced stop is kept, once",
          w.conversation_history[before:] == [
              {"role": "user", "content": "one that will not stop"},
              {"role": "assistant", "content": "Probe "}], repr(w.conversation_history[before:]))
    check("STOP is hidden after a forced stop", w.stop_btn.isHidden())
    STATE["hold"], STATE["ignore_cancel"] = False, False
    STATE["cancel"].set()                          # let the server's thread go

    # ── 6. commands ──────────────────────────────────────────────────────
    step('6' + '. ' + 'commands')
    n, sent = len(bubbles(w)), len(STATE["chats"])
    send(w, "/help")
    check("/help answers locally", len(bubbles(w)) == n + 1 and "/sessions" in texts(w)[-1]
          and len(STATE["chats"]) == sent)
    send(w, "/hlep")
    check("a typo is not sent to the agent",
          len(STATE["chats"]) == sent and "unknown command" in texts(w)[-1])
    check("…and stays in the input to be corrected", w.text_input.text() == "/hlep")
    send(w, "/shutdown")
    check("/shutdown once does nothing", powered == [])
    send(w, "/help")
    send(w, "/shutdown")
    check("/shutdown, something else, /shutdown does nothing", powered == [])
    send(w, "/shutdown")
    check("/shutdown twice powers off",
          powered == [["sudo", "shutdown", "-h", "now"]], str(powered))
    send(w, "/stop it")
    check("a stray word after a command runs nothing and keeps the text",
          "takes nothing after it" in texts(w)[-1] and w.text_input.text() == "/stop it")
    w.text_input.clear()
    send(w, "/bright")
    check("/bright reports the level", re.search(r"brightness \d+ of \d+", texts(w)[-1]) is not None,
          texts(w)[-1])
    n = len(bubbles(w))
    send(w, "/vol")
    await until(lambda: len(bubbles(w)) > n)
    check("/vol reports the volume",
          re.search(r"volume \d+%|no audio output found", texts(w)[-1]) is not None, texts(w)[-1])
    before_sid, n_chats = w.session_id, len(STATE["chats"])
    send(w, "/new idea: use a queue instead")
    await until(lambda: not w._busy())
    check("a sentence that starts with a command word is a MESSAGE",
          len(STATE["chats"]) == n_chats + 1 and w.session_id == before_sid
          and STATE["chats"][-1]["body"]["messages"][-1]["content"] == "/new idea: use a queue instead")

    # ── 7. a new conversation is a new session ───────────────────────────
    step('7' + '. ' + 'a new conversation is a new session')
    before = w.session_id
    send(w, "/new")
    check("/new changes the session id", w.session_id != before
          and agentapi.load_session_id() == w.session_id)
    check("/new empties the conversation", w.conversation_history == []
          and not w.good_btn.isEnabled())

    # ── 8. restoring a stored conversation ───────────────────────────────
    step('8' + '. ' + 'restoring a stored conversation')
    STATE["sessions"]["cw-probe-stored"] = [
        {"role": "system", "content": "never shown"},
        {"role": "user", "content": "an earlier question"},
        {"role": "assistant", "content": "an earlier answer ![pic](/api/download/x.png)"},
    ]
    w._begin_session("cw-probe-stored")
    STATE["gets"].clear()
    await w.restore_session()
    check("the stored conversation is drawn",
          texts(w, "user") == ["an earlier question"]
          and any("an earlier answer" in t for t in texts(w, "agent")), str(texts(w)))
    check("…and loaded as history",
          [m["role"] for m in w.conversation_history] == ["user", "assistant"])
    await asyncio.sleep(0.1)
    check("a restored image is not popped open",
          popped == [] and "/api/download/x.png" in w.shown_images, str(popped))
    STATE["sessions"]["cw-probe-other"] = [
        {"role": "user", "content": "from the browser"},
        {"role": "assistant", "content": "continued on the handheld"}]
    await w._show_sessions()
    await w._open_session("2")
    check("/open switches to the listed conversation",
          w.session_id == "cw-probe-other"
          and texts(w, "user") == ["from the browser"], f"{w.session_id} {texts(w)}")

    # ── 9. notifications ─────────────────────────────────────────────────
    step('9' + '. ' + 'notifications')
    STATE["pending"] = [{"ts": time.time(), "phase": "scheduled_task",
                         "summary": "backup finished", "severity": "notify", "meta": {}}]
    import httpx
    poller = agentapi.NotifyPoller(client.AGENT_BASE, client.GHOST_API_KEY)
    async with httpx.AsyncClient(timeout=5.0) as hc:
        delivered = await poller.cycle(hc, w._deliver_notifications)
    check("a pending notification is shown", delivered == 1
          and "backup finished" in texts(w)[-1], texts(w)[-1])
    check("…and acked with the server's watermark",
          STATE["acks"] and STATE["acks"][-1] == {"consumer": "clockwork", "watermark": 7},
          str(STATE["acks"]))

    # ── 10. scrolling ────────────────────────────────────────────────────
    step('10' + '. ' + 'scrolling')
    for i in range(40):
        log.add(f"line {i}", "system")
    await asyncio.sleep(0.3)
    bar = log.verticalScrollBar()
    check("the view follows new content while at the bottom",
          bar.maximum() > 0 and bar.value() == bar.maximum(), f"{bar.value()}/{bar.maximum()}")
    bar.setValue(0)
    log.add("arrived while reading back", "system")
    await asyncio.sleep(0.3)
    check("…and stays put once the operator has scrolled up", bar.value() == 0,
          f"value={bar.value()}")
    w.text_input.setFocus()
    QApplication.sendEvent(w.text_input, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier))
    check("PgDn scrolls the transcript from the input", bar.value() > 0, f"value={bar.value()}")
    QApplication.sendEvent(w.text_input, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier))
    up = bar.value()
    QApplication.sendEvent(w.text_input, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier))
    check("Shift+Up / Shift+Down scroll too", bar.value() > up, f"{up} -> {bar.value()}")

    # ── 11. voice review countdown ───────────────────────────────────────
    step('11' + '. ' + 'voice review countdown')
    w.text_input.setText("a misheard transcript")
    w._after_transcript()
    check("a transcript waits before sending",
          w.review_timer.isActive() and w.ptt_btn.text().startswith("↵") and not w._busy())
    QApplication.sendEvent(w.text_input, QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier))
    check("moving the caret cancels the countdown (not only typing)",
          not w.review_timer.isActive() and "PTT" in w.ptt_btn.text())
    w._after_transcript()
    w._on_text_edited("a corrected transcript")
    check("an edit cancels the countdown", not w.review_timer.isActive())
    w.text_input.clear()

    # ── 11b. push-to-talk ────────────────────────────────────────────────
    step("11b. push-to-talk")
    check("a held key is ONE action, not twenty-five a second",
          not any(sc.autoRepeat() for sc in (
              w.esc_shortcut, w.tts_shortcut, w.stop_shortcut, w.snap_shortcut,
              w.fullscreen_shortcut, w.good_shortcut, w.bad_shortcut)))

    def let_go_of_esc():
        QApplication.sendEvent(w.text_input, QKeyEvent(
            QEvent.Type.KeyRelease, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))

    w.toggle_ptt()
    check("Esc starts a recording, in a file of its own",
          w.is_recording and len(recorded) == 1 and os.path.exists(recorded[0]))
    let_go_of_esc()
    check("a TAP leaves the recording running (press again to send)", w.is_recording)
    w._rec_started -= 1.0                       # …as if Esc had been held for a second
    let_go_of_esc()
    check("letting go of a HELD Esc ends the recording", not w.is_recording)
    check("the clip reaches the transcriber", await until(lambda: STATE["stt"], 5.0))
    up = STATE["stt"][-1] if STATE["stt"] else {}
    check("…whole: the bytes sent are the bytes declared",
          up.get("declared") == up.get("got") and up.get("clip") == 1, str(up))
    check("the transcript lands in the input, waiting to be sent",
          await until(lambda: w.review_timer.isActive(), 3.0)
          and w.text_input.text() == "hello from the mic", w.text_input.text())
    w._cancel_review()
    w.text_input.clear()
    check("the clip's file is removed once it is read", not os.path.exists(recorded[0]))
    sent = len(STATE["stt"])
    w.toggle_ptt()
    w.toggle_ptt()                              # on and off again at once: a slip of the key
    await asyncio.sleep(0.4)
    check("a slip of the key is not uploaded",
          len(STATE["stt"]) == sent and len(recorded) == 2 and not os.path.exists(recorded[1])
          and not w.is_recording, f"uploads={len(STATE['stt'])}")
    check("each recording has a file of its OWN (an upload must never read the next one)",
          len(set(recorded)) == len(recorded), str(recorded))

    # ── 12. face frame rate ──────────────────────────────────────────────
    step('12' + '. ' + 'face frame rate')
    w._last_input_at = time.monotonic()
    w._apply_face_rate()
    w._last_input_at = time.monotonic() - 100
    w._apply_face_rate()
    w._last_input_at = time.monotonic() - 700
    w._apply_face_rate()
    w._note_activity()
    rates = [c[1] for c in face.calls if c[0] == "set_rate"]
    check("the face slows when idle, pauses when left, wakes on input",
          len(rates) >= 4 and rates[-4] == 0 and rates[-3] in (5, 10)
          and rates[-2] == -1 and rates[-1] == 0, str(rates[-4:]))

    # ── 13. status readout ───────────────────────────────────────────────
    step('13' + '. ' + 'status readout')
    w.update_stats()
    check("the status readout draws", "●" in w.stats_label.text(), w.stats_label.text())

    # ── 14. errors are said in words ─────────────────────────────────────
    step('14' + '. ' + 'errors are said in words')
    STATE["chat_status"] = 503
    send(w, "while the model loads")
    await until(lambda: not w._busy())
    check("an HTTP error carries the server's message",
          "HTTP 503" in texts(w)[-1] and "model is loading" in texts(w)[-1], texts(w)[-1])
    check("a failed turn cannot be rated", w._last_reply_rid is None
          and not w.good_btn.isEnabled())
    check("a failed turn does not leave the face 'thinking'", w._face_mood == "idle", w._face_mood)
    STATE["chat_status"] = 200
    step("14b. an unreachable agent")
    real_base, client.AGENT_BASE = client.AGENT_BASE, "http://127.0.0.1:9"
    send(w, "into the void")
    await until(lambda: not w._busy())
    check("an unreachable agent is named as such",
          "cannot reach 127.0.0.1:9" in texts(w)[-1], texts(w)[-1])
    check("…and the link dot goes red", w._agent_ok is False)
    client.AGENT_BASE = real_base

    step("15. closing")
    await asyncio.sleep(0.5)       # let the reply-landed helpers finish
    check("every call the client made to the face is one the real widget accepts",
          FACE_ERRORS == [], str(FACE_ERRORS[:3]))
    w.close()
    step("16. closed")


if __name__ == "__main__":
    application = QApplication(sys.argv)
    loop = qasync.QEventLoop(application)
    asyncio.set_event_loop(loop)
    try:
        with loop:
            loop.run_until_complete(main(application))
    except Exception as exc:  # noqa: BLE001 — a crash IS the finding
        import traceback
        traceback.print_exc()
        FAILS.append(f"probe crashed: {type(exc).__name__}: {exc}")
    total = PASSED + len(FAILS)
    if FAILS:
        print(f"  device probe: {len(FAILS)} of {total} checks FAILED", flush=True)
        for f in FAILS:
            print(f"    - {f}", flush=True)
        sys.exit(1)
    print(f"  device probe: all {total} checks passed", flush=True)
