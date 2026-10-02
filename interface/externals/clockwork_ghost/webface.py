"""Web-backed face for the uConsole client — the SAME face the web UI runs.

Rather than reimplementing the web UI's particle field in QPainter (a ~2,700
line port that would immediately start drifting from the original), this hosts
the real ``matrix_graph.js`` in a ``QWebEngineView``. Every form, the thermal
palette, the immersion dive, pulses and error flinches come along for free, and
future work on the web face lands on the handheld automatically.

**Measured on the device (ClockworkPi CM4, XWayland, 1280x720 panel):**
WebGL 1.0 is available (three r160 falls back to it), and cost is purely
fill-rate bound — 16 fps at full resolution, 32 fps at ``zoom 2.0``. The zoom
trick is why this is viable: a zoom factor shrinks the CSS viewport, so the
drawing buffer shrinks with it and the browser upscales the result. At
640x329 the face also trips its own ``IS_MOBILE`` query (``max-height: 600px``),
which halves node count (250 -> 120) and bloom scale — the exact profile that
exists for weak GPUs. Two knobs, both env-overridable, so this can be tuned on
the device without a redeploy.

**Why a loopback HTTP server rather than file://** — ES module imports are
blocked under ``file://`` by Chromium's origin rules. The failure is silent
(the module simply never executes, no console error), which cost real time to
diagnose, so the server is not optional. It binds 127.0.0.1 only and serves a
single directory.
"""

from __future__ import annotations

import functools
import http.server
import json
import logging
import os
import socket
import threading
from pathlib import Path

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout

import facestate

# MODULE-LEVEL import, and it must stay that way. Qt refuses to create a
# QWebEngineView unless QtWebEngineWidgets was imported (or
# AA_ShareOpenGLContexts was set) BEFORE the QCoreApplication exists —
# importing it lazily inside __init__ raises "QtWebEngineWidgets must be
# imported ... before a QCoreApplication instance is created" and the face
# silently degrades to a blank panel. client.py imports this module at the top,
# well before it constructs QApplication, so this import is what makes the face
# work at all. Guarded so a machine without QtWebEngine can still run the
# client with the other faces.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

FACE_DIR = Path(__file__).resolve().parent / "webface"

# 2.0 -> 640x329 CSS viewport -> mobile profile -> ~32 fps on the CM4.
# Lower it for a crisper face on stronger hardware.
ZOOM = float(os.environ.get("GHOST_FACE_ZOOM", "2.0"))

# The face opens on whatever the ◈ chip last selected (2026-08-03). The JS
# already persists that choice to localStorage — and on this device that can
# never survive a restart: the profile is off-the-record AND the loopback
# origin's port changes every boot. facestate.py keeps the memory on the Python
# side instead; see its docstring.
DEFAULT_FORM = facestate.FALLBACK_FORM


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003 — silence per-request logging
        pass


def _serve(directory: Path) -> int:
    """Serve ``directory`` on a loopback port; returns the port.

    Daemon thread: the client exits without waiting on it.
    """
    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True,
                     name="ghost-face-http").start()
    return port


class WebFaceWidget(QWidget):
    """Duck-typed drop-in for the existing face widgets.

    Implements the same surface the client already calls — ``set_mood``,
    ``wake``, ``startle``, ``pulse``, ``feed_audio`` — so it can be added to
    the face QStackedWidget with no changes at the call sites.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = None
        self._ready = False
        # Three states the client re-asserts often (the frame-rate policy runs
        # every few seconds), so only a CHANGE crosses into the page. `_want`
        # is what the client last asked for, `_sent` what the page was told —
        # kept apart because a call made before the page exists goes nowhere
        # (`window.ghostFace &&` is false), and recording it as sent meant an
        # early "busy" or "rate" was never repeated.
        self._want = {"rate": 0, "busy": False, "gaze": False}
        self._sent = {"rate": 0, "busy": False, "gaze": False}
        self._polls = 0
        self._errors_seen = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not WEBENGINE_AVAILABLE:
            logger.warning("QtWebEngine is not installed; the Ghost face will "
                           "be blank — switch faces with the face button.")
            return
        try:
            port = _serve(FACE_DIR)
            self._view = QWebEngineView(self)
            self._view.setZoomFactor(ZOOM)
            self._view.setUrl(QUrl(f"http://127.0.0.1:{port}/face.html"))
            layout.addWidget(self._view)
            # The face reports readiness itself; poll briefly rather than
            # guessing a fixed delay, since shader compile time varies.
            self._poll = QTimer(self)
            self._poll.timeout.connect(self._check_ready)
            self._poll.start(500)
        except Exception as exc:  # noqa: BLE001
            # No QtWebEngine (or no GL): the client must still start. The old
            # QPainter faces remain in the stack, so the operator just switches
            # to one of those.
            logger.warning("web face unavailable (%s); falling back to a blank "
                           "widget — use the face switch button", exc)

    # ── readiness ────────────────────────────────────────────────────────
    # How long to keep asking before saying, once, that the face never came up.
    READY_POLLS = 120          # x 500 ms

    def _check_ready(self):
        if not self._view:
            self._poll.stop()
            return

        def _cb(raw):
            try:
                state = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                state = {}
            # A face that failed to start is otherwise SILENT: the page stays
            # black, every call is queued forever, and nothing on the Python
            # side ever read `window.__faceErrors`. deploy.sh greps for this.
            errors = state.get("errors") or []
            for err in errors[self._errors_seen:]:
                print(f"[face] ERROR {err}", flush=True)
            self._errors_seen = len(errors)
            # `not self._ready` is load-bearing: runJavaScript is ASYNC, so the
            # 500 ms poll can fire again before the first answer comes back and
            # a second callback would apply (and log) the form twice. The
            # doubled `[face]` line is what revealed it.
            if state.get("ready") and not self._ready:
                self._ready = True
                self._poll.stop()
                # Resolved HERE, not at import: the operator may have cycled
                # the form during a previous run seconds ago, and this is the
                # last moment before the face is visible.
                form = facestate.startup_form(FACE_DIR)
                # Printed, not just applied: which form the face opened on —
                # and whether it came from the remembered choice — is otherwise
                # only observable by LOOKING at the panel, which is no help
                # over ssh (and none at all when the screen is blanked).
                print(f"[face] opening on {form!r} "
                      f"(remembered={facestate.load_form(FACE_DIR)!r})",
                      flush=True)
                self.set_form(form)
                self._sync()

        self._polls += 1
        if self._polls == self.READY_POLLS and not self._ready:
            print(f"[face] NOT READY after {self.READY_POLLS // 2} s — the face is "
                  f"blank (panel asleep, or the page failed; see [face] ERROR above)",
                  flush=True)
        self._view.page().runJavaScript(
            "JSON.stringify({ready: !!window.__faceReady, "
            "errors: window.__faceErrors || []})", _cb)

    def _js(self, script: str):
        """Fire-and-forget JS. The harness queues calls made before init, so
        an early mood change is not lost."""
        if self._view is None:
            return
        try:
            self._view.page().runJavaScript(script)
        except Exception:  # noqa: BLE001 — the face must never break the UI
            pass

    # ── the face API the client already speaks ───────────────────────────
    def set_mood(self, mood: str):
        self._js(f"window.ghostFace && ghostFace.mood({mood!r})")

    def wake(self):
        self._js("window.ghostFace && ghostFace.wake()")

    def startle(self):
        self._js("window.ghostFace && ghostFace.startle()")

    def pulse(self, *_args, **_kwargs):
        # The QPainter faces take an optional colour; the web face derives its
        # own from the thermal palette, so extra args are accepted and ignored.
        self._js("window.ghostFace && ghostFace.pulse()")

    def feed_audio(self, level=0.0):
        try:
            lvl = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            lvl = 0.0
        self._js(f"window.ghostFace && ghostFace.audio({lvl})")

    # ── the signal layer (2026-10-01) ────────────────────────────────────
    # What the browser's face has reacted to since 2026-09-11. Arguments go
    # through json.dumps, not repr(): an error message is arbitrary text, and
    # Python's repr of a string is not always a JavaScript literal.
    def _call(self, op: str, *args):
        self._js(f"window.ghostFace && ghostFace.{op}("
                 f"{', '.join(json.dumps(a) for a in args)})")

    def set_phase(self, name):
        self._call("phase", name)

    def note_tool(self):
        self._call("tool")

    def note_recall(self):
        self._call("recall")

    def note_verdict(self, kind: str):
        self._call("verdict", kind)

    def note_error(self, message: str = ""):
        self._call("error", str(message or "")[:300])

    def _sync(self):
        """Tell the page whichever of rate / busy / gaze it has not been told."""
        if not self._ready:
            return
        for key, op in (("rate", "rate"), ("busy", "busy"), ("gaze", "gaze")):
            if self._want[key] != self._sent[key]:
                self._sent[key] = self._want[key]
                self._call(op, self._want[key])
                if key == "rate":
                    rate = self._want[key]
                    print(f"[face] rate → "
                          f"{'paused' if rate < 0 else ('full' if rate == 0 else f'{rate} fps')}",
                          flush=True)

    def set_background_busy(self, busy: bool):
        self._want["busy"] = bool(busy)
        self._sync()

    def set_gaze(self, active: bool):
        self._want["gaze"] = bool(active)
        self._sync()

    def set_rate(self, rate: int):
        """0 = full rate, n > 0 = at most n fps, n < 0 = paused."""
        self._want["rate"] = int(rate)
        self._sync()

    # ── extras specific to this face ─────────────────────────────────────
    def set_form(self, name: str):
        self._js(f"window.ghostFace && ghostFace.form({name!r})")

    def cycle_form(self, on_name=None):
        """Advance to the next form and REMEMBER it for the next start.

        The name has to be read back from the face rather than tracked here:
        the JS owns the FORMS order, and duplicating it on this side is exactly
        the kind of second copy that drifts the first time a form is added.
        ``on_name`` (optional) receives the new form's name — handy for a
        button tooltip.
        """
        self._js("window.ghostFace && ghostFace.cycle()")
        if self._view is None:
            return

        def _got(name):
            # Persist unconditionally, not only when a caller wants the name:
            # the ◈ press IS the operator choosing a face, and it is the only
            # thing this memory exists to capture.
            saved = facestate.save_form(name)
            print(f"[face] cycled to {name!r} (remembered={saved})", flush=True)
            if on_name:
                on_name(name)

        QTimer.singleShot(120, lambda: self._view.page().runJavaScript(
            "window.__face ? window.__face.getForm() : ''", _got))
