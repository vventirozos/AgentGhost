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
import logging
import os
import socket
import threading
from pathlib import Path

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtWidgets import QWidget, QVBoxLayout

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
DEFAULT_FORM = os.environ.get("GHOST_FACE_FORM", "vortex")


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
    def _check_ready(self):
        if not self._view:
            self._poll.stop()
            return

        def _cb(val):
            if val:
                self._ready = True
                self._poll.stop()
                self.set_form(DEFAULT_FORM)

        self._view.page().runJavaScript("!!window.__faceReady", _cb)

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

    # ── extras specific to this face ─────────────────────────────────────
    def set_form(self, name: str):
        self._js(f"window.ghostFace && ghostFace.form({name!r})")

    def cycle_form(self, on_name=None):
        """Advance to the next form. ``on_name`` (optional) receives the new
        form's name asynchronously — handy for a button tooltip."""
        self._js("window.ghostFace && ghostFace.cycle()")
        if on_name and self._view is not None:
            QTimer.singleShot(120, lambda: self._view.page().runJavaScript(
                "window.__face ? window.__face.getForm() : ''", on_name))

    def set_rendering(self, active: bool):
        """Pause/resume the render loop.

        A QStackedWidget HIDES pages, it does not stop their timers — a
        documented trap with the existing faces, and far more expensive here
        where the hidden page is a whole browser compositing WebGL. Hidden
        pages are throttled to a stopped animation loop instead.
        """
        if self._view is None:
            return
        # `visible` on the page drives Chromium's own rAF throttling.
        try:
            self._view.page().setLifecycleState(
                self._view.page().LifecycleState.Active if active
                else self._view.page().LifecycleState.Frozen)
        except Exception:  # noqa: BLE001 — older Qt lacks lifecycle control
            self._js("window.__face && window.__face.setWorkingState(false)")
