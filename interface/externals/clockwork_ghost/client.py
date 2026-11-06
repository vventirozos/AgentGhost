import sys
import os
import socket
import asyncio
import base64
import json
import httpx
import re
import datetime
import subprocess
import time
import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QSize, QTimer
from PyQt6.QtGui import QPixmap, QFont, QShortcut, QKeySequence, QImage, QCursor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QTextBrowser, QLineEdit, QDialog, QLabel, QPushButton, QFileDialog, QStackedWidget
)
import qasync

from webface import WebFaceWidget
from chatlog import ChatLog
from turnstatus import (
    TurnTicker, caption_html as _caption_html, log_ws_url, stream_log_lines,
    face_signals_for_ticker,
)
import agentapi
import icons
import commands
import devstatus
from markup import (
    escape_user, render_reply, reply_images, transcript_items,
)
from speech import SpeechChunker, chime_wav, clock_label, should_chime

audio_queue = asyncio.Queue()
playback_queue = asyncio.Queue()
# Bumped by MainWindow._silence(): a sentence whose audio was requested before
# a silence is stale when it arrives, and is dropped rather than played.
_speech_epoch = 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# The agent itself (chat, sessions, cancel, feedback, notifications). Plain
# HTTP inside the tailnet, as it always was — only the voice and log channels
# go through the interface's TLS port.
AGENT_BASE = os.environ.get("GHOST_AGENT_BASE", "http://eva:8000").rstrip("/")

# A voice transcript sits in the input for this long before it sends itself,
# so a misheard word can be fixed (any key cancels the countdown). 0 sends at
# once.
STT_REVIEW_S = _env_float("GHOST_STT_REVIEW_S", 2.5)
# A recording that was never stopped (Shift+Esc is a TOGGLE) ends itself here.
PTT_MAX_S = _env_float("GHOST_PTT_MAX_S", 120)
# Shift+Esc held at least this long is push-to-talk: letting go sends. A
# shorter press is a tap, and toggles (press again to send).
PTT_HOLD_S = _env_float("GHOST_PTT_HOLD_S", 0.4)
# A recording shorter than this is a slip of the key, not speech: not uploaded.
PTT_MIN_S = 0.3
# Chime when a reply lands after at least this long. 0 disables the chime.
CHIME_AFTER_S = _env_float("GHOST_CHIME_AFTER_S", 20)
# Seconds between polls: agent reachability + who holds the turn lock, and
# the agent's pending notifications. 0 disables a poll.
LINK_POLL_S = _env_float("GHOST_LINK_POLL_S", 15)
NOTIFY_POLL_S = _env_float("GHOST_NOTIFY_POLL_S", 60)
MOOD_POLL_S = _env_float("GHOST_MOOD_POLL_S", 25)
# Camera capture size. It was forced to 1920x1080@60 with a 16 ms timer: a
# 6 MB frame decoded and colour-converted on the UI thread, on a CM4, up to
# 60 times a second, to show a 640x480 preview. 720p is ample for the vision
# model and for framing a shot.
CAM_W = int(_env_float("GHOST_CAM_W", 1280))
CAM_H = int(_env_float("GHOST_CAM_H", 720))
CAM_FPS = int(_env_float("GHOST_CAM_FPS", 30))
CAM_PREVIEW_MS = 66        # ~15 previews a second is plenty for framing a shot

INPUT_PLACEHOLDER = "Message Ghost…   /help"


def _power(argv):
    """Run a command that ends the session (shutdown, reboot).

    One function so there is exactly one place that can power the device off —
    and so `device_probe.py`, which drives the real /shutdown path on the real
    device at every deploy, can do it with GHOST_DRY_POWER=1 and get a log
    line instead of a dark screen.
    """
    if os.environ.get("GHOST_DRY_POWER") == "1":
        print(f"[power] DRY RUN: {' '.join(argv)}", flush=True)
        return
    subprocess.Popen(argv)

# ── Voice endpoints (repointed 2026-08-02) ──────────────────────────────
# WAS `http://192.168.0.24:8000/{tts,stt}` — a Raspberry-Pi voice server
# (Whisper + Piper) that NO LONGER EXISTS. Both calls had been failing
# silently against a dead LAN host, which is why voice on this device was
# "unused". Voice now runs on eva itself: STT = ffmpeg + the Gemma 4 audio
# node, TTS = the macOS speech synthesiser (see interface/voice.py).
#
# Two things differ from the old Pi server and both are load-bearing:
#   1. These endpoints live on the INTERFACE (port 8080, TLS), not the agent
#      (port 8000, plain HTTP) that chat uses — so the URLs carry https and
#      their own port.
#   2. They require `X-Ghost-Key`, exactly like /api/chat already does.
# TLS VERIFICATION IS ON as of 2026-08-15, and the host default is the FQDN.
# It was off because the interface served a SELF-SIGNED cert (CN=localhost,
# no subjectAltName) that nothing could verify — so this client accepted ANY
# certificate on both TLS channels (voice AND the log websocket). Inside the
# tailnet that is a small hole, but it is a real one: WireGuard authenticates
# the PEERS, it does not stop a compromised tailnet node from answering on
# :8080 and collecting the X-Ghost-Key these calls send.
#
# The interface now serves a genuine Let's Encrypt cert issued by Tailscale
# for `eva.taila2b1d.ts.net` (see bin/start-ghost-client.sh). Verification
# only works against the name in the SAN, hence the FQDN default — the short
# `eva` would fail hostname matching even though it reaches the same box.
#
# ⚠ IF VOICE OR THE TURN-STATUS CAPTION GOES QUIET AFTER A DEPLOY, this is
# the first thing to check: the FQDN must resolve ON THE DEVICE (Tailscale
# MagicDNS provides it; `getent hosts eva.taila2b1d.ts.net` confirms it).
# Both switches are env-overridable, so the escape hatch needs no redeploy —
# add to ~/bin/launch_ghost.sh before the python3 line:
#     export GHOST_HOST=eva
#     export GHOST_VOICE_VERIFY_TLS=0
# Chat is unaffected either way: it talks plain HTTP to eva:8000.
GHOST_HOST = os.environ.get("GHOST_HOST", "eva.taila2b1d.ts.net")
VOICE_BASE_URL = os.environ.get("GHOST_VOICE_BASE", f"https://{GHOST_HOST}:8080")
TTS_SERVER_URL = f"{VOICE_BASE_URL}/api/tts"
STT_SERVER_URL = f"{VOICE_BASE_URL}/api/stt"
VOICE_VERIFY_TLS = os.environ.get("GHOST_VOICE_VERIFY_TLS", "1").lower() in ("1", "true", "yes")

# One line at startup, into /tmp/ghost_ui.log — the file deploy.sh tails when
# a deploy looks wrong. This exists because the voice endpoints spent weeks
# "unused" while silently failing against a host that no longer existed, and
# turning verification ON adds exactly one new way to reproduce that: an FQDN
# that does not resolve on the device. Name the failure at boot instead of
# leaving a quiet caption and dead voice to be discovered later.
print(f"[tls] voice+log host={GHOST_HOST} verify="
      f"{'ON' if VOICE_VERIFY_TLS else 'OFF'}", flush=True)
if VOICE_VERIFY_TLS:
    try:
        socket.getaddrinfo(GHOST_HOST, 8080)
    except OSError as _dns_err:
        print(f"[tls] WARNING: {GHOST_HOST} does not resolve here "
              f"({_dns_err}) — voice and the turn-status caption will be "
              f"DEAD (chat still works, it uses eva:8000 over plain HTTP). "
              f"Fall back without a redeploy by adding to "
              f"~/bin/launch_ghost.sh:  export GHOST_HOST=eva ; "
              f"export GHOST_VOICE_VERIFY_TLS=0", flush=True)

# ── Live turn status (2026-08-03) ───────────────────────────────────────────
# The waiting bubble narrates the agent's CURRENT step instead of showing a
# static "cogitating". The step names come from the interface's log broadcast —
# same host, port and (since 2026-08-15) VERIFIED cert as the voice endpoints
# above, so the same TLS switch applies. See turnstatus.py for why the chat stream cannot supply
# this itself.
LOG_WS_URL = os.environ.get("GHOST_LOG_WS")


def _resolve_ghost_api_key() -> str:
    """Agent API key (X-Ghost-Key). The agent enforces a real key since
    2026-07-13 — the old hardcoded placeholder only worked because auth
    used to be disabled. Resolution order: GHOST_API_KEY env, then
    ~/.ghost_api_key on the device, then a .ghost_api_key next to this
    file. Deploy: copy the key file from eva
    (~/Data/AI/.ghost_api_key) to the uConsole as ~/.ghost_api_key
    (chmod 600)."""
    env = os.environ.get("GHOST_API_KEY")
    if env:
        return env
    for path in (
        os.path.expanduser("~/.ghost_api_key"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ghost_api_key"),
    ):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    return ""


GHOST_API_KEY = _resolve_ghost_api_key()


# ============================================================================
# THEME — central palette + stylesheet builders
# ============================================================================

# §4KP: the agent's `ghost.reasoning_unparsed` frame says this reply's opening
# may be reasoning that ends in an orphan </think> (the model server did not
# route it to the reasoning channel). Printed text cannot be taken back, so
# the reply is held until complete and stripped here — the mirror of
# agent._strip_orphan_think_close (a parity test pins the two together).
_ORPHAN_CLOSE_RE = re.compile(r'\A.*?(?:\A|\n)</think(?:ing)?[ \t]*>[ \t]*\r?\n[ \t]*\r?\n',
                              re.DOTALL | re.IGNORECASE)
_FENCE_SPAN_RES = (re.compile(r"```.*?```", re.DOTALL), re.compile(r"~~~.*?~~~", re.DOTALL))


def strip_orphan_think_close(text):
    if not isinstance(text, str) or "</think" not in text.lower() or text.lstrip().startswith("{"):
        return text
    fences = []

    def _shield(m):
        fences.append(m.group(0))
        return f"\x00FENCE{len(fences) - 1}\x00"
    shielded = text
    for rx in _FENCE_SPAN_RES:
        shielded = rx.sub(_shield, shielded)
    out = _ORPHAN_CLOSE_RE.sub("", shielded, count=1)
    for i in range(len(fences) - 1, -1, -1):          # outer shields hold inner ones
        out = out.replace(f"\x00FENCE{i}\x00", fences[i])
    return out

class T:
    """Glass UI over the live face (2026-08-02 restyle).

    Nothing is opaque any more: the face fills the window and every panel is
    tinted glass on top of it, so the thermal palette reads through the whole
    interface instead of being boxed into one half.

    The old scheme was navy panels with a CYAN accent (#7be0ff). Cyan fights
    the face directly — the face's ring runs blue → violet → crimson with no
    green channel to speak of, so a cold cyan chrome sat outside that range and
    made the two look like different applications. The accent is now violet,
    lifted from the middle of the face's own palette, and the warm user colour
    sits with its crimson core. No hard fills, hairline borders, larger radii.
    """

    # Panel fills. Qt widgets cannot do backdrop-blur, so readability comes
    # from the tint alone — hence a heavier fill behind long-form text than
    # behind chips, rather than the uniform low alpha the web UI can afford.
    GLASS       = "rgba(9, 11, 22, 0.46)"     # chat surface (text legibility)
    GLASS_SOFT  = "rgba(255, 255, 255, 0.07)" # inputs, chips
    GLASS_HOT   = "rgba(255, 255, 255, 0.16)" # pressed
    HAIRLINE    = "rgba(255, 255, 255, 0.08)"
    HAIRLINE_HOT = "rgba(201, 166, 255, 0.55)"

    TEXT        = "#ecebf6"
    TEXT_DIM    = "rgba(236, 235, 246, 0.52)"
    TEXT_SOFT   = "rgba(236, 235, 246, 0.82)"  # controls
    ACCENT_FILL = "rgba(201, 166, 255, 0.18)"
    USER        = "#ffc08a"   # warm sand — sits with the face's crimson core
    ASSISTANT   = "#e9e6ff"
    ACCENT      = "#c9a6ff"   # violet, from the face's mid palette
    ACCENT_WARM = "#ffc08a"
    OK          = "#9fe3b8"
    DANGER      = "#ff7b91"
    REC         = "#ff5470"
    SCROLL      = "rgba(255, 255, 255, 0.12)"
    SCROLL_HOT  = "rgba(201, 166, 255, 0.45)"

    # Kept as aliases so any straggling reference still resolves.
    BG          = "transparent"
    BG_PANEL    = GLASS
    BG_INPUT    = GLASS_SOFT
    BORDER      = HAIRLINE
    BORDER_HOT  = HAIRLINE_HOT

    FONT        = "'Fira Code', 'JetBrains Mono', 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', monospace"
    # The interface is sans (operator, 2026-11-06: "modern"); FONT stays mono, for code.
    SANS        = "'Noto Sans', 'Cantarell', 'Liberation Sans', 'Noto Color Emoji', sans-serif"


_CHIP_SHAPE = f"""
            border-radius: 18px;
            padding: 8px 15px;
            font-family: {T.SANS};
            font-size: 17px;
            font-weight: 500;
"""


def set_chip(btn, icon_name, text, color=T.TEXT_SOFT):
    """A control's icon and word. The colour is the icon's: the text takes the style's."""
    btn.setIcon(icons.icon(icon_name, color))
    btn.setIconSize(QSize(20, 20))
    btn.setText(f" {text}")


def chip_style(fg=T.TEXT_SOFT, hover=T.GLASS_HOT):
    return f"""
        QPushButton {{
            background-color: {T.GLASS_SOFT};
            color: {fg};
            border: 1px solid transparent;
            {_CHIP_SHAPE}
        }}
        QPushButton:hover {{
            border: 1px solid {T.HAIRLINE_HOT};
        }}
        QPushButton:pressed {{
            background-color: {hover};
            color: {T.TEXT};
        }}
        QPushButton:disabled {{
            background-color: rgba(255, 255, 255, 0.03);
            color: rgba(236, 235, 246, 0.26);
        }}
    """


def chip_style_on(fg=None, fill="rgba(159, 227, 184, 0.16)"):
    """A control whose function is ON (spoken replies, a latched rating).

    A FILL means on, and nothing else does: on a device whose pointer is a
    trackball that stays where it was left, a hover fill looked switched on.
    """
    fg = fg or T.OK
    return f"""
        QPushButton {{
            background-color: {fill};
            color: {fg};
            border: 1px solid transparent;
            {_CHIP_SHAPE}
        }}
    """


def chip_style_hot(fg, border):
    """Armed state (recording): only the colour changes, not the shape."""
    return f"""
        QPushButton {{
            background-color: rgba(255, 84, 112, 0.22);
            color: {fg};
            border: 1px solid {border};
            {_CHIP_SHAPE}
        }}
    """


INPUT_STYLE = f"""
    QLineEdit {{
        background-color: {T.GLASS_SOFT};
        color: {T.TEXT};
        border: 1px solid transparent;
        border-radius: 20px;
        padding: 10px 18px;
        font-family: {T.SANS};
        font-size: 20px;
        selection-background-color: {T.HAIRLINE_HOT};
        selection-color: {T.TEXT};
    }}
    QLineEdit:focus {{
        border: 1px solid {T.HAIRLINE_HOT};
    }}
"""

DIALOG_STYLE = f"""
    QDialog {{
        background-color: {T.BG_PANEL};
        border: 1px solid {T.BORDER_HOT};
        border-radius: 14px;
    }}
"""

# Chat bubbles moved to chatlog.py (2026-08-02, second pass): they are
# real QLabel widgets now, so they get true rounded/notched corners and
# hug their content — neither of which QTextDocument can do.

def caption_html(ticker):
    """Waiting-bubble caption, in this client's palette.

    The building (field order, escaping, one-line elide) lives in turnstatus.py
    where it is Qt-free and unit-tested; this only supplies the theme.
    """
    return _caption_html(ticker, dim=T.TEXT_DIM, mono=T.FONT)


NOTE_DIM = f"<div style='color:{T.TEXT_DIM};'><i>"
NOTE_OK = f"<div style='color:{T.OK};'><i>"
NOTE_WARN = f"<div style='color:{T.ACCENT_WARM};'><i>"
NOTE_ERR = f"<div style='color:{T.DANGER};'><i>"


class ImageViewer(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        # Not always-on-top (2026-10-17): a dialog with the kiosk as its parent is
        # stacked above it by labwc anyway (tested on the device); on-top made it a
        # peer of the kiosk, free to end up under it and block input invisibly.
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setStyleSheet(DIALOG_STYLE)

        top_bar = QHBoxLayout()
        self.close_btn = QPushButton()
        set_chip(self.close_btn, "x", "Close")
        self.zoom_in_btn = QPushButton()
        set_chip(self.zoom_in_btn, "zoom-in", "Zoom in")
        self.zoom_out_btn = QPushButton()
        set_chip(self.zoom_out_btn, "zoom-out", "Zoom out")

        for btn in (self.close_btn, self.zoom_in_btn, self.zoom_out_btn):
            btn.setStyleSheet(chip_style())
        
        top_bar.addStretch()
        top_bar.addWidget(self.zoom_out_btn)
        top_bar.addWidget(self.zoom_in_btn)
        top_bar.addWidget(self.close_btn)
        
        self.close_btn.clicked.connect(self.close)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_pixmap = pixmap
        self.scale_factor = 1.0
        
        self.update_image()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.addLayout(top_bar)
        layout.addWidget(self.lbl)
        
        scaled = self.original_pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.resize(scaled.width() + 60, scaled.height() + 100)

    def update_image(self):
        w = int(800 * self.scale_factor)
        h = int(600 * self.scale_factor)
        scaled = self.original_pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(scaled)

    def zoom_in(self):
        self.scale_factor *= 1.25
        self.update_image()

    def zoom_out(self):
        self.scale_factor /= 1.25
        self.update_image()


class CameraPreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Not always-on-top (2026-10-17): a dialog with the kiosk as its parent is
        # stacked above it by labwc anyway (tested on the device); on-top made it a
        # peer of the kiosk, free to end up under it and block input invisibly.
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setStyleSheet(DIALOG_STYLE)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 18, 18, 18)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("Camera")
        self.title_label.setStyleSheet(f"color: {T.TEXT}; font-family: {T.SANS}; font-size: 20px; font-weight: 600;")
        self.close_btn = QPushButton()
        set_chip(self.close_btn, "x", "Close")
        self.close_btn.setStyleSheet(chip_style())
        self.close_btn.clicked.connect(self.close_and_stop)
        top_bar.addWidget(self.title_label)
        top_bar.addStretch()
        top_bar.addWidget(self.close_btn)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet(f"background-color: #000; border: 1px solid {T.BORDER_HOT}; border-radius: 8px;")

        self.live_controls = QWidget()
        live_layout = QHBoxLayout(self.live_controls)
        live_layout.setContentsMargins(0, 8, 0, 0)
        self.capture_btn = QPushButton()
        set_chip(self.capture_btn, "camera", "Capture", T.ACCENT)
        self.capture_btn.setStyleSheet(chip_style_on(T.ACCENT, T.ACCENT_FILL))
        self.capture_btn.clicked.connect(self.take_picture)
        live_layout.addStretch()
        live_layout.addWidget(self.capture_btn)
        live_layout.addStretch()

        self.review_controls = QWidget()
        rev_layout = QHBoxLayout(self.review_controls)
        rev_layout.setContentsMargins(0, 8, 0, 0)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("annotate the capture…")
        self.prompt_input.setStyleSheet(INPUT_STYLE)

        self.upload_btn = QPushButton()
        set_chip(self.upload_btn, "send", "Send", T.ACCENT)
        self.upload_btn.setStyleSheet(chip_style_on(T.ACCENT, T.ACCENT_FILL))
        self.upload_btn.clicked.connect(self.upload_picture)

        self.download_btn = QPushButton()
        set_chip(self.download_btn, "download", "Save")
        self.download_btn.setStyleSheet(chip_style())
        self.download_btn.clicked.connect(self.download_picture)
        
        rev_layout.addWidget(self.prompt_input, 1)
        rev_layout.addWidget(self.download_btn)
        rev_layout.addWidget(self.upload_btn)
        self.review_controls.hide()
        
        self.layout.addLayout(top_bar)
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.live_controls)
        self.layout.addWidget(self.review_controls)
        
        self.resize(700, 600)

        self.cap = cv2.VideoCapture(0)

        # MJPG so USB bandwidth allows a usable frame rate at this size; the
        # size itself is modest on purpose (see CAM_W above).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.result_data = None
        self._opened_at = time.monotonic()
        self._released = False

        # No camera is an ordinary state on this device (the BRIO is on a USB
        # cable): say so. It used to be a black rectangle and a CAPTURE button
        # that did nothing.
        if not self.cap.isOpened():
            self._no_camera("no camera found — is it plugged in?")
        else:
            self.timer.start(CAM_PREVIEW_MS)

        self.snap_state = 0
        self.snap_shortcut = QShortcut(QKeySequence("Ctrl+Escape"), self)
        self.snap_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.snap_shortcut.activated.connect(self.handle_snap_shortcut)

    def _no_camera(self, message):
        self.timer.stop()
        self.capture_btn.setEnabled(False)
        self.video_label.setText(message)
        self.video_label.setStyleSheet(
            f"background-color: #000; color: {T.TEXT_DIM}; font-family: {T.FONT};"
            f" font-size: 18px; border: 1px solid {T.BORDER_HOT}; border-radius: 8px;")

    def update_frame(self):
        if not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # Preview remains safely scaled down for UI
            self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))
        elif self.current_frame is None and time.monotonic() - self._opened_at > 3.0:
            # Opened, but never delivered a frame (busy device, wrong node).
            self._no_camera("the camera opened but sent no picture")

    def handle_snap_shortcut(self):
        if self.snap_state == 0:
            if self.take_picture():
                self.snap_state = 1
        elif self.snap_state == 1:
            self.upload_picture()

    def take_picture(self):
        if self.current_frame is None:
            return False          # nothing to freeze on yet
        self.timer.stop()
        self.live_controls.hide()
        self.review_controls.show()
        self.prompt_input.setFocus()
        return True

    def download_picture(self):
        if self.current_frame is not None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Picture", "/home/vasilis/snapshot.jpg", "Images (*.jpg)")
            if filename:
                cv2.imwrite(filename, self.current_frame)

    def upload_picture(self):
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            if ret:
                b64_str = base64.b64encode(buffer).decode('utf-8')
                self.result_data = (b64_str, self.prompt_input.text().strip())
                self.accept()

    def _release(self):
        if self._released:
            return
        self._released = True
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()

    def done(self, result):
        # EVERY way out of a QDialog ends here — accept, the close chip, and
        # the Escape key, which calls reject() directly and so used to skip
        # close_and_stop(): the timer kept reading and the camera stayed
        # claimed (its light on) until the client was restarted.
        self._release()
        super().done(result)

    def close_and_stop(self):
        self.reject()

class MainWindow(QWidget):
    update_chat_signal = pyqtSignal(str, str)
    show_image_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_response_text = ""
        self.shown_images = set()
        
        # Context and History
        self.conversation_history = []
        self.input_history = []
        self.history_index = -1
        self.is_recording = False
        self._ptt_press_open = False     # the Shift+Esc that started it is still down
        self._forwarding_key = False     # keyPressEvent is handing a key to the input

        # ── the turn in flight ───────────────────────────────────────────
        # ONE at a time. There was no guard: Enter during a reply started a
        # second stream that shared `current_response_text` and the open
        # bubble with the first, and the two interleaved.
        self._turn_task = None
        self._turn_rid = None          # this turn's request id (minted here)
        self._turn_verdict = None      # the verifier's verdict, if it logged one
        self._stop_asked = 0           # 0 none, 1 asked, 2 forced
        self._stop_took = False        # the agent accepted the stop (or it was forced)
        self._stop_pending = False     # a stop was sent and not yet answered
        self._busy_noted = False
        # The last finished reply's id, when it can take a rating.
        self._last_reply_rid = None
        self._rated = None             # "positive" / "negative" once rated

        # ── the conversation's durable id (see agentapi) ─────────────────
        self.session_id = agentapi.load_session_id() or agentapi.new_session_id()
        agentapi.save_session_id(self.session_id)
        self._sessions_listed = []     # what /sessions last showed, for /open

        self.confirmer = commands.Confirmer()
        self.speech = SpeechChunker()
        self._voice_fault_shown = False
        self._agent_ok = None          # None until the first poll answers
        self._last_input_at = time.monotonic()
        self._last_cursor = None
        self._rec_started = 0.0
        self._rec_path = None          # the file THIS recording is written to
        self._rec_seq = 0
        self._transcribing = False
        self._review_left = 0.0
        self._on_battery = False

        self.initUI()
        self.update_chat_signal.connect(self._update_chat)
        self.show_image_signal.connect(self._show_image_popup)

        # The waiting bubble's caption: elapsed clock + what the agent is doing
        # right now, fed by the log socket (see turnstatus.py). The timer only
        # advances the CLOCK — the description changes when a log line arrives.
        self.ticker = TurnTicker()
        self.ticker.on_step = self._on_turn_step
        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self._animate_thinking)
        self.is_thinking = False

        # Monitor TTS queue drain to return faces to idle after speak mode
        self.tts_monitor = QTimer(self)
        self.tts_monitor.timeout.connect(self._check_tts_done)
        self.tts_monitor.start(500)

        # Recording clock (the chip shows elapsed time, and a recording that
        # was never stopped ends itself) and the voice-review countdown.
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._tick_recording)
        self.review_timer = QTimer(self)
        self.review_timer.timeout.connect(self._tick_review)

        # Face frame rate follows activity (devstatus.face_rate). 3 s is the
        # resolution of "idle for 45 s" that matters; waking is immediate
        # because every input path calls _note_activity() itself.
        self.face_timer = QTimer(self)
        self.face_timer.timeout.connect(self._apply_face_rate)
        self.face_timer.start(3000)

    def initUI(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        win_w, win_h = screen_geometry.width(), screen_geometry.height()
        self.setFixedSize(win_w, win_h)
        self.move(0, 0)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        # Only the window itself gets a fill, and only so there is no flash of
        # nothing before the face paints — the face covers it entirely.
        self.setObjectName("root")
        self.setStyleSheet("QWidget#root { background-color: #05060c; }")

        # ── The face is the BACKGROUND of the whole window (2026-08-02) ────
        # It used to sit in the right half with opaque panels beside it. Now
        # every panel is glass and floats over it, so the thermal palette
        # reads through the entire interface.
        #
        # Explicit geometry + raise_() instead of a layout: this is the exact
        # arrangement proven to composite correctly over a GPU-backed
        # QWebEngineView on this device. The window is a fixed size (it is a
        # frameless full-screen kiosk), so nothing has to react to a resize.
        self.web_face = WebFaceWidget(self)
        self.web_face.setGeometry(0, 0, win_w, win_h)
        # The face is decorative — it must never take focus: key presses would
        # go to the web view instead of the input.
        self.web_face.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.faces = (self.web_face,)
        # Face state the client owns (the face itself is async JS now).
        self._face_mood = "idle"

        # Everything else lives on a transparent sheet ON TOP of the face.
        self.overlay = QWidget(self)
        self.overlay.setGeometry(0, 0, win_w, win_h)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # ONE full-width column, not a left/right split (2026-08-02). The
        # split existed to keep the transcript clear of the face; now the face
        # is behind everything and the messages are bubbles that place
        # themselves — operator right, agent left — so a container column would
        # only reserve dead space.
        main_layout = QVBoxLayout(self.overlay)
        main_layout.setContentsMargins(26, 14, 26, 12)
        main_layout.setSpacing(10)

        left_widget = QWidget()
        # Holds the ONLY stretch factor in the main column: hiding it collapses
        # the bottom bar. Toggle self.chat_display instead.
        self.left_widget = left_widget
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Real widgets, not a rich-text document: QTextDocument cannot do
        # border-radius, and its table cells take a fixed percentage width
        # instead of hugging short messages. See chatlog.py.
        self.chat_display = ChatLog(T)
        self.chat_display.link_clicked.connect(self.handle_link_clicked)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(INPUT_PLACEHOLDER)
        self.text_input.setStyleSheet(INPUT_STYLE)
        self.text_input.returnPressed.connect(self.handle_input)
        self.text_input.installEventFilter(self)
        # textEdited fires for the OPERATOR's edits only (not setText), which
        # is exactly what cancels a voice-review countdown; textChanged drives
        # the face's lean toward the composer.
        self.text_input.textEdited.connect(self._on_text_edited)
        self.text_input.textChanged.connect(
            lambda t: self.web_face.set_gaze(bool(t.strip())))

        # Only the transcript lives in this container now; the input moved to
        # the bottom bar so it can share that row with the action chips.
        left_layout.addWidget(self.chat_display)

        # Every control carries an icon and a word: a tooltip needs a hover, on a
        # device with a trackball.
        self.animations_btn = QPushButton()
        self.animations_btn.setFixedSize(34, 34)
        self.animations_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 17px; }}
            QPushButton:hover {{ border: 1px solid {T.HAIRLINE}; }}
            QPushButton:pressed {{ background-color: {T.GLASS_SOFT}; }}
        """)
        self._show_animations(self.web_face.animations_on_at_start())
        self.animations_btn.clicked.connect(self.toggle_animations)

        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_layout.setSpacing(8)

        # Rate the last reply.
        self.good_btn = QPushButton()
        self.good_btn.setToolTip("The last reply was right (Ctrl+Up)")
        self.good_btn.clicked.connect(lambda: self.rate_last("positive"))
        self.bad_btn = QPushButton()
        self.bad_btn.setToolTip("The last reply was wrong (Ctrl+Down)")
        self.bad_btn.clicked.connect(lambda: self.rate_last("negative"))
        top_right_layout.addWidget(self.good_btn)
        top_right_layout.addWidget(self.bad_btn)
        top_right_layout.addStretch()


        # Bottom row: the input keeps its left position, the action chips and
        # the status readout keep theirs on the right — now sharing one row
        # instead of sitting in two separate columns.
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(8)
        stats_layout.addWidget(self.text_input, 1)

        # Shown only while a turn runs — the one moment it means anything.
        self.stop_btn = QPushButton()
        set_chip(self.stop_btn, "stop", "Stop", T.DANGER)
        self.stop_btn.setStyleSheet(chip_style(fg=T.DANGER))
        self.stop_btn.setToolTip("Stop the running turn (Esc)")
        self.stop_btn.clicked.connect(self.request_stop)
        self.stop_btn.hide()

        self.snap_btn = QPushButton()
        set_chip(self.snap_btn, "camera", "Camera")
        self.snap_btn.setStyleSheet(chip_style())
        self.snap_btn.clicked.connect(self.take_picture)

        self.ptt_btn = QPushButton()
        set_chip(self.ptt_btn, "mic", "Talk")
        self.ptt_btn.setStyleSheet(chip_style())
        self.ptt_btn.pressed.connect(self.start_recording)
        self.ptt_btn.released.connect(self.stop_recording)

        self.tts_btn = QPushButton()
        set_chip(self.tts_btn, "volume-x", "Voice off")
        self.tts_btn.setStyleSheet(chip_style())
        self.tts_btn.clicked.connect(self.toggle_tts)

        stats_layout.addWidget(self.stop_btn)
        stats_layout.addWidget(self.snap_btn)
        stats_layout.addWidget(self.ptt_btn)
        stats_layout.addWidget(self.tts_btn)
        stats_layout.addWidget(self.animations_btn)

        self.stats_label = QLabel("●   --%   ··:··")
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)
        self.stats_label.setToolTip("agent link · wifi · battery · time")
        self.stats_label.setStyleSheet(f"color: {T.TEXT_SOFT}; font-family: {T.SANS}; font-size: 17px; font-weight: 500; padding-left: 8px;")
        stats_layout.addWidget(self.stats_label)

        # top chips · transcript (stretch) · input + actions
        main_layout.addLayout(top_right_layout)
        main_layout.addWidget(left_widget, 1)
        main_layout.addLayout(stats_layout)

        # Keep the glass UI above the face at all times.
        self.overlay.raise_()

        # Focus text input on startup
        self.text_input.setFocus()
        self.tts_enabled = False
        
        # Start stats loop
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(5000) # Every 5s
        self.update_stats()
        self._refresh_rating_chips()

        # The Esc family (operator, 2026-10-17): Shift+Esc talks, Esc cancels,
        # Alt+Esc speaks, Ctrl+Esc looks. Both are WINDOW shortcuts, not
        # application ones: an application-wide Esc would also fire inside the
        # camera dialog, where Esc must close the dialog. The window holds the
        # face.
        self.esc_shortcut = QShortcut(QKeySequence("Shift+Escape"), self)
        self.esc_shortcut.activated.connect(self._ptt_key)

        self.tts_shortcut = QShortcut(QKeySequence("Alt+Escape"), self)
        self.tts_shortcut.activated.connect(self.toggle_tts)

        self.stop_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.stop_shortcut.activated.connect(self.cancel_key)

        self.good_shortcut = QShortcut(QKeySequence("Ctrl+Up"), self)
        self.good_shortcut.activated.connect(lambda: self.rate_last("positive"))
        self.bad_shortcut = QShortcut(QKeySequence("Ctrl+Down"), self)
        self.bad_shortcut.activated.connect(lambda: self.rate_last("negative"))

        # ⚠ ONE action per key press. A QShortcut auto-repeats while its key
        # is held, and every one of these is a TOGGLE: holding Esc — the
        # natural way to use a chip that says "PTT" — started and stopped the
        # recording some twenty-five times a second, each stop uploading a
        # file the next start was already rewriting ("STT Error: Too much
        # data for declared Content-Length", eleven in a row on the device).
        for _sc in (self.esc_shortcut, self.tts_shortcut, self.stop_shortcut,
                    self.good_shortcut, self.bad_shortcut):
            _sc.setAutoRepeat(False)

        self.snap_shortcut = QShortcut(QKeySequence("Ctrl+Escape"), self)
        self.snap_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.snap_shortcut.activated.connect(self.take_picture)
        self.snap_shortcut.setAutoRepeat(False)


    # ── conversation: sessions, history, notes ───────────────────────────
    def _note(self, text, tone=None):
        """One dim line in the transcript. `text` is plain; it is escaped."""
        self.chat_display.add(f"{tone or NOTE_DIM}{escape_user(text)}</i></div>", "system")

    def _begin_session(self, session_id, history=None):
        """Switch to `session_id` with `history` as the local conversation."""
        self.session_id = session_id
        agentapi.save_session_id(session_id)
        self.conversation_history = list(history or [])
        self.current_response_text = ""
        self._last_reply_rid = None
        self._rated = None
        self.chat_display.clear()
        self._refresh_rating_chips()

    def _render_history(self, messages):
        """Draw a stored conversation. Its images are NOT popped open again —
        restoring a session used to mean a stack of image dialogs, one per
        picture ever generated in it; each stays a tap away on its 🖼️."""
        for role, text in transcript_items(messages):
            if role == "user":
                self.chat_display.add(escape_user(text), "user")
            else:
                self.shown_images.update(reply_images(text))
                self.chat_display.add(render_reply(text), "agent")
        self.chat_display.scroll_to_end()

    async def restore_session(self):
        """At startup: bring back the conversation this device was having.

        The transcript lived only in this process, so a restart, a crash or a
        reboot lost it — while the agent had the whole thing stored under the
        session id all along.
        """
        sid = self.session_id
        async with httpx.AsyncClient(timeout=20.0) as client:
            messages, status = await agentapi.fetch_session(
                client, AGENT_BASE, GHOST_API_KEY, sid)
        # Re-checked AFTER the await: the operator may have typed, or started
        # a /new conversation, while the fetch was out — and the stored
        # conversation must not be poured into a different session.
        if (status != "ok" or not messages or sid != self.session_id
                or self.conversation_history or self._busy()):
            return
        self.conversation_history = agentapi.history_for_model(messages)
        self._render_history(messages)
        self._note("conversation restored — /new starts a fresh one")

    async def _show_sessions(self):
        async with httpx.AsyncClient(timeout=20.0) as client:
            rows = await agentapi.list_sessions(client, AGENT_BASE, GHOST_API_KEY)
        self._sessions_listed = rows
        self.chat_display.scroll_to_end()
        if not rows:
            self._note("no stored conversations (or the agent did not answer)", NOTE_WARN)
            return
        lines = []
        for i, s in enumerate(rows, 1):
            here = "  ← this one" if s.get("id") == self.session_id else ""
            title = escape_user(str(s.get("title") or "untitled")[:60])
            lines.append(f"<b>{i}</b>&nbsp; {title} "
                         f"<span style='color:{T.TEXT_DIM};'>· {s.get('message_count', 0)} msgs"
                         f" · {agentapi.age_text(s.get('updated_at'))}{here}</span>")
        self.chat_display.add("<br>".join(lines) + f"<br>{NOTE_DIM}/open N to continue one</i></div>",
                              "system")

    async def _open_session(self, arg):
        try:
            n = int(arg)
        except ValueError:
            n = 0
        if not 1 <= n <= len(self._sessions_listed):
            self._note("usage: /open N — a number from /sessions", NOTE_WARN)
            return
        row = self._sessions_listed[n - 1]
        async with httpx.AsyncClient(timeout=20.0) as client:
            messages, status = await agentapi.fetch_session(
                client, AGENT_BASE, GHOST_API_KEY, row["id"])
        if status != "ok":
            self._note(f"could not open that conversation ({status})", NOTE_ERR)
            return
        if self._busy():
            # A message was sent while the fetch was out. Switching now would
            # swap the session under the running turn.
            self._note("a turn started meanwhile — the conversation was not opened", NOTE_WARN)
            return
        self._begin_session(row["id"], agentapi.history_for_model(messages))
        self._render_history(messages)
        self._note(f"opened: {str(row.get('title') or 'untitled')[:60]}")

    def toggle_ptt(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _ptt_key(self):
        """Shift+Esc pressed. Remember that THIS press has not been let go
        yet: only its own release may end a held recording. Without that, a
        tap followed later by Esc (cancel) would have sent the recording on
        the cancel key's release."""
        self.toggle_ptt()
        self._ptt_press_open = self.is_recording

    def cancel_key(self):
        """Esc: cancel whatever is in progress — a recording (thrown away,
        not sent), else a transcript's send countdown, else the running turn."""
        self._note_activity()
        if self.is_recording:
            self.discard_recording()
            return
        if self.review_timer.isActive():
            self._cancel_review()
            self._note("not sent — Enter sends it")
            return
        self.request_stop()

    # ── activity → face frame rate ───────────────────────────────────────
    def _note_activity(self):
        """The operator did something: the face is back at full rate NOW,
        not at the next 3-second tick."""
        self._last_input_at = time.monotonic()
        self._apply_face_rate()

    def _apply_face_rate(self):
        # The pointer is the one input no key handler sees. Polled here (one
        # call every 3 s) rather than tracked by an application-wide event
        # filter, which would put a Python call on every paint and timer
        # event of a 60 fps window.
        pos = QCursor.pos()
        if self._last_cursor is not None and pos != self._last_cursor:
            self._last_input_at = time.monotonic()
        self._last_cursor = pos
        busy = (self._busy() or self.is_recording or self.review_timer.isActive()
                or self._face_mood in ("think", "speak", "listen"))
        idle_s = time.monotonic() - self._last_input_at
        self.web_face.set_rate(
            devstatus.face_rate(idle_s, busy, self._on_battery))

    def _silence(self):
        """Stop speaking now and forget what was queued to be said."""
        global _speech_epoch
        # A sentence already being fetched lands AFTER the purge below; the
        # epoch lets audio_fetch_task see that it is stale and drop it.
        _speech_epoch += 1
        while not audio_queue.empty():
            try: audio_queue.get_nowait(); audio_queue.task_done()
            except Exception: break
        while not playback_queue.empty():
            try: playback_queue.get_nowait(); playback_queue.task_done()
            except Exception: break
        self._kill_playback()

    def _kill_playback(self):
        # Its own method so device_probe.py can run the real _silence() beside
        # the live client without killing the live client's audio.
        subprocess.Popen(['pkill', 'aplay'])

    def start_recording(self):
        """Triggered when the PTT button is held down."""
        if self.is_recording:
            return
        if self._transcribing:
            self._note("still transcribing the last recording…")
            return
        self._cancel_review()
        self._note_activity()
        # Barge-in: the agent stops talking when the operator starts. It used
        # to keep speaking, straight into the microphone it was recording.
        self._silence()
        self.is_recording = True
        self._rec_started = time.monotonic()
        self.ptt_btn.setStyleSheet(chip_style_hot(T.REC, T.REC))
        set_chip(self.ptt_btn, "mic", "0:00", T.REC)
        self.rec_timer.start(500)
        self.set_face_mood("listen")
        # Its OWN file per recording. They all used to be written to
        # /tmp/ghost_stt.wav, so an upload still reading the last one saw the
        # next one being written underneath it.
        self._rec_seq += 1
        self._rec_path = f"/tmp/ghost_stt_{os.getpid()}_{self._rec_seq}.wav"
        self._start_arecord(self._rec_path)

    def _start_arecord(self, path):
        # Kill any lingering recording processes just in case
        subprocess.Popen(['pkill', 'arecord']).wait()
        # Start recording 16kHz mono audio to a temporary file
        self.record_proc = subprocess.Popen(
            ['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', path],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )

    def _stop_arecord(self):
        if hasattr(self, 'record_proc') and self.record_proc:
            self.record_proc.terminate()
            self.record_proc.wait()

    def _tick_recording(self):
        if not self.is_recording:
            self.rec_timer.stop()
            return
        elapsed = time.monotonic() - self._rec_started
        self.ptt_btn.setText(f" {clock_label(elapsed)}")
        if PTT_MAX_S > 0 and elapsed >= PTT_MAX_S:
            # Shift+Esc is a toggle, so a recording can be left running by accident.
            self._note(f"recording stopped at the {int(PTT_MAX_S)} s limit")
            self.stop_recording()

    def discard_recording(self):
        """End the recording without sending it (Esc)."""
        if not self.is_recording:
            return
        self.is_recording = False
        self._ptt_press_open = False
        self.rec_timer.stop()
        self.ptt_btn.setStyleSheet(chip_style())
        set_chip(self.ptt_btn, "mic", "Talk")
        self._stop_arecord()
        path, self._rec_path = self._rec_path, None
        self._discard_recording(path)
        self.set_face_mood("idle")
        self._note("recording discarded")

    def stop_recording(self):
        """Triggered when the PTT button is released."""
        if not self.is_recording:
            return
        self.is_recording = False
        self._ptt_press_open = False
        self.rec_timer.stop()
        self.ptt_btn.setStyleSheet(chip_style())
        set_chip(self.ptt_btn, "mic", "Talk")
        self._stop_arecord()
        path, self._rec_path = self._rec_path, None
        if time.monotonic() - self._rec_started < PTT_MIN_S:
            # A slip of the key, not speech: nothing worth a transcription.
            self._discard_recording(path)
            self.set_face_mood("idle")
            return
        # Trigger the async upload task
        asyncio.ensure_future(self.process_stt_audio(path))

    @staticmethod
    def _discard_recording(path):
        try:
            if path:
                os.unlink(path)
        except OSError:
            pass

    def _ptt_released(self, event):
        """The Esc of a Shift+Esc was let go. A TAP toggles (press again to
        send); a HOLD is push-to-talk — letting go sends. True if this ended a
        recording. Only the release of the press that STARTED the recording
        counts (`_ptt_press_open`), and Shift may be let go first or last, so
        the modifiers are not checked."""
        if (event.key() != Qt.Key.Key_Escape or event.isAutoRepeat()
                or not self._ptt_press_open):
            return False
        self._ptt_press_open = False
        if self.is_recording and time.monotonic() - self._rec_started >= PTT_HOLD_S:
            self.stop_recording()
            return True
        return False

    def keyReleaseEvent(self, event):
        if not self._ptt_released(event):
            super().keyReleaseEvent(event)

    def take_picture(self):
        if self._busy():
            self._note("still working on the last message — Esc stops it", NOTE_WARN)
            return
        self._note_activity()
        dialog = CameraPreviewDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_data:
            b64_img, prompt_text = dialog.result_data
            
            if not prompt_text:
                prompt_text = "I just took a picture with my camera. What do you see?"
                
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
            # _submit re-checks: a turn may have started while the dialog was
            # open (it is modal, but timers and tasks keep running under it).
            if not self._submit(content, f"{escape_user(prompt_text)}<br><span style='color:{T.TEXT_DIM};'><i>[ optic capture attached ]</i></span>"):
                self._note("the picture was not sent — a turn was already running", NOTE_WARN)

    async def process_stt_audio(self, path):
        """Uploads the audio and forwards the transcribed text to the chat."""
        # The whole clip is read into memory FIRST (it is small: 16 kHz mono,
        # ~32 KB a second) and the file removed. Handing httpx the open file
        # let the declared Content-Length and the bytes actually read differ
        # whenever the file changed in between.
        try:
            with open(path, 'rb') as f:
                audio = f.read()
        except OSError:
            self.set_face_mood("idle")
            return
        finally:
            self._discard_recording(path)

        self._transcribing = True
        self.text_input.setPlaceholderText("Transcribing audio...")
        self.text_input.setEnabled(False)

        try:
            # 60s (was 30): transcription runs on the audio node, and a long
            # held PTT plus node queueing can exceed 30s. arecord already
            # writes 16kHz mono WAV — exactly what the endpoint wants — so the
            # server-side transcode is a cheap passthrough.
            async with httpx.AsyncClient(timeout=60.0, verify=VOICE_VERIFY_TLS) as client:
                # Standard multipart file upload format
                files = {'file': ('ghost_stt.wav', audio, 'audio/wav')}
                response = await client.post(
                    STT_SERVER_URL, files=files,
                    headers={"X-Ghost-Key": GHOST_API_KEY})

                if response.status_code == 200:
                    data = response.json()
                    # Assuming the server returns a JSON with a 'text' key
                    text = data.get("text", "").strip()
                    if text:
                        self.text_input.setText(text)
                        self._after_transcript()
                    else:
                        # Empty transcription — return to idle
                        self.set_face_mood("idle")
                else:
                    # Surface the server's OWN message, not just the status.
                    # The endpoint explains real causes (missing binary under
                    # a daemon PATH, clip too long); "HTTP 503" alone sent the
                    # last diagnosis down the wrong path entirely.
                    try:
                        detail = response.json().get("error") or response.text[:160]
                    except Exception:
                        detail = response.text[:160]
                    self.update_chat_signal.emit(
                        "error", f"STT failed: HTTP {response.status_code} — {detail}")
                    self.set_face_mood("idle")
        except Exception as e:
            self.update_chat_signal.emit("error", f"STT Error: {str(e)}")
            self.set_face_mood("idle")
        finally:
            self._transcribing = False
            # The REAL placeholder, not "": this used to blank it for the rest
            # of the session after the first transcription.
            self.text_input.setPlaceholderText(INPUT_PLACEHOLDER)
            self.text_input.setEnabled(True)
            self.text_input.setFocus()

    # ── voice review: a moment to fix a misheard word ────────────────────
    def _after_transcript(self):
        """The transcript is in the input. Send it — after a short, visible
        countdown when the operator can see and fix it."""
        if STT_REVIEW_S <= 0:
            # No review window configured: send.
            self.handle_input()
            return
        self.set_face_mood("idle")
        self._review_left = STT_REVIEW_S
        self.ptt_btn.setStyleSheet(chip_style_on())
        self._show_review()
        self.review_timer.start(250)

    def _show_review(self):
        self.ptt_btn.setText(f"↵  {max(0.0, self._review_left):.0f}s")

    def _tick_review(self):
        self._review_left -= 0.25
        if self._review_left <= 0:
            self._cancel_review()
            self.handle_input()
        else:
            self._show_review()

    def _cancel_review(self):
        if self.review_timer.isActive():
            self.review_timer.stop()
            if not self.is_recording:
                self.ptt_btn.setStyleSheet(chip_style())
                set_chip(self.ptt_btn, "mic", "Talk")

    def _on_text_edited(self, _text):
        # The operator touched the transcript: it is theirs now, and it goes
        # when they press Enter.
        self._note_activity()
        self._cancel_review()

    def update_stats(self):
        now = datetime.datetime.now().strftime("%I:%M %p")
        pct, state = devstatus.read_battery()
        self._on_battery = devstatus.on_battery(state)
        self.stats_label.setText(devstatus.status_html(
            self._agent_ok, devstatus.read_wifi(), pct, state, now,
            ok=T.OK, danger=T.DANGER, dim=T.TEXT_DIM))

    def _scroll_key(self, event):
        """PgUp / PgDn (and Shift+Up / Shift+Down, for keyboards where the
        page keys sit behind Fn) scroll the transcript. True if handled.

        The input keeps the focus, so the transcript never saw a key: reading
        back meant steering the trackball onto an 8-pixel scrollbar.
        """
        key = event.key()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_PageUp or (shift and key == Qt.Key.Key_Up):
            self.chat_display.scroll_page(-1)
            return True
        if key == Qt.Key.Key_PageDown or (shift and key == Qt.Key.Key_Down):
            self.chat_display.scroll_page(+1)
            return True
        return False

    def eventFilter(self, obj, event):
        if obj == self.text_input and event.type() == QEvent.Type.KeyRelease:
            if self._ptt_released(event):
                return True
        if obj == self.text_input and event.type() == QEvent.Type.KeyPress:
            self._note_activity()
            if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # ANY key but Enter stops a voice transcript's countdown:
                # moving the caret to the misheard word must not race it, and
                # recalling an old message with Up must not be auto-sent.
                self._cancel_review()
            if self._scroll_key(event):
                return True
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if ctrl and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                # Normally the Ctrl+Up / Ctrl+Down shortcuts fire first; this
                # is the same action if the key reaches the input instead.
                self.rate_last("positive" if event.key() == Qt.Key.Key_Up else "negative")
                return True
            if event.key() == Qt.Key.Key_Up:
                if self.input_history:
                    if self.history_index == -1:
                        self.history_index = len(self.input_history) - 1
                    elif self.history_index > 0:
                        self.history_index -= 1
                    self.text_input.setText(self.input_history[self.history_index])
                return True
            elif event.key() == Qt.Key.Key_Down:
                if self.input_history and self.history_index != -1:
                    if self.history_index < len(self.input_history) - 1:
                        self.history_index += 1
                        self.text_input.setText(self.input_history[self.history_index])
                    else:
                        self.history_index = -1
                        self.text_input.clear()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        self._note_activity()
        if self._scroll_key(event):
            return

        if (not self.text_input.hasFocus() and len(event.text()) > 0
                and event.text().isprintable()):
            # Hand the key to the input ONCE. An input that cannot take it
            # (disabled while a clip transcribes, or a key it ignores such as
            # Alt+letter) lets Qt pass it back up to this window, which used
            # to hand it over again — RecursionError, the client aborted
            # (rc 134), and the relaunched window came up without keyboard
            # focus: "the keyboard stopped working" (2026-10-17, twice).
            if self.text_input.isEnabled() and not self._forwarding_key:
                self._forwarding_key = True
                try:
                    self.text_input.setFocus()
                    QApplication.sendEvent(self.text_input, event)
                finally:
                    self._forwarding_key = False
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_link_clicked(self, url):
        # ChatLog emits a plain string (QLabel.linkActivated); the old
        # QTextBrowser emitted a QUrl. Accept either.
        link = url if isinstance(url, str) else url.toString()
        if link.startswith("/api/download/"):
            asyncio.ensure_future(self._download_and_show_image(link))
        else:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl as _QUrl
            QDesktopServices.openUrl(_QUrl(link))

    def _check_tts_done(self):
        """Poll TTS queues; when both drain and faces are still in speak, go idle."""
        if (self._face_mood == "speak"
                and audio_queue.empty() and playback_queue.empty()):
            self.set_face_mood("idle")

    def toggle_tts(self):
        self.tts_enabled = not self.tts_enabled
        self._voice_fault_shown = False
        if self.tts_enabled:
            set_chip(self.tts_btn, "volume", "Voice on", T.OK)
            self.tts_btn.setStyleSheet(chip_style_on())
        else:
            set_chip(self.tts_btn, "volume-x", "Voice off")
            self.tts_btn.setStyleSheet(chip_style())
            self._silence()

    def set_face_mood(self, mood):
        """Set the face mood, and remember it.

        The mood is tracked here because the face now lives in a browser: its
        state is only reachable through async JavaScript, so a caller cannot
        ask "are we still speaking?" synchronously the way it could of the old
        QPainter widgets. One local string replaces that read.
        """
        self._face_mood = mood
        try:
            self.web_face.set_mood(mood)
        except Exception:  # noqa: BLE001 — a face must never break a turn
            pass

    def _show_animations(self, on):
        # Discreet (operator, 2026-11-06): a faint icon, no word and no fill.
        word = "Animations on" if on else "Animations off"
        self.animations_btn.setIcon(icons.icon("sparkles" if on else "sparkles-off", T.TEXT_DIM, 18))
        self.animations_btn.setIconSize(QSize(18, 18))
        self.animations_btn.setToolTip(word)
        self.animations_btn.setAccessibleName(word)

    def toggle_animations(self):
        """Animations on (the mood tesseract) or off (the empty form), as the
        web UI's button. The web face owns the two forms; the chip shows what
        it reports back."""
        def _got(name):
            if name:
                self._show_animations(name != "empty")
        try:
            self.web_face.cycle_form(_got)
        except Exception:  # noqa: BLE001
            pass

    # ── one turn at a time ───────────────────────────────────────────────
    def _busy(self):
        return self._turn_task is not None and not self._turn_task.done()

    def _submit(self, content, shown_html):
        """Send a message. The ONLY way one is sent — and it refuses while a
        turn is running.

        The check is HERE, at the moment of commitment, not at each caller:
        the camera dialog is modal, and a voice transcript's countdown can
        start a turn while it is open; a caller that checked `_busy()` before
        opening it would then start a second stream into the first one's
        reply bubble. Returns whether the message went.
        """
        if self._busy():
            if not self._busy_noted:
                self._busy_noted = True
                self._note("still working on the last message — Esc (or /stop) "
                           "stops the turn", NOTE_WARN)
            return False
        self._silence()
        self.update_chat_signal.emit("user", shown_html)
        self.conversation_history.append({"role": "user", "content": content})
        self.web_face.wake()
        self._start_turn()
        return True

    def _start_turn(self):
        self._stop_asked = 0
        self._stop_took = False
        self._stop_pending = False
        self._busy_noted = False
        # The chips rate the LAST FINISHED reply; while a new one is arriving
        # they would label the previous turn, under the operator's eyes on the
        # new one. Off until this turn ends.
        self._last_reply_rid = None
        self._rated = None
        self._refresh_rating_chips()
        self.stop_btn.show()
        self._turn_task = asyncio.ensure_future(self.send_chat_request())

    def request_stop(self):
        """Stop the running turn — on the AGENT, not just here.

        The first press asks the agent to stop at its next boundary (it
        returns what it has). A second press forces it: the agent cancels the
        task outright and this client stops listening.
        """
        self._note_activity()
        if not self._busy():
            self._note("nothing is running")
            return
        self._stop_asked += 1
        hard = self._stop_asked >= 2
        self._stop_pending = True
        self._silence()                 # stop means stop talking, too
        self._note("forcing the stop…" if hard else "stopping… (again to force)")
        asyncio.ensure_future(self._stop_turn(hard))

    async def _stop_turn(self, hard):
        task = self._turn_task
        async with httpx.AsyncClient(timeout=8.0) as client:
            # By THIS turn's request id and nothing else — never "whatever is
            # running", which can be a dream or another client's turn.
            outcome, detail = await agentapi.cancel_turn(
                client, AGENT_BASE, GHOST_API_KEY, self._turn_rid, hard=hard)
        if task is None or task is not self._turn_task:
            return                      # a different turn by now
        self._stop_pending = False
        if outcome == "cancelled":
            self._stop_took = True
        if task.done():
            return                      # it ended while we were asking
        if outcome == "failed":
            self._note(f"the agent did not take the stop ({detail})", NOTE_ERR)
        elif outcome == "unknown":
            self._note("this turn has no id yet — nothing was cancelled on the agent",
                       NOTE_WARN)
        if hard:
            self._stop_took = True
            task.cancel()

    # ── rating the last reply ────────────────────────────────────────────
    def _refresh_rating_chips(self):
        can = bool(self._last_reply_rid)
        for btn, signal, icon_name, label, fg, fill in (
                (self.good_btn, "positive", "thumbs-up", "Good", T.OK, "rgba(159, 227, 184, 0.16)"),
                (self.bad_btn, "negative", "thumbs-down", "Bad", T.DANGER, "rgba(255, 123, 145, 0.16)")):
            latched = can and self._rated == signal
            btn.setEnabled(can)
            set_chip(btn, icon_name, label, fg if latched else T.TEXT_SOFT)
            btn.setStyleSheet(chip_style_on(fg, fill) if latched else chip_style())

    def rate_last(self, signal, note=""):
        """A human label for the last reply — the scarcest signal the agent's
        learning has, and one this device never produced."""
        self._note_activity()
        if not self._last_reply_rid:
            self._note("no reply to rate yet (a quick greeting cannot be rated)")
            return
        if self._rated == signal and not note:
            return
        asyncio.ensure_future(self._send_rating(self._last_reply_rid, signal, note))

    async def _send_rating(self, rid, signal, note):
        async with httpx.AsyncClient(timeout=15.0) as client:
            ok, detail = await agentapi.send_feedback(
                client, AGENT_BASE, GHOST_API_KEY, rid, signal, note)
        if rid != self._last_reply_rid:
            return              # a newer reply owns the chips now
        if ok:
            self._rated = signal
            self._refresh_rating_chips()
        else:
            self._note(f"rating not recorded: {detail}", NOTE_ERR)

    # ── device controls ──────────────────────────────────────────────────
    def _set_brightness(self, arg):
        result = devstatus.set_backlight(arg)
        if result is None:
            self._note("usage: /bright 1-9, /bright + or /bright -  "
                       "(or this device has no backlight control)", NOTE_WARN)
        else:
            self._note(f"brightness {result[0]} of {result[1]}")

    async def _set_volume(self, arg):
        env = devstatus.panel_env()

        async def run(*argv):
            proc = await asyncio.create_subprocess_exec(
                *argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = await proc.communicate()
            return out.decode(errors="replace")

        try:
            current = devstatus.parse_volume(
                await run("wpctl", "get-volume", devstatus.SINK))
            if current is None:
                self._note("no audio output found", NOTE_WARN)
                return
            argv = devstatus.volume_command(arg, current)
            if argv is None and (arg or "").strip():
                self._note("usage: /vol 0-100, /vol + or /vol -", NOTE_WARN)
                return
            reading = ""
            if argv:
                await run(*argv)
            reading = await run("wpctl", "get-volume", devstatus.SINK)
            current = devstatus.parse_volume(reading)
            self.chat_display.scroll_to_end()
            self._note(f"volume {current}%"
                       + ("  (muted)" if devstatus.volume_muted(reading) else ""))
        except OSError as e:
            self._note(f"volume control unavailable: {e}", NOTE_ERR)

    def _new_conversation(self):
        if self._busy():
            self._note("a turn is running — /stop it first", NOTE_WARN)
            return
        # A NEW session id, not just an empty screen: the agent keeps the
        # conversation under the id, and reusing it would bring every earlier
        # message back into the next turn.
        self._begin_session(agentapi.new_session_id())
        self._note("new conversation.", NOTE_WARN)
        self.set_face_mood("idle")
        self._silence()

    def _run_command(self, cmd):
        name, arg = cmd
        if name in commands.CONFIRM:
            if not self.confirmer.confirm(name):
                self._note(f"/{name} — type it again within "
                           f"{int(commands.CONFIRM_WINDOW_S)} s to confirm", NOTE_WARN)
                return
        else:
            self.confirmer.disarm()

        # A command is the operator asking for something: its answer is
        # shown, wherever the transcript happened to be scrolled.
        self.chat_display.scroll_to_end()
        if name == "extra":
            self._note(f"/{arg} takes nothing after it — nothing was done", NOTE_WARN)
        elif name == "help":
            self.chat_display.add(commands.help_html(T.TEXT_DIM, T.ACCENT, T.FONT), "system")
        elif name in ("new", "clear"):
            self._new_conversation()
        elif name == "stop":
            self.request_stop()
        elif name == "sessions":
            asyncio.ensure_future(self._show_sessions())
        elif name == "open":
            if self._busy():
                self._note("a turn is running — /stop it first", NOTE_WARN)
            else:
                asyncio.ensure_future(self._open_session(arg))
        elif name == "good":
            self.rate_last("positive")
        elif name == "bad":
            self.rate_last("negative", arg)
        elif name == "bright":
            self._set_brightness(arg)
        elif name == "vol":
            asyncio.ensure_future(self._set_volume(arg))
        elif name == "tts":
            self.toggle_tts()
        elif name == "face":
            self.toggle_animations()
        elif name == "shutdown":
            self._note("powering down hardware…", NOTE_WARN)
            _power(['sudo', 'shutdown', '-h', 'now'])
        elif name == "reboot":
            self._note("rebooting hardware…", NOTE_WARN)
            _power(['sudo', 'reboot'])
        elif name == "exit":
            self._note("detaching from cyberdeck…", NOTE_WARN)
            QApplication.quit()
        else:
            near = commands.suggestion(arg)
            self._note(f"unknown command /{arg}"
                       + (f" — did you mean /{near}?" if near else "")
                       + "   (/help lists them)", NOTE_WARN)

    def handle_input(self):
        self._cancel_review()
        self._note_activity()
        text = self.text_input.text().strip()
        if not text:
            return

        cmd = commands.parse(text)
        if cmd is not None:
            if cmd.name not in ("unknown", "extra"):
                # A mistyped command stays in the input to be corrected — it
                # may be a whole sentence that only LOOKED like a command.
                self.input_history.append(text)
                self.history_index = -1
                self.text_input.clear()
            self._run_command(cmd)
            return
        self.confirmer.disarm()

        # While a turn runs the text STAYS in the input: sending it would
        # start a second stream into the reply that is still arriving.
        if self._submit(text, escape_user(text)):
            self.input_history.append(text)
            self.history_index = -1
            self.text_input.clear()

    def _say(self, sentences):
        # Not while recording: barge-in silenced what was queued, and a reply
        # still streaming must not start talking again into the open
        # microphone (it would be transcribed and sent back as the operator's
        # words).
        if self.tts_enabled and not self.is_recording:
            for sentence in sentences:
                audio_queue.put_nowait(sentence)

    async def send_chat_request(self):
        url = f"{AGENT_BASE}/api/chat"
        # The request id is minted HERE, so the turn can be stopped before the
        # agent has sent a frame (the whole thinking phase). The agent may
        # uniquify it; the frames carry the id it actually used.
        self._turn_rid = agentapi.new_request_id()
        self._turn_verdict = None
        headers = agentapi.headers(GHOST_API_KEY, self._turn_rid)
        payload = {
            # model omitted on purpose — the agent uses its configured model;
            # pinning a name here 404s (ModelNotFound) whenever the model is upgraded
            "messages": self.conversation_history,
            # Durable session: the agent stores the conversation under this id
            # and merges a replayed history tolerantly, so sending the whole
            # local history (as the web UI does) can never double it.
            "session_id": self.session_id,
            "stream": True
        }
        
        self.update_chat_signal.emit("start_response", "")
        self.set_face_mood("think")
        self.speech = SpeechChunker()
        started = time.monotonic()
        frame_rid = None          # the id the agent filed this turn under
        unlabelable = False
        writing = False
        recorded = False
        saw_done = got_error = False   # read by the finally (§4ML)
        
        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")
                        self.web_face.note_error(body)
                        self.update_chat_signal.emit(
                            "error", agentapi.describe_http_error(response.status_code, body))
                        return
                    self._set_agent_ok(True)

                    held = None           # §4KP: a list while the reply is held
                    saw_done = got_error = False
                    err_msg = None        # the FIRST fault, shown once after the reply
                    after_err = []        # content the agent sends after a fault (its fallback)
                    # one SSE line per item (§4KP: aiter_text yields socket chunks,
                    # and two frames in one chunk failed json.loads together)
                    #
                    # ⚠ The loop does NOT `break` at [DONE]; it reads on to the end of
                    # the stream (which the agent closes right after). A `break` leaves
                    # httpx's chain of async generators suspended, and under qasync —
                    # which installs no async-generator finalizer — they are torn down
                    # by the garbage collector mid-await: "async generator ignored
                    # GeneratorExit" and anyio's "exit cancel scope in a different
                    # task", on every turn. Found by device_probe.py; the old client
                    # did it silently, its stderr going nowhere. Closing the line
                    # iterator does not help: the generators nested under it are not
                    # closed with it.
                    async for chunk in response.aiter_lines():
                        if saw_done:
                            continue
                        if chunk.startswith("data: "):
                            data_str = chunk[6:].strip()
                            if data_str == "[DONE]":
                                saw_done = True
                                continue
                            try:
                                data = json.loads(data_str)
                                if not isinstance(data, dict):
                                    continue       # `data: 12` is not a frame
                                _rid = agentapi.frame_request_id(data)
                                if _rid:
                                    frame_rid = self._turn_rid = _rid
                                if agentapi.frame_unlabelable(data):
                                    unlabelable = True
                                _err = agentapi.frame_error(data)
                                if _err is not None:
                                    got_error = True       # §4KP: a cut reply is not released
                                    if err_msg is None:
                                        err_msg = _err
                                    continue       # the agent may still send its fallback sentence
                                if agentapi.frame_retract(data):
                                    # §4MP: drop what was shown and said; the
                                    # reply that follows replaces it
                                    if held is not None:
                                        held.clear()
                                    elif self.current_response_text:
                                        self._silence()
                                        self.speech = SpeechChunker()
                                        self.update_chat_signal.emit("retract_response", "")
                                    continue
                                if (data.get("ghost") or {}).get("reasoning_unparsed") is True and held is None:
                                    held = []
                                # Any frame shape: a usage-only frame has an
                                # EMPTY `choices` (see agentapi.frame_content).
                                content = agentapi.frame_content(data)

                                if content and held is not None:
                                    # spoken and shown once complete; after a fault
                                    # only the agent's own fallback is kept
                                    (after_err if got_error else held).append(content)
                                    continue
                                if content:
                                    if not writing:
                                        writing = True
                                        self.web_face.set_phase("write")
                                    self.update_chat_signal.emit("update_response", content)
                                    self.web_face.pulse()
                                    # Network auto-spawns its own pulses in think mode;
                                    # just feed it a token-activity signal instead of
                                    # stacking extra full MoE cascades on every token.
                                    self.web_face.feed_audio(0.5)
                                    self._say(self.speech.feed(content))
                            except json.JSONDecodeError:
                                pass
                    if held is not None and saw_done:
                        text = strip_orphan_think_close("".join(after_err if got_error else held))
                        if text:
                            self.update_chat_signal.emit("update_response", text)
                            self._say(self.speech.feed(text))

                    if err_msg is not None:
                        self.web_face.note_error(err_msg)
                        self.update_chat_signal.emit("error", err_msg)   # once, after the reply

            self._say(self.speech.flush())
            if self.current_response_text:
                self.conversation_history.append({"role": "assistant", "content": self.current_response_text})
            recorded = True
            
        except asyncio.CancelledError:
            # A forced stop (request_stop, second press). Not an error: the
            # operator asked for exactly this.
            pass
        except Exception as e:
            # only a stop the agent TOOK, or a FORCED one still on its way,
            # breaks the stream on purpose — a cooperative stop pending (it
            # may yet be refused) never hides a real drop (fresh-reader R2)
            if self._stop_took or (self._stop_pending and self._stop_asked >= 2):
                # §4ML: a hard stop kills the agent's task mid-stream, so the
                # link breaks — the operator asked for that; it is not "eva
                # dropped the connection" (the error flinch, then "stopped.")
                print(f"[stop] stream ended by the stop: {type(e).__name__}", flush=True)
            else:
                self.web_face.note_error(f"{type(e).__name__}: {e}")
                if agentapi.is_unreachable(e):
                    self._set_agent_ok(False)
                self.update_chat_signal.emit("error", agentapi.describe_error(e, AGENT_BASE))
        finally:
            if not recorded and self.current_response_text:
                # A reply that was cut (stopped, or the link dropped) is still
                # what the operator read: it stays in the conversation.
                self.conversation_history.append(
                    {"role": "assistant", "content": self.current_response_text})
            self.update_chat_signal.emit("stop_thinking", "")
            self.stop_btn.hide()
            # §4ML: a stop still on its way when the reply finished ON ITS OWN
            # (a clean [DONE]) did not stop anything — the whole reply arrived
            if self._stop_took or (self._stop_pending and not (saw_done and not got_error)):
                # When the agent TOOK the stop (or this client forced it) —
                # or the stream ended while the stop was still on its way,
                # which is the same thing arriving in the other order. A stop
                # that was REFUSED, followed by a normal finish, is not
                # "stopped".
                self._note("stopped.")
            # How the turn ended shapes how the face lets go, then the gait clears.
            self.web_face.note_verdict(self._turn_verdict or "stop")
            self.web_face.set_phase(None)
            # The reply can be rated when the agent filed a trajectory for it.
            self._last_reply_rid = (frame_rid if frame_rid and not unlabelable
                                    and self.current_response_text else None)
            self._rated = None
            self._refresh_rating_chips()
            # ALWAYS leave "think". A failed turn used to skip this (so the
            # error flinch would show) and the mood stayed "think" until the
            # next successful turn — a face that looks busy forever, and one
            # the frame-rate policy therefore never slowed. The flinch is its
            # own signal (note_error) and does not need the mood held.
            if self.tts_enabled and (not audio_queue.empty() or not playback_queue.empty()):
                self.set_face_mood("speak")
            else:
                self.set_face_mood("idle")
            self._last_input_at = time.monotonic()
            asyncio.ensure_future(self._reply_landed(time.monotonic() - started))

    # ── when a reply lands: wake the panel, and say so ───────────────────
    async def _run_quiet(self, *argv, env=None, stdin=None):
        """Run a small helper; returns (returncode, stdout). Never raises."""
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, env=env,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout=6)
            return proc.returncode, out.decode(errors="replace")
        except Exception:  # noqa: BLE001 — tool missing, not Wayland, timeout
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()             # a timed-out helper is not left running
                except ProcessLookupError:
                    pass
            return None, ""

    async def _reply_landed(self, elapsed_s):
        """A long turn used to end in silence on a dark screen: the panel
        blanks after 10 idle minutes whether or not a turn is running, and
        nothing brought it back. Wake it, and chime if the wait was long."""
        env = devstatus.panel_env()
        _rc, out = await self._run_quiet("wlopm", env=env)
        was_off = devstatus.panel_is_off(out) is True
        if was_off:
            await self._run_quiet("wlopm", "--on", "*", env=env)
        if CHIME_AFTER_S > 0 and should_chime(elapsed_s, was_off, self.tts_enabled,
                                              min_elapsed_s=CHIME_AFTER_S):
            await self._run_quiet("aplay", "-q", "-", stdin=chime_wav())

    # ── the face's signal layer ──────────────────────────────────────────
    def _on_turn_step(self, title, icon, detail):
        """One step line of THIS turn (TurnTicker has already filtered out
        other corridors and plumbing): give the face its gait and kicks."""
        f = face_signals_for_ticker(title, icon, detail)
        if f["phase"]:
            self.web_face.set_phase(f["phase"])
        if f["tool"]:
            self.web_face.note_tool()
        if f["recall"]:
            now = time.monotonic()
            if now - getattr(self, "_last_recall_at", 0.0) > 1.5:
                self._last_recall_at = now
                self.web_face.note_recall()
        if f["verdict"]:
            self._turn_verdict = f["verdict"]

    # ── link: is the agent there, and is it busy with itself? ────────────
    def _set_agent_ok(self, ok):
        if ok != self._agent_ok:
            self._agent_ok = ok
            self.update_stats()

    async def link_loop(self):
        """Every LINK_POLL_S: one cheap call that answers two questions — is
        the agent reachable (the status dot), and does a turn that is not
        ours hold the lock (the face's second, slower breath)."""
        if LINK_POLL_S <= 0:
            return
        async with httpx.AsyncClient(timeout=6.0) as client:
            while True:
                try:
                    payload = await agentapi.fetch_turns(client, AGENT_BASE, GHOST_API_KEY)
                    # While our own turn streams, the stream is the evidence.
                    if not self._busy():
                        self._set_agent_ok(payload is not None)
                    self.web_face.set_background_busy(agentapi.background_busy(
                        payload, self.session_id, self._turn_rid if self._busy() else None))
                except Exception as e:  # noqa: BLE001 — a poll must never end the loop
                    print(f"[link] poll failed: {e}", flush=True)
                await asyncio.sleep(LINK_POLL_S)

    # ── what the agent did while nobody was asking ───────────────────────
    def _deliver_notifications(self, records):
        for rec in records:
            line = agentapi.format_notification(rec)
            self.chat_display.add(
                f"<span style='color:{T.ACCENT};'>•</span>&nbsp; {escape_user(line)}", "system")
            self._say([str(rec.get("summary") or "")] if rec.get("summary") else [])
        self.web_face.wake()

    async def notify_loop(self):
        if NOTIFY_POLL_S <= 0:
            return
        poller = agentapi.NotifyPoller(AGENT_BASE, GHOST_API_KEY)
        async with httpx.AsyncClient(timeout=20.0) as client:
            while True:
                try:
                    await poller.cycle(client, self._deliver_notifications)
                except Exception as e:  # noqa: BLE001
                    print(f"[notify] poll failed: {e}", flush=True)
                await asyncio.sleep(NOTIFY_POLL_S)

    async def mood_loop(self):
        """Every MOOD_POLL_S: hand the agent's mood to the face, as the web UI's
        status poll does. A failed poll keeps the look it has."""
        if MOOD_POLL_S <= 0:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                mood = await agentapi.fetch_mood(client, AGENT_BASE, GHOST_API_KEY)
                if mood:
                    try:
                        self.web_face.set_agent_mood(*mood)
                    except Exception:  # noqa: BLE001 — a face must never break the client
                        pass
                await asyncio.sleep(MOOD_POLL_S)

    def _animate_thinking(self):
        if getattr(self, 'is_thinking', False):
            self._render_thinking()

    def _render_thinking(self):
        """Paint the waiting caption into the streaming bubble."""
        # `/clear` mid-turn drops the streaming bubble but leaves this timer
        # running — without the guard, update_agent() would open a BRAND NEW
        # bubble and paint the caption into the transcript the operator just
        # wiped.
        if not self.chat_display.has_open_agent():
            return
        self.chat_display.update_agent(caption_html(self.ticker))

    def note_log_line(self, line):
        """One line from the interface's log broadcast (the ticker's only feed).

        Runs on the qasync loop, i.e. the Qt main thread, so it touches widgets
        directly. Repaint only when the caption actually changed: the socket
        carries every line the agent logs (including other corridors' and the
        continuation lines of long thinking blocks), and re-fitting the bubble
        on each one would be constant churn for no visible difference.
        """
        if self.ticker.note_line(line) and getattr(self, 'is_thinking', False):
            self._render_thinking()

    def note_log_state(self, connected):
        """Log socket came up / went down — swap the pre-corridor placeholder."""
        if self.ticker.set_connected(connected) and getattr(self, 'is_thinking', False):
            self._render_thinking()

    def _close_thinking(self):
        """Stop the caption and discard the bubble if nothing ever arrived.

        The placeholder holds the status caption, so it is never literally
        empty — it has to be blanked before the log can decide to drop it.
        """
        self.ticker.stop()
        if getattr(self, 'is_thinking', False):
            self.is_thinking = False
            self.thinking_timer.stop()
            if not self.current_response_text:
                self.chat_display.update_agent("")
        self.chat_display.end_agent(drop_if_empty=True)

    def _update_chat(self, action, data):
        # Cursor arithmetic is gone: the transcript is widgets now, and the
        # streaming bubble is addressed directly instead of by document offset.
        if action == "append":
            self.chat_display.add(data, "system")
        elif action == "user":
            self.chat_display.add(data, "user")
        elif action == "agent":
            self.chat_display.add(data, "agent")
        elif action == "start_response":
            self.current_response_text = ""
            self.chat_display.start_agent()
            self.is_thinking = True
            # start() BEFORE the first render: it resets the elapsed clock and
            # re-arms corridor adoption, so the caption belongs to THIS turn.
            self.ticker.start(self._turn_rid)
            self._render_thinking()
            # 1 s — the clock has second granularity and each repaint re-fits
            # the bubble, which is not free on the CM4.
            self.thinking_timer.start(1000)
        elif action == "update_response":
            if getattr(self, 'is_thinking', False):
                self.is_thinking = False
                self.thinking_timer.stop()
                self.ticker.stop()
            self.current_response_text += data

            self.chat_display.update_agent(render_reply(self.current_response_text))

            for image_path in reply_images(self.current_response_text):
                self.show_image_signal.emit(image_path)

        elif action == "retract_response":
            # §4MP: the streamed text was retracted — the bubble starts over
            self.current_response_text = ""
            self.chat_display.update_agent(render_reply(""))
        elif action == "stop_thinking":
            self._close_thinking()
            return
        elif action == "error":
            self._close_thinking()
            # `data` is a server message or an exception's text — escaped, or a
            # `<` in it would be swallowed as markup.
            self.chat_display.add(
                f"<span style='color:{T.DANGER};'>{escape_user(data)}</span>", "system")

    def _show_image_popup(self, image_path):
        if image_path in self.shown_images:
            return
        self.shown_images.add(image_path)
        asyncio.ensure_future(self._download_and_show_image(image_path))

    async def _download_and_show_image(self, image_path):
        url = f"{AGENT_BASE}{image_path}"
        headers = {"X-Ghost-Key": GHOST_API_KEY}
        # §4SX B0: a bare name resolves under THIS session's focused project
        if getattr(self, "session_id", None):
            headers["X-Ghost-Session"] = str(self.session_id)
        # A failed fetch used to be a print() nobody saw: the reply said an
        # image was attached and tapping it did nothing.
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.get(url, headers=headers)
                if r.status_code == 200:
                    pixmap = QPixmap()
                    if pixmap.loadFromData(r.content):
                        self._display_image_dialog(pixmap)
                    else:
                        self._note("that image could not be decoded", NOTE_ERR)
                else:
                    self._note(f"image not available (HTTP {r.status_code})", NOTE_ERR)
        except Exception as e:
            self._note(f"image fetch failed: {agentapi.describe_error(e, AGENT_BASE)}", NOTE_ERR)

    # ── voice faults, said once ──────────────────────────────────────────
    def note_voice_fault(self, message):
        """Spoken replies failed. It used to be a print() per sentence into a
        log that did not exist: TTS on, silence, and no reason given."""
        if self._voice_fault_shown:
            return
        self._voice_fault_shown = True
        self._silence()
        self._note(f"voice unavailable — {message}", NOTE_ERR)

    def note_voice_ok(self):
        self._voice_fault_shown = False

    def _display_image_dialog(self, pixmap):
        dialog = ImageViewer(pixmap, self)
        dialog.show()

# Set in __main__ to the window's note_voice_fault / note_voice_ok. Module
# level because the two audio tasks below are free functions.
_voice_fault = lambda _message: None   # noqa: E731
_voice_ok = lambda: None               # noqa: E731


async def audio_fetch_task():
    # verify=VOICE_VERIFY_TLS: the voice endpoints moved to the interface's
    # self-signed TLS port (see the VOICE_BASE_URL block at the top).
    async with httpx.AsyncClient(timeout=60.0, verify=VOICE_VERIFY_TLS) as client:
        while True:
            try:
                text_chunk = await audio_queue.get()
                if not text_chunk:
                    audio_queue.task_done()
                    continue
                
                payload = {"text": text_chunk}
                epoch = _speech_epoch
                resp = await client.post(
                    TTS_SERVER_URL, json=payload, timeout=60.0,
                    headers={"X-Ghost-Key": GHOST_API_KEY})
                if resp.status_code == 200:
                    # audio/wav from the macOS synthesiser; `aplay -q -` reads
                    # a WAV header off stdin, same as the old Piper output.
                    if epoch == _speech_epoch:      # not silenced meanwhile
                        await playback_queue.put(resp.content)
                    _voice_ok()
                else:
                    print(f"TTS Fetch Err: HTTP {resp.status_code} "
                          f"{(resp.text or '')[:160]}", flush=True)
                    _voice_fault(agentapi.describe_http_error(resp.status_code, resp.text or ""))
            except Exception as e:
                print(f"TTS Fetch Err: {e}", flush=True)
                _voice_fault(agentapi.describe_error(e, VOICE_BASE_URL))
            finally:
                try:
                    audio_queue.task_done()
                except:
                    pass

async def audio_worker_task():
    while True:
        try:
            audio_bytes = await playback_queue.get()
            if not audio_bytes:
                playback_queue.task_done()
                continue
                
            proc = await asyncio.create_subprocess_exec(
                'aplay', '-q', '-', 
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if proc.stdin:
                proc.stdin.write(audio_bytes)
                await proc.stdin.drain()
                proc.stdin.close()
            await proc.wait()
        except Exception as e:
            print(f"TTS Play Err: {e}", flush=True)
            _voice_fault(f"playback failed ({e})")
        finally:
            try:
                playback_queue.task_done()
            except:
                pass

if __name__ == "__main__":
    # Belt-and-braces for the web face: QtWebEngine needs shared GL contexts
    # established BEFORE the QApplication exists. The module-level import in
    # webface.py already satisfies Qt's requirement; this attribute is the
    # documented second half of the same contract and costs nothing when the
    # web face is unavailable.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    # PyQt ABORTS the process on an exception raised inside a Qt handler
    # (keyPressEvent, eventFilter, a slot) unless sys.excepthook has been
    # replaced. One bad key used to kill the client this way. Log it, loudly,
    # and keep the window — the launcher's CRASHED line is no longer the only
    # trace, and the operator keeps a working keyboard.
    def _log_unhandled(exc_type, exc, tb):
        import traceback
        print(f"[error] unhandled {exc_type.__name__} in a Qt handler — the client keeps running",
              flush=True)
        traceback.print_exception(exc_type, exc, tb)
        sys.stdout.flush()
    sys.excepthook = _log_unhandled

    app = QApplication(sys.argv)
    # Where keyboard focus goes, one line per change of WINDOW (not widget):
    # the evidence the next "keyboard stopped" needs, since a reboot clears
    # /tmp and this device keeps no journal.
    def _log_focus_window(win):
        title = (win.title() or win.objectName() or type(win).__name__) if win else "none"
        print(f"[focus] keyboard window → {title}", flush=True)
    app.focusWindowChanged.connect(_log_focus_window)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindow()
    window.show()
    _voice_fault, _voice_ok = window.note_voice_fault, window.note_voice_ok
    
    loop.create_task(audio_fetch_task())
    loop.create_task(audio_worker_task())
    # The conversation this device was having, the agent's reachability, and
    # what it did while nobody was asking. Each loop survives its own errors.
    loop.create_task(window.restore_session())
    loop.create_task(window.link_loop())
    loop.create_task(window.notify_loop())
    loop.create_task(window.mood_loop())
    # Live turn status. Started unconditionally: the reader reconnects forever
    # and never raises, so an interface that is down (or a device without the
    # `websockets` package) just leaves the waiting bubble on its offline
    # placeholder — chat itself talks to the agent directly and is unaffected.
    loop.create_task(stream_log_lines(
        LOG_WS_URL or log_ws_url(GHOST_HOST, GHOST_API_KEY),
        on_line=window.note_log_line,
        on_state=window.note_log_state,
        verify_tls=VOICE_VERIFY_TLS,
    ))

    with loop:
        loop.run_forever()
        