import sys
import os
import asyncio
import base64
import json
import httpx
import re
import datetime
import subprocess
import cv2
from PyQt6.QtCore import Qt, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import QPixmap, QFont, QShortcut, QKeySequence, QImage
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QTextBrowser, QLineEdit, QDialog, QLabel, QPushButton, QFileDialog, QStackedWidget
)
import markdown
import qasync

from webface import WebFaceWidget
from chatlog import ChatLog

audio_queue = asyncio.Queue()
playback_queue = asyncio.Queue()

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
# The interface serves a self-signed cert, so verification is off by default;
# the hop is inside the tailnet (WireGuard-authenticated) and carries the same
# key the chat calls already send over plain HTTP to :8000.
GHOST_HOST = os.environ.get("GHOST_HOST", "eva")
VOICE_BASE_URL = os.environ.get("GHOST_VOICE_BASE", f"https://{GHOST_HOST}:8080")
TTS_SERVER_URL = f"{VOICE_BASE_URL}/api/tts"
STT_SERVER_URL = f"{VOICE_BASE_URL}/api/stt"
VOICE_VERIFY_TLS = os.environ.get("GHOST_VOICE_VERIFY_TLS", "0").lower() in ("1", "true", "yes")


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
    GLASS_SOFT  = "rgba(9, 11, 22, 0.30)"     # inputs, chips
    GLASS_HOT   = "rgba(40, 26, 60, 0.55)"    # hover / active
    HAIRLINE    = "rgba(255, 255, 255, 0.10)"
    HAIRLINE_HOT = "rgba(201, 166, 255, 0.55)"

    TEXT        = "#ecebf6"
    TEXT_DIM    = "rgba(236, 235, 246, 0.52)"
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


def chip_style(fg=T.TEXT_DIM, border=T.HAIRLINE, hover=T.GLASS_HOT):
    return f"""
        QPushButton {{
            background-color: {T.GLASS_SOFT};
            color: {fg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 9px 16px;
            font-family: {T.FONT};
            font-size: 18px;
            font-weight: bold;
            letter-spacing: 1px;
        }}
        QPushButton:hover {{
            background-color: {hover};
            color: {T.ACCENT};
            border: 1px solid {T.HAIRLINE_HOT};
        }}
        QPushButton:pressed {{
            background-color: {T.GLASS_HOT};
            color: {T.TEXT};
        }}
    """


def chip_style_hot(fg, border):
    """Armed state (recording). Same glass geometry as chip_style so the chip
    does not change shape when it lights up — only its colour does."""
    return f"""
        QPushButton {{
            background-color: rgba(255, 84, 112, 0.20);
            color: {fg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 9px 16px;
            font-family: {T.FONT};
            font-size: 18px;
            font-weight: bold;
            letter-spacing: 1px;
        }}
    """


INPUT_STYLE = f"""
    QLineEdit {{
        background-color: {T.GLASS_SOFT};
        color: {T.TEXT};
        border: 1px solid {T.HAIRLINE};
        border-radius: 16px;
        padding: 16px 20px;
        font-family: {T.FONT};
        font-size: 22px;
        selection-background-color: {T.GLASS_HOT};
        selection-color: {T.TEXT};
    }}
    QLineEdit:focus {{
        border: 1px solid {T.HAIRLINE_HOT};
        background-color: {T.GLASS};
    }}
"""

DIALOG_STYLE = f"""
    QDialog {{
        background-color: {T.BG_PANEL};
        border: 1px solid {T.BORDER_HOT};
        border-radius: 14px;
    }}
"""

FILEDIALOG_STYLE = f"""
    QFileDialog, QListView, QTreeView, QLineEdit, QComboBox, QPushButton, QLabel {{
        background-color: {T.BG_PANEL};
        color: {T.TEXT};
        border-color: {T.BORDER};
        font-family: {T.FONT};
    }}
"""

# Chat bubbles moved to chatlog.py (2026-08-02, second pass): they are
# real QLabel widgets now, so they get true rounded/notched corners and
# hug their content — neither of which QTextDocument can do.

NOTE_DIM = f"<div style='color:{T.TEXT_DIM};'><i>"
NOTE_OK = f"<div style='color:{T.OK};'><i>"
NOTE_WARN = f"<div style='color:{T.ACCENT_WARM};'><i>"
NOTE_ERR = f"<div style='color:{T.DANGER};'><i>"

class ImageViewer(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DIALOG_STYLE)

        top_bar = QHBoxLayout()
        self.close_btn = QPushButton("✕  CLOSE")
        self.zoom_in_btn = QPushButton("+  ZOOM")
        self.zoom_out_btn = QPushButton("−  ZOOM")

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
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DIALOG_STYLE)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 18, 18, 18)

        top_bar = QHBoxLayout()
        self.title_label = QLabel("◉ OPTIC FEED")
        self.title_label.setStyleSheet(f"color: {T.ACCENT}; font-family: {T.FONT}; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        self.close_btn = QPushButton("✕  CLOSE")
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
        self.capture_btn = QPushButton("◉  CAPTURE")
        self.capture_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 51, 68, 0.15);
                color: {T.REC};
                border: 1px solid {T.REC};
                border-radius: 8px;
                padding: 14px 28px;
                font-family: {T.FONT};
                font-size: 22px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ background-color: rgba(255, 51, 68, 0.28); }}
        """)
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

        self.upload_btn = QPushButton("↑  TRANSMIT")
        self.upload_btn.setStyleSheet(chip_style(fg=T.OK, border="#2a5a3a"))
        self.upload_btn.clicked.connect(self.upload_picture)

        self.download_btn = QPushButton("↓  STASH")
        self.download_btn.setStyleSheet(chip_style(fg=T.ACCENT, border=T.BORDER_HOT))
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

        # --- HIGH QUALITY WEBCAM CONFIGURATION ---
        # 1. Force MJPG codec so USB bandwidth allows high FPS at high resolutions (fixes shakiness)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # 2. Set to 1080p resolution (increase to 3840x2160 if you want full 4K and your machine can handle it)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 3. Attempt to lock in a smooth 60 FPS
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # 4. Ensure autofocus is on
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        # -----------------------------------------

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16) # 16ms target for ~60 FPS
        self.current_frame = None
        self.result_data = None
        
        self.snap_state = 0
        self.snap_shortcut = QShortcut(QKeySequence("Ctrl+Escape"), self)
        self.snap_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.snap_shortcut.activated.connect(self.handle_snap_shortcut)

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
            
    def handle_snap_shortcut(self):
        if self.snap_state == 0:
            self.take_picture()
            self.snap_state = 1
        elif self.snap_state == 1:
            self.upload_picture()

    def take_picture(self):
        self.timer.stop()
        self.live_controls.hide()
        self.review_controls.show()
        self.prompt_input.setFocus()
        
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
                if self.cap.isOpened():
                    self.cap.release()
                self.accept()
                
    def close_and_stop(self):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.reject()

class MainWindow(QWidget):
    update_chat_signal = pyqtSignal(str, str)
    show_image_signal = pyqtSignal(str)
    update_workspace_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_response_text = ""
        self.shown_images = set()
        
        # Context and History
        self.conversation_history = []
        self.input_history = []
        self.history_index = -1
        self.is_recording = False
        
        self.initUI()
        self.update_chat_signal.connect(self._update_chat)
        self.show_image_signal.connect(self._show_image_popup)
        self.update_workspace_signal.connect(self.update_workspace_btn_state)
        
        self.thinking_timer = QTimer(self)
        self.thinking_timer.timeout.connect(self._animate_thinking)
        self.thinking_dots = 0
        self.is_thinking = False

        # Monitor TTS queue drain to return faces to idle after speak mode
        self.tts_monitor = QTimer(self)
        self.tts_monitor.timeout.connect(self._check_tts_done)
        self.tts_monitor.start(500)

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
        self.faces = (self.web_face,)
        # Face state the client owns (the face itself is async JS now).
        self._face_mood = "idle"
        self._face_error = False

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
        self.left_widget = left_widget       # still the fullscreen-face toggle target
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Real widgets, not a rich-text document: QTextDocument cannot do
        # border-radius, and its table cells take a fixed percentage width
        # instead of hugging short messages. See chatlog.py.
        self.chat_display = ChatLog(T)
        self.chat_display.link_clicked.connect(self.handle_link_clicked)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("speak to the ghost…")
        self.text_input.setStyleSheet(INPUT_STYLE)
        self.text_input.returnPressed.connect(self.handle_input)
        self.text_input.installEventFilter(self)
        
        # Only the transcript lives in this container now; the input moved to
        # the bottom bar so it can share that row with the action chips.
        left_layout.addWidget(self.chat_display)

        self.fs_btn = QPushButton("◐")
        self.fs_btn.setStyleSheet(chip_style())
        self.fs_btn.setToolTip("Toggle fullscreen face")
        self.fs_btn.clicked.connect(self.toggle_fullscreen_face)

        self.switch_face_btn = QPushButton("◈")
        self.switch_face_btn.setStyleSheet(chip_style())
        self.switch_face_btn.setToolTip("Cycle face form")
        self.switch_face_btn.clicked.connect(self.toggle_face_style)

        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 0, 0, 0)
        top_right_layout.setSpacing(8)
        top_right_layout.addStretch()

        self.workspace_btn = QPushButton("◇")
        self.workspace_btn.setStyleSheet(chip_style())
        self.workspace_btn.setToolTip("Load Workspace")
        self.workspace_btn.clicked.connect(self.handle_workspace)

        top_right_layout.addWidget(self.workspace_btn)
        top_right_layout.addWidget(self.switch_face_btn)
        top_right_layout.addWidget(self.fs_btn)
        
        # Bottom row: the input keeps its left position, the action chips and
        # the status readout keep theirs on the right — now sharing one row
        # instead of sitting in two separate columns.
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(8)
        stats_layout.addWidget(self.text_input, 1)

        self.snap_btn = QPushButton("◉  SNAP")
        self.snap_btn.setStyleSheet(chip_style())
        self.snap_btn.clicked.connect(self.take_picture)

        self.ptt_btn = QPushButton("●  PTT")
        self.ptt_btn.setStyleSheet(chip_style())
        self.ptt_btn.pressed.connect(self.start_recording)
        self.ptt_btn.released.connect(self.stop_recording)

        self.tts_btn = QPushButton("◌  TTS")
        self.tts_btn.setStyleSheet(chip_style(fg=T.TEXT_DIM))
        self.tts_btn.clicked.connect(self.toggle_tts)

        stats_layout.addWidget(self.snap_btn)
        stats_layout.addWidget(self.ptt_btn)
        stats_layout.addWidget(self.tts_btn)

        self.stats_label = QLabel("⚡ --%   ··:··")
        self.stats_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-family: {T.FONT}; font-size: 18px; font-weight: bold; padding: 0 12px; letter-spacing: 1px;")
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

        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.toggle_ptt)
        
        self.tts_shortcut = QShortcut(QKeySequence("Alt+Escape"), self)
        self.tts_shortcut.activated.connect(self.toggle_tts)
        
        self.snap_shortcut = QShortcut(QKeySequence("Ctrl+Escape"), self)
        self.snap_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.snap_shortcut.activated.connect(self.take_picture)

    def update_workspace_btn_state(self):
        if not hasattr(self, 'workspace_btn'):
            return
        if not self.conversation_history:
            self.workspace_btn.setToolTip("Load Workspace")
            self.workspace_btn.setText("◇")
            self.workspace_btn.setStyleSheet(chip_style(fg=T.OK))
        else:
            self.workspace_btn.setToolTip("Save Workspace")
            self.workspace_btn.setText("◆")
            self.workspace_btn.setStyleSheet(chip_style(fg=T.ACCENT_WARM))

    def handle_workspace(self):
        options = QFileDialog.Option.DontUseNativeDialog
        dialog_style = FILEDIALOG_STYLE
        if not self.conversation_history:
            dialog = QFileDialog(self, "Load Workspace", os.path.expanduser("~"), "Zip Files (*.zip)")
            dialog.setOption(options)
            dialog.setStyleSheet(dialog_style)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                filename = dialog.selectedFiles()[0]
                asyncio.ensure_future(self._async_load_workspace(filename))
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(os.path.expanduser("~"), f"ghost_workspace_{timestamp}.zip")
            dialog = QFileDialog(self, "Save Workspace", default_path, "Zip Files (*.zip)")
            dialog.setOption(options)
            dialog.setStyleSheet(dialog_style)
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setDefaultSuffix("zip")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                filename = dialog.selectedFiles()[0]
                asyncio.ensure_future(self._async_save_workspace(filename))

    async def _async_save_workspace(self, filename):
        url = "http://eva:8000/api/workspace/save"
        headers = {"X-Ghost-Key": GHOST_API_KEY}
        payload = {"chat_history": self.conversation_history}
        self.update_chat_signal.emit("append", f"<br>{NOTE_DIM}archiving workspace…</i></div>")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    self.update_chat_signal.emit("append", f"<br>{NOTE_OK}archived → {filename}</i></div>")
                else:
                    self.update_chat_signal.emit("error", f"Save failed: HTTP {response.status_code}")
        except Exception as e:
            self.update_chat_signal.emit("error", f"Save error: {str(e)}")

    async def _async_load_workspace(self, filename):
        url = "http://eva:8000/api/workspace/load"
        headers = {"X-Ghost-Key": GHOST_API_KEY}
        self.update_chat_signal.emit("append", f"<br>{NOTE_DIM}restoring workspace…</i></div>")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(filename, 'rb') as f:
                    files = {'file': (os.path.basename(filename), f, 'application/zip')}
                    response = await client.post(url, files=files, headers=headers)
                    
                if response.status_code == 200:
                    data = response.json()
                    self.conversation_history = data.get("chat_history", [])
                    self.chat_display.clear()
                    self.chat_display.add(f"{NOTE_OK}workspace restored.</i></div>", "system")
                    for msg in self.conversation_history:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role == "user":
                            if isinstance(content, list):
                                text_part = next((item["text"] for item in content if item.get("type") == "text"), "[Image Attached]")
                                self.update_chat_signal.emit("user", (text_part))
                            else:
                                self.update_chat_signal.emit("user", (content))
                        elif role == "assistant":
                            display_content = re.sub(r'<tool_call[\s\S]*?(?:</tool_call>|$)', '', content, flags=re.IGNORECASE | re.DOTALL).strip()
                            if display_content:
                                processed_text = re.sub(
                                    r'!\[(.*?)\]\((/api/download/[^\)]+)\)',
                                    r'<br><a href="\2" style="text-decoration:none; font-size:28px;" title="View Image: \1">🖼️</a>',
                                    display_content
                                )
                                html = markdown.markdown(processed_text, extensions=['fenced_code', 'tables'])
                                self.update_chat_signal.emit("agent", html)
                                matches = re.findall(r'!\[.*?\]\((/api/download/[^\)]+)\)', display_content)
                                for image_path in matches:
                                    self.show_image_signal.emit(image_path)
                    self.update_workspace_signal.emit()
                else:
                    self.update_chat_signal.emit("error", f"Load failed: HTTP {response.status_code}")
        except Exception as e:
            self.update_chat_signal.emit("error", f"Load error: {str(e)}")

    def toggle_ptt(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Triggered when the PTT button is held down."""
        if self.is_recording:
            return
        self.is_recording = True
        self.ptt_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(255, 51, 68, 0.18);
                color: {T.REC};
                border: 1px solid {T.REC};
                border-radius: 6px;
                padding: 6px 12px;
                font-family: {T.FONT};
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """)
        self.ptt_btn.setText("●  REC")
        self.set_face_mood("listen")

        # Kill any lingering recording processes just in case
        subprocess.Popen(['pkill', 'arecord']).wait()
        
        # Start recording 16kHz mono audio to a temporary file
        self.record_proc = subprocess.Popen(
            ['arecord', '-f', 'S16_LE', '-r', '16000', '-c', '1', '/tmp/ghost_stt.wav'],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )

    def stop_recording(self):
        """Triggered when the PTT button is released."""
        if not self.is_recording:
            return
        self.is_recording = False
        self.ptt_btn.setStyleSheet(chip_style())
        self.ptt_btn.setText("●  PTT")
        
        # Stop recording
        if hasattr(self, 'record_proc') and self.record_proc:
            self.record_proc.terminate()
            self.record_proc.wait()
            
        # Trigger the async upload task
        asyncio.ensure_future(self.process_stt_audio())

    def take_picture(self):
        dialog = CameraPreviewDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_data:
            b64_img, prompt_text = dialog.result_data
            
            if not prompt_text:
                prompt_text = "I just took a picture with my camera. What do you see?"
                
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
            
            self.update_chat_signal.emit("user", (f"{prompt_text}<br><span style='color:{T.TEXT_DIM};'><i>[ optic capture attached ]</i></span>"))
            
            self.conversation_history.append({"role": "user", "content": content})
            self.update_workspace_signal.emit()
            asyncio.ensure_future(self.send_chat_request())

    async def process_stt_audio(self):
        """Uploads the audio and forwards the transcribed text to the chat."""
        if not os.path.exists('/tmp/ghost_stt.wav'):
            self.set_face_mood("idle")
            return

        self.text_input.setPlaceholderText("Transcribing audio...")
        self.text_input.setEnabled(False)

        try:
            # 60s (was 30): transcription runs on the audio node, and a long
            # held PTT plus node queueing can exceed 30s. arecord already
            # writes 16kHz mono WAV — exactly what the endpoint wants — so the
            # server-side transcode is a cheap passthrough.
            async with httpx.AsyncClient(timeout=60.0, verify=VOICE_VERIFY_TLS) as client:
                with open('/tmp/ghost_stt.wav', 'rb') as f:
                    # Standard multipart file upload format
                    files = {'file': ('ghost_stt.wav', f, 'audio/wav')}
                    response = await client.post(
                        STT_SERVER_URL, files=files,
                        headers={"X-Ghost-Key": GHOST_API_KEY})

                if response.status_code == 200:
                    data = response.json()
                    # Assuming the server returns a JSON with a 'text' key
                    text = data.get("text", "").strip()
                    if text:
                        self.text_input.setText(text)
                        # immediately send it as a message
                        self.handle_input()
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
            self.text_input.setPlaceholderText("")
            self.text_input.setEnabled(True)
            self.text_input.setFocus()

    def update_stats(self):
        now = datetime.datetime.now().strftime("%I:%M %p")
        
        bat_pct = "--"
        try:
            for ps in os.listdir("/sys/class/power_supply/"):
                if "bat" in ps.lower() or "axp" in ps.lower():
                    cap_path = f"/sys/class/power_supply/{ps}/capacity"
                    if os.path.exists(cap_path):
                        with open(cap_path, 'r') as f:
                            bat_pct = f.read().strip()
                        break
        except Exception:
            pass

        self.stats_label.setText(f"⚡ {bat_pct}%   {now}")

    def eventFilter(self, obj, event):
        if obj == self.text_input and event.type() == QEvent.Type.KeyPress:
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
        if not self.text_input.hasFocus() and len(event.text()) > 0 and event.text().isprintable():
            self.text_input.setFocus()
            QApplication.sendEvent(self.text_input, event)
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
        if self.tts_enabled:
            self.tts_btn.setText("◉  TTS")
            self.tts_btn.setStyleSheet(chip_style(fg=T.OK))
        else:
            self.tts_btn.setText("◌  TTS")
            self.tts_btn.setStyleSheet(chip_style(fg=T.TEXT_DIM))
            # Clear queue immediately
            subprocess.Popen(['pkill', 'aplay'])
            while not audio_queue.empty():
                try: audio_queue.get_nowait(); audio_queue.task_done()
                except: pass
            while not playback_queue.empty():
                try: playback_queue.get_nowait(); playback_queue.task_done()
                except: pass

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

    def toggle_face_style(self):
        """Cycle the face FORM (vortex → cortex → lattice → …).

        This used to swap between three separate QPainter renderers. Those are
        gone; the web face carries the same eight forms the browser has, so the
        button now walks that list and the two clients stay in step.
        """
        try:
            self.web_face.cycle_form()
        except Exception:  # noqa: BLE001
            pass

    def toggle_fullscreen_face(self):
        if self.left_widget.isVisible():
            self.left_widget.hide()
            self.fs_btn.setText("📖")
        else:
            self.left_widget.show()
            self.fs_btn.setText("👁️")
            self.text_input.setFocus()

    def handle_input(self):
        text = self.text_input.text().strip()
        if not text:
            return
            
        if text.startswith('/clear'):
            self.conversation_history.clear()
            self.update_workspace_signal.emit()
            self.chat_display.clear()
            self.chat_display.add(f"{NOTE_WARN}context wiped.</i></div>", "system")
            self.text_input.clear()
            self.set_face_mood("idle")
            
            while not audio_queue.empty():
                try: audio_queue.get_nowait(); audio_queue.task_done()
                except: pass
            while not playback_queue.empty():
                try: playback_queue.get_nowait(); playback_queue.task_done()
                except: pass
            subprocess.Popen(['pkill', 'aplay'])
            return

        if text.startswith('/shutdown'):
            self.update_chat_signal.emit("append", f"{NOTE_WARN}powering down hardware…</i></div>")
            self.text_input.clear()
            subprocess.Popen(['sudo', 'shutdown', '-h', 'now'])
            return

        if text.startswith('/reboot'):
            self.update_chat_signal.emit("append", f"{NOTE_WARN}rebooting hardware…</i></div>")
            self.text_input.clear()
            subprocess.Popen(['sudo', 'reboot'])
            return

        if text.startswith('/exit'):
            self.update_chat_signal.emit("append", f"{NOTE_WARN}detaching from cyberdeck…</i></div>")
            self.text_input.clear()
            QApplication.quit()
            return

        while not audio_queue.empty():
            try: audio_queue.get_nowait(); audio_queue.task_done()
            except: pass
        while not playback_queue.empty():
            try: playback_queue.get_nowait(); playback_queue.task_done()
            except: pass
        subprocess.Popen(['pkill', 'aplay'])

        self.input_history.append(text)
        self.history_index = -1
        self.text_input.clear()

        self.update_chat_signal.emit("user", (text))

        self.conversation_history.append({"role": "user", "content": text})
        self.web_face.wake()
        self.update_workspace_signal.emit()
        
        asyncio.ensure_future(self.send_chat_request())

    async def send_chat_request(self):
        url = "http://eva:8000/api/chat"
        headers = {
            "X-Ghost-Key": GHOST_API_KEY
        }
        # Get the text directly from the last user input
        text = self.conversation_history[-1]["content"] if self.conversation_history else ""
        payload = {
            # model omitted on purpose — the agent uses its configured model;
            # pinning a name here 404s (ModelNotFound) whenever the model is upgraded
            "messages": self.conversation_history,
            "stream": True
        }
        
        self.update_chat_signal.emit("start_response", "")
        self._face_error = False
        self.set_face_mood("think")
        self.tts_buffer = ""
        
        try:
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        self.update_chat_signal.emit("error", f"HTTP {response.status_code}")
                        return

                    async for chunk in response.aiter_text():
                        if chunk.startswith("data: "):
                            data_str = chunk[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                content = data.get("message", {}).get("content", "")
                                if not content and "choices" in data:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    
                                if content:
                                    self.update_chat_signal.emit("update_response", content)
                                    self.web_face.pulse()
                                    # Network auto-spawns its own pulses in think mode;
                                    # just feed it a token-activity signal instead of
                                    # stacking extra full MoE cascades on every token.
                                    self.web_face.feed_audio(0.5)
                                    self.tts_buffer += content
                                    
                                    match = re.search(r'([.?!]+[\s\n]+)', self.tts_buffer)
                                    while match:
                                        split_idx = match.end()
                                        sentence = self.tts_buffer[:split_idx].strip()
                                        if sentence:
                                            clean = re.sub(r'!\[.*?\]\(.*?\)', '', sentence)
                                            clean = re.sub(r'[*`_#]', '', clean)
                                            if clean.strip() and self.tts_enabled:
                                                audio_queue.put_nowait(clean.strip())
                                        self.tts_buffer = self.tts_buffer[split_idx:]
                                        match = re.search(r'([.?!]+[\s\n]+)', self.tts_buffer)
                            except json.JSONDecodeError:
                                pass
                                
            final_sentence = self.tts_buffer.strip()
            if final_sentence:
                clean = re.sub(r'!\[.*?\]\(.*?\)', '', final_sentence)
                clean = re.sub(r'[*`_#]', '', clean)
                if clean.strip() and self.tts_enabled:
                    audio_queue.put_nowait(clean.strip())
                                
            self.conversation_history.append({"role": "assistant", "content": self.current_response_text})
            self.update_workspace_signal.emit()
            
        except Exception as e:
            self.web_face.startle()
            self._face_error = True
            self.update_chat_signal.emit("error", f"{type(e).__name__}: {str(e)}")
        finally:
            self.update_chat_signal.emit("stop_thinking", "")
            if not self._face_error:
                if self.tts_enabled and (not audio_queue.empty() or not playback_queue.empty()):
                    self.set_face_mood("speak")
                else:
                    self.set_face_mood("idle")

    def _animate_thinking(self):
        if getattr(self, 'is_thinking', False):
            self.thinking_dots = (self.thinking_dots % 3) + 1
            self._render_thinking()

    def _render_thinking(self):
        dots = "·" * self.thinking_dots
        self.chat_display.update_agent(
            f"<span style='color:{T.TEXT_DIM};'><i>cogitating {dots}</i></span>")

    def _close_thinking(self):
        """Stop the dots and discard the bubble if nothing ever arrived.

        The placeholder holds "cogitating …", so it is never literally empty —
        it has to be blanked before the log can decide to drop it.
        """
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
            self.thinking_dots = 1
            self._render_thinking()
            self.thinking_timer.start(500)
        elif action == "update_response":
            if getattr(self, 'is_thinking', False):
                self.is_thinking = False
                self.thinking_timer.stop()
            self.current_response_text += data

            processed_text = re.sub(
                r'!\[(.*?)\]\((/api/download/[^\)]+)\)',
                r'<br><a href="\2" style="text-decoration:none; font-size:28px;" title="View Image: \1">🖼️</a>',
                self.current_response_text
            )
            html = markdown.markdown(processed_text, extensions=['fenced_code', 'tables'])
            self.chat_display.update_agent(html)

            matches = re.findall(r'!\[.*?\]\((/api/download/[^\)]+)\)', self.current_response_text)
            for image_path in matches:
                self.show_image_signal.emit(image_path)

        elif action == "stop_thinking":
            self._close_thinking()
            return
        elif action == "error":
            self._close_thinking()
            self.chat_display.add(
                f"<span style='color:{T.DANGER};'>fault → {data}</span>", "system")

    def _show_image_popup(self, image_path):
        if image_path in self.shown_images:
            return
        self.shown_images.add(image_path)
        asyncio.ensure_future(self._download_and_show_image(image_path))

    async def _download_and_show_image(self, image_path):
        url = f"http://eva:8000{image_path}"
        headers = {"X-Ghost-Key": GHOST_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.get(url, headers=headers)
                if r.status_code == 200:
                    pixmap = QPixmap()
                    pixmap.loadFromData(r.content)
                    self._display_image_dialog(pixmap)
        except Exception as e:
            print(f"Image fetch failed: {e}")

    def _display_image_dialog(self, pixmap):
        dialog = ImageViewer(pixmap, self)
        dialog.show()

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
                resp = await client.post(
                    TTS_SERVER_URL, json=payload, timeout=60.0,
                    headers={"X-Ghost-Key": GHOST_API_KEY})
                if resp.status_code == 200:
                    # audio/wav from the macOS synthesiser; `aplay -q -` reads
                    # a WAV header off stdin, same as the old Piper output.
                    await playback_queue.put(resp.content)
                else:
                    print(f"TTS Fetch Err: HTTP {resp.status_code} "
                          f"{(resp.text or '')[:160]}")
            except Exception as e:
                print(f"TTS Fetch Err: {e}")
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
            print(f"TTS Play Err: {e}")
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
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindow()
    window.show()
    
    loop.create_task(audio_fetch_task())
    loop.create_task(audio_worker_task())

    with loop:
        loop.run_forever()
        