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

from face import NeuralFaceWidget as MatrixFaceWidget, FluidFaceWidget, NetworkFaceWidget

audio_queue = asyncio.Queue()
playback_queue = asyncio.Queue()

# Adjust these URLs if the TTS/STT services are hosted centrally on ghost or remotely:
TTS_SERVER_URL = "http://192.168.0.24:8000/tts"
STT_SERVER_URL = "http://192.168.0.24:8000/stt"


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
    BG          = "#04060e"
    BG_PANEL    = "#080c18"
    BG_INPUT    = "#0c1224"
    BORDER      = "#172240"
    BORDER_HOT  = "#2a3a60"
    TEXT        = "#e6edf8"
    TEXT_DIM    = "#7e94b8"
    USER        = "#ffb86b"
    ASSISTANT   = "#e6edf8"
    ACCENT      = "#7be0ff"
    ACCENT_WARM = "#ffb86b"
    OK          = "#88ff9c"
    DANGER      = "#ff5a6e"
    REC         = "#ff3344"
    SCROLL      = "#1a2440"
    SCROLL_HOT  = "#2c3e6e"
    FONT        = "'Fira Code', 'JetBrains Mono', 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', monospace"


def chip_style(fg=T.TEXT_DIM, border=T.BORDER, hover=T.BG_INPUT):
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {fg};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 6px 12px;
            font-family: {T.FONT};
            font-size: 18px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }}
        QPushButton:hover {{
            background-color: {hover};
            color: {T.ACCENT};
            border: 1px solid {T.ACCENT};
        }}
        QPushButton:pressed {{
            background-color: {T.BORDER};
        }}
    """


def chip_style_hot(fg, border):
    return f"""
        QPushButton {{
            background-color: rgba(255, 90, 110, 0.12);
            color: {fg};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 6px 12px;
            font-family: {T.FONT};
            font-size: 18px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }}
    """


CHAT_STYLE = f"""
    QTextBrowser {{
        background-color: {T.BG_PANEL};
        color: {T.TEXT};
        border: 1px solid {T.BORDER};
        border-radius: 10px;
        padding: 18px;
        font-family: {T.FONT};
        font-size: 22px;
        selection-background-color: {T.BORDER_HOT};
    }}
    QScrollBar:vertical {{
        border: none;
        background: {T.BG_PANEL};
        width: 12px;
        margin: 4px 4px 4px 0px;
        border-radius: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {T.SCROLL};
        min-height: 40px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {T.SCROLL_HOT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px; border: none; background: none;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}
"""

INPUT_STYLE = f"""
    QLineEdit {{
        background-color: {T.BG_INPUT};
        color: {T.TEXT};
        border: 1px solid {T.BORDER};
        border-radius: 10px;
        padding: 14px 16px;
        font-family: {T.FONT};
        font-size: 22px;
        selection-background-color: {T.BORDER_HOT};
    }}
    QLineEdit:focus {{
        border: 1px solid {T.ACCENT};
        background-color: #0d142a;
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

USER_BUBBLE_PREFIX = f"<br><div style='color:{T.USER};'><b>&gt; </b>"
ASSISTANT_BUBBLE_PREFIX = f"<br><div style='color:{T.ACCENT};'><b>&lt;&lt; </b>"
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
        self.response_start_position = 0
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
        self.setFixedSize(screen_geometry.width(), screen_geometry.height())
        self.move(0, 0)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"background-color: {T.BG};")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_widget = QWidget()
        self.left_widget = left_widget
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.setOpenLinks(False)
        self.chat_display.anchorClicked.connect(self.handle_link_clicked)
        self.chat_display.setStyleSheet(CHAT_STYLE)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("speak to the ghost…")
        self.text_input.setStyleSheet(INPUT_STYLE)
        self.text_input.returnPressed.connect(self.handle_input)
        self.text_input.installEventFilter(self)
        
        left_layout.addWidget(self.chat_display)
        left_layout.addWidget(self.text_input, stretch=0)
        
        # Wrapper for Right Panel to allow bottom bar
        right_widget = QWidget()
        right_widget.setStyleSheet(f"background-color: {T.BG};")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.stacked_widget = QStackedWidget()

        self.fluid_face = FluidFaceWidget()
        self.network_face = NetworkFaceWidget()
        self.matrix_face = MatrixFaceWidget()

        self.stacked_widget.addWidget(self.network_face)  # 0 — MoE Network (default)
        self.stacked_widget.addWidget(self.fluid_face)    # 1 — Smoke Oracle
        self.stacked_widget.addWidget(self.matrix_face)   # 2 — Iris

        self.stacked_widget.setCurrentIndex(0)

        self.fs_btn = QPushButton("◐")
        self.fs_btn.setStyleSheet(chip_style())
        self.fs_btn.setToolTip("Toggle fullscreen face")
        self.fs_btn.clicked.connect(self.toggle_fullscreen_face)

        self.switch_face_btn = QPushButton("◈")
        self.switch_face_btn.setStyleSheet(chip_style())
        self.switch_face_btn.setToolTip("Switch face")
        self.switch_face_btn.clicked.connect(self.toggle_face_style)

        top_right_layout = QHBoxLayout()
        top_right_layout.setContentsMargins(0, 15, 20, 0)
        top_right_layout.addStretch()

        self.workspace_btn = QPushButton("◇")
        self.workspace_btn.setStyleSheet(chip_style())
        self.workspace_btn.setToolTip("Load Workspace")
        self.workspace_btn.clicked.connect(self.handle_workspace)

        top_right_layout.addWidget(self.workspace_btn)
        top_right_layout.addWidget(self.switch_face_btn)
        top_right_layout.addWidget(self.fs_btn)
        
        right_layout.addLayout(top_right_layout)
        right_layout.addWidget(self.stacked_widget, 1)
        
        # Discreet Bottom Right Clock/Battery/TTS
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(0, 0, 20, 5)
        
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

        stats_layout.addStretch()
        stats_layout.addWidget(self.snap_btn)
        stats_layout.addWidget(self.ptt_btn)
        stats_layout.addWidget(self.tts_btn)

        self.stats_label = QLabel("⚡ --%   ··:··")
        self.stats_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-family: {T.FONT}; font-size: 18px; font-weight: bold; padding: 0 12px; letter-spacing: 1px;")
        stats_layout.addWidget(self.stats_label)
        
        right_layout.addLayout(stats_layout)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)

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
                    self.chat_display.insertHtml(f"{NOTE_OK}workspace restored.</i></div><br>")
                    for msg in self.conversation_history:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role == "user":
                            if isinstance(content, list):
                                text_part = next((item["text"] for item in content if item.get("type") == "text"), "[Image Attached]")
                                self.update_chat_signal.emit("append", f"{USER_BUBBLE_PREFIX}{text_part}</div>")
                            else:
                                self.update_chat_signal.emit("append", f"{USER_BUBBLE_PREFIX}{content}</div>")
                        elif role == "assistant":
                            display_content = re.sub(r'<tool_call[\s\S]*?(?:</tool_call>|$)', '', content, flags=re.IGNORECASE | re.DOTALL).strip()
                            if display_content:
                                processed_text = re.sub(
                                    r'!\[(.*?)\]\((/api/download/[^\)]+)\)',
                                    r'<br><a href="\2" style="text-decoration:none; font-size:28px;" title="View Image: \1">🖼️</a>',
                                    display_content
                                )
                                html = markdown.markdown(processed_text, extensions=['fenced_code', 'tables'])
                                styled_html = f"<div style='color:{T.TEXT};'>{html}</div>"
                                self.update_chat_signal.emit("append", f"{ASSISTANT_BUBBLE_PREFIX}{styled_html}</div>")
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
        self.fluid_face.set_mood("listen")
        self.network_face.set_mood("listen")
        self.matrix_face.set_mood("listen")

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
            
            self.update_chat_signal.emit("append", f"{USER_BUBBLE_PREFIX}{prompt_text}<br><span style='color:{T.TEXT_DIM};'><i>[ optic capture attached ]</i></span></div>")
            
            self.conversation_history.append({"role": "user", "content": content})
            self.update_workspace_signal.emit()
            asyncio.ensure_future(self.send_chat_request())

    async def process_stt_audio(self):
        """Uploads the audio and forwards the transcribed text to the chat."""
        if not os.path.exists('/tmp/ghost_stt.wav'):
            self.fluid_face.set_mood("idle")
            self.network_face.set_mood("idle")
            self.matrix_face.set_mood("idle")
            return

        self.text_input.setPlaceholderText("Transcribing audio...")
        self.text_input.setEnabled(False)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open('/tmp/ghost_stt.wav', 'rb') as f:
                    # Standard multipart file upload format
                    files = {'file': ('ghost_stt.wav', f, 'audio/wav')}
                    response = await client.post(STT_SERVER_URL, files=files)

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
                        self.fluid_face.set_mood("idle")
                        self.network_face.set_mood("idle")
                        self.matrix_face.set_mood("idle")
                else:
                    self.update_chat_signal.emit("error", f"STT failed: HTTP {response.status_code}")
                    self.fluid_face.set_mood("idle")
                    self.network_face.set_mood("idle")
                    self.matrix_face.set_mood("idle")
        except Exception as e:
            self.update_chat_signal.emit("error", f"STT Error: {str(e)}")
            self.fluid_face.set_mood("idle")
            self.network_face.set_mood("idle")
            self.matrix_face.set_mood("idle")
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
        link = url.toString()
        if link.startswith("/api/download/"):
            asyncio.ensure_future(self._download_and_show_image(link))
        else:
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(url)

    def _check_tts_done(self):
        """Poll TTS queues; when both drain and faces are still in speak, go idle."""
        if (self.network_face.target_mood.name == "speak"
                and audio_queue.empty() and playback_queue.empty()):
            self.fluid_face.set_mood("idle")
            self.network_face.set_mood("idle")
            self.matrix_face.set_mood("idle")

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

    def toggle_face_style(self):
        idx = (self.stacked_widget.currentIndex() + 1) % self.stacked_widget.count()
        self.stacked_widget.setCurrentIndex(idx)

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
            self.chat_display.insertHtml(f"{NOTE_WARN}context wiped.</i></div><br>")
            self.text_input.clear()
            self.fluid_face.set_mood("idle")
            self.network_face.set_mood("idle")
            self.matrix_face.set_mood("idle")
            
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

        self.update_chat_signal.emit("append", f"{USER_BUBBLE_PREFIX}{text}</div>")

        self.conversation_history.append({"role": "user", "content": text})
        self.fluid_face.wake()
        self.network_face.wake()
        self.matrix_face.wake()
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
            "model": "qwen",
            "messages": self.conversation_history,
            "stream": True
        }
        
        self.update_chat_signal.emit("start_response", "")
        self.fluid_face.set_mood("think")
        self.network_face.set_mood("think")
        self.matrix_face.set_mood("think")
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
                                    self.fluid_face.pulse()
                                    self.matrix_face.pulse()
                                    # Network auto-spawns its own pulses in think mode;
                                    # just feed it a token-activity signal instead of
                                    # stacking extra full MoE cascades on every token.
                                    self.network_face.feed_audio(0.5)
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
            self.fluid_face.startle()
            self.network_face.startle()
            self.matrix_face.startle()
            self.update_chat_signal.emit("error", f"{type(e).__name__}: {str(e)}")
        finally:
            self.update_chat_signal.emit("stop_thinking", "")
            if not self.matrix_face.state_error:
                if self.tts_enabled and (not audio_queue.empty() or not playback_queue.empty()):
                    self.fluid_face.set_mood("speak")
                    self.network_face.set_mood("speak")
                    self.matrix_face.set_mood("speak")
                else:
                    self.fluid_face.set_mood("idle")
                    self.network_face.set_mood("idle")
                    self.matrix_face.set_mood("idle")

    def _animate_thinking(self):
        if getattr(self, 'is_thinking', False):
            self.thinking_dots = (self.thinking_dots % 3) + 1
            self._render_thinking()

    def _render_thinking(self):
        cursor = self.chat_display.textCursor()
        cursor.setPosition(self.response_start_position)
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        dots = "·" * self.thinking_dots
        styled_html = f"<div style='color:{T.TEXT_DIM};'><i>cogitating {dots}</i></div>"
        cursor.insertHtml(styled_html)

    def _update_chat(self, action, data):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        
        if action == "append":
            self.chat_display.insertHtml(data)
        elif action == "start_response":
            self.chat_display.insertHtml(f"{ASSISTANT_BUBBLE_PREFIX}</div>")
            self.response_start_position = self.chat_display.textCursor().position()
            self.current_response_text = ""
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
            
            # Wraps the markdown in a div to constraint formatting
            styled_html = f"<div style='color:{T.TEXT};'>{html}</div>"
            
            cursor.setPosition(self.response_start_position)
            cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
            cursor.insertHtml(styled_html)

            matches = re.findall(r'!\[.*?\]\((/api/download/[^\)]+)\)', self.current_response_text)
            for image_path in matches:
                self.show_image_signal.emit(image_path)
                
        elif action == "stop_thinking":
            if getattr(self, 'is_thinking', False):
                self.is_thinking = False
                self.thinking_timer.stop()
                if not self.current_response_text:
                    cursor.setPosition(self.response_start_position)
                    cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
                    cursor.insertText("")
            return
        elif action == "error":
            if getattr(self, 'is_thinking', False):
                self.is_thinking = False
                self.thinking_timer.stop()
                if not self.current_response_text:
                    cursor.setPosition(self.response_start_position)
                    cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
                    cursor.insertText("")
            self.chat_display.insertHtml(f"<br>{NOTE_ERR}fault → {data}</i></div><br>")
            
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

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
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                text_chunk = await audio_queue.get()
                if not text_chunk:
                    audio_queue.task_done()
                    continue
                
                payload = {"text": text_chunk}
                resp = await client.post(TTS_SERVER_URL, json=payload, timeout=60.0)
                if resp.status_code == 200:
                    await playback_queue.put(resp.content)
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
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindow()
    window.show()
    
    loop.create_task(audio_fetch_task())
    loop.create_task(audio_worker_task())

    with loop:
        loop.run_forever()
        