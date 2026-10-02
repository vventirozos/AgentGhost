"""Chat transcript as real widgets — one QLabel per message.

**Why this replaced the QTextBrowser.** The transcript used to be a single
rich-text document with each message drawn as an HTML table. That worked, but
Qt's rich-text engine (``QTextDocument``) supports no ``border-radius`` at all,
so every bubble was square, and table cells take a fixed percentage width
rather than hugging their content — a one-word reply got the same 54% slab as a
paragraph.

Real widgets fix both: a QLabel honours the full stylesheet (per-corner radii,
rgba fills, per-side accent rails) and, with ``setMaximumWidth`` plus a
content-hugging size policy, a short message occupies exactly as much room as
it needs while a long one wraps at the cap.

Layout mirrors the web UI: the operator's messages align RIGHT, the agent's
LEFT, system notes centre, and everything floats over the face with nothing
opaque behind it.

The typography (``style_markup``) lives in ``markup.py``, which is Qt-free and
therefore unit-tested; what is left here is the part only a real Qt can check,
and ``device_probe.py`` checks it on the handheld at every deploy.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from markup import SANS, soften_long_tokens, style_markup

# Qt's QWIDGETSIZE_MAX. Spelled out because PyQt6 does not export the macro on
# every build, and a missing name here would be an ImportError at startup.
_NO_MAX = 16777215


def _ratio(name: str, default: float) -> float:
    try:
        return max(0.3, min(0.95, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


# How much of the window's width a bubble may take. The agent's is wider than
# the operator's: its messages are the long ones, and at 0.56 a briefing left
# nearly half the panel empty and took twice the scrolling. Both are env knobs
# (tune on the device, no redeploy).
AGENT_RATIO = _ratio("GHOST_BUBBLE_AGENT", 0.70)
USER_RATIO = _ratio("GHOST_BUBBLE_USER", 0.56)

# Within this many pixels of the end counts as "at the bottom".
STICK_PX = 28


class _Bubble(QLabel):
    """One message. Rich text, wraps, hugs its content up to a max width.

    Getting "hug the content, but never exceed the cap" out of a QLabel needs
    an explicit width, and this is the part that is not obvious: a
    word-wrapped QLabel reports a SMALL ``sizeHint().width()`` no matter what
    ``maximumWidth`` says, so any size policy that defers to the hint (Maximum,
    Preferred) leaves every bubble as a narrow ribbon — raising the cap changes
    nothing, because the cap was never the binding constraint.

    :meth:`fit` therefore measures the text unwrapped, then pins minimum ==
    maximum width to ``min(natural, cap)``, which forces the layout to grant
    exactly that.
    """

    def __init__(self, html: str, style: str, max_width: int, parent=None):
        super().__init__(html, parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setOpenExternalLinks(False)
        self.setStyleSheet(style)
        self.fit(max_width)

    def fit(self, cap: int) -> None:
        """Size to the content, bounded by `cap`.

        ⚠ The pin from the PREVIOUS fit must be released before measuring.
        ``QLabel.sizeHint()`` is expanded to the widget's ``minimumSize()``, so
        with the old pin in place the "natural" width could never come back
        smaller than it — a bubble could grow and never shrink. Every reply
        opens as a waiting caption (~540 px) and so every short answer stayed
        540 px wide: measured on the device 2026-10-01, "Hello!" was 540 px
        after a caption and 120 px in a fresh bubble.
        """
        self.setMinimumWidth(0)
        self.setMaximumWidth(_NO_MAX)
        self.setWordWrap(False)
        natural = self.sizeHint().width()      # width if it never wrapped
        self.setWordWrap(True)
        target = max(120, min(natural, cap))
        self.setMinimumWidth(target)
        self.setMaximumWidth(target)
        self.updateGeometry()


class ChatLog(QScrollArea):
    """Scrolling transcript of bubbles. Drop-in for the old chat_display."""

    link_clicked = pyqtSignal(str)

    def __init__(self, palette, max_width_ratio: float | None = None, parent=None):
        super().__init__(parent)
        self.T = palette
        # An explicit ratio pins BOTH roles (the old single-knob behaviour).
        self._agent_ratio = max_width_ratio or AGENT_RATIO
        self._user_ratio = max_width_ratio or USER_RATIO
        self._current_agent: _Bubble | None = None
        # True while the view follows new content. Cleared by scrolling up,
        # set again by returning to the bottom (or by sending a message).
        self._stick = True

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        # The transcript is scrolled from the keyboard by the client (the
        # input keeps the focus); it must not take focus itself.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # The container must be transparent all the way down — a scroll area
        # paints its viewport, so styling only the QScrollArea leaves an
        # opaque rectangle over the face.
        self.setStyleSheet(f"""
            QScrollArea, QScrollArea > QWidget > QWidget {{
                background: transparent; border: none;
            }}
            QScrollBar:vertical {{
                border: none; background: transparent; width: 8px; margin: 4px 0 4px 0;
            }}
            QScrollBar::handle:vertical {{
                background: {palette.SCROLL}; min-height: 48px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {palette.SCROLL_HOT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px; border: none; background: none;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        """)
        self.viewport().setAutoFillBackground(False)

        self._body = QWidget()
        self._body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._col = QVBoxLayout(self._body)
        self._col.setContentsMargins(2, 2, 8, 2)
        self._col.setSpacing(10)
        # Push the first message to the bottom so a short conversation sits
        # near the input instead of floating at the top of a tall empty area.
        self._col.addStretch(1)
        self.setWidget(self._body)

        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scroll_value)
        bar.rangeChanged.connect(self._on_scroll_range)

    # ── styling ──────────────────────────────────────────────────────────
    def _max_width(self, role: str = "agent") -> int:
        """Width cap for a bubble.

        Falls back to the SCREEN width when the scroll area has not been laid
        out yet. Without that fallback every bubble added before the first
        show() got the minimum cap and wrapped into a narrow ribbon — and it
        never recovered, because a wrapped QLabel's size hint is computed from
        the width it wrapped at.
        """
        w = self.width()
        if w < 400:
            screen = QApplication.primaryScreen()
            w = screen.geometry().width() if screen else 1280
        ratio = self._user_ratio if role == "user" else self._agent_ratio
        return max(320, int(w * ratio))

    def _style(self, role: str) -> str:
        T = self.T
        # Padding lives here: 13/17 reads as a bubble rather than a slab.
        # The operator's own words stay MONOSPACE (they are commands, and it
        # matches the input field they were typed into); the agent's prose is
        # SANS, because a briefing in monospace is what made long answers look
        # wrong in the first place.
        pad = "padding: 13px 17px;"
        if role == "user":
            base = f"{pad} font-family: {T.FONT}; font-size: 19px;"
            return (f"QLabel {{ {base} color: {T.TEXT};"
                    f" background-color: rgba(46, 30, 20, 0.42);"
                    f" border: 1px solid rgba(255, 192, 138, 0.22);"
                    f" border-right: 2px solid {T.USER};"
                    # Notched on the speaker's corner, exactly like the web UI.
                    f" border-top-left-radius: 16px; border-top-right-radius: 4px;"
                    f" border-bottom-left-radius: 16px; border-bottom-right-radius: 16px; }}")
        if role == "agent":
            base = f"{pad} font-family: {SANS}; font-size: 20px;"
            return (f"QLabel {{ {base} color: {T.TEXT};"
                    f" background-color: rgba(26, 16, 44, 0.42);"
                    f" border: 1px solid rgba(201, 166, 255, 0.20);"
                    f" border-left: 2px solid {T.ACCENT};"
                    f" border-top-left-radius: 4px; border-top-right-radius: 16px;"
                    f" border-bottom-left-radius: 16px; border-bottom-right-radius: 16px; }}")
        return (f"QLabel {{ padding: 4px 10px; font-family: {SANS}; font-size: 16px;"
                f" color: {T.TEXT_DIM}; background: transparent; border: none; }}")

    def _styled(self, html: str) -> str:
        return style_markup(html, self.T.FONT, self.T.ACCENT, self.T.TEXT_DIM)

    # ── public API ───────────────────────────────────────────────────────
    def add(self, html: str, role: str = "system") -> _Bubble:
        # The operator's text is already escaped by the caller; it still needs
        # break opportunities, or a pasted URL runs off the bubble.
        html = soften_long_tokens(html) if role == "user" else self._styled(html)
        bubble = _Bubble(html, self._style(role), self._max_width(role))
        bubble._role = role
        bubble.linkActivated.connect(self.link_clicked.emit)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if role == "user":
            row.addStretch(1)
            row.addWidget(bubble)
        elif role == "agent":
            row.addWidget(bubble)
            row.addStretch(1)
        else:
            row.addStretch(1)
            row.addWidget(bubble)
            row.addStretch(1)
        # insert before the trailing stretch that bottom-aligns the log
        self._col.insertLayout(self._col.count() - 1, row)
        bubble._row = row          # end_agent() removes exactly this row
        if role == "user":
            # Sending is "take me to the end" — whatever was being re-read.
            self.scroll_to_end()
        return bubble

    def start_agent(self, placeholder: str = "") -> None:
        """Open a streaming agent bubble; `update_agent` fills it."""
        self._current_agent = self.add(placeholder, "agent")

    def update_agent(self, html: str) -> None:
        if self._current_agent is None:
            self.start_agent()
        self._current_agent.setText(self._styled(html))
        # Re-fit on every token: a reply that ends up short must not be left
        # in a bubble sized for the longest line it briefly had.
        self._current_agent.fit(self._max_width("agent"))

    def end_agent(self, drop_if_empty: bool = True) -> None:
        """Close the streaming bubble, discarding it if nothing arrived."""
        bubble = self._current_agent
        self._current_agent = None
        if bubble is not None and drop_if_empty and not bubble.text().strip():
            # ⚠ Remove the bubble's OWN row. This used to look the row up as
            # `bubble.parentWidget().layout()` — which is not the row at all
            # but the transcript's whole column layout — and then call
            # `setParent(None)` on it. The next message added to the log
            # SEGFAULTED the client (reproduced on the device against the
            # build then live, 2026-10-01). And this path runs whenever a turn
            # ends with no text: the agent unreachable, an HTTP error, a
            # refused key — exactly when the operator most needs the client
            # to stay up and say what went wrong.
            row = getattr(bubble, "_row", None)
            if row is not None:
                self._col.removeItem(row)
                while row.count():
                    row.takeAt(0)
                row.deleteLater()
            bubble.setParent(None)
            bubble.deleteLater()

    def has_open_agent(self) -> bool:
        return self._current_agent is not None

    def clear(self) -> None:
        self._current_agent = None
        self._stick = True
        while self._col.count() > 1:          # keep the trailing stretch
            item = self._col.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
                item.layout().setParent(None)

    # ── scrolling ────────────────────────────────────────────────────────
    # The view FOLLOWS new content only while the operator is at the bottom.
    # It used to jump to the end on every streamed token, so scrolling up to
    # re-read something during a long reply was undone a few times a second.
    def _on_scroll_value(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - STICK_PX

    def _on_scroll_range(self, _lo: int, hi: int) -> None:
        # The content grew (or shrank). A range change does not move the
        # value, so `_stick` still says where the operator was BEFORE it.
        if self._stick:
            self.verticalScrollBar().setValue(hi)

    def is_following(self) -> bool:
        return self._stick

    def scroll_to_end(self) -> None:
        self._stick = True
        # Deferred as well: the layout has not resized yet when a bubble is
        # added, so scrolling immediately lands short of the true bottom.
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def scroll_page(self, direction: int) -> None:
        """Scroll most of a screenful: -1 up, +1 down (PgUp / PgDn)."""
        bar = self.verticalScrollBar()
        step = max(60, int(bar.pageStep() * 0.85))
        bar.setValue(bar.value() + (step if direction > 0 else -step))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for bubble in self._body.findChildren(_Bubble):
            bubble.fit(self._max_width(getattr(bubble, "_role", "agent")))
