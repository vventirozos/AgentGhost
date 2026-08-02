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
"""

from __future__ import annotations

import re

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)


# Prose is set in a SANS face; only the operator's own input, inline code and
# code blocks stay monospace. This mirrors the web UI, where agent messages
# inherit the sans body font and `.message.user` overrides to mono — reading a
# briefing in 21px monospace is what made long answers feel wrong.
SANS = "'DejaVu Sans', 'Liberation Sans', 'Cantarell', 'Noto Sans', sans-serif"

# Markdown → HTML arrives as bare <h2>/<p>/<ol>, and Qt's rich text applies its
# OWN defaults to those: <h1> renders at roughly 2x the base size and <h2> at
# 1.5x, so a briefing's headings came out enormous next to 21px body text.
# QLabel gives no hook for a document stylesheet, so the styles are injected
# inline, per tag. Headings sit only slightly above body size — in a chat
# bubble a heading is a label, not a page title.
_TAG_STYLES = {
    "h1": "font-size:22px; font-weight:700; margin:12px 0 6px 0;",
    "h2": "font-size:21px; font-weight:700; margin:11px 0 5px 0;",
    "h3": "font-size:20px; font-weight:700; margin:10px 0 4px 0;",
    "h4": "font-size:19px; font-weight:700; margin:9px 0 4px 0;",
    "h5": "font-size:19px; font-weight:600; margin:8px 0 3px 0;",
    "h6": "font-size:19px; font-weight:600; margin:8px 0 3px 0;",
    "p": "margin:0 0 9px 0; line-height:148%;",
    "ul": "margin:2px 0 9px 0; -qt-list-indent:1;",
    "ol": "margin:2px 0 9px 0; -qt-list-indent:1;",
    "li": "margin:0 0 4px 0; line-height:145%;",
    "blockquote": "margin:6px 0 8px 6px; padding-left:11px;",
    "hr": "margin:10px 0;",
    "table": "margin:6px 0 9px 0;",
    "th": "padding:3px 11px 3px 0; font-weight:700;",
    "td": "padding:3px 11px 3px 0;",
}
_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)((?:\s[^>]*?)?)(/?)>")


def style_markup(html: str, mono_font: str, accent: str, dim: str) -> str:
    """Give markdown-generated HTML sane, chat-sized typography."""
    code_style = (f"font-family:{mono_font}; font-size:18px;"
                  f" background-color:rgba(0,0,0,0.30);")
    per_tag = dict(_TAG_STYLES)
    per_tag["code"] = code_style + " padding:0 3px;"
    per_tag["pre"] = (f"font-family:{mono_font}; font-size:17px;"
                      f" background-color:rgba(0,0,0,0.32); margin:7px 0 9px 0;")
    per_tag["a"] = f"color:{accent}; text-decoration:none;"
    per_tag["blockquote"] = (_TAG_STYLES["blockquote"] + f" color:{dim};"
                             f" border-left:2px solid {accent};")

    def _inject(m):
        tag, attrs, close = m.group(1).lower(), m.group(2) or "", m.group(3)
        style = per_tag.get(tag)
        # Leave anything that already carries a style alone — the image links
        # client.py builds are styled at the source.
        if not style or "style=" in attrs.lower():
            return m.group(0)
        return f"<{m.group(1)}{attrs} style=\"{style}\"{close}>"

    return _TAG_RE.sub(_inject, html)


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
        """Size to the content, bounded by `cap`."""
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

    def __init__(self, palette, max_width_ratio: float = 0.56, parent=None):
        super().__init__(parent)
        self.T = palette
        self._ratio = max_width_ratio
        self._current_agent: _Bubble | None = None

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
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

    # ── styling ──────────────────────────────────────────────────────────
    def _max_width(self) -> int:
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
        return max(320, int(w * self._ratio))

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
        if role != "user":
            html = self._styled(html)
        bubble = _Bubble(html, self._style(role), self._max_width())
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
        self._scroll_soon()
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
        self._current_agent.fit(self._max_width())
        self._scroll_soon()

    def end_agent(self, drop_if_empty: bool = True) -> None:
        """Close the streaming bubble, discarding it if nothing arrived."""
        bubble = self._current_agent
        self._current_agent = None
        if bubble is not None and drop_if_empty and not bubble.text().strip():
            row = bubble.parentWidget().layout() if bubble.parentWidget() else None
            bubble.setParent(None)
            bubble.deleteLater()
            if row is not None:
                row.setParent(None)

    def has_open_agent(self) -> bool:
        return self._current_agent is not None

    def clear(self) -> None:
        self._current_agent = None
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
    def _scroll_soon(self) -> None:
        # Deferred: the layout has not resized yet when a bubble is added, so
        # scrolling immediately lands short of the true bottom.
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cap = self._max_width()
        for bubble in self._body.findChildren(_Bubble):
            bubble.fit(cap)
