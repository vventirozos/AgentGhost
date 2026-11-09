"""Notifications panel header (2026-11-08): both buttons grouped on the
right as SVG icons. Before, three children under justify-content:
space-between floated the clear button mid-header, and the ⌫ / ✕ glyphs
rendered per-font (odd on iOS)."""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "interface" / "static"


def _header():
    html = (STATIC / "index.html").read_text()
    m = re.search(r'<div id="notif-panel".*?<div id="notif-list">', html, re.S)
    assert m
    return m.group(0)


def test_buttons_grouped_in_actions_wrapper():
    h = _header()
    actions = h[h.index('class="panel-actions"'):]
    assert 'id="notif-clear"' in actions and 'id="notif-close"' in actions


def test_buttons_are_svg_not_font_glyphs():
    h = _header()
    assert "⌫" not in h and "✕" not in h
    assert h.count("<svg") == 2


def test_actions_wrapper_is_styled():
    css = (STATIC / "style.css").read_text()
    assert re.search(r"\.panel-actions\s*\{[^}]*display:\s*flex", css)
