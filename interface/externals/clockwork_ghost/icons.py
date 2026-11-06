"""The handheld's line icons, drawn with QPainter.

PyQt6's QtSvg is not on the device, so the paths (Lucide, ISC licence, on its
24×24 grid) are parsed here into moves, lines and curves. ``parse`` is pure and
testable off the device; Qt is imported only when an icon is drawn.
"""

from __future__ import annotations

import math
import re

_SPEAKER = ("M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6"
            "a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z")
_SPARKLES = ("M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5"
             "l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964"
             "L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z")

PATHS = {
    "thumbs-up": ["M7 10v12", "M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8"
                  "a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"],
    "thumbs-down": ["M17 14V2", "M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8"
                    "a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"],
    "sparkles": [_SPARKLES, "M20 3v4", "M22 5h-4"],
    "sparkles-off": [_SPARKLES, "M2 2l20 20"],
    "camera": ["M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z",
               "M15 13a3 3 0 1 1-6 0a3 3 0 1 1 6 0"],
    "mic": ["M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z", "M19 10v2a7 7 0 0 1-14 0v-2", "M12 19v3"],
    "volume": [_SPEAKER, "M16 9a5 5 0 0 1 0 6", "M19.364 18.364a9 9 0 0 0 0-12.728"],
    "volume-x": [_SPEAKER, "M22 9l-6 6", "M16 9l6 6"],
    "stop": ["M7 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"],
    "x": ["M18 6 6 18", "M6 6l12 12"],
    "zoom-in": ["M19 11a8 8 0 1 1-16 0a8 8 0 1 1 16 0", "M21 21l-4.3-4.3", "M11 8v6", "M8 11h6"],
    "zoom-out": ["M19 11a8 8 0 1 1-16 0a8 8 0 1 1 16 0", "M21 21l-4.3-4.3", "M8 11h6"],
    "send": ["M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937"
             "l7.93 3.18a2 2 0 0 1 1.112 1.11z", "M21.854 2.147l-10.94 10.939"],
    "download": ["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", "M7 10l5 5 5-5", "M12 15V3"],
}

_TOKEN = re.compile(r"[MmLlHhVvCcAaZz]|-?(?:\d+\.\d*|\.\d+|\d+)")


def _arc(x0, y0, rx, ry, large, sweep, x, y):
    """An SVG arc (no rotation: Lucide's are circular) as cubic curves."""
    if rx == 0 or ry == 0 or (x0 == x and y0 == y):
        return [("L", x, y)]
    dx, dy = (x0 - x) / 2, (y0 - y) / 2
    scale = max(1.0, math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2))
    rx, ry = rx * scale, ry * scale
    num = max(0.0, rx * rx * ry * ry - rx * rx * dy * dy - ry * ry * dx * dx)
    den = rx * rx * dy * dy + ry * ry * dx * dx
    k = math.sqrt(num / den) if den else 0.0
    if large == sweep:
        k = -k
    cxp, cyp = k * rx * dy / ry, -k * ry * dx / rx
    cx, cy = cxp + (x0 + x) / 2, cyp + (y0 + y) / 2
    a0 = math.atan2((dy - cyp) / ry, (dx - cxp) / rx)
    a1 = math.atan2((-dy - cyp) / ry, (-dx - cxp) / rx)
    delta = a1 - a0
    if sweep and delta < 0:
        delta += 2 * math.pi
    elif not sweep and delta > 0:
        delta -= 2 * math.pi
    n = max(1, math.ceil(abs(delta) / (math.pi / 2) - 1e-9))
    step, out = delta / n, []
    t = 4 / 3 * math.tan(step / 4)
    for i in range(n):
        a, b = a0 + i * step, a0 + (i + 1) * step
        out.append(("C", cx + rx * (math.cos(a) - t * math.sin(a)), cy + ry * (math.sin(a) + t * math.cos(a)),
                    cx + rx * (math.cos(b) + t * math.sin(b)), cy + ry * (math.sin(b) - t * math.cos(b)),
                    cx + rx * math.cos(b), cy + ry * math.sin(b)))
    return out


def parse(path: str) -> list:
    """An SVG path as ``("M"|"L", x, y)``, ``("C", x1, y1, x2, y2, x, y)``, ``("Z",)``."""
    tok, i, out = _TOKEN.findall(path), 0, []
    x = y = sx = sy = 0.0
    cmd = ""

    def num():
        nonlocal i
        i += 1
        return float(tok[i - 1])

    while i < len(tok):
        if tok[i].isalpha():
            cmd = tok[i]
            i += 1
            if cmd in "Zz":
                out.append(("Z",))
                x, y = sx, sy
                continue
        rel, c = cmd.islower(), cmd.upper()
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if c in "ML":
            x, y = ox + num(), oy + num()
            out.append((c, x, y))
            if c == "M":
                sx, sy = x, y
                cmd = "l" if rel else "L"       # more pairs after a move are lines
        elif c == "H":
            x = ox + num()
            out.append(("L", x, y))
        elif c == "V":
            y = oy + num()
            out.append(("L", x, y))
        elif c == "C":
            a, b, d, e = ox + num(), oy + num(), ox + num(), oy + num()
            x, y = ox + num(), oy + num()
            out.append(("C", a, b, d, e, x, y))
        elif c == "A":
            rx, ry = num(), num()
            num()                               # rotation: 0 throughout
            large, sweep = num() != 0, num() != 0
            nx, ny = ox + num(), oy + num()
            out.extend(_arc(x, y, rx, ry, large, sweep, nx, ny))
            x, y = nx, ny
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    return out


def rgba(color: str):
    """``(r, g, b, a)`` 0–255 for a stylesheet colour. The theme's colours are CSS
    strings, and QColor reads ``rgba(…)`` as invalid, which draws black."""
    m = re.fullmatch(r"rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)", color.strip())
    if m:
        return int(m[1]), int(m[2]), int(m[3]), round(float(m[4]) * 255)
    h = color.strip().lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255


def icon(name: str, color: str, size: int = 20):
    """A QIcon of ``name``; its disabled state is the same shape, faint."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

    def draw(col, opacity=1.0):
        scale = 2                               # sharp on any display
        pm = QPixmap(size * scale, size * scale)
        pm.setDevicePixelRatio(scale)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(opacity)
        p.scale(size / 24, size / 24)
        pen = QPen(QColor(*rgba(col)), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        for d in PATHS[name]:
            path = QPainterPath()
            for seg in parse(d):
                if seg[0] == "M":
                    path.moveTo(seg[1], seg[2])
                elif seg[0] == "L":
                    path.lineTo(seg[1], seg[2])
                elif seg[0] == "C":
                    path.cubicTo(*seg[1:])
                else:
                    path.closeSubpath()
            p.drawPath(path)
        p.end()
        return pm

    ic = QIcon()
    ic.addPixmap(draw(color), QIcon.Mode.Normal)
    ic.addPixmap(draw("#ffffff", 0.22), QIcon.Mode.Disabled)
    return ic
