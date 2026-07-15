"""Guard: the web face's ICON_CLASS map must cover the agent's live glyphs.

The agent emits log-line icons from utils/logging.py's Icons class; app.js
maps each glyph to a face colour/priority class. When Icons was migrated to
wide-base glyphs (WARN ⚠️→🔶, SHIELD 🛡️→🔒, …) the JS map wasn't updated, so
21 currently-emitted glyphs fell through to the 'think' floor — WARN and
tool/memory lines silently lost their colour (found 2026-07-15). This test
fails if the two drift again.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGGING_PY = REPO / "src" / "ghost_agent" / "utils" / "logging.py"
APP_JS = REPO / "interface" / "static" / "app.js"

# Icons that are defined in the class but never reach the face log stream
# (or are intentionally unstyled) — exempt from the coverage requirement.
EXEMPT = set()


def _python_icon_glyphs():
    """Every single-glyph value assigned in the Icons class body."""
    src = LOGGING_PY.read_text(encoding="utf-8")
    body = src.split("class Icons", 1)[1]
    # Stop at the next top-level class/def so we only scan Icons.
    body = re.split(r"\nclass \w|\ndef \w", body)[0]
    glyphs = set()
    for m in re.finditer(r'^\s+[A-Z_]+\s*=\s*"([^"]+)"', body, re.M):
        val = m.group(1)
        # Only emoji-ish values (skip plain ASCII separators, if any).
        if val and any(ord(c) > 0x2000 for c in val):
            glyphs.add(val)
    return glyphs


def _js_icon_keys():
    """The key set of app.js's ICON_CLASS object literal."""
    src = APP_JS.read_text(encoding="utf-8")
    block = src.split("const ICON_CLASS = {", 1)[1].split("};", 1)[0]
    return {m.group(1) for m in re.finditer(r"'([^']+)':\s*'\w+'", block)}


def test_every_emitted_glyph_is_classified():
    py = _python_icon_glyphs()
    js = _js_icon_keys()
    missing = (py - js) - EXEMPT
    assert not missing, (
        "app.js ICON_CLASS is missing glyphs the agent emits "
        f"(they fall through to 'think'): {sorted(missing)}"
    )


def test_no_unreachable_vs16_keys():
    # extractIcon matches \p{Extended_Pictographic}, which strips a trailing
    # U+FE0F — so any two-codepoint VS16 key in the map is permanently dead.
    dead = [k for k in _js_icon_keys() if "️" in k]
    assert not dead, f"unreachable VS16 keys in ICON_CLASS: {dead}"
