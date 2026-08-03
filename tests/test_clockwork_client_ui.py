"""uConsole (ClockworkPi) client — UI layout guards.

Source-level, not widget-level, on purpose: PyQt6 is not installed in this
project's venv (the client runs on the handheld, against its own ~/gui_env),
so importing `client.py` here is impossible. The existing clockwork guard in
tests/test_interface_voice.py uses the same technique.

The behaviour these guards encode was measured on the device with real Qt
6.4.2 rather than reasoned about — see test_fullscreen_toggle_hides_the_
transcript_not_its_container for the numbers.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CLIENT = _ROOT / "interface" / "externals" / "clockwork_ghost" / "client.py"


def _func_source(name: str) -> str:
    """Return the source of a top-level method by name."""
    tree = ast.parse(_CLIENT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(_CLIENT.read_text(), node) or ""
    raise AssertionError(f"{name}() not found in client.py")


def _statements(name: str) -> str:
    """Source of a method with comments and docstring stripped.

    The docstring for this particular method *describes* the bug, naming
    `left_widget.hide()` as the thing that caused it. A naive substring check
    over the raw source would match that prose and fail on the fix itself.
    """
    src = _func_source(name)
    tree = ast.parse(src)
    fn = tree.body[0]
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           and isinstance(fn.body[0].value.value, str)) else fn.body
    return "\n".join(ast.unparse(stmt) for stmt in body)


def test_client_file_exists():
    assert _CLIENT.exists(), "the uConsole client moved — update these guards"


def test_fullscreen_toggle_hides_the_whole_overlay():
    """"Hide UI" must hide ALL the glass UI, not one panel of it.

    `self.overlay` is the single translucent sheet carrying the top chips, the
    transcript and the bottom bar. Two narrower targets were tried and both
    left controls on screen:

      * `left_widget`  — the transcript's container. Also carries the main
        column's ONLY stretch factor, so hiding it handed the freed space to
        the bottom bar, which grew from 25px to 659px tall and vertically
        centred its fixed-height chips. Measured on-device at the panel's real
        1280x720 (XWAYLAND0, a 720x1280 display rotated right), Qt 6.4.2:
        bottom bar y=683 h=25 (96% down) -> y=49 h=659 (52% down, dead centre).
      * `chat_display` — fixed that placement but the chips and bar stayed up,
        so the button still did not hide the UI.
    """
    body = _statements("toggle_fullscreen_face")

    assert "self.overlay.hide()" in body, (
        "the toggle must hide the whole overlay — hiding a panel inside it "
        "leaves the top chips and the bottom bar on screen"
    )
    assert "self.left_widget.hide()" not in body, (
        "hiding left_widget releases the main column's only stretch factor — "
        "the bottom bar grows into the freed space and its chips centre "
        "vertically, landing mid-deck"
    )


def test_hidden_overlay_has_at_least_two_ways_back():
    """A frameless always-on-top kiosk with the UI hidden has NO visible
    control to restore it. Losing the way back is worse than the bug this
    fixes, so the escape hatch is redundant by design: an F11
    ApplicationShortcut *and* an any-key fallback in keyPressEvent."""
    src = _CLIENT.read_text()

    assert "Key_F11" in src, "the documented F11 restore shortcut is gone"
    assert "self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen_face)" in src

    # ApplicationShortcut, not the default WindowShortcut: with the overlay
    # hidden the QWebEngineView is all that is left to hold focus.
    fs = src[src.find("self.fullscreen_shortcut = QShortcut"):][:400]
    assert "ShortcutContext.ApplicationShortcut" in fs, (
        "a window-scoped shortcut can be swallowed by the focused web view"
    )

    key_body = _statements("keyPressEvent")
    assert "self.overlay.isVisible()" in key_body and "_restore_overlay" in key_body, (
        "keyPressEvent must restore the UI on any key while it is hidden"
    )


def test_face_never_takes_focus():
    """If the decorative web view could take focus it would swallow key
    presses, and the any-key escape from fullscreen mode would never fire."""
    src = _CLIENT.read_text()
    idx = src.find("self.web_face = WebFaceWidget(self)")
    assert idx != -1, "web_face construction moved — re-point this guard"
    assert "setFocusPolicy(Qt.FocusPolicy.NoFocus)" in src[idx:idx + 600], (
        "the face must be NoFocus so keys reach the window"
    )


def test_stretch_carrying_container_is_documented_as_such():
    """The trap is invisible at the definition site: `left_widget` looks like
    an ordinary container. Whoever next reaches for a 'hide the UI' target
    needs the warning where they will be standing."""
    src = _CLIENT.read_text()
    idx = src.find("self.left_widget = left_widget")
    assert idx != -1, "left_widget assignment moved — re-point this guard"
    context = src[max(0, idx - 400):idx + 200]
    assert "stretch" in context.lower(), (
        "left_widget's stretch role must stay documented at its definition"
    )


def test_toggle_uses_the_chip_glyph_set_not_emoji():
    """Every other control is a geometric glyph (◇ ◈ ◐ ◉ ● ◌) rendered in the
    chip style. The toggle used to swap in 📖/👁️ on first press, so the button
    left the design language and never came back — the initial ◐ was
    unreachable after one click."""
    body = _statements("toggle_fullscreen_face")
    for emoji in ("📖", "👁"):
        assert emoji not in body, f"{emoji} breaks the chip glyph set"


@pytest.mark.parametrize("name", ["toggle_fullscreen_face"])
def test_toggle_is_reachable_from_its_button(name):
    """A layout fix is worthless if the button stops being wired to it."""
    src = _CLIENT.read_text()
    assert f"self.fs_btn.clicked.connect(self.{name})" in src


# ── webface.py ──────────────────────────────────────────────────────────────
_WEBFACE = _ROOT / "interface" / "externals" / "clockwork_ghost" / "webface.py"


def test_the_face_form_is_never_hardcoded_at_the_call_site():
    """The startup form is resolved by facestate (remembered ◈ choice → env
    override → fallback). A literal here would quietly outrank the operator's
    last pick — which is precisely the bug the memory was added to fix.

    The FALLBACK_FORM-is-a-real-form guard lives with the rest of that logic in
    tests/test_clockwork_facestate.py, where the FORMS list is already parsed.
    """
    src = _WEBFACE.read_text()
    assert "facestate.startup_form(FACE_DIR)" in src
    assert 'os.environ.get("GHOST_FACE_FORM"' not in src, (
        "the env override belongs in facestate.startup_form, which weighs it "
        "against the remembered form"
    )
