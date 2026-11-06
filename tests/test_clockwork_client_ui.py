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
import re
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


# ──────────────────────────────────────────────────────────────────────
# TLS verification on the two interface channels (2026-08-15)
# ──────────────────────────────────────────────────────────────────────

def _env_default(var: str) -> str:
    """The DEFAULT argument of `os.environ.get("<var>", <default>)`.

    Extracted from the call node, not by substring. The first version of
    this asked whether "'1'" appeared anywhere in the assignment source —
    and `... .lower() in ('1', 'true', 'yes')` contains '1' on the
    membership side, so flipping the default to '0' left the guard green.
    """
    tree = ast.parse(_CLIENT.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == var):
            return node.args[1].value
    raise AssertionError(f"os.environ.get({var!r}, ...) not found in client.py")


def test_voice_and_log_channels_VERIFY_tls():
    """They accepted ANY certificate until 2026-08-15, because the
    interface served a self-signed CN=localhost cert with no SAN that
    nothing could verify. WireGuard authenticates tailnet PEERS; it does
    not stop a compromised node answering on :8080 and harvesting the
    X-Ghost-Key these calls carry. The interface now serves a real
    Let's Encrypt cert (issued by Tailscale), so verification is possible
    and therefore mandatory."""
    default = _env_default("GHOST_VOICE_VERIFY_TLS")
    assert default == "1", (
        f"TLS verification is off by default again (default={default!r})")


def test_the_host_default_MATCHES_THE_CERT_SAN():
    """Verification only works against the name in the certificate. The
    SAN is `DNS:eva.taila2b1d.ts.net` — the short `eva` reaches the same
    box but fails hostname matching, which was proven live: with
    verify=True, https://eva:8080 is refused while the FQDN returns 200.
    A short-name default here would silently disable voice."""
    assert _env_default("GHOST_HOST") == "eva.taila2b1d.ts.net"


def test_the_ONE_switch_still_governs_BOTH_channels():
    """Voice (httpx) and the log websocket must not drift apart — a
    half-verified client is the worst of both."""
    text = _CLIENT.read_text()
    assert "verify=VOICE_VERIFY_TLS" in text          # the httpx calls
    assert "verify_tls=VOICE_VERIFY_TLS" in text      # stream_log_lines


def test_a_NON_RESOLVING_host_is_announced_at_startup():
    """The documented failure mode on this device is silence: the voice
    endpoints spent weeks "unused" while failing against a host that no
    longer existed. Turning verification on adds exactly one new way to
    reproduce that — an FQDN the handheld cannot resolve — so it must be
    named at boot, in the log deploy.sh tails, not discovered later."""
    text = _CLIENT.read_text()
    assert "getaddrinfo" in text, "no resolve preflight"
    assert "[tls]" in text, "the preflight is not greppable in ghost_ui.log"
    # …and it must name the escape hatch, since the device is often remote.
    assert "GHOST_VOICE_VERIFY_TLS=0" in text


# ── the modern interface (operator, 2026-11-06) ───────────────────────────────
def test_save_load_and_the_face_only_mode_are_gone():
    src = (Path(__file__).resolve().parent.parent / "interface/externals/clockwork_ghost/client.py").read_text(encoding="utf-8")
    for gone in ("load_btn", "workspace_btn", "save_workspace", "load_workspace", "/api/workspace",
                 "fs_btn", "toggle_fullscreen_face", "_restore_overlay", "fullscreen_shortcut", "Key_F11",
                 "self.overlay.hide"):
        assert not re.search(r"(?<![a-z])" + re.escape(gone), src), gone   # upload_btn is not load_btn


def test_no_control_is_a_capitalised_glyph_any_more():
    src = (Path(__file__).resolve().parent.parent / "interface/externals/clockwork_ghost/client.py").read_text(encoding="utf-8")
    assert not re.search(r'QPushButton\("[^"]+"\)', src), "a button text bypasses set_chip"
    assert not re.search(r'set(?:Text)?\("[◉◆◈◐◇○◌●■▲▼✕]', src)
    for word in ("Good", "Bad", "Animations on", "Animations off", "Camera", "Talk", "Voice on", "Voice off", "Stop"):
        assert f'"{word}"' in src, word


def test_the_icons_parse_to_real_shapes_and_theme_colours_read():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface/externals/clockwork_ghost"))
    import icons
    for name, paths in icons.PATHS.items():
        for d in paths:
            segs = icons.parse(d)
            assert segs[0][0] == "M", (name, d)
            for s in segs:
                assert all(-1 <= v <= 25 for v in s[1:]), (name, s)     # stays on the grid
    assert icons.rgba("rgba(236, 235, 246, 0.82)") == (236, 235, 246, 209)     # QColor would draw black
    assert icons.rgba("#c9a6ff") == (201, 166, 255, 255)
    # a half-circle arc ends where it says, through the far side
    arc = icons.parse("M15 13a3 3 0 1 1-6 0")
    assert arc[-1][-2:] == pytest.approx((9, 13))


def test_deploy_ships_the_icons():
    assert "icons.py" in (Path(__file__).resolve().parent.parent / "interface/externals/clockwork_ghost/deploy.sh").read_text(encoding="utf-8")
