"""uConsole client — the face form is remembered across restarts (facestate.py).

The JS already persists the ◈ choice to localStorage, and on this device that
can never survive a restart: the QWebEngineView profile is off-the-record, AND
the face is served from ``http://127.0.0.1:<free port>`` — a different origin
every boot. So the memory lives on the Python side, and these are its guards.

Like turnstatus.py, facestate.py is deliberately Qt-free, so this imports the
real module instead of reading its source.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CLIENT_DIR = _ROOT / "interface" / "externals" / "clockwork_ghost"
_FACE_DIR = _CLIENT_DIR / "webface"

sys.path.insert(0, str(_CLIENT_DIR))
import facestate as fs  # noqa: E402


@pytest.fixture
def state(tmp_path):
    return tmp_path / "face_form"


# ── round trip ──────────────────────────────────────────────────────────────
def test_a_saved_form_comes_back(state):
    assert fs.save_form("lattice", path=state) is True
    assert fs.load_form(path=state) == "lattice"


def test_saving_twice_keeps_the_last_one(state):
    fs.save_form("cube", path=state)
    fs.save_form("horizon", path=state)
    assert fs.load_form(path=state) == "horizon"


def test_startup_prefers_the_remembered_form(state):
    fs.save_form("cortex", path=state)
    assert fs.startup_form(path=state, env={}) == "cortex"


def test_startup_falls_back_when_nothing_is_remembered(state):
    assert not state.exists()
    assert fs.startup_form(path=state, env={}) == fs.FALLBACK_FORM


def test_env_override_beats_the_remembered_form(state):
    """`GHOST_FACE_FORM` is an explicit instruction for THIS run; it must not
    be silently outvoted by a months-old ◈ press."""
    fs.save_form("cube", path=state)
    assert fs.startup_form(path=state, env={"GHOST_FACE_FORM": "abyssal"}) == "abyssal"


def test_blank_env_override_is_not_an_override(state):
    fs.save_form("cube", path=state)
    assert fs.startup_form(path=state, env={"GHOST_FACE_FORM": "  "}) == "cube"


# ── the face must never break on bad state ──────────────────────────────────
def test_missing_file_is_not_an_error(tmp_path):
    assert fs.load_form(path=tmp_path / "nope") is None


def test_unreadable_state_is_not_an_error(tmp_path):
    """A directory where the file should be — load must shrug, not raise."""
    d = tmp_path / "as_a_dir"
    d.mkdir()
    assert fs.load_form(path=d) is None


@pytest.mark.parametrize("junk", ["", "   ", "\x00\x01", "rm -rf /", "Vortex",
                                  "'; DROP TABLE", "a" * 500])
def test_corrupt_state_is_ignored(state, junk):
    state.write_text(junk)
    assert fs.load_form(path=state) is None
    assert fs.startup_form(path=state, env={}) == fs.FALLBACK_FORM


@pytest.mark.parametrize("junk", ["", "  ", "not a form!", None, 7])
def test_junk_is_never_written(state, junk):
    assert fs.save_form(junk, path=state) is False
    assert not state.exists()


def test_save_failure_is_reported_not_raised(tmp_path):
    assert fs.save_form("cube", path=tmp_path / "no" / "such" / "dir" / "x") in (True, False)


def test_write_is_atomic_and_leaves_no_litter(state):
    fs.save_form("stack", path=state)
    strays = [p.name for p in state.parent.iterdir() if p.name.startswith(".face-")]
    assert strays == [], f"temp files left behind: {strays}"


# ── validation against the face's own FORMS list ────────────────────────────
def test_known_forms_are_read_from_the_deployed_face_module():
    forms = fs.known_forms(_FACE_DIR)
    assert "descent" in forms and "vortex" in forms and len(forms) >= 8


def test_the_fallback_is_a_form_the_face_actually_has():
    assert fs.FALLBACK_FORM in fs.known_forms(_FACE_DIR)


def test_a_form_that_no_longer_exists_is_dropped(state):
    """A form renamed in matrix_graph.js would otherwise be handed to
    `setForm`, which no-ops silently and leaves the face on the JS default."""
    fs.save_form("tesseract", path=state)
    assert fs.load_form(face_dir=_FACE_DIR, path=state) is None
    assert fs.startup_form(face_dir=_FACE_DIR, path=state, env={}) == fs.FALLBACK_FORM


def test_a_real_form_passes_validation(state):
    fs.save_form("embedding", path=state)
    assert fs.load_form(face_dir=_FACE_DIR, path=state) == "embedding"


def test_unparseable_face_module_does_not_reject_the_memory(state, tmp_path):
    """Cannot-validate must not mean reject: the name came from the face's own
    getForm(), so dropping it on a read failure would lose a good choice."""
    fs.save_form("lattice", path=state)
    assert fs.known_forms(tmp_path) == []
    assert fs.load_form(face_dir=tmp_path, path=state) == "lattice"


# ── the wiring in webface.py (source-level: it imports Qt) ──────────────────
def test_webface_persists_on_cycle_and_reads_at_startup():
    src = (_CLIENT_DIR / "webface.py").read_text()
    assert "facestate.save_form(name)" in src, (
        "the ◈ press is the only thing that records a choice — if cycle_form "
        "stops saving, the memory silently stops updating"
    )
    assert "facestate.startup_form(FACE_DIR)" in src, (
        "the remembered form must be resolved when the face becomes ready"
    )


def test_deploy_kills_hand_started_clients_too():
    """`pkill -f 'bin/client.py'` misses `python3 client.py` (started by hand
    from ~/bin). One such instance survived five deploys on 2026-08-03; two
    clients compete for the CM4's single GPU and the newer face never gets a
    WebGL context."""
    deploy = (_CLIENT_DIR / "deploy.sh").read_text()
    assert "pkill -f 'python3.*client\\.py'" in deploy
    assert "pkill -f 'bin/client.py'" not in deploy


def test_deploy_does_not_count_webengine_helpers_as_clients():
    """The QtWebEngineProcess helpers carry `--application-name=client.py`, so a
    bare `client\\.py` pattern counts three of them as extra clients and the
    single-instance check cries wolf on every deploy."""
    deploy = (_CLIENT_DIR / "deploy.sh").read_text()
    assert "pgrep -fc 'python3.*client\\.py'" in deploy


def test_deploy_ships_facestate():
    """A module the client imports but deploy.sh does not copy is an ImportError
    on the device and a client that will not start."""
    deploy = (_CLIENT_DIR / "deploy.sh").read_text()
    for mod in ("client.py", "webface.py", "chatlog.py", "turnstatus.py",
                "facestate.py"):
        assert mod in deploy, f"{mod} is not deployed"
