"""Web UI, 2026-09-05 (operator): the last face used is the default until
changed, and the header loses the model pill + the agent-status panel.

Two properties, pinned so that each has a world in which it fails:

1. FACE PERSISTENCE. The pick used to live only in localStorage — one
   origin, one browser, one device, purged by Safari after a week — so
   "the last face used" was never what the phone PWA or a LAN-IP tab came
   up with. Now `POST /api/ui/prefs` stores it in ONE server-side file,
   `GET /` injects it as `<meta name="ghost-face-form">`, and
   matrix_graph.js boots into the server's record before localStorage
   before the default. Server side is exercised through the real FastAPI
   app; the client precedence and the save call are EXECUTED under node.

2. THE CHIP IS A LIGHT. `#model-pill`, `#status-panel`, the palette's
   'Agent status' / 'Turn queue…' and status.js's /api/turns half are
   gone; what survives is the DEGRADED tag, now with the reason on the
   chip's title. The tag logic is executed under node with a fake DOM
   and four fetch worlds, because the text pins that guarded it before
   could not tell a toggled class from a deleted one.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

import interface.server as server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tests.helpers import eval_js, extract_js_function, strip_js_comments  # noqa: E402

_STATIC = _ROOT / "interface" / "static"
KEY = server.GHOST_API_KEY
HDR = {"X-Ghost-Key": KEY}


def _js(name: str) -> str:
    return strip_js_comments((_STATIC / name).read_text(encoding="utf-8"))


def _raw(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


@pytest.fixture
def prefs_file(tmp_path, monkeypatch):
    path = tmp_path / "ui_prefs.json"
    monkeypatch.setenv("GHOST_UI_PREFS_FILE", str(path))
    return path


@pytest.fixture
def client():
    return TestClient(server.app)


# ═══════════════════════════════════════════════════════════════════════
#  1. Server: the preference store and the <meta> it feeds
# ═══════════════════════════════════════════════════════════════════════

class TestPrefsRoutes:

    def test_both_routes_require_the_key(self, client, prefs_file):
        assert client.get("/api/ui/prefs").status_code == 401
        r = client.post("/api/ui/prefs", json={"face_form": "lattice"})
        assert r.status_code == 401
        assert not prefs_file.exists(), "an unauthenticated POST wrote the file"

    def test_a_pick_round_trips_through_the_file(self, client, prefs_file):
        assert client.get("/api/ui/prefs", headers=HDR).json() == {}
        r = client.post("/api/ui/prefs", headers=HDR, json={"face_form": "lattice"})
        assert r.status_code == 200, r.text
        assert r.json() == {"face_form": "lattice"}
        assert json.loads(prefs_file.read_text()) == {"face_form": "lattice"}
        assert client.get("/api/ui/prefs", headers=HDR).json() == {"face_form": "lattice"}
        # A second pick REPLACES — "last used", not first.
        client.post("/api/ui/prefs", headers=HDR, json={"face_form": "cube"})
        assert json.loads(prefs_file.read_text()) == {"face_form": "cube"}

    def test_the_page_carries_the_pick_only_once_one_exists(self, client, prefs_file):
        before = client.get(f"/?key={KEY}")
        assert before.status_code == 200
        assert "ghost-face-form" not in before.text, (
            "no pick saved yet, but the page injects a meta — the client "
            "would treat an empty/placeholder value as the server's word")
        client.post("/api/ui/prefs", headers=HDR, json={"face_form": "descent"})
        after = client.get(f"/?key={KEY}").text
        tag = '<meta name="ghost-face-form" content="descent">'
        assert tag in after
        assert after.index(tag) < after.index("</head>"), "meta must be in <head>"
        # It rides the same injection as the key; the unauthenticated
        # static copy must never carry it (it is served from a mount).
        assert "ghost-face-form" not in _raw("index.html")

    @pytest.mark.parametrize("body", [
        {"face_form": "Vortex"},            # case: the roster is lowercase
        {"face_form": "a\"b"},              # attribute breakout
        {"face_form": "<script>"},          # markup
        {"face_form": "x" * 33},            # length cap
        {"face_form": 7},                   # type
        {"face_form": ""},                  # empty
        {"colour": "red"},                  # unknown key
        {"face_form": "cube", "x": 1},      # partial-valid must not half-apply
        ["cube"],                           # not an object
    ])
    def test_bad_values_are_refused_and_the_old_pick_survives(self, client, prefs_file, body):
        client.post("/api/ui/prefs", headers=HDR, json={"face_form": "cube"})
        r = client.post("/api/ui/prefs", headers=HDR, json=body)
        assert r.status_code == 400, (body, r.text)
        assert json.loads(prefs_file.read_text()) == {"face_form": "cube"}
        assert '<meta name="ghost-face-form" content="cube">' in client.get(f"/?key={KEY}").text

    def test_malformed_json_is_a_400_not_a_502(self, client, prefs_file):
        r = client.post("/api/ui/prefs", headers={**HDR, "Content-Type": "application/json"},
                        content=b"{not json")
        assert r.status_code == 400

    def test_a_tampered_file_cannot_reach_the_page(self, client, prefs_file):
        """The read side re-validates: a hand-edited (or corrupted) file must
        neither inject markup nor break the page."""
        prefs_file.write_text(json.dumps(
            {"face_form": '"><script>alert(1)</script>'}))
        page = client.get(f"/?key={KEY}")
        assert page.status_code == 200
        assert "alert(1)" not in page.text
        assert "ghost-face-form" not in page.text
        assert client.get("/api/ui/prefs", headers=HDR).json() == {}
        prefs_file.write_text("{ this is not json")
        assert client.get(f"/?key={KEY}").status_code == 200
        assert client.get("/api/ui/prefs", headers=HDR).json() == {}
        # And a valid pick written over the junk is honoured again.
        client.post("/api/ui/prefs", headers=HDR, json={"face_form": "stack"})
        assert '<meta name="ghost-face-form" content="stack">' in client.get(f"/?key={KEY}").text

    def test_write_creates_parents_and_leaves_no_temp_file(self, client, tmp_path, monkeypatch):
        path = tmp_path / "deeper" / "still" / "ui_prefs.json"
        monkeypatch.setenv("GHOST_UI_PREFS_FILE", str(path))
        r = client.post("/api/ui/prefs", headers=HDR, json={"face_form": "abyssal"})
        assert r.status_code == 200
        assert json.loads(path.read_text()) == {"face_form": "abyssal"}
        assert [p.name for p in path.parent.iterdir()] == ["ui_prefs.json"], (
            "the atomic-write temp file was left behind")

    def test_an_unwritable_store_is_a_503_and_says_so(self, client, tmp_path, monkeypatch):
        """The face still switched locally; the server must say the SAVE
        failed (503), not crash into a 500 the client reads as 'HTTP 500'."""
        monkeypatch.setenv("GHOST_UI_PREFS_FILE", str(tmp_path))   # a directory
        r = client.post("/api/ui/prefs", headers=HDR, json={"face_form": "cube"})
        assert r.status_code == 503, r.text
        assert "save" in r.json()["error"].lower()

    def test_the_validator_is_shared_by_read_and_write(self):
        """One function both ways — otherwise a value that cannot be
        written can still be read (the tampered-file test above would then
        depend on the read path having its own copy of the rule)."""
        src = (_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
        body_read = src[src.index("def _read_ui_prefs("):src.index("def _write_ui_prefs(")]
        body_set = src[src.index("async def ui_prefs_set("):src.index('@app.get("/")')]
        assert "_sanitize_ui_prefs(" in body_read
        assert "_sanitize_ui_prefs(" in body_set


# ═══════════════════════════════════════════════════════════════════════
#  2. Client: boot precedence and the save call — EXECUTED
# ═══════════════════════════════════════════════════════════════════════

def _forms_js() -> str:
    m = re.search(r"const FORMS = \[(.*?)\];", _raw("matrix_graph.js"), re.DOTALL)
    assert m, "FORMS roster not found"
    return f"const FORMS = [{m.group(1)}];\n"


def _forms() -> list:
    return re.findall(r"'(\w+)'", _forms_js())


class TestBootPrecedence:

    @pytest.fixture(scope="class")
    def resolver(self):
        return _forms_js() + extract_js_function(_js("matrix_graph.js"), "resolveInitialForm")

    def test_the_servers_record_beats_this_browsers(self, resolver):
        assert eval_js(resolver, "resolveInitialForm('lattice', 'cube', 'vortex')") == "lattice"

    def test_local_storage_is_the_fallback(self, resolver):
        for server_value in ("null", "undefined", "''", "'bogus'", "42", "{}"):
            got = eval_js(resolver, f"resolveInitialForm({server_value}, 'cube', 'vortex')")
            assert got == "cube", f"server={server_value} -> {got!r}"

    def test_the_default_is_last(self, resolver):
        for stored in ("null", "'nope'", "''", "7"):
            got = eval_js(resolver, f"resolveInitialForm(null, {stored}, 'vortex')")
            assert got == "vortex", f"stored={stored} -> {got!r}"

    def test_every_roster_form_is_accepted_from_the_server(self, resolver):
        for name in _forms():
            assert eval_js(resolver, f"resolveInitialForm('{name}', 'cube', 'vortex')") == name

    def test_boot_actually_uses_it(self):
        """The resolver is wired at module level — a pure function nobody
        calls is documentation. Its inputs must be the meta, then
        localStorage, then the previous default."""
        js = _js("matrix_graph.js")
        boot = js[js.index("export function resolveInitialForm"):js.index("let formBlend")]
        assert "formIndex = FORMS.indexOf(resolveInitialForm(" in boot
        call = boot[boot.index("formIndex = FORMS.indexOf(resolveInitialForm("):]
        call = call[:call.index(";")]
        assert "_meta" in call and "_stored" in call and "FORMS[formIndex]" in call
        assert "meta[name=\"ghost-face-form\"]" in boot
        assert "localStorage.getItem('ghost_face_form')" in boot

    def test_meta_name_is_one_string_on_both_sides(self):
        """R5: one input, one story. The server writes the tag name it
        defines; the client queries the name it hard-codes."""
        src = (_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
        m = re.search(r'_FACE_FORM_META = "([a-z-]+)"', src)
        assert m
        assert f'meta[name="{m.group(1)}"]' in _js("matrix_graph.js")
        assert re.search(r"FACE_FORM_RE = re\.compile\(r\"\[a-z\]\[a-z0-9_-\]\{0,31\}\"\)", src)
        # Every roster name passes the server's charset — a form the server
        # refuses could be picked but never remembered.
        for name in _forms():
            assert re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", name), name


_SAVE_HARNESS = """
const captured = [];
globalThis.__fetchResult = { ok: true, status: 200 };
globalThis.fetch = async (url, init) => {
    captured.push({ url, init });
    if (globalThis.__fetchResult instanceof Error) throw globalThis.__fetchResult;
    return globalThis.__fetchResult;
};
const warned = [];
globalThis.console = { warn: (...a) => warned.push(a.join(' ')) };
"""


class TestRememberFaceForm:

    @pytest.fixture(scope="class")
    def fn(self):
        return _SAVE_HARNESS + extract_js_function(_js("app.js"), "rememberFaceForm")

    def test_it_posts_the_pick_to_the_prefs_route(self, fn):
        out = eval_js(fn, "await rememberFaceForm('lattice').then(ok => ({ok, c: captured, warned}))")
        assert out["ok"] is True
        assert len(out["c"]) == 1
        req = out["c"][0]
        assert req["url"] == "/api/ui/prefs"
        assert req["init"]["method"] == "POST"
        assert json.loads(req["init"]["body"]) == {"face_form": "lattice"}
        assert req["init"]["headers"]["Content-Type"] == "application/json"
        assert out["warned"] == []

    def test_a_refused_save_resolves_false_and_warns_without_throwing(self, fn):
        src = fn + "\nglobalThis.__fetchResult = { ok: false, status: 503 };\n"
        out = eval_js(src, "await rememberFaceForm('cube').then(ok => ({ok, warned}))")
        assert out["ok"] is False
        assert len(out["warned"]) == 1 and "503" in out["warned"][0]

    def test_a_dead_socket_resolves_false_too(self, fn):
        src = fn + "\nglobalThis.__fetchResult = new TypeError('Failed to fetch');\n"
        out = eval_js(src, "await rememberFaceForm('cube').then(ok => ({ok, warned}))")
        assert out["ok"] is False
        assert "Failed to fetch" in out["warned"][0]

    def test_both_pick_paths_call_it(self):
        """The menu pick AND the stale-cache cycle fallback. A save call
        nobody makes is the whole feature missing."""
        js = _js("app.js")
        menu = js[js.index("function buildFaceFormMenu("):js.index("function markActiveFaceForm(")]
        assert "const picked = activeFace.setForm(name);" in menu
        assert "rememberFaceForm(picked);" in menu
        cycle = js[js.index("faceFormBtn.addEventListener('click'"):]
        cycle = cycle[:cycle.index("const menu = buildFaceFormMenu()")]
        assert "const name = activeFace.cycleForm();" in cycle
        assert "rememberFaceForm(name);" in cycle


# ═══════════════════════════════════════════════════════════════════════
#  3. The header chip: pill and panel gone, DEGRADED tag executed
# ═══════════════════════════════════════════════════════════════════════

class TestRemoval:
    """Pin the deletion: a revert of any one file brings the surface back."""

    def test_index_has_no_pill_and_no_panel(self):
        """The chip stayed; the pill and the panel did not. (Later on
        2026-09-05 the chip became the live-log toggle — a <button> — which
        tests/test_interface_header_simplify.py pins; what must NOT return
        is a STATUS surface behind it.)"""
        html = _raw("index.html")
        for gone in ('id="model-pill"', 'id="status-panel"', 'id="status-body"',
                     'id="status-close"', "AGENT STATUS"):
            assert gone not in html, f"{gone} is back"
        assert re.search(r'<button id="status-indicator"[^>]*>', html), (
            "the ONLINE chip itself must stay")
        assert 'id="status-text">ONLINE<' in html

    def test_palette_lost_the_two_panel_commands(self):
        js = _js("palette.js")
        assert "'Agent status'" not in js
        assert "Turn queue" not in js
        assert "status.togglePanel" not in js and "status.openPanel" not in js
        assert "Stop MY turn" in js, "the composer-scoped stop must survive"
        # The destructure must not name a ctx member that is no longer passed.
        head = js[:js.index("function commandList")]
        assert "status" not in head.split("= ctx;")[0].rsplit("{", 1)[1]

    def test_workspace_no_longer_hands_status_to_the_palette(self):
        js = _js("workspace.js")
        assert "initPalette({ ...ctx, sessions, notifications });" in js
        assert "const status = initStatus(ctx);" in js, "the health poll must still boot"

    def test_css_lost_the_pill_and_panel_rules_but_kept_the_tag(self):
        css = _raw("style.css")
        for gone in ("#model-pill", "#status-panel", "#status-body", ".status-section",
                     ".status-turn", ".status-row", "#status-indicator { cursor: pointer; }"):
            assert gone not in css, f"{gone} is back"
        assert "#status-indicator.degraded::after" in css
        assert "content: 'DEGRADED'" in css
        # The notifications panel must not have lost its chrome in the edit.
        assert "#notif-panel {" in css and "#notif-panel.hidden { display: none; }" in css

    def test_status_js_has_no_panel_and_no_turn_half(self):
        js = _js("status.js")
        for gone in ("/api/turns", "/api/turn/cancel", "renderPanel", "openPanel",
                     "modelPill", "pillText", "status-panel", "cancelTurn"):
            assert gone not in js, f"{gone} is back"

    def test_the_server_still_proxies_health_for_the_tag(self):
        src = (_ROOT / "interface" / "server.py").read_text(encoding="utf-8")
        assert '@app.get("/api/health", dependencies=[Depends(verify_interface_key)])' in src


_CHIP_HARNESS = """
const HEALTH_POLL_MS = 25_000;
const indicator = {
    classes: new Set(),
    title: 'Live log',
    getAttribute: (n) => (n === 'data-title' ? 'Live log' : null),
    classList: {
        toggle(c, on) { if (on) indicator.classes.add(c); else indicator.classes.delete(c); return !!on; },
    },
};
const listeners = {};
globalThis.document = {
    getElementById: (id) => (id === 'status-indicator' ? indicator : null),
    addEventListener: (ev, fn) => { listeners[ev] = fn; },
    visibilityState: 'visible',
};
globalThis.window = {};
globalThis.setInterval = () => 0;
const calls = [];
globalThis.fetch = async (url) => {
    calls.push(url);
    if (globalThis.__health instanceof Error) throw globalThis.__health;
    return globalThis.__health;
};
const okHealth = (h) => ({ ok: true, status: 200, json: async () => h });
"""


def _run_chip(status_js: str, world: str):
    src = (_CHIP_HARNESS + f"globalThis.__health = {world};\n"
           + extract_js_function(status_js, "initStatus")
           + "\nconst api = initStatus({});\n"
             "await new Promise(r => setTimeout(r, 30));\n")
    return eval_js(src, "({ degraded: indicator.classes.has('degraded'), title: indicator.title,"
                        " calls, health: api.health(), stored: globalThis.window.__ghostHealth })")


class TestStatusChip:

    @pytest.fixture(scope="class")
    def status_js(self):
        return _js("status.js")

    def test_a_healthy_agent_is_calm(self, status_js):
        out = _run_chip(status_js, "okHealth({memory_system_loaded: true, biological_watchdog_alive: true})")
        assert out["degraded"] is False
        # Healthy = the RESTING tooltip (the chip is the live-log button),
        # not an empty string that wipes the button's name on hover.
        assert out["title"] == "Live log"
        assert out["calls"] == ["/api/health"]
        assert out["health"] == {"memory_system_loaded": True, "biological_watchdog_alive": True}
        assert out["stored"] == out["health"]

    def test_a_dead_memory_system_lights_the_tag_with_the_reason(self, status_js):
        out = _run_chip(status_js, "okHealth({memory_system_loaded: false, biological_watchdog_alive: true})")
        assert out["degraded"] is True
        assert out["title"].startswith("DEGRADED")
        assert "memory system" in out["title"].lower()
        assert "watchdog" not in out["title"].lower()

    def test_a_dead_watchdog_lights_it_too(self, status_js):
        out = _run_chip(status_js, "okHealth({memory_system_loaded: true, biological_watchdog_alive: false})")
        assert out["degraded"] is True
        assert "watchdog" in out["title"].lower()

    def test_a_401_says_not_authorised_on_the_chip(self, status_js):
        """The panel used to render the classified note; the chip's title
        carries it now. A key rotation must NOT read as 'agent unreachable'."""
        out = _run_chip(status_js, "({ ok: false, status: 401 })")
        assert out["degraded"] is True
        assert "NOT AUTHORISED" in out["title"]
        assert "unreachable" not in out["title"].lower()
        assert out["health"] is None and out["stored"] is None

    def test_a_dead_socket_says_unreachable(self, status_js):
        out = _run_chip(status_js, "new TypeError('Failed to fetch')")
        assert out["degraded"] is True
        assert "unreachable" in out["title"].lower()

    def test_recovery_clears_the_tag_and_the_reason(self, status_js):
        """Degraded → healthy on the next poll must clear BOTH the class and
        the title; a stale reason on a calm chip is a lie."""
        src = (_CHIP_HARNESS + "globalThis.__health = ({ ok: false, status: 503 });\n"
               + extract_js_function(status_js, "initStatus")
               + "\nconst api = initStatus({});\n"
                 "await new Promise(r => setTimeout(r, 20));\n"
                 "const first = { degraded: indicator.classes.has('degraded'), title: indicator.title };\n"
                 "globalThis.__health = okHealth({memory_system_loaded: true, biological_watchdog_alive: true});\n"
                 "listeners.visibilitychange();\n"
                 "await new Promise(r => setTimeout(r, 20));\n")
        out = eval_js(src, "({ first, degraded: indicator.classes.has('degraded'), title: indicator.title, calls })")
        assert out["first"] == {"degraded": True, "title": out["first"]["title"]}
        assert out["first"]["title"].startswith("DEGRADED")
        assert out["degraded"] is False and out["title"] == "Live log"
        assert out["calls"] == ["/api/health", "/api/health"], "tab-visible must re-poll"
