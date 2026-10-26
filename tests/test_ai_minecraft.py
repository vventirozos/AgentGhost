"""Eva's World (interface/externals/ai_minecraft) — the side project where the
27B lives in a Minecraft world through a body driven by a small model on nova.

The agent itself is untouched by that project; these pins cover the code that
shares the agent's resources:

* ``mac/eva_mind.py`` calls the agent's llama-server directly. Its busy gate
  and its abort-on-owner-request are what keep it from delaying a real
  request, so they are exercised against a fake streaming server, not mocks
  of the functions under test.
* the hub's own unit tests (``nova/hub/test``, node:test) run here too when
  node is installed, so the regular suite covers both halves.
* config pins: no code execution, no outbound wiki fetch, a monitor page with
  no external requests.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "interface" / "externals" / "ai_minecraft"


def _load_mind():
    spec = importlib.util.spec_from_file_location("eva_mind", ROOT / "mac" / "eva_mind.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mind = _load_mind()


# ── decision parsing ──────────────────────────────────────────────────────

def test_parse_decision_takes_the_last_json_object_with_a_known_key():
    reply = ('I think I will build. Example of format: {"note": 1}\n'
             '{"goal": "build a hut", "wake_in_minutes": 12, "extra": "dropped"}')
    d, err = mind.parse_decision(reply)
    assert err is None
    assert d == {"goal": "build a hut", "wake_in_minutes": 12}


def test_parse_decision_handles_fences_and_braces_inside_strings():
    reply = 'Plan:\n```json\n{"journal": "a {weird} day", "stop": true}\n```'
    d, err = mind.parse_decision(reply)
    assert d == {"journal": "a {weird} day", "stop": True}


@pytest.mark.parametrize("reply,want", [
    # a stray brace in the prose must not swallow the decision after it
    ('A smiley :-{ then\n{"goal": "a"}', {"goal": "a"}),
    ('He said "hi {there" and {"goal": "a"}', {"goal": "a"}),
    # the agent's tool-call habit, arguments as an object or a JSON string
    ('<tool_call>\n{"name": "decide", "arguments": {"goal": "g", "wake_in_minutes": 5}}\n</tool_call>',
     {"goal": "g", "wake_in_minutes": 5}),
    ('{"name": "decide", "arguments": "{\\"stop\\": true}"}', {"stop": True}),
    # a nested key does not count; the last top-level decision wins
    ('{"a": {"goal": "inner"}} {"journal": "j"}', {"journal": "j"}),
])
def test_parse_decision_survives_what_the_model_actually_writes(reply, want):
    assert mind.parse_decision(reply) == (want, None)


def test_parse_decision_reports_why_it_failed():
    assert mind.parse_decision("") == (None, "empty reply")
    assert mind.parse_decision("just prose, no json")[1] == "no decision JSON found in the reply"
    assert mind.parse_decision('{"goal": "unterminated')[0] is None


def test_split_reasoning_field_inline_and_unclosed():
    assert mind.split_reasoning("thought", "reply") == ("thought", "reply")
    assert mind.split_reasoning("", "<think>hmm</think>\nreply") == ("hmm", "reply")
    # the chat template opens <think> itself, so the reply may carry only the close
    assert mind.split_reasoning("", "musing</think>\n{\"goal\": \"x\"}") == ("musing", '{"goal": "x"}')


# ── the busy gate ─────────────────────────────────────────────────────────

_IDLE = {"foreground_requests": 0, "slot_keep_left_s": 0.0}


@pytest.mark.parametrize("health,slots,busy", [
    (_IDLE, [{"is_processing": False}], False),
    ({**_IDLE, "foreground_requests": 1}, [{"is_processing": False}], True),
    (_IDLE, [{"is_processing": True}], True),
    (None, [{"is_processing": False}], True),                 # agent down: unknown counts as busy
    ({"status": "ok"}, [{"is_processing": False}], True),     # plain /health has no counter
    ({"foreground_requests": 0}, [{"is_processing": False}], True),  # no keep field: unknown = busy
    (_IDLE, None, True),                                      # llama unreachable
])
def test_agent_busy_unless_both_signals_say_idle(health, slots, busy):
    assert mind.agent_busy(health, slots)[0] is busy


# ── streaming call + abort, against a fake llama ─────────────────────────

class _FakeLlama(BaseHTTPRequestHandler):
    slow = False
    cancelled = threading.Event()

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "let me think"}}]},
                  {"choices": [{"delta": {"content": '{"goal": "explore"}'}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"prompt_n": 10, "predicted_n": 5}}]
        try:
            if type(self).slow:  # a long prefill: nothing arrives for a while
                for _ in range(100):
                    time.sleep(0.1)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            type(self).cancelled.set()


@pytest.fixture
def fake_llama(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(mind, "LLAMA", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(mind, "WATCH_S", 0.1)
    _FakeLlama.slow = False
    _FakeLlama.cancelled.clear()
    yield _FakeLlama
    srv.shutdown()


def test_stream_chat_collects_reasoning_content_and_timings(fake_llama):
    out = mind.stream_chat([{"role": "user", "content": "hi"}], lambda: None)
    assert "".join(out["reasoning"]) == "let me think"
    assert "".join(out["content"]) == '{"goal": "explore"}'
    assert out["finish"] == "stop"
    assert out["timings"]["prompt_n"] == 10


def test_stream_chat_aborts_when_the_owner_sends_a_request(fake_llama):
    """The world where this fails: Eva keeps the only slot while the operator
    waits. The abort must close the connection (llama cancels on disconnect)."""
    fake_llama.slow = True
    calls = {"n": 0}

    def owner():
        calls["n"] += 1
        return "the operator sent a request" if calls["n"] >= 3 else None

    t0 = time.time()
    with pytest.raises(mind.Interrupted, match="operator"):
        mind.stream_chat([{"role": "user", "content": "hi"}], owner)
    assert time.time() - t0 < 5
    assert fake_llama.cancelled.wait(5), "the server never saw the client go away"


# ── one look, end to end with a fake hub ─────────────────────────────────

class _Hub:
    def __init__(self, due=True):
        self.due = due
        self.posts = []
        self.statuses = []

    def get(self, path, timeout=15):
        if path == "/api/mind/poll":
            return {"due": self.due}
        if path == "/api/mind/context":
            return {"look_ts": 123, "diary": "WORLD TIME: day 0", "journal": "", "reasons": ["a new session began"]}
        raise AssertionError(path)

    def post(self, path, body, timeout=30):
        self.posts.append((path, body))
        return {"applied": ["set the goal"], "wake_at": 0}

    def status(self, status, detail=""):
        self.statuses.append(status)


def test_look_once_not_due_does_not_touch_the_model(monkeypatch):
    monkeypatch.setattr(mind, "stream_chat", lambda *a: pytest.fail("called the model"))
    hub = _Hub(due=False)
    assert mind.look_once(hub, "SYS", "h") == mind.POLL_S
    assert hub.posts == []


@pytest.fixture
def agent_says(monkeypatch):
    """The agent's /api/slot/outside-call: records every state, refuses
    "start" when told to."""
    calls, answer, details = [], {"start": (True, "")}, {}
    monkeypatch.setattr(mind, "agent_outside_call",
                        lambda state, detail="": calls.append(state) or details.__setitem__(state, detail)
                        or answer.get(state, (True, "")))
    return calls, answer, details


def test_look_once_skips_while_the_agent_is_busy(monkeypatch):
    monkeypatch.setattr(mind, "read_agent_health", lambda: {"foreground_requests": 1})
    monkeypatch.setattr(mind, "read_slots", lambda: [{"is_processing": False}])
    monkeypatch.setattr(mind, "stream_chat", lambda *a: pytest.fail("called the model"))
    hub = _Hub()
    assert mind.look_once(hub, "SYS", "h") == mind.BUSY_RETRY_S
    assert hub.statuses == ["skipped: agent busy"]


def test_look_once_posts_the_decision_with_reasoning_and_diary(monkeypatch, fake_llama, agent_says):
    monkeypatch.setattr(mind, "read_agent_health", lambda: dict(_IDLE))
    monkeypatch.setattr(mind, "read_slots", lambda: [{"is_processing": False}])
    hub = _Hub()
    mind.look_once(hub, "SYS", "abc")
    assert agent_says[0] == ["start", "end", "result"]
    # the agent's log shows why she woke, how the call ended and what she did
    assert agent_says[2] == {"start": "woke because: a new session began", "end": "finished", "result": "set the goal"}
    path, body = hub.posts[-1]
    assert path == "/api/mind/decision"
    assert body["decision"] == {"goal": "explore"}
    assert body["reasoning"] == "let me think"
    assert body["diary"] == "WORLD TIME: day 0"
    assert body["look_ts"] == 123 and body["prompt_hash"] == "abc"


def test_look_once_asks_again_after_fetching_the_context(monkeypatch, agent_says):
    """The context fetch takes seconds; an owner request in that gap must
    still stop the look before the model is called."""
    healths = iter([dict(_IDLE), {**_IDLE, "foreground_requests": 1}])
    monkeypatch.setattr(mind, "read_agent_health", lambda: next(healths))
    monkeypatch.setattr(mind, "read_slots", lambda: [{"is_processing": False}])
    monkeypatch.setattr(mind, "stream_chat", lambda *a: pytest.fail("called the model"))
    hub = _Hub()
    assert mind.look_once(hub, "SYS", "h") == mind.BUSY_RETRY_S
    assert agent_says[0] == []                       # never even asked the agent to start


def test_look_once_obeys_the_agents_refusal(monkeypatch, agent_says):
    monkeypatch.setattr(mind, "read_agent_health", lambda: dict(_IDLE))
    monkeypatch.setattr(mind, "read_slots", lambda: [{"is_processing": False}])
    monkeypatch.setattr(mind, "stream_chat", lambda *a: pytest.fail("called the model"))
    agent_says[1]["start"] = (False, "1 request(s) live")
    hub = _Hub()
    assert mind.look_once(hub, "SYS", "h") == mind.BUSY_RETRY_S
    assert agent_says[0] == ["start"]                # refused: no "end" for a call never made


def test_an_interrupted_look_still_tells_the_agent_it_ended(monkeypatch, agent_says):
    monkeypatch.setattr(mind, "read_agent_health", lambda: dict(_IDLE))
    monkeypatch.setattr(mind, "read_slots", lambda: [{"is_processing": False}])

    def cut(*a):
        raise mind.Interrupted("the operator sent a request")
    monkeypatch.setattr(mind, "stream_chat", cut)
    hub = _Hub()
    assert mind.look_once(hub, "SYS", "h") == mind.BUSY_RETRY_S
    assert agent_says[0] == ["start", "end"]          # no "result": nothing was decided
    assert agent_says[2]["end"] == "cut off: the operator sent a request"


def test_user_message_carries_journal_then_diary():
    m = mind.build_messages("SYS", {"journal": "", "diary": "DIARY"})
    assert m[0] == {"role": "system", "content": "SYS"}
    assert m[1]["content"].startswith("YOUR JOURNAL:\n(empty")
    assert m[1]["content"].endswith("DIARY")


def test_prompt_files_exist_and_hash_is_stable():
    persona, world = mind.load_prompt_parts()
    assert persona.startswith("You are Eva")
    assert "Nobody has given you a task" in world
    s = mind.system_prompt(persona, world)
    assert mind.prompt_hash(s) == mind.prompt_hash(s) and len(mind.prompt_hash(s)) == 16


# ── config pins ───────────────────────────────────────────────────────────

def test_mindcraft_settings_forbid_code_and_outbound_wiki():
    s = (ROOT / "nova" / "config" / "mindcraft_settings.js").read_text()
    assert '"allow_insecure_coding": false' in s
    for blocked in ("!searchWiki", "!newAction"):
        assert f'"{blocked}"' in s
    assert '"auth": "offline"' in s and '"host": "@BIND_IP@"' in s


def test_body_profile_stays_local():
    p = json.loads((ROOT / "nova" / "config" / "eva_body.json").read_text())
    # an embedding block without a url makes Mindcraft call api.openai.com
    assert p["embedding"]["url"].startswith("http://127.0.0.1")
    assert p["model"]["url"].startswith("http://127.0.0.1")
    # 10-26 operator: the body model gets swapped on nova; the profile names
    # no model family (llama-server serves its one model under any name)
    assert p["model"]["model"] == "body"
    assert p["model"]["params"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_world_difficulty_matches_what_eva_is_told():
    # peaceful: the body's 4-8 s per move is too slow to fight; the prompt must agree
    props = (ROOT / "nova" / "config" / "server.properties.template").read_text()
    assert "\ndifficulty=peaceful\n" in props
    assert "peaceful difficulty" in (ROOT / "mac" / "world_prompt.txt").read_text()


def _patch_tree(tmp_path, viewer_line):
    """A WORLD with our config and a Mindcraft whose viewer has `viewer_line`."""
    world = tmp_path / "World"
    cfg = world / "app" / "config"
    cfg.mkdir(parents=True)
    for name in ("mindcraft_settings.js", "eva_body.json", "keys.json"):
        shutil.copy(ROOT / "nova" / "config" / name, cfg / name)
    (cfg / "world.json").write_text('{"bind": "100.64.0.9"}')
    pv = world / "mindcraft" / "node_modules" / "prismarine-viewer" / "lib"
    pv.mkdir(parents=True)
    (pv / "mineflayer.js").write_text(viewer_line + "\n")
    return world, pv / "mineflayer.js"


def _apply(world):
    return subprocess.run(["bash", str(ROOT / "nova" / "scripts" / "apply_patches.sh")],
                          env={**os.environ, "WORLD": str(world)},
                          capture_output=True, text=True, timeout=60)


def test_apply_patches_refuses_to_start_when_the_viewer_patch_cannot_apply(tmp_path):
    # upstream changed the line: the viewer would listen on every interface
    world, viewer = _patch_tree(tmp_path, "server.listen(port)")
    r = _apply(world)
    assert r.returncode != 0
    assert "viewer localhost patch did not apply" in r.stderr
    assert '"host": "100.64.0.9"' in (world / "mindcraft" / "settings.js").read_text()


def test_apply_patches_binds_the_stock_viewer_to_localhost(tmp_path):
    world, viewer = _patch_tree(tmp_path, "http.listen(port, () => {")
    r = _apply(world)
    assert "http.listen(port, '127.0.0.1', () => {" in viewer.read_text()
    # the next patch's file is missing from this tree, so it must still stop
    assert r.returncode != 0


def _full_patch_tree(tmp_path):
    """A WORLD where every earlier patch is already in place, plus viewer skins."""
    world, viewer = _patch_tree(tmp_path, "http.listen(port, '127.0.0.1', () => {")
    mc = world / "mindcraft"
    (world / "app" / "scripts").mkdir(parents=True)
    for name in ("patch_physics.cjs", "patch_viewer.cjs", "patch_craft.cjs"):
        shutil.copy(ROOT / "nova" / "scripts" / name, world / "app" / "scripts")
    bv = mc / "src" / "agent" / "vision"
    bv.mkdir(parents=True)
    (bv / "browser_viewer.js").write_text("mineflayerViewer(bot, { port: 3000+count_id, firstPerson: false, prefix: '/viewer' });\n")
    ph = mc / "node_modules" / "mineflayer" / "lib" / "plugins"
    ph.mkdir(parents=True)
    (ph / "physics.js").write_text("// EVA-WORLD-GUARD v2 already here\n")
    tex = mc / "node_modules" / "prismarine-viewer" / "public" / "textures"
    (tex / "1.21.4" / "entity" / "player" / "wide").mkdir(parents=True)
    (tex / "1.16.4" / "entity").mkdir(parents=True)
    (tex / "1.21.4" / "entity" / "player" / "wide" / "noor.png").write_bytes(b"NOOR")
    (tex / "1.16.4" / "entity" / "steve.png").write_bytes(b"STEVE")
    (tex.parent / "index.js").write_text(_BUNDLE)
    sk = mc / "src" / "agent" / "library"
    sk.mkdir(parents=True)
    (sk / "skills.js").write_text(_CRAFT_FN)
    return world, tex


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_apply_patches_draws_the_body_with_eva_s_skin_and_keeps_steve(tmp_path, monkeypatch):
    world, tex = _full_patch_tree(tmp_path)
    monkeypatch.setenv("NODE", shutil.which("node"))
    for _ in range(2):  # idempotent: the second run must not back up Noor as Steve
        r = _apply(world)
        assert r.returncode == 0, r.stderr
    assert (tex / "1.16.4" / "entity" / "steve.png").read_bytes() == b"NOOR"
    assert (tex / "1.16.4" / "entity" / "steve.png.orig").read_bytes() == b"STEVE"


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_apply_patches_refuses_to_start_without_the_skin(tmp_path, monkeypatch):
    world, tex = _full_patch_tree(tmp_path)
    monkeypatch.setenv("NODE", shutil.which("node"))
    (tex / "1.21.4" / "entity" / "player" / "wide" / "noor.png").unlink()
    r = _apply(world)
    assert r.returncode != 0 and "viewer skin files missing" in r.stderr


_BUNDLE = ('x=1;const u=new r(l);let h=new THREE.OrbitControls(u.camera,l.domElement);y=2;'
           'o.on("position",(({pos:t,addMesh:i,yaw:r,pitch:o})=>{z=3}));')


def _patch_viewer(path):
    return subprocess.run(["node", str(ROOT / "nova" / "scripts" / "patch_viewer.cjs"), str(path)],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_find_eva_hook_aims_close_and_from_above(tmp_path):
    """Run the patched bundle with a stub THREE: the button must land the camera
    12 blocks from her chest, never below her, keeping the side you looked from."""
    f = tmp_path / "index.js"
    f.write_text(_BUNDLE)
    assert _patch_viewer(f).returncode == 0
    once = f.read_text()
    assert _patch_viewer(f).returncode == 0 and f.read_text() == once  # idempotent
    harness = tmp_path / "run.cjs"
    harness.write_text(r"""
class V { constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z}
  clone(){return new V(this.x,this.y,this.z)} copy(v){this.x=v.x;this.y=v.y;this.z=v.z;return this}
  sub(v){this.x-=v.x;this.y-=v.y;this.z-=v.z;return this} set(x,y,z){this.x=x;this.y=y;this.z=z;return this}
  lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}
  normalize(){const l=Math.sqrt(this.lengthSq());this.x/=l;this.y/=l;this.z/=l;return this}
  addScaledVector(v,s){this.x+=v.x*s;this.y+=v.y*s;this.z+=v.z*s;return this} }
global.window = {};
global.THREE = { Vector3: V, OrbitControls: function (cam) { this.target = new V(); this.update = () => {}; global.__ctl = this; } };
const r = function () { this.camera = { position: new V(-30, 50, 0) }; global.__view = this; };  // camera far, behind (-x)
let l = {}, o = { on: (n, f) => f({ pos: { x: 100, y: 64, z: 7 } }) };
eval(require('fs').readFileSync(process.argv[2], 'utf8'));
const ok = window.__evaRecenter();
const cam = new V().copy(__view.camera.position), tgt = __ctl.target;
console.log(JSON.stringify({ ok, tgt, dist: Math.sqrt(cam.clone().sub(tgt).lengthSq()), cam }));
""")
    r = subprocess.run(["node", str(harness), str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ok"] is True
    assert out["tgt"] == {"x": 100, "y": 65, "z": 7}                # her chest
    assert abs(out["dist"] - 12) < 1e-6                              # close, not 28 blocks
    assert out["cam"]["y"] >= 65 + 4 - 1e-6                          # from above
    assert out["cam"]["x"] < 100                                     # same side as before


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_find_eva_hook_upgrades_v1_and_refuses_a_changed_bundle(tmp_path):
    f = tmp_path / "index.js"
    f.write_text(_BUNDLE)
    _patch_viewer(f)
    fresh = f.read_text()
    v1 = _BUNDLE.replace("let h=new THREE.OrbitControls(u.camera,l.domElement);",
                         "let h=new THREE.OrbitControls(u.camera,l.domElement);/*EVA-WORLD-RECENTER v1*/"
                         "window.__evaRecenter=()=>{h.update();return!0};").replace(
                         "pitch:o})=>{", "pitch:o})=>{window.__evaLast=t;")
    f.write_text(v1)
    assert _patch_viewer(f).returncode == 0 and f.read_text() == fresh
    f.write_text("upstream rewrote the viewer")
    r = _patch_viewer(f)
    assert r.returncode != 0 and "anchor found 0 times" in r.stderr


_CRAFT_FN = """async function craftRecipe(bot, itemName, num, recipe, craftLimit, craftingTable, placedTable) {
    await bot.craft(recipe, Math.min(craftLimit.num, num), craftingTable);
    log(bot, `Successfully crafted ${itemName}, you now have ${world.getInventoryCounts(bot)[itemName]} ${itemName}.`);
    return true;
}
module.exports = craftRecipe;
"""

_CRAFT_HARNESS = r"""
// A bot whose server undoes the crafts numbered in `undo` (1-based): mineflayer's
// own inventory shows the item at once, the server's correction removes it 20 ms
// later. Craft number `throwAt` throws mineflayer's "missing ingredient"; during
// craft number `stopAt` Mindcraft asks the action to stop (bot.interrupt_code).
const [file, item, num, limit, undo, throwAt, stopAt] = [process.argv[2], process.argv[3], Number(process.argv[4]),
  Number(process.argv[5]), JSON.parse(process.argv[6]), Number(process.argv[7]), Number(process.argv[8])];
const inv = {}; let crafts = 0, collected = 0; const out = [], counts = [];
global.world = { getInventoryCounts: () => ({ ...inv }) };
global.mc = { getItemId: () => 1 };
global.log = (bot, m) => out.push(m);
global.collectBlock = async () => { collected++; };
const bot = { interrupt_code: false, recipesFor: () => [{ again: true }],
  craft: async (recipe, count) => { crafts++; counts.push(count);
    if (crafts === stopAt) setTimeout(() => { bot.interrupt_code = true; }, 10);
    if (crafts === throwAt) throw new Error('Error: missing ingredient');
    inv[item] = (inv[item] || 0) + count;
    if (undo.includes(crafts)) setTimeout(() => { inv[item] -= count; }, 20); } };
const craftRecipe = require(file);
const t0 = Date.now();
craftRecipe(bot, item, num, { first: true }, { num: limit }, { table: true }, true).then(
  (ok) => console.log(JSON.stringify({ ok, crafts, counts, have: inv[item] || 0, collected, out, ms: Date.now() - t0 })),
  (err) => console.log(JSON.stringify({ threw: String(err.message), crafts, counts, out })));
"""


def _craft_run(tmp_path, undo=(), num=1, limit=None, throw_at=0, item="wooden_pickaxe", source=None,
               stop_at=0, settle_ms=80):
    f = tmp_path / "craft.cjs"
    f.write_text(source or _CRAFT_FN)
    r = subprocess.run(["node", str(ROOT / "nova" / "scripts" / "patch_craft.cjs"), str(f)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    h = tmp_path / "h.cjs"
    h.write_text(_CRAFT_HARNESS)
    r = subprocess.run(["node", str(h), str(f), item, str(num), str(limit or num), json.dumps(list(undo)), str(throw_at),
                        str(stop_at)],
                       capture_output=True, text=True, timeout=30, env={**os.environ, "EVA_CRAFT_SETTLE_MS": str(settle_ms)})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_a_craft_the_server_undoes_is_retried_then_reported_as_not_done(tmp_path):
    every = _craft_run(tmp_path, undo=range(1, 10))   # 10-26: every table craft rolled back
    assert every["ok"] is False and every["crafts"] == 2 and every["have"] == 0
    assert every["out"] == ["Crafting wooden_pickaxe did not go through: the server undid it, so nothing was used up."]
    assert every["collected"] == 1                    # a table it placed itself is picked back up
    once = _craft_run(tmp_path, undo=[1])             # transient: the retry lands
    assert once["ok"] is True and once["crafts"] == 2 and once["have"] == 1
    assert once["out"] == ["Successfully crafted wooden_pickaxe, you now have 1 wooden_pickaxe."]
    clean = _craft_run(tmp_path)
    assert clean["ok"] is True and clean["crafts"] == 1


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_several_crafts_go_one_at_a_time_each_confirmed(tmp_path):
    """10-26 12:23: !craftRecipe("torch", 4) → mineflayer's back-to-back multi-craft
    made one, then threw "missing ingredient" from its stale inventory."""
    four = _craft_run(tmp_path, item="torch", num=4)
    assert four["ok"] is True and four["counts"] == [1, 1, 1, 1] and four["have"] == 4
    assert four["out"] == ["Successfully crafted torch, you now have 4 torch."]
    undone = _craft_run(tmp_path, item="torch", num=4, undo=[2])   # one undone mid-way: retried, all four land
    assert undone["ok"] is True and undone["crafts"] == 5 and undone["have"] == 4
    part = _craft_run(tmp_path, item="torch", num=4, undo=[3, 4])  # the third never lands
    assert part["ok"] is True and part["crafts"] == 4 and part["have"] == 2
    assert part["out"] == ["Crafted torch 2 of 4 times; the server undid the rest. You now have 2 torch."]
    late = _craft_run(tmp_path, item="torch", num=4, throw_at=2)   # an error after one landed: say so
    assert late["ok"] is True and late["have"] == 1
    assert late["out"] == ["Crafted torch 1 of 4 times; then: missing ingredient. You now have 1 torch."]
    first = _craft_run(tmp_path, item="torch", num=4, throw_at=1)  # nothing made: upstream's error path
    assert "missing ingredient" in first["threw"] and first["crafts"] == 1
    capped = _craft_run(tmp_path, item="torch", num=4, limit=2)    # ingredients for 2: upstream's message follows
    assert capped["counts"] == [1, 1] and capped["have"] == 2
    # a partial success is not worded as a failed command (the hub's FAIL_RUN pattern)
    from re import search
    attention = (ROOT / "nova" / "hub" / "lib" / "attention.js").read_text()
    assert "'did not go through'" in attention
    assert "did not go through" not in part["out"][0] and "did not go through" not in late["out"][0]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_a_stop_request_ends_the_crafts_at_once(tmp_path):
    """Review 10-26: Mindcraft's stop() kills the body process when an action
    ignores bot.interrupt_code for 10 s; 8 crafts at 3 s each would."""
    run = _craft_run(tmp_path, item="stick", num=8, stop_at=2, settle_ms=3000)
    assert run["crafts"] == 2 and run["ms"] < 4000      # one full settle, then out of the second (unchecked: 24 s)
    assert run["ok"] is True and run["out"] == ["Crafting stick was stopped after 2 of 8. You now have 2 stick."]
    assert run["collected"] == 0                                  # no walking to the table after a stop
    first = _craft_run(tmp_path, item="stick", num=8, stop_at=1, settle_ms=3000)
    assert first["crafts"] == 1 and first["ms"] < 1500 and first["out"][0].startswith("Crafting stick was stopped after 1 of 8")



def _old_block(name):
    """A previous version's inserted block, as the patcher keeps it."""
    src = (ROOT / "nova" / "scripts" / "patch_craft.cjs").read_text()
    start = src.index(f"const {name} = `") + len(f"const {name} = `")
    return src[start:src.index("`;\n", start)].replace("\\`", "`").replace("\\$", "$")


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("old", ["V1", "V2"])
def test_an_older_craft_check_is_upgraded(tmp_path, old):
    """nova's skills.js carried v1, then v2, when the next version shipped."""
    anchor = "    await bot.craft(recipe, Math.min(craftLimit.num, num), craftingTable);\n"
    fn = _CRAFT_FN.replace(anchor, _old_block(old))
    assert f"EVA-WORLD-CRAFT-CHECK v{old[1]}" in fn
    four = _craft_run(tmp_path, item="torch", num=4, source=fn)
    assert four["counts"] == [1, 1, 1, 1] and four["have"] == 4
    patched = (tmp_path / "craft.cjs").read_text()
    assert "CHECK v1" not in patched and "CHECK v2" not in patched
    assert patched.count("EVA-WORLD-CRAFT-CHECK v3") == 1


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_the_craft_check_is_idempotent_and_refuses_a_changed_skills_file(tmp_path):
    f = tmp_path / "skills.js"
    f.write_text(_CRAFT_FN)
    cmd = ["node", str(ROOT / "nova" / "scripts" / "patch_craft.cjs"), str(f)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    once = f.read_text()
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    assert f.read_text() == once
    f.write_text("upstream rewrote craftRecipe")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "anchor found 0 times" in r.stderr


# ── the eva-world command (Mac) and the reset script (nova) ─────────────

def _fake_hub(phase="running"):
    """A hub on a free localhost port that records POSTs."""
    import http.server, threading
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("X-World-Key") != "K":
                return self._send(401, {"error": "key required"})
            self._send(200, {"phase": phase, "session": "s1", "now": 0, "wakeAt": 300000, "currentGoal": "chop",
                             "mind": {"status": "idle"},
                             "state": {"gameplay": {"position": {"x": 1.4, "y": 64, "z": -2.6}, "biome": "plains",
                                                    "health": 20, "hunger": 19},
                                       "action": {"current": "action:stay"}}})

        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            seen.append(body)
            if body.get("action") == "pause":
                return self._send(409, {"error": "world is stopped"})
            self._send(200, {"result": "ok"})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _eva_world(tmp_path, srv, *args):
    key = tmp_path / "key"
    key.write_text("K")
    env = {**os.environ, "EVA_WORLD_HUB": f"http://127.0.0.1:{srv.server_address[1]}", "EVA_WORLD_KEY_FILE": str(key)}
    return subprocess.run(["bash", str(ROOT / "mac" / "eva-world"), *args], capture_output=True, text=True,
                          timeout=60, env=env)


def test_eva_world_status_start_and_errors(tmp_path):
    srv, seen = _fake_hub()
    try:
        r = _eva_world(tmp_path, srv, "status")
        assert r.returncode == 0, r.stderr
        assert "world:   running  (session s1)" in r.stdout
        assert "eva:     at 1, 64, -3 (plains), health 20/20, food 19/20" in r.stdout
        assert "doing:   stay" in r.stdout and "looks:   next in ~5 min" in r.stdout
        assert _eva_world(tmp_path, srv, "start").returncode == 0
        assert seen[-1] == {"action": "start"}
        bad = _eva_world(tmp_path, srv, "pause")          # the hub's refusal reaches the user
        assert bad.returncode == 1 and "world is stopped (HTTP 409)" in bad.stderr
        assert _eva_world(tmp_path, srv, "dance").returncode == 2
    finally:
        srv.shutdown()


def test_eva_world_reset_needs_the_word_yes(tmp_path):
    srv, seen = _fake_hub()
    try:
        r = subprocess.run(["bash", str(ROOT / "mac" / "eva-world"), "reset"], input="no\n", capture_output=True,
                           text=True, timeout=30, env={**os.environ, "EVA_WORLD_HUB": f"http://127.0.0.1:{srv.server_address[1]}",
                                                       "EVA_WORLD_KEY_FILE": str((tmp_path / "k").write_text("K") and tmp_path / "k")})
        assert r.returncode == 1 and "cancelled" in r.stdout
        assert seen == []                                  # nothing stopped, nothing wiped
    finally:
        srv.shutdown()


def _world_to_reset(tmp_path, phase, live_pid=None):
    w = tmp_path / "World"
    for d in ("server/evaworld/region", "server/evaworld_nether", "server/evaworld_the_end", "data/sessions/s1",
              "data/eva/prompts", "mindcraft/bots/Eva", "run", "app"):
        (w / d).mkdir(parents=True)
    (w / "data/sessions/s1/events.jsonl").write_text("{}")
    (w / "data/eva/journal.md").write_text("## day 1")
    (w / "mindcraft/bots/Eva/memory.json").write_text("{}")
    (w / "server/server.properties").write_text("difficulty=peaceful")
    (w / "run/state.json").write_text(json.dumps({"phase": phase}))
    if live_pid:
        (w / "run/server.pid").write_text(str(live_pid))
    return w


def _reset(w, *args):
    return subprocess.run(["bash", str(ROOT / "nova" / "scripts" / "reset_world.sh"), *args], capture_output=True,
                          text=True, timeout=30, env={**os.environ, "WORLD": str(w), "HUB_RESTART_CMD": "echo hub-kicked"})


def test_reset_wipes_everything_but_only_when_stopped(tmp_path):
    w = _world_to_reset(tmp_path / "a", "running")
    r = _reset(w, "--yes")
    assert r.returncode == 1 and "the world is running" in r.stderr
    assert (w / "data/eva/journal.md").exists()
    w = _world_to_reset(tmp_path / "b", "stopped", live_pid=os.getpid())   # a server still alive
    r = _reset(w, "--yes")
    assert r.returncode == 1 and "still running" in r.stderr and (w / "server/evaworld").exists()
    w = _world_to_reset(tmp_path / "c", "stopped")
    assert _reset(w).returncode == 2 and (w / "server/evaworld").exists()  # no --yes, no wipe
    r = _reset(w, "--yes")
    assert r.returncode == 0, r.stderr
    for gone in ("server/evaworld", "server/evaworld_nether", "server/evaworld_the_end", "data/eva/journal.md",
                 "mindcraft/bots/Eva", "data/sessions/s1"):
        assert not (w / gone).exists(), gone
    assert (w / "data/sessions").is_dir() and (w / "data/eva/prompts").is_dir()   # the hub needs them
    assert (w / "server/server.properties").read_text() == "difficulty=peaceful"   # settings stay
    assert "hub-kicked" in r.stdout


def test_eva_is_a_woman_in_both_prompts():
    assert "The body is a woman, a player named Eva." in (ROOT / "mac" / "world_prompt.txt").read_text()
    body = json.loads((ROOT / "nova" / "config" / "eva_body.json").read_text())
    assert "You are $NAME, a woman and a player" in body["conversing"]


def test_the_body_waits_between_model_calls():
    """10-26 (operator, "slow it down"): back to back, the body kept nova's GPU at 100%."""
    body = json.loads((ROOT / "nova" / "config" / "eva_body.json").read_text())
    assert body["cooldown"] == 30000          # Mindcraft prompter: ms between call starts


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_hub_node_unit_tests_pass():
    hub = ROOT / "nova" / "hub"
    files = sorted(str(p) for p in (hub / "test").glob("*.test.js"))
    r = subprocess.run(["node", "--test", *files], cwd=hub, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


# ── §4RQ: the agent tells outside callers the slot is kept ───────────────

def _health(llm):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.api import routes
    agent = SimpleNamespace(context=SimpleNamespace(self_model=None, llm_client=llm, scheduler=None, memory_system=None))
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=agent, biological_task=None, boot_monotonic=None, resolved_config={}))
    resp = asyncio.new_event_loop().run_until_complete(routes.api_health(req))
    return json.loads(resp.body)


def test_health_reports_seconds_the_slot_is_still_kept(monkeypatch):
    """§4RP keeps the slot 600 s for the conversation that last used it; Eva's
    mind calls llama directly, so without this field it would push a fresh
    conversation out and the owner's next turn would re-process it."""
    from types import SimpleNamespace
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    now = time.monotonic()
    kept = _health(SimpleNamespace(foreground_requests=0, _slot_session="abc", _slot_session_used=now - 100))
    assert 495 <= kept["slot_keep_left_s"] <= 500
    expired = _health(SimpleNamespace(foreground_requests=0, _slot_session="abc", _slot_session_used=now - 700))
    assert expired["slot_keep_left_s"] == 0.0
    nobody = _health(SimpleNamespace(foreground_requests=0, _slot_session=None, _slot_session_used=now))
    assert nobody["slot_keep_left_s"] == 0.0
    assert _health(None)["slot_keep_left_s"] == 0.0          # a degraded boot never 500s


def test_agent_busy_while_the_slot_is_kept():
    h = {"foreground_requests": 0, "slot_keep_left_s": 312.7}
    busy, why = mind.agent_busy(h, [{"is_processing": False}])
    assert busy and "313 s" in why
    assert mind.agent_busy({"foreground_requests": 0, "slot_keep_left_s": 0.0}, [{"is_processing": False}])[0] is False
    # an agent from before §4RQ has no field: unknown counts as busy
    assert mind.agent_busy({"foreground_requests": 0}, [{"is_processing": False}])[0] is True


def _outside_call(llm, body):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.api import routes
    req = MagicMock()
    req.app = SimpleNamespace(state=SimpleNamespace(agent=SimpleNamespace(context=SimpleNamespace(llm_client=llm))))

    async def _json():
        return body
    req.json = _json
    resp = asyncio.new_event_loop().run_until_complete(routes.api_slot_outside_call(req))
    return resp.status_code, json.loads(resp.body)


def test_the_agent_has_the_last_word_on_an_outside_call(monkeypatch):
    """Eva's mind asks before every look: refused while a request is live or a
    conversation keeps the slot, so a request in the mind's check-to-call gap
    still wins."""
    from types import SimpleNamespace
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    now = time.monotonic()
    live = SimpleNamespace(foreground_requests=1, _main_streams_open=0, _slot_session=None, _head_ref=None)
    assert _outside_call(live, {"state": "start"})[0] == 409
    kept = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session="a",
                           _slot_session_used=now - 10, _head_ref=None)
    code, body = _outside_call(kept, {"state": "start"})
    assert code == 409 and "keeps the slot" in body["reason"]
    assert not getattr(kept, "_outside_slot_dirty", False)     # refused: nothing marked
    idle = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session=None, _head_ref=None)
    assert _outside_call(idle, {"state": "start", "label": "eva_world"}) == (200, {"ok": True})
    assert idle._outside_call_live is True and idle._outside_slot_dirty is True
    # "end" is always taken, even with a request live (that is when it matters)
    idle.foreground_requests = 1
    assert _outside_call(idle, {"state": "end"})[0] == 200
    assert idle._outside_call_live is False and idle._outside_slot_dirty is True
    assert _outside_call(idle, {"state": "nonsense"})[0] == 400


def test_every_outside_call_is_a_background_line_in_the_agents_log(monkeypatch):
    """10-26 (operator): Eva's looks were invisible in the agent's log."""
    from types import SimpleNamespace
    from ghost_agent.api import routes
    from ghost_agent.utils.logging import request_id_context, Icons
    lines = []
    monkeypatch.setattr(routes, "pretty_log",
                        lambda title, text, icon=None, **k: lines.append((request_id_context.get(), title, text, icon)))
    monkeypatch.delenv("GHOST_SLOT_KEEP_S", raising=False)
    tok = request_id_context.set("probe-abc")               # the HTTP call's own context
    try:
        live = SimpleNamespace(foreground_requests=1, _main_streams_open=0, _slot_session=None, _head_ref=None)
        _outside_call(live, {"state": "start"})
        idle = SimpleNamespace(foreground_requests=0, _main_streams_open=0, _slot_session=None, _head_ref=None)
        _outside_call(idle, {"state": "start", "detail": "woke because: your body died"})
        idle._outside_call_started -= 52
        _outside_call(idle, {"state": "end", "detail": "finished"})
        assert _outside_call(idle, {"state": "result", "detail": "set the goal: \"chop\"\n wrote in your journal"}) == (200, {"ok": True})
    finally:
        request_id_context.reset(tok)
    assert [l[2] for l in lines] == ["look refused: 1 request(s) live",
                                      "look started — woke because: your body died",
                                      "look ended after 52s (finished)",
                                      'decided: set the goal: "chop" wrote in your journal']
    assert {(l[0], l[1], l[3]) for l in lines} == {("SYSTEM", "eva's world", Icons.OUTSIDE_LOOK)}
    assert idle._outside_slot_dirty is True                 # "result" changed nothing about the slot
