"""§4VP — one command to switch the model (`scripts/use_model.py`) and the
functional scorecard (`scripts/model_scorecard.py`).

The switch is tested against a directory of made-up launchers, a stand-in
model server and agent, and a "restart" that is a small script flipping what
the stand-in server says it runs. The scorecard's checks, card and table are
tested on canned replies; its live run is the acceptance test, not a unit test.
"""
from __future__ import annotations

import base64
import http.server
import importlib.util
import json
import os
import stat
import struct
import sys
import threading
import time
import types
import zlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


UM = _load("use_model")
SC = _load("model_scorecard")


class _World:
    """A model server (/props) and an agent (/api/health) that follows it."""

    def __init__(self, tmp_path, follows=True, busy=0, status=200, key=None, no_count=False):
        self.state = tmp_path / "running.txt"          # what the "server" runs: the restart script writes it
        self.follows, self.busy, self.health_hits = follows, busy, 0
        self.status, self.key, self.no_count, self.keys_seen = status, key, no_count, []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                running = outer.state.read_text().strip() if outer.state.exists() else ""
                code = 200
                if self.path == "/props":
                    body = {"model_path": running} if running else None
                elif self.path == "/api/health":
                    outer.health_hits += 1
                    outer.keys_seen.append(self.headers.get("X-Ghost-Key"))
                    assert self.headers.get("X-Ghost-Origin") == "probe"
                    busy = 1 if outer.health_hits <= outer.busy else 0
                    seen = os.path.basename(running) if outer.follows else "Old-Model.gguf"
                    body = {"model_profile": {"name": "p", "why": "w", "server_model": seen, "tokens": {"exact": True}}}
                    if not outer.no_count:
                        body["foreground_requests"] = busy
                    if outer.status != 200:
                        code, body = outer.status, {"detail": "no"}
                    elif outer.key is not None and self.headers.get("X-Ghost-Key") != outer.key:
                        code, body = 403, {"detail": "bad key"}
                else:
                    body = None
                self.send_response(code if body is not None else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body or {}).encode())

            def log_message(self, *a):
                pass
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def world(tmp_path):
    made = []

    def make(**kw):
        made.append(_World(tmp_path, **kw))
        return made[-1]
    yield make
    for w in made:
        w.close()


@pytest.fixture(autouse=True)
def _no_key_from_the_machine(monkeypatch, tmp_path):
    """The key comes from what a test sets, never from this machine."""
    monkeypatch.delenv("GHOST_KEY_FILE", raising=False)
    monkeypatch.setenv("GHOST_API_KEY", "test-key")
    monkeypatch.setattr(UM, "REPO", tmp_path / "norepo" / "tree")
    monkeypatch.setattr(SC, "REPO", tmp_path / "norepo" / "tree")


@pytest.fixture
def bindir(tmp_path):
    """bin/ with a live launcher (model A), named launchers for A and B, and
    a restart script that makes the stand-in server run the live launcher's
    model. Words in the live launcher make the restart misbehave: REFUSE (exit
    3, server untouched), LATE (exit 3, but the server does come up), NEVERUP
    (exit 0, server down). Every run is counted in restarts.txt."""
    b = tmp_path / "bin"
    b.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    for n in ("Model-A.gguf", "Model-B.gguf"):
        (models / n).write_bytes(b"gguf")

    def launcher(model, extra=""):
        return f'#!/bin/bash\n# a launcher\nLLAMA_SERVER="/usr/bin/true"\nMODEL="{models / model}"\n{extra}exec true\n'
    (b / "start-llama-server.sh").write_text(launcher("Model-A.gguf"))
    (b / "start-llama-server.sh").chmod(0o755)
    (b / "start-llama-server.sh.a").write_text(launcher("Model-A.gguf"))
    (b / "start-llama-server.sh.b").write_text(launcher("Model-B.gguf"))
    (b / "start-llama-server.sh.gone").write_text(launcher("Missing.gguf"))
    (b / "start-llama-server.sh.nomodel").write_text("#!/bin/bash\nexec true\n")
    (b / "start-llama-server.sh.dense.daisy_rose-2").write_text(launcher("Model-A.gguf", "# another\n"))
    (b / "start-llama-server.sh.bad name").write_text(launcher("Model-A.gguf"))
    restart = b / "restart-llama-server.sh"
    restart.write_text(f"""#!{sys.executable}
import re, sys
with open({str(tmp_path / 'restarts.txt')!r}, "a") as f:
    f.write("x")
live = open({str(b / 'start-llama-server.sh')!r}).read()
model = re.search(r'^MODEL="(.+?)"', live, re.M).group(1)
if "REFUSE" in live:
    sys.stderr.write("Still alive after 120s")
    sys.exit(3)
open({str(tmp_path / 'running.txt')!r}, "w").write("" if "NEVERUP" in live else model)
if "LATE" in live:
    sys.exit(3)
""")
    restart.chmod(restart.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "running.txt").write_text(str(models / "Model-A.gguf"))
    return b


def _restarts(tmp_path) -> int:
    f = tmp_path / "restarts.txt"
    return len(f.read_text()) if f.exists() else 0


def _launcher_with(bindir, name, word):
    f = bindir / f"start-llama-server.sh.{name}"
    f.write_text(f.read_text().replace("exec true", f"# {word}\nexec true"))


def _no_leftovers(bindir):
    return not [p.name for p in bindir.iterdir() if ".tmp-" in p.name]


FAST = dict(idle_wait=2.0, load_wait=2.0, follow_wait=2.0, step=0.05)


# --- the launchers -----------------------------------------------------------------------

def test_the_launchers_and_what_each_starts(bindir):
    rows = {r["name"]: r for r in UM.launchers(bindir)}
    assert set(rows) == {"a", "b", "gone", "nomodel", "dense.daisy_rose-2"}   # every well-formed name; the live file is not one
    assert rows["a"]["live"] and not rows["b"]["live"] and not rows["dense.daisy_rose-2"]["live"]
    assert rows["b"]["model"].endswith("Model-B.gguf") and rows["b"]["exists"]
    assert rows["gone"]["exists"] is False and rows["nomodel"]["model"] == "" and rows["nomodel"]["exists"] is False


@pytest.mark.parametrize("text, want", [
    ('MODEL="/m/a.gguf"\n', "/m/a.gguf"),
    ("MODEL='/m/a b.gguf'\n", "/m/a b.gguf"),
    ("  MODEL=/m/a.gguf   # the model\n", "/m/a.gguf"),
    ('export MODEL="/m/a.gguf";\n', "/m/a.gguf"),
    ('# MODEL="/m/old.gguf"\nMODEL="/m/new.gguf"\n', "/m/new.gguf"),
    ('MODEL="/m/first.gguf"\nMODEL="/m/last.gguf"\n', "/m/last.gguf"),      # as the shell: the last one stands
    ('DIR=/m\nMODEL="$DIR/a.gguf"\n', ""),                                  # cannot be checked: not guessed
    ('MMPROJ="/m/p.gguf"\nMODEL_DIR=/x\n', ""),
    ("", ""),
])
def test_the_model_line_of_a_launcher(tmp_path, text, want, monkeypatch):
    monkeypatch.delenv("DIR", raising=False)
    f = tmp_path / "l.sh"
    f.write_text(text)
    assert UM.launcher_model(f) == want
    assert UM.launcher_model(tmp_path / "missing.sh") == ""


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a b", ".hidden", "x" * 65])
def test_a_name_that_is_not_a_launcher_name(bindir, name):
    with pytest.raises(UM.SwitchError):
        UM.launcher_path(bindir, name)


# --- the switch ----------------------------------------------------------------------------

def test_a_switch_end_to_end(bindir, world, tmp_path):
    w = world()
    said = []
    got = UM.switch(bindir, "b", server=w.url, agent=w.url, say=said.append, **FAST)
    live = bindir / "start-llama-server.sh"
    assert live.read_text() == (bindir / "start-llama-server.sh.b").read_text()
    assert stat.S_IMODE((bindir / "start-llama-server.sh.b").stat().st_mode) & 0o111 == 0       # the named one is not executable
    assert stat.S_IMODE(live.stat().st_mode) & 0o111 == 0o111                                  # launchd runs the live one: it stays so
    assert (bindir / "start-llama-server.sh.previous").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert stat.S_IMODE((bindir / "start-llama-server.sh.previous").stat().st_mode) == 0o755      # what the live one had
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf") and _restarts(tmp_path) == 1
    assert got["server_model"] == "Model-B.gguf" and any("Model-B.gguf" in s for s in said)
    assert any("token counts exact" in s for s in said) and _no_leftovers(bindir)
    # … and back
    UM.switch(bindir, "a", server=w.url, agent=w.url, say=said.append, **FAST)
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")
    assert (bindir / "start-llama-server.sh.previous").read_text() == (bindir / "start-llama-server.sh.b").read_text()


def test_switch_previous_swaps_the_two(bindir, world, tmp_path):
    """Review: the old launcher was copied over `previous` before `previous` was
    read — it destroyed its own source and restarted the server twice for nothing."""
    w = world()
    a_text = (bindir / "start-llama-server.sh.a").read_text()
    b_text = (bindir / "start-llama-server.sh.b").read_text()
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    UM.switch(bindir, "previous", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert (bindir / "start-llama-server.sh").read_text() == a_text
    assert (bindir / "start-llama-server.sh.previous").read_text() == b_text
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf") and _restarts(tmp_path) == 2
    UM.switch(bindir, "previous", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert (bindir / "start-llama-server.sh").read_text() == b_text
    assert (bindir / "start-llama-server.sh.previous").read_text() == a_text


def test_a_dry_run_changes_nothing(bindir, world, tmp_path):
    w = world()
    before = {p.name: p.read_bytes() for p in bindir.iterdir()}
    said = []
    assert UM.switch(bindir, "b", server=w.url, agent=w.url, dry_run=True, say=said.append, **FAST) == {}
    assert {p.name: p.read_bytes() for p in bindir.iterdir()} == before and _restarts(tmp_path) == 0
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf") and any("would" in s for s in said)


def test_another_launcher_for_the_same_model_is_a_switch(bindir, world, tmp_path):
    """Same model file, other flags: the launcher differs, so it is put in place and the server restarted."""
    w = world()
    UM.switch(bindir, "dense.daisy_rose-2", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.dense.daisy_rose-2").read_text()
    assert _restarts(tmp_path) == 1 and (bindir / "start-llama-server.sh.previous").exists()


def test_what_is_already_live_is_left_alone(bindir, world, tmp_path):
    w = world()
    said = []
    UM.switch(bindir, "a", server=w.url, agent=w.url, say=said.append, **FAST)
    assert not (bindir / "start-llama-server.sh.previous").exists() and any("nothing to do" in s for s in said)
    assert _restarts(tmp_path) == 0


@pytest.mark.parametrize("running", ["", "Model-B.gguf"])
def test_the_live_launcher_with_a_server_on_something_else_is_restarted_and_previous_kept(bindir, world, tmp_path, running):
    """Review: the launcher is the live one but the server is down or runs another
    model. The restart is the whole job — `previous` must not become a copy of live."""
    w = world()
    (bindir / "start-llama-server.sh.previous").write_text("the operator's real previous")
    (tmp_path / "running.txt").write_text(str(tmp_path / "models" / running) if running else "")
    UM.switch(bindir, "a", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert _restarts(tmp_path) == 1 and (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")
    assert (bindir / "start-llama-server.sh.previous").read_text() == "the operator's real previous"


@pytest.mark.parametrize("name, why", [("zzz", "no launcher"), ("gone", "not there"), ("nomodel", "no MODEL= line")])
def test_a_switch_that_cannot_be_made_touches_nothing(bindir, world, name, why, tmp_path):
    w = world()
    before = {p.name: p.read_bytes() for p in bindir.iterdir()}
    with pytest.raises(UM.SwitchError, match=why):
        UM.switch(bindir, name, server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert {p.name: p.read_bytes() for p in bindir.iterdir()} == before and _restarts(tmp_path) == 0


def test_no_restart_command_is_found_out_before_anything_is_changed(bindir, world, tmp_path):
    w = world()
    (bindir / "restart-llama-server.sh").unlink()
    before = {p.name: p.read_bytes() for p in bindir.iterdir()}
    with pytest.raises(UM.SwitchError, match="no restart command"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert {p.name: p.read_bytes() for p in bindir.iterdir()} == before


def test_it_waits_for_the_agent_to_be_idle_and_gives_up_if_it_never_is(bindir, world, tmp_path):
    w = world(busy=4)                                                       # busy for the first four looks
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf") and w.health_hits > 4
    stuck = world(busy=10 ** 9)
    (tmp_path / "running.txt").write_text(str(tmp_path / "models" / "Model-B.gguf"))
    before = (bindir / "start-llama-server.sh").read_bytes()
    with pytest.raises(UM.SwitchError, match="stayed busy"):
        UM.switch(bindir, "a", server=stuck.url, agent=stuck.url, say=lambda s: None, **{**FAST, "idle_wait": 0.3})
    assert (bindir / "start-llama-server.sh").read_bytes() == before        # not switched under a request
    UM.switch(bindir, "a", server=stuck.url, agent=stuck.url, force=True, say=lambda s: None,
              **{**FAST, "follow_wait": 1.0})                               # --force does not wait
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")


@pytest.mark.parametrize("kind", ["refused key", "server error", "no count in the answer"])
def test_an_agent_whose_state_cannot_be_read_is_not_taken_for_idle(bindir, world, tmp_path, kind):
    """Review (MAJOR): a 403 or a timeout was read as "the agent is down, nothing
    in flight" and the server was restarted under whatever the agent was doing."""
    w = world(**{"refused key": {"key": "the-right-one"}, "server error": {"status": 500},
                 "no count in the answer": {"no_count": True}}[kind])
    before = {p.name: p.read_bytes() for p in bindir.iterdir()}
    with pytest.raises(UM.SwitchError, match="stayed busy" if kind == "no count in the answer" else "cannot tell whether the agent is idle"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "idle_wait": 0.3})
    assert {p.name: p.read_bytes() for p in bindir.iterdir()} == before and _restarts(tmp_path) == 0
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")


class _Silent:
    """Accepts a connection and never answers."""

    def __init__(self):
        import socket
        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(5)
        self.url = f"http://127.0.0.1:{self.s.getsockname()[1]}"

    def close(self):
        self.s.close()


def test_only_a_refused_connection_is_an_agent_that_is_down(bindir, world, tmp_path):
    """Second review: a mutant that read a timeout as "down" passed every test."""
    silent = _Silent()
    try:
        state, health, why = UM.ask_agent(silent.url, timeout=0.3)
        assert state == UM.UNREADABLE and health == {}
        assert UM.ask_agent("http://127.0.0.1:9")[0] == UM.DOWN
        assert UM.ask_agent("http://no-such-host.invalid:9", timeout=2)[0] == UM.UNREADABLE
        w = world()
        monkeypatch_get = UM._get
        try:
            UM._get = lambda *a, **k: (_ for _ in ()).throw(ValueError("not JSON"))
            assert UM.ask_agent(w.url)[0] == UM.UNREADABLE
            UM._get = lambda *a, **k: {}                                    # answered, but with nothing in it
            state, health, why = UM.ask_agent(w.url)
            assert state == UM.UNREADABLE and "empty" in why
        finally:
            UM._get = monkeypatch_get
    finally:
        silent.close()


def test_an_agent_that_goes_down_while_waited_for_is_idle(bindir, world, tmp_path, monkeypatch):
    w = world()
    real, asked = UM.ask_agent, []

    def ask(agent, timeout=5.0):
        asked.append(1)
        if _restarts(tmp_path):                                              # after the switch: the real stand-in
            return real(agent, timeout)
        return (UM.ANSWERED, {"foreground_requests": 1}, "") if len(asked) <= 2 else (UM.DOWN, {}, "nothing listens there")
    monkeypatch.setattr(UM, "ask_agent", ask)
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "idle_wait": 0.5})
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf")


def test_an_agent_that_becomes_unreadable_while_waited_for_is_not_idle(bindir, world, tmp_path, monkeypatch):
    """Third review: a mutant that took "not answered" for idle during the wait passed."""
    w = world()
    asked = []

    def ask(agent, timeout=5.0):
        asked.append(1)
        return (UM.ANSWERED, {"foreground_requests": 1}, "") if len(asked) <= 2 else (UM.UNREADABLE, {}, "timed out")
    monkeypatch.setattr(UM, "ask_agent", ask)
    with pytest.raises(UM.SwitchError, match="stayed busy"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "idle_wait": 0.4})
    assert _restarts(tmp_path) == 0 and len(asked) > 3


@pytest.mark.parametrize("count", [False, 0.0, "0", None, [], 1])
def test_only_the_number_zero_is_idle(bindir, world, tmp_path, monkeypatch, count):
    w = world()
    monkeypatch.setattr(UM, "ask_agent", lambda agent, timeout=5.0: (UM.ANSWERED, {"foreground_requests": count}, ""))
    with pytest.raises(UM.SwitchError, match="stayed busy"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "idle_wait": 0.2})
    assert _restarts(tmp_path) == 0


def test_the_key_the_agent_took_is_tried_first_from_then_on(world, tmp_path, monkeypatch):
    w = world(key="from-the-file")
    keyfile = tmp_path / "norepo" / ".ghost_api_key"
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("from-the-file\n")
    monkeypatch.setenv("GHOST_API_KEY", "stale")
    monkeypatch.setattr(UM, "_KEY_TAKEN", {})
    for _ in range(3):
        assert UM.ask_agent(w.url)[0] == UM.ANSWERED
    assert w.keys_seen == ["stale", "from-the-file", "from-the-file", "from-the-file"]


def test_the_next_key_is_tried_when_the_agent_refuses_one(bindir, world, tmp_path, monkeypatch):
    w = world(key="from-the-file")
    keyfile = tmp_path / "norepo" / ".ghost_api_key"                         # beside the (made-up) repository
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("from-the-file\n")
    monkeypatch.setenv("GHOST_API_KEY", "stale")
    assert UM.api_keys() == ["stale", "from-the-file"]
    monkeypatch.setattr(UM, "_KEY_TAKEN", {})
    state, health, why = UM.ask_agent(w.url)
    assert state == UM.ANSWERED and health["foreground_requests"] == 0 and w.keys_seen == ["stale", "from-the-file"]
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf")
    named = tmp_path / "k"
    named.write_text("named")
    monkeypatch.setenv("GHOST_KEY_FILE", str(named))
    assert UM.api_keys() == ["named", "stale", "from-the-file"] and UM.api_key() == "named"
    assert UM.ask_agent("http://127.0.0.1:9")[0] == UM.DOWN
    refusing = world(key="nobody-has-it")
    state, health, why = UM.ask_agent(refusing.url)
    assert state == UM.UNREADABLE and health == {} and "403" in why and "key" in why


def test_a_server_that_does_not_come_up_gets_the_previous_launcher_back(bindir, world, tmp_path):
    w = world()
    _launcher_with(bindir, "b", "NEVERUP")
    b_text = (bindir / "start-llama-server.sh.b").read_text()
    said = []
    with pytest.raises(UM.SwitchError, match="did not come up on Model-B.gguf"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=said.append, **{**FAST, "load_wait": 0.4})
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert (bindir / "start-llama-server.sh.b").read_text() == b_text
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")  # restarted on the launcher it had
    assert _restarts(tmp_path) == 2 and any("previous launcher is back" in s for s in said) and _no_leftovers(bindir)


@pytest.mark.parametrize("word", ["NEVERUP", "REFUSE"])
def test_a_switch_that_does_not_take_leaves_the_directory_as_it_was(bindir, world, tmp_path, word):
    """Second review: a failed switch still overwrote `previous` — a launcher kept
    only there was gone though nothing had been switched."""
    w = world()
    (bindir / "start-llama-server.sh.previous").write_text("#!/bin/bash\nMODEL=\"/only/kept/here.gguf\"\n")
    (bindir / "start-llama-server.sh.previous").chmod(0o750)
    _launcher_with(bindir, "b", word)
    before = {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in bindir.iterdir()}
    with pytest.raises(UM.SwitchError):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in bindir.iterdir()} == before


def test_the_live_launcher_is_always_runnable_and_a_rollback_gives_it_its_own_mode_back(bindir, world, tmp_path):
    w = world()
    live = bindir / "start-llama-server.sh"
    live.chmod(0o640)                                                        # an odd one: not executable at all
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert stat.S_IMODE(live.stat().st_mode) == 0o744                        # b's 0644 + the owner may run it
    assert stat.S_IMODE((bindir / "start-llama-server.sh.previous").stat().st_mode) == 0o640
    live.chmod(0o750)
    _launcher_with(bindir, "a", "NEVERUP")
    with pytest.raises(UM.SwitchError):
        UM.switch(bindir, "a", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert stat.S_IMODE(live.stat().st_mode) == 0o750                        # exactly what it had, not the mode of a switch


def test_a_rollback_that_fails_says_what_to_check(bindir, world, tmp_path, monkeypatch):
    w = world()
    _launcher_with(bindir, "b", "NEVERUP")
    real, n = UM._put, []

    def third_write_fails(data, mode, dst):
        n.append(dst.name)
        if len(n) >= 3:
            raise OSError(28, "No space left on device")
        return real(data, mode, dst)
    monkeypatch.setattr(UM, "_put", third_write_fails)
    with pytest.raises(UM.SwitchError, match="could NOT be put back.*check .* by hand"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})


def test_a_failed_switch_with_no_previous_leaves_none(bindir, world, tmp_path):
    w = world()
    _launcher_with(bindir, "b", "NEVERUP")
    with pytest.raises(UM.SwitchError):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert not (bindir / "start-llama-server.sh.previous").exists()


def test_a_write_that_fails_half_way_is_rolled_back(bindir, world, tmp_path, monkeypatch):
    """Second review: the two launcher writes were outside the rollback."""
    w = world()
    UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)        # live = b, previous = a
    before = {p.name: p.read_bytes() for p in bindir.iterdir()}
    real, n = UM._put, []

    def second_write_fails(data, mode, dst):
        n.append(dst.name)
        if len(n) == 2:
            raise OSError(28, "No space left on device")
        return real(data, mode, dst)
    monkeypatch.setattr(UM, "_put", second_write_fails)
    with pytest.raises(OSError):
        UM.switch(bindir, "previous", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert {p.name: p.read_bytes() for p in bindir.iterdir()} == before and _restarts(tmp_path) == 1


def test_the_live_launcher_that_is_the_one_asked_for_and_does_not_come_up_is_not_restarted_twice(bindir, world, tmp_path):
    w = world()
    (bindir / "start-llama-server.sh").write_text((bindir / "start-llama-server.sh").read_text().replace("exec true", "# NEVERUP\nexec true"))
    (bindir / "start-llama-server.sh.a").write_text((bindir / "start-llama-server.sh").read_text())
    (tmp_path / "running.txt").write_text("")
    with pytest.raises(UM.SwitchError, match="did not come up"):
        UM.switch(bindir, "a", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert _restarts(tmp_path) == 1                                          # there is no other launcher to go back to


def test_a_restart_command_that_fails_is_never_run_a_second_time(bindir, world, tmp_path):
    """Review: after "Still alive after 120s" the rollback called the restart
    script again — the second signal that script exists to prevent."""
    w = world()
    _launcher_with(bindir, "b", "REFUSE")
    with pytest.raises(UM.SwitchError, match=r"failed \(3\): Still alive.*NOT signalled again"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.4})
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert _restarts(tmp_path) == 1 and _no_leftovers(bindir)


def test_a_restart_command_that_gave_up_on_a_server_that_did_come_up_is_a_switch(bindir, world, tmp_path):
    """The restart script stops waiting at 300 s; a slow loader finishes after it."""
    w = world()
    _launcher_with(bindir, "b", "LATE")
    got = UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **FAST)
    assert got["server_model"] == "Model-B.gguf" and _restarts(tmp_path) == 1
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.b").read_text()


def test_a_server_that_comes_up_after_the_restart_command_gave_up_is_waited_for(bindir, world, tmp_path):
    """Third review: the stand-in was up before the command returned, so the watch never ran."""
    w = world()
    _launcher_with(bindir, "b", "REFUSE")                                    # exits 3; the server still runs model A
    target = str(tmp_path / "models" / "Model-B.gguf")
    looks = []

    def sleeping(seconds):
        looks.append(1)
        if len(looks) == 3:                                                  # the slow loader finishes on the third look
            (tmp_path / "running.txt").write_text(target)
    got = UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, sleep=sleeping, **FAST)
    assert got["server_model"] == "Model-B.gguf" and _restarts(tmp_path) == 1 and len(looks) >= 3


def test_a_restart_command_that_cannot_be_run_leaves_the_launcher_as_it_was(bindir, world, tmp_path):
    """It never ran, so nothing was signalled: put back at once — and never "it came up on
    its own" because another launcher for the SAME model file was being switched to."""
    w = world()
    (bindir / "restart-llama-server.sh").chmod(0o644)
    slept = []
    with pytest.raises(UM.SwitchError, match="could not be run.*Nothing was signalled"):
        UM.switch(bindir, "dense.daisy_rose-2", server=w.url, agent=w.url, say=lambda s: None, sleep=slept.append, **FAST)
    assert slept == []                                                        # no watching
    with pytest.raises(UM.SwitchError, match="could not be run"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert _no_leftovers(bindir)


def test_an_interrupt_puts_the_launcher_back_and_signals_nothing(bindir, world, tmp_path):
    w = world()
    _launcher_with(bindir, "b", "NEVERUP")

    def interrupted(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, sleep=interrupted,
                  **{k: v for k, v in FAST.items()})
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert _restarts(tmp_path) == 1 and _no_leftovers(bindir)


def test_with_no_live_launcher_a_failed_switch_leaves_none(bindir, world, tmp_path):
    w = world()
    (bindir / "start-llama-server.sh").unlink()
    (tmp_path / "running.txt").write_text("")
    _launcher_with(bindir, "b", "NEVERUP")
    with pytest.raises(UM.SwitchError, match="did not come up"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "load_wait": 0.3})
    assert not (bindir / "start-llama-server.sh").exists() and not (bindir / "start-llama-server.sh.previous").exists()
    assert _restarts(tmp_path) == 1


def test_an_agent_that_does_not_follow_is_said_so(bindir, world, tmp_path):
    w = world(follows=False)
    with pytest.raises(UM.SwitchError, match="has not followed"):
        UM.switch(bindir, "b", server=w.url, agent=w.url, say=lambda s: None, **{**FAST, "follow_wait": 0.4})
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf")  # the server DID switch: not rolled back


def test_an_agent_that_is_down_does_not_block_the_switch(bindir, world, tmp_path):
    w = world()
    with pytest.raises(UM.SwitchError, match="has not followed"):
        UM.switch(bindir, "b", server=w.url, agent="http://127.0.0.1:9", say=lambda s: None, **{**FAST, "follow_wait": 0.3})
    assert (tmp_path / "running.txt").read_text().endswith("Model-B.gguf")


def test_the_command_line(bindir, world, capsys, monkeypatch, tmp_path):
    w = world()
    monkeypatch.setattr(UM, "profile_for", lambda path: "p")
    base = ["--bin", str(bindir), "--server", w.url, "--agent", w.url]
    assert UM.main(base + ["list"]) == 0
    out = capsys.readouterr().out
    assert "* a " in out and "  b " in out and "MODEL FILE MISSING" in out and "Model-B.gguf" in out
    assert UM.main(base + ["status"]) == 0
    out = capsys.readouterr().out
    assert "server: Model-A.gguf" in out and "launcher: a" in out and "agent: profile p" in out
    assert UM.main(base[:4] + ["--agent", "http://127.0.0.1:9", "status"]) == 0
    assert "agent: down" in capsys.readouterr().out
    assert UM.main(base + ["save", "kept"]) == 0 and (bindir / "start-llama-server.sh.kept").is_file()
    assert UM.main(base + ["save", "kept"]) == 1 and "exists" in capsys.readouterr().err
    assert UM.main(base + ["save", "previous"]) == 1 and not (bindir / "start-llama-server.sh.previous").exists()
    capsys.readouterr()
    assert UM.main(base + ["switch", "zzz"]) == 1 and "no launcher" in capsys.readouterr().err
    assert UM.main(base + ["switch", "b", "--dry-run"]) == 0
    assert (tmp_path / "running.txt").read_text().endswith("Model-A.gguf")


def test_an_interrupt_or_a_kill_ends_with_its_own_exit_code_and_the_launcher_back(bindir, world, tmp_path, monkeypatch, capsys):
    import signal
    w = world()
    base = ["--bin", str(bindir), "--server", w.url, "--agent", w.url]
    monkeypatch.setattr(UM, "profile_for", lambda path: "p")
    before = signal.getsignal(signal.SIGTERM)
    real = UM.subprocess.run

    def killed_during_restart(*a, **k):
        real(*a, **k)
        os.kill(os.getpid(), signal.SIGTERM)                                 # a kill / a closed terminal, mid-switch
        time.sleep(0.5)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(UM.subprocess, "run", killed_during_restart)
    assert UM.main(base + ["switch", "b"]) == 130 and "interrupted" in capsys.readouterr().err
    assert (bindir / "start-llama-server.sh").read_text() == (bindir / "start-llama-server.sh.a").read_text()
    assert not (bindir / "start-llama-server.sh.previous").exists()
    assert signal.getsignal(signal.SIGTERM) is before                        # the handler is the caller's again


def test_describe():
    assert UM.describe({}) == "the agent does not answer"
    assert "exact" in UM.describe({"name": "p", "why": "w", "tokens": {"exact": True}})
    assert "2.6% off" in UM.describe({"name": "p", "tokens": {"local_tokens": 413, "drift": -0.0259}})
    assert "estimated (2.7" in UM.describe({"name": "p", "tokens": {"chars_per_token": 2.7}})
    assert "not checked" in UM.describe({"name": "p"})


# --- the scorecard -------------------------------------------------------------------------

@pytest.mark.parametrize("text, ok", [
    ("101\n103\n107", True), (" 101 \n\n103\n107\n", True),
    ("101\n103", False), ("101\n103\n105", False), ("97\n101\n103", False),
    ("101, 103, 107", False), ("- 101\n- 103\n- 107", False), ("101\n103\n107\n109", False), ("", False),
])
def test_the_exact_format_check(text, ok):
    assert SC.check_three_primes(text) is ok


@pytest.mark.parametrize("text, ok", [
    ('{"city": "Tokyo", "population_millions": 14.0}', True),
    (' {"city":"tokyo","population_millions":37} ', True),
    ('```json\n{"city": "Tokyo", "population_millions": 14}\n```', False),
    ('Here: {"city": "Tokyo", "population_millions": 14}', False),
    ('{"city": "Kyoto", "population_millions": 1.4}', False),
    ('{"city": "Tokyo", "population_millions": "14"}', False),
    ('{"city": "Tokyo", "population_millions": true}', False),
    ('["Tokyo", 14]', False), ("", False),
])
def test_the_json_only_check(text, ok):
    assert SC.check_json_city(text) is ok


@pytest.mark.parametrize("text, n, ok", [
    ("8", 8, True), ("Total: 8", 8, True), ("it is 8.", 8, True), ("8.0", 8, True), ("5 + 3 = 8 items", 8, True),
    ("18", 8, False), ("80", 8, False), ("1.8", 8, False), ("8.5", 8, False), ("", 8, False),
    ("The bill was 120 euros.", 120, True), ("€120.", 120, True), ("**120**", 120, True),
    ("120%", 120, False), ("1,120", 120, False), ("120,000", 120, False), ("1200", 120, False),
    ("40% of 100 is 40", 100, True), ("that is 100% certain", 100, False),
    ("Title: Example Domain, value: 42", 42, True), ("14 42", 42, True), ("142", 42, False),
])
def test_a_number_counts_only_as_a_number_of_its_own(text, n, ok):
    """Review: "8" in the line passed 18 and 80; "100" passed "100%"."""
    assert SC.has_number(text, n) is ok


def test_the_edit_check_wants_one_line_changed_and_forgives_only_the_last_newline():
    after = SC.X3_BEFORE.replace("port = 8080", "port = 9090")
    assert SC.check_one_line_changed(after) and SC.check_one_line_changed(after.rstrip("\n")) and SC.check_one_line_changed(after + "\n")
    assert not SC.check_one_line_changed(SC.X3_BEFORE)                                   # not changed at all
    assert not SC.check_one_line_changed(after.replace("debug = false", "debug = true"))
    assert not SC.check_one_line_changed(after.replace("\n\n[client]", "\n[client]"))
    assert not SC.check_one_line_changed("port = 9090\n") and not SC.check_one_line_changed("")


@pytest.mark.parametrize("rc, out, exactly, ok", [
    (0, "3 passed in 0.01s", None, True), (0, "3 passed in 0.01s", 3, True), (0, "4 passed in 0.01s", 3, False),
    (1, "2 passed, 1 failed in 0.01s", None, False), (0, "2 passed, 1 error in 0.01s", None, False),
    (1, "3 passed in 0.01s", None, False), (0, "no tests ran in 0.00s", None, False), (0, "", None, False),
    (2, "ERROR: file or directory not found", None, False),
])
def test_a_pytest_run_passed_only_by_exit_code_and_count(rc, out, exactly, ok):
    """Review: the exit code read was `tail`'s, and "2 passed, 1 error" passed."""
    assert SC.pytest_passed(rc, out, exactly) is ok


def test_a_reply_is_read_from_both_kinds_of_answer():
    plain = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": {"completion_tokens": 3}}).encode()
    assert SC.parse_reply(plain, False, 1.5) == {"content": "hi", "usage": {"completion_tokens": 3}, "s": 1.5, "error": None}
    assert SC.parse_reply(json.dumps({"error": {"message": "x"}}).encode(), False, 0.1)["error"] == {"message": "x"}
    frames = [{"choices": [{"delta": {"reasoning_content": "hmm "}}]}, {"choices": [{"delta": {"content": "he"}}]},
              {"choices": [{"delta": {"content": "llo"}}], "usage": {"completion_tokens": 2}}, {"choices": [{"delta": {}}]}]
    raw = ("".join(f"data: {json.dumps(f)}\n\n" for f in frames) + ": keepalive\n\ndata: not json\n\ndata: [DONE]\n\n").encode()
    got = SC.parse_reply(raw, True, 2.0)
    assert got["content"] == "hello" and got["reasoning_chars"] == 4 and got["frames"] == 4 and got["usage"] == {"completion_tokens": 2}


def test_the_picture_is_a_real_png_of_the_right_size():
    url = SC.png_with_digits("7391", scale=10)
    assert url.startswith("data:image/png;base64,")
    png = base64.b64decode(url.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    w, h, depth, colour = struct.unpack(">IIBB", png[16:26])
    assert (w, h, depth, colour) == ((4 * 6 + 1) * 10, 90, 8, 0)
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    rows = zlib.decompress(idat)
    assert len(rows) == h * (w + 1) and rows.count(b"\x00") > h and b"\xff" in rows       # ink and paper
    assert SC.png_with_digits("7391") != SC.png_with_digits("7397")


LOG = """2026-01-01 10:00:00 - S - INFO - [probe-aa +0.1s] llm request — Turn 1 | Temp 1.00
2026-01-01 10:00:01 - S - INFO - [probe-aa +1.0s] tool call — execute · command=date
2026-01-01 10:00:02 - S - INFO - [probe-aa +2.0s] agent parser — Extracting XML tool call...
2026-01-01 10:00:02 - S - INFO - [probe-aa +2.0s] tool call — browser · operation=interact
2026-01-01 10:00:03 - S - WARNING - [probe-aa +3.0s] tool warning — browser -> STATUS: ERROR
2026-01-01 10:00:04 - S - ERROR - [probe-aa +4.0s] system fault — HTTP 503
2026-01-01 10:00:05 - S - INFO - [abc123 +0.1s] llm request — Turn 1 | Temp 1.00
2026-01-01 10:00:06 - S - WARNING - [abc123 +1.0s] tool warning — someone else's turn
"""


def test_the_log_is_counted_for_probe_turns_only():
    assert SC.log_counts(LOG) == {"model_calls": 1, "tool_calls": 2, "text_format_tool_calls": 1, "warnings": 1, "errors": 1}
    assert SC.log_counts("") == {"model_calls": 0, "tool_calls": 0, "text_format_tool_calls": 0, "warnings": 0, "errors": 0}


def test_a_tool_task_answered_from_memory_is_not_a_pass():
    """Second review: the date, the page, the search and the browser task passed on
    the reply alone — with the browser down, a model that remembers example.com scored 3/3."""
    assert SC.used_a_tool({"usage": {"ghost_llm_calls": 2}}) and SC.used_a_tool({"usage": {"ghost_llm_calls": 7}})
    for usage in ({"ghost_llm_calls": 1}, {"ghost_llm_calls": 0}, {}, None, {"ghost_llm_calls": "2"}):
        assert not SC.used_a_tool({"usage": usage})
    by_id = {s[0]: s[3] for s in SC.SCENARIOS}
    replies = {"x6": time.strftime("%Y-%m-%d"), "w1": "Its title is Example Domain.", "w2": "2015, per rust-lang.org",
               "w3": "Example Domain and 42"}

    class E:
        def __init__(self, calls):
            self.calls = calls

        def chat(self, *a, **k):
            return {"content": replies[self.sid], "usage": {"ghost_llm_calls": self.calls}}
    for sid in replies:
        for calls, want in ((1, False), (3, True)):
            e = E(calls)
            e.sid = sid
            assert bool(by_id[sid](e)[1]) is want, (sid, calls)


class _Scripted:
    """An Env whose agent says what a test tells it to, and whose sandbox is a dict."""

    def __init__(self, content="", calls=3, files=None, box=None):
        self.content, self.calls, self.files, self.boxes, self.sent = content, calls, dict(files or {}), list(box or []), []
        self.ran = []

    def chat(self, messages, stream=False, arm=None):
        self.sent.append({"messages": messages, "stream": stream, "arm": arm})
        return {"content": self.content, "usage": {"ghost_llm_calls": self.calls}, "frames": 9, "s": 1.0, "error": None}

    def read(self, *p):
        return self.files.get("/".join(p))

    def seed(self, rel, text):
        self.files[rel] = text

    def box(self, cmd, timeout=180):
        self.ran.append(cmd)
        return self.boxes.pop(0) if self.boxes else (0, "")


def _run(sid, **kw):
    e = _Scripted(**kw)
    fn = {s[0]: s[3] for s in SC.SCENARIOS}[sid]
    return bool(fn(e)[1]), e


def test_the_scenarios_ask_what_they_say_and_check_what_they_asked():
    today = time.strftime("%Y-%m-%d")
    assert _run("x6", content=f"It is {today}.")[0] and not _run("x6", content="It is 1999-01-01.")[0]
    assert not _run("x6", content=f"It is {today}.", calls=1)[0]             # from memory
    assert _run("e1", content="120 euros")[1].sent[0]["arm"] == "effort_low"
    assert _run("e2", content="120 euros")[1].sent[0]["arm"] == "effort_xhigh"
    assert _run("e2", content="120 euros")[0] and not _run("e2", content="100 euros")[0]
    assert _run("t5", content="Shorter wavelengths scatter more in the air, so blue light reaches the eye from every direction.")[1].sent[0]["stream"] is True
    assert _run("x1", content="digest: " + __import__("hashlib").sha256(b"ghost-scorecard").hexdigest().upper())[0]
    assert not _run("x1", content="digest: " + "0" * 64)[0]
    # c4: a short, correct program passes; the checks are the ones the task states
    prog = "import argparse\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"
    ok = dict(content="total prints 8", files={"c4/inventory.py": prog}, box=[(0, ""), (0, "added\nadded\n8")])
    assert _run("c4", **ok)[0]
    assert not _run("c4", **{**ok, "box": [(0, ""), (0, "added\nadded\n18")]})[0]
    assert not _run("c4", **{**ok, "box": [(0, ""), (1, "8")]})[0]
    assert not _run("c4", **{**ok, "files": {"c4/inventory.py": prog.replace("argparse", "sys")}})[0]
    assert not _run("c4", **{**ok, "files": {"c4/inventory.py": prog.replace("__main__", "main")}})[0]
    assert not _run("c4", **{**ok, "content": "total prints 2"})[0]
    # c3: the given test file must be untouched and exactly its three tests pass
    assert _run("c3", content="fixed", box=[(0, "3 passed in 0.01s")])[0]

    class Tampering(_Scripted):                                              # the "fix" that edits the test instead
        def chat(self, *a, **k):
            self.files["c3/test_stats.py"] += "\n# weakened\n"
            return super().chat(*a, **k)
    e = Tampering(content="fixed", box=[(0, "3 passed in 0.01s")])
    assert not {s[0]: s[3] for s in SC.SCENARIOS}["c3"](e)[1]
    # c2: its tests pass AND the function itself is right (checked with our own cases)
    assert _run("c2", content="done", box=[(0, "5 passed in 0.02s"), (0, "OK")])[0]
    assert not _run("c2", content="done", box=[(0, "5 passed in 0.02s"), (1, "AssertionError")])[0]
    assert not _run("c2", content="done", box=[(1, "1 failed, 4 passed"), (0, "OK")])[0]
    assert not _run("c3", content="fixed", box=[(1, "2 passed, 1 failed")])[0]
    assert not _run("c3", content="fixed", box=[(0, "4 passed in 0.01s")])[0]
    # x3: seeded, then judged on the file
    assert not _run("x3", content="done")[0]                                 # the file was left as seeded
    # nothing a check runs in the container writes outside the run's folder
    for sid in ("c1", "c2", "c3", "c4"):
        e = _run(sid, content="x")[1]
        assert e.ran and all("/tmp" not in cmd and SC.ROOT in cmd for cmd in e.ran), sid


@pytest.mark.parametrize("leak", ["</think>", "<think>", "<tool_call>", "</tool_call>", "<|end_message|>", "/no_think",
                                  "<function=execute>", "content_invoke_tool"])
def test_every_kind_of_leak_fails_a_right_answer(leak):
    ok = {"content": f"42 {leak}", "usage": {}, "s": 1.0}
    assert SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e: (ok, True))["ok"] is False


def test_a_log_that_was_rotated_is_not_read_as_a_quiet_one(tmp_path):
    log = tmp_path / "agent.log"
    log.write_text("x" * 100 + LOG)
    assert SC.read_log_since(log, 100) == LOG and SC.read_log_since(log, 0).startswith("x")
    assert SC.read_log_since(log, log.stat().st_size) == ""                 # nothing written since: quiet, not unknown
    log.write_text("short")                                                # rotated: smaller than where the run began
    assert SC.read_log_since(log, 100) is None and SC.read_log_since(tmp_path / "none.log", 0) is None


class _FakeEnv:
    def __init__(self, idle=True):
        self._idle = idle

    def idle(self):
        return self._idle


def test_one_scenario_passes_only_with_a_clean_reply():
    ok = {"content": "42", "usage": {"ghost_llm_calls": 2, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 100}}, "s": 3.2}
    r = SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e: (ok, True))
    assert r["ok"] and (r["calls"], r["out_tokens"], r["cached_tokens"], r["s"], r["reply"]) == (2, 9, 100, 3.2, "42")
    assert SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e: (ok, False))["ok"] is False
    for leak in ("42</think>", "<tool_call>\n<function=execute>", "ok <|end_message|>", "42 /no_think"):
        r = SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e, t=leak: ({**ok, "content": t}, True))
        assert r["ok"] is False and r["check"] is True and r["leaks"], leak            # right answer, leaked markup: a fail
    assert SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e: ({**ok, "error": {"m": 1}}, True))["ok"] is False
    boom = SC.run_one(_FakeEnv(), "t9", "talk", "x", lambda e: 1 / 0)
    assert boom["ok"] is False and "ZeroDivisionError" in boom["note"]
    assert SC.run_one(_FakeEnv(idle=False), "t9", "talk", "x", lambda e: (ok, True))["note"] == SC.NEVER_IDLE


def _results():
    return [{"id": "t1", "area": "talk", "what": "a fact", "ok": True, "s": 10.0},
            {"id": "x1", "area": "tools", "what": "run", "ok": False, "s": 30.5},
            {"id": "x2", "area": "tools", "what": "write", "ok": True, "s": 20.0}]


def test_a_card_and_the_table_of_cards(tmp_path):
    model = {"server_model": "Model-A v2 (Q4).gguf", "name": "pa", "tokens": {"exact": True}}
    card = SC.summarise(_results(), model, SC.log_counts(LOG), "20260101-100000")
    assert (card["passed"], card["of"], card["seconds"]) == (2, 3, 60) and card["areas"] == {"talk": "1/1", "tools": "1/2"}
    assert card["profile"] == "pa" and card["tokens_exact"] is True and card["text_format_tool_calls"] == 1
    assert SC.card_name(card["model"], card["started"]) == "Model-A_v2_Q4--20260101-100000.json"
    assert SC.card_name("../../etc/x.gguf", "t") == ".._.._etc_x--t.json" and "/" not in SC.card_name("a/b", "t")
    d = tmp_path / "cards"
    assert SC.newest_cards(d) == [] and "no cards yet" in SC.table([])
    d.mkdir()
    older = {**card, "started": "20251231-090000", "passed": 1}
    other = SC.summarise(_results()[:1] + [{"id": "z9", "area": "talk", "what": "new", "ok": True, "s": 1.0}],
                         {"server_model": "Model-B.gguf", "name": "pb"}, {}, "20260102-080000")
    for c in (older, card, other):
        (d / SC.card_name(c["model"], c["started"])).write_text(json.dumps(c))
    (d / "junk.json").write_text("{not json")
    (d / "list.json").write_text("[1]")
    (d / "damaged.json").write_text(json.dumps({"model": "Model-C.gguf", "started": "20270101-000000"}))      # no results
    (d / "damaged2.json").write_text(json.dumps({"model": "Model-A v2 (Q4).gguf", "started": 7, "results": []}))
    newest = SC.newest_cards(d)
    assert [c["model"] for c in newest] == ["Model-A v2 (Q4).gguf", "Model-B.gguf"] and newest[0]["passed"] == 2
    t = SC.table(newest)
    lines = t.splitlines()
    assert "Model-A v2 (Q4)" in lines[0] and "Model-B" in lines[0]
    row = {ln.split("  ")[0]: ln for ln in lines}
    assert "2/3" in row["passed"] and "2/2" in row["passed"]
    assert "1/2" in row["tool calls as text"] and "?/?" in row["tool calls as text"]
    assert "FAIL 30.5s" in row["x1 run"] and row["x1 run"].rstrip().endswith("-")       # the other model never ran it
    assert "pass 1.0s" in row["z9 new"] and "yes" in row["exact token counts"] and "no" in row["exact token counts"]
    # a card someone edited by hand prints, it does not crash the table
    assert "?/?" in SC.table([{"model": "M.gguf", "results": [{"id": "t1"}, "junk", {"what": "no id"}]}])


def test_the_scenarios_are_well_formed():
    ids = [s[0] for s in SC.SCENARIOS]
    assert len(ids) == len(set(ids)) == 23 and {s[1] for s in SC.SCENARIOS} == {"talk", "tools", "coding", "outside", "effort"}
    assert all(callable(s[3]) and s[2] for s in SC.SCENARIOS)
    assert SC.main(["list"]) == 0


def test_everything_a_run_creates_is_under_one_folder_and_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    env = SC.Env()
    assert env.sandbox == tmp_path / "sandbox" and env.cards == tmp_path / "system" / "eval" / "scorecards"
    env.seed("x3/config.ini", "a=1\n")
    assert env.read("x3", "config.ini") == "a=1\n" and env.read("x3", "missing") is None
    assert (tmp_path / "sandbox" / SC.ROOT / "x3" / "config.ini").is_file()
    (tmp_path / "sandbox" / "owner-file.txt").write_text("keep")
    env.clean()
    assert not (tmp_path / "sandbox" / SC.ROOT).exists() and (tmp_path / "sandbox" / "owner-file.txt").read_text() == "keep"


# --- a run, against a stand-in agent ----------------------------------------------------------

class _Agent:
    """Answers /api/health, /api/turns and /v1/chat/completions; records what it was sent."""

    def __init__(self, key="test-key", model="Model-A.gguf"):
        self.key, self.model, self.busy, self.running, self.queued = key, model, 0, None, 0
        self.chats, self.unlabelled, self.reply, self.after_chat = [], [], "Canberra", None
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def _send(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def _gate(self):
                if self.headers.get("X-Ghost-Origin") != "probe":
                    outer.unlabelled.append(self.path)
                if self.headers.get("X-Ghost-Key") != outer.key:
                    self._send(403, {"detail": "bad key"})
                    return False
                return True

            def do_GET(self):
                if not self._gate():
                    return
                if self.path == "/api/health":
                    self._send(200, {"foreground_requests": outer.busy,
                                     "model_profile": {"name": "pa", "server_model": outer.model, "tokens": {"exact": True}}})
                elif self.path == "/api/turns":
                    self._send(200, {"turns": [], "running": outer.running, "queued": outer.queued})
                else:
                    self._send(404, {})

            def do_POST(self):
                if not self._gate():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.chats.append({"body": body, "arm": self.headers.get("X-Ghost-Arm")})
                if "model" in body:                                         # as the real agent: any other name is refused
                    self._send(404, {"error": {"message": "model not found", "type": "ModelNotFound"}})
                    return
                if outer.after_chat:
                    outer.after_chat()
                self._send(200, {"choices": [{"message": {"content": outer.reply}}],
                                 "usage": {"completion_tokens": 3, "ghost_llm_calls": 1}})

            def log_message(self, *a):
                pass
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def agent(tmp_path, monkeypatch):
    a = _Agent()
    monkeypatch.setenv("GHOST_URL", a.url)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GHOST_SANDBOX_CONTAINER", "a-container")
    monkeypatch.setattr(SC.Env, "idle_wait", 0.4)
    monkeypatch.setattr(SC.Env, "idle_step", 0.05)
    (tmp_path / "home" / "system").mkdir(parents=True)
    (tmp_path / "home" / "sandbox").mkdir()
    yield a
    a.close()


def _cards(tmp_path):
    d = tmp_path / "home" / "system" / "eval" / "scorecards"
    return sorted(d.glob("*.json")) if d.is_dir() else []


def test_a_partial_run_asks_as_a_probe_names_no_model_and_writes_no_card(agent, tmp_path, capsys):
    """Review (MAJOR): every request carried `"model": "default"`, which the agent
    refuses with 404 — the card would have said 0/23 for every model."""
    assert SC.main(["run", "t1"]) == 0                                       # passed: exit 0
    assert len(agent.chats) == 1 and "model" not in agent.chats[0]["body"] and agent.chats[0]["body"]["stream"] is False
    assert agent.unlabelled == [] and _cards(tmp_path) == []
    out = capsys.readouterr().out
    assert "t1  PASS" in out and "1/1 passed" in out and "partial run" in out
    agent.reply = "Sydney"
    assert SC.main(["run", "t1"]) == 1                                       # failed: exit 1
    assert SC.main(["run", "nope"]) == 2 and "unknown scenario" in capsys.readouterr().err
    assert SC.main(["run", "e1"]) == 1 and agent.chats[-1]["arm"] == "effort_low"


def test_a_full_run_writes_one_card_and_leaves_nothing_in_the_sandbox(agent, tmp_path, monkeypatch, capsys):
    stale = tmp_path / "home" / "sandbox" / SC.ROOT / "x2" / "notes.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("alpha\nbeta\ngamma\n")                                 # a killed run's file: must not pass for this model
    seen = {}

    def writes(e):
        seen["stale"] = e.read("x2", "notes.txt")
        e.seed("mine/out.txt", "x")
        return e.chat(SC.U("hi")), True

    def fails(e):
        return e.chat(SC.U("hi"), arm="effort_low"), False
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "writes a file", writes), ("a2", "tools", "fails", fails)])
    log = tmp_path / "home" / "system" / "ghost-agent.log"
    log.write_text("older line [probe-zz] tool call \u2014 x\n")
    agent.after_chat = lambda: log.open("a").write(LOG)
    assert SC.main(["run"]) == 1                                             # one failed
    cards = _cards(tmp_path)
    assert len(cards) == 1 and cards[0].name.startswith("Model-A--")
    card = json.loads(cards[0].read_text())
    assert (card["passed"], card["of"], card["model"], card["profile"]) == (1, 2, "Model-A.gguf", "pa")
    assert card["tool_calls"] == 4 and card["text_format_tool_calls"] == 2  # two chats' worth, not the older line
    assert [r["ok"] for r in card["results"]] == [True, False]
    assert seen["stale"] is None and not (tmp_path / "home" / "sandbox" / SC.ROOT).exists()
    assert agent.unlabelled == [] and all("model" not in c["body"] for c in agent.chats)
    capsys.readouterr()
    assert SC.main(["show"]) == 0 and "1/2" in capsys.readouterr().out
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "writes a file", writes)])
    assert SC.main(["run"]) == 0 and len(_cards(tmp_path)) == 2              # all passed: exit 0


def test_a_rotated_log_leaves_the_counts_out_of_the_card(agent, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.chat(SC.U("hi")), True))])
    log = tmp_path / "home" / "system" / "ghost-agent.log"
    log.write_text("x" * 500)
    agent.after_chat = lambda: log.write_text("rotated")
    assert SC.main(["run"]) == 0
    card = json.loads(_cards(tmp_path)[0].read_text())
    assert "tool_calls" not in card and "text_format_tool_calls" not in card and card["passed"] == 1
    assert "tool calls as text ?/?" in capsys.readouterr().out


def test_naming_every_scenario_is_still_a_chosen_run(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.chat(SC.U("hi")), True))])
    assert SC.main(["run", "a1"]) == 0 and _cards(tmp_path) == []


def test_a_run_folder_that_is_a_link_is_removed_not_followed(agent, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    root = tmp_path / "home" / "sandbox" / SC.ROOT
    root.parent.mkdir(parents=True, exist_ok=True)
    root.symlink_to(elsewhere)
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.seed("f.txt", "x"), e.chat(SC.U("hi")))[1:] + (True,))])
    assert SC.main(["run"]) == 0
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"] and not root.exists()


def test_a_folder_that_cannot_be_removed_is_said_and_the_card_still_written(agent, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.chat(SC.U("hi")), True))])
    real, calls = SC.shutil.rmtree, []

    def rmtree(path, *a, **k):
        calls.append(k)
        if len(calls) == 1:                                                  # the one at the end (at the start there is no folder)
            raise PermissionError(13, "Permission denied")
        return real(path, *a, **k)
    monkeypatch.setattr(SC.shutil, "rmtree", rmtree)
    assert SC.main(["run"]) == 0 and len(_cards(tmp_path)) == 1
    assert "could not remove" in capsys.readouterr().err and all(not k.get("ignore_errors") for k in calls)
    monkeypatch.setattr(SC.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "Permission denied")))
    with pytest.raises(PermissionError):                                     # at the START it stops: nothing was sent
        SC.main(["run"])
    assert len(agent.chats) == 1


def test_a_run_whose_last_task_was_never_sent_writes_no_card(agent, tmp_path, monkeypatch, capsys):
    """Third review: busy before the LAST scenario only — every scenario had a row, so a
    card was written with a FAIL for a task the model never saw."""
    def then_busy(e):
        r = e.chat(SC.U("hi"))
        agent.busy = 1
        return r, True
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", then_busy), ("a2", "talk", "y", lambda e: (e.chat(SC.U("hi")), True))])
    assert SC.main(["run"]) == 1 and _cards(tmp_path) == [] and len(agent.chats) == 1
    assert "stopped before its end" in capsys.readouterr().out


def test_one_slow_answer_at_the_end_does_not_cost_the_card(agent, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.chat(SC.U("hi")), True))])
    real, fail = SC.Env.health, []

    def health(self):
        if agent.chats and len(fail) < 2:                                    # the first two looks after the run fail
            fail.append(1)
            return {}
        return real(self)
    monkeypatch.setattr(SC.Env, "health", health)
    assert SC.main(["run"]) == 0 and len(_cards(tmp_path)) == 1 and len(fail) == 2
    monkeypatch.setattr(SC.Env, "health", lambda self: {} if agent.chats[1:] else real(self))
    assert SC.main(["run"]) == 1 and len(_cards(tmp_path)) == 1             # it never answers again: said so, no card
    assert "stopped answering" in capsys.readouterr().err


def test_a_run_needs_the_agents_sandbox(agent, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path / "not-the-agents-home"))
    assert SC.main(["run", "t1"]) == 2 and "no sandbox at" in capsys.readouterr().err and agent.chats == []


def test_a_model_that_changed_under_the_run_gets_no_card(agent, tmp_path, monkeypatch, capsys):
    def ok(e):
        return e.chat(SC.U("hi")), True
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", ok)])
    agent.after_chat = lambda: setattr(agent, "model", "Model-B.gguf")
    assert SC.main(["run"]) == 1 and _cards(tmp_path) == []
    assert "the model changed during the run" in capsys.readouterr().err


@pytest.mark.parametrize("what", ["a request in flight", "a turn running", "a turn queued"])
def test_nothing_is_sent_while_the_agent_is_busy(agent, tmp_path, monkeypatch, capsys, what):
    if what == "a request in flight":
        agent.busy = 1
    elif what == "a turn running":
        agent.running = {"id": "abc"}
    else:
        agent.queued = 2
    assert SC.Env().idle() is False
    monkeypatch.setattr(SC, "SCENARIOS", [("a1", "talk", "x", lambda e: (e.chat(SC.U("hi")), True)),
                                          ("a2", "talk", "y", lambda e: (e.chat(SC.U("hi")), True))])
    assert SC.main(["run"]) == 1 and agent.chats == [] and _cards(tmp_path) == []
    assert "stopping here" in capsys.readouterr().err                        # not 2 × the whole wait
    agent.busy, agent.running, agent.queued = 0, None, 0
    assert SC.Env().idle() is True


def test_a_run_needs_an_agent_and_for_code_a_container(agent, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GHOST_SANDBOX_CONTAINER", "")
    monkeypatch.setattr(SC.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="other-box\n", returncode=0, stderr=""))
    assert SC.main(["run", "c1"]) == 2 and "no running sandbox container" in capsys.readouterr().err and agent.chats == []
    monkeypatch.setattr(SC.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="zeta\nghost-agent-sandbox-b\nghost-agent-sandbox-a\n", returncode=0, stderr=""))
    assert SC.Env().container() == "ghost-agent-sandbox-a"
    monkeypatch.setenv("GHOST_URL", "http://127.0.0.1:9")
    assert SC.main(["run", "t1"]) == 2 and "does not answer" in capsys.readouterr().err


def test_the_scorecard_finds_the_key_the_agent_takes(agent, tmp_path, monkeypatch):
    agent.key = "from-the-file"
    keyfile = tmp_path / "norepo" / ".ghost_api_key"
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("from-the-file\n")
    monkeypatch.setenv("GHOST_API_KEY", "stale")
    env = SC.Env()
    assert env.keys() == ["stale", "from-the-file"] and env.key() == "from-the-file"
    assert env.health()["model_profile"]["server_model"] == "Model-A.gguf"
    assert SC.main(["run", "t1"]) == 0
