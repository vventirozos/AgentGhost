#!/usr/bin/env python3
"""§4VP (phase 5): one command to put another model behind the agent.

A model server launcher is a file ``start-llama-server.sh.<name>`` beside the
live ``start-llama-server.sh`` (the operator's ``bin`` directory). Switching
used to be: copy the file by hand, restart llama, wait, hope the agent follows.
This does it in one step and says what the agent ended up with:

    use_model.py list                  the launchers, the model each starts, the profile it would get
    use_model.py status                what runs now: server model, agent profile, tokenizer, budget
    use_model.py save NAME             keep the live launcher as start-llama-server.sh.NAME
    use_model.py switch NAME           switch to it (waits for the agent to be idle first)
    use_model.py switch NAME --dry-run say what would be done, do nothing

``switch`` refuses when the launcher's model file is missing, while the
agent has a request in flight, or when it cannot TELL whether it has (the
agent answers but not with its health: a refused key, a timeout). ``--force``
skips that wait. It keeps the launcher it replaces as
``start-llama-server.sh.previous`` (one slot; ``switch previous`` swaps the
two), restarts llama with the directory's own ``restart-llama-server.sh``
(ONE signal, then wait — see that script), waits until the server names the
new model file, and then until the agent's ``/api/health`` shows it followed.

When it does not take, the launcher that was live is put back — whatever
went wrong, an interrupt included. The server is restarted on it only when
the restart command itself had succeeded (the server came up, on something
else): after a restart command that FAILED the server may still be shutting
down, and a second signal is what that script exists to prevent.

Stdlib only. The ``bin`` directory is ``--bin`` / ``$GHOST_BIN`` (default:
``bin`` beside the repository); the agent's key is the file
``$GHOST_KEY_FILE``, else ``$GHOST_API_KEY``, else ``.ghost_api_key`` beside
the repository — the next one is tried when the agent refuses one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
LIVE = "start-llama-server.sh"
PREVIOUS = "previous"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MODEL_RE = re.compile(r"""^\s*(?:export\s+)?MODEL=(["']?)(.+?)\1\s*;?\s*(?:#.*)?$""", re.M)


class SwitchError(RuntimeError):
    """The switch cannot be made, or did not take."""


class RestartFailed(SwitchError):
    """The restart command ran and gave up."""


class NotRun(SwitchError):
    """The restart command could not be started: the server was never signalled."""


def bin_dir(arg: Optional[str] = None) -> Path:
    return Path(arg or os.environ.get("GHOST_BIN") or REPO.parent / "bin").expanduser()


def launcher_path(bindir: Path, name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise SwitchError(f"not a launcher name: {name!r}")
    return bindir / f"{LIVE}.{name}"


def launcher_model(path: Path) -> str:
    """The model file a launcher starts: its last ``MODEL=`` line, as the
    shell would have it. "" when it has none, or when the line uses a variable
    this process cannot resolve (it cannot be checked, so it is not guessed)."""
    try:
        found = _MODEL_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""
    if not found:
        return ""
    value = os.path.expandvars(os.path.expanduser(found[-1][1]))
    return "" if "$" in value else value


def launchers(bindir: Path) -> List[Dict[str, object]]:
    """Every named launcher: its name, model file, whether that file exists,
    and whether it is the one that is live (same bytes)."""
    live = bindir / LIVE
    live_bytes = live.read_bytes() if live.is_file() else None
    out = []
    for p in sorted(bindir.glob(LIVE + ".*")):
        name = p.name[len(LIVE) + 1:]
        if not p.is_file() or not _NAME_RE.match(name):
            continue
        model = launcher_model(p)
        out.append({"name": name, "model": model, "exists": bool(model) and os.path.isfile(model),
                    "live": live_bytes is not None and p.read_bytes() == live_bytes})
    return out


def profile_for(model_path: str) -> str:
    """The profile the agent would choose for a model file, by its name and —
    when the name fits nothing — by what the file says it is. "?" when the
    agent's code cannot be imported here."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from ghost_agent.core import model_profile as mp
        from ghost_agent.utils import gguf_tokenizer as gg
        known = mp.profiles()
        meta = None
        if mp.resolve(known, model_path) is None and os.path.isfile(model_path):
            meta = gg.read_meta(model_path)
        return mp.choose_main("", model_path, known, "", meta=meta)[0].name
    except Exception:  # noqa: BLE001
        return "?"
    finally:
        if sys.path and sys.path[0] == str(REPO / "src"):
            sys.path.pop(0)


def _get(url: str, headers: Optional[dict] = None, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        got = json.loads(r.read().decode("utf-8", "replace"))
    return got if isinstance(got, dict) else {}


def server_model(server: str) -> str:
    """The model file the server says it runs ("" = it does not answer)."""
    try:
        return str(_get(server.rstrip("/") + "/props").get("model_path") or "")
    except Exception:  # noqa: BLE001
        return ""


def api_keys() -> List[str]:
    """The agent's key, every place it may be, in the order tried."""
    out: List[str] = []
    for f in (os.environ.get("GHOST_KEY_FILE", "").strip(), None, str(REPO.parent / ".ghost_api_key")):
        if f is None:
            k = os.environ.get("GHOST_API_KEY", "").strip()
        else:
            try:
                k = Path(f).expanduser().read_text().strip() if f else ""
            except OSError:
                k = ""
        if k and k not in out:
            out.append(k)
    return out


def api_key() -> str:
    keys = api_keys()
    return keys[0] if keys else ""


_KEY_TAKEN: Dict[str, str] = {}

#: What asking the agent for its health can come to.
ANSWERED, DOWN, UNREADABLE = "answered", "down", "unreadable"


def ask_agent(agent: str, timeout: float = 5.0):
    """``(state, health, why)``: ANSWERED with its health; DOWN when nothing
    listens there (nothing can be in flight); UNREADABLE when something does
    but its health did not come back — a refused key, a timeout, not JSON —
    and so nothing is known about what it is doing. Labelled a probe."""
    url = agent.rstrip("/") + "/api/health"
    why = "no key"
    keys = api_keys() or [""]
    if _KEY_TAKEN.get(agent) in keys:               # the one it took last time first: no refusal per poll
        keys.remove(_KEY_TAKEN[agent])
        keys.insert(0, _KEY_TAKEN[agent])
    for key in keys:
        try:
            got = _get(url, {"X-Ghost-Key": key, "X-Ghost-Origin": "probe"}, timeout)
            if got:
                _KEY_TAKEN[agent] = key
            return (ANSWERED, got, "") if got else (UNREADABLE, {}, "its health answer is empty")
        except urllib.error.HTTPError as exc:
            why = f"HTTP {exc.code}" + (" — the key was refused" if exc.code in (401, 403) else "")
            if exc.code not in (401, 403):
                return UNREADABLE, {}, why
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                return DOWN, {}, "nothing listens there"
            return UNREADABLE, {}, f"{type(exc.reason).__name__}: {exc.reason}"
        except ConnectionRefusedError:
            return DOWN, {}, "nothing listens there"
        except Exception as exc:  # noqa: BLE001 — a timeout, a reset, not JSON
            return UNREADABLE, {}, f"{type(exc).__name__}: {exc}"
    return UNREADABLE, {}, why


def agent_health(agent: str) -> dict:
    """The agent's health answer ({} = none came back)."""
    return ask_agent(agent)[1]


def _put(data: bytes, mode: int, dst: Path) -> None:
    """``dst`` becomes ``data`` in one step: launchd may start the live launcher
    at any moment, and must never find half of one."""
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def _wait(what: Callable[[], bool], seconds: float, step: float, sleep=time.sleep, clock=time.monotonic) -> bool:
    end = clock() + seconds
    while True:
        if what():
            return True
        if clock() >= end:
            return False
        sleep(step)


def switch(bindir: Path, name: str, *, server: str, agent: str, restart: Optional[List[str]] = None,
           dry_run: bool = False, force: bool = False, say: Callable[[str], None] = print,
           idle_wait: float = 600.0, load_wait: float = 180.0, follow_wait: float = 600.0,
           step: float = 5.0, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Switch the live launcher to ``name``. Returns what the agent reports
    afterwards ({} with ``dry_run``). Raises ``SwitchError`` with the reason
    when it cannot be done or did not take."""
    src = launcher_path(bindir, name)
    live = bindir / LIVE
    if not src.is_file():
        have = ", ".join(str(x["name"]) for x in launchers(bindir)) or "none"
        raise SwitchError(f"no launcher {src.name} in {bindir} (there: {have})")
    model = launcher_model(src)
    if not model:
        raise SwitchError(f"{src.name} has no MODEL= line")
    if not os.path.isfile(model):
        raise SwitchError(f"{src.name} starts {model}, which is not there")
    restart = restart or [str(bindir / "restart-llama-server.sh")]
    if not dry_run and not Path(restart[0]).exists() and shutil.which(restart[0]) is None:
        raise SwitchError(f"no restart command: {restart[0]}")
    running = server_model(server)
    new_bytes, new_mode = src.read_bytes(), src.stat().st_mode & 0o7777
    had_live = live.is_file()
    old_bytes = live.read_bytes() if had_live else b""
    old_mode = (live.stat().st_mode & 0o7777) if had_live else new_mode
    live_mode = new_mode | (old_mode & 0o111) | 0o100       # launchd executes it: never left unrunnable
    same_launcher = had_live and old_bytes == new_bytes
    if same_launcher and running and os.path.realpath(running) == os.path.realpath(model):
        say(f"{name} is already live and the server runs {os.path.basename(model)} — nothing to do")
        return agent_health(agent).get("model_profile") or {}
    say(f"switch to {name}: {os.path.basename(model)} (profile: {profile_for(model)})")
    previous = bindir / f"{LIVE}.{PREVIOUS}"
    had_previous = previous.is_file()
    prev_bytes = previous.read_bytes() if had_previous else b""
    prev_mode = (previous.stat().st_mode & 0o7777) if had_previous else 0o644
    if dry_run:
        keep = "" if same_launcher else f"keep {LIVE} as {previous.name}, copy {src.name} over it, "
        say(f"  would wait for the agent to be idle, {keep}run {' '.join(restart)},")
        say("  wait for the server to name the model and for the agent to follow")
        return {}
    if not force:
        state, health, why = ask_agent(agent)
        if state == UNREADABLE:
            raise SwitchError(f"cannot tell whether the agent is idle ({why}) — not switched (use --force to switch anyway)")
        if state == ANSWERED:                       # an agent that is DOWN has nothing in flight
            say("  waiting for the agent to be idle …")

            def _idle() -> bool:
                st, h, _ = ask_agent(agent)
                n = h.get("foreground_requests")
                return st == DOWN or (st == ANSWERED and type(n) is int and n == 0)
            if not _wait(_idle, idle_wait, step, sleep, clock):
                raise SwitchError("the agent stayed busy — not switched (use --force to switch under a request)")
    def _back() -> None:
        """The directory as it was: the live launcher, and what `previous` held."""
        if same_launcher:
            return
        try:
            if had_live:
                _put(old_bytes, old_mode, live)
            else:
                live.unlink(missing_ok=True)
            if had_previous:
                _put(prev_bytes, prev_mode, previous)
            else:
                previous.unlink(missing_ok=True)
        except OSError as exc:
            raise SwitchError(f"the launchers could NOT be put back ({exc}): {live.name} may be {src.name}'s, and "
                              f"{previous.name} may hold the launcher that was live — check {bindir} by hand") from None

    def _restart() -> None:
        try:
            r = subprocess.run(restart, capture_output=True, text=True)
        except OSError as exc:
            raise NotRun(f"{' '.join(restart)} could not be run: {exc}") from None
        if r.returncode != 0:
            raise RestartFailed(f"{' '.join(restart)} failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")

    def _up() -> bool:
        return os.path.realpath(server_model(server) or "/") == os.path.realpath(model)
    try:
        if not same_launcher:
            if had_live:
                _put(old_bytes, old_mode, previous)     # (every launcher was read first: `switch previous` swaps the two)
            _put(new_bytes, live_mode, live)
        say("  restarting the model server …")
        try:
            _restart()
        except NotRun as exc:
            _back()
            raise SwitchError(f"{exc}. Nothing was signalled; the launcher that was live is back") from None
        except RestartFailed as exc:
            # the command gave up; the server may still be loading the new model. Watch — never signal again.
            say(f"  {exc}; watching whether the server comes up on its own …")
            if not _wait(_up, load_wait, step, sleep, clock):
                _back()
                raise SwitchError(f"{exc}. The launcher that was live is back, and the server was NOT signalled "
                                  "again — look at it before restarting anything") from None
        else:
            if not _wait(_up, load_wait, step, sleep, clock):
                _back()
                gone = f"the server did not come up on {os.path.basename(model)} in {int(load_wait)} s"
                if not had_live or same_launcher:
                    raise SwitchError(gone)
                say(f"  {gone} — the previous launcher is back; restarting on it")
                try:
                    _restart()
                except RestartFailed as again:
                    raise SwitchError(f"{gone}; and the restart on the previous launcher failed too: {again}") from None
                raise SwitchError(gone)
    except SwitchError:
        raise
    except BaseException:                           # an interrupt, a full disk: the launcher is put back, the server left alone
        _back()
        say("  interrupted — the launcher that was live is back; the server was not signalled again")
        raise
    say(f"  the server runs {os.path.basename(model)}; waiting for the agent to follow …")
    want = os.path.basename(model)
    got: dict = {}

    def _followed() -> bool:
        nonlocal got
        got = agent_health(agent).get("model_profile") or {}
        return got.get("server_model") == want
    if not _wait(_followed, follow_wait, step, sleep, clock):
        raise SwitchError(f"the server runs {want} but the agent has not followed after {int(follow_wait)} s "
                          f"(it follows at its next head warm-up; it reports {got.get('server_model') or 'nothing'})")
    say("  " + describe(got))
    return got


def describe(model_profile: dict) -> str:
    """One line on what the agent reports about its model."""
    if not model_profile:
        return "the agent does not answer"
    tok = model_profile.get("tokens") or {}
    if tok.get("exact"):
        count = "token counts exact (the model file's own tokenizer)"
    elif tok.get("local_tokens"):
        count = f"token counts from a tokenizer {abs(tok.get('drift') or 0) * 100:.1f}% off the server's"
    elif tok:
        count = f"token counts estimated ({tok.get('chars_per_token')} chars per token)"
    else:
        count = "token counts not checked"
    return (f"agent: profile {model_profile.get('name') or '?'} "
            f"({model_profile.get('why') or 'no reason given'}); {count}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Put another model behind the agent.")
    ap.add_argument("--bin", default=None, help="directory of the launchers (default: $GHOST_BIN, else bin beside the repository)")
    ap.add_argument("--server", default=os.environ.get("GHOST_LLAMA_URL", "http://127.0.0.1:8088"))
    ap.add_argument("--agent", default=os.environ.get("GHOST_URL", "http://127.0.0.1:8000"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    sv = sub.add_parser("save")
    sv.add_argument("name")
    sw = sub.add_parser("switch")
    sw.add_argument("name")
    sw.add_argument("--dry-run", action="store_true")
    sw.add_argument("--force", action="store_true", help="do not wait for the agent to be idle")
    a = ap.parse_args(argv)
    bindir = bin_dir(a.bin)

    def _stop(signum, frame):                       # a terminal closing, a kill: undo as for Ctrl-C
        raise KeyboardInterrupt
    old_handlers = {}
    if a.cmd == "switch":
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                old_handlers[sig] = signal.signal(sig, _stop)
            except (ValueError, OSError):           # not the main thread
                pass
    try:
        if a.cmd == "list":
            rows = launchers(bindir)
            if not rows:
                print(f"no launchers ({LIVE}.<name>) in {bindir}")
            for r in rows:
                model = str(r["model"])
                note = " (what the last switch replaced)" if r["name"] == PREVIOUS else ""
                print(f"{'*' if r['live'] else ' '} {r['name']:<24} {os.path.basename(model) or '(no MODEL= line it can read)':<52} "
                      f"{'profile ' + profile_for(model) if r['exists'] else 'MODEL FILE MISSING'}{note}")
        elif a.cmd == "status":
            running = server_model(a.server)
            print(f"server: {os.path.basename(running) if running else 'does not answer'}")
            live = [r["name"] for r in launchers(bindir) if r["live"]]
            print(f"launcher: {', '.join(map(str, live)) if live else 'the live one matches no named launcher (use: save NAME)'}")
            state, health, why = ask_agent(a.agent)
            print(describe(health.get("model_profile") or {}) if state == ANSWERED else f"agent: {state} ({why})")
        elif a.cmd == "save":
            dst = launcher_path(bindir, a.name)
            if dst.exists() or a.name == PREVIOUS:
                raise SwitchError(f"{dst.name} exists or is kept for the switch — choose another name")
            if not (bindir / LIVE).is_file():
                raise SwitchError(f"no {LIVE} in {bindir}")
            shutil.copy2(bindir / LIVE, dst)
            print(f"kept the live launcher as {dst.name}")
        else:
            switch(bindir, a.name, server=a.server, agent=a.agent, dry_run=a.dry_run, force=a.force)
    except SwitchError as exc:
        print(f"use_model: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("use_model: interrupted", file=sys.stderr)
        return 130
    finally:
        for sig, h in old_handlers.items():
            signal.signal(sig, h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
