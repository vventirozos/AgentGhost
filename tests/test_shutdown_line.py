"""§4IC — the shutdown line says how long the process ran and what it was doing.

Three graceful SIGTERM shutdowns on 2026-09-17 (13:02, 13:50, 16:44) came
from outside the codebase (launchctl: last exit status -15; no launchctl /
sudo entry in the unified log; no script, cron or plist on the box sends
one) and the log said only "draining background work…". The line now
carries uptime and the in-flight turns so a deploy can be told from an
interruption mid-request.
"""
import ast
import inspect
import types
from unittest.mock import patch

from ghost_agent import main as mn
from ghost_agent.main import shutdown_line


def _reg(turns):
    return types.SimpleNamespace(list=lambda: turns)


def _turn(rid, running=True):
    return types.SimpleNamespace(req_id=rid, running=running)


def test_uptime_and_in_flight_turns_are_named(monkeypatch):
    monkeypatch.setattr(mn, "_BOOT_MONO", mn.time.monotonic() - 125.0)
    with patch("ghost_agent.core.turns.get_turn_registry",
               return_value=_reg([_turn("3cb143fc01"), _turn("queued01", running=False),
                                  _turn("21b295ef02")])):
        line = shutdown_line(object())
    assert line.startswith("draining background work…")
    assert "uptime 2.1 min" in line
    assert "2 turn(s) in flight (3cb143fc, 21b295ef)" in line


def test_no_boot_stamp_and_no_turns_still_says_something(monkeypatch):
    monkeypatch.setattr(mn, "_BOOT_MONO", None)
    with patch("ghost_agent.core.turns.get_turn_registry", return_value=_reg([])):
        line = shutdown_line(object())
    assert "uptime" not in line and "0 turn(s) in flight" in line


def test_registry_failure_never_raises(monkeypatch):
    monkeypatch.setattr(mn, "_BOOT_MONO", None)
    with patch("ghost_agent.core.turns.get_turn_registry", side_effect=RuntimeError("boom")):
        line = shutdown_line(object())
    assert "turn registry unreadable" in line


def test_lifespan_stamps_boot_and_uses_the_line():
    tree = ast.parse(inspect.getsource(mn))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    stamps = [s for s in ast.walk(fn) if isinstance(s, ast.Assign)
              and any(getattr(t, "id", "") == "_BOOT_MONO" for t in s.targets)]
    assert len(stamps) == 1
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "pretty_log"
             and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "System Shutdown"]
    assert len(calls) == 1
    assert getattr(calls[0].args[1].func, "id", "") == "shutdown_line"
