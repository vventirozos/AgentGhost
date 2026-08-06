"""Log beautification (2026-08-05): console repeat-collapse + verify
purpose tags.

The two invariants that matter:

* the collapse is CONSOLE-ONLY — the durable GhostStream mirror records
  every occurrence, because several instruments grep-COUNT mirror lines
  (the escalation-overturn double-count lesson: one logger, complete);
* the purpose tag rides a contextvar (`verify_purpose_context`) so the
  llm-routing log sites, several layers below the caller, can say WHY a
  verification call is happening.
"""
from __future__ import annotations

import logging as _logging
from pathlib import Path

import pytest

from ghost_agent.utils import logging as glog

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_collapse(monkeypatch):
    monkeypatch.setattr(glog, "_COLLAPSE_STATE", None)
    monkeypatch.delenv("GHOST_LOG_COLLAPSE", raising=False)
    yield
    glog._COLLAPSE_STATE = None


class _MirrorSpy:
    def __init__(self):
        self.lines = []

    def log(self, level, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


@pytest.fixture
def mirror(monkeypatch):
    spy = _MirrorSpy()
    monkeypatch.setattr(glog, "_MIRROR_LOGGER", spy)
    return spy


def test_consecutive_repeats_print_once_with_summary(capsys, mirror):
    for _ in range(5):
        glog.pretty_log("Critic Compute", "Routing verification to X",
                        icon="🧪")
    glog.pretty_log("Other Line", "different", icon="🔧")
    out = capsys.readouterr().out
    assert out.count("Routing verification to X") == 1     # printed once
    assert "repeated ×5" in out                            # run summarised
    assert "different" in out
    # The mirror got EVERY occurrence — the instrument stays complete.
    assert sum("routing verification to x" in l.lower()
               for l in mirror.lines) == 5


def test_non_repeating_lines_all_print(capsys, mirror):
    for i in range(3):
        glog.pretty_log("Title", f"content {i}", icon="🔧")
    out = capsys.readouterr().out
    for i in range(3):
        assert f"content {i}" in out
    assert "repeated" not in out


def test_kill_switch_disables_collapse(capsys, mirror, monkeypatch):
    monkeypatch.setenv("GHOST_LOG_COLLAPSE", "0")
    for _ in range(3):
        glog.pretty_log("Critic Compute", "Routing verification to X",
                        icon="🧪")
    out = capsys.readouterr().out
    assert out.count("Routing verification to X") == 3


def test_different_request_ids_do_not_collapse(capsys, mirror):
    tok = glog.request_id_context.set("req-A")
    glog.pretty_log("T", "same content", icon="🔧")
    glog.request_id_context.reset(tok)
    tok = glog.request_id_context.set("req-B")
    glog.pretty_log("T", "same content", icon="🔧")
    glog.request_id_context.reset(tok)
    out = capsys.readouterr().out
    assert out.count("same content") == 2


def test_frame_markers_are_exempt(capsys, mirror):
    # BEGIN/END take early-return branches — collapse must not touch them.
    glog.pretty_log("ignored", special_marker="BEGIN")
    glog.pretty_log("ignored", special_marker="END")
    out = capsys.readouterr().out
    assert "request started" in out and "request finished" in out


def test_verify_purpose_contextvar_set_and_reset():
    assert glog.verify_purpose_context.get() == ""
    with glog.verify_purpose("turn gate"):
        assert glog.verify_purpose_context.get() == "turn gate"
    assert glog.verify_purpose_context.get() == ""


def test_purpose_reaches_routing_log_lines():
    """Source fence: both llm.py routing sites read the contextvar, and
    the two callers that know their purpose set it."""
    llm = (REPO / "src" / "ghost_agent" / "core" / "llm.py").read_text()
    assert llm.count("verify_purpose_context.get()") >= 2
    agent = (REPO / "src" / "ghost_agent" / "core" / "agent.py").read_text()
    assert agent.count('with verify_purpose("turn gate")') == 3
    main = (REPO / "src" / "ghost_agent" / "main.py").read_text()
    assert 'verify_purpose("reflection plan-verify")' in main
