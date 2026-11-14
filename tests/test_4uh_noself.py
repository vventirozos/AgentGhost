"""§4UH-noself (2026-11-14): the daily self-look is removed, code and data.

A calibration rated 40 samples of six replies with the look's own prompt — replies written with no character,
with three different dial settings and with a strong "short and dry, blunt" sentence. Every pool came back
about (+1, +1, +1), where blind readers had told those characters apart 75–97 %. The main model rating
itself cannot see the character, so the look carried nothing (its vote was already off).

Pinned here: the module is gone and nothing imports it, the daily rule takes no votes, the `mirror` banner,
the character view and the introspect line say nothing about a look, and the idle phase makes no model call.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as A
from ghost_agent.selfhood import character as C
from ghost_agent.selfhood import loop as L

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(C.ENV, raising=False)
    C._MEMO.clear()
    yield
    C._MEMO.clear()


def _home(tmp_path) -> Path:
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "selfhood").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_the_selfview_module_cannot_be_imported():
    assert not (SRC / "selfhood" / "selfview.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ghost_agent.selfhood.selfview")


def test_nothing_in_src_imports_it():
    """ast walk over every module: no import of `selfview` (relative or absolute)."""
    bad = []
    for p in SRC.rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                if mod.endswith("selfview") or any(a.name == "selfview" for a in n.names):
                    bad.append(f"{p.relative_to(SRC)}:{n.lineno}")
            elif isinstance(n, ast.Import) and any(a.name.endswith("selfview") for a in n.names):
                bad.append(f"{p.relative_to(SRC)}:{n.lineno}")
    assert bad == []


def test_the_daily_rule_takes_no_votes():
    assert "votes" not in inspect.signature(C.daily).parameters
    assert "votes" not in inspect.signature(C.propose).parameters


def test_the_reply_picker_went_with_the_look():
    for name in ("pick_replies", "_replies_since", "written_under_strong_mood", "_look_epoch", "_mood_date",
                 "REPLIES_PER_LOOK", "REPLY_WINDOW_DAYS", "MIN_REPLY_CHARS", "REPLY_MAX_CHARS", "STRONG_MOOD_MIN"):
        assert not hasattr(L, name), name


def test_the_mirror_banner_has_no_look_section(tmp_path):
    home = _home(tmp_path)
    b = L.banner(home)
    assert b.startswith("**The strange loop**") and "**Now:**" in b
    assert "What I see" not in b and "look" not in b.lower() and "vote" not in b.lower()


def test_the_view_and_the_introspect_line_say_no_look(tmp_path):
    home = _home(tmp_path)
    (home / "system" / "selfhood" / "selfview.jsonl").write_text(          # a leftover file is never read
        '{"at": "2026-11-12T10:00:00Z", "date": "2026-11-12", "observed": {"warmth": 1, "humour": 1, '
        '"directness": 0}, "dials": {"warmth": -1, "humour": 1, "directness": 0}, "replies": 4}\n')
    assert "look" not in C.view(home).lower() and "look" not in C.line(home).lower()
    assert "how my days go" in C.line(home)


@pytest.mark.asyncio
async def test_the_idle_phase_makes_no_model_call(tmp_path):
    asked = []

    class _LLM:
        async def chat_completion(self, payload, **kw):
            asked.append(payload)
            return {"choices": [{"message": {"content": "0 0 0"}}]}
    home = _home(tmp_path)
    ctx = SimpleNamespace(memory_dir=str(home / "system" / "memory"), llm_client=_LLM(), args=None, self_model=None)
    ran, notes = [], {}
    await A.GhostAgent.__new__(A.GhostAgent)._character_phase(home, ran, notes, ctx)
    assert asked == [] and not (home / "system" / "selfhood" / "selfview.jsonl").exists()


def test_the_idle_job_no_longer_says_it_rates_its_replies():
    purpose = A.IDLE_JOB_PURPOSE["character"]
    assert "rates" not in purpose and "replies" not in purpose
