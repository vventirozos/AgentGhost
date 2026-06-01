"""Tests for the GEPA read-site wiring (quick win c).

GEPA tuned the `reflection.critique` and `tool_selection.pick` signatures
but nothing READ them at inference (only `planning.decompose` was wired).
These tests cover the new `reflection.critique` read-site in
build_reflection_prompt. The `tool_selection.pick` read lives in the
planner prompt assembly in core/agent.py (alongside planning.decompose).
"""

import ghost_agent.optim.loader as loader
from ghost_agent.reflection import prompts as rp
from ghost_agent.distill.schema import Trajectory, Outcome


def _traj():
    return Trajectory(user_request="parse a log", failure_reason="boom",
                      outcome=Outcome.FAILED.value)


def test_reflection_prompt_prepends_tuned_critique(monkeypatch):
    monkeypatch.setattr(
        loader, "tuned_instruction",
        lambda name, default="": ("TUNED-CRITIQUE-INSTRUCTION"
                                  if name == "reflection.critique" else default),
    )
    out = rp.build_reflection_prompt(_traj())
    assert out.startswith("TUNED-CRITIQUE-INSTRUCTION")
    # The baseline template still follows.
    assert "DIAGNOSIS:" in out


def test_reflection_prompt_baseline_when_no_tuned(monkeypatch):
    monkeypatch.setattr(loader, "tuned_instruction", lambda name, default="": default)
    out = rp.build_reflection_prompt(_traj())
    assert not out.startswith("TUNED")
    assert out.lstrip().startswith("You are reviewing")


def test_reflection_prompt_survives_loader_error(monkeypatch):
    def _boom(name, default=""):
        raise RuntimeError("loader down")
    monkeypatch.setattr(loader, "tuned_instruction", _boom)
    out = rp.build_reflection_prompt(_traj())  # must not raise
    assert "DIAGNOSIS:" in out
