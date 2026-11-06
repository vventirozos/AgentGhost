"""§4PJ (2026-10-16, operator: "cut these, we don't need them any more").

Removed: the composite confidence score (core/confidence.py), its inputs (the
per-domain competence profile, memory/competence.py, and the per-token
logprob entropy tracker, core/entropy.py), the calibration corpus + hourly
idle refit (core/calibration.py), the dual-solver arbiter (core/arbiter.py),
the low-confidence deep-verification trigger, the once-a-day "was it right?"
label ask, the failure-report calibration tier, and every log line,
introspect section and liveness probe that described them.

Measured before the cut (2026-10-16): on the current epoch the score ranked
outcomes at AUC 0.48 [0.30, 0.67] and lost to a constant as a probability;
its only live consumer (deep verification on the lowest 15%) sat behind the
verifier judge, which has been off since 2026-10-14.

Each pin names the world in which it fails.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


@pytest.mark.parametrize("mod", [
    "ghost_agent.core.calibration",
    "ghost_agent.core.confidence",
    "ghost_agent.core.entropy",
    "ghost_agent.core.arbiter",
    "ghost_agent.memory.competence",
])
def test_the_removed_modules_are_gone(mod):
    """Fails where a removed module is restored or a stale .py lingers."""
    with pytest.raises(ImportError):
        importlib.import_module(mod)


# ── the launcher flag that stays accepted ────────────────────────────────

def _parse(monkeypatch, argv):
    import ghost_agent.main as M
    monkeypatch.setattr(sys, "argv", ["ghost_agent.main", *argv])
    return M.parse_args()


def test_an_old_launcher_passing_the_refit_cooldown_still_boots(monkeypatch):
    """Fails where the flag is deleted while an old launcher may still pass
    it: argparse exits and launchd respawns a process that dies at boot."""
    args = _parse(monkeypatch, ["--enable-metacog", "--calib-refit-cooldown", "21600"])
    assert args.calib_refit_cooldown == 21600


def test_the_resolved_config_labels_the_refit_cooldown_ignored(monkeypatch):
    """Fails where /api/health and last_config.json show the old value as if
    it configured something."""
    import ghost_agent.main as M
    args = _parse(monkeypatch, ["--calib-refit-cooldown", "21600"])
    cfg = M._build_resolved_config(args, types.SimpleNamespace())
    assert "ignored" in str(cfg["arg.calib_refit_cooldown"])


def test_nothing_reads_the_refit_cooldown():
    """Fails where a reader of the ignored flag reappears (it would act on a
    value the config dump says is ignored). Every attribute read, all of src."""
    # Both read shapes: `args.calib_refit_cooldown` and the getattr string
    # the removed refit actually used. The one sanctioned mention is the
    # key of main.py's deprecated-no-op table.
    reads = []
    for p in SRC.rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr == "calib_refit_cooldown":
                reads.append(f"{p.relative_to(SRC)}:{n.lineno}")
            elif (isinstance(n, ast.Constant) and n.value == "calib_refit_cooldown"
                  and p.name != "main.py"):
                reads.append(f"{p.relative_to(SRC)}:{n.lineno} (string)")
    assert reads == [], reads
    import ghost_agent.main as M
    assert "calib_refit_cooldown" in M._DEPRECATED_NOOP_ARGS


# ── the idle scheduler, the activity ledger, the liveness view ───────────

def test_no_calibration_job_is_scheduled_even_with_a_tracker_attached():
    """Fails where the idle refit comes back keyed on a context attribute."""
    import ghost_agent.core.agent as A
    ctx = types.SimpleNamespace(journal=object(), 
                                calibration_tracker=object(), project_store=object(),
                                self_model=types.SimpleNamespace(enabled=True),
                                workspace_model=types.SimpleNamespace(enabled=False))
    on = A.idle_scheduler_jobs(types.SimpleNamespace(no_dream=False, autoadvance_idle=False), ctx)
    assert not any("calibration" in j for j in on), on
    assert "calibration" not in {label for label, _t, _f in A.IDLE_SCHEDULER_JOBS}


def test_old_calibration_rows_still_render_but_are_never_expected():
    """Fails both ways: a missing label renders old ledger rows as a raw
    slug; a kept expectation pages the operator about a loop that is gone."""
    from ghost_agent.core import autonomous_activity as AA
    assert AA._PHASE_LABELS.get("calibration") == "calibration"
    assert "calibration" not in AA.PHASE_EXPECTATION


def test_no_liveness_probe_describes_a_removed_mechanism():
    from ghost_agent.core import liveness as LV
    names = {p.name for p in LV.PROBES}
    assert not names & {"calibration.fit", "metacog.arbitration"}, names


# ── the verifier keeps its own logprob probe ─────────────────────────────

async def test_the_verdict_score_probe_still_asks_for_logprobs():
    """The probe's payload helper lived in the removed entropy module. Fails
    where the inlined fields are dropped: the probe then reads only the one
    emitted digit and never its distribution."""
    from ghost_agent.core.verifier import Verifier
    seen = {}

    class _Client:
        critic_clients = None

        async def chat_completion(self, payload, **kw):
            seen.update(payload)
            return {"choices": [{"message": {"content": "7"},
                                 "logprobs": {"content": []}}]}

    v = Verifier(llm_client=_Client())
    score = await v._verdict_score_probe("claim", "evidence", "context")
    assert seen.get("logprobs") is True and seen.get("top_logprobs") == 10
    assert "tools" not in seen and "n_probs" not in seen
    assert score == pytest.approx(7 / 9.0)


# ── owner feedback and the context ───────────────────────────────────────

def test_a_thumb_never_touches_a_calibration_tracker(monkeypatch):
    """Fails where the relabel comes back: a leftover tracker attribute that
    raises on ANY access must stay untouched, and the line says nothing
    about calibration."""
    from ghost_agent.core import feedback as FB
    from tests.test_4ee_feedback_pins import _DayCol, _Traj, _day

    class _Tripwire:
        def __getattr__(self, name):
            raise AssertionError(f"calibration tracker touched: .{name}")

    seen = []
    monkeypatch.setattr(FB, "pretty_log", lambda t, m, **kw: seen.append((t, m)))
    col = _DayCol({_day(0): [_Traj("t", "r")]})
    ctx = types.SimpleNamespace(trajectory_collector=col, calibration_tracker=_Tripwire(),
                                _recent_trajectories_for_correction=None,
                                self_model=None, args=None)
    out = FB.apply_human_label(types.SimpleNamespace(context=ctx), "r", "negative", "wrong")
    assert out["ok"] is True
    line = [m for t, m in seen if t == "Human Feedback"][-1]
    assert "calibration" not in line, line


def test_the_context_carries_no_confidence_state():
    """Fails where a writer of the per-turn confidence / calibration stashes
    is restored on the shared context."""
    from ghost_agent.core.agent import GhostContext
    ctx = GhostContext(args=None, sandbox_dir=None, memory_dir=None, tor_proxy=None)
    for gone in ("calibration_tracker", "last_confidence", "last_confidence_req",
                 "last_entropy_reading", "_calib_pending"):
        assert not hasattr(ctx, gone), gone


async def test_a_pasted_traceback_is_no_longer_filed_anywhere(tmp_path):
    """The failure-report tier was calibration-only. Fails where a
    non-correction message starts writing a label again: the corpus row of
    the prior turn must stay untouched."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory, Outcome
    from collections import OrderedDict
    col = TrajectoryCollector(root=tmp_path / "t", session_id="s")
    recorded = []
    a = GhostAgent.__new__(GhostAgent)
    # the shape the removed tier needed — a tracker and a stash keyed by the
    # prior reply's fingerprint — so a restored tier would have a target
    fp = GhostAgent._response_fingerprint("Done — the game is ready.")
    a.context = types.SimpleNamespace(
        trajectory_collector=col, last_user_content="", self_model=None, args=None,
        calibration_tracker=types.SimpleNamespace(record=lambda **kw: recorded.append(kw)),
        _recent_calib_for_correction=OrderedDict({fp: {"composite": 0.9}}))
    traj = Trajectory(id="prior", user_request="build the game",
                      final_response="Done — the game is ready.",
                      outcome=Outcome.UNKNOWN.value)
    col.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)
    msgs = [{"role": "user", "content": "build the game"},
            {"role": "assistant", "content": traj.final_response},
            {"role": "user", "content": "game.js:45 Uncaught TypeError: x is not a function"}]
    a._maybe_promote_prior_turn_via_user_correction(
        msgs, "game.js:45 Uncaught TypeError: x is not a function")
    assert col.latest_correction("prior") is None
    assert traj.outcome == Outcome.UNKNOWN.value
    assert recorded == [], recorded


def test_the_retired_calibration_phase_is_not_an_unregistered_row(tmp_path):
    """§4PJ follow-up review: 1,815 old calibration rows printed
    "calibration … ← UNREGISTERED phase" in the learning report for 7 days."""
    import json
    import time
    from ghost_agent.core.learning_health import activity_liveness
    p = tmp_path / "autonomous_activity.jsonl"
    p.write_text(json.dumps({"phase": "calibration", "ts": time.time(), "summary": "x", "severity": "info"}) + "\n")
    assert all(r["phase"] != "calibration" for r in activity_liveness(p)["rows"])
