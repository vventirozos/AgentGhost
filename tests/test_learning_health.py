"""Learning-health telemetry (2026-07-26 improvement #1)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ghost_agent.core.learning_health import (
    collect_learning_health, render_learning_health,
)


def _seed(md: Path):
    md.mkdir(parents=True, exist_ok=True)
    (md.parent / "calibration").mkdir(parents=True, exist_ok=True)
    # Playbook: one fail-only lesson, one pass-only, one stale.
    playbook = [
        {"trigger": "a", "succeeded_retrievals": 5, "failed_retrievals": 0,
         "retrievals": 6, "helpful_retrievals": 5, "verified": True},
        {"trigger": "b", "succeeded_retrievals": 0, "failed_retrievals": 4,
         "retrievals": 8, "helpful_retrievals": 0},
        {"trigger": "c", "retrievals": 10, "helpful_retrievals": 1,
         "graduated": True},
    ]
    (md / "skills_playbook.json").write_text(json.dumps(playbook))
    # Competence: one domain crossing the gate, one below.
    comp = {
        "shell|*": {"alpha": 90, "beta": 10, "n": 100},
        "sql|*": {"alpha": 3, "beta": 1, "n": 4},
        "*|*": {"alpha": 1, "beta": 1, "n": 104},
    }
    (md / "competence_profile.json").write_text(json.dumps(comp))
    (md / "adaptive_threshold.json").write_text(
        json.dumps({"threshold": 0.6, "window": [1, 2, 3]}))
    (md / "auto_skills.json").write_text(json.dumps({"sig1": {}}))
    # Calibration: degenerate entropy (all 0.5).
    calib = md.parent / "calibration"
    (calib / "calibration_params.json").write_text(json.dumps({
        "w_entropy": 0.0, "w_competence": 1.0, "brier": 0.07,
        "threshold": 0.7, "n_samples": 3}))
    with (calib / "calibration.jsonl").open("w") as fh:
        for _ in range(3):
            fh.write(json.dumps({"entropy_component": 0.5, "outcome": 1.0}) + "\n")
    # Episodes db.
    conn = sqlite3.connect(md / "episodic_memory.db")
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, context TEXT "
                 "DEFAULT '', cluster_id TEXT DEFAULT '', outcome_success INT "
                 "DEFAULT 0, consolidated INT DEFAULT 0)")
    conn.execute("INSERT INTO episodes (context, cluster_id, outcome_success, consolidated) "
                 "VALUES ('tools: x', 'shell', 1, 0)")
    conn.execute("INSERT INTO episodes (context, cluster_id, outcome_success, consolidated) "
                 "VALUES ('', '', 1, 1)")
    conn.commit()
    conn.close()


def test_collect_sections(tmp_path):
    md = tmp_path / "memory"
    _seed(md)
    r = collect_learning_health(md)
    assert r["lessons"]["total"] == 3
    assert r["lessons"]["present_on_failure_only"] == 1
    assert r["lessons"]["present_on_pass_only"] == 1
    assert r["lessons"]["stale_prune_candidates"] == 2   # b and c: ret≥5, low hit-rate
    assert r["competence"]["domains_injecting"] == ["shell"]   # sql below gate
    assert r["competence"]["injects_into_prompt"] is True
    assert r["episodes"]["total"] == 2
    assert r["episodes"]["with_context"] == 1
    assert r["calibration"]["entropy_learnable"] is False       # degenerate
    assert r["calibration"]["w_entropy"] == 0.0
    assert r["auto_skills"]["graduated"] == 1


def test_render_is_a_string(tmp_path):
    md = tmp_path / "memory"
    _seed(md)
    out = render_learning_health(md)
    assert "LEARNING HEALTH" in out
    assert "COMPETENCE" in out
    assert "CALIBRATION" in out


def test_failure_arm_inert_warning(tmp_path):
    """The arm is flagged inert only when the success side clearly flows
    (≥20 ticks) while not a single failure tick ever landed. (Rewritten
    2026-07-27: the old fail-ONLY-lesson test was a metric artifact — at
    a ~96% turn pass rate any retrieved lesson accrues a success, so the
    bucket was near-impossible by construction.)"""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    (md / "skills_playbook.json").write_text(json.dumps([
        {"trigger": f"t{i}", "succeeded_retrievals": 5, "failed_retrievals": 0,
         "retrievals": 5, "helpful_retrievals": 4} for i in range(6)]))
    out = render_learning_health(md)
    assert "ZERO failed-retrieval ticks" in out


def test_failure_arm_flowing_is_not_flagged(tmp_path):
    """Failure ticks on mixed lessons = the arm is alive; the report must
    say so instead of warning (the live 2026-07-27 state: 29 failed ticks,
    0 fail-only lessons)."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    (md / "skills_playbook.json").write_text(json.dumps([
        {"trigger": f"t{i}", "succeeded_retrievals": 4, "failed_retrievals": 2,
         "retrievals": 6, "helpful_retrievals": 3} for i in range(5)]))
    out = render_learning_health(md)
    assert "ZERO failed-retrieval ticks" not in out
    assert "failure ticks flow" in out


def test_failure_arm_thin_corpus_not_flagged(tmp_path):
    """Zero failure ticks with a THIN success side (<20 ticks) is 'not
    enough signal', not inertness — no warning either way."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    (md / "skills_playbook.json").write_text(json.dumps([
        {"trigger": "t0", "succeeded_retrievals": 5, "failed_retrievals": 0,
         "retrievals": 5, "helpful_retrievals": 4}]))
    out = render_learning_health(md)
    assert "ZERO failed-retrieval ticks" not in out


def test_missing_stores_degrade_gracefully(tmp_path):
    md = tmp_path / "empty"
    md.mkdir()
    r = collect_learning_health(md)          # no stores at all
    assert isinstance(r, dict)
    assert render_learning_health(md)         # never raises


def test_cognitive_wiring_section(tmp_path):
    md = tmp_path / "memory"
    md.mkdir()
    r = collect_learning_health(md)
    cw = r["cognitive_wiring"]
    assert "selfhood" in cw and "calibration" in cw
    assert cw["calibration"]["write_only"] is False
    # 2026-07-27: retired (module removed); the entry stays to document
    # the decision.
    assert "RETIRED" in cw["self_consistency"]["status"]
