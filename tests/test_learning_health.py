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
    # The inject gate is on TOTAL observations across domain rollups
    # (mirrors agent.py, which then renders EVERY domain — there is no
    # per-domain gate): shell 100 + sql 4 = 104 >= 20.
    assert r["competence"]["total_observations"] == 104
    assert r["competence"]["min_obs_gate"] == 20
    assert r["competence"]["injects_into_prompt"] is True
    assert "domains_injecting" not in r["competence"]  # mirrored no mechanism
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


def test_competence_gate_is_total_not_per_domain(tmp_path):
    """The live divergence case (bug found 2026-07-27): several domains all
    below the per-domain count, but the TOTAL crosses the gate — agent.py
    injects the block every turn, so the instrument must say INJECTING.
    The old per-domain mirror reported 'NONE (block not injecting yet)'."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    comp = {f"d{i}|*": {"alpha": 5, "beta": 5, "n": 8} for i in range(4)}
    (md / "competence_profile.json").write_text(json.dumps(comp))
    r = collect_learning_health(md)
    assert r["competence"]["total_observations"] == 32
    assert r["competence"]["injects_into_prompt"] is True
    assert "INJECTING" in render_learning_health(md)


def test_competence_below_total_gate_not_injecting(tmp_path):
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    (md / "competence_profile.json").write_text(json.dumps(
        {"shell|*": {"alpha": 5, "beta": 5, "n": 10}}))
    r = collect_learning_health(md)
    assert r["competence"]["injects_into_prompt"] is False
    assert "not injecting yet" in render_learning_health(md)


def test_competence_gate_reads_the_mechanisms_constant():
    """The gate must come from GhostAgent._COMPETENCE_MIN_OBS (the code
    that actually gates injection), not a hand-copied literal."""
    from ghost_agent.core.learning_health import _live_competence_gate
    from ghost_agent.core.agent import GhostAgent
    assert _live_competence_gate() == GhostAgent._COMPETENCE_MIN_OBS


def _calib_row(**fields):
    """A calibration row in the CURRENT epoch. Rows without an epoch tag are
    derived from `ts`, and an undated row is treated as legacy and excluded
    from both the fit and this report — correct, but it means a fixture that
    omits both describes an empty corpus."""
    from ghost_agent.core.calibration import CURRENT_EPOCH
    fields.setdefault("epoch", CURRENT_EPOCH)
    return json.dumps(fields) + "\n"


def test_entropy_learnable_mirrors_the_fit_gate(tmp_path):
    """calibration.py fits w_entropy only when >= _MIN_ENTROPY_SAMPLES
    observed samples exist AND both outcome classes are represented among
    them. 40 observed one-class samples must NOT read as LEARNABLE (the
    old distinct>=3 formula said it was)."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    calib = md.parent / "calibration"
    calib.mkdir(parents=True)
    with (calib / "calibration.jsonl").open("w") as fh:
        for i in range(40):  # varied entropy, all positive-class
            fh.write(_calib_row(entropy_component=0.3 + (i % 10) / 20.0,
                                entropy_observed=True, outcome=1.0))
    r = collect_learning_health(md)
    cal = r["calibration"]
    assert cal["entropy_observed_samples"] == 40
    assert cal["entropy_observed_pos"] == 40
    assert cal["entropy_observed_neg"] == 0
    assert cal["entropy_learnable"] is False  # one-class → fit pins w_e=0


def test_entropy_learnable_true_with_both_classes(tmp_path):
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    calib = md.parent / "calibration"
    calib.mkdir(parents=True)
    with (calib / "calibration.jsonl").open("w") as fh:
        for i in range(35):
            ok = bool(i % 5)
            # Entropy must actually SEPARATE the classes, not merely vary —
            # that is the second half of the fit gate.
            fh.write(_calib_row(
                entropy_component=(0.80 if ok else 0.20) + (i % 3) / 100.0,
                entropy_observed=True,
                outcome=1.0 if ok else 0.0))
    cal = collect_learning_health(md)["calibration"]
    assert cal["entropy_observed_pos"] > 0 and cal["entropy_observed_neg"] > 0
    assert cal["entropy_learnable"] is True
    # And the floor itself is the mechanism's constant, not a copy.
    from ghost_agent.core.calibration import _MIN_ENTROPY_SAMPLES
    assert cal["entropy_min_samples_gate"] == _MIN_ENTROPY_SAMPLES


def test_entropy_not_learnable_when_it_varies_but_does_not_separate(tmp_path):
    """The half of the gate this report used to omit. Plenty of samples,
    both classes, 30 distinct values — and no ability to tell the classes
    apart, so the fit pins w_entropy to 0. Reporting LEARNABLE here is the
    exact lie the mirror-the-gate rule exists to prevent."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    calib = md.parent / "calibration"
    calib.mkdir(parents=True)
    # Both classes drawn from an IDENTICAL spread of 40 values, so the
    # separation is exactly zero by construction. Random noise would be
    # seed-luck: a 2.5σ test admits ~5% of pure-noise corpora by design,
    # and this test must assert the mechanism, not a lucky draw.
    values = [j / 40.0 for j in range(40)]
    with (calib / "calibration.jsonl").open("w") as fh:
        for v in values:                       # 40 failures
            fh.write(_calib_row(entropy_component=v,
                                entropy_observed=True, outcome=0.0))
        for v in values * 2:                   # 80 successes, same spread
            fh.write(_calib_row(entropy_component=v,
                                entropy_observed=True, outcome=1.0))
    cal = collect_learning_health(md)["calibration"]
    assert cal["entropy_observed_samples"] == 120
    assert cal["entropy_observed_pos"] > 0 and cal["entropy_observed_neg"] > 0
    assert cal["entropy_distinct_values"] > 30      # it VARIES
    assert cal["entropy_separation_sigmas"] < cal["separation_min_sigmas"]
    assert cal["entropy_learnable"] is False        # …and still teaches nothing


def test_calibration_telemetry_is_epoch_scoped(tmp_path):
    """Counts must describe the population the fit reads. Pooling epochs is
    what made `competence_component` report separation 0.0023 / verdict
    'dead' — a number measured across a label-scheme change."""
    md = tmp_path / "memory"
    md.mkdir(parents=True)
    calib = md.parent / "calibration"
    calib.mkdir(parents=True)
    with (calib / "calibration.jsonl").open("w") as fh:
        for _ in range(60):   # legacy: undated, so derived as the old epoch
            fh.write(json.dumps({"entropy_component": 0.5, "outcome": 1.0,
                                 "ts": "2026-07-10T00:00:00Z"}) + "\n")
        for i in range(20):
            fh.write(_calib_row(entropy_component=0.4, entropy_observed=True,
                                outcome=1.0 if i % 4 else 0.0))
    cal = collect_learning_health(md)["calibration"]
    assert cal["samples_on_disk"] == 80        # the file, unchanged
    assert cal["samples_this_epoch"] == 20     # what the fit actually reads
    assert cal["samples_other_epochs"] == 60
    assert cal["entropy_observed_samples"] == 20


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


class TestVerifierEscalationHealth:
    """§4F false-positive watch metric surfaced from the durable ledger.

    Before this, the rate lived only in `VerifyResult.to_dict()` — which has
    no production caller — so reading it meant grepping the log, where a
    naive count is 9 points high (the OVERTURNED line is a WARNING mirrored
    to a second logger; "verdict stands" is INFO).
    """

    def _ledger(self, md: Path, rows):
        p = md.parent / "verifier" / "escalations.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows),
                     encoding="utf-8")
        return p

    def test_absent_ledger_names_which_silence(self, tmp_path):
        md = tmp_path / "memory"
        _seed(md)
        r = collect_learning_health(md)
        esc = r["verifier_escalation"]
        assert esc["present"] is False
        assert esc["reason"] == "no ledger file"
        out = render_learning_health(md)
        assert "no ledger file" in out
        assert "first escalation after a boot" in out

    def test_empty_ledger_is_distinct_from_absent(self, tmp_path):
        md = tmp_path / "memory"
        _seed(md)
        self._ledger(md, [])
        assert collect_learning_health(md)["verifier_escalation"]["reason"] \
            == "ledger empty"

    def test_rate_is_per_route_and_kind(self, tmp_path):
        """claim-refute and code-refute are different populations: measured
        2026-08-04, claim refutes overturn 84% while all 7 live code refutes
        were upheld on replay. Averaging them hides both."""
        md = tmp_path / "memory"
        _seed(md)
        self._ledger(md, (
            [{"route": "claim", "kind": "refute", "outcome": "overturned"}] * 42
            + [{"route": "claim", "kind": "refute", "outcome": "upheld"}] * 8
            + [{"route": "code", "kind": "refute", "outcome": "upheld"}] * 7))
        arms = collect_learning_health(md)["verifier_escalation"]["arms"]
        assert arms["claim/refute"]["overturn_rate"] == 0.84
        assert arms["code/refute"]["overturn_rate"] == 0.0
        out = render_learning_health(md)
        assert "claim/refute: 50 escalations — 84% overturned" in out

    def test_unavailable_stays_in_the_denominator_of_n(self, tmp_path):
        """A strong-model error still spent a call. It must not vanish from
        the count — but it cannot be scored as overturned or upheld either,
        so it is excluded from the RATE and reported separately."""
        md = tmp_path / "memory"
        _seed(md)
        self._ledger(md, [
            {"route": "claim", "kind": "refute", "outcome": "overturned"},
            {"route": "claim", "kind": "refute", "outcome": "upheld"},
            {"route": "claim", "kind": "refute", "outcome": "unavailable"},
        ])
        arm = collect_learning_health(md)["verifier_escalation"]["arms"]["claim/refute"]
        assert arm["n"] == 3
        assert arm["overturn_rate"] == 0.5
        assert "unavailable (call spent)" in render_learning_health(md)

    def test_withheld_confirmations_are_reported(self, tmp_path):
        md = tmp_path / "memory"
        _seed(md)
        self._ledger(md, [
            {"route": "claim", "kind": "confirm", "outcome": "withheld"},
            {"route": "claim", "kind": "confirm", "outcome": "upheld"},
        ])
        out = render_learning_health(md)
        assert "claim/confirm" in out
        assert "1 withheld" in out

    def test_malformed_rows_do_not_break_the_report(self, tmp_path):
        md = tmp_path / "memory"
        _seed(md)
        p = md.parent / "verifier" / "escalations.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"route": "claim", "kind": "refute", '
                     '"outcome": "overturned"}\nnot json at all\n{}\n',
                     encoding="utf-8")
        arms = collect_learning_health(md)["verifier_escalation"]["arms"]
        assert arms["claim/refute"]["overturned"] == 1
        assert "?/?" in arms  # the bare {} row is counted, not dropped
        render_learning_health(md)
