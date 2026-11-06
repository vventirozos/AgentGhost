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
    # §4TD: an old playbook on disk is not read — the report must not render it.
    (md / "skills_playbook.json").write_text(json.dumps([{"trigger": "a", "retrievals": 6}]))
    # Competence + calibration: §4PJ removed both, but a live install still
    # has these files on disk — the report must ignore them, not render them.
    comp = {
        "shell|*": {"alpha": 90, "beta": 10, "n": 100},
        "sql|*": {"alpha": 3, "beta": 1, "n": 4},
        "*|*": {"alpha": 1, "beta": 1, "n": 104},
    }
    (md / "competence_profile.json").write_text(json.dumps(comp))
    (md / "adaptive_threshold.json").write_text(
        json.dumps({"threshold": 0.6, "window": [1, 2, 3]}))
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
    assert "lessons" not in r
    assert r["episodes"]["total"] == 2
    assert r["episodes"]["with_context"] == 1
    # §4PJ: leftover competence/calibration files are NOT reported — a
    # section over a store nothing writes would read as a live mechanism
    assert "competence" not in r and "calibration" not in r


def test_render_is_a_string(tmp_path):
    md = tmp_path / "memory"
    _seed(md)
    out = render_learning_health(md)
    assert "LEARNING HEALTH" in out and "LESSONS:" not in out
    assert "COMPETENCE" not in out and "CALIBRATION" not in out   # §4PJ


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
    assert "selfhood" in cw
    assert "calibration" not in cw and "metacog_arbiter" not in cw   # §4PJ
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


# ── the lesson hit-rate's contaminated denominator ──────────────────────────
#
# (§4BM R1/R2/R3's PRM wiring-row pins — the `.uncertainty()` flag read,
# `_flag_state`, the --frontier-selfplay / --prm-online-update parser types —
# went with ghost_agent/prm and the PRM section of learning_health, §4NM.)


@pytest.mark.asyncio
async def test_introspect_learning_renders_the_wiring_rows_that_remain(tmp_path):
    """§4BM R6 CRIT-2 — BEHAVIOURAL, not a source string: the wiring rows
    must REACH the operator surface. (The PRM rows went with
    ghost_agent/prm, §4NM; the metacog-arbiter and calibration rows went
    with the confidence score, §4PJ — and must not come back as rows
    describing a mechanism that does not exist.)"""
    import types
    from ghost_agent.tools.introspect import tool_introspect
    md = tmp_path / "memory"
    md.mkdir(parents=True, exist_ok=True)
    context = types.SimpleNamespace(
        memory_dir=md, self_model=None, args=types.SimpleNamespace())
    # verbose=True: the wiring rows are indented detail, which the
    # 2026-09-05 brief (the tool's default view) folds away by design.
    out = await tool_introspect(action="learning", context=context,
                                verbose=True)
    assert "selfhood: writes every turn" in out, out[:400]
    assert "metacog arbiter" not in out and "calibration:" not in out


def test_sibling_wiring_rows_are_tri_state_too():
    """§4BM R3 MAJ-3 (second unpinned half): reverting the SIBLING rows
    to bare truthiness broke zero tests, so the selfhood prefix / metacog
    arbiter rows would print OFF for a None that means 'gate not read'."""
    from ghost_agent.core.learning_health import _consumer_state
    assert _consumer_state(None) == "unknown"
    assert _consumer_state(None, "unknown (gate not read)") == \
        "unknown (gate not read)"
    assert _consumer_state(True) == "ON"
    assert _consumer_state(False) == "OFF"
    import inspect as _inspect
    from ghost_agent.core import learning_health as _lh
    src = _inspect.getsource(_lh.render_learning_health)
    # the remaining row (selfhood prefix) must route through the
    # tri-state renderer; the PRM rows left in §4NM, the arbiter's in §4PJ
    assert src.count("_consumer_state(") >= 1, \
        "a wiring row bypasses the tri-state renderer"


# ── Introspect fresh-eye review (2026-08-17) ───────────────────────────────

def test_activity_counts_SKIPS_unparseable_timestamps(tmp_path):
    """m4: `if ts and ts < cutoff` kept a garbage-ts row in EVERY window —
    one such row would keep a dead PERIODIC phase looking alive in the
    liveness table forever, and that table's job is telling benign zeros
    from fatal ones."""
    import json as _json
    from ghost_agent.core.learning_health import _activity_counts
    ledger = tmp_path / "autonomous_activity.jsonl"
    rows = [
        {"ts": "garbage", "phase": "dream"},
        {"ts": None, "phase": "dream"},
        {"phase": "dream"},                       # absent ts
        {"ts": 0, "phase": "dream"},
    ]
    ledger.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    counts = _activity_counts(ledger, window_hours=24.0)
    assert counts.get("dream", 0) == 0, (
        f"rows with no usable timestamp counted as recent activity: {counts}")
