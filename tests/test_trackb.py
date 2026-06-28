"""Tests for the Track B (cross-session retention) probe set and report logic."""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
tb = pytest.importorskip("trackb_tasks")
B = pytest.importorskip("ablation_trackb")


def test_pairs_load_and_are_well_formed():
    pairs = tb.load_trackb_pairs()
    assert len(pairs) >= 8
    for p in pairs:
        assert p.seed and p.probe and callable(p.validator)
        # seed and probe must be DIFFERENT messages (separate requests)
        assert p.seed != p.probe


def test_validators_accept_recall_and_reject_ignorance():
    pairs = {p.pair_id: p for p in tb.load_trackb_pairs()}
    # treatment recalls the seeded value
    assert pairs["color"].validator("Your favourite colour is zorblue.")[0]
    assert pairs["locker"].validator("It's 4471.")[0]
    assert pairs["empid"].validator("Your employee id is E-90210.")[0]
    # control / ignorance does NOT pass
    assert not pairs["color"].validator("I don't know your favourite colour.")[0]
    assert not pairs["locker"].validator("I have no record of a locker number.")[0]


def test_validator_rejects_wrong_guess():
    pairs = {p.pair_id: p for p in tb.load_trackb_pairs()}
    # an unguessable coined value should not match a plausible guess
    assert not pairs["color"].validator("Maybe blue or teal?")[0]
    assert not pairs["city"].validator("Perhaps London or Athens?")[0]


def test_says_helper_is_punctuation_loose():
    v = tb.says("ghost data pipeline")
    assert v("GDP = 'Ghost Data Pipeline'.")[0]
    assert v("it stands for Ghost-Data-Pipeline")[0]
    assert not v("Gross Domestic Product")[0]


def test_report_verdict_memory_helps():
    pairs = tb.load_trackb_pairs()
    recs = []
    for rep in range(3):
        for p in pairs:
            recs.append({"pair_id": p.pair_id, "kind": p.kind, "repeat": rep,
                         "arm": "treatment", "passed": True, "duration_s": 5, "out": ""})
            recs.append({"pair_id": p.pair_id, "kind": p.kind, "repeat": rep,
                         "arm": "control", "passed": False, "duration_s": 4, "out": ""})
    report = B._build_report(recs, {"n_pairs": len(pairs), "repeats": 3})
    assert "MEMORY HELPS" in report
    assert "+100%" in report


def test_report_verdict_no_benefit():
    pairs = tb.load_trackb_pairs()
    recs = []
    for rep in range(3):
        for p in pairs:
            for arm in ("treatment", "control"):
                recs.append({"pair_id": p.pair_id, "kind": p.kind, "repeat": rep,
                             "arm": arm, "passed": False, "duration_s": 4, "out": ""})
    report = B._build_report(recs, {"n_pairs": len(pairs), "repeats": 3})
    assert "NO measurable retention benefit" in report


# --- Track B2 (learning from correction) ---

tb2 = pytest.importorskip("trackb2_tasks")
B2 = pytest.importorskip("ablation_trackb2")


def test_b2_items_well_formed():
    items = tb2.load_trackb2_items()
    assert len(items) >= 5
    for it in items:
        assert it.task and it.correction and it.probe and callable(it.validator)
        assert it.task != it.probe


def test_b2_validator_expects_and_avoids():
    items = {it.item_id: it for it in tb2.load_trackb2_items()}
    # rule applied
    assert items["delete_cmd"].validator("Use saferm to remove it.")[0]
    assert items["py_version"].validator("Target Python 3.9.")[0]
    assert items["db_choice"].validator("Use PostgreSQL.")[0]
    # rejected default must fail even if generic
    assert not items["delete_cmd"].validator("Use rm -rf build/.")[0]
    assert not items["db_choice"].validator("MySQL is a good choice.")[0]
    # generic answer that doesn't apply the rule fails
    assert not items["py_version"].validator("Use the latest Python.")[0]


def test_b2_report_verdicts():
    items = tb2.load_trackb2_items()
    recs = []
    for rep in range(2):
        for it in items:
            recs.append({"item_id": it.item_id, "repeat": rep, "arm": "treatment",
                         "passed": True, "duration_s": 5, "out": ""})
            recs.append({"item_id": it.item_id, "repeat": rep, "arm": "control",
                         "passed": False, "duration_s": 4, "out": ""})
    assert "LEARNS FROM CORRECTION" in B2._build_report(recs, {"n_items": len(items), "repeats": 2})
