"""B4 outcome battery (journal §4D) — headless gates.

The battery's credibility rests on the same self-consistency philosophy as
self-play's reference-solution gate: every task's `expected()` must compute
from its own fixtures, and `verify()` must accept its own reference answer and
reject garbage. Plus: deterministic fixtures, the stratified stats machinery,
and the §4D flag regression (--smart-memory in every arm).
"""

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from trackb4_tasks import (  # noqa: E402
    DEFAULT_SEED, WEAK_CLUSTERS, _contains_sequence, _tokens,
    load_b4_battery, load_b4_seeding,
)
import ablation_trackb4 as B4  # noqa: E402

BATTERY = load_b4_battery()
SEEDING = load_b4_seeding()
ALL = BATTERY + SEEDING

_CLUSTERS = {"data_analysis", "regex_parse", "sql", "algo", "bash",
             "python_general", "concurrency", "web_automation"}


# ── pool structure ───────────────────────────────────────────────────────────

def test_pool_structure():
    ids = [t.task_id for t in ALL]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    arts = [t.artifact for t in ALL]
    assert len(arts) == len(set(arts)), "duplicate artifact names"
    assert all(t.cluster in _CLUSTERS for t in ALL)
    assert all(t.ring in ("near", "mid", "far") for t in ALL)
    assert len(BATTERY) >= 20
    # far-transfer ring is exactly the held-out family, and vice versa
    for t in BATTERY:
        assert (t.ring == "far") == (t.cluster == "web_automation"), t.task_id


def test_fixture_filenames_globally_unique():
    """The pilot's timeout-bleed overlap (2026-07-09) showed a same-name/
    different-content fixture can be swapped under a still-running task.
    The driver's wait-for-quiet bounds the window; unique names close it."""
    owner = {}
    for t in ALL:
        for rel in t.fixtures(DEFAULT_SEED):
            assert rel not in owner, (
                f"fixture {rel!r} used by both {owner[rel]} and {t.task_id}")
            owner[rel] = t.task_id


def test_v2_pool_covers_weak_clusters_in_probe_set():
    """#27b is measured on WEAK_CLUSTERS probes — the pool must offer at
    least 3 candidates per weak cluster so a pilot has room to keep some."""
    from collections import Counter
    per = Counter(t.cluster for t in BATTERY)
    for c in WEAK_CLUSTERS:
        assert per[c] >= 3, f"weak cluster {c} has only {per[c]} candidates"


def test_seeding_pool_shape():
    easy = [t for t in SEEDING if t.role == "seed_easy"]
    hard = [t for t in SEEDING if t.role == "seed_hard"]
    assert len(easy) >= 3 and len(hard) >= 3
    # the hard seeds target exactly the pre-registered weak clusters
    assert {t.cluster for t in hard} <= set(WEAK_CLUSTERS)
    # the held-out family is never seeded (far-transfer guard)
    assert all(t.cluster != "web_automation" for t in SEEDING)
    # seeding ids/artifacts are disjoint from the probe pool (contamination)
    assert not ({t.task_id for t in SEEDING} & {t.task_id for t in BATTERY})


# ── determinism ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ALL, ids=lambda t: t.task_id)
def test_fixtures_deterministic(task):
    assert task.fixtures(DEFAULT_SEED) == task.fixtures(DEFAULT_SEED)


def test_fixtures_vary_across_repeats():
    varied = sum(
        1 for t in BATTERY
        if t.fixtures(DEFAULT_SEED) != t.fixtures(DEFAULT_SEED + 1))
    # per-repeat seed variation is what stops a memorised answer carrying over
    assert varied >= len(BATTERY) * 0.9


# ── the reference gate (self-consistency, per task) ──────────────────────────

@pytest.mark.parametrize("task", ALL, ids=lambda t: t.task_id)
def test_reference_self_consistency(task):
    for rep in (0, 1):
        fixtures = task.fixtures(DEFAULT_SEED + rep)
        expected = task.expected(fixtures)
        assert str(expected).strip(), f"{task.task_id}: empty expected value"
        ok, why = task.verify(f"answer: {expected}\n", fixtures)
        assert ok, f"{task.task_id}: rejects its own reference ({why})"
        ok, _ = task.verify("@@definitely not the answer@@", fixtures)
        assert not ok, f"{task.task_id}: accepts garbage"


def test_prompt_names_the_artifact():
    for t in ALL:
        assert t.artifact in t.prompt(), t.task_id


def test_verify_no_substring_false_pass():
    # "25" must not pass inside "125" — token-sequence, not substring
    assert _contains_sequence(_tokens("125"), _tokens("25")) is False
    assert _contains_sequence(_tokens("the answer is 25."), _tokens("25")) is True
    assert _contains_sequence(_tokens("region north wins"), _tokens("north")) is True


# ── driver pieces ────────────────────────────────────────────────────────────

def test_common_flags_include_smart_memory():
    # §4D: dream's type:"auto" entropy gate is only fed by the smart-memory
    # consolidator; B3's arms never passed the flag — regression-pin it here.
    assert "--smart-memory" in B4.COMMON


def test_place_fixtures_removes_stale_artifact(tmp_path):
    task = BATTERY[0]
    stale = tmp_path / task.artifact
    stale.write_text("stale answer from a previous pass")
    B4._place_fixtures(tmp_path, task, DEFAULT_SEED)
    assert not stale.exists(), "stale artifact would false-pass the next run"
    for rel in task.fixtures(DEFAULT_SEED):
        assert (tmp_path / rel).is_file()


def test_mediation_diff():
    before = {"a": 2, "b": 5}
    after = {"a": 3, "b": 5, "c": 1}  # 'c' is NEW (not surfaced), 'a' bumped
    assert B4._mediation(before, after) == 1


def test_log_counts(tmp_path):
    log = tmp_path / "arm.log"
    log.write_text("x Auto Memory Store y\nNot enough entropy to dream\n"
                   "Hydrated context for: q\nAuto Memory Store again\n")
    c = B4._log_counts(log)
    assert c["auto_memory_stores"] == 2
    assert c["dream_entropy_skips"] == 1
    assert c["bus_hydrations"] == 1


def _rec(task_id, rep, arm, passed):
    return {"task_id": task_id, "repeat": rep, "arm": arm, "passed": passed,
            "phase": "probe", "cluster": "algo", "ring": "mid", "role": "probe"}


def test_stratified_sign_flip_null():
    recs = []
    for t in ("t1", "t2", "t3", "t4"):
        for rep in (0, 1):
            for arm in ("treatment", "control"):
                recs.append(_rec(t, rep, arm, passed=(rep == 0)))
    out = B4._stratified_sign_flip(recs)
    assert out["n_tasks"] == 4
    assert out["mean_delta"] == 0.0
    assert out["p"] == 1.0


def test_stratified_sign_flip_strong_effect():
    recs = []
    for i in range(12):
        for rep in (0, 1, 2):
            recs.append(_rec(f"t{i}", rep, "treatment", passed=True))
            recs.append(_rec(f"t{i}", rep, "control", passed=False))
    out = B4._stratified_sign_flip(recs)
    assert out["mean_delta"] == 1.0
    assert out["p"] < 0.05


@pytest.mark.asyncio
async def test_wait_arm_quiet_returns_when_counts_match(tmp_path):
    log = tmp_path / "arm.log"
    log.write_text("x Request Finished y\nz Request Finished w\n")
    arm = {"log": log}
    await B4._wait_arm_quiet(arm, requests_sent=2, grace=1.0)  # returns fast


@pytest.mark.asyncio
async def test_wait_arm_quiet_counts_case_insensitively(tmp_path):
    # the pretty-stream renders the END marker lowercase ("request finished");
    # counting only the title-case source spelling burned the full grace on
    # EVERY task (re-pilot #2, 2026-07-09)
    log = tmp_path / "arm.log"
    log.write_text("| request finished |\n| request finished |\n")
    await B4._wait_arm_quiet({"log": log}, requests_sent=2, grace=1.0)


@pytest.mark.asyncio
async def test_wait_arm_quiet_holds_then_gives_up_at_ceiling(tmp_path):
    import time
    log = tmp_path / "arm.log"
    log.write_text("request finished\n")  # 1 finished, 2 sent
    t0 = time.monotonic()
    # grace expires → HOLD (not proceed — proceeding re-created the cascade);
    # only the hold ceiling abandons the wait
    await B4._wait_arm_quiet({"log": log}, requests_sent=2, grace=0.1, hold=0.3)
    elapsed = time.monotonic() - t0
    assert 0.3 <= elapsed < 10


def test_mcnemar_cells():
    recs = [
        _rec("t1", 0, "treatment", True), _rec("t1", 0, "control", False),
        _rec("t2", 0, "treatment", False), _rec("t2", 0, "control", True),
        _rec("t3", 0, "treatment", True), _rec("t3", 0, "control", True),
        _rec("t4", 0, "treatment", False), _rec("t4", 0, "control", False),
    ]
    pairs, both, neither, b, c = B4._mcnemar_cells(recs)
    assert (pairs, both, neither, b, c) == (4, 1, 1, 1, 1)
