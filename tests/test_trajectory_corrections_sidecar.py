"""Tests for the TrajectoryCollector corrections sidecar.

The sidecar is what makes outcome promotion (e.g. user-correction
detection) durable: instead of rewriting the original JSONL line,
``update_outcome`` appends a record to ``corrections.jsonl`` and
``iter_trajectories`` overlays it on read. These tests pin:

  * append-only semantics (original line untouched)
  * last-write-wins on repeat updates for the same id
  * graceful no-op when the sidecar is missing
  * disabled collector refuses writes
  * malformed lines in the sidecar don't poison the overlay
"""

from __future__ import annotations

import json
from pathlib import Path

from ghost_agent.distill.collector import (
    TrajectoryCollector,
    CORRECTIONS_FILENAME,
)
from ghost_agent.distill.schema import Trajectory, Outcome


def _t(traj_id: str, **overrides) -> Trajectory:
    base = dict(
        id=traj_id,
        user_request="ask",
        final_response="answer",
        outcome=Outcome.UNKNOWN.value,
    )
    base.update(overrides)
    return Trajectory(**base)


# ----------------------------------------------------- sidecar file shape


def test_update_outcome_creates_sidecar_under_root(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1"))
    assert c.update_outcome("t1", Outcome.FAILED.value, "user-correction")
    sidecar = tmp_path / CORRECTIONS_FILENAME
    assert sidecar.exists()
    line = sidecar.read_text().strip()
    rec = json.loads(line)
    assert rec["trajectory_id"] == "t1"
    assert rec["outcome"] == Outcome.FAILED.value
    assert rec["reason"] == "user-correction"
    assert "timestamp" in rec


def test_update_outcome_records_optional_source_label(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1"))
    c.update_outcome(
        "t1", Outcome.FAILED.value, "abort", source="user_correction"
    )
    sidecar = tmp_path / CORRECTIONS_FILENAME
    rec = json.loads(sidecar.read_text().strip())
    assert rec["source"] == "user_correction"


# --------------------------------------------- original JSONL is untouched


def test_original_trajectory_jsonl_line_is_not_rewritten(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    p = c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    original = p.read_text()
    c.update_outcome("t1", Outcome.FAILED.value, "user-correction")
    after = p.read_text()
    # Audit trail must stay byte-identical — only the sidecar grows.
    assert after == original
    assert '"outcome": "unknown"' in after  # the on-disk write was UNKNOWN


# --------------------------------------------- iter_trajectories overlay


def test_iter_trajectories_overlays_corrected_outcome(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    c.update_outcome("t1", Outcome.FAILED.value, "selectors thrashed")

    c2 = TrajectoryCollector(root=tmp_path, session_id="s2")
    found = list(c2.iter_trajectories())
    assert len(found) == 1
    assert found[0].id == "t1"
    assert found[0].outcome == Outcome.FAILED.value
    assert found[0].failure_reason == "selectors thrashed"


def test_iter_trajectories_does_not_overwrite_pre_existing_failure_reason(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1",
                outcome=Outcome.UNKNOWN.value,
                failure_reason="original on-disk reason"))
    c.update_outcome("t1", Outcome.FAILED.value, "sidecar reason")

    c2 = TrajectoryCollector(root=tmp_path, session_id="s2")
    found = list(c2.iter_trajectories())
    assert len(found) == 1
    # Sidecar promotes outcome, but does not clobber pre-existing reason.
    assert found[0].outcome == Outcome.FAILED.value
    assert found[0].failure_reason == "original on-disk reason"


def test_iter_trajectories_passes_through_uncorrected(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.PASSED.value))
    found = list(TrajectoryCollector(root=tmp_path, session_id="x").iter_trajectories())
    assert len(found) == 1
    assert found[0].outcome == Outcome.PASSED.value


# ------------------------------------------------- last-write-wins semantics


def test_repeat_update_for_same_id_uses_latest(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    c.update_outcome("t1", Outcome.FAILED.value, "first try")
    c.update_outcome("t1", Outcome.PASSED.value, "second try")

    # Sidecar contains both lines (append-only audit trail)...
    sidecar = tmp_path / CORRECTIONS_FILENAME
    lines = [l for l in sidecar.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    # ...but the overlay yields the latest verdict.
    found = list(TrajectoryCollector(root=tmp_path, session_id="x").iter_trajectories())
    assert len(found) == 1
    assert found[0].outcome == Outcome.PASSED.value


# ----------------------------------------------- defensive edge cases


def test_update_outcome_returns_false_when_disabled(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1", enabled=False)
    assert not c.update_outcome("t1", Outcome.FAILED.value, "x")
    assert not (tmp_path / CORRECTIONS_FILENAME).exists()


def test_update_outcome_rejects_empty_id(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    assert not c.update_outcome("", Outcome.FAILED.value, "x")
    assert not c.update_outcome(None, Outcome.FAILED.value, "x")  # type: ignore[arg-type]


def test_iter_skips_malformed_sidecar_lines(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    sidecar = tmp_path / CORRECTIONS_FILENAME
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "{not json\n"
        '{"trajectory_id": "", "outcome": "failed"}\n'  # missing id
        '{"trajectory_id": "t1", "outcome": "failed", "reason": "ok"}\n'
        "\n"
    )
    found = list(TrajectoryCollector(root=tmp_path, session_id="x").iter_trajectories())
    assert len(found) == 1
    assert found[0].outcome == Outcome.FAILED.value


def test_iter_when_sidecar_missing_passes_through(tmp_path):
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    # Make sure no sidecar exists.
    assert not (tmp_path / CORRECTIONS_FILENAME).exists()
    found = list(TrajectoryCollector(root=tmp_path, session_id="x").iter_trajectories())
    assert len(found) == 1
    assert found[0].outcome == Outcome.UNKNOWN.value


def test_correction_for_unknown_id_is_silently_ignored_on_read(tmp_path):
    """A correction for a trajectory id that doesn't exist on disk
    is harmless — overlay just doesn't apply. Useful for late-
    arriving corrections that race with file rotation."""
    c = TrajectoryCollector(root=tmp_path, session_id="s1")
    c.append(_t("t1", outcome=Outcome.UNKNOWN.value))
    c.update_outcome("orphan-id", Outcome.FAILED.value, "ghost")
    found = list(TrajectoryCollector(root=tmp_path, session_id="x").iter_trajectories())
    assert len(found) == 1
    assert found[0].id == "t1"
    assert found[0].outcome == Outcome.UNKNOWN.value
