"""A quarantined lesson must never absorb a later correct re-learn.

Before the fix, `_find_duplicate_lesson` had no quarantine awareness: a
re-learn matched the quarantined row, returned "reinforced", bumped ITS
frequency — and `_filter_quarantined` then kept that row out of every prompt.
The new correct lesson was written nowhere and retrievable nowhere, so the
topic became permanently un-learnable, while each bump raised the quarantined
row's utility and protected it from `prune_low_utility`.
"""
from __future__ import annotations

import pytest

from ghost_agent.memory.skills import SkillMemory


@pytest.fixture
def mem(tmp_path):
    return SkillMemory(tmp_path)


def test_relearning_a_quarantined_lesson_creates_a_live_entry(mem):
    mem.learn_lesson(task="deploy the web app", mistake="used port 80",
                     solution="ask the supervisor for a port")
    assert mem.quarantine_lesson("deploy the web app") >= 1

    # The corrected lesson, learned again from a real failure.
    mem.learn_lesson(task="deploy the web app", mistake="hardcoded the port",
                     solution="call manage_services and use $PORT")

    ctx = mem.get_playbook_context("deploy the web app")
    assert "manage_services" in ctx, (
        "the re-learned lesson must be retrievable; it was swallowed by the "
        "quarantined row")


def test_the_quarantined_row_is_still_suppressed(mem):
    mem.learn_lesson(task="deploy the web app", mistake="used port 80",
                     solution="POISONED ADVICE")
    mem.quarantine_lesson("deploy the web app")
    mem.learn_lesson(task="deploy the web app", mistake="hardcoded the port",
                     solution="call manage_services and use $PORT")
    ctx = mem.get_playbook_context("deploy the web app")
    assert "POISONED ADVICE" not in ctx


def test_quarantined_row_does_not_absorb_the_frequency_bump(mem):
    mem.learn_lesson(task="deploy the web app", mistake="a", solution="b")
    mem.quarantine_lesson("deploy the web app")
    mem.learn_lesson(task="deploy the web app", mistake="c", solution="d")
    playbook = mem._load_playbook()
    quarantined = [p for p in playbook if p.get("quarantined")]
    assert quarantined, "the quarantined row should still exist for review"
    assert all(int(p.get("frequency") or 1) == 1 for p in quarantined), (
        "a quarantined row must not accumulate frequency from re-learns")


def _prunable(tmp_path, n=14):
    mem = SkillMemory(tmp_path)
    for i in range(n):
        mem.learn_lesson(task=f"task {i}", mistake=f"m{i}", solution=f"s{i}")
    pb = mem._load_playbook()
    for row in pb:
        row["retrievals"] = 12
        row["helpful_retrievals"] = 0
        row["verified"] = False
    mem.save_playbook(pb)
    return mem


def test_prune_is_off_unless_explicitly_enabled(tmp_path, monkeypatch):
    """The prune destroyed 13 real lessons across its first three unattended
    runs, one scoring retrievals=277 succ=77 fail=32 — a 70%-success lesson
    dropped as "low utility". The cutoff is a RELATIVE bottom quartile, so it
    always finds victims however good the playbook is. Default OFF."""
    monkeypatch.delenv("GHOST_SKILL_PRUNE", raising=False)
    mem = _prunable(tmp_path)
    before = len(mem._load_playbook())
    assert mem.prune_low_utility() == 0
    assert len(mem._load_playbook()) == before, "nothing may be deleted"


def test_prune_archives_what_it_deletes(tmp_path, monkeypatch):
    """`prune_low_utility` is destructive, unattended, and takes vector twins
    with it. Its first real production run (armed by the 2026-08-04
    component-guard fix) dropped 8 lessons and recorded only a COUNT — the
    content was unrecoverable. It must leave an audit trail."""
    import json
    monkeypatch.setenv("GHOST_SKILL_PRUNE", "1")
    mem = _prunable(tmp_path)

    dropped = mem.prune_low_utility()
    assert dropped > 0, "fixture should produce prunable lessons"

    archive = tmp_path / "skills_pruned_archive.jsonl"
    assert archive.exists(), "a destructive prune must archive what it drops"
    rows = [json.loads(l) for l in archive.read_text().splitlines() if l.strip()]
    assert len(rows) == dropped
    assert all("lesson" in r and "pruned_at" in r for r in rows)
    # The archived content must be enough to reinstate a lesson by hand.
    assert any(r["lesson"].get("solution") for r in rows)


def test_prune_aborts_when_the_archive_cannot_be_written(tmp_path, monkeypatch):
    """A fix whose only purpose is recoverability must not proceed when
    recoverability is what failed. The first version warned and deleted
    anyway — probed at 7 lessons lost with the archive path unwritable."""
    monkeypatch.setenv("GHOST_SKILL_PRUNE", "1")
    mem = _prunable(tmp_path)
    before = mem._load_playbook()
    # Occupy the archive path with a directory: append raises IsADirectoryError.
    (tmp_path / "skills_pruned_archive.jsonl").mkdir()

    assert mem.prune_low_utility() == 0
    after = mem._load_playbook()
    assert len(after) == len(before), (
        "an unwritable archive must abort the prune, not delete unarchived")


def test_prune_never_deletes_a_quarantined_lesson(tmp_path, monkeypatch):
    """Quarantine holds a row on disk *for the operator to review or
    reinstate*, and excludes it from prompt injection — so its retrievals
    stop accruing and its utility decays into the prune's bottom quartile by
    construction. Probed: 4 of 5 quarantined rows deleted."""
    monkeypatch.setenv("GHOST_SKILL_PRUNE", "1")
    mem = _prunable(tmp_path)
    # Quarantine the rows the prune would ACTUALLY pick. Quarantining an
    # arbitrary five was vacuous: with the guard removed the fixture still
    # dropped three OTHER rows, so the assertion could not discriminate.
    from ghost_agent.memory.skills import (compute_lesson_utility,
                                           _normalize_lesson)
    scored = sorted(((compute_lesson_utility(_normalize_lesson(r)), r)
                     for r in mem._load_playbook()), key=lambda kv: kv[0])
    victims = [r.get("trigger") or r.get("task") for _s, r in scored[:3]]
    for t in victims:
        assert mem.quarantine_lesson(t) >= 1

    dropped = mem.prune_low_utility()

    survivors = mem._load_playbook()
    still = {p.get("trigger") or p.get("task")
             for p in survivors if p.get("quarantined")}
    assert set(victims) <= still, (
        f"the prune deleted quarantined rows it had ranked lowest: "
        f"{set(victims) - still} (dropped {dropped})")
