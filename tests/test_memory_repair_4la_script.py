"""The §4LA episode cleanup the operator confirmed ("yes to all, proceed"):
listed ids only; a moved store applies nothing; backups pruned to the newest
two folders and never a live file."""
import os
import runpy
import time
from pathlib import Path

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "memory_repair_4la.py"), run_name="not_main")


def _ids():
    return set(M["RELABEL"]) | {i for g in M["DELETE"].values() for i in g}


def test_a_deleted_episode_is_not_also_relabelled():
    relabel, dele = M["check"](_ids())
    assert not set(relabel) & set(dele) and len(dele) == sum(len(g) for g in M["DELETE"].values())


def test_a_missing_id_applies_nothing():
    with pytest.raises(SystemExit, match="nothing applied"):
        M["check"](_ids() - {M["DELETE"]["sensitive"][0]})


def test_the_ambiguous_owner_requests_are_kept():
    dele = {i for g in M["DELETE"].values() for i in g}
    assert not {211, 212, 264} & dele


def test_backups_keep_the_newest_two_folders_and_no_live_file(tmp_path):
    system, mem = tmp_path, tmp_path / "memory"
    mem.mkdir()
    for n in ("episodic_memory.db", "user_profile.json", "skills_playbook.json", "graph_pruned_archive.jsonl"):
        (mem / n).write_text("x")
    for n in ("episodic_memory.db.pre-f8.bak", "user_profile.json.pre-temporal-20260904T070844",
              "auto_skills.json.bak-4fk-20260908-003615"):
        (mem / n).write_text("x")
    for i, n in enumerate(("memory.pre-a.bak", "memory.pre-b.bak", "memory.pre-c.bak")):
        d = system / n
        d.mkdir()
        os.utime(d, (time.time() - 100 + i, time.time() - 100 + i))
    gone = {p.name for p in M["backups_to_delete"](system, mem)}
    assert gone == {"memory.pre-a.bak", "episodic_memory.db.pre-f8.bak",
                    "user_profile.json.pre-temporal-20260904T070844", "auto_skills.json.bak-4fk-20260908-003615"}



def test_the_backup_just_made_is_never_pruned(tmp_path):
    """Fails where copytree's copied mtime made the fresh backup look oldest."""
    mem = tmp_path / "memory"
    mem.mkdir()
    old = tmp_path / "memory.pre-4ky-20261003T164929.bak"
    newer = tmp_path / "memory.pre-4kz-20261003T183121.bak"
    mine = tmp_path / "memory.pre-4la-20261003T200000.bak"
    for d in (old, newer, mine):
        d.mkdir()
    os.utime(mine, (1, 1))                      # looks ancient by mtime
    gone = {p.name for p in M["backups_to_delete"](tmp_path, mem, keep=mine)}
    assert gone == {old.name}
