"""WorkspaceActivity log: append, iter, recent, filter-by-kind, kind tallies,
corruption tolerance."""

from pathlib import Path

from ghost_agent.workspace.activity import WorkspaceActivity
from ghost_agent.workspace.schema import WorkspaceEvent


def test_append_and_iter(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="research", summary="pulled X"))
    act.append(WorkspaceEvent(kind="task_outcome", summary="cron OK"))
    events = list(act.iter_events())
    assert len(events) == 2
    assert {e.kind for e in events} == {"research", "task_outcome"}


def test_recent_returns_tail_newest_last(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    for i in range(5):
        act.append(WorkspaceEvent(kind="note", summary=f"n{i}"))
    tail = act.recent(limit=3)
    assert [e.summary for e in tail] == ["n2", "n3", "n4"]


def test_recent_filter_by_kind(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="research", summary="r1"))
    act.append(WorkspaceEvent(kind="task_outcome", summary="t1"))
    act.append(WorkspaceEvent(kind="research", summary="r2"))
    tasks = act.recent(limit=10, kind="task_outcome")
    assert len(tasks) == 1
    assert tasks[0].summary == "t1"


def test_append_with_empty_kind_refused(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="", summary="bad"))
    assert act.count() == 0


def test_disabled_activity_is_noop(tmp_path: Path):
    act = WorkspaceActivity(tmp_path, enabled=False)
    act.append(WorkspaceEvent(kind="note", summary="x"))
    assert act.count() == 0
    assert not (tmp_path / "activity.jsonl").exists()


def test_corrupt_line_skipped_not_raised(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="note", summary="ok"))
    # Inject a malformed line.
    with (tmp_path / "activity.jsonl").open("a", encoding="utf-8") as f:
        f.write("{ this is not json\n")
    act.append(WorkspaceEvent(kind="note", summary="after"))
    summaries = [e.summary for e in act.iter_events()]
    assert summaries == ["ok", "after"]


def test_kinds_tally(tmp_path: Path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="research", summary="r"))
    act.append(WorkspaceEvent(kind="research", summary="r2"))
    act.append(WorkspaceEvent(kind="task_outcome", summary="t"))
    tally = act.kinds()
    assert tally == {"research": 2, "task_outcome": 1}
