"""Queue #10 — the stale-constraint reconciler.

Since 2026-08-01 every DONE transition retires the project's constraints (the
fix for a constraint that replayed into every request for four days after the
work closed, driving verifier refutes whose follow-ups reopened the project).
Retirement fires ON THE TRANSITION, so a project whose last DONE predates the
fix still carries them armed — live: WebOS, DONE, 7 active, last rolled up to
DONE on 2026-07-31, one day before the fix.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
from pathlib import Path

import pytest

from ghost_agent.memory.projects import ProjectStore


def _load():
    path = (Path(__file__).resolve().parent.parent / "scripts"
            / "retire_stale_constraints.py")
    spec = importlib.util.spec_from_file_location("retire_stale", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()


def _home(tmp_path):
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sandbox").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _store(tmp_path):
    return ProjectStore(tmp_path / "system" / "memory",
                        sandbox_root=tmp_path / "sandbox")


def _run(tmp_path, *argv):
    old = sys.argv
    sys.argv = ["retire_stale_constraints.py", "--home", str(tmp_path)] + list(argv)
    try:
        return R.main()
    finally:
        sys.argv = old


class TestDetection:
    def test_it_reads_the_PARSED_metadata_the_store_returns(self, tmp_path):
        """⚠ The bug this script shipped with. `list_projects` returns
        `metadata` already parsed; the first version read `metadata_json`
        (the SQL column name), found nothing, and printed "nothing to
        retire" against a store with seven live examples. A reconciler that
        silently finds nothing looks exactly like a clean store."""
        _home(tmp_path)
        st = _store(tmp_path)
        pid = st.create_project(
            title="P", kind="CODING", goal="g",
            metadata={"constraints": ["dark mode", "have a browser"]})
        rows = [p for p in st.list_projects() if p["id"] == pid]

        assert rows, "precondition"
        assert R._constraints(st, rows[0]) == ["dark mode", "have a browser"]

    def test_a_project_with_no_constraints_is_not_a_target(self, tmp_path):
        _home(tmp_path)
        st = _store(tmp_path)
        pid = st.create_project(title="P", kind="CODING", goal="g")
        rows = [p for p in st.list_projects() if p["id"] == pid]

        assert R._constraints(st, rows[0]) == []

    def test_a_string_metadata_blob_is_still_parsed(self, tmp_path):
        assert R._constraints(None, {"metadata": '{"constraints": ["x"]}'}) == ["x"]

    def test_a_broken_metadata_blob_yields_nothing_rather_than_raising(self):
        assert R._constraints(None, {"metadata": "{not json"}) == []


class TestDryRunAndApply:
    def _stale(self, tmp_path):
        _home(tmp_path)
        st = _store(tmp_path)
        pid = st.create_project(
            title="WebOS", kind="CODING", goal="g",
            metadata={"constraints": ["dark mode", "have a browser"]})
        # Force the pre-fix shape: terminal WITH constraints still active.
        with st._lock, st._connect() as conn:
            conn.execute("UPDATE projects SET status='DONE' WHERE id=?", (pid,))
        return st, pid

    def test_dry_run_changes_nothing(self, tmp_path, capsys):
        st, pid = self._stale(tmp_path)

        assert _run(tmp_path) == 0
        out = capsys.readouterr().out

        assert "DRY RUN" in out and "WebOS" in out
        row = [p for p in st.list_projects() if p["id"] == pid][0]
        assert R._constraints(st, row) == ["dark mode", "have a browser"]

    def test_apply_retires_through_the_stores_own_path(self, tmp_path, capsys):
        """The audit trail must be written by the same code the live DONE
        transition uses — not re-implemented here."""
        st, pid = self._stale(tmp_path)

        assert _run(tmp_path, "--apply") == 0

        row = [p for p in st.list_projects() if p["id"] == pid][0]
        assert R._constraints(st, row) == []
        meta = row.get("metadata") or {}
        assert set(meta.get("constraints_retired") or []) == {
            "dark mode", "have a browser"}

    def test_a_clean_store_says_so_and_exits_zero(self, tmp_path, capsys):
        _home(tmp_path)
        st = _store(tmp_path)
        st.create_project(title="P", kind="CODING", goal="g")

        assert _run(tmp_path) == 0
        assert "nothing to retire" in capsys.readouterr().out

    def test_an_ACTIVE_project_is_never_touched(self, tmp_path):
        """Constraints on live work are the whole point of constraints."""
        _home(tmp_path)
        st = _store(tmp_path)
        pid = st.create_project(title="Live", kind="CODING", goal="g",
                                metadata={"constraints": ["dark mode"]})

        assert _run(tmp_path, "--apply") == 0

        row = [p for p in st.list_projects() if p["id"] == pid][0]
        assert R._constraints(st, row) == ["dark mode"]


class TestGuards:
    def test_missing_home_is_an_error(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        old = sys.argv
        sys.argv = ["retire_stale_constraints.py"]
        try:
            assert R.main() == 2
        finally:
            sys.argv = old
