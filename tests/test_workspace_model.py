"""Tests for the Workspace State Model module."""

import os
import pytest
import tempfile
from pathlib import Path

from ghost_agent.core.workspace_model import WorkspaceModel, FileState, WorkspaceDiff


@pytest.fixture
def workspace_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d)


@pytest.fixture
def model(workspace_dir):
    return WorkspaceModel(sandbox_dir=workspace_dir)


class TestFileState:
    def test_to_dict(self):
        fs = FileState(path="test.py", size=100, file_type="py", summary="A test file")
        d = fs.to_dict()
        assert d["path"] == "test.py"
        assert d["size"] == 100
        assert d["summary"] == "A test file"


class TestWorkspaceDiff:
    def test_has_changes(self):
        assert WorkspaceDiff().has_changes() is False
        assert WorkspaceDiff(added=["file.txt"]).has_changes() is True
        assert WorkspaceDiff(removed=["old.txt"]).has_changes() is True
        assert WorkspaceDiff(modified=["edit.py"]).has_changes() is True

    def test_summary(self):
        diff = WorkspaceDiff(added=["new.py"], modified=["edit.py"])
        s = diff.summary()
        assert "new.py" in s
        assert "edit.py" in s

    def test_summary_no_changes(self):
        assert "No changes" in WorkspaceDiff().summary()

    def test_to_dict(self):
        diff = WorkspaceDiff(added=["a"], removed=["b"])
        d = diff.to_dict()
        assert d["added"] == ["a"]
        assert d["removed"] == ["b"]


class TestWorkspaceModel:
    def test_scan_empty_dir(self, model, workspace_dir):
        diff = model.scan()
        assert not diff.has_changes()
        assert len(model.files) == 0

    def test_scan_detects_new_files(self, model, workspace_dir):
        (workspace_dir / "test.py").write_text("print('hello')")
        (workspace_dir / "data.csv").write_text("a,b,c\n1,2,3")
        diff = model.scan()
        assert len(diff.added) == 2
        assert "test.py" in diff.added
        assert "data.csv" in diff.added
        assert len(model.files) == 2
        assert model.files["test.py"].file_type == "py"

    def test_scan_detects_modifications(self, model, workspace_dir):
        f = workspace_dir / "test.py"
        f.write_text("v1")
        model.scan()

        import time
        time.sleep(0.05)
        f.write_text("v2 with more content")
        diff = model.scan()
        assert "test.py" in diff.modified

    def test_scan_detects_removals(self, model, workspace_dir):
        f = workspace_dir / "temp.txt"
        f.write_text("temporary")
        model.scan()
        assert "temp.txt" in model.files

        f.unlink()
        diff = model.scan()
        assert "temp.txt" in diff.removed
        assert "temp.txt" not in model.files

    def test_scan_skips_hidden_files(self, model, workspace_dir):
        (workspace_dir / ".hidden").write_text("secret")
        (workspace_dir / "visible.txt").write_text("public")
        model.scan()
        assert ".hidden" not in model.files
        assert "visible.txt" in model.files

    def test_scan_handles_subdirectories(self, model, workspace_dir):
        sub = workspace_dir / "subdir"
        sub.mkdir()
        (sub / "nested.py").write_text("code")
        model.scan()
        assert os.path.join("subdir", "nested.py") in model.files

    def test_record_read(self, model, workspace_dir):
        (workspace_dir / "data.csv").write_text("a,b\n1,2")
        model.scan()
        model.record_read("data.csv", summary="2 columns, 1 row", columns=["a", "b"], row_count=1)
        assert model.files["data.csv"].summary == "2 columns, 1 row"
        assert model.files["data.csv"].columns == ["a", "b"]
        assert model.files["data.csv"].row_count == 1

    def test_record_write_creates_entry(self, model):
        model.record_write("output.json", summary="Analysis results")
        assert "output.json" in model.files
        assert model.files["output.json"].summary == "Analysis results"

    def test_what_do_i_know(self, model, workspace_dir):
        (workspace_dir / "report.csv").write_text("x")
        model.scan()
        model.current_turn = 5
        model.record_read("report.csv", summary="Sales data", columns=["date", "amount"], row_count=100)

        info = model.what_do_i_know("report.csv")
        assert "Sales data" in info
        assert "date" in info
        assert "100" in info

    def test_what_do_i_know_empty_for_unknown_file(self, model):
        assert model.what_do_i_know("nonexistent.py") == ""

    def test_what_am_i_missing(self, model, workspace_dir):
        (workspace_dir / "sales_data.csv").write_text("data")
        (workspace_dir / "config.json").write_text("{}")
        (workspace_dir / "analysis.py").write_text("code")
        model.scan()
        # Only read the config
        model.record_read("config.json", summary="config loaded")

        missing = model.what_am_i_missing("analyze sales data")
        assert "sales_data.csv" in missing
        assert "config.json" not in missing  # already read

    def test_advance_turn(self, model):
        assert model.current_turn == 0
        model.advance_turn()
        assert model.current_turn == 1

    def test_get_recently_modified(self, model, workspace_dir):
        (workspace_dir / "old.py").write_text("old")
        model.scan()
        model.current_turn = 10
        model.files["old.py"].last_read_turn = 2  # Old read

        (workspace_dir / "new.py").write_text("new")
        model.scan()
        model.record_read("new.py", summary="fresh")

        recent = model.get_recently_modified(turns_back=3)
        names = [f.path for f in recent]
        assert "new.py" in names

    def test_get_context_for_prompt(self, model, workspace_dir):
        (workspace_dir / "test.py").write_text("code")
        (workspace_dir / "data.csv").write_text("csv data")
        model.scan()
        model.record_read("test.py", summary="Test script")

        ctx = model.get_context_for_prompt()
        assert "WORKSPACE STATE" in ctx
        assert "test.py" in ctx
        assert "data.csv" in ctx

    def test_get_context_empty(self, model):
        assert model.get_context_for_prompt() == ""

    def test_to_dict(self, model, workspace_dir):
        (workspace_dir / "file.txt").write_text("content")
        model.scan()
        d = model.to_dict()
        assert d["file_count"] == 1
        assert "file.txt" in d["files"]

    def test_scan_nonexistent_dir(self):
        model = WorkspaceModel(sandbox_dir=Path("/nonexistent/path"))
        diff = model.scan()
        assert not diff.has_changes()

    def test_scan_no_dir(self):
        model = WorkspaceModel(sandbox_dir=None)
        diff = model.scan()
        assert not diff.has_changes()

    def test_modification_invalidates_summary(self, model, workspace_dir):
        f = workspace_dir / "data.csv"
        f.write_text("a,b\n1,2")
        model.scan()
        model.record_read("data.csv", summary="Original", columns=["a", "b"])

        import time
        time.sleep(0.05)
        f.write_text("a,b,c\n1,2,3")
        model.scan()
        assert model.files["data.csv"].summary == ""
        assert model.files["data.csv"].columns == []
