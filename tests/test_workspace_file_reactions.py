"""Tests for file_changed reactions — broken-import/parse flags (feature 2A).

The wake-up scan records file_changed events; this is the first consumer.
A changed .py that no longer parses must be flagged; clean files, deleted
files, and non-Python files must stay silent. The flag must reach the
workspace wake-up prefix.
"""

from ghost_agent.workspace.reactions import check_changed_python_files
from ghost_agent.workspace.recognition import build_workspace_prefix


def _change(path, change="modified (+10 bytes)", label=""):
    return {"path": str(path), "change": change, "label": label}


def test_broken_python_file_is_flagged(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def oops(:\n    pass\n")  # syntax error
    warnings = check_changed_python_files([_change(f)])
    assert len(warnings) == 1
    assert warnings[0]["path"] == str(f)
    assert "line" in warnings[0]["error"].lower()


def test_clean_python_file_is_silent(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("def fine():\n    return 42\n")
    assert check_changed_python_files([_change(f)]) == []


def test_non_python_file_skipped(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("def oops(:")  # would be a syntax error IF parsed
    assert check_changed_python_files([_change(f)]) == []


def test_deleted_file_skipped(tmp_path):
    f = tmp_path / "gone.py"  # never created
    assert check_changed_python_files([_change(f, change="deleted")]) == []


def test_missing_file_skipped(tmp_path):
    f = tmp_path / "missing.py"  # change reported but file not on disk
    assert check_changed_python_files([_change(f)]) == []


def test_duplicate_paths_checked_once(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("x = (")
    warnings = check_changed_python_files([_change(f), _change(f)])
    assert len(warnings) == 1


def test_warning_reaches_prefix(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def oops(:\n")
    warnings = check_changed_python_files([_change(f)])
    prefix = build_workspace_prefix(
        activity=None, state=None,
        file_changes=[_change(f)], file_warnings=warnings,
    )
    assert "DO NOT PARSE" in prefix
    assert "broken.py" in prefix


def test_no_warnings_no_broken_section(tmp_path):
    f = tmp_path / "ok.py"
    f.write_text("ok = True\n")
    prefix = build_workspace_prefix(
        activity=None, state=None,
        file_changes=[_change(f)], file_warnings=[],
    )
    # The change is still listed, but no broken-file section appears.
    assert "DO NOT PARSE" not in prefix
    assert "ok.py" in prefix
